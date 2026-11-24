package qwen

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/sypra-labs/sypra-ai/internal/chattemplate"
	"github.com/sypra-labs/sypra-ai/internal/domain"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/sypra-labs/sypra-ai/internal/adapter"
)

// Runner exposes a loaded Qwen checkpoint through the small OpenAI-compatible
// surface used by Sypra. CPU execution is deliberately serialized: it keeps
// checkpoint memory bounded and prevents two large generations competing for
// the same machine.
type Runner struct {
	Model      *Model
	ModelID    string
	MaxContext int
	DefaultMax int
	Logger     *slog.Logger
	Parallel   int
	slots      chan struct{}
	slotsMu    sync.Mutex
}

type ChatMessage struct {
	Role    string               `json:"role"`
	Content string               `json:"-"`
	Parts   []domain.ContentPart `json:"-"`
}

func (m *ChatMessage) UnmarshalJSON(data []byte) error {
	var raw struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	m.Role, m.Content, m.Parts = raw.Role, "", nil
	if err := json.Unmarshal(raw.Content, &m.Content); err == nil {
		return nil
	}
	var parts []struct {
		Type     string `json:"type"`
		Text     string `json:"text"`
		ImageURL struct {
			URL string `json:"url"`
		} `json:"image_url"`
		VideoURL struct {
			URL string `json:"url"`
		} `json:"video_url"`
		InputAudio struct{ Data, Format string } `json:"input_audio"`
		File       struct {
			Data string `json:"file_data"`
			Name string `json:"filename"`
		} `json:"file"`
		URL      string `json:"url"`
		Name     string `json:"name"`
		MIMEType string `json:"mime_type"`
	}
	if err := json.Unmarshal(raw.Content, &parts); err != nil {
		return fmt.Errorf("invalid message content: %w", err)
	}
	for _, rawPart := range parts {
		part := domain.ContentPart{Type: rawPart.Type, Text: rawPart.Text}
		switch part.Type {
		case "image_url":
			part.URL = rawPart.ImageURL.URL
		case "video_url":
			part.URL = rawPart.VideoURL.URL
		case "input_audio":
			part.Data, part.MIMEType = rawPart.InputAudio.Data, rawPart.InputAudio.Format
		case "file":
			part.Data, part.Name = rawPart.File.Data, rawPart.File.Name
		case "markdown", "diagram":
			part.Name = rawPart.Name
		case "image", "audio", "video", "artifact":
			part.URL, part.Name, part.MIMEType = rawPart.URL, rawPart.Name, rawPart.MIMEType
		}
		m.Parts = append(m.Parts, part)
	}
	if err := domain.ValidateContentParts(m.Parts); err != nil {
		return err
	}
	m.Content = domain.ContentText("", m.Parts)
	return nil
}

type ChatRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens"`
	Temperature float64       `json:"temperature"`
	Stream      bool          `json:"stream,omitempty"`
	Adapter     string        `json:"adapter,omitempty"`
}

func NewRunner(model *Model, modelID string) *Runner {
	return &Runner{Model: model, ModelID: modelID, MaxContext: model.Config.MaxPositionEmbeddings, DefaultMax: model.Config.MaxPositionEmbeddings, Logger: slog.Default(), Parallel: 10}
}

func (r *Runner) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", r.health)
	mux.HandleFunc("/v1/models", r.models)
	mux.HandleFunc("/v1/tokenize", r.tokenize)
	mux.HandleFunc("/v1/chat/completions", r.chat)
	return mux
}

func (r *Runner) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "engine": "sypra-qwen-cpu", "contextTokens": r.MaxContext, "defaultMaxTokens": r.DefaultMax, "maxParallelRequests": r.Parallel})
}
func (r *Runner) models(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": []map[string]string{{"id": r.ModelID, "object": "model"}}})
}
func (r *Runner) tokenize(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var input struct {
		Messages []ChatMessage `json:"messages"`
	}
	if err := json.NewDecoder(req.Body).Decode(&input); err != nil {
		http.Error(w, "invalid tokenize request: "+err.Error(), http.StatusBadRequest)
		return
	}
	ids, err := r.promptIDs(input.Messages)
	if err != nil {
		http.Error(w, "tokenize failed: "+err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, http.StatusOK, map[string]int{"promptTokens": len(ids), "contextTokens": r.MaxContext})
}

func (r *Runner) chat(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var input ChatRequest
	if err := json.NewDecoder(req.Body).Decode(&input); err != nil {
		http.Error(w, "invalid chat request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if len(input.Messages) == 0 {
		http.Error(w, "messages are required", http.StatusBadRequest)
		return
	}
	maxTokens := input.MaxTokens
	if maxTokens < 1 {
		maxTokens = r.DefaultMax
	}
	start := time.Now()
	r.ensureSlots()
	r.logger().Info("runner generation requested", "model", r.ModelID, "messageCount", len(input.Messages), "requestedMaxTokens", input.MaxTokens, "effectiveMaxTokens", maxTokens, "contextLimit", r.MaxContext)
	queuedAt := time.Now()
	r.slots <- struct{}{}
	defer func() { <-r.slots }()
	r.logger().Info("runner generation started", "model", r.ModelID, "queueWait", time.Since(queuedAt))
	var writeDelta func(string)
	if input.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}
		writeDelta = func(delta string) {
			b, _ := json.Marshal(map[string]any{"choices": []map[string]any{{"delta": map[string]string{"content": delta}}}})
			_, _ = fmt.Fprintf(w, "data: %s\n\n", b)
			flusher.Flush()
		}
	}
	var lora adapter.QwenProjectionAdapter
	if input.Adapter != "" {
		loaded, loadErr := adapter.LoadQwenAdapter(input.Adapter)
		if loadErr != nil {
			http.Error(w, "load Qwen adapter: "+loadErr.Error(), http.StatusBadRequest)
			return
		}
		lora = loaded
	}
	content, promptTokens, completionTokens, err := r.generateStream(req.Context(), input.Messages, maxTokens, writeDelta, lora)
	if err != nil {
		status := http.StatusInternalServerError
		if strings.Contains(err.Error(), "context limit") {
			status = http.StatusUnprocessableEntity
		}
		r.logger().Error("runner generation failed", "model", r.ModelID, "status", status, "duration", time.Since(start), "error", err)
		http.Error(w, "Qwen inference failed: "+err.Error(), status)
		return
	}
	r.logger().Info("runner generation completed", "model", r.ModelID, "promptTokens", promptTokens, "completionTokens", completionTokens, "duration", time.Since(start))
	if input.Stream {
		b, _ := json.Marshal(map[string]any{"choices": []map[string]any{{"delta": map[string]string{}, "finish_reason": "stop"}}, "usage": map[string]int{"prompt_tokens": promptTokens, "completion_tokens": completionTokens, "total_tokens": promptTokens + completionTokens}})
		_, _ = fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", b)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"id": fmt.Sprintf("sypra-%d", start.UnixNano()), "object": "chat.completion", "created": start.Unix(), "model": r.ModelID, "choices": []map[string]any{{"index": 0, "message": map[string]string{"role": "assistant", "content": content}, "finish_reason": "stop"}}, "usage": map[string]int{"prompt_tokens": promptTokens, "completion_tokens": completionTokens, "total_tokens": promptTokens + completionTokens}})
}

func (r *Runner) Generate(ctx context.Context, messages []ChatMessage, maxTokens int) (string, int, int, error) {
	return r.GenerateStream(ctx, messages, maxTokens, nil)
}
func (r *Runner) GenerateStream(ctx context.Context, messages []ChatMessage, maxTokens int, onDelta func(string)) (string, int, int, error) {
	return r.generateStream(ctx, messages, maxTokens, onDelta, nil)
}

func (r *Runner) generateStream(ctx context.Context, messages []ChatMessage, maxTokens int, onDelta func(string), lora adapter.QwenProjectionAdapter) (string, int, int, error) {
	ids, err := r.promptIDs(messages)
	if err != nil {
		return "", 0, 0, err
	}
	if len(ids) >= r.MaxContext {
		return "", 0, 0, fmt.Errorf("prompt has %d tokens; local runner context limit is %d", len(ids), r.MaxContext-1)
	}
	r.logger().Info("runner prompt tokenized", "model", r.ModelID, "promptTokens", len(ids), "remainingContextTokens", r.MaxContext-len(ids), "requestedMaxTokens", maxTokens)
	state := NewState(r.Model.Config.NumHiddenLayers)
	var logits []float32
	for position, id := range ids {
		if err = ctx.Err(); err != nil {
			return "", 0, 0, err
		}
		logits, err = r.Model.ForwardWithAdapter(id, position, state, lora)
		if err != nil {
			return "", 0, 0, err
		}
	}
	output := make([]int, 0, maxTokens)
	for len(output) < maxTokens && len(ids)+len(output) < r.MaxContext {
		if err = ctx.Err(); err != nil {
			return "", 0, 0, err
		}
		next := argmax(logits)
		if next == 151643 || next == 151645 {
			break
		}
		output = append(output, next)
		if onDelta != nil {
			onDelta(r.Model.Tokenizer.Decode([]int{next}))
		}
		logits, err = r.Model.ForwardWithAdapter(next, len(ids)+len(output)-1, state, lora)
		if err != nil {
			return "", 0, 0, err
		}
	}
	return strings.TrimSpace(r.Model.Tokenizer.Decode(output)), len(ids), len(output), nil
}
func (r *Runner) ensureSlots() {
	r.slotsMu.Lock()
	defer r.slotsMu.Unlock()
	if r.Parallel < 1 {
		r.Parallel = 10
	}
	if r.slots == nil {
		r.slots = make(chan struct{}, r.Parallel)
	}
}
func (r *Runner) logger() *slog.Logger {
	if r.Logger != nil {
		return r.Logger
	}
	return slog.Default()
}

func (r *Runner) promptIDs(messages []ChatMessage) ([]int, error) {
	converted := make([]chattemplate.Message, len(messages))
	for index, message := range messages {
		if domain.HasNonTextContent(message.Parts) {
			return nil, fmt.Errorf("the local Qwen text runner cannot process non-text content; select a multimodal endpoint for images, audio, video, or files")
		}
		converted[index] = chattemplate.Message{Role: message.Role, Content: message.Content, Parts: message.Parts}
	}
	rendered, err := chattemplate.Render(chattemplate.FamilyQwen, converted, false)
	if err != nil {
		return nil, err
	}
	ids, _, err := chattemplate.ResponseTokenMask(rendered.Segments, func(segment chattemplate.Segment) ([]int, error) {
		if segment.Control != "" {
			id, found := r.Model.Tokenizer.EncodeSpecial(segment.Control)
			if !found {
				return nil, fmt.Errorf("Qwen tokenizer has no control token %q", segment.Control)
			}
			return []int{id}, nil
		}
		return r.Model.Tokenizer.Encode(segment.Text)
	})
	return ids, err
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
