package inference

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"github.com/sypra-labs/sypra-ai/internal/domain"
	"github.com/sypra-labs/sypra-ai/internal/finetune"
	"io"
	"net/http"
	"strings"
	"time"
)

// Client is intentionally provider-neutral. Any local runner exposing the
// OpenAI-compatible /chat/completions contract (including a custom runner) works.
type Client struct {
	HTTP                    *http.Client
	DefaultEndpoint, APIKey string
}
type ChatRequest struct {
	Model       string           `json:"model"`
	Messages    []Message        `json:"messages"`
	Temperature float64          `json:"temperature,omitempty"`
	MaxTokens   int              `json:"max_tokens,omitempty"`
	Stream      bool             `json:"stream,omitempty"`
	Adapter     string           `json:"adapter,omitempty"`
	Tools       []ToolDefinition `json:"tools,omitempty"`
	ToolChoice  string           `json:"tool_choice,omitempty"`
	APIKey      string           `json:"-"`
}

// ToolDefinition and ToolCall mirror the OpenAI function-calling wire format
// without binding the inference client to a particular provider SDK.
type ToolDefinition struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}
type ToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
}
type ToolCall struct {
	ID        string         `json:"id"`
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// MarshalJSON emits the assistant-message form used by OpenAI-compatible
// providers. ToolCall stays ergonomic internally while the provider receives
// the required nested function and JSON-encoded arguments fields.
func (c ToolCall) MarshalJSON() ([]byte, error) {
	arguments, err := json.Marshal(c.Arguments)
	if err != nil {
		return nil, err
	}
	return json.Marshal(struct {
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	}{ID: c.ID, Type: "function", Function: struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	}{Name: c.Name, Arguments: string(arguments)}})
}

// ChatStream delivers local-runner deltas as they are generated. It uses the
// OpenAI-compatible SSE shape and is deliberately optional for cloud endpoints.
func (c *Client) ChatStream(ctx context.Context, endpoint string, req ChatRequest, onDelta func(string)) (ChatResponse, error) {
	base := strings.TrimRight(endpoint, "/")
	if base == "" {
		base = strings.TrimRight(c.DefaultEndpoint, "/")
	}
	if base == "" || strings.HasPrefix(base, "native://") {
		return c.Chat(ctx, endpoint, req)
	}
	req.Stream = true
	payload, err := json.Marshal(req)
	if err != nil {
		return ChatResponse{}, err
	}
	start := time.Now()
	h, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return ChatResponse{}, err
	}
	h.Header.Set("Content-Type", "application/json")
	apiKey := req.APIKey
	if apiKey == "" {
		apiKey = c.APIKey
	}
	if apiKey != "" {
		h.Header.Set("Authorization", "Bearer "+apiKey)
	}
	client := c.HTTP
	if client == nil {
		client = http.DefaultClient
	}
	response, err := client.Do(h)
	if err != nil {
		return ChatResponse{}, fmt.Errorf("local inference request: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode > 299 {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 64<<10))
		return ChatResponse{}, fmt.Errorf("local inference service returned %s: %s", response.Status, strings.TrimSpace(string(body)))
	}
	var out strings.Builder
	var usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	}
	var outputParts []domain.ContentPart
	scanner := bufio.NewScanner(response.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}
		var event struct {
			Choices []struct {
				Delta Message `json:"delta"`
			} `json:"choices"`
			Parts []domain.ContentPart `json:"parts"`
			Usage struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			return ChatResponse{}, err
		}
		usage.PromptTokens, usage.CompletionTokens = event.Usage.PromptTokens, event.Usage.CompletionTokens
		if len(event.Parts) > 0 {
			outputParts = event.Parts
		}
		if event.Choices != nil && event.Choices[0].Delta.Content != "" {
			delta := event.Choices[0].Delta.Content
			out.WriteString(delta)
			if onDelta != nil {
				onDelta(delta)
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return ChatResponse{}, err
	}
	return ChatResponse{Content: strings.TrimSpace(out.String()), Parts: outputParts, InputTokens: usage.PromptTokens, OutputTokens: usage.CompletionTokens, Latency: time.Since(start)}, nil
}

// Message uses the lowercase OpenAI-compatible wire names. Go's default
// `Role`/`Content` serialization happens to work with our permissive local
// runner, but cloud compatibility endpoints (notably Gemini) reject it.
type Message struct {
	Role       string               `json:"role"`
	Content    string               `json:"-"`
	Parts      []domain.ContentPart `json:"-"`
	ToolCallID string               `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCall           `json:"tool_calls,omitempty"`
}

// MarshalJSON uses the OpenAI-compatible multimodal content-array shape when
// attachments exist, and retains the simple string form for text-only models.
func (m Message) MarshalJSON() ([]byte, error) {
	if len(m.Parts) == 0 {
		return json.Marshal(struct {
			Role       string     `json:"role"`
			Content    string     `json:"content"`
			ToolCallID string     `json:"tool_call_id,omitempty"`
			ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
		}{m.Role, m.Content, m.ToolCallID, m.ToolCalls})
	}
	parts := make([]map[string]any, 0, len(m.Parts)+1)
	if strings.TrimSpace(m.Content) != "" {
		parts = append(parts, map[string]any{"type": "text", "text": m.Content})
	}
	for _, part := range m.Parts {
		switch part.Type {
		case "text":
			parts = append(parts, map[string]any{"type": "text", "text": part.Text})
		case "image_url", "video_url":
			url := part.URL
			if url == "" {
				url = part.Data
			}
			parts = append(parts, map[string]any{"type": part.Type, part.Type: map[string]any{"url": url}})
		case "input_audio":
			data := part.Data
			if data == "" {
				data = part.URL
			}
			parts = append(parts, map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": data, "format": part.MIMEType}})
		case "file":
			data := part.Data
			if data == "" {
				data = part.URL
			}
			parts = append(parts, map[string]any{"type": "file", "file": map[string]any{"file_data": data, "filename": part.Name}})
		case "markdown", "diagram":
			parts = append(parts, map[string]any{"type": part.Type, "text": part.Text, "name": part.Name})
		case "image", "audio", "video", "artifact":
			data := part.Data
			if data == "" {
				data = part.URL
			}
			parts = append(parts, map[string]any{"type": part.Type, "url": data, "name": part.Name, "mime_type": part.MIMEType})
		}
	}
	return json.Marshal(struct {
		Role       string     `json:"role"`
		Content    any        `json:"content"`
		ToolCallID string     `json:"tool_call_id,omitempty"`
		ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	}{m.Role, parts, m.ToolCallID, m.ToolCalls})
}

func (m *Message) UnmarshalJSON(data []byte) error {
	var raw struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	m.Role, m.Content, m.Parts = raw.Role, "", nil
	if len(raw.Content) == 0 || string(raw.Content) == "null" {
		return nil
	}
	if err := json.Unmarshal(raw.Content, &m.Content); err == nil {
		return nil
	}
	var values []map[string]json.RawMessage
	if err := json.Unmarshal(raw.Content, &values); err != nil {
		return fmt.Errorf("invalid message content: %w", err)
	}
	for _, value := range values {
		var kind string
		_ = json.Unmarshal(value["type"], &kind)
		part := domain.ContentPart{Type: kind}
		switch kind {
		case "text":
			_ = json.Unmarshal(value["text"], &part.Text)
			m.Content += part.Text
		case "image_url", "video_url":
			var nested struct {
				URL string `json:"url"`
			}
			_ = json.Unmarshal(value[kind], &nested)
			part.URL = nested.URL
		case "input_audio":
			var nested struct {
				Data   string `json:"data"`
				Format string `json:"format"`
			}
			_ = json.Unmarshal(value["input_audio"], &nested)
			part.Data, part.MIMEType = nested.Data, nested.Format
		case "file":
			var nested struct {
				Data string `json:"file_data"`
				Name string `json:"filename"`
			}
			_ = json.Unmarshal(value["file"], &nested)
			part.Data, part.Name = nested.Data, nested.Name
		case "markdown", "diagram":
			_ = json.Unmarshal(value["text"], &part.Text)
			_ = json.Unmarshal(value["name"], &part.Name)
		case "image", "audio", "video", "artifact":
			_ = json.Unmarshal(value["url"], &part.URL)
			_ = json.Unmarshal(value["data"], &part.Data)
			_ = json.Unmarshal(value["name"], &part.Name)
			_ = json.Unmarshal(value["mime_type"], &part.MIMEType)
		}
		m.Parts = append(m.Parts, part)
	}
	return domain.ValidateContentParts(m.Parts)
}

type ChatResponse struct {
	Content                   string               `json:"content"`
	Parts                     []domain.ContentPart `json:"parts,omitempty"`
	ToolCalls                 []ToolCall           `json:"toolCalls,omitempty"`
	InputTokens, OutputTokens int
	Latency                   time.Duration
}

// CountTokens asks Sypra's local runner to tokenize messages without running
// inference. Third-party OpenAI-compatible endpoints do not implement this
// optional endpoint, in which case supported is false.
func (c *Client) CountTokens(ctx context.Context, endpoint string, messages []Message) (tokens, contextLimit int, supported bool, err error) {
	base := strings.TrimRight(endpoint, "/")
	if base == "" {
		base = strings.TrimRight(c.DefaultEndpoint, "/")
	}
	if base == "" || strings.HasPrefix(base, "native://") {
		return 0, 0, false, nil
	}
	payload, err := json.Marshal(struct {
		Messages []Message `json:"messages"`
	}{Messages: messages})
	if err != nil {
		return 0, 0, false, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/tokenize", bytes.NewReader(payload))
	if err != nil {
		return 0, 0, false, err
	}
	request.Header.Set("Content-Type", "application/json")
	client := c.HTTP
	if client == nil {
		client = http.DefaultClient
	}
	response, err := client.Do(request)
	if err != nil {
		return 0, 0, false, err
	}
	defer response.Body.Close()
	if response.StatusCode == http.StatusNotFound {
		return 0, 0, false, nil
	}
	if response.StatusCode < 200 || response.StatusCode > 299 {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 64<<10))
		return 0, 0, false, fmt.Errorf("local tokenizer returned %s: %s", response.Status, strings.TrimSpace(string(body)))
	}
	var result struct {
		PromptTokens  int `json:"promptTokens"`
		ContextTokens int `json:"contextTokens"`
	}
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		return 0, 0, false, err
	}
	if result.ContextTokens < 1 {
		return 0, 0, false, fmt.Errorf("local tokenizer returned an invalid context size")
	}
	return result.PromptTokens, result.ContextTokens, true, nil
}

func (c *Client) Chat(ctx context.Context, endpoint string, req ChatRequest) (ChatResponse, error) {
	base := strings.TrimRight(endpoint, "/")
	if base == "" {
		base = strings.TrimRight(c.DefaultEndpoint, "/")
	}
	if base == "" {
		return ChatResponse{}, fmt.Errorf("no local model endpoint configured")
	}
	if strings.HasPrefix(base, "native://") {
		prompt := lastUserPrompt(req.Messages)
		artifact, err := finetune.LoadArtifact(strings.TrimPrefix(base, "native://"))
		if err != nil {
			return ChatResponse{}, fmt.Errorf("load fine-tuned SLM adapter: %w", err)
		}
		content := artifact.Generate(prompt)
		return ChatResponse{Content: content, InputTokens: len(strings.Fields(prompt)), OutputTokens: len(strings.Fields(content)), Latency: time.Millisecond}, nil
	}
	payload, e := json.Marshal(req)
	if e != nil {
		return ChatResponse{}, e
	}
	start := time.Now()
	h, e := http.NewRequestWithContext(ctx, http.MethodPost, base+"/chat/completions", bytes.NewReader(payload))
	if e != nil {
		return ChatResponse{}, e
	}
	h.Header.Set("Content-Type", "application/json")
	apiKey := req.APIKey
	if apiKey == "" {
		apiKey = c.APIKey
	}
	if apiKey != "" {
		h.Header.Set("Authorization", "Bearer "+apiKey)
	}
	r, e := c.HTTP.Do(h)
	if e != nil {
		return ChatResponse{}, fmt.Errorf("local inference request: %w", e)
	}
	defer r.Body.Close()
	if r.StatusCode < 200 || r.StatusCode > 299 {
		body, _ := io.ReadAll(io.LimitReader(r.Body, 64<<10))
		message := strings.TrimSpace(string(body))
		if message != "" {
			return ChatResponse{}, fmt.Errorf("local inference service returned %s: %s", r.Status, message)
		}
		return ChatResponse{}, fmt.Errorf("local inference service returned %s", r.Status)
	}
	var raw struct {
		Choices []struct {
			Message struct {
				Role      string          `json:"role"`
				Content   json.RawMessage `json:"content"`
				ToolCalls []struct {
					ID       string `json:"id"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if e = json.NewDecoder(r.Body).Decode(&raw); e != nil {
		return ChatResponse{}, e
	}
	if len(raw.Choices) == 0 {
		return ChatResponse{}, fmt.Errorf("inference response contained no choices")
	}
	encodedMessage, err := json.Marshal(map[string]any{"role": raw.Choices[0].Message.Role, "content": raw.Choices[0].Message.Content})
	if err != nil {
		return ChatResponse{}, err
	}
	var message Message
	if err := json.Unmarshal(encodedMessage, &message); err != nil {
		return ChatResponse{}, err
	}
	toolCalls, parseErr := parseToolCalls(raw.Choices[0].Message.ToolCalls)
	if parseErr != nil {
		return ChatResponse{}, parseErr
	}
	return ChatResponse{Content: message.Content, Parts: message.Parts, ToolCalls: toolCalls, InputTokens: raw.Usage.PromptTokens, OutputTokens: raw.Usage.CompletionTokens, Latency: time.Since(start)}, nil
}

func parseToolCalls(raw []struct {
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}) ([]ToolCall, error) {
	result := make([]ToolCall, 0, len(raw))
	for _, call := range raw {
		arguments := map[string]any{}
		if strings.TrimSpace(call.Function.Arguments) != "" && json.Unmarshal([]byte(call.Function.Arguments), &arguments) != nil {
			return nil, fmt.Errorf("tool call %s has invalid JSON arguments", call.Function.Name)
		}
		if strings.TrimSpace(call.Function.Name) == "" {
			return nil, fmt.Errorf("tool call is missing a function name")
		}
		result = append(result, ToolCall{ID: call.ID, Name: call.Function.Name, Arguments: arguments})
	}
	return result, nil
}

func lastUserPrompt(messages []Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return messages[i].Content
		}
	}
	return ""
}
