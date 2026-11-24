// Package mockrunner provides an independently runnable, OpenAI-compatible
// test model. It is intentionally separate from the API process so tests use
// the same network, streaming, cancellation, and runner lifecycle as a real
// local model.
package mockrunner

import (
	"crypto/rand"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/sypra-labs/sypra-ai/internal/domain"
)

// Handler exposes a small OpenAI-compatible local model service.
func Handler() http.Handler { return http.HandlerFunc(route) }

func route(w http.ResponseWriter, r *http.Request) {
	switch {
	case r.URL.Path == "/v1/models" && r.Method == http.MethodGet:
		writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": []map[string]any{{"id": "sypra-mock", "object": "model"}}})
	case r.URL.Path == "/v1/tokenize" && r.Method == http.MethodPost:
		var input chatRequest
		if json.NewDecoder(r.Body).Decode(&input) != nil {
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		writeJSON(w, http.StatusOK, map[string]int{"promptTokens": len(strings.Fields(lastUser(input.Messages))), "contextTokens": 131072})
	case r.URL.Path == "/v1/chat/completions" && r.Method == http.MethodPost:
		var input chatRequest
		if json.NewDecoder(r.Body).Decode(&input) != nil {
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		response := fixture(lastUser(input.Messages))
		if input.Stream {
			stream(w, r, response, len(strings.Fields(lastUser(input.Messages))))
			return
		}
		writeJSON(w, http.StatusOK, completion(response, len(strings.Fields(lastUser(input.Messages)))))
	default:
		http.NotFound(w, r)
	}
}

type chatRequest struct {
	Messages []message `json:"messages"`
	Stream   bool      `json:"stream"`
}
type message struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}
type response struct {
	Content string
	Parts   []domain.ContentPart
}

func lastUser(messages []message) string {
	for index := len(messages) - 1; index >= 0; index-- {
		if messages[index].Role != "user" {
			continue
		}
		var text string
		if json.Unmarshal(messages[index].Content, &text) == nil {
			return text
		}
		var parts []struct{ Type, Text string }
		if json.Unmarshal(messages[index].Content, &parts) == nil {
			var out strings.Builder
			for _, part := range parts {
				if part.Type == "text" {
					out.WriteString(part.Text)
				}
			}
			return out.String()
		}
	}
	return ""
}

func completion(result response, promptTokens int) map[string]any {
	return map[string]any{
		"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": outputContent(result)}}},
		"usage":   map[string]int{"prompt_tokens": promptTokens, "completion_tokens": len(strings.Fields(result.Content)) + len(result.Parts)*4},
	}
}

func stream(w http.ResponseWriter, r *http.Request, result response, promptTokens int) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unavailable", http.StatusInternalServerError)
		return
	}
	words := strings.Fields(result.Content)
	duration := fixtureDuration()
	delay := duration
	if len(words) > 0 {
		delay = duration / time.Duration(len(words))
	}
	for index, word := range words {
		select {
		case <-r.Context().Done():
			return
		case <-time.After(delay):
		}
		if index < len(words)-1 {
			word += " "
		}
		writeSSE(w, map[string]any{"choices": []map[string]any{{"delta": map[string]string{"content": word}}}})
		flusher.Flush()
	}
	writeSSE(w, map[string]any{"choices": []map[string]any{{"delta": map[string]string{}}}, "parts": result.Parts, "usage": map[string]int{"prompt_tokens": promptTokens, "completion_tokens": len(words) + len(result.Parts)*4}})
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func writeSSE(w http.ResponseWriter, value any) {
	bytes, _ := json.Marshal(value)
	fmt.Fprintf(w, "data: %s\n\n", bytes)
}
func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
func fixtureDuration() time.Duration {
	var value [1]byte
	if _, err := rand.Read(value[:]); err != nil {
		return 10 * time.Second
	}
	return time.Duration(5+int(value[0])%11) * time.Second
}

func outputContent(result response) any {
	if len(result.Parts) == 0 {
		return result.Content
	}
	parts := []map[string]any{{"type": "text", "text": result.Content}}
	for _, part := range result.Parts {
		value := map[string]any{"type": part.Type, "name": part.Name, "mime_type": part.MIMEType}
		switch part.Type {
		case "diagram", "markdown":
			value["text"] = part.Text
		case "image_url", "video_url":
			value[part.Type] = map[string]string{"url": part.Data}
		case "input_audio":
			value["input_audio"] = map[string]string{"data": part.Data, "format": part.MIMEType}
		default:
			value["data"] = part.Data
			value["url"] = part.URL
		}
		parts = append(parts, value)
	}
	return parts
}

func fixture(prompt string) response {
	lower := strings.ToLower(prompt)
	plain := response{Content: "Test response: the mock model received your prompt: " + prompt}
	switch {
	case strings.Contains(lower, "mock: markdown"):
		return response{Content: "# Mock Markdown\n\nThis is **bold**, this is `inline code`, and this list is rendered locally.\n\n- First item\n- Second item"}
	case strings.Contains(lower, "mock: table"):
		return response{Content: "# Mock table\n\n| Model | Status | Score |\n| --- | --- | ---: |\n| Qwen | Active | 0.92 |\n| Mock | Ready | 1.00 |"}
	case strings.Contains(lower, "mock: code"):
		return response{Content: "# Mock code\n\n```go\nfunc greet(name string) string {\n    return \"Hello, \" + name\n}\n```"}
	case strings.Contains(lower, "mock: diagram"):
		return response{Content: "Mock diagram response.", Parts: []domain.ContentPart{{Type: "diagram", Name: "retrieval-flow.mmd", Text: "flowchart LR\n  Query --> Retrieve\n  Retrieve --> Rerank\n  Rerank --> Answer"}}}
	case strings.Contains(lower, "mock: image"):
		return response{Content: "Mock generated image.", Parts: []domain.ContentPart{{Type: "image_url", Name: "mock-image.svg", MIMEType: "image/svg+xml", Data: mockSVG("Mock image")}}}
	case strings.Contains(lower, "mock: audio"):
		return response{Content: "Mock audio output.", Parts: []domain.ContentPart{{Type: "input_audio", Name: "mock-audio.wav", MIMEType: "audio/wav", Data: "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="}}}
	case strings.Contains(lower, "mock: video"):
		return response{Content: "Mock video output.", Parts: []domain.ContentPart{{Type: "video_url", Name: "mock-video.webm", MIMEType: "video/webm", Data: "data:video/webm;base64,GkXfo0AgQoaBAULygQFC8oEEQvOBCEKCQAR3ZWJtQoeBAg=="}}}
	case strings.Contains(lower, "mock: artifact"):
		return response{Content: "Mock artifact output.", Parts: []domain.ContentPart{{Type: "artifact", Name: "mock-report.md", MIMEType: "text/markdown", Data: "data:text/markdown;charset=utf-8,%23%20Mock%20artifact%0A%0AGenerated%20locally."}}}
	case strings.Contains(lower, "mock: mixed"):
		return response{Content: "# Mock mixed response\n\nA response can carry Markdown plus multiple independently downloadable media parts.", Parts: []domain.ContentPart{{Type: "diagram", Name: "flow.mmd", Text: "flowchart LR\n  Prompt --> Model\n  Model --> Text\n  Model --> Media"}, {Type: "image_url", Name: "mock-image.svg", MIMEType: "image/svg+xml", Data: mockSVG("Mixed image")}, {Type: "artifact", Name: "mock-notes.md", MIMEType: "text/markdown", Data: "data:text/markdown,Mock%20artifact"}, {Type: "input_audio", Name: "mock-audio.wav", MIMEType: "audio/wav", Data: "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="}, {Type: "video_url", Name: "mock-video.webm", MIMEType: "video/webm", Data: "data:video/webm;base64,GkXfo0AgQoaBAULygQFC8oEEQvOBCEKCQAR3ZWJtQoeBAg=="}}}
	default:
		return plain
	}
}
func mockSVG(label string) string {
	return "data:image/svg+xml," + strings.ReplaceAll("<svg xmlns='http://www.w3.org/2000/svg' width='640' height='360'><rect width='100%' height='100%' fill='%231d4ed8'/><text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' font-family='sans-serif' font-size='36' fill='white'>"+label+"</text></svg>", "#", "%23")
}
