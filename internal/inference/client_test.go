package inference

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/sypra-labs/sypra-ai/internal/domain"
	"github.com/sypra-labs/sypra-ai/pkg/mockrunner"
)

func TestChatUsesOpenAICompatibleLowercaseMessageFields(t *testing.T) {
	var payload map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{}}`))
	}))
	defer server.Close()
	client := &Client{HTTP: server.Client(), DefaultEndpoint: server.URL}
	if _, err := client.Chat(context.Background(), "", ChatRequest{Model: "gemini-2.5-pro", Messages: []Message{{Role: "user", Content: "hello"}}}); err != nil {
		t.Fatal(err)
	}
	messages, ok := payload["messages"].([]any)
	if !ok || len(messages) != 1 {
		t.Fatalf("invalid messages payload: %#v", payload)
	}
	message, ok := messages[0].(map[string]any)
	if !ok || message["role"] != "user" || message["content"] != "hello" {
		t.Fatalf("message is not OpenAI-compatible: %#v", messages[0])
	}
	if _, exists := message["Role"]; exists {
		t.Fatalf("message used Go field name: %#v", message)
	}
}

func TestChatPreservesFunctionCalls(t *testing.T) {
	var payload map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call-1","function":{"name":"sypra_browser_search","arguments":"{\"query\":\"Sypra\"}"}}]}}],"usage":{}}`))
	}))
	defer server.Close()
	client := &Client{HTTP: server.Client(), DefaultEndpoint: server.URL}
	response, err := client.Chat(context.Background(), "", ChatRequest{
		Messages: []Message{{Role: "user", Content: "search"}},
		Tools: []ToolDefinition{{
			Type: "function",
			Function: ToolFunction{
				Name:       "sypra_browser_search",
				Parameters: map[string]any{"type": "object"},
			},
		}},
		ToolChoice: "auto",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(response.ToolCalls) != 1 || response.ToolCalls[0].Name != "sypra_browser_search" || response.ToolCalls[0].Arguments["query"] != "Sypra" {
		t.Fatalf("tool call was not decoded: %#v", response.ToolCalls)
	}
	tools, ok := payload["tools"].([]any)
	if !ok || len(tools) != 1 || payload["tool_choice"] != "auto" {
		t.Fatalf("tools were not sent in OpenAI-compatible shape: %#v", payload)
	}
}

func TestChatPreservesMultimodalContentAndResponseParts(t *testing.T) {
	var payload map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":[{"type":"text","text":"I can see it."},{"type":"image_url","image_url":{"url":"https://example.test/result.png"}}]}}],"usage":{}}`))
	}))
	defer server.Close()
	client := &Client{HTTP: server.Client(), DefaultEndpoint: server.URL}
	response, err := client.Chat(context.Background(), "", ChatRequest{Messages: []Message{{Role: "user", Content: "Describe this", Parts: []domain.ContentPart{{Type: "image_url", URL: "data:image/png;base64,abc"}}}}})
	if err != nil {
		t.Fatal(err)
	}
	messages := payload["messages"].([]any)
	content := messages[0].(map[string]any)["content"].([]any)
	if len(content) != 2 || content[1].(map[string]any)["type"] != "image_url" {
		t.Fatalf("multimodal request was not preserved: %#v", content)
	}
	if response.Content != "I can see it." || len(response.Parts) != 2 || response.Parts[1].URL != "https://example.test/result.png" {
		t.Fatalf("multimodal response was not decoded: %#v", response)
	}
}

func TestMockRichOutputFixtures(t *testing.T) {
	server := httptest.NewServer(mockrunner.Handler())
	defer server.Close()
	client := &Client{HTTP: server.Client(), DefaultEndpoint: server.URL + "/v1"}
	for _, prompt := range []string{"mock: markdown", "mock: table", "mock: code", "mock: diagram", "mock: image", "mock: audio", "mock: video", "mock: artifact", "mock: mixed"} {
		response, err := client.Chat(context.Background(), "", ChatRequest{Messages: []Message{{Role: "user", Content: prompt}}})
		if err != nil || (response.Content == "" && len(response.Parts) == 0) {
			t.Fatalf("fixture %q response=%#v err=%v", prompt, response, err)
		}
		if len(response.Parts) > 0 && domain.ValidateContentParts(response.Parts) != nil {
			t.Fatalf("fixture %q produced invalid parts: %#v", prompt, response.Parts)
		}
	}
}
