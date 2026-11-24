package api

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/sypra-labs/sypra-ai/internal/domain"
	"github.com/sypra-labs/sypra-ai/internal/inference"
	"github.com/sypra-labs/sypra-ai/internal/rag"
	"github.com/sypra-labs/sypra-ai/internal/store"
	"github.com/sypra-labs/sypra-ai/internal/training"
	"github.com/sypra-labs/sypra-ai/internal/websearch"
	"github.com/sypra-labs/sypra-ai/pkg/mockrunner"
)

func TestRAGLifecycle(t *testing.T) {
	t.Parallel()
	st, err := store.Open(filepath.Join(t.TempDir(), "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	srv := &Server{
		Store: st, RAG: rag.Service{Store: st}, Trainer: training.New(st),
		Inference: mockInference(t), CORSOrigin: "http://localhost:5173",
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
	h := srv.Handler()

	create := request(h, http.MethodPost, "/api/v1/rag/collections", map[string]any{
		"name": "handbook", "chunkSize": 100, "chunkOverlap": 20,
		"documents": []map[string]string{{"source": "policy.txt", "text": "Expenses require receipts. Travel needs manager approval."}},
	})
	if create.Code != http.StatusCreated {
		t.Fatalf("create status %d: %s", create.Code, create.Body.String())
	}
	var collection struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(create.Body).Decode(&collection); err != nil {
		t.Fatal(err)
	}
	if collection.ID == "" {
		t.Fatal("collection ID was empty")
	}

	query := request(h, http.MethodPost, "/api/v1/rag/query", map[string]any{"collectionID": collection.ID, "query": "travel"})
	if query.Code != http.StatusOK {
		t.Fatalf("query status %d: %s", query.Code, query.Body.String())
	}
	var chunks []struct{ Source, Content string }
	if err := json.NewDecoder(query.Body).Decode(&chunks); err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 1 || chunks[0].Source != "policy.txt" {
		t.Fatalf("unexpected retrieval: %#v", chunks)
	}

	chat := request(h, http.MethodPost, "/api/v1/chat", map[string]any{"prompt": "Is travel approved?"})
	if chat.Code != http.StatusOK {
		t.Fatalf("chat status %d: %s", chat.Code, chat.Body.String())
	}
	var response struct {
		Content string `json:"content"`
	}
	if err := json.NewDecoder(chat.Body).Decode(&response); err != nil {
		t.Fatal(err)
	}
	if response.Content != "Test response: the mock model received your prompt: Is travel approved?" {
		t.Fatalf("unexpected mock response: %q", response.Content)
	}

	workspace := request(h, http.MethodPost, "/api/v1/chats", map[string]any{})
	if workspace.Code != http.StatusCreated {
		t.Fatalf("chat create status %d", workspace.Code)
	}
	var workspaceChat struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(workspace.Body).Decode(&workspaceChat); err != nil {
		t.Fatal(err)
	}
	bound := request(h, http.MethodPatch, "/api/v1/chats/"+workspaceChat.ID, map[string]any{"collectionId": collection.ID})
	if bound.Code != http.StatusOK {
		t.Fatalf("chat bind status %d: %s", bound.Code, bound.Body.String())
	}
	persisted := request(h, http.MethodPost, "/api/v1/chat", map[string]any{"chatID": workspaceChat.ID, "prompt": "What needs approval?"})
	if persisted.Code != http.StatusOK {
		t.Fatalf("persisted chat status %d: %s", persisted.Code, persisted.Body.String())
	}
	history := request(h, http.MethodGet, "/api/v1/chats/"+workspaceChat.ID+"/messages", nil)
	var stored []domain.ChatMessage
	if err := json.NewDecoder(history.Body).Decode(&stored); err != nil {
		t.Fatal(err)
	}
	if len(stored) != 2 || stored[0].Role != "user" || stored[1].Role != "assistant" {
		t.Fatalf("unexpected chat history: %#v", stored)
	}
}

func TestChatRAGModesUseFocusedOrFullContext(t *testing.T) {
	st, err := store.Open(filepath.Join(t.TempDir(), "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	service := rag.Service{Store: st}
	collection, err := service.Create(context.Background(), rag.CreateInput{
		Name:      "knowledge",
		ChunkSize: 1000,
		Documents: []rag.Document{
			{Source: "relevant.txt", Text: "The needle is stored in the blue cabinet."},
			{Source: "other.txt", Text: "The garden is watered every morning."},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	received := make(chan []inference.Message, 2)
	runner := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload inference.ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		received <- payload.Messages
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"answer"}}],"usage":{"prompt_tokens":4,"completion_tokens":1}}`))
	}))
	defer runner.Close()
	srv := &Server{Store: st, RAG: service, Trainer: training.New(st), Inference: &inference.Client{HTTP: runner.Client(), DefaultEndpoint: runner.URL}, CORSOrigin: "http://localhost:5173", Logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	created := request(srv.Handler(), http.MethodPost, "/api/v1/chats", map[string]any{"collectionId": collection.ID, "ragMode": "focused", "ragTopK": 1})
	var chat domain.Chat
	if created.Code != http.StatusCreated || json.NewDecoder(created.Body).Decode(&chat) != nil {
		t.Fatalf("chat setup failed: %d %s", created.Code, created.Body.String())
	}
	response := request(srv.Handler(), http.MethodPost, "/api/v1/chat", map[string]any{"chatID": chat.ID, "prompt": "Where is the needle?"})
	if response.Code != http.StatusOK {
		t.Fatalf("focused chat failed: %d %s", response.Code, response.Body.String())
	}
	focused := <-received
	if len(focused) == 0 || !strings.Contains(focused[0].Content, "needle") || strings.Contains(focused[0].Content, "garden") {
		t.Fatalf("focused RAG did not isolate relevant context: %#v", focused)
	}
	updated := request(srv.Handler(), http.MethodPatch, "/api/v1/chats/"+chat.ID, map[string]any{"ragMode": "full"})
	if updated.Code != http.StatusOK {
		t.Fatalf("switch to full RAG failed: %d %s", updated.Code, updated.Body.String())
	}
	response = request(srv.Handler(), http.MethodPost, "/api/v1/chat", map[string]any{"chatID": chat.ID, "prompt": "Summarize the knowledge base."})
	if response.Code != http.StatusOK {
		t.Fatalf("full chat failed: %d %s", response.Code, response.Body.String())
	}
	full := <-received
	if len(full) == 0 || !strings.Contains(full[0].Content, "needle") || !strings.Contains(full[0].Content, "garden") {
		t.Fatalf("full RAG did not include all context: %#v", full)
	}
}

func TestAsyncChatPersistsWorkingStateAndCompletes(t *testing.T) {
	st, err := store.Open(filepath.Join(t.TempDir(), "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	started, release := make(chan struct{}), make(chan struct{})
	runner := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		close(started)
		<-release
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"completed after refresh"}}],"usage":{"prompt_tokens":3,"completion_tokens":2}}`))
	}))
	defer runner.Close()
	srv := &Server{Store: st, RAG: rag.Service{Store: st}, Trainer: training.New(st), Inference: &inference.Client{HTTP: runner.Client(), DefaultEndpoint: runner.URL}, CORSOrigin: "http://localhost:5173", Logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	created := request(srv.Handler(), http.MethodPost, "/api/v1/chats", map[string]any{})
	var chat domain.Chat
	if created.Code != http.StatusCreated || json.NewDecoder(created.Body).Decode(&chat) != nil {
		t.Fatalf("chat setup failed: %d %s", created.Code, created.Body.String())
	}
	response := request(srv.Handler(), http.MethodPost, "/api/v1/chat", map[string]any{"chatID": chat.ID, "prompt": "continue in background", "async": true})
	if response.Code != http.StatusAccepted {
		t.Fatalf("async chat status %d: %s", response.Code, response.Body.String())
	}
	<-started
	working, err := st.Messages(context.Background(), chat.ID)
	if err != nil || len(working) != 2 || working[1].Status != "working" {
		t.Fatalf("working state was not persisted: %#v err=%v", working, err)
	}
	close(release)
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		messages, messageErr := st.Messages(context.Background(), chat.ID)
		if messageErr != nil {
			t.Fatal(messageErr)
		}
		if len(messages) == 2 && messages[1].Status == "completed" {
			if messages[1].Content != "completed after refresh" {
				t.Fatalf("unexpected completed content: %#v", messages[1])
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("background chat did not complete")
}

func TestExplicitWebRequestAddsTemporaryCitedContext(t *testing.T) {
	var webServer *httptest.Server
	webServer = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/search":
			_, _ = w.Write([]byte(`<a class="result__a" href="` + webServer.URL + `/article">Live release</a><div class="result__snippet">release summary</div>`))
		case "/article":
			_, _ = w.Write([]byte(`<html><body>The release happened today.</body></html>`))
		}
	}))
	defer webServer.Close()
	received := make(chan []inference.Message, 1)
	runner := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload inference.ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		received <- payload.Messages
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"live answer"}}],"usage":{"prompt_tokens":4,"completion_tokens":2}}`))
	}))
	defer runner.Close()
	st, err := store.Open(filepath.Join(t.TempDir(), "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	srv := &Server{Store: st, RAG: rag.Service{Store: st}, Trainer: training.New(st), Inference: &inference.Client{HTTP: runner.Client(), DefaultEndpoint: runner.URL}, CORSOrigin: "http://localhost:5173", Logger: slog.New(slog.NewTextHandler(io.Discard, nil)), Web: websearch.Service{SearchEndpoint: webServer.URL + "/search", HTTP: webServer.Client(), MaxResults: 1, MaxPageBytes: 1 << 20}}
	response := request(srv.Handler(), http.MethodPost, "/api/v1/chat", map[string]any{"prompt": "Please search the web for the release."})
	if response.Code != http.StatusOK {
		t.Fatalf("web chat status %d: %s", response.Code, response.Body.String())
	}
	messages := <-received
	if len(messages) == 0 || !strings.Contains(messages[0].Content, "The release happened today.") {
		t.Fatalf("web context was not sent to model: %#v", messages)
	}
	if !strings.Contains(response.Body.String(), "Web: Live release") {
		t.Fatalf("web citation missing: %s", response.Body.String())
	}
}

func TestNativeFineTuningLifecycle(t *testing.T) {
	dir := t.TempDir()
	st, err := store.Open(filepath.Join(dir, "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	dataset := filepath.Join(dir, "dataset.csv")
	csv := "instruction,response,category,intent\nhello,Hi there!,general,greeting\nhello friend,Hi there!,general,greeting\ngoodbye,Goodbye!,general,farewell\nbye now,Goodbye!,general,farewell\nthanks,You are welcome!,general,thanks\n"
	if err = os.WriteFile(dataset, []byte(csv), 0600); err != nil {
		t.Fatal(err)
	}
	srv := &Server{Store: st, RAG: rag.Service{Store: st}, Trainer: training.New(st), Inference: mockInference(t), CORSOrigin: "http://localhost:5173", Logger: slog.New(slog.NewTextHandler(io.Discard, nil)), UploadDir: filepath.Join(dir, "models")}
	started := request(srv.Handler(), http.MethodPost, "/api/v1/training/jobs", map[string]any{"name": "support-skill", "baseModel": "C:/models/Qwen3-0.6B", "datasetPath": dataset, "method": "lora", "parameters": map[string]any{"epochs": 15, "batchSize": 1, "learningRate": .2}})
	if started.Code != http.StatusAccepted {
		t.Fatalf("start status %d: %s", started.Code, started.Body.String())
	}
	var startedJob domain.TrainingJob
	if err = json.NewDecoder(started.Body).Decode(&startedJob); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(3 * time.Second)
	var job domain.TrainingJob
	for time.Now().Before(deadline) {
		jobs, jobsErr := st.Jobs(context.Background())
		if jobsErr != nil {
			t.Fatal(jobsErr)
		}
		if len(jobs) == 1 && (jobs[0].Status == "completed" || jobs[0].Status == "failed") {
			job = jobs[0]
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if job.Status != "completed" || job.TrainLoss == nil || job.EvalLoss == nil {
		t.Fatalf("job was not completed with metrics: %#v", job)
	}
	models, err := st.Models(context.Background())
	if err != nil || len(models) != 1 || models[0].Kind != "slm" {
		t.Fatalf("trained SLM registration models=%#v error=%v", models, err)
	}
	chat := request(srv.Handler(), http.MethodPost, "/api/v1/chat", map[string]any{"modelID": models[0].ID, "prompt": "bye now"})
	if chat.Code != http.StatusOK {
		t.Fatalf("SLM chat status %d: %s", chat.Code, chat.Body.String())
	}
	var answer struct{ Content string }
	if err = json.NewDecoder(chat.Body).Decode(&answer); err != nil {
		t.Fatal(err)
	}
	if answer.Content != "Goodbye!" {
		t.Fatalf("unexpected trained SLM answer %q", answer.Content)
	}
}

func TestCloudModelUsesRemoteIDAndServerSideProviderKey(t *testing.T) {
	var received inference.ChatRequest
	var authorization string
	runner := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorization = r.Header.Get("Authorization")
		if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"cloud answer"}}],"usage":{"prompt_tokens":2,"completion_tokens":2}}`))
	}))
	defer runner.Close()
	st, err := store.Open(filepath.Join(t.TempDir(), "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	srv := &Server{Store: st, RAG: rag.Service{Store: st}, Trainer: training.New(st), Inference: &inference.Client{HTTP: runner.Client()}, ProviderAPIKeys: map[string]string{"gemini": "test-gemini-key"}, CORSOrigin: "http://localhost:5173", Logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	created := request(srv.Handler(), http.MethodPost, "/api/v1/models", map[string]any{"name": "Gemini test", "kind": "base", "provider": "gemini", "remoteId": "Gemini 2.5 Pro", "endpoint": runner.URL})
	if created.Code != http.StatusCreated {
		t.Fatalf("model creation status %d: %s", created.Code, created.Body.String())
	}
	var model domain.Model
	if err := json.NewDecoder(created.Body).Decode(&model); err != nil {
		t.Fatal(err)
	}
	if model.Path != "cloud://gemini/Gemini 2.5 Pro" || model.Format != "api" {
		t.Fatalf("unexpected cloud model: %#v", model)
	}
	response := request(srv.Handler(), http.MethodPost, "/api/v1/chat", map[string]any{"modelID": model.ID, "prompt": "hello"})
	if response.Code != http.StatusOK {
		t.Fatalf("cloud chat status %d: %s", response.Code, response.Body.String())
	}
	if received.Model != "gemini-2.5-pro" || authorization != "Bearer test-gemini-key" {
		t.Fatalf("cloud model request model=%q authorization=%q", received.Model, authorization)
	}
}

func TestExternalLocalEndpointCanBeRegisteredAndUsed(t *testing.T) {
	runner := httptest.NewServer(mockrunner.Handler())
	defer runner.Close()
	st, err := store.Open(filepath.Join(t.TempDir(), "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	srv := &Server{Store: st, RAG: rag.Service{Store: st}, Trainer: training.New(st), Inference: &inference.Client{HTTP: runner.Client()}, CORSOrigin: "http://localhost:5173", Logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	registered := request(srv.Handler(), http.MethodPost, "/api/v1/models", map[string]any{"name": "External test model", "endpoint": runner.URL + "/v1", "format": "api", "family": "generic", "kind": "base"})
	if registered.Code != http.StatusCreated {
		t.Fatalf("register endpoint status %d: %s", registered.Code, registered.Body.String())
	}
	var model domain.Model
	if err := json.NewDecoder(registered.Body).Decode(&model); err != nil {
		t.Fatal(err)
	}
	chat := request(srv.Handler(), http.MethodPost, "/api/v1/chat", map[string]any{"modelID": model.ID, "prompt": "endpoint smoke check"})
	if chat.Code != http.StatusOK {
		t.Fatalf("external endpoint chat status %d: %s", chat.Code, chat.Body.String())
	}
}

func TestChatPreservesLocalRunnerContextFailure(t *testing.T) {
	runner := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "Qwen inference failed: prompt has 40960 tokens; local runner context limit is 40959", http.StatusUnprocessableEntity)
	}))
	defer runner.Close()
	st, err := store.Open(filepath.Join(t.TempDir(), "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	srv := &Server{Store: st, RAG: rag.Service{Store: st}, Trainer: training.New(st), Inference: &inference.Client{HTTP: runner.Client(), DefaultEndpoint: runner.URL}, CORSOrigin: "http://localhost:5173", Logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	response := request(srv.Handler(), http.MethodPost, "/api/v1/chat", map[string]any{"prompt": "use web sources"})
	if response.Code != http.StatusUnprocessableEntity {
		t.Fatalf("chat status %d: %s", response.Code, response.Body.String())
	}
	if !strings.Contains(response.Body.String(), "context limit") {
		t.Fatalf("context detail missing: %s", response.Body.String())
	}
}

func TestChatRequestIncludesItsPersistedSessionHistory(t *testing.T) {
	st, err := store.Open(filepath.Join(t.TempDir(), "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	chat := domain.Chat{ID: "chat-history", Title: "History", CreatedAt: store.Now(), UpdatedAt: store.Now()}
	if err = st.AddChat(context.Background(), chat); err != nil {
		t.Fatal(err)
	}
	if err = st.AddMessage(context.Background(), domain.ChatMessage{ID: "m1", ChatID: chat.ID, Role: "user", Content: "First question", CreatedAt: store.Now()}); err != nil {
		t.Fatal(err)
	}
	if err = st.AddMessage(context.Background(), domain.ChatMessage{ID: "m2", ChatID: chat.ID, Role: "assistant", Content: "First answer", CreatedAt: store.Now()}); err != nil {
		t.Fatal(err)
	}
	received := make(chan []inference.Message, 1)
	runner := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload inference.ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		received <- payload.Messages
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"follow-up answer"}}],"usage":{"prompt_tokens":4,"completion_tokens":2}}`))
	}))
	defer runner.Close()
	srv := &Server{Store: st, RAG: rag.Service{Store: st}, Trainer: training.New(st), Inference: &inference.Client{HTTP: runner.Client(), DefaultEndpoint: runner.URL}, CORSOrigin: "http://localhost:5173", Logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	response := request(srv.Handler(), http.MethodPost, "/api/v1/chat", map[string]any{"chatID": chat.ID, "prompt": "Follow-up question"})
	if response.Code != http.StatusOK {
		t.Fatalf("chat status %d: %s", response.Code, response.Body.String())
	}
	messages := <-received
	if len(messages) != 3 || messages[0].Content != "First question" || messages[1].Content != "First answer" || messages[2].Content != "Follow-up question" {
		t.Fatalf("runner received wrong session history: %#v", messages)
	}
}

func TestCreatedChatUsesV4UUID(t *testing.T) {
	st, err := store.Open(filepath.Join(t.TempDir(), "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	srv := &Server{Store: st, RAG: rag.Service{Store: st}, Trainer: training.New(st), Inference: mockInference(t), CORSOrigin: "http://localhost:5173", Logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	response := request(srv.Handler(), http.MethodPost, "/api/v1/chats", map[string]any{})
	if response.Code != http.StatusCreated {
		t.Fatalf("chat create status %d: %s", response.Code, response.Body.String())
	}
	var chat domain.Chat
	if err = json.NewDecoder(response.Body).Decode(&chat); err != nil {
		t.Fatal(err)
	}
	if !regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`).MatchString(chat.ID) {
		t.Fatalf("chat id is not UUID v4: %q", chat.ID)
	}
}

func TestStartedChatCanChangeModelButCannotChangeRAGBinding(t *testing.T) {
	st, err := store.Open(filepath.Join(t.TempDir(), "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	chat := domain.Chat{ID: "locked-chat", Title: "Locked", ModelID: "base-model", CollectionID: "rag-a", CreatedAt: store.Now(), UpdatedAt: store.Now()}
	if err = st.AddChat(context.Background(), chat); err != nil {
		t.Fatal(err)
	}
	if err = st.AddMessage(context.Background(), domain.ChatMessage{ID: "first-user-message", ChatID: chat.ID, Role: "user", Content: "Start this chat", CreatedAt: store.Now()}); err != nil {
		t.Fatal(err)
	}
	srv := &Server{Store: st, RAG: rag.Service{Store: st}, Trainer: training.New(st), Inference: mockInference(t), CORSOrigin: "http://localhost:5173", Logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	response := request(srv.Handler(), http.MethodPatch, "/api/v1/chats/"+chat.ID, map[string]any{"modelId": "other-model"})
	if response.Code != http.StatusOK {
		t.Fatalf("model update status %d: %s", response.Code, response.Body.String())
	}
	response = request(srv.Handler(), http.MethodPatch, "/api/v1/chats/"+chat.ID, map[string]any{"collectionId": "rag-b"})
	if response.Code != http.StatusConflict {
		t.Fatalf("RAG update status %d: %s", response.Code, response.Body.String())
	}
	boundBeforeMessage := domain.Chat{ID: "bound-before-message", Title: "Bound", CollectionID: "rag-a", CreatedAt: store.Now(), UpdatedAt: store.Now()}
	if err = st.AddChat(context.Background(), boundBeforeMessage); err != nil {
		t.Fatal(err)
	}
	response = request(srv.Handler(), http.MethodPatch, "/api/v1/chats/"+boundBeforeMessage.ID, map[string]any{"collectionId": "rag-b"})
	if response.Code != http.StatusConflict {
		t.Fatalf("pre-message RAG update status %d: %s", response.Code, response.Body.String())
	}
	response = request(srv.Handler(), http.MethodPatch, "/api/v1/chats/"+boundBeforeMessage.ID, map[string]any{"ragEnabled": false})
	if response.Code != http.StatusOK {
		t.Fatalf("RAG behavior update status %d: %s", response.Code, response.Body.String())
	}
}

func TestNewChatsUseTheWorkspaceDefaultBaseModel(t *testing.T) {
	st, err := store.Open(filepath.Join(t.TempDir(), "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	srv := &Server{Store: st, RAG: rag.Service{Store: st}, Trainer: training.New(st), Inference: mockInference(t), CORSOrigin: "http://localhost:5173", Logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	firstResponse := request(srv.Handler(), http.MethodPost, "/api/v1/models", map[string]any{"name": "First base", "path": "C:/models/first", "kind": "base"})
	if firstResponse.Code != http.StatusCreated {
		t.Fatalf("first model status %d: %s", firstResponse.Code, firstResponse.Body.String())
	}
	var first domain.Model
	if err = json.NewDecoder(firstResponse.Body).Decode(&first); err != nil || !first.Default {
		t.Fatalf("first=%#v err=%v", first, err)
	}
	secondResponse := request(srv.Handler(), http.MethodPost, "/api/v1/models", map[string]any{"name": "Second base", "path": "C:/models/second", "kind": "base"})
	var second domain.Model
	if secondResponse.Code != http.StatusCreated || json.NewDecoder(secondResponse.Body).Decode(&second) != nil || second.Default {
		t.Fatalf("second model status %d: %s", secondResponse.Code, secondResponse.Body.String())
	}
	changed := request(srv.Handler(), http.MethodPost, "/api/v1/models/"+second.ID+"/default", map[string]any{})
	if changed.Code != http.StatusOK {
		t.Fatalf("default update status %d: %s", changed.Code, changed.Body.String())
	}
	created := request(srv.Handler(), http.MethodPost, "/api/v1/chats", map[string]any{})
	if created.Code != http.StatusCreated {
		t.Fatalf("chat create status %d: %s", created.Code, created.Body.String())
	}
	var chat domain.Chat
	if err = json.NewDecoder(created.Body).Decode(&chat); err != nil || chat.ModelID != second.ID {
		t.Fatalf("chat=%#v err=%v", chat, err)
	}
}

func TestModelCatalogDiscoversDownloadedCheckpointDirectories(t *testing.T) {
	root := t.TempDir()
	checkpoint := filepath.Join(root, "Qwen3-0.6B")
	if err := os.MkdirAll(checkpoint, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(checkpoint, "config.json"), []byte(`{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}`), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(checkpoint, "model.safetensors"), nil, 0600); err != nil {
		t.Fatal(err)
	}
	st, err := store.Open(filepath.Join(root, "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	if _, err = st.AddModel(context.Background(), domain.Model{ID: "qwen", Name: "Qwen3 0.6B", Path: checkpoint, Family: "qwen", Kind: "base", CreatedAt: store.Now()}); err != nil {
		t.Fatal(err)
	}
	srv := &Server{Store: st, RAG: rag.Service{Store: st}, Trainer: training.New(st), Inference: mockInference(t), CORSOrigin: "http://localhost:5173", Logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	response := request(srv.Handler(), http.MethodGet, "/api/v1/models/catalog?directory="+url.QueryEscape(root), nil)
	if response.Code != http.StatusOK {
		t.Fatalf("catalog status %d: %s", response.Code, response.Body.String())
	}
	var entries []localModelCatalogEntry
	if err = json.NewDecoder(response.Body).Decode(&entries); err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Path != checkpoint || entries[0].Family != "qwen" || !entries[0].Registered {
		t.Fatalf("unexpected catalog entries: %#v", entries)
	}
}

func TestModelCatalogDiscoversGGUFAndONNXFiles(t *testing.T) {
	root := t.TempDir()
	gguf := filepath.Join(root, "small-qwen.gguf")
	onnx := filepath.Join(root, "encoder.onnx")
	for _, path := range []string{gguf, onnx} {
		if err := os.WriteFile(path, []byte("model"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	entries, err := discoverLocalModels(root, []domain.Model{{Path: gguf}})
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Fatalf("catalog entries = %#v", entries)
	}
	byPath := map[string]localModelCatalogEntry{}
	for _, entry := range entries {
		byPath[entry.Path] = entry
	}
	if entry := byPath[gguf]; entry.Format != "gguf" || !entry.Registered {
		t.Fatalf("GGUF entry = %#v", entry)
	}
	if entry := byPath[onnx]; entry.Format != "onnx" || entry.Registered {
		t.Fatalf("ONNX entry = %#v", entry)
	}
}

func request(h http.Handler, method, path string, body any) *httptest.ResponseRecorder {
	b, _ := json.Marshal(body)
	r := httptest.NewRequest(method, path, bytes.NewReader(b))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}

func mockInference(t *testing.T) *inference.Client {
	t.Helper()
	runner := httptest.NewServer(mockrunner.Handler())
	t.Cleanup(runner.Close)
	return &inference.Client{HTTP: runner.Client(), DefaultEndpoint: runner.URL + "/v1"}
}
