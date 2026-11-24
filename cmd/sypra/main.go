package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"github.com/sypra-labs/sypra-ai/internal/adapter"
	"github.com/sypra-labs/sypra-ai/internal/agent"
	"github.com/sypra-labs/sypra-ai/internal/api"
	"github.com/sypra-labs/sypra-ai/internal/chattemplate"
	"github.com/sypra-labs/sypra-ai/internal/clisession"
	"github.com/sypra-labs/sypra-ai/internal/config"
	"github.com/sypra-labs/sypra-ai/internal/domain"
	"github.com/sypra-labs/sypra-ai/internal/finetune"
	"github.com/sypra-labs/sypra-ai/internal/inference"
	"github.com/sypra-labs/sypra-ai/internal/media"
	"github.com/sypra-labs/sypra-ai/internal/observability"
	"github.com/sypra-labs/sypra-ai/internal/qwen"
	"github.com/sypra-labs/sypra-ai/internal/rag"
	"github.com/sypra-labs/sypra-ai/internal/runnermanager"
	"github.com/sypra-labs/sypra-ai/internal/store"
	"github.com/sypra-labs/sypra-ai/internal/training"
	"github.com/sypra-labs/sypra-ai/internal/websearch"
	"github.com/sypra-labs/sypra-ai/internal/webui"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// version is injected by the release build. Development builds retain dev.
var version = "dev"

func main() {
	if len(os.Args) < 2 {
		usage()
		return
	}
	cfg := config.Load()
	st, e := store.OpenConfigured(context.Background(), cfg.DatabaseURL, cfg.DataPath)
	if e != nil {
		fatal(e)
	}
	defer st.Close()
	switch os.Args[1] {
	case "version", "--version", "-version":
		fmt.Println(version)
	case "serve":
		fatalIf(serve(cfg, st, false))
	case "start":
		fatalIf(serve(cfg, st, true))
	case "models":
		models(st, cfg)
	case "rag":
		ragCmd(st)
	case "train":
		train(st, cfg)
	case "train-worker":
		trainWorker()
	case "metrics":
		v, err := st.Summary(context.Background())
		fatalIf(err)
		print(v)
	case "chats":
		chats(st)
	case "chat":
		chat(st)
	case "web":
		web(cfg)
	case "reset":
		resetWorkspace(st, cfg)
	case "backup":
		backupWorkspace(st)
	case "restore":
		restoreWorkspace(st)
	case "runner":
		runner(st)
	default:
		usage()
	}
}

func chats(st *store.Store) {
	if len(os.Args) < 3 || os.Args[2] != "cancel" {
		fmt.Println("sypra chats cancel --chat <chat UUID>")
		return
	}
	fs := flag.NewFlagSet("chats cancel", flag.ExitOnError)
	chatID := fs.String("chat", "", "chat UUID")
	fs.Parse(os.Args[3:])
	if *chatID == "" {
		fatal(fmt.Errorf("--chat is required"))
	}
	messages, err := st.Messages(context.Background(), *chatID)
	fatalIf(err)
	cancelled := 0
	for _, message := range messages {
		if message.Role != "assistant" || message.Status != "working" {
			continue
		}
		message.Status, message.Content = "cancelled", "Generation cancelled."
		fatalIf(st.UpdateMessage(context.Background(), message))
		cancelled++
	}
	print(map[string]any{"chatID": *chatID, "cancelled": cancelled})
}

// chat sends one terminal turn and persists its history under .sypra/chats in
// the current directory. A session ID is deliberately scoped to that folder.
func chat(st *store.Store) {
	fs := flag.NewFlagSet("chat", flag.ExitOnError)
	prompt := fs.String("prompt", "", "message to send")
	sessionID := fs.String("session", "", "session ID from this folder")
	endpoint := fs.String("endpoint", "http://127.0.0.1:8081/v1", "local runner endpoint")
	model := fs.String("model", "", "model name or ID")
	adapter := fs.String("adapter", "", "optional local Sypra adapter artifact")
	maxTokens := fs.Int("max-tokens", 0, "maximum response tokens; 0 uses the runner default")
	list := fs.Bool("list", false, "list sessions created in this folder")
	closeSession := fs.Bool("close", false, "close a folder-bound CLI session")
	fs.Parse(os.Args[2:])
	workspace, err := os.Getwd()
	fatalIf(err)
	if *list {
		sessions, listErr := clisession.List(workspace)
		fatalIf(listErr)
		print(sessions)
		return
	}
	if *closeSession {
		if *sessionID == "" {
			fatal(fmt.Errorf("--close requires --session"))
		}
		session, loadErr := clisession.Load(workspace, *sessionID)
		fatalIf(loadErr)
		chatRecord, chatErr := st.Chat(context.Background(), session.ChatID)
		fatalIf(chatErr)
		chatRecord.CLISessionActive, chatRecord.UpdatedAt = false, store.Now()
		fatalIf(st.UpdateChat(context.Background(), chatRecord))
		fatalIf(clisession.Close(workspace, *sessionID))
		fmt.Println("CLI chat session closed:", *sessionID)
		return
	}
	if fs.NArg() > 0 || (*prompt == "" && *sessionID == "") {
		fmt.Println("sypra chat --prompt <message> [--session ID] [--endpoint URL] [--model NAME] [--adapter PATH]\nsypra chat --session ID --prompt <message>\nsypra chat --list")
		return
	}
	var session clisession.Session
	var chatRecord domain.Chat
	if *sessionID != "" {
		session, err = clisession.Load(workspace, *sessionID)
		fatalIf(err)
		chatRecord, err = st.Chat(context.Background(), session.ChatID)
		fatalIf(err)
		if !chatRecord.CLISessionActive {
			fatal(fmt.Errorf("CLI chat session %s is closed", *sessionID))
		}
	} else {
		var selected domain.Model
		if *model == "" {
			defaultModel, ok, defaultErr := st.DefaultBaseModel(context.Background())
			fatalIf(defaultErr)
			if !ok {
				fatal(fmt.Errorf("no default base model is registered; pass --model"))
			}
			selected, *model = defaultModel, defaultModel.Name
		} else {
			registered, modelErr := st.Models(context.Background())
			fatalIf(modelErr)
			for _, candidate := range registered {
				if candidate.ID == *model || candidate.Name == *model {
					selected = candidate
					*model = candidate.Name
					break
				}
			}
		}
		session, err = clisession.New(workspace, *endpoint, *model, *adapter)
		fatalIf(err)
		chatRecord = domain.Chat{ID: newID(), Title: shortCLITitle(*prompt), ModelID: selected.ID, WorkspacePath: session.Workspace, CLISessionActive: true, CreatedAt: store.Now(), UpdatedAt: store.Now()}
		fatalIf(st.AddChat(context.Background(), chatRecord))
		session.ID, session.ChatID = chatRecord.ID, chatRecord.ID
		fatalIf(clisession.Save(&session))
	}
	if *prompt == "" {
		fatal(fmt.Errorf("--prompt is required"))
	}
	user := domain.ChatMessage{ID: newID(), ChatID: chatRecord.ID, Role: "user", Content: *prompt, CreatedAt: store.Now()}
	fatalIf(st.AddMessage(context.Background(), user))
	history, historyErr := st.Messages(context.Background(), chatRecord.ID)
	fatalIf(historyErr)
	messages := make([]inference.Message, 0, len(history))
	for _, message := range history {
		messages = append(messages, inference.Message{Role: message.Role, Content: message.Content, Parts: message.Parts})
	}
	client := &inference.Client{HTTP: &http.Client{}}
	response, err := client.ChatStream(context.Background(), session.Endpoint, inference.ChatRequest{Model: session.Model, Messages: messages, MaxTokens: *maxTokens, Adapter: session.Adapter}, func(delta string) { fmt.Print(delta) })
	if err != nil {
		fmt.Println()
		fatal(err)
	}
	if response.Content != "" {
		fmt.Println()
	}
	fatalIf(st.AddMessage(context.Background(), domain.ChatMessage{ID: newID(), ChatID: chatRecord.ID, Role: "assistant", Content: response.Content, Parts: response.Parts, Status: "completed", CreatedAt: store.Now()}))
	chatRecord.UpdatedAt = store.Now()
	fatalIf(st.UpdateChat(context.Background(), chatRecord))
	fmt.Fprintln(os.Stderr, "CLI chat session:", session.ID, "(folder:", session.Workspace+")")
}

func shortCLITitle(prompt string) string {
	words := strings.Fields(prompt)
	if len(words) == 0 {
		return "CLI chat"
	}
	title := strings.Join(words, " ")
	if len([]rune(title)) > 48 {
		return string([]rune(title)[:48]) + "…"
	}
	return title
}
func web(cfg config.Config) {
	if len(os.Args) < 3 || os.Args[2] != "search" {
		fmt.Println("sypra web search --query <query> [--full]")
		return
	}
	fs := flag.NewFlagSet("web search", flag.ExitOnError)
	query := fs.String("query", "", "internet search query")
	full := fs.Bool("full", false, "print the full retrieved text for every source")
	fs.Parse(os.Args[3:])
	if strings.TrimSpace(*query) == "" {
		fatal(fmt.Errorf("--query is required"))
	}
	results, err := websearch.New(cfg.WebSearchEndpoint).Search(context.Background(), *query)
	fatalIf(err)
	if *full {
		print(results)
		return
	}
	type summary struct {
		Title          string `json:"title"`
		URL            string `json:"url"`
		Snippet        string `json:"snippet,omitempty"`
		RetrievedChars int    `json:"retrievedChars"`
	}
	output := make([]summary, 0, len(results))
	for _, result := range results {
		output = append(output, summary{Title: result.Title, URL: result.URL, Snippet: result.Snippet, RetrievedChars: len([]rune(result.Content))})
	}
	print(output)
}
func resetWorkspace(st *store.Store, cfg config.Config) {
	if err := st.ResetWorkspaceData(context.Background()); err != nil {
		fatal(err)
	}
	root := filepath.Dir(cfg.DataPath)
	for _, name := range []string{"documents", "datasets", "runs"} {
		path := filepath.Join(root, name)
		if err := os.RemoveAll(path); err != nil {
			fatal(fmt.Errorf("clear %s: %w", path, err))
		}
	}
	fmt.Println("Cleared chats, projects, RAG data, cloud models, fine-tuning data, jobs, and usage. Downloaded local base models were preserved.")
}

func runner(st *store.Store) {
	fs := flag.NewFlagSet("runner", flag.ExitOnError)
	modelPath := fs.String("model", "", "Qwen SafeTensors model directory; defaults to the first base model")
	listen := fs.String("listen", "127.0.0.1:8081", "listen address")
	contextSize := fs.Int("context", 0, "maximum context tokens; 0 uses the model's full supported context")
	maxTokens := fs.Int("max-tokens", 0, "completion tokens; 0 uses all remaining model context")
	parallel := fs.Int("parallel", 10, "maximum concurrent local chat generations")
	backend := fs.String("backend", "auto", "matrix backend: auto, cpu, cuda, or metal")
	fs.Parse(os.Args[2:])
	modelID := "qwen3-0.6b"
	if *modelPath == "" {
		models, err := st.Models(context.Background())
		fatalIf(err)
		for _, model := range models {
			if model.Kind == "base" {
				*modelPath, modelID = model.Path, model.Name
				break
			}
		}
	}
	if *modelPath == "" {
		fatal(fmt.Errorf("no base model is registered; pass --model or add one with `sypra models add`"))
	}
	model, err := qwen.LoadModel(*modelPath)
	fatalIf(err)
	defer model.Close()
	if *backend != "cpu" {
		useMetal := *backend == "metal" || (*backend == "auto" && runtime.GOOS == "darwin")
		if useMetal {
			executor, metalErr := qwen.NewMetalExecutor()
			if metalErr != nil {
				if *backend == "metal" {
					fatal(fmt.Errorf("initialize Metal backend: %w", metalErr))
				}
				fmt.Fprintln(os.Stderr, "Metal unavailable; using CPU matrix backend:", metalErr)
			} else {
				model.SetMatrixExecutor(executor)
				fmt.Fprintln(os.Stderr, "Using Metal matrix backend for frozen Qwen projections.")
			}
		} else {
			executor, cudaErr := qwen.NewCUDAExecutor()
			if cudaErr != nil {
				if *backend == "cuda" {
					fatal(fmt.Errorf("initialize CUDA backend: %w", cudaErr))
				}
				fmt.Fprintln(os.Stderr, "CUDA unavailable; using CPU matrix backend:", cudaErr)
			} else {
				model.SetMatrixExecutor(executor)
				fmt.Fprintln(os.Stderr, "Using CUDA matrix backend for frozen Qwen projections.")
			}
		}
	}
	local := qwen.NewRunner(model, modelID)
	if *contextSize > 0 {
		if *contextSize > model.Config.MaxPositionEmbeddings {
			fatal(fmt.Errorf("--context cannot exceed this model's %d-token context window", model.Config.MaxPositionEmbeddings))
		}
		local.MaxContext = *contextSize
	}
	if *maxTokens > 0 {
		local.DefaultMax = *maxTokens
	}
	if *parallel < 1 || *parallel > 10 {
		fatal(fmt.Errorf("--parallel must be between 1 and 10"))
	}
	local.Parallel = *parallel
	if local.MaxContext < 8 || local.DefaultMax < 1 {
		fatal(fmt.Errorf("--context must be at least 8 and --max-tokens cannot be negative"))
	}
	fmt.Printf("Sypra Qwen CPU runner listening at http://%s/v1 (model: %s)\n", *listen, modelID)
	fatal(http.ListenAndServe(*listen, local.Handler()))
}
func serve(c config.Config, st *store.Store, openBrowser bool) error {
	logger, logCloser, err := observability.NewLogger(observability.Config{Format: c.LogFormat, File: c.LogFile, MaxBytes: c.LogMaxBytes, Backups: c.LogBackups})
	if err != nil {
		return fmt.Errorf("configure structured logger: %w", err)
	}
	defer logCloser.Close()
	telemetry, err := observability.NewTelemetry(observability.OTLPConfig{Endpoint: c.OTLPEndpoint, ServiceName: c.OTLPServiceName, Logger: logger})
	if err != nil {
		return fmt.Errorf("configure OTLP telemetry: %w", err)
	}
	if telemetry != nil {
		defer func() {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if err := telemetry.Shutdown(ctx); err != nil {
				logger.Warn("flush OTLP telemetry", "error", err)
			}
		}()
	}
	runners := runnermanager.New("", logger, runnermanager.Options{
		MaxActive:   c.MaxActiveRunners,
		GPUBudgetMB: c.RunnerGPUBudgetMB,
		StatePath:   filepath.Join(filepath.Dir(c.DataPath), "runners.json"),
	})
	defer func() {
		if err := runners.Close(); err != nil {
			logger.Error("stop managed model runners", "error", err)
		}
	}()
	ragService := rag.Service{Store: st}
	mediaRoot := filepath.Join(filepath.Dir(c.DataPath), "media")
	mediaService := media.New(st, filepath.Join(mediaRoot, "outputs"))
	webService := websearch.New(c.WebSearchEndpoint)
	agentService := agent.New(st, webService, parseStringList(c.AgentTerminalAllowlist), agent.Options{
		MCPAllowlist:   parseStringList(c.AgentMCPAllowlist),
		WorkRoot:       c.AgentWorkDir,
		Timeout:        time.Duration(c.AgentTimeoutSeconds) * time.Second,
		MaxOutputBytes: c.AgentMaxOutputBytes,
	})
	trainer := training.New(st, c.MaxConcurrentJobs)
	if recovered, recoverErr := trainer.Recover(context.Background()); recoverErr != nil {
		logger.Error("recover interrupted training jobs", "error", recoverErr)
	} else if recovered > 0 {
		logger.Warn("marked interrupted training jobs after restart", "count", recovered)
	}
	server := &api.Server{Store: st, RAG: ragService, Trainer: trainer, Inference: &inference.Client{HTTP: &http.Client{}, DefaultEndpoint: c.ModelEndpoint, APIKey: c.ModelAPIKey}, ProviderAPIKeys: map[string]string{"openai": c.OpenAIAPIKey, "xai": c.XAIAPIKey, "gemini": c.GeminiAPIKey}, CORSOrigin: c.CORSOrigin, Logger: logger, UploadDir: filepath.Join(filepath.Dir(c.DataPath), "models"), Web: webService, Media: mediaService, MediaRoot: mediaRoot, Agent: agentService, Runners: runners, Metrics: observability.NewMetrics(), Telemetry: telemetry}
	apiHandler := server.Handler()
	ui, hasUI := webui.Handler()
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !hasUI || strings.HasPrefix(r.URL.Path, "/api/") || r.URL.Path == "/healthz" || r.URL.Path == "/readyz" || r.URL.Path == "/metrics" {
			apiHandler.ServeHTTP(w, r)
			return
		}
		ui.ServeHTTP(w, r)
	})
	httpServer := &http.Server{Addr: c.ListenAddr, Handler: handler}
	shutdown, stopSignals := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stopSignals()
	go func() {
		ticker := time.NewTicker(time.Minute)
		defer ticker.Stop()
		for {
			select {
			case <-shutdown.Done():
				return
			case <-ticker.C:
				for _, err := range ragService.RefreshDue(context.Background()) {
					logger.Error("scheduled RAG URL refresh failed", "error", err)
				}
			}
		}
	}()
	go func() {
		<-shutdown.Done()
		logger.Info("Sypra API shutdown requested; stopping managed model runners")
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := httpServer.Shutdown(ctx); err != nil {
			logger.Error("gracefully stop Sypra API", "error", err)
		}
	}()
	logger.Info("Sypra API listening", "address", c.ListenAddr)
	if openBrowser && hasUI {
		go openLocalBrowser(c.ListenAddr)
	}
	err = httpServer.ListenAndServe()
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}

func openLocalBrowser(address string) {
	port := "8080"
	if _, value, found := strings.Cut(address, ":"); found && value != "" {
		port = value
	}
	url := "http://127.0.0.1:" + port
	time.Sleep(400 * time.Millisecond)
	var command *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		command = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	case "darwin":
		command = exec.Command("open", url)
	default:
		command = exec.Command("xdg-open", url)
	}
	_ = command.Start()
}
func models(st *store.Store, cfg config.Config) {
	if len(os.Args) < 3 {
		fmt.Println("sypra models [list|add|import-peft|promote]")
		return
	}
	switch os.Args[2] {
	case "list":
		v, e := st.Models(context.Background())
		fatalIf(e)
		print(v)
	case "add":
		fs := flag.NewFlagSet("models add", flag.ExitOnError)
		name := fs.String("name", "", "display name")
		path := fs.String("path", "", "local model path")
		format := fs.String("format", "gguf", "model format")
		endpoint := fs.String("endpoint", "", "local runner endpoint")
		kind := fs.String("kind", "base", "base|slm")
		family := fs.String("family", "", "model family: qwen, llama, mistral, gemma, phi, or generic")
		modalities := fs.String("modalities", "text", "comma-separated capabilities: text,image,audio,video,file")
		fs.Parse(os.Args[3:])
		if *name == "" || *path == "" {
			fatal(fmt.Errorf("--name and --path are required"))
		}
		resolvedFamily := chattemplate.NormalizeFamily(*family)
		if *family == "" {
			resolvedFamily = chattemplate.NormalizeFamily(*name + " " + *path)
		}
		m := domain.Model{ID: newID(), Name: *name, Path: *path, Format: *format, Endpoint: *endpoint, Family: resolvedFamily, Modalities: parseModalities(*modalities), Kind: *kind, CreatedAt: store.Now()}
		stored, err := st.AddModel(context.Background(), m)
		fatalIf(err)
		print(stored)
	case "import-peft":
		fs := flag.NewFlagSet("models import-peft", flag.ExitOnError)
		name := fs.String("name", "", "display name")
		path := fs.String("path", "", "directory containing adapter_config.json and adapter_model.safetensors")
		parentModel := fs.String("parent-model", "", "registered base model ID; inferred from adapter metadata when omitted")
		method := fs.String("method", "lora", "adapter method label: lora or qlora")
		version := fs.String("version", "", "optional adapter version label")
		fs.Parse(os.Args[3:])
		if *name == "" || *path == "" {
			fatal(fmt.Errorf("--name and --path are required"))
		}
		artifact, err := adapter.ImportPEFT(*path)
		fatalIf(err)
		registered, err := st.Models(context.Background())
		fatalIf(err)
		parent, err := resolveAdapterParent(registered, *parentModel, artifact.BaseModel)
		fatalIf(err)
		modelID := newID()
		target := filepath.Join(filepath.Dir(cfg.DataPath), "adapters", modelID, "sypra-adapter.json")
		fatalIf(adapter.SaveTransformerLoRA(target, artifact))
		if *version == "" {
			*version = filepath.Base(filepath.Clean(*path))
		}
		m := domain.Model{ID: modelID, Name: *name, Path: target, Format: adapter.TransformerLoRAType, Family: "qwen", Modalities: []string{"text"}, Kind: "slm", ParentModel: parent.ID, Method: strings.ToLower(strings.TrimSpace(*method)), Version: *version, Status: "candidate", CreatedAt: store.Now()}
		stored, err := st.AddModel(context.Background(), m)
		fatalIf(err)
		print(stored)
	case "promote":
		fs := flag.NewFlagSet("models promote", flag.ExitOnError)
		modelID := fs.String("id", "", "SLM model ID")
		fs.Parse(os.Args[3:])
		if *modelID == "" {
			fatal(fmt.Errorf("--id is required"))
		}
		model, err := st.PromoteModel(context.Background(), *modelID)
		fatalIf(err)
		print(model)
	default:
		fmt.Println("sypra models [list|add|import-peft|promote]")
	}
}

func resolveAdapterParent(models []domain.Model, requestedID, baseReference string) (domain.Model, error) {
	if requestedID != "" {
		for _, model := range models {
			if model.ID == requestedID && model.Kind == "base" {
				return model, nil
			}
		}
		return domain.Model{}, fmt.Errorf("registered base model %q was not found", requestedID)
	}
	normalize := func(value string) string {
		return strings.Map(func(r rune) rune {
			if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
				return r
			}
			return -1
		}, strings.ToLower(value))
	}
	reference := normalize(baseReference)
	var matches []domain.Model
	for _, model := range models {
		if model.Kind != "base" {
			continue
		}
		candidate := normalize(model.Name + " " + model.Path)
		if strings.Contains(candidate, reference) || strings.Contains(reference, normalize(model.Name)) {
			matches = append(matches, model)
		}
	}
	if len(matches) == 1 {
		return matches[0], nil
	}
	if len(matches) == 0 {
		return domain.Model{}, fmt.Errorf("no registered base model matches adapter base %q; pass --parent-model", baseReference)
	}
	return domain.Model{}, fmt.Errorf("multiple registered base models match adapter base %q; pass --parent-model", baseReference)
}

func backupWorkspace(st *store.Store) {
	fs := flag.NewFlagSet("backup", flag.ExitOnError)
	output := fs.String("output", "", "destination .sypra-backup archive")
	fs.Parse(os.Args[2:])
	if *output == "" {
		fatal(fmt.Errorf("backup requires --output <path>"))
	}
	fatalIf(st.BackupTo(context.Background(), *output))
	fmt.Println("workspace backup created:", *output)
}

func restoreWorkspace(st *store.Store) {
	fs := flag.NewFlagSet("restore", flag.ExitOnError)
	input := fs.String("input", "", "source .sypra-backup archive")
	confirm := fs.Bool("confirm", false, "confirm replacement of the current workspace")
	fs.Parse(os.Args[2:])
	if *input == "" || !*confirm {
		fatal(fmt.Errorf("restore requires --input <backup> --confirm"))
	}
	fatalIf(st.RestoreFrom(context.Background(), *input))
	fmt.Println("workspace restored from:", *input)
}
func ragCmd(st *store.Store) {
	if len(os.Args) < 3 {
		fmt.Println("sypra rag [list|create|documents|query|reindex|remove|refresh|evaluate]")
		return
	}
	svc := rag.Service{Store: st}
	switch os.Args[2] {
	case "list":
		v, e := st.Collections(context.Background())
		fatalIf(e)
		print(v)
	case "create":
		fs := flag.NewFlagSet("rag create", flag.ExitOnError)
		name := fs.String("name", "", "collection name")
		file := fs.String("file", "", "RAG source file (DOCX or supported text format)")
		url := fs.String("url", "", "web page to ingest")
		refresh := fs.Int("refresh-minutes", 0, "URL refresh cadence; 0 disables scheduled refresh")
		tags := fs.String("tags", "", "comma-separated routing tags")
		chunk := fs.Int("chunk-size", 192, "embedding tokens per chunk")
		fs.Parse(os.Args[3:])
		if *file == "" && *url == "" {
			fatal(fmt.Errorf("--file or --url is required"))
		}
		c, e := svc.Create(context.Background(), rag.CreateInput{Name: *name, ChunkSize: *chunk, RoutingTags: parseStringList(*tags), Documents: []rag.Document{{Path: *file, URL: *url, RefreshIntervalMinutes: *refresh}}})
		fatalIf(e)
		print(c)
	case "documents":
		fs := flag.NewFlagSet("rag documents", flag.ExitOnError)
		collection := fs.String("collection", "", "collection ID")
		fs.Parse(os.Args[3:])
		v, e := st.Documents(context.Background(), *collection)
		fatalIf(e)
		print(v)
	case "query":
		fs := flag.NewFlagSet("rag query", flag.ExitOnError)
		collection := fs.String("collection", "", "collection ID")
		q := fs.String("query", "", "search text")
		limit := fs.Int("limit", 5, "number of results; 0 returns all")
		inspect := fs.Bool("inspect", false, "print scores, neighboring chunks and document provenance")
		rerank := fs.Bool("rerank", false, "apply local query-coverage reranking")
		metadata := fs.String("metadata", "", "comma-separated exact filters, e.g. department=legal")
		fs.Parse(os.Args[3:])
		if *inspect {
			v, e := svc.RetrieveDetailed(context.Background(), *collection, *q, rag.RetrievalOptions{Limit: *limit, Metadata: metadataFilters(*metadata), Rerank: *rerank, IncludeNeighbors: true})
			fatalIf(e)
			print(v)
			return
		}
		v, e := svc.Retrieve(context.Background(), *collection, *q, *limit)
		fatalIf(e)
		print(v)
	case "reindex":
		fs := flag.NewFlagSet("rag reindex", flag.ExitOnError)
		document := fs.String("document", "", "document ID")
		fs.Parse(os.Args[3:])
		v, e := svc.ReindexDocument(context.Background(), *document, nil)
		fatalIf(e)
		print(v)
	case "remove":
		fs := flag.NewFlagSet("rag remove", flag.ExitOnError)
		document := fs.String("document", "", "document ID")
		fs.Parse(os.Args[3:])
		fatalIf(st.DeleteDocument(context.Background(), *document))
		print(map[string]string{"removed": *document})
	case "refresh":
		problems := svc.RefreshDue(context.Background())
		if len(problems) > 0 {
			fatal(problems[0])
		}
		print(map[string]string{"status": "scheduled URL refreshes completed"})
	case "evaluate":
		fs := flag.NewFlagSet("rag evaluate", flag.ExitOnError)
		set := fs.String("set", "", "evaluation set ID")
		limit := fs.Int("limit", 5, "retrieval depth")
		rerank := fs.Bool("rerank", false, "apply local reranking")
		fs.Parse(os.Args[3:])
		v, e := svc.Evaluate(context.Background(), *set, *limit, rag.RetrievalOptions{Rerank: *rerank})
		fatalIf(e)
		print(v)
	default:
		fmt.Println("sypra rag [list|create|documents|query|reindex|remove|refresh|evaluate]")
	}
}

func metadataFilters(value string) map[string]string {
	filters := map[string]string{}
	for _, pair := range parseStringList(value) {
		parts := strings.SplitN(pair, "=", 2)
		if len(parts) == 2 && strings.TrimSpace(parts[0]) != "" {
			filters[strings.TrimSpace(parts[0])] = strings.TrimSpace(parts[1])
		}
	}
	return filters
}
func train(st *store.Store, cfg config.Config) {
	if len(os.Args) < 3 {
		fmt.Println("sypra train [list|start]")
		return
	}
	if os.Args[2] == "list" {
		v, e := st.Jobs(context.Background())
		fatalIf(e)
		print(v)
		return
	}
	if os.Args[2] != "start" {
		fmt.Println("sypra train [list|start]")
		return
	}
	fs := flag.NewFlagSet("train start", flag.ExitOnError)
	name := fs.String("name", "", "job name")
	model := fs.String("model", "", "base model location")
	data := fs.String("data", "", "dataset path")
	method := fs.String("method", "lora", "lora|qlora|full")
	command := fs.String("command", "", "optional external training shell command")
	epochs := fs.Int("epochs", 3, "native trainer epochs")
	batchSize := fs.Int("batch-size", 1, "native trainer batch size")
	learningRate := fs.Float64("learning-rate", .0002, "native trainer learning rate")
	loraRank := fs.Int("lora-rank", 16, "Qwen LoRA rank")
	loraAlpha := fs.Int("lora-alpha", 32, "Qwen LoRA alpha")
	seed := fs.Uint64("seed", 1, "deterministic data split and run seed")
	checkpointEvery := fs.Int("checkpoint-every", 1, "save an adapter checkpoint every N epochs")
	earlyStopping := fs.Int("early-stopping-patience", 0, "stop after N non-improving validation epochs; 0 disables")
	minImprovement := fs.Float64("min-validation-improvement", 0, "minimum validation-loss reduction to count as improvement")
	resume := fs.String("resume-from", "", "checkpoint directory or adapter path to resume")
	targets := fs.String("targets", "all", "comma-separated LoRA targets: all, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head")
	weightDecay := fs.Float64("weight-decay", .01, "AdamW weight decay")
	maxGradNorm := fs.Float64("max-grad-norm", 1, "global gradient clipping norm; 0 disables")
	maxSequence := fs.Int("max-seq-length", 0, "maximum training sequence tokens; 0 uses model context")
	backend := fs.String("backend", "auto", "matrix backend: auto, cpu, or cuda")
	gpuDevices := fs.String("gpu-devices", "", "comma-separated local CUDA device IDs for data-parallel workers")
	workers := fs.String("workers", "", "comma-separated remote gradient worker endpoints")
	fs.Parse(os.Args[3:])
	if *name == "" || *model == "" || *data == "" {
		fatal(fmt.Errorf("--name, --model, and --data are required"))
	}
	j := domain.TrainingJob{ID: newID(), Name: *name, BaseModel: *model, DatasetPath: *data, Method: *method, Parameters: map[string]any{}, Status: "queued", CreatedAt: store.Now()}
	service := training.New(st)
	if *command != "" {
		fatalIf(service.Start(context.Background(), j, *command))
	} else {
		native := finetune.DefaultConfig()
		native.Method, native.ModelDir = *method, *model
		native.Epochs, native.BatchSize, native.EvalBatchSize, native.LearningRate = *epochs, *batchSize, *batchSize, *learningRate
		native.LoRARank, native.LoRAAlpha = *loraRank, *loraAlpha
		native.Seed, native.CheckpointEvery, native.EarlyStoppingPatience = *seed, *checkpointEvery, *earlyStopping
		native.MinValidationImprovement, native.ResumeFrom = *minImprovement, *resume
		native.WeightDecay, native.MaxGradientNorm, native.MaxSequenceLength = *weightDecay, *maxGradNorm, *maxSequence
		native.ComputeBackend = *backend
		native.GPUDevices = parseIntList(*gpuDevices)
		native.WorkerEndpoints = parseStringList(*workers)
		if strings.TrimSpace(*targets) != "" && strings.TrimSpace(*targets) != "all" {
			native.TargetModules = strings.Split(*targets, ",")
		}
		directory := filepath.Join(filepath.Dir(cfg.DataPath), "runs", j.ID)
		fatalIf(service.StartNative(context.Background(), j, native, directory))
	}
	completed, err := service.Wait(context.Background(), j.ID)
	fatalIf(err)
	print(completed)
}

func trainWorker() {
	fs := flag.NewFlagSet("train-worker", flag.ExitOnError)
	model := fs.String("model", "", "local Qwen model directory")
	method := fs.String("method", "qlora", "lora or qlora")
	backend := fs.String("backend", "auto", "matrix backend: auto, cpu, or cuda")
	device := fs.Int("device", -1, "CUDA device ID; -1 uses CPU/automatic backend")
	listen := fs.String("listen", "0.0.0.0:8091", "worker listen address")
	fs.Parse(os.Args[2:])
	if *model == "" {
		fatal(fmt.Errorf("--model is required"))
	}
	cfg := finetune.DefaultConfig()
	cfg.ModelDir, cfg.Method, cfg.ComputeBackend = *model, *method, *backend
	var devicePointer *int
	if *device >= 0 {
		devicePointer = device
	}
	worker, err := finetune.NewGradientWorkerServer(cfg, devicePointer)
	fatalIf(err)
	defer worker.Close()
	fmt.Fprintln(os.Stderr, "Sypra gradient worker listening on", *listen)
	fatal(http.ListenAndServe(*listen, worker.Handler()))
}

func parseIntList(value string) []int {
	var result []int
	for _, part := range parseStringList(value) {
		parsed, err := strconv.Atoi(part)
		if err == nil && parsed >= 0 {
			result = append(result, parsed)
		}
	}
	return result
}

func parseStringList(value string) []string {
	var result []string
	for _, part := range strings.Split(value, ",") {
		if part = strings.TrimSpace(part); part != "" {
			result = append(result, part)
		}
	}
	return result
}
func parseModalities(value string) []string {
	allowed := map[string]bool{"text": true, "image": true, "audio": true, "video": true, "file": true}
	seen, result := map[string]bool{}, []string{}
	for _, item := range parseStringList(value) {
		item = strings.ToLower(item)
		if allowed[item] && !seen[item] {
			seen[item] = true
			result = append(result, item)
		}
	}
	if len(result) == 0 {
		return []string{"text"}
	}
	return result
}
func print(v any) { b, _ := json.MarshalIndent(v, "", "  "); fmt.Println(string(b)) }
func fatalIf(e error) {
	if e != nil {
		fatal(e)
	}
}
func fatal(e error) { fmt.Fprintln(os.Stderr, "error:", e); os.Exit(1) }
func newID() string { return strings.ReplaceAll(fmt.Sprintf("%d", time.Now().UnixNano()), "-", "") }
func usage() {
	fmt.Println("Sypra: local SLM inference, RAG, training, web retrieval, and native runners\n\nCommands: start | serve | version | runner | chat | models | rag | train | chats | web | metrics | backup | restore")
}
