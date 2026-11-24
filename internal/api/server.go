package api

import (
	"archive/zip"
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"github.com/sypra-labs/sypra-ai/internal/adapter"
	"github.com/sypra-labs/sypra-ai/internal/agent"
	"github.com/sypra-labs/sypra-ai/internal/chattemplate"
	"github.com/sypra-labs/sypra-ai/internal/domain"
	"github.com/sypra-labs/sypra-ai/internal/finetune"
	"github.com/sypra-labs/sypra-ai/internal/inference"
	"github.com/sypra-labs/sypra-ai/internal/media"
	"github.com/sypra-labs/sypra-ai/internal/observability"
	"github.com/sypra-labs/sypra-ai/internal/rag"
	"github.com/sypra-labs/sypra-ai/internal/runnermanager"
	"github.com/sypra-labs/sypra-ai/internal/store"
	"github.com/sypra-labs/sypra-ai/internal/training"
	"github.com/sypra-labs/sypra-ai/internal/websearch"
	"io"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"sync"
	"time"
)

type Server struct {
	Store           *store.Store
	RAG             rag.Service
	Trainer         *training.Service
	Inference       *inference.Client
	CORSOrigin      string
	Logger          *slog.Logger
	UploadDir       string
	Web             websearch.Service
	Media           *media.Service
	MediaRoot       string
	Agent           *agent.Service
	Runners         *runnermanager.Manager
	Metrics         *observability.Metrics
	Telemetry       *observability.Telemetry
	ProviderAPIKeys map[string]string
	asyncMu         sync.Mutex
	asyncCancel     map[string]asyncRequest
	streamMu        sync.RWMutex
	chatStreams     map[string]map[chan chatStreamEvent]struct{}
}
type asyncRequest struct {
	chatID string
	cancel context.CancelFunc
}
type chatStreamEvent struct {
	MessageID string               `json:"messageID"`
	Content   string               `json:"content"`
	Parts     []domain.ContentPart `json:"parts,omitempty"`
	Status    string               `json:"status"`
	Sources   []string             `json:"sources,omitempty"`
	LatencyMS int64                `json:"latencyMs,omitempty"`
}

type chatInput struct {
	ChatID, ModelID, Endpoint, CollectionID, Prompt, RAGMode string
	Parts                                                    []domain.ContentPart
	Temperature                                              float64
	MaxTokens, RAGTopK                                       int
	Async                                                    bool
	ToolsEnabled                                             bool
	WebSearch, RAGEnabled                                    *bool
	remoteModelID, apiKey, provider, adapter, projectID      string
}

func (s *Server) Handler() http.Handler {
	return s.recover(s.cors(s.log(http.HandlerFunc(s.route))))
}
func (s *Server) route(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/healthz" {
		write(w, 200, map[string]string{"status": "ok"})
		return
	}
	if r.URL.Path == "/readyz" {
		if err := s.Store.Health(r.Context()); err != nil {
			s.logger().Error("API readiness check failed", "error", err)
			errJSON(w, http.StatusServiceUnavailable, "service is not ready")
			return
		}
		write(w, http.StatusOK, map[string]string{"status": "ready"})
		return
	}
	if r.URL.Path == "/metrics" {
		if s.Metrics == nil {
			http.Error(w, "metrics unavailable", http.StatusServiceUnavailable)
			return
		}
		s.Metrics.ServeHTTP(w, r)
		return
	}
	if !strings.HasPrefix(r.URL.Path, "/api/v1/") {
		errJSON(w, 404, "not found")
		return
	}
	p := strings.TrimPrefix(r.URL.Path, "/api/v1/")
	switch {
	case p == "models" && r.Method == http.MethodGet:
		s.models(w, r)
	case p == "models/catalog" && r.Method == http.MethodGet:
		s.modelCatalog(w, r)
	case p == "models" && r.Method == http.MethodPost:
		s.addModel(w, r)
	case p == "runners" && r.Method == http.MethodGet:
		s.runners(w, r)
	case strings.HasPrefix(p, "models/") && strings.HasSuffix(p, "/start") && r.Method == http.MethodPost:
		s.startRunner(w, r)
	case strings.HasPrefix(p, "models/") && strings.HasSuffix(p, "/stop") && r.Method == http.MethodPost:
		s.stopRunner(w, r)
	case strings.HasPrefix(p, "models/") && strings.HasSuffix(p, "/default") && r.Method == http.MethodPost:
		s.setDefaultBaseModel(w, r)
	case strings.HasPrefix(p, "models/") && strings.HasSuffix(p, "/promote") && r.Method == http.MethodPost:
		s.promoteModel(w, r)
	case strings.HasPrefix(p, "models/") && r.Method == http.MethodDelete:
		s.deleteModel(w, r)
	case p == "models/upload" && r.Method == http.MethodPost:
		s.uploadModel(w, r)
	case p == "training/datasets/upload" && r.Method == http.MethodPost:
		s.uploadDataset(w, r)
	case p == "rag/collections" && r.Method == http.MethodGet:
		s.collections(w, r)
	case p == "rag/collections" && r.Method == http.MethodPost:
		s.createCollection(w, r)
	case strings.HasPrefix(p, "rag/collections/") && strings.HasSuffix(p, "/documents") && r.Method == http.MethodGet:
		s.collectionDocuments(w, r)
	case strings.HasPrefix(p, "rag/collections/") && strings.HasSuffix(p, "/documents") && r.Method == http.MethodPost:
		s.addCollectionDocuments(w, r)
	case strings.HasPrefix(p, "rag/collections/") && r.Method == http.MethodDelete:
		s.deleteCollection(w, r)
	case p == "rag/documents/upload" && r.Method == http.MethodPost:
		s.uploadRAGDocument(w, r)
	case strings.HasPrefix(p, "rag/documents/") && strings.HasSuffix(p, "/reindex") && r.Method == http.MethodPost:
		s.reindexRAGDocument(w, r)
	case strings.HasPrefix(p, "rag/documents/") && strings.HasSuffix(p, "/refresh") && r.Method == http.MethodPost:
		s.refreshRAGDocument(w, r)
	case strings.HasPrefix(p, "rag/documents/") && r.Method == http.MethodPatch:
		s.renameRAGDocument(w, r)
	case strings.HasPrefix(p, "rag/documents/") && r.Method == http.MethodDelete:
		s.deleteRAGDocument(w, r)
	case p == "rag/query" && r.Method == http.MethodPost:
		s.queryRAG(w, r)
	case p == "rag/evaluations" && r.Method == http.MethodGet:
		s.evaluationSets(w, r)
	case p == "rag/evaluations" && r.Method == http.MethodPost:
		s.createEvaluationSet(w, r)
	case strings.HasPrefix(p, "rag/evaluations/") && strings.HasSuffix(p, "/run") && r.Method == http.MethodPost:
		s.runEvaluation(w, r)
	case strings.HasPrefix(p, "rag/evaluations/") && strings.HasSuffix(p, "/runs") && r.Method == http.MethodGet:
		s.evaluationRuns(w, r)
	case p == "media/jobs" && r.Method == http.MethodGet:
		s.mediaJobs(w, r)
	case p == "media/jobs" && r.Method == http.MethodPost:
		s.startMediaJob(w, r)
	case strings.HasPrefix(p, "media/jobs/") && strings.HasSuffix(p, "/cancel") && r.Method == http.MethodPost:
		s.cancelMediaJob(w, r)
	case p == "media/upload" && r.Method == http.MethodPost:
		s.uploadMedia(w, r)
	case strings.HasPrefix(p, "media/files/") && r.Method == http.MethodGet:
		s.mediaFile(w, r)
	case p == "tools/catalog" && r.Method == http.MethodGet:
		s.toolCatalog(w, r)
	case p == "tools/connections" && r.Method == http.MethodGet:
		s.toolConnections(w, r)
	case p == "tools/connections" && r.Method == http.MethodPost:
		s.addToolConnection(w, r)
	case strings.HasPrefix(p, "tools/connections/") && r.Method == http.MethodDelete:
		s.deleteToolConnection(w, r)
	case p == "tools/permissions" && r.Method == http.MethodGet:
		s.toolPermissions(w, r)
	case p == "tools/permissions" && r.Method == http.MethodPost:
		s.addToolPermission(w, r)
	case p == "agents/runs" && r.Method == http.MethodGet:
		s.agentRuns(w, r)
	case p == "agents/runs" && r.Method == http.MethodPost:
		s.runAgentTool(w, r)
	case strings.HasPrefix(p, "agents/runs/") && strings.HasSuffix(p, "/approve") && r.Method == http.MethodPost:
		s.approveAgentRun(w, r)
	case p == "projects" && r.Method == http.MethodGet:
		s.projects(w, r)
	case p == "projects" && r.Method == http.MethodPost:
		s.createProject(w, r)
	case p == "chats" && r.Method == http.MethodGet:
		s.chats(w, r)
	case p == "chats" && r.Method == http.MethodPost:
		s.createChat(w, r)
	case strings.HasPrefix(p, "chats/") && r.Method == http.MethodDelete:
		s.deleteChat(w, r)
	case strings.HasPrefix(p, "chats/") && strings.HasSuffix(p, "/events") && r.Method == http.MethodGet:
		s.chatEvents(w, r)
	case strings.HasPrefix(p, "chats/") && strings.HasSuffix(p, "/messages") && r.Method == http.MethodGet:
		s.messages(w, r)
	case strings.HasPrefix(p, "chats/") && strings.HasSuffix(p, "/cancel") && r.Method == http.MethodPost:
		s.cancelChat(w, r)
	case strings.HasPrefix(p, "chats/") && r.Method == http.MethodPatch:
		s.updateChat(w, r)
	case p == "chat" && r.Method == http.MethodPost:
		s.chat(w, r)
	case p == "training/jobs" && r.Method == http.MethodGet:
		s.jobs(w, r)
	case p == "training/jobs" && r.Method == http.MethodPost:
		s.startJob(w, r)
	case strings.HasPrefix(p, "training/jobs/") && strings.HasSuffix(p, "/cancel") && r.Method == http.MethodPost:
		s.cancelJob(w, r)
	case p == "metrics" && r.Method == http.MethodGet:
		s.metrics(w, r)
	case p == "diagnostics/bundle" && r.Method == http.MethodGet:
		s.diagnosticBundle(w, r)
	case p == "client-logs" && r.Method == http.MethodPost:
		s.clientLog(w, r)
	default:
		errJSON(w, 404, "not found")
	}
}
func (s *Server) models(w http.ResponseWriter, r *http.Request) {
	v, e := s.Store.Models(r.Context())
	respond(w, v, e)
}

type localModelCatalogEntry struct {
	Name       string `json:"name"`
	Path       string `json:"path"`
	Family     string `json:"family"`
	Format     string `json:"format"`
	Registered bool   `json:"registered"`
}

func (s *Server) modelCatalog(w http.ResponseWriter, r *http.Request) {
	directory := strings.TrimSpace(r.URL.Query().Get("directory"))
	if directory == "" {
		errJSON(w, http.StatusBadRequest, "directory is required")
		return
	}
	models, err := s.Store.Models(r.Context())
	if err != nil {
		errJSON(w, 500, err.Error())
		return
	}
	entries, err := discoverLocalModels(directory, models)
	if err != nil {
		errJSON(w, http.StatusBadRequest, err.Error())
		return
	}
	write(w, http.StatusOK, entries)
}
func discoverLocalModels(directory string, registered []domain.Model) ([]localModelCatalogEntry, error) {
	root, err := filepath.Abs(directory)
	if err != nil {
		return nil, fmt.Errorf("resolve model directory: %w", err)
	}
	info, err := os.Stat(root)
	if err != nil {
		return nil, fmt.Errorf("read model directory: %w", err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("model catalog path must be a directory")
	}
	registeredPaths := map[string]bool{}
	for _, model := range registered {
		registeredPaths[filepath.Clean(model.Path)] = true
	}
	entries := []localModelCatalogEntry{}
	seen := map[string]bool{}
	err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		if !entry.IsDir() {
			extension := strings.ToLower(filepath.Ext(path))
			if extension != ".gguf" && extension != ".onnx" {
				return nil
			}
			cleanPath := filepath.Clean(path)
			if seen[cleanPath] {
				return nil
			}
			seen[cleanPath] = true
			format := strings.TrimPrefix(extension, ".")
			name := strings.TrimSuffix(filepath.Base(path), extension)
			entries = append(entries, localModelCatalogEntry{
				Name:       name,
				Path:       path,
				Family:     chattemplate.NormalizeFamily(name),
				Format:     format,
				Registered: registeredPaths[cleanPath],
			})
			return nil
		}
		relative, relativeErr := filepath.Rel(root, path)
		if relativeErr == nil && len(strings.Split(relative, string(os.PathSeparator))) > 6 {
			return filepath.SkipDir
		}
		configPath := filepath.Join(path, "config.json")
		configBytes, configErr := os.ReadFile(configPath)
		if configErr != nil {
			return nil
		}
		var config struct {
			ModelType     string   `json:"model_type"`
			Architectures []string `json:"architectures"`
		}
		if json.Unmarshal(configBytes, &config) != nil {
			return nil
		}
		if seen[path] {
			return filepath.SkipDir
		}
		seen[path] = true
		family := chattemplate.NormalizeFamily(config.ModelType + " " + strings.Join(config.Architectures, " ") + " " + filepath.Base(path))
		format := "unknown"
		if _, statErr := os.Stat(filepath.Join(path, "model.safetensors")); statErr == nil {
			format = "safetensors"
		} else if matches, _ := filepath.Glob(filepath.Join(path, "*.safetensors")); len(matches) > 0 {
			format = "safetensors"
		}
		entries = append(entries, localModelCatalogEntry{Name: filepath.Base(path), Path: path, Family: family, Format: format, Registered: registeredPaths[filepath.Clean(path)]})
		return filepath.SkipDir
	})
	if err != nil {
		return nil, fmt.Errorf("scan model directory: %w", err)
	}
	return entries, nil
}
func (s *Server) runners(w http.ResponseWriter, r *http.Request) {
	if s.Runners == nil {
		write(w, http.StatusOK, []runnermanager.Status{})
		return
	}
	s.Runners.Refresh(r.Context())
	write(w, http.StatusOK, s.Runners.Statuses())
}
func (s *Server) startRunner(w http.ResponseWriter, r *http.Request) {
	if s.Runners == nil {
		errJSON(w, http.StatusServiceUnavailable, "managed local runners are unavailable in this API process")
		return
	}
	modelID := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/api/v1/models/"), "/start")
	models, err := s.Store.Models(r.Context())
	if err != nil {
		errJSON(w, 500, err.Error())
		return
	}
	for _, model := range models {
		if model.ID != modelID {
			continue
		}
		status, startErr := s.Runners.Start(model, models)
		if startErr != nil {
			errJSON(w, http.StatusBadRequest, startErr.Error())
			return
		}
		write(w, http.StatusAccepted, status)
		return
	}
	errJSON(w, http.StatusNotFound, "model "+modelID+" not found")
}
func (s *Server) stopRunner(w http.ResponseWriter, r *http.Request) {
	if s.Runners == nil {
		errJSON(w, http.StatusServiceUnavailable, "managed local runners are unavailable in this API process")
		return
	}
	modelID := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/api/v1/models/"), "/stop")
	status, err := s.Runners.Stop(modelID)
	if err != nil {
		errJSON(w, 500, err.Error())
		return
	}
	write(w, http.StatusOK, status)
}
func (s *Server) deleteModel(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/v1/models/")
	if err := s.Store.DeleteModel(r.Context(), id); err != nil {
		errJSON(w, 404, err.Error())
		return
	}
	write(w, http.StatusNoContent, nil)
}
func (s *Server) promoteModel(w http.ResponseWriter, r *http.Request) {
	modelID := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/api/v1/models/"), "/promote")
	model, err := s.Store.PromoteModel(r.Context(), modelID)
	if err != nil {
		errJSON(w, 400, err.Error())
		return
	}
	write(w, http.StatusOK, model)
}
func (s *Server) setDefaultBaseModel(w http.ResponseWriter, r *http.Request) {
	modelID := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/api/v1/models/"), "/default")
	model, err := s.Store.SetDefaultBaseModel(r.Context(), modelID)
	if err != nil {
		errJSON(w, http.StatusBadRequest, err.Error())
		return
	}
	write(w, http.StatusOK, model)
}
func (s *Server) addModel(w http.ResponseWriter, r *http.Request) {
	var x struct {
		Name, Path, Format, Endpoint, Kind, Provider, RemoteID, Family string
		Modalities                                                     []string
	}
	if !decode(w, r, &x) {
		return
	}
	x.Provider = strings.ToLower(strings.TrimSpace(x.Provider))
	if x.Kind == "" {
		x.Kind = "base"
	}
	if x.Kind != "base" && x.Kind != "slm" {
		errJSON(w, 400, "kind must be base or slm")
		return
	}
	if x.Provider != "" {
		if x.Kind != "base" {
			errJSON(w, 400, "cloud providers can only be added as base chat models")
			return
		}
		if x.RemoteID == "" {
			errJSON(w, 400, "remoteId is required for a cloud model")
			return
		}
		endpoint, ok := cloudProviderEndpoint(x.Provider)
		if !ok {
			errJSON(w, 400, "provider must be openai, xai, or gemini")
			return
		}
		if x.Endpoint == "" {
			x.Endpoint = endpoint
		}
		if x.Name == "" {
			x.Name = x.RemoteID
		}
		x.Path, x.Format = "cloud://"+x.Provider+"/"+x.RemoteID, "api"
	} else {
		if x.Name == "" {
			errJSON(w, 400, "name is required for a local model")
			return
		}
		if x.Path == "" && x.Endpoint == "" {
			errJSON(w, 400, "path is required unless an external local endpoint is supplied")
			return
		}
		if x.Endpoint != "" && x.Path == "" {
			x.Path, x.Format = x.Endpoint, "api"
		} else if x.Format == "" {
			x.Format = "gguf"
		}
	}
	if strings.TrimSpace(x.Family) == "" {
		x.Family = chattemplate.NormalizeFamily(x.Name + " " + x.RemoteID + " " + x.Path)
	} else {
		x.Family = chattemplate.NormalizeFamily(x.Family)
	}
	m := domain.Model{ID: id(), Name: x.Name, Path: x.Path, Format: x.Format, Endpoint: x.Endpoint, Provider: x.Provider, RemoteID: x.RemoteID, Family: x.Family, Modalities: normalizeModalities(x.Modalities), Kind: x.Kind, CreatedAt: store.Now()}
	stored, e := s.Store.AddModel(r.Context(), m)
	if e != nil {
		errJSON(w, 500, e.Error())
		return
	}
	write(w, 201, stored)
}
func (s *Server) uploadModel(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 10<<30)
	if e := r.ParseMultipartForm(32 << 20); e != nil {
		errJSON(w, 400, "invalid model upload: "+e.Error())
		return
	}
	file, header, e := r.FormFile("file")
	if e != nil {
		errJSON(w, 400, "model file is required")
		return
	}
	defer file.Close()
	kind := r.FormValue("kind")
	if kind == "" {
		kind = "slm"
	}
	if kind != "base" && kind != "slm" {
		errJSON(w, 400, "kind must be base or slm")
		return
	}
	dir := s.UploadDir
	if dir == "" {
		dir = "data/models"
	}
	if e = os.MkdirAll(dir, 0755); e != nil {
		errJSON(w, 500, e.Error())
		return
	}
	target := filepath.Join(dir, filepath.Base(header.Filename))
	out, e := os.Create(target)
	if e != nil {
		errJSON(w, 500, e.Error())
		return
	}
	_, copyErr := io.Copy(out, file)
	closeErr := out.Close()
	if copyErr != nil || closeErr != nil {
		_ = os.Remove(target)
		errJSON(w, 500, "save model upload failed")
		return
	}
	name := r.FormValue("name")
	if name == "" {
		name = header.Filename
	}
	family := chattemplate.NormalizeFamily(r.FormValue("family") + " " + name)
	m := domain.Model{ID: id(), Name: name, Path: target, Format: strings.TrimPrefix(strings.ToLower(filepath.Ext(header.Filename)), "."), Endpoint: r.FormValue("endpoint"), Family: family, Modalities: normalizeModalities(strings.Split(r.FormValue("modalities"), ",")), Kind: kind, CreatedAt: store.Now()}
	m, e = s.Store.AddModel(r.Context(), m)
	if e != nil {
		errJSON(w, 500, e.Error())
		return
	}
	write(w, 201, m)
}
func (s *Server) uploadDataset(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 2<<30)
	if e := r.ParseMultipartForm(32 << 20); e != nil {
		errJSON(w, 400, "invalid dataset upload: "+e.Error())
		return
	}
	file, header, e := r.FormFile("file")
	if e != nil {
		errJSON(w, 400, "training dataset file is required")
		return
	}
	defer file.Close()
	extension := strings.ToLower(filepath.Ext(header.Filename))
	if extension != ".csv" && extension != ".jsonl" && extension != ".json" {
		errJSON(w, 400, "training data must be CSV or JSONL")
		return
	}
	if _, e = file.Seek(0, 0); e != nil {
		errJSON(w, 500, "rewind dataset upload: "+e.Error())
		return
	}
	root := s.UploadDir
	if root == "" {
		root = "data/models"
	}
	dir := filepath.Join(filepath.Dir(root), "datasets")
	if e = os.MkdirAll(dir, 0755); e != nil {
		errJSON(w, 500, e.Error())
		return
	}
	target := filepath.Join(dir, filepath.Base(header.Filename))
	out, e := os.Create(target)
	if e != nil {
		errJSON(w, 500, e.Error())
		return
	}
	_, copyErr := io.Copy(out, file)
	closeErr := out.Close()
	if copyErr != nil || closeErr != nil {
		_ = os.Remove(target)
		errJSON(w, 500, "save dataset upload failed")
		return
	}
	if e = finetune.ValidateCSV(target); e != nil {
		_ = os.Remove(target)
		errJSON(w, 400, "invalid training dataset: "+e.Error())
		return
	}
	write(w, 201, map[string]string{"path": target, "format": strings.TrimPrefix(extension, ".")})
}
func (s *Server) collections(w http.ResponseWriter, r *http.Request) {
	v, e := s.Store.Collections(r.Context())
	respond(w, v, e)
}
func (s *Server) deleteCollection(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/v1/rag/collections/")
	if err := s.Store.DeleteCollection(r.Context(), id); err != nil {
		errJSON(w, 404, err.Error())
		return
	}
	write(w, http.StatusNoContent, nil)
}
func (s *Server) addCollectionDocuments(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/api/v1/rag/collections/"), "/documents")
	var x struct{ Documents []rag.Document }
	if !decode(w, r, &x) {
		return
	}
	collection, err := s.RAG.AddDocuments(r.Context(), id, x.Documents)
	if err != nil {
		errJSON(w, http.StatusBadRequest, err.Error())
		return
	}
	write(w, http.StatusOK, collection)
}
func (s *Server) collectionDocuments(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/api/v1/rag/collections/"), "/documents")
	v, err := s.Store.Documents(r.Context(), id)
	respond(w, v, err)
}
func (s *Server) createCollection(w http.ResponseWriter, r *http.Request) {
	var x struct {
		Name, EmbeddingModel    string
		ChunkSize, ChunkOverlap int
		Documents               []rag.Document
		RoutingTags             []string
	}
	if !decode(w, r, &x) {
		return
	}
	c, e := s.RAG.Create(r.Context(), rag.CreateInput{Name: x.Name, EmbeddingModel: x.EmbeddingModel, ChunkSize: x.ChunkSize, ChunkOverlap: x.ChunkOverlap, Documents: x.Documents, RoutingTags: x.RoutingTags})
	if e != nil {
		errJSON(w, 400, e.Error())
		return
	}
	write(w, 201, c)
}
func (s *Server) uploadRAGDocument(w http.ResponseWriter, r *http.Request) {
	// Store source files locally so binary formats (DOCX) are extracted by the
	// backend rather than coerced into browser text.
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		errJSON(w, 400, "invalid document upload: "+err.Error())
		return
	}
	file, header, err := r.FormFile("file")
	if err != nil {
		errJSON(w, 400, "document file is required")
		return
	}
	defer file.Close()
	// Any file can be stored and classified. RAG extraction chooses native text,
	// Office parsing, OCR, transcription, or a configured local extractor.
	root := s.UploadDir
	if root == "" {
		root = "data/models"
	}
	dir := filepath.Join(filepath.Dir(root), "documents")
	if err = os.MkdirAll(dir, 0755); err != nil {
		errJSON(w, 500, err.Error())
		return
	}
	target := filepath.Join(dir, id()+"-"+filepath.Base(header.Filename))
	out, err := os.Create(target)
	if err != nil {
		errJSON(w, 500, err.Error())
		return
	}
	_, copyErr := io.Copy(out, file)
	closeErr := out.Close()
	if copyErr != nil || closeErr != nil {
		_ = os.Remove(target)
		errJSON(w, 500, "save RAG document failed")
		return
	}
	write(w, 201, map[string]string{"source": header.Filename, "path": target})
}
func (s *Server) queryRAG(w http.ResponseWriter, r *http.Request) {
	var x struct {
		CollectionID, Query string
		CollectionIDs       []string
		RoutingTags         []string
		Metadata            map[string]string
		Limit               int
		Rerank              bool
		IncludeNeighbors    *bool
	}
	if !decode(w, r, &x) {
		return
	}
	// Keep the original compact array response for existing CLI/API clients.
	// The inspect controls opt into the evidence-rich response below.
	if x.CollectionID != "" && x.Metadata == nil && !x.Rerank && x.IncludeNeighbors == nil {
		v, err := s.RAG.Retrieve(r.Context(), x.CollectionID, x.Query, x.Limit)
		respond(w, v, err)
		return
	}
	options := rag.RetrievalOptions{Limit: x.Limit, Metadata: x.Metadata, Rerank: x.Rerank, IncludeNeighbors: true}
	if x.IncludeNeighbors != nil {
		options.IncludeNeighbors = *x.IncludeNeighbors
	}
	if x.CollectionID != "" {
		v, e := s.RAG.RetrieveDetailed(r.Context(), x.CollectionID, x.Query, options)
		respond(w, map[string]any{"collectionId": x.CollectionID, "results": v}, e)
		return
	}
	collectionID, v, e := s.RAG.Route(r.Context(), x.Query, x.CollectionIDs, x.RoutingTags, options)
	respond(w, map[string]any{"collectionId": collectionID, "results": v, "routed": true}, e)
}

func ragDocumentID(path, suffix string) string {
	return strings.TrimSuffix(strings.TrimPrefix(path, "/api/v1/rag/documents/"), suffix)
}

func (s *Server) renameRAGDocument(w http.ResponseWriter, r *http.Request) {
	var input struct{ Name string }
	if !decode(w, r, &input) {
		return
	}
	document, err := s.Store.RenameDocument(r.Context(), ragDocumentID(r.URL.Path, ""), input.Name)
	respond(w, document, err)
}

func (s *Server) deleteRAGDocument(w http.ResponseWriter, r *http.Request) {
	if err := s.Store.DeleteDocument(r.Context(), ragDocumentID(r.URL.Path, "")); err != nil {
		errJSON(w, http.StatusNotFound, err.Error())
		return
	}
	write(w, http.StatusNoContent, nil)
}

func (s *Server) reindexRAGDocument(w http.ResponseWriter, r *http.Request) {
	var input struct{ Document *rag.Document }
	if !decode(w, r, &input) {
		return
	}
	document, err := s.RAG.ReindexDocument(r.Context(), ragDocumentID(r.URL.Path, "/reindex"), input.Document)
	respond(w, document, err)
}

func (s *Server) refreshRAGDocument(w http.ResponseWriter, r *http.Request) {
	document, err := s.RAG.ReindexDocument(r.Context(), ragDocumentID(r.URL.Path, "/refresh"), nil)
	respond(w, document, err)
}

func (s *Server) evaluationSets(w http.ResponseWriter, r *http.Request) {
	sets, err := s.Store.EvaluationSets(r.Context(), r.URL.Query().Get("collectionId"))
	respond(w, sets, err)
}

func (s *Server) createEvaluationSet(w http.ResponseWriter, r *http.Request) {
	var input struct {
		CollectionID, Name string
		Cases              []domain.RAGEvaluationCase
	}
	if !decode(w, r, &input) {
		return
	}
	set, err := s.RAG.CreateEvaluationSet(r.Context(), input.CollectionID, input.Name, input.Cases)
	respond(w, set, err)
}

func (s *Server) runEvaluation(w http.ResponseWriter, r *http.Request) {
	var input struct {
		Limit    int
		Metadata map[string]string
		Rerank   bool
	}
	if !decode(w, r, &input) {
		return
	}
	id := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/api/v1/rag/evaluations/"), "/run")
	run, err := s.RAG.Evaluate(r.Context(), id, input.Limit, rag.RetrievalOptions{Metadata: input.Metadata, Rerank: input.Rerank})
	respond(w, run, err)
}

func (s *Server) evaluationRuns(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/api/v1/rag/evaluations/"), "/runs")
	runs, err := s.Store.EvaluationRuns(r.Context(), id)
	respond(w, runs, err)
}

func (s *Server) mediaJobs(w http.ResponseWriter, r *http.Request) {
	if s.Media == nil {
		errJSON(w, http.StatusServiceUnavailable, "media workflows are unavailable")
		return
	}
	jobs, err := s.Store.MediaJobs(r.Context())
	respond(w, jobs, err)
}

func (s *Server) startMediaJob(w http.ResponseWriter, r *http.Request) {
	if s.Media == nil {
		errJSON(w, http.StatusServiceUnavailable, "media workflows are unavailable")
		return
	}
	var input struct{ Kind, Prompt, InputPath, InputName string }
	if !decode(w, r, &input) {
		return
	}
	job, err := s.Media.Start(r.Context(), input.Kind, input.Prompt, input.InputPath, input.InputName)
	respond(w, job, err)
}

func (s *Server) cancelMediaJob(w http.ResponseWriter, r *http.Request) {
	if s.Media == nil {
		errJSON(w, http.StatusServiceUnavailable, "media workflows are unavailable")
		return
	}
	id := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/api/v1/media/jobs/"), "/cancel")
	job, err := s.Media.Cancel(r.Context(), id)
	respond(w, job, err)
}

func (s *Server) uploadMedia(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(256 << 20); err != nil {
		errJSON(w, 400, "invalid media upload: "+err.Error())
		return
	}
	file, header, err := r.FormFile("file")
	if err != nil {
		errJSON(w, 400, "media file is required")
		return
	}
	defer file.Close()
	root := s.MediaRoot
	if root == "" {
		root = filepath.Join(filepath.Dir(s.UploadDir), "media")
	}
	inputDir := filepath.Join(root, "inputs")
	if err := os.MkdirAll(inputDir, 0755); err != nil {
		errJSON(w, 500, err.Error())
		return
	}
	name := id() + "-" + filepath.Base(header.Filename)
	path := filepath.Join(inputDir, name)
	out, err := os.Create(path)
	if err != nil {
		errJSON(w, 500, err.Error())
		return
	}
	_, copyErr := io.Copy(out, file)
	closeErr := out.Close()
	if copyErr != nil || closeErr != nil {
		_ = os.Remove(path)
		errJSON(w, 500, "save media upload failed")
		return
	}
	write(w, http.StatusCreated, map[string]string{"path": path, "name": header.Filename, "mimeType": header.Header.Get("Content-Type")})
}

func (s *Server) mediaFile(w http.ResponseWriter, r *http.Request) {
	root := s.MediaRoot
	if root == "" {
		root = filepath.Join(filepath.Dir(s.UploadDir), "media")
	}
	name := filepath.Base(strings.TrimPrefix(r.URL.Path, "/api/v1/media/files/"))
	if name == "." || name == "" {
		errJSON(w, 404, "media artifact not found")
		return
	}
	path := filepath.Join(root, "outputs", name)
	if _, err := os.Stat(path); err != nil {
		errJSON(w, 404, "media artifact not found")
		return
	}
	http.ServeFile(w, r, path)
}

func (s *Server) toolCatalog(w http.ResponseWriter, r *http.Request) {
	if s.Agent == nil {
		errJSON(w, http.StatusServiceUnavailable, "agent tools are unavailable")
		return
	}
	tools, err := s.Agent.Catalog(r.Context())
	respond(w, tools, err)
}

func (s *Server) toolConnections(w http.ResponseWriter, r *http.Request) {
	connections, err := s.Store.ToolConnections(r.Context())
	respond(w, connections, err)
}

func (s *Server) addToolConnection(w http.ResponseWriter, r *http.Request) {
	var input struct {
		Name, Kind, Command string
		Args, ToolIDs       []string
		Enabled             bool
	}
	if !decode(w, r, &input) {
		return
	}
	input.Name, input.Kind = strings.TrimSpace(input.Name), strings.ToLower(strings.TrimSpace(input.Kind))
	if input.Name == "" || (input.Kind != "mcp" && input.Kind != "pipeline") {
		errJSON(w, 400, "connection requires a name and kind mcp or pipeline")
		return
	}
	if input.Kind == "mcp" && strings.TrimSpace(input.Command) == "" {
		errJSON(w, 400, "MCP connection command is required")
		return
	}
	if input.Kind == "pipeline" && len(input.ToolIDs) == 0 {
		errJSON(w, 400, "pipeline needs at least one tool ID")
		return
	}
	now := store.Now()
	connection := domain.ToolConnection{ID: id(), Name: input.Name, Kind: input.Kind, Command: input.Command, Args: input.Args, ToolIDs: input.ToolIDs, Enabled: input.Enabled, Status: "configured", CreatedAt: now, UpdatedAt: now}
	if connection.Kind == "mcp" && connection.Enabled && s.Agent != nil {
		discovered, discoverErr := s.Agent.Discover(r.Context(), connection)
		if discoverErr != nil {
			connection.Status, connection.Error = "failed", discoverErr.Error()
		} else {
			connection.Status, connection.ToolSchemas = "ready", map[string]map[string]any{}
			connection.ToolIDs = make([]string, 0, len(discovered))
			for _, tool := range discovered {
				connection.ToolIDs = append(connection.ToolIDs, tool.Name)
				connection.ToolSchemas[tool.Name] = tool.InputSchema
			}
		}
	}
	if err := s.Store.AddToolConnection(r.Context(), connection); err != nil {
		errJSON(w, 500, err.Error())
		return
	}
	write(w, http.StatusCreated, connection)
}

func (s *Server) deleteToolConnection(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/v1/tools/connections/")
	if err := s.Store.DeleteToolConnection(r.Context(), id); err != nil {
		errJSON(w, 404, err.Error())
		return
	}
	write(w, http.StatusNoContent, nil)
}

func (s *Server) toolPermissions(w http.ResponseWriter, r *http.Request) {
	permissions, err := s.Store.ToolPermissions(r.Context())
	respond(w, permissions, err)
}

func (s *Server) addToolPermission(w http.ResponseWriter, r *http.Request) {
	var input struct {
		ToolID, ModelID, ProjectID string
		Enabled, RequireApproval   bool
	}
	if !decode(w, r, &input) {
		return
	}
	if strings.TrimSpace(input.ToolID) == "" {
		errJSON(w, 400, "tool ID is required")
		return
	}
	permission := domain.ToolPermission{ID: id(), ToolID: input.ToolID, ModelID: input.ModelID, ProjectID: input.ProjectID, Enabled: input.Enabled, RequireApproval: input.RequireApproval, CreatedAt: store.Now()}
	if err := s.Store.AddToolPermission(r.Context(), permission); err != nil {
		errJSON(w, 500, err.Error())
		return
	}
	write(w, http.StatusCreated, permission)
}

func (s *Server) agentRuns(w http.ResponseWriter, r *http.Request) {
	runs, err := s.Store.AgentRuns(r.Context())
	respond(w, runs, err)
}

func (s *Server) runAgentTool(w http.ResponseWriter, r *http.Request) {
	if s.Agent == nil {
		errJSON(w, http.StatusServiceUnavailable, "agent tools are unavailable")
		return
	}
	var input struct {
		ModelID, ProjectID, ToolID string
		Arguments                  map[string]any
		Approved                   bool
	}
	if !decode(w, r, &input) {
		return
	}
	run, err := s.Agent.Execute(r.Context(), input.ModelID, input.ProjectID, input.ToolID, input.Arguments, input.Approved)
	if err != nil {
		errJSON(w, http.StatusBadRequest, err.Error())
		return
	}
	write(w, http.StatusOK, run)
}

func (s *Server) approveAgentRun(w http.ResponseWriter, r *http.Request) {
	if s.Agent == nil {
		errJSON(w, http.StatusServiceUnavailable, "agent tools are unavailable")
		return
	}
	runID := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/api/v1/agents/runs/"), "/approve")
	run, err := s.Agent.Approve(r.Context(), runID)
	if err != nil {
		errJSON(w, http.StatusBadRequest, err.Error())
		return
	}
	write(w, http.StatusOK, run)
}
func (s *Server) projects(w http.ResponseWriter, r *http.Request) {
	v, e := s.Store.Projects(r.Context())
	respond(w, v, e)
}
func (s *Server) createProject(w http.ResponseWriter, r *http.Request) {
	var x struct{ Name string }
	if !decode(w, r, &x) {
		return
	}
	if strings.TrimSpace(x.Name) == "" {
		errJSON(w, 400, "project name is required")
		return
	}
	p := domain.Project{ID: id(), Name: x.Name, CreatedAt: store.Now()}
	if e := s.Store.AddProject(r.Context(), p); e != nil {
		errJSON(w, 500, e.Error())
		return
	}
	write(w, 201, p)
}
func (s *Server) chats(w http.ResponseWriter, r *http.Request) {
	v, e := s.Store.Chats(r.Context(), r.URL.Query().Get("projectId"), r.URL.Query().Get("q"))
	respond(w, v, e)
}
func (s *Server) deleteChat(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/v1/chats/")
	s.cancelRunningChat(id)
	if err := s.Store.DeleteChat(r.Context(), id); err != nil {
		errJSON(w, 404, err.Error())
		return
	}
	write(w, http.StatusNoContent, nil)
}
func (s *Server) createChat(w http.ResponseWriter, r *http.Request) {
	var x struct {
		Title, ProjectID, ModelID, CollectionID, RAGMode string
		RAGTopK                                          int
		WebEnabled                                       bool
		RAGEnabled                                       *bool
	}
	if !decode(w, r, &x) {
		return
	}
	now := store.Now()
	if x.Title == "" {
		x.Title = "New chat"
	}
	mode, topK, e := ragSettings(x.RAGMode, x.RAGTopK)
	if e != nil {
		errJSON(w, 400, e.Error())
		return
	}
	if x.ModelID == "" {
		if defaultModel, ok, defaultErr := s.Store.DefaultBaseModel(r.Context()); defaultErr != nil {
			errJSON(w, 500, defaultErr.Error())
			return
		} else if ok {
			x.ModelID = defaultModel.ID
		}
	}
	if x.CollectionID != "" && x.RAGEnabled == nil {
		enabled := true
		x.RAGEnabled = &enabled
	}
	c := domain.Chat{ID: id(), Title: x.Title, ProjectID: x.ProjectID, ModelID: x.ModelID, CollectionID: x.CollectionID, RAGMode: mode, RAGTopK: topK, RAGEnabled: x.RAGEnabled, WebEnabled: x.WebEnabled, CreatedAt: now, UpdatedAt: now}
	if e := s.Store.AddChat(r.Context(), c); e != nil {
		errJSON(w, 500, e.Error())
		return
	}
	write(w, 201, c)
}
func (s *Server) messages(w http.ResponseWriter, r *http.Request) {
	chatID := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/api/v1/chats/"), "/messages")
	v, e := s.Store.Messages(r.Context(), chatID)
	respond(w, v, e)
}

// chatEvents exposes the in-progress assistant message as server-sent events.
// The runner itself streams token deltas; this channel carries the progressively
// accumulated message to the browser while the JSON store remains durable for
// reloads and reconnections.
func (s *Server) chatEvents(w http.ResponseWriter, r *http.Request) {
	chatID := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/api/v1/chats/"), "/events")
	if _, err := s.Store.Chat(r.Context(), chatID); err != nil {
		errJSON(w, http.StatusNotFound, err.Error())
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		errJSON(w, http.StatusInternalServerError, "streaming is unavailable")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	channel := s.subscribeChatStream(chatID)
	defer s.unsubscribeChatStream(chatID, channel)

	writeEvent := func(event chatStreamEvent) bool {
		data, err := json.Marshal(event)
		if err != nil {
			return false
		}
		_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
		return true
	}
	// Send the current persisted partial first, so reconnecting after a refresh
	// displays what has already been generated before later deltas arrive.
	if messages, err := s.Store.Messages(r.Context(), chatID); err == nil {
		for index := len(messages) - 1; index >= 0; index-- {
			message := messages[index]
			if message.Role == "assistant" && message.Status == "working" {
				writeEvent(chatStreamEvent{MessageID: message.ID, Content: message.Content, Status: message.Status, LatencyMS: message.LatencyMS})
				break
			}
		}
	}
	for {
		select {
		case <-r.Context().Done():
			return
		case event := <-channel:
			if !writeEvent(event) {
				return
			}
		}
	}
}

func (s *Server) subscribeChatStream(chatID string) chan chatStreamEvent {
	channel := make(chan chatStreamEvent, 64)
	s.streamMu.Lock()
	if s.chatStreams == nil {
		s.chatStreams = map[string]map[chan chatStreamEvent]struct{}{}
	}
	if s.chatStreams[chatID] == nil {
		s.chatStreams[chatID] = map[chan chatStreamEvent]struct{}{}
	}
	s.chatStreams[chatID][channel] = struct{}{}
	s.streamMu.Unlock()
	return channel
}

func (s *Server) unsubscribeChatStream(chatID string, channel chan chatStreamEvent) {
	s.streamMu.Lock()
	defer s.streamMu.Unlock()
	listeners := s.chatStreams[chatID]
	delete(listeners, channel)
	if len(listeners) == 0 {
		delete(s.chatStreams, chatID)
	}
}

func (s *Server) publishChatStream(chatID string, message domain.ChatMessage) {
	event := chatStreamEvent{MessageID: message.ID, Content: message.Content, Parts: message.Parts, Status: message.Status, Sources: message.Sources, LatencyMS: message.LatencyMS}
	s.streamMu.RLock()
	defer s.streamMu.RUnlock()
	for channel := range s.chatStreams[chatID] {
		// The next event contains the complete accumulated response, so dropping
		// a slow client's intermediate event cannot corrupt its displayed text.
		select {
		case channel <- event:
		default:
		}
	}
}
func (s *Server) updateChat(w http.ResponseWriter, r *http.Request) {
	chatID := strings.TrimPrefix(r.URL.Path, "/api/v1/chats/")
	c, e := s.Store.Chat(r.Context(), chatID)
	if e != nil {
		errJSON(w, 404, e.Error())
		return
	}
	var x struct {
		Title, ProjectID, ModelID, CollectionID, RAGMode *string
		RAGTopK                                          *int
		RAGEnabled                                       *bool
		WebEnabled                                       *bool
		Pinned                                           *bool
	}
	if !decode(w, r, &x) {
		return
	}
	if x.CollectionID != nil && c.CollectionID != "" && *x.CollectionID != c.CollectionID {
		errJSON(w, http.StatusConflict, "this chat is permanently bound to its RAG collection; start a new chat to use different knowledge")
		return
	}
	if x.Title != nil {
		c.Title = *x.Title
	}
	if x.ProjectID != nil {
		c.ProjectID = *x.ProjectID
	}
	if x.ModelID != nil {
		c.ModelID = *x.ModelID
	}
	if x.CollectionID != nil {
		c.CollectionID = *x.CollectionID
	}
	if x.RAGMode != nil || x.RAGTopK != nil {
		mode, topK, settingErr := ragSettings(c.RAGMode, c.RAGTopK)
		if x.RAGMode != nil {
			mode = *x.RAGMode
		}
		if x.RAGTopK != nil {
			topK = *x.RAGTopK
		}
		mode, topK, settingErr = ragSettings(mode, topK)
		if settingErr != nil {
			errJSON(w, 400, settingErr.Error())
			return
		}
		c.RAGMode, c.RAGTopK = mode, topK
	}
	if x.RAGEnabled != nil {
		if c.CollectionID == "" {
			errJSON(w, http.StatusConflict, "attach a RAG collection before enabling or disabling RAG retrieval")
			return
		}
		c.RAGEnabled = x.RAGEnabled
	}
	if x.WebEnabled != nil {
		c.WebEnabled = *x.WebEnabled
	}
	if x.Pinned != nil {
		c.Pinned = *x.Pinned
	}
	c.UpdatedAt = store.Now()
	if e = s.Store.UpdateChat(r.Context(), c); e != nil {
		errJSON(w, 500, e.Error())
		return
	}
	write(w, 200, c)
}
func (s *Server) cancelChat(w http.ResponseWriter, r *http.Request) {
	chatID := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/api/v1/chats/"), "/cancel")
	s.cancelRunningChat(chatID)
	messages, err := s.Store.Messages(r.Context(), chatID)
	if err != nil {
		errJSON(w, 500, err.Error())
		return
	}
	cancelled := 0
	for _, message := range messages {
		if message.Role != "assistant" || message.Status != "working" {
			continue
		}
		message.Status = "cancelled"
		message.Content = "Generation cancelled."
		if err := s.Store.UpdateMessage(context.Background(), message); err != nil {
			errJSON(w, 500, err.Error())
			return
		}
		s.publishChatStream(chatID, message)
		cancelled++
	}
	write(w, http.StatusOK, map[string]int{"cancelled": cancelled})
}
func (s *Server) cancelRunningChat(chatID string) {
	s.asyncMu.Lock()
	defer s.asyncMu.Unlock()
	for _, request := range s.asyncCancel {
		if request.chatID == chatID {
			request.cancel()
		}
	}
}
func (s *Server) chat(w http.ResponseWriter, r *http.Request) {
	var x chatInput
	if !decode(w, r, &x) {
		return
	}
	if strings.TrimSpace(x.Prompt) == "" && len(x.Parts) == 0 {
		errJSON(w, 400, "prompt or content parts are required")
		return
	}
	if err := domain.ValidateContentParts(x.Parts); err != nil {
		errJSON(w, 400, "invalid content parts: "+err.Error())
		return
	}
	var activeChat domain.Chat
	messages := []inference.Message{}
	if x.ChatID != "" {
		var chatErr error
		activeChat, chatErr = s.Store.Chat(r.Context(), x.ChatID)
		if chatErr != nil {
			errJSON(w, 404, chatErr.Error())
			return
		}
		if activeChat.CLISessionActive {
			errJSON(w, http.StatusConflict, "this chat is active in the CLI session bound to "+activeChat.WorkspacePath+"; close it from that folder before continuing in the web app")
			return
		}
		if x.ModelID == "" {
			x.ModelID = activeChat.ModelID
		}
		if x.CollectionID == "" {
			x.CollectionID = activeChat.CollectionID
		}
		if x.RAGMode == "" {
			x.RAGMode = activeChat.RAGMode
		}
		if x.RAGTopK == 0 {
			x.RAGTopK = activeChat.RAGTopK
		}
		if x.WebSearch == nil {
			value := activeChat.WebEnabled
			x.WebSearch = &value
		}
		if x.RAGEnabled == nil {
			value := activeChat.CollectionID != ""
			if activeChat.RAGEnabled != nil {
				value = *activeChat.RAGEnabled
			}
			x.RAGEnabled = &value
		}
		x.projectID = activeChat.ProjectID
		history, historyErr := s.Store.Messages(r.Context(), x.ChatID)
		if historyErr != nil {
			errJSON(w, 500, historyErr.Error())
			return
		}
		for _, message := range history {
			// A cancelled/failed background generation is UI state, not useful
			// conversational context for a later retry.
			if message.Role == "assistant" && (message.Status == "working" || message.Status == "failed") {
				continue
			}
			messages = append(messages, inference.Message{Role: message.Role, Content: message.Content, Parts: message.Parts})
		}
	}
	if x.ModelID == "" {
		if defaultModel, ok, defaultErr := s.Store.DefaultBaseModel(r.Context()); defaultErr != nil {
			errJSON(w, 500, defaultErr.Error())
			return
		} else if ok {
			x.ModelID = defaultModel.ID
		}
	}
	endpoint := x.Endpoint
	x.remoteModelID = x.ModelID
	selectedModalities := []string{"text"}
	if x.ModelID != "" {
		models, e := s.Store.Models(r.Context())
		if e != nil {
			errJSON(w, 500, e.Error())
			return
		}
		for _, m := range models {
			if m.ID != x.ModelID {
				continue
			}
			if m.Endpoint != "" {
				endpoint = m.Endpoint
			}
			if s.Runners != nil && m.Provider == "" && m.Endpoint == "" {
				runnerModelID := m.ID
				// A native LoRA adapter is applied per request by the Qwen
				// runner. It therefore shares its registered parent base
				// runner instead of requiring a duplicate loaded checkpoint.
				if m.Kind == "slm" && m.ParentModel != "" {
					runnerModelID = m.ParentModel
				}
				managedEndpoint, active := s.Runners.Endpoint(runnerModelID)
				if !active {
					errJSON(w, http.StatusConflict, "local model is not active; start it from the Models catalog before sending a chat")
					return
				}
				endpoint = managedEndpoint
			}
			if len(m.Modalities) > 0 {
				selectedModalities = m.Modalities
			}
			if m.Format == adapter.QwenLMHeadLoRAType || m.Format == adapter.TransformerLoRAType {
				x.adapter = m.Path
			}
			if m.RemoteID != "" {
				x.remoteModelID = providerModelID(m.Provider, m.RemoteID)
			}
			if m.Provider != "" {
				x.provider = m.Provider
				x.apiKey = s.providerAPIKey(m.Provider)
				if x.apiKey == "" {
					errJSON(w, 400, "no API key configured for "+m.Provider+"; add the provider key to .env and restart the API")
					return
				}
			}
		}
	}
	if err := supportsParts(selectedModalities, x.Parts); err != nil {
		errJSON(w, http.StatusBadRequest, err.Error())
		return
	}
	rawPrompt := x.Prompt
	x.Prompt = domain.ContentText(rawPrompt, x.Parts)
	messages = append(messages, inference.Message{Role: "user", Content: rawPrompt, Parts: x.Parts})
	s.logger().Info("chat request prepared", "chatID", x.ChatID, "modelID", x.ModelID, "remoteModelID", x.remoteModelID, "provider", x.provider, "endpoint", endpoint, "historyMessages", len(messages)-1, "promptChars", len([]rune(x.Prompt)), "async", x.Async)
	sources := []domain.Chunk{}
	useRAG := x.CollectionID != "" && (x.RAGEnabled == nil || *x.RAGEnabled)
	if useRAG {
		mode, topK, settingsErr := ragSettings(x.RAGMode, x.RAGTopK)
		if settingsErr != nil {
			errJSON(w, 400, settingsErr.Error())
			return
		}
		x.RAGMode, x.RAGTopK = mode, topK
		var e error
		if mode == "full" {
			sources, e = s.RAG.All(r.Context(), x.CollectionID)
		} else {
			sources, e = s.RAG.Retrieve(r.Context(), x.CollectionID, x.Prompt, topK)
		}
		if e != nil {
			s.logger().Error("RAG retrieval failed", "chatID", x.ChatID, "collectionID", x.CollectionID, "error", e)
			errJSON(w, 400, e.Error())
			return
		}
		s.logger().Info("RAG retrieval completed", "chatID", x.ChatID, "collectionID", x.CollectionID, "mode", mode, "sourceCount", len(sources), "contextChars", chunkCharacters(sources))
		var packed bool
		messages, sources, packed = s.fitLocalRAGContext(r.Context(), endpoint, messages, sources)
		if packed {
			s.logger().Info("packed RAG context for local model", "chatID", x.ChatID, "sourceCount", len(sources))
		}
	}
	useWeb := x.WebSearch != nil && *x.WebSearch
	if explicitWebRequest(x.Prompt) {
		useWeb = true
	}
	if useWeb {
		s.logger().Info("web retrieval starting", "chatID", x.ChatID, "query", x.Prompt)
		results, webErr := s.Web.Search(r.Context(), x.Prompt)
		if webErr != nil {
			s.logger().Error("web retrieval failed", "chatID", x.ChatID, "error", webErr)
			errJSON(w, 502, webErr.Error())
			return
		}
		webParts := make([]string, 0, len(results))
		for _, result := range results {
			if strings.TrimSpace(result.Content) == "" {
				continue
			}
			source := "Web: " + result.Title + " — " + result.URL
			sources = append(sources, domain.Chunk{Source: source, Content: result.Content})
			webParts = append(webParts, fmt.Sprintf("[%s]\n%s", source, result.Content))
		}
		if len(webParts) == 0 {
			s.logger().Warn("web retrieval returned no readable sources", "chatID", x.ChatID, "resultCount", len(results))
			errJSON(w, 502, "web search returned no readable sources")
			return
		}
		webContext := "Use these live web sources when relevant. Cite source names in square brackets. Web sources:\n" + strings.Join(webParts, "\n---\n")
		var packed bool
		messages, packed = s.fitLocalWebContext(r.Context(), endpoint, messages, webContext)
		s.logger().Info("web retrieval completed", "chatID", x.ChatID, "resultCount", len(results), "readableSourceCount", len(webParts), "retrievedChars", chunkCharacters(sources), "promptContextPacked", packed)
	}
	if x.Async {
		if x.ChatID == "" {
			errJSON(w, 400, "async chat requires chatID")
			return
		}
		now := store.Now()
		assistant := domain.ChatMessage{ID: id(), ChatID: x.ChatID, Role: "assistant", Content: "Working…", Status: "working", CreatedAt: now}
		if e := s.Store.AddMessage(context.Background(), domain.ChatMessage{ID: id(), ChatID: x.ChatID, Role: "user", Content: x.Prompt, Parts: x.Parts, CreatedAt: now}); e != nil {
			errJSON(w, 500, e.Error())
			return
		}
		if e := s.Store.AddMessage(context.Background(), assistant); e != nil {
			errJSON(w, 500, e.Error())
			return
		}
		s.publishChatStream(assistant.ChatID, assistant)
		activeChat.ModelID, activeChat.CollectionID = x.ModelID, x.CollectionID
		activeChat.RAGMode, activeChat.RAGTopK = x.RAGMode, x.RAGTopK
		activeChat.RAGEnabled = x.RAGEnabled
		activeChat.WebEnabled = x.WebSearch != nil && *x.WebSearch
		activeChat.UpdatedAt = now
		if activeChat.Title == "New chat" {
			activeChat.Title = shortTitle(x.Prompt)
		}
		if e := s.Store.UpdateChat(context.Background(), activeChat); e != nil {
			errJSON(w, 500, e.Error())
			return
		}
		runCtx, cancel := context.WithCancel(context.Background())
		s.asyncMu.Lock()
		if s.asyncCancel == nil {
			s.asyncCancel = map[string]asyncRequest{}
		}
		s.asyncCancel[assistant.ID] = asyncRequest{chatID: assistant.ChatID, cancel: cancel}
		s.asyncMu.Unlock()
		go s.finishAsyncChat(runCtx, endpoint, x, messages, sources, assistant)
		s.logger().Info("async chat queued", "chatID", x.ChatID, "messageID", assistant.ID, "modelID", x.ModelID, "sourceCount", len(sources))
		write(w, http.StatusAccepted, map[string]any{"status": "working", "messageID": assistant.ID})
		return
	}
	out, e := s.completeWithTools(r.Context(), endpoint, x, messages)
	if e != nil {
		s.logger().Error("chat inference failed", "chatID", x.ChatID, "modelID", x.ModelID, "endpoint", endpoint, "error", e)
		errJSON(w, inferenceErrorStatus(e), e.Error())
		return
	}
	s.logger().Info("chat inference completed", "chatID", x.ChatID, "modelID", x.ModelID, "remoteModelID", x.remoteModelID, "provider", x.provider, "inputTokens", out.InputTokens, "outputTokens", out.OutputTokens, "latency", out.Latency)
	_ = s.Store.AddUsage(context.Background(), domain.UsageEvent{ID: id(), Kind: "chat", ModelID: x.ModelID, InputTokens: out.InputTokens, OutputTokens: out.OutputTokens, LatencyMS: out.Latency.Milliseconds(), CreatedAt: store.Now()})
	if x.ChatID != "" {
		now := store.Now()
		_ = s.Store.AddMessage(context.Background(), domain.ChatMessage{ID: id(), ChatID: x.ChatID, Role: "user", Content: x.Prompt, Parts: x.Parts, CreatedAt: now})
		_ = s.Store.AddMessage(context.Background(), domain.ChatMessage{ID: id(), ChatID: x.ChatID, Role: "assistant", Content: out.Content, Parts: out.Parts, Sources: sourceNames(sources), LatencyMS: out.Latency.Milliseconds(), CreatedAt: now})
		activeChat.ModelID = x.ModelID
		activeChat.CollectionID = x.CollectionID
		activeChat.RAGMode = x.RAGMode
		activeChat.RAGTopK = x.RAGTopK
		activeChat.RAGEnabled = x.RAGEnabled
		activeChat.WebEnabled = x.WebSearch != nil && *x.WebSearch
		activeChat.UpdatedAt = now
		if activeChat.Title == "New chat" {
			activeChat.Title = shortTitle(x.Prompt)
		}
		_ = s.Store.UpdateChat(context.Background(), activeChat)
	}
	write(w, 200, map[string]any{"content": out.Content, "parts": out.Parts, "sources": sources, "usage": map[string]any{"inputTokens": out.InputTokens, "outputTokens": out.OutputTokens, "latencyMs": out.Latency.Milliseconds()}})
}
func (s *Server) finishAsyncChat(ctx context.Context, endpoint string, x chatInput, messages []inference.Message, sources []domain.Chunk, assistant domain.ChatMessage) {
	defer func() { s.asyncMu.Lock(); delete(s.asyncCancel, assistant.ID); s.asyncMu.Unlock() }()
	s.logger().Info("async chat inference starting", "chatID", assistant.ChatID, "messageID", assistant.ID, "modelID", x.ModelID, "endpoint", endpoint, "messageCount", len(messages), "sourceCount", len(sources))
	request := inference.ChatRequest{Model: x.remoteModelID, Messages: messages, Temperature: x.Temperature, MaxTokens: x.MaxTokens, APIKey: x.apiKey, Adapter: x.adapter}
	var out inference.ChatResponse
	var err error
	if x.ToolsEnabled {
		out, err = s.completeWithTools(ctx, endpoint, x, messages)
	} else if strings.HasPrefix(endpoint, "http://127.0.0.1:") || strings.HasPrefix(endpoint, "http://localhost:") {
		partial, lastPersist := "", time.Time{}
		out, err = s.Inference.ChatStream(ctx, endpoint, request, func(delta string) {
			partial += delta
			assistant.Content = partial
			assistant.Status = "working"
			s.publishChatStream(assistant.ChatID, assistant)
			if time.Since(lastPersist) < 150*time.Millisecond {
				return
			}
			lastPersist = time.Now()
			_ = s.Store.UpdateMessage(context.Background(), assistant)
		})
	} else {
		out, err = s.Inference.Chat(ctx, endpoint, request)
	}
	if err != nil {
		if ctx.Err() != nil {
			assistant.Content, assistant.Status = "Generation cancelled.", "cancelled"
			_ = s.Store.UpdateMessage(context.Background(), assistant)
			s.publishChatStream(assistant.ChatID, assistant)
			return
		}
		s.logger().Error("async chat inference failed", "chatID", assistant.ChatID, "messageID", assistant.ID, "modelID", x.ModelID, "endpoint", endpoint, "error", err)
		assistant.Content = "Unable to get a response: " + err.Error()
		assistant.Status = "failed"
		_ = s.Store.UpdateMessage(context.Background(), assistant)
		s.publishChatStream(assistant.ChatID, assistant)
		return
	}
	assistant.Content = out.Content
	assistant.Parts = out.Parts
	assistant.Sources = sourceNames(sources)
	assistant.Status = "completed"
	assistant.LatencyMS = out.Latency.Milliseconds()
	_ = s.Store.UpdateMessage(context.Background(), assistant)
	s.publishChatStream(assistant.ChatID, assistant)
	_ = s.Store.AddUsage(context.Background(), domain.UsageEvent{ID: id(), Kind: "chat", ModelID: x.ModelID, InputTokens: out.InputTokens, OutputTokens: out.OutputTokens, LatencyMS: out.Latency.Milliseconds(), CreatedAt: store.Now()})
	s.logger().Info("async chat inference completed", "chatID", assistant.ChatID, "messageID", assistant.ID, "modelID", x.ModelID, "inputTokens", out.InputTokens, "outputTokens", out.OutputTokens, "latency", out.Latency)
}

// completeWithTools runs the provider's native function-calling loop. The
// model only receives tools permitted for this model/project; every returned
// call is schema-validated and executed by the agent service. A policy that
// needs approval becomes an auditable pending run rather than an implicit
// execution. Eight rounds bound recursive/tool-driven prompt growth.
func (s *Server) completeWithTools(ctx context.Context, endpoint string, x chatInput, messages []inference.Message) (inference.ChatResponse, error) {
	request := inference.ChatRequest{Model: x.remoteModelID, Messages: messages, Temperature: x.Temperature, MaxTokens: x.MaxTokens, APIKey: x.apiKey, Adapter: x.adapter}
	if !x.ToolsEnabled || s.Agent == nil {
		return s.Inference.Chat(ctx, endpoint, request)
	}
	catalog, err := s.Agent.Callable(ctx, x.ModelID, x.projectID)
	if err != nil {
		return inference.ChatResponse{}, err
	}
	if len(catalog) == 0 {
		return s.Inference.Chat(ctx, endpoint, request)
	}
	byName := make(map[string]string, len(catalog))
	request.Tools = make([]inference.ToolDefinition, 0, len(catalog))
	for _, tool := range catalog {
		name := wireToolName(tool.ID)
		byName[name] = tool.ID
		request.Tools = append(request.Tools, inference.ToolDefinition{Type: "function", Function: inference.ToolFunction{Name: name, Description: tool.Description, Parameters: tool.InputSchema}})
	}
	request.ToolChoice = "auto"
	var total inference.ChatResponse
	for round := 0; round < 8; round++ {
		out, callErr := s.Inference.Chat(ctx, endpoint, request)
		if callErr != nil {
			return inference.ChatResponse{}, callErr
		}
		total.InputTokens += out.InputTokens
		total.OutputTokens += out.OutputTokens
		total.Latency += out.Latency
		if len(out.ToolCalls) == 0 {
			out.InputTokens, out.OutputTokens, out.Latency = total.InputTokens, total.OutputTokens, total.Latency
			return out, nil
		}
		request.Messages = append(request.Messages, inference.Message{Role: "assistant", Content: out.Content, ToolCalls: out.ToolCalls})
		for _, call := range out.ToolCalls {
			toolID, known := byName[call.Name]
			result := ""
			if !known {
				result = "Tool call rejected: unknown tool " + call.Name
			} else {
				run, runErr := s.Agent.Execute(ctx, x.ModelID, x.projectID, toolID, call.Arguments, false)
				if runErr != nil {
					result = "Tool call rejected: " + runErr.Error()
				} else if run.Status == "awaiting_approval" {
					result = "Tool call is awaiting user approval. Run ID: " + run.ID
				} else if run.Error != "" {
					result = "Tool call failed: " + run.Error
				} else {
					result = run.Output
				}
			}
			callID := call.ID
			if callID == "" {
				callID = id()
			}
			request.Messages = append(request.Messages, inference.Message{Role: "tool", ToolCallID: callID, Content: result})
		}
	}
	return inference.ChatResponse{}, fmt.Errorf("tool-calling loop reached its maximum of 8 rounds")
}

func wireToolName(toolID string) string {
	replacer := strings.NewReplacer("/", "_", ".", "_", "-", "_")
	return "sypra_" + replacer.Replace(toolID)
}
func explicitWebRequest(prompt string) bool {
	text := strings.ToLower(prompt)
	for _, phrase := range []string{"search the web", "search web", "search the internet", "search online", "web search", "look up online", "browse the web", "on the internet"} {
		if strings.Contains(text, phrase) {
			return true
		}
	}
	return false
}
func inferenceErrorStatus(err error) int {
	// A full local-model context is a valid, actionable request error. Preserve
	// it instead of reporting it as an opaque upstream/server failure.
	if strings.Contains(strings.ToLower(err.Error()), "context limit") {
		return http.StatusUnprocessableEntity
	}
	return http.StatusBadGateway
}
func cloudProviderEndpoint(provider string) (string, bool) {
	switch provider {
	case "openai":
		return "https://api.openai.com/v1", true
	case "xai":
		return "https://api.x.ai/v1", true
	case "gemini":
		return "https://generativelanguage.googleapis.com/v1beta/openai", true
	default:
		return "", false
	}
}
func (s *Server) providerAPIKey(provider string) string {
	if s.ProviderAPIKeys == nil {
		return ""
	}
	return s.ProviderAPIKeys[strings.ToLower(provider)]
}
func providerModelID(provider, modelID string) string {
	modelID = strings.TrimSpace(modelID)
	if strings.ToLower(provider) != "gemini" {
		return modelID
	}
	// Gemini's API model identifiers use lowercase hyphenated names. Accepting
	// a display label here also repairs cloud models created before the UI made
	// that distinction explicit (for example, "Gemini 2.5 Pro").
	return strings.ToLower(strings.Join(strings.Fields(modelID), "-"))
}
func normalizeModalities(values []string) []string {
	allowed := map[string]bool{"text": true, "image": true, "audio": true, "video": true, "file": true}
	seen, result := map[string]bool{}, []string{}
	for _, value := range values {
		value = strings.ToLower(strings.TrimSpace(value))
		if allowed[value] && !seen[value] {
			seen[value] = true
			result = append(result, value)
		}
	}
	if len(result) == 0 {
		return []string{"text"}
	}
	return result
}
func supportsParts(modalities []string, parts []domain.ContentPart) error {
	available := map[string]bool{}
	for _, modality := range modalities {
		available[strings.ToLower(modality)] = true
	}
	for _, part := range parts {
		required := map[string]string{"image_url": "image", "input_audio": "audio", "video_url": "video", "file": "file"}[part.Type]
		if required != "" && !available[required] {
			return fmt.Errorf("the selected model does not advertise %s input support; choose a compatible endpoint or update its modalities", required)
		}
	}
	return nil
}
func ragSettings(mode string, topK int) (string, int, error) {
	mode = strings.ToLower(strings.TrimSpace(mode))
	if mode == "" {
		mode = "focused"
	}
	if mode != "focused" && mode != "full" {
		return "", 0, fmt.Errorf("ragMode must be focused or full")
	}
	if topK < 0 {
		return "", 0, fmt.Errorf("ragTopK cannot be negative")
	}
	if topK == 0 && mode == "focused" {
		// A useful default should be selective. Zero continues to mean all
		// chunks only for the explicit full mode and diagnostic RAG endpoint.
		topK = 8
	}
	return mode, topK, nil
}
func sourceNames(chunks []domain.Chunk) []string {
	seen, out := map[string]bool{}, []string{}
	for _, chunk := range chunks {
		if !seen[chunk.Source] {
			seen[chunk.Source] = true
			out = append(out, chunk.Source)
		}
	}
	return out
}
func chunkCharacters(chunks []domain.Chunk) int {
	total := 0
	for _, chunk := range chunks {
		total += len([]rune(chunk.Content))
	}
	return total
}

func ragContext(chunks []domain.Chunk) string {
	parts := make([]string, len(chunks))
	for i, chunk := range chunks {
		parts[i] = fmt.Sprintf("[%s | passage %d]\n%s", chunk.Source, chunk.Position+1, chunk.Content)
	}
	return "Answer using the supplied sources when they are relevant. Do not invent facts outside them. If the sources do not answer the question, say so plainly. Cite the source name in square brackets for every factual claim grounded in this context. Sources:\n" + strings.Join(parts, "\n---\n")
}

// fitLocalRAGContext uses the active local runner's tokenizer and context
// window to select whole retrieved passages. It is not a character limit: the
// actual model's token budget decides how many ranked passages can be used.
func (s *Server) fitLocalRAGContext(ctx context.Context, endpoint string, messages []inference.Message, chunks []domain.Chunk) ([]inference.Message, []domain.Chunk, bool) {
	withRAG := func(passages []domain.Chunk) []inference.Message {
		if len(passages) == 0 {
			return messages
		}
		return append([]inference.Message{{Role: "system", Content: ragContext(passages)}}, messages...)
	}
	full := withRAG(chunks)
	if strings.TrimSpace(endpoint) == "" || len(chunks) == 0 {
		return full, chunks, false
	}
	total, limit, supported, err := s.Inference.CountTokens(ctx, endpoint, full)
	if err != nil || !supported {
		if err != nil {
			s.logger().Warn("local RAG context tokenizer failed; using retrieved passages", "endpoint", endpoint, "error", err)
		}
		return full, chunks, false
	}
	target := limit - maxInt(1, limit/8)
	if total <= target {
		return full, chunks, false
	}
	baseTokens, _, baseSupported, baseErr := s.Inference.CountTokens(ctx, endpoint, messages)
	if baseErr != nil || !baseSupported || baseTokens >= target {
		s.logger().Warn("local model context has no capacity for RAG passages", "endpoint", endpoint, "promptTokens", total, "baseTokens", baseTokens, "contextTokens", limit)
		return messages, nil, true
	}
	kept := make([]domain.Chunk, 0, len(chunks))
	for _, chunk := range chunks {
		candidate := append(append([]domain.Chunk{}, kept...), chunk)
		candidateMessages := withRAG(candidate)
		tokens, _, ok, countErr := s.Inference.CountTokens(ctx, endpoint, candidateMessages)
		if countErr != nil || !ok {
			break
		}
		if tokens <= target {
			kept = candidate
		}
	}
	if len(kept) == 0 {
		return messages, nil, true
	}
	return withRAG(kept), kept, true
}

// fitLocalWebContext retains all downloaded sources for the request and packs
// the text sent to the local runner into its actual context window. A
// model cannot generate any answer when its prompt exceeds that fixed hardware/
// checkpoint boundary. Cloud endpoints and third-party local servers keep their
// original context untouched because they do not expose Sypra's tokenizer API.
func (s *Server) fitLocalWebContext(ctx context.Context, endpoint string, messages []inference.Message, webContext string) ([]inference.Message, bool) {
	withWeb := func(content string) []inference.Message {
		return append([]inference.Message{{Role: "system", Content: content}}, messages...)
	}
	full := withWeb(webContext)
	// An implicit default may be any OpenAI-compatible endpoint. Only probe an
	// explicitly selected local runner for Sypra's optional /tokenize route.
	if strings.TrimSpace(endpoint) == "" {
		return full, false
	}
	total, limit, supported, err := s.Inference.CountTokens(ctx, endpoint, full)
	if err != nil {
		s.logger().Warn("local web context tokenizer failed; using complete context", "endpoint", endpoint, "error", err)
		return full, false
	}
	if !supported {
		return full, false
	}
	// Leave one eighth of the model window for generation. This is derived from
	// the active model capacity, not an application-wide retrieval cap.
	target := limit - maxInt(1, limit/8)
	if total <= target {
		return full, false
	}
	baseTokens, _, baseSupported, baseErr := s.Inference.CountTokens(ctx, endpoint, messages)
	if baseErr != nil || !baseSupported || baseTokens >= target {
		s.logger().Warn("local model context has no capacity for web text", "endpoint", endpoint, "promptTokens", total, "baseTokens", baseTokens, "contextTokens", limit)
		return messages, true
	}
	fullRunes := []rune(webContext)
	webTokens := total - baseTokens
	allowedTokens := target - baseTokens
	chars := len(fullRunes) * allowedTokens / maxInt(1, webTokens)
	chars = minInt(len(fullRunes), maxInt(1, chars))
	for attempts := 0; attempts < 6; attempts++ {
		content := truncateRunes(webContext, chars)
		candidate := withWeb(content)
		candidateTokens, _, ok, countErr := s.Inference.CountTokens(ctx, endpoint, candidate)
		if countErr != nil || !ok {
			break
		}
		if candidateTokens <= target {
			s.logger().Info("packed web context for local model", "contextTokens", limit, "originalPromptTokens", total, "packedPromptTokens", candidateTokens, "originalChars", len(fullRunes), "packedChars", chars)
			return candidate, true
		}
		chars = maxInt(1, chars*target/candidateTokens)
	}
	// A final, small context preserves a useful live-source signal even when
	// source markup tokenizes more densely than plain text.
	return withWeb(truncateRunes(webContext, maxInt(1, len(fullRunes)/16))), true
}
func truncateRunes(value string, count int) string {
	runes := []rune(value)
	if count >= len(runes) {
		return value
	}
	return string(runes[:count]) + "\n[Additional retrieved web content was omitted because it does not fit this model's context window.]"
}
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
func shortTitle(v string) string {
	v = strings.TrimSpace(v)
	r := []rune(v)
	if len(r) > 42 {
		return string(r[:42]) + "…"
	}
	return v
}
func (s *Server) jobs(w http.ResponseWriter, r *http.Request) {
	v, e := s.Store.Jobs(r.Context())
	respond(w, v, e)
}
func (s *Server) startJob(w http.ResponseWriter, r *http.Request) {
	var x struct {
		Name, BaseModel, DatasetPath, Method, Command string
		Parameters                                    map[string]any
	}
	if !decode(w, r, &x) {
		return
	}
	if x.Name == "" || x.BaseModel == "" || x.DatasetPath == "" {
		errJSON(w, 400, "name, baseModel, and datasetPath are required")
		return
	}
	if strings.HasPrefix(x.BaseModel, "cloud://") {
		errJSON(w, 400, "cloud chat models cannot be used for native local fine-tuning")
		return
	}
	if x.Parameters == nil {
		x.Parameters = map[string]any{}
	}
	if x.Method == "" {
		x.Method = "lora"
	}
	j := domain.TrainingJob{ID: id(), Name: x.Name, BaseModel: x.BaseModel, DatasetPath: x.DatasetPath, Method: x.Method, Parameters: x.Parameters, Status: "queued", CreatedAt: store.Now()}
	if x.Command == "" {
		if x.Method != "lora" && x.Method != "qlora" {
			errJSON(w, 400, "native fine-tuning supports lora or qlora")
			return
		}
		cfg := finetune.DefaultConfig()
		cfg.Method, cfg.ModelDir = x.Method, x.BaseModel
		if v, ok := x.Parameters["epochs"].(float64); ok {
			cfg.Epochs = int(v)
		}
		if v, ok := x.Parameters["batchSize"].(float64); ok {
			cfg.BatchSize, cfg.EvalBatchSize = int(v), int(v)
		}
		if v, ok := x.Parameters["learningRate"].(float64); ok {
			cfg.LearningRate = v
		}
		if v, ok := x.Parameters["loraRank"].(float64); ok {
			cfg.LoRARank = int(v)
		}
		if v, ok := x.Parameters["loraAlpha"].(float64); ok {
			cfg.LoRAAlpha = int(v)
		}
		if v, ok := x.Parameters["seed"].(float64); ok && v >= 0 {
			cfg.Seed = uint64(v)
		}
		if v, ok := x.Parameters["checkpointEvery"].(float64); ok {
			cfg.CheckpointEvery = int(v)
		}
		if v, ok := x.Parameters["earlyStoppingPatience"].(float64); ok {
			cfg.EarlyStoppingPatience = int(v)
		}
		if v, ok := x.Parameters["minValidationImprovement"].(float64); ok {
			cfg.MinValidationImprovement = v
		}
		if v, ok := x.Parameters["resumeFrom"].(string); ok {
			cfg.ResumeFrom = v
		}
		if v, ok := x.Parameters["weightDecay"].(float64); ok {
			cfg.WeightDecay = v
		}
		if v, ok := x.Parameters["maxGradientNorm"].(float64); ok {
			cfg.MaxGradientNorm = v
		}
		if v, ok := x.Parameters["maxSequenceLength"].(float64); ok {
			cfg.MaxSequenceLength = int(v)
		}
		if v, ok := x.Parameters["computeBackend"].(string); ok {
			cfg.ComputeBackend = strings.ToLower(strings.TrimSpace(v))
		}
		if values, ok := x.Parameters["targetModules"].([]any); ok {
			cfg.TargetModules = cfg.TargetModules[:0]
			for _, value := range values {
				if target, ok := value.(string); ok && strings.TrimSpace(target) != "" {
					cfg.TargetModules = append(cfg.TargetModules, strings.TrimSpace(target))
				}
			}
		}
		if values, ok := x.Parameters["gpuDevices"].([]any); ok {
			cfg.GPUDevices = cfg.GPUDevices[:0]
			for _, value := range values {
				if device, ok := value.(float64); ok && device >= 0 {
					cfg.GPUDevices = append(cfg.GPUDevices, int(device))
				}
			}
		}
		if values, ok := x.Parameters["workerEndpoints"].([]any); ok {
			cfg.WorkerEndpoints = cfg.WorkerEndpoints[:0]
			for _, value := range values {
				if endpoint, ok := value.(string); ok && strings.TrimSpace(endpoint) != "" {
					cfg.WorkerEndpoints = append(cfg.WorkerEndpoints, strings.TrimRight(strings.TrimSpace(endpoint), "/"))
				}
			}
		}
		directory := filepath.Join(filepath.Dir(s.UploadDir), "runs", j.ID)
		if e := s.Trainer.StartNative(r.Context(), j, cfg, directory); e != nil {
			errJSON(w, 400, "start native training: "+e.Error())
			return
		}
	} else if e := s.Trainer.Start(r.Context(), j, x.Command); e != nil {
		errJSON(w, 400, e.Error())
		return
	}
	write(w, 202, j)
}
func (s *Server) cancelJob(w http.ResponseWriter, r *http.Request) {
	p := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/api/v1/training/jobs/"), "/cancel")
	if !s.Trainer.Cancel(p) {
		errJSON(w, 404, "job is not running")
		return
	}
	write(w, 202, map[string]string{"status": "cancelling"})
}
func (s *Server) metrics(w http.ResponseWriter, r *http.Request) {
	v, e := s.Store.Summary(r.Context())
	respond(w, v, e)
}
func (s *Server) diagnosticBundle(w http.ResponseWriter, r *http.Request) {
	summary, err := s.Store.Summary(r.Context())
	if err != nil {
		s.logger().Error("build diagnostic summary", "error", err)
		errJSON(w, http.StatusInternalServerError, "unable to build diagnostic bundle")
		return
	}
	jobs, err := s.Store.Jobs(r.Context())
	if err != nil {
		s.logger().Error("build diagnostic jobs", "error", err)
		errJSON(w, http.StatusInternalServerError, "unable to build diagnostic bundle")
		return
	}
	type jobStatus struct {
		ID, Status, Phase string
		Progress          float64
		CreatedAt         time.Time
	}
	safeJobs := make([]jobStatus, 0, len(jobs))
	for _, job := range jobs {
		safeJobs = append(safeJobs, jobStatus{ID: job.ID, Status: job.Status, Phase: job.Phase, Progress: job.Progress, CreatedAt: job.CreatedAt})
	}
	w.Header().Set("Content-Type", "application/zip")
	w.Header().Set("Content-Disposition", `attachment; filename="sypra-diagnostics.zip"`)
	archive := zip.NewWriter(w)
	defer archive.Close()
	writeEntry := func(name string, value any) {
		entry, createErr := archive.Create(name)
		if createErr == nil {
			_ = json.NewEncoder(entry).Encode(value)
		}
	}
	writeEntry("manifest.json", map[string]any{"createdAt": time.Now().UTC(), "redacted": true, "schema": 1})
	writeEntry("usage-summary.json", summary)
	writeEntry("training-status.json", safeJobs)
	if s.Runners != nil {
		writeEntry("runner-status.json", s.Runners.Statuses())
	}
}
func (s *Server) clientLog(w http.ResponseWriter, r *http.Request) {
	var entry struct {
		Level, Event, At, Method, URL, StatusText, Error string
		Status, DurationMs                               int
	}
	if !decode(w, r, &entry) {
		return
	}
	s.logger().Error("UI client diagnostic", "level", entry.Level, "event", entry.Event, "at", entry.At, "method", entry.Method, "url", entry.URL, "status", entry.Status, "statusText", entry.StatusText, "durationMs", entry.DurationMs, "error", entry.Error)
	w.WriteHeader(http.StatusNoContent)
}
func (s *Server) cors(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", s.CORSOrigin)
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
		if r.Method == "OPTIONS" {
			w.WriteHeader(204)
			return
		}
		next.ServeHTTP(w, r)
	})
}
func (s *Server) log(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		traceID := observability.TraceID(r)
		telemetryRequest := s.Telemetry.BeginHTTP(r)
		if telemetryRequest != nil {
			traceID = telemetryRequest.TraceID()
			w.Header().Set("traceparent", telemetryRequest.Traceparent())
		}
		w.Header().Set("X-Request-ID", traceID)
		response := &statusResponseWriter{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(response, r)
		duration := time.Since(start)
		if telemetryRequest != nil {
			telemetryRequest.End(response.status, duration)
		}
		if s.Metrics != nil {
			s.Metrics.Observe(r.Method, r.URL.Path, response.status, duration)
		}
		attributes := []any{"traceID", traceID, "method", r.Method, "path", r.URL.Path, "status", response.status, "bytes", response.bytes, "duration", duration}
		if response.status >= http.StatusInternalServerError {
			s.logger().Error("API request completed with server error", attributes...)
		} else if response.status >= http.StatusBadRequest {
			s.logger().Warn("API request completed with client error", attributes...)
		} else {
			s.logger().Info("API request completed", attributes...)
		}
	})
}
func (s *Server) recover(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if recovered := recover(); recovered != nil {
				s.logger().Error("unhandled API panic", "traceID", w.Header().Get("X-Request-ID"), "panic", recovered, "stack", string(debug.Stack()))
				errJSON(w, http.StatusInternalServerError, "internal server error")
			}
		}()
		next.ServeHTTP(w, r)
	})
}

type statusResponseWriter struct {
	http.ResponseWriter
	status int
	bytes  int
}

func (w *statusResponseWriter) WriteHeader(status int) {
	w.status = status
	w.ResponseWriter.WriteHeader(status)
}
func (w *statusResponseWriter) Write(data []byte) (int, error) {
	n, err := w.ResponseWriter.Write(data)
	w.bytes += n
	return n, err
}
func (s *Server) logger() *slog.Logger {
	if s.Logger != nil {
		return s.Logger
	}
	return slog.Default()
}
func decode(w http.ResponseWriter, r *http.Request, v any) bool {
	d := json.NewDecoder(r.Body)
	d.DisallowUnknownFields()
	if e := d.Decode(v); e != nil {
		errJSON(w, 400, "invalid JSON: "+e.Error())
		return false
	}
	return true
}
func respond(w http.ResponseWriter, v any, e error) {
	if e != nil {
		errJSON(w, 500, e.Error())
		return
	}
	write(w, 200, v)
}
func write(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
func errJSON(w http.ResponseWriter, status int, msg string) {
	if status >= http.StatusInternalServerError {
		msg = "internal server error"
	}
	write(w, status, map[string]string{"error": msg})
}
func id() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	// RFC 4122 variant and version 4 bits make chat URLs valid UUIDs.
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
}
