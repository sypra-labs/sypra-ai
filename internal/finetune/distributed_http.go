package finetune

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/sypra-labs/sypra-ai/internal/adapter"
)

const distributedGradientPath = "/v1/training/gradients"

type distributedGradientRequest struct {
	Artifact          adapter.TransformerLoRA `json:"artifact"`
	Examples          []Example               `json:"examples"`
	MaxSequenceLength int                     `json:"maxSequenceLength"`
}

type remoteGradientWorker struct {
	endpoint string
	http     *http.Client
}

func NewRemoteGradientWorker(endpoint string) (GradientWorker, error) {
	endpoint = strings.TrimRight(strings.TrimSpace(endpoint), "/")
	if endpoint == "" {
		return nil, fmt.Errorf("distributed worker endpoint is required")
	}
	parsed, err := url.ParseRequestURI(endpoint)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return nil, fmt.Errorf("distributed worker endpoint must be an absolute http(s) URL")
	}
	return &remoteGradientWorker{endpoint: endpoint, http: &http.Client{}}, nil
}

func (w *remoteGradientWorker) Compute(ctx context.Context, artifact adapter.TransformerLoRA, examples []Example, maximumSequenceLength int) (GradientBatch, error) {
	payload, err := json.Marshal(distributedGradientRequest{Artifact: artifact, Examples: examples, MaxSequenceLength: maximumSequenceLength})
	if err != nil {
		return GradientBatch{}, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, w.endpoint+distributedGradientPath, bytes.NewReader(payload))
	if err != nil {
		return GradientBatch{}, err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := w.http.Do(request)
	if err != nil {
		return GradientBatch{}, err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode > 299 {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 64<<10))
		return GradientBatch{}, fmt.Errorf("distributed worker %s returned %s: %s", w.endpoint, response.Status, strings.TrimSpace(string(body)))
	}
	var result GradientBatch
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		return GradientBatch{}, err
	}
	return result, nil
}

func (w *remoteGradientWorker) Close() error { return nil }

// GradientWorkerServer serves one local frozen-model replica. It is intended
// for trusted private networks; place authentication/TLS in front of it before
// exposing it beyond the training cluster.
type GradientWorkerServer struct{ worker GradientWorker }

func NewGradientWorkerServer(cfg Config, device *int) (*GradientWorkerServer, error) {
	worker, err := NewLocalGradientWorker(cfg, device)
	if err != nil {
		return nil, err
	}
	return &GradientWorkerServer{worker: worker}, nil
}

func (s *GradientWorkerServer) Close() error { return s.worker.Close() }

func (s *GradientWorkerServer) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) })
	mux.HandleFunc(distributedGradientPath, s.gradients)
	return mux
}

func (s *GradientWorkerServer) gradients(w http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	request.Body = http.MaxBytesReader(w, request.Body, 2<<30)
	var input distributedGradientRequest
	if err := json.NewDecoder(request.Body).Decode(&input); err != nil {
		http.Error(w, "invalid gradient request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if err := input.Artifact.Validate(); err != nil {
		http.Error(w, "invalid adapter: "+err.Error(), http.StatusBadRequest)
		return
	}
	batch, err := s.worker.Compute(request.Context(), input.Artifact, input.Examples, input.MaxSequenceLength)
	if err != nil {
		http.Error(w, "gradient computation failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(batch)
}
