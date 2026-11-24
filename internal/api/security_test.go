package api

import (
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sypra-labs/sypra-ai/internal/observability"
	"github.com/sypra-labs/sypra-ai/internal/rag"
	"github.com/sypra-labs/sypra-ai/internal/store"
	"github.com/sypra-labs/sypra-ai/internal/training"
)

func TestOperationalEndpointsRedactErrorsAndExposeMetrics(t *testing.T) {
	st, err := store.Open(filepath.Join(t.TempDir(), "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	srv := &Server{Store: st, RAG: rag.Service{Store: st}, Trainer: training.New(st), CORSOrigin: "http://localhost:5173", Logger: slog.New(slog.NewTextHandler(io.Discard, nil)), Metrics: observability.NewMetrics()}
	handler := srv.Handler()

	ready := httptest.NewRecorder()
	handler.ServeHTTP(ready, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if ready.Code != http.StatusOK || ready.Header().Get("X-Request-ID") == "" {
		t.Fatalf("readyz status=%d requestID=%q", ready.Code, ready.Header().Get("X-Request-ID"))
	}
	metrics := httptest.NewRecorder()
	handler.ServeHTTP(metrics, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if metrics.Code != http.StatusOK || !strings.Contains(metrics.Body.String(), "sypra_http_requests_total") {
		t.Fatalf("metrics status=%d body=%q", metrics.Code, metrics.Body.String())
	}
	errorResponse := httptest.NewRecorder()
	errJSON(errorResponse, http.StatusInternalServerError, `database password=not-for-clients`)
	if strings.Contains(errorResponse.Body.String(), "password") || !strings.Contains(errorResponse.Body.String(), "internal server error") {
		t.Fatalf("server error leaked detail: %s", errorResponse.Body.String())
	}
}
