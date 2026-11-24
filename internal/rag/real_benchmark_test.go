//go:build realmodel

package rag

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/sypra-labs/sypra-ai/internal/domain"
	"github.com/sypra-labs/sypra-ai/internal/embedding"
	"github.com/sypra-labs/sypra-ai/internal/store"
)

// TestRealMiniLMRetrievalBenchmark is intentionally opt-in. It loads the
// downloaded all-MiniLM-L6-v2 checkpoint rather than the deterministic hash
// fallback and fails when labelled retrieval quality regresses.
func TestRealMiniLMRetrievalBenchmark(t *testing.T) {
	modelPath := os.Getenv("SYPRA_RAG_BENCHMARK_MODEL_PATH")
	if modelPath == "" {
		t.Skip("set SYPRA_RAG_BENCHMARK_MODEL_PATH to run the real MiniLM benchmark")
	}
	t.Setenv("SYPRA_EMBEDDING_MODEL_PATH", modelPath)
	ctx := context.Background()
	st, err := store.Open(filepath.Join(t.TempDir(), "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	service := Service{Store: st}
	started := time.Now()
	collection, err := service.Create(ctx, CreateInput{Name: "benchmark", EmbeddingModel: embedding.DefaultModel, ChunkSize: 64, ChunkOverlap: 8, Documents: []Document{
		{Source: "travel.md", Text: "Employees must obtain manager approval before booking business travel."},
		{Source: "expenses.md", Text: "Expense reimbursement requires an itemized receipt submitted within thirty days."},
		{Source: "security.md", Text: "Security incidents must be reported to the incident response team immediately."},
	}})
	if err != nil {
		t.Fatal(err)
	}
	documents, err := st.Documents(ctx, collection.ID)
	if err != nil || len(documents) != 3 {
		t.Fatalf("benchmark documents=%#v err=%v", documents, err)
	}
	bySource := map[string]string{}
	for _, document := range documents {
		bySource[document.Name] = document.ID
	}
	evaluation, err := service.CreateEvaluationSet(ctx, collection.ID, "real-minilm", []domain.RAGEvaluationCase{
		{Query: "Who approves a business trip?", RelevantDocumentIDs: []string{bySource["travel.md"]}},
		{Query: "When is a receipt needed for reimbursement?", RelevantDocumentIDs: []string{bySource["expenses.md"]}},
		{Query: "How do I report a security incident?", RelevantDocumentIDs: []string{bySource["security.md"]}},
	})
	if err != nil {
		t.Fatal(err)
	}
	run, err := service.Evaluate(ctx, evaluation.ID, 1, RetrievalOptions{Rerank: true})
	if err != nil {
		t.Fatal(err)
	}
	threshold := 0.80
	if configured, parseErr := strconv.ParseFloat(os.Getenv("SYPRA_RAG_BENCHMARK_MIN_HIT_RATE"), 64); parseErr == nil && configured > 0 {
		threshold = configured
	}
	if run.Metrics["hitRate"] < threshold {
		t.Fatalf("real MiniLM hit rate %.3f is below threshold %.3f; metrics=%v", run.Metrics["hitRate"], threshold, run.Metrics)
	}
	t.Logf("real MiniLM benchmark: documents=%d duration=%s metrics=%v", len(documents), time.Since(started).Round(time.Millisecond), run.Metrics)
}
