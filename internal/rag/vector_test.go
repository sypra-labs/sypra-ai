package rag

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sypra-labs/sypra-ai/internal/embedding"
	"github.com/sypra-labs/sypra-ai/internal/store"
)

func TestCollectionPersistsVectorsAndUsesVectorRetrieval(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	st, err := store.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	svc := Service{Store: st}
	collection, err := svc.Create(context.Background(), CreateInput{Name: "notes", ChunkSize: 100, ChunkOverlap: 10, Documents: []Document{{Source: "notes.txt", Text: "Revenue forecast and quarterly financial report.\n\nMountain bicycle maintenance checklist and tire pressure."}}})
	if err != nil {
		t.Fatal(err)
	}
	if collection.EmbeddingModel != embedding.DefaultModel {
		t.Fatalf("embedding model = %q", collection.EmbeddingModel)
	}
	chunks, err := st.CollectionChunks(context.Background(), collection.ID)
	if err != nil || len(chunks) == 0 || len(chunks[0].Embedding) != 0 {
		t.Fatalf("chunk metadata should not retain vector values: chunks=%#v error=%v", chunks, err)
	}
	if count := st.VectorCount(context.Background(), collection.ID); count != len(chunks) {
		t.Fatalf("vector database count = %d, want %d", count, len(chunks))
	}
	jsonState, err := os.ReadFile(path)
	if err != nil || strings.Contains(string(jsonState), "\"embedding\"") {
		t.Fatalf("workspace JSON retained embeddings: %v", err)
	}
	reopened, err := store.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	results, err := (Service{Store: reopened}).Retrieve(context.Background(), collection.ID, "financial revenue report", 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Content[:7] != "Revenue" {
		t.Fatalf("unexpected ranked result: %#v", results)
	}
}
