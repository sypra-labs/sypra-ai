package rag

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/sypra-labs/sypra-ai/internal/embedding"
	"github.com/sypra-labs/sypra-ai/internal/store"
)

func TestDocumentLibraryLifecycleAndInspectableRetrieval(t *testing.T) {
	st, err := store.Open(filepath.Join(t.TempDir(), "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	svc := Service{Store: st}
	collection, err := svc.Create(context.Background(), CreateInput{
		Name:           "handbook",
		EmbeddingModel: embedding.HashModel,
		ChunkSize:      64,
		ChunkOverlap:   0,
		Documents:      []Document{{Source: "handbook.txt", Text: "Vacation requests require manager approval. Submit requests through the HR portal. Payroll closes every Friday."}},
	})
	if err != nil {
		t.Fatal(err)
	}
	documents, err := st.Documents(context.Background(), collection.ID)
	if err != nil || len(documents) != 1 || documents[0].ChunkCount == 0 {
		t.Fatalf("documents=%#v err=%v", documents, err)
	}
	results, err := svc.RetrieveDetailed(context.Background(), collection.ID, "How are vacation requests approved?", RetrievalOptions{Limit: 1, IncludeNeighbors: true, Rerank: true})
	if err != nil || len(results) != 1 || results[0].Document == nil || results[0].Chunk.DocumentID != documents[0].ID {
		t.Fatalf("results=%#v err=%v", results, err)
	}
	renamed, err := st.RenameDocument(context.Background(), documents[0].ID, "Employee handbook")
	if err != nil || renamed.Name != "Employee handbook" {
		t.Fatalf("rename=%#v err=%v", renamed, err)
	}
	if err := st.DeleteDocument(context.Background(), documents[0].ID); err != nil {
		t.Fatal(err)
	}
	if count := st.VectorCount(context.Background(), collection.ID); count != 0 {
		t.Fatalf("vectors after document removal=%d, want 0", count)
	}
}
