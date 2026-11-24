package store

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/sypra-labs/sypra-ai/internal/domain"
)

func TestWorkspaceBackupRestorePreservesStateAndVectors(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	source, err := Open(filepath.Join(dir, "source.json"))
	if err != nil {
		t.Fatal(err)
	}
	collection := domain.Collection{ID: "collection", Name: "Policies", EmbeddingModel: "local-hash-384", ChunkSize: 32, ChunkOverlap: 4, CreatedAt: Now()}
	chunk := domain.Chunk{ID: "chunk", CollectionID: collection.ID, Source: "policy.txt", Content: "Travel needs approval.", Embedding: []float32{1, 0, 0}, CreatedAt: Now()}
	if err = source.AddCollection(ctx, collection, []domain.Chunk{chunk}); err != nil {
		t.Fatal(err)
	}
	if _, err = source.AddModel(ctx, domain.Model{ID: "base", Name: "Base", Path: "C:/models/base", Kind: "base", CreatedAt: Now()}); err != nil {
		t.Fatal(err)
	}
	backup := filepath.Join(dir, "workspace.sypra-backup")
	if err = source.BackupTo(ctx, backup); err != nil {
		t.Fatal(err)
	}

	target, err := Open(filepath.Join(dir, "target.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = target.RestoreFrom(ctx, backup); err != nil {
		t.Fatal(err)
	}
	models, err := target.Models(ctx)
	if err != nil || len(models) != 1 || models[0].ID != "base" {
		t.Fatalf("models after restore = %#v, %v", models, err)
	}
	matches, err := target.SearchVectorChunks(ctx, collection.ID, "travel", []float32{1, 0, 0}, 1)
	if err != nil || len(matches) != 1 || matches[0].ID != chunk.ID {
		t.Fatalf("vectors after restore = %#v, %v", matches, err)
	}
}
