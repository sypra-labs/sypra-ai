package store

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/sypra-labs/sypra-ai/internal/domain"
)

func TestPromoteModelArchivesPriorProductionVersion(t *testing.T) {
	st, err := Open(filepath.Join(t.TempDir(), "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	first := domain.Model{ID: "adapter-1", Name: "support", Path: "one.json", Format: "sypra-qwen-lm-head-lora", Kind: "slm", ParentModel: "qwen", Status: "production", CreatedAt: Now()}
	second := domain.Model{ID: "adapter-2", Name: "support", Path: "two.json", Format: "sypra-qwen-lm-head-lora", Kind: "slm", ParentModel: "qwen", Status: "candidate", CreatedAt: Now()}
	if _, err := st.AddModel(context.Background(), first); err != nil {
		t.Fatal(err)
	}
	if _, err := st.AddModel(context.Background(), second); err != nil {
		t.Fatal(err)
	}
	promoted, err := st.PromoteModel(context.Background(), second.ID)
	if err != nil || promoted.Status != "production" || promoted.PromotedAt == nil {
		t.Fatalf("promote=%#v err=%v", promoted, err)
	}
	models, err := st.Models(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if models[0].Status != "archived" || models[1].Status != "production" {
		t.Fatalf("unexpected deployment statuses: %#v", models)
	}
}

func TestFirstBaseModelBecomesDefaultAndCanBeChanged(t *testing.T) {
	st, err := Open(filepath.Join(t.TempDir(), "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	first, err := st.AddModel(context.Background(), domain.Model{ID: "base-1", Name: "First", Path: "first", Kind: "base", CreatedAt: Now()})
	if err != nil {
		t.Fatal(err)
	}
	if !first.Default {
		t.Fatal("first base model should become the default")
	}
	second, err := st.AddModel(context.Background(), domain.Model{ID: "base-2", Name: "Second", Path: "second", Kind: "base", CreatedAt: Now()})
	if err != nil {
		t.Fatal(err)
	}
	if second.Default {
		t.Fatal("subsequent base model must not replace the default automatically")
	}
	defaultModel, ok, err := st.DefaultBaseModel(context.Background())
	if err != nil || !ok || defaultModel.ID != first.ID {
		t.Fatalf("default model=%#v ok=%t err=%v", defaultModel, ok, err)
	}
	updated, err := st.SetDefaultBaseModel(context.Background(), second.ID)
	if err != nil || !updated.Default || updated.ID != second.ID {
		t.Fatalf("updated=%#v err=%v", updated, err)
	}
	defaultModel, ok, err = st.DefaultBaseModel(context.Background())
	if err != nil || !ok || defaultModel.ID != second.ID {
		t.Fatalf("default model=%#v ok=%t err=%v", defaultModel, ok, err)
	}
}
