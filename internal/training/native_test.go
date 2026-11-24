package training

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/sypra-labs/sypra-ai/internal/domain"
	"github.com/sypra-labs/sypra-ai/internal/finetune"
	"github.com/sypra-labs/sypra-ai/internal/inference"
	"github.com/sypra-labs/sypra-ai/internal/store"
)

func TestNativeRunRegistersUsableSLM(t *testing.T) {
	dir := t.TempDir()
	dataset := filepath.Join(dir, "training.csv")
	csv := "instruction,response,category,intent\nhello,Hi there!,general,greeting\nhello friend,Hi there!,general,greeting\ngoodbye,Goodbye!,general,farewell\nbye now,Goodbye!,general,farewell\nthanks,You are welcome!,general,thanks\n"
	if err := os.WriteFile(dataset, []byte(csv), 0600); err != nil {
		t.Fatal(err)
	}
	st, err := store.Open(filepath.Join(dir, "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	cfg := finetune.DefaultConfig()
	cfg.ModelDir, cfg.Epochs, cfg.LearningRate = "base-model", 15, .2
	job := domain.TrainingJob{ID: "job-1", Name: "support-adapter", BaseModel: cfg.ModelDir, DatasetPath: dataset, Method: "lora", Parameters: map[string]any{}, Status: "queued", CreatedAt: store.Now()}
	service := New(st)
	if err = service.StartNative(context.Background(), job, cfg, filepath.Join(dir, "run")); err != nil {
		t.Fatal(err)
	}
	completed, err := service.Wait(context.Background(), job.ID)
	if err != nil {
		t.Fatal(err)
	}
	if completed.Status != "completed" {
		t.Fatalf("training did not complete: %#v", completed)
	}
	models, err := st.Models(context.Background())
	if err != nil || len(models) != 1 || models[0].Kind != "slm" {
		t.Fatalf("expected registered SLM, models=%#v error=%v", models, err)
	}
	response, err := (&inference.Client{}).Chat(context.Background(), models[0].Endpoint, inference.ChatRequest{Messages: []inference.Message{{Role: "user", Content: "bye now"}}})
	if err != nil || response.Content != "Goodbye!" {
		t.Fatalf("native SLM response=%q error=%v", response.Content, err)
	}
}
