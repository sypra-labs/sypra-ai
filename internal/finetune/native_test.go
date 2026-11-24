package finetune

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestTrainNativeAndGenerate(t *testing.T) {
	dir := t.TempDir()
	dataset := filepath.Join(dir, "data.csv")
	data := "instruction,response,category,intent\nhello,Hi there!,general,greeting\nhello friend,Hi there!,general,greeting\ngoodbye,Goodbye!,general,farewell\nbye now,Goodbye!,general,farewell\nthanks,You are welcome!,general,thanks\n"
	if err := os.WriteFile(dataset, []byte(data), 0600); err != nil {
		t.Fatal(err)
	}
	cfg := DefaultConfig()
	cfg.ModelDir, cfg.Epochs, cfg.LearningRate = "models/qwen", 25, .2
	var reports []Progress
	result, err := TrainNative(context.Background(), dataset, filepath.Join(dir, "artifacts"), cfg, func(progress Progress) { reports = append(reports, progress) })
	if err != nil {
		t.Fatal(err)
	}
	artifact, err := LoadArtifact(result.ArtifactPath)
	if err != nil {
		t.Fatal(err)
	}
	if got := artifact.Generate("goodbye now"); got != "Goodbye!" {
		t.Fatalf("unexpected adapter response %q", got)
	}
	if len(reports) != cfg.Epochs || !reports[len(reports)-1].ValidationComplete || reports[len(reports)-1].Epoch != cfg.Epochs || reports[len(reports)-1].Steps != cfg.Epochs {
		t.Fatalf("unexpected training telemetry: %#v", reports)
	}
}
