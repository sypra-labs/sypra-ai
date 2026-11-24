package finetune

import (
	"path/filepath"
	"testing"

	"github.com/sypra-labs/sypra-ai/internal/adapter"
)

func TestAdamWUpdatesAndRestoresProjectionState(t *testing.T) {
	cfg := DefaultConfig()
	artifact := adapter.TransformerLoRA{Version: adapter.QwenAdapterArtifactV1, Type: adapter.TransformerLoRAType, Targets: map[string]adapter.ProjectionLoRA{"q.weight": {Rows: 1, Cols: 1, Rank: 1, Alpha: 1, A: []float32{.2}, B: []float32{.1}}}}
	optimizer := NewAdamW(cfg, &artifact)
	gradients := adapter.NewTransformerGradients(&artifact)
	gradient := gradients["q.weight"]
	gradient.A[0], gradient.B[0] = .5, -.25
	gradients["q.weight"] = gradient
	beforeA, beforeB := artifact.Targets["q.weight"].A[0], artifact.Targets["q.weight"].B[0]
	if err := optimizer.Step(&artifact, gradients, .01, 1); err != nil {
		t.Fatal(err)
	}
	if artifact.Targets["q.weight"].A[0] == beforeA || artifact.Targets["q.weight"].B[0] == beforeB {
		t.Fatal("AdamW did not update LoRA parameters")
	}
	directory := t.TempDir()
	if err := writeAdamWState(filepath.Join(directory, optimizerStateFile), optimizer.Snapshot()); err != nil {
		t.Fatal(err)
	}
	state, err := readAdamWState(directory)
	if err != nil {
		t.Fatal(err)
	}
	restored := NewAdamW(cfg, &artifact)
	if err := restored.Restore(state, &artifact); err != nil {
		t.Fatal(err)
	}
	if restored.step != optimizer.step || restored.m["q.weight"].A[0] != optimizer.m["q.weight"].A[0] {
		t.Fatalf("optimizer state was not restored: %#v %#v", restored, optimizer)
	}
}
