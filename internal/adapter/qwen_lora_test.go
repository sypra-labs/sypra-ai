package adapter

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestQwenLMHeadLoRAAppliesLowRankDelta(t *testing.T) {
	artifact := QwenLMHeadLoRA{
		Version: 1, Type: QwenLMHeadLoRAType, Target: "lm_head", Rank: 1, Alpha: 1,
		HiddenSize: 2, VocabSize: 2, A: []float32{2, 3}, B: []float32{1, -1},
	}
	logits := []float32{10, 20}
	if err := artifact.Apply([]float32{4, 5}, logits); err != nil {
		t.Fatal(err)
	}
	// A*h = 23, therefore the two rows receive +23 and -23.
	if logits[0] != 33 || logits[1] != -3 {
		t.Fatalf("logits = %v", logits)
	}
}

func TestLoadQwenLMHeadLoRAFromArtifactDirectory(t *testing.T) {
	artifact := QwenLMHeadLoRA{
		Version: 1, Type: QwenLMHeadLoRAType, Target: "lm_head", Rank: 1, Alpha: 1,
		HiddenSize: 1, VocabSize: 1, A: []float32{1}, B: []float32{1},
	}
	directory := t.TempDir()
	data, err := json.Marshal(artifact)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "adapter.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadQwenLMHeadLoRA(directory)
	if err != nil || loaded.Type != QwenLMHeadLoRAType {
		t.Fatalf("loaded=%#v err=%v", loaded, err)
	}
}

func TestTransformerLoRAAppliesProjectionDelta(t *testing.T) {
	artifact := TransformerLoRA{
		Version: QwenAdapterArtifactV1,
		Type:    TransformerLoRAType,
		Targets: map[string]ProjectionLoRA{
			"model.layers.0.self_attn.q_proj.weight": {
				Rows: 2, Cols: 2, Rank: 1, Alpha: 1,
				A: []float32{2, 3},
				B: []float32{1, -1},
			},
		},
	}
	output := []float32{10, 20}
	if err := artifact.ApplyProjection("model.layers.0.self_attn.q_proj.weight", []float32{4, 5}, output); err != nil {
		t.Fatal(err)
	}
	if output[0] != 33 || output[1] != -3 {
		t.Fatalf("projection delta was not applied: %v", output)
	}
}

func TestLoadQwenAdapterLoadsTransformerArtifact(t *testing.T) {
	directory := t.TempDir()
	artifact := TransformerLoRA{
		Version: QwenAdapterArtifactV1,
		Type:    TransformerLoRAType,
		Targets: map[string]ProjectionLoRA{
			"lm_head.weight": {Rows: 1, Cols: 1, Rank: 1, Alpha: 1, A: []float32{1}, B: []float32{1}},
		},
	}
	data, err := json.Marshal(artifact)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "adapter.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadQwenAdapter(directory)
	if err != nil {
		t.Fatal(err)
	}
	output := []float32{0}
	if err := loaded.ApplyProjection("lm_head.weight", []float32{2}, output); err != nil {
		t.Fatal(err)
	}
	if output[0] != 2 {
		t.Fatalf("loaded adapter did not apply: %v", output)
	}
}

func TestPEFTRoundTripPreservesTransformerTargets(t *testing.T) {
	directory := t.TempDir()
	artifact := TransformerLoRA{
		Version:   QwenAdapterArtifactV1,
		Type:      TransformerLoRAType,
		BaseModel: "Qwen/Qwen3-0.6B",
		Targets: map[string]ProjectionLoRA{
			"model.layers.0.self_attn.q_proj.weight": {Rows: 2, Cols: 3, Rank: 1, Alpha: 2, A: []float32{1, 2, 3}, B: []float32{4, 5}},
			"lm_head.weight":                         {Rows: 2, Cols: 3, Rank: 1, Alpha: 2, A: []float32{6, 7, 8}, B: []float32{9, 10}},
		},
	}
	if err := ExportTransformerPEFT(directory, artifact); err != nil {
		t.Fatal(err)
	}
	loaded, err := ImportPEFT(directory)
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range artifact.Targets {
		got, ok := loaded.Targets[name]
		if !ok || got.Rows != want.Rows || got.Cols != want.Cols || got.Rank != want.Rank || got.Alpha != want.Alpha || !slicesEqual(got.A, want.A) || !slicesEqual(got.B, want.B) {
			t.Fatalf("target %s was not preserved: got=%#v want=%#v", name, got, want)
		}
	}
}

func slicesEqual(left, right []float32) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}
