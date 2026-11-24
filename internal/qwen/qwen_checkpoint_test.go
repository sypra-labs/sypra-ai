package qwen

import (
	"os"
	"testing"
)

// Opt-in checkpoint validation for a locally downloaded model. It reads only
// metadata and tokenizer assets, never model.safetensors payloads.
func TestLocalQwenCheckpointMetadata(t *testing.T) {
	dir := os.Getenv("SYPRA_QWEN_MODEL_TEST_DIR")
	if dir == "" {
		t.Skip("set SYPRA_QWEN_MODEL_TEST_DIR to validate a local Qwen checkpoint")
	}
	model, err := LoadModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	if model.Config.HiddenSize != 1024 || model.Config.NumHiddenLayers != 28 || model.Config.VocabSize != 151936 {
		t.Fatalf("unexpected Qwen3-0.6B config: %#v", model.Config)
	}
	ids, err := model.Tokenizer.Encode("user\nHello from Sypra")
	if err != nil || len(ids) == 0 {
		t.Fatalf("tokenizer did not encode prompt: ids=%v error=%v", ids, err)
	}
}
