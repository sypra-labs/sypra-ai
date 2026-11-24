//go:build realmodel

package finetune

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestRealQwenFineTuneE2E validates loading a downloaded Qwen checkpoint,
// response-only training, validation/test loss, and a generated adapter. It is
// opt-in because it is compute intensive.
func TestRealQwenFineTuneE2E(t *testing.T) {
	modelDir := os.Getenv("SYPRA_REAL_MODEL_DIR")
	dataset := os.Getenv("SYPRA_REAL_MODEL_DATASET")
	if modelDir == "" || dataset == "" {
		t.Skip("set SYPRA_REAL_MODEL_DIR and SYPRA_REAL_MODEL_DATASET to run real fine-tuning validation")
	}
	config := DefaultConfig()
	config.ModelDir, config.Method, config.ComputeBackend = modelDir, "lora", "cpu"
	config.Epochs, config.LoRARank, config.LoRAAlpha, config.BatchSize = 1, 4, 8, 1
	result, err := TrainQwenTransformerLoRA(context.Background(), dataset, filepath.Join(t.TempDir(), "artifacts"), config, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.ArtifactPath == "" || math.IsNaN(result.TrainLoss) || math.IsNaN(result.EvalLoss) || math.IsNaN(result.TestLoss) {
		t.Fatalf("invalid real training result: %#v", result)
	}
	if _, err = os.Stat(result.ArtifactPath); err != nil {
		t.Fatalf("adapter artifact was not written: %v", err)
	}
	t.Logf("real Qwen fine-tuning: train=%f validation=%f test=%f artifact=%s", result.TrainLoss, result.EvalLoss, result.TestLoss, result.ArtifactPath)
}
