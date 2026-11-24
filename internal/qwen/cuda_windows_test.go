//go:build windows

package qwen

import (
	"math"
	"os"
	"testing"
	"time"

	"github.com/sypra-labs/sypra-ai/internal/adapter"
)

// Opt-in because it initializes a real CUDA context and compiles embedded PTX
// through the installed driver. Set SYPRA_CUDA_TEST=1 on a CUDA machine.
func TestCUDAExecutorMatVec(t *testing.T) {
	if os.Getenv("SYPRA_CUDA_TEST") != "1" {
		t.Skip("set SYPRA_CUDA_TEST=1 to run against an installed NVIDIA driver")
	}
	model := tinyTrainingModel()
	input := []float32{.3, -.2}
	want, err := model.matrixVec("model.layers.0.self_attn.q_proj.weight", input)
	if err != nil {
		t.Fatal(err)
	}
	executor, err := NewCUDAExecutor()
	if err != nil {
		t.Fatal(err)
	}
	defer executor.Close()
	model.SetMatrixExecutor(executor)
	got, err := model.matrixVec("model.layers.0.self_attn.q_proj.weight", input)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("CUDA result shape %v, want %v", got, want)
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("CUDA result %v, want %v", got, want)
		}
	}
	model.SetMatrixExecutor(nil)
	wantTranspose, err := model.matrixVecTranspose("model.layers.0.self_attn.q_proj.weight", []float32{.1, -.4})
	if err != nil {
		t.Fatal(err)
	}
	model.SetMatrixExecutor(executor)
	gotTranspose, err := model.matrixVecTranspose("model.layers.0.self_attn.q_proj.weight", []float32{.1, -.4})
	if err != nil {
		t.Fatal(err)
	}
	for i := range wantTranspose {
		if math.Abs(float64(gotTranspose[i]-wantTranspose[i])) > 1e-5 {
			t.Fatalf("CUDA transpose result %v, want %v", gotTranspose, wantTranspose)
		}
	}
	// Fine-tuning workers execute from coordinator goroutines, not the thread
	// that created the CUDA context. Verify that path explicitly.
	result := make(chan error, 1)
	go func() {
		_, err := model.matrixVec("model.layers.0.self_attn.q_proj.weight", input)
		result <- err
	}()
	if err := <-result; err != nil {
		t.Fatalf("cross-goroutine CUDA matrix operation: %v", err)
	}
}

// TestCUDAFullCheckpointForward exercises the same complete frozen projection
// path used by fine-tuning. It is opt-in because loading a real checkpoint
// consumes several gigabytes of RAM/VRAM.
func TestCUDAFullCheckpointForward(t *testing.T) {
	modelDir := os.Getenv("SYPRA_QWEN_MODEL")
	if os.Getenv("SYPRA_CUDA_FULL_TEST") != "1" || modelDir == "" {
		t.Skip("set SYPRA_CUDA_FULL_TEST=1 and SYPRA_QWEN_MODEL to run against a real checkpoint")
	}
	model, err := LoadModel(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	executor, err := NewCUDAExecutor()
	if err != nil {
		t.Fatal(err)
	}
	model.SetMatrixExecutor(executor)
	tokens, err := model.Tokenizer.Encode("GPU smoke test")
	if err != nil || len(tokens) == 0 {
		t.Fatalf("tokenize: %v", err)
	}
	state := NewState(model.Config.NumHiddenLayers)
	if _, err = model.ForwardWithAdapter(tokens[0], 0, state, nil); err != nil {
		t.Fatal(err)
	}
}

// TestCUDAFullCheckpointBackward covers the transpose-matrix calls that only
// happen during reverse-mode fine-tuning. It deliberately adapts one Q
// projection so the test stays small while traversing the entire decoder.
func TestCUDAFullCheckpointBackward(t *testing.T) {
	modelDir := os.Getenv("SYPRA_QWEN_MODEL")
	if os.Getenv("SYPRA_CUDA_FULL_TEST") != "1" || modelDir == "" {
		t.Skip("set SYPRA_CUDA_FULL_TEST=1 and SYPRA_QWEN_MODEL to run against a real checkpoint")
	}
	model, err := LoadModel(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	executor, err := NewCUDAExecutor()
	if err != nil {
		t.Fatal(err)
	}
	model.SetMatrixExecutor(executor)
	name := "model.layers.0.self_attn.q_proj.weight"
	rows, cols, err := model.ProjectionShape(name)
	if err != nil {
		t.Fatal(err)
	}
	artifact := adapter.TransformerLoRA{Version: adapter.QwenAdapterArtifactV1, Type: adapter.TransformerLoRAType, BaseModel: modelDir, CreatedAt: time.Now(), Targets: map[string]adapter.ProjectionLoRA{name: {Rows: rows, Cols: cols, Rank: 1, Alpha: 1, A: make([]float32, cols), B: make([]float32, rows)}}}
	tokens, err := model.Tokenizer.Encode("short training response")
	if err != nil || len(tokens) < 2 {
		t.Fatalf("tokenize: %v", err)
	}
	if _, _, _, err = model.CausalLoRAGradients(tokens, 1, &artifact); err != nil {
		t.Fatal(err)
	}
}
