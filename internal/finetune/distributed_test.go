package finetune

import (
	"context"
	"strings"
	"testing"

	"github.com/sypra-labs/sypra-ai/internal/adapter"
)

type testGradientWorker struct{ value float32 }

func (w testGradientWorker) Compute(_ context.Context, artifact adapter.TransformerLoRA, examples []Example, _ int) (GradientBatch, error) {
	gradients := adapter.NewTransformerGradients(&artifact)
	gradient := gradients["q.weight"]
	gradient.A[0] = w.value * float32(len(examples))
	gradient.B[0] = -w.value * float32(len(examples))
	gradients["q.weight"] = gradient
	return GradientBatch{Loss: float64(w.value) * float64(len(examples)), Tokens: len(examples), Gradients: gradients}, nil
}
func (testGradientWorker) Close() error { return nil }

func TestAllReduceGradientBatch(t *testing.T) {
	artifact := adapter.TransformerLoRA{Version: adapter.QwenAdapterArtifactV1, Type: adapter.TransformerLoRAType, Targets: map[string]adapter.ProjectionLoRA{"q.weight": {Rows: 1, Cols: 1, Rank: 1, Alpha: 1, A: []float32{1}, B: []float32{1}}}}
	examples := []Example{{ID: "1"}, {ID: "2"}, {ID: "3"}, {ID: "4"}}
	result, err := allReduceGradientBatch(context.Background(), []GradientWorker{testGradientWorker{value: 1}, testGradientWorker{value: 2}}, artifact, examples, 0)
	if err != nil {
		t.Fatal(err)
	}
	gradient := result.Gradients["q.weight"]
	// Each worker receives two records: 2*1 + 2*2.
	if result.Tokens != 4 || result.Loss != 6 || gradient.A[0] != 6 || gradient.B[0] != -6 {
		t.Fatalf("unexpected all-reduce result: %#v", result)
	}
}

func TestNewRemoteGradientWorkerRejectsInvalidEndpoint(t *testing.T) {
	_, err := NewRemoteGradientWorker("worker-one:8091")
	if err == nil || !strings.Contains(err.Error(), "absolute http(s) URL") {
		t.Fatalf("expected endpoint validation error, got %v", err)
	}
}
