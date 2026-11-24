package qwen

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/sypra-labs/sypra-ai/internal/adapter"
)

func TestCausalLoRAGradientMatchesFiniteDifference(t *testing.T) {
	model := tinyTrainingModel()
	for _, targetName := range []string{"model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.down_proj.weight"} {
		target := adapter.ProjectionLoRA{Rows: 2, Cols: 2, Rank: 1, Alpha: 1, A: []float32{.12, -.08}, B: []float32{.04, -.03}}
		artifact := adapter.TransformerLoRA{Version: adapter.QwenAdapterArtifactV1, Type: adapter.TransformerLoRAType, BaseModel: "tiny", Targets: map[string]adapter.ProjectionLoRA{targetName: target}}
		loss, _, gradients, err := model.CausalLoRAGradients([]int{0, 1, 2}, 1, &artifact)
		if err != nil {
			t.Fatalf("%s: %v", targetName, err)
		}
		analytic := gradients[targetName].B[0]
		epsilon := float32(1e-3)
		plus, minus := copyTinyArtifact(artifact), copyTinyArtifact(artifact)
		plusTarget, minusTarget := plus.Targets[targetName], minus.Targets[targetName]
		plusTarget.B[0] += epsilon
		minusTarget.B[0] -= epsilon
		plus.Targets[targetName], minus.Targets[targetName] = plusTarget, minusTarget
		plusLoss, _, err := model.CausalLoRALoss([]int{0, 1, 2}, 1, &plus)
		if err != nil {
			t.Fatal(err)
		}
		minusLoss, _, err := model.CausalLoRALoss([]int{0, 1, 2}, 1, &minus)
		if err != nil {
			t.Fatal(err)
		}
		numeric := float32((plusLoss - minusLoss) / float64(2*epsilon))
		if math.Abs(float64(analytic-numeric)) > 2e-3 {
			t.Fatalf("%s B[0] gradient mismatch: analytic=%f numeric=%f loss=%f", targetName, analytic, numeric, loss)
		}
	}
}

func copyTinyArtifact(artifact adapter.TransformerLoRA) adapter.TransformerLoRA {
	copy := artifact
	copy.Targets = make(map[string]adapter.ProjectionLoRA, len(artifact.Targets))
	for name, target := range artifact.Targets {
		target.A, target.B = append([]float32(nil), target.A...), append([]float32(nil), target.B...)
		copy.Targets[name] = target
	}
	return copy
}

func tinyTrainingModel() *Model {
	model := &Model{Config: Config{HiddenSize: 2, IntermediateSize: 2, NumAttentionHeads: 1, NumKeyValueHeads: 1, NumHiddenLayers: 1, HeadDim: 2, VocabSize: 3, RMSNormEps: 1e-5, RopeTheta: 10000, MaxPositionEmbeddings: 16}, cache: map[string]weight{}, nf4Cache: map[string]*nf4Weight{}}
	setMatrix := func(name string, rows, cols int, values ...float32) {
		model.cache[name] = weight{data: testBF16(values), shape: []int{rows, cols}}
	}
	setVector := func(name string, values ...float32) {
		model.cache[name] = weight{data: testBF16(values), shape: []int{len(values)}}
	}
	setMatrix("model.embed_tokens.weight", 3, 2, .2, -.1, -.15, .25, .05, .3)
	setVector("model.layers.0.input_layernorm.weight", 1, .9)
	setMatrix("model.layers.0.self_attn.q_proj.weight", 2, 2, .3, -.2, .1, .4)
	setMatrix("model.layers.0.self_attn.k_proj.weight", 2, 2, -.25, .2, .35, .1)
	setMatrix("model.layers.0.self_attn.v_proj.weight", 2, 2, .2, .15, -.1, .3)
	setVector("model.layers.0.self_attn.q_norm.weight", 1, 1)
	setVector("model.layers.0.self_attn.k_norm.weight", 1, 1)
	setMatrix("model.layers.0.self_attn.o_proj.weight", 2, 2, .25, -.1, .15, .2)
	setVector("model.layers.0.post_attention_layernorm.weight", 1, 1)
	setMatrix("model.layers.0.mlp.gate_proj.weight", 2, 2, .2, .1, -.15, .3)
	setMatrix("model.layers.0.mlp.up_proj.weight", 2, 2, -.1, .25, .3, -.2)
	setMatrix("model.layers.0.mlp.down_proj.weight", 2, 2, .15, .2, .1, -.25)
	setVector("model.norm.weight", 1, 1)
	setMatrix("lm_head.weight", 3, 2, .2, -.15, -.1, .3, .25, .1)
	return model
}

func testBF16(values []float32) []byte {
	data := make([]byte, len(values)*2)
	for index, value := range values {
		binary.LittleEndian.PutUint16(data[index*2:], uint16(math.Float32bits(value)>>16))
	}
	return data
}
