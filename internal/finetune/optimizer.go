package finetune

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/sypra-labs/sypra-ai/internal/adapter"
)

const optimizerStateFile = "optimizer_state.json"

// AdamW owns optimizer moments separately from adapter artifacts, allowing
// artifacts to remain portable while trainer state is local to a run.
type AdamW struct {
	step                               int
	beta1, beta2, epsilon, weightDecay float64
	maxNorm                            float64
	m, v                               map[string]adapter.ProjectionGradient
}

type AdamWState struct {
	Version int                                   `json:"version"`
	Step    int                                   `json:"step"`
	M       map[string]adapter.ProjectionGradient `json:"m"`
	V       map[string]adapter.ProjectionGradient `json:"v"`
}

func NewAdamW(cfg Config, artifact *adapter.TransformerLoRA) *AdamW {
	return &AdamW{beta1: cfg.AdamBeta1, beta2: cfg.AdamBeta2, epsilon: cfg.AdamEpsilon, weightDecay: cfg.WeightDecay, maxNorm: cfg.MaxGradientNorm, m: adapter.NewTransformerGradients(artifact), v: adapter.NewTransformerGradients(artifact)}
}

func (o *AdamW) Snapshot() AdamWState {
	return AdamWState{Version: 1, Step: o.step, M: copyGradients(o.m), V: copyGradients(o.v)}
}

func (o *AdamW) Restore(state AdamWState, artifact *adapter.TransformerLoRA) error {
	if state.Version != 1 || state.Step < 0 {
		return fmt.Errorf("invalid AdamW state")
	}
	for name, target := range artifact.Targets {
		first, firstOK := state.M[name]
		second, secondOK := state.V[name]
		if !firstOK || !secondOK || len(first.A) != len(target.A) || len(first.B) != len(target.B) || len(second.A) != len(target.A) || len(second.B) != len(target.B) {
			return fmt.Errorf("incompatible AdamW state for %s", name)
		}
	}
	o.step, o.m, o.v = state.Step, copyGradients(state.M), copyGradients(state.V)
	return nil
}

func writeAdamWState(path string, state AdamWState) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	data, err := json.Marshal(state)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0600)
}

func readAdamWState(directory string) (AdamWState, error) {
	data, err := os.ReadFile(filepath.Join(directory, optimizerStateFile))
	if err != nil {
		return AdamWState{}, err
	}
	var state AdamWState
	if err := json.Unmarshal(data, &state); err != nil {
		return AdamWState{}, err
	}
	return state, nil
}

// Step applies accumulated gradients. gradientScale is normally one divided
// by the number of response labels in the effective batch.
func (o *AdamW) Step(artifact *adapter.TransformerLoRA, gradients map[string]adapter.ProjectionGradient, learningRate, gradientScale float64) error {
	if artifact == nil || learningRate <= 0 || gradientScale <= 0 {
		return fmt.Errorf("invalid AdamW step")
	}
	norm := gradientNorm(gradients, gradientScale)
	clip := float64(1)
	if o.maxNorm > 0 && norm > o.maxNorm {
		clip = o.maxNorm / norm
	}
	o.step++
	bias1 := 1 - math.Pow(o.beta1, float64(o.step))
	bias2 := 1 - math.Pow(o.beta2, float64(o.step))
	for name, target := range artifact.Targets {
		gradient, ok := gradients[name]
		if !ok {
			return fmt.Errorf("missing gradient for %s", name)
		}
		first, second := o.m[name], o.v[name]
		updateAdamW(target.A, gradient.A, first.A, second.A, learningRate, gradientScale*clip, o.weightDecay, o.beta1, o.beta2, bias1, bias2, o.epsilon)
		updateAdamW(target.B, gradient.B, first.B, second.B, learningRate, gradientScale*clip, o.weightDecay, o.beta1, o.beta2, bias1, bias2, o.epsilon)
		artifact.Targets[name] = target
		o.m[name], o.v[name] = first, second
	}
	return nil
}

func gradientNorm(gradients map[string]adapter.ProjectionGradient, scale float64) float64 {
	var sum float64
	for _, gradient := range gradients {
		for _, value := range gradient.A {
			scaled := float64(value) * scale
			sum += scaled * scaled
		}
		for _, value := range gradient.B {
			scaled := float64(value) * scale
			sum += scaled * scaled
		}
	}
	return math.Sqrt(sum)
}

func updateAdamW(parameters, gradients, first, second []float32, learningRate, gradientScale, weightDecay, beta1, beta2, bias1, bias2, epsilon float64) {
	for index := range parameters {
		gradient := float64(gradients[index]) * gradientScale
		first[index] = float32(beta1*float64(first[index]) + (1-beta1)*gradient)
		second[index] = float32(beta2*float64(second[index]) + (1-beta2)*gradient*gradient)
		moment := float64(first[index]) / bias1
		variance := float64(second[index]) / bias2
		parameters[index] -= float32(learningRate * (moment/(math.Sqrt(variance)+epsilon) + weightDecay*float64(parameters[index])))
	}
}

func copyGradients(source map[string]adapter.ProjectionGradient) map[string]adapter.ProjectionGradient {
	copy := make(map[string]adapter.ProjectionGradient, len(source))
	for name, gradient := range source {
		copy[name] = adapter.ProjectionGradient{A: append([]float32(nil), gradient.A...), B: append([]float32(nil), gradient.B...)}
	}
	return copy
}
