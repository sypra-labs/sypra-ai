// Package adapter defines portable, local fine-tuning artifacts for Sypra's
// native Qwen runner. The frozen checkpoint is never modified.
package adapter

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

const (
	QwenLMHeadLoRAType    = "sypra-qwen-lm-head-lora"
	TransformerLoRAType   = "sypra-qwen-transformer-lora"
	QwenAdapterArtifactV1 = 1
)

// QwenLMHeadLoRA applies scale * B * A to Qwen's frozen lm_head projection.
// A has rank*hidden values and B has vocab*rank values, in row-major order.
type QwenLMHeadLoRA struct {
	Version    int       `json:"version"`
	Type       string    `json:"type"`
	BaseModel  string    `json:"baseModel"`
	Target     string    `json:"target"`
	Rank       int       `json:"rank"`
	Alpha      float32   `json:"alpha"`
	HiddenSize int       `json:"hiddenSize"`
	VocabSize  int       `json:"vocabSize"`
	CreatedAt  time.Time `json:"createdAt"`
	A          []float32 `json:"a"`
	B          []float32 `json:"b"`
}

// QwenProjectionAdapter allows the runner to inject low-rank deltas into any
// frozen Qwen linear projection, including attention and MLP matrices.
type QwenProjectionAdapter interface {
	ApplyProjection(name string, input, output []float32) error
}

func (a QwenLMHeadLoRA) Validate() error {
	if a.Version != 1 || a.Type != QwenLMHeadLoRAType || a.Target != "lm_head" {
		return fmt.Errorf("unsupported Qwen LoRA artifact")
	}
	if a.Rank < 1 || a.HiddenSize < 1 || a.VocabSize < 1 || a.Alpha <= 0 {
		return fmt.Errorf("invalid Qwen LoRA dimensions")
	}
	if len(a.A) != a.Rank*a.HiddenSize || len(a.B) != a.VocabSize*a.Rank {
		return fmt.Errorf("invalid Qwen LoRA tensor sizes")
	}
	return nil
}

func LoadQwenLMHeadLoRA(path string) (QwenLMHeadLoRA, error) {
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		path = filepath.Join(path, "adapter.json")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return QwenLMHeadLoRA{}, err
	}
	var artifact QwenLMHeadLoRA
	if err := json.Unmarshal(data, &artifact); err != nil {
		return QwenLMHeadLoRA{}, fmt.Errorf("read Qwen LoRA artifact: %w", err)
	}
	if err := artifact.Validate(); err != nil {
		return QwenLMHeadLoRA{}, err
	}
	return artifact, nil
}

// LoadQwenAdapter loads either the original output-head artifact or the
// projection-level artifact. Callers only need the projection interface and
// therefore do not need to special-case adapter generations.
func LoadQwenAdapter(path string) (QwenProjectionAdapter, error) {
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		path = filepath.Join(path, "adapter.json")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var header struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &header); err != nil {
		return nil, fmt.Errorf("read Qwen adapter: %w", err)
	}
	switch header.Type {
	case QwenLMHeadLoRAType:
		var artifact QwenLMHeadLoRA
		if err := json.Unmarshal(data, &artifact); err != nil {
			return nil, fmt.Errorf("read Qwen LoRA artifact: %w", err)
		}
		if err := artifact.Validate(); err != nil {
			return nil, err
		}
		return artifact, nil
	case TransformerLoRAType:
		var artifact TransformerLoRA
		if err := json.Unmarshal(data, &artifact); err != nil {
			return nil, fmt.Errorf("read Qwen transformer LoRA artifact: %w", err)
		}
		if err := artifact.Validate(); err != nil {
			return nil, err
		}
		return artifact, nil
	default:
		return nil, fmt.Errorf("unsupported Qwen adapter type %q", header.Type)
	}
}

// SaveTransformerLoRA persists a projection-level artifact that the local
// Qwen runner can load directly. It complements PEFT interchange files with
// Sypra's fast local runtime representation.
func SaveTransformerLoRA(path string, artifact TransformerLoRA) error {
	if err := artifact.Validate(); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(artifact, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0600)
}

func (a QwenLMHeadLoRA) Apply(hidden, logits []float32) error {
	if err := a.Validate(); err != nil {
		return err
	}
	if len(hidden) != a.HiddenSize || len(logits) != a.VocabSize {
		return fmt.Errorf("adapter dimensions do not match Qwen model")
	}
	projected := make([]float32, a.Rank)
	for rank := range projected {
		start := rank * a.HiddenSize
		for i, value := range hidden {
			projected[rank] += a.A[start+i] * value
		}
	}
	scale := a.Alpha / float32(a.Rank)
	for token := range logits {
		start := token * a.Rank
		var delta float32
		for rank, value := range projected {
			delta += a.B[start+rank] * value
		}
		logits[token] += scale * delta
	}
	return nil
}

func (a QwenLMHeadLoRA) ApplyProjection(name string, input, output []float32) error {
	if name != "lm_head.weight" {
		return nil
	}
	return a.Apply(input, output)
}

// ProjectionLoRA is a row-major low-rank delta for one Qwen projection.
type ProjectionLoRA struct {
	Rows  int       `json:"rows"`
	Cols  int       `json:"cols"`
	Rank  int       `json:"rank"`
	Alpha float32   `json:"alpha"`
	A     []float32 `json:"a"`
	B     []float32 `json:"b"`
}

// ProjectionGradient holds the accumulated derivatives for one LoRA pair.
// The frozen projection is never represented here or updated by Sypra.
type ProjectionGradient struct {
	A []float32
	B []float32
}

// NewTransformerGradients allocates a gradient buffer for every trainable
// target in an artifact. It is used by the native reverse-mode Qwen trainer.
func NewTransformerGradients(artifact *TransformerLoRA) map[string]ProjectionGradient {
	gradients := make(map[string]ProjectionGradient, len(artifact.Targets))
	for name, target := range artifact.Targets {
		gradients[name] = ProjectionGradient{A: make([]float32, len(target.A)), B: make([]float32, len(target.B))}
	}
	return gradients
}

func AddTransformerGradients(destination, source map[string]ProjectionGradient) error {
	for name, sourceGradient := range source {
		destinationGradient, ok := destination[name]
		if !ok || len(destinationGradient.A) != len(sourceGradient.A) || len(destinationGradient.B) != len(sourceGradient.B) {
			return fmt.Errorf("incompatible LoRA gradient for %s", name)
		}
		for i := range destinationGradient.A {
			destinationGradient.A[i] += sourceGradient.A[i]
		}
		for i := range destinationGradient.B {
			destinationGradient.B[i] += sourceGradient.B[i]
		}
		destination[name] = destinationGradient
	}
	return nil
}

func (p ProjectionLoRA) Validate() error {
	if p.Rows < 1 || p.Cols < 1 || p.Rank < 1 || p.Alpha <= 0 {
		return fmt.Errorf("invalid LoRA projection dimensions")
	}
	if len(p.A) != p.Rank*p.Cols || len(p.B) != p.Rows*p.Rank {
		return fmt.Errorf("invalid LoRA projection tensor sizes")
	}
	return nil
}

// TransformerLoRA is the multi-layer artifact shape used for adapters on
// q/k/v/o and MLP projections. It is intentionally separate from the legacy
// output-head artifact so old trained models remain loadable.
type TransformerLoRA struct {
	Version   int                       `json:"version"`
	Type      string                    `json:"type"`
	BaseModel string                    `json:"baseModel"`
	CreatedAt time.Time                 `json:"createdAt"`
	Targets   map[string]ProjectionLoRA `json:"targets"`
}

func (a TransformerLoRA) ApplyProjection(name string, input, output []float32) error {
	target, ok := a.Targets[name]
	if !ok {
		return nil
	}
	if err := target.Validate(); err != nil {
		return fmt.Errorf("invalid LoRA target %s: %w", name, err)
	}
	if target.Rows != len(output) || target.Cols != len(input) {
		return fmt.Errorf("invalid LoRA target %s", name)
	}
	projected := make([]float32, target.Rank)
	for rank := range projected {
		for column, value := range input {
			projected[rank] += target.A[rank*target.Cols+column] * value
		}
	}
	scale := target.Alpha / float32(target.Rank)
	for row := range output {
		for rank, value := range projected {
			output[row] += scale * target.B[row*target.Rank+rank] * value
		}
	}
	return nil
}

func (a TransformerLoRA) Validate() error {
	if a.Version != QwenAdapterArtifactV1 || a.Type != TransformerLoRAType || len(a.Targets) == 0 {
		return fmt.Errorf("unsupported Qwen transformer LoRA artifact")
	}
	for name, target := range a.Targets {
		if name == "" {
			return fmt.Errorf("LoRA target name is required")
		}
		if err := target.Validate(); err != nil {
			return fmt.Errorf("invalid LoRA target %s: %w", name, err)
		}
	}
	return nil
}
