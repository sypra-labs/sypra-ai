package qwen

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/sypra-labs/sypra-ai/internal/adapter"
)

type Config struct {
	HiddenSize            int      `json:"hidden_size"`
	IntermediateSize      int      `json:"intermediate_size"`
	NumAttentionHeads     int      `json:"num_attention_heads"`
	NumKeyValueHeads      int      `json:"num_key_value_heads"`
	NumHiddenLayers       int      `json:"num_hidden_layers"`
	HeadDim               int      `json:"head_dim"`
	VocabSize             int      `json:"vocab_size"`
	RMSNormEps            float32  `json:"rms_norm_eps"`
	RopeTheta             float64  `json:"rope_theta"`
	MaxPositionEmbeddings int      `json:"max_position_embeddings"`
	Architectures         []string `json:"architectures"`
}

type Model struct {
	Config    Config
	Tokenizer *Tokenizer
	weights   *SafeTensorFile
	cache     map[string]weight
	nf4Cache  map[string]*nf4Weight
	useNF4    bool
	doubleNF4 bool
	executor  MatrixExecutor
	mu        sync.Mutex
}

// MatrixExecutor optionally accelerates frozen projection products. Adapter
// updates and control-plane work stay in Go on the CPU.
type MatrixExecutor interface {
	MatrixVec(model *Model, name string, input []float32) ([]float32, error)
	MatrixVecTranspose(model *Model, name string, outputGradient []float32) ([]float32, error)
	Close() error
}

type weight struct {
	data  []byte
	shape []int
}

func LoadModel(modelDir string) (*Model, error) {
	configData, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return nil, err
	}
	var config Config
	if err = json.Unmarshal(configData, &config); err != nil {
		return nil, fmt.Errorf("read Qwen config: %w", err)
	}
	if len(config.Architectures) == 0 || config.Architectures[0] != "Qwen3ForCausalLM" {
		return nil, fmt.Errorf("only Qwen3ForCausalLM is supported")
	}
	if config.HiddenSize < 1 || config.HeadDim < 1 || config.NumAttentionHeads < 1 || config.NumKeyValueHeads < 1 || config.NumAttentionHeads%config.NumKeyValueHeads != 0 {
		return nil, fmt.Errorf("invalid Qwen attention configuration")
	}
	if config.MaxPositionEmbeddings < 1 {
		return nil, fmt.Errorf("invalid Qwen maximum context length")
	}
	weights, err := OpenSafeTensorFile(filepath.Join(modelDir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	tokenizer, err := LoadTokenizer(modelDir)
	if err != nil {
		_ = weights.Close()
		return nil, err
	}
	return &Model{Config: config, Tokenizer: tokenizer, weights: weights, cache: map[string]weight{}, nf4Cache: map[string]*nf4Weight{}}, nil
}

func (m *Model) Close() error {
	if m.executor != nil {
		_ = m.executor.Close()
	}
	return m.weights.Close()
}

// SetMatrixExecutor attaches an optional accelerator to the model instance.
// It must be configured before the model is used by a runner or trainer.
func (m *Model) SetMatrixExecutor(executor MatrixExecutor) { m.executor = executor }

func (m *Model) tensor(name string) (weight, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if cached, ok := m.cache[name]; ok {
		return cached, nil
	}
	dtype, _, _, err := m.weights.Entry(name)
	if err != nil {
		return weight{}, err
	}
	if dtype != "BF16" {
		return weight{}, fmt.Errorf("tensor %s has %s; only BF16 Qwen weights are supported", name, dtype)
	}
	data, shape, err := m.weights.ReadBytes(name)
	if err != nil {
		return weight{}, err
	}
	w := weight{data: data, shape: shape}
	m.cache[name] = w
	return w, nil
}

func (m *Model) vector(name string) ([]float32, error) {
	w, err := m.tensor(name)
	if err != nil {
		return nil, err
	}
	if len(w.shape) != 1 {
		return nil, fmt.Errorf("%s is not a vector", name)
	}
	result := make([]float32, w.shape[0])
	for i := range result {
		result[i] = bfloat(w.data, i)
	}
	return result, nil
}

func (m *Model) matrixVec(name string, input []float32) ([]float32, error) {
	if m.executor != nil {
		return m.executor.MatrixVec(m, name, input)
	}
	if m.useNF4 {
		quantized, err := m.nf4(name)
		if err != nil {
			return nil, err
		}
		return quantized.matrixVec(input)
	}
	w, err := m.tensor(name)
	if err != nil {
		return nil, err
	}
	if len(w.shape) != 2 || w.shape[1] != len(input) {
		return nil, fmt.Errorf("matrix %s shape %v cannot multiply %d values", name, w.shape, len(input))
	}
	result := make([]float32, w.shape[0])
	parallelRows(w.shape[0], func(row int) {
		start := row * w.shape[1]
		for column, value := range input {
			result[row] += bfloat(w.data, start+column) * value
		}
	})
	return result, nil
}

// parallelRows uses the host's available CPU cores for independent output
// rows. Small projections remain single-threaded to avoid scheduling cost.
func parallelRows(rows int, work func(row int)) {
	workers := runtime.GOMAXPROCS(0)
	if rows < 1024 || workers < 2 {
		for row := 0; row < rows; row++ {
			work(row)
		}
		return
	}
	if workers > rows {
		workers = rows
	}
	chunk := (rows + workers - 1) / workers
	var group sync.WaitGroup
	for start := 0; start < rows; start += chunk {
		end := start + chunk
		if end > rows {
			end = rows
		}
		group.Add(1)
		go func(start, end int) {
			defer group.Done()
			for row := start; row < end; row++ {
				work(row)
			}
		}(start, end)
	}
	group.Wait()
}

// matrixVecWithAdapter applies a frozen projection then, when present, its
// low-rank delta. The base implementation is shared by LoRA and QLoRA; QLoRA
// changes the frozen matrix implementation through EnableNF4.
func (m *Model) matrixVecWithAdapter(name string, input []float32, lora adapter.QwenProjectionAdapter) ([]float32, error) {
	output, err := m.matrixVec(name, input)
	if err != nil {
		return nil, err
	}
	if lora != nil {
		if err := lora.ApplyProjection(name, input, output); err != nil {
			return nil, err
		}
	}
	return output, nil
}

// EnableNF4 switches this model instance to lazy NF4 matrix quantization.
// Training loads an independent model instance, so chat inference is not
// affected. Base matrices are discarded from the in-memory BF16 cache after
// their 4-bit form is built.
func (m *Model) EnableNF4(doubleQuant bool) {
	m.mu.Lock()
	m.useNF4, m.doubleNF4 = true, doubleQuant
	m.mu.Unlock()
}

func (m *Model) nf4(name string) (*nf4Weight, error) {
	m.mu.Lock()
	if cached := m.nf4Cache[name]; cached != nil {
		m.mu.Unlock()
		return cached, nil
	}
	doubleQuant := m.doubleNF4
	m.mu.Unlock()
	w, err := m.tensor(name)
	if err != nil {
		return nil, err
	}
	quantized := quantizeNF4(w, doubleQuant)
	m.mu.Lock()
	if cached := m.nf4Cache[name]; cached != nil {
		m.mu.Unlock()
		return cached, nil
	}
	m.nf4Cache[name] = quantized
	delete(m.cache, name)
	m.mu.Unlock()
	return quantized, nil
}

func (m *Model) embedding(id int) ([]float32, error) {
	w, err := m.tensor("model.embed_tokens.weight")
	if err != nil {
		return nil, err
	}
	if len(w.shape) != 2 || id < 0 || id >= w.shape[0] {
		return nil, fmt.Errorf("token %d is outside Qwen vocabulary", id)
	}
	value := make([]float32, w.shape[1])
	for i := range value {
		value[i] = bfloat(w.data, id*w.shape[1]+i)
	}
	return value, nil
}

func bfloat(data []byte, index int) float32 {
	return math.Float32frombits(uint32(binary.LittleEndian.Uint16(data[index*2:])) << 16)
}

type kvCache struct{ keys, values []float32 }

// State holds a single sequence's attention cache. It is intentionally not
// shared across training examples or concurrent chat requests.
type State struct{ layers []kvCache }

func NewState(layers int) *State { return &State{layers: make([]kvCache, layers)} }

func (m *Model) Forward(token, position int, s *State) ([]float32, error) {
	return m.ForwardWithAdapter(token, position, s, nil)
}

func (m *Model) ForwardWithAdapter(token, position int, s *State, lora adapter.QwenProjectionAdapter) ([]float32, error) {
	x, err := m.forwardHidden(token, position, s, lora)
	if err != nil {
		return nil, err
	}
	logits, err := m.matrixVecWithAdapter("lm_head.weight", x, lora)
	if err != nil {
		return nil, err
	}
	return logits, nil
}

// ForwardHidden runs Qwen up to its final normalized hidden representation.
// It is used by the LoRA trainer while the base checkpoint remains frozen.
func (m *Model) ForwardHidden(token, position int, s *State) ([]float32, error) {
	return m.forwardHidden(token, position, s, nil)
}

func (m *Model) forwardHidden(token, position int, s *State, lora adapter.QwenProjectionAdapter) ([]float32, error) {
	x, err := m.embedding(token)
	if err != nil {
		return nil, err
	}
	for layer := 0; layer < m.Config.NumHiddenLayers; layer++ {
		prefix := fmt.Sprintf("model.layers.%d.", layer)
		residual := append([]float32(nil), x...)
		norm, err := m.vector(prefix + "input_layernorm.weight")
		if err != nil {
			return nil, err
		}
		x = rmsNorm(x, norm, m.Config.RMSNormEps)
		q, err := m.matrixVecWithAdapter(prefix+"self_attn.q_proj.weight", x, lora)
		if err != nil {
			return nil, err
		}
		k, err := m.matrixVecWithAdapter(prefix+"self_attn.k_proj.weight", x, lora)
		if err != nil {
			return nil, err
		}
		v, err := m.matrixVecWithAdapter(prefix+"self_attn.v_proj.weight", x, lora)
		if err != nil {
			return nil, err
		}
		qNorm, err := m.vector(prefix + "self_attn.q_norm.weight")
		if err != nil {
			return nil, err
		}
		kNorm, err := m.vector(prefix + "self_attn.k_norm.weight")
		if err != nil {
			return nil, err
		}
		normalizeHeads(q, qNorm, m.Config.RMSNormEps)
		normalizeHeads(k, kNorm, m.Config.RMSNormEps)
		applyRoPE(q, position, m.Config.HeadDim, m.Config.RopeTheta)
		applyRoPE(k, position, m.Config.HeadDim, m.Config.RopeTheta)
		cache := &s.layers[layer]
		cache.keys, cache.values = append(cache.keys, k...), append(cache.values, v...)
		attention := m.attention(q, cache)
		projected, err := m.matrixVecWithAdapter(prefix+"self_attn.o_proj.weight", attention, lora)
		if err != nil {
			return nil, err
		}
		add(residual, projected)
		norm, err = m.vector(prefix + "post_attention_layernorm.weight")
		if err != nil {
			return nil, err
		}
		x = rmsNorm(residual, norm, m.Config.RMSNormEps)
		gate, err := m.matrixVecWithAdapter(prefix+"mlp.gate_proj.weight", x, lora)
		if err != nil {
			return nil, err
		}
		up, err := m.matrixVecWithAdapter(prefix+"mlp.up_proj.weight", x, lora)
		if err != nil {
			return nil, err
		}
		for i := range gate {
			gate[i] = silu(gate[i]) * up[i]
		}
		down, err := m.matrixVecWithAdapter(prefix+"mlp.down_proj.weight", gate, lora)
		if err != nil {
			return nil, err
		}
		add(residual, down)
		x = residual
	}
	norm, err := m.vector("model.norm.weight")
	if err != nil {
		return nil, err
	}
	x = rmsNorm(x, norm, m.Config.RMSNormEps)
	return x, nil
}

// LMHeadLogit returns a frozen base-model output score for one vocabulary ID.
// Sampled-softmax LoRA training only needs a target and a small set of
// negatives, avoiding allocation of a full vocabulary gradient per token.
func (m *Model) LMHeadLogit(hidden []float32, tokenID int) (float32, error) {
	if m.useNF4 {
		weights, err := m.nf4("lm_head.weight")
		if err != nil {
			return 0, err
		}
		if len(weights.shape) != 2 || len(hidden) != weights.shape[1] || tokenID < 0 || tokenID >= weights.shape[0] {
			return 0, fmt.Errorf("lm_head token or hidden dimensions are invalid")
		}
		var value float32
		start := tokenID * weights.shape[1]
		for i, feature := range hidden {
			value += weights.value(start+i) * feature
		}
		return value, nil
	}
	weights, err := m.tensor("lm_head.weight")
	if err != nil {
		return 0, err
	}
	if len(weights.shape) != 2 || len(hidden) != weights.shape[1] || tokenID < 0 || tokenID >= weights.shape[0] {
		return 0, fmt.Errorf("lm_head token or hidden dimensions are invalid")
	}
	var value float32
	start := tokenID * weights.shape[1]
	for i, feature := range hidden {
		value += bfloat(weights.data, start+i) * feature
	}
	return value, nil
}

// LMHeadLogits returns the full vocabulary logits from the frozen output
// projection. Fine-tuning uses this for exact causal cross-entropy instead of
// a sampled vocabulary approximation.
func (m *Model) LMHeadLogits(hidden []float32) ([]float32, error) {
	return m.matrixVec("lm_head.weight", hidden)
}

func (m *Model) attention(q []float32, cache *kvCache) []float32 {
	headDim, heads, kvHeads := m.Config.HeadDim, m.Config.NumAttentionHeads, m.Config.NumKeyValueHeads
	steps := len(cache.keys) / (kvHeads * headDim)
	output := make([]float32, len(q))
	for head := 0; head < heads; head++ {
		kvHead := head / (heads / kvHeads)
		scores := make([]float32, steps)
		max := float32(-math.MaxFloat32)
		for step := 0; step < steps; step++ {
			keyOffset := step*kvHeads*headDim + kvHead*headDim
			var score float32
			for i := 0; i < headDim; i++ {
				score += q[head*headDim+i] * cache.keys[keyOffset+i]
			}
			scores[step] = score / float32(math.Sqrt(float64(headDim)))
			if scores[step] > max {
				max = scores[step]
			}
		}
		var total float32
		for i := range scores {
			scores[i] = float32(math.Exp(float64(scores[i] - max)))
			total += scores[i]
		}
		for step, score := range scores {
			valueOffset := step*kvHeads*headDim + kvHead*headDim
			weight := score / total
			for i := 0; i < headDim; i++ {
				output[head*headDim+i] += weight * cache.values[valueOffset+i]
			}
		}
	}
	return output
}

func rmsNorm(value, weight []float32, eps float32) []float32 {
	var sum float32
	for _, v := range value {
		sum += v * v
	}
	scale := float32(1 / math.Sqrt(float64(sum/float32(len(value))+eps)))
	result := make([]float32, len(value))
	for i := range result {
		result[i] = value[i] * scale * weight[i]
	}
	return result
}

func normalizeHeads(value, weight []float32, eps float32) {
	for start := 0; start < len(value); start += len(weight) {
		var sum float32
		for i := range weight {
			sum += value[start+i] * value[start+i]
		}
		scale := float32(1 / math.Sqrt(float64(sum/float32(len(weight))+eps)))
		for i := range weight {
			value[start+i] *= scale * weight[i]
		}
	}
}

func applyRoPE(value []float32, position, headDim int, theta float64) {
	for head := 0; head < len(value)/headDim; head++ {
		half := headDim / 2
		for i := 0; i < half; i++ {
			angle := float64(position) / math.Pow(theta, float64(2*i)/float64(headDim))
			cos, sin := float32(math.Cos(angle)), float32(math.Sin(angle))
			left, right := head*headDim+i, head*headDim+half+i
			a, b := value[left], value[right]
			value[left], value[right] = a*cos-b*sin, a*sin+b*cos
		}
	}
}

func add(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}
func silu(x float32) float32 { return x / (1 + float32(math.Exp(float64(-x)))) }

func argmax(values []float32) int {
	best := 0
	for i := 1; i < len(values); i++ {
		if values[i] > values[best] {
			best = i
		}
	}
	return best
}
