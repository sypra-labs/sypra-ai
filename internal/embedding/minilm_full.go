package embedding

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"

	"github.com/sypra-labs/sypra-ai/internal/qwen"
)

// fullMiniLM runs the complete all-MiniLM-L6-v2 BERT encoder: token/position
// embeddings, six self-attention layers, feed-forward blocks, residuals and
// layer normalisation, followed by sentence-transformers mean pooling.
type fullMiniLM struct {
	vocab                     map[string]int
	word, position, tokenType []float32
	embeddingNorm             miniNorm
	layers                    []miniLayer
	dims, heads, maximum      int
	unknown, cls, sep         int
}

type miniLinear struct {
	in, out int
	weight  []float32 // SafeTensors / PyTorch layout: [out, in]
	bias    []float32
}

type miniNorm struct {
	weight, bias []float32
	epsilon      float32
}

type miniLayer struct {
	query, key, value, attentionOut miniLinear
	attentionNorm                   miniNorm
	intermediate, output            miniLinear
	outputNorm                      miniNorm
}

var fullMiniLMCache struct {
	once  sync.Once
	model *fullMiniLM
	err   error
}

func newFullMiniLMEngine() (Engine, error) {
	fullMiniLMCache.once.Do(func() { fullMiniLMCache.model, fullMiniLMCache.err = loadFullMiniLM(defaultMiniLMPath()) })
	if fullMiniLMCache.err != nil {
		return Engine{}, fullMiniLMCache.err
	}
	model := fullMiniLMCache.model
	return Engine{Dimensions: model.dims, MaxTokens: model.maximum, embed: model.embed, tokens: model.tokenCount}, nil
}

func loadFullMiniLM(directory string) (*fullMiniLM, error) {
	if directory == "" {
		return nil, fmt.Errorf("%s needs a local all-MiniLM-L6-v2 checkpoint; set SYPRA_EMBEDDING_MODEL_PATH", DefaultModel)
	}
	var config struct {
		HiddenSize            int     `json:"hidden_size"`
		IntermediateSize      int     `json:"intermediate_size"`
		MaxPositionEmbeddings int     `json:"max_position_embeddings"`
		NumAttentionHeads     int     `json:"num_attention_heads"`
		NumHiddenLayers       int     `json:"num_hidden_layers"`
		LayerNormEpsilon      float64 `json:"layer_norm_eps"`
	}
	configBytes, err := os.ReadFile(filepath.Join(directory, "config.json"))
	if err != nil {
		return nil, err
	}
	if err = json.Unmarshal(configBytes, &config); err != nil {
		return nil, fmt.Errorf("parse MiniLM config: %w", err)
	}
	if config.HiddenSize < 1 || config.IntermediateSize < 1 || config.NumAttentionHeads < 1 || config.NumHiddenLayers < 1 || config.HiddenSize%config.NumAttentionHeads != 0 {
		return nil, fmt.Errorf("invalid MiniLM transformer configuration")
	}
	vocab, err := loadVocab(filepath.Join(directory, "vocab.txt"))
	if err != nil {
		return nil, err
	}
	file, err := qwen.OpenSafeTensorFile(filepath.Join(directory, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer file.Close()
	matrix := func(name string, rows, columns int) ([]float32, error) { return miniMatrix(file, name, rows, columns) }
	vector := func(name string, size int) ([]float32, error) { return miniVector(file, name, size) }
	word, err := miniAnyRowsMatrix(file, "embeddings.word_embeddings.weight", config.HiddenSize)
	if err != nil {
		return nil, err
	}
	position, err := matrix("embeddings.position_embeddings.weight", config.MaxPositionEmbeddings, config.HiddenSize)
	if err != nil {
		return nil, err
	}
	tokenType, err := miniAnyRowsMatrix(file, "embeddings.token_type_embeddings.weight", config.HiddenSize)
	if err != nil {
		return nil, err
	}
	embeddingNorm, err := miniReadNorm(vector, "embeddings.LayerNorm", config.HiddenSize, config.LayerNormEpsilon)
	if err != nil {
		return nil, err
	}
	model := &fullMiniLM{vocab: vocab, word: word, position: position, tokenType: tokenType, embeddingNorm: embeddingNorm, dims: config.HiddenSize, heads: config.NumAttentionHeads, maximum: miniMaximum(directory, config.MaxPositionEmbeddings), unknown: tokenID(vocab, "[UNK]", 100), cls: tokenID(vocab, "[CLS]", 101), sep: tokenID(vocab, "[SEP]", 102)}
	for index := 0; index < config.NumHiddenLayers; index++ {
		prefix := fmt.Sprintf("encoder.layer.%d.", index)
		layer := miniLayer{}
		if layer.query, err = miniReadLinear(matrix, vector, prefix+"attention.self.query", config.HiddenSize, config.HiddenSize); err != nil {
			return nil, err
		}
		if layer.key, err = miniReadLinear(matrix, vector, prefix+"attention.self.key", config.HiddenSize, config.HiddenSize); err != nil {
			return nil, err
		}
		if layer.value, err = miniReadLinear(matrix, vector, prefix+"attention.self.value", config.HiddenSize, config.HiddenSize); err != nil {
			return nil, err
		}
		if layer.attentionOut, err = miniReadLinear(matrix, vector, prefix+"attention.output.dense", config.HiddenSize, config.HiddenSize); err != nil {
			return nil, err
		}
		if layer.attentionNorm, err = miniReadNorm(vector, prefix+"attention.output.LayerNorm", config.HiddenSize, config.LayerNormEpsilon); err != nil {
			return nil, err
		}
		if layer.intermediate, err = miniReadLinear(matrix, vector, prefix+"intermediate.dense", config.IntermediateSize, config.HiddenSize); err != nil {
			return nil, err
		}
		if layer.output, err = miniReadLinear(matrix, vector, prefix+"output.dense", config.HiddenSize, config.IntermediateSize); err != nil {
			return nil, err
		}
		if layer.outputNorm, err = miniReadNorm(vector, prefix+"output.LayerNorm", config.HiddenSize, config.LayerNormEpsilon); err != nil {
			return nil, err
		}
		model.layers = append(model.layers, layer)
	}
	return model, nil
}

func miniMaximum(directory string, maximum int) int {
	limit := 256
	if data, err := os.ReadFile(filepath.Join(directory, "sentence_bert_config.json")); err == nil {
		var config struct {
			Max int `json:"max_seq_length"`
		}
		if json.Unmarshal(data, &config) == nil && config.Max > 0 {
			limit = config.Max
		}
	}
	if maximum > 0 && limit > maximum {
		limit = maximum
	}
	return limit
}

func miniF32(file *qwen.SafeTensorFile, name string) ([]float32, []int, error) {
	dtype, _, _, err := file.Entry(name)
	if err != nil {
		return nil, nil, err
	}
	if dtype != "F32" {
		return nil, nil, fmt.Errorf("MiniLM tensor %s has dtype %s; F32 is required", name, dtype)
	}
	raw, shape, err := file.ReadBytes(name)
	if err != nil {
		return nil, nil, err
	}
	if len(raw)%4 != 0 {
		return nil, nil, fmt.Errorf("invalid MiniLM tensor size for %s", name)
	}
	values := make([]float32, len(raw)/4)
	for index := range values {
		values[index] = math.Float32frombits(binary.LittleEndian.Uint32(raw[index*4:]))
	}
	return values, shape, nil
}

func miniMatrix(file *qwen.SafeTensorFile, name string, rows, columns int) ([]float32, error) {
	values, shape, err := miniF32(file, name)
	if err != nil {
		return nil, err
	}
	if len(shape) != 2 || shape[0] != rows || shape[1] != columns {
		return nil, fmt.Errorf("MiniLM tensor %s has shape %v, want [%d %d]", name, shape, rows, columns)
	}
	return values, nil
}

func miniAnyRowsMatrix(file *qwen.SafeTensorFile, name string, columns int) ([]float32, error) {
	values, shape, err := miniF32(file, name)
	if err != nil {
		return nil, err
	}
	if len(shape) != 2 || shape[0] < 1 || shape[1] != columns {
		return nil, fmt.Errorf("MiniLM tensor %s has shape %v", name, shape)
	}
	return values, nil
}

func miniVector(file *qwen.SafeTensorFile, name string, size int) ([]float32, error) {
	values, shape, err := miniF32(file, name)
	if err != nil {
		return nil, err
	}
	if len(shape) != 1 || shape[0] != size {
		return nil, fmt.Errorf("MiniLM tensor %s has shape %v, want [%d]", name, shape, size)
	}
	return values, nil
}

func miniReadLinear(matrix func(string, int, int) ([]float32, error), vector func(string, int) ([]float32, error), prefix string, out, in int) (miniLinear, error) {
	weight, err := matrix(prefix+".weight", out, in)
	if err != nil {
		return miniLinear{}, err
	}
	bias, err := vector(prefix+".bias", out)
	if err != nil {
		return miniLinear{}, err
	}
	return miniLinear{in: in, out: out, weight: weight, bias: bias}, nil
}

func miniReadNorm(vector func(string, int) ([]float32, error), prefix string, size int, epsilon float64) (miniNorm, error) {
	weight, err := vector(prefix+".weight", size)
	if err != nil {
		return miniNorm{}, err
	}
	bias, err := vector(prefix+".bias", size)
	if err != nil {
		return miniNorm{}, err
	}
	return miniNorm{weight: weight, bias: bias, epsilon: float32(epsilon)}, nil
}

func (m *fullMiniLM) tokenCount(text string) int { return len(m.ids(text)) }

func (m *fullMiniLM) ids(text string) []int {
	ids := []int{m.cls}
	for _, word := range miniLMWords(text) {
		pieces := m.wordPieces(word)
		if len(ids)+len(pieces)+1 > m.maximum {
			break
		}
		ids = append(ids, pieces...)
	}
	return append(ids, m.sep)
}

func (m *fullMiniLM) embed(text string) []float32 {
	ids := m.ids(text)
	sequence := len(ids)
	hidden := make([]float32, sequence*m.dims)
	for position, id := range ids {
		if id < 0 || (id+1)*m.dims > len(m.word) {
			id = m.unknown
		}
		for dimension := 0; dimension < m.dims; dimension++ {
			hidden[position*m.dims+dimension] = m.word[id*m.dims+dimension] + m.position[position*m.dims+dimension] + m.tokenType[dimension]
		}
	}
	hidden = miniNormalize(hidden, sequence, m.embeddingNorm)
	for _, layer := range m.layers {
		hidden = layer.forward(hidden, sequence, m.dims, m.heads)
	}
	pooled := make([]float32, m.dims)
	for position := 0; position < sequence; position++ {
		for dimension := 0; dimension < m.dims; dimension++ {
			pooled[dimension] += hidden[position*m.dims+dimension]
		}
	}
	for dimension := range pooled {
		pooled[dimension] /= float32(sequence)
	}
	normalize(pooled)
	return pooled
}

func (layer miniLayer) forward(input []float32, sequence, dims, heads int) []float32 {
	query, key, value := miniProject(input, sequence, layer.query), miniProject(input, sequence, layer.key), miniProject(input, sequence, layer.value)
	headSize := dims / heads
	context := make([]float32, len(input))
	scores := make([]float32, sequence)
	for head := 0; head < heads; head++ {
		for row := 0; row < sequence; row++ {
			maximum := float32(-math.MaxFloat32)
			for column := 0; column < sequence; column++ {
				var dot float32
				for dimension := 0; dimension < headSize; dimension++ {
					dot += query[row*dims+head*headSize+dimension] * key[column*dims+head*headSize+dimension]
				}
				scores[column] = dot / float32(math.Sqrt(float64(headSize)))
				if scores[column] > maximum {
					maximum = scores[column]
				}
			}
			var total float32
			for column := range scores {
				scores[column] = float32(math.Exp(float64(scores[column] - maximum)))
				total += scores[column]
			}
			for column := range scores {
				scores[column] /= total
			}
			for column := 0; column < sequence; column++ {
				for dimension := 0; dimension < headSize; dimension++ {
					context[row*dims+head*headSize+dimension] += scores[column] * value[column*dims+head*headSize+dimension]
				}
			}
		}
	}
	attention := miniProject(context, sequence, layer.attentionOut)
	for index := range attention {
		attention[index] += input[index]
	}
	attention = miniNormalize(attention, sequence, layer.attentionNorm)
	feedForward := miniProject(attention, sequence, layer.intermediate)
	for index := range feedForward {
		feedForward[index] = miniGELU(feedForward[index])
	}
	output := miniProject(feedForward, sequence, layer.output)
	for index := range output {
		output[index] += attention[index]
	}
	return miniNormalize(output, sequence, layer.outputNorm)
}

func miniProject(input []float32, sequence int, layer miniLinear) []float32 {
	output := make([]float32, sequence*layer.out)
	for position := 0; position < sequence; position++ {
		values := input[position*layer.in : (position+1)*layer.in]
		for out := 0; out < layer.out; out++ {
			total := layer.bias[out]
			weights := layer.weight[out*layer.in : (out+1)*layer.in]
			for in, value := range values {
				total += value * weights[in]
			}
			output[position*layer.out+out] = total
		}
	}
	return output
}

func miniNormalize(input []float32, sequence int, norm miniNorm) []float32 {
	output, dims := make([]float32, len(input)), len(norm.weight)
	for position := 0; position < sequence; position++ {
		row := input[position*dims : (position+1)*dims]
		var mean float32
		for _, value := range row {
			mean += value
		}
		mean /= float32(dims)
		var variance float32
		for _, value := range row {
			diff := value - mean
			variance += diff * diff
		}
		scale := float32(1 / math.Sqrt(float64(variance/float32(dims)+norm.epsilon)))
		for dimension, value := range row {
			output[position*dims+dimension] = (value-mean)*scale*norm.weight[dimension] + norm.bias[dimension]
		}
	}
	return output
}

func miniGELU(value float32) float32 {
	return .5 * value * (1 + float32(math.Tanh(math.Sqrt(2/math.Pi)*(float64(value)+.044715*math.Pow(float64(value), 3)))))
}

func (m *fullMiniLM) wordPieces(word string) []int {
	if id, ok := m.vocab[word]; ok {
		return []int{id}
	}
	runes, output := []rune(word), []int{}
	for start := 0; start < len(runes); {
		end, matched := len(runes), -1
		for end > start {
			piece := string(runes[start:end])
			if start > 0 {
				piece = "##" + piece
			}
			if id, ok := m.vocab[piece]; ok {
				matched = id
				break
			}
			end--
		}
		if matched < 0 {
			return []int{m.unknown}
		}
		output, start = append(output, matched), end
	}
	return output
}
