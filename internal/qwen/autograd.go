package qwen

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/sypra-labs/sypra-ai/internal/adapter"
)

// ProjectionShape returns the frozen matrix dimensions for an adapter target.
func (m *Model) ProjectionShape(name string) (rows, cols int, err error) {
	if m.useNF4 {
		quantized, err := m.nf4(name)
		if err != nil {
			return 0, 0, err
		}
		if len(quantized.shape) != 2 {
			return 0, 0, fmt.Errorf("%s is not a matrix", name)
		}
		return quantized.shape[0], quantized.shape[1], nil
	}
	w, err := m.tensor(name)
	if err != nil {
		return 0, 0, err
	}
	if len(w.shape) != 2 {
		return 0, 0, fmt.Errorf("%s is not a matrix", name)
	}
	return w.shape[0], w.shape[1], nil
}

type trainingLinearTape struct {
	name  string
	input []float32
}

type trainingLayerTape struct {
	input, norm1, residual, norm2 []float32
	q, k, v, qRaw, kRaw           []float32 // q and k are post-RoPE.
	qLinear, kLinear, vLinear     trainingLinearTape
	oLinear, gateLinear           trainingLinearTape
	upLinear, downLinear          trainingLinearTape
	attention, gateRaw, upRaw     []float32
	activation                    []float32
	weights                       [][]float32
	dOutput, dQ, dK, dV           []float32
}

type trainingSequence struct {
	layers                  [][]*trainingLayerTape
	finalInput, finalHidden [][]float32
	finalGradient           [][]float32
}

// CausalLoRAGradients performs a full reverse pass for one teacher-forced
// sequence. Labels start at labelStart, so prompt tokens build causal state
// but contribute no loss. Every configured adapter projection receives A/B
// gradients; frozen base tensors are read-only.
func (m *Model) CausalLoRAGradients(tokens []int, labelStart int, artifact *adapter.TransformerLoRA) (float64, int, map[string]adapter.ProjectionGradient, error) {
	sequence, err := m.forwardTrainingSequence(tokens, artifact)
	if err != nil {
		return 0, 0, nil, err
	}
	if labelStart < 1 || labelStart >= len(tokens) {
		return 0, 0, nil, fmt.Errorf("label start must follow at least one prompt token")
	}
	gradients := adapter.NewTransformerGradients(artifact)
	var totalLoss float64
	count := 0
	for position := labelStart - 1; position < len(tokens)-1; position++ {
		target := tokens[position+1]
		logits, tape, err := m.forwardTrainingLinear("lm_head.weight", sequence.finalHidden[position], artifact)
		if err != nil {
			return 0, 0, nil, err
		}
		probabilities := trainingSoftmax(logits)
		if target < 0 || target >= len(probabilities) {
			return 0, 0, nil, fmt.Errorf("target token %d is outside Qwen vocabulary", target)
		}
		totalLoss -= math.Log(math.Max(float64(probabilities[target]), 1e-12))
		probabilities[target]--
		dHidden, err := m.backwardTrainingLinear(tape, probabilities, artifact, gradients)
		if err != nil {
			return 0, 0, nil, err
		}
		addTraining(sequence.finalGradient[position], dHidden)
		count++
	}
	if count == 0 {
		return 0, 0, nil, fmt.Errorf("sequence has no response labels")
	}
	if err := m.backwardTrainingSequence(sequence, artifact, gradients); err != nil {
		return 0, 0, nil, err
	}
	return totalLoss, count, gradients, nil
}

// CausalLoRALoss is the validation counterpart of CausalLoRAGradients. It
// uses the same causal forward path without allocating gradient buffers.
func (m *Model) CausalLoRALoss(tokens []int, labelStart int, artifact *adapter.TransformerLoRA) (float64, int, error) {
	sequence, err := m.forwardTrainingSequence(tokens, artifact)
	if err != nil {
		return 0, 0, err
	}
	if labelStart < 1 || labelStart >= len(tokens) {
		return 0, 0, fmt.Errorf("label start must follow at least one prompt token")
	}
	var total float64
	count := 0
	for position := labelStart - 1; position < len(tokens)-1; position++ {
		logits, _, err := m.forwardTrainingLinear("lm_head.weight", sequence.finalHidden[position], artifact)
		if err != nil {
			return 0, 0, err
		}
		probabilities := trainingSoftmax(logits)
		target := tokens[position+1]
		if target < 0 || target >= len(probabilities) {
			return 0, 0, fmt.Errorf("target token %d is outside Qwen vocabulary", target)
		}
		total -= math.Log(math.Max(float64(probabilities[target]), 1e-12))
		count++
	}
	return total, count, nil
}

func (m *Model) forwardTrainingSequence(tokens []int, artifact *adapter.TransformerLoRA) (*trainingSequence, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("training sequence is empty")
	}
	if len(tokens) > m.Config.MaxPositionEmbeddings {
		return nil, fmt.Errorf("training sequence has %d tokens; Qwen context limit is %d", len(tokens), m.Config.MaxPositionEmbeddings)
	}
	sequence := &trainingSequence{
		layers:        make([][]*trainingLayerTape, m.Config.NumHiddenLayers),
		finalInput:    make([][]float32, len(tokens)),
		finalHidden:   make([][]float32, len(tokens)),
		finalGradient: make([][]float32, len(tokens)),
	}
	finalNorm, err := m.vector("model.norm.weight")
	if err != nil {
		return nil, err
	}
	for position, token := range tokens {
		x, err := m.embedding(token)
		if err != nil {
			return nil, err
		}
		for layer := 0; layer < m.Config.NumHiddenLayers; layer++ {
			prefix := fmt.Sprintf("model.layers.%d.", layer)
			tape, output, err := m.forwardTrainingLayer(prefix, x, position, sequence.layers[layer], artifact)
			if err != nil {
				return nil, err
			}
			sequence.layers[layer] = append(sequence.layers[layer], tape)
			x = output
		}
		sequence.finalInput[position] = x
		sequence.finalHidden[position] = rmsNorm(x, finalNorm, m.Config.RMSNormEps)
		sequence.finalGradient[position] = make([]float32, len(x))
	}
	return sequence, nil
}

func (m *Model) forwardTrainingLayer(prefix string, input []float32, position int, prior []*trainingLayerTape, artifact *adapter.TransformerLoRA) (*trainingLayerTape, []float32, error) {
	tape := &trainingLayerTape{input: input}
	inputNormWeight, err := m.vector(prefix + "input_layernorm.weight")
	if err != nil {
		return nil, nil, err
	}
	tape.norm1 = rmsNorm(input, inputNormWeight, m.Config.RMSNormEps)
	if tape.q, tape.qLinear, err = m.forwardTrainingLinear(prefix+"self_attn.q_proj.weight", tape.norm1, artifact); err != nil {
		return nil, nil, err
	}
	if tape.k, tape.kLinear, err = m.forwardTrainingLinear(prefix+"self_attn.k_proj.weight", tape.norm1, artifact); err != nil {
		return nil, nil, err
	}
	if tape.v, tape.vLinear, err = m.forwardTrainingLinear(prefix+"self_attn.v_proj.weight", tape.norm1, artifact); err != nil {
		return nil, nil, err
	}
	tape.qRaw = append([]float32(nil), tape.q...)
	tape.kRaw = append([]float32(nil), tape.k...)
	qNorm, err := m.vector(prefix + "self_attn.q_norm.weight")
	if err != nil {
		return nil, nil, err
	}
	kNorm, err := m.vector(prefix + "self_attn.k_norm.weight")
	if err != nil {
		return nil, nil, err
	}
	normalizeHeads(tape.q, qNorm, m.Config.RMSNormEps)
	normalizeHeads(tape.k, kNorm, m.Config.RMSNormEps)
	applyRoPE(tape.q, position, m.Config.HeadDim, m.Config.RopeTheta)
	applyRoPE(tape.k, position, m.Config.HeadDim, m.Config.RopeTheta)
	tape.attention, tape.weights = m.trainingAttention(tape.q, append(prior, tape))
	projected, linear, err := m.forwardTrainingLinear(prefix+"self_attn.o_proj.weight", tape.attention, artifact)
	if err != nil {
		return nil, nil, err
	}
	tape.oLinear = linear
	tape.residual = append([]float32(nil), input...)
	addTraining(tape.residual, projected)
	postNormWeight, err := m.vector(prefix + "post_attention_layernorm.weight")
	if err != nil {
		return nil, nil, err
	}
	tape.norm2 = rmsNorm(tape.residual, postNormWeight, m.Config.RMSNormEps)
	if tape.gateRaw, tape.gateLinear, err = m.forwardTrainingLinear(prefix+"mlp.gate_proj.weight", tape.norm2, artifact); err != nil {
		return nil, nil, err
	}
	if tape.upRaw, tape.upLinear, err = m.forwardTrainingLinear(prefix+"mlp.up_proj.weight", tape.norm2, artifact); err != nil {
		return nil, nil, err
	}
	tape.activation = make([]float32, len(tape.gateRaw))
	for i := range tape.activation {
		tape.activation[i] = silu(tape.gateRaw[i]) * tape.upRaw[i]
	}
	down, linear, err := m.forwardTrainingLinear(prefix+"mlp.down_proj.weight", tape.activation, artifact)
	if err != nil {
		return nil, nil, err
	}
	tape.downLinear = linear
	output := append([]float32(nil), tape.residual...)
	addTraining(output, down)
	tape.dOutput = make([]float32, len(output))
	tape.dQ, tape.dK, tape.dV = make([]float32, len(tape.q)), make([]float32, len(tape.k)), make([]float32, len(tape.v))
	return tape, output, nil
}

func (m *Model) forwardTrainingLinear(name string, input []float32, artifact *adapter.TransformerLoRA) ([]float32, trainingLinearTape, error) {
	output, err := m.matrixVec(name, input)
	if err != nil {
		return nil, trainingLinearTape{}, err
	}
	if artifact != nil {
		if err := artifact.ApplyProjection(name, input, output); err != nil {
			return nil, trainingLinearTape{}, err
		}
	}
	return output, trainingLinearTape{name: name, input: input}, nil
}

func (m *Model) trainingAttention(q []float32, steps []*trainingLayerTape) ([]float32, [][]float32) {
	headDim, heads, kvHeads := m.Config.HeadDim, m.Config.NumAttentionHeads, m.Config.NumKeyValueHeads
	output := make([]float32, len(q))
	weights := make([][]float32, heads)
	for head := 0; head < heads; head++ {
		kvHead := head / (heads / kvHeads)
		scores := make([]float32, len(steps))
		max := float32(-math.MaxFloat32)
		for step, source := range steps {
			keyOffset := kvHead * headDim
			for index := 0; index < headDim; index++ {
				scores[step] += q[head*headDim+index] * source.k[keyOffset+index]
			}
			scores[step] /= float32(math.Sqrt(float64(headDim)))
			if scores[step] > max {
				max = scores[step]
			}
		}
		var total float32
		for index := range scores {
			scores[index] = float32(math.Exp(float64(scores[index] - max)))
			total += scores[index]
		}
		for step, source := range steps {
			scores[step] /= total
			valueOffset := kvHead * headDim
			for index := 0; index < headDim; index++ {
				output[head*headDim+index] += scores[step] * source.v[valueOffset+index]
			}
		}
		weights[head] = scores
	}
	return output, weights
}

func (m *Model) backwardTrainingSequence(sequence *trainingSequence, artifact *adapter.TransformerLoRA, gradients map[string]adapter.ProjectionGradient) error {
	finalNorm, err := m.vector("model.norm.weight")
	if err != nil {
		return err
	}
	for position := len(sequence.finalInput) - 1; position >= 0; position-- {
		gradient := rmsNormBackward(sequence.finalInput[position], finalNorm, sequence.finalGradient[position], m.Config.RMSNormEps)
		addTraining(sequence.layers[m.Config.NumHiddenLayers-1][position].dOutput, gradient)
	}
	for layer := m.Config.NumHiddenLayers - 1; layer >= 0; layer-- {
		prefix := fmt.Sprintf("model.layers.%d.", layer)
		for position := len(sequence.layers[layer]) - 1; position >= 0; position-- {
			inputGradient, err := m.backwardTrainingLayer(prefix, position, sequence.layers[layer], artifact, gradients)
			if err != nil {
				return err
			}
			if layer > 0 {
				addTraining(sequence.layers[layer-1][position].dOutput, inputGradient)
			}
		}
	}
	return nil
}

func (m *Model) backwardTrainingLayer(prefix string, position int, layers []*trainingLayerTape, artifact *adapter.TransformerLoRA, gradients map[string]adapter.ProjectionGradient) ([]float32, error) {
	tape := layers[position]
	dResidual := append([]float32(nil), tape.dOutput...)
	dActivation, err := m.backwardTrainingLinear(tape.downLinear, tape.dOutput, artifact, gradients)
	if err != nil {
		return nil, err
	}
	dGate, dUp := make([]float32, len(dActivation)), make([]float32, len(dActivation))
	for index, value := range dActivation {
		dGate[index] = value * tape.upRaw[index] * siluDerivative(tape.gateRaw[index])
		dUp[index] = value * silu(tape.gateRaw[index])
	}
	dNorm2, err := m.backwardTrainingLinear(tape.gateLinear, dGate, artifact, gradients)
	if err != nil {
		return nil, err
	}
	gradient, err := m.backwardTrainingLinear(tape.upLinear, dUp, artifact, gradients)
	if err != nil {
		return nil, err
	}
	addTraining(dNorm2, gradient)
	postNorm, err := m.vector(prefix + "post_attention_layernorm.weight")
	if err != nil {
		return nil, err
	}
	addTraining(dResidual, rmsNormBackward(tape.residual, postNorm, dNorm2, m.Config.RMSNormEps))
	dInput := append([]float32(nil), dResidual...)
	dAttention, err := m.backwardTrainingLinear(tape.oLinear, dResidual, artifact, gradients)
	if err != nil {
		return nil, err
	}
	m.backwardTrainingAttention(position, layers, dAttention)
	inverseRoPEGradient(tape.dQ, position, m.Config.HeadDim, m.Config.RopeTheta)
	inverseRoPEGradient(tape.dK, position, m.Config.HeadDim, m.Config.RopeTheta)
	qNorm, err := m.vector(prefix + "self_attn.q_norm.weight")
	if err != nil {
		return nil, err
	}
	kNorm, err := m.vector(prefix + "self_attn.k_norm.weight")
	if err != nil {
		return nil, err
	}
	dQRaw := rmsHeadsBackward(tape.qRaw, qNorm, tape.dQ, m.Config.RMSNormEps)
	dKRaw := rmsHeadsBackward(tape.kRaw, kNorm, tape.dK, m.Config.RMSNormEps)
	dNorm1, err := m.backwardTrainingLinear(tape.qLinear, dQRaw, artifact, gradients)
	if err != nil {
		return nil, err
	}
	gradient, err = m.backwardTrainingLinear(tape.kLinear, dKRaw, artifact, gradients)
	if err != nil {
		return nil, err
	}
	addTraining(dNorm1, gradient)
	gradient, err = m.backwardTrainingLinear(tape.vLinear, tape.dV, artifact, gradients)
	if err != nil {
		return nil, err
	}
	addTraining(dNorm1, gradient)
	inputNorm, err := m.vector(prefix + "input_layernorm.weight")
	if err != nil {
		return nil, err
	}
	addTraining(dInput, rmsNormBackward(tape.input, inputNorm, dNorm1, m.Config.RMSNormEps))
	return dInput, nil
}

func (m *Model) backwardTrainingAttention(position int, layers []*trainingLayerTape, dAttention []float32) {
	tape := layers[position]
	headDim, heads, kvHeads := m.Config.HeadDim, m.Config.NumAttentionHeads, m.Config.NumKeyValueHeads
	scale := float32(1 / math.Sqrt(float64(headDim)))
	for head := 0; head < heads; head++ {
		kvHead := head / (heads / kvHeads)
		dWeights := make([]float32, position+1)
		for step := 0; step <= position; step++ {
			valueOffset := kvHead * headDim
			for index := 0; index < headDim; index++ {
				dWeights[step] += dAttention[head*headDim+index] * layers[step].v[valueOffset+index]
				layers[step].dV[valueOffset+index] += tape.weights[head][step] * dAttention[head*headDim+index]
			}
		}
		var weighted float32
		for step, value := range dWeights {
			weighted += tape.weights[head][step] * value
		}
		for step, value := range dWeights {
			dScore := tape.weights[head][step] * (value - weighted)
			keyOffset := kvHead * headDim
			for index := 0; index < headDim; index++ {
				tape.dQ[head*headDim+index] += dScore * layers[step].k[keyOffset+index] * scale
				layers[step].dK[keyOffset+index] += dScore * tape.q[head*headDim+index] * scale
			}
		}
	}
}

func (m *Model) backwardTrainingLinear(tape trainingLinearTape, outputGradient []float32, artifact *adapter.TransformerLoRA, gradients map[string]adapter.ProjectionGradient) ([]float32, error) {
	inputGradient, err := m.matrixVecTranspose(tape.name, outputGradient)
	if err != nil {
		return nil, err
	}
	if artifact == nil {
		return inputGradient, nil
	}
	target, ok := artifact.Targets[tape.name]
	if !ok {
		return inputGradient, nil
	}
	gradient := gradients[tape.name]
	projected := make([]float32, target.Rank)
	for rank := 0; rank < target.Rank; rank++ {
		for column, value := range tape.input {
			projected[rank] += target.A[rank*target.Cols+column] * value
		}
	}
	scale := target.Alpha / float32(target.Rank)
	dProjected := make([]float32, target.Rank)
	for row, value := range outputGradient {
		for rank := 0; rank < target.Rank; rank++ {
			dProjected[rank] += scale * target.B[row*target.Rank+rank] * value
			gradient.B[row*target.Rank+rank] += scale * value * projected[rank]
		}
	}
	for rank, value := range dProjected {
		for column, input := range tape.input {
			gradient.A[rank*target.Cols+column] += value * input
			inputGradient[column] += target.A[rank*target.Cols+column] * value
		}
	}
	gradients[tape.name] = gradient
	return inputGradient, nil
}

func (m *Model) matrixVecTranspose(name string, outputGradient []float32) ([]float32, error) {
	if m.executor != nil {
		return m.executor.MatrixVecTranspose(m, name, outputGradient)
	}
	if m.useNF4 {
		quantized, err := m.nf4(name)
		if err != nil {
			return nil, err
		}
		if len(quantized.shape) != 2 || len(outputGradient) != quantized.shape[0] {
			return nil, fmt.Errorf("matrix %s transpose shape cannot multiply %d values", name, len(outputGradient))
		}
		return transposeNF4Parallel(quantized, outputGradient), nil
	}
	w, err := m.tensor(name)
	if err != nil {
		return nil, err
	}
	if len(w.shape) != 2 || len(outputGradient) != w.shape[0] {
		return nil, fmt.Errorf("matrix %s transpose shape cannot multiply %d values", name, len(outputGradient))
	}
	return transposeBF16Parallel(w, outputGradient), nil
}

func transposeNF4Parallel(w *nf4Weight, outputGradient []float32) []float32 {
	return transposeParallel(w.shape[0], w.shape[1], outputGradient, func(row, column int) float32 {
		return w.value(row*w.shape[1] + column)
	})
}

func transposeBF16Parallel(w weight, outputGradient []float32) []float32 {
	return transposeParallel(w.shape[0], w.shape[1], outputGradient, func(row, column int) float32 {
		return bfloat(w.data, row*w.shape[1]+column)
	})
}

func transposeParallel(rows, columns int, outputGradient []float32, value func(row, column int) float32) []float32 {
	// A transpose product reduces over rows, so each worker accumulates a local
	// vector and the caller combines it without locks.
	if rows < 1024 || runtime.GOMAXPROCS(0) < 2 {
		result := make([]float32, columns)
		for row, gradient := range outputGradient {
			for column := range result {
				result[column] += value(row, column) * gradient
			}
		}
		return result
	}
	workers := runtime.GOMAXPROCS(0)
	if workers > rows {
		workers = rows
	}
	partials := make([][]float32, workers)
	chunk := (rows + workers - 1) / workers
	var group sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		start, end := worker*chunk, (worker+1)*chunk
		if start >= rows {
			break
		}
		if end > rows {
			end = rows
		}
		partials[worker] = make([]float32, columns)
		group.Add(1)
		go func(partial []float32, start, end int) {
			defer group.Done()
			for row := start; row < end; row++ {
				for column := range partial {
					partial[column] += value(row, column) * outputGradient[row]
				}
			}
		}(partials[worker], start, end)
	}
	group.Wait()
	result := make([]float32, columns)
	for _, partial := range partials {
		for column, value := range partial {
			result[column] += value
		}
	}
	return result
}

func rmsNormBackward(input, weight, outputGradient []float32, eps float32) []float32 {
	var squareSum, weightedDot float32
	for index, value := range input {
		squareSum += value * value
		weightedDot += outputGradient[index] * weight[index] * value
	}
	scale := float32(1 / math.Sqrt(float64(squareSum/float32(len(input))+eps)))
	result := make([]float32, len(input))
	coefficient := scale * scale * scale / float32(len(input))
	for index, value := range input {
		result[index] = outputGradient[index]*weight[index]*scale - value*weightedDot*coefficient
	}
	return result
}

func rmsHeadsBackward(input, weight, outputGradient []float32, eps float32) []float32 {
	result := make([]float32, len(input))
	for start := 0; start < len(input); start += len(weight) {
		part := rmsNormBackward(input[start:start+len(weight)], weight, outputGradient[start:start+len(weight)], eps)
		copy(result[start:], part)
	}
	return result
}

func inverseRoPEGradient(gradient []float32, position, headDim int, theta float64) {
	for head := 0; head < len(gradient)/headDim; head++ {
		half := headDim / 2
		for index := 0; index < half; index++ {
			angle := float64(position) / math.Pow(theta, float64(2*index)/float64(headDim))
			cos, sin := float32(math.Cos(angle)), float32(math.Sin(angle))
			left, right := head*headDim+index, head*headDim+half+index
			a, b := gradient[left], gradient[right]
			gradient[left], gradient[right] = a*cos+b*sin, -a*sin+b*cos
		}
	}
}

func trainingSoftmax(values []float32) []float32 {
	result := make([]float32, len(values))
	if len(values) == 0 {
		return result
	}
	max := values[0]
	for _, value := range values[1:] {
		if value > max {
			max = value
		}
	}
	var total float64
	for index, value := range values {
		result[index] = float32(math.Exp(float64(value - max)))
		total += float64(result[index])
	}
	for index := range result {
		result[index] /= float32(total)
	}
	return result
}

func siluDerivative(value float32) float32 {
	sigmoid := float32(1 / (1 + math.Exp(float64(-value))))
	return sigmoid + value*sigmoid*(1-sigmoid)
}

func addTraining(destination, source []float32) {
	for index := range destination {
		destination[index] += source[index]
	}
}
