package finetune

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/sypra-labs/sypra-ai/internal/adapter"
	"github.com/sypra-labs/sypra-ai/internal/qwen"
)

// TrainQwenLMHeadLoRA performs dependency-free supervised LoRA training over
// Qwen's frozen final hidden states. Loss is exact response-only causal
// cross-entropy over the full output vocabulary; prompt tokens advance the
// frozen decoder state but never contribute to the loss.
func TrainQwenLMHeadLoRA(ctx context.Context, datasetPath, outputDir string, cfg Config, report func(Progress)) (Result, error) {
	if err := cfg.Validate(); err != nil {
		return Result{}, err
	}
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return Result{}, err
	}
	model, err := qwen.LoadModel(cfg.ModelDir)
	if err != nil {
		return Result{}, fmt.Errorf("load base Qwen model: %w", err)
	}
	defer model.Close()
	if cfg.Method == "qlora" {
		model.EnableNF4(cfg.DoubleQuant)
	}
	examples, err := ReadDataset(datasetPath)
	if err != nil {
		return Result{}, err
	}
	train, validation, test := SplitDataset3(examples, cfg.Seed)
	artifact := newQwenArtifact(cfg, model)
	startEpoch := 0
	bestEpoch := 0
	bestLoss := math.Inf(1)
	if cfg.ResumeFrom != "" {
		artifact, startEpoch, bestEpoch, bestLoss, err = resumeQwenTraining(cfg.ResumeFrom, model, cfg)
		if err != nil {
			return Result{}, err
		}
	}
	bestArtifact := cloneQwenArtifact(artifact)

	var trainLoss, evalLoss, testLoss float64
	withoutImprovement := 0
	for epoch := startEpoch; epoch < cfg.Epochs; epoch++ {
		trainLoss = 0
		trained := 0
		for exampleIndex, example := range train {
			if err := ctx.Err(); err != nil {
				return Result{}, err
			}
			loss, tokens, err := trainQwenExample(ctx, model, &artifact, example, cfg.LearningRate, cfg.Seed+uint64(epoch*len(train)+exampleIndex+1))
			if err != nil {
				return Result{}, err
			}
			trainLoss += loss
			trained += tokens
		}
		if trained == 0 {
			return Result{}, fmt.Errorf("dataset produced no trainable Qwen response tokens")
		}
		trainLoss /= float64(trained)
		evalLoss, err = evaluateQwen(ctx, model, artifact, validation)
		if err != nil {
			return Result{}, err
		}
		improved := evalLoss < bestLoss-cfg.MinValidationImprovement
		if improved {
			bestArtifact, bestLoss, bestEpoch = cloneQwenArtifact(artifact), evalLoss, epoch+1
			withoutImprovement = 0
			if err := writeQwenArtifact(filepath.Join(outputDir, "best", "adapter.json"), bestArtifact); err != nil {
				return Result{}, err
			}
		} else {
			withoutImprovement++
		}
		state := qwenTrainerState{Version: 1, CompletedEpoch: epoch + 1, BestEpoch: bestEpoch, BestValidationLoss: bestLoss, UpdatedAt: time.Now().UTC()}
		if err := writeQwenTrainerState(filepath.Join(outputDir, "trainer_state.json"), state); err != nil {
			return Result{}, err
		}
		if (epoch+1)%cfg.CheckpointEvery == 0 {
			checkpoint := filepath.Join(outputDir, "checkpoints", fmt.Sprintf("epoch-%04d", epoch+1))
			if err := writeQwenArtifact(filepath.Join(checkpoint, "adapter.json"), artifact); err != nil {
				return Result{}, err
			}
			if err := writeQwenTrainerState(filepath.Join(checkpoint, "trainer_state.json"), state); err != nil {
				return Result{}, err
			}
		}
		if report != nil {
			report(Progress{Progress: float64(epoch+1) / float64(cfg.Epochs), Phase: "training", Epoch: epoch + 1, Epochs: cfg.Epochs, Step: epoch + 1, Steps: cfg.Epochs, TrainLoss: trainLoss, EvalLoss: evalLoss, ValidationComplete: true})
		}
		if cfg.EarlyStoppingPatience > 0 && withoutImprovement >= cfg.EarlyStoppingPatience {
			break
		}
	}
	if bestEpoch == 0 {
		return Result{}, fmt.Errorf("training completed without a validation checkpoint")
	}
	testLoss, err = evaluateQwen(ctx, model, bestArtifact, test)
	if err != nil {
		return Result{}, err
	}
	path := filepath.Join(outputDir, "best", "adapter.json")
	if err := writeQwenArtifact(path, bestArtifact); err != nil {
		return Result{}, err
	}
	if err := adapter.ExportPEFT(filepath.Join(outputDir, "best", "peft"), bestArtifact); err != nil {
		return Result{}, fmt.Errorf("export PEFT adapter: %w", err)
	}
	if err := writeEvaluation(filepath.Join(outputDir, "evaluation.json"), DatasetSplitReport(examples, cfg.Seed), trainLoss, bestLoss, testLoss); err != nil {
		return Result{}, err
	}
	return Result{ArtifactPath: path, ArtifactFormat: adapter.QwenLMHeadLoRAType, TrainLoss: trainLoss, EvalLoss: bestLoss, TestLoss: testLoss}, nil
}

type qwenTrainerState struct {
	Version            int       `json:"version"`
	CompletedEpoch     int       `json:"completedEpoch"`
	BestEpoch          int       `json:"bestEpoch"`
	BestValidationLoss float64   `json:"bestValidationLoss"`
	UpdatedAt          time.Time `json:"updatedAt"`
}

func cloneQwenArtifact(artifact adapter.QwenLMHeadLoRA) adapter.QwenLMHeadLoRA {
	artifact.A = append([]float32(nil), artifact.A...)
	artifact.B = append([]float32(nil), artifact.B...)
	return artifact
}

func resumeQwenTraining(path string, model *qwen.Model, cfg Config) (adapter.QwenLMHeadLoRA, int, int, float64, error) {
	artifact, err := adapter.LoadQwenLMHeadLoRA(path)
	if err != nil {
		return adapter.QwenLMHeadLoRA{}, 0, 0, 0, fmt.Errorf("load resume adapter: %w", err)
	}
	if artifact.BaseModel != cfg.ModelDir || artifact.HiddenSize != model.Config.HiddenSize || artifact.VocabSize != model.Config.VocabSize || artifact.Rank != cfg.LoRARank || artifact.Alpha != float32(cfg.LoRAAlpha) {
		return adapter.QwenLMHeadLoRA{}, 0, 0, 0, fmt.Errorf("resume adapter is incompatible with this model or LoRA configuration")
	}
	directory := path
	if info, statErr := os.Stat(path); statErr == nil && !info.IsDir() {
		directory = filepath.Dir(path)
	}
	state, err := readQwenTrainerState(filepath.Join(directory, "trainer_state.json"))
	if err != nil {
		return adapter.QwenLMHeadLoRA{}, 0, 0, 0, fmt.Errorf("load resume state: %w", err)
	}
	return artifact, state.CompletedEpoch, state.BestEpoch, state.BestValidationLoss, nil
}

func writeQwenTrainerState(path string, state qwenTrainerState) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0600)
}

func readQwenTrainerState(path string) (qwenTrainerState, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return qwenTrainerState{}, err
	}
	var state qwenTrainerState
	if err := json.Unmarshal(data, &state); err != nil {
		return qwenTrainerState{}, err
	}
	if state.Version != 1 || state.CompletedEpoch < 1 || state.BestEpoch < 1 || math.IsInf(state.BestValidationLoss, 0) || math.IsNaN(state.BestValidationLoss) {
		return qwenTrainerState{}, fmt.Errorf("invalid trainer state")
	}
	return state, nil
}

func newQwenArtifact(cfg Config, model *qwen.Model) adapter.QwenLMHeadLoRA {
	rank := cfg.LoRARank
	artifact := adapter.QwenLMHeadLoRA{
		Version: 1, Type: adapter.QwenLMHeadLoRAType, BaseModel: cfg.ModelDir, Target: "lm_head",
		Rank: rank, Alpha: float32(cfg.LoRAAlpha), HiddenSize: model.Config.HiddenSize,
		VocabSize: model.Config.VocabSize, CreatedAt: time.Now().UTC(),
		A: make([]float32, rank*model.Config.HiddenSize), B: make([]float32, model.Config.VocabSize*rank),
	}
	for i := range artifact.A {
		artifact.A[i] = float32(math.Sin(float64(i+1)*.61803398875)) * .01
	}
	return artifact
}

func trainQwenExample(ctx context.Context, model *qwen.Model, artifact *adapter.QwenLMHeadLoRA, example Example, learningRate float64, seed uint64) (float64, int, error) {
	prompt, err := model.Tokenizer.Encode(example.Instruction + "\n")
	if err != nil {
		return 0, 0, err
	}
	response, err := model.Tokenizer.Encode(example.Response)
	if err != nil {
		return 0, 0, err
	}
	if len(prompt) == 0 || len(response) == 0 {
		return 0, 0, nil
	}
	state := qwen.NewState(model.Config.NumHiddenLayers)
	position := 0
	var hidden []float32
	for _, token := range prompt {
		if err := ctx.Err(); err != nil {
			return 0, 0, err
		}
		hidden, err = model.ForwardHidden(token, position, state)
		if err != nil {
			return 0, 0, err
		}
		position++
	}
	var loss float64
	for tokenIndex, target := range response {
		if err := ctx.Err(); err != nil {
			return 0, tokenIndex, err
		}
		stepLoss, err := updateQwenLoRA(model, artifact, hidden, target, learningRate, seed+uint64(tokenIndex))
		if err != nil {
			return 0, tokenIndex, err
		}
		loss += stepLoss
		hidden, err = model.ForwardHidden(target, position, state)
		if err != nil {
			return 0, tokenIndex, err
		}
		position++
	}
	return loss, len(response), nil
}

func updateQwenLoRA(model *qwen.Model, artifact *adapter.QwenLMHeadLoRA, hidden []float32, target int, learningRate float64, seed uint64) (float64, error) {
	_ = seed // retained in the function signature for deterministic call sites.
	logits, err := model.LMHeadLogits(hidden)
	if err != nil {
		return 0, err
	}
	if err := artifact.Apply(hidden, logits); err != nil {
		return 0, err
	}
	probabilities := softmax32(logits)
	gradientProjected := make([]float32, artifact.Rank)
	for token, probability := range probabilities {
		gradient := probability
		if token == target {
			gradient--
		}
		row := token * artifact.Rank
		scale := artifact.Alpha / float32(artifact.Rank)
		for rank := 0; rank < artifact.Rank; rank++ {
			gradientProjected[rank] += gradient * scale * artifact.B[row+rank]
		}
	}
	projected := projectLoRA(artifact.A, artifact.Rank, artifact.HiddenSize, hidden)
	for token, probability := range probabilities {
		gradient := probability
		if token == target {
			gradient--
		}
		row := token * artifact.Rank
		scale := artifact.Alpha / float32(artifact.Rank)
		for rank, value := range projected {
			artifact.B[row+rank] -= float32(learningRate) * gradient * scale * value
		}
	}
	for rank, gradient := range gradientProjected {
		row := rank * artifact.HiddenSize
		for feature, value := range hidden {
			artifact.A[row+feature] -= float32(learningRate) * gradient * value
		}
	}
	return -math.Log(math.Max(float64(probabilities[target]), 1e-12)), nil
}

func evaluateQwen(ctx context.Context, model *qwen.Model, artifact adapter.QwenLMHeadLoRA, examples []Example) (float64, error) {
	var total float64
	count := 0
	for exampleIndex, example := range examples {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		loss, tokens, err := scoreQwenExample(ctx, model, artifact, example, uint64(exampleIndex+1))
		if err != nil {
			return 0, err
		}
		total += loss
		count += tokens
	}
	if count == 0 {
		return 0, fmt.Errorf("validation split produced no response tokens")
	}
	return total / float64(count), nil
}

func scoreQwenExample(ctx context.Context, model *qwen.Model, artifact adapter.QwenLMHeadLoRA, example Example, seed uint64) (float64, int, error) {
	prompt, err := model.Tokenizer.Encode(example.Instruction + "\n")
	if err != nil {
		return 0, 0, err
	}
	response, err := model.Tokenizer.Encode(example.Response)
	if err != nil || len(prompt) == 0 || len(response) == 0 {
		return 0, 0, err
	}
	state := qwen.NewState(model.Config.NumHiddenLayers)
	position := 0
	var hidden []float32
	for _, token := range prompt {
		hidden, err = model.ForwardHidden(token, position, state)
		if err != nil {
			return 0, 0, err
		}
		position++
	}
	var total float64
	for index, target := range response {
		if err := ctx.Err(); err != nil {
			return 0, index, err
		}
		_ = seed
		logits, err := model.LMHeadLogits(hidden)
		if err != nil {
			return 0, index, err
		}
		if err = artifact.Apply(hidden, logits); err != nil {
			return 0, index, err
		}
		probabilities := softmax32(logits)
		total -= math.Log(math.Max(float64(probabilities[target]), 1e-12))
		hidden, err = model.ForwardHidden(target, position, state)
		if err != nil {
			return 0, index, err
		}
		position++
	}
	return total, len(response), nil
}

func projectLoRA(values []float32, rank, hiddenSize int, hidden []float32) []float32 {
	result := make([]float32, rank)
	for row := range result {
		for feature, value := range hidden {
			result[row] += values[row*hiddenSize+feature] * value
		}
	}
	return result
}

func softmax32(values []float32) []float32 {
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
	for i, value := range values {
		result[i] = float32(math.Exp(float64(value - max)))
		total += float64(result[i])
	}
	for i := range result {
		result[i] /= float32(total)
	}
	return result
}

func writeQwenArtifact(path string, artifact adapter.QwenLMHeadLoRA) error {
	data, err := json.MarshalIndent(artifact, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0600)
}

func writeEvaluation(path string, split SplitReport, trainLoss, validationLoss, testLoss float64) error {
	data, err := json.MarshalIndent(struct {
		Split          SplitReport `json:"split"`
		TrainLoss      float64     `json:"trainLoss"`
		ValidationLoss float64     `json:"validationLoss"`
		TestLoss       float64     `json:"testLoss"`
	}{split, trainLoss, validationLoss, testLoss}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0600)
}
