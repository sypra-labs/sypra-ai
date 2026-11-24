package finetune

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sypra-labs/sypra-ai/internal/adapter"
	"github.com/sypra-labs/sypra-ai/internal/chattemplate"
	"github.com/sypra-labs/sypra-ai/internal/domain"
	"github.com/sypra-labs/sypra-ai/internal/qwen"
)

// TrainQwenTransformerLoRA is the native full-transformer LoRA/QLoRA path.
// It computes response-only causal gradients through all selected Qwen decoder
// projections and updates only LoRA A/B matrices with AdamW.
func TrainQwenTransformerLoRA(ctx context.Context, datasetPath, outputDir string, cfg Config, report func(Progress)) (Result, error) {
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
	distributed := len(cfg.GPUDevices) > 0 || len(cfg.WorkerEndpoints) > 0
	if !distributed && cfg.ComputeBackend != "cpu" {
		executor, cudaErr := qwen.NewCUDAExecutor()
		if cudaErr != nil {
			if cfg.ComputeBackend == "cuda" {
				return Result{}, fmt.Errorf("initialize CUDA backend: %w", cudaErr)
			}
		} else {
			model.SetMatrixExecutor(executor)
		}
	}
	if cfg.Method == "qlora" {
		model.EnableNF4(cfg.DoubleQuant)
	}
	examples, err := ReadDataset(datasetPath)
	if err != nil {
		return Result{}, err
	}
	train, validation, test := SplitDataset3(examples, cfg.Seed)
	artifact, err := newQwenTransformerArtifact(cfg, model)
	if err != nil {
		return Result{}, err
	}
	workers, err := newDistributedWorkers(cfg)
	if err != nil {
		return Result{}, err
	}
	defer closeGradientWorkers(workers)
	startEpoch, bestEpoch, bestLoss := 0, 0, math.Inf(1)
	var resumeOptimizer AdamWState
	if cfg.ResumeFrom != "" {
		artifact, startEpoch, bestEpoch, bestLoss, resumeOptimizer, err = resumeTransformerTraining(cfg.ResumeFrom, model, cfg)
		if err != nil {
			return Result{}, err
		}
	}
	optimizer := NewAdamW(cfg, &artifact)
	if cfg.ResumeFrom != "" {
		if err := optimizer.Restore(resumeOptimizer, &artifact); err != nil {
			return Result{}, fmt.Errorf("restore optimizer state: %w", err)
		}
	}
	bestArtifact := cloneTransformerArtifact(artifact)
	bestOptimizer := optimizer.Snapshot()
	var trainLoss, validationLoss, testLoss float64
	withoutImprovement := 0
	effectiveBatch := cfg.BatchSize * cfg.GradientAccumulation
	batchesPerEpoch := (len(train) + effectiveBatch - 1) / effectiveBatch
	totalSteps := batchesPerEpoch * cfg.Epochs
	completedTokens := 0
	startedAt := time.Now()
	for epoch := startEpoch; epoch < cfg.Epochs; epoch++ {
		trainLoss = 0
		epochTokens := 0
		for start, batchStep := 0, 0; start < len(train); start, batchStep = start+effectiveBatch, batchStep+1 {
			if err := ctx.Err(); err != nil {
				return Result{}, err
			}
			end := start + effectiveBatch
			if end > len(train) {
				end = len(train)
			}
			batch := train[start:end]
			var gradients map[string]adapter.ProjectionGradient
			var loss float64
			var tokenCount int
			if len(workers) > 0 {
				result, workerErr := allReduceGradientBatch(ctx, workers, artifact, batch, cfg.MaxSequenceLength)
				if workerErr != nil {
					return Result{}, workerErr
				}
				gradients, loss, tokenCount = result.Gradients, result.Loss, result.Tokens
			} else {
				localResult, localErr := localGradientBatch(ctx, model, artifact, batch, cfg.MaxSequenceLength)
				if localErr != nil {
					return Result{}, localErr
				}
				gradients, loss, tokenCount = localResult.Gradients, localResult.Loss, localResult.Tokens
			}
			if err := optimizer.Step(&artifact, gradients, cfg.LearningRate, 1/float64(tokenCount)); err != nil {
				return Result{}, err
			}
			trainLoss += loss
			epochTokens += tokenCount
			completedTokens += tokenCount
			if report != nil && epochTokens > 0 {
				step := epoch*batchesPerEpoch + batchStep + 1
				report(Progress{Progress: float64(step) / float64(totalSteps), Phase: "training", Epoch: epoch + 1, Epochs: cfg.Epochs, Step: step, Steps: totalSteps, Tokens: completedTokens, TokensPerSecond: float64(completedTokens) / maxDurationSeconds(time.Since(startedAt)), TrainLoss: trainLoss / float64(epochTokens)})
			}
		}
		trainTokens := responseTokenCount(model, train, cfg.MaxSequenceLength)
		if trainTokens == 0 {
			return Result{}, fmt.Errorf("dataset produced no trainable Qwen response tokens")
		}
		trainLoss /= float64(trainTokens)
		if report != nil {
			report(Progress{Progress: float64((epoch+1)*batchesPerEpoch) / float64(totalSteps), Phase: "validating", Epoch: epoch + 1, Epochs: cfg.Epochs, Step: (epoch + 1) * batchesPerEpoch, Steps: totalSteps, Tokens: completedTokens, TokensPerSecond: float64(completedTokens) / maxDurationSeconds(time.Since(startedAt)), TrainLoss: trainLoss})
		}
		validationLoss, err = evaluateTransformer(ctx, model, artifact, validation, cfg.MaxSequenceLength)
		if err != nil {
			return Result{}, err
		}
		improved := validationLoss < bestLoss-cfg.MinValidationImprovement
		if improved {
			bestArtifact, bestLoss, bestEpoch = cloneTransformerArtifact(artifact), validationLoss, epoch+1
			bestOptimizer = optimizer.Snapshot()
			withoutImprovement = 0
			if err := saveTransformerCheckpoint(filepath.Join(outputDir, "best"), bestArtifact, qwenTrainerState{Version: 1, CompletedEpoch: epoch + 1, BestEpoch: bestEpoch, BestValidationLoss: bestLoss, UpdatedAt: time.Now().UTC()}, optimizer); err != nil {
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
			if err := saveTransformerCheckpoint(filepath.Join(outputDir, "checkpoints", fmt.Sprintf("epoch-%04d", epoch+1)), artifact, state, optimizer); err != nil {
				return Result{}, err
			}
		}
		if report != nil {
			report(Progress{Progress: float64((epoch+1)*batchesPerEpoch) / float64(totalSteps), Phase: "training", Epoch: epoch + 1, Epochs: cfg.Epochs, Step: (epoch + 1) * batchesPerEpoch, Steps: totalSteps, Tokens: completedTokens, TokensPerSecond: float64(completedTokens) / maxDurationSeconds(time.Since(startedAt)), TrainLoss: trainLoss, EvalLoss: validationLoss, ValidationComplete: true})
		}
		if cfg.EarlyStoppingPatience > 0 && withoutImprovement >= cfg.EarlyStoppingPatience {
			break
		}
	}
	if bestEpoch == 0 {
		return Result{}, fmt.Errorf("training completed without a validation checkpoint")
	}
	testLoss, err = evaluateTransformer(ctx, model, bestArtifact, test, cfg.MaxSequenceLength)
	if err != nil {
		return Result{}, err
	}
	bestDir := filepath.Join(outputDir, "best")
	bestOptimizerRuntime := NewAdamW(cfg, &bestArtifact)
	if err := bestOptimizerRuntime.Restore(bestOptimizer, &bestArtifact); err != nil {
		return Result{}, fmt.Errorf("restore best optimizer state: %w", err)
	}
	if err := saveTransformerCheckpoint(bestDir, bestArtifact, qwenTrainerState{Version: 1, CompletedEpoch: bestEpoch, BestEpoch: bestEpoch, BestValidationLoss: bestLoss, UpdatedAt: time.Now().UTC()}, bestOptimizerRuntime); err != nil {
		return Result{}, err
	}
	if err := adapter.ExportTransformerPEFT(filepath.Join(bestDir, "peft"), bestArtifact); err != nil {
		return Result{}, fmt.Errorf("export PEFT adapter: %w", err)
	}
	if err := writeEvaluation(filepath.Join(outputDir, "evaluation.json"), DatasetSplitReport(examples, cfg.Seed), trainLoss, bestLoss, testLoss); err != nil {
		return Result{}, err
	}
	return Result{ArtifactPath: filepath.Join(bestDir, "adapter.json"), ArtifactFormat: adapter.TransformerLoRAType, TrainLoss: trainLoss, EvalLoss: bestLoss, TestLoss: testLoss}, nil
}

func maxDurationSeconds(value time.Duration) float64 {
	if value <= 0 {
		return 1
	}
	return value.Seconds()
}

func newQwenTransformerArtifact(cfg Config, model *qwen.Model) (adapter.TransformerLoRA, error) {
	targets := map[string]adapter.ProjectionLoRA{}
	for targetIndex, name := range qwenTargetNames(model.Config.NumHiddenLayers) {
		if !qwenTargetSelected(name, cfg.TargetModules) {
			continue
		}
		rows, cols, err := model.ProjectionShape(name)
		if err != nil {
			return adapter.TransformerLoRA{}, err
		}
		target := adapter.ProjectionLoRA{Rows: rows, Cols: cols, Rank: cfg.LoRARank, Alpha: float32(cfg.LoRAAlpha), A: make([]float32, cfg.LoRARank*cols), B: make([]float32, rows*cfg.LoRARank)}
		for index := range target.A {
			target.A[index] = float32(math.Sin(float64(index+1+targetIndex*7919)*.61803398875)) * .01
		}
		targets[name] = target
	}
	artifact := adapter.TransformerLoRA{Version: adapter.QwenAdapterArtifactV1, Type: adapter.TransformerLoRAType, BaseModel: cfg.ModelDir, CreatedAt: time.Now().UTC(), Targets: targets}
	if err := artifact.Validate(); err != nil {
		return adapter.TransformerLoRA{}, err
	}
	return artifact, nil
}

func qwenTargetNames(layers int) []string {
	targets := make([]string, 0, layers*7+1)
	for layer := 0; layer < layers; layer++ {
		prefix := fmt.Sprintf("model.layers.%d.", layer)
		for _, module := range []string{"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"} {
			targets = append(targets, prefix+module)
		}
	}
	return append(targets, "lm_head.weight")
}

func qwenTargetSelected(name string, configured []string) bool {
	if len(configured) == 0 {
		return true
	}
	for _, candidate := range configured {
		candidate = strings.TrimSpace(candidate)
		if candidate == "all" || candidate == name || strings.HasSuffix(name, "."+strings.TrimSuffix(candidate, ".weight")+".weight") {
			return true
		}
	}
	return false
}

func encodeQwenExample(model *qwen.Model, example Example, maximum int) ([]int, int, error) {
	messages := make([]chattemplate.Message, 0, len(example.Messages)+2)
	if len(example.Messages) == 0 {
		messages = append(messages, chattemplate.Message{Role: "user", Content: example.Instruction}, chattemplate.Message{Role: "assistant", Content: example.Response})
	} else {
		for _, message := range example.Messages {
			if domain.HasNonTextContent(message.Parts) {
				return nil, 0, fmt.Errorf("example %s contains non-text data; the native Qwen text trainer requires a multimodal model endpoint", example.ID)
			}
			messages = append(messages, chattemplate.Message{Role: message.Role, Content: message.Content, Parts: message.Parts})
		}
	}
	rendered, err := chattemplate.Render(chattemplate.FamilyQwen, messages, true)
	if err != nil {
		return nil, 0, fmt.Errorf("render chat template: %w", err)
	}
	tokens, mask, err := chattemplate.ResponseTokenMask(rendered.Segments, func(segment chattemplate.Segment) ([]int, error) {
		if segment.Control != "" {
			id, found := model.Tokenizer.EncodeSpecial(segment.Control)
			if !found {
				return nil, fmt.Errorf("Qwen tokenizer has no control token %q", segment.Control)
			}
			return []int{id}, nil
		}
		return model.Tokenizer.Encode(segment.Text)
	})
	if err != nil {
		return nil, 0, err
	}
	labelStart := len(mask)
	for index, value := range mask {
		if value {
			labelStart = index
			break
		}
	}
	if labelStart == len(mask) || labelStart == 0 || len(tokens) <= labelStart {
		return nil, 0, fmt.Errorf("example %s tokenized to an empty prompt or response", example.ID)
	}
	if maximum > 0 && len(tokens) > maximum {
		return nil, 0, fmt.Errorf("example %s has %d tokens, exceeding configured max sequence length %d", example.ID, len(tokens), maximum)
	}
	return tokens, labelStart, nil
}

func evaluateTransformer(ctx context.Context, model *qwen.Model, artifact adapter.TransformerLoRA, examples []Example, maximum int) (float64, error) {
	var total float64
	tokens := 0
	for _, example := range examples {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		sequence, labelStart, err := encodeQwenExample(model, example, maximum)
		if err != nil {
			return 0, err
		}
		loss, count, err := model.CausalLoRALoss(sequence, labelStart, &artifact)
		if err != nil {
			return 0, err
		}
		total += loss
		tokens += count
	}
	if tokens == 0 {
		return 0, fmt.Errorf("evaluation split produced no response tokens")
	}
	return total / float64(tokens), nil
}

func responseTokenCount(model *qwen.Model, examples []Example, maximum int) int {
	count := 0
	for _, example := range examples {
		sequence, labelStart, err := encodeQwenExample(model, example, maximum)
		if err == nil {
			count += len(sequence) - labelStart
		}
	}
	return count
}

func cloneTransformerArtifact(artifact adapter.TransformerLoRA) adapter.TransformerLoRA {
	clone := artifact
	clone.Targets = make(map[string]adapter.ProjectionLoRA, len(artifact.Targets))
	for name, target := range artifact.Targets {
		target.A, target.B = append([]float32(nil), target.A...), append([]float32(nil), target.B...)
		clone.Targets[name] = target
	}
	return clone
}

func saveTransformerCheckpoint(directory string, artifact adapter.TransformerLoRA, state qwenTrainerState, optimizer *AdamW) error {
	if err := adapter.SaveTransformerLoRA(filepath.Join(directory, "adapter.json"), artifact); err != nil {
		return err
	}
	if err := writeQwenTrainerState(filepath.Join(directory, "trainer_state.json"), state); err != nil {
		return err
	}
	if optimizer == nil {
		return fmt.Errorf("optimizer is required for a transformer checkpoint")
	}
	return writeAdamWState(filepath.Join(directory, optimizerStateFile), optimizer.Snapshot())
}

func resumeTransformerTraining(path string, model *qwen.Model, cfg Config) (adapter.TransformerLoRA, int, int, float64, AdamWState, error) {
	loaded, err := adapter.LoadQwenAdapter(path)
	if err != nil {
		return adapter.TransformerLoRA{}, 0, 0, 0, AdamWState{}, fmt.Errorf("load resume adapter: %w", err)
	}
	artifact, ok := loaded.(adapter.TransformerLoRA)
	if !ok || artifact.BaseModel != cfg.ModelDir {
		return adapter.TransformerLoRA{}, 0, 0, 0, AdamWState{}, fmt.Errorf("resume adapter is not a compatible transformer LoRA artifact")
	}
	for _, name := range qwenTargetNames(model.Config.NumHiddenLayers) {
		if target, found := artifact.Targets[name]; found && (target.Rank != cfg.LoRARank || target.Alpha != float32(cfg.LoRAAlpha)) {
			return adapter.TransformerLoRA{}, 0, 0, 0, AdamWState{}, fmt.Errorf("resume adapter target %s does not match the LoRA configuration", name)
		}
	}
	directory := path
	if info, statErr := os.Stat(path); statErr == nil && !info.IsDir() {
		directory = filepath.Dir(path)
	}
	state, err := readQwenTrainerState(filepath.Join(directory, "trainer_state.json"))
	if err != nil {
		return adapter.TransformerLoRA{}, 0, 0, 0, AdamWState{}, err
	}
	optimizer, err := readAdamWState(directory)
	if err != nil {
		return adapter.TransformerLoRA{}, 0, 0, 0, AdamWState{}, err
	}
	return artifact, state.CompletedEpoch, state.BestEpoch, state.BestValidationLoss, optimizer, nil
}
