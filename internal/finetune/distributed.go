package finetune

import (
	"context"
	"fmt"
	"sync"

	"github.com/sypra-labs/sypra-ai/internal/adapter"
	"github.com/sypra-labs/sypra-ai/internal/qwen"
)

// GradientWorker owns an independent frozen-model replica. Every worker sees
// the same immutable LoRA snapshot for a batch; the coordinator all-reduces
// gradients before applying the next AdamW update.
type GradientWorker interface {
	Compute(context.Context, adapter.TransformerLoRA, []Example, int) (GradientBatch, error)
	Close() error
}

type GradientBatch struct {
	Loss      float64                               `json:"loss"`
	Tokens    int                                   `json:"tokens"`
	Gradients map[string]adapter.ProjectionGradient `json:"gradients"`
}

type localGradientWorker struct{ model *qwen.Model }

func NewLocalGradientWorker(cfg Config, device *int) (GradientWorker, error) {
	model, err := qwen.LoadModel(cfg.ModelDir)
	if err != nil {
		return nil, err
	}
	if cfg.Method == "qlora" {
		model.EnableNF4(cfg.DoubleQuant)
	}
	if device != nil {
		executor, err := qwen.NewCUDAExecutorForDevice(*device)
		if err != nil {
			_ = model.Close()
			return nil, err
		}
		model.SetMatrixExecutor(executor)
	} else if cfg.ComputeBackend != "cpu" {
		executor, err := qwen.NewCUDAExecutor()
		if err != nil {
			if cfg.ComputeBackend == "cuda" {
				_ = model.Close()
				return nil, err
			}
		} else {
			model.SetMatrixExecutor(executor)
		}
	}
	return &localGradientWorker{model: model}, nil
}

func (w *localGradientWorker) Compute(ctx context.Context, artifact adapter.TransformerLoRA, examples []Example, maximumSequenceLength int) (GradientBatch, error) {
	result := GradientBatch{Gradients: adapter.NewTransformerGradients(&artifact)}
	for _, example := range examples {
		if err := ctx.Err(); err != nil {
			return GradientBatch{}, err
		}
		tokens, labelStart, err := encodeQwenExample(w.model, example, maximumSequenceLength)
		if err != nil {
			return GradientBatch{}, err
		}
		loss, count, gradients, err := w.model.CausalLoRAGradients(tokens, labelStart, &artifact)
		if err != nil {
			return GradientBatch{}, err
		}
		if err := adapter.AddTransformerGradients(result.Gradients, gradients); err != nil {
			return GradientBatch{}, err
		}
		result.Loss += loss
		result.Tokens += count
	}
	return result, nil
}

func (w *localGradientWorker) Close() error { return w.model.Close() }

func localGradientBatch(ctx context.Context, model *qwen.Model, artifact adapter.TransformerLoRA, examples []Example, maximumSequenceLength int) (GradientBatch, error) {
	return (&localGradientWorker{model: model}).Compute(ctx, artifact, examples, maximumSequenceLength)
}

func newDistributedWorkers(cfg Config) ([]GradientWorker, error) {
	workers := make([]GradientWorker, 0, len(cfg.GPUDevices)+len(cfg.WorkerEndpoints))
	for _, device := range cfg.GPUDevices {
		worker, err := NewLocalGradientWorker(cfg, &device)
		if err != nil {
			closeGradientWorkers(workers)
			return nil, fmt.Errorf("start GPU worker %d: %w", device, err)
		}
		workers = append(workers, worker)
	}
	for _, endpoint := range cfg.WorkerEndpoints {
		worker, err := NewRemoteGradientWorker(endpoint)
		if err != nil {
			closeGradientWorkers(workers)
			return nil, err
		}
		workers = append(workers, worker)
	}
	return workers, nil
}

func closeGradientWorkers(workers []GradientWorker) {
	for _, worker := range workers {
		_ = worker.Close()
	}
}

func allReduceGradientBatch(ctx context.Context, workers []GradientWorker, artifact adapter.TransformerLoRA, examples []Example, maximumSequenceLength int) (GradientBatch, error) {
	if len(workers) == 0 {
		return GradientBatch{}, fmt.Errorf("at least one worker is required")
	}
	partitions := make([][]Example, len(workers))
	for index, example := range examples {
		partitions[index%len(workers)] = append(partitions[index%len(workers)], example)
	}
	result := GradientBatch{Gradients: adapter.NewTransformerGradients(&artifact)}
	type response struct {
		batch GradientBatch
		err   error
	}
	responses := make(chan response, len(workers))
	var group sync.WaitGroup
	for index, worker := range workers {
		if len(partitions[index]) == 0 {
			continue
		}
		group.Add(1)
		go func(worker GradientWorker, partition []Example) {
			defer group.Done()
			batch, err := worker.Compute(ctx, artifact, partition, maximumSequenceLength)
			responses <- response{batch: batch, err: err}
		}(worker, partitions[index])
	}
	group.Wait()
	close(responses)
	for response := range responses {
		if response.err != nil {
			return GradientBatch{}, response.err
		}
		if err := adapter.AddTransformerGradients(result.Gradients, response.batch.Gradients); err != nil {
			return GradientBatch{}, err
		}
		result.Loss += response.batch.Loss
		result.Tokens += response.batch.Tokens
	}
	if result.Tokens == 0 {
		return GradientBatch{}, fmt.Errorf("distributed batch produced no response tokens")
	}
	return result, nil
}
