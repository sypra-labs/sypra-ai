package finetune

import (
	"fmt"
	"os"
	"path/filepath"
)

type Run struct {
	ID, DatasetPath, Directory string
	Config                     Config
}

func (r Run) Prepare() (string, error) {
	if e := r.Config.Validate(); e != nil {
		return "", e
	}
	if e := ValidateCSV(r.DatasetPath); e != nil {
		return "", e
	}
	if e := os.MkdirAll(filepath.Join(r.Directory, "data", "raw"), 0755); e != nil {
		return "", e
	}
	data, err := os.ReadFile(r.DatasetPath)
	if err != nil {
		return "", err
	}
	raw := filepath.Join(r.Directory, "data", "raw", "bitext.csv")
	if err = os.WriteFile(raw, data, 0600); err != nil {
		return "", err
	}
	configPath := filepath.Join(r.Directory, "run.yaml")
	return configPath, os.WriteFile(configPath, []byte(r.YAML()), 0600)
}

func (r Run) YAML() string {
	c := r.Config
	return fmt.Sprintf("model:\n  name: %s\n  local_path: %q\n  max_seq_length: %d\nruntime:\n  engine: sypra-native-cpu\n  external_dependencies: false\nlora:\n  r: %d\n  alpha: %d\n  dropout: %.4f\ntraining:\n  method: %s\n  epochs: %d\n  learning_rate: %.8f\n  batch_size: %d\n  eval_batch_size: %d\n  gradient_accumulation_steps: %d\n  seed: %d\n  checkpoint_every: %d\n  early_stopping_patience: %d\n  min_validation_improvement: %.8f\n  resume_from: %q\nvalidation:\n  split: grouped-content-hash-80-10-10\nserving:\n  adapter_path: %q\n  variant: %q\n", c.ModelName, c.ModelDir, c.MaxSequenceLength, c.LoRARank, c.LoRAAlpha, c.LoRADropout, c.Method, c.Epochs, c.LearningRate, c.BatchSize, c.EvalBatchSize, c.GradientAccumulation, c.Seed, c.CheckpointEvery, c.EarlyStoppingPatience, c.MinValidationImprovement, c.ResumeFrom, filepath.Join(r.Directory, "artifacts", "best"), r.ID)
}
