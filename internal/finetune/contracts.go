// Package finetune owns Sypra's native Qwen fine-tuning contract.
package finetune

import (
	"fmt"
)

const DefaultBaseModel = "Qwen/Qwen3-0.6B"

var DatasetContract = []string{"instruction", "response"}

type Config struct {
	ModelName, ModelDir, CacheDir, Method, ComputeBackend                                          string
	Epochs, BatchSize, EvalBatchSize, GradientAccumulation, LoRARank, LoRAAlpha, MaxSequenceLength int
	LearningRate, LoRADropout, WeightDecay, MaxGradientNorm, AdamBeta1, AdamBeta2, AdamEpsilon     float64
	UseNF4, DoubleQuant                                                                            bool
	Seed                                                                                           uint64
	CheckpointEvery, EarlyStoppingPatience                                                         int
	MinValidationImprovement                                                                       float64
	ResumeFrom                                                                                     string
	TargetModules                                                                                  []string
	GPUDevices                                                                                     []int
	WorkerEndpoints                                                                                []string
}

func DefaultConfig() Config {
	return Config{ModelName: DefaultBaseModel, Method: "qlora", ComputeBackend: "auto", Epochs: 2, LearningRate: .0002, BatchSize: 1, EvalBatchSize: 1, GradientAccumulation: 1, LoRARank: 16, LoRAAlpha: 32, LoRADropout: 0, MaxSequenceLength: 0, UseNF4: true, DoubleQuant: true, Seed: 1, CheckpointEvery: 1, EarlyStoppingPatience: 0, MinValidationImprovement: 0, WeightDecay: .01, MaxGradientNorm: 1, AdamBeta1: .9, AdamBeta2: .999, AdamEpsilon: 1e-8}
}
func (c Config) Validate() error {
	if c.ModelName == "" {
		return fmt.Errorf("base model is required")
	}
	if c.Method != "lora" && c.Method != "qlora" {
		return fmt.Errorf("method must be lora or qlora")
	}
	if c.ComputeBackend != "auto" && c.ComputeBackend != "cpu" && c.ComputeBackend != "cuda" {
		return fmt.Errorf("compute backend must be auto, cpu, or cuda")
	}
	if len(c.GPUDevices) > 0 && c.ComputeBackend == "cpu" {
		return fmt.Errorf("GPU devices require auto or cuda backend")
	}
	if c.Epochs < 1 || c.BatchSize < 1 || c.GradientAccumulation < 1 || c.LearningRate <= 0 || c.LoRARank < 1 || c.LoRAAlpha < 1 || c.MaxSequenceLength < 0 {
		return fmt.Errorf("invalid training parameters")
	}
	if c.CheckpointEvery < 1 || c.EarlyStoppingPatience < 0 || c.MinValidationImprovement < 0 {
		return fmt.Errorf("invalid checkpoint or early stopping parameters")
	}
	if c.WeightDecay < 0 || c.MaxGradientNorm < 0 || c.AdamBeta1 <= 0 || c.AdamBeta1 >= 1 || c.AdamBeta2 <= 0 || c.AdamBeta2 >= 1 || c.AdamEpsilon <= 0 {
		return fmt.Errorf("invalid AdamW parameters")
	}
	if c.Method == "qlora" && !c.UseNF4 {
		return fmt.Errorf("QLoRA requires NF4 quantization")
	}
	return nil
}
func ValidateCSV(path string) error {
	_, err := ReadDataset(path)
	return err
}
