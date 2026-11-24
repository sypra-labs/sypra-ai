package finetune

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode"
)

const nativeFeatureCount = 512

// Artifact is a portable, dependency-free local adapter. It is deliberately
// separate from the frozen base checkpoint, mirroring how a LoRA adapter is
// selected alongside a base model at serving time.
type Artifact struct {
	Version      int         `json:"version"`
	Type         string      `json:"type"`
	BaseModel    string      `json:"baseModel"`
	Method       string      `json:"method"`
	CreatedAt    time.Time   `json:"createdAt"`
	FeatureCount int         `json:"featureCount"`
	Responses    []string    `json:"responses"`
	Weights      [][]float32 `json:"weights"`
	Fallback     string      `json:"fallback"`
}

type Progress struct {
	Progress           float64
	Phase              string
	Epoch, Epochs      int
	Step, Steps        int
	Tokens             int
	TokensPerSecond    float64
	TrainLoss          float64
	EvalLoss           float64
	ValidationComplete bool
}

type Result struct {
	ArtifactPath   string
	ArtifactFormat string
	TrainLoss      float64
	EvalLoss       float64
	TestLoss       float64
}

// TrainNative builds a small trainable response adapter entirely in Go. The
// base checkpoint remains immutable; this creates the adapter artifact which
// is registered as an SLM after validation succeeds.
func TrainNative(ctx context.Context, datasetPath, outputDir string, cfg Config, report func(Progress)) (Result, error) {
	if err := cfg.Validate(); err != nil {
		return Result{}, err
	}
	examples, err := ReadDataset(datasetPath)
	if err != nil {
		return Result{}, err
	}
	train, validation := SplitDataset(examples)
	// Build the output vocabulary from the full labelled corpus so validation
	// examples whose response appears only in that split retain a valid class.
	classes := uniqueResponses(examples)
	if len(classes) < 2 {
		return Result{}, fmt.Errorf("training rows need at least two distinct responses")
	}
	classByResponse := make(map[string]int, len(classes))
	for i, response := range classes {
		classByResponse[response] = i
	}
	weights := make([][]float32, len(classes))
	for i := range weights {
		weights[i] = make([]float32, nativeFeatureCount)
	}

	var trainLoss, evalLoss float64
	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		if err := ctx.Err(); err != nil {
			return Result{}, err
		}
		trainLoss = 0
		for _, example := range train {
			if err := ctx.Err(); err != nil {
				return Result{}, err
			}
			features := featuresFor(example.Instruction, nativeFeatureCount)
			probabilities := softmax(scores(weights, features))
			target := classByResponse[example.Response]
			trainLoss -= math.Log(math.Max(float64(probabilities[target]), 1e-8))
			for class, probability := range probabilities {
				gradient := probability
				if class == target {
					gradient -= 1
				}
				for index, value := range features {
					weights[class][index] -= float32(cfg.LearningRate) * gradient * value
				}
			}
		}
		trainLoss /= float64(len(train))
		evalLoss = loss(weights, validation, classByResponse)
		if report != nil {
			report(Progress{Progress: float64(epoch+1) / float64(cfg.Epochs), Phase: "training", Epoch: epoch + 1, Epochs: cfg.Epochs, Step: epoch + 1, Steps: cfg.Epochs, TrainLoss: trainLoss, EvalLoss: evalLoss, ValidationComplete: true})
		}
	}

	artifact := Artifact{Version: 1, Type: "sypra-native-response-adapter", BaseModel: cfg.ModelDir, Method: cfg.Method, CreatedAt: time.Now().UTC(), FeatureCount: nativeFeatureCount, Responses: classes, Weights: weights, Fallback: "This fine-tuned local adapter has no matching example for that request."}
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return Result{}, err
	}
	path := filepath.Join(outputDir, "adapter.json")
	data, err := json.MarshalIndent(artifact, "", "  ")
	if err != nil {
		return Result{}, err
	}
	if err = os.WriteFile(path, data, 0600); err != nil {
		return Result{}, err
	}
	return Result{ArtifactPath: path, ArtifactFormat: "sypra-adapter", TrainLoss: trainLoss, EvalLoss: evalLoss}, nil
}

func LoadArtifact(path string) (Artifact, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Artifact{}, err
	}
	var artifact Artifact
	if err = json.Unmarshal(data, &artifact); err != nil {
		return Artifact{}, fmt.Errorf("read native adapter: %w", err)
	}
	if artifact.Type != "sypra-native-response-adapter" || artifact.FeatureCount < 1 || len(artifact.Responses) == 0 || len(artifact.Weights) != len(artifact.Responses) {
		return Artifact{}, fmt.Errorf("invalid native adapter artifact")
	}
	return artifact, nil
}

func (a Artifact) Generate(prompt string) string {
	if len(a.Responses) == 0 {
		return a.Fallback
	}
	index := bestClass(scores(a.Weights, featuresFor(prompt, a.FeatureCount)))
	if index < 0 || index >= len(a.Responses) || len(words(prompt)) == 0 {
		return a.Fallback
	}
	return a.Responses[index]
}

func uniqueResponses(examples []Example) []string {
	set := map[string]bool{}
	for _, example := range examples {
		set[example.Response] = true
	}
	responses := make([]string, 0, len(set))
	for response := range set {
		responses = append(responses, response)
	}
	sort.Strings(responses)
	return responses
}

func featuresFor(text string, size int) map[int]float32 {
	features := map[int]float32{}
	for _, token := range words(text) {
		h := fnv.New32a()
		_, _ = h.Write([]byte(token))
		features[int(h.Sum32()%uint32(size))] += 1
	}
	if len(features) == 0 {
		return features
	}
	norm := float32(0)
	for _, value := range features {
		norm += value * value
	}
	norm = float32(math.Sqrt(float64(norm)))
	for index, value := range features {
		features[index] = value / norm
	}
	return features
}

func words(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsNumber(r) })
}

func scores(weights [][]float32, features map[int]float32) []float32 {
	values := make([]float32, len(weights))
	for class, row := range weights {
		for index, value := range features {
			if index < len(row) {
				values[class] += row[index] * value
			}
		}
	}
	return values
}

func softmax(values []float32) []float32 {
	probabilities := make([]float32, len(values))
	if len(values) == 0 {
		return probabilities
	}
	max := values[0]
	for _, value := range values[1:] {
		if value > max {
			max = value
		}
	}
	var total float64
	for i, value := range values {
		probabilities[i] = float32(math.Exp(float64(value - max)))
		total += float64(probabilities[i])
	}
	for i := range probabilities {
		probabilities[i] /= float32(total)
	}
	return probabilities
}

func loss(weights [][]float32, examples []Example, classByResponse map[string]int) float64 {
	var total float64
	for _, example := range examples {
		probability := softmax(scores(weights, featuresFor(example.Instruction, len(weights[0]))))[classByResponse[example.Response]]
		total -= math.Log(math.Max(float64(probability), 1e-8))
	}
	return total / float64(len(examples))
}

func bestClass(values []float32) int {
	if len(values) == 0 {
		return -1
	}
	best := 0
	for i := 1; i < len(values); i++ {
		if values[i] > values[best] {
			best = i
		}
	}
	return best
}
