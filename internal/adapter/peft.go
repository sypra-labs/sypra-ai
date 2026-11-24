package adapter

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	PEFTConfigFile      = "adapter_config.json"
	PEFTWeightsFile     = "adapter_model.safetensors"
	peftTensorNameStart = "base_model.model."
)

// ExportPEFT writes a Hugging Face PEFT-compatible LoRA adapter directory for
// the native output-head artifact.
func ExportPEFT(directory string, artifact QwenLMHeadLoRA) error {
	if err := artifact.Validate(); err != nil {
		return err
	}
	return ExportTransformerPEFT(directory, TransformerLoRA{
		Version:   QwenAdapterArtifactV1,
		Type:      TransformerLoRAType,
		BaseModel: artifact.BaseModel,
		CreatedAt: artifact.CreatedAt,
		Targets: map[string]ProjectionLoRA{
			"lm_head.weight": {Rows: artifact.VocabSize, Cols: artifact.HiddenSize, Rank: artifact.Rank, Alpha: artifact.Alpha, A: artifact.A, B: artifact.B},
		},
	})
}

// ExportTransformerPEFT writes the common PEFT LoRA interchange files. It
// intentionally stores adapter tensors as F32 so the artifact can be used by
// standard Transformers/PEFT tooling without a Sypra runtime dependency.
func ExportTransformerPEFT(directory string, artifact TransformerLoRA) error {
	if err := artifact.Validate(); err != nil {
		return err
	}
	if err := os.MkdirAll(directory, 0755); err != nil {
		return err
	}
	targets := make([]string, 0, len(artifact.Targets))
	tensors := map[string]peftTensor{}
	for targetName, target := range artifact.Targets {
		module := strings.TrimSuffix(targetName, ".weight")
		targets = append(targets, shortTargetName(module))
		prefix := peftTensorNameStart + module
		tensors[prefix+".lora_A.weight"] = peftTensor{Shape: []int{target.Rank, target.Cols}, Data: target.A}
		tensors[prefix+".lora_B.weight"] = peftTensor{Shape: []int{target.Rows, target.Rank}, Data: target.B}
	}
	sort.Strings(targets)
	targets = compactStrings(targets)
	config := struct {
		BaseModelNameOrPath string   `json:"base_model_name_or_path"`
		Bias                string   `json:"bias"`
		FanInFanOut         bool     `json:"fan_in_fan_out"`
		InferenceMode       bool     `json:"inference_mode"`
		LoraAlpha           float32  `json:"lora_alpha"`
		LoraDropout         float64  `json:"lora_dropout"`
		PEFTType            string   `json:"peft_type"`
		R                   int      `json:"r"`
		TargetModules       []string `json:"target_modules"`
		TaskType            string   `json:"task_type"`
	}{BaseModelNameOrPath: artifact.BaseModel, Bias: "none", FanInFanOut: false, InferenceMode: true, LoraAlpha: firstAlpha(artifact.Targets), LoraDropout: 0, PEFTType: "LORA", R: firstRank(artifact.Targets), TargetModules: targets, TaskType: "CAUSAL_LM"}
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(directory, PEFTConfigFile), data, 0600); err != nil {
		return err
	}
	return writePEFTSafeTensors(filepath.Join(directory, PEFTWeightsFile), tensors)
}

// ImportPEFT reads a float32 Hugging Face PEFT LoRA adapter. The resulting
// runtime artifact supports the attention, MLP, and lm_head targets used by
// Sypra's Qwen runner.
func ImportPEFT(directory string) (TransformerLoRA, error) {
	configData, err := os.ReadFile(filepath.Join(directory, PEFTConfigFile))
	if err != nil {
		return TransformerLoRA{}, err
	}
	var config struct {
		BaseModelNameOrPath string  `json:"base_model_name_or_path"`
		LoraAlpha           float32 `json:"lora_alpha"`
		R                   int     `json:"r"`
		PEFTType            string  `json:"peft_type"`
	}
	if err := json.Unmarshal(configData, &config); err != nil {
		return TransformerLoRA{}, fmt.Errorf("read PEFT config: %w", err)
	}
	if !strings.EqualFold(config.PEFTType, "LORA") || config.R < 1 || config.LoraAlpha <= 0 {
		return TransformerLoRA{}, fmt.Errorf("only a valid PEFT LoRA adapter is supported")
	}
	tensors, err := readPEFTSafeTensors(filepath.Join(directory, PEFTWeightsFile))
	if err != nil {
		return TransformerLoRA{}, err
	}
	targets := map[string]ProjectionLoRA{}
	for name, a := range tensors {
		if !strings.HasPrefix(name, peftTensorNameStart) || !strings.HasSuffix(name, ".lora_A.weight") {
			continue
		}
		prefix := strings.TrimSuffix(name, ".lora_A.weight")
		b, ok := tensors[prefix+".lora_B.weight"]
		if !ok || len(a.Shape) != 2 || len(b.Shape) != 2 || a.Shape[0] != b.Shape[1] || a.Shape[0] != config.R {
			return TransformerLoRA{}, fmt.Errorf("invalid PEFT LoRA pair for %s", prefix)
		}
		module := strings.TrimPrefix(prefix, peftTensorNameStart)
		targetName := module + ".weight"
		targets[targetName] = ProjectionLoRA{Rows: b.Shape[0], Cols: a.Shape[1], Rank: a.Shape[0], Alpha: config.LoraAlpha, A: a.Data, B: b.Data}
	}
	artifact := TransformerLoRA{Version: QwenAdapterArtifactV1, Type: TransformerLoRAType, BaseModel: config.BaseModelNameOrPath, CreatedAt: time.Now().UTC(), Targets: targets}
	if err := artifact.Validate(); err != nil {
		return TransformerLoRA{}, err
	}
	return artifact, nil
}

type peftTensor struct {
	Shape []int
	Data  []float32
}

type peftSafeEntry struct {
	DType   string `json:"dtype"`
	Shape   []int  `json:"shape"`
	Offsets []int  `json:"data_offsets"`
}

func writePEFTSafeTensors(path string, tensors map[string]peftTensor) error {
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	header := map[string]any{"__metadata__": map[string]string{"format": "pt"}}
	offset := 0
	for _, name := range names {
		tensor := tensors[name]
		if tensorElements(tensor.Shape) != len(tensor.Data) {
			return fmt.Errorf("invalid PEFT tensor %s", name)
		}
		size := len(tensor.Data) * 4
		header[name] = peftSafeEntry{DType: "F32", Shape: tensor.Shape, Offsets: []int{offset, offset + size}}
		offset += size
	}
	encoded, err := json.Marshal(header)
	if err != nil {
		return err
	}
	for len(encoded)%8 != 0 {
		encoded = append(encoded, ' ')
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer file.Close()
	if err := binary.Write(file, binary.LittleEndian, uint64(len(encoded))); err != nil {
		return err
	}
	if _, err := file.Write(encoded); err != nil {
		return err
	}
	buf := make([]byte, 4)
	for _, name := range names {
		for _, value := range tensors[name].Data {
			binary.LittleEndian.PutUint32(buf, math.Float32bits(value))
			if _, err := file.Write(buf); err != nil {
				return err
			}
		}
	}
	return file.Close()
}

func readPEFTSafeTensors(path string) (map[string]peftTensor, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	var headerSize uint64
	if err := binary.Read(file, binary.LittleEndian, &headerSize); err != nil {
		return nil, err
	}
	if headerSize > 64<<20 {
		return nil, fmt.Errorf("invalid PEFT safetensors header")
	}
	header := make([]byte, headerSize)
	if _, err := io.ReadFull(file, header); err != nil {
		return nil, err
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(header, &raw); err != nil {
		return nil, err
	}
	dataStart := int64(8 + headerSize)
	result := map[string]peftTensor{}
	for name, value := range raw {
		if name == "__metadata__" {
			continue
		}
		var entry peftSafeEntry
		if err := json.Unmarshal(value, &entry); err != nil {
			return nil, fmt.Errorf("read PEFT tensor header %s: %w", name, err)
		}
		bytesPerElement, supported := peftDTypeBytes(entry.DType)
		if !supported || len(entry.Offsets) != 2 || entry.Offsets[0] < 0 || entry.Offsets[1] < entry.Offsets[0] || entry.Offsets[1]-entry.Offsets[0] != tensorElements(entry.Shape)*bytesPerElement {
			return nil, fmt.Errorf("unsupported or invalid PEFT tensor %s", name)
		}
		bytes := make([]byte, entry.Offsets[1]-entry.Offsets[0])
		if _, err := file.ReadAt(bytes, dataStart+int64(entry.Offsets[0])); err != nil {
			return nil, err
		}
		values, err := decodePEFTTensor(entry.DType, bytes)
		if err != nil {
			return nil, fmt.Errorf("decode PEFT tensor %s: %w", name, err)
		}
		result[name] = peftTensor{Shape: entry.Shape, Data: values}
	}
	return result, nil
}

func peftDTypeBytes(dtype string) (int, bool) {
	switch strings.ToUpper(dtype) {
	case "F32":
		return 4, true
	case "BF16", "F16":
		return 2, true
	default:
		return 0, false
	}
}

// decodePEFTTensor normalizes common Hugging Face LoRA tensor precisions to
// F32. Sypra keeps its runtime adapter representation in F32, while allowing
// imports produced by BF16/FP16 QLoRA training jobs.
func decodePEFTTensor(dtype string, bytes []byte) ([]float32, error) {
	width, supported := peftDTypeBytes(dtype)
	if !supported || len(bytes)%width != 0 {
		return nil, fmt.Errorf("unsupported PEFT dtype %q", dtype)
	}
	values := make([]float32, len(bytes)/width)
	switch strings.ToUpper(dtype) {
	case "F32":
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(bytes[i*4:]))
		}
	case "BF16":
		for i := range values {
			values[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(bytes[i*2:])) << 16)
		}
	case "F16":
		for i := range values {
			values[i] = float32FromHalf(binary.LittleEndian.Uint16(bytes[i*2:]))
		}
	}
	return values, nil
}

func float32FromHalf(value uint16) float32 {
	sign := uint32(value&0x8000) << 16
	exponent := uint32(value >> 10 & 0x1f)
	mantissa := uint32(value & 0x03ff)
	switch exponent {
	case 0:
		if mantissa == 0 {
			return math.Float32frombits(sign)
		}
		// Normalize a subnormal half-precision value.
		exponent = 1
		for mantissa&0x0400 == 0 {
			mantissa <<= 1
			exponent--
		}
		mantissa &= 0x03ff
		exponent += 112
	case 31:
		return math.Float32frombits(sign | 0x7f800000 | mantissa<<13)
	default:
		exponent += 112
	}
	return math.Float32frombits(sign | exponent<<23 | mantissa<<13)
}

func tensorElements(shape []int) int {
	count := 1
	for _, dimension := range shape {
		if dimension < 1 {
			return 0
		}
		count *= dimension
	}
	return count
}

func shortTargetName(module string) string {
	parts := strings.Split(module, ".")
	return parts[len(parts)-1]
}

func compactStrings(values []string) []string {
	if len(values) == 0 {
		return values
	}
	result := values[:1]
	for _, value := range values[1:] {
		if value != result[len(result)-1] {
			result = append(result, value)
		}
	}
	return result
}

func firstAlpha(targets map[string]ProjectionLoRA) float32 {
	for _, target := range targets {
		return target.Alpha
	}
	return 0
}

func firstRank(targets map[string]ProjectionLoRA) int {
	for _, target := range targets {
		return target.Rank
	}
	return 0
}
