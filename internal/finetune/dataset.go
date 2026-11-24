package finetune

import (
	"bufio"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/sypra-labs/sypra-ai/internal/domain"
)

// Example is a normalized supervised example. Category and intent remain
// optional metadata; instruction and response are the only required training
// values across CSV and JSONL input formats.
type Example struct {
	ID          string            `json:"id"`
	Instruction string            `json:"instruction"`
	Response    string            `json:"response"`
	Category    string            `json:"category,omitempty"`
	Intent      string            `json:"intent,omitempty"`
	Messages    []TrainingMessage `json:"messages,omitempty"`
}

// TrainingMessage keeps a conversation intact so a model-family template can
// render roles and response-token masks correctly instead of flattening chat
// records into an ad-hoc "role: text" prompt.
type TrainingMessage struct {
	Role    string               `json:"role"`
	Content string               `json:"content"`
	Parts   []domain.ContentPart `json:"parts,omitempty"`
}

// SplitReport makes dataset partitioning inspectable and reproducible.
type SplitReport struct {
	Method          string `json:"method"`
	Seed            uint64 `json:"seed"`
	TrainExamples   int    `json:"trainExamples"`
	ValidationCount int    `json:"validationExamples"`
	TestExamples    int    `json:"testExamples"`
	Groups          int    `json:"groups"`
}

// ReadDataset accepts Sypra CSV and common instruction/chat JSONL records.
// CSV requires instruction and response columns. JSONL accepts instruction /
// response, prompt / completion, or messages with a final assistant message.
func ReadDataset(path string) ([]Example, error) {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".json":
		return readJSON(path)
	case ".jsonl":
		return readJSONL(path)
	default:
		return readCSV(path)
	}
}

func readJSON(path string) ([]Example, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var rows []json.RawMessage
	if err := json.Unmarshal(data, &rows); err != nil {
		return nil, fmt.Errorf("JSON training data must be an array: %w", err)
	}
	examples := make([]Example, 0, len(rows))
	for index, row := range rows {
		example, err := parseJSONLExample(row)
		if err != nil {
			return nil, fmt.Errorf("read JSON row %d: %w", index+1, err)
		}
		if err := normalizeExample(&example, fmt.Sprintf("JSON row %d", index+1)); err != nil {
			return nil, err
		}
		examples = append(examples, example)
	}
	return ensureMinimumExamples(examples)
}

func readCSV(path string) ([]Example, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.FieldsPerRecord = -1
	header, err := r.Read()
	if err != nil {
		return nil, fmt.Errorf("read dataset header: %w", err)
	}
	columns := map[string]int{}
	for i, value := range header {
		columns[strings.ToLower(strings.TrimSpace(value))] = i
	}
	for _, name := range []string{"instruction", "response"} {
		if _, ok := columns[name]; !ok {
			return nil, fmt.Errorf("CSV is missing required column %q", name)
		}
	}

	var examples []Example
	for line := 2; ; line++ {
		record, readErr := r.Read()
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return nil, fmt.Errorf("read dataset row %d: %w", line, readErr)
		}
		field := func(name string) string {
			i, ok := columns[name]
			if !ok || i >= len(record) {
				return ""
			}
			return strings.TrimSpace(record[i])
		}
		example := Example{Instruction: field("instruction"), Response: field("response"), Category: field("category"), Intent: field("intent")}
		if err := normalizeExample(&example, fmt.Sprintf("CSV row %d", line)); err != nil {
			return nil, err
		}
		examples = append(examples, example)
	}
	return ensureMinimumExamples(examples)
}

func readJSONL(path string) ([]Example, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var examples []Example
	scanner := bufio.NewScanner(f)
	// Training examples can contain large retrieved contexts or conversations.
	scanner.Buffer(make([]byte, 64<<10), 8<<20)
	for line := 1; scanner.Scan(); line++ {
		text := strings.TrimSpace(scanner.Text())
		if text == "" {
			continue
		}
		example, err := parseJSONLExample([]byte(text))
		if err != nil {
			return nil, fmt.Errorf("read JSONL row %d: %w", line, err)
		}
		if err := normalizeExample(&example, fmt.Sprintf("JSONL row %d", line)); err != nil {
			return nil, err
		}
		examples = append(examples, example)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read JSONL dataset: %w", err)
	}
	return ensureMinimumExamples(examples)
}

func parseJSONLExample(data []byte) (Example, error) {
	var value struct {
		ID          string `json:"id"`
		Instruction string `json:"instruction"`
		Response    string `json:"response"`
		Prompt      string `json:"prompt"`
		Completion  string `json:"completion"`
		Category    string `json:"category"`
		Intent      string `json:"intent"`
		Messages    []struct {
			Role    string               `json:"role"`
			Content json.RawMessage      `json:"content"`
			Parts   []domain.ContentPart `json:"parts"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(data, &value); err != nil {
		return Example{}, err
	}
	example := Example{ID: value.ID, Instruction: value.Instruction, Response: value.Response, Category: value.Category, Intent: value.Intent}
	if example.Instruction == "" {
		example.Instruction = value.Prompt
	}
	if example.Response == "" {
		example.Response = value.Completion
	}
	if example.Instruction != "" || example.Response != "" {
		return example, nil
	}
	if len(value.Messages) < 2 || value.Messages[len(value.Messages)-1].Role != "assistant" {
		return Example{}, fmt.Errorf("needs instruction/response, prompt/completion, or messages ending in assistant")
	}
	for _, message := range value.Messages {
		parsed, err := parseTrainingContent(message.Content, message.Parts)
		if err != nil {
			return Example{}, err
		}
		example.Messages = append(example.Messages, TrainingMessage{Role: strings.TrimSpace(message.Role), Content: parsed, Parts: message.Parts})
	}
	var prompt strings.Builder
	for _, message := range example.Messages[:len(example.Messages)-1] {
		role, content := strings.TrimSpace(message.Role), domain.ContentText(message.Content, message.Parts)
		if role == "" || content == "" {
			return Example{}, fmt.Errorf("messages contain an empty role or content")
		}
		if prompt.Len() > 0 {
			prompt.WriteByte('\n')
		}
		prompt.WriteString(role)
		prompt.WriteString(": ")
		prompt.WriteString(content)
	}
	example.Instruction = prompt.String()
	example.Response = domain.ContentText(example.Messages[len(example.Messages)-1].Content, example.Messages[len(example.Messages)-1].Parts)
	return example, nil
}

func parseTrainingContent(raw json.RawMessage, parts []domain.ContentPart) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return domain.ContentText("", parts), domain.ValidateContentParts(parts)
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, domain.ValidateContentParts(parts)
	}
	var contentParts []domain.ContentPart
	if err := json.Unmarshal(raw, &contentParts); err != nil {
		return "", fmt.Errorf("invalid message content: %w", err)
	}
	if err := domain.ValidateContentParts(contentParts); err != nil {
		return "", err
	}
	return domain.ContentText("", contentParts), nil
}

func normalizeExample(example *Example, location string) error {
	example.Instruction = strings.TrimSpace(example.Instruction)
	example.Response = strings.TrimSpace(example.Response)
	example.Category = strings.TrimSpace(example.Category)
	example.Intent = strings.TrimSpace(example.Intent)
	if len(example.Messages) > 0 {
		for _, message := range example.Messages {
			if err := domain.ValidateContentParts(message.Parts); err != nil {
				return fmt.Errorf("%s has invalid message content: %w", location, err)
			}
		}
	}
	if example.Instruction == "" || example.Response == "" {
		return fmt.Errorf("%s needs non-empty instruction and response", location)
	}
	if example.ID == "" {
		sum := sha256.Sum256([]byte(example.Instruction + "\x00" + example.Response))
		example.ID = hex.EncodeToString(sum[:8])
	}
	return nil
}

func ensureMinimumExamples(examples []Example) ([]Example, error) {
	if len(examples) < 3 {
		return nil, fmt.Errorf("dataset needs at least three examples for train, validation, and test")
	}
	return examples, nil
}

// SplitDataset retains the original public shape for the trainer, while using
// a content-derived grouping key so duplicate prompt/response pairs cannot be
// split across train and validation.
func SplitDataset(examples []Example) (train, validation []Example) {
	train, validation, _ = SplitDataset3(examples, 1)
	return train, validation
}

// SplitDataset3 creates stable 80/10/10 grouped partitions. Identical pairs
// always land in the same split, avoiding the most basic form of leakage.
func SplitDataset3(examples []Example, seed uint64) (train, validation, test []Example) {
	for _, example := range examples {
		bucket := splitBucket(example.ID, seed)
		switch {
		case bucket < 80:
			train = append(train, example)
		case bucket < 90:
			validation = append(validation, example)
		default:
			test = append(test, example)
		}
	}
	if len(validation) == 0 && len(train) > 1 {
		validation, train = append(validation, train[len(train)-1]), train[:len(train)-1]
	}
	if len(test) == 0 && len(train) > 1 {
		test, train = append(test, train[len(train)-1]), train[:len(train)-1]
	}
	return train, validation, test
}

func DatasetSplitReport(examples []Example, seed uint64) SplitReport {
	train, validation, test := SplitDataset3(examples, seed)
	groups := map[string]struct{}{}
	for _, example := range examples {
		groups[example.ID] = struct{}{}
	}
	return SplitReport{Method: "content-hash-80-10-10", Seed: seed, TrainExamples: len(train), ValidationCount: len(validation), TestExamples: len(test), Groups: len(groups)}
}

func splitBucket(key string, seed uint64) uint64 {
	sum := sha256.Sum256([]byte(fmt.Sprintf("%d:%s", seed, key)))
	return uint64(sum[0]) * 100 / 256
}
