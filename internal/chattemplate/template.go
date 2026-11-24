// Package chattemplate renders the role/control-token layouts expected by
// common causal chat model families. It is deliberately tokenizer-agnostic:
// runtimes encode the returned segments with their own vocabulary, while the
// Train flag identifies the exact assistant-response tokens to optimize.
package chattemplate

import (
	"fmt"
	"strings"

	"github.com/sypra-labs/sypra-ai/internal/domain"
)

type Message struct {
	Role    string
	Content string
	Parts   []domain.ContentPart
}

type Segment struct {
	Text    string
	Control string
	Train   bool
}

type Rendered struct{ Segments []Segment }

const (
	FamilyQwen    = "qwen"
	FamilyLlama   = "llama"
	FamilyMistral = "mistral"
	FamilyGemma   = "gemma"
	FamilyPhi     = "phi"
	FamilyGeneric = "generic"
)

func NormalizeFamily(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	switch {
	case strings.Contains(value, "qwen"):
		return FamilyQwen
	case strings.Contains(value, "llama"):
		return FamilyLlama
	case strings.Contains(value, "mistral"), strings.Contains(value, "mixtral"):
		return FamilyMistral
	case strings.Contains(value, "gemma"):
		return FamilyGemma
	case strings.Contains(value, "phi"):
		return FamilyPhi
	default:
		return FamilyGeneric
	}
}

// Render produces a generation prompt or, when training is true, marks the
// final assistant response and its closing token as loss-bearing segments.
func Render(family string, messages []Message, training bool) (Rendered, error) {
	if len(messages) == 0 {
		return Rendered{}, fmt.Errorf("at least one chat message is required")
	}
	for _, message := range messages {
		if err := validRole(message.Role); err != nil {
			return Rendered{}, err
		}
		if err := domain.ValidateContentParts(message.Parts); err != nil {
			return Rendered{}, err
		}
	}
	if training && strings.ToLower(messages[len(messages)-1].Role) != "assistant" {
		return Rendered{}, fmt.Errorf("a training conversation must end with an assistant message")
	}
	switch NormalizeFamily(family) {
	case FamilyQwen:
		return renderQwen(messages, training), nil
	case FamilyLlama:
		return renderLlama(messages, training), nil
	case FamilyMistral:
		return renderMistral(messages, training), nil
	case FamilyGemma:
		return renderGemma(messages, training), nil
	case FamilyPhi:
		return renderPhi(messages, training), nil
	default:
		return renderGeneric(messages, training), nil
	}
}

func validRole(role string) error {
	switch strings.ToLower(strings.TrimSpace(role)) {
	case "system", "user", "assistant", "tool":
		return nil
	default:
		return fmt.Errorf("unsupported message role %q", role)
	}
}

func content(message Message) string {
	return domain.ContentText(message.Content, message.Parts)
}

func lastTrain(messages []Message, index int, training bool) bool {
	return training && index == len(messages)-1 && strings.EqualFold(messages[index].Role, "assistant")
}

func renderQwen(messages []Message, training bool) Rendered {
	var out Rendered
	for index, message := range messages {
		out.Segments = append(out.Segments, Segment{Control: "<|im_start|>"}, Segment{Text: strings.ToLower(message.Role) + "\n"}, Segment{Text: content(message), Train: lastTrain(messages, index, training)}, Segment{Control: "<|im_end|>", Train: lastTrain(messages, index, training)}, Segment{Text: "\n", Train: lastTrain(messages, index, training)})
	}
	if !training {
		out.Segments = append(out.Segments, Segment{Control: "<|im_start|>"}, Segment{Text: "assistant\n"}, Segment{Control: "<think>"}, Segment{Text: "\n\n"}, Segment{Control: "</think>"}, Segment{Text: "\n\n"})
	}
	return out
}

func renderLlama(messages []Message, training bool) Rendered {
	out := Rendered{Segments: []Segment{{Control: "<|begin_of_text|>"}}}
	for index, message := range messages {
		train := lastTrain(messages, index, training)
		out.Segments = append(out.Segments, Segment{Control: "<|start_header_id|>"}, Segment{Text: strings.ToLower(message.Role)}, Segment{Control: "<|end_header_id|>"}, Segment{Text: "\n\n"}, Segment{Text: content(message), Train: train}, Segment{Control: "<|eot_id|>", Train: train})
	}
	if !training {
		out.Segments = append(out.Segments, Segment{Control: "<|start_header_id|>"}, Segment{Text: "assistant"}, Segment{Control: "<|end_header_id|>"}, Segment{Text: "\n\n"})
	}
	return out
}

func renderMistral(messages []Message, training bool) Rendered {
	var out Rendered
	for index, message := range messages {
		role, train := strings.ToLower(message.Role), lastTrain(messages, index, training)
		if role == "assistant" {
			out.Segments = append(out.Segments, Segment{Text: content(message), Train: train}, Segment{Control: "</s>", Train: train})
			continue
		}
		out.Segments = append(out.Segments, Segment{Control: "<s>"}, Segment{Text: "[INST] " + role + "\n" + content(message) + " [/INST] "})
	}
	return out
}

func renderGemma(messages []Message, training bool) Rendered {
	var out Rendered
	for index, message := range messages {
		train := lastTrain(messages, index, training)
		role := strings.ToLower(message.Role)
		if role == "assistant" {
			role = "model"
		}
		out.Segments = append(out.Segments, Segment{Control: "<start_of_turn>"}, Segment{Text: role + "\n"}, Segment{Text: content(message), Train: train}, Segment{Control: "<end_of_turn>", Train: train}, Segment{Text: "\n", Train: train})
	}
	if !training {
		out.Segments = append(out.Segments, Segment{Control: "<start_of_turn>"}, Segment{Text: "model\n"})
	}
	return out
}

func renderPhi(messages []Message, training bool) Rendered {
	var out Rendered
	for index, message := range messages {
		train := lastTrain(messages, index, training)
		out.Segments = append(out.Segments, Segment{Control: "<|" + strings.ToLower(message.Role) + "|>"}, Segment{Text: "\n"}, Segment{Text: content(message), Train: train}, Segment{Control: "<|end|>", Train: train}, Segment{Text: "\n", Train: train})
	}
	if !training {
		out.Segments = append(out.Segments, Segment{Control: "<|assistant|>"}, Segment{Text: "\n"})
	}
	return out
}

func renderGeneric(messages []Message, training bool) Rendered {
	var out Rendered
	for index, message := range messages {
		train := lastTrain(messages, index, training)
		out.Segments = append(out.Segments, Segment{Text: strings.ToLower(message.Role) + ": "}, Segment{Text: content(message), Train: train}, Segment{Text: "\n", Train: train})
	}
	if !training {
		out.Segments = append(out.Segments, Segment{Text: "assistant: "})
	}
	return out
}

// ResponseTokenMask converts encoded template segments into a loss mask. A
// tokenizer supplies segment token IDs so the mask always aligns with actual
// model tokens rather than byte or character offsets.
func ResponseTokenMask(segments []Segment, encode func(Segment) ([]int, error)) ([]int, []bool, error) {
	var ids []int
	var mask []bool
	for _, segment := range segments {
		encoded, err := encode(segment)
		if err != nil {
			return nil, nil, err
		}
		ids = append(ids, encoded...)
		for range encoded {
			mask = append(mask, segment.Train)
		}
	}
	return ids, mask, nil
}
