// Package embedding provides offline vectorizers for local retrieval.
package embedding

import (
	"fmt"
	"math"
	"strings"
	"unicode"
)

const (
	// DefaultModel uses the locally cached all-MiniLM-L6-v2 checkpoint.
	DefaultModel = "all-minilm-l6-v2-token-mean"
	HashModel    = "local-hash-384"
)

type Engine struct {
	Dimensions int
	MaxTokens  int
	embed      func(string) []float32
	tokens     func(string) int
}

func New(name string) (Engine, error) {
	if name == "" {
		name = DefaultModel
	}
	switch name {
	case DefaultModel:
		return newMiniLMEngine()
	case HashModel:
		return Engine{Dimensions: 384, MaxTokens: 512, embed: hashEmbed, tokens: func(text string) int { return len(terms(text)) + 2 }}, nil
	default:
		return Engine{}, fmt.Errorf("unsupported local embedding model %q", name)
	}
}

// TokenCount reports encoder tokens, including special tokens. RAG uses it
// for chunk boundaries rather than approximating token capacity by characters.
func (e Engine) TokenCount(text string) int {
	if e.tokens != nil {
		return e.tokens(text)
	}
	return len(terms(text)) + 2
}

// Embed combines whole-word and character-trigram features into a normalized
// dense vector. Whole words favour exact terminology; trigrams make related
// spelling/forms closer than plain keyword counting. It is intentionally a
// local fallback, not a claim of learned semantic understanding.
func (e Engine) Embed(text string) []float32 {
	if e.embed != nil {
		return e.embed(text)
	}
	return hashEmbed(text)
}

func hashEmbed(text string) []float32 {
	vector := make([]float32, 384)
	for _, term := range terms(text) {
		add(vector, "w:"+term, 1)
		bounded := "^" + term + "$"
		runes := []rune(bounded)
		for i := 0; i+3 <= len(runes); i++ {
			add(vector, "g:"+string(runes[i:i+3]), .35)
		}
	}
	normalize(vector)
	return vector
}

func Cosine(left, right []float32) float64 {
	if len(left) == 0 || len(left) != len(right) {
		return 0
	}
	var total float64
	for i := range left {
		total += float64(left[i]) * float64(right[i])
	}
	return total
}

func terms(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsNumber(r) })
}
func add(vector []float32, feature string, weight float32) {
	first := hash(feature, 2166136261)
	second := hash(feature, 16777619)
	index := int(first % uint32(len(vector)))
	if second&1 == 0 {
		vector[index] += weight
	} else {
		vector[index] -= weight
	}
}
func hash(value string, seed uint32) uint32 {
	h := seed
	for _, b := range []byte(value) {
		h ^= uint32(b)
		h *= 16777619
	}
	return h
}
func normalize(vector []float32) {
	var sum float64
	for _, value := range vector {
		sum += float64(value * value)
	}
	if sum == 0 {
		return
	}
	scale := float32(1 / math.Sqrt(sum))
	for i := range vector {
		vector[i] *= scale
	}
}
