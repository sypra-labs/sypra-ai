package qwen

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"unicode"
)

// Tokenizer is a compact byte-level BPE implementation for the Qwen2/Qwen3
// tokenizer files distributed with the checkpoint. It intentionally has no
// Python or native runtime dependency.
type Tokenizer struct {
	encoder    map[string]int
	decoder    map[int]string
	merges     map[string]int
	byteToRune [256]rune
	runeToByte map[rune]byte
	special    map[string]int
}

func LoadTokenizer(modelDir string) (*Tokenizer, error) {
	vocabData, err := os.ReadFile(modelDir + string(os.PathSeparator) + "vocab.json")
	if err != nil {
		return nil, err
	}
	encoder := map[string]int{}
	if err = json.Unmarshal(vocabData, &encoder); err != nil {
		return nil, fmt.Errorf("read Qwen vocabulary: %w", err)
	}
	mergesData, err := os.ReadFile(modelDir + string(os.PathSeparator) + "merges.txt")
	if err != nil {
		return nil, err
	}
	merges := map[string]int{}
	lineNo := 0
	for _, line := range strings.Split(string(mergesData), "\n") {
		lineNo++
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Split(line, " ")
		if len(parts) != 2 {
			continue
		}
		merges[parts[0]+"\x00"+parts[1]] = lineNo
	}
	t := &Tokenizer{encoder: encoder, decoder: map[int]string{}, merges: merges, runeToByte: map[rune]byte{}, special: map[string]int{"<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645, "<think>": 151667, "</think>": 151668}}
	for token, id := range encoder {
		t.decoder[id] = token
	}
	t.initByteMap()
	return t, nil
}

func (t *Tokenizer) initByteMap() {
	kept := map[int]bool{}
	for i := 33; i <= 126; i++ {
		kept[i] = true
	}
	for i := 161; i <= 172; i++ {
		kept[i] = true
	}
	for i := 174; i <= 255; i++ {
		kept[i] = true
	}
	next := 0
	for value := 0; value < 256; value++ {
		code := value
		if !kept[value] {
			code, next = 256+next, next+1
		}
		t.byteToRune[value] = rune(code)
		t.runeToByte[rune(code)] = byte(value)
	}
}

func (t *Tokenizer) Encode(text string) ([]int, error) {
	var ids []int
	for _, piece := range preTokens(text) {
		for _, token := range t.bpe(piece) {
			id, ok := t.encoder[token]
			if !ok {
				return nil, fmt.Errorf("Qwen vocabulary has no token %q", token)
			}
			ids = append(ids, id)
		}
	}
	return ids, nil
}

func (t *Tokenizer) EncodeSpecial(value string) (int, bool) {
	id, ok := t.special[value]
	return id, ok
}

func (t *Tokenizer) Decode(ids []int) string {
	var data []byte
	for _, id := range ids {
		value, ok := t.decoder[id]
		if !ok {
			continue
		}
		for _, r := range value {
			if b, exists := t.runeToByte[r]; exists {
				data = append(data, b)
			} else {
				data = append(data, string(r)...)
			}
		}
	}
	return string(data)
}

func (t *Tokenizer) bpe(piece string) []string {
	bytes := []byte(piece)
	parts := make([]string, len(bytes))
	for i, b := range bytes {
		parts[i] = string(t.byteToRune[b])
	}
	for len(parts) > 1 {
		index, rank := -1, int(^uint(0)>>1)
		for i := 0; i < len(parts)-1; i++ {
			if r, ok := t.merges[parts[i]+"\x00"+parts[i+1]]; ok && r < rank {
				index, rank = i, r
			}
		}
		if index < 0 {
			break
		}
		parts[index] += parts[index+1]
		parts = append(parts[:index+1], parts[index+2:]...)
	}
	return parts
}

// Qwen's full tokenizer regex is optimized for multilingual text. This
// standard-library version preserves all bytes and follows the same broad
// word/number/punctuation boundaries, which is sufficient for local chat.
func preTokens(text string) []string {
	var pieces []string
	runes := []rune(text)
	for i := 0; i < len(runes); {
		start := i
		space := unicode.IsSpace(runes[i])
		letter := unicode.IsLetter(runes[i])
		digit := unicode.IsNumber(runes[i])
		i++
		for i < len(runes) {
			next := runes[i]
			if space != unicode.IsSpace(next) {
				break
			}
			if !space && ((letter && !unicode.IsLetter(next)) || (digit && !unicode.IsNumber(next)) || (!letter && !digit && (unicode.IsLetter(next) || unicode.IsNumber(next)))) {
				break
			}
			i++
		}
		pieces = append(pieces, string(runes[start:i]))
	}
	return pieces
}

func (t *Tokenizer) VocabularySize() int { return len(t.encoder) }

func (t *Tokenizer) TokenIDs() []int {
	ids := make([]int, 0, len(t.decoder))
	for id := range t.decoder {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	return ids
}
