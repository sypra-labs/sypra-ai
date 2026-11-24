package embedding

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
	"unicode"
)

func newMiniLMEngine() (Engine, error) {
	return newFullMiniLMEngine()
}

func defaultMiniLMPath() string {
	if path := strings.TrimSpace(os.Getenv("SYPRA_EMBEDDING_MODEL_PATH")); path != "" {
		return path
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	root := filepath.Join(home, ".cache", "huggingface", "hub", "models--sentence-transformers--all-MiniLM-L6-v2", "snapshots")
	entries, err := os.ReadDir(root)
	if err != nil {
		return ""
	}
	for _, entry := range entries {
		if entry.IsDir() {
			return filepath.Join(root, entry.Name())
		}
	}
	return ""
}

func loadVocab(path string) (map[string]int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	vocab := map[string]int{}
	scanner, id := bufio.NewScanner(file), 0
	for scanner.Scan() {
		vocab[scanner.Text()] = id
		id++
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return vocab, nil
}
func tokenID(vocab map[string]int, token string, fallback int) int {
	if id, ok := vocab[token]; ok {
		return id
	}
	return fallback
}

func miniLMWords(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsNumber(r) })
}
