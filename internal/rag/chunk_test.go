package rag

import (
	"strings"
	"testing"

	"github.com/sypra-labs/sypra-ai/internal/embedding"
)

func TestSplitKeepsCompleteSentences(t *testing.T) {
	text := "Alpha begins here. Beta follows after it. Gamma closes the thought."
	got := split(text, 30, 0)
	want := []string{"Alpha begins here.", "Beta follows after it.", "Gamma closes the thought."}
	if len(got) != len(want) {
		t.Fatalf("chunk count = %d, want %d: %#v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("chunk %d = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestSplitDoesNotCutLongSentence(t *testing.T) {
	text := "This deliberately long sentence must remain whole even when the configured chunk target is smaller."
	got := split(text, 24, 0)
	if len(got) != 1 || got[0] != text {
		t.Fatalf("long sentence was split: %#v", got)
	}
}

func TestSplitOverlapUsesWholeSentences(t *testing.T) {
	text := "One short sentence. Two short sentence. Three short sentence."
	got := split(text, 42, 20)
	for _, chunk := range got {
		if chunk[len(chunk)-1] != '.' {
			t.Fatalf("chunk ends mid-sentence: %q", chunk)
		}
	}
}

func TestSplitTokensUsesTokenizerBudgetAndWholeSentences(t *testing.T) {
	engine, err := embedding.New(embedding.HashModel)
	if err != nil {
		t.Fatal(err)
	}
	text := "One two three. Four five six. Seven eight nine."
	chunks := splitTokens(text, engine, 8, 0)
	if len(chunks) != 2 {
		t.Fatalf("got %d chunks, want 2: %#v", len(chunks), chunks)
	}
	for _, chunk := range chunks {
		if !strings.HasSuffix(chunk, ".") {
			t.Fatalf("chunk split a sentence: %q", chunk)
		}
		if tokens := engine.TokenCount(chunk); tokens > 8 {
			t.Fatalf("chunk has %d tokens, want at most 8: %q", tokens, chunk)
		}
	}
}
