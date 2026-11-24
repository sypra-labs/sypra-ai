package embedding

import "testing"

func TestLocalHashEmbeddingsAreNormalizedAndRelevant(t *testing.T) {
	engine, err := New(HashModel)
	if err != nil {
		t.Fatal(err)
	}
	query := engine.Embed("quarterly revenue report")
	related := engine.Embed("revenue report for the quarter")
	unrelated := engine.Embed("mountain bicycle maintenance")
	if len(query) != 384 {
		t.Fatalf("embedding dimensions = %d", len(query))
	}
	if Cosine(query, related) <= Cosine(query, unrelated) {
		t.Fatalf("related text was not closer than unrelated text")
	}
}
