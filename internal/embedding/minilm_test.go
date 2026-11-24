package embedding

import "testing"

func TestMiniLMLocalCheckpointEmbedsText(t *testing.T) {
	engine, err := New(DefaultModel)
	if err != nil {
		t.Fatal(err)
	}
	vector := engine.Embed("local document retrieval with semantic embeddings")
	if len(vector) != 384 {
		t.Fatalf("MiniLM vector dimensions = %d", len(vector))
	}
	if Cosine(vector, vector) < .99 {
		t.Fatalf("MiniLM vector was not normalized")
	}
}
