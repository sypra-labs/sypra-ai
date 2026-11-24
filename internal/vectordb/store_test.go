package vectordb

import (
	"context"
	"path/filepath"
	"testing"
)

func TestPersistentCollectionSearch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vectors.db")
	db, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := db.Upsert(context.Background(), []Record{{CollectionID: "rag-a", ID: "one", Vector: []float32{1, 0}}, {CollectionID: "rag-a", ID: "two", Vector: []float32{0, 1}}, {CollectionID: "rag-b", ID: "other", Vector: []float32{1, 0}}}); err != nil {
		t.Fatal(err)
	}
	db, err = Open(path)
	if err != nil {
		t.Fatal(err)
	}
	matches, err := db.Search(context.Background(), "rag-a", []float32{.9, .1}, 1)
	if err != nil || len(matches) != 1 || matches[0].ID != "one" {
		t.Fatalf("matches=%#v err=%v", matches, err)
	}
}
