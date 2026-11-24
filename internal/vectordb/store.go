// Package vectordb is Sypra's embedded, persistent vector database. It keeps
// embeddings outside workspace JSON and provides isolated collection searches.
package vectordb

import (
	"context"
	"encoding/gob"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
)

type Record struct {
	CollectionID string
	ID           string
	Vector       []float32
}

type Match struct {
	ID    string
	Score float64
}

type DB struct {
	path    string
	mu      sync.RWMutex
	vectors map[string]map[string][]float32
}

type diskState struct {
	Version int
	Vectors map[string]map[string][]float32
}

func Open(path string) (*DB, error) {
	if path == "" {
		return nil, fmt.Errorf("vector database path is required")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return nil, err
	}
	db := &DB{path: path, vectors: map[string]map[string][]float32{}}
	file, err := os.Open(path)
	if os.IsNotExist(err) {
		return db, nil
	}
	if err != nil {
		return nil, err
	}
	defer file.Close()
	var state diskState
	if err := gob.NewDecoder(file).Decode(&state); err != nil {
		return nil, fmt.Errorf("read vector database: %w", err)
	}
	if state.Version != 1 {
		return nil, fmt.Errorf("unsupported vector database version %d", state.Version)
	}
	if state.Vectors != nil {
		db.vectors = state.Vectors
	}
	return db, nil
}

func (db *DB) Upsert(_ context.Context, records []Record) error {
	db.mu.Lock()
	defer db.mu.Unlock()
	for _, record := range records {
		if record.CollectionID == "" || record.ID == "" || len(record.Vector) == 0 {
			return fmt.Errorf("vector records require a collection, id, and values")
		}
		if db.vectors[record.CollectionID] == nil {
			db.vectors[record.CollectionID] = map[string][]float32{}
		}
		db.vectors[record.CollectionID][record.ID] = append([]float32(nil), record.Vector...)
	}
	return db.saveLocked()
}

func (db *DB) DeleteCollection(_ context.Context, collectionID string) error {
	db.mu.Lock()
	defer db.mu.Unlock()
	delete(db.vectors, collectionID)
	return db.saveLocked()
}

// Delete removes individual vectors without disturbing the rest of a
// collection. It is used for document replacement and deletion.
func (db *DB) Delete(_ context.Context, collectionID string, ids []string) error {
	db.mu.Lock()
	defer db.mu.Unlock()
	for _, id := range ids {
		delete(db.vectors[collectionID], id)
	}
	return db.saveLocked()
}

func (db *DB) Reset(_ context.Context) error {
	db.mu.Lock()
	defer db.mu.Unlock()
	db.vectors = map[string]map[string][]float32{}
	return db.saveLocked()
}

func (db *DB) Search(_ context.Context, collectionID string, query []float32, limit int) ([]Match, error) {
	if limit < 0 {
		return nil, fmt.Errorf("result limit cannot be negative")
	}
	db.mu.RLock()
	defer db.mu.RUnlock()
	matches := make([]Match, 0, len(db.vectors[collectionID]))
	for id, vector := range db.vectors[collectionID] {
		matches = append(matches, Match{ID: id, Score: cosine(query, vector)})
	}
	sort.SliceStable(matches, func(i, j int) bool { return matches[i].Score > matches[j].Score })
	if limit > 0 && len(matches) > limit {
		matches = matches[:limit]
	}
	return matches, nil
}

func (db *DB) Count(_ context.Context, collectionID string) int {
	db.mu.RLock()
	defer db.mu.RUnlock()
	return len(db.vectors[collectionID])
}

// Records returns a deep copy suitable for an encrypted/operator-managed
// workspace backup. It intentionally does not expose the mutable map.
func (db *DB) Records(_ context.Context) []Record {
	db.mu.RLock()
	defer db.mu.RUnlock()
	result := make([]Record, 0)
	for collectionID, vectors := range db.vectors {
		for id, vector := range vectors {
			result = append(result, Record{CollectionID: collectionID, ID: id, Vector: append([]float32(nil), vector...)})
		}
	}
	return result
}

// Replace atomically replaces all persisted vector records after the caller
// has validated an import. It is used only by workspace restore.
func (db *DB) Replace(_ context.Context, records []Record) error {
	db.mu.Lock()
	defer db.mu.Unlock()
	next := map[string]map[string][]float32{}
	for _, record := range records {
		if record.CollectionID == "" || record.ID == "" || len(record.Vector) == 0 {
			return fmt.Errorf("vector records require a collection, id, and values")
		}
		if next[record.CollectionID] == nil {
			next[record.CollectionID] = map[string][]float32{}
		}
		next[record.CollectionID][record.ID] = append([]float32(nil), record.Vector...)
	}
	db.vectors = next
	return db.saveLocked()
}

func (db *DB) saveLocked() error {
	temporary := db.path + ".tmp"
	file, err := os.Create(temporary)
	if err != nil {
		return err
	}
	if err := gob.NewEncoder(file).Encode(diskState{Version: 1, Vectors: db.vectors}); err != nil {
		_ = file.Close()
		_ = os.Remove(temporary)
		return err
	}
	if err := file.Close(); err != nil {
		_ = os.Remove(temporary)
		return err
	}
	return os.Rename(temporary, db.path)
}

func cosine(left, right []float32) float64 {
	if len(left) == 0 || len(left) != len(right) {
		return 0
	}
	var dot, leftNorm, rightNorm float64
	for i := range left {
		dot += float64(left[i]) * float64(right[i])
		leftNorm += float64(left[i]) * float64(left[i])
		rightNorm += float64(right[i]) * float64(right[i])
	}
	if leftNorm == 0 || rightNorm == 0 {
		return 0
	}
	return dot / (math.Sqrt(leftNorm) * math.Sqrt(rightNorm))
}
