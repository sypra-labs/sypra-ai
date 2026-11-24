// Package store provides a repository boundary over durable workspace state.
// It supports atomic JSON compatibility storage plus SQLite and PostgreSQL
// backends selected by configuration.
package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/sypra-labs/sypra-ai/internal/domain"
	"github.com/sypra-labs/sypra-ai/internal/vectordb"
)

type state struct {
	Models          []domain.Model            `json:"models"`
	Collections     []domain.Collection       `json:"collections"`
	Documents       []domain.RAGDocument      `json:"documents"`
	Chunks          []domain.Chunk            `json:"chunks"`
	Evaluations     []domain.RAGEvaluationSet `json:"evaluations"`
	EvaluationRuns  []domain.RAGEvaluationRun `json:"evaluationRuns"`
	MediaJobs       []domain.MediaJob         `json:"mediaJobs"`
	ToolConnections []domain.ToolConnection   `json:"toolConnections"`
	ToolPermissions []domain.ToolPermission   `json:"toolPermissions"`
	AgentRuns       []domain.AgentRun         `json:"agentRuns"`
	Jobs            []domain.TrainingJob      `json:"jobs"`
	Usage           []domain.UsageEvent       `json:"usage"`
	Projects        []domain.Project          `json:"projects"`
	Chats           []domain.Chat             `json:"chats"`
	Messages        []domain.ChatMessage      `json:"messages"`
}
type Store struct {
	path    string
	vectors *vectordb.DB
	backend stateBackend
	mu      sync.RWMutex
	data    state
}

func Open(path string) (*Store, error) {
	return openWithBackend(context.Background(), path, fileBackend{path: path}, false)
}

// OpenConfigured opens the configured durable state backend. SQLite is the
// default single-node backend; PostgreSQL is supported for centrally managed
// workspace state. Existing JSON state is imported once when a SQL backend is
// first initialized.
func OpenConfigured(ctx context.Context, databaseURL, legacyPath string) (*Store, error) {
	backend, err := openStateBackend(ctx, databaseURL)
	if err != nil {
		return nil, err
	}
	store, err := openWithBackend(ctx, legacyPath, backend, true)
	if err != nil {
		_ = backend.Close()
		return nil, err
	}
	return store, nil
}

func openWithBackend(ctx context.Context, path string, backend stateBackend, importLegacy bool) (*Store, error) {
	if path == "" {
		return nil, fmt.Errorf("data path is required")
	}
	if dir := filepath.Dir(path); dir != "." {
		if e := os.MkdirAll(dir, 0755); e != nil {
			return nil, e
		}
	}
	vectors, e := vectordb.Open(filepath.Join(filepath.Dir(path), "vector-db", "vectors.db"))
	if e != nil {
		return nil, e
	}
	s := &Store{path: path, vectors: vectors, backend: backend}
	b, e := backend.Load(ctx)
	if errors.Is(e, errStateNotFound) && importLegacy {
		b, e = os.ReadFile(path)
		if os.IsNotExist(e) {
			e = errStateNotFound
		}
		if e == nil {
			if e = json.Unmarshal(b, &s.data); e != nil {
				return nil, fmt.Errorf("read legacy state: %w", e)
			}
			if e = s.saveLocked(); e != nil {
				return nil, fmt.Errorf("import legacy state: %w", e)
			}
			return s.finishOpen(ctx)
		}
	}
	if errors.Is(e, errStateNotFound) || os.IsNotExist(e) {
		return s, nil
	}
	if e != nil {
		return nil, e
	}
	if e = json.Unmarshal(b, &s.data); e != nil {
		return nil, fmt.Errorf("read state: %w", e)
	}
	return s.finishOpen(ctx)
}

func (s *Store) finishOpen(ctx context.Context) (*Store, error) {
	if e := s.migrateLegacyEmbeddings(ctx); e != nil {
		return nil, e
	}
	if s.migrateLegacyDocumentsLocked() {
		if err := s.saveLocked(); err != nil {
			return nil, err
		}
	}
	if s.ensureDefaultBaseModelLocked() {
		if err := s.saveLocked(); err != nil {
			return nil, err
		}
	}
	return s, nil
}
func (s *Store) Close() error {
	if s.backend == nil {
		return nil
	}
	return s.backend.Close()
}

// Health verifies that the persistence backend is reachable without exposing
// implementation details to callers.
func (s *Store) Health(ctx context.Context) error {
	if s.backend == nil {
		return fmt.Errorf("state backend is not configured")
	}
	return s.backend.Health(ctx)
}

// ResetWorkspaceData removes all user workspace state while preserving base
// model registrations. The runner can therefore remain available after reset.
func (s *Store) ResetWorkspaceData(_ context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.vectors.Reset(context.Background()); err != nil {
		return err
	}
	baseModels := make([]domain.Model, 0, len(s.data.Models))
	for _, model := range s.data.Models {
		// Preserve downloaded local checkpoints, not cloud connection records.
		if model.Kind == "base" && model.Provider == "" {
			baseModels = append(baseModels, model)
		}
	}
	s.data = state{Models: baseModels}
	return s.saveLocked()
}
func (s *Store) saveLocked() error {
	b, e := json.MarshalIndent(s.data, "", "  ")
	if e != nil {
		return e
	}
	return s.backend.Save(context.Background(), b)
}
func (s *Store) AddModel(_ context.Context, m domain.Model) (domain.Model, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if m.Kind == "base" && !hasDefaultBaseModel(s.data.Models) {
		m.Default = true
	}
	s.data.Models = append(s.data.Models, m)
	if err := s.saveLocked(); err != nil {
		return domain.Model{}, err
	}
	return m, nil
}

func hasDefaultBaseModel(models []domain.Model) bool {
	for _, model := range models {
		if model.Kind == "base" && model.Default {
			return true
		}
	}
	return false
}

func (s *Store) ensureDefaultBaseModelLocked() bool {
	if hasDefaultBaseModel(s.data.Models) {
		return false
	}
	for index := range s.data.Models {
		if s.data.Models[index].Kind == "base" {
			s.data.Models[index].Default = true
			return true
		}
	}
	return false
}

// DefaultBaseModel reports the workspace default base model, if one exists.
func (s *Store) DefaultBaseModel(_ context.Context) (domain.Model, bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, model := range s.data.Models {
		if model.Kind == "base" && model.Default {
			return model, true, nil
		}
	}
	return domain.Model{}, false, nil
}

// SetDefaultBaseModel switches the workspace's implicit model for new chats.
func (s *Store) SetDefaultBaseModel(_ context.Context, modelID string) (domain.Model, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var selected *domain.Model
	for index := range s.data.Models {
		model := &s.data.Models[index]
		if model.ID != modelID {
			continue
		}
		if model.Kind != "base" {
			return domain.Model{}, fmt.Errorf("only a base model can be the workspace default")
		}
		selected = model
		break
	}
	if selected == nil {
		return domain.Model{}, fmt.Errorf("model %s not found", modelID)
	}
	for index := range s.data.Models {
		if s.data.Models[index].Kind == "base" {
			s.data.Models[index].Default = s.data.Models[index].ID == selected.ID
		}
	}
	if err := s.saveLocked(); err != nil {
		return domain.Model{}, err
	}
	return *selected, nil
}

// PromoteModel marks one locally trained SLM as production. A previous
// production SLM for the same parent checkpoint remains available as a
// rollback candidate rather than being deleted.
func (s *Store) PromoteModel(_ context.Context, modelID string) (domain.Model, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var promoted *domain.Model
	for index := range s.data.Models {
		model := &s.data.Models[index]
		if model.ID == modelID {
			if model.Kind != "slm" {
				return domain.Model{}, fmt.Errorf("only an SLM can be promoted")
			}
			promoted = model
			break
		}
	}
	if promoted == nil {
		return domain.Model{}, fmt.Errorf("model %s not found", modelID)
	}
	for index := range s.data.Models {
		model := &s.data.Models[index]
		if model.ID != promoted.ID && model.Kind == "slm" && model.ParentModel == promoted.ParentModel && model.Status == "production" {
			model.Status = "archived"
		}
	}
	now := Now()
	promoted.Status, promoted.PromotedAt = "production", &now
	if err := s.saveLocked(); err != nil {
		return domain.Model{}, err
	}
	return *promoted, nil
}
func (s *Store) Models(_ context.Context) ([]domain.Model, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]domain.Model, len(s.data.Models))
	copy(out, s.data.Models)
	return out, nil
}
func (s *Store) DeleteModel(_ context.Context, modelID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	kept := s.data.Models[:0]
	for _, model := range s.data.Models {
		if model.ID != modelID {
			kept = append(kept, model)
		}
	}
	if len(kept) == len(s.data.Models) {
		return fmt.Errorf("model %s not found", modelID)
	}
	s.data.Models = kept
	s.ensureDefaultBaseModelLocked()
	for i := range s.data.Chats {
		if s.data.Chats[i].ModelID == modelID {
			s.data.Chats[i].ModelID = ""
		}
	}
	return s.saveLocked()
}
func (s *Store) AddCollection(_ context.Context, c domain.Collection, chunks []domain.Chunk) error {
	return s.AddCollectionDocuments(context.Background(), c, nil, chunks)
}

// AddCollectionDocuments persists the source records alongside the chunks in
// a single workspace update. Vectors are intentionally stored separately.
func (s *Store) AddCollectionDocuments(_ context.Context, c domain.Collection, documents []domain.RAGDocument, chunks []domain.Chunk) error {
	if err := s.upsertVectors(context.Background(), chunks); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data.Collections = append(s.data.Collections, c)
	s.data.Documents = append(s.data.Documents, documents...)
	s.data.Chunks = append(s.data.Chunks, chunks...)
	return s.saveLocked()
}
func (s *Store) Collections(_ context.Context) ([]domain.Collection, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]domain.Collection, len(s.data.Collections))
	copy(out, s.data.Collections)
	for i := range out {
		count := 0
		for _, document := range s.data.Documents {
			if document.CollectionID == out[i].ID {
				count++
			}
		}
		if count == 0 { // legacy safety during the one-time migration.
			sources := map[string]struct{}{}
			for _, k := range s.data.Chunks {
				if k.CollectionID == out[i].ID {
					sources[k.Source] = struct{}{}
				}
			}
			count = len(sources)
		}
		out[i].DocumentCount = count
	}
	return out, nil
}
func (s *Store) AddCollectionChunks(_ context.Context, collectionID string, chunks []domain.Chunk) error {
	if err := s.upsertVectors(context.Background(), chunks); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	found := false
	for _, collection := range s.data.Collections {
		if collection.ID == collectionID {
			found = true
			break
		}
	}
	if !found {
		return fmt.Errorf("collection %s not found", collectionID)
	}
	s.data.Chunks = append(s.data.Chunks, chunks...)
	return s.saveLocked()
}

// AddDocuments appends durable document records and their chunks to an
// existing collection.
func (s *Store) AddDocuments(_ context.Context, collectionID string, documents []domain.RAGDocument, chunks []domain.Chunk) error {
	if err := s.upsertVectors(context.Background(), chunks); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.hasCollectionLocked(collectionID) {
		return fmt.Errorf("collection %s not found", collectionID)
	}
	s.data.Documents = append(s.data.Documents, documents...)
	s.data.Chunks = append(s.data.Chunks, chunks...)
	return s.saveLocked()
}

func (s *Store) hasCollectionLocked(collectionID string) bool {
	for _, collection := range s.data.Collections {
		if collection.ID == collectionID {
			return true
		}
	}
	return false
}

func (s *Store) Documents(_ context.Context, collectionID string) ([]domain.RAGDocument, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]domain.RAGDocument, 0)
	for _, document := range s.data.Documents {
		if collectionID == "" || document.CollectionID == collectionID {
			out = append(out, cloneDocument(document))
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].CreatedAt.Before(out[j].CreatedAt) })
	return out, nil
}

func cloneDocument(document domain.RAGDocument) domain.RAGDocument {
	if document.Metadata != nil {
		document.Metadata = mapsClone(document.Metadata)
	}
	return document
}

func mapsClone(values map[string]string) map[string]string {
	copy := make(map[string]string, len(values))
	for key, value := range values {
		copy[key] = value
	}
	return copy
}

func (s *Store) Document(_ context.Context, documentID string) (domain.RAGDocument, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, document := range s.data.Documents {
		if document.ID == documentID {
			return cloneDocument(document), nil
		}
	}
	return domain.RAGDocument{}, fmt.Errorf("document %s not found", documentID)
}

func (s *Store) RenameDocument(_ context.Context, documentID, name string) (domain.RAGDocument, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return domain.RAGDocument{}, fmt.Errorf("document name is required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for index := range s.data.Documents {
		if s.data.Documents[index].ID != documentID {
			continue
		}
		s.data.Documents[index].Name = name
		s.data.Documents[index].UpdatedAt = Now()
		for chunkIndex := range s.data.Chunks {
			if s.data.Chunks[chunkIndex].DocumentID == documentID {
				s.data.Chunks[chunkIndex].Source = name
			}
		}
		if err := s.saveLocked(); err != nil {
			return domain.RAGDocument{}, err
		}
		return cloneDocument(s.data.Documents[index]), nil
	}
	return domain.RAGDocument{}, fmt.Errorf("document %s not found", documentID)
}

// ReplaceDocument atomically swaps a document's chunks after extraction and
// embedding have completed. The old vectors are deleted only after new vectors
// are durable, avoiding an empty index if indexing is interrupted.
func (s *Store) ReplaceDocument(_ context.Context, document domain.RAGDocument, chunks []domain.Chunk) error {
	if err := s.upsertVectors(context.Background(), chunks); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	oldIDs := make([]string, 0)
	updated := false
	for index := range s.data.Documents {
		if s.data.Documents[index].ID == document.ID {
			s.data.Documents[index] = document
			updated = true
			break
		}
	}
	if !updated {
		return fmt.Errorf("document %s not found", document.ID)
	}
	kept := s.data.Chunks[:0]
	for _, chunk := range s.data.Chunks {
		if chunk.DocumentID == document.ID {
			oldIDs = append(oldIDs, chunk.ID)
			continue
		}
		kept = append(kept, chunk)
	}
	s.data.Chunks = append(kept, chunks...)
	if err := s.vectors.Delete(context.Background(), document.CollectionID, oldIDs); err != nil {
		return err
	}
	return s.saveLocked()
}

func (s *Store) DeleteDocument(_ context.Context, documentID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	var collectionID string
	documents := s.data.Documents[:0]
	found := false
	for _, document := range s.data.Documents {
		if document.ID == documentID {
			collectionID, found = document.CollectionID, true
			continue
		}
		documents = append(documents, document)
	}
	if !found {
		return fmt.Errorf("document %s not found", documentID)
	}
	ids := make([]string, 0)
	chunks := s.data.Chunks[:0]
	for _, chunk := range s.data.Chunks {
		if chunk.DocumentID == documentID {
			ids = append(ids, chunk.ID)
			continue
		}
		chunks = append(chunks, chunk)
	}
	if err := s.vectors.Delete(context.Background(), collectionID, ids); err != nil {
		return err
	}
	s.data.Documents, s.data.Chunks = documents, chunks
	return s.saveLocked()
}
func (s *Store) DeleteCollection(_ context.Context, collectionID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	keptCollections := s.data.Collections[:0]
	for _, collection := range s.data.Collections {
		if collection.ID != collectionID {
			keptCollections = append(keptCollections, collection)
		}
	}
	if len(keptCollections) == len(s.data.Collections) {
		return fmt.Errorf("collection %s not found", collectionID)
	}
	if err := s.vectors.DeleteCollection(context.Background(), collectionID); err != nil {
		return err
	}
	s.data.Collections = keptCollections
	keptDocuments := s.data.Documents[:0]
	for _, document := range s.data.Documents {
		if document.CollectionID != collectionID {
			keptDocuments = append(keptDocuments, document)
		}
	}
	s.data.Documents = keptDocuments
	keptChunks := s.data.Chunks[:0]
	for _, chunk := range s.data.Chunks {
		if chunk.CollectionID != collectionID {
			keptChunks = append(keptChunks, chunk)
		}
	}
	s.data.Chunks = keptChunks
	for i := range s.data.Chats {
		if s.data.Chats[i].CollectionID == collectionID {
			s.data.Chats[i].CollectionID = ""
		}
	}
	return s.saveLocked()
}
func (s *Store) SearchChunks(_ context.Context, collectionID, query string, limit int) ([]domain.Chunk, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if limit == 0 {
		out := []domain.Chunk{}
		for _, c := range s.data.Chunks {
			if c.CollectionID == collectionID {
				out = append(out, c)
			}
		}
		return out, nil
	}
	terms := strings.FieldsFunc(strings.ToLower(query), func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsNumber(r) })
	type ranked struct {
		chunk domain.Chunk
		score int
	}
	rankedChunks := []ranked{}
	for _, c := range s.data.Chunks {
		if c.CollectionID != collectionID {
			continue
		}
		content, score := strings.ToLower(c.Content), 0
		for _, term := range terms {
			score += strings.Count(content, term)
		}
		rankedChunks = append(rankedChunks, ranked{chunk: c, score: score})
	}
	sort.SliceStable(rankedChunks, func(i, j int) bool { return rankedChunks[i].score > rankedChunks[j].score })
	out := make([]domain.Chunk, 0, len(rankedChunks))
	for _, item := range rankedChunks {
		if limit > 0 && len(out) >= limit {
			break
		}
		out = append(out, item.chunk)
	}
	return out, nil
}

// SearchVectorChunks queries the persistent embedded vector database, then
// applies a small lexical tie-breaker from the JSON-backed chunk metadata.
func (s *Store) SearchVectorChunks(_ context.Context, collectionID, query string, vector []float32, limit int) ([]domain.Chunk, error) {
	matches, err := s.SearchVectorChunkMatches(context.Background(), collectionID, query, vector, limit, nil)
	if err != nil {
		return nil, err
	}
	out := make([]domain.Chunk, len(matches))
	for index, match := range matches {
		out[index] = match.Chunk
	}
	return out, nil
}

// SearchVectorChunkMatches exposes the hybrid score components used to rank
// evidence. Metadata filters are exact AND filters; a key prefixed with
// "documentId" matches the durable source identity.
func (s *Store) SearchVectorChunkMatches(_ context.Context, collectionID, query string, vector []float32, limit int, filters map[string]string) ([]domain.ChunkMatch, error) {
	if limit < 0 {
		return nil, fmt.Errorf("result limit cannot be negative")
	}
	matches, err := s.vectors.Search(context.Background(), collectionID, vector, 0)
	if err != nil {
		return nil, err
	}
	semanticScores := make(map[string]float64, len(matches))
	for _, match := range matches {
		semanticScores[match.ID] = match.Score
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	terms := strings.FieldsFunc(strings.ToLower(query), func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsNumber(r) })
	// BM25 supplies the exact-term signal which dense embeddings alone often
	// miss (identifiers, product names, dates and numbers). It is deliberately
	// a secondary, normalized score; semantic similarity remains primary.
	type documentStats struct {
		chunk domain.Chunk
		terms []string
	}
	documents := make([]documentStats, 0)
	documentFrequency := map[string]int{}
	totalLength := 0
	for _, chunk := range s.data.Chunks {
		if chunk.CollectionID != collectionID {
			continue
		}
		chunkTerms := strings.FieldsFunc(strings.ToLower(chunk.Content), func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsNumber(r) })
		seen := map[string]bool{}
		for _, term := range chunkTerms {
			if !seen[term] {
				documentFrequency[term]++
				seen[term] = true
			}
		}
		totalLength += len(chunkTerms)
		documents = append(documents, documentStats{chunk: chunk, terms: chunkTerms})
	}
	averageLength := float64(totalLength) / math.Max(1, float64(len(documents)))
	type ranked struct {
		match domain.ChunkMatch
	}
	rankedChunks := make([]ranked, 0, len(documents))
	for _, document := range documents {
		chunk := document.chunk
		if !matchesMetadata(chunk, filters) {
			continue
		}
		semantic := semanticScores[chunk.ID]
		frequencies := map[string]int{}
		for _, term := range document.terms {
			frequencies[term]++
		}
		var lexical float64
		for _, term := range terms {
			frequency := float64(frequencies[term])
			if frequency == 0 {
				continue
			}
			idf := math.Log(1 + (float64(len(documents))-float64(documentFrequency[term])+0.5)/(float64(documentFrequency[term])+0.5))
			lengthNorm := frequency + 1.2*(1-.75+.75*float64(len(document.terms))/averageLength)
			lexical += idf * frequency * 2.2 / lengthNorm
		}
		// Keep exact matching useful while ensuring vector similarity remains the
		// primary rank signal. tanh prevents a long keyword-stuffed chunk from
		// dominating a genuinely semantic match.
		lexicalScore := math.Tanh(lexical)
		rankedChunks = append(rankedChunks, ranked{match: domain.ChunkMatch{Chunk: chunk, SemanticScore: semantic, LexicalScore: lexicalScore, Score: semantic + .18*lexicalScore}})
	}
	sort.SliceStable(rankedChunks, func(i, j int) bool { return rankedChunks[i].match.Score > rankedChunks[j].match.Score })
	out := make([]domain.ChunkMatch, 0, len(rankedChunks))
	for _, item := range rankedChunks {
		if limit > 0 && len(out) >= limit {
			break
		}
		out = append(out, item.match)
	}
	return out, nil
}

func matchesMetadata(chunk domain.Chunk, filters map[string]string) bool {
	for key, want := range filters {
		if key == "documentId" {
			if chunk.DocumentID != want {
				return false
			}
			continue
		}
		if chunk.Metadata == nil || chunk.Metadata[key] != want {
			return false
		}
	}
	return true
}

func (s *Store) CollectionChunks(_ context.Context, collectionID string) ([]domain.Chunk, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]domain.Chunk, 0)
	for _, chunk := range s.data.Chunks {
		if chunk.CollectionID == collectionID {
			out = append(out, chunk)
		}
	}
	return out, nil
}

func cosine(left, right []float32) float64 {
	if len(left) == 0 || len(left) != len(right) {
		return 0
	}
	var total float64
	for i := range left {
		total += float64(left[i]) * float64(right[i])
	}
	return total
}

func (s *Store) VectorCount(ctx context.Context, collectionID string) int {
	return s.vectors.Count(ctx, collectionID)
}

func (s *Store) AddEvaluationSet(_ context.Context, set domain.RAGEvaluationSet) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data.Evaluations = append(s.data.Evaluations, set)
	return s.saveLocked()
}

func (s *Store) EvaluationSets(_ context.Context, collectionID string) ([]domain.RAGEvaluationSet, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	sets := make([]domain.RAGEvaluationSet, 0)
	for _, set := range s.data.Evaluations {
		if collectionID == "" || set.CollectionID == collectionID {
			sets = append(sets, set)
		}
	}
	return sets, nil
}

func (s *Store) EvaluationSet(_ context.Context, id string) (domain.RAGEvaluationSet, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, set := range s.data.Evaluations {
		if set.ID == id {
			return set, nil
		}
	}
	return domain.RAGEvaluationSet{}, fmt.Errorf("evaluation set %s not found", id)
}

func (s *Store) AddEvaluationRun(_ context.Context, run domain.RAGEvaluationRun) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data.EvaluationRuns = append(s.data.EvaluationRuns, run)
	return s.saveLocked()
}

func (s *Store) AddMediaJob(_ context.Context, job domain.MediaJob) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data.MediaJobs = append(s.data.MediaJobs, job)
	return s.saveLocked()
}

func (s *Store) UpdateMediaJob(_ context.Context, job domain.MediaJob) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for index := range s.data.MediaJobs {
		if s.data.MediaJobs[index].ID == job.ID {
			s.data.MediaJobs[index] = job
			return s.saveLocked()
		}
	}
	return fmt.Errorf("media job %s not found", job.ID)
}

func (s *Store) MediaJobs(_ context.Context) ([]domain.MediaJob, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	jobs := append([]domain.MediaJob(nil), s.data.MediaJobs...)
	sort.SliceStable(jobs, func(i, j int) bool { return jobs[i].CreatedAt.After(jobs[j].CreatedAt) })
	return jobs, nil
}

func (s *Store) MediaJob(_ context.Context, id string) (domain.MediaJob, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, job := range s.data.MediaJobs {
		if job.ID == id {
			return job, nil
		}
	}
	return domain.MediaJob{}, fmt.Errorf("media job %s not found", id)
}

func (s *Store) AddToolConnection(_ context.Context, connection domain.ToolConnection) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data.ToolConnections = append(s.data.ToolConnections, connection)
	return s.saveLocked()
}

func (s *Store) ToolConnections(_ context.Context) ([]domain.ToolConnection, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return append([]domain.ToolConnection(nil), s.data.ToolConnections...), nil
}

func (s *Store) UpdateToolConnection(_ context.Context, connection domain.ToolConnection) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for index := range s.data.ToolConnections {
		if s.data.ToolConnections[index].ID == connection.ID {
			s.data.ToolConnections[index] = connection
			return s.saveLocked()
		}
	}
	return fmt.Errorf("tool connection %s not found", connection.ID)
}

func (s *Store) DeleteToolConnection(_ context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	kept := s.data.ToolConnections[:0]
	for _, connection := range s.data.ToolConnections {
		if connection.ID != id {
			kept = append(kept, connection)
		}
	}
	if len(kept) == len(s.data.ToolConnections) {
		return fmt.Errorf("tool connection %s not found", id)
	}
	s.data.ToolConnections = kept
	return s.saveLocked()
}

func (s *Store) AddToolPermission(_ context.Context, permission domain.ToolPermission) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data.ToolPermissions = append(s.data.ToolPermissions, permission)
	return s.saveLocked()
}

func (s *Store) ToolPermissions(_ context.Context) ([]domain.ToolPermission, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return append([]domain.ToolPermission(nil), s.data.ToolPermissions...), nil
}

func (s *Store) AddAgentRun(_ context.Context, run domain.AgentRun) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data.AgentRuns = append(s.data.AgentRuns, run)
	return s.saveLocked()
}

func (s *Store) UpdateAgentRun(_ context.Context, run domain.AgentRun) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for index := range s.data.AgentRuns {
		if s.data.AgentRuns[index].ID == run.ID {
			s.data.AgentRuns[index] = run
			return s.saveLocked()
		}
	}
	return fmt.Errorf("agent run %s not found", run.ID)
}

func (s *Store) AgentRuns(_ context.Context) ([]domain.AgentRun, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	runs := append([]domain.AgentRun(nil), s.data.AgentRuns...)
	sort.SliceStable(runs, func(i, j int) bool { return runs[i].CreatedAt.After(runs[j].CreatedAt) })
	return runs, nil
}

func (s *Store) EvaluationRuns(_ context.Context, evaluationID string) ([]domain.RAGEvaluationRun, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	runs := make([]domain.RAGEvaluationRun, 0)
	for _, run := range s.data.EvaluationRuns {
		if evaluationID == "" || run.EvaluationID == evaluationID {
			runs = append(runs, run)
		}
	}
	sort.SliceStable(runs, func(i, j int) bool { return runs[i].CreatedAt.After(runs[j].CreatedAt) })
	return runs, nil
}

func (s *Store) upsertVectors(ctx context.Context, chunks []domain.Chunk) error {
	records := make([]vectordb.Record, 0, len(chunks))
	for i := range chunks {
		if len(chunks[i].Embedding) > 0 {
			records = append(records, vectordb.Record{CollectionID: chunks[i].CollectionID, ID: chunks[i].ID, Vector: chunks[i].Embedding})
		}
		chunks[i].Embedding = nil
	}
	if len(records) == 0 {
		return nil
	}
	return s.vectors.Upsert(ctx, records)
}

func (s *Store) migrateLegacyEmbeddings(ctx context.Context) error {
	hasLegacyEmbeddings := false
	for _, chunk := range s.data.Chunks {
		if len(chunk.Embedding) > 0 {
			hasLegacyEmbeddings = true
			break
		}
	}
	if !hasLegacyEmbeddings {
		return nil
	}
	if err := s.upsertVectors(ctx, s.data.Chunks); err != nil {
		return err
	}
	// Embeddings were decoded from older workspace JSON. Rewrite metadata
	// without them after their one-time vector database migration.
	return s.saveLocked()
}

// migrateLegacyDocuments supplies a stable source document for older JSON
// workspaces that predate document IDs. It is deliberately idempotent.
func (s *Store) migrateLegacyDocumentsLocked() bool {
	known := map[string]bool{}
	for _, document := range s.data.Documents {
		known[document.ID] = true
	}
	bySource := map[string]string{}
	changed := false
	for index := range s.data.Chunks {
		chunk := &s.data.Chunks[index]
		if chunk.DocumentID != "" {
			continue
		}
		key := chunk.CollectionID + "\x00" + chunk.Source
		documentID := bySource[key]
		if documentID == "" {
			documentID = fmt.Sprintf("legacy-%x", len(s.data.Documents)+1)
			for known[documentID] {
				documentID += "x"
			}
			known[documentID] = true
			metadata := mapsClone(chunk.Metadata)
			status := "complete"
			if metadata != nil && metadata["extractionStatus"] != "" {
				status = metadata["extractionStatus"]
			}
			now := Now()
			s.data.Documents = append(s.data.Documents, domain.RAGDocument{ID: documentID, CollectionID: chunk.CollectionID, Name: chunk.Source, Source: chunk.Source, Path: metadata["path"], ContentType: chunk.ContentType, Metadata: metadata, ExtractionStatus: status, ExtractionError: metadata["extractionError"], Extractor: metadata["extractor"], CreatedAt: now, UpdatedAt: now})
			bySource[key] = documentID
		}
		chunk.DocumentID = documentID
		changed = true
	}
	if !changed {
		return false
	}
	counts := map[string]int{}
	for _, chunk := range s.data.Chunks {
		counts[chunk.DocumentID]++
	}
	for index := range s.data.Documents {
		s.data.Documents[index].ChunkCount = counts[s.data.Documents[index].ID]
	}
	return true
}
func (s *Store) AddTrainingJob(_ context.Context, j domain.TrainingJob) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data.Jobs = append(s.data.Jobs, j)
	return s.saveLocked()
}
func (s *Store) Jobs(_ context.Context) ([]domain.TrainingJob, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]domain.TrainingJob, len(s.data.Jobs))
	copy(out, s.data.Jobs)
	return out, nil
}
func (s *Store) UpdateJob(_ context.Context, j domain.TrainingJob) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.data.Jobs {
		if s.data.Jobs[i].ID == j.ID {
			s.data.Jobs[i] = j
			return s.saveLocked()
		}
	}
	return fmt.Errorf("job %s not found", j.ID)
}
func (s *Store) AddUsage(_ context.Context, u domain.UsageEvent) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data.Usage = append(s.data.Usage, u)
	return s.saveLocked()
}
func (s *Store) Summary(_ context.Context) (domain.UsageSummary, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	x := domain.UsageSummary{TrainingJobs: len(s.data.Jobs)}
	var ms int64
	for _, u := range s.data.Usage {
		x.Requests++
		x.InputTokens += u.InputTokens
		x.OutputTokens += u.OutputTokens
		ms += u.LatencyMS
	}
	if x.Requests > 0 {
		x.AverageLatencyMS = float64(ms) / float64(x.Requests)
	}
	return x, nil
}
func (s *Store) AddProject(_ context.Context, p domain.Project) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data.Projects = append(s.data.Projects, p)
	return s.saveLocked()
}
func (s *Store) Projects(_ context.Context) ([]domain.Project, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]domain.Project, len(s.data.Projects))
	copy(out, s.data.Projects)
	return out, nil
}
func (s *Store) AddChat(_ context.Context, c domain.Chat) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data.Chats = append(s.data.Chats, c)
	return s.saveLocked()
}
func (s *Store) Chats(_ context.Context, projectID, query string) ([]domain.Chat, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := []domain.Chat{}
	q := strings.ToLower(query)
	for _, c := range s.data.Chats {
		if projectID != "" && c.ProjectID != projectID {
			continue
		}
		if q != "" && !strings.Contains(strings.ToLower(c.Title), q) {
			continue
		}
		out = append(out, c)
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Pinned != out[j].Pinned {
			return out[i].Pinned
		}
		return out[i].UpdatedAt.After(out[j].UpdatedAt)
	})
	return out, nil
}
func (s *Store) DeleteChat(_ context.Context, chatID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	keptChats := s.data.Chats[:0]
	for _, chat := range s.data.Chats {
		if chat.ID != chatID {
			keptChats = append(keptChats, chat)
		}
	}
	if len(keptChats) == len(s.data.Chats) {
		return fmt.Errorf("chat %s not found", chatID)
	}
	s.data.Chats = keptChats
	keptMessages := s.data.Messages[:0]
	for _, message := range s.data.Messages {
		if message.ChatID != chatID {
			keptMessages = append(keptMessages, message)
		}
	}
	s.data.Messages = keptMessages
	return s.saveLocked()
}
func (s *Store) Chat(_ context.Context, id string) (domain.Chat, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, c := range s.data.Chats {
		if c.ID == id {
			return c, nil
		}
	}
	return domain.Chat{}, fmt.Errorf("chat %s not found", id)
}
func (s *Store) UpdateChat(_ context.Context, c domain.Chat) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.data.Chats {
		if s.data.Chats[i].ID == c.ID {
			s.data.Chats[i] = c
			return s.saveLocked()
		}
	}
	return fmt.Errorf("chat %s not found", c.ID)
}
func (s *Store) AddMessage(_ context.Context, m domain.ChatMessage) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data.Messages = append(s.data.Messages, m)
	return s.saveLocked()
}
func (s *Store) UpdateMessage(_ context.Context, m domain.ChatMessage) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.data.Messages {
		if s.data.Messages[i].ID == m.ID {
			s.data.Messages[i] = m
			return s.saveLocked()
		}
	}
	return fmt.Errorf("message %s not found", m.ID)
}
func (s *Store) Messages(_ context.Context, chatID string) ([]domain.ChatMessage, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := []domain.ChatMessage{}
	for _, m := range s.data.Messages {
		if m.ChatID == chatID {
			out = append(out, m)
		}
	}
	return out, nil
}
func Now() time.Time { return time.Now().UTC() }
