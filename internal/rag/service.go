package rag

import (
	"context"
	"crypto/rand"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/sypra-labs/sypra-ai/internal/domain"
	"github.com/sypra-labs/sypra-ai/internal/embedding"
	"github.com/sypra-labs/sypra-ai/internal/store"
)

type CreateInput struct {
	Name, EmbeddingModel    string
	ChunkSize, ChunkOverlap int
	Documents               []Document
	RoutingTags             []string
}
type Document struct {
	Source, Text, Path, Folder, URL, ContentType string
	RefreshIntervalMinutes                       int
	Metadata                                     map[string]string
}
type Service struct{ Store *store.Store }

// RetrievalOptions allows callers to make retrieval explicit and observable.
// Metadata is an exact AND filter. Rerank is local and dependency-free: it
// strengthens passages that cover more of the query terms after hybrid search.
type RetrievalOptions struct {
	Limit            int               `json:"limit"`
	Metadata         map[string]string `json:"metadata,omitempty"`
	Rerank           bool              `json:"rerank"`
	IncludeNeighbors bool              `json:"includeNeighbors"`
}

type RetrievalResult struct {
	Chunk         domain.Chunk        `json:"chunk"`
	Score         float64             `json:"score"`
	SemanticScore float64             `json:"semanticScore"`
	LexicalScore  float64             `json:"lexicalScore"`
	RerankScore   float64             `json:"rerankScore,omitempty"`
	Previous      *domain.Chunk       `json:"previous,omitempty"`
	Next          *domain.Chunk       `json:"next,omitempty"`
	Document      *domain.RAGDocument `json:"document,omitempty"`
	Location      string              `json:"location,omitempty"`
}

func (s Service) Create(ctx context.Context, in CreateInput) (domain.Collection, error) {
	if in.Name == "" {
		return domain.Collection{}, fmt.Errorf("collection name is required")
	}
	if in.ChunkSize == 0 {
		in.ChunkSize = 192
	}
	if in.ChunkOverlap == 0 {
		in.ChunkOverlap = 32
	}
	if in.ChunkSize < 16 || in.ChunkOverlap >= in.ChunkSize {
		return domain.Collection{}, fmt.Errorf("chunk size must be at least 16 tokens and overlap smaller than size")
	}
	if in.EmbeddingModel == "" {
		in.EmbeddingModel = embedding.DefaultModel
	}
	engine, err := embedding.New(in.EmbeddingModel)
	if err != nil {
		return domain.Collection{}, err
	}
	id := newID()
	c := domain.Collection{ID: id, Name: in.Name, EmbeddingModel: in.EmbeddingModel, ChunkUnit: "tokens", EmbeddingVersion: "minilm-transformer-v1", ChunkSize: in.ChunkSize, ChunkOverlap: in.ChunkOverlap, RoutingTags: in.RoutingTags, CreatedAt: store.Now()}
	documents, chunks, err := s.indexDocuments(ctx, c, in.Documents, engine)
	if err != nil {
		return c, err
	}
	if len(chunks) == 0 {
		return c, fmt.Errorf("provide at least one non-empty document")
	}
	if err := embedAll(ctx, chunks, engine); err != nil {
		return c, err
	}
	return c, s.Store.AddCollectionDocuments(ctx, c, documents, chunks)
}

// AddDocuments appends new content to an existing collection. Existing chunks
// remain intact; the collection's original embedding model and chunk settings
// are used so its vector space stays consistent.
func (s Service) AddDocuments(ctx context.Context, collectionID string, documents []Document) (domain.Collection, error) {
	collections, err := s.Store.Collections(ctx)
	if err != nil {
		return domain.Collection{}, err
	}
	var collection domain.Collection
	found := false
	for _, item := range collections {
		if item.ID == collectionID {
			collection, found = item, true
			break
		}
	}
	if !found {
		return domain.Collection{}, fmt.Errorf("collection %s not found", collectionID)
	}
	model := collection.EmbeddingModel
	if model == "" {
		model = embedding.DefaultModel
	}
	engine, err := embedding.New(model)
	if err != nil {
		return domain.Collection{}, err
	}
	documentRecords, chunks, err := s.indexDocuments(ctx, collection, documents, engine)
	if err != nil {
		return collection, err
	}
	if len(chunks) == 0 {
		return collection, fmt.Errorf("provide at least one non-empty document")
	}
	if err := embedAll(ctx, chunks, engine); err != nil {
		return collection, err
	}
	if err := s.Store.AddDocuments(ctx, collection.ID, documentRecords, chunks); err != nil {
		return collection, err
	}
	collection.DocumentCount += countSources(chunks)
	return collection, nil
}

func (s Service) indexDocuments(ctx context.Context, collection domain.Collection, inputs []Document, engine embedding.Engine) ([]domain.RAGDocument, []domain.Chunk, error) {
	documents := []domain.RAGDocument{}
	chunks := []domain.Chunk{}
	chunkSize, overlap := collection.ChunkSize, collection.ChunkOverlap
	if collection.ChunkUnit != "tokens" {
		chunkSize, overlap = 192, 32
	}
	for _, input := range inputs {
		loaded, err := load(ctx, input)
		if err != nil {
			return nil, nil, err
		}
		bySource := map[string]int{}
		for _, item := range loaded {
			metadata := cloneMetadata(item.Metadata)
			documentIndex, exists := bySource[item.Source]
			if !exists {
				now := store.Now()
				status := metadata["extractionStatus"]
				if status == "" {
					status = "complete"
				}
				document := domain.RAGDocument{ID: newID(), CollectionID: collection.ID, Name: displayName(item.Source), Source: item.Source, Path: item.Path, URL: item.URL, Text: item.InlineText, ContentType: item.ContentType, Metadata: metadata, ExtractionStatus: status, ExtractionError: metadata["extractionError"], Extractor: metadata["extractor"], RefreshIntervalMinutes: input.RefreshIntervalMinutes, CreatedAt: now, UpdatedAt: now}
				if document.URL != "" && document.RefreshIntervalMinutes > 0 {
					next := now.Add(time.Duration(document.RefreshIntervalMinutes) * time.Minute)
					document.LastRefreshedAt, document.NextRefreshAt = &now, &next
				}
				documents = append(documents, document)
				documentIndex, bySource[item.Source] = len(documents)-1, len(documents)-1
			}
			document := &documents[documentIndex]
			parts := splitTokens(item.Text, engine, chunkSize, overlap)
			for position, part := range parts {
				chunks = append(chunks, domain.Chunk{ID: newID(), CollectionID: collection.ID, DocumentID: document.ID, Source: document.Name, Content: part, ContentType: item.ContentType, Metadata: metadata, Position: document.ChunkCount + position, CreatedAt: document.CreatedAt})
			}
			document.ChunkCount += len(parts)
		}
	}
	return documents, chunks, nil
}

func cloneMetadata(metadata map[string]string) map[string]string {
	copy := make(map[string]string, len(metadata))
	for key, value := range metadata {
		copy[key] = value
	}
	return copy
}

func displayName(source string) string {
	if parsed, err := url.Parse(source); err == nil && parsed.Host != "" {
		return parsed.Host + parsed.Path
	}
	if name := filepath.Base(source); name != "." && name != "" {
		return name
	}
	return source
}

func countSources(chunks []domain.Chunk) int {
	sources := map[string]struct{}{}
	for _, chunk := range chunks {
		sources[chunk.Source] = struct{}{}
	}
	return len(sources)
}

// embedAll parallelizes independent encoder calls. The transformer is
// dependency-free and each embedding is independent, so this lets capable
// CPUs index a document set without changing vector results or their order.
func embedAll(ctx context.Context, chunks []domain.Chunk, engine embedding.Engine) error {
	if len(chunks) == 0 {
		return nil
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	if workers > len(chunks) {
		workers = len(chunks)
	}
	if workers > 8 {
		workers = 8
	}
	jobs := make(chan int)
	var wait sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for index := range jobs {
				if ctx.Err() != nil {
					return
				}
				chunks[index].Embedding = engine.Embed(chunks[index].Content)
			}
		}()
	}
	for index := range chunks {
		if ctx.Err() != nil {
			break
		}
		jobs <- index
	}
	close(jobs)
	wait.Wait()
	return ctx.Err()
}

type loadedDocument struct {
	Source, Text, ContentType string
	Path, URL, InlineText     string
	Metadata                  map[string]string
}

func load(ctx context.Context, d Document) ([]loadedDocument, error) {
	if d.Folder != "" {
		out := []loadedDocument{}
		e := filepath.WalkDir(d.Folder, func(path string, entry os.DirEntry, e error) error {
			if e != nil {
				return e
			}
			if entry.IsDir() {
				return nil
			}
			text, contentType, metadata, e := extractFile(ctx, path)
			if e != nil {
				return e
			}
			out = append(out, expandExtractedFile(path, path, text, contentType, metadata)...)
			return nil
		})
		if e != nil {
			return nil, fmt.Errorf("read folder %s: %w", d.Folder, e)
		}
		return out, nil
	}
	if d.Path != "" {
		text, contentType, metadata, e := extractFile(ctx, d.Path)
		if e != nil {
			return nil, fmt.Errorf("read %s: %w", d.Path, e)
		}
		if d.ContentType != "" {
			contentType = d.ContentType
		}
		for key, value := range d.Metadata {
			metadata[key] = value
		}
		return expandExtractedFile(d.Path, d.Path, text, contentType, metadata), nil
	}
	if d.URL != "" {
		text, metadata, err := loadURL(ctx, d.URL)
		if err != nil {
			return nil, err
		}
		for key, value := range d.Metadata {
			metadata[key] = value
		}
		return []loadedDocument{{Source: d.URL, URL: d.URL, Text: text, ContentType: "web", Metadata: metadata}}, nil
	}
	if strings.TrimSpace(d.Text) == "" {
		return nil, nil
	}
	if d.Source == "" {
		d.Source = "inline"
	}
	metadata := cloneMetadata(d.Metadata)
	metadata["extractionStatus"] = "complete"
	metadata["extractor"] = "inline"
	return []loadedDocument{{Source: d.Source, Text: d.Text, InlineText: d.Text, ContentType: "text", Metadata: metadata}}, nil
}

// pdftotext separates pages with form-feed characters. Retaining those page
// boundaries makes a citation actionable while still using ordinary text
// chunks for the vector database.
func expandExtractedFile(source, path, text, contentType string, metadata map[string]string) []loadedDocument {
	if contentType != "pdf" || !strings.Contains(text, "\f") {
		return []loadedDocument{{Source: source, Path: path, Text: text, ContentType: contentType, Metadata: metadata}}
	}
	pages := strings.Split(text, "\f")
	out := make([]loadedDocument, 0, len(pages))
	for index, page := range pages {
		if strings.TrimSpace(page) == "" {
			continue
		}
		pageMetadata := cloneMetadata(metadata)
		pageMetadata["page"] = fmt.Sprint(index + 1)
		out = append(out, loadedDocument{Source: source, Path: path, Text: page, ContentType: contentType, Metadata: pageMetadata})
	}
	return out
}

func loadURL(ctx context.Context, rawURL string) (string, map[string]string, error) {
	parsed, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return "", nil, fmt.Errorf("URL must be an absolute http or https address")
	}
	// Fetching loopback services through a RAG document would make the API an
	// SSRF primitive. Local files remain available through the explicit path
	// and folder import modes instead.
	host := strings.ToLower(parsed.Hostname())
	if host == "localhost" || strings.HasPrefix(host, "127.") || host == "::1" {
		return "", nil, fmt.Errorf("loopback URLs are not allowed for RAG ingestion")
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, parsed.String(), nil)
	if err != nil {
		return "", nil, err
	}
	request.Header.Set("User-Agent", "Sypra-RAG/1.0")
	response, err := (&http.Client{Timeout: 30 * time.Second}).Do(request)
	if err != nil {
		return "", nil, fmt.Errorf("fetch %s: %w", parsed.Host, err)
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return "", nil, fmt.Errorf("fetch %s: HTTP %d", parsed.Host, response.StatusCode)
	}
	bytes, err := io.ReadAll(io.LimitReader(response.Body, 16<<20))
	if err != nil {
		return "", nil, err
	}
	text := strings.TrimSpace(stripMarkup(string(bytes)))
	if text == "" {
		return "", nil, fmt.Errorf("web page contained no readable text")
	}
	return text, map[string]string{"url": parsed.String(), "path": parsed.String(), "contentType": "web", "extractor": "http", "extractionStatus": "complete"}, nil
}

func stripMarkup(value string) string {
	var out strings.Builder
	inTag := false
	for _, r := range value {
		switch r {
		case '<':
			inTag = true
		case '>':
			inTag = false
			out.WriteByte(' ')
		default:
			if !inTag && (unicode.IsPrint(r) || unicode.IsSpace(r)) {
				out.WriteRune(r)
			}
		}
	}
	return strings.Join(strings.Fields(out.String()), " ")
}
func (s Service) Retrieve(ctx context.Context, collectionID, query string, n int) ([]domain.Chunk, error) {
	results, err := s.RetrieveDetailed(ctx, collectionID, query, RetrievalOptions{Limit: n, IncludeNeighbors: true})
	if err != nil {
		return nil, err
	}
	chunks := make([]domain.Chunk, len(results))
	for index, result := range results {
		chunks[index] = result.Chunk
	}
	return chunks, nil
}

// RetrieveDetailed returns the exact text passed to retrieval, its hybrid
// score, immediate document context, and durable document/location metadata.
func (s Service) RetrieveDetailed(ctx context.Context, collectionID, query string, options RetrievalOptions) ([]RetrievalResult, error) {
	if strings.TrimSpace(query) == "" {
		return nil, fmt.Errorf("query is required")
	}
	if options.Limit < 0 {
		return nil, fmt.Errorf("result limit cannot be negative")
	}
	collections, err := s.Store.Collections(ctx)
	if err != nil {
		return nil, err
	}
	model := embedding.DefaultModel
	for _, collection := range collections {
		if collection.ID == collectionID && collection.EmbeddingModel != "" {
			model = collection.EmbeddingModel
			break
		}
	}
	engine, err := embedding.New(model)
	if err != nil {
		return nil, err
	}
	// A zero limit deliberately means all indexed chunks for diagnostic/full
	// retrieval. Chat requests set an explicit focused limit.
	if options.Limit == 0 {
		matches, err := s.Store.SearchVectorChunkMatches(ctx, collectionID, query, engine.Embed(query), 0, options.Metadata)
		if err != nil {
			return nil, err
		}
		return s.retrievalResults(ctx, matches, false)
	}
	// Retrieve a larger candidate pool first. The final pass diversifies sources
	// and adds neighboring chunks, which preserves the local passage around a
	// hit instead of sending near-identical slices of the same paragraph.
	candidates, err := s.Store.SearchVectorChunkMatches(ctx, collectionID, query, engine.Embed(query), max(options.Limit*4, 24), options.Metadata)
	if err != nil {
		return nil, err
	}
	if options.Rerank {
		rerankMatches(query, candidates)
	}
	selected := diversifyMatches(candidates, options.Limit)
	return s.retrievalResults(ctx, selected, options.IncludeNeighbors)
}

func rerankMatches(query string, matches []domain.ChunkMatch) {
	terms := strings.FieldsFunc(strings.ToLower(query), func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsNumber(r) })
	if len(terms) == 0 {
		return
	}
	for index := range matches {
		content := strings.ToLower(matches[index].Chunk.Content)
		coverage := 0
		for _, term := range terms {
			if strings.Contains(content, term) {
				coverage++
			}
		}
		// The local reranker favours complete query coverage without replacing the
		// dense/hybrid rank. The extra score stays in the public evidence model.
		matches[index].Score += .12 * float64(coverage) / float64(len(terms))
	}
	sort.SliceStable(matches, func(i, j int) bool { return matches[i].Score > matches[j].Score })
}

func diversifyMatches(candidates []domain.ChunkMatch, limit int) []domain.ChunkMatch {
	if limit <= 0 || len(candidates) <= limit {
		return candidates
	}
	selected := make([]domain.ChunkMatch, 0, limit)
	perDocument := map[string]int{}
	for _, match := range candidates {
		if len(selected) >= limit {
			break
		}
		key := match.Chunk.DocumentID
		if key == "" {
			key = match.Chunk.Source
		}
		if perDocument[key] >= 2 {
			continue
		}
		selected, perDocument[key] = append(selected, match), perDocument[key]+1
	}
	for _, match := range candidates {
		if len(selected) >= limit {
			break
		}
		found := false
		for _, current := range selected {
			if current.Chunk.ID == match.Chunk.ID {
				found = true
				break
			}
		}
		if !found {
			selected = append(selected, match)
		}
	}
	return selected
}

func (s Service) retrievalResults(ctx context.Context, matches []domain.ChunkMatch, neighbors bool) ([]RetrievalResult, error) {
	if len(matches) == 0 {
		return []RetrievalResult{}, nil
	}
	collectionID := matches[0].Chunk.CollectionID
	all, err := s.Store.CollectionChunks(ctx, collectionID)
	if err != nil {
		return nil, err
	}
	documents, err := s.Store.Documents(ctx, collectionID)
	if err != nil {
		return nil, err
	}
	byDocument := map[string]domain.RAGDocument{}
	for _, document := range documents {
		byDocument[document.ID] = document
	}
	byPosition := map[string]map[int]domain.Chunk{}
	for _, chunk := range all {
		key := chunk.DocumentID
		if key == "" {
			key = chunk.Source
		}
		if byPosition[key] == nil {
			byPosition[key] = map[int]domain.Chunk{}
		}
		byPosition[key][chunk.Position] = chunk
	}
	results := make([]RetrievalResult, 0, len(matches))
	for _, match := range matches {
		result := RetrievalResult{Chunk: match.Chunk, Score: match.Score, SemanticScore: match.SemanticScore, LexicalScore: match.LexicalScore, Location: locationFor(match.Chunk)}
		if document, ok := byDocument[match.Chunk.DocumentID]; ok {
			copy := document
			result.Document = &copy
		}
		if neighbors {
			key := match.Chunk.DocumentID
			if key == "" {
				key = match.Chunk.Source
			}
			if previous, ok := byPosition[key][match.Chunk.Position-1]; ok {
				copy := previous
				result.Previous = &copy
			}
			if next, ok := byPosition[key][match.Chunk.Position+1]; ok {
				copy := next
				result.Next = &copy
			}
		}
		results = append(results, result)
	}
	return results, nil
}

func locationFor(chunk domain.Chunk) string {
	if chunk.Metadata != nil && chunk.Metadata["page"] != "" {
		return "page " + chunk.Metadata["page"]
	}
	return fmt.Sprintf("chunk %d", chunk.Position+1)
}

func diversify(candidates []domain.Chunk, limit int) []domain.Chunk {
	if limit <= 0 || len(candidates) <= limit {
		return candidates
	}
	selected := make([]domain.Chunk, 0, limit)
	perSource := map[string]int{}
	for _, chunk := range candidates {
		if len(selected) >= limit {
			break
		}
		// Two passages per source keeps a single large file useful while making
		// room for independently relevant documents.
		if perSource[chunk.Source] >= 2 {
			continue
		}
		selected = append(selected, chunk)
		perSource[chunk.Source]++
	}
	for _, chunk := range candidates {
		if len(selected) >= limit {
			break
		}
		found := false
		for _, current := range selected {
			if current.ID == chunk.ID {
				found = true
				break
			}
		}
		if !found {
			selected = append(selected, chunk)
		}
	}
	return selected
}

func addNeighbors(selected, all []domain.Chunk, limit int) []domain.Chunk {
	if len(selected) == 0 {
		return selected
	}
	bySource := map[string][]domain.Chunk{}
	seen := map[string]bool{}
	for _, chunk := range all {
		bySource[chunk.Source] = append(bySource[chunk.Source], chunk)
	}
	for source := range bySource {
		sort.SliceStable(bySource[source], func(i, j int) bool { return bySource[source][i].Position < bySource[source][j].Position })
	}
	// Keep the top result first, then place its adjacent passage after it. This
	// is deliberately bounded by the requested retrieval count.
	out := make([]domain.Chunk, 0, limit)
	for _, chunk := range selected {
		if len(out) >= limit {
			break
		}
		if !seen[chunk.ID] {
			out, seen[chunk.ID] = append(out, chunk), true
		}
		for _, neighbor := range bySource[chunk.Source] {
			if len(out) >= limit {
				break
			}
			if !seen[neighbor.ID] && abs(neighbor.Position-chunk.Position) == 1 {
				out, seen[neighbor.ID] = append(out, neighbor), true
			}
		}
	}
	return out
}

func abs(value int) int {
	if value < 0 {
		return -value
	}
	return value
}
func (s Service) All(ctx context.Context, collectionID string) ([]domain.Chunk, error) {
	return s.Store.CollectionChunks(ctx, collectionID)
}

// ReindexDocument re-extracts the persisted original input and replaces only
// that document's chunks. It never silently changes another source.
func (s Service) ReindexDocument(ctx context.Context, documentID string, replacement *Document) (domain.RAGDocument, error) {
	document, err := s.Store.Document(ctx, documentID)
	if err != nil {
		return domain.RAGDocument{}, err
	}
	input := Document{Source: document.Source, Path: document.Path, URL: document.URL, Text: document.Text, ContentType: document.ContentType, RefreshIntervalMinutes: document.RefreshIntervalMinutes, Metadata: document.Metadata}
	if replacement != nil {
		input = *replacement
		if input.Source == "" {
			input.Source = document.Source
		}
	}
	collections, err := s.Store.Collections(ctx)
	if err != nil {
		return domain.RAGDocument{}, err
	}
	var collection domain.Collection
	for _, candidate := range collections {
		if candidate.ID == document.CollectionID {
			collection = candidate
			break
		}
	}
	if collection.ID == "" {
		return domain.RAGDocument{}, fmt.Errorf("collection %s not found", document.CollectionID)
	}
	engine, err := embedding.New(collection.EmbeddingModel)
	if err != nil {
		return domain.RAGDocument{}, err
	}
	loaded, err := load(ctx, input)
	if err != nil {
		return domain.RAGDocument{}, err
	}
	if len(loaded) == 0 {
		return domain.RAGDocument{}, fmt.Errorf("replacement has no extractable document")
	}
	item := loaded[0]
	now := store.Now()
	document.Name, document.Source, document.Path, document.URL, document.Text = displayName(item.Source), item.Source, item.Path, item.URL, item.InlineText
	document.ContentType, document.Metadata, document.UpdatedAt = item.ContentType, cloneMetadata(item.Metadata), now
	document.ExtractionStatus, document.ExtractionError, document.Extractor = document.Metadata["extractionStatus"], document.Metadata["extractionError"], document.Metadata["extractor"]
	if document.ExtractionStatus == "" {
		document.ExtractionStatus = "complete"
	}
	if document.URL != "" && document.RefreshIntervalMinutes > 0 {
		next := now.Add(time.Duration(document.RefreshIntervalMinutes) * time.Minute)
		document.LastRefreshedAt, document.NextRefreshAt = &now, &next
	}
	chunkSize, overlap := collection.ChunkSize, collection.ChunkOverlap
	if collection.ChunkUnit != "tokens" {
		chunkSize, overlap = 192, 32
	}
	chunks := []domain.Chunk{}
	for _, page := range loaded {
		basePosition := len(chunks)
		for position, text := range splitTokens(page.Text, engine, chunkSize, overlap) {
			chunks = append(chunks, domain.Chunk{ID: newID(), CollectionID: collection.ID, DocumentID: document.ID, Source: document.Name, Content: text, ContentType: document.ContentType, Metadata: cloneMetadata(page.Metadata), Position: basePosition + position, CreatedAt: now})
		}
	}
	if len(chunks) == 0 {
		return document, fmt.Errorf("document has no extractable text")
	}
	if err := embedAll(ctx, chunks, engine); err != nil {
		return document, err
	}
	document.ChunkCount = len(chunks)
	if err := s.Store.ReplaceDocument(ctx, document, chunks); err != nil {
		return document, err
	}
	return document, nil
}

// RefreshDue updates scheduled URL sources. A failing source remains indexed;
// the previous chunks are left intact and its failure is returned to the
// caller/logger rather than replacing useful knowledge with an empty document.
func (s Service) RefreshDue(ctx context.Context) []error {
	documents, err := s.Store.Documents(ctx, "")
	if err != nil {
		return []error{err}
	}
	now := store.Now()
	problems := []error{}
	for _, document := range documents {
		if document.URL == "" || document.NextRefreshAt == nil || document.NextRefreshAt.After(now) {
			continue
		}
		if _, err := s.ReindexDocument(ctx, document.ID, nil); err != nil {
			problems = append(problems, fmt.Errorf("refresh %s: %w", document.Name, err))
		}
	}
	return problems
}

func (s Service) CreateEvaluationSet(ctx context.Context, collectionID, name string, cases []domain.RAGEvaluationCase) (domain.RAGEvaluationSet, error) {
	if strings.TrimSpace(collectionID) == "" || strings.TrimSpace(name) == "" {
		return domain.RAGEvaluationSet{}, fmt.Errorf("collection ID and evaluation set name are required")
	}
	if len(cases) == 0 {
		return domain.RAGEvaluationSet{}, fmt.Errorf("evaluation set requires at least one case")
	}
	for index := range cases {
		if strings.TrimSpace(cases[index].Query) == "" {
			return domain.RAGEvaluationSet{}, fmt.Errorf("evaluation case %d requires a query", index+1)
		}
		if cases[index].ID == "" {
			cases[index].ID = newID()
		}
		if len(cases[index].RelevantDocumentIDs) == 0 && len(cases[index].RelevantChunkIDs) == 0 {
			return domain.RAGEvaluationSet{}, fmt.Errorf("evaluation case %d needs a relevant document or chunk", index+1)
		}
	}
	now := store.Now()
	set := domain.RAGEvaluationSet{ID: newID(), CollectionID: collectionID, Name: name, Cases: cases, CreatedAt: now, UpdatedAt: now}
	if err := s.Store.AddEvaluationSet(ctx, set); err != nil {
		return domain.RAGEvaluationSet{}, err
	}
	return set, nil
}

// Evaluate measures retriever quality rather than answer quality: hit rate,
// Recall@K, Precision@K and MRR are calculated from explicitly labelled
// document/chunk relevance.
func (s Service) Evaluate(ctx context.Context, evaluationID string, limit int, options RetrievalOptions) (domain.RAGEvaluationRun, error) {
	set, err := s.Store.EvaluationSet(ctx, evaluationID)
	if err != nil {
		return domain.RAGEvaluationRun{}, err
	}
	if limit <= 0 {
		limit = 5
	}
	options.Limit, options.IncludeNeighbors = limit, false
	results := make([]domain.RAGEvaluationResult, 0, len(set.Cases))
	var hits, recallTotal, precisionTotal, reciprocalTotal float64
	for _, test := range set.Cases {
		retrieved, err := s.RetrieveDetailed(ctx, set.CollectionID, test.Query, options)
		if err != nil {
			return domain.RAGEvaluationRun{}, fmt.Errorf("evaluate %q: %w", test.Query, err)
		}
		relevantDocs, relevantChunks := stringSet(test.RelevantDocumentIDs), stringSet(test.RelevantChunkIDs)
		retrievedIDs := make([]string, 0, len(retrieved))
		matched, firstRank := 0, 0
		for index, result := range retrieved {
			retrievedIDs = append(retrievedIDs, result.Chunk.ID)
			if relevantDocs[result.Chunk.DocumentID] || relevantChunks[result.Chunk.ID] {
				matched++
				if firstRank == 0 {
					firstRank = index + 1
				}
			}
		}
		relevantCount := len(relevantDocs) + len(relevantChunks)
		if relevantCount == 0 {
			relevantCount = 1
		}
		recall := float64(matched) / float64(relevantCount)
		if recall > 1 {
			recall = 1
		}
		precision := float64(matched) / float64(max(1, len(retrieved)))
		reciprocal := 0.0
		if firstRank > 0 {
			reciprocal, hits = 1/float64(firstRank), hits+1
		}
		results = append(results, domain.RAGEvaluationResult{CaseID: test.ID, Query: test.Query, Retrieved: retrievedIDs, Hit: firstRank > 0, Rank: firstRank, Recall: recall, Precision: precision, Reciprocal: reciprocal})
		recallTotal, precisionTotal, reciprocalTotal = recallTotal+recall, precisionTotal+precision, reciprocalTotal+reciprocal
	}
	denominator := float64(max(1, len(results)))
	run := domain.RAGEvaluationRun{ID: newID(), EvaluationID: set.ID, CollectionID: set.CollectionID, Limit: limit, Metrics: map[string]float64{"hitRate": hits / denominator, "recallAtK": recallTotal / denominator, "precisionAtK": precisionTotal / denominator, "mrr": reciprocalTotal / denominator}, Results: results, CreatedAt: store.Now()}
	if err := s.Store.AddEvaluationRun(ctx, run); err != nil {
		return domain.RAGEvaluationRun{}, err
	}
	return run, nil
}

func stringSet(values []string) map[string]bool {
	set := map[string]bool{}
	for _, value := range values {
		set[value] = true
	}
	return set
}

// Route selects the most relevant permitted collection for an unbound search.
// It is opt-in; chats retain their immutable collection binding policy. Tags
// are compared case-insensitively against collection routing tags.
func (s Service) Route(ctx context.Context, query string, collectionIDs, tags []string, options RetrievalOptions) (string, []RetrievalResult, error) {
	collections, err := s.Store.Collections(ctx)
	if err != nil {
		return "", nil, err
	}
	allowed, wantedTags := stringSet(collectionIDs), stringSet(lowerAll(tags))
	bestID, bestScore := "", -1.0
	var best []RetrievalResult
	for _, collection := range collections {
		if len(allowed) > 0 && !allowed[collection.ID] {
			continue
		}
		if len(wantedTags) > 0 && !hasRoutingTag(collection.RoutingTags, wantedTags) {
			continue
		}
		candidateOptions := options
		if candidateOptions.Limit == 0 {
			candidateOptions.Limit = 1
		}
		results, err := s.RetrieveDetailed(ctx, collection.ID, query, candidateOptions)
		if err != nil || len(results) == 0 {
			continue
		}
		if results[0].Score > bestScore {
			bestID, bestScore, best = collection.ID, results[0].Score, results
		}
	}
	if bestID == "" {
		return "", nil, fmt.Errorf("no routed collection produced a result")
	}
	return bestID, best, nil
}

func lowerAll(values []string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		out = append(out, strings.ToLower(value))
	}
	return out
}

func hasRoutingTag(tags []string, wanted map[string]bool) bool {
	for _, tag := range tags {
		if wanted[strings.ToLower(tag)] {
			return true
		}
	}
	return false
}
func split(text string, size, overlap int) []string {
	sentences := sentenceUnits(text)
	if len(sentences) == 0 {
		return nil
	}
	// Chunk size is a target, not a reason to cut a word or sentence in half.
	// A single sentence longer than the target stays intact by design.
	if size <= 0 {
		return []string{joinSentences(sentences)}
	}
	out := make([]string, 0)
	current := make([]string, 0)
	for _, sentence := range sentences {
		if len(current) == 0 || runeLen(joinSentences(append(append([]string{}, current...), sentence))) <= size {
			current = append(current, sentence)
			continue
		}
		out = append(out, joinSentences(current))
		current = sentenceOverlap(current, overlap)
		for len(current) > 0 && runeLen(joinSentences(append(append([]string{}, current...), sentence))) > size {
			current = current[1:]
		}
		current = append(current, sentence)
	}
	if len(current) > 0 {
		out = append(out, joinSentences(current))
	}
	return out
}

// splitTokens keeps complete sentences together while sizing chunks with the
// tokenizer of the embedding model. This prevents a multi-byte character
// count, or a cut in the middle of a word, from defining retrieval units.
func splitTokens(text string, engine embedding.Engine, size, overlap int) []string {
	sentences := sentenceUnits(text)
	if len(sentences) == 0 {
		return nil
	}
	if size <= 0 {
		return []string{joinSentences(sentences)}
	}
	out := make([]string, 0)
	current := make([]string, 0)
	for _, sentence := range sentences {
		candidate := append(append([]string{}, current...), sentence)
		if len(current) == 0 || engine.TokenCount(joinSentences(candidate)) <= size {
			current = append(current, sentence)
			continue
		}
		out = append(out, joinSentences(current))
		current = tokenOverlap(current, engine, overlap)
		for len(current) > 0 && engine.TokenCount(joinSentences(append(append([]string{}, current...), sentence))) > size {
			current = current[1:]
		}
		current = append(current, sentence)
	}
	if len(current) > 0 {
		out = append(out, joinSentences(current))
	}
	return out
}

func tokenOverlap(sentences []string, engine embedding.Engine, limit int) []string {
	if limit <= 0 {
		return nil
	}
	overlap := make([]string, 0)
	for i := len(sentences) - 1; i >= 0; i-- {
		candidate := append([]string{sentences[i]}, overlap...)
		if engine.TokenCount(joinSentences(candidate)) > limit {
			break
		}
		overlap = candidate
	}
	return overlap
}

func sentenceUnits(text string) []string {
	runes := []rune(strings.TrimSpace(text))
	if len(runes) == 0 {
		return nil
	}
	units := make([]string, 0)
	start := 0
	for i, r := range runes {
		if !strings.ContainsRune(".!?。！？", r) {
			continue
		}
		end := i + 1
		for end < len(runes) && strings.ContainsRune("\"'”’)]}", runes[end]) {
			end++
		}
		if end < len(runes) && !unicode.IsSpace(runes[end]) {
			continue
		}
		if sentence := strings.TrimSpace(string(runes[start:end])); sentence != "" {
			units = append(units, sentence)
		}
		start = end
	}
	if trailing := strings.TrimSpace(string(runes[start:])); trailing != "" {
		units = append(units, trailing)
	}
	return units
}

func sentenceOverlap(sentences []string, limit int) []string {
	if limit <= 0 {
		return nil
	}
	overlap := make([]string, 0)
	for i := len(sentences) - 1; i >= 0; i-- {
		candidate := append([]string{sentences[i]}, overlap...)
		if runeLen(joinSentences(candidate)) > limit {
			break
		}
		overlap = candidate
	}
	return overlap
}

func joinSentences(sentences []string) string { return strings.Join(sentences, " ") }
func runeLen(text string) int                 { return len([]rune(text)) }
func newID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
}
