package api

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/sypra-labs/sypra-ai/internal/rag"
	"github.com/sypra-labs/sypra-ai/internal/store"
	"github.com/sypra-labs/sypra-ai/internal/training"
)

func TestUploadDOCXAndIndexIt(t *testing.T) {
	dir := t.TempDir()
	st, err := store.Open(filepath.Join(dir, "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	srv := &Server{
		Store:      st,
		RAG:        rag.Service{Store: st},
		Trainer:    training.New(st),
		Inference:  mockInference(t),
		CORSOrigin: "http://localhost:5173",
		Logger:     slog.New(slog.NewTextHandler(io.Discard, nil)),
		UploadDir:  filepath.Join(dir, "models"),
	}

	var upload bytes.Buffer
	form := multipart.NewWriter(&upload)
	part, err := form.CreateFormFile("file", "handbook.docx")
	if err != nil {
		t.Fatal(err)
	}
	archive := zip.NewWriter(part)
	entry, err := archive.Create("word/document.xml")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = entry.Write([]byte(`<w:document xmlns:w="urn:test"><w:body><w:p><w:r><w:t>Travel requires approval.</w:t></w:r></w:p></w:body></w:document>`)); err != nil {
		t.Fatal(err)
	}
	if err = archive.Close(); err != nil {
		t.Fatal(err)
	}
	if err = form.Close(); err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/v1/rag/documents/upload", &upload)
	req.Header.Set("Content-Type", form.FormDataContentType())
	response := httptest.NewRecorder()
	srv.Handler().ServeHTTP(response, req)
	if response.Code != http.StatusCreated {
		t.Fatalf("upload status %d: %s", response.Code, response.Body.String())
	}
	var stored struct{ Path string }
	if err = json.NewDecoder(response.Body).Decode(&stored); err != nil {
		t.Fatal(err)
	}

	collection := request(srv.Handler(), http.MethodPost, "/api/v1/rag/collections", map[string]any{
		"name":      "handbook",
		"documents": []map[string]string{{"path": stored.Path}},
	})
	if collection.Code != http.StatusCreated {
		t.Fatalf("collection status %d: %s", collection.Code, collection.Body.String())
	}
	var saved struct{ ID string }
	if err = json.NewDecoder(collection.Body).Decode(&saved); err != nil {
		t.Fatal(err)
	}
	query := request(srv.Handler(), http.MethodPost, "/api/v1/rag/query", map[string]any{"collectionID": saved.ID, "query": "approval", "limit": 0})
	if query.Code != http.StatusOK || !bytes.Contains(query.Body.Bytes(), []byte("Travel requires approval.")) {
		t.Fatalf("DOCX was not indexed: %d %s", query.Code, query.Body.String())
	}
}
