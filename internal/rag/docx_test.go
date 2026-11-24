package rag

import (
	"archive/zip"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExtractDOCX(t *testing.T) {
	path := filepath.Join(t.TempDir(), "handbook.docx")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	archive := zip.NewWriter(file)
	entry, err := archive.Create("word/document.xml")
	if err != nil {
		t.Fatal(err)
	}
	_, err = entry.Write([]byte(`<?xml version="1.0"?><w:document xmlns:w="urn:test"><w:body><w:p><w:r><w:t>Expenses require receipts.</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Travel needs approval.</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>`))
	if err != nil {
		t.Fatal(err)
	}
	if err = archive.Close(); err != nil {
		t.Fatal(err)
	}
	if err = file.Close(); err != nil {
		t.Fatal(err)
	}

	text, err := extractDOCX(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(text, "Expenses require receipts.") || !strings.Contains(text, "Travel needs approval.") {
		t.Fatalf("extracted text missing expected content: %q", text)
	}
}

func TestUnknownBinaryIsPreservedWithPendingExtractionMetadata(t *testing.T) {
	path := filepath.Join(t.TempDir(), "recording.custombin")
	if err := os.WriteFile(path, []byte{0, 1, 2}, 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("SYPRA_RAG_EXTRACT_COMMAND", "")
	text, contentType, metadata, err := extractFile(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	if contentType != "binary" || metadata["extractionStatus"] != "pending" || !strings.Contains(text, "extraction is pending") {
		t.Fatalf("unexpected fallback: %q %q %#v", text, contentType, metadata)
	}
}
