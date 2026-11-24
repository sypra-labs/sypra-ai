package store

import (
	"archive/zip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/sypra-labs/sypra-ai/internal/vectordb"
)

const backupSchema = 1

// BackupTo writes a portable, atomic workspace backup. The archive contains
// state and vectors together, so a restored RAG collection is immediately
// queryable without re-embedding documents.
func (s *Store) BackupTo(ctx context.Context, destination string) error {
	if destination == "" {
		return fmt.Errorf("backup destination is required")
	}
	s.mu.RLock()
	workspace, err := json.Marshal(s.data)
	s.mu.RUnlock()
	if err != nil {
		return fmt.Errorf("serialize workspace backup: %w", err)
	}
	vectors, err := json.Marshal(s.vectors.Records(ctx))
	if err != nil {
		return fmt.Errorf("serialize vector backup: %w", err)
	}
	if err = os.MkdirAll(filepath.Dir(destination), 0755); err != nil && filepath.Dir(destination) != "." {
		return err
	}
	temporary := destination + ".tmp"
	file, err := os.OpenFile(temporary, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	archive := zip.NewWriter(file)
	write := func(name string, data []byte) error {
		entry, writeErr := archive.Create(name)
		if writeErr != nil {
			return writeErr
		}
		_, writeErr = entry.Write(data)
		return writeErr
	}
	manifest, _ := json.Marshal(map[string]any{"schema": backupSchema, "createdAt": time.Now().UTC(), "application": "sypra"})
	if err = write("manifest.json", manifest); err == nil {
		err = write("workspace.json", workspace)
	}
	if err == nil {
		err = write("vectors.json", vectors)
	}
	if closeErr := archive.Close(); err == nil {
		err = closeErr
	}
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		_ = os.Remove(temporary)
		return fmt.Errorf("write workspace backup: %w", err)
	}
	return os.Rename(temporary, destination)
}

// RestoreFrom replaces the workspace only after the complete archive has been
// parsed and validated. Operators should take a fresh backup before restore.
func (s *Store) RestoreFrom(ctx context.Context, source string) error {
	file, err := os.Open(source)
	if err != nil {
		return err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return err
	}
	if info.Size() > 512<<20 {
		return fmt.Errorf("backup is larger than the 512 MiB restore safety limit")
	}
	archive, err := zip.NewReader(file, info.Size())
	if err != nil {
		return fmt.Errorf("open workspace backup: %w", err)
	}
	entries := map[string][]byte{}
	for _, entry := range archive.File {
		if entry.UncompressedSize64 > 512<<20 {
			return fmt.Errorf("backup entry %s exceeds restore safety limit", entry.Name)
		}
		reader, openErr := entry.Open()
		if openErr != nil {
			return openErr
		}
		payload, readErr := io.ReadAll(io.LimitReader(reader, 512<<20))
		_ = reader.Close()
		if readErr != nil {
			return readErr
		}
		entries[entry.Name] = payload
	}
	var manifest struct {
		Schema int `json:"schema"`
	}
	if err = json.Unmarshal(entries["manifest.json"], &manifest); err != nil || manifest.Schema != backupSchema {
		return fmt.Errorf("unsupported or invalid Sypra workspace backup")
	}
	var next state
	if err = json.Unmarshal(entries["workspace.json"], &next); err != nil {
		return fmt.Errorf("read workspace backup: %w", err)
	}
	var vectors []vectordb.Record
	if err = json.Unmarshal(entries["vectors.json"], &vectors); err != nil {
		return fmt.Errorf("read vector backup: %w", err)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err = s.vectors.Replace(ctx, vectors); err != nil {
		return fmt.Errorf("restore vectors: %w", err)
	}
	s.data = next
	if err = s.saveLocked(); err != nil {
		return fmt.Errorf("restore workspace state: %w", err)
	}
	return nil
}
