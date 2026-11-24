//go:build integration

package store

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/sypra-labs/sypra-ai/internal/domain"
)

// TestSQLiteImportsLegacyWorkspace is an upgrade gate. It uses the actual
// SQLite driver and verifies that a pre-SQL JSON workspace is available after
// migration without manual export/import work.
func TestSQLiteImportsLegacyWorkspace(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	legacy := filepath.Join(dir, "sypra.json")
	old, err := Open(legacy)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = old.AddModel(ctx, domain.Model{ID: "base", Name: "Migrated", Path: "model", Kind: "base", CreatedAt: Now()}); err != nil {
		t.Fatal(err)
	}
	if err = old.Close(); err != nil {
		t.Fatal(err)
	}
	upgraded, err := OpenConfigured(ctx, "sqlite://"+filepath.ToSlash(filepath.Join(dir, "sypra.db")), legacy)
	if err != nil {
		t.Fatal(err)
	}
	defer upgraded.Close()
	models, err := upgraded.Models(ctx)
	if err != nil || len(models) != 1 || models[0].ID != "base" {
		t.Fatalf("migrated models=%#v err=%v", models, err)
	}
	if _, err = os.Stat(filepath.Join(dir, "sypra.db")); err != nil {
		t.Fatalf("SQLite state was not created: %v", err)
	}
}
