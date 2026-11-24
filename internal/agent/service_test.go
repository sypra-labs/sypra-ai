package agent

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/sypra-labs/sypra-ai/internal/domain"
	"github.com/sypra-labs/sypra-ai/internal/store"
	"github.com/sypra-labs/sypra-ai/internal/websearch"
)

func TestCatalogAndDenyByDefault(t *testing.T) {
	st, err := store.Open(filepath.Join(t.TempDir(), "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	service := New(st, websearch.New(), nil)
	catalog, err := service.Catalog(context.Background())
	if err != nil || len(catalog) < 2 {
		t.Fatalf("catalog=%#v err=%v", catalog, err)
	}
	if _, err := service.Execute(context.Background(), "model-a", "project-a", "terminal.run", map[string]any{"command": "echo"}, true); err == nil {
		t.Fatal("unpermitted terminal tool was allowed")
	}
	permission := domain.ToolPermission{ID: "permission", ToolID: "terminal.run", ModelID: "model-a", ProjectID: "project-a", Enabled: true, RequireApproval: true, CreatedAt: store.Now()}
	if err := st.AddToolPermission(context.Background(), permission); err != nil {
		t.Fatal(err)
	}
	run, err := service.Execute(context.Background(), "model-a", "project-a", "terminal.run", map[string]any{"command": "echo"}, false)
	if err != nil || run.Status != "awaiting_approval" {
		t.Fatalf("approval-required terminal tool should queue for approval: run=%#v err=%v", run, err)
	}
}
