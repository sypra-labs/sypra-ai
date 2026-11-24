package clisession

import (
	"path/filepath"
	"testing"
)

func TestSessionIsBoundToCreationFolder(t *testing.T) {
	first := t.TempDir()
	second := t.TempDir()
	session, err := New(first, "http://127.0.0.1:8081/v1", "qwen", "")
	if err != nil {
		t.Fatal(err)
	}
	if err = Save(&session); err != nil {
		t.Fatal(err)
	}
	if _, err = Load(second, session.ID); err == nil {
		t.Fatal("expected session to be unavailable from another folder")
	}
	loaded, err := Load(filepath.Clean(first), session.ID)
	if err != nil || loaded.Workspace != session.Workspace {
		t.Fatalf("load=%#v err=%v", loaded, err)
	}
}
