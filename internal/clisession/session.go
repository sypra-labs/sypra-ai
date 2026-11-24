// Package clisession stores terminal chat history beside the work it belongs
// to. A session cannot be continued from a different working folder.
package clisession

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/sypra-labs/sypra-ai/internal/inference"
)

type Session struct {
	ID        string              `json:"id"`
	ChatID    string              `json:"chatId"`
	Workspace string              `json:"workspace"`
	Endpoint  string              `json:"endpoint"`
	Model     string              `json:"model"`
	Adapter   string              `json:"adapter,omitempty"`
	Messages  []inference.Message `json:"messages"`
	CreatedAt time.Time           `json:"createdAt"`
	UpdatedAt time.Time           `json:"updatedAt"`
}

func Directory(workspace string) string { return filepath.Join(workspace, ".sypra", "chats") }

func New(workspace, endpoint, model, adapter string) (Session, error) {
	root, err := filepath.Abs(workspace)
	if err != nil {
		return Session{}, err
	}
	now := time.Now().UTC()
	return Session{ID: fmt.Sprintf("%d", now.UnixNano()), Workspace: filepath.Clean(root), Endpoint: endpoint, Model: model, Adapter: adapter, CreatedAt: now, UpdatedAt: now}, nil
}

func Load(workspace, id string) (Session, error) {
	if strings.ContainsAny(id, `/\\`) || id == "" {
		return Session{}, fmt.Errorf("session must be a session ID, not a path")
	}
	root, err := filepath.Abs(workspace)
	if err != nil {
		return Session{}, err
	}
	data, err := os.ReadFile(filepath.Join(Directory(root), id+".json"))
	if err != nil {
		return Session{}, err
	}
	var session Session
	if err = json.Unmarshal(data, &session); err != nil {
		return Session{}, fmt.Errorf("read CLI chat session: %w", err)
	}
	if filepath.Clean(session.Workspace) != filepath.Clean(root) {
		return Session{}, fmt.Errorf("session %s belongs to %s; continue it from that folder", id, session.Workspace)
	}
	return session, nil
}

func Save(session *Session) error {
	if session.ID == "" || session.Workspace == "" {
		return fmt.Errorf("CLI chat session is incomplete")
	}
	session.UpdatedAt = time.Now().UTC()
	directory := Directory(session.Workspace)
	if err := os.MkdirAll(directory, 0700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(session, "", "  ")
	if err != nil {
		return err
	}
	temporary := filepath.Join(directory, session.ID+".tmp")
	if err = os.WriteFile(temporary, data, 0600); err != nil {
		return err
	}
	return os.Rename(temporary, filepath.Join(directory, session.ID+".json"))
}

func List(workspace string) ([]Session, error) {
	root, err := filepath.Abs(workspace)
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(Directory(root))
	if os.IsNotExist(err) {
		return []Session{}, nil
	}
	if err != nil {
		return nil, err
	}
	result := make([]Session, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		data, readErr := os.ReadFile(filepath.Join(Directory(root), entry.Name()))
		var session Session
		if readErr == nil && json.Unmarshal(data, &session) == nil && filepath.Clean(session.Workspace) == filepath.Clean(root) {
			result = append(result, session)
		}
	}
	sort.Slice(result, func(i, j int) bool { return result[i].UpdatedAt.After(result[j].UpdatedAt) })
	return result, nil
}

func Close(workspace, id string) error {
	session, err := Load(workspace, id)
	if err != nil {
		return err
	}
	return os.Remove(filepath.Join(Directory(session.Workspace), session.ID+".json"))
}
