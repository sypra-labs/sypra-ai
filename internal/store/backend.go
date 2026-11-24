package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	_ "github.com/jackc/pgx/v5/stdlib"
	_ "modernc.org/sqlite"
)

var errStateNotFound = errors.New("workspace state not found")

type stateBackend interface {
	Load(context.Context) ([]byte, error)
	Save(context.Context, []byte) error
	Health(context.Context) error
	Close() error
}

type fileBackend struct{ path string }

func (b fileBackend) Load(_ context.Context) ([]byte, error) {
	payload, err := os.ReadFile(b.path)
	if os.IsNotExist(err) {
		return nil, errStateNotFound
	}
	return payload, err
}
func (b fileBackend) Save(_ context.Context, payload []byte) error {
	temporary := b.path + ".tmp"
	if err := os.WriteFile(temporary, payload, 0600); err != nil {
		return err
	}
	return os.Rename(temporary, b.path)
}
func (b fileBackend) Health(_ context.Context) error {
	if b.path == "" {
		return fmt.Errorf("state file path is empty")
	}
	return nil
}
func (b fileBackend) Close() error { return nil }

type sqlBackend struct {
	db           *sql.DB
	placeholder  string
	databaseName string
}

func openStateBackend(ctx context.Context, rawURL string) (stateBackend, error) {
	rawURL = strings.TrimSpace(rawURL)
	if rawURL == "" {
		return nil, fmt.Errorf("database URL is required")
	}
	if strings.HasPrefix(rawURL, "sqlite://") {
		path := strings.TrimPrefix(rawURL, "sqlite://")
		if path == "" {
			return nil, fmt.Errorf("SQLite database path is required")
		}
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil && filepath.Dir(path) != "." {
			return nil, fmt.Errorf("create SQLite directory: %w", err)
		}
		dsn := "file:" + filepath.ToSlash(path) + "?_pragma=busy_timeout(5000)&_pragma=journal_mode(WAL)&_pragma=foreign_keys(1)"
		db, err := sql.Open("sqlite", dsn)
		if err != nil {
			return nil, fmt.Errorf("open SQLite database: %w", err)
		}
		db.SetMaxOpenConns(1)
		backend := &sqlBackend{db: db, placeholder: "?", databaseName: "sqlite"}
		if err = backend.initialize(ctx); err != nil {
			_ = db.Close()
			return nil, err
		}
		return backend, nil
	}
	parsed, err := url.Parse(rawURL)
	if err != nil || (parsed.Scheme != "postgres" && parsed.Scheme != "postgresql") {
		return nil, fmt.Errorf("database URL must use sqlite://, postgres://, or postgresql://")
	}
	db, err := sql.Open("pgx", rawURL)
	if err != nil {
		return nil, fmt.Errorf("open PostgreSQL database: %w", err)
	}
	db.SetMaxOpenConns(12)
	db.SetMaxIdleConns(4)
	db.SetConnMaxLifetime(30 * time.Minute)
	backend := &sqlBackend{db: db, placeholder: "$1", databaseName: "postgresql"}
	if err = backend.initialize(ctx); err != nil {
		_ = db.Close()
		return nil, err
	}
	return backend, nil
}

func (b *sqlBackend) initialize(ctx context.Context) error {
	if err := b.db.PingContext(ctx); err != nil {
		return fmt.Errorf("connect %s database: %w", b.databaseName, err)
	}
	_, err := b.db.ExecContext(ctx, `CREATE TABLE IF NOT EXISTS sypra_workspace_state (
id BIGINT PRIMARY KEY,
payload BYTEA NOT NULL,
updated_at TIMESTAMP NOT NULL
)`)
	if err != nil {
		return fmt.Errorf("initialize %s schema: %w", b.databaseName, err)
	}
	return nil
}

func (b *sqlBackend) Load(ctx context.Context) ([]byte, error) {
	query := "SELECT payload FROM sypra_workspace_state WHERE id = " + b.placeholder
	var payload []byte
	err := b.db.QueryRowContext(ctx, query, 1).Scan(&payload)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, errStateNotFound
	}
	if err != nil {
		return nil, fmt.Errorf("load workspace state: %w", err)
	}
	return payload, nil
}

func (b *sqlBackend) Save(ctx context.Context, payload []byte) error {
	var query string
	if b.databaseName == "postgresql" {
		query = `INSERT INTO sypra_workspace_state (id, payload, updated_at) VALUES ($1, $2, $3)
ON CONFLICT (id) DO UPDATE SET payload = EXCLUDED.payload, updated_at = EXCLUDED.updated_at`
	} else {
		query = `INSERT INTO sypra_workspace_state (id, payload, updated_at) VALUES (?, ?, ?)
ON CONFLICT(id) DO UPDATE SET payload = excluded.payload, updated_at = excluded.updated_at`
	}
	if _, err := b.db.ExecContext(ctx, query, 1, payload, time.Now().UTC()); err != nil {
		return fmt.Errorf("save workspace state: %w", err)
	}
	return nil
}
func (b *sqlBackend) Health(ctx context.Context) error { return b.db.PingContext(ctx) }
func (b *sqlBackend) Close() error                     { return b.db.Close() }
