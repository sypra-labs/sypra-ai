// Package observability provides dependency-free structured logging, bounded
// log retention, request metrics, and W3C-compatible request correlation.
package observability

import (
	"crypto/rand"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type Config struct {
	Format   string
	File     string
	MaxBytes int
	Backups  int
}

// NewLogger writes structured records to stdout and, when configured, a
// rotation-bounded local file. It never logs configuration values or secrets.
func NewLogger(config Config) (*slog.Logger, io.Closer, error) {
	var output io.Writer = os.Stdout
	var closer io.Closer = nopCloser{}
	if strings.TrimSpace(config.File) != "" {
		writer, err := newRotatingWriter(config.File, config.MaxBytes, config.Backups)
		if err != nil {
			return nil, nil, err
		}
		output, closer = io.MultiWriter(os.Stdout, writer), writer
	}
	options := &slog.HandlerOptions{Level: slog.LevelInfo}
	if strings.EqualFold(config.Format, "text") {
		return slog.New(slog.NewTextHandler(output, options)), closer, nil
	}
	return slog.New(slog.NewJSONHandler(output, options)), closer, nil
}

type nopCloser struct{}

func (nopCloser) Close() error { return nil }

type rotatingWriter struct {
	mu       sync.Mutex
	path     string
	maxBytes int64
	backups  int
	file     *os.File
}

func newRotatingWriter(path string, maxBytes, backups int) (*rotatingWriter, error) {
	if maxBytes <= 0 {
		maxBytes = 10 * 1024 * 1024
	}
	if backups < 1 {
		backups = 1
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil && filepath.Dir(path) != "." {
		return nil, fmt.Errorf("create log directory: %w", err)
	}
	writer := &rotatingWriter{path: path, maxBytes: int64(maxBytes), backups: backups}
	if err := writer.open(); err != nil {
		return nil, err
	}
	return writer, nil
}
func (w *rotatingWriter) open() error {
	file, err := os.OpenFile(w.path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	w.file = file
	return nil
}
func (w *rotatingWriter) Write(payload []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if info, err := w.file.Stat(); err == nil && info.Size()+int64(len(payload)) > w.maxBytes {
		if err := w.rotate(); err != nil {
			return 0, err
		}
	}
	return w.file.Write(payload)
}
func (w *rotatingWriter) rotate() error {
	if err := w.file.Close(); err != nil {
		return err
	}
	for index := w.backups - 1; index >= 1; index-- {
		from, to := w.path+"."+strconv.Itoa(index), w.path+"."+strconv.Itoa(index+1)
		if index == w.backups-1 {
			_ = os.Remove(to)
		}
		if _, err := os.Stat(from); err == nil {
			if err := os.Rename(from, to); err != nil {
				return err
			}
		}
	}
	if err := os.Rename(w.path, w.path+".1"); err != nil && !os.IsNotExist(err) {
		return err
	}
	return w.open()
}
func (w *rotatingWriter) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.file == nil {
		return nil
	}
	return w.file.Close()
}

type Metrics struct {
	mu       sync.RWMutex
	started  time.Time
	requests map[string]requestMetric
}
type requestMetric struct {
	Count    uint64
	Duration float64
}

func NewMetrics() *Metrics {
	return &Metrics{started: time.Now().UTC(), requests: map[string]requestMetric{}}
}
func (m *Metrics) Observe(method, path string, status int, duration time.Duration) {
	if m == nil {
		return
	}
	key := method + "\x00" + route(path) + "\x00" + strconv.Itoa(status)
	m.mu.Lock()
	value := m.requests[key]
	value.Count++
	value.Duration += duration.Seconds()
	m.requests[key] = value
	m.mu.Unlock()
}
func (m *Metrics) ServeHTTP(w http.ResponseWriter, _ *http.Request) {
	if m == nil {
		http.Error(w, "metrics unavailable", http.StatusServiceUnavailable)
		return
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	m.mu.RLock()
	keys := make([]string, 0, len(m.requests))
	for key := range m.requests {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	_, _ = fmt.Fprintln(w, "# HELP sypra_process_start_time_seconds Unix time at process start.")
	_, _ = fmt.Fprintln(w, "# TYPE sypra_process_start_time_seconds gauge")
	_, _ = fmt.Fprintf(w, "sypra_process_start_time_seconds %.3f\n", float64(m.started.UnixNano())/1e9)
	_, _ = fmt.Fprintln(w, "# HELP sypra_http_requests_total Completed HTTP requests.")
	_, _ = fmt.Fprintln(w, "# TYPE sypra_http_requests_total counter")
	_, _ = fmt.Fprintln(w, "# HELP sypra_http_request_duration_seconds_total Aggregate completed request duration.")
	_, _ = fmt.Fprintln(w, "# TYPE sypra_http_request_duration_seconds_total counter")
	for _, key := range keys {
		parts := strings.Split(key, "\x00")
		value := m.requests[key]
		labels := fmt.Sprintf(`method=%q,path=%q,status=%q`, parts[0], parts[1], parts[2])
		_, _ = fmt.Fprintf(w, "sypra_http_requests_total{%s} %d\n", labels, value.Count)
		_, _ = fmt.Fprintf(w, "sypra_http_request_duration_seconds_total{%s} %.6f\n", labels, value.Duration)
	}
	m.mu.RUnlock()
}

func route(path string) string {
	for _, prefix := range []string{"/api/v1/chats/", "/api/v1/models/", "/api/v1/rag/collections/", "/api/v1/training/jobs/"} {
		if strings.HasPrefix(path, prefix) {
			return prefix + ":id"
		}
	}
	return path
}

// TraceID preserves a valid W3C trace ID when supplied by a trusted caller or
// creates a new opaque ID. It enables log/API correlation without leaking user
// content into traces.
func TraceID(request *http.Request) string {
	if header := request.Header.Get("traceparent"); len(header) >= 35 {
		parts := strings.Split(header, "-")
		if len(parts) == 4 && len(parts[1]) == 32 {
			return strings.ToLower(parts[1])
		}
	}
	bytes := make([]byte, 16)
	if _, err := rand.Read(bytes); err != nil {
		return fmt.Sprintf("%032x", time.Now().UnixNano())
	}
	return fmt.Sprintf("%x", bytes)
}
