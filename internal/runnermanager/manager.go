// Package runnermanager supervises local Sypra model-runner processes.
package runnermanager

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/sypra-labs/sypra-ai/internal/domain"
)

// Status is the runtime state of a registered local model.
type Status struct {
	ModelID           string     `json:"modelId"`
	BaseModelID       string     `json:"baseModelId,omitempty"`
	Runtime           string     `json:"runtime,omitempty"`
	State             string     `json:"state"`
	Endpoint          string     `json:"endpoint,omitempty"`
	PID               int        `json:"pid,omitempty"`
	Error             string     `json:"error,omitempty"`
	StartedAt         *time.Time `json:"startedAt,omitempty"`
	EstimatedMemoryMB int        `json:"estimatedMemoryMB,omitempty"`
}

type process struct {
	command *exec.Cmd
	status  Status
	stopped bool
}

// Manager supervises native and explicitly configured external local runners.
// Native Qwen uses this executable's runner mode; other formats receive a
// direct trusted executable command and retain process isolation from the API.
type Manager struct {
	mu          sync.RWMutex
	startMu     sync.Mutex
	executable  string
	http        *http.Client
	logger      *slog.Logger
	processes   map[string]*process
	maxActive   int
	gpuBudgetMB int
	statePath   string
}

type Options struct {
	MaxActive   int
	GPUBudgetMB int
	StatePath   string
}

func New(executable string, logger *slog.Logger, options ...Options) *Manager {
	if executable == "" {
		executable, _ = os.Executable()
	}
	if logger == nil {
		logger = slog.Default()
	}
	option := Options{MaxActive: 2}
	if len(options) > 0 {
		option = options[0]
	}
	if option.MaxActive < 1 {
		option.MaxActive = 1
	}
	manager := &Manager{
		executable:  executable,
		http:        &http.Client{Timeout: 800 * time.Millisecond},
		logger:      logger,
		processes:   map[string]*process{},
		maxActive:   option.MaxActive,
		gpuBudgetMB: option.GPUBudgetMB,
		statePath:   option.StatePath,
	}
	manager.loadManifest()
	_ = manager.saveManifestLocked()
	return manager
}

// Start launches the runtime appropriate for model. A fine-tuned model resolves
// its parent checkpoint by the recorded parent path.
func (m *Manager) Start(model domain.Model, models []domain.Model) (Status, error) {
	// Serializing admission and process launch prevents concurrent requests from
	// both observing spare capacity and exceeding the configured ceiling.
	m.startMu.Lock()
	defer m.startMu.Unlock()
	base, err := resolveBase(model, models)
	if err != nil {
		return m.unsupported(model.ID, err)
	}
	m.mu.RLock()
	if existing := m.processes[model.ID]; existing != nil && (existing.status.State == "starting" || existing.status.State == "running") {
		status := existing.status
		m.mu.RUnlock()
		return status, nil
	}
	m.mu.RUnlock()
	plan, err := m.launchPlan(base, addressPlaceholder{})
	if err != nil {
		return m.unsupported(model.ID, err)
	}
	if err := m.admit(base); err != nil {
		return Status{}, err
	}

	address, endpoint, err := reserveLoopbackEndpoint()
	if err != nil {
		return Status{}, err
	}
	started := time.Now().UTC()
	plan, err = m.launchPlan(base, addressPlaceholder{address: address})
	if err != nil {
		return m.unsupported(model.ID, err)
	}
	command := exec.Command(plan.command, plan.args...)
	command.Stdout = os.Stdout
	command.Stderr = os.Stderr
	if err = command.Start(); err != nil {
		return Status{}, fmt.Errorf("start local runner: %w", err)
	}
	status := Status{ModelID: model.ID, BaseModelID: base.ID, Runtime: plan.runtime, State: "starting", Endpoint: endpoint, PID: command.Process.Pid, StartedAt: &started, EstimatedMemoryMB: estimateMemoryMB(base.Path)}
	m.mu.Lock()
	m.processes[model.ID] = &process{command: command, status: status}
	_ = m.saveManifestLocked()
	m.mu.Unlock()
	m.logger.Info("local model runner starting", "modelID", model.ID, "baseModelID", base.ID, "pid", status.PID, "endpoint", endpoint)
	go m.wait(model.ID, command)
	return status, nil
}

func (m *Manager) unsupported(modelID string, err error) (Status, error) {
	status := Status{ModelID: modelID, State: "unsupported", Error: err.Error()}
	m.mu.Lock()
	m.processes[modelID] = &process{status: status}
	_ = m.saveManifestLocked()
	m.mu.Unlock()
	return status, err
}

func resolveBase(model domain.Model, models []domain.Model) (domain.Model, error) {
	if model.Provider != "" {
		return domain.Model{}, fmt.Errorf("cloud models are configured externally and cannot be started locally")
	}
	base := model
	if model.Kind == "slm" {
		for _, candidate := range models {
			if candidate.Kind == "base" && (candidate.ID == model.ParentModel || candidate.Path == model.ParentModel) {
				base = candidate
				break
			}
		}
		if base.ID == model.ID {
			return domain.Model{}, fmt.Errorf("fine-tuned model has no registered parent base model")
		}
	}
	info, err := os.Stat(base.Path)
	if err != nil {
		return domain.Model{}, fmt.Errorf("base model path is unavailable: %w", err)
	}
	if base.Path == "" || (!info.IsDir() && !info.Mode().IsRegular()) {
		return domain.Model{}, fmt.Errorf("base model path must be a model directory or model file")
	}
	return base, nil
}

type addressPlaceholder struct{ address string }

type launchPlan struct {
	runtime string
	command string
	args    []string
}

func (m *Manager) launchPlan(base domain.Model, target addressPlaceholder) (launchPlan, error) {
	format := strings.ToLower(strings.TrimSpace(base.Format))
	family := strings.ToLower(strings.TrimSpace(base.Family))
	if format == "safetensors" || (format == "" && (family == "qwen" || strings.Contains(strings.ToLower(base.Name), "qwen"))) {
		info, err := os.Stat(base.Path)
		if err != nil || !info.IsDir() {
			return launchPlan{}, fmt.Errorf("native Qwen runner requires a SafeTensors model directory")
		}
		return launchPlan{runtime: "native-qwen", command: m.executable, args: []string{"runner", "--model", base.Path, "--listen", target.address, "--backend", "auto"}}, nil
	}
	return launchPlan{}, fmt.Errorf("unsupported local model format %q; the native runner currently supports Qwen SafeTensors checkpoints", base.Format)
}

func reserveLoopbackEndpoint() (address, endpoint string, err error) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", "", err
	}
	address = listener.Addr().String()
	if err = listener.Close(); err != nil {
		return "", "", err
	}
	return address, "http://" + address + "/v1", nil
}

func (m *Manager) wait(modelID string, command *exec.Cmd) {
	err := command.Wait()
	m.mu.Lock()
	defer m.mu.Unlock()
	process := m.processes[modelID]
	if process == nil || process.command != command {
		return
	}
	if process.stopped {
		process.status.State, process.status.Error, process.status.PID = "stopped", "", 0
		_ = m.saveManifestLocked()
		return
	}
	process.status.State, process.status.PID = "failed", 0
	if err != nil {
		process.status.Error = err.Error()
	}
	m.logger.Error("local model runner exited", "modelID", modelID, "error", err)
	_ = m.saveManifestLocked()
}

// Stop terminates a managed process. It intentionally does not attempt to
// stop arbitrary external endpoints.
func (m *Manager) Stop(modelID string) (Status, error) {
	m.mu.Lock()
	process := m.processes[modelID]
	if process == nil {
		m.mu.Unlock()
		return Status{ModelID: modelID, State: "stopped"}, nil
	}
	if process.command == nil || process.command.Process == nil || process.status.State == "stopped" || process.status.State == "unsupported" {
		status := process.status
		m.mu.Unlock()
		return status, nil
	}
	process.stopped = true
	_ = m.saveManifestLocked()
	command := process.command
	m.mu.Unlock()
	if err := command.Process.Kill(); err != nil {
		return Status{}, fmt.Errorf("stop local runner: %w", err)
	}
	// Kill has been accepted by the operating system.  Report the intended
	// terminal state immediately instead of racing the asynchronous Wait call
	// and returning the stale "running" status to the UI or CLI.
	m.mu.Lock()
	if process := m.processes[modelID]; process != nil && process.command == command {
		process.status.State, process.status.Error, process.status.PID = "stopped", "", 0
		_ = m.saveManifestLocked()
		status := process.status
		m.mu.Unlock()
		return status, nil
	}
	m.mu.Unlock()
	return Status{ModelID: modelID, State: "stopped"}, nil
}

// Close stops every runner created by this manager. It never touches an
// arbitrary endpoint: only child processes recorded by Start are terminated.
// The API calls Close as part of its graceful shutdown lifecycle.
func (m *Manager) Close() error {
	m.mu.RLock()
	modelIDs := make([]string, 0, len(m.processes))
	for modelID := range m.processes {
		modelIDs = append(modelIDs, modelID)
	}
	m.mu.RUnlock()

	var failures []error
	for _, modelID := range modelIDs {
		if _, err := m.Stop(modelID); err != nil && !errors.Is(err, os.ErrProcessDone) {
			failures = append(failures, fmt.Errorf("stop runner %s: %w", modelID, err))
		}
	}
	if len(modelIDs) > 0 {
		m.logger.Info("managed local runners stopped", "count", len(modelIDs))
	}
	return errors.Join(failures...)
}

func (m *Manager) Status(modelID string) Status {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if process := m.processes[modelID]; process != nil {
		return process.status
	}
	return Status{ModelID: modelID, State: "stopped"}
}

func (m *Manager) Statuses() []Status {
	m.mu.RLock()
	defer m.mu.RUnlock()
	statuses := make([]Status, 0, len(m.processes))
	for _, process := range m.processes {
		statuses = append(statuses, process.status)
	}
	return statuses
}

// Refresh updates liveness using the runner's OpenAI-compatible models route.
func (m *Manager) Refresh(ctx context.Context) {
	for _, status := range m.Statuses() {
		if status.State != "starting" && status.State != "running" {
			continue
		}
		request, err := http.NewRequestWithContext(ctx, http.MethodGet, status.Endpoint+"/models", nil)
		if err != nil {
			continue
		}
		response, err := m.http.Do(request)
		if response != nil {
			_ = response.Body.Close()
		}
		m.mu.Lock()
		process := m.processes[status.ModelID]
		if process != nil && (process.status.State == "starting" || process.status.State == "running") {
			if err == nil && response.StatusCode >= 200 && response.StatusCode < 300 {
				process.status.State, process.status.Error = "running", ""
			} else if process.status.State == "running" {
				process.status.State = "failed"
				if err != nil {
					process.status.Error = err.Error()
				} else {
					process.status.Error = fmt.Sprintf("runner health check returned %s", response.Status)
				}
			}
			_ = m.saveManifestLocked()
		}
		m.mu.Unlock()
	}
}

// Endpoint returns a healthy managed runner endpoint for modelID.
func (m *Manager) Endpoint(modelID string) (string, bool) {
	status := m.Status(modelID)
	return status.Endpoint, status.State == "running"
}

func (m *Manager) admit(base domain.Model) error {
	return m.admitMemory(estimateMemoryMB(base.Path))
}

func (m *Manager) admitMemory(estimate int) error {
	m.mu.RLock()
	defer m.mu.RUnlock()
	active, reserved := 0, 0
	for _, process := range m.processes {
		if process.status.State == "starting" || process.status.State == "running" {
			active++
			reserved += process.status.EstimatedMemoryMB
		}
	}
	if active >= m.maxActive {
		return fmt.Errorf("runner concurrency limit reached (%d active runner(s))", m.maxActive)
	}
	if m.gpuBudgetMB > 0 && reserved+estimate > m.gpuBudgetMB {
		return fmt.Errorf("runner admission denied: estimated %d MiB would exceed configured GPU budget of %d MiB", reserved+estimate, m.gpuBudgetMB)
	}
	return nil
}

func estimateMemoryMB(path string) int {
	var bytes int64
	_ = filepath.Walk(path, func(name string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() && (strings.HasSuffix(strings.ToLower(name), ".safetensors") || strings.HasSuffix(strings.ToLower(name), ".gguf") || strings.HasSuffix(strings.ToLower(name), ".onnx")) {
			bytes += info.Size()
		}
		return nil
	})
	return int((bytes + (1 << 20) - 1) / (1 << 20))
}

func (m *Manager) loadManifest() {
	if m.statePath == "" {
		return
	}
	payload, err := os.ReadFile(m.statePath)
	if err != nil {
		return
	}
	var statuses []Status
	if json.Unmarshal(payload, &statuses) != nil {
		return
	}
	for _, status := range statuses {
		if status.State == "starting" || status.State == "running" {
			status.State, status.PID, status.Error = "stopped", 0, "API restarted; start the model runner again"
		}
		m.processes[status.ModelID] = &process{status: status}
	}
}

func (m *Manager) saveManifestLocked() error {
	if m.statePath == "" {
		return nil
	}
	statuses := make([]Status, 0, len(m.processes))
	for _, process := range m.processes {
		statuses = append(statuses, process.status)
	}
	payload, err := json.MarshalIndent(statuses, "", "  ")
	if err != nil {
		return err
	}
	if err = os.MkdirAll(filepath.Dir(m.statePath), 0755); err != nil {
		return err
	}
	temporary := m.statePath + ".tmp"
	if err = os.WriteFile(temporary, payload, 0600); err != nil {
		return err
	}
	return os.Rename(temporary, m.statePath)
}
