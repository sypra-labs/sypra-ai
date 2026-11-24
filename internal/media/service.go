// Package media runs optional local multimodal workflows while persisting
// progress/status in the workspace. It deliberately shells out only to tools
// the user configured or installed locally; Sypra never uploads media itself.
package media

import (
	"context"
	"crypto/rand"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	"github.com/sypra-labs/sypra-ai/internal/domain"
	"github.com/sypra-labs/sypra-ai/internal/store"
)

type Service struct {
	Store  *store.Store
	Root   string
	mu     sync.Mutex
	cancel map[string]context.CancelFunc
}

func New(st *store.Store, root string) *Service {
	return &Service{Store: st, Root: root, cancel: map[string]context.CancelFunc{}}
}

func (s *Service) Start(ctx context.Context, kind, prompt, inputPath, inputName string) (domain.MediaJob, error) {
	kind = strings.ToLower(strings.TrimSpace(kind))
	if !allowed(kind) {
		return domain.MediaJob{}, fmt.Errorf("unsupported media job kind %q", kind)
	}
	if (kind == "image_generation" || kind == "image_edit") && strings.TrimSpace(prompt) == "" {
		return domain.MediaJob{}, fmt.Errorf("an image prompt is required")
	}
	if kind != "image_generation" && strings.TrimSpace(inputPath) == "" {
		return domain.MediaJob{}, fmt.Errorf("an input file is required")
	}
	if inputPath != "" {
		if _, err := os.Stat(inputPath); err != nil {
			return domain.MediaJob{}, fmt.Errorf("input file: %w", err)
		}
	}
	if err := os.MkdirAll(s.Root, 0755); err != nil {
		return domain.MediaJob{}, err
	}
	now := store.Now()
	job := domain.MediaJob{ID: jobID(), Kind: kind, Prompt: prompt, InputPath: inputPath, InputName: inputName, Status: "queued", Phase: "queued", CreatedAt: now}
	if err := s.Store.AddMediaJob(ctx, job); err != nil {
		return domain.MediaJob{}, err
	}
	runCtx, cancel := context.WithCancel(context.Background())
	s.mu.Lock()
	s.cancel[job.ID] = cancel
	s.mu.Unlock()
	go s.run(runCtx, job)
	return job, nil
}

func (s *Service) Cancel(ctx context.Context, id string) (domain.MediaJob, error) {
	job, err := s.Store.MediaJob(ctx, id)
	if err != nil {
		return domain.MediaJob{}, err
	}
	s.mu.Lock()
	cancel := s.cancel[id]
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	if job.Status == "queued" || job.Status == "running" {
		now := store.Now()
		job.Status, job.Phase, job.FinishedAt = "cancelled", "cancelled", &now
		_ = s.Store.UpdateMediaJob(ctx, job)
	}
	return job, nil
}

func (s *Service) run(ctx context.Context, job domain.MediaJob) {
	defer func() { s.mu.Lock(); delete(s.cancel, job.ID); s.mu.Unlock() }()
	now := store.Now()
	job.Status, job.Phase, job.Progress, job.StartedAt = "running", "processing", .1, &now
	_ = s.Store.UpdateMediaJob(context.Background(), job)
	var err error
	switch job.Kind {
	case "ocr":
		err = s.extractText(ctx, &job, "SYPRA_MEDIA_OCR_COMMAND", []string{"tesseract", "{input}", "stdout"})
	case "transcription":
		err = s.extractText(ctx, &job, "SYPRA_MEDIA_TRANSCRIBE_COMMAND", nil)
	case "image_generation":
		err = s.generateImage(ctx, &job, "SYPRA_IMAGE_GENERATE_COMMAND")
	case "image_edit":
		err = s.generateImage(ctx, &job, "SYPRA_IMAGE_EDIT_COMMAND")
	}
	now = store.Now()
	if ctx.Err() != nil {
		job.Status, job.Phase, job.Error = "cancelled", "cancelled", ""
	} else if err != nil {
		job.Status, job.Phase, job.Error = "failed", "failed", err.Error()
	} else {
		job.Status, job.Phase, job.Progress = "completed", "completed", 1
	}
	job.FinishedAt = &now
	_ = s.Store.UpdateMediaJob(context.Background(), job)
}

func (s *Service) extractText(ctx context.Context, job *domain.MediaJob, environment string, fallback []string) error {
	output, err := run(ctx, environment, fallback, job.Prompt, job.InputPath, "")
	if err != nil {
		return err
	}
	if strings.TrimSpace(string(output)) == "" {
		return fmt.Errorf("%s returned no text", job.Kind)
	}
	path := filepath.Join(s.Root, job.ID+".txt")
	if err := os.WriteFile(path, output, 0600); err != nil {
		return err
	}
	job.OutputPath, job.OutputURL, job.OutputMIME, job.Text, job.Progress = path, "/api/v1/media/files/"+filepath.Base(path), "text/plain", string(output), .9
	return s.Store.UpdateMediaJob(context.Background(), *job)
}

func (s *Service) generateImage(ctx context.Context, job *domain.MediaJob, environment string) error {
	path := filepath.Join(s.Root, job.ID+".png")
	if _, err := run(ctx, environment, nil, job.Prompt, job.InputPath, path); err != nil {
		return err
	}
	if info, err := os.Stat(path); err != nil || info.Size() == 0 {
		return fmt.Errorf("image command did not create %s", filepath.Base(path))
	}
	job.OutputPath, job.OutputURL, job.OutputMIME, job.Progress = path, "/api/v1/media/files/"+filepath.Base(path), "image/png", .9
	return s.Store.UpdateMediaJob(context.Background(), *job)
}

func run(ctx context.Context, environment string, fallback []string, prompt, input, output string) ([]byte, error) {
	args := fallback
	if configured := strings.TrimSpace(os.Getenv(environment)); configured != "" {
		args = strings.Fields(configured)
	}
	if len(args) == 0 {
		return nil, fmt.Errorf("no local workflow configured; set %s", environment)
	}
	for index, value := range args {
		args[index] = strings.NewReplacer("{prompt}", prompt, "{input}", input, "{output}", output).Replace(value)
	}
	if _, err := exec.LookPath(args[0]); err != nil {
		return nil, fmt.Errorf("workflow command %q is unavailable", args[0])
	}
	outputBytes, err := exec.CommandContext(ctx, args[0], args[1:]...).CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("workflow failed: %w: %s", err, strings.TrimSpace(string(outputBytes)))
	}
	return outputBytes, nil
}

func allowed(kind string) bool {
	return kind == "ocr" || kind == "transcription" || kind == "image_generation" || kind == "image_edit"
}
func jobID() string {
	bytes := make([]byte, 16)
	_, _ = rand.Read(bytes)
	return fmt.Sprintf("%x-%x-%x-%x-%x", bytes[0:4], bytes[4:6], bytes[6:8], bytes[8:10], bytes[10:])
}
