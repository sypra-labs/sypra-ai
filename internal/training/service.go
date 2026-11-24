package training

import (
	"bufio"
	"context"
	"fmt"
	"github.com/sypra-labs/sypra-ai/internal/adapter"
	"github.com/sypra-labs/sypra-ai/internal/domain"
	"github.com/sypra-labs/sypra-ai/internal/finetune"
	"github.com/sypra-labs/sypra-ai/internal/store"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

type Service struct {
	Store         *store.Store
	mu            sync.Mutex
	running       map[string]context.CancelFunc
	maxConcurrent int
}

func New(st *store.Store, limits ...int) *Service {
	maxConcurrent := 1
	if len(limits) > 0 && limits[0] > 0 {
		maxConcurrent = limits[0]
	}
	return &Service{Store: st, running: map[string]context.CancelFunc{}, maxConcurrent: maxConcurrent}
}
func (s *Service) Start(ctx context.Context, j domain.TrainingJob, command string) error {
	if command == "" {
		return fmt.Errorf("training command is required (for example: python train.py --config config.json)")
	}
	if j.Method != "lora" && j.Method != "qlora" && j.Method != "full" {
		return fmt.Errorf("method must be lora, qlora, or full")
	}
	runCtx, release, err := s.reserve(j.ID)
	if err != nil {
		return err
	}
	if e := s.Store.AddTrainingJob(ctx, j); e != nil {
		release()
		return e
	}
	go s.run(runCtx, j, command)
	return nil
}

// StartNative executes Sypra's built-in CPU adapter trainer. It is intentionally
// command-free, so UI and CLI jobs have identical lifecycle and metrics.
func (s *Service) StartNative(ctx context.Context, j domain.TrainingJob, cfg finetune.Config, directory string) error {
	if j.Method != "lora" && j.Method != "qlora" {
		return fmt.Errorf("native training supports lora or qlora")
	}
	if directory == "" {
		return fmt.Errorf("native run directory is required")
	}
	run := finetune.Run{ID: j.ID, DatasetPath: j.DatasetPath, Directory: directory, Config: cfg}
	configPath, err := run.Prepare()
	if err != nil {
		return err
	}
	if j.Parameters == nil {
		j.Parameters = map[string]any{}
	}
	j.Parameters["runDirectory"] = run.Directory
	j.Parameters["configPath"] = configPath
	runCtx, release, err := s.reserve(j.ID)
	if err != nil {
		return err
	}
	if err = s.Store.AddTrainingJob(ctx, j); err != nil {
		release()
		return err
	}
	go s.runNative(runCtx, j, cfg, filepath.Join(run.Directory, "artifacts"))
	return nil
}

func (s *Service) reserve(id string) (context.Context, func(), error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.running[id]; exists {
		return nil, nil, fmt.Errorf("training job %s is already running", id)
	}
	if len(s.running) >= s.maxConcurrent {
		return nil, nil, fmt.Errorf("training concurrency limit reached (%d active job(s))", s.maxConcurrent)
	}
	ctx, cancel := context.WithCancel(context.Background())
	s.running[id] = cancel
	return ctx, func() {
		s.mu.Lock()
		delete(s.running, id)
		s.mu.Unlock()
	}, nil
}

func (s *Service) runNative(ctx context.Context, j domain.TrainingJob, cfg finetune.Config, artifactDir string) {
	now := store.Now()
	j.Status, j.Phase, j.StartedAt = "running", "initializing", &now
	_ = s.Store.UpdateJob(context.Background(), j)
	train := finetune.TrainNative
	if info, statErr := os.Stat(filepath.Join(cfg.ModelDir, "config.json")); statErr == nil && !info.IsDir() {
		train = finetune.TrainQwenTransformerLoRA
	}
	lastPersist := time.Time{}
	result, err := train(ctx, j.DatasetPath, artifactDir, cfg, func(metric finetune.Progress) {
		j.Progress = metric.Progress
		j.Phase, j.Epoch, j.Epochs = metric.Phase, metric.Epoch, metric.Epochs
		j.Step, j.Steps, j.Tokens, j.TokensPerSecond = metric.Step, metric.Steps, metric.Tokens, metric.TokensPerSecond
		trainLoss := metric.TrainLoss
		j.TrainLoss = &trainLoss
		if metric.ValidationComplete {
			evalLoss := metric.EvalLoss
			j.EvalLoss = &evalLoss
		}
		if !metric.ValidationComplete && time.Since(lastPersist) < time.Second {
			return
		}
		lastPersist = time.Now()
		_ = s.Store.UpdateJob(context.Background(), j)
	})
	now = store.Now()
	j.FinishedAt = &now
	if ctx.Err() != nil {
		j.Status, j.Phase, j.Error = "cancelled", "cancelled", "cancelled"
	} else if err != nil {
		j.Status, j.Phase, j.Error = "failed", "failed", err.Error()
	} else {
		j.Status, j.Phase, j.Progress = "completed", "completed", 1
		trainLoss, evalLoss := result.TrainLoss, result.EvalLoss
		j.TrainLoss, j.EvalLoss = &trainLoss, &evalLoss
		j.Parameters["testLoss"] = result.TestLoss
		endpoint := "native://" + filepath.ToSlash(result.ArtifactPath)
		if result.ArtifactFormat == adapter.QwenLMHeadLoRAType || result.ArtifactFormat == adapter.TransformerLoRAType {
			endpoint = ""
		}
		model := domain.Model{ID: j.ID + "-adapter", Name: j.Name, Path: result.ArtifactPath, Format: result.ArtifactFormat, Endpoint: endpoint, Kind: "slm", ParentModel: j.BaseModel, Method: j.Method, Version: j.ID, Status: "candidate", Metrics: map[string]float64{"trainLoss": result.TrainLoss, "validationLoss": result.EvalLoss, "testLoss": result.TestLoss}, CreatedAt: now}
		if _, addErr := s.Store.AddModel(context.Background(), model); addErr != nil {
			j.Status, j.Error = "failed", "register fine-tuned SLM: "+addErr.Error()
		} else {
			j.Parameters["modelID"] = model.ID
		}
	}
	_ = s.Store.UpdateJob(context.Background(), j)
	s.mu.Lock()
	delete(s.running, j.ID)
	s.mu.Unlock()
}
func (s *Service) run(ctx context.Context, j domain.TrainingJob, command string) {
	now := store.Now()
	j.Status = "running"
	j.StartedAt = &now
	_ = s.Store.UpdateJob(context.Background(), j)
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.CommandContext(ctx, "cmd", "/C", command)
	} else {
		cmd = exec.CommandContext(ctx, "sh", "-c", command)
	}
	stdout, e := cmd.StdoutPipe()
	if e != nil {
		s.fail(&j, e)
		return
	}
	cmd.Stderr = cmd.Stdout
	if e = cmd.Start(); e != nil {
		s.fail(&j, e)
		return
	}
	scan := bufio.NewScanner(stdout)
	for scan.Scan() {
		s.metric(&j, scan.Text())
	}
	e = cmd.Wait()
	now = store.Now()
	j.FinishedAt = &now
	if ctx.Err() != nil {
		j.Status = "cancelled"
		j.Error = "cancelled"
	} else if e != nil {
		j.Status = "failed"
		j.Error = e.Error()
	} else {
		j.Status = "completed"
		j.Progress = 1
	}
	_ = s.Store.UpdateJob(context.Background(), j)
	s.mu.Lock()
	delete(s.running, j.ID)
	s.mu.Unlock()
}

var progressRE = regexp.MustCompile(`(?:progress|percent)\s*[=:]\s*(\d+(?:\.\d+)?)%?`)
var trainRE = regexp.MustCompile(`(?:train_?loss|loss)\s*[=:]\s*(\d+(?:\.\d+)?)`)
var evalRE = regexp.MustCompile(`(?:eval_?loss|validation_?loss)\s*[=:]\s*(\d+(?:\.\d+)?)`)

func (s *Service) metric(j *domain.TrainingJob, line string) {
	line = strings.ToLower(line)
	if m := progressRE.FindStringSubmatch(line); len(m) > 1 {
		v, _ := strconv.ParseFloat(m[1], 64)
		if v > 1 {
			v /= 100
		}
		j.Progress = v
	}
	if m := trainRE.FindStringSubmatch(line); len(m) > 1 {
		v, _ := strconv.ParseFloat(m[1], 64)
		j.TrainLoss = &v
	}
	if m := evalRE.FindStringSubmatch(line); len(m) > 1 {
		v, _ := strconv.ParseFloat(m[1], 64)
		j.EvalLoss = &v
	}
	_ = s.Store.UpdateJob(context.Background(), *j)
}
func (s *Service) fail(j *domain.TrainingJob, e error) {
	now := store.Now()
	j.Status = "failed"
	j.Error = e.Error()
	j.FinishedAt = &now
	_ = s.Store.UpdateJob(context.Background(), *j)
}
func (s *Service) Cancel(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if f := s.running[id]; f != nil {
		f()
		return true
	}
	return false
}

// Recover marks jobs that were executing when the API stopped as interrupted.
// The in-memory process and cancellation context no longer exist after a
// crash, so silently reporting such work as running would be misleading. The
// persisted run directory/checkpoint remains available for a later resume.
func (s *Service) Recover(ctx context.Context) (int, error) {
	jobs, err := s.Store.Jobs(ctx)
	if err != nil {
		return 0, err
	}
	recovered := 0
	for _, job := range jobs {
		if job.Status != "running" && job.Status != "starting" {
			continue
		}
		now := store.Now()
		job.Status, job.Phase, job.Error, job.FinishedAt = "interrupted", "recovery", "API restarted before this job completed; resume from its checkpoint if available", &now
		if err := s.Store.UpdateJob(ctx, job); err != nil {
			return recovered, err
		}
		recovered++
	}
	return recovered, nil
}

// Wait blocks until a queued/running job reaches a terminal state. The API
// does not use this, while the foreground CLI uses it so its child goroutine
// is not terminated when the command process exits.
func (s *Service) Wait(ctx context.Context, id string) (domain.TrainingJob, error) {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		jobs, err := s.Store.Jobs(ctx)
		if err != nil {
			return domain.TrainingJob{}, err
		}
		for _, job := range jobs {
			if job.ID != id {
				continue
			}
			if job.Status == "completed" || job.Status == "failed" || job.Status == "cancelled" {
				return job, nil
			}
			break
		}
		select {
		case <-ctx.Done():
			return domain.TrainingJob{}, ctx.Err()
		case <-ticker.C:
		}
	}
}
