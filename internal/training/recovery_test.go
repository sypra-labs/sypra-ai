package training

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/sypra-labs/sypra-ai/internal/domain"
	"github.com/sypra-labs/sypra-ai/internal/store"
)

func TestRecoverMarksOrphanedJobsInterrupted(t *testing.T) {
	ctx := context.Background()
	st, err := store.Open(filepath.Join(t.TempDir(), "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = st.AddTrainingJob(ctx, domain.TrainingJob{ID: "orphaned", Name: "Interrupted", Status: "running", CreatedAt: store.Now()}); err != nil {
		t.Fatal(err)
	}
	service := New(st, 1)
	count, err := service.Recover(ctx)
	if err != nil || count != 1 {
		t.Fatalf("recovered=%d err=%v", count, err)
	}
	jobs, err := st.Jobs(ctx)
	if err != nil || len(jobs) != 1 || jobs[0].Status != "interrupted" || jobs[0].FinishedAt == nil {
		t.Fatalf("jobs=%#v err=%v", jobs, err)
	}
}

func TestTrainingAdmissionLimit(t *testing.T) {
	st, err := store.Open(filepath.Join(t.TempDir(), "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	service := New(st, 1)
	_, release, err := service.reserve("first")
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	if _, _, err = service.reserve("second"); err == nil {
		t.Fatal("second job bypassed the configured concurrency limit")
	}
}
