package runnermanager

import (
	"os"
	"os/exec"
	"testing"
	"time"
)

func TestCloseStopsManagedChild(t *testing.T) {
	command := exec.Command(os.Args[0], "-test.run=TestRunnerManagerHelper", "--")
	command.Env = append(os.Environ(), "SYPRA_RUNNER_MANAGER_HELPER=1")
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	manager := New("", nil)
	manager.processes["test"] = &process{command: command, status: Status{ModelID: "test", State: "running", PID: command.Process.Pid}}
	go manager.wait("test", command)
	if err := manager.Close(); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if manager.Status("test").State == "stopped" {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("runner state = %q, want stopped", manager.Status("test").State)
}

func TestStopReportsStoppedImmediately(t *testing.T) {
	command := exec.Command(os.Args[0], "-test.run=TestRunnerManagerHelper", "--")
	command.Env = append(os.Environ(), "SYPRA_RUNNER_MANAGER_HELPER=1")
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	manager := New("", nil)
	manager.processes["test"] = &process{command: command, status: Status{ModelID: "test", State: "running", PID: command.Process.Pid}}
	go manager.wait("test", command)

	status, err := manager.Stop("test")
	if err != nil {
		t.Fatal(err)
	}
	if status.State != "stopped" {
		t.Fatalf("stop state = %q, want stopped", status.State)
	}
}

func TestRunnerManagerHelper(t *testing.T) {
	if os.Getenv("SYPRA_RUNNER_MANAGER_HELPER") != "1" {
		return
	}
	for {
		time.Sleep(time.Second)
	}
}
