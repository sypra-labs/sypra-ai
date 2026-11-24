package agent

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

var errOutputLimit = errors.New("tool output exceeded configured limit")

// sandboxCommand establishes the portable isolation boundary available without
// requiring a container runtime: no shell, a fresh working directory, a small
// inherited environment, direct arguments only, a process timeout supplied by
// the caller, and bounded output. OS-level container/VM isolation can be
// layered outside Sypra for untrusted code.
func (s *Service) sandboxCommand(contextCommand *exec.Cmd) (func(), error) {
	root := s.WorkRoot
	if root != "" {
		if err := os.MkdirAll(root, 0700); err != nil {
			return nil, fmt.Errorf("create agent work root: %w", err)
		}
	}
	directory, err := os.MkdirTemp(root, "sypra-tool-")
	if err != nil {
		return nil, fmt.Errorf("create isolated tool directory: %w", err)
	}
	contextCommand.Dir = directory
	contextCommand.Env = safeEnvironment()
	return func() { _ = os.RemoveAll(directory) }, nil
}

func safeEnvironment() []string {
	values := []string{}
	for _, key := range []string{"PATH", "SystemRoot", "WINDIR", "LANG", "LC_ALL", "TZ"} {
		if value := os.Getenv(key); value != "" {
			values = append(values, key+"="+value)
		}
	}
	return values
}

func (s *Service) trustedExecutable(command string, allowed map[string]bool, kind string) (string, error) {
	command = strings.TrimSpace(command)
	if command == "" {
		return "", fmt.Errorf("%s command is required", kind)
	}
	clean := filepath.Clean(command)
	if allowed[strings.ToLower(clean)] {
		return command, nil
	}
	resolved, err := exec.LookPath(command)
	if err != nil {
		return "", fmt.Errorf("resolve %s command %q: %w", kind, command, err)
	}
	if allowed[strings.ToLower(filepath.Clean(resolved))] {
		return resolved, nil
	}
	return "", fmt.Errorf("%s command %q is not trusted; add its exact name or path to the configured allowlist", kind, command)
}

type boundedBuffer struct {
	bytes.Buffer
	limit    int
	exceeded bool
}

func (b *boundedBuffer) Write(value []byte) (int, error) {
	remaining := b.limit - b.Len()
	if remaining <= 0 {
		b.exceeded = true
		return 0, errOutputLimit
	}
	if len(value) > remaining {
		b.exceeded = true
		_, _ = b.Buffer.Write(value[:remaining])
		return remaining, errOutputLimit
	}
	return b.Buffer.Write(value)
}
