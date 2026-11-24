//go:build !windows && !(linux && cgo)

package qwen

import "fmt"

// NewCUDAExecutor is unavailable when this build has no native CUDA binding.
// The CPU backend remains the automatic fallback.
func NewCUDAExecutor() (MatrixExecutor, error) {
	return nil, fmt.Errorf("CUDA driver execution is unavailable in this build")
}

func NewCUDAExecutorForDevice(_ int) (MatrixExecutor, error) { return NewCUDAExecutor() }

func CUDADeviceCount() (int, error) { return 0, fmt.Errorf("CUDA is unavailable in this build") }
