//go:build !darwin || !cgo

package qwen

import "fmt"

// NewMetalExecutor is available in macOS builds with CGO enabled. Other
// platforms retain the portable CPU executor.
func NewMetalExecutor() (MatrixExecutor, error) {
	return nil, fmt.Errorf("Metal execution is available only in macOS CGO builds")
}
