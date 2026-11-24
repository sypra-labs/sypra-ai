//go:build darwin && cgo

package qwen

/*
#cgo LDFLAGS: -framework Metal -framework Foundation
#include <stdint.h>
#include <stddef.h>
int sypra_metal_create(void **context);
int sypra_metal_upload(void *context, const float *values, size_t count, void **buffer);
int sypra_metal_run(void *context, void *weights, unsigned int rows, unsigned int cols, const float *input, size_t inputCount, int transpose, float *output, size_t outputCount);
void sypra_metal_buffer_free(void *buffer);
void sypra_metal_destroy(void *context);
*/
import "C"

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// metalExecutor compiles the small projection kernels through the system Metal
// framework. It keeps immutable Qwen projections resident in Metal buffers
// while inputs and outputs remain short-lived host-visible buffers.
type metalExecutor struct {
	context  unsafe.Pointer
	matrices map[string]metalMatrix
	mu       sync.Mutex
}

type metalMatrix struct {
	buffer     unsafe.Pointer
	rows, cols int
}

func NewMetalExecutor() (MatrixExecutor, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	var context unsafe.Pointer
	if code := C.sypra_metal_create(&context); code != 0 {
		return nil, metalError("create Metal device", code)
	}
	return &metalExecutor{context: context, matrices: map[string]metalMatrix{}}, nil
}

func (e *metalExecutor) MatrixVec(model *Model, name string, input []float32) ([]float32, error) {
	return e.matrixVec(model, name, input, false)
}

func (e *metalExecutor) MatrixVecTranspose(model *Model, name string, input []float32) ([]float32, error) {
	return e.matrixVec(model, name, input, true)
}

func (e *metalExecutor) matrixVec(model *Model, name string, input []float32, transpose bool) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.context == nil {
		return nil, fmt.Errorf("Metal executor is closed")
	}
	matrix, err := e.matrix(model, name)
	if err != nil {
		return nil, err
	}
	expected := matrix.cols
	if transpose {
		expected = matrix.rows
	}
	if len(input) != expected {
		return nil, fmt.Errorf("Metal matrix %s requires %d values, got %d", name, expected, len(input))
	}
	outputCount := matrix.rows
	if transpose {
		outputCount = matrix.cols
	}
	output := make([]float32, outputCount)
	if code := C.sypra_metal_run(e.context, matrix.buffer, C.uint(matrix.rows), C.uint(matrix.cols), (*C.float)(unsafe.Pointer(&input[0])), C.size_t(len(input)), boolToCInt(transpose), (*C.float)(unsafe.Pointer(&output[0])), C.size_t(len(output))); code != 0 {
		return nil, metalError("run Metal matrix kernel", code)
	}
	runtime.KeepAlive(input)
	runtime.KeepAlive(output)
	return output, nil
}

func (e *metalExecutor) matrix(model *Model, name string) (metalMatrix, error) {
	if matrix, ok := e.matrices[name]; ok {
		return matrix, nil
	}
	rows, cols, values, err := acceleratorMatrixValues(model, name)
	if err != nil {
		return metalMatrix{}, err
	}
	var buffer unsafe.Pointer
	if code := C.sypra_metal_upload(e.context, (*C.float)(unsafe.Pointer(&values[0])), C.size_t(len(values)), &buffer); code != 0 {
		return metalMatrix{}, metalError("upload Metal matrix", code)
	}
	matrix := metalMatrix{buffer: buffer, rows: rows, cols: cols}
	e.matrices[name] = matrix
	runtime.KeepAlive(values)
	return matrix, nil
}

func (e *metalExecutor) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.context == nil {
		return nil
	}
	for _, matrix := range e.matrices {
		C.sypra_metal_buffer_free(matrix.buffer)
	}
	e.matrices = map[string]metalMatrix{}
	C.sypra_metal_destroy(e.context)
	e.context = nil
	return nil
}

func boolToCInt(value bool) C.int {
	if value {
		return 1
	}
	return 0
}

func metalError(action string, code C.int) error {
	return fmt.Errorf("%s: Metal runtime result %d", action, int(code))
}
