//go:build windows

package qwen

import (
	"fmt"
	"runtime"
	"sync"
	"syscall"
	"unsafe"
)

// cudaExecutor uses the installed NVIDIA driver API directly. It deliberately
// avoids CGO and the CUDA toolkit so a Windows host with a compatible NVIDIA
// driver can accelerate frozen projection products without adding a Python or
// GoMLX runtime dependency.
type cudaExecutor struct {
	driver                 *syscall.LazyDLL
	init, deviceGet        *syscall.LazyProc
	ctxCreate, ctxSet      *syscall.LazyProc
	ctxDestroy             *syscall.LazyProc
	moduleLoad, moduleFunc *syscall.LazyProc
	memAlloc, memFree      *syscall.LazyProc
	copyHtoD, copyDtoH     *syscall.LazyProc
	launch                 *syscall.LazyProc
	context, module        uintptr
	matvec, transpose      uintptr
	matrices               map[string]cudaMatrix
	mu                     sync.Mutex
}

type cudaMatrix struct {
	device     uintptr
	rows, cols int
}

func NewCUDAExecutor() (MatrixExecutor, error) {
	return NewCUDAExecutorForDevice(0)
}

func NewCUDAExecutorForDevice(deviceOrdinal int) (MatrixExecutor, error) {
	if deviceOrdinal < 0 {
		return nil, fmt.Errorf("CUDA device ordinal must be non-negative")
	}
	// CUDA driver contexts are bound to OS threads. Keep construction on one
	// thread and detach it before returning so later Go worker goroutines can
	// make the context current on their own locked thread.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	driver := syscall.NewLazyDLL("nvcuda.dll")
	executor := &cudaExecutor{
		driver:     driver,
		init:       driver.NewProc("cuInit"),
		deviceGet:  driver.NewProc("cuDeviceGet"),
		ctxCreate:  driver.NewProc("cuCtxCreate_v2"),
		ctxSet:     driver.NewProc("cuCtxSetCurrent"),
		ctxDestroy: driver.NewProc("cuCtxDestroy_v2"),
		moduleLoad: driver.NewProc("cuModuleLoadData"),
		moduleFunc: driver.NewProc("cuModuleGetFunction"),
		memAlloc:   driver.NewProc("cuMemAlloc_v2"),
		memFree:    driver.NewProc("cuMemFree_v2"),
		copyHtoD:   driver.NewProc("cuMemcpyHtoD_v2"),
		copyDtoH:   driver.NewProc("cuMemcpyDtoH_v2"),
		launch:     driver.NewProc("cuLaunchKernel"),
		matrices:   map[string]cudaMatrix{},
	}
	if err := executor.call(executor.init, 0); err != nil {
		return nil, fmt.Errorf("initialize CUDA driver: %w", err)
	}
	var device int32
	if err := executor.call(executor.deviceGet, uintptr(unsafe.Pointer(&device)), uintptr(deviceOrdinal)); err != nil {
		return nil, fmt.Errorf("select CUDA device: %w", err)
	}
	if err := executor.call(executor.ctxCreate, uintptr(unsafe.Pointer(&executor.context)), 0, uintptr(device)); err != nil {
		return nil, fmt.Errorf("create CUDA context: %w", err)
	}
	ptx := append([]byte(cudaMatrixPTX), 0)
	if err := executor.call(executor.moduleLoad, uintptr(unsafe.Pointer(&executor.module)), uintptr(unsafe.Pointer(&ptx[0]))); err != nil {
		_ = executor.Close()
		return nil, fmt.Errorf("load CUDA matrix kernels: %w", err)
	}
	if err := executor.loadFunction(&executor.matvec, "sypra_matvec_f32"); err != nil {
		_ = executor.Close()
		return nil, err
	}
	if err := executor.loadFunction(&executor.transpose, "sypra_matvec_transpose_f32"); err != nil {
		_ = executor.Close()
		return nil, err
	}
	if err := executor.clearCurrent(); err != nil {
		_ = executor.Close()
		return nil, fmt.Errorf("detach CUDA context: %w", err)
	}
	return executor, nil
}

func CUDADeviceCount() (int, error) {
	driver := syscall.NewLazyDLL("nvcuda.dll")
	init := driver.NewProc("cuInit")
	count := driver.NewProc("cuDeviceGetCount")
	result, _, callErr := init.Call(0)
	if result != 0 {
		return 0, fmt.Errorf("initialize CUDA driver result %d: %v", result, callErr)
	}
	var devices int32
	result, _, callErr = count.Call(uintptr(unsafe.Pointer(&devices)))
	if result != 0 {
		return 0, fmt.Errorf("enumerate CUDA devices result %d: %v", result, callErr)
	}
	return int(devices), nil
}

func (e *cudaExecutor) MatrixVec(model *Model, name string, input []float32) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := e.makeCurrent(); err != nil {
		return nil, err
	}
	defer e.clearCurrent()
	matrix, err := e.matrix(model, name)
	if err != nil {
		return nil, fmt.Errorf("CUDA cache matrix %s: %w", name, err)
	}
	if len(input) != matrix.cols {
		return nil, fmt.Errorf("CUDA matrix %s has %d columns, got %d input values", name, matrix.cols, len(input))
	}
	output, err := e.run(matrix, input, false)
	if err != nil {
		return nil, fmt.Errorf("CUDA matvec %s: %w", name, err)
	}
	return output, nil
}

func (e *cudaExecutor) MatrixVecTranspose(model *Model, name string, outputGradient []float32) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := e.makeCurrent(); err != nil {
		return nil, err
	}
	defer e.clearCurrent()
	matrix, err := e.matrix(model, name)
	if err != nil {
		return nil, fmt.Errorf("CUDA cache matrix %s: %w", name, err)
	}
	if len(outputGradient) != matrix.rows {
		return nil, fmt.Errorf("CUDA matrix %s has %d rows, got %d output gradients", name, matrix.rows, len(outputGradient))
	}
	output, err := e.run(matrix, outputGradient, true)
	if err != nil {
		return nil, fmt.Errorf("CUDA transpose matvec %s: %w", name, err)
	}
	return output, nil
}

func (e *cudaExecutor) matrix(model *Model, name string) (cudaMatrix, error) {
	if matrix, ok := e.matrices[name]; ok {
		return matrix, nil
	}
	var rows, cols int
	var values []float32
	if model.useNF4 {
		quantized, err := model.nf4(name)
		if err != nil {
			return cudaMatrix{}, err
		}
		if len(quantized.shape) != 2 {
			return cudaMatrix{}, fmt.Errorf("CUDA target %s is not a matrix", name)
		}
		rows, cols = quantized.shape[0], quantized.shape[1]
		values = make([]float32, rows*cols)
		for index := range values {
			values[index] = quantized.value(index)
		}
	} else {
		w, err := model.tensor(name)
		if err != nil {
			return cudaMatrix{}, err
		}
		if len(w.shape) != 2 {
			return cudaMatrix{}, fmt.Errorf("CUDA target %s is not a matrix", name)
		}
		rows, cols = w.shape[0], w.shape[1]
		values = make([]float32, rows*cols)
		for index := range values {
			values[index] = bfloat(w.data, index)
		}
	}
	var device uintptr
	if err := e.call(e.memAlloc, uintptr(unsafe.Pointer(&device)), uintptr(len(values)*4)); err != nil {
		return cudaMatrix{}, fmt.Errorf("allocate %d-byte matrix buffer: %w", len(values)*4, err)
	}
	if err := e.copyHostToDevice(device, values); err != nil {
		_ = e.call(e.memFree, device)
		return cudaMatrix{}, fmt.Errorf("upload matrix buffer: %w", err)
	}
	matrix := cudaMatrix{device: device, rows: rows, cols: cols}
	e.matrices[name] = matrix
	return matrix, nil
}

func (e *cudaExecutor) run(matrix cudaMatrix, input []float32, transpose bool) ([]float32, error) {
	inputBytes := len(input) * 4
	outputCount := matrix.rows
	function := e.matvec
	if transpose {
		outputCount, function = matrix.cols, e.transpose
	}
	var deviceInput, deviceOutput uintptr
	if err := e.call(e.memAlloc, uintptr(unsafe.Pointer(&deviceInput)), uintptr(inputBytes)); err != nil {
		return nil, fmt.Errorf("allocate %d-byte input buffer: %w", inputBytes, err)
	}
	defer e.call(e.memFree, deviceInput)
	if err := e.call(e.memAlloc, uintptr(unsafe.Pointer(&deviceOutput)), uintptr(outputCount*4)); err != nil {
		return nil, fmt.Errorf("allocate %d-byte output buffer: %w", outputCount*4, err)
	}
	defer e.call(e.memFree, deviceOutput)
	if err := e.copyHostToDevice(deviceInput, input); err != nil {
		return nil, fmt.Errorf("upload input buffer: %w", err)
	}
	rows, cols := uint32(matrix.rows), uint32(matrix.cols)
	params := []unsafe.Pointer{unsafe.Pointer(&matrix.device), unsafe.Pointer(&deviceInput), unsafe.Pointer(&deviceOutput), unsafe.Pointer(&rows), unsafe.Pointer(&cols)}
	grid := uintptr((outputCount + 255) / 256)
	if err := e.call(e.launch, function, grid, 1, 1, 256, 1, 1, 0, 0, uintptr(unsafe.Pointer(&params[0])), 0); err != nil {
		return nil, fmt.Errorf("launch kernel: %w", err)
	}
	output := make([]float32, outputCount)
	if err := e.copyDeviceToHost(output, deviceOutput); err != nil {
		return nil, fmt.Errorf("download output buffer: %w", err)
	}
	runtime.KeepAlive(params)
	runtime.KeepAlive(input)
	return output, nil
}

func (e *cudaExecutor) loadFunction(destination *uintptr, name string) error {
	bytes := append([]byte(name), 0)
	if err := e.call(e.moduleFunc, uintptr(unsafe.Pointer(destination)), e.module, uintptr(unsafe.Pointer(&bytes[0]))); err != nil {
		return fmt.Errorf("load CUDA kernel %s: %w", name, err)
	}
	return nil
}

func (e *cudaExecutor) copyHostToDevice(destination uintptr, values []float32) error {
	if len(values) == 0 {
		return nil
	}
	return e.call(e.copyHtoD, destination, uintptr(unsafe.Pointer(&values[0])), uintptr(len(values)*4))
}

func (e *cudaExecutor) copyDeviceToHost(destination []float32, source uintptr) error {
	if len(destination) == 0 {
		return nil
	}
	return e.call(e.copyDtoH, uintptr(unsafe.Pointer(&destination[0])), source, uintptr(len(destination)*4))
}

func (e *cudaExecutor) makeCurrent() error  { return e.call(e.ctxSet, e.context) }
func (e *cudaExecutor) clearCurrent() error { return e.call(e.ctxSet, 0) }

func (e *cudaExecutor) call(proc *syscall.LazyProc, arguments ...uintptr) error {
	result, _, callErr := proc.Call(arguments...)
	if result != 0 {
		return fmt.Errorf("CUDA driver %s (%d): %v", cudaErrorName(result), result, callErr)
	}
	return nil
}

func cudaErrorName(code uintptr) string {
	switch code {
	case 1:
		return "CUDA_ERROR_INVALID_VALUE"
	case 2:
		return "CUDA_ERROR_OUT_OF_MEMORY"
	case 201:
		return "CUDA_ERROR_INVALID_CONTEXT"
	case 700:
		return "CUDA_ERROR_ILLEGAL_ADDRESS"
	case 719:
		return "CUDA_ERROR_LAUNCH_FAILED"
	default:
		return "CUDA_ERROR_UNKNOWN"
	}
}

func (e *cudaExecutor) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if e.context == 0 {
		return nil
	}
	if err := e.makeCurrent(); err != nil {
		return err
	}
	for _, matrix := range e.matrices {
		_ = e.call(e.memFree, matrix.device)
	}
	e.matrices = map[string]cudaMatrix{}
	if err := e.call(e.ctxDestroy, e.context); err != nil {
		return err
	}
	e.context = 0
	return nil
}

const cudaMatrixPTX = `.version 7.0
.target sm_52
.address_size 64

.visible .entry sypra_matvec_f32(
    .param .u64 weights,
    .param .u64 input,
    .param .u64 output,
    .param .u32 rows,
    .param .u32 cols
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd1, [weights];
    ld.param.u64 %rd2, [input];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [rows];
    ld.param.u32 %r5, [cols];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r0, %r1, %r2, %r3;
    setp.ge.u32 %p1, %r0, %r4;
    @%p1 bra MV_DONE;
    cvt.u64.u32 %rd4, %r0;
    cvt.u64.u32 %rd5, %r5;
    mul.lo.u64 %rd6, %rd4, %rd5;
    mov.u32 %r6, 0;
    mov.f32 %f0, 0f00000000;
MV_LOOP:
    setp.ge.u32 %p1, %r6, %r5;
    @%p1 bra MV_STORE;
    cvt.u64.u32 %rd7, %r6;
    add.u64 %rd8, %rd6, %rd7;
    mul.lo.u64 %rd8, %rd8, 4;
    add.u64 %rd9, %rd1, %rd8;
    ld.global.f32 %f1, [%rd9];
    mul.lo.u64 %rd7, %rd7, 4;
    add.u64 %rd9, %rd2, %rd7;
    ld.global.f32 %f2, [%rd9];
    fma.rn.f32 %f0, %f1, %f2, %f0;
    add.u32 %r6, %r6, 1;
    bra MV_LOOP;
MV_STORE:
    mul.wide.u32 %rd7, %r0, 4;
    add.u64 %rd8, %rd3, %rd7;
    st.global.f32 [%rd8], %f0;
MV_DONE:
    ret;
}

.visible .entry sypra_matvec_transpose_f32(
    .param .u64 weights,
    .param .u64 input,
    .param .u64 output,
    .param .u32 rows,
    .param .u32 cols
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd1, [weights];
    ld.param.u64 %rd2, [input];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [rows];
    ld.param.u32 %r5, [cols];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r0, %r1, %r2, %r3;
    setp.ge.u32 %p1, %r0, %r5;
    @%p1 bra MT_DONE;
    mov.u32 %r6, 0;
    mov.f32 %f0, 0f00000000;
MT_LOOP:
    setp.ge.u32 %p1, %r6, %r4;
    @%p1 bra MT_STORE;
    cvt.u64.u32 %rd4, %r6;
    cvt.u64.u32 %rd5, %r5;
    mul.lo.u64 %rd6, %rd4, %rd5;
    cvt.u64.u32 %rd7, %r0;
    add.u64 %rd6, %rd6, %rd7;
    mul.lo.u64 %rd6, %rd6, 4;
    add.u64 %rd8, %rd1, %rd6;
    ld.global.f32 %f1, [%rd8];
    cvt.u64.u32 %rd7, %r6;
    mul.lo.u64 %rd7, %rd7, 4;
    add.u64 %rd8, %rd2, %rd7;
    ld.global.f32 %f2, [%rd8];
    fma.rn.f32 %f0, %f1, %f2, %f0;
    add.u32 %r6, %r6, 1;
    bra MT_LOOP;
MT_STORE:
    mul.wide.u32 %rd7, %r0, 4;
    add.u64 %rd8, %rd3, %rd7;
    st.global.f32 [%rd8], %f0;
MT_DONE:
    ret;
}`
