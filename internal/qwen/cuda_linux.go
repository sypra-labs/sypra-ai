//go:build linux && cgo

package qwen

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>
#include <stdint.h>
#include <stddef.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef uint64_t CUdeviceptr;
typedef void *CUmodule;
typedef void *CUfunction;
typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDeviceGet_t)(CUdevice *, int);
typedef CUresult (*cuDeviceGetCount_t)(int *);
typedef CUresult (*cuCtxCreate_t)(CUcontext *, unsigned int, CUdevice);
typedef CUresult (*cuCtxSetCurrent_t)(CUcontext);
typedef CUresult (*cuCtxDestroy_t)(CUcontext);
typedef CUresult (*cuModuleLoadData_t)(CUmodule *, const void *);
typedef CUresult (*cuModuleGetFunction_t)(CUfunction *, CUmodule, const char *);
typedef CUresult (*cuMemAlloc_t)(CUdeviceptr *, size_t);
typedef CUresult (*cuMemFree_t)(CUdeviceptr);
typedef CUresult (*cuMemcpyHtoD_t)(CUdeviceptr, const void *, size_t);
typedef CUresult (*cuMemcpyDtoH_t)(void *, CUdeviceptr, size_t);
typedef CUresult (*cuLaunchKernel_t)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, void *, void **, void **);

static void *sypra_cuda_driver;
static cuInit_t sypra_cuInit;
static cuDeviceGet_t sypra_cuDeviceGet;
static cuDeviceGetCount_t sypra_cuDeviceGetCount;
static cuCtxCreate_t sypra_cuCtxCreate;
static cuCtxSetCurrent_t sypra_cuCtxSetCurrent;
static cuCtxDestroy_t sypra_cuCtxDestroy;
static cuModuleLoadData_t sypra_cuModuleLoadData;
static cuModuleGetFunction_t sypra_cuModuleGetFunction;
static cuMemAlloc_t sypra_cuMemAlloc;
static cuMemFree_t sypra_cuMemFree;
static cuMemcpyHtoD_t sypra_cuMemcpyHtoD;
static cuMemcpyDtoH_t sypra_cuMemcpyDtoH;
static cuLaunchKernel_t sypra_cuLaunchKernel;

static int sypra_cuda_load(void) {
  if (sypra_cuda_driver) return 0;
  sypra_cuda_driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!sypra_cuda_driver) return -1;
  #define SYPRA_CUDA_SYM(name) do { *(void **)(&sypra_##name) = dlsym(sypra_cuda_driver, #name); if (!sypra_##name) return -2; } while (0)
  SYPRA_CUDA_SYM(cuInit); SYPRA_CUDA_SYM(cuDeviceGet); SYPRA_CUDA_SYM(cuDeviceGetCount);
  SYPRA_CUDA_SYM(cuCtxCreate); SYPRA_CUDA_SYM(cuCtxSetCurrent); SYPRA_CUDA_SYM(cuCtxDestroy);
  SYPRA_CUDA_SYM(cuModuleLoadData); SYPRA_CUDA_SYM(cuModuleGetFunction);
  SYPRA_CUDA_SYM(cuMemAlloc); SYPRA_CUDA_SYM(cuMemFree); SYPRA_CUDA_SYM(cuMemcpyHtoD);
  SYPRA_CUDA_SYM(cuMemcpyDtoH); SYPRA_CUDA_SYM(cuLaunchKernel);
  #undef SYPRA_CUDA_SYM
  return 0;
}
static int sypra_cuda_init(void) { int e = sypra_cuda_load(); return e ? e : sypra_cuInit(0); }
static int sypra_cuda_device_count(int *count) { return sypra_cuDeviceGetCount(count); }
static int sypra_cuda_device_get(CUdevice *device, int ordinal) { return sypra_cuDeviceGet(device, ordinal); }
static int sypra_cuda_context_create(CUcontext *context, CUdevice device) { return sypra_cuCtxCreate(context, 0, device); }
static int sypra_cuda_context_set(CUcontext context) { return sypra_cuCtxSetCurrent(context); }
static int sypra_cuda_context_destroy(CUcontext context) { return sypra_cuCtxDestroy(context); }
static int sypra_cuda_module_load(CUmodule *module, const void *ptx) { return sypra_cuModuleLoadData(module, ptx); }
static int sypra_cuda_function(CUfunction *function, CUmodule module, const char *name) { return sypra_cuModuleGetFunction(function, module, name); }
static int sypra_cuda_alloc(CUdeviceptr *pointer, size_t size) { return sypra_cuMemAlloc(pointer, size); }
static int sypra_cuda_free(CUdeviceptr pointer) { return sypra_cuMemFree(pointer); }
static int sypra_cuda_upload(CUdeviceptr pointer, const void *source, size_t size) { return sypra_cuMemcpyHtoD(pointer, source, size); }
static int sypra_cuda_download(void *destination, CUdeviceptr pointer, size_t size) { return sypra_cuMemcpyDtoH(destination, pointer, size); }
static int sypra_cuda_launch(CUfunction function, unsigned int grid, CUdeviceptr weights, CUdeviceptr input, CUdeviceptr output, unsigned int rows, unsigned int cols) {
  void *parameters[] = { &weights, &input, &output, &rows, &cols };
  return sypra_cuLaunchKernel(function, grid, 1, 1, 256, 1, 1, 0, 0, parameters, 0);
}
*/
import "C"

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// cudaExecutor uses the installed NVIDIA driver directly through dlopen. No
// CUDA toolkit or link-time CUDA dependency is required, so CPU fallback
// remains available on hosts without the driver.
type cudaExecutor struct {
	context, module   unsafe.Pointer
	matvec, transpose unsafe.Pointer
	matrices          map[string]cudaMatrix
	mu                sync.Mutex
}

type cudaMatrix struct {
	device     C.uint64_t
	rows, cols int
}

func NewCUDAExecutor() (MatrixExecutor, error) { return NewCUDAExecutorForDevice(0) }

func NewCUDAExecutorForDevice(deviceOrdinal int) (MatrixExecutor, error) {
	if deviceOrdinal < 0 {
		return nil, fmt.Errorf("CUDA device ordinal must be non-negative")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if code := C.sypra_cuda_init(); code != 0 {
		return nil, cudaError("initialize CUDA driver", code)
	}
	var device C.CUdevice
	if code := C.sypra_cuda_device_get(&device, C.int(deviceOrdinal)); code != 0 {
		return nil, cudaError("select CUDA device", code)
	}
	executor := &cudaExecutor{matrices: map[string]cudaMatrix{}}
	if code := C.sypra_cuda_context_create(&executor.context, device); code != 0 {
		return nil, cudaError("create CUDA context", code)
	}
	ptx := append([]byte(cudaMatrixPTX), 0)
	if code := C.sypra_cuda_module_load(&executor.module, unsafe.Pointer(&ptx[0])); code != 0 {
		_ = executor.Close()
		return nil, cudaError("load CUDA matrix kernels", code)
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
		return nil, err
	}
	return executor, nil
}

func CUDADeviceCount() (int, error) {
	if code := C.sypra_cuda_init(); code != 0 {
		return 0, cudaError("initialize CUDA driver", code)
	}
	var count C.int
	if code := C.sypra_cuda_device_count(&count); code != 0 {
		return 0, cudaError("enumerate CUDA devices", code)
	}
	return int(count), nil
}

func (e *cudaExecutor) MatrixVec(model *Model, name string, input []float32) ([]float32, error) {
	return e.matrixVec(model, name, input, false)
}
func (e *cudaExecutor) MatrixVecTranspose(model *Model, name string, input []float32) ([]float32, error) {
	return e.matrixVec(model, name, input, true)
}
func (e *cudaExecutor) matrixVec(model *Model, name string, input []float32, transpose bool) ([]float32, error) {
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
		return nil, err
	}
	expected := matrix.cols
	if transpose {
		expected = matrix.rows
	}
	if len(input) != expected {
		return nil, fmt.Errorf("CUDA matrix %s requires %d values, got %d", name, expected, len(input))
	}
	return e.run(matrix, input, transpose)
}

func (e *cudaExecutor) matrix(model *Model, name string) (cudaMatrix, error) {
	if matrix, ok := e.matrices[name]; ok {
		return matrix, nil
	}
	rows, cols, values, err := acceleratorMatrixValues(model, name)
	if err != nil {
		return cudaMatrix{}, err
	}
	var device C.uint64_t
	if code := C.sypra_cuda_alloc(&device, C.size_t(len(values)*4)); code != 0 {
		return cudaMatrix{}, cudaError("allocate CUDA matrix", code)
	}
	if code := C.sypra_cuda_upload(device, unsafe.Pointer(&values[0]), C.size_t(len(values)*4)); code != 0 {
		_ = C.sypra_cuda_free(device)
		return cudaMatrix{}, cudaError("upload CUDA matrix", code)
	}
	matrix := cudaMatrix{device: device, rows: rows, cols: cols}
	e.matrices[name] = matrix
	return matrix, nil
}

func (e *cudaExecutor) run(matrix cudaMatrix, input []float32, transpose bool) ([]float32, error) {
	var deviceInput, deviceOutput C.uint64_t
	if code := C.sypra_cuda_alloc(&deviceInput, C.size_t(len(input)*4)); code != 0 {
		return nil, cudaError("allocate CUDA input", code)
	}
	defer C.sypra_cuda_free(deviceInput)
	outputCount, function := matrix.rows, e.matvec
	if transpose {
		outputCount, function = matrix.cols, e.transpose
	}
	if code := C.sypra_cuda_alloc(&deviceOutput, C.size_t(outputCount*4)); code != 0 {
		return nil, cudaError("allocate CUDA output", code)
	}
	defer C.sypra_cuda_free(deviceOutput)
	if code := C.sypra_cuda_upload(deviceInput, unsafe.Pointer(&input[0]), C.size_t(len(input)*4)); code != 0 {
		return nil, cudaError("upload CUDA input", code)
	}
	if code := C.sypra_cuda_launch(function, C.uint((outputCount+255)/256), matrix.device, deviceInput, deviceOutput, C.uint(matrix.rows), C.uint(matrix.cols)); code != 0 {
		return nil, cudaError("launch CUDA matrix kernel", code)
	}
	output := make([]float32, outputCount)
	if code := C.sypra_cuda_download(unsafe.Pointer(&output[0]), deviceOutput, C.size_t(outputCount*4)); code != 0 {
		return nil, cudaError("download CUDA output", code)
	}
	runtime.KeepAlive(input)
	return output, nil
}

func (e *cudaExecutor) loadFunction(destination *unsafe.Pointer, name string) error {
	bytes := append([]byte(name), 0)
	if code := C.sypra_cuda_function(destination, e.module, (*C.char)(unsafe.Pointer(&bytes[0]))); code != 0 {
		return cudaError("load CUDA kernel "+name, code)
	}
	return nil
}
func (e *cudaExecutor) makeCurrent() error {
	if code := C.sypra_cuda_context_set(e.context); code != 0 {
		return cudaError("make CUDA context current", code)
	}
	return nil
}
func (e *cudaExecutor) clearCurrent() error {
	if code := C.sypra_cuda_context_set(nil); code != 0 {
		return cudaError("detach CUDA context", code)
	}
	return nil
}
func (e *cudaExecutor) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if e.context == nil {
		return nil
	}
	if err := e.makeCurrent(); err != nil {
		return err
	}
	for _, matrix := range e.matrices {
		_ = C.sypra_cuda_free(matrix.device)
	}
	e.matrices = map[string]cudaMatrix{}
	if code := C.sypra_cuda_context_destroy(e.context); code != 0 {
		return cudaError("destroy CUDA context", code)
	}
	e.context = nil
	return nil
}
func cudaError(action string, code C.int) error {
	return fmt.Errorf("%s: CUDA driver result %d", action, int(code))
}

const cudaMatrixPTX = `.version 7.0
.target sm_52
.address_size 64

.visible .entry sypra_matvec_f32(
    .param .u64 weights, .param .u64 input, .param .u64 output,
    .param .u32 rows, .param .u32 cols
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd1, [weights]; ld.param.u64 %rd2, [input];
    ld.param.u64 %rd3, [output]; ld.param.u32 %r4, [rows]; ld.param.u32 %r5, [cols];
    mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x;
    mad.lo.u32 %r0, %r1, %r2, %r3; setp.ge.u32 %p1, %r0, %r4; @%p1 bra MV_DONE;
    cvt.u64.u32 %rd4, %r0; cvt.u64.u32 %rd5, %r5; mul.lo.u64 %rd6, %rd4, %rd5;
    mov.u32 %r6, 0; mov.f32 %f0, 0f00000000;
MV_LOOP:
    setp.ge.u32 %p1, %r6, %r5; @%p1 bra MV_STORE;
    cvt.u64.u32 %rd7, %r6; add.u64 %rd8, %rd6, %rd7; mul.lo.u64 %rd8, %rd8, 4;
    add.u64 %rd9, %rd1, %rd8; ld.global.f32 %f1, [%rd9];
    mul.lo.u64 %rd7, %rd7, 4; add.u64 %rd9, %rd2, %rd7; ld.global.f32 %f2, [%rd9];
    fma.rn.f32 %f0, %f1, %f2, %f0; add.u32 %r6, %r6, 1; bra MV_LOOP;
MV_STORE:
    mul.wide.u32 %rd7, %r0, 4; add.u64 %rd8, %rd3, %rd7; st.global.f32 [%rd8], %f0;
MV_DONE: ret;
}

.visible .entry sypra_matvec_transpose_f32(
    .param .u64 weights, .param .u64 input, .param .u64 output,
    .param .u32 rows, .param .u32 cols
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd1, [weights]; ld.param.u64 %rd2, [input];
    ld.param.u64 %rd3, [output]; ld.param.u32 %r4, [rows]; ld.param.u32 %r5, [cols];
    mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x;
    mad.lo.u32 %r0, %r1, %r2, %r3; setp.ge.u32 %p1, %r0, %r5; @%p1 bra MT_DONE;
    mov.u32 %r6, 0; mov.f32 %f0, 0f00000000;
MT_LOOP:
    setp.ge.u32 %p1, %r6, %r4; @%p1 bra MT_STORE;
    cvt.u64.u32 %rd4, %r6; cvt.u64.u32 %rd5, %r5; mul.lo.u64 %rd6, %rd4, %rd5;
    cvt.u64.u32 %rd7, %r0; add.u64 %rd6, %rd6, %rd7; mul.lo.u64 %rd6, %rd6, 4;
    add.u64 %rd8, %rd1, %rd6; ld.global.f32 %f1, [%rd8];
    cvt.u64.u32 %rd7, %r6; mul.lo.u64 %rd7, %rd7, 4; add.u64 %rd8, %rd2, %rd7;
    ld.global.f32 %f2, [%rd8]; fma.rn.f32 %f0, %f1, %f2, %f0;
    add.u32 %r6, %r6, 1; bra MT_LOOP;
MT_STORE:
    mul.wide.u32 %rd7, %r0, 4; add.u64 %rd8, %rd3, %rd7; st.global.f32 [%rd8], %f0;
MT_DONE: ret;
}`
