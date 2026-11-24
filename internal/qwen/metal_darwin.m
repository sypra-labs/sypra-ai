#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

typedef struct {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> matvec;
  id<MTLComputePipelineState> transpose;
} SypraMetalContext;

static NSString *const SypraMetalSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void sypra_matvec_f32(device const float *weights [[buffer(0)]],"
" device const float *input [[buffer(1)]], device float *output [[buffer(2)]],"
" constant uint &rows [[buffer(3)]], constant uint &cols [[buffer(4)]],"
" uint row [[thread_position_in_grid]]) {"
" if (row >= rows) return; float sum = 0.0f; uint base = row * cols;"
" for (uint column = 0; column < cols; ++column) sum += weights[base + column] * input[column];"
" output[row] = sum; }\n"
"kernel void sypra_matvec_transpose_f32(device const float *weights [[buffer(0)]],"
" device const float *input [[buffer(1)]], device float *output [[buffer(2)]],"
" constant uint &rows [[buffer(3)]], constant uint &cols [[buffer(4)]],"
" uint column [[thread_position_in_grid]]) {"
" if (column >= cols) return; float sum = 0.0f;"
" for (uint row = 0; row < rows; ++row) sum += weights[row * cols + column] * input[row];"
" output[column] = sum; }\n";

int sypra_metal_create(void **out) {
  if (!out) return -1;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return -2;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) return -3;
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:SypraMetalSource options:nil error:&error];
    if (!library) { [queue release]; return -4; }
    id<MTLFunction> matvecFunction = [library newFunctionWithName:@"sypra_matvec_f32"];
    id<MTLFunction> transposeFunction = [library newFunctionWithName:@"sypra_matvec_transpose_f32"];
    id<MTLComputePipelineState> matvec = [device newComputePipelineStateWithFunction:matvecFunction error:&error];
    id<MTLComputePipelineState> transpose = [device newComputePipelineStateWithFunction:transposeFunction error:&error];
    [matvecFunction release]; [transposeFunction release]; [library release];
    if (!matvec || !transpose) { [matvec release]; [transpose release]; [queue release]; return -5; }
    SypraMetalContext *context = calloc(1, sizeof(SypraMetalContext));
    if (!context) { [matvec release]; [transpose release]; [queue release]; return -6; }
    context->device = [device retain]; context->queue = queue; context->matvec = matvec; context->transpose = transpose;
    *out = context;
    return 0;
  }
}

int sypra_metal_upload(void *raw, const float *values, size_t count, void **out) {
  if (!raw || !values || !count || !out) return -1;
  SypraMetalContext *context = raw;
  id<MTLBuffer> buffer = [context->device newBufferWithBytes:values length:count * sizeof(float) options:MTLResourceStorageModeShared];
  if (!buffer) return -2;
  *out = buffer;
  return 0;
}

int sypra_metal_run(void *raw, void *rawWeights, unsigned int rows, unsigned int cols, const float *input, size_t inputCount, int transpose, float *output, size_t outputCount) {
  if (!raw || !rawWeights || !input || !output || !inputCount || !outputCount) return -1;
  @autoreleasepool {
    SypraMetalContext *context = raw;
    id<MTLBuffer> weights = rawWeights;
    id<MTLBuffer> inputBuffer = [context->device newBufferWithBytes:input length:inputCount * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [context->device newBufferWithLength:outputCount * sizeof(float) options:MTLResourceStorageModeShared];
    if (!inputBuffer || !outputBuffer) { [inputBuffer release]; [outputBuffer release]; return -2; }
    id<MTLCommandBuffer> command = [context->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:transpose ? context->transpose : context->matvec];
    [encoder setBuffer:weights offset:0 atIndex:0]; [encoder setBuffer:inputBuffer offset:0 atIndex:1]; [encoder setBuffer:outputBuffer offset:0 atIndex:2];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:3]; [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
    NSUInteger width = transpose ? cols : rows;
    NSUInteger threadWidth = MIN((NSUInteger)256, (transpose ? context->transpose : context->matvec).maxTotalThreadsPerThreadgroup);
    if (threadWidth == 0) threadWidth = 1;
    [encoder dispatchThreads:MTLSizeMake(width, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) { [inputBuffer release]; [outputBuffer release]; return -3; }
    memcpy(output, [outputBuffer contents], outputCount * sizeof(float));
    [inputBuffer release]; [outputBuffer release];
    return 0;
  }
}

void sypra_metal_buffer_free(void *raw) { if (raw) [(id<MTLBuffer>)raw release]; }
void sypra_metal_destroy(void *raw) {
  if (!raw) return;
  SypraMetalContext *context = raw;
  [context->matvec release]; [context->transpose release]; [context->queue release]; [context->device release];
  free(context);
}
