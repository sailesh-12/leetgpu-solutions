#include <cuda_runtime.h>
#define THREADS 512
#define REDUCE_THREADS 1024
__global__ void partialMax(
    const float* input,
    float* partial,
    int N
) {
    __shared__ float sdata[THREADS];

