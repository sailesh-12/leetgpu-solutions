#include <cuda_runtime.h>
#define THREADS 512
#define REDUCE_THREADS 1024
__global__ void partialMax(
    const float* input,
    float* partial,
    int N
) {
    __shared__ float sdata[THREADS];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float value = (i < N) ? input[i] : -INFINITY;

    sdata[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2) {

        if (tid < stride) {
            sdata[tid] = fmaxf(
                sdata[tid],
                sdata[tid + stride]
            );
        }

        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}
__global__ void finalMax(
    const float* partial,
    float* globalMax,
    int numBlocks
) {
    __shared__ float sdata[REDUCE_THREADS];

    int tid = threadIdx.x;

    sdata[tid] =
        (tid < numBlocks) ? partial[tid] : -INFINITY;

    __syncthreads();

    for (int stride = REDUCE_THREADS / 2;
         stride > 0;
         stride /= 2) {

        if (tid < stride) {
            sdata[tid] = fmaxf(
                sdata[tid],
                sdata[tid + stride]
            );
        }

        __syncthreads();
    }

    if (tid == 0) {
        *globalMax = sdata[0];
    }
}
__global__ void partialSum(
    const float* input,
    float* output,
    float* partial,
    const float* globalMax,
    int N
) {
    __shared__ float sdata[THREADS];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float value = 0.0f;

    if (i < N) {
        value = expf(input[i] - *globalMax);
        output[i] = value;
    }

    sdata[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2) {

        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}
__global__ void finalSum(
    const float* partial,
    float* globalSum,
    int numBlocks
) {
    __shared__ float sdata[REDUCE_THREADS];

    int tid = threadIdx.x;

    sdata[tid] =
        (tid < numBlocks) ? partial[tid] : 0.0f;

    __syncthreads();

    for (int stride = REDUCE_THREADS / 2;
         stride > 0;
         stride /= 2) {

        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        *globalSum = sdata[0];
    }
}
__global__ void normalize(
    float* output,
    const float* globalSum,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        output[i] /= *globalSum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(
    const float* input,
    float* output,
    int N
) {
    if (N <= 0) return;

    int numBlocks = (N + THREADS - 1) / THREADS;

    float *d_partial, *d_max, *d_sum;

    cudaMalloc(&d_partial, numBlocks * sizeof(float));
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));

    dim3 blocks(numBlocks);

    partialMax<<<blocks, THREADS>>>(
        input, d_partial, N
    );

    finalMax<<<1, REDUCE_THREADS>>>(
        d_partial, d_max, numBlocks
    );

    partialSum<<<blocks, THREADS>>>(
        input, output, d_partial, d_max, N
    );

    finalSum<<<1, REDUCE_THREADS>>>(
        d_partial, d_sum, numBlocks
    );

    normalize<<<blocks, THREADS>>>(
        output, d_sum, N
    );

    cudaFree(d_partial);
    cudaFree(d_max);
    cudaFree(d_sum);
}