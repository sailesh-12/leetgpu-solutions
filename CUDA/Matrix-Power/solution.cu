#include <cuda_runtime.h>

// input, output are device pointers


__global__ void matrix_multiplication_kernel(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= N || col >= N) return;

    float sum = 0.0f;

    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

__global__ void identity_kernel(float* I, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        I[row * N + col] = (row == col) ? 1.0f : 0.0f;
    }
}

extern "C" void solve(
    const float* input,
    float* output,
    int N,
    int P
) {
    if (N <= 0) return;

    dim3 threads(32, 32);
    dim3 blocks(
        (N + 31) / 32,
        (N + 31) / 32
    );

    size_t bytes = (size_t)N * N * sizeof(float);

    float *result, *base, *temp;

    cudaMalloc(&result, bytes);
    cudaMalloc(&base, bytes);
    cudaMalloc(&temp, bytes);

    // result = identity matrix
    identity_kernel<<<blocks, threads>>>(result, N);

    // base = input matrix
    cudaMemcpy(
        base, input, bytes,
        cudaMemcpyDeviceToDevice
    );

    int exponent = P;

    while (exponent > 0) {
        // If exponent is odd, multiply result by base.
        if (exponent % 2 == 1) {
            matrix_multiplication_kernel<<<blocks, threads>>>(
                result, base, temp, N
            );

            float* swap = result;
            result = temp;
            temp = swap;
        }

        exponent /= 2;

        // Square base for the next power.
        if (exponent > 0) {
            matrix_multiplication_kernel<<<blocks, threads>>>(
                base, base, temp, N
            );

            float* swap = base;
            base = temp;
            temp = swap;
        }
    }

    cudaMemcpy(
        output, result, bytes,
        cudaMemcpyDeviceToDevice
    );

    cudaFree(result);
    cudaFree(base);
    cudaFree(temp);
}
