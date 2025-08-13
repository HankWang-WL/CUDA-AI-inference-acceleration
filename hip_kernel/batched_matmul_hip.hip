// hip_kernel/batched_matmul_hip.hip
#include <hip/hip_runtime.h>

extern "C" __global__
void batched_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch, int M, int K, int N
) {
    int b   = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
            sum += A[b*M*K + row*K + kk] * B[b*K*N + kk*N + col];
        }
        C[b*M*N + row*N + col] = sum;
    }
}
