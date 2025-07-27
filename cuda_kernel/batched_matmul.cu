extern "C" __global__
void batched_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch, int M, int K, int N
) {
    // batch parallel
    int b = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[b*M*K + row*K + k] * B[b*K*N + k*N + col];
        }
        C[b*M*N + row*N + col] = sum;
    }
}
