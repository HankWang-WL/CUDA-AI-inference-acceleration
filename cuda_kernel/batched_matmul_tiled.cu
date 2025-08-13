#ifndef TILE
#define TILE 16
#endif

extern "C" __global__
void batched_matmul_tiled_kernel(
    const float* __restrict__ A,   // [B, M, K]
    const float* __restrict__ B,   // [B, K, N]
    float* __restrict__ C,         // [B, M, N]
    int BATCH, int M, int K, int N)
{
    const int b   = blockIdx.z;
    const int row = blockIdx.y * TILE + threadIdx.y; // M
    const int col = blockIdx.x * TILE + threadIdx.x; // N

    
    const float* Ab = A + b * (M * K);
    const float* Bb = B + b * (K * N);
    float*       Cb = C + b * (M * N);

    // +1 padding avoid bank conflict
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    float acc = 0.0f;

    // step: TILE
    for (int kt = 0; kt < K; kt += TILE) {
        // A[row, kt + tx]
        const int a_col = kt + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K && b < BATCH) ? Ab[row * K + a_col] : 0.0f;

        // B[kt + ty, col]
        const int b_row = kt + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N && b < BATCH) ? Bb[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int kInner = 0; kInner < TILE; ++kInner) {
            acc += As[threadIdx.y][kInner] * Bs[kInner][threadIdx.x];
        }

        __syncthreads();
    }
    if (b < BATCH && row < M && col < N) {
        Cb[row * N + col] = acc;
    }
}


