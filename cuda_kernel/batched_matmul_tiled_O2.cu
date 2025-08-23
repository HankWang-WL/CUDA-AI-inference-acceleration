#ifndef TILE
#define TILE 32
#endif

extern "C" __global__
void batched_matmul_tiled_kernel_O2(
    const float* __restrict__ A,   // [B, M, K]
    const float* __restrict__ B,   // [B, K, N]
    float* __restrict__ C,         // [B, M, N]
    int BATCH, int M, int K, int N)
{
    const int b   = blockIdx.z;

    // One block computes a TILE x TILE output tile of C.
    // Each thread computes two output columns (col0, col1), so threadIdx.x spans TILE/2.
    const int tx  = threadIdx.x;           // 0 .. TILE/2-1
    const int ty  = threadIdx.y;           // 0 .. TILE-1

    const int row  = blockIdx.y * TILE + ty;           // M dimension
    const int col0 = blockIdx.x * TILE + (tx << 1);    // N dimension (2 columns per thread)
    const int col1 = col0 + 1;

    const float* Ab = A + b * (M * K);
    const float* Bb = B + b * (K * N);
    float*       Cb = C + b * (M * N);

    // +1 padding to avoid shared memory bank conflicts (preserves your original approach)
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    float acc0 = 0.0f;  // accumulator for col0
    float acc1 = 0.0f;  // accumulator for col1

    // Sweep K in steps of TILE
    for (int kt = 0; kt < K; kt += TILE) {
        // ===== Load A into shared (prefer float2; fall back to scalar near edges) =====
        // As[ty][0..TILE-1] is cooperatively filled by half the threads in x using float2 stores.
        int a_col0 = kt + (tx << 1);
        int a_col1 = a_col0 + 1;

        if (row < M && b < BATCH) {
            if (a_col1 < K) {
                // Vectorized load (2 elements along K)
                const float2* __restrict__ Arow = reinterpret_cast<const float2*>(Ab + row * K + a_col0);
                float2 va = *Arow; // May be misaligned; CUDA still executes (often with benefit).
                As[ty][a_col0 - kt] = va.x;
                As[ty][a_col1 - kt] = va.y;
            } else {
                // Boundary handling
                As[ty][a_col0 - kt] = (a_col0 < K) ? Ab[row * K + a_col0] : 0.0f;
                if (a_col1 - kt < TILE) {
                    As[ty][a_col1 - kt] = (a_col1 < K) ? Ab[row * K + a_col1] : 0.0f;
                }
            }
        } else {
            // Out of row/batch range: fill zeros to avoid dirty values
            if (a_col0 - kt < TILE) As[ty][a_col0 - kt] = 0.0f;
            if (a_col1 - kt < TILE) As[ty][a_col1 - kt] = 0.0f;
        }

        // ===== Load B into shared (float2) =====
        // Each ty corresponds to a row of the current K-slice; x threads write two columns.
        int b_row = kt + ty;
        if (b_row < K && b < BATCH) {
            if (col1 < N) {
                const float2* __restrict__ Brow = reinterpret_cast<const float2*>(Bb + b_row * N + col0);
                float2 vb = *Brow;
                Bs[ty][col0 - blockIdx.x * TILE] = vb.x;
                Bs[ty][col1 - blockIdx.x * TILE] = vb.y;
            } else {
                Bs[ty][col0 - blockIdx.x * TILE] = (col0 < N) ? Bb[b_row * N + col0] : 0.0f;
                if (col1 - blockIdx.x * TILE < TILE) {
                    Bs[ty][col1 - blockIdx.x * TILE] = (col1 < N) ? Bb[b_row * N + col1] : 0.0f;
                }
            }
        } else {
            if (col0 - blockIdx.x * TILE < TILE) Bs[ty][col0 - blockIdx.x * TILE] = 0.0f;
            if (col1 - blockIdx.x * TILE < TILE) Bs[ty][col1 - blockIdx.x * TILE] = 0.0f;
        }

        __syncthreads();

        // ===== Accumulate: each thread computes two output columns of C =====
        #pragma unroll
        for (int kInner = 0; kInner < TILE; ++kInner) {
            float a = As[ty][kInner];
            acc0 += a * Bs[kInner][(tx << 1) + 0];
            acc1 += a * Bs[kInner][(tx << 1) + 1];
        }

        __syncthreads();
    }

    // ===== Write back with boundary checks =====
    if (b < BATCH && row < M) {
        if (col0 < N) Cb[row * N + col0] = acc0;
        if (col1 < N) Cb[row * N + col1] = acc1;
    }
}
