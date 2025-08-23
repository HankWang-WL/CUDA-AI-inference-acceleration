#ifndef TILE
#define TILE 32
#endif

extern "C" __global__
void batched_matmul_tiled_kernel_O2_1(
    const float* __restrict__ A,   // [B, M, K]
    const float* __restrict__ B,   // [B, K, N]
    float* __restrict__ C,         // [B, M, N]
    int BATCH, int M, int K, int N)
{
    const int b  = blockIdx.z;

    // 2x2 register blocking: each thread computes a 2x2 C block
    const int tx = threadIdx.x;               // 0..TILE/2-1
    const int ty = threadIdx.y;               // 0..TILE/2-1

    const int tile_x0 = (tx << 1);
    const int tile_y0 = (ty << 1);
    const int tile_x1 = tile_x0 + 1;
    const int tile_y1 = tile_y0 + 1;

    const int row0 = blockIdx.y * TILE + tile_y0;
    const int row1 = row0 + 1;
    const int col0 = blockIdx.x * TILE + tile_x0;
    const int col1 = col0 + 1;

    const float* Ab = A + b * (M * K);
    const float* Bb = B + b * (K * N);
    float*       Cb = C + b * (M * N);

    // +1 padding to mitigate bank conflicts
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    float acc00 = 0.f, acc01 = 0.f;
    float acc10 = 0.f, acc11 = 0.f;

    for (int kt = 0; kt < K; kt += TILE) {
        // -------- Load A (two rows per thread) using float4 when aligned --------
        // Use only even tx to avoid overlap: each loads 4 consecutive k's.
        const int even_tx = (tx & 1) == 0;
        const int px = tx >> 1;                 // 0..TILE/4-1 when even
        const int k4 = kt + (px << 2);          // start column (k) for float4

        if (even_tx) {
            // row0
            if (row0 < M) {
                if (k4 + 3 < K) {
                    const float4* __restrict__ p = reinterpret_cast<const float4*>(Ab + row0 * K + k4);
                    float4 v = *p;
                    As[tile_y0][k4 - kt + 0] = v.x;
                    As[tile_y0][k4 - kt + 1] = v.y;
                    As[tile_y0][k4 - kt + 2] = v.z;
                    As[tile_y0][k4 - kt + 3] = v.w;
                } else {
                    // tail
                    for (int t = 0; t < 4; ++t) {
                        int kc = k4 + t;
                        if (kc - kt < TILE)
                            As[tile_y0][kc - kt] = (kc < K) ? Ab[row0 * K + kc] : 0.f;
                    }
                }
            } else {
                for (int t = 0; t < 4; ++t) if (k4 - kt + t < TILE) As[tile_y0][k4 - kt + t] = 0.f;
            }
            // row1
            if (row1 < M) {
                if (k4 + 3 < K) {
                    const float4* __restrict__ p = reinterpret_cast<const float4*>(Ab + row1 * K + k4);
                    float4 v = *p;
                    As[tile_y1][k4 - kt + 0] = v.x;
                    As[tile_y1][k4 - kt + 1] = v.y;
                    As[tile_y1][k4 - kt + 2] = v.z;
                    As[tile_y1][k4 - kt + 3] = v.w;
                } else {
                    for (int t = 0; t < 4; ++t) {
                        int kc = k4 + t;
                        if (kc - kt < TILE)
                            As[tile_y1][kc - kt] = (kc < K) ? Ab[row1 * K + kc] : 0.f;
                    }
                }
            } else {
                for (int t = 0; t < 4; ++t) if (k4 - kt + t < TILE) As[tile_y1][k4 - kt + t] = 0.f;
            }
        }

        // -------- Load B (two k-rows per thread) using float4 --------
        const int b_row0 = kt + tile_y0;
        const int b_row1 = b_row0 + 1;
        const int cx4 = (tx & 1) == 0 ? (blockIdx.x * TILE + (tx >> 1 << 2)) : -1; // starting col for float4

        if ((tx & 1) == 0) {
            const int off = cx4 - blockIdx.x * TILE; // 0..TILE-1
            // row0
            if (b_row0 < K) {
                if (cx4 + 3 < N) {
                    const float4* __restrict__ p = reinterpret_cast<const float4*>(Bb + b_row0 * N + cx4);
                    float4 v = *p;
                    Bs[b_row0 - kt][off + 0] = v.x;
                    Bs[b_row0 - kt][off + 1] = v.y;
                    Bs[b_row0 - kt][off + 2] = v.z;
                    Bs[b_row0 - kt][off + 3] = v.w;
                } else {
                    for (int t = 0; t < 4; ++t) {
                        int nc = cx4 + t;
                        if (off + t < TILE)
                            Bs[b_row0 - kt][off + t] = (nc < N) ? Bb[b_row0 * N + nc] : 0.f;
                    }
                }
            } else {
                for (int t = 0; t < 4; ++t) if (off + t < TILE) Bs[b_row0 - kt][off + t] = 0.f;
            }
            // row1
            if (b_row1 < K) {
                if (cx4 + 3 < N) {
                    const float4* __restrict__ p = reinterpret_cast<const float4*>(Bb + b_row1 * N + cx4);
                    float4 v = *p;
                    Bs[b_row1 - kt][off + 0] = v.x;
                    Bs[b_row1 - kt][off + 1] = v.y;
                    Bs[b_row1 - kt][off + 2] = v.z;
                    Bs[b_row1 - kt][off + 3] = v.w;
                } else {
                    for (int t = 0; t < 4; ++t) {
                        int nc = cx4 + t;
                        if (off + t < TILE)
                            Bs[b_row1 - kt][off + t] = (nc < N) ? Bb[b_row1 * N + nc] : 0.f;
                    }
                }
            } else {
                for (int t = 0; t < 4; ++t) if (off + t < TILE) Bs[b_row1 - kt][off + t] = 0.f;
            }
        }

        __syncthreads();

        // -------- Compute 2x2 block --------
        #pragma unroll
        for (int kInner = 0; kInner < TILE; ++kInner) {
            const float a0 = As[tile_y0][kInner];
            const float a1 = As[tile_y1][kInner];
            const float b0 = Bs[kInner][tile_x0];
            const float b1 = Bs[kInner][tile_x1];
            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }

        __syncthreads();
    }

    // -------- Write back (vectorize to float2 when possible) --------
    if (b < BATCH) {
        if (row0 < M) {
            if (col1 < N) {
                float2* __restrict__ pc = reinterpret_cast<float2*>(Cb + row0 * N + col0);
                *pc = make_float2(acc00, acc01);
            } else if (col0 < N) {
                Cb[row0 * N + col0] = acc00;
            }
        }
        if (row1 < M) {
            if (col1 < N) {
                float2* __restrict__ pc = reinterpret_cast<float2*>(Cb + row1 * N + col0);
                *pc = make_float2(acc10, acc11);
            } else if (col0 < N) {
                Cb[row1 * N + col0] = acc10;
            }
        }
    }
}
