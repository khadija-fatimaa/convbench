// src/gemm.cuh -- shared GEMM kernels (owner: Mania). Section 5.1.
// The GEMM computes, for each batch index b, C_b = A * B_b.
// A is M x Kd, row-major, the same for every b.
// B_b is Kd x Ncols; elements supplied by a loader object: loadB(b, k, col).
// C_b is M x Ncols, row-major, stored at C + b * M * Ncols.
// Two GEMM kernels:
//   gemm_simple: 16x16 shared-memory tile, one element per thread.
//   gemm_opt:    64x64 tile, 256 threads (16x16), 4x4 register tiling per thread.
#pragma once
#include <cuda_runtime.h>

#define GEMM_USE_OPT 1   // set to 0 only under the fallback rule in Section 9.1

// ---------------------------------------------------------------------------
// Step 1: gemm_simple
// Block of 16 x 16 threads; grid (ceil(Ncols/16), ceil(M/16), batch); b = blockIdx.z.
// Thread (tx, ty) computes C[row][col] with row = blockIdx.y*16 + ty, col = blockIdx.x*16 + tx.
// Shared arrays As[16][16] and Bs[16][16].
// ---------------------------------------------------------------------------
template <class Loader>
__global__ void gemm_simple(const float* __restrict__ A, Loader loadB,
                            float* __restrict__ C,
                            int M, int Ncols, int Kd) {
    const int TILE = 16;
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    int b   = blockIdx.z;

    float* C_b = C + (size_t)b * M * Ncols;

    float acc = 0.0f;

    for (int k0 = 0; k0 < Kd; k0 += TILE) {
        int ak = k0 + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && ak < Kd) ? A[(size_t)row * Kd + ak] : 0.0f;

        int bk = k0 + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (bk < Kd && col < Ncols) ? loadB(b, bk, col) : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < Ncols) {
        C_b[(size_t)row * Ncols + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// Step 2: gemm_opt (Section 5.1)
// Output tile 64 x 64 (BM = 64 rows, BN = 64 columns), BK = 16.
// Block of 256 threads seen as 16 x 16 (tid = ty * 16 + tx).
// Grid (ceil(Ncols/64), ceil(M/64), batch).
// Each thread computes 16 outputs at rows rowBase + ty + 16*i and
// columns colBase + tx + 16*j for i, j = 0..3, kept in a 4 x 4 register array.
// Shared arrays As[16][64] (A stored transposed, As[kk][m]) and Bs[16][64].
// ---------------------------------------------------------------------------
template <class Loader>
__global__ void gemm_opt(const float* __restrict__ A, Loader loadB,
                         float* __restrict__ C,
                         int M, int Ncols, int Kd) {
    const int BM = 64, BN = 64, BK = 16;

    __shared__ float As[BK][BM];   // As[kk][m]
    __shared__ float Bs[BK][BN];   // Bs[kk][c]

    int b = blockIdx.z;
    float* C_b = C + (size_t)b * M * Ncols;

    int rowBase = blockIdx.y * BM;
    int colBase = blockIdx.x * BN;

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * 16 + tx;   // 0..255

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < Kd; k0 += BK) {
        // Loading for each k0: for l = 0..3, idx = tid + 256*l
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            int idx = tid + 256 * l;  // 0..1023

            // A tile: m = idx / 16, kk = idx % 16, As[kk][m] = A[(rowBase + m)*Kd + k0 + kk]
            int m_a  = idx / 16;
            int kk_a = idx % 16;
            int gm   = rowBase + m_a;
            int gk_a = k0 + kk_a;
            As[kk_a][m_a] = (gm < M && gk_a < Kd) ? A[(size_t)gm * Kd + gk_a] : 0.0f;

            // B tile: kk = idx / 64, c = idx % 64, Bs[kk][c] = loadB(b, k0 + kk, colBase + c)
            int kk_b = idx / 64;
            int c_b  = idx % 64;
            int gk_b = k0 + kk_b;
            int gc   = colBase + c_b;
            Bs[kk_b][c_b] = (gk_b < Kd && gc < Ncols) ? loadB(b, gk_b, gc) : 0.0f;
        }

        __syncthreads();

        // Compute: for kk = 0..15, read a[i] = As[kk][ty + 16*i] and v[j] = Bs[kk][tx + 16*j]
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a_reg[4];
            float v_reg[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                a_reg[i] = As[kk][ty + 16 * i];
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                v_reg[j] = Bs[kk][tx + 16 * j];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += a_reg[i] * v_reg[j];
                }
            }
        }

        __syncthreads();
    }

    // Write back: rowBase + ty + 16*i, colBase + tx + 16*j
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int gm = rowBase + ty + 16 * i;
        if (gm < M) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int gn = colBase + tx + 16 * j;
                if (gn < Ncols) {
                    C_b[(size_t)gm * Ncols + gn] = acc[i][j];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// gemm_launch: dispatch to gemm_opt or gemm_simple based on GEMM_USE_OPT
// ---------------------------------------------------------------------------
template <class Loader>
void gemm_launch(const float* A, Loader loadB, float* C,
                 int M, int Ncols, int Kd, int batch) {
#if GEMM_USE_OPT
    const int BM = 64, BN = 64;
    dim3 block(16, 16);
    dim3 grid((Ncols + BN - 1) / BN, (M + BM - 1) / BM, batch);
    gemm_opt<<<grid, block>>>(A, loadB, C, M, Ncols, Kd);
#else
    const int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((Ncols + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch);
    gemm_simple<<<grid, block>>>(A, loadB, C, M, Ncols, Kd);
#endif
}

