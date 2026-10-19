// src/v4_cublas.cu -- V4 im2col + cuBLAS (owner: Mania). Section 5.5.
// Same im2col as V1, but uses cuBLAS SGEMM instead of our custom GEMM.
// cublasSgemmStridedBatched handles all N images in one call.
#include "common.h"
#include <cublas_v2.h>

static cublasHandle_t g_handle = nullptr;

void v4_init() {
    if (!g_handle) {
        cublasStatus_t st = cublasCreate(&g_handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS create failed: %d\n", (int)st);
            exit(1);
        }
    }
}

void v4_destroy() {
    if (g_handle) {
        cublasDestroy(g_handle);
        g_handle = nullptr;
    }
}

// Workspace = same as V1: im2col columns buffer
size_t v4_workspace_bytes(const ConvParams& p) {
    return (size_t)p.N * p.Cin * p.K * p.K * p.Ho * p.Wo * sizeof(float);
}

// ---------------------------------------------------------------------------
// V4: im2col + cuBLAS batched SGEMM
//
// After im2col, for each image n:
//   out[n] = w * col[n]
//   out[n]: [Cout x (Ho*Wo)]
//   w:      [Cout x (Cin*K*K)]
//   col[n]: [(Cin*K*K) x (Ho*Wo)]
//
// cuBLAS uses column-major, so we compute:
//   C^T = B^T * A^T
// where A^T = w^T: [(Cin*K*K) x Cout]  -- stride 0 (same weights for all images)
//       B^T = col[n]^T: [(Ho*Wo) x (Cin*K*K)]
//       C^T = out[n]^T: [(Ho*Wo) x Cout]
//
// In column-major terms:
//   M = Ho*Wo, N_blas = Cout, K_blas = Cin*K*K
//   lda = Ho*Wo (leading dim of B^T = col stored row-major)
//   ldb = Cin*K*K (leading dim of A^T = w stored row-major)
//   ldc = Ho*Wo (leading dim of C^T = out stored row-major)
// ---------------------------------------------------------------------------
void v4_run(const ConvParams& p, const float* in, const float* w, float* out, float* ws) {
    // Step 1: im2col
    im2col_launch(p, in, ws);

    // Step 2: cuBLAS batched SGEMM
    int M_blas = p.Ho * p.Wo;           // rows of C in col-major
    int N_blas = p.Cout;                 // cols of C in col-major
    int K_blas = p.Cin * p.K * p.K;     // inner dimension

    float alpha = 1.0f, beta = 0.0f;

    // col is stored as: col[n][k][col_idx] in row-major
    // In column-major, col[n] is a (K_blas x M_blas) matrix with leading dim M_blas when transposed
    // Actually: col[n] row-major [K_blas x M_blas] => col-major view is [M_blas x K_blas] with ld=M_blas
    // w row-major [Cout x K_blas] => col-major view is [K_blas x Cout] with ld=K_blas

    // We want: out[n] = w * col[n]  (row-major)
    // In col-major: out[n]^T = col[n]^T * w^T
    // col[n]^T in col-major: M_blas x K_blas, ld = M_blas
    // w^T in col-major: K_blas x N_blas, ld = K_blas
    // out[n]^T in col-major: M_blas x N_blas, ld = M_blas

    long long strideA = (long long)K_blas * M_blas;  // stride between col[n] and col[n+1]
    long long strideB = 0;                            // same weights for all images
    long long strideC = (long long)N_blas * M_blas;   // stride between out[n] and out[n+1]

    cublasStatus_t st = cublasSgemmStridedBatched(
        g_handle,
        CUBLAS_OP_N,    // op on col[n]^T: no transpose
        CUBLAS_OP_N,    // op on w^T: no transpose
        M_blas,         // M
        N_blas,         // N
        K_blas,         // K
        &alpha,
        ws,             // col (acts as A in col-major)
        M_blas,         // lda
        strideA,
        w,              // weights (acts as B in col-major)
        K_blas,         // ldb
        strideB,
        &beta,
        out,            // output (acts as C in col-major)
        M_blas,         // ldc
        strideC,
        p.N             // batch count
    );

    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS SGEMM failed: %d\n", (int)st);
        exit(1);
    }
}
