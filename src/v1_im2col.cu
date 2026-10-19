// src/v1_im2col.cu -- im2col kernel + V1 (owner: Mania). Section 5.2.
// im2col transforms the input so that convolution becomes matrix multiplication.
// The im2col columns matrix has shape [N x (Cin*K*K) x (Ho*Wo)].
// V1 uses im2col + our custom GEMM from gemm.cuh.
// The weight matrix is reshaped as [Cout x (Cin*K*K)] and treated as A.
// The im2col matrix is B, so C = A * B => [Cout x (Ho*Wo)] per image.
#include "common.h"
#include "gemm.cuh"

// ---------------------------------------------------------------------------
// im2col kernel: grid-stride loop
// For each output position (n, c*K*K_index, ho*Wo+wo_index), write the
// corresponding input element (with padding check).
// col layout: col[n][ c*K*K + r*K + s ][ oy * Wo + ox ] = in[n][c][iy][ix]
// Total elements: N * Cin * K * K * Ho * Wo
// ---------------------------------------------------------------------------
__global__ void im2col_kernel(const ConvParams p, const float* __restrict__ in,
                              float* __restrict__ col) {
    const size_t total = (size_t)p.N * p.Cin * p.K * p.K * p.Ho * p.Wo;
    const size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        // Decompose idx -> n, c, r, s, oy, ox
        size_t rem = idx;
        int ox = (int)(rem % p.Wo); rem /= p.Wo;
        int oy = (int)(rem % p.Ho); rem /= p.Ho;
        int s  = (int)(rem % p.K);  rem /= p.K;
        int r  = (int)(rem % p.K);  rem /= p.K;
        int c  = (int)(rem % p.Cin); rem /= p.Cin;
        int n  = (int)rem;

        int iy = oy * p.S - p.P + r;
        int ix = ox * p.S - p.P + s;

        float val = 0.0f;
        if (iy >= 0 && iy < p.H && ix >= 0 && ix < p.W) {
            val = in[(((size_t)n * p.Cin + c) * p.H + iy) * p.W + ix];
        }

        // col[n][c*K*K + r*K + s][oy*Wo + ox]
        int Kd = p.Cin * p.K * p.K;
        int Ncols = p.Ho * p.Wo;
        int k_idx = c * p.K * p.K + r * p.K + s;
        col[((size_t)n * Kd + k_idx) * Ncols + oy * p.Wo + ox] = val;
    }
}

// ---------------------------------------------------------------------------
// Workspace = im2col columns buffer: N * Cin * K * K * Ho * Wo floats
// ---------------------------------------------------------------------------
size_t v1_workspace_bytes(const ConvParams& p) {
    return (size_t)p.N * p.Cin * p.K * p.K * p.Ho * p.Wo * sizeof(float);
}

void im2col_launch(const ConvParams& p, const float* in, float* col) {
    size_t total = (size_t)p.N * p.Cin * p.K * p.K * p.Ho * p.Wo;
    int threads = 256;
    int blocks = (int)std::min((total + 255) / 256, (size_t)65535);
    im2col_kernel<<<blocks, threads>>>(p, in, col);
}

// ---------------------------------------------------------------------------
// DenseLoader: loads from the pre-materialized im2col columns buffer.
// col layout: col[n][k][col_idx] where k = c*K*K+r*K+s, col_idx = oy*Wo+ox
// ---------------------------------------------------------------------------
struct DenseLoader {
    const float* col;
    int Kd, Ncols;

    __device__ float operator()(int b, int k, int col_idx) const {
        return col[((size_t)b * Kd + k) * Ncols + col_idx];
    }
};

// ---------------------------------------------------------------------------
// V1: im2col + custom GEMM
// weight [Cout x Cin*K*K] is A.  im2col columns [Cin*K*K x Ho*Wo] is B.
// output [Cout x Ho*Wo] is C.  Batched over N.
// ---------------------------------------------------------------------------
void v1_run(const ConvParams& p, const float* in, const float* w, float* out, float* ws) {
    // Step 1: im2col
    im2col_launch(p, in, ws);

    // Step 2: batched GEMM:  out[n] = w * col[n]
    // A = w: [Cout x Kd],  B = col[n]: [Kd x Ncols],  C = out[n]: [Cout x Ncols]
    int Kd = p.Cin * p.K * p.K;
    int Ncols = p.Ho * p.Wo;

    DenseLoader loader{ws, Kd, Ncols};
    gemm_launch(w, loader, out, p.Cout, Ncols, Kd, p.N);
}
