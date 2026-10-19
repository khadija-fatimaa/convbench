// src/v2_implicit.cu -- V2 implicit GEMM (owner: Mania). Section 5.3.
// Instead of materialising the im2col columns, we compute B[k][col] on the fly
// using an Im2colLoader functor that maps (batch, k, col) -> input element.
// No workspace is needed.
#include "common.h"
#include "gemm.cuh"

// ---------------------------------------------------------------------------
// Im2colLoader: computes im2col elements on the fly.
// Given (batch_idx, k, col):
//   k   = c * K * K + r * K + s   (flattened input channel and filter position)
//   col = oy * Wo + ox            (flattened output spatial position)
// Returns in[n][c][iy][ix] where iy = oy*S - P + r, ix = ox*S - P + s.
// Returns 0.0f if outside the input image (padding).
// ---------------------------------------------------------------------------
struct Im2colLoader {
    const float* in;
    int Cin, H, W, K, S, P, Ho, Wo;

    __device__ float operator()(int n, int k, int col) const {
        int KK = K * K;
        int c = k / KK;
        int rs = k % KK;
        int r = rs / K;
        int s = rs % K;

        int oy = col / Wo;
        int ox = col % Wo;

        int iy = oy * S - P + r;
        int ix = ox * S - P + s;

        if (iy < 0 || iy >= H || ix < 0 || ix >= W) return 0.0f;

        return in[(((size_t)n * Cin + c) * H + iy) * W + ix];
    }
};

// ---------------------------------------------------------------------------
// V2 needs no workspace
// ---------------------------------------------------------------------------
size_t v2_workspace_bytes(const ConvParams& p) { (void)p; return 0; }

// ---------------------------------------------------------------------------
// V2: implicit GEMM convolution
// weight [Cout x Cin*K*K] is A,  im2col on-the-fly is B,  output is C.
// ---------------------------------------------------------------------------
void v2_run(const ConvParams& p, const float* in, const float* w, float* out, float* ws) {
    (void)ws;   // V2 needs no workspace

    int Kd = p.Cin * p.K * p.K;
    int Ncols = p.Ho * p.Wo;

    Im2colLoader loader{in, p.Cin, p.H, p.W, p.K, p.S, p.P, p.Ho, p.Wo};
    gemm_launch(w, loader, out, p.Cout, Ncols, Kd, p.N);
}
