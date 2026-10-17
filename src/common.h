// src/common.h -- shared by all files. Do NOT edit without group agreement.
#pragma once
#include <cstddef>
#include <cstdio>
#include <cstdlib>

struct ConvParams {
    // batch, in-channels, height, width, out-channels, filter size, stride, padding
    int N, Cin, H, W, Cout, K, S, P;
    int Ho, Wo;   // output height and width (computed)
};

inline ConvParams make_params(int N, int Cin, int H, int W, int Cout, int K, int S, int P) {
    ConvParams p{N, Cin, H, W, Cout, K, S, P, 0, 0};
    p.Ho = (H + 2 * P - K) / S + 1;
    p.Wo = (W + 2 * P - K) / S + 1;
    return p;
}
inline size_t input_elems (const ConvParams& p) { return (size_t)p.N * p.Cin * p.H * p.W; }
inline size_t weight_elems(const ConvParams& p) { return (size_t)p.Cout * p.Cin * p.K * p.K; }
inline size_t output_elems(const ConvParams& p) { return (size_t)p.N * p.Cout * p.Ho * p.Wo; }
// input  index: ((n*Cin + c)*H + y)*W + x
// weight index: ((co*Cin + c)*K + r)*K + s
// output index: ((n*Cout + co)*Ho + oy)*Wo + ox

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(1); } } while (0)
#endif

// GPU versions. All pointers are DEVICE pointers.
// ws = workspace (nullptr if its size is 0).
// Rules: only launch kernels. No cudaMalloc, cudaFree,
// cudaMemcpy or cudaDeviceSynchronize inside.
size_t v1_workspace_bytes(const ConvParams& p);
void   v1_run(const ConvParams& p, const float* in, const float* w, float* out, float* ws);
size_t v2_workspace_bytes(const ConvParams& p);  // returns 0
void   v2_run(const ConvParams& p, const float* in, const float* w, float* out, float* ws);
size_t v3_workspace_bytes(const ConvParams& p);   // returns 0
// cob: 0 = naive kernel; 4, 8 or 16 = tiled kernel
void   v3_set_cob(int cob);
void   v3_run(const ConvParams& p, const float* in, const float* w, float* out, float* ws);
size_t v4_workspace_bytes(const ConvParams& p);   // same as V1
void   v4_init();     // create cuBLAS handle; called once by main
void   v4_destroy();
void   v4_run(const ConvParams& p, const float* in, const float* w, float* out, float* ws);
// in v1_im2col.cu, used by V1 and V4:
void   im2col_launch(const ConvParams& p, const float* in, float* col);
// in peak.cu:
void   run_peaks(const char* out_csv);

// CPU versions. All pointers are HOST pointers.
void v6_run(const ConvParams& p, const float* in, const float* w, float* out);
void v7_run(const ConvParams& p, const float* in, const float* w, float* out, int threads);
