// src/gemm.cuh -- shared GEMM kernels (owner: Mania). STUB: declarations only, see Section 5.1.
#pragma once
#define GEMM_USE_OPT 1   // set to 0 only under the fallback rule in Section 9.1
// Loader: a struct with  __device__ float operator()(int b, int k, int col) const;
// The GEMM kernels check k < Kd and col < Ncols before calling the loader.
template <class Loader> __global__ void gemm_simple(const float* A, Loader loadB, float* C, int M, int Ncols, int Kd);
template <class Loader> __global__ void gemm_opt   (const float* A, Loader loadB, float* C, int M, int Ncols, int Kd);
template <class Loader> void gemm_launch(const float* A, Loader loadB, float* C, int M, int Ncols, int Kd, int batch);
