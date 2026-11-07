// src/peak.cu -- peak bandwidth and compute benchmarks (owner: Faizan). Section 4.11.
// Writes results/peaks.csv with header metric,value and the rows
// peak_bandwidth_gbs and peak_gflops. Each is the highest of the 20 timed runs.
#include "common.h"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <chrono>

// Written only if the impossible sum below occurs, so the loop cannot be removed.
__device__ float g_peak_sink = 0.0f;

// ---------------------------------------------------------------------------
// Bandwidth kernel: copy float4 elements with a grid-stride loop
// ---------------------------------------------------------------------------
__global__ void copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst, size_t n4) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        dst[i] = src[i];
    }
}

// ---------------------------------------------------------------------------
// Compute kernel: 8 independent accumulators, ITERS fused multiply-adds on each.
// ---------------------------------------------------------------------------
__global__ void compute_kernel(int iters) {
    const float seed = (float)(blockIdx.x * blockDim.x + threadIdx.x);
    float a0 = seed + 1.0f, a1 = seed + 2.0f, a2 = seed + 3.0f, a3 = seed + 4.0f;
    float a4 = seed + 5.0f, a5 = seed + 6.0f, a6 = seed + 7.0f, a7 = seed + 8.0f;

    for (int i = 0; i < iters; ++i) {
        a0 = fmaf(a0, 0.9999f, 0.0001f);
        a1 = fmaf(a1, 0.9999f, 0.0001f);
        a2 = fmaf(a2, 0.9999f, 0.0001f);
        a3 = fmaf(a3, 0.9999f, 0.0001f);
        a4 = fmaf(a4, 0.9999f, 0.0001f);
        a5 = fmaf(a5, 0.9999f, 0.0001f);
        a6 = fmaf(a6, 0.9999f, 0.0001f);
        a7 = fmaf(a7, 0.9999f, 0.0001f);
    }

    float sum = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    if (sum == -1.0f) g_peak_sink = sum;
}

// ---------------------------------------------------------------------------
// gpu_spinup: keep the GPU busy so it has left its idle clocks. A few warm-up
// runs of a short kernel last milliseconds, which is too short for that.
// Alternates compute and memory traffic: the core clock and the memory clock
// ramp up separately, and compute load alone leaves the memory clock low.
// ---------------------------------------------------------------------------
void gpu_spinup(double seconds) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int blocks = 8 * prop.multiProcessorCount;
    const size_t bytes = (size_t)256 << 20;
    float4 *src = nullptr, *dst = nullptr;
    CUDA_CHECK(cudaMalloc(&src, bytes));
    CUDA_CHECK(cudaMalloc(&dst, bytes));
    CUDA_CHECK(cudaMemset(src, 1, bytes));
    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
        compute_kernel<<<blocks, 256>>>(8192);
        copy_kernel<<<1024, 256>>>(src, dst, bytes / sizeof(float4));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaFree(dst));
    CUDA_CHECK(cudaFree(src));
}

// ---------------------------------------------------------------------------
// run_peaks
// ---------------------------------------------------------------------------
static const int WARMUP = 5, REPS = 20;
static const double SPINUP_S = 3.0;

// Runs launch() 5 + 20 times and returns the shortest timed run, in ms.
template <class F>
static double fastest_ms(F launch) {
    gpu_spinup(SPINUP_S);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < WARMUP; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    double best = 1e30;
    for (int i = 0; i < REPS; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_CHECK(cudaGetLastError());
        best = std::min(best, (double)ms);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return best;
}

void run_peaks(const char* out_csv) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int sms = prop.multiProcessorCount;

    int clock_khz = 0, mem_clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0);
    printf("Device: %s, SMs: %d, clock %d MHz, memory clock %d MHz, bus %d bits\n",
           prop.name, sms, clock_khz / 1000, mem_clock_khz / 1000, prop.memoryBusWidth);

    // --- Bandwidth: two buffers of 64 x 2^20 floats (256 MiB each) -----------
    const size_t floats = (size_t)64 << 20;
    const size_t bytes  = floats * sizeof(float);
    const size_t n4     = floats / 4;
    float4 *src = nullptr, *dst = nullptr;
    CUDA_CHECK(cudaMalloc(&src, bytes));
    CUDA_CHECK(cudaMalloc(&dst, bytes));
    CUDA_CHECK(cudaMemset(src, 1, bytes));

    double bw_ms = fastest_ms([&]() { copy_kernel<<<1024, 256>>>(src, dst, n4); });
    double bw_gbs = 536870912.0 / (bw_ms * 1e6);    // read + write
    printf("Peak bandwidth: %.2f GB/s (fastest run %.4f ms)\n", bw_gbs, bw_ms);

    CUDA_CHECK(cudaFree(dst));
    CUDA_CHECK(cudaFree(src));

    // --- Compute --------------------------------------------------------------
    const int blocks = 8 * sms, iters = 4096;
    double flops = (double)blocks * 256 * iters * 8 * 2;
    double c_ms = fastest_ms([&]() { compute_kernel<<<blocks, 256>>>(iters); });
    double gflops = flops / (c_ms * 1e6);
    printf("Peak compute: %.2f GFLOP/s (fastest run %.4f ms, %d blocks)\n", gflops, c_ms, blocks);

    FILE* f = fopen(out_csv, "w");
    if (!f) { fprintf(stderr, "Cannot open %s for writing\n", out_csv); exit(1); }
    fprintf(f, "metric,value\n");
    fprintf(f, "peak_bandwidth_gbs,%.4f\n", bw_gbs);
    fprintf(f, "peak_gflops,%.4f\n", gflops);
    fclose(f);
    printf("Results written to %s\n", out_csv);
}
