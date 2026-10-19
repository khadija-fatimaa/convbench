// src/peak.cu -- peak bandwidth and compute benchmarks (owner: Faizan). Section 4.11.
// Measures:
//   1. Peak memory bandwidth (GB/s) using a float4 grid-stride copy kernel
//   2. Peak compute throughput (GFLOP/s) using fmaf in a tight loop
#include "common.h"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// Bandwidth kernel: copy float4 arrays with grid-stride loop
// ---------------------------------------------------------------------------
__global__ void copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst, size_t n4) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = idx; i < n4; i += stride) {
        dst[i] = src[i];
    }
}

// ---------------------------------------------------------------------------
// Compute kernel: fmaf loop. Each thread does ITERS fused multiply-adds.
// The result is written to prevent the compiler from optimising it away.
// ---------------------------------------------------------------------------
static const int COMPUTE_ITERS = 4096;

__global__ void compute_kernel(float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float a = 1.0f, b = 0.5f, c = 0.0f;
    #pragma unroll 32
    for (int i = 0; i < COMPUTE_ITERS; ++i) {
        c = fmaf(a, b, c);   // 2 FLOPs (1 mul + 1 add)
    }
    out[idx] = c;
}

// ---------------------------------------------------------------------------
// run_peaks: measure and write CSV
// ---------------------------------------------------------------------------
void run_peaks(const char* out_csv) {
    printf("=== Peak Hardware Benchmarks ===\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int clockRate = 0, memoryClockRate = 0;
    cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, 0);
    cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, 0);
    printf("Device: %s\n", prop.name);
    printf("SMs: %d, Clock: %d MHz, Mem Clock: %d MHz, Bus Width: %d bits\n",
           prop.multiProcessorCount, clockRate / 1000,
           memoryClockRate / 1000, prop.memoryBusWidth);

    // Theoretical peaks
    double theo_bw_gbps = 2.0 * memoryClockRate * 1e3 * (prop.memoryBusWidth / 8.0) / 1e9;
    printf("Theoretical peak bandwidth: %.1f GB/s\n", theo_bw_gbps);

    int warmup = 5, reps = 20;

    // --- Bandwidth benchmark: 256 MiB buffer ---
    {
        size_t bytes = 256 * 1024 * 1024;  // 256 MiB
        size_t n4 = bytes / sizeof(float4);
        float4 *d_src, *d_dst;
        CUDA_CHECK(cudaMalloc(&d_src, bytes));
        CUDA_CHECK(cudaMalloc(&d_dst, bytes));
        CUDA_CHECK(cudaMemset(d_src, 1, bytes));

        int threads = 256;
        int blocks = (int)std::min(n4 / threads, (size_t)prop.multiProcessorCount * 8);

        // Warmup
        for (int i = 0; i < warmup; ++i) {
            copy_kernel<<<blocks, threads>>>(d_src, d_dst, n4);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed runs
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        std::vector<float> times(reps);
        for (int i = 0; i < reps; ++i) {
            CUDA_CHECK(cudaEventRecord(start));
            copy_kernel<<<blocks, threads>>>(d_src, d_dst, n4);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
        }

        std::sort(times.begin(), times.end());
        float median_ms = times[reps / 2];
        double bw_gbps = 2.0 * bytes / (median_ms / 1e3) / 1e9;  // read + write

        printf("Measured bandwidth: %.1f GB/s (median %.3f ms for 256 MiB copy)\n",
               bw_gbps, median_ms);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        cudaFree(d_dst);
        cudaFree(d_src);

        // --- Compute benchmark ---
        int compute_threads = 256;
        int compute_blocks = prop.multiProcessorCount * 8;
        int total_threads = compute_blocks * compute_threads;
        float* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, total_threads * sizeof(float)));

        for (int i = 0; i < warmup; ++i) {
            compute_kernel<<<compute_blocks, compute_threads>>>(d_out, total_threads);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        std::vector<float> ctimes(reps);
        for (int i = 0; i < reps; ++i) {
            CUDA_CHECK(cudaEventRecord(start));
            compute_kernel<<<compute_blocks, compute_threads>>>(d_out, total_threads);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ctimes[i], start, stop));
        }

        std::sort(ctimes.begin(), ctimes.end());
        float compute_median_ms = ctimes[reps / 2];
        // Each thread does COMPUTE_ITERS fmaf = 2 FLOPs each
        double total_flops = (double)total_threads * COMPUTE_ITERS * 2.0;
        double gflops = total_flops / (compute_median_ms / 1e3) / 1e9;

        printf("Measured compute: %.1f GFLOP/s (median %.3f ms)\n", gflops, compute_median_ms);

        // Write CSV
        FILE* f = fopen(out_csv, "w");
        if (!f) { fprintf(stderr, "Cannot open %s for writing\n", out_csv); exit(1); }
        fprintf(f, "metric,value,unit\n");
        fprintf(f, "device,%s,\n", prop.name);
        fprintf(f, "sm_count,%d,\n", prop.multiProcessorCount);
        fprintf(f, "clock_mhz,%d,MHz\n", clockRate / 1000);
        fprintf(f, "mem_clock_mhz,%d,MHz\n", memoryClockRate / 1000);
        fprintf(f, "bus_width,%d,bits\n", prop.memoryBusWidth);
        fprintf(f, "theo_bw,%.2f,GB/s\n", theo_bw_gbps);
        fprintf(f, "measured_bw,%.2f,GB/s\n", bw_gbps);
        fprintf(f, "bw_median_ms,%.4f,ms\n", median_ms);
        fprintf(f, "measured_compute,%.2f,GFLOP/s\n", gflops);
        fprintf(f, "compute_median_ms,%.4f,ms\n", compute_median_ms);
        fclose(f);
        printf("Results written to %s\n", out_csv);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        cudaFree(d_out);
    }
}
