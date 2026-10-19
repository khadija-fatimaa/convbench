// src/main.cu -- test harness (owner: Faizan). Section 4.9.
// Modes:
//   check  – load reference data, run each version, compare element-wise
//   bench  – warm up, then time kernel-only and end-to-end (pageable + pinned)
// Usage:
//   convbench --configs configs/test.csv --versions 1,2,3,4,6,7 --mode check --out results/check.csv
//   convbench --configs configs/test.csv --versions 1,2,3 --mode bench --out results/bench.csv
//   convbench --peaks --out results/peaks.csv
#include "common.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

// ---------------------------------------------------------------------------
// Command-line arguments
// ---------------------------------------------------------------------------
struct Args {
    std::string configs;
    std::string versions_str;
    std::string mode;         // "check" or "bench"
    std::string out;
    std::string datadir = "data";
    int threads  = 8;
    int warmup   = 3;
    int reps     = 10;
    int v3cob    = 0;         // 0 = naive, 4/8/16 = tiled
    bool peaks   = false;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto match = [&](const char* name) { return strcmp(argv[i], name) == 0; };
        if (match("--configs")  && i+1 < argc) { a.configs  = argv[++i]; continue; }
        if (match("--versions") && i+1 < argc) { a.versions_str = argv[++i]; continue; }
        if (match("--mode")     && i+1 < argc) { a.mode     = argv[++i]; continue; }
        if (match("--out")      && i+1 < argc) { a.out      = argv[++i]; continue; }
        if (match("--datadir")  && i+1 < argc) { a.datadir  = argv[++i]; continue; }
        if (match("--threads")  && i+1 < argc) { a.threads  = atoi(argv[++i]); continue; }
        if (match("--warmup")   && i+1 < argc) { a.warmup   = atoi(argv[++i]); continue; }
        if (match("--reps")     && i+1 < argc) { a.reps     = atoi(argv[++i]); continue; }
        if (match("--v3cob")    && i+1 < argc) { a.v3cob    = atoi(argv[++i]); continue; }
        if (match("--peaks"))   { a.peaks = true; continue; }
        fprintf(stderr, "Unknown argument: %s\n", argv[i]);
        exit(1);
    }
    return a;
}

static std::vector<int> parse_versions(const std::string& s) {
    std::vector<int> v;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) v.push_back(atoi(tok.c_str()));
    }
    return v;
}

// ---------------------------------------------------------------------------
// Config loader
// ---------------------------------------------------------------------------
struct ConfigRow {
    std::string id;
    int N, Cin, H, W, Cout, K, S, P;
};

static std::vector<ConfigRow> load_configs(const std::string& path) {
    std::vector<ConfigRow> rows;
    std::ifstream f(path);
    if (!f.is_open()) { fprintf(stderr, "Cannot open config file: %s\n", path.c_str()); exit(1); }
    std::string line;
    std::getline(f, line); // skip header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        ConfigRow r;
        // Replace commas with spaces for easy parsing
        for (char& c : line) if (c == ',') c = ' ';
        std::istringstream ss(line);
        ss >> r.id >> r.N >> r.Cin >> r.H >> r.W >> r.Cout >> r.K >> r.S >> r.P;
        rows.push_back(r);
    }
    return rows;
}

// ---------------------------------------------------------------------------
// Binary file loader
// ---------------------------------------------------------------------------
static std::vector<float> load_bin(const std::string& path, size_t expected_elems) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); exit(1); }
    std::vector<float> buf(expected_elems);
    size_t got = fread(buf.data(), sizeof(float), expected_elems, f);
    fclose(f);
    if (got != expected_elems) {
        fprintf(stderr, "%s: expected %zu floats, got %zu\n", path.c_str(), expected_elems, got);
        exit(1);
    }
    return buf;
}

// ---------------------------------------------------------------------------
// Correctness check: |out - ref| <= atol + rtol * |ref|
// ---------------------------------------------------------------------------
static bool check_output(const float* out, const float* ref, size_t n,
                          float atol, float rtol, float& max_err) {
    max_err = 0.0f;
    bool pass = true;
    for (size_t i = 0; i < n; ++i) {
        float err = fabsf(out[i] - ref[i]);
        float tol = atol + rtol * fabsf(ref[i]);
        if (err > tol) { pass = false; }
        if (err > max_err) max_err = err;
    }
    return pass;
}

// ---------------------------------------------------------------------------
// Version name helper
// ---------------------------------------------------------------------------
static const char* version_name(int v) {
    switch (v) {
        case 1: return "V1_im2col";
        case 2: return "V2_implicit";
        case 3: return "V3_direct";
        case 4: return "V4_cublas";
        case 6: return "V6_cpu";
        case 7: return "V7_omp";
        default: return "unknown";
    }
}

// ---------------------------------------------------------------------------
// Run a GPU version: allocate device memory, copy in, run, copy out
// ---------------------------------------------------------------------------
static void run_gpu_version(int ver, const ConvParams& p,
                            const float* h_in, const float* h_w,
                            float* h_out) {
    size_t in_bytes  = input_elems(p) * sizeof(float);
    size_t w_bytes   = weight_elems(p) * sizeof(float);
    size_t out_bytes = output_elems(p) * sizeof(float);

    // Workspace
    size_t ws_bytes = 0;
    switch (ver) {
        case 1: ws_bytes = v1_workspace_bytes(p); break;
        case 2: ws_bytes = v2_workspace_bytes(p); break;
        case 3: ws_bytes = v3_workspace_bytes(p); break;
        case 4: ws_bytes = v4_workspace_bytes(p); break;
    }

    float *d_in = nullptr, *d_w = nullptr, *d_out = nullptr, *d_ws = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  in_bytes));
    CUDA_CHECK(cudaMalloc(&d_w,   w_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    if (ws_bytes > 0) CUDA_CHECK(cudaMalloc(&d_ws, ws_bytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w,  h_w,  w_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, out_bytes));

    switch (ver) {
        case 1: v1_run(p, d_in, d_w, d_out, d_ws); break;
        case 2: v2_run(p, d_in, d_w, d_out, d_ws); break;
        case 3: v3_run(p, d_in, d_w, d_out, d_ws); break;
        case 4: v4_run(p, d_in, d_w, d_out, d_ws); break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));

    if (d_ws) cudaFree(d_ws);
    cudaFree(d_out);
    cudaFree(d_w);
    cudaFree(d_in);
}

// ---------------------------------------------------------------------------
// Run a CPU version
// ---------------------------------------------------------------------------
static void run_cpu_version(int ver, const ConvParams& p,
                            const float* h_in, const float* h_w,
                            float* h_out, int threads) {
    memset(h_out, 0, output_elems(p) * sizeof(float));
    switch (ver) {
        case 6: v6_run(p, h_in, h_w, h_out); break;
        case 7: v7_run(p, h_in, h_w, h_out, threads); break;
    }
}

// ---------------------------------------------------------------------------
// Bench a GPU version: returns kernel-only time in ms (median of reps)
// ---------------------------------------------------------------------------
static double bench_gpu_kernel(int ver, const ConvParams& p,
                               const float* d_in, const float* d_w,
                               float* d_out, float* d_ws,
                               int warmup, int reps) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto run_ver = [&]() {
        CUDA_CHECK(cudaMemset(d_out, 0, output_elems(p) * sizeof(float)));
        switch (ver) {
            case 1: v1_run(p, d_in, d_w, d_out, d_ws); break;
            case 2: v2_run(p, d_in, d_w, d_out, d_ws); break;
            case 3: v3_run(p, d_in, d_w, d_out, d_ws); break;
            case 4: v4_run(p, d_in, d_w, d_out, d_ws); break;
        }
    };

    // Warmup
    for (int i = 0; i < warmup; ++i) { run_ver(); }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    std::vector<float> times(reps);
    for (int i = 0; i < reps; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        run_ver();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(times.begin(), times.end());
    return (double)times[reps / 2];
}

// ---------------------------------------------------------------------------
// Bench a GPU version: end-to-end (pageable host memory)
// ---------------------------------------------------------------------------
static double bench_gpu_e2e_pageable(int ver, const ConvParams& p,
                                     const float* h_in, const float* h_w,
                                     float* h_out,
                                     int warmup, int reps) {
    size_t in_bytes  = input_elems(p) * sizeof(float);
    size_t w_bytes   = weight_elems(p) * sizeof(float);
    size_t out_bytes = output_elems(p) * sizeof(float);
    size_t ws_bytes  = 0;
    switch (ver) {
        case 1: ws_bytes = v1_workspace_bytes(p); break;
        case 2: ws_bytes = v2_workspace_bytes(p); break;
        case 3: ws_bytes = v3_workspace_bytes(p); break;
        case 4: ws_bytes = v4_workspace_bytes(p); break;
    }

    float *d_in, *d_w, *d_out, *d_ws = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  in_bytes));
    CUDA_CHECK(cudaMalloc(&d_w,   w_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    if (ws_bytes > 0) CUDA_CHECK(cudaMalloc(&d_ws, ws_bytes));

    auto run_e2e = [&]() {
        CUDA_CHECK(cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w,  h_w,  w_bytes,  cudaMemcpyHostToDevice));
        switch (ver) {
            case 1: v1_run(p, d_in, d_w, d_out, d_ws); break;
            case 2: v2_run(p, d_in, d_w, d_out, d_ws); break;
            case 3: v3_run(p, d_in, d_w, d_out, d_ws); break;
            case 4: v4_run(p, d_in, d_w, d_out, d_ws); break;
        }
        CUDA_CHECK(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));
    };

    // Warmup
    for (int i = 0; i < warmup; ++i) run_e2e();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times(reps);
    for (int i = 0; i < reps; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        run_e2e();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    if (d_ws) cudaFree(d_ws);
    cudaFree(d_out); cudaFree(d_w); cudaFree(d_in);

    std::sort(times.begin(), times.end());
    return (double)times[reps / 2];
}

// ---------------------------------------------------------------------------
// Bench a CPU version: returns wall-clock time in ms (median of reps)
// ---------------------------------------------------------------------------
static double bench_cpu(int ver, const ConvParams& p,
                        const float* h_in, const float* h_w,
                        float* h_out, int threads,
                        int warmup, int reps) {
    auto run_ver = [&]() {
        memset(h_out, 0, output_elems(p) * sizeof(float));
        switch (ver) {
            case 6: v6_run(p, h_in, h_w, h_out); break;
            case 7: v7_run(p, h_in, h_w, h_out, threads); break;
        }
    };

    for (int i = 0; i < warmup; ++i) run_ver();

    std::vector<double> times(reps);
    for (int i = 0; i < reps; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_ver();
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    std::sort(times.begin(), times.end());
    return times[reps / 2];
}

// ---------------------------------------------------------------------------
// FLOPs for a convolution: 2 * N * Cout * Ho * Wo * Cin * K * K
// ---------------------------------------------------------------------------
static double conv_flops(const ConvParams& p) {
    return 2.0 * p.N * p.Cout * p.Ho * p.Wo * p.Cin * p.K * p.K;
}

// ---------------------------------------------------------------------------
// Memory check: can we fit this config on the GPU?
// ---------------------------------------------------------------------------
static bool check_gpu_memory(const ConvParams& p, int ver) {
    size_t in_bytes  = input_elems(p) * sizeof(float);
    size_t w_bytes   = weight_elems(p) * sizeof(float);
    size_t out_bytes = output_elems(p) * sizeof(float);
    size_t ws_bytes  = 0;
    switch (ver) {
        case 1: ws_bytes = v1_workspace_bytes(p); break;
        case 2: ws_bytes = v2_workspace_bytes(p); break;
        case 3: ws_bytes = v3_workspace_bytes(p); break;
        case 4: ws_bytes = v4_workspace_bytes(p); break;
    }
    size_t total_needed = in_bytes + w_bytes + out_bytes + ws_bytes;

    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    // Leave a 64 MiB safety margin
    return total_needed + 64 * 1024 * 1024 < free_mem;
}

// ===================================================================
// MAIN
// ===================================================================
int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    // --peaks mode
    if (a.peaks) {
        if (a.out.empty()) { fprintf(stderr, "--peaks requires --out\n"); return 1; }
        run_peaks(a.out.c_str());
        return 0;
    }

    if (a.configs.empty() || a.versions_str.empty() || a.mode.empty()) {
        fprintf(stderr, "Usage: convbench --configs FILE --versions LIST --mode check|bench --out FILE [options]\n");
        fprintf(stderr, "  --threads N     CPU thread count for V7 (default 8)\n");
        fprintf(stderr, "  --warmup N      warmup iterations (default 3)\n");
        fprintf(stderr, "  --reps N        timed repetitions (default 10)\n");
        fprintf(stderr, "  --v3cob N       V3 channel block: 0=naive, 4, 8, 16 (default 0)\n");
        fprintf(stderr, "  --datadir DIR   reference data directory (default data)\n");
        fprintf(stderr, "  --peaks         run peak bandwidth/compute benchmark\n");
        return 1;
    }

    std::vector<int> versions = parse_versions(a.versions_str);
    std::vector<ConfigRow> configs = load_configs(a.configs);

    // Set V3 COB
    v3_set_cob(a.v3cob);

    // Init cuBLAS if V4 is requested
    bool need_v4 = false;
    for (int v : versions) if (v == 4) { need_v4 = true; break; }
    if (need_v4) v4_init();

    // Open output CSV
    FILE* fout = nullptr;
    if (!a.out.empty()) {
        fout = fopen(a.out.c_str(), "w");
        if (!fout) { fprintf(stderr, "Cannot open output file: %s\n", a.out.c_str()); return 1; }
    }

    bool is_check = (a.mode == "check");
    bool is_bench = (a.mode == "bench");

    if (fout && is_check) {
        fprintf(fout, "config,version,pass,max_err\n");
    }
    if (fout && is_bench) {
        fprintf(fout, "config,version,threads,v3cob,kernel_ms,e2e_pageable_ms,gflops_kernel,gflops_e2e\n");
    }

    int total_checks = 0, total_pass = 0;

    for (const auto& cfg : configs) {
        ConvParams p = make_params(cfg.N, cfg.Cin, cfg.H, cfg.W, cfg.Cout, cfg.K, cfg.S, cfg.P);
        printf("Config %s: N=%d Cin=%d H=%d W=%d Cout=%d K=%d S=%d P=%d => Ho=%d Wo=%d\n",
               cfg.id.c_str(), p.N, p.Cin, p.H, p.W, p.Cout, p.K, p.S, p.P, p.Ho, p.Wo);

        // Load reference data for check mode
        std::vector<float> h_in, h_w, h_ref;
        if (is_check) {
            std::string in_path  = a.datadir + "/" + cfg.id + "_input.bin";
            std::string w_path   = a.datadir + "/" + cfg.id + "_weight.bin";
            std::string ref_path = a.datadir + "/" + cfg.id + "_output.bin";
            h_in  = load_bin(in_path,  input_elems(p));
            h_w   = load_bin(w_path,   weight_elems(p));
            h_ref = load_bin(ref_path, output_elems(p));
        }

        for (int ver : versions) {
            bool is_gpu = (ver >= 1 && ver <= 4);
            bool is_cpu = (ver == 6 || ver == 7);

            // GPU memory check
            if (is_gpu && !check_gpu_memory(p, ver)) {
                printf("  %s: SKIPPED (OOM)\n", version_name(ver));
                if (fout && is_check) fprintf(fout, "%s,%s,OOM,0\n", cfg.id.c_str(), version_name(ver));
                if (fout && is_bench) fprintf(fout, "%s,%s,%d,%d,OOM,OOM,0,0\n",
                    cfg.id.c_str(), version_name(ver), is_cpu ? a.threads : 0, a.v3cob);
                continue;
            }

            if (is_check) {
                // ---- CHECK MODE ----
                std::vector<float> h_out(output_elems(p), 0.0f);
                total_checks++;

                if (is_gpu) {
                    run_gpu_version(ver, p, h_in.data(), h_w.data(), h_out.data());
                } else {
                    run_cpu_version(ver, p, h_in.data(), h_w.data(), h_out.data(), a.threads);
                }

                float max_err = 0.0f;
                bool pass = check_output(h_out.data(), h_ref.data(), output_elems(p),
                                         0.001f, 0.001f, max_err);
                if (pass) total_pass++;

                printf("  %s: %s (max_err=%.6e)\n", version_name(ver),
                       pass ? "PASS" : "FAIL", max_err);
                if (fout) fprintf(fout, "%s,%s,%s,%.6e\n", cfg.id.c_str(), version_name(ver),
                                  pass ? "PASS" : "FAIL", max_err);

            } else if (is_bench) {
                // ---- BENCH MODE ----
                double flops = conv_flops(p);
                int threads_used = is_cpu ? a.threads : 0;

                if (is_gpu) {
                    // Allocate device memory
                    size_t in_bytes  = input_elems(p) * sizeof(float);
                    size_t w_bytes   = weight_elems(p) * sizeof(float);
                    size_t out_bytes = output_elems(p) * sizeof(float);
                    size_t ws_bytes  = 0;
                    switch (ver) {
                        case 1: ws_bytes = v1_workspace_bytes(p); break;
                        case 2: ws_bytes = v2_workspace_bytes(p); break;
                        case 3: ws_bytes = v3_workspace_bytes(p); break;
                        case 4: ws_bytes = v4_workspace_bytes(p); break;
                    }
                    float *d_in, *d_w, *d_out, *d_ws = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_in,  in_bytes));
                    CUDA_CHECK(cudaMalloc(&d_w,   w_bytes));
                    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
                    if (ws_bytes > 0) CUDA_CHECK(cudaMalloc(&d_ws, ws_bytes));

                    // Upload data (generate random for bench mode if reference not loaded)
                    std::vector<float> tmp_in(input_elems(p));
                    std::vector<float> tmp_w(weight_elems(p));
                    for (size_t i = 0; i < tmp_in.size(); ++i) tmp_in[i] = (float)(rand() % 1000) / 500.0f - 1.0f;
                    for (size_t i = 0; i < tmp_w.size();  ++i) tmp_w[i]  = (float)(rand() % 1000) / 500.0f - 1.0f;
                    CUDA_CHECK(cudaMemcpy(d_in, tmp_in.data(), in_bytes, cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_w,  tmp_w.data(),  w_bytes,  cudaMemcpyHostToDevice));

                    double kernel_ms = bench_gpu_kernel(ver, p, d_in, d_w, d_out, d_ws,
                                                        a.warmup, a.reps);

                    std::vector<float> tmp_out(output_elems(p));
                    double e2e_ms = bench_gpu_e2e_pageable(ver, p, tmp_in.data(), tmp_w.data(),
                                                           tmp_out.data(), a.warmup, a.reps);

                    double gflops_kernel = (flops / 1e9) / (kernel_ms / 1e3);
                    double gflops_e2e    = (flops / 1e9) / (e2e_ms / 1e3);

                    printf("  %s: kernel=%.3f ms (%.1f GFLOP/s), e2e_pageable=%.3f ms (%.1f GFLOP/s)\n",
                           version_name(ver), kernel_ms, gflops_kernel, e2e_ms, gflops_e2e);
                    if (fout) fprintf(fout, "%s,%s,%d,%d,%.4f,%.4f,%.2f,%.2f\n",
                                      cfg.id.c_str(), version_name(ver), threads_used, a.v3cob,
                                      kernel_ms, e2e_ms, gflops_kernel, gflops_e2e);

                    if (d_ws) cudaFree(d_ws);
                    cudaFree(d_out); cudaFree(d_w); cudaFree(d_in);

                } else {
                    // CPU bench
                    std::vector<float> tmp_in(input_elems(p));
                    std::vector<float> tmp_w(weight_elems(p));
                    std::vector<float> tmp_out(output_elems(p));
                    for (size_t i = 0; i < tmp_in.size(); ++i) tmp_in[i] = (float)(rand() % 1000) / 500.0f - 1.0f;
                    for (size_t i = 0; i < tmp_w.size();  ++i) tmp_w[i]  = (float)(rand() % 1000) / 500.0f - 1.0f;

                    double cpu_ms = bench_cpu(ver, p, tmp_in.data(), tmp_w.data(),
                                              tmp_out.data(), a.threads, a.warmup, a.reps);
                    double gflops = (flops / 1e9) / (cpu_ms / 1e3);

                    printf("  %s (threads=%d): %.3f ms (%.1f GFLOP/s)\n",
                           version_name(ver), a.threads, cpu_ms, gflops);
                    if (fout) fprintf(fout, "%s,%s,%d,%d,%.4f,%.4f,%.2f,%.2f\n",
                                      cfg.id.c_str(), version_name(ver), threads_used, a.v3cob,
                                      cpu_ms, cpu_ms, gflops, gflops);
                }
            }
        }
    }

    if (is_check) {
        printf("\n=== CHECK SUMMARY: %d / %d passed ===\n", total_pass, total_checks);
    }

    if (fout) fclose(fout);
    if (need_v4) v4_destroy();

    return (is_check && total_pass < total_checks) ? 1 : 0;
}
