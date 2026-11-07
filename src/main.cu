// src/main.cu -- test harness (owner: Faizan). Section 4.9.
//   convbench --configs <csv> --versions <list> --mode check|bench --out <csv>
//             [--data data] [--threads 1,4,8,12] [--warmup 5] [--reps 10] [--v3cob <0|4|8|16>]
//   convbench --peaks --out results/peaks.csv
// V5 is benchmarked by python/bench_cudnn.py, which writes the same CSV columns.
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
#include <limits>

// ---------------------------------------------------------------------------
// Command-line arguments
// ---------------------------------------------------------------------------
struct Args {
    std::string configs;
    std::string versions_str;
    std::string mode;         // "check" or "bench"
    std::string out;
    std::string data = "data";
    std::string threads_str = "12";
    int warmup = 5;
    int reps   = 10;
    int v3cob  = 16;          // 0 = naive, 4/8/16 = tiled (16 after tuning)
    bool peaks = false;
};

static void usage() {
    fprintf(stderr,
        "Usage: convbench --configs FILE --versions LIST --mode check|bench --out FILE\n"
        "                 [--data DIR] [--threads 1,4,8,12] [--warmup N] [--reps N] [--v3cob 0|4|8|16]\n"
        "       convbench --peaks --out FILE\n");
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto match = [&](const char* name) { return strcmp(argv[i], name) == 0; };
        if (match("--configs")  && i+1 < argc) { a.configs      = argv[++i]; continue; }
        if (match("--versions") && i+1 < argc) { a.versions_str = argv[++i]; continue; }
        if (match("--mode")     && i+1 < argc) { a.mode         = argv[++i]; continue; }
        if (match("--out")      && i+1 < argc) { a.out          = argv[++i]; continue; }
        if (match("--data")     && i+1 < argc) { a.data         = argv[++i]; continue; }
        if (match("--threads")  && i+1 < argc) { a.threads_str  = argv[++i]; continue; }
        if (match("--warmup")   && i+1 < argc) { a.warmup       = atoi(argv[++i]); continue; }
        if (match("--reps")     && i+1 < argc) { a.reps         = atoi(argv[++i]); continue; }
        if (match("--v3cob")    && i+1 < argc) { a.v3cob        = atoi(argv[++i]); continue; }
        if (match("--peaks"))   { a.peaks = true; continue; }
        fprintf(stderr, "Unknown argument: %s\n", argv[i]);
        usage();
        exit(1);
    }
    return a;
}

static std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> v;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) v.push_back(atoi(tok.c_str()));
    }
    return v;
}

// ---------------------------------------------------------------------------
// Config and binary loaders
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
        if (line.empty() || line == "\r") continue;
        ConfigRow r;
        for (char& c : line) if (c == ',') c = ' ';
        std::istringstream ss(line);
        ss >> r.id >> r.N >> r.Cin >> r.H >> r.W >> r.Cout >> r.K >> r.S >> r.P;
        if (r.K > 7 || r.S > 2) {
            fprintf(stderr, "Config %s has K=%d, S=%d: kernels require K <= 7 and S <= 2\n",
                    r.id.c_str(), r.K, r.S);
            exit(1);
        }
        rows.push_back(r);
    }
    return rows;
}

static std::vector<float> load_bin(const std::string& path, size_t expected_elems) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Missing data file: %s\n", path.c_str()); exit(1); }
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
// Correctness rule (Section 4.8): |out - ref| <= 0.001 + 0.001 * |ref|
// ---------------------------------------------------------------------------
static bool check_output(const float* out, const float* ref, size_t n, double& max_err) {
    max_err = 0.0;
    bool pass = true;
    for (size_t i = 0; i < n; ++i) {
        double err = fabs((double)out[i] - (double)ref[i]);
        double tol = 0.001 + 0.001 * fabs((double)ref[i]);
        if (!(err <= tol)) pass = false;      // written this way so NaN fails
        if (!(err <= max_err)) max_err = err;
    }
    return pass;
}

// Median: for an even count, the average of the two middle values.
static double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ---------------------------------------------------------------------------
// CSV output, Table 5. NaN in a numeric field is written as NA.
// ---------------------------------------------------------------------------
static const double NA = std::numeric_limits<double>::quiet_NaN();

struct Row {
    std::string config_id;
    int version = 0, threads = 0;
    std::string status = "OK";
    double kmed = NA, kmin = NA, kmax = NA;
    double e2e_pageable = NA, e2e_pinned = NA;
    double peak_mem_mb = NA, memgetinfo_mb = NA;
    double max_abs_err = NA;
};

static void put_num(FILE* f, double v) {
    if (std::isnan(v)) fprintf(f, ",NA");
    else               fprintf(f, ",%.6g", v);
}

static void write_row(FILE* f, const Row& r) {
    printf("  V%d%s: %s", r.version,
           r.threads ? (" t=" + std::to_string(r.threads)).c_str() : "", r.status.c_str());
    if (!std::isnan(r.kmed))        printf("  kernel=%.3f ms", r.kmed);
    if (!std::isnan(r.e2e_pinned))  printf("  e2e=%.3f/%.3f ms", r.e2e_pageable, r.e2e_pinned);
    if (!std::isnan(r.peak_mem_mb)) printf("  mem=%.1f MB (getinfo %.1f)", r.peak_mem_mb, r.memgetinfo_mb);
    if (!std::isnan(r.max_abs_err)) printf("  max_err=%.3e", r.max_abs_err);
    printf("\n");
    fflush(stdout);

    if (!f) return;
    fprintf(f, "%s,%d,%d,%s", r.config_id.c_str(), r.version, r.threads, r.status.c_str());
    put_num(f, r.kmed); put_num(f, r.kmin); put_num(f, r.kmax);
    put_num(f, r.e2e_pageable); put_num(f, r.e2e_pinned);
    put_num(f, r.peak_mem_mb); put_num(f, r.memgetinfo_mb);
    put_num(f, r.max_abs_err);
    fprintf(f, "\n");
    fflush(f);     // keep completed rows if a long run is interrupted
}

// ---------------------------------------------------------------------------
// Version dispatch
// ---------------------------------------------------------------------------
static size_t workspace_bytes(int ver, const ConvParams& p) {
    switch (ver) {
        case 1: return v1_workspace_bytes(p);
        case 2: return v2_workspace_bytes(p);
        case 3: return v3_workspace_bytes(p);
        case 4: return v4_workspace_bytes(p);
    }
    return 0;
}

static void run_gpu(int ver, const ConvParams& p, const float* in, const float* w,
                    float* out, float* ws) {
    switch (ver) {
        case 1: v1_run(p, in, w, out, ws); break;
        case 2: v2_run(p, in, w, out, ws); break;
        case 3: v3_run(p, in, w, out, ws); break;
        case 4: v4_run(p, in, w, out, ws); break;
    }
}

// ---------------------------------------------------------------------------
// Device buffers. cudaMalloc failure is reported, not fatal: OOM is a result.
// ---------------------------------------------------------------------------
struct DeviceBuffers {
    float* in = nullptr; float* w = nullptr; float* out = nullptr; float* ws = nullptr;

    void release() {
        if (ws)  cudaFree(ws);
        if (out) cudaFree(out);
        if (w)   cudaFree(w);
        if (in)  cudaFree(in);
        ws = out = w = in = nullptr;
    }
};

static bool try_alloc(float** ptr, size_t bytes) {
    cudaError_t e = cudaMalloc(ptr, bytes);
    if (e == cudaSuccess) return true;
    *ptr = nullptr;
    if (e != cudaErrorMemoryAllocation) {
        fprintf(stderr, "cudaMalloc error: %s\n", cudaGetErrorString(e));
        exit(1);
    }
    cudaGetLastError();   // clear the error
    return false;
}

static bool alloc_buffers(DeviceBuffers& b, size_t in_bytes, size_t w_bytes,
                          size_t out_bytes, size_t ws_bytes) {
    bool ok = try_alloc(&b.in, in_bytes) && try_alloc(&b.w, w_bytes)
           && try_alloc(&b.out, out_bytes)
           && (ws_bytes == 0 || try_alloc(&b.ws, ws_bytes));
    if (!ok) b.release();
    return ok;
}

// ---------------------------------------------------------------------------
// End-to-end time (pageable or pinned, depending on the host buffers passed in)
// ---------------------------------------------------------------------------
static double e2e_median(int ver, const ConvParams& p, const DeviceBuffers& d,
                         const float* h_in, const float* h_w, float* h_out,
                         size_t in_bytes, size_t w_bytes, size_t out_bytes,
                         int warmup, int reps) {
    std::vector<double> times;
    times.reserve(reps);
    for (int i = 0; i < warmup + reps; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaMemcpy(d.in, h_in, in_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.w,  h_w,  w_bytes,  cudaMemcpyHostToDevice));
        run_gpu(ver, p, d.in, d.w, d.out, d.ws);
        CUDA_CHECK(cudaMemcpy(h_out, d.out, out_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaGetLastError());
        if (i >= warmup)
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return median_of(times);
}

// GPU spin-up before timing: the first config must not run at idle clocks.
static const double START_SPINUP_S = 3.0, ROW_SPINUP_S = 0.5;

// ---------------------------------------------------------------------------
// One GPU version on one config (Section 4.9, steps 2 to 11)
// ---------------------------------------------------------------------------
static Row do_gpu(int ver, const ConfigRow& cfg, const ConvParams& p,
                  const float* h_in, const float* h_w, const float* h_ref,
                  bool bench, int warmup, int reps) {
    Row r;
    r.config_id = cfg.id;
    r.version = ver;

    size_t in_bytes  = input_elems(p)  * sizeof(float);
    size_t w_bytes   = weight_elems(p) * sizeof(float);
    size_t out_bytes = output_elems(p) * sizeof(float);
    size_t ws_bytes  = workspace_bytes(ver, p);

    size_t free_before = 0, free_after = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total));

    DeviceBuffers d;
    if (!alloc_buffers(d, in_bytes, w_bytes, out_bytes, ws_bytes)) {
        r.status = "OOM";
        return r;
    }

    CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
    r.memgetinfo_mb = ((double)free_before - (double)free_after) / 1048576.0;
    r.peak_mem_mb   = (double)(in_bytes + w_bytes + out_bytes + ws_bytes) / 1048576.0;

    CUDA_CHECK(cudaMemcpy(d.in, h_in, in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.w,  h_w,  w_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d.out, 0, out_bytes));   // untimed; a stale buffer must not pass the check

    std::vector<float> h_out(output_elems(p));

    if (!bench) {
        run_gpu(ver, p, d.in, d.w, d.out, d.ws);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        gpu_spinup(ROW_SPINUP_S);
        for (int i = 0; i < warmup; ++i) {
            run_gpu(ver, p, d.in, d.w, d.out, d.ws);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        std::vector<double> times;
        times.reserve(reps);
        for (int i = 0; i < reps; ++i) {
            CUDA_CHECK(cudaEventRecord(start));
            run_gpu(ver, p, d.in, d.w, d.out, d.ws);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            CUDA_CHECK(cudaGetLastError());
            times.push_back(ms);
        }
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        r.kmed = median_of(times);
        r.kmin = *std::min_element(times.begin(), times.end());
        r.kmax = *std::max_element(times.begin(), times.end());
    }

    CUDA_CHECK(cudaMemcpy(h_out.data(), d.out, out_bytes, cudaMemcpyDeviceToHost));
    bool pass = check_output(h_out.data(), h_ref, output_elems(p), r.max_abs_err);
    r.status = pass ? "OK" : "FAIL";

    if (bench) {
        // End-to-end, pageable: ordinary host buffers.
        std::vector<float> pg_in(h_in, h_in + input_elems(p));
        std::vector<float> pg_w(h_w, h_w + weight_elems(p));
        std::vector<float> pg_out(output_elems(p));
        r.e2e_pageable = e2e_median(ver, p, d, pg_in.data(), pg_w.data(), pg_out.data(),
                                    in_bytes, w_bytes, out_bytes, warmup, reps);

        // End-to-end, pinned: host buffers from cudaMallocHost.
        float *pin_in = nullptr, *pin_w = nullptr, *pin_out = nullptr;
        CUDA_CHECK(cudaMallocHost(&pin_in,  in_bytes));
        CUDA_CHECK(cudaMallocHost(&pin_w,   w_bytes));
        CUDA_CHECK(cudaMallocHost(&pin_out, out_bytes));
        memcpy(pin_in, h_in, in_bytes);
        memcpy(pin_w,  h_w,  w_bytes);
        r.e2e_pinned = e2e_median(ver, p, d, pin_in, pin_w, pin_out,
                                  in_bytes, w_bytes, out_bytes, warmup, reps);
        CUDA_CHECK(cudaFreeHost(pin_out));
        CUDA_CHECK(cudaFreeHost(pin_w));
        CUDA_CHECK(cudaFreeHost(pin_in));
    }

    d.release();
    return r;
}

// ---------------------------------------------------------------------------
// One CPU version on one config
// ---------------------------------------------------------------------------
static Row do_cpu(int ver, int threads, const ConfigRow& cfg, const ConvParams& p,
                  const float* h_in, const float* h_w, const float* h_ref,
                  bool bench, int warmup, int reps) {
    Row r;
    r.config_id = cfg.id;
    r.version = ver;
    r.threads = (ver == 7) ? threads : 0;

    std::vector<float> h_out(output_elems(p), 0.0f);
    auto run = [&]() {
        if (ver == 6) v6_run(p, h_in, h_w, h_out.data());
        else          v7_run(p, h_in, h_w, h_out.data(), threads);
    };

    if (!bench) {
        run();
    } else {
        std::vector<double> times;
        times.reserve(reps);
        for (int i = 0; i < warmup + reps; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            run();
            auto t1 = std::chrono::steady_clock::now();
            if (i >= warmup)
                times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        r.kmed = median_of(times);
        r.kmin = *std::min_element(times.begin(), times.end());
        r.kmax = *std::max_element(times.begin(), times.end());
    }

    bool pass = check_output(h_out.data(), h_ref, output_elems(p), r.max_abs_err);
    r.status = pass ? "OK" : "FAIL";
    return r;
}

// ===================================================================
// MAIN
// ===================================================================
int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    if (a.peaks) {
        if (a.out.empty()) { fprintf(stderr, "--peaks requires --out\n"); return 1; }
        run_peaks(a.out.c_str());
        return 0;
    }

    if (a.configs.empty() || a.versions_str.empty() ||
        (a.mode != "check" && a.mode != "bench")) {
        usage();
        return 1;
    }
    if (a.v3cob != 0 && a.v3cob != 4 && a.v3cob != 8 && a.v3cob != 16) {
        fprintf(stderr, "--v3cob must be 0, 4, 8 or 16\n");
        return 1;
    }

    std::vector<int> versions = parse_int_list(a.versions_str);
    std::vector<int> thread_counts = parse_int_list(a.threads_str);
    std::vector<ConfigRow> configs = load_configs(a.configs);
    bool bench = (a.mode == "bench");

    for (int v : versions) {
        if (v == 5) { fprintf(stderr, "V5 is run by python/bench_cudnn.py\n"); return 1; }
        if (v < 1 || v > 7) { fprintf(stderr, "Unknown version %d\n", v); return 1; }
    }
    if (thread_counts.empty()) { fprintf(stderr, "--threads is empty\n"); return 1; }

    v3_set_cob(a.v3cob);
    v4_init();

    FILE* fout = nullptr;
    if (!a.out.empty()) {
        fout = fopen(a.out.c_str(), "w");
        if (!fout) { fprintf(stderr, "Cannot open output file: %s\n", a.out.c_str()); return 1; }
        fprintf(fout, "config_id,version,threads,status,kernel_ms_median,kernel_ms_min,kernel_ms_max,"
                      "e2e_pageable_ms_median,e2e_pinned_ms_median,peak_mem_mb,memgetinfo_mb,max_abs_err\n");
    }

    int rows = 0, ok = 0;

    bool any_gpu = std::any_of(versions.begin(), versions.end(), [](int v) { return v <= 4; });
    if (bench && any_gpu) gpu_spinup(START_SPINUP_S);

    for (const auto& cfg : configs) {
        ConvParams p = make_params(cfg.N, cfg.Cin, cfg.H, cfg.W, cfg.Cout, cfg.K, cfg.S, cfg.P);
        printf("Config %s: N=%d Cin=%d H=%d W=%d Cout=%d K=%d S=%d P=%d => Ho=%d Wo=%d\n",
               cfg.id.c_str(), p.N, p.Cin, p.H, p.W, p.Cout, p.K, p.S, p.P, p.Ho, p.Wo);

        std::string base = a.data + "/" + cfg.id;
        std::vector<float> h_in  = load_bin(base + "_input.bin",  input_elems(p));
        std::vector<float> h_w   = load_bin(base + "_weight.bin", weight_elems(p));
        std::vector<float> h_ref = load_bin(base + "_output.bin", output_elems(p));

        for (int ver : versions) {
            if (ver >= 1 && ver <= 4) {
                Row r = do_gpu(ver, cfg, p, h_in.data(), h_w.data(), h_ref.data(),
                               bench, a.warmup, a.reps);
                write_row(fout, r);
                ++rows; if (r.status == "OK") ++ok;
            } else if (ver == 6) {
                Row r = do_cpu(6, 0, cfg, p, h_in.data(), h_w.data(), h_ref.data(),
                               bench, a.warmup, a.reps);
                write_row(fout, r);
                ++rows; if (r.status == "OK") ++ok;
            } else {   // ver == 7: one row per thread count
                for (int t : thread_counts) {
                    Row r = do_cpu(7, t, cfg, p, h_in.data(), h_w.data(), h_ref.data(),
                                   bench, a.warmup, a.reps);
                    write_row(fout, r);
                    ++rows; if (r.status == "OK") ++ok;
                }
            }
        }
    }

    printf("\n=== SUMMARY: %d / %d rows OK ===\n", ok, rows);
    if (fout) fclose(fout);
    v4_destroy();
    return (ok == rows) ? 0 : 1;
}
