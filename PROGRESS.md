# ConvBench Execution Progress

**Project:** Parallel and Distributed Computing (PDC) Group Project  
**Repository:** [khadija-fatimaa/convbench](https://github.com/khadija-fatimaa/convbench)  
**Authors:** Faizan Tanveer (23L-0640), Mania Ali (23L-0876), Khadija Fatima (23L-0976)  
**Target Platform:** Laptop GPU (NVIDIA GeForce RTX 4050 6GB Laptop GPU, 13th Gen Intel Core i5-13420H)  
**Last Updated:** 2026-11-07  

---

## 1. Overall Milestone Dashboard

| Step | Phase Description | Owner | Status | Deliverables / Artifacts |
| :--- | :--- | :--- | :---: | :--- |
| **Step 1** | Foundation & Reference Data | Faizan | **COMPLETED** | Repo layout, `src/common.h`, config files, `data/*.bin` generated for Test, Set A, Set B |
| **Step 2** | CPU Baseline & Direct Kernels | Khadija | **COMPLETED** | `src/v6_cpu.cpp` (C++ single-core), `src/v7_omp.cpp` (OpenMP), `src/v3_direct.cu` (Naive + Tiled) |
| **Step 3** | Test Harness & Benchmarks | Faizan | **COMPLETED** (rewritten 2026-11-07 to match Sections 4.9 to 4.11) | `src/main.cu`, `src/peak.cu`, `python/bench_cudnn.py` (V5 cuDNN, TF32 disabled), `build.bat` |
| **Step 4** | Shared GEMM & Matrix Convolutions | Mania / Faizan | **COMPLETED** | `src/gemm.cuh` (`gemm_simple` + `gemm_opt`), `src/v1_im2col.cu` (V1), `src/v2_implicit.cu` (V2), `src/v4_cublas.cu` (V4) |
| **Step 5** | V3 Tuning Sweep (COB Parameter) | Faizan | **COMPLETED** | Rerun 2026-11-07 on the fixed harness; $CO_B=16$ stays the default -> `results/tune_cob*.csv` |
| **Step 6** | CPU Benchmark Sweeps | Faizan | **COMPLETED** | `results/A_cpu.csv`, `results/B_cpu.csv` |
| **Step 7** | Final GPU Benchmark Sweeps | Faizan | **COMPLETED** | `results/A_gpu.csv`, `B_gpu.csv`, `A_v5.csv`, `B_v5.csv`, repeat pass in `results/repeat/` |
| **Step 8** | Analysis, Roofline & Plots | Mania & Khadija | **COMPLETED** | `python/roofline.py`, `plots.py`, `sanity_checks.py`, `repeatability.py`, `figures/fig01`-`fig07`, `tables/*` |
| **Step 9** | Final Report Assembly & Submission | All | **NEXT UP** | Word documents in `report/`, assembled final paper, clean zip bundle |

---

## 2. Hardware & Environment Specifications

Recorded in [`results/environment.txt`](results/environment.txt):
* **OS:** Microsoft Windows 11 Home (Build 26200)
* **CPU:** 13th Gen Intel(R) Core(TM) i5-13420H (8 Cores: 4 Performance + 4 Efficiency, 12 Threads)
* **System RAM:** 15.6 GB
* **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6141 MiB GDDR6, Driver: 592.82)
* **CUDA / Compiler Toolchain:** NVIDIA CUDA Toolkit 13.4 (`nvcc` 13.4.59), Visual Studio 2026 Developer Tools (MSVC 19.51; the plan said VS 2022), Windows 11 SDK (Build 22621)
* **Python Environment:** Python 3.11 (the plan said 3.12) with PyTorch 2.5.1+cu121, cuDNN 90100, NumPy 2.3.5
* **GPU power limit:** 42.15 to 50 W across sessions, read with `nvidia-smi -q -d POWER`

---

## 3. Peak Hardware Benchmarks ([results/peaks.csv](results/peaks.csv))

Measured with `convbench.exe --peaks`, highest of 20 timed runs after a 3 s spin-up (Section 4.11):

| Metric | Value | Notes |
| :--- | :--- | :--- |
| Peak bandwidth | **175.5 GB/s** | Theoretical 192.0 GB/s (96-bit GDDR6, 8001 MHz) |
| Peak compute | **10,799 GFLOP/s** | 160 blocks x 256 threads x 4096 iterations x 8 accumulators x 2 FLOPs |
| Ridge point | **61.5 FLOP/byte** | Peak compute / peak bandwidth |

The earlier figures (138.85 GB/s, 2428.95 GFLOP/s) were too low: they used a median, one accumulator and an idle GPU.

---

## 4. Correctness Verification Log

All 7 versions pass all 29 configs (T, Set A, Set B), and V3 passes with $CO_B$ = 4, 8 and 16: 377 / 377 rows OK. Largest error 3.13e-4 ([tables/tab_correctness.csv](tables/tab_correctness.csv)). Acceptance: $|out - ref| \le 10^{-3} + 10^{-3} |ref|$. Bench mode repeats this comparison after timing.

---

## 5. Architectural & Implementation Highlights

1. **Shared Weight Indexing in `gemm.cuh`:**
   - In standard CNN convolutions, the weight tensor $A$ ($C_{\text{out}} \times C_{\text{in}} K^2$) is shared across all images in the mini-batch $N$. Fixed offset indexing in `gemm_simple` and `gemm_opt` so batch offsets apply exclusively to virtual matrix $B$ and output tensor $C$.
2. **Optimized GEMM (`gemm_opt` in `gemm.cuh`):**
   - Output tile: $64 \times 64$ with $BK = 16$.
   - Thread block: 256 threads ($16 \times 16$), each calculating a $4 \times 4$ register-tiled sub-tile ($TM=4, TN=4$).
   - Transposed shared memory layout $A_s[16][64]$ eliminates bank conflicts.
3. **Zero-Workspace Implicit GEMM (`v2_implicit.cu`):**
   - Implemented `Im2colLoader` functor which computes spatial coordinates and input indices on-the-fly directly inside the inner GEMM loop, bypassing memory allocations completely.
4. **Strided Batched cuBLAS (`v4_cublas.cu`):**
   - Formulated as column-major $C^T = B^T A^T$ with batch stride $A$ as $K \times M$, stride $B$ as 0 (shared weights), and stride $C$ as $N_{\text{blas}} \times M$.
5. **Modernized CUDA 13 Device Queries (`peak.cu`):**
   - Replaced deprecated `cudaDeviceProp::clockRate` and `memoryClockRate` with standard `cudaDeviceGetAttribute` queries.

---

## 6. V3 Tuning Sweep Results (Section 6.3)

| $CO_B$ | A00 (ms) | B04 (ms) | Sum (ms) |
| :---: | :---: | :---: | :---: |
| 4 | 2.461 | 39.017 | 41.478 |
| 8 | 1.938 | 30.094 | 32.032 |
| **16** | **1.809** | **27.956** | **29.765** |

$CO_B = 16$ is 7.1% faster than $CO_B = 8$, which is outside the 2% rule, so it stays the default. The earlier table in this file was measured with a `cudaMemset` inside the timed section.

---

## 7. Result Summary

* **Fastest overall:** V5 on 20 of 23 layers, V4 on 3. Median speed relative to V5 on Set B: V4 0.48, V2 0.24, V1 0.22, V3 0.15.
* **Our methods (V1-V3):** V2 is fastest on 18 of 23 layers, V1 on 4, V3 on 1. V2 and V3 use the least memory of V1-V3 on every layer. Recommendations: [tables/tab_recommendation.csv](tables/tab_recommendation.csv).
* **Roofline:** V1 and V4 are memory-bound on 22 of 23 layers; V2, V3 and V5 are compute-bound on 18 of 23 ([tables/roofline.csv](tables/roofline.csv)).
* **CPU-GPU crossover:** every GPU method beats best OpenMP end to end on every Set A layer, so the crossover is below the smallest layer tested (0.116 GFLOP).
* **OpenMP scaling:** at 12 threads, 3.7x on A00 and 4.9x on B04.
* **Repeatability:** between two full passes V1-V3 differ by a median of under 1% (at most 9.3%). V4 and V5 vary more on small layers ([tables/tab_repeatability.csv](tables/tab_repeatability.csv)).

### Section 8 sanity checks ([tables/sanity_checks.csv](tables/sanity_checks.csv))

| # | Check | Result | Explanation |
| :---: | :--- | :---: | :--- |
| 1 | All rows OK | PASS | |
| 2 | Below measured peaks | 2 over | A06 V4 175.7 and A06 V5 180.8 GB/s, against 175.5. This 12 MB layer may be partly served from GPU cache, which the byte estimate does not model. Unconfirmed. |
| 3 | V1, V4 memory >= V2, V3 | PASS | |
| 4 | memgetinfo within 10% or 2 MB | 20 over | Every flagged row reads high by a few MB, which looks like allocation rounding across up to 4 buffers. |
| 5 | V1 not over 1.5x faster than V4 | PASS | |
| 6 | Ours not over 2x faster than V5 | 3 flagged | All on B11, where V5 took 0.077 ms here and 0.024 ms in the repeat. The outputs are correct; V5 timing is unstable on this tiny layer. |
| 7 | Over 10x faster than V6 on B02, B04 | PASS | |
| 8 | Pinned faster than pageable | PASS | |
| 9 | max within 1.5x of median | 14 flagged | All very short kernels, so this is jitter; the median is not affected. |

---

## 8. Next Actions

1. Done: the group reviewed and approved the results and the changes listed in [DECISIONS.md](DECISIONS.md).
2. Draft and assemble the report sections (Table 9). Methodology must state the deviations recorded on 2026-11-07.
3. Done: code package built (`convbench_code.zip`, Section 9.3). Rebuild it once `report/` is filled.

---

## 9. Daily Session & Check-in Logs

### Session Log: 19 October 2026 (Day 5 – Code Freeze & Verification)

* **Session Date:** Monday 19 October 2026  
* **Session Lead:** Faizan Tanveer  
* **Protocol Checklist (Section 6.1):**
  - [x] Charger plugged in continuously throughout session
  - [x] OMEN Gaming Hub set to Performance Mode; Windows Settings set to Best Performance
  - [x] Background applications closed; OneDrive sync and Windows Update paused
  - [x] Screen and Sleep timeouts set to *Never* while plugged in
  - [x] Laptop placed on a hard, flat surface with unimpeded cooling vents
  - [x] Reference datasets verified in `data/` for Set A and Set B

#### Hardware Telemetry at Session Start (from `results/session_log.txt`)
```text
19/10/2026 23:30:15
name, driver_version, power.limit [W], temperature.gpu, clocks.max.sm [MHz]
NVIDIA GeForce RTX 4050 Laptop GPU, 592.82, [N/A], 38, 3105 MHz
```

#### Daily 9:00 PM Check-in Meeting Summary (Day 5 Protocol)
* **Faizan Tanveer (23L-0640):**
  - **Finished:** Assembled portable CUDA 13.4 (`C:\cuda`) and Windows 11 SDK (`C:\winsdk`) environment. Implemented full test harness in `src/main.cu` (CLI parser, check mode, bench mode with pageable and pinned memory transfers, memory tracking via `cudaMemGetInfo`), peak hardware benchmarks in `src/peak.cu`, and `python/bench_cudnn.py` (V5 cuDNN benchmark with TF32 disabled per Section 4.10). Generated double-precision reference binary files in `data/` for Set A and Set B.
  - **Blocked:** None.
  - **Plan for Tomorrow (Day 6):** Run V3 COB tuning sweep on `configs/focus.csv`, followed by full CPU benchmarks (`A_cpu.csv`, `B_cpu.csv`) and final GPU benchmarks (`A_gpu.csv`, `B_gpu.csv`, `A_v5.csv`, `B_v5.csv`) on the RTX 4050 laptop.
* **Mania Ali (23L-0876):**
  - **Finished:** Implemented `src/gemm.cuh` including `gemm_simple` (16×16 shared-memory tile) and `gemm_opt` (64×64 tile, 16×16 thread block, 4×4 register tiling per thread with conflict-free transposed shared memory $A_s[16][64]$). Implemented `src/v1_im2col.cu` (grid-stride im2col kernel + GEMM dispatch), `src/v2_implicit.cu` (on-the-fly implicit GEMM via `Im2colLoader` with 0 workspace overhead), and `src/v4_cublas.cu` (im2col + `cublasSgemmStridedBatched` with row/column major transposition). All 4 versions verified 100% correct on test configs.
  - **Blocked:** Waiting for Faizan's final benchmark CSV outputs to compute roofline metrics.
  - **Plan for Tomorrow (Day 6):** Run `python/roofline.py` on Faizan's result CSVs to generate `figures/fig05_roofline.png` and `tables/roofline.csv`.
* **Khadija Fatima (23L-0976):**
  - **Finished:** Implemented single-core baseline `src/v6_cpu.cpp` and OpenMP multi-core `src/v7_omp.cpp`. Implemented direct convolution `src/v3_direct.cu` with both naive (`cob=0`) and tiled shared-memory kernels ($16 \times 16$ spatial tile with $CO_B \in \{4, 8, 16\}$ output channels per block). All kernels passed 100% correctness checks.
  - **Blocked:** Waiting for Faizan's final benchmark CSV outputs to plot comparative graphs.
  - **Plan for Tomorrow (Day 6):** Execute `python/plots.py` on Faizan's benchmark results to generate figures `fig01`–`fig04`, `fig06`, `fig07`, and tables `tab_correctness.csv`, `tab_setB.csv`, `tab_speedup.csv`, and `tab_crossover.csv`.

#### Milestone Evaluation (Day 5 Code Freeze)
* **Code Freeze Status:** **ACHIEVED & CONFIRMED**.
* All 7 convolution versions (V1–V7) are implemented, compiled via `build.bat`, and validated with **42 / 42 passed tests** on `configs/test.csv`. Zero breaking changes or pending code modifications remain before starting final benchmark runs.

---

### Session Log: 20 October 2026 (Step 5: V3 Tuning Sweep Completed)

* **Session Date:** Tuesday 20 October 2026 (02:27 PKT)  
* **Session Lead:** Faizan Tanveer  
* **Action:** Executed Step 5 parameter tuning sweep for V3 direct convolution across $CO_B \in \{4, 8, 16\}$ on `configs/focus.csv` (layers A00 and B04).
* **Hardware Telemetry:**
```text
20/10/2026 02:27:08
name, driver_version, power.limit [W], temperature.gpu, clocks.max.sm [MHz]
NVIDIA GeForce RTX 4050 Laptop GPU, 592.82, [N/A], 39, 3105 MHz
```
* **Results:**
  - $CO_B = 4$: Sum of medians = 52.89 ms (A00: 4.15 ms / 445.9 GFLOP/s; B04: 48.74 ms / 607.2 GFLOP/s)
  - $CO_B = 8$: Sum of medians = 38.84 ms (A00: 1.99 ms / 927.8 GFLOP/s; B04: 36.84 ms / 803.3 GFLOP/s)
  - $CO_B = 16$: Sum of medians = **33.03 ms** (A00: **1.85 ms / 999.1 GFLOP/s**; B04: **31.18 ms / 949.2 GFLOP/s**)
* **Decision:** $CO_B = 16$ selected (17.6% faster than $CO_B = 8$); set as permanent default `--v3cob` in `src/main.cu`; `convbench.exe` rebuilt.

---

### Session Log: 7 November 2026 (Harness fixes and final benchmarks)

* **Found:** the harness did not match Section 4.9 (no pinned timing, no memory columns, one thread count only, `cudaMemset` inside the timed section, no comparison in bench mode), `peak.cu` did not match Section 4.11, and the earlier Day 5 code freeze covered the test configs only.
* **Done:** harness, V5 script and peak benchmark fixed; correctness on all configs; V3 re-tuned; all CPU and GPU benchmarks; all figures and tables; sanity checks; recommendation table.
* **Deviations:** listed in [DECISIONS.md](DECISIONS.md). Approved by the group.
