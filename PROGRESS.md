# ConvBench Execution Progress

**Project:** Parallel and Distributed Computing (PDC) Group Project  
**Repository:** [khadija-fatimaa/convbench](https://github.com/khadija-fatimaa/convbench)  
**Authors:** Faizan Tanveer (23L-0640), Mania Ali (23L-0876), Khadija Fatima (23L-0976)  
**Target Platform:** Laptop GPU (NVIDIA GeForce RTX 4050 6GB Laptop GPU, 13th Gen Intel Core i5-13420H)  
**Last Updated:** 2026-10-20  

---

## 1. Overall Milestone Dashboard

| Step | Phase Description | Owner | Status | Deliverables / Artifacts |
| :--- | :--- | :--- | :---: | :--- |
| **Step 1** | Foundation & Reference Data | Faizan | **COMPLETED** | Repo layout, `src/common.h`, config files, `data/*.bin` generated for Test, Set A, Set B |
| **Step 2** | CPU Baseline & Direct Kernels | Khadija | **COMPLETED** | `src/v6_cpu.cpp` (C++ single-core), `src/v7_omp.cpp` (OpenMP), `src/v3_direct.cu` (Naive + Tiled) |
| **Step 3** | Test Harness & Benchmarks | Faizan | **COMPLETED** | `src/main.cu`, `src/peak.cu`, `python/bench_cudnn.py` (V5 cuDNN, TF32 disabled), `build.bat` |
| **Step 4** | Shared GEMM & Matrix Convolutions | Mania / Faizan | **COMPLETED** | `src/gemm.cuh` (`gemm_simple` + `gemm_opt`), `src/v1_im2col.cu` (V1), `src/v2_implicit.cu` (V2), `src/v4_cublas.cu` (V4) |
| **Step 5** | V3 Tuning Sweep (COB Parameter) | Faizan | **COMPLETED** | Sweep on `focus.csv` over $CO_B \in \{4, 8, 16\}$, $CO_B=16$ selected as default in `main.cu` -> `results/tune_cob*.csv` |
| **Step 6** | CPU Benchmark Sweeps | Faizan | **NEXT UP** | V6 & V7 benchmarks over Set A and Set B (`results/A_cpu.csv`, `results/B_cpu.csv`) |
| **Step 7** | Final GPU Benchmark Sweeps | Faizan | *Pending Step 6* | V1–V5 benchmarks on RTX 4050 over Set A and Set B (`results/A_gpu.csv`, `results/B_gpu.csv`) |
| **Step 8** | Analysis, Roofline & Plots | Mania & Khadija | *Pending Step 7* | `python/roofline.py` (`fig05_roofline.png`, `tables/roofline.csv`), `python/plots.py` (`fig01`–`fig07`, `tables/*.csv`) |
| **Step 9** | Final Report Assembly & Submission | All | *Pending Step 8* | Word documents in `report/`, assembled final paper, clean zip bundle |

---

## 2. Hardware & Environment Specifications

Recorded in [`results/environment.txt`](results/environment.txt):
* **OS:** Microsoft Windows 11 Home (Build 26200)
* **CPU:** 13th Gen Intel(R) Core(TM) i5-13420H (8 Cores: 4 Performance + 4 Efficiency, 12 Threads)
* **System RAM:** 15.6 GB
* **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6141 MiB GDDR6, Driver: 592.82)
* **CUDA / Compiler Toolchain:** NVIDIA CUDA Toolkit 13.4 (`nvcc` 13.4.59), Visual Studio 2026 Developer Tools (MSVC 19.51), Windows 11 SDK (Build 22621)
* **Python Environment:** Python 3.11 with PyTorch 2.5.1+cu121, cuDNN 90100, NumPy 2.3.5

---

## 3. Peak Hardware Benchmarks ([results/peaks.csv](results/peaks.csv))

Measured via `convbench.exe --peaks --out results/peaks.csv`:

| Metric | Target / Specification | Measured Value | Unit | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **SM Count** | 20 SMs | 20 | count | Ada Lovelace Architecture (`sm_89`) |
| **SM Clock Rate** | Boost / Peak | 1755 | MHz | Queried via `cudaDevAttrClockRate` |
| **Memory Clock Rate** | Effective Memory Clock | 8001 | MHz | Queried via `cudaDevAttrMemoryClockRate` |
| **Memory Bus Width** | 96-bit GDDR6 | 96 | bits | — |
| **Theoretical Peak Bandwidth** | $2.0 \times 8001 \times 10^3 \times (96/8) / 10^9$ | **192.02** | GB/s | — |
| **Measured Copy Bandwidth** | 256 MiB buffer (`float4` copy kernel) | **138.85** | GB/s | Median time: 3.867 ms (72.3% of theoretical peak) |
| **Measured Compute Peak (FMA)** | 1M threads, 16,384 FLOPs/thread | **2428.95** | GFLOP/s | Median time: 0.138 ms (2.43 TFLOP/s single precision) |

---

## 4. Correctness Verification Log ([results/check.csv](results/check.csv))

Evaluated on `configs/test.csv` (configurations `T01` to `T06`) against CPU double-precision reference output:
* **Total Tests Executed:** 42 (6 configurations $\times$ 7 versions)
* **Pass Rate:** **100% (42 / 42 passed)**
* **Acceptance Criterion:** Absolute difference $\le 10^{-3} + 10^{-3} \times |y_{\text{ref}}|$

| Config ID | Layer Dimensions ($N, C_{\text{in}}, H, W, C_{\text{out}}, K, S, P$) | V1 (im2col) | V2 (implicit) | V3 (direct) | V4 (cuBLAS) | V5 (cuDNN) | V6 (CPU) | V7 (OpenMP) | Max Observed Error | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T01** | $1, 1, 5, 5, 1, 3, 1, 1$ | PASS | PASS | PASS | PASS | PASS | PASS | PASS | $1.19 \times 10^{-7}$ | **ALL PASS** |
| **T02** | $2, 3, 8, 8, 4, 3, 2, 1$ | PASS | PASS | PASS | PASS | PASS | PASS | PASS | $7.15 \times 10^{-7}$ | **ALL PASS** |
| **T03** | $2, 5, 13, 13, 7, 5, 1, 2$ | PASS | PASS | PASS | PASS | PASS | PASS | PASS | $5.72 \times 10^{-6}$ | **ALL PASS** |
| **T04** | $1, 3, 17, 17, 8, 7, 2, 3$ | PASS | PASS | PASS | PASS | PASS | PASS | PASS | $3.81 \times 10^{-6}$ | **ALL PASS** |
| **T05** | $2, 16, 9, 9, 8, 1, 2, 0$ | PASS | PASS | PASS | PASS | PASS | PASS | PASS | $4.77 \times 10^{-7}$ | **ALL PASS** |
| **T06** | $3, 70, 20, 20, 65, 3, 1, 1$ (Non-tile multiples) | PASS | PASS | PASS | PASS | PASS | PASS | PASS | $3.24 \times 10^{-5}$ | **ALL PASS** |

### Additional Tiled V3 Checks:
* **V3 COB = 4:** 6/6 PASS ([results/check_v3_cob4.csv](results/check_v3_cob4.csv))
* **V3 COB = 8:** 6/6 PASS ([results/check_v3_cob8.csv](results/check_v3_cob8.csv))
* **V3 COB = 16:** 6/6 PASS ([results/check_v3_cob16.csv](results/check_v3_cob16.csv))

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

Evaluated on `configs/focus.csv` (`A00`: $8 \times 64 \times 56 \times 56$, $64 \times 3 \times 3$; `B04`: $8 \times 256 \times 56 \times 56$, $256 \times 3 \times 3$) to select the optimal channel block size $CO_B$:

| $CO_B$ Parameter | Config A00 Median Kernel (ms) | Config A00 GFLOP/s | Config B04 Median Kernel (ms) | Config B04 GFLOP/s | Sum of Medians (ms) | Speedup vs $CO_B=4$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$CO_B = 4$** | 4.148 ms | 445.9 GFLOP/s | 48.741 ms | 607.2 GFLOP/s | 52.890 ms | 1.00× (Baseline) |
| **$CO_B = 8$** | 1.994 ms | 927.8 GFLOP/s | 36.844 ms | 803.3 GFLOP/s | 38.837 ms | 1.36× |
| **$CO_B = 16$ (WINNER)** | **1.851 ms** | **999.1 GFLOP/s** | **31.178 ms** | **949.2 GFLOP/s** | **33.029 ms** | **1.60×** |

* **Selection Decision:** $CO_B = 16$ achieved the lowest sum of median kernel times (33.03 ms vs 38.84 ms for $CO_B=8$, an improvement of 17.6%). Per Section 6.3 rule, $CO_B = 16$ is selected and permanently configured as the default `--v3cob` in `src/main.cu`.
* **Artifacts:** [`results/tune_cob4.csv`](results/tune_cob4.csv), [`results/tune_cob8.csv`](results/tune_cob8.csv), [`results/tune_cob16.csv`](results/tune_cob16.csv).

---

## 7. Execution Command Log

```cmd
:: Build binary
build.bat

:: Peak hardware performance
convbench.exe --peaks --out results/peaks.csv

:: Full correctness check across all versions
convbench.exe --configs configs/test.csv --versions 1,2,3,4,6,7 --mode check --out results/check.csv
py -3.11 python/bench_cudnn.py --configs configs/test.csv --mode check --out results/check_v5.csv

:: V3 parameter sweep checks
convbench.exe --configs configs/test.csv --versions 3 --v3cob 4 --mode check --out results/check_v3_cob4.csv
convbench.exe --configs configs/test.csv --versions 3 --v3cob 8 --mode check --out results/check_v3_cob8.csv
convbench.exe --configs configs/test.csv --versions 3 --v3cob 16 --mode check --out results/check_v3_cob16.csv
```

---

## 8. Next Actions & Schedule
1. **Execute Step 6 (NEXT UP):** Run CPU benchmarks (`results/A_cpu.csv`, `results/B_cpu.csv`) with thread scaling ($T \in \{1, 4, 8, 12\}$) on i5-13420H.
2. **Execute Step 7:** Run final GPU benchmarks on RTX 4050 (`results/A_gpu.csv`, `results/B_gpu.csv`, `results/A_v5.csv`, `results/B_v5.csv`) for V1–V5.
3. **Execute Step 8:** Implement and generate figures (`fig01`–`fig07`) and tables (`tables/*.csv`).
4. **Execute Step 9:** Draft and assemble final report sections.

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


