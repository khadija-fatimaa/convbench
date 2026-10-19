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
| **Step 5** | V3 Tuning Sweep (COB Parameter) | Faizan | **IN PROGRESS** | Run `focus.csv` over COB = 4, 8, 16 -> set default in `src/main.cu` -> `results/tune_cob*.csv` |
| **Step 6** | CPU Benchmark Sweeps | Faizan | *Pending Step 5* | V6 & V7 benchmarks over Set A and Set B (`results/A_cpu.csv`, `results/B_cpu.csv`) |
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

## 6. Execution Command Log

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

## 7. Next Actions & Schedule
1. **Execute Step 5:** Run `configs/focus.csv` on V3 with COB $\in \{4, 8, 16\}$ to select optimal tiling parameter.
2. **Execute Step 6:** Run CPU benchmarks (`results/A_cpu.csv`, `results/B_cpu.csv`) with thread scaling on i5-13420H.
3. **Execute Step 7:** Run final GPU benchmarks on RTX 4050 (`results/A_gpu.csv`, `results/B_gpu.csv`) for V1–V5.
4. **Execute Step 8:** Implement and generate figures (`fig01`–`fig07`) and tables (`tables/*.csv`).
