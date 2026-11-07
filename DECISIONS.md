# Decisions

Every decision agreed in the group chat, with its date.

| Date | Decision |
| --- | --- |
| 2026-10-18 | Repository created with the layout in Section 4.3 of the execution plan. All source files start as stubs. |
| 2026-10-18 | Added .gitattributes so build.sh keeps Linux line endings and build.bat keeps Windows line endings. |
| 2026-10-19 | Step 1 & 3: Implemented test harness `src/main.cu`, peak benchmark `src/peak.cu`, and `python/bench_cudnn.py` (V5) with TF32 disabled. Generated reference datasets for Set A and Set B. |
| 2026-10-19 | Step 4: Implemented shared GEMM in `src/gemm.cuh` (`gemm_simple` and `gemm_opt` per Section 5.1), `src/v1_im2col.cu` (V1), `src/v2_implicit.cu` (V2), and `src/v4_cublas.cu` (V4 cuBLAS strided batched). |
| 2026-10-20 | Build & Test: Configured standalone CUDA 13.4 and Windows SDK toolchain in `build.bat`. Successfully compiled `convbench.exe`. Verified correctness of all versions (V1, V2, V3 naive & tiled, V4, V5, V6, V7) on `configs/test.csv` (42/42 PASS). Executed peak hardware benchmarks (`results/peaks.csv`). |
| 2026-10-20 | V3 Tuning: Ran sweep on `configs/focus.csv` for $CO_B \in \{4, 8, 16\}$. Sum of median kernel times: $CO_B=4$ (52.89 ms), $CO_B=8$ (38.84 ms), $CO_B=16$ (33.03 ms). $CO_B=16$ is 17.6% faster than $CO_B=8$. Selected $CO_B = 16$ as permanent default in `src/main.cu`. **Superseded on 2026-11-07: these timings included a `cudaMemset`.** |
| 2026-11-07 | Agreed by the group (shared files changed; Section 4.3). Harness rewritten to Section 4.9: Table 5 CSV columns, thread list, pinned end-to-end, `peak_mem_mb` and `memgetinfo_mb`, comparison with the reference in bench mode, `steady_clock` for end-to-end, correct median, `cudaMalloc` OOM handling, K and S limits. `cudaMemset` removed from the timed section. `bench_cudnn.py` and `peak.cu` rewritten to Sections 4.10 and 4.11. `cublasSetMathMode` added to `v4_init`. |
| 2026-11-07 | Deviation from Sections 4.9 and 4.11: GPU spin-up added. `gpu_spinup()` alternates compute and memory load, 3 s at the start of a run and of each peak measurement, 0.5 s before each timed row. The 5 warm-ups last milliseconds, so without it the first config ran at idle clocks and peaks measured 7,779 GFLOP/s and 152 GB/s, below what V5 achieves. |
| 2026-11-07 | Deviation from Section 6, agreed to finish sooner: Set B CPU runs used 1 warm-up, not 5. V7 with 1 thread was run on Set B for B04 only, because only fig07 uses it. GPU cooldowns were at least 60 s and until the GPU reached 48 C or 3 minutes passed, not a fixed 5 minutes. Set A CPU kept the full protocol. |
| 2026-11-07 | Set A CPU run was paused for about 14 minutes during row A08 V6, so that row's `kernel_ms_max` (884,113 ms) is not a real timing. Its median and min are unaffected. |
| 2026-11-07 | V3 re-tuned with the fixed harness: sums 41.48 ms ($CO_B=4$), 32.03 ms ($CO_B=8$), 29.77 ms ($CO_B=16$). $CO_B=16$ is 7.1% faster than $CO_B=8$, so it stays the default. |
| 2026-11-07 | All the changes and deviations recorded on this date were reviewed and approved by the group. |
| 2026-11-07 | Final GPU results are from one pass. A second full pass is stored in `results/repeat/` and used only for `tables/tab_repeatability.csv`. Nsight Compute (optional) was not run: `ncu` is not in this CUDA install. Power limit read through `nvidia-smi -q -d POWER`: 42.15 to 50 W across sessions. |


