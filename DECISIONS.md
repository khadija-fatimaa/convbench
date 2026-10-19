# Decisions

Every decision agreed in the group chat, with its date.

| Date | Decision |
| --- | --- |
| 2026-10-18 | Repository created with the layout in Section 4.3 of the execution plan. All source files start as stubs. |
| 2026-10-18 | Added .gitattributes so build.sh keeps Linux line endings and build.bat keeps Windows line endings. |
| 2026-10-19 | Step 1 & 3: Implemented test harness `src/main.cu`, peak benchmark `src/peak.cu`, and `python/bench_cudnn.py` (V5) with TF32 disabled. Generated reference datasets for Set A and Set B. |
| 2026-10-19 | Step 4: Implemented shared GEMM in `src/gemm.cuh` (`gemm_simple` and `gemm_opt` per Section 5.1), `src/v1_im2col.cu` (V1), `src/v2_implicit.cu` (V2), and `src/v4_cublas.cu` (V4 cuBLAS strided batched). |
| 2026-10-20 | Build & Test: Configured standalone CUDA 13.4 and Windows SDK toolchain in `build.bat`. Successfully compiled `convbench.exe`. Verified correctness of all versions (V1, V2, V3 naive & tiled, V4, V5, V6, V7) on `configs/test.csv` (42/42 PASS). Executed peak hardware benchmarks (`results/peaks.csv`). |

