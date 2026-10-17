#!/bin/bash
set -e
nvcc -O3 -std=c++17 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -Xcompiler -fopenmp -o convbench src/main.cu src/peak.cu src/v1_im2col.cu src/v2_implicit.cu src/v3_direct.cu src/v4_cublas.cu src/v6_cpu.cpp src/v7_omp.cpp -lcublas -lgomp
