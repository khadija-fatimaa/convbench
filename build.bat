@echo off
set "PATH=C:\cuda\bin;C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64;%PATH%"
set "INCLUDE=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\include;C:\winsdk\include\ucrt;C:\winsdk\include\um;C:\winsdk\include\shared;C:\cuda\include"
set "LIB=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\lib\x64;C:\winsdk\lib\x64;C:\cuda\lib\x64"

nvcc -O3 -std=c++17 -gencode arch=compute_89,code=sm_89 -Xcompiler /openmp -o convbench.exe src\main.cu src\peak.cu src\v1_im2col.cu src\v2_implicit.cu src\v3_direct.cu src\v4_cublas.cu src\v6_cpu.cpp src\v7_omp.cpp -lcublas
