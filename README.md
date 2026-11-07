# convbench

Parallel and Distributed Computing group project: comparing ways to speed up the convolution layer of a CNN on the GPU.

Members: Faizan Tanveer (23L-0640), Mania Ali (23L-0876), Khadija Fatima (23L-0976).

## Build

Windows laptop (paths to CUDA, MSVC and the Windows SDK are set inside build.bat; edit them for another machine):

    .\build.bat

The program needs `C:\cuda\bin` on PATH to find the cuBLAS DLL.

Google Colab (T4 GPU runtime):

    !bash build.sh

## Generate reference data

    python python/make_reference.py --configs configs/test.csv

Data is written to data/ and is never committed.

## Run a check

    convbench --configs configs/test.csv --versions 3 --mode check --out scratch/check.csv

## Run the benchmarks

Follow Section 6 of the execution plan. Set A and Set B data must exist first.

    convbench --peaks --out results/peaks.csv
    convbench --configs configs/setA.csv --versions 6,7 --threads 1,4,8,12 --mode bench --out results/A_cpu.csv
    convbench --configs configs/setB.csv --versions 6,7 --threads 1,4,8,12 --mode bench --out results/B_cpu.csv
    convbench --configs configs/setA.csv --versions 1,2,3,4 --mode bench --out results/A_gpu.csv
    convbench --configs configs/setB.csv --versions 1,2,3,4 --mode bench --out results/B_gpu.csv
    py -3.11 python/bench_cudnn.py --configs configs/setA.csv --mode bench --out results/A_v5.csv
    py -3.11 python/bench_cudnn.py --configs configs/setB.csv --mode bench --out results/B_v5.csv

## Produce the figures and tables

    py -3.11 python/roofline.py
    py -3.11 python/plots.py
    py -3.11 python/sanity_checks.py
    py -3.11 python/repeatability.py

## Build the report

    py -3.11 python/make_report_tables.py
    py -3.11 python/check_report_numbers.py
    cd report
    xelatex report.tex
    xelatex report.tex

## Rules

Edit only the files you own. Shared files (src/common.h, src/main.cu, configs/, the CSV format) change only after all three members agree in the group chat. Run git pull before starting work and before every push. Never push code that does not build. Only Faizan commits to results/.

Full instructions are in the 8-Day Execution Plan.
