# convbench

Parallel and Distributed Computing group project: comparing ways to speed up the convolution layer of a CNN on the GPU.

Members: Faizan Tanveer (23L-0640), Mania Ali (23L-0876), Khadija Fatima (23L-0976).

## Build

Windows laptop (run in "x64 Native Tools Command Prompt for VS 2022" from the repository folder):

    build.bat

Google Colab (T4 GPU runtime):

    !bash build.sh

## Generate reference data

    python python/make_reference.py --configs configs/test.csv

Data is written to data/ and is never committed.

## Run a check

    convbench --configs configs/test.csv --versions 3 --mode check --out scratch/check.csv

## Rules

Edit only the files you own. Shared files (src/common.h, src/main.cu, configs/, the CSV format) change only after all three members agree in the group chat. Run git pull before starting work and before every push. Never push code that does not build. Only Faizan commits to results/.

Full instructions are in the 8-Day Execution Plan.
