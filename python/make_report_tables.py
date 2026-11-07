# python/make_report_tables.py. Writes the data tables of the report as LaTeX, directly from
# the CSV files, so no number in them is typed by hand. Output: report/generated/*.tex
#   py -3.11 python/make_report_tables.py
import math
import os

import numpy as np
import pandas as pd

from analysis_common import load_configs, load_cpu, load_gpu, load_peaks, only_ok

OUT = "report/generated"
os.makedirs(OUT, exist_ok=True)


def num(x, sf=3):
    """Format to `sf` significant figures, with thousands separators."""
    if x is None or pd.isna(x):
        return "--"
    if x == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(x))))
    decimals = max(sf - 1 - exp, 0)
    rounded = round(float(x), sf - 1 - exp)
    return f"{rounded:,.{decimals}f}".replace(",", "{,}")


def sci(x):
    """3 significant figures in the form 3.13 x 10^-4."""
    exp = int(math.floor(math.log10(abs(x))))
    return rf"${x / 10 ** exp:.2f} \times 10^{{{exp}}}$"


def write(name, columns, header, rows, over=None):
    """columns: LaTeX column spec. header: list of cells. rows: list of lists of cells.
    over: optional row of LaTeX cells above the header, for grouped columns."""
    lines = [r"\begin{tabular}{" + columns + "}", r"\hline", r"\rowcolor{headgrey}"]
    if over:
        lines += [over + r" \\", r"\rowcolor{headgrey}"]
    lines.append(" & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\ \hline")
    lines += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    lines += [r"\hline", r"\end{tabular}"]
    with open(os.path.join(OUT, name + ".tex"), "w", encoding="utf8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", name)


cfg = load_configs("configs")
gpu = only_ok(load_gpu("results"))
cpu = only_ok(load_cpu("results"))
peak_bw, peak_gf = load_peaks("results")
kernel = gpu.pivot(index="config_id", columns="version", values="kernel_ms_median")
memory = gpu.pivot(index="config_id", columns="version", values="peak_mem_mb")
page = gpu.pivot(index="config_id", columns="version", values="e2e_pageable_ms_median")
pin = gpu.pivot(index="config_id", columns="version", values="e2e_pinned_ms_median")
v6 = cpu[cpu.version == 6].set_index("config_id")["kernel_ms_median"]
best_omp = cpu[cpu.version == 7].groupby("config_id")["kernel_ms_median"].min()
SET_B = [f"B{i:02d}" for i in range(1, 13)]

# V3 tuning
rows = []
for c in (4, 8, 16):
    d = pd.read_csv(f"results/tune_cob{c}.csv").set_index("config_id")["kernel_ms_median"]
    rows.append([c, num(d["A00"]), num(d["B04"]), num(d.sum())])
write("tuning", "cccc", ["CO\\_B", "A00 (ms)", "B04 (ms)", "Sum (ms)"], rows)

# Correctness
t = pd.read_csv("tables/tab_correctness.csv")
write("correctness", "ccc", ["Version", "Largest max\\_abs\\_err", "Configurations passed"],
      [[f"V{r.version}", sci(r.max_abs_err), f"{r.configs_passed} / {r.configs_total}"]
       for r in t.itertuples()])

# Set A kernel time and memory
SET_A = [f"A{i:02d}" for i in range(0, 11)]
write("setA", "lrrrrrrrrrrr",
      ["Layer", "GFLOP", "V1", "V2", "V3", "V4", "V5", "V1", "V2", "V3", "V4", "V5"],
      [[c, num(cfg.at[c, "gflop"])] + [num(kernel.at[c, v]) for v in range(1, 6)]
       + [num(memory.at[c, v]) for v in range(1, 6)] for c in SET_A],
      over=r"& & \multicolumn{5}{c}{\textbf{Kernel time (ms)}} & \multicolumn{5}{c}{\textbf{Peak memory (MB)}}")

# Set B kernel time, including the CPU versions
write("setB_time", "lrrrrrrr", ["Layer", "V1", "V2", "V3", "V4", "V5", "V6", "Best V7"],
      [[c] + [num(kernel.at[c, v]) for v in range(1, 6)] + [num(v6[c]), num(best_omp[c])] for c in SET_B])

# Set B memory
write("setB_memory", "lrrrrr", ["Layer", "V1", "V2", "V3", "V4", "V5"],
      [[c] + [num(memory.at[c, v]) for v in range(1, 6)] for c in SET_B])

# Speedup
s = pd.read_csv("tables/tab_speedup.csv").set_index("config_id")
write("speedup", "lrrrrrrrrr",
      ["Layer", "V1", "V2", "V3", "V4", "V5", "V1", "V2", "V3", "V4"],
      [[c] + [num(s.at[c, f"V{v}_over_V6"]) for v in range(1, 6)]
       + [num(s.at[c, f"V{v}_over_V5"]) for v in range(1, 5)] for c in SET_B],
      over=r"& \multicolumn{5}{c}{\textbf{Speedup over V6}} & \multicolumn{4}{c}{\textbf{Speedup over V5}}")

# Roofline summary per version
r = pd.read_csv("tables/roofline.csv")
rows = []
for v in range(1, 6):
    d = r[r.version == v]
    rows.append([f"V{v}", (d.bound == "memory").sum(), (d.bound == "compute").sum(),
                 num(d.ai.median()), num(d.gflops.median()), num(d.gflops.median() / peak_gf * 100),
                 num(d.bandwidth_gbs.median()), num(d.bandwidth_gbs.median() / peak_bw * 100)])
write("roofline", "ccccrrrr",
      ["Version", "Memory-bound", "Compute-bound", "Median AI", "GFLOP/s", r"\% peak", "GB/s", r"\% peak"], rows)

# Crossover detail, Set A sorted by size
order = cfg[cfg.set == "A"].sort_values("gflop").index
fast = kernel.loc[order, [1, 2, 3]].idxmin(axis=1)
write("crossover", "lrrcrrr",
      ["Layer", "GFLOP", "Best V7", "Fastest", "Pageable", "Pinned", "V5 pageable"],
      [[c, num(cfg.at[c, "gflop"]), num(best_omp[c]), f"V{fast[c]}", num(page.at[c, fast[c]]),
        num(pin.at[c, fast[c]]), num(page.at[c, 5])] for c in order])

# End-to-end on large layers
rows = []
for c in ("B02", "B04"):
    for v in range(1, 6):
        rows.append([c, f"V{v}", num(kernel.at[c, v]), num(page.at[c, v]), num(pin.at[c, v])])
write("e2e", "cclll", ["Layer", "Version", "Kernel (ms)", "Pageable (ms)", "Pinned (ms)"], rows)

# Repeatability
t = pd.read_csv("tables/tab_repeatability.csv")
write("repeatability", "ccccc",
      ["Version", r"Median diff.\ (\%)", r"Max diff.\ (\%)", "Layer of max", r"Rows over 10\%"],
      [[f"V{x.version}", num(x.median_diff_pct), num(x.max_diff_pct), x.max_diff_config,
        f"{x.rows_over_10pct} / {x.rows}"] for x in t.itertuples()])

# Recommendation
t = pd.read_csv("tables/tab_recommendation.csv")
write("recommendation", "lcccrc",
      ["Layer", "Fastest", "Lowest memory", "Second", r"Lead (\%)", "Recommended"],
      [[x.config_id, x.fastest, x.lowest_memory, x.second_fastest, num(x.faster_by_pct),
        rf"\textbf{{{x.recommended}}}"] for x in t.itertuples()])

# OpenMP thread table
v7 = cpu[cpu.version == 7].pivot(index="config_id", columns="threads", values="kernel_ms_median")
write("openmp", "crrrr", ["Layer", "1 thread", "4 threads", "8 threads", "12 threads"],
      [[c] + [num(v7.at[c, n]) for n in (1, 4, 8, 12)] for c in ("A00", "B04")])

# Print the values the prose draws on, for checking.
print("speedup at 12 threads", {c: num(v7.at[c, 1] / v7.at[c, 12]) for c in ("A00", "B04")})
print("speedup at 4 threads", {c: num(v7.at[c, 1] / v7.at[c, 4]) for c in ("A00", "B04")})
print("best OpenMP over V6: min", num((v6 / best_omp).min()), "max", num((v6 / best_omp).max()))
print("peaks", num(peak_bw, 4), num(peak_gf, 5), "ridge", num(peak_gf / peak_bw))
g = gpu.assign(ratio=gpu.e2e_pageable_ms_median / gpu.e2e_pinned_ms_median)
print("pageable/pinned", {k: num(v) for k, v in g.groupby("version").ratio.median().items()})
print("A01,A03 omp", num(best_omp["A01"]), num(best_omp["A03"]))
print("lowest GPU win margin pageable", num((best_omp[order] / pd.Series({c: page.at[c, fast[c]] for c in order})).min()))
