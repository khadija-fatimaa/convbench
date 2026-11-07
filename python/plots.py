# python/plots.py (owner: Khadija). Section 7.4 and 7.5.
# Output: figures/fig01..04, fig06, fig07 and the tab_*.csv tables. fig05 comes from roofline.py.
#   py -3.11 python/plots.py [--results results] [--configs configs] [--figures figures] [--tables tables]
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_common import (FIG_WIDTH, GPU_VERSIONS, INK_MUTED, VERSION_COLORS, VERSION_MARKERS,
                             VERSION_NAMES, apply_style, load_configs, load_cpu, load_gpu,
                             only_ok, save_fig, save_table, sig3)

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--configs", default="configs")
ap.add_argument("--figures", default="figures")
ap.add_argument("--tables", default="tables")
args = ap.parse_args()

apply_style()
cfg = load_configs(args.configs)
gpu = only_ok(load_gpu(args.results))
cpu = only_ok(load_cpu(args.results))

kernel = gpu.pivot(index="config_id", columns="version", values="kernel_ms_median")
memory = gpu.pivot(index="config_id", columns="version", values="peak_mem_mb")
e2e_page = gpu.pivot(index="config_id", columns="version", values="e2e_pageable_ms_median")
e2e_pin = gpu.pivot(index="config_id", columns="version", values="e2e_pinned_ms_median")
v6 = cpu[cpu.version == 6].set_index("config_id")["kernel_ms_median"]
best_omp = cpu[cpu.version == 7].groupby("config_id")["kernel_ms_median"].min()

SET_B = [f"B{i:02d}" for i in range(1, 13)]
BATCH8 = ["B02", "B04", "B06", "B08", "B10", "B12"]
SWEEPS = [
    ("(a) Input size", "Input height and width", ["A01", "A02", "A00"], [14, 28, 56]),
    ("(b) Channels", "Input and output channels", ["A03", "A00", "A04", "A05"], [16, 64, 128, 256]),
    ("(c) Filter size", "Filter size K", ["A06", "A00", "A07", "A08"], [1, 3, 5, 7]),
    ("(d) Batch size", "Batch size", ["A09", "A00", "A10"], [1, 8, 32]),
]


# ---------------------------------------------------------------------------
# Figures 1 and 2: Set A sweeps
# ---------------------------------------------------------------------------
def sweep_figure(values, ylabel, name, ylog, sizes=None):
    # sizes: marker size per version, so series with identical values stay visible.
    sizes = sizes or {}
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, 5.4))
    for ax, (title, xlabel, ids, xs) in zip(axes.flat, SWEEPS):
        for v in GPU_VERSIONS:
            ys = [values.at[c, v] if c in values.index else np.nan for c in ids]
            ax.plot(xs, ys, color=VERSION_COLORS[v], marker=VERSION_MARKERS[v], ms=sizes.get(v, 5),
                    lw=1.5, markeredgecolor="white", markeredgewidth=0.5, label=VERSION_NAMES[v])
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.minorticks_off()
        if ylog:
            ax.set_yscale("log")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save_fig(fig, args.figures, name)


sweep_figure(kernel, "Kernel time (ms)", "fig01_setA_time.png", True)
# V1 and V4 have equal memory, as do V2 and V3: the earlier version is drawn larger behind.
sweep_figure(memory, "Peak memory (MB)", "fig02_setA_memory.png", True, sizes={1: 9, 2: 9})


# ---------------------------------------------------------------------------
# Figures 3 and 4: Set B, batch 8, grouped bars
# ---------------------------------------------------------------------------
def setb_figure(values, ylabel, name):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.6))
    group = 0.82
    width = group / len(GPU_VERSIONS)
    x = np.arange(len(BATCH8))
    for i, v in enumerate(GPU_VERSIONS):
        ys = [values.at[c, v] if c in values.index else np.nan for c in BATCH8]
        ax.bar(x + (i - 2) * width, ys, width * 0.88, color=VERSION_COLORS[v],
               label=VERSION_NAMES[v], zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(BATCH8)
    ax.set_yscale("log")
    ax.set_xlabel("Layer (batch 8)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    save_fig(fig, args.figures, name)


setb_figure(kernel, "Kernel time (ms)", "fig03_setB_time.png")
setb_figure(memory, "Peak memory (MB)", "fig04_setB_memory.png")

# ---------------------------------------------------------------------------
# Figure 6 and tab_crossover: CPU-GPU crossover on Set A
# ---------------------------------------------------------------------------
set_a = cfg[cfg.set == "A"].sort_values(["gflop"]).index.tolist()
fastest_123 = kernel.loc[set_a, [1, 2, 3]].idxmin(axis=1)
fast_page = pd.Series({c: e2e_page.at[c, fastest_123[c]] for c in set_a})
fast_pin = pd.Series({c: e2e_pin.at[c, fastest_123[c]] for c in set_a})

series = [
    ("Best OpenMP (V7)", best_omp.reindex(set_a), 1, "o"),
    ("Fastest of V1-V3, end-to-end pageable", fast_page, 2, "s"),
    ("Fastest of V1-V3, end-to-end pinned", fast_pin, 3, "^"),
    ("V5 cuDNN, end-to-end pageable", e2e_page[5].reindex(set_a), 4, "D"),
]
fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4.2))
xs = cfg.loc[set_a, "gflop"].values
for label, ys, slot, marker in series:
    ax.plot(xs, ys.values, color=VERSION_COLORS[slot], marker=marker, ms=5, lw=1.5,
            markeredgecolor="white", markeredgewidth=0.5, label=label)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Layer size (GFLOP)")
ax.set_ylabel("Time (ms)")
ax.legend(loc="upper left")
fig.tight_layout()
save_fig(fig, args.figures, "fig06_crossover.png")


def crossover(times):
    """Smallest layer on which `times` beats best OpenMP, and whether every larger layer does too."""
    wins = [c for c in set_a if times.get(c, np.inf) < best_omp[c]]
    if not wins:
        return np.nan, "", False
    first = min(wins, key=lambda c: cfg.at[c, "gflop"])
    size = cfg.at[first, "gflop"]
    larger = [c for c in set_a if cfg.at[c, "gflop"] > size]
    return size, first, all(c in wins for c in larger)


rows = []
targets = [("Fastest of V1-V3", fast_page, fast_pin)]
targets += [(VERSION_NAMES[v], e2e_page[v], e2e_pin[v]) for v in GPU_VERSIONS]
for name, page, pin in targets:
    s1, c1, all1 = crossover(page)
    s2, c2, all2 = crossover(pin)
    rows.append({"method": name,
                 "crossover_pageable_gflop": sig3(s1), "config_pageable": c1,
                 "wins_all_larger_pageable": all1,
                 "crossover_pinned_gflop": sig3(s2), "config_pinned": c2,
                 "wins_all_larger_pinned": all2})
save_table(pd.DataFrame(rows), args.tables, "tab_crossover.csv")

# ---------------------------------------------------------------------------
# Figure 7: OpenMP scaling
# ---------------------------------------------------------------------------
v7 = cpu[cpu.version == 7]
fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.6))
threads = [1, 4, 8, 12]
ax.plot(threads, threads, color=INK_MUTED, lw=1, ls="--", label="Ideal (y = x)")
for cid, slot, marker in (("A00", 1, "o"), ("B04", 2, "s")):
    d = v7[v7.config_id == cid].set_index("threads")["kernel_ms_median"]
    ys = [d[1] / d[t] if t in d.index else np.nan for t in threads]
    ax.plot(threads, ys, color=VERSION_COLORS[slot], marker=marker, ms=5, lw=1.5,
            markeredgecolor="white", markeredgewidth=0.5, label=cid)
ax.set_xticks(threads)
ax.set_xlabel("OpenMP threads")
ax.set_ylabel("Speedup over V7 with 1 thread")
ax.legend(loc="upper left")
fig.tight_layout()
save_fig(fig, args.figures, "fig07_openmp_scaling.png")

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
# tab_correctness: from every check file, per config so V7 thread rows count once.
checks = []
for pattern in ("check.csv", "check_A.csv", "check_B.csv", "check_v5.csv", "check_v5_A.csv", "check_v5_B.csv"):
    for path in glob.glob(os.path.join(args.results, pattern)):
        checks.append(pd.read_csv(path))
if checks:
    chk = pd.concat(checks, ignore_index=True)
    per_cfg = chk.groupby(["version", "config_id"]).agg(ok=("status", lambda s: (s == "OK").all()),
                                                        err=("max_abs_err", "max")).reset_index()
    tab = per_cfg.groupby("version").agg(max_abs_err=("err", "max"), configs_passed=("ok", "sum"),
                                         configs_total=("ok", "size")).reset_index()
    tab["max_abs_err"] = tab["max_abs_err"].map(sig3)
    save_table(tab, args.tables, "tab_correctness.csv")

# tab_setB: kernel time and peak memory.
rows = []
for c in SET_B:
    row = {"config_id": c}
    for v in GPU_VERSIONS:
        row[f"V{v}_kernel_ms"] = sig3(kernel.at[c, v])
        row[f"V{v}_peak_mem_mb"] = sig3(memory.at[c, v])
    row["V6_ms"] = sig3(v6.get(c, np.nan))
    row["V7_best_ms"] = sig3(best_omp.get(c, np.nan))
    rows.append(row)
save_table(pd.DataFrame(rows), args.tables, "tab_setB.csv")

# tab_speedup: over V6 (V1-V5) and over V5 (V1-V4).
rows = []
for c in SET_B:
    row = {"config_id": c}
    for v in GPU_VERSIONS:
        row[f"V{v}_over_V6"] = sig3(v6.get(c, np.nan) / kernel.at[c, v])
    for v in (1, 2, 3, 4):
        row[f"V{v}_over_V5"] = sig3(kernel.at[c, 5] / kernel.at[c, v])
    rows.append(row)
save_table(pd.DataFrame(rows), args.tables, "tab_speedup.csv")

# tab_recommendation (Section 7.5).
rows = []
for c in SET_B:
    t = kernel.loc[c, [1, 2, 3]].sort_values()
    first, second = int(t.index[0]), int(t.index[1])
    m = memory.loc[c, [1, 2, 3]]
    lowest = "/".join(f"V{v}" for v in m.index[m == m.min()])
    faster_by = 1 - t.iloc[0] / t.iloc[1]
    switch = m[first] > 2 * m[second] and faster_by < 0.10
    rec = second if switch else first
    p = cfg.loc[c]
    rows.append({
        "config_id": c,
        "layer": f"N={p.N} Cin={p.Cin} H={p.H} Cout={p.Cout} K={p.K} S={p.S}",
        "fastest": f"V{first}",
        "fastest_ms": sig3(t.iloc[0]),
        "second_fastest": f"V{second}",
        "second_ms": sig3(t.iloc[1]),
        "lowest_memory": lowest,
        "lowest_memory_mb": sig3(m.min()),
        "faster_by_pct": sig3(faster_by * 100),
        "recommended": f"V{rec}",
    })
save_table(pd.DataFrame(rows), args.tables, "tab_recommendation.csv")
