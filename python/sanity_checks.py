# python/sanity_checks.py. Section 8: the nine checks on the final results.
# Run after roofline.py. Writes tables/sanity_checks.csv and prints one line per check.
#   py -3.11 python/sanity_checks.py [--results results] [--configs configs] [--tables tables]
import argparse

import pandas as pd

from analysis_common import load_configs, load_cpu, load_gpu, load_peaks, save_table

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--configs", default="configs")
ap.add_argument("--tables", default="tables")
args = ap.parse_args()

gpu = load_gpu(args.results)
cpu = load_cpu(args.results)
peak_bw, peak_gf = load_peaks(args.results)
roof = pd.read_csv(f"{args.tables}/roofline.csv")
cfg = load_configs(args.configs)
rows = []


def report(num, name, kind, hits):
    """kind: 'must' checks fail on any hit; 'usually' checks only need an explanation."""
    result = "PASS" if not hits else ("FAIL" if kind == "must" else "REVIEW")
    print(f"{num}. {name}: {result} ({len(hits)})")
    for h in hits[:40]:
        print("     ", h)
    rows.append({"check": num, "name": name, "result": result, "count": len(hits), "details": "; ".join(hits)})


both = pd.concat([gpu, cpu], ignore_index=True)
bad = both[both.status != "OK"]
report(1, "Every row has status OK", "must",
       [f"{r.config_id} V{r.version} t={r.threads}: {r.status}" for r in bad.itertuples()])

hits = [f"{r.config_id} V{r.version}: {r.gflops:.0f} GFLOP/s" for r in roof.itertuples() if r.gflops > peak_gf]
hits += [f"{r.config_id} V{r.version}: {r.bandwidth_gbs:.1f} GB/s" for r in roof.itertuples() if r.bandwidth_gbs > peak_bw]
report(2, "No version exceeds the measured peaks", "must", hits)

mem = gpu.pivot(index="config_id", columns="version", values="peak_mem_mb")
hits = [c for c in mem.index if max(mem.at[c, 2], mem.at[c, 3]) > min(mem.at[c, 1], mem.at[c, 4])]
report(3, "V1 and V4 use at least as much memory as V2 and V3", "must", hits)

hits = []
for r in gpu[gpu.version <= 4].itertuples():
    diff = abs(r.memgetinfo_mb - r.peak_mem_mb)
    if diff > 2 and diff > 0.10 * r.peak_mem_mb:
        hits.append(f"{r.config_id} V{r.version}: getinfo {r.memgetinfo_mb:.1f} vs {r.peak_mem_mb:.1f} MB")
report(4, "memgetinfo_mb within 10% or 2 MB of peak_mem_mb", "must", hits)

k = gpu.pivot(index="config_id", columns="version", values="kernel_ms_median")
hits = [f"{c}: V1 {k.at[c, 1]:.3g} ms, V4 {k.at[c, 4]:.3g} ms" for c in k.index if k.at[c, 4] > 1.5 * k.at[c, 1]]
report(5, "V1 is not more than 1.5x faster than V4", "usually", hits)

hits = [f"{c} V{v}: {k.at[c, 5] / k.at[c, v]:.2f}x" for c in k.index for v in (1, 2, 3, 4)
        if k.at[c, 5] > 2 * k.at[c, v]]
report(6, "None of our versions is more than 2x faster than V5", "usually", hits)

v6 = cpu[cpu.version == 6].set_index("config_id")["kernel_ms_median"]
hits = [f"{c} V{v}: {v6[c] / k.at[c, v]:.1f}x" for c in ("B02", "B04") for v in (1, 2, 3, 4, 5)
        if v6[c] / k.at[c, v] <= 10]
report(7, "Every GPU version is over 10x faster than V6 on B02 and B04", "usually", hits)

hits = [f"{r.config_id} V{r.version}: pinned {r.e2e_pinned_ms_median:.3g}, pageable {r.e2e_pageable_ms_median:.3g} ms"
        for r in gpu.itertuples() if r.e2e_pinned_ms_median >= r.e2e_pageable_ms_median]
report(8, "Pinned end-to-end is lower than pageable", "usually", hits)

hits = [f"{r.config_id} V{r.version}: max {r.kernel_ms_max:.3g}, median {r.kernel_ms_median:.3g} ms"
        for r in gpu.itertuples() if r.kernel_ms_max > 1.5 * r.kernel_ms_median]
report(9, "kernel_ms_max within 1.5x of the median", "usually", hits)

save_table(pd.DataFrame(rows), args.tables, "sanity_checks.csv")
