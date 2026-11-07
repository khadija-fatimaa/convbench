# python/roofline.py (owner: Mania). Section 7.3.
# Output: tables/roofline.csv and figures/fig05_roofline.png
#   py -3.11 python/roofline.py [--results results] [--configs configs] [--figures figures] [--tables tables]
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_common import (FIG_WIDTH, GPU_VERSIONS, INK_MUTED, VERSION_COLORS, VERSION_MARKERS,
                             VERSION_NAMES, apply_style, load_configs, load_gpu, load_peaks,
                             only_ok, save_fig, save_table)

FIG_CONFIGS = ["A00", "B02", "B04", "B06", "B08", "B10", "B12"]

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--configs", default="configs")
ap.add_argument("--figures", default="figures")
ap.add_argument("--tables", default="tables")
args = ap.parse_args()

apply_style()
cfg = load_configs(args.configs)
gpu = only_ok(load_gpu(args.results))
peak_bw, peak_gf = load_peaks(args.results)
ridge = peak_gf / peak_bw

rows = []
for r in gpu.itertuples():
    c = cfg.loc[r.config_id]
    flops = int(c.flops)
    nbytes = int(c.bytes_im2col if r.version in (1, 4) else c.bytes_base)
    ai = flops / nbytes
    rows.append({
        "config_id": r.config_id,
        "version": r.version,
        "flops": flops,
        "bytes": nbytes,
        "ai": ai,
        "gflops": flops / (r.kernel_ms_median * 1e6),
        "bandwidth_gbs": nbytes / (r.kernel_ms_median * 1e6),
        "bound": "memory" if ai < ridge else "compute",
    })
table = pd.DataFrame(rows).sort_values(["config_id", "version"])
save_table(table, args.tables, "roofline.csv")

# ---------------------------------------------------------------------------
# Figure: log-log, bandwidth line and compute line meeting at the ridge point.
# ---------------------------------------------------------------------------
sel = table[table.config_id.isin(FIG_CONFIGS)]
x_lo = min(sel.ai.min(), ridge) / 2
x_hi = max(sel.ai.max(), ridge) * 2

fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4.6))
xs = np.array([x_lo, ridge])
ax.plot(xs, peak_bw * xs, color=INK_MUTED, lw=1.5, zorder=2)
ax.plot([ridge, x_hi], [peak_gf, peak_gf], color=INK_MUTED, lw=1.5, zorder=2)
ax.axvline(ridge, color=INK_MUTED, lw=0.8, ls=":", zorder=1)
ax.annotate(f"peak compute {peak_gf:,.0f} GFLOP/s", (ridge, peak_gf), xytext=(4, 4),
            textcoords="offset points", ha="left", fontsize=8, color=INK_MUTED)
ax.annotate(f"peak bandwidth {peak_bw:.0f} GB/s", (x_lo, peak_bw * x_lo), xytext=(4, 9),
            textcoords="offset points", ha="left", fontsize=8, color=INK_MUTED, rotation=33,
            rotation_mode="anchor")
ax.annotate(f"ridge {ridge:.1f} FLOP/byte", (ridge, peak_gf * 0.0), xycoords=("data", "axes fraction"),
            xytext=(3, 4), textcoords="offset points", fontsize=8, color=INK_MUTED)

for v in GPU_VERSIONS:
    d = sel[sel.version == v]
    ax.scatter(d.ai, d.gflops, s=30, marker=VERSION_MARKERS[v], color=VERSION_COLORS[v],
               edgecolors="white", linewidths=0.6, label=VERSION_NAMES[v], zorder=3)

# One label per point cluster: versions that share a config and an intensity.
# Offsets (points) move labels that would otherwise overlap; False = the V2/V3/V5 cluster.
LABEL_OFFSETS = {
    ("B08", True): (-11, -2), ("A00", True): (-3, 7), ("B02", True): (0, -24), ("B12", True): (7, 5),
    ("B02", False): (0, -11), ("B04", False): (0, -11),
}
sel = sel.assign(group=sel.version.isin([1, 4]))
for (cid, is_im2col), g in sel.groupby(["config_id", "group"]):
    top = g.loc[g.gflops.idxmax()]
    offset = LABEL_OFFSETS.get((cid, bool(is_im2col)), (0, 6))
    ax.annotate(cid, (top.ai, top.gflops), xytext=offset, textcoords="offset points",
                ha="center", fontsize=7, color=INK_MUTED)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(x_lo, x_hi)
ax.set_ylim(sel.gflops.min() / 2, peak_gf * 2)
ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
ax.set_ylabel("Throughput (GFLOP/s)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
fig.tight_layout()
save_fig(fig, args.figures, "fig05_roofline.png")
