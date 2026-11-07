# python/analysis_common.py -- shared by roofline.py, plots.py and sanity_checks.py.
# Loads the configs and result CSVs (Table 5 columns) and holds the chart style.
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

VERSION_NAMES = {
    1: "V1 im2col + GEMM",
    2: "V2 implicit GEMM",
    3: "V3 direct",
    4: "V4 im2col + cuBLAS",
    5: "V5 cuDNN",
}
# Fixed categorical order, so a version keeps its colour in every figure.
# Shape is a second encoding for greyscale printing and colour-blind readers.
VERSION_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100", 5: "#e87ba4"}
VERSION_MARKERS = {1: "o", 2: "s", 3: "^", 4: "D", 5: "v"}
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
GPU_VERSIONS = [1, 2, 3, 4, 5]
FIG_WIDTH = 6.5   # inches (Section 7.4)


def apply_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    })


def load_configs(cfg_dir):
    """Set A and Set B configs with output size, FLOPs and bytes (Section 7.1, 7.2)."""
    frames = []
    for name, tag in (("setA.csv", "A"), ("setB.csv", "B")):
        df = pd.read_csv(os.path.join(cfg_dir, name))
        df["set"] = tag
        frames.append(df)
    c = pd.concat(frames, ignore_index=True)
    c = c.rename(columns={"id": "config_id"})
    c["Ho"] = (c.H + 2 * c.P - c.K) // c.S + 1
    c["Wo"] = (c.W + 2 * c.P - c.K) // c.S + 1
    k2 = c.K * c.K
    c["flops"] = 2 * c.N * c.Cout * c.Ho * c.Wo * c.Cin * k2
    c["gflop"] = c.flops / 1e9
    c["bytes_base"] = 4 * (c.N * c.Cin * c.H * c.W + c.Cout * c.Cin * k2 + c.N * c.Cout * c.Ho * c.Wo)
    c["bytes_im2col"] = c.bytes_base + 2 * 4 * c.N * c.Cin * k2 * c.Ho * c.Wo
    return c.set_index("config_id")


def load_gpu(res_dir):
    """V1 to V5 rows for Set A and Set B."""
    names = ["A_gpu.csv", "B_gpu.csv", "A_v5.csv", "B_v5.csv"]
    return pd.concat([pd.read_csv(os.path.join(res_dir, n)) for n in names], ignore_index=True)


def load_cpu(res_dir):
    """V6 and V7 rows for Set A and Set B."""
    names = ["A_cpu.csv", "B_cpu.csv"]
    return pd.concat([pd.read_csv(os.path.join(res_dir, n)) for n in names], ignore_index=True)


def load_peaks(res_dir):
    p = pd.read_csv(os.path.join(res_dir, "peaks.csv")).set_index("metric")["value"]
    return float(p["peak_bandwidth_gbs"]), float(p["peak_gflops"])


def only_ok(df):
    return df[df.status == "OK"]


def sig3(x):
    """Round to 3 significant figures (Section 9.2 writing rule)."""
    if x is None or pd.isna(x) or x == 0:
        return x
    return round(float(x), 2 - int(math.floor(math.log10(abs(x)))))


def save_fig(fig, fig_dir, name):
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, name)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", path)


def save_table(df, tab_dir, name):
    os.makedirs(tab_dir, exist_ok=True)
    path = os.path.join(tab_dir, name)
    df.to_csv(path, index=False)
    print("wrote", path)
