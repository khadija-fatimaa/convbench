# python/check_report_numbers.py. Section 9.2 requires every number in the report to come from
# the CSV files. Lists numbers in report/report.tex that match no result at 3 significant
# figures, so each can be checked by hand. Tables in report/generated/ come from the CSVs.
#   py -3.11 python/check_report_numbers.py
import glob
import math
import re

import pandas as pd

from analysis_common import load_configs, load_cpu, load_gpu, load_peaks


def sf3(x):
    if x == 0 or pd.isna(x):
        return None
    return round(float(x), 2 - int(math.floor(math.log10(abs(x)))))


known = set()


def add(values):
    for v in values:
        try:
            s = sf3(float(v))
        except (TypeError, ValueError):
            continue
        if s is not None:
            known.add(s)


# Every numeric cell in results and tables.
for path in glob.glob("results/*.csv") + glob.glob("tables/*.csv") + glob.glob("results/repeat/*.csv"):
    df = pd.read_csv(path)
    for col in df.select_dtypes("number"):
        add(df[col])

# Derived values.
cfg = load_configs("configs")
gpu = load_gpu("results")
cpu = load_cpu("results")
peak_bw, peak_gf = load_peaks("results")
add(cfg.gflop)
add([peak_gf / peak_bw, peak_bw, peak_gf])
k = gpu.pivot(index="config_id", columns="version", values="kernel_ms_median")
add(k.loc["A05", [2, 3, 4]].values / k.loc["A03", [2, 3, 4]].values)
add((gpu.e2e_pageable_ms_median / gpu.e2e_pinned_ms_median).groupby(gpu.version).median())
r = pd.read_csv("tables/roofline.csv")
for v, d in r.groupby("version"):
    add([d.ai.median(), d.gflops.median() / peak_gf * 100, d.bandwidth_gbs.median() / peak_bw * 100])
s = pd.read_csv("tables/tab_speedup.csv")
add(s.filter(like="over_V5").median())
v7 = cpu[cpu.version == 7].pivot(index="config_id", columns="threads", values="kernel_ms_median")
add([v7.at[c, 1] / v7.at[c, 12] for c in ("A00", "B04")])
best = v7.min(axis=1)
v6 = cpu[cpu.version == 6].set_index("config_id").kernel_ms_median
add([(v6 / best).min(), (v6 / best).max()])
page = gpu.pivot(index="config_id", columns="version", values="e2e_pageable_ms_median")
fast = k.loc[cfg[cfg.set == "A"].index, [1, 2, 3]].idxmin(axis=1)
add([(best[fast.index] / pd.Series({c: page.at[c, fast[c]] for c in fast.index})).min()])
t = {c: pd.read_csv(f"results/tune_cob{c}.csv").kernel_ms_median.sum() for c in (8, 16)}
add([(t[8] - t[16]) / t[8] * 100])

# Prose: drop preamble, references, comments and citation markers.
text = open("report/report.tex", encoding="utf8").read()
text = text[text.index(r"\begin{document}"):text.index(r"\section*{References}")]
text = re.sub(r"\\(label|ref|gentable|fig|input|section|subsection)(\[[^\]]*\])?\{[^}]*\}", " ", text)
text = re.sub(r"\[\d+\]", " ", text)
text = re.sub(r"\b[VABT]\d+\b|\bGap \d\b|\bO\d\b", " ", text)
found = re.findall(r"(?<![\w.])\d[\d,]*\.?\d*", text)
seen, missing = set(), []
for tok in found:
    tok = tok.rstrip(".,")
    if tok in seen or not tok:
        continue
    seen.add(tok)
    value = float(tok.replace(",", ""))
    if sf3(value) not in known:
        missing.append(tok)
print(f"{len(seen)} distinct numbers in the prose, {len(missing)} not found in the results:")
print(", ".join(missing))

# Abstract length (Section 9.2: 150 to 200 words).
a = text[text.index("Abstract"):text.index(r"\newpage")]
print("abstract words:", len(a.split()) - 2)
