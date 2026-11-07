# python/repeatability.py. Compares the final GPU results with an independent repeat
# of the same runs (results/repeat/). The repeat is used for this table only.
# Output: tables/tab_repeatability.csv
#   py -3.11 python/repeatability.py
import pandas as pd

from analysis_common import load_gpu, save_table, sig3

final = load_gpu("results").set_index(["config_id", "version"])
repeat = load_gpu("results/repeat").set_index(["config_id", "version"])

rows = []
for v in sorted(final.index.get_level_values("version").unique()):
    a = final.xs(v, level="version").kernel_ms_median
    b = repeat.xs(v, level="version").kernel_ms_median
    diff = ((a / b - 1).abs() * 100).dropna()
    rows.append({
        "version": v,
        "rows": len(diff),
        "median_diff_pct": sig3(diff.median()),
        "max_diff_pct": sig3(diff.max()),
        "max_diff_config": diff.idxmax(),
        "rows_over_10pct": int((diff > 10).sum()),
    })
save_table(pd.DataFrame(rows), "tables", "tab_repeatability.csv")
print(pd.DataFrame(rows).to_string(index=False))
