# python/bench_cudnn.py (owner: Faizan). Section 4.10.
# Benchmarks V5: PyTorch F.conv2d (cuDNN) on the GPU, writing the Table 5 CSV columns.
# Usage:
#   py -3.11 python/bench_cudnn.py --configs configs/test.csv --mode check --out results/check_v5.csv
#   py -3.11 python/bench_cudnn.py --configs configs/setA.csv --mode bench --out results/A_v5.csv
import argparse, csv, os, statistics, time
import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True

ap = argparse.ArgumentParser()
ap.add_argument("--configs", required=True)
ap.add_argument("--mode", required=True, choices=["check", "bench"])
ap.add_argument("--out", required=True)
ap.add_argument("--data", default="data")
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--reps", type=int, default=10)
args = ap.parse_args()

COLUMNS = ["config_id", "version", "threads", "status", "kernel_ms_median", "kernel_ms_min",
           "kernel_ms_max", "e2e_pageable_ms_median", "e2e_pinned_ms_median", "peak_mem_mb",
           "memgetinfo_mb", "max_abs_err"]


def blank_row(cid):
    row = {c: "NA" for c in COLUMNS}
    row.update(config_id=cid, version=5, threads=0, status="OK")
    return row


def load(path, shape):
    if not os.path.exists(path):
        raise SystemExit(f"Missing data file: {path}")
    return np.fromfile(path, dtype=np.float32).reshape(shape)


START_SPINUP_S, ROW_SPINUP_S = 3.0, 0.5


def gpu_spinup(seconds):
    """Keep the GPU busy so timing does not start at idle clocks, as main.cu does."""
    a = torch.rand(1024, 1024, device="cuda")
    src = torch.zeros(64 << 20, device="cuda")     # 256 MiB of memory traffic
    dst = torch.empty_like(src)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        b = a @ a
        dst.copy_(src)
        torch.cuda.synchronize()
    del a, b, src, dst


def e2e_median(xh, wh, yh, S, P):
    """Median end-to-end time in ms, using the given host tensors (Section 4.10)."""
    times = []
    for i in range(args.warmup + args.reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        xd = xh.to("cuda")
        wd = wh.to("cuda")
        yd = F.conv2d(xd, wd, stride=S, padding=P)
        yh.copy_(yd)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        del xd, wd, yd
        if i >= args.warmup:
            times.append((t1 - t0) * 1000.0)
    return statistics.median(times)


def run_config(row):
    cid = row["id"]
    N, Cin, H, W, Cout, K, S, P = (int(row[x]) for x in ["N", "Cin", "H", "W", "Cout", "K", "S", "P"])
    Ho = (H + 2 * P - K) // S + 1
    Wo = (W + 2 * P - K) // S + 1
    print(f"Config {cid}: N={N} Cin={Cin} H={H} W={W} Cout={Cout} K={K} S={S} P={P} => Ho={Ho} Wo={Wo}", flush=True)

    x_np = load(os.path.join(args.data, f"{cid}_input.bin"), (N, Cin, H, W))
    w_np = load(os.path.join(args.data, f"{cid}_weight.bin"), (Cout, Cin, K, K))
    ref_np = load(os.path.join(args.data, f"{cid}_output.bin"), (N, Cout, Ho, Wo))

    out = blank_row(cid)
    x = torch.from_numpy(x_np).to("cuda")
    w = torch.from_numpy(w_np).to("cuda")

    if args.mode == "bench":
        gpu_spinup(ROW_SPINUP_S)

    # Warm-up: cuDNN chooses its algorithm during the first call.
    y = None
    for _ in range(args.warmup):
        y = F.conv2d(x, w, stride=S, padding=P)

    # Memory: input, weights, output and cuDNN workspace.
    del y
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    y = F.conv2d(x, w, stride=S, padding=P)
    torch.cuda.synchronize()
    out["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 1048576

    # Correctness (Section 4.8).
    y_np = y.cpu().numpy()
    diff = np.abs(y_np.astype(np.float64) - ref_np.astype(np.float64))
    tol = 0.001 + 0.001 * np.abs(ref_np.astype(np.float64))
    out["max_abs_err"] = float(np.max(diff))
    out["status"] = "OK" if bool(np.all(diff <= tol)) else "FAIL"

    if args.mode == "check":
        return out

    # Kernel time.
    del y
    times = []
    for _ in range(args.reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        y = F.conv2d(x, w, stride=S, padding=P)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        del y
    out["kernel_ms_median"] = statistics.median(times)
    out["kernel_ms_min"] = min(times)
    out["kernel_ms_max"] = max(times)
    del x, w

    # End-to-end: pageable, then the same tensors pinned.
    xh = torch.from_numpy(x_np.copy())
    wh = torch.from_numpy(w_np.copy())
    yh = torch.empty(N, Cout, Ho, Wo, dtype=torch.float32)
    out["e2e_pageable_ms_median"] = e2e_median(xh, wh, yh, S, P)
    out["e2e_pinned_ms_median"] = e2e_median(xh.pin_memory(), wh.pin_memory(), yh.pin_memory(), S, P)
    return out


def fmt(v):
    return f"{v:.6g}" if isinstance(v, float) else str(v)


print(f"{torch.cuda.get_device_name(0)}; PyTorch {torch.__version__}, CUDA {torch.version.cuda}, "
      f"cuDNN {torch.backends.cudnn.version()}", flush=True)

with open(args.configs) as f:
    rows = list(csv.DictReader(f))

if args.mode == "bench":
    gpu_spinup(START_SPINUP_S)

passed = 0
with open(args.out, "w", newline="") as fout:
    fout.write(",".join(COLUMNS) + "\n")
    for row in rows:
        try:
            res = run_config(row)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            res = blank_row(row["id"])
            res["status"] = "OOM"
        fout.write(",".join(fmt(res[c]) for c in COLUMNS) + "\n")
        fout.flush()
        print(f"  V5: {res['status']}  kernel={fmt(res['kernel_ms_median'])} ms  "
              f"e2e={fmt(res['e2e_pageable_ms_median'])}/{fmt(res['e2e_pinned_ms_median'])} ms  "
              f"mem={fmt(res['peak_mem_mb'])} MB  max_err={fmt(res['max_abs_err'])}", flush=True)
        passed += res["status"] == "OK"

print(f"\n=== SUMMARY: {passed} / {len(rows)} rows OK ===")
raise SystemExit(0 if passed == len(rows) else 1)
