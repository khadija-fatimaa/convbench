# python/bench_cudnn.py (owner: Faizan). Section 4.10.
# Benchmarks PyTorch cuDNN convolution (V5) on the GPU.
# Also verifies correctness against a CPU double-precision reference.
# Usage:
#   py -3.11 python/bench_cudnn.py --configs configs/test.csv --mode check --out results/check_v5.csv
#   py -3.11 python/bench_cudnn.py --configs configs/setA.csv --mode bench --out results/A_v5.csv
import argparse, csv, os, time, sys
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--configs", required=True, help="Path to config CSV")
ap.add_argument("--mode", default="bench", choices=["check", "bench"])
ap.add_argument("--out", required=True, help="Output CSV path")
ap.add_argument("--datadir", default="data", help="Reference data directory")
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--reps", type=int, default=20)
args = ap.parse_args()

# Disable TF32 for fair comparison (Section 4.10 rule)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = True

device = torch.device("cuda")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, cuDNN: {torch.backends.cudnn.version()}")

rows = []
with open(args.configs) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

fout = open(args.out, "w", newline="")
if args.mode == "check":
    fout.write("config,version,pass,max_err\n")
else:
    fout.write("config,version,threads,v3cob,kernel_ms,e2e_pageable_ms,gflops_kernel,gflops_e2e\n")

for row in rows:
    cid = row["id"]
    N    = int(row["N"])
    Cin  = int(row["Cin"])
    H    = int(row["H"])
    W    = int(row["W"])
    Cout = int(row["Cout"])
    K    = int(row["K"])
    S    = int(row["S"])
    P    = int(row["P"])
    Ho = (H + 2*P - K) // S + 1
    Wo = (W + 2*P - K) // S + 1
    flops = 2.0 * N * Cout * Ho * Wo * Cin * K * K

    print(f"Config {cid}: N={N} Cin={Cin} H={H} W={W} Cout={Cout} K={K} S={S} P={P} => Ho={Ho} Wo={Wo}")

    if args.mode == "check":
        # Load reference data
        inp_path = os.path.join(args.datadir, f"{cid}_input.bin")
        wt_path  = os.path.join(args.datadir, f"{cid}_weight.bin")
        ref_path = os.path.join(args.datadir, f"{cid}_output.bin")
        import numpy as np
        x_np = np.fromfile(inp_path, dtype=np.float32).reshape(N, Cin, H, W)
        w_np = np.fromfile(wt_path, dtype=np.float32).reshape(Cout, Cin, K, K)
        ref_np = np.fromfile(ref_path, dtype=np.float32).reshape(N, Cout, Ho, Wo)

        x_gpu = torch.from_numpy(x_np).to(device)
        w_gpu = torch.from_numpy(w_np).to(device)
        y_gpu = F.conv2d(x_gpu, w_gpu, stride=S, padding=P)
        y_out = y_gpu.cpu().numpy()

        diff = np.abs(y_out - ref_np)
        tol  = 0.001 + 0.001 * np.abs(ref_np)
        max_err = float(np.max(diff))
        passed = bool(np.all(diff <= tol))

        status = "PASS" if passed else "FAIL"
        print(f"  V5_cudnn: {status} (max_err={max_err:.6e})")
        fout.write(f"{cid},V5_cudnn,{status},{max_err:.6e}\n")

    else:
        # Bench mode
        g = torch.Generator().manual_seed(42)
        x_cpu = torch.rand(N, Cin, H, W, generator=g) * 2 - 1
        w_cpu = torch.rand(Cout, Cin, K, K, generator=g) * 2 - 1

        x_gpu = x_cpu.to(device)
        w_gpu = w_cpu.to(device)

        # Warmup
        for _ in range(args.warmup):
            _ = F.conv2d(x_gpu, w_gpu, stride=S, padding=P)
        torch.cuda.synchronize()

        # Kernel-only timing (data already on GPU)
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev   = torch.cuda.Event(enable_timing=True)
        kernel_times = []
        for _ in range(args.reps):
            start_ev.record()
            _ = F.conv2d(x_gpu, w_gpu, stride=S, padding=P)
            end_ev.record()
            torch.cuda.synchronize()
            kernel_times.append(start_ev.elapsed_time(end_ev))

        kernel_times.sort()
        kernel_ms = kernel_times[len(kernel_times) // 2]

        # End-to-end timing (pageable: CPU -> GPU -> compute -> GPU -> CPU)
        e2e_times = []
        for _ in range(args.reps):
            start_ev.record()
            x_d = x_cpu.to(device)
            w_d = w_cpu.to(device)
            y_d = F.conv2d(x_d, w_d, stride=S, padding=P)
            y_c = y_d.cpu()
            end_ev.record()
            torch.cuda.synchronize()
            e2e_times.append(start_ev.elapsed_time(end_ev))

        e2e_times.sort()
        e2e_ms = e2e_times[len(e2e_times) // 2]

        gflops_kernel = (flops / 1e9) / (kernel_ms / 1e3)
        gflops_e2e    = (flops / 1e9) / (e2e_ms / 1e3)

        print(f"  V5_cudnn: kernel={kernel_ms:.3f} ms ({gflops_kernel:.1f} GFLOP/s), "
              f"e2e_pageable={e2e_ms:.3f} ms ({gflops_e2e:.1f} GFLOP/s)")
        fout.write(f"{cid},V5_cudnn,0,0,{kernel_ms:.4f},{e2e_ms:.4f},{gflops_kernel:.2f},{gflops_e2e:.2f}\n")

fout.close()
print(f"\nResults written to {args.out}")
