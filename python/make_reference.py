# python/make_reference.py
import argparse, csv, os, torch, torch.nn.functional as F
ap = argparse.ArgumentParser()
ap.add_argument("--configs", required=True)
ap.add_argument("--outdir", default="data")
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
with open(a.configs) as f:
    for row in csv.DictReader(f):
        cid = row["id"]
        N, Cin, H, W, Cout, K, S, P = (int(row[x]) for x in ["N","Cin","H","W","Cout","K","S","P"])
        g = torch.Generator().manual_seed(12345)
        x = torch.rand(N, Cin, H, W, generator=g) * 2 - 1
        w = torch.rand(Cout, Cin, K, K, generator=g) * 2 - 1
        y = F.conv2d(x.double(), w.double(), stride=S, padding=P).float()
        x.numpy().tofile(os.path.join(a.outdir, f"{cid}_input.bin"))
        w.numpy().tofile(os.path.join(a.outdir, f"{cid}_weight.bin"))
        y.numpy().tofile(os.path.join(a.outdir, f"{cid}_output.bin"))
        print(cid, tuple(y.shape))
