"""Crossover measured in the real training loop -- the only version that counts.

Same harness as realloop.py (dataloader, head, loss, optimizer.step, cudagraph-fallback
reporting), swept over sequence length instead of batch size. Everything reported here is
step/s of a full training step, not forward+backward on a static synthetic tensor.

Batch 1 to stay comparable with the historical crossover numbers, plus batch 16 because the
ranking at batch 1 is not guaranteed to hold where the GPU is saturated -- at batch 64 the
Jamba advantage over ViT already shrinks from 1.3x to 2.1x against.
"""

import sys, time, json, math, warnings
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/content")
import bench
import sscanop2 as co  # fused backward; sscanop.py (recompute backward) was superseded

DEVICE, DTYPE = "cuda", torch.bfloat16
GRID = [196, 256, 320, 392, 512, 640, 784, 1024, 1568, 3136, 6272]
STEPS, WARM = 25, 8


class Head(nn.Module):
    def __init__(self, dim, out=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, out), nn.GELU(), nn.Linear(out, out))

    def forward(self, x):
        return self.net(x)


class Wrapped(nn.Module):
    def __init__(self, encoder, dim):
        super().__init__()
        self.encoder, self.head = encoder, Head(dim)

    def forward(self, x):
        return self.head(self.encoder(x))


def make_vit(hidden=328, layers=10):
    layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=8, dim_feedforward=hidden * 4,
                                       dropout=0.0, activation="gelu", batch_first=True,
                                       norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=layers)


class ViTEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(bench.HIDDEN_SIZE, 328)
        self.enc = torch.compile(make_vit(), mode="reduce-overhead")

    def forward(self, x):
        return self.enc(self.proj(x)).mean(dim=1)


def step_rate(build, dim, L, batch, name):
    torch.cuda.empty_cache(); torch._dynamo.reset()
    n = (STEPS + WARM + 2) * batch
    dl = DataLoader(TensorDataset(torch.randn(n, L, bench.HIDDEN_SIZE), torch.randn(n, 256)),
                    batch_size=batch, shuffle=True, pin_memory=True, drop_last=True)
    model = build().to(DEVICE, DTYPE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lossf = nn.MSELoss()
    it = iter(dl)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in range(WARM + STEPS):
            if i == WARM:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(dl); x, y = next(it)
            x = x.to(DEVICE, dtype=DTYPE, non_blocking=True)
            y = y.to(DEVICE, dtype=DTYPE, non_blocking=True)
            torch.compiler.cudagraph_mark_step_begin()
            opt.zero_grad(set_to_none=True)
            lossf(model(x), y).backward()
            opt.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        fb = sum("fast path of CUDAGraphs" in str(w.message) for w in caught)
    mem = torch.cuda.max_memory_allocated() / 1e9
    del model, opt
    torch.cuda.empty_cache(); torch._dynamo.reset()
    flag = f"  CGFALLBACK x{fb}" if fb else ""
    print(f"  {name:<24} L={L:<6} b={batch:<3} {STEPS/dt:8.2f} step/s  {mem:5.2f} GB{flag}",
          flush=True)
    return STEPS / dt, mem, fb


def crossover(j, v):
    shared = sorted(set(j) & set(v))
    d = [(L, j[L] - v[L]) for L in shared]
    if not d:
        return None
    if all(x > 0 for _, x in d):
        return f"<{shared[0]}"
    if all(x < 0 for _, x in d):
        return f">{shared[-1]}"
    for (L0, d0), (L1, d1) in zip(d, d[1:]):
        if d0 <= 0 < d1 or d0 < 0 <= d1:
            t = -d0 / (d1 - d0)
            return round(math.exp(math.log(L0) + t * (math.log(L1) - math.log(L0))))
    return None


def main():
    orig, shim = co.register()
    if not co.verify(orig, shim):
        print("STOPPING: equivalence failed"); return

    variants = {
        "ViT matched": (lambda: Wrapped(ViTEnc(), 328), 328),
        "Jamba single-graph": (lambda: Wrapped(
            torch.compile(bench.BidirectionalJambaBatched(), mode="reduce-overhead"),
            bench.HIDDEN_SIZE), bench.HIDDEN_SIZE),
        "Jamba surgical": (lambda: Wrapped(
            bench.BidirectionalJambaBatchedFullCompiledGlue(), bench.HIDDEN_SIZE),
            bench.HIDDEN_SIZE),
    }

    out = {}
    for batch in (1, 16):
        print(f"\n=== REAL TRAINING STEP, batch={batch} ===", flush=True)
        for name, (build, dim) in variants.items():
            curve = {}
            for L in GRID:
                try:
                    r, mem, fb = step_rate(build, dim, L, batch, name)
                    curve[L] = (r, mem, fb)
                except Exception as e:  # noqa: BLE001
                    print(f"  {name:<24} L={L:<6} b={batch:<3} FAILED {type(e).__name__}: "
                          f"{str(e)[:120]}", flush=True)
                    torch.cuda.empty_cache(); torch._dynamo.reset()
            out[f"{name}|{batch}"] = curve
        vit = {L: v[0] for L, v in out[f"ViT matched|{batch}"].items()}
        for name in ("Jamba single-graph", "Jamba surgical"):
            j = {L: v[0] for L, v in out[f"{name}|{batch}"].items()}
            print(f"  CROSSOVER batch={batch}  {name:<22} = {crossover(j, vit)}", flush=True)

    print("RESULT_JSON " + json.dumps(
        {k: {str(L): v for L, v in c.items()} for k, c in out.items()}), flush=True)


# Guarded so other harnesses can import ViTEnc/Wrapped/Head instead of re-declaring the
# baseline -- that kind of duplication is what produced a 21%-undersized ViT once already.
# Run it as a sweep with runpy.run_path(path, run_name="__main__"), not `colab exec -f`.
if __name__ == "__main__":
    main()
