"""Does padding the scan to the kernel's chunk boundary move the crossover?

The scan is 42-46% of Jamba's training step, and it is 1.5-1.7x more expensive at 196/392
than at 256/512 because mamba's dispatch charges for a whole block of 128/256/512/1024/2048
timesteps. 196 and 392 are the per-frame token counts this project actually trains on, and
both sit on the wrong side of a boundary. sscanop2._pad_len rounds up; this measures what
that is worth end to end.

Same measurement standard as realcross.py -- dataloader, head, real loss, backward,
optimizer step, cudagraph-fallback count -- because nothing else counts as evidence here.
Both arms are the identical model and the identical op; the only difference is PAD_SCAN.
"""

import sys, time, json, math, warnings, importlib.util
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/content")
_spec = importlib.util.spec_from_file_location("bench", "/content/benchmark_seq_scaling.py")
bench = importlib.util.module_from_spec(_spec)
sys.modules["bench"] = bench
_spec.loader.exec_module(bench)

import sscanop2 as co
import realcross as rc

DEVICE, DTYPE = "cuda", torch.bfloat16
# Extended past 1868 and repeated. The previous grid stopped at 1568 and reported
# ">1568" for both arms, which is consistent with the recorded crossover of 1868 but does
# not confirm it -- and the measured ratios extrapolated past 2000. A headline number that
# a rerun cannot reproduce is worse than no number, so this pins it with repeats.
GRID = [1024, 1568, 1868, 2048, 2560, 3136]
BATCH = 16
STEPS, WARM = 40, 10
REPEATS = 2


def step_rate(build, L, name):
    torch.cuda.empty_cache(); torch._dynamo.reset()
    n = (STEPS + WARM + 2) * BATCH
    dl = DataLoader(TensorDataset(torch.randn(n, L, bench.HIDDEN_SIZE), torch.randn(n, 256)),
                    batch_size=BATCH, shuffle=True, pin_memory=True, drop_last=True)
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
    print(f"  {name:<22} L={L:<5} {STEPS/dt:8.2f} step/s  {mem:5.2f} GB{flag}", flush=True)
    return STEPS / dt


def main():
    orig, shim = co.register()

    print("equivalence, padding ON (this is the gate):", flush=True)
    co.PAD_SCAN = True
    if not co.verify(orig, shim):
        print("STOPPING: padded scan is not equivalent", flush=True); return
    print(f"  _pad_len: " + ", ".join(f"{L}->{co._pad_len(L)}" for L in GRID), flush=True)

    jamba = lambda: rc.Wrapped(
        torch.compile(bench.BidirectionalJambaBatched(), mode="reduce-overhead"),
        bench.HIDDEN_SIZE)
    vit = lambda: rc.Wrapped(rc.ViTEnc(), 328)

    curves, spread = {}, {}
    for label, pad in (("ViT matched", None), ("Jamba pad=off", False), ("Jamba pad=ON", True)):
        c, s = {}, {}
        for L in GRID:
            if pad is not None:
                co.PAD_SCAN = pad
            reps = []
            for _ in range(REPEATS):
                try:
                    reps.append(step_rate(vit if pad is None else jamba, L, label))
                except Exception as e:  # noqa: BLE001
                    print(f"  {label:<22} L={L:<5} FAILED {type(e).__name__}: "
                          f"{str(e)[:120]}", flush=True)
                    torch.cuda.empty_cache(); torch._dynamo.reset()
            if reps:
                # Best of N. A slow rep is contention or a compile artefact, never the model
                # being genuinely faster, so the max is the less noisy estimator here.
                c[L], s[L] = max(reps), (max(reps) - min(reps)) / max(reps)
                print(f"  {label:<22} L={L:<5} best {c[L]:7.2f} step/s  "
                      f"spread {100*s[L]:4.1f}%", flush=True)
        curves[label], spread[label] = c, s

    v = curves["ViT matched"]
    for label in ("Jamba pad=off", "Jamba pad=ON"):
        n = rc.crossover(curves[label], v)
        print(f"  CROSSOVER  {label:<16} = {n}", flush=True)
    for L in GRID:
        if L in curves["Jamba pad=off"] and L in curves["Jamba pad=ON"]:
            a, b = curves["Jamba pad=off"][L], curves["Jamba pad=ON"][L]
            print(f"  speedup from padding  L={L:<5} {b/a:5.3f}x", flush=True)

    print("PAD_JSON " + json.dumps({k: {str(a): b for a, b in c.items()}
                                    for k, c in curves.items()}), flush=True)


main()
