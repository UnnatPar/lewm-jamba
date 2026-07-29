"""How many 196-token frames can we encode at once and still beat the ViT?

The crossover is 1,723 (measured, padded scan). Multiples of 196 bracketing it:

    8 frames = 1568   below  -> ViT wins
    9 frames = 1764   the smallest above the crossover
   10 frames = 1960   also above
   11 frames = 2156   above, but crosses into the next scan chunk

The interesting part is the padding interaction. sscanop2._pad_len rounds the scan up to
128/256/512/1024/2048, so 1764 and 1960 BOTH pad to 2048 and pay identical scan cost -- 10
frames is 11% more context than 9 for nothing. 2156 spills into a 4096 chunk and pays nearly
double. So the right frame count is the largest multiple of 196 that still fits under a chunk
boundary, not the smallest one above the crossover.

1723 is interpolated between measured points at 1568 and 1868, and 1764 sits only 2.4% above
it. Tonight already produced one wrong headline number by trusting an interpolation across a
grid gap (1868, actually 1946), so this measures the frame counts directly.
"""

import sys, time, json, warnings, importlib.util
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
FRAME = 196
FRAMES = [8, 9, 10, 11, 12]
BATCH = 16
STEPS, WARM, REPEATS = 40, 10, 2


def rate(build, L):
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
                torch.cuda.synchronize(); t0 = time.perf_counter()
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
    return STEPS / dt, mem, fb


def main():
    orig, shim = co.register()
    co.PAD_SCAN = True
    if not co.verify(orig, shim):
        print("STOPPING: equivalence failed"); return

    jamba = lambda: rc.Wrapped(
        torch.compile(bench.BidirectionalJambaBatched(), mode="reduce-overhead"),
        bench.HIDDEN_SIZE)
    vit = lambda: rc.Wrapped(rc.ViTEnc(), 328)

    print(f"\n  {'frames':>6} {'tokens':>7} {'padded':>7} {'ViT':>9} {'Jamba':>9} "
          f"{'ratio':>7}  {'GB':>5}", flush=True)
    out = {}
    for f in FRAMES:
        L = f * FRAME
        row = {}
        for name, build in (("vit", vit), ("jamba", jamba)):
            best, mem, fb = None, 0, 0
            for _ in range(REPEATS):
                try:
                    r, m, b = rate(build, L)
                except Exception as e:  # noqa: BLE001
                    print(f"  {f:>6} {L:>7}  {name} FAILED {type(e).__name__}: "
                          f"{str(e)[:90]}", flush=True)
                    torch.cuda.empty_cache(); torch._dynamo.reset()
                    continue
                if best is None or r > best:
                    best, mem, fb = r, m, b
            row[name] = best
            row[name + "_mem"] = mem
            row[name + "_fb"] = fb
        if row.get("vit") and row.get("jamba"):
            ratio = row["jamba"] / row["vit"]
            flag = "  JAMBA WINS" if ratio > 1 else ""
            cg = f"  CGFALLBACK x{row['jamba_fb']}" if row["jamba_fb"] else ""
            print(f"  {f:>6} {L:>7} {co._pad_len(L):>7} {row['vit']:>9.2f} "
                  f"{row['jamba']:>9.2f} {ratio:>7.3f}  {row['jamba_mem']:>5.2f}"
                  f"{flag}{cg}", flush=True)
        out[str(L)] = row
    print("FRAME_JSON " + json.dumps(out), flush=True)


main()
