"""Why is the measured crossover 1868 when the FLOP model says 53?

flopmodel.py (CPU, no credits) says Jamba and the matched ViT have near-identical per-token
MAC counts (13.2M vs 12.9M) while ViT's quadratic coefficient is 11x larger. On arithmetic
alone Jamba should win by L~53. It measures 1868. So the gap is not compute, it is achieved
efficiency -- and that is a property of Jamba's kernels specifically, not a global overhead
that rescales both curves.

This prints the number that decides where to spend the remaining credits: achieved TFLOP/s
for each model, and the CUDA-time breakdown underneath it.
"""

import sys, time, json
import torch
from torch import nn
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, "/content")

# `bench` is benchmark_seq_scaling.py under its in-repo name. colab-run.sh uploads by
# basename, so register the alias here rather than renaming the file on the way up --
# realcross.py's own `import bench` then resolves to the same module object.
import importlib.util
_spec = importlib.util.spec_from_file_location("bench", "/content/benchmark_seq_scaling.py")
bench = importlib.util.module_from_spec(_spec)
sys.modules["bench"] = bench
_spec.loader.exec_module(bench)

import sscanop2 as co
import realcross as rc

DEVICE, DTYPE = "cuda", torch.bfloat16
LS = [392, 1868]
ITERS = 12

# MACs/token from flopmodel.py, kept in sync by construction below.
H, INTER, VH, VL = bench.HIDDEN_SIZE, bench.INTERMEDIATE_SIZE, 328, 10
N_MAMBA = bench.NUM_HIDDEN_LAYERS - 1
DI, DS, DC, DTR = bench.MAMBA_EXPAND * H, bench.MAMBA_D_STATE, bench.MAMBA_D_CONV, -(-H // 16)

JAMBA_TOK = (N_MAMBA * (H * DI * 2 + DI * H + 2 * DI * DC + 2 * DI * (DTR + 2 * DS)
                        + 2 * DTR * DI + 2 * DI * DS * 4 + 3 * H * INTER)
             + 4 * H * H + 3 * H * INTER)
JAMBA_QUAD = 2 * H
VIT_TOK = VL * 4 * VH * VH + VL * 2 * VH * (4 * VH)
VIT_QUAD = VL * 2 * VH


def macs(model, L, batch):
    tok, quad = (JAMBA_TOK, JAMBA_QUAD) if model == "jamba" else (VIT_TOK, VIT_QUAD)
    return batch * L * (tok + quad * L)


def build(which):
    if which == "jamba":
        enc, dim = torch.compile(bench.BidirectionalJambaBatched(),
                                 mode="reduce-overhead"), bench.HIDDEN_SIZE
    else:
        enc, dim = rc.ViTEnc(), 328
    return rc.Wrapped(enc, dim).to(DEVICE, DTYPE)


def run(which, L, batch=16):
    torch.cuda.empty_cache(); torch._dynamo.reset()
    model = build(which)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lossf = nn.MSELoss()
    x = torch.randn(batch, L, bench.HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    y = torch.randn(batch, 256, device=DEVICE, dtype=DTYPE)

    def step():
        torch.compiler.cudagraph_mark_step_begin()
        opt.zero_grad(set_to_none=True)
        lossf(model(x + torch.randn_like(x) * 1e-6), y).backward()
        opt.step()

    for _ in range(8):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / ITERS

    # forward+backward ~ 3x forward MACs, 2 FLOP per MAC
    tflops = 3 * 2 * macs(which, L, batch) / dt / 1e12
    print(f"  {which:<6} L={L:<5} {1/dt:7.2f} step/s   {tflops:6.2f} TFLOP/s achieved",
          flush=True)

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
    ev = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    tot = sum(e.self_device_time_total for e in ev) or 1
    print(f"    top CUDA time ({which}, L={L}):", flush=True)
    for e in sorted(ev, key=lambda e: -e.self_device_time_total)[:12]:
        print(f"      {100*e.self_device_time_total/tot:5.1f}%  {e.key[:70]}", flush=True)

    del model, opt
    torch.cuda.empty_cache(); torch._dynamo.reset()
    return {"step_s": dt, "tflops": tflops}


def main():
    orig, shim = co.register()
    if not co.verify(orig, shim):
        print("STOPPING: equivalence failed"); return
    print(f"MAC model: jamba {JAMBA_TOK/1e6:.2f} M/tok + {JAMBA_QUAD}L, "
          f"vit {VIT_TOK/1e6:.2f} M/tok + {VIT_QUAD}L", flush=True)
    out = {}
    for L in LS:
        for which in ("vit", "jamba"):
            try:
                out[f"{which}|{L}"] = run(which, L)
            except Exception as e:  # noqa: BLE001
                print(f"  {which} L={L} FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
                torch.cuda.empty_cache(); torch._dynamo.reset()
    print("PROF_JSON " + json.dumps(out), flush=True)


main()
