"""Is the scan occupancy-starved, and has the benchmark been measuring the wrong regime?

Two things point the same way.

`ceiling.py` found the scan machinery costs 1.93x of step time but only ~1.27x of that is FLOPs
(d_state->1 reaches 0.826; deleting the machinery reaches 1.258). So ~1.52x is fixed cost, and a
fixed cost that does not shrink with FLOPs usually means the kernel is not filling the GPU.

The arithmetic agrees: the selective scan parallelises over batch * d_inner. At batch 8 with
bidirectional doubling and d_inner=288 that is 16*288 ~ 4.6k concurrent lanes, against roughly
221k thread slots on an A100. ~2% occupancy. The ViT's attention and GEMMs have no such problem,
so any measurement at small batch flatters the ViT.

And the real training loop does NOT run at batch 8 through the encoder. Each sample is
history_size+num_preds = 5 windows, so batch_size=4 sends 20 windows -- 40 after doubling. Every
crossover number this session was taken at 8 or 16 windows, i.e. below the regime that matters.

So sweep batch and watch the ratio. If it climbs, the crossover has been understated all session
and the honest number is the one at the real training batch. The ViT is re-measured at every
batch, so nothing here is a free lunch -- both models get the same parallelism.
"""

import json
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/content")

import module as mod
import crossover2 as c2

DEVICE = "cuda"
IMG, EMB, FRAMES = 224, 192, 10
HID, LAYERS = 288, 10
BEST = dict(expand=1, d_state=32, inter=1424)   # SSM state 9216, params ~15.50M
TARGET_PARAMS = 15_497_922
BATCHES = [4, 8, 16, 20, 24, 32]
STEPS, WARM, REPEATS = 10, 4, 2


def enc():
    return mod.JambaEncoder(
        image_size=IMG, output_dim=EMB, hidden_size=HID, intermediate_size=BEST["inter"],
        num_hidden_layers=LAYERS, num_attention_heads=8, num_key_value_heads=4,
        attn_layer_period=10, attn_layer_offset=9, mamba_d_state=BEST["d_state"],
        mamba_d_conv=4, mamba_expand=BEST["expand"], max_frames=FRAMES,
        bidir_mode="inner", mixer_impl="fused")


def rate(build, batch, compile_it=False):
    best, mem = None, 0.0
    for _ in range(REPEATS):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch._dynamo.reset()
        n = (STEPS + WARM + 1) * batch
        dl = DataLoader(TensorDataset(torch.randn(n, FRAMES, 3, IMG, IMG),
                                      torch.randn(n, FRAMES, EMB)),
                        batch_size=batch, shuffle=False, pin_memory=True, drop_last=True)
        m = build().to(DEVICE)
        if compile_it:
            m = torch.compile(m, mode="max-autotune")
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        lf, it = nn.MSELoss(), iter(dl)
        for i in range(WARM + STEPS):
            if i == WARM:
                torch.cuda.synchronize(); t0 = time.perf_counter()
            x, y = next(it)
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            if compile_it:
                torch.compiler.cudagraph_mark_step_begin()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = m(x)
                o = o.last_hidden_state if hasattr(o, "last_hidden_state") else o
                loss = lf(o.float(), y)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        r = STEPS / (time.perf_counter() - t0)
        best = r if best is None else max(best, r)
        mem = max(mem, torch.cuda.max_memory_allocated() / 1e9)
        del m, opt
        torch.cuda.empty_cache(); torch._dynamo.reset()
    return best, mem


def main():
    vh, vpar = c2.match_hidden(TARGET_PARAMS, FRAMES)
    print(f"\n  {FRAMES} frames = {FRAMES * 196} tokens.  Jamba expand=1 d_state=32 "
          f"intermediate=1424 (state 9216, ~15.50M)")
    print(f"  ViT hidden={vh} ({vpar:,}).  windows/step = batch; lanes = 2*batch*288\n",
          flush=True)
    print(f"  {'batch':>5} {'lanes':>7} {'ViT/s':>7} {'Jam/s':>7} {'ratio':>7} "
          f"{'+compile':>9} {'ratio':>7} {'J GB':>6}", flush=True)
    out = {}
    for b in BATCHES:
        try:
            v, _ = rate(lambda: c2.ViTEncoder(vh, FRAMES), b)
            j, jm = rate(enc, b)
        except torch.cuda.OutOfMemoryError:
            print(f"  {b:>5} OOM", flush=True); torch.cuda.empty_cache(); continue
        row = {"vit": v, "jamba": j, "ratio": j / v, "gb": jm}
        # Compile only where it can matter -- it costs minutes per config.
        cj = cr = None
        if b in (16, 20, 24, 32):
            try:
                cj, _ = rate(enc, b, compile_it=True)
                cv, _ = rate(lambda: c2.ViTEncoder(vh, FRAMES), b, compile_it=True)
                cr = cj / cv
                row.update({"jamba_c": cj, "vit_c": cv, "ratio_c": cr})
            except Exception as e:  # noqa: BLE001
                print(f"        compile failed: {type(e).__name__}: {str(e)[:50]}", flush=True)
                torch.cuda.empty_cache(); torch._dynamo.reset()
        print(f"  {b:>5} {2 * b * 288:>7} {v:>7.2f} {j:>7.2f} {j / v:>7.3f} "
              f"{(f'{cj:9.2f}' if cj else '        -')} "
              f"{(f'{cr:7.3f}' if cr else '      -')} {jm:>6.2f}"
              f"{'   JAMBA WINS' if (cr or j / v) > 1 else ''}", flush=True)
        out[str(b)] = row
    print("BATCH_JSON " + json.dumps({"vit_hidden": vh, "rows": out}), flush=True)


if __name__ == "__main__":
    main()
