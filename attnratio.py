"""How many of the 10 layers should be attention rather than mamba?

diag.py found that REMOVING the single attention layer makes the encoder slower (0.621 vs 0.652):
with attn_layer_period > num_hidden_layers, layer 9 becomes a 10th mamba layer, and at 1,960
tokens a Jamba attention layer is cheaper than a Jamba mamba layer. So the swap runs the other
way too -- trading mamba layers for attention layers should be faster.

It is also more faithful. Jamba's published design is a 1:7 attention:mamba ratio; this config is
1:9, which is thinner on attention than the architecture it is named after. Moving to 2 or 3
attention layers out of 10 is closer to Jamba, not further from it.

What it does trade: total recurrent state across the model drops (fewer mamba layers), while
global mixing capacity rises (more attention). That is a shift in the hybrid balance, not a
straight loss -- but it is a real change and is reported as one. Per-mamba-layer SSM state stays
9216 and total parameters stay ~15.50M, with intermediate_size re-solved for every ratio because
attention layers and mamba layers do not have the same parameter count.

The quadratic term grows with each swap, so a win here is specific to ~1,960 tokens and would
erode at longer windows. That is the honest scope of the result.
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
D_STATE, EXPAND = 32, 1
TARGET_PARAMS = 15_497_922
BATCHES = [16, 20, 24]
STEPS, WARM, REPEATS = 10, 4, 3


def enc(period, offset, inter):
    return mod.JambaEncoder(
        image_size=IMG, output_dim=EMB, hidden_size=HID, intermediate_size=inter,
        num_hidden_layers=LAYERS, num_attention_heads=8, num_key_value_heads=4,
        attn_layer_period=period, attn_layer_offset=offset, mamba_d_state=D_STATE,
        mamba_d_conv=4, mamba_expand=EXPAND, max_frames=FRAMES,
        bidir_mode="inner", mixer_impl="alternate")


def n_attn(period, offset):
    return sum(1 for i in range(LAYERS) if (i % period) == (offset % period))


def solve_inter(period, offset):
    f = lambda i: sum(p.numel() for p in enc(period, offset, i).parameters())
    i0, i1 = 512, 2048
    p0, p1 = f(i0), f(i1)
    want = i0 + (TARGET_PARAMS - p0) * (i1 - i0) / (p1 - p0)
    best = None
    for i in range(max(64, int(want) - 40), int(want) + 41, 8):
        p = f(i)
        if best is None or abs(p - TARGET_PARAMS) < abs(best[1] - TARGET_PARAMS):
            best = (i, p)
    return best


def rate(build, batch):
    best, mem = None, 0.0
    for _ in range(REPEATS):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        n = (STEPS + WARM + 1) * batch
        dl = DataLoader(TensorDataset(torch.randn(n, FRAMES, 3, IMG, IMG),
                                      torch.randn(n, FRAMES, EMB)),
                        batch_size=batch, shuffle=False, pin_memory=True, drop_last=True)
        m = build().to(DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        lf, it = nn.MSELoss(), iter(dl)
        for i in range(WARM + STEPS):
            if i == WARM:
                torch.cuda.synchronize(); t0 = time.perf_counter()
            x, y = next(it)
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
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
        torch.cuda.empty_cache()
    return best, mem


def main():
    vh, vpar = c2.match_hidden(TARGET_PARAMS, FRAMES)
    print(f"\n  {FRAMES} frames = {FRAMES * 196} tokens.  alternate, expand=1, d_state=32.")
    print(f"  ViT hidden={vh} ({vpar:,}).  best of {REPEATS}\n", flush=True)

    # (period, offset): 1, 2, 3 and 5 attention layers out of 10.
    schedules = [(10, 9), (5, 4), (4, 3), (2, 1)]
    vits = {}
    out = {"vit_hidden": vh, "rows": {}}
    print(f"  {'attn':>4} {'inter':>6} {'params':>11} {'batch':>6} {'ViT':>7} {'Jamba':>7} "
          f"{'ratio':>7} {'GB':>6}", flush=True)
    for period, offset in schedules:
        na = n_attn(period, offset)
        inter, p = solve_inter(period, offset)
        for b in BATCHES:
            if b not in vits:
                vits[b] = rate(lambda: c2.ViTEncoder(vh, FRAMES), b)[0]
            try:
                r, mem = rate(lambda: enc(period, offset, inter), b)
            except Exception as e:  # noqa: BLE001
                print(f"  {na:>4} {inter:>6} {p:>11,} {b:>6}  FAILED "
                      f"{type(e).__name__}: {str(e)[:40]}", flush=True)
                torch.cuda.empty_cache(); continue
            v = vits[b]
            print(f"  {na:>4} {inter:>6} {p:>11,} {b:>6} {v:>7.2f} {r:>7.2f} {r / v:>7.3f} "
                  f"{mem:>6.2f}{'   JAMBA WINS' if r > v else ''}", flush=True)
            out["rows"][f"a{na}|b{b}"] = {"vit": v, "jamba": r, "ratio": r / v,
                                         "inter": inter, "params": p, "gb": mem}
    print("ATTN_JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
