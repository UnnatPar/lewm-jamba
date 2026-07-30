"""Alternating-direction Jamba vs a matched ViT: push batch and window length to the crossover.

Alternating the scan direction by depth reached 0.832 of the matched ViT at batch 8 and 0.892 at
batch 20, at SSM state 9216 and ~15.50M parameters -- both held. Unlike per-layer bidirectionality
its ratio IMPROVES with batch, because the mixer no longer runs at 2B, so the trend points at the
crossover instead of away from it. Backward information flow is intact (0.002654 vs 0.002511 for
per-layer bidirectional; forward-only is 0.0).

Two axes left, both free of capability cost:
  batch  -- alternating scales better than the ViT here, so find where it crosses
  frames -- more tokens means more of the ViT's quadratic term, which is the whole thesis

The ViT is re-measured at every point. 10 frames = 1,960 tokens is the target configuration; 12
is included to see the slope, not as a proposal (it spills the scan's 2,048 chunk and costs memory
the real training loop cannot spare).
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
IMG, EMB = 224, 192
HID, LAYERS = 288, 10
BEST = dict(expand=1, d_state=32, inter=1424)
TARGET_PARAMS = 15_497_922
STEPS, WARM, REPEATS = 10, 4, 3          # 3 repeats: chasing effects near the +-5% noise floor


def enc(frames, impl="alternate"):
    return mod.JambaEncoder(
        image_size=IMG, output_dim=EMB, hidden_size=HID, intermediate_size=BEST["inter"],
        num_hidden_layers=LAYERS, num_attention_heads=8, num_key_value_heads=4,
        attn_layer_period=10, attn_layer_offset=9, mamba_d_state=BEST["d_state"],
        mamba_d_conv=4, mamba_expand=BEST["expand"], max_frames=frames,
        bidir_mode="inner", mixer_impl=impl)


def rate(build, batch, frames):
    best, mem = None, 0.0
    for _ in range(REPEATS):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        n = (STEPS + WARM + 1) * batch
        dl = DataLoader(TensorDataset(torch.randn(n, frames, 3, IMG, IMG),
                                      torch.randn(n, frames, EMB)),
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
    vh, _ = c2.match_hidden(TARGET_PARAMS, 12)
    print(f"\n  alternate, expand=1 d_state=32 intermediate=1424 (state 9216, ~15.50M) "
          f"vs ViT hidden={vh}")
    print(f"  best of {REPEATS}\n", flush=True)
    out = {"vit_hidden": vh, "rows": {}}
    for frames in (10, 12):
        print(f"  === {frames} frames = {frames * 196} tokens ===", flush=True)
        print(f"  {'batch':>5} {'ViT':>7} {'altern':>7} {'ratio':>7} {'GB':>6}", flush=True)
        for b in (8, 16, 20, 24, 32, 40, 48):
            try:
                v, _ = rate(lambda: c2.ViTEncoder(vh, frames), b, frames)
                a, am = rate(lambda: enc(frames), b, frames)
            except torch.cuda.OutOfMemoryError:
                print(f"  {b:>5} OOM", flush=True); torch.cuda.empty_cache(); continue
            print(f"  {b:>5} {v:>7.2f} {a:>7.2f} {a / v:>7.3f} {am:>6.2f}"
                  f"{'   JAMBA WINS' if a > v else ''}", flush=True)
            out["rows"][f"{frames}|{b}"] = {"vit": v, "alt": a, "ratio": a / v, "gb": am}
    print("ALTB_JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
