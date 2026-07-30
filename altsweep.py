"""Alternate the scan direction by depth instead of doing both directions in every layer.

Everything else is exhausted. Measured on an A100 at 1,960 tokens with SSM state (9216) and
parameters (~15.50M) both held fixed: fused inner-bidirectional reaches 0.643 of a matched ViT at
batch 8 and 0.598 at the real training batch of 20 windows. The shape trade, the fused kernel and
the padding path are all in. compile makes the RATIO worse once the ViT is compiled too. Larger
batch makes it worse. A free scan would reach 1.258, so the headroom is entirely in the mixer's
fixed cost -- which is ~49% of step time and is paid TWICE per layer for bidirectionality.

So stop paying it twice. Even layers scan forward, odd layers backward: the stack stays
bidirectional through depth, no layer pays for two scans, and no parameter changes.

Correctness matters more than speed here, so this also checks that the alternating stack really
does propagate information both ways -- an encoder whose first frame cannot see its last frame is
useless for this task no matter how fast it is. forward_only is included as the bound and as the
negative control: it MUST fail the backward-propagation check, and if it passes, the check is
broken rather than the model being good.
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
BEST = dict(expand=1, d_state=32, inter=1424)
TARGET_PARAMS = 15_497_922
STEPS, WARM, REPEATS = 10, 4, 2


def enc(impl):
    return mod.JambaEncoder(
        image_size=IMG, output_dim=EMB, hidden_size=HID, intermediate_size=BEST["inter"],
        num_hidden_layers=LAYERS, num_attention_heads=8, num_key_value_heads=4,
        attn_layer_period=10, attn_layer_offset=9, mamba_d_state=BEST["d_state"],
        mamba_d_conv=4, mamba_expand=BEST["expand"], max_frames=FRAMES,
        bidir_mode="inner", mixer_impl=impl)


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


def backward_flow(impl):
    """Relative change at frame 0 when only the LAST frame is perturbed.

    Deliberately measured on a TINY encoder (32px -> 4 tokens/frame, 2 frames = 8 tokens), not
    the real one. A single mixer at random init has no coupling past ~128 tokens in either
    direction, so probing across a 1,764-token gap in the real encoder returns 0.0 for every
    variant including ones known to be bidirectional -- that measures SSM decay, not whether a
    path exists. At 8 tokens nothing has decayed and the structural question is the only one
    left. Depth, layer count and direction schedule are all preserved.
    """
    e = mod.JambaEncoder(
        image_size=32, patch_size=16, output_dim=EMB, hidden_size=HID,
        intermediate_size=BEST["inter"], num_hidden_layers=LAYERS, num_attention_heads=8,
        num_key_value_heads=4, attn_layer_period=10, attn_layer_offset=9,
        mamba_d_state=BEST["d_state"], mamba_d_conv=4, mamba_expand=BEST["expand"],
        max_frames=2, bidir_mode="inner", mixer_impl=impl).to(DEVICE)
    px = torch.randn(1, 2, 3, 32, 32, device=DEVICE)
    with torch.no_grad():
        base = e(px).last_hidden_state
        b = px.clone(); b[0, -1] += 5.0
        moved = (e(b).last_hidden_state - base).abs()
    far, near = moved[0, 0].max().item(), moved[0, -1].max().item()
    del e
    torch.cuda.empty_cache()
    return far / max(near, 1e-12)


def main():
    vh, vpar = c2.match_hidden(TARGET_PARAMS, FRAMES)
    print(f"\n  {FRAMES} frames = {FRAMES * 196} tokens.  expand=1 d_state=32 "
          f"intermediate=1424 (state 9216, ~15.50M).  ViT hidden={vh}\n", flush=True)

    print("  backward information flow (frame 0's response to perturbing the LAST frame):",
          flush=True)
    flows = {}
    for impl in ("fused", "alternate", "forward_only"):
        flows[impl] = backward_flow(impl)
        print(f"    {impl:<14} {flows[impl]:.6f}", flush=True)
    ok = flows["alternate"] > 1e-5
    ctl = flows["forward_only"] < flows["alternate"] / 10
    print(f"    [{'PASS' if ok else 'FAIL'}] alternate propagates backward", flush=True)
    print(f"    [{'PASS' if ctl else 'FAIL'}] forward_only does not (negative control)",
          flush=True)
    if not (ok and ctl):
        print("\n  Correctness failed -- not reporting speed.\n", flush=True)
        return

    print(f"\n  {'batch':>5} {'ViT':>7} {'fused':>7} {'altern':>7} {'fwdonly':>8} "
          f"{'alt/fus':>8} {'alt vs ViT':>10}", flush=True)
    out = {"vit_hidden": vh, "flows": flows, "rows": {}}
    for b in (8, 20):
        v, _ = rate(lambda: c2.ViTEncoder(vh, FRAMES), b)
        f, _ = rate(lambda: enc("fused"), b)
        a, am = rate(lambda: enc("alternate"), b)
        fo, _ = rate(lambda: enc("forward_only"), b)
        print(f"  {b:>5} {v:>7.2f} {f:>7.2f} {a:>7.2f} {fo:>8.2f} {a / f:>7.2f}x "
              f"{a / v:>10.3f}{'   JAMBA WINS' if a > v else ''}", flush=True)
        out["rows"][str(b)] = {"vit": v, "fused": f, "alternate": a, "forward_only": fo,
                              "alt_vs_vit": a / v, "gb": am}
    print("ALT_JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
