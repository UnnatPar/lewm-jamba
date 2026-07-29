"""Where does the remaining time actually go? Ablate one component at a time.

Two FLOP-based predictions this session were badly wrong (1.9x predicted / 1.17x measured, and
a "crossover at 1,723" that was really "far past 2,352"), so stop predicting and measure the
decomposition directly.

Method: take the fused inner-bidirectional encoder at the real config and knock out one
component at a time. The drop in step time is that component's share. Not additive in general,
but good enough to rank what is worth attacking.

  d_state 16 -> 8 -> 4 -> 1   : the scan's share (scan FLOPs are d_inner*d_state*L exactly)
  intermediate 1152 -> 576    : the MLP's share
  attn period 10 -> 100       : the single attention layer's share (quadratic in L)
  compile on                  : how much is left for the compiler to fuse now the mixer is
                                one kernel -- a different question from the earlier
                                whole-model compile, which favoured the ViT

Everything at 1,960 tokens, batch 8, real training step. The ViT baseline is printed as the
target to beat, not as an ablation.

Nothing here is a proposed config -- d_state 1 is not a model, it is a probe. Reductions in
d_state shrink SSM state capacity and are ruled out as real options for that reason.
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
IMG, EMB, FRAMES, BATCH = 224, 192, 10, 8
STEPS, WARM, REPEATS = 12, 4, 2


def enc(**over):
    kw = dict(image_size=IMG, output_dim=EMB, hidden_size=288, intermediate_size=1152,
              num_hidden_layers=10, num_attention_heads=8, num_key_value_heads=4,
              attn_layer_period=10, attn_layer_offset=9, mamba_d_state=16, mamba_d_conv=4,
              mamba_expand=2, max_frames=FRAMES, bidir_mode="inner", mixer_impl="fused")
    kw.update(over)
    return mod.JambaEncoder(**kw)


def rate(build, compile_it=False):
    best, params = None, None
    for _ in range(REPEATS):
        torch.cuda.empty_cache(); torch._dynamo.reset()
        n = (STEPS + WARM + 1) * BATCH
        dl = DataLoader(TensorDataset(torch.randn(n, FRAMES, 3, IMG, IMG),
                                      torch.randn(n, FRAMES, EMB)),
                        batch_size=BATCH, shuffle=False, pin_memory=True, drop_last=True)
        m = build().to(DEVICE)
        params = sum(p.numel() for p in m.parameters())
        if compile_it:
            m = torch.compile(m, mode="reduce-overhead")
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        lf = nn.MSELoss()
        it = iter(dl)
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
        del m, opt
        torch.cuda.empty_cache(); torch._dynamo.reset()
    return best, params


def main():
    print(f"\n  {FRAMES} frames = {FRAMES * 196} tokens, batch {BATCH}, "
          f"best of {REPEATS}\n", flush=True)

    base, bp = rate(lambda: enc())
    vp_target = bp
    vh, vpar = c2.match_hidden(vp_target, FRAMES)
    vit, _ = rate(lambda: c2.ViTEncoder(vh, FRAMES))
    print(f"  {'variant':<34} {'params':>11} {'it/s':>8} {'vs base':>8} {'vs ViT':>8}",
          flush=True)
    print(f"  {'ViT matched (target to beat)':<34} {vpar:>11,} {vit:>8.2f} "
          f"{'-':>8} {1.0:>8.3f}", flush=True)
    print(f"  {'fused baseline':<34} {bp:>11,} {base:>8.2f} {1.0:>8.2f} "
          f"{base / vit:>8.3f}", flush=True)

    out = {"vit": vit, "vit_hidden": vh, "vit_params": vpar, "base": base, "base_params": bp}
    ablations = [
        ("d_state 16->8  (probe only)", dict(mamba_d_state=8), False),
        ("d_state 16->4  (probe only)", dict(mamba_d_state=4), False),
        ("d_state 16->1  (probe only)", dict(mamba_d_state=1), False),
        ("intermediate 1152->576", dict(intermediate_size=576), False),
        ("no attention layer", dict(attn_layer_period=100, attn_layer_offset=99), False),
        ("expand 2->1", dict(mamba_expand=1), False),
        ("unfused mixer", dict(mixer_impl="unfused"), False),
        ("compile(fused baseline)", dict(), True),
    ]
    for label, over, comp in ablations:
        try:
            r, p = rate(lambda o=over: enc(**o), compile_it=comp)
        except Exception as e:  # noqa: BLE001
            print(f"  {label:<34} FAILED {type(e).__name__}: {str(e)[:60]}", flush=True)
            torch.cuda.empty_cache(); torch._dynamo.reset()
            continue
        print(f"  {label:<34} {p:>11,} {r:>8.2f} {r / base:>8.2f} {r / vit:>8.3f}"
              f"{'   BEATS ViT' if r > vit else ''}", flush=True)
        out[label] = {"it_s": r, "params": p}

    print("\n  Read the 'vs base' column as an upper bound on what removing that component "
          "could ever buy.", flush=True)
    print("DIAG_JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
