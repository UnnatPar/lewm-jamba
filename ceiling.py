"""What is the BEST Jamba could possibly do at 1,960 tokens? Bound it before optimising further.

Optimisation has gone 0.430 -> 0.705 vs a parameter-matched ViT (fused mixer, inner
bidirectionality, d_inner traded for d_state at fixed SSM state, compile). Each further step is
getting smaller. Before spending more, bound the target: if Jamba cannot reach 1.0 even with its
scan deleted, then no scan work -- SSD, a Triton rewrite, kernel tuning -- can get there, and the
honest answer is that a 1,960-token crossover against a matched ViT is not reachable in this
architecture family.

Three probes, each a strictly-unphysical upper bound rather than a proposed config:

  d_state -> 1        scan FLOPs ~ 0, but the kernel still streams u/delta/out
  scan deleted        mixer becomes in_proj -> out_proj. No conv, no scan, no gate. Bounds
                      EVERYTHING mamba-shaped at once.
  mixer deleted       the mamba layers become their MLP plus a residual. Bounds the whole
                      mixer including its projections, leaving patch embed + MLPs + the one
                      attention layer. If THIS loses to the ViT, the problem was never mamba.

Plus two real options: dropping the single attention layer, and max-autotune compile.
"""

import json
import sys
import time
import types

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/content")

import module as mod
import crossover2 as c2
from transformers.models.jamba import modeling_jamba as mj

DEVICE = "cuda"
IMG, EMB, FRAMES, BATCH = 224, 192, 10, 8
HID, LAYERS = 288, 10
BEST = dict(expand=1, d_state=32, inter=1424)   # from shapesweep: state 9216, params 15.50M
TARGET_PARAMS = 15_497_922
STEPS, WARM, REPEATS = 12, 4, 2


def enc(expand=None, d_state=None, inter=None, attn_period=10, attn_offset=9, impl="fused"):
    return mod.JambaEncoder(
        image_size=IMG, output_dim=EMB, hidden_size=HID,
        intermediate_size=inter or BEST["inter"], num_hidden_layers=LAYERS,
        num_attention_heads=8, num_key_value_heads=4, attn_layer_period=attn_period,
        attn_layer_offset=attn_offset, mamba_d_state=d_state or BEST["d_state"],
        mamba_d_conv=4, mamba_expand=expand or BEST["expand"], max_frames=FRAMES,
        bidir_mode="inner", mixer_impl=impl)


def _proj_only(self, hidden_states, cache_params=None, attention_mask=None, **kw):
    """Upper bound: keep in_proj/out_proj, delete conv + scan + gate entirely."""
    x, _ = self.in_proj(hidden_states).transpose(1, 2).chunk(2, dim=1)
    return self.out_proj(x.transpose(1, 2))


def _identity(self, hidden_states, cache_params=None, attention_mask=None, **kw):
    """Upper bound: delete the mixer. The layer keeps only its MLP and residuals."""
    return torch.zeros_like(hidden_states)


def strip(model, fn):
    for m in model.modules():
        if isinstance(m, mj.JambaMambaMixer):
            m.cuda_kernels_forward = types.MethodType(fn, m)
    return model


def rate(build, compile_it=False, mode="reduce-overhead"):
    best = None
    for _ in range(REPEATS):
        torch.cuda.empty_cache(); torch._dynamo.reset()
        n = (STEPS + WARM + 1) * BATCH
        dl = DataLoader(TensorDataset(torch.randn(n, FRAMES, 3, IMG, IMG),
                                      torch.randn(n, FRAMES, EMB)),
                        batch_size=BATCH, shuffle=False, pin_memory=True, drop_last=True)
        m = build().to(DEVICE)
        if compile_it:
            m = torch.compile(m, mode=mode)
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
        del m, opt
        torch.cuda.empty_cache(); torch._dynamo.reset()
    return best


def main():
    vh, vpar = c2.match_hidden(TARGET_PARAMS, FRAMES)
    vit = rate(lambda: c2.ViTEncoder(vh, FRAMES))
    print(f"\n  {FRAMES} frames = {FRAMES * 196} tokens, batch {BATCH}, best of {REPEATS}")
    print(f"  shape: expand={BEST['expand']} d_state={BEST['d_state']} "
          f"intermediate={BEST['inter']} (SSM state 9216, params ~15.50M)\n", flush=True)
    print(f"  {'probe':<38} {'it/s':>7} {'vs ViT':>7}  kind", flush=True)
    print(f"  {'ViT matched (hidden=' + str(vh) + ')':<38} {vit:>7.2f} {1.0:>7.3f}  target",
          flush=True)

    rows = [
        ("best real config", lambda: enc(), False, "real", "reduce-overhead"),
        ("  + compile", lambda: enc(), True, "real", "reduce-overhead"),
        ("  + compile max-autotune", lambda: enc(), True, "real", "max-autotune"),
        ("no attention layer", lambda: enc(attn_period=99, attn_offset=98), False, "real",
         "reduce-overhead"),
        ("d_state -> 1", lambda: enc(d_state=1), False, "BOUND", "reduce-overhead"),
        ("scan deleted (proj only)", lambda: strip(enc(), _proj_only), False, "BOUND",
         "reduce-overhead"),
        ("mixer deleted entirely", lambda: strip(enc(), _identity), False, "BOUND",
         "reduce-overhead"),
        ("mixer deleted + compile", lambda: strip(enc(), _identity), True, "BOUND",
         "reduce-overhead"),
    ]
    out = {"vit": vit, "vit_hidden": vh}
    for label, build, comp, kind, cmode in rows:
        try:
            r = rate(build, compile_it=comp, mode=cmode)
        except Exception as e:  # noqa: BLE001
            print(f"  {label:<38} FAILED {type(e).__name__}: {str(e)[:50]}", flush=True)
            torch.cuda.empty_cache(); torch._dynamo.reset(); continue
        print(f"  {label:<38} {r:>7.2f} {r / vit:>7.3f}  {kind}"
              f"{'   BEATS ViT' if r > vit else ''}", flush=True)
        out[label.strip()] = {"it_s": r, "vs_vit": r / vit, "kind": kind}
    print("\n  A BOUND row is not a model. If the best BOUND still loses, no amount of scan or "
          "mixer work reaches parity at this length.", flush=True)
    print("CEIL_JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
