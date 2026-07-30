"""SSD mixer vs the Mamba-1 alternating mixer vs the matched ViT. Correctness gates speed.

Checks, in order:
  1. the swap actually happened (a silent zero would benchmark the wrong model)
  2. backward information flow through the alternating stack, on a TINY encoder -- probing the
     real 1,764-token gap returns 0.0 for every variant because a mixer at random init couples
     over ~128 tokens at most, which measures decay rather than connectivity
  3. forward-only SSD as the negative control: it MUST show zero backward flow
  4. trains 12 steps without NaN
Then speed at 1,960 tokens across batch, with parameters re-solved to ~15.50M and SSM state held
at 288*32 = 9216.
"""

import json
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/content")

import crossover2 as c2
import module as mod
import ssdmixer

DEVICE = "cuda"
IMG, EMB, FRAMES = 224, 192, 10
HID, LAYERS = 288, 10
TARGET_PARAMS = 15_497_922
SSD_KW = dict(expand=1, d_state=32, headdim=48, chunk_size=256)
STEPS, WARM, REPEATS = 10, 4, 3
PASS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}", flush=True)
    PASS.append(ok)


def build(inter, frames=FRAMES, img=IMG, ssd=True, alternate=True):
    e = mod.JambaEncoder(
        image_size=img, patch_size=16, output_dim=EMB, hidden_size=HID,
        intermediate_size=inter, num_hidden_layers=LAYERS, num_attention_heads=8,
        num_key_value_heads=4, attn_layer_period=10, attn_layer_offset=9,
        mamba_d_state=32, mamba_d_conv=4, mamba_expand=1, max_frames=frames,
        bidir_mode="inner", mixer_impl="alternate")
    if ssd:
        n = ssdmixer.swap_into(e.jamba, HID, alternate=alternate, **SSD_KW)
        assert n == LAYERS - 1, f"swapped {n}, expected {LAYERS - 1}"
    return e


def solve_inter(ssd=True):
    f = lambda i: sum(p.numel() for p in build(i, ssd=ssd).parameters())
    i0, i1 = 512, 2048
    p0, p1 = f(i0), f(i1)
    want = i0 + (TARGET_PARAMS - p0) * (i1 - i0) / (p1 - p0)
    best = None
    for i in range(max(64, int(want) - 48), int(want) + 49, 8):
        p = f(i)
        if best is None or abs(p - TARGET_PARAMS) < abs(best[1] - TARGET_PARAMS):
            best = (i, p)
    return best


def flow(inter, alternate=True):
    e = build(inter, frames=2, img=32, ssd=True, alternate=alternate).to(DEVICE)
    px = torch.randn(1, 2, 3, 32, 32, device=DEVICE)
    with torch.no_grad():
        base = e(px).last_hidden_state
        b = px.clone(); b[0, -1] += 5.0
        moved = (e(b).last_hidden_state - base).abs()
    r = moved[0, 0].max().item() / max(moved[0, -1].max().item(), 1e-12)
    del e; torch.cuda.empty_cache()
    return r


def rate(bld, batch):
    best, mem = None, 0.0
    for _ in range(REPEATS):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        n = (STEPS + WARM + 1) * batch
        dl = DataLoader(TensorDataset(torch.randn(n, FRAMES, 3, IMG, IMG),
                                      torch.randn(n, FRAMES, EMB)),
                        batch_size=batch, shuffle=False, pin_memory=True, drop_last=True)
        m = bld().to(DEVICE)
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
    si, sp = solve_inter(ssd=True)
    mi, mp = solve_inter(ssd=False)
    e = build(si)
    m0 = e.jamba.layers[0].mamba
    print(f"\n  SSD: nheads={m0.nheads} headdim={m0.headdim} d_state={m0.d_state} "
          f"SSM state={m0.d_inner * m0.d_state}  A params={m0.A_log.numel()} "
          f"(Mamba-1 had {288 * 32})")
    print(f"  intermediate: SSD {si} ({sp:,})   Mamba-1 {mi} ({mp:,})\n", flush=True)
    check("mixers swapped", isinstance(m0, ssdmixer.SSDMixer), type(m0).__name__)
    del e

    fa, ff = flow(si, alternate=True), flow(si, alternate=False)
    check("SSD alternating propagates backward", fa > 1e-5, f"{fa:.6f}")
    check("SSD forward-only does not (control)", ff < fa / 10, f"{ff:.6f}")

    enc = build(si).to(DEVICE)
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-4)
    bad, l0, ln = [], None, None
    for i in range(12):
        px = torch.randn(2, 4, 3, IMG, IMG, device=DEVICE)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = enc(px).last_hidden_state.float().pow(2).mean()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step()
        l0 = l0 if l0 is not None else loss.item()
        ln = loss.item()
        if not (torch.isfinite(loss) and torch.isfinite(gn)):
            bad.append(i)
    check("12 steps finite", not bad, f"loss {l0:.4f} -> {ln:.4f}")
    del enc, opt
    torch.cuda.empty_cache()

    if not all(PASS):
        print("\n  CORRECTNESS FAILED -- not reporting speed.\n"); return

    vh, _ = c2.match_hidden(TARGET_PARAMS, FRAMES)
    print(f"\n  {FRAMES} frames = {FRAMES * 196} tokens.  ViT hidden={vh}.  best of {REPEATS}\n")
    print(f"  {'batch':>5} {'ViT':>7} {'mamba1':>7} {'SSD':>7} {'ssd/m1':>7} "
          f"{'SSD vs ViT':>10} {'GB':>6}", flush=True)
    out = {"ssd_inter": si, "m1_inter": mi, "flows": {"alt": fa, "fwd": ff}, "rows": {}}
    for b in (8, 16, 20, 24, 32):
        try:
            v, _ = rate(lambda: c2.ViTEncoder(vh, FRAMES), b)
            m1, _ = rate(lambda: build(mi, ssd=False), b)
            s, sm = rate(lambda: build(si, ssd=True), b)
        except torch.cuda.OutOfMemoryError:
            print(f"  {b:>5} OOM", flush=True); torch.cuda.empty_cache(); continue
        print(f"  {b:>5} {v:>7.2f} {m1:>7.2f} {s:>7.2f} {s / m1:>6.2f}x {s / v:>10.3f} "
              f"{sm:>6.2f}{'   JAMBA WINS' if s > v else ''}", flush=True)
        out["rows"][str(b)] = {"vit": v, "mamba1": m1, "ssd": s, "ratio": s / v, "gb": sm}
    print("SSD_JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
