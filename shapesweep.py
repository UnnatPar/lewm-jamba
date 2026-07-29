"""Trade d_inner for d_state at fixed SSM state, and put the freed parameters in the MLP.

The ablation (diag.py, A100, 1,960 tokens, batch 8) established three things:

  * d_state 16 -> 1, i.e. an essentially free scan, only reaches 0.817 vs the matched ViT. No
    scan optimisation -- SSD included -- can reach parity at this length. Stop chasing the scan.
  * Halving intermediate_size buys 1.05x despite the MLP being 67% of the FLOPs. MLP width is
    nearly free: it is a dense GEMM on tensor cores.
  * expand 2->1 (1.41x) beats d_state 16->8 (1.21x) even though both cut scan FLOPs similarly.

The third point is the useful one. Scan cost splits into FLOPs ~ d_inner*d_state -- which IS the
SSM state size, so shrinking it is a real capability loss -- and STREAMING ~ d_inner alone,
which is not. Lower d_inner and raise d_state to compensate and you hold state and scan FLOPs
fixed while halving the streaming, the conv, in_proj and out_proj.

So every row here is pinned to two invariants and only the shape varies:

  SSM state  d_inner * d_state = 9216   (== today's 576 * 16)
  parameters ~= 15.50M                  (== today's, by solving intermediate_size)

`intermediate_size` is solved per row rather than guessed, because that is what makes the
comparison a shape comparison instead of a capacity comparison. Rows that break an invariant are
included and LABELLED as such, for reference only -- they are not candidates.
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
HID, LAYERS = 288, 10
TARGET_PARAMS = 15_497_922      # the expand=2 / d_state=16 / intermediate=1152 baseline
TARGET_STATE = 576 * 16         # 9216
STEPS, WARM, REPEATS = 12, 4, 2


def enc(expand, d_state, inter, impl="fused"):
    return mod.JambaEncoder(
        image_size=IMG, output_dim=EMB, hidden_size=HID, intermediate_size=inter,
        num_hidden_layers=LAYERS, num_attention_heads=8, num_key_value_heads=4,
        attn_layer_period=10, attn_layer_offset=9, mamba_d_state=d_state, mamba_d_conv=4,
        mamba_expand=expand, max_frames=FRAMES, bidir_mode="inner", mixer_impl=impl)


def nparams(expand, d_state, inter):
    return sum(p.numel() for p in enc(expand, d_state, inter).parameters())


def solve_inter(expand, d_state):
    """Parameter count is affine in intermediate_size, so two probes pin it exactly."""
    i0, i1 = 512, 2048
    p0, p1 = nparams(expand, d_state, i0), nparams(expand, d_state, i1)
    slope = (p1 - p0) / (i1 - i0)
    want = i0 + (TARGET_PARAMS - p0) / slope
    best = None
    for i in range(max(64, int(want) - 40), int(want) + 41, 8):
        p = nparams(expand, d_state, i)
        if best is None or abs(p - TARGET_PARAMS) < abs(best[1] - TARGET_PARAMS):
            best = (i, p)
    return best


def rate(build, compile_it=False):
    best = None
    for _ in range(REPEATS):
        torch.cuda.empty_cache(); torch._dynamo.reset()
        n = (STEPS + WARM + 1) * BATCH
        dl = DataLoader(TensorDataset(torch.randn(n, FRAMES, 3, IMG, IMG),
                                      torch.randn(n, FRAMES, EMB)),
                        batch_size=BATCH, shuffle=False, pin_memory=True, drop_last=True)
        m = build().to(DEVICE)
        if compile_it:
            m = torch.compile(m, mode="reduce-overhead")
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
    # (expand, d_state, note). State-preserving rows first; the rest are labelled reference.
    cands = [
        (2, 16, ""),                       # baseline shape
        (1, 32, ""),                       # half the streaming, same state and scan FLOPs
        (1, 64, "2x SSM state"),           # more capable, more scan FLOPs
        (4, 8, ""),                        # opposite direction, expect worse
        (1, 16, "HALF SSM state (ref)"),   # not a candidate: capability loss
    ]
    vit = None
    print(f"\n  {FRAMES} frames = {FRAMES * 196} tokens, batch {BATCH}, best of {REPEATS}",
          flush=True)
    print(f"  invariants: SSM state {TARGET_STATE}, params ~{TARGET_PARAMS:,}\n", flush=True)
    print(f"  {'e':>2} {'d_st':>5} {'d_in':>5} {'inter':>6} {'state':>6} {'params':>11} "
          f"{'it/s':>7} {'vs ViT':>7}  note", flush=True)
    out = {}
    for e, ds, note in cands:
        inter, p = solve_inter(e, ds)
        d_inner, state = e * HID, e * HID * ds
        if vit is None:
            vh, vpar = c2.match_hidden(TARGET_PARAMS, FRAMES)
            vit = rate(lambda: c2.ViTEncoder(vh, FRAMES))
            print(f"  {'--':>2} {'--':>5} {'--':>5} {'--':>6} {'--':>6} {vpar:>11,} "
                  f"{vit:>7.2f} {1.0:>7.3f}  ViT matched (hidden={vh})", flush=True)
        try:
            r = rate(lambda: enc(e, ds, inter))
        except Exception as ex:  # noqa: BLE001
            print(f"  {e:>2} {ds:>5} FAILED {type(ex).__name__}: {str(ex)[:50]}", flush=True)
            torch.cuda.empty_cache(); continue
        tag = note or ("state+params held" if state == TARGET_STATE else "")
        print(f"  {e:>2} {ds:>5} {d_inner:>5} {inter:>6} {state:>6} {p:>11,} {r:>7.2f} "
              f"{r / vit:>7.3f}  {tag}{'  BEATS ViT' if r > vit else ''}", flush=True)
        out[f"e{e}_ds{ds}"] = {"it_s": r, "params": p, "inter": inter, "state": state,
                              "vs_vit": r / vit}

    # Compile the best state-and-params-preserving row.
    ok = {k: v for k, v in out.items() if v["state"] >= TARGET_STATE
          and abs(v["params"] - TARGET_PARAMS) / TARGET_PARAMS < 0.02}
    if ok:
        bk = max(ok, key=lambda k: ok[k]["it_s"])
        e, ds = (int(x) for x in bk.replace("e", "").split("_ds"))
        r = rate(lambda: enc(e, ds, out[bk]["inter"]), compile_it=True)
        print(f"\n  compile({bk}): {r:>.2f} it/s  vs ViT {r / vit:.3f}"
              f"{'  BEATS ViT' if r > vit else ''}", flush=True)
        out[f"compile_{bk}"] = {"it_s": r, "vs_vit": r / vit}
    print("SHAPE_JSON " + json.dumps({"vit": vit, "rows": out}), flush=True)


if __name__ == "__main__":
    main()
