"""Does the inner-bidirectional mixer actually compute bidirectional attention, and is it faster?

Must run on CUDA: the mamba kernels bidir.py rests on do not exist on CPU, so test_multiframe.py
cannot reach any of this. Correctness gates speed -- if any check fails the timings are not
printed at all, because a fast encoder that quietly became unidirectional is the exact failure
this is guarding against and a number beside it would invite quoting it.

The checks, strongest first:

1. REVERSAL EQUIVARIANCE, y(flip(x)) == flip(y(x)). Analytically exact for this construction:
   y = out_proj(S(x) + flip(S(flip(x)))), so y(flip(x)) = out_proj(S(flip x) + flip(S(x))) which
   is flip(y(x)) term for term. Any error in which tensor gets flipped, or a conv applied to the
   wrong direction, breaks it. This is the test that catches the conv(flip) != flip(conv) trap.

2. BATCHING IS A NO-OP. The two directions are concatenated into one set of kernel launches.
   Compared against an obviously-correct reference that just calls the scan branch twice.

3. GENUINELY BIDIRECTIONAL. Perturb the LAST token, require the FIRST position's output to move.
   A unidirectional encoder passes 1 and 2 and fails this.

4. Parameter count identical to the outer scheme -- the speedup must not come from a smaller
   model.
"""

import sys

import torch

sys.path.insert(0, "/content")

import bidir
import module as mod
import sscanop2 as co

DEVICE = "cuda"
IMG, EMB, HID = 224, 192, 288
PASS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}", flush=True)
    PASS.append(ok)
    return ok


def build(mode, max_frames=10, expand=1):
    return mod.JambaEncoder(
        image_size=IMG, output_dim=EMB, hidden_size=HID, intermediate_size=1152,
        num_hidden_layers=10, num_attention_heads=8, num_key_value_heads=4,
        attn_layer_period=10, attn_layer_offset=9, mamba_d_state=16, mamba_d_conv=4,
        mamba_expand=expand, max_frames=max_frames, bidir_mode=mode).to(DEVICE)


def ref_forward(mixer, h):
    """Inner bidirectionality written the obvious way: two separate scan-branch calls."""
    projected = mixer.in_proj(h).transpose(1, 2)
    x, gate = projected.chunk(2, dim=1)
    f = bidir._scan_both(mixer, x.contiguous(), gate.contiguous())
    b = bidir._scan_both(mixer, x.flip(-1).contiguous(), gate.flip(-1).contiguous())
    return mixer.out_proj((f + b.flip(-1)).transpose(1, 2))


def main():
    torch.manual_seed(0)
    orig, shim = co.register()
    co.PAD_SCAN = True
    if not co.verify(orig, shim):
        print("STOPPING: scan padding equivalence failed"); return 1

    enc = build("inner")
    mixer = next(m for m in enc.jamba.modules()
                 if type(m).__name__ == "JambaMambaMixer")
    h = torch.randn(2, 392, HID, device=DEVICE, dtype=torch.bfloat16)

    print("\n1. reversal equivariance of the mixer:  y(flip(x)) == flip(y(x))")
    with torch.no_grad():
        y = mixer.cuda_kernels_forward(h)
        y_flip = mixer.cuda_kernels_forward(h.flip(1))
    d = (y_flip - y.flip(1)).abs().max().item()
    scale = y.abs().max().item()
    check("exact under sequence reversal", d <= 2e-2 * scale, f"max|diff|={d:.3e} (scale {scale:.3f})")

    print("\n2. batching the two directions is a no-op")
    with torch.no_grad():
        d2 = (mixer.cuda_kernels_forward(h) - ref_forward(mixer, h)).abs().max().item()
    check("batched == two separate scan calls", d2 <= 1e-2 * scale, f"max|diff|={d2:.3e}")

    # An absolute threshold here is meaningless: `outer` sums two full residual streams so its
    # outputs are inherently larger, and at init the scan contributes little next to the
    # residual. What matters is whether information moves BACKWARD about as well as it moves
    # FORWARD within the same model -- a ratio, which no scale difference can fake.
    # Coupling vs DISTANCE, in fp32. A single SSM's influence decays with distance by
    # construction -- at random init only the slowest channel (A ~ -1, dt ~ 0.01) survives
    # 392 steps, at roughly exp(-0.01*392) ~ 2%, which bf16 rounds to zero at this output
    # scale. So a null at long range says nothing. What proves bidirectionality is SYMMETRY at
    # short range: backward coupling must match forward coupling where neither has decayed.
    # A unidirectional mixer gives back == 0 at every distance including the shortest.
    # bidir applies the gate outside the scan so sscanop2's padding path is reachable at all.
    # That is only legitimate if it is exactly the same arithmetic the kernel does internally.
    print("\n2b. gating outside the scan == gating fused inside it")
    with torch.no_grad():
        proj = mixer.in_proj(h).transpose(1, 2)
        xx, gg = proj.chunk(2, dim=1)
        xx, gg = xx.contiguous(), gg.contiguous()
        manual = bidir._scan_both(mixer, xx, gg)
        # The unmodified upstream call: gate fused in, last state returned.
        import bidir as _b
        cw = mixer.conv1d.weight.view(mixer.conv1d.weight.size(0), mixer.conv1d.weight.size(2))
        xc = _b.mj.causal_conv1d_fn(xx, cw, mixer.conv1d.bias, activation=mixer.activation)
        sp = mixer.x_proj(xc.transpose(1, 2))
        ts, Bm, Cm = torch.split(
            sp, [mixer.time_step_rank, mixer.ssm_state_size, mixer.ssm_state_size], dim=-1)
        dts = torch.nn.functional.linear(
            mixer.dt_layernorm(ts), mixer.dt_proj.weight).transpose(1, 2)
        fused, _ = _b.mj.selective_scan_fn(
            xc, dts, -torch.exp(mixer.A_log.float()),
            mixer.b_layernorm(Bm).transpose(1, 2), mixer.c_layernorm(Cm).transpose(1, 2),
            mixer.D.float(), gg, mixer.dt_proj.bias.float(),
            delta_softplus=True, return_last_state=True)
    dg = (manual - fused).abs().max().item()
    check("manual silu gate == fused gate", dg <= 1e-2 * max(fused.abs().max().item(), 1e-9),
          f"max|diff|={dg:.3e}")

    print("\n3. mixer: coupling vs distance, fp32 (decay with distance is expected)")
    import copy
    m32 = copy.deepcopy(mixer).float()
    print(f"    {'L':>5} {'backward':>11} {'forward':>11} {'back/fwd':>9}", flush=True)
    short_ok, sym = None, []
    for L in (8, 32, 128, 392):
        h = torch.randn(1, L, HID, device=DEVICE, dtype=torch.float32)
        with torch.no_grad():
            y = m32.cuda_kernels_forward(h)
            hb = h.clone(); hb[0, -1] += 3.0      # perturb LAST  -> look at position 0
            back = (m32.cuda_kernels_forward(hb) - y).abs()[0, 0].max().item()
            hf = h.clone(); hf[0, 0] += 3.0       # perturb FIRST -> look at position L-1
            fwd = (m32.cuda_kernels_forward(hf) - y).abs()[0, -1].max().item()
        r = back / max(fwd, 1e-30)
        print(f"    {L:>5} {back:>11.3e} {fwd:>11.3e} {r:>9.3f}", flush=True)
        if L == 8:
            short_ok = (back > 0, r)
        # The invariant that actually distinguishes a bug from decay: the two directions must
        # die together. Measured, both hit exactly 0 at L>=128 -- no single mixer at random
        # init reaches 128 tokens in EITHER direction. Requiring backward coupling where
        # forward has none would be testing something false.
        sym.append(((back > 0) == (fwd > 0)) and (not fwd > 0 or 0.05 < r < 20))
    check("backward coupling exists at short range", short_ok[0], "L=8")
    check("symmetric at short range (back ~ fwd)", 0.1 < short_ok[1] < 10,
          f"ratio={short_ok[1]:.3f}")
    check("directions decay together at every distance", all(sym),
          f"{sum(sym)}/{len(sym)} distances symmetric")
    del m32

    print("\n4. encoder end to end: frame 0 must react to frame 1")
    for mode in ("inner", "outer"):
        e = build(mode)
        px = torch.randn(1, 2, 3, IMG, IMG, device=DEVICE)
        with torch.no_grad():
            base = e(px).last_hidden_state
            bumped = px.clone()
            bumped[0, 1] += 5.0                  # perturb the LAST frame only
            moved = (e(bumped).last_hidden_state - base).abs()
        far, near = moved[0, 0].max().item(), moved[0, 1].max().item()
        # Relative to the direct effect on the frame actually perturbed. The bar is low on
        # purpose: `outer` couples strongly at init because the reversed stack carries frame 1
        # in its RESIDUAL stream, while `inner` couples only through the scan output, which is
        # small until trained. That gap is an initialisation property, not a correctness one --
        # what has to hold is that the path exists.
        check(f"{mode}: frame 0 sees frame 1", far / max(near, 1e-12) > 1e-5,
              f"far={far:.3e} near={near:.3e} ratio={far / max(near, 1e-12):.5f}")
        del e

    print("\n5. parameter counts identical")
    pi = sum(p.numel() for p in build("inner").parameters())
    po = sum(p.numel() for p in build("outer").parameters())
    check("inner and outer have the same parameters", pi == po, f"{pi:,} vs {po:,}")

    if not all(PASS):
        print("\nCORRECTNESS FAILED -- not reporting speed.\n")
        return 1

    print("\n6. speed: inner vs outer vs the matched ViT (batch 8, real training step)")
    import crossover2 as c2
    for expand in (1, 2):
        jp = sum(p.numel() for p in build("outer", 12, expand).parameters())
        vh, vp = c2.match_hidden(jp, 12)
        print(f"\n  expand={expand}: Jamba {jp:,}  ViT(hidden={vh}) {vp:,}", flush=True)
        print(f"  {'frames':>6} {'tokens':>7} {'ViT':>8} {'outer':>8} {'inner':>8} "
              f"{'in/out':>7} {'vs ViT':>8}", flush=True)
        for f in (8, 10, 12):
            try:
                v, _ = c2.rate(lambda: c2.ViTEncoder(vh, f), f, 8)
                o, _ = c2.rate(lambda: build("outer", f, expand), f, 8)
                i, _ = c2.rate(lambda: build("inner", f, expand), f, 8)
            except torch.cuda.OutOfMemoryError:
                print(f"  {f:>6} OOM", flush=True); torch.cuda.empty_cache(); continue
            print(f"  {f:>6} {f * 196:>7} {v:>8.2f} {o:>8.2f} {i:>8.2f} {i / o:>7.2f}x "
                  f"{i / v:>7.3f}{'  JAMBA WINS' if i > v else ''}", flush=True)
    print("\nALL CORRECTNESS PASSED\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
