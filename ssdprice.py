"""Would Mamba-2's SSD even be faster at OUR sequence lengths? Settle that before debating
what it costs in capability.

SSD reaches tensor cores by making the decay constant across the channels it contracts --
both A and dt become per-head instead of per-channel. That is a real capability change and
it is Unnat's call. But the change is only worth discussing if it is actually faster at
196-392 tokens, and a first measurement said it is not: 802 us against mamba-1's 282 us at
L=392, at chunk_size=128 and head dim 32, untuned.

SSD's win comes from amortising a chunked matmul over sequence length. At 196-392 there is
barely more than one chunk, so it may simply not pay here no matter how it is configured.
This sweeps the three knobs that decide it -- chunk size, head dim, and d_state -- against
the padded mamba-1 baseline that is currently the SOTA.

Head dim is the capacity dial: at d_inner=288, P=64 gives ~5 decay rates, P=16 gives 18,
against mamba-1's 4608. P=16 is the practical floor because the tensor core tile is 16.
d_state is the compensation: SSD makes N cheap, so N=64-128 buys back state size that
mamba-1 cannot afford. Nothing here is adopted -- this prices the option.
"""

import sys, time, json
import torch

sys.path.insert(0, "/content")
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

DEV, DT = "cuda", torch.bfloat16
D, B_ = 288, 32          # d_inner, and batch 16 x 2 directions
LS = [196, 256, 392, 784]
IT = 30


def timed(fn, iters=IT):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def pad_len(L):
    for c in (128, 256, 512, 1024):
        if L <= c:
            return c
    return ((L + 2047) // 2048) * 2048


def mamba1(L, n=16, pad=True):
    Lp = pad_len(L) if pad else L
    g = lambda *s: torch.randn(*s, device=DEV, dtype=DT, requires_grad=True)
    u, delta, Bm, Cm = g(B_, D, Lp), g(B_, D, Lp), g(B_, n, Lp), g(B_, n, Lp)
    A = (-torch.rand(D, n, device=DEV, dtype=torch.float32).exp()).requires_grad_(True)
    Dp = torch.ones(D, device=DEV, dtype=torch.float32, requires_grad=True)
    bias = torch.zeros(D, device=DEV, dtype=torch.float32, requires_grad=True)
    f = lambda: selective_scan_fn(u, delta, A, Bm, Cm, Dp, z=None, delta_bias=bias,
                                  delta_softplus=True)
    tf = timed(f)
    out = f()
    go = torch.randn_like(out)
    tb = timed(lambda: torch.autograd.grad(out, [u, delta, A, Bm, Cm, Dp, bias], go,
                                           retain_graph=True))
    return tf + tb


def ssd(L, P, n, chunk):
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    H = D // P
    x = torch.randn(B_, L, H, P, device=DEV, dtype=DT, requires_grad=True)
    dt = torch.randn(B_, L, H, device=DEV, dtype=DT, requires_grad=True)
    A = (-torch.rand(H, device=DEV, dtype=torch.float32).exp()).requires_grad_(True)
    Bm = torch.randn(B_, L, 1, n, device=DEV, dtype=DT, requires_grad=True)
    Cm = torch.randn(B_, L, 1, n, device=DEV, dtype=DT, requires_grad=True)
    f = lambda: mamba_chunk_scan_combined(x, dt, A, Bm, Cm, chunk_size=chunk, D=None)
    tf = timed(f)
    out = f()
    tb = timed(lambda: torch.autograd.grad(out, [x, dt, A, Bm, Cm],
                                           torch.randn_like(out), retain_graph=True))
    return tf + tb


def main():
    print(f"device {torch.cuda.get_device_name(0)}  d_inner={D} batch={B_} (16 x 2 dirs)",
          flush=True)
    out = {}
    for L in LS:
        base = mamba1(L)
        out[f"m1|{L}"] = base
        print(f"\n  L={L}  mamba-1 padded (current SOTA): {base*1e6:8.1f} us  fwd+bwd",
              flush=True)
        best = None
        for P in (64, 32, 16):
            for n in (16, 64, 128):
                for chunk in (32, 64, 128, 256):
                    if chunk > L:
                        continue
                    try:
                        t = ssd(L, P, n, chunk)
                    except Exception as e:  # noqa: BLE001
                        print(f"    P={P} N={n} chunk={chunk} FAILED "
                              f"{type(e).__name__}: {str(e)[:70]}", flush=True)
                        continue
                    out[f"ssd|{L}|{P}|{n}|{chunk}"] = t
                    if best is None or t < best[0]:
                        best = (t, P, n, chunk)
        if best:
            t, P, n, chunk = best
            rates = D // P
            print(f"    BEST SSD  P={P} (={rates} decay rates & dt values, vs mamba-1's "
                  f"{D*16}/{D})  N={n}  chunk={chunk}", flush=True)
            print(f"              {t*1e6:8.1f} us   {base/t:5.2f}x vs mamba-1   "
                  f"state {D*n:,} vs {D*16:,}", flush=True)
    print("SSD_JSON " + json.dumps(out), flush=True)


main()
