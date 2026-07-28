"""How much headroom is in the selective scan, in isolation.

profstep.py found the scan is 42-46% of Jamba's step time while being 2.5% of its MACs.
That makes the scan kernel -- not the projections, not launch overhead -- the only term
large enough to move the crossover, and it is Jamba-specific, so improving it does not have
to be handed to the ViT baseline.

This measures, at the exact shapes the mixer uses:
  1. fwd and bwd time, and achieved HBM bandwidth against the A100's ~1555 GB/s. If the
     kernel is far under peak it is latency-bound and there is headroom; if it is near peak
     the scan is done and the crossover is near its floor.
  2. whether fusing the z-gating into the kernel (which the mixer currently does as a
     separate elementwise pass) pays, and whether it is numerically identical.
  3. what a tensor-core chunked scan (Mamba-2 SSD) achieves at matched shapes. SSD shares
     one decay rate per head and so is NOT adoptable -- A must stay (d_inner, d_state).
     It is measured only to price the ceiling.
"""

import sys, time, json
import torch

sys.path.insert(0, "/content")
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import selective_scan_cuda

DEV, DT = "cuda", torch.bfloat16
PEAK_GBS = 1555.0
B, D, N, DCONV = 32, 288, 16, 4   # 32 = 2 directions x batch 16
LS = [392, 1868]
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


def inputs(L, with_z):
    g = lambda *s: torch.randn(*s, device=DEV, dtype=DT, requires_grad=True)
    u, delta = g(B, D, L), g(B, D, L)
    Bm, Cm = g(B, N, L), g(B, N, L)
    A = -torch.rand(D, N, device=DEV, dtype=torch.float32).exp()
    A.requires_grad_(True)
    Dp = torch.ones(D, device=DEV, dtype=torch.float32, requires_grad=True)
    bias = torch.zeros(D, device=DEV, dtype=torch.float32, requires_grad=True)
    z = g(B, D, L) if with_z else None
    return u, delta, A, Bm, Cm, Dp, z, bias


def bytes_moved(L, with_z):
    """u, delta, out (+z, +out_z) at 2 B; B and C are N-wide not D-wide."""
    big = 3 + (2 if with_z else 0)
    return B * L * 2 * (big * D + 2 * N)


def bench_scan(L, with_z):
    u, delta, A, Bm, Cm, Dp, z, bias = inputs(L, with_z)
    f = lambda: selective_scan_fn(u, delta, A, Bm, Cm, Dp, z=z, delta_bias=bias,
                                  delta_softplus=True)
    tf = timed(f)
    out = f()
    go = torch.randn_like(out)
    tb = timed(lambda: torch.autograd.grad(out, [u, delta, A, Bm, Cm, Dp, bias],
                                           go, retain_graph=True))
    gbs = bytes_moved(L, with_z) / tf / 1e9
    tag = "z fused " if with_z else "z separate"
    print(f"  scan {tag} L={L:<5} fwd {tf*1e6:8.1f} us  bwd {tb*1e6:8.1f} us  "
          f"bwd/fwd {tb/tf:4.2f}x  fwd {gbs:7.1f} GB/s = {100*gbs/PEAK_GBS:4.1f}% peak",
          flush=True)
    return {"fwd": tf, "bwd": tb, "gbs": gbs}


def check_z_equivalence(L):
    """(y_fwd + flip(y_bwd)) * silu(z) == y_fwd*silu(z) + flip(y_bwd*silu(z_flipped)).
    Gating is elementwise and distributes over the direction sum, so folding it into the
    kernel is algebraically free -- confirm numerically before trusting it."""
    import torch.nn.functional as F
    u, delta, A, Bm, Cm, Dp, z, bias = inputs(L, True)
    sep = selective_scan_fn(u, delta, A, Bm, Cm, Dp, z=None, delta_bias=bias,
                            delta_softplus=True) * F.silu(z)
    fused = selective_scan_fn(u, delta, A, Bm, Cm, Dp, z=z, delta_bias=bias,
                              delta_softplus=True)
    err = (sep.float() - fused.float()).abs().max().item()
    print(f"  z-fusion equivalence L={L}: max abs err {err:.3e}", flush=True)
    return err


def bench_ssd(L):
    """Ceiling only. Not adoptable: A is per-head here, not (d_inner, d_state)."""
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    except Exception as e:  # noqa: BLE001
        print(f"  ssd unavailable: {type(e).__name__}: {e}", flush=True)
        return None
    P, HG = 32, D // 32
    x = torch.randn(B, L, HG, P, device=DEV, dtype=DT, requires_grad=True)
    dt = torch.randn(B, L, HG, device=DEV, dtype=DT, requires_grad=True)
    A = -torch.rand(HG, device=DEV, dtype=torch.float32).exp().requires_grad_(True)
    Bm = torch.randn(B, L, 1, N, device=DEV, dtype=DT, requires_grad=True)
    Cm = torch.randn(B, L, 1, N, device=DEV, dtype=DT, requires_grad=True)
    f = lambda: mamba_chunk_scan_combined(x, dt, A, Bm, Cm, chunk_size=128, D=None)
    tf = timed(f)
    out = f()
    tb = timed(lambda: torch.autograd.grad(out, [x, dt, A, Bm, Cm],
                                           torch.randn_like(out), retain_graph=True))
    print(f"  SSD ceiling  L={L:<5} fwd {tf*1e6:8.1f} us  bwd {tb*1e6:8.1f} us "
          f"(NOT adoptable: one decay per head)", flush=True)
    return {"fwd": tf, "bwd": tb}


def main():
    print(f"device {torch.cuda.get_device_name(0)}  shapes B={B} D={D} N={N}", flush=True)
    out = {}
    for L in LS:
        out[f"sep|{L}"] = bench_scan(L, False)
        out[f"fused|{L}"] = bench_scan(L, True)
        out[f"zerr|{L}"] = check_z_equivalence(L)
        out[f"ssd|{L}"] = bench_ssd(L)
    print("SCAN_JSON " + json.dumps(out), flush=True)


main()
