"""Is the selective scan parallelism-starved, or is the kernel just slow?

scanbench.py found it running at 5-12% of the A100's HBM roofline -- latency-bound, not
bandwidth-bound. That leaves exactly two explanations, and they imply opposite next moves:

  (a) not enough concurrent work to fill the GPU  -> a chunked parallel scan wins, because
      splitting L into K chunks multiplies the block count by K. The recurrence is linear
      with diagonal A, so chunking is exact and costs no capacity.
  (b) the kernel is slow per unit work regardless -> chunking buys nothing and the crossover
      is near its floor without writing a new kernel.

The discriminator is cheap: run the SAME total work as (B*K, D, L/K) and see whether it gets
faster. Splitting the sequence like this is numerically WRONG (it drops the cross-chunk
state) -- that is fine here, the point is only to price the parallelism, and the real version
would add the state-passing correction. Nothing in this file is a proposed model change.

Also sweeps batch, because "does the ranking hold at real training batch size" is a standing
open question and the answer follows from the same curve.
"""

import sys, time, json
import torch

sys.path.insert(0, "/content")
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

DEV, DT = "cuda", torch.bfloat16
D, N = 288, 16
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


def scan_time(b, L, want_bwd=True):
    g = lambda *s: torch.randn(*s, device=DEV, dtype=DT, requires_grad=True)
    u, delta, Bm, Cm = g(b, D, L), g(b, D, L), g(b, N, L), g(b, N, L)
    A = (-torch.rand(D, N, device=DEV, dtype=torch.float32).exp()).requires_grad_(True)
    Dp = torch.ones(D, device=DEV, dtype=torch.float32, requires_grad=True)
    bias = torch.zeros(D, device=DEV, dtype=torch.float32, requires_grad=True)
    f = lambda: selective_scan_fn(u, delta, A, Bm, Cm, Dp, z=None, delta_bias=bias,
                                  delta_softplus=True)
    tf = timed(f)
    if not want_bwd:
        return tf, 0.0
    out = f()
    go = torch.randn_like(out)
    tb = timed(lambda: torch.autograd.grad(out, [u, delta, A, Bm, Cm, Dp, bias], go,
                                           retain_graph=True))
    return tf, tb


def main():
    print(f"device {torch.cuda.get_device_name(0)}  D={D} N={N}", flush=True)
    out = {}

    for L in (392, 1868):
        print(f"\n--- chunking: same total work, L={L}, batch 16 x 2 directions ---", flush=True)
        base = None
        for K in (1, 2, 4, 8, 16):
            if L % K:
                continue
            b, l = 32 * K, L // K
            tf, tb = scan_time(b, l)
            tot = tf + tb
            base = base or tot
            print(f"  K={K:<3} shape=({b},{D},{l})  fwd {tf*1e6:8.1f} us  bwd {tb*1e6:8.1f} us"
                  f"  total {tot*1e6:8.1f} us  speedup {base/tot:5.2f}x", flush=True)
            out[f"chunk|{L}|{K}"] = {"fwd": tf, "bwd": tb}

    print("\n--- batch scaling at L=392 (2 directions, so b = 2 x train batch) ---", flush=True)
    for tb_ in (8, 16, 32, 64, 128):
        tf, tbw = scan_time(2 * tb_, 392)
        per = (tf + tbw) / tb_
        print(f"  train batch {tb_:<4} shape=({2*tb_},{D},392)  total {(tf+tbw)*1e6:8.1f} us"
              f"  per-sample {per*1e6:7.2f} us", flush=True)
        out[f"batch|{tb_}"] = {"fwd": tf, "bwd": tbw}

    print("SCALE_JSON " + json.dumps(out), flush=True)


main()
