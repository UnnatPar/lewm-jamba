"""A Triton selective scan, forward only. Step 1 of the kernel rewrite.

Why a rewrite is the only lever left: the mamba-1 CUDA scan runs at 5-12% of the A100's
bandwidth roofline and ~5% of fp32 peak while already saturating the machine, and it is
42-46% of Jamba's training step for 2.5% of its arithmetic. Chunking it, fusing the gating,
and swapping in Mamba-2's SSD were all measured and all lost (ledger, 2026-07-28).

The bet this kernel makes: the CUDA kernel buys intra-sequence parallelism with a
shuffle-heavy chunked warp scan, and scanscale.py showed that parallelism is worthless here
(K=8 chunks ran at 0.49x). So drop it. Run a plain sequential loop over L with a whole
(BLOCK_D, N) tile of state live in registers, which turns each timestep into a dense 256-FMA
tile instead of a warp shuffle, and reuse B/C across every channel in the tile -- they are
(b, N, L), shared by all d_inner channels, which the per-channel decomposition cannot exploit.

Semantics are mamba-1's exactly, so there is no capacity question to answer:
    dt   = softplus(delta + delta_bias)
    h[t] = exp(dt[t] * A) * h[t-1] + dt[t] * B[t] * u[t]     A is (d_inner, d_state)
    y[t] = sum_n C[n,t] * h[:,n,t] + D * u[t]
A stays (d_inner, d_state) -- 4608 independent decay rates, unchanged.

Correctness gate before any timing is believed: agreement with a naive per-timestep fp32
reference AND with selective_scan_cuda on identical inputs.
"""

import sys, time, json, math
import torch
import triton
import triton.language as tl

sys.path.insert(0, "/content")

DEV = "cuda"
# Must be a tl.constexpr instance, not a plain float: a @triton.jit body cannot close over
# an ordinary module global at all.
LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _scan_fwd(u_ptr, dl_ptr, A_ptr, B_ptr, C_ptr, D_ptr, bias_ptr, y_ptr,
              seqlen, dim, dstate,
              su_b, su_d, su_l,
              sb_b, sb_n, sb_l,
              sy_b, sy_d, sy_l,
              BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr, SOFTPLUS: tl.constexpr,
              UNROLL: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    mask_d = offs_d < dim
    mask_n = offs_n < dstate

    # A is (dim, dstate), contiguous. Held in registers for the whole scan -- this is the
    # tile that makes each timestep dense instead of shuffle-bound.
    A = tl.load(A_ptr + offs_d[:, None] * dstate + offs_n[None, :],
                mask=mask_d[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
    Dv = tl.load(D_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    u_p = u_ptr + pid_b * su_b + offs_d * su_d
    dl_p = dl_ptr + pid_b * su_b + offs_d * su_d
    b_p = B_ptr + pid_b * sb_b + offs_n * sb_n
    c_p = C_ptr + pid_b * sb_b + offs_n * sb_n
    y_p = y_ptr + pid_b * sy_b + offs_d * sy_d

    h = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)

    # The state update h = h*dA + dBu is a serial dependency of length seqlen, and at ~5% of
    # both peak and roofline this kernel is bound by that chain, not by work. Unrolling with
    # static_range puts UNROLL timesteps in one basic block, so the loads and the exp2 -- none
    # of which depend on h -- can be hoisted above the chain and overlapped with it.
    for t0 in range(0, seqlen, UNROLL):
        for r in tl.static_range(UNROLL):
            t = t0 + r
            live = t < seqlen
            u = tl.load(u_p + t * su_l, mask=mask_d & live, other=0.0).to(tl.float32)
            dt = tl.load(dl_p + t * su_l, mask=mask_d & live, other=0.0).to(tl.float32) + bias
            if SOFTPLUS:
                # log1p(exp(x)), guarded the way the reference kernel guards it
                dt = tl.where(dt <= 20.0, tl.log(1.0 + tl.exp(dt)), dt)
            Bv = tl.load(b_p + t * sb_l, mask=mask_n & live, other=0.0).to(tl.float32)
            Cv = tl.load(c_p + t * sb_l, mask=mask_n & live, other=0.0).to(tl.float32)

            # exp2 rather than exp: same value, SFU-cheaper.
            dA = tl.math.exp2(dt[:, None] * A * LOG2E)
            # A masked-off tail step must leave the state untouched, not zero it.
            h = tl.where(live, h * dA + (dt[:, None] * u[:, None]) * Bv[None, :], h)
            y = tl.sum(h * Cv[None, :], axis=1) + Dv * u
            tl.store(y_p + t * sy_l, y, mask=mask_d & live)


def triton_scan(u, delta, A, B, C, D, delta_bias, delta_softplus=True,
                block_d=16, unroll=4, num_warps=4, num_stages=2):
    b, dim, L = u.shape
    dstate = A.shape[1]
    y = torch.empty_like(u)
    grid = (b, triton.cdiv(dim, block_d))
    _scan_fwd[grid](
        u, delta, A, B, C, D, delta_bias, y,
        L, dim, dstate,
        u.stride(0), u.stride(1), u.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_D=block_d, BLOCK_N=triton.next_power_of_2(dstate),
        SOFTPLUS=delta_softplus, UNROLL=unroll,
        num_warps=num_warps, num_stages=num_stages,
    )
    return y


@triton.jit
def _scan_bidir(u_ptr, dl_ptr, B_ptr, C_ptr,
                u2_ptr, dl2_ptr, B2_ptr, C2_ptr,
                A_ptr, D_ptr, bias_ptr, yf_ptr, yb_ptr,
                seqlen, dim, dstate,
                su_b, su_d, su_l,
                sb_b, sb_n, sb_l,
                sy_b, sy_d, sy_l,
                BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr, SOFTPLUS: tl.constexpr):
    """Both directions in one program, two state tiles live in registers at once.

    The bidirectional mixer currently builds cat([x, flip(x)]) and runs the scan on 2b rows,
    so the two chains are separate kernel-launches' worth of work with nothing shared. They
    are independent, which is exactly the resource a latency-bound serial recurrence is short
    of: interleaving them doubles the work in flight per thread while A, D and delta_bias are
    loaded once instead of twice.

    This is the kernel-level half of LBMamba's idea (arXiv 2506.15976). Their alternating
    scan directions across layers are NOT adopted -- that changes what each layer sees.

    The second chain gets its OWN u/delta/B/C pointers rather than reverse-indexing the
    first's. That is not an optimisation detail, it is correctness: the reverse direction's
    dt/B/C are produced by x_proj on the *flipped* sequence after the causal conv, and
    conv(flip(x)) != flip(conv(x)). Both chains therefore run t = 0..L-1 in their own order,
    exactly as the (2b, d, L) tensor already stores them, and the caller still finishes with
    y_fwd + flip(y_bwd).
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    mask_d = offs_d < dim
    mask_n = offs_n < dstate

    A = tl.load(A_ptr + offs_d[:, None] * dstate + offs_n[None, :],
                mask=mask_d[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
    Dv = tl.load(D_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    uo, bo = pid_b * su_b + offs_d * su_d, pid_b * sb_b + offs_n * sb_n
    u_p, dl_p, b_p, c_p = u_ptr + uo, dl_ptr + uo, B_ptr + bo, C_ptr + bo
    u2_p, dl2_p, b2_p, c2_p = u2_ptr + uo, dl2_ptr + uo, B2_ptr + bo, C2_ptr + bo
    yf_p = yf_ptr + pid_b * sy_b + offs_d * sy_d
    yb_p = yb_ptr + pid_b * sy_b + offs_d * sy_d

    hf = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
    hb = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)

    for t in range(0, seqlen):
        uf = tl.load(u_p + t * su_l, mask=mask_d, other=0.0).to(tl.float32)
        df = tl.load(dl_p + t * su_l, mask=mask_d, other=0.0).to(tl.float32) + bias
        ub = tl.load(u2_p + t * su_l, mask=mask_d, other=0.0).to(tl.float32)
        db = tl.load(dl2_p + t * su_l, mask=mask_d, other=0.0).to(tl.float32) + bias
        if SOFTPLUS:
            df = tl.where(df <= 20.0, tl.log(1.0 + tl.exp(df)), df)
            db = tl.where(db <= 20.0, tl.log(1.0 + tl.exp(db)), db)
        Bf = tl.load(b_p + t * sb_l, mask=mask_n, other=0.0).to(tl.float32)
        Cf = tl.load(c_p + t * sb_l, mask=mask_n, other=0.0).to(tl.float32)
        Bb = tl.load(b2_p + t * sb_l, mask=mask_n, other=0.0).to(tl.float32)
        Cb = tl.load(c2_p + t * sb_l, mask=mask_n, other=0.0).to(tl.float32)

        hf = hf * tl.math.exp2(df[:, None] * A * LOG2E) + (df[:, None] * uf[:, None]) * Bf[None, :]
        hb = hb * tl.math.exp2(db[:, None] * A * LOG2E) + (db[:, None] * ub[:, None]) * Bb[None, :]

        tl.store(yf_p + t * sy_l, tl.sum(hf * Cf[None, :], axis=1) + Dv * uf, mask=mask_d)
        tl.store(yb_p + t * sy_l, tl.sum(hb * Cb[None, :], axis=1) + Dv * ub, mask=mask_d)


def triton_bidir(u2, delta2, A, B2, C2, D, delta_bias, delta_softplus=True,
                 block_d=8, num_warps=1, num_stages=2):
    """Takes the (2b, ...) tensors the mixer already builds and splits them into the two
    chains. Returns (y_fwd, y_bwd) stacked back to (2b, d, L), so the caller's existing
    y_cat[:b] + y_cat[b:].flip(-1) still applies unchanged."""
    b2_, dim, L = u2.shape
    b = b2_ // 2
    yf, yb = torch.empty_like(u2[:b]), torch.empty_like(u2[:b])
    _scan_bidir[(b, triton.cdiv(dim, block_d))](
        u2[:b], delta2[:b], B2[:b], C2[:b],
        u2[b:], delta2[b:], B2[b:], C2[b:],
        A, D, delta_bias, yf, yb,
        L, dim, A.shape[1],
        u2.stride(0), u2.stride(1), u2.stride(2),
        B2.stride(0), B2.stride(1), B2.stride(2),
        yf.stride(0), yf.stride(1), yf.stride(2),
        BLOCK_D=block_d, BLOCK_N=triton.next_power_of_2(A.shape[1]),
        SOFTPLUS=delta_softplus, num_warps=num_warps, num_stages=num_stages,
    )
    return torch.cat([yf, yb], dim=0)


def reference(u, delta, A, B, C, D, delta_bias, delta_softplus=True):
    """Naive per-timestep fp32. Slow on purpose -- this is the thing being trusted."""
    u, delta = u.float(), delta.float()
    Bf, Cf = B.float(), C.float()
    b, dim, L = u.shape
    n = A.shape[1]
    dt = delta + delta_bias[None, :, None]
    if delta_softplus:
        dt = torch.nn.functional.softplus(dt)
    h = torch.zeros(b, dim, n, device=u.device, dtype=torch.float32)
    ys = []
    for t in range(L):
        dA = torch.exp(dt[:, :, t][:, :, None] * A[None])
        h = h * dA + dt[:, :, t][:, :, None] * Bf[:, :, t][:, None, :] * u[:, :, t][:, :, None]
        ys.append((h * Cf[:, :, t][:, None, :]).sum(-1) + D[None, :] * u[:, :, t])
    return torch.stack(ys, dim=-1)


def make(b, dim, L, n, dtype):
    g = lambda *s: torch.randn(*s, device=DEV, dtype=dtype)
    u, delta = g(b, dim, L), g(b, dim, L)
    Bm, Cm = g(b, n, L), g(b, n, L)
    A = -torch.rand(dim, n, device=DEV, dtype=torch.float32).exp()
    D = torch.ones(dim, device=DEV, dtype=torch.float32)
    bias = torch.zeros(dim, device=DEV, dtype=torch.float32)
    return u, delta, A, Bm, Cm, D, bias


def validate():
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    ok = True
    for dtype in (torch.float32, torch.bfloat16):
        # small L so the naive reference is affordable
        args = make(2, 64, 64, 16, dtype)
        ref = reference(*args)
        tri = triton_scan(*args).float()
        cud = selective_scan_fn(args[0], args[1], args[2], args[3], args[4], args[5],
                                z=None, delta_bias=args[6], delta_softplus=True).float()
        e_tri = (tri - ref).abs().max().item()
        e_cud = (cud - ref).abs().max().item()
        rel = e_tri / (ref.abs().max().item() + 1e-12)
        tol = 2e-4 if dtype == torch.float32 else 5e-2
        good = e_tri <= max(tol, 2 * e_cud)
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} {str(dtype):<16} triton-vs-ref {e_tri:.3e}  "
              f"cuda-vs-ref {e_cud:.3e}  rel {rel:.3e}", flush=True)
    return ok


def timed(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bench():
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    out = {}
    for L in (392, 1868):
        args = make(32, 288, L, 16, torch.bfloat16)
        t_cuda = timed(lambda: selective_scan_fn(args[0], args[1], args[2], args[3], args[4],
                                                 args[5], z=None, delta_bias=args[6],
                                                 delta_softplus=True))
        print(f"\n  L={L}  cuda fwd {t_cuda*1e6:8.1f} us", flush=True)
        best = None
        for bd in (2, 4, 8):
            for un in (1, 8, 16):
                for nw in (1, 2):
                    key = f"{L}|d{bd}|u{un}|w{nw}"
                    try:
                        t = timed(lambda: triton_scan(*args, block_d=bd, unroll=un,
                                                      num_warps=nw))
                    except Exception as e:  # noqa: BLE001
                        print(f"    d={bd} u={un} w={nw}  FAILED "
                              f"{type(e).__name__}: {str(e)[:80]}", flush=True)
                        continue
                    out[key] = t
                    if best is None or t < best[0]:
                        best = (t, key)
                    print(f"    d={bd:<3} unroll={un:<2} warps={nw:<2} {t*1e6:8.1f} us  "
                          f"{t_cuda/t:5.2f}x vs cuda", flush=True)
        out[f"{L}|cuda"] = t_cuda
        if best:
            print(f"    BEST {best[1]}  {best[0]*1e6:.1f} us  {t_cuda/best[0]:.2f}x vs cuda",
                  flush=True)
    print("TSCAN_JSON " + json.dumps(out), flush=True)


def short_sequence_sweep():
    """The one place left where the CUDA kernel should be beatable: short sequences.

    Fitting the measured scan times (899 us at L=392, 3487 us at L=1868) to s0 + s*L gives
    s0 = 212 us per layer-call -- an L-independent cost that is 24% of the whole scan at
    L=392. It is quantisation: mamba dispatches kNThreads=32, kNItems=16 below L=512, so a
    block covers 512 timesteps and L=392 pays for 512. At L=196 -- the real per-frame token
    count for this project -- it pays for 512 to use 196, wasting 62%.

    The Triton kernel loops exactly seqlen times and has no such quantisation. Nothing has
    ever measured either kernel below 392, and 196-392 is precisely the target regime.
    """
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    print("\n  short sequences (the target regime; nothing has measured here before):",
          flush=True)
    for L in (128, 196, 256, 320, 392, 512, 640, 784):
        a = make(32, 288, L, 16, torch.bfloat16)
        t_cuda = timed(lambda: selective_scan_fn(a[0], a[1], a[2], a[3], a[4], a[5], z=None,
                                                 delta_bias=a[6], delta_softplus=True))
        best = None
        for bd in (4, 8, 16):
            for un in (1, 4, 8):
                try:
                    t = timed(lambda: triton_scan(*a, block_d=bd, unroll=un, num_warps=1))
                except Exception:  # noqa: BLE001
                    continue
                if best is None or t < best[0]:
                    best = (t, bd, un)
        ns_cuda = t_cuda * 1e9 / (32 * 288 * L)
        print(f"    L={L:<5} cuda {t_cuda*1e6:8.1f} us ({ns_cuda:5.3f} ns/elem)   "
              f"triton {best[0]*1e6:8.1f} us (d={best[1]} u={best[2]})   "
              f"{t_cuda/best[0]:5.2f}x", flush=True)


def occupancy_probe():
    """Is this kernel starved of parallel work, or already saturating the GPU?

    It decides the next move and the two answers are opposite. The grid is
    (batch, dim/BLOCK_D), so scaling batch scales the program count directly. Sublinear
    time means idle SMs, which means a chunked scan -- more programs, each shorter -- is
    worth building despite needing a state-passing correction. Linear time means the
    machine is full and chunking only adds the correction's cost.

    scanscale.py already answered this for the CUDA kernel (linear, saturated at batch 8).
    This kernel has a completely different decomposition, so it needs its own answer.
    """
    print("\n  occupancy probe, L=392, d=8 unroll=8 warps=1:", flush=True)
    base = None
    for b in (8, 32, 128, 256):
        args = make(b, 288, 392, 16, torch.bfloat16)
        t = timed(lambda: triton_scan(*args, block_d=8, unroll=8, num_warps=1))
        per = t / b
        base = base or per
        print(f"    b={b:<4} programs={b*36:<6} {t*1e6:8.1f} us  per-sample {per*1e6:6.2f} us"
              f"  ({base/per:4.2f}x better than b=8)", flush=True)


def bidir():
    """The interleaved two-chain kernel, against the CUDA kernel doing the same (2b) work."""
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    print("\n  bidirectional (both chains in one program):", flush=True)
    args = make(4, 64, 96, 16, torch.float32)
    ref = selective_scan_fn(args[0], args[1], args[2], args[3], args[4], args[5],
                            z=None, delta_bias=args[6], delta_softplus=True)
    got = triton_bidir(*args)
    err = (got.float() - ref.float()).abs().max().item()
    good = err < 2e-4
    print(f"    {'PASS' if good else 'FAIL'} vs selective_scan_fn on identical (2b) input: "
          f"max abs err {err:.3e}", flush=True)
    if not good:
        return
    for L in (392, 1868):
        a = make(32, 288, L, 16, torch.bfloat16)
        t_cuda = timed(lambda: selective_scan_fn(a[0], a[1], a[2], a[3], a[4], a[5], z=None,
                                                 delta_bias=a[6], delta_softplus=True))
        best = None
        for bd in (2, 4, 8, 16):
            for nw in (1, 2):
                try:
                    t = timed(lambda: triton_bidir(*a, block_d=bd, num_warps=nw))
                except Exception as e:  # noqa: BLE001
                    print(f"    L={L} d={bd} w={nw} FAILED {type(e).__name__}", flush=True)
                    continue
                if best is None or t < best[0]:
                    best = (t, bd, nw)
        if best:
            print(f"    L={L:<5} cuda {t_cuda*1e6:8.1f} us   bidir {best[0]*1e6:8.1f} us "
                  f"(d={best[1]} w={best[2]})  {t_cuda/best[0]:5.2f}x vs cuda", flush=True)


def main():
    print(f"device {torch.cuda.get_device_name(0)}  triton {triton.__version__}", flush=True)
    print("validation:", flush=True)
    if not validate():
        print("STOPPING: triton scan does not match the reference", flush=True)
        return
    short_sequence_sweep()


main()
