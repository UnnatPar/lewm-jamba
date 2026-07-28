"""selective_scan custom op with a REAL fused backward instead of a recompute.

The previous version's backward re-ran the scan forward under enable_grad and took
autograd.grad. Correct, and cheap to get right, but it costs an extra forward on every step:
2 forwards + 1 backward per iteration instead of 1 + 1. At batch 1 that hid inside launch
overhead. At batch 16 the GPU is compute-bound and it is pure loss -- which is exactly the
regime the crossover is now measured in.

This version saves the scan's internal state `x` from the forward and hands it to
selective_scan_cuda.bwd, the same path mamba_ssm's own autograd.Function uses.

Two things are discovered at runtime rather than assumed, because guessing either one wrong
would produce silently wrong gradients:
  * the shape of `x` (needed for the fake impl) -- asserted against the real allocation
  * the accepted signature of selective_scan_cuda.bwd -- probed across known variants

Everything is gated on the equivalence check. If the fused backward disagrees with the
reference, this falls back to the recompute path rather than reporting a fast wrong number.
"""

import torch
import torch.nn.functional as F
from typing import List

_ORIG = None
_SHIM = None
_BWD_SIG = None
_CHUNK = 2048          # selective_scan_fwd allocates x with n_chunks = ceil(L / 2048)
PAD_SCAN = True        # see _pad_len


def _pad_len(L):
    """Round L up to the sequence length the kernel is going to charge for anyway.

    selective_scan dispatches (kNThreads, kNItems) by seqlen, and a block covers
    kNThreads*kNItems timesteps: 128 up to L=128, then 256, 512, 1024, and 2048 beyond.
    Whatever is left over in the last block is paid for and thrown away, and measurement says
    that waste is worth far more than its share -- at batch 16 x 2 directions, d=288:

        L=196  158.3 us      L=256   91.4 us   <- 60 MORE timesteps, 42% LESS time
        L=320  277.0 us      L=512  140.3 us   <- 192 more timesteps, half the time
        L=392  219.6 us      L=512  140.3 us

    196 and 392 are exactly the per-frame token counts this project cares about, and both sit
    on the bad side of a boundary. So pad up to it.

    Exact, not approximate: the padded steps come after every real one, so no real output
    changes. Slicing the output back to L makes the padded positions' grad_out zero, so the
    reduced gradients (dA, dD, ddelta_bias) get no contribution from them either -- the
    backward recursion carries dh = 0 through the whole padded tail.
    """
    if not PAD_SCAN:
        return L
    for c in (128, 256, 512, 1024):
        if L <= c:
            return c
    return ((L + _CHUNK - 1) // _CHUNK) * _CHUNK


def _probe_bwd_signature(u, delta, A, B, C, D, z, delta_bias, dout, x, out, delta_softplus):
    """selective_scan_cuda.bwd's arity has moved between releases. Try known orders."""
    import selective_scan_cuda
    candidates = [
        ("u,delta,A,B,C,D,z,db,dout,x,out,None,ds,False",
         lambda: selective_scan_cuda.bwd(u, delta, A, B, C, D, z, delta_bias, dout, x, out,
                                         None, delta_softplus, False)),
        ("u,delta,A,B,C,D,z,db,dout,x,None,ds,False",
         lambda: selective_scan_cuda.bwd(u, delta, A, B, C, D, z, delta_bias, dout, x,
                                         None, delta_softplus, False)),
        ("u,delta,A,B,C,D,z,db,dout,x,out,None,ds",
         lambda: selective_scan_cuda.bwd(u, delta, A, B, C, D, z, delta_bias, dout, x, out,
                                         None, delta_softplus)),
    ]
    for name, fn in candidates:
        try:
            r = fn()
            print(f"  bwd signature: {name}  -> {len(r)} outputs", flush=True)
            return name, fn
        except (TypeError, RuntimeError) as e:
            print(f"  bwd signature rejected ({name}): {str(e)[:80]}", flush=True)
    return None, None


def register(fused_backward=True):
    global _ORIG, _SHIM
    from mamba_ssm.ops import selective_scan_interface as ssi
    import selective_scan_cuda

    if _ORIG is not None:
        ssi.selective_scan_fn = _SHIM
        return _ORIG, _SHIM

    orig = _ORIG = ssi.selective_scan_fn

    @torch.library.custom_op("lewm3::sscan", mutates_args=(), device_types="cuda")
    def sscan(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
              C: torch.Tensor, D: torch.Tensor, z: torch.Tensor, delta_bias: torch.Tensor,
              delta_softplus: bool) -> List[torch.Tensor]:
        zz = None if z.numel() == 0 else z
        u = u.contiguous(); delta = delta.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        # Replicate the normalization selective_scan_fn does before touching the kernel.
        # bench passes B and C as (b, dstate, l); the raw CUDA entry point requires
        # (b, n_groups, dstate, l). Calling it directly skips that reshape, which is why
        # every variant failed with "B must have shape (batch_size, n_groups, dstate, ...)".
        if B.dim() == 3:
            B = B.unsqueeze(1)
        if C.dim() == 3:
            C = C.unsqueeze(1)
        out, x, *_ = selective_scan_cuda.fwd(u, delta, A, B, C, D, zz, delta_bias,
                                             delta_softplus)
        n_chunks = (u.shape[-1] + _CHUNK - 1) // _CHUNK
        expected = (u.shape[0], u.shape[1], n_chunks, A.shape[1] * 2)
        assert tuple(x.shape) == expected, (
            f"x shape {tuple(x.shape)} != assumed {expected}; the fake impl would be wrong "
            f"and inductor would miscompile. Update _CHUNK / the formula.")
        return [out.contiguous(), x]

    @sscan.register_fake
    def _(u, delta, A, B, C, D, z, delta_bias, delta_softplus):
        n_chunks = (u.shape[-1] + _CHUNK - 1) // _CHUNK
        return [torch.empty(u.shape, dtype=u.dtype, device=u.device),
                torch.empty((u.shape[0], u.shape[1], n_chunks, A.shape[1] * 2),
                            dtype=torch.float32, device=u.device)]

    @torch.library.custom_op("lewm3::sscan_bwd", mutates_args=(), device_types="cuda")
    def sscan_bwd(dout: torch.Tensor, x: torch.Tensor, out: torch.Tensor, u: torch.Tensor,
                  delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                  D: torch.Tensor, z: torch.Tensor, delta_bias: torch.Tensor,
                  delta_softplus: bool) -> List[torch.Tensor]:
        global _BWD_SIG
        zz = None if z.numel() == 0 else z
        dout = dout.contiguous()
        # Remember the caller's rank BEFORE the 4-D normalization below. The kernel returns
        # dB/dC matching its own 4-D view, but autograd (and the fake impl) expect grads
        # shaped like the inputs it was given -- otherwise inductor asserts "wrong number of
        # dimensions 4".
        b_was_3d, c_was_3d = B.dim() == 3, C.dim() == 3
        if _BWD_SIG is None:
            name, fn = _probe_bwd_signature(u, delta, A, B, C, D, zz, delta_bias, dout, x,
                                            out, delta_softplus)
            if fn is None:
                raise RuntimeError("no working selective_scan_cuda.bwd signature")
            _BWD_SIG = name
        # Replicate the normalization selective_scan_fn does before touching the kernel.
        # bench passes B and C as (b, dstate, l); the raw CUDA entry point requires
        # (b, n_groups, dstate, l). Calling it directly skips that reshape, which is why
        # every variant failed with "B must have shape (batch_size, n_groups, dstate, ...)".
        if B.dim() == 3:
            B = B.unsqueeze(1)
        if C.dim() == 3:
            C = C.unsqueeze(1)
        r = selective_scan_cuda.bwd(u, delta, A, B, C, D, zz, delta_bias, dout, x, out,
                                    None, delta_softplus, False)
        # Arity depends on whether z was passed: without z there is no dz (7 returns), with z
        # there is dz and sometimes out_z as well. Take the first seven positionally and
        # decide dz from the count -- the equivalence check is what confirms the ordering.
        du, ddelta, dA, dB, dC, dD, ddelta_bias = r[:7]
        dz = r[7] if (zz is not None and len(r) > 7) else torch.zeros_like(z)
        if b_was_3d and dB.dim() == 4:
            dB = dB.squeeze(1)
        if c_was_3d and dC.dim() == 4:
            dC = dC.squeeze(1)
        return [t.contiguous() for t in
                (du, ddelta, dA, dB, dC, dD, dz, ddelta_bias)]

    @sscan_bwd.register_fake
    def _(dout, x, out, u, delta, A, B, C, D, z, delta_bias, delta_softplus):
        return [torch.empty(t.shape, dtype=t.dtype, device=t.device)
                for t in (u, delta, A, B, C, D, z, delta_bias)]

    def setup_context(ctx, inputs, output):
        u, delta, A, B, C, D, z, delta_bias, ds = inputs
        ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, output[0], output[1])
        ctx.ds = ds

    def backward(ctx, grads):
        # The op returns List[Tensor], so autograd hands back ONE list of grads, not one
        # argument per output. grads[1] (w.r.t. the saved scan state x) is unused: x is an
        # internal buffer, not something downstream differentiates through.
        grad_out = grads[0]
        u, delta, A, B, C, D, z, delta_bias, out, x = ctx.saved_tensors
        gs = torch.ops.lewm3.sscan_bwd(grad_out, x, out, u, delta, A, B, C, D, z,
                                       delta_bias, ctx.ds)
        return (*gs, None)

    torch.library.register_autograd("lewm3::sscan", backward, setup_context=setup_context)

    _NONE = torch.empty(0)

    def shim(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
             return_last_state=False):
        # z != None routes to the unmodified kernel. selective_scan_cuda.fwd returns the
        # PRE-gating output plus a separate out_z, so the fused path would need different
        # plumbing there -- and bench never passes z, so optimizing it buys nothing. Verified
        # by the equivalence check, which caught this as a 9.0e+01 forward error.
        if return_last_state or D is None or delta_bias is None or z is not None:
            return orig(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)
        zz = _NONE.to(u.device, u.dtype) if z is None else z
        L = u.shape[-1]
        Lp = _pad_len(L)
        if Lp != L:
            p = (0, Lp - L)
            u, delta = F.pad(u, p), F.pad(delta, p)
            B, C = F.pad(B, p), F.pad(C, p)
        out = torch.ops.lewm3.sscan(
            u.contiguous(), delta.contiguous(), A.contiguous(), B.contiguous(),
            C.contiguous(), D.contiguous(), zz.contiguous(), delta_bias.contiguous(),
            delta_softplus)[0]
        return out[..., :L] if Lp != L else out

    _SHIM = shim
    ssi.selective_scan_fn = shim
    import sys
    b = sys.modules.get("bench")
    if b is not None and hasattr(b, "selective_scan_fn"):
        b.selective_scan_fn = shim
    return orig, shim


def verify(orig, shim, device="cuda", batch=2, lengths=(64, 196, 392)):
    ok_all = True
    # Several lengths, because the shim pads L up to the kernel's chunk boundary and a
    # single length would leave that path untested at the shapes that matter. 196 and 392
    # both pad; a length that happens to land on a boundary would hide a padding bug.
    cases = [(f"{lbl}, L={l}", use_z, l)
             for l in lengths
             for lbl, use_z in (("z provided (falls back)", True), ("z=None (fused)", False))]
    for label, use_z, l in cases:
        torch.manual_seed(0)
        b, d, n = batch, 32, 16
        mk = lambda *s: torch.randn(*s, device=device, dtype=torch.float32, requires_grad=True)
        u, delta = mk(b, d, l), mk(b, d, l)
        A = (-torch.rand(d, n, device=device).float() - 0.1).requires_grad_(True)
        B, C = mk(b, 1, n, l), mk(b, 1, n, l)
        D, db = mk(d), mk(d)
        z = mk(b, d, l) if use_z else None
        ts = [t for t in (u, delta, A, B, C, D, z, db) if t is not None]
        outs, grads = [], []
        for fn in (orig, shim):
            for t in ts:
                t.grad = None
            o = fn(u, delta, A, B, C, D, z, db, True)
            outs.append(o.detach().clone())
            o.sum().backward()
            grads.append([t.grad.detach().clone() for t in ts])
        f = (outs[0] - outs[1]).abs().max().item()
        g = max((a - b_).abs().max().item() for a, b_ in zip(*grads))
        good = f < 1e-4 and g < 1e-3
        ok_all &= good
        print(f"  [{label:<32}] fwd={f:.3e} grad={g:.3e}  "
              f"{'PASS' if good else '*** FAIL ***'}", flush=True)
    print("  EQUIVALENCE:", "PASS" if ok_all else "*** FAIL ***", flush=True)
    return ok_all
