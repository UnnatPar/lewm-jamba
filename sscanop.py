"""selective_scan as a torch.library custom op. Import-safe, idempotent, no side effects.

Split out of customop2.py because that file called main() at module scope: importing it ran the
whole benchmark AND registered the op, so a second register() captured the shim as `orig` and
recursed until the stack died. register() here is safe to call any number of times.

Why this exists at all: Dynamo graph-breaks at every selective_scan_fn call (9 breaks, 10 graphs
for a 10-layer stack), so torch.compile can only ever optimize fragments and inductor can never
cover the model end to end. Registering the kernel as a real op with a fake implementation
removes every break. Same kernel, same values -- verified numerically by verify().
"""

import torch
from typing import List

_ORIG = None
_SHIM = None


def register():
    """Patch selective_scan_fn to route through lewm2::sscan. Returns (orig, shim)."""
    global _ORIG, _SHIM
    from mamba_ssm.ops import selective_scan_interface as ssi

    if _ORIG is not None:                      # already registered; do not re-capture
        ssi.selective_scan_fn = _SHIM
        return _ORIG, _SHIM

    orig = _ORIG = ssi.selective_scan_fn

    @torch.library.custom_op("lewm2::sscan", mutates_args=(), device_types="cuda")
    def sscan(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
              C: torch.Tensor, D: torch.Tensor, z: torch.Tensor, delta_bias: torch.Tensor,
              delta_softplus: bool) -> torch.Tensor:
        zz = None if z.numel() == 0 else z
        with torch.no_grad():
            return orig(u, delta, A, B, C, D, zz, delta_bias, delta_softplus).clone()

    @sscan.register_fake
    def _(u, delta, A, B, C, D, z, delta_bias, delta_softplus):
        # Contiguous, NOT empty_like: the real op returns a contiguous tensor (it ends in
        # .clone()), while u at trace time can be a permuted view. A fake whose strides differ
        # from the real output trips an inductor stride assertion at runtime.
        return torch.empty(u.shape, dtype=u.dtype, device=u.device)

    @torch.library.custom_op("lewm2::sscan_bwd", mutates_args=(), device_types="cuda")
    def sscan_bwd(g: torch.Tensor, u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
                  B: torch.Tensor, C: torch.Tensor, D: torch.Tensor, z: torch.Tensor,
                  delta_bias: torch.Tensor, delta_softplus: bool) -> List[torch.Tensor]:
        ins = (u, delta, A, B, C, D, z, delta_bias)
        with torch.enable_grad():
            dup = [t.detach().requires_grad_(True) for t in ins]
            args = list(dup)
            if args[6].numel() == 0:          # z was None on the way in
                args[6] = None
            out = orig(*args, delta_softplus)
            dup = [t for t in dup]
            grads = torch.autograd.grad(out, dup, g, allow_unused=True)
        return [(gr if gr is not None else torch.zeros(t.shape, dtype=t.dtype, device=t.device)).contiguous()
                for gr, t in zip(grads, ins)]

    @sscan_bwd.register_fake
    def _(g, u, delta, A, B, C, D, z, delta_bias, delta_softplus):
        return [torch.empty(t.shape, dtype=t.dtype, device=t.device)
                for t in (u, delta, A, B, C, D, z, delta_bias)]

    def setup_context(ctx, inputs, output):
        u, delta, A, B, C, D, z, delta_bias, ds = inputs
        ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias)
        ctx.ds = ds

    def backward(ctx, g):
        return (*torch.ops.lewm2.sscan_bwd(g, *ctx.saved_tensors, ctx.ds), None)

    torch.library.register_autograd("lewm2::sscan", backward, setup_context=setup_context)

    # bench calls this with z=None. Custom-op schemas cannot take Optional[Tensor] here, so
    # None travels as a 0-element tensor and is restored inside the op. Without this the shim
    # fell back to the unregistered kernel on every real call and the graph never healed --
    # breaks stayed at 9 while the equivalence test (which passes z) reported success.
    _NONE = torch.empty(0)

    def shim(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
             return_last_state=False):
        if return_last_state or D is None or delta_bias is None:
            return orig(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)
        zz = _NONE.to(u.device, u.dtype) if z is None else z
        # Make layout explicit in the traced graph. Inductor plans a stride for every tensor
        # entering an opaque op; the underlying kernel silently calls .contiguous() itself, so
        # the plan and the reality disagreed and runtime asserted
        # ("expected size 2==2, stride 196==56448 at dim=0"). Doing it here puts the
        # contiguous() in the graph where inductor can see it. No-op when already contiguous.
        return torch.ops.lewm2.sscan(
            u.contiguous(), delta.contiguous(), A.contiguous(), B.contiguous(),
            C.contiguous(), D.contiguous(), zz.contiguous(), delta_bias.contiguous(),
            delta_softplus)

    _SHIM = shim
    ssi.selective_scan_fn = shim
    try:
        import sys
        bench = sys.modules.get("bench")
        if bench is not None and hasattr(bench, "selective_scan_fn"):
            bench.selective_scan_fn = shim
    except Exception:  # noqa: BLE001
        pass
    return orig, shim


def verify(orig, shim, device="cuda", batch=2):
    """A faster mixer that computes a different function is not an optimization.

    Covers BOTH z paths. The z=None case is the one bench actually uses, and an earlier version
    of this check only exercised z-provided -- so it reported PASS while the real model was
    silently falling back to the unregistered kernel on every call. Also runs at batch>1, since
    a batch-1-only check cannot catch a batching bug.
    """
    ok_all = True
    for label, use_z in (("z provided", True), ("z=None (the path bench uses)", False)):
        torch.manual_seed(0)
        b, d, l, n = batch, 32, 64, 16
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
        print(f"  [{label:<28}] fwd={f:.3e} grad={g:.3e}  "
              f"{'PASS' if good else '*** FAIL ***'}", flush=True)
    print("  EQUIVALENCE:", "PASS" if ok_all else "*** FAIL ***", flush=True)
    return ok_all
