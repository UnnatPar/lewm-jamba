"""Single-graph Jamba, take 2: make the BACKWARD fake-safe too.

Take 1 got breaks 9 -> 0 and graphs 10 -> 1, with equivalence exact on forward and 3.8e-6 on
gradients. Inductor then failed compiling the backward: "Cannot access data pointer of Tensor
(FakeTensor)". The recompute backward called the raw CUDA kernel directly, so when inductor
traced the backward graph it handed that kernel fake tensors and the kernel tried to read real
memory.

Fix: the backward is its own custom op with its own fake impl. Forward and backward are then
both opaque-but-traceable, and the whole training step -- not just the forward -- compiles as
one graph.

Still no math change, and still no CUDA graph capture by hand: this is torch.compile only,
which recompiles on shape change and runs under an ordinary training loop.
"""

import sys, time, json
from typing import List
import torch
from torch import nn

sys.path.insert(0, "/content")
import bench

DEVICE, DTYPE = "cuda", torch.bfloat16
SCREEN = [392, 1568, 6272]
TIMED, WARMUP = 20, 5
_ORIG = None


def register():
    global _ORIG
    from mamba_ssm.ops import selective_scan_interface as ssi
    orig = _ORIG = ssi.selective_scan_fn

    @torch.library.custom_op("lewm2::sscan", mutates_args=(), device_types="cuda")
    def sscan(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
              C: torch.Tensor, D: torch.Tensor, z: torch.Tensor, delta_bias: torch.Tensor,
              delta_softplus: bool) -> torch.Tensor:
        with torch.no_grad():
            return orig(u, delta, A, B, C, D, z, delta_bias, delta_softplus).clone()

    @sscan.register_fake
    def _(u, delta, A, B, C, D, z, delta_bias, delta_softplus):
        return torch.empty_like(u)

    @torch.library.custom_op("lewm2::sscan_bwd", mutates_args=(), device_types="cuda")
    def sscan_bwd(g: torch.Tensor, u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
                  B: torch.Tensor, C: torch.Tensor, D: torch.Tensor, z: torch.Tensor,
                  delta_bias: torch.Tensor, delta_softplus: bool) -> List[torch.Tensor]:
        # Recompute rather than hand-transcribing selective_scan_cuda.bwd. Costs one extra
        # scan forward; buys a backward that cannot silently compute the wrong gradient.
        ins = (u, delta, A, B, C, D, z, delta_bias)
        with torch.enable_grad():
            dup = [t.detach().requires_grad_(True) for t in ins]
            out = orig(*dup, delta_softplus)
            grads = torch.autograd.grad(out, dup, g, allow_unused=True)
        return [gr if gr is not None else torch.zeros_like(t) for gr, t in zip(grads, ins)]

    @sscan_bwd.register_fake
    def _(g, u, delta, A, B, C, D, z, delta_bias, delta_softplus):
        return [torch.empty_like(t) for t in (u, delta, A, B, C, D, z, delta_bias)]

    def setup_context(ctx, inputs, output):
        u, delta, A, B, C, D, z, delta_bias, ds = inputs
        ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias)
        ctx.ds = ds

    def backward(ctx, g):
        gs = torch.ops.lewm2.sscan_bwd(g, *ctx.saved_tensors, ctx.ds)
        return (*gs, None)

    torch.library.register_autograd("lewm2::sscan", backward, setup_context=setup_context)

    def shim(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
             return_last_state=False):
        # The op signature requires real tensors; anything else falls back to the original.
        if return_last_state or any(t is None for t in (D, z, delta_bias)):
            return orig(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)
        return torch.ops.lewm2.sscan(u, delta, A, B, C, D, z, delta_bias, delta_softplus)

    ssi.selective_scan_fn = shim
    if hasattr(bench, "selective_scan_fn"):
        bench.selective_scan_fn = shim
    return orig, shim


def verify(orig, shim):
    torch.manual_seed(0)
    b, d, l, n = 1, 32, 64, 16
    mk = lambda *s: torch.randn(*s, device=DEVICE, dtype=torch.float32, requires_grad=True)
    u, delta = mk(b, d, l), mk(b, d, l)
    A = (-torch.rand(d, n, device=DEVICE).float() - 0.1).requires_grad_(True)
    B, C = mk(b, 1, n, l), mk(b, 1, n, l)
    D, z, db = mk(d), mk(b, d, l), mk(d)
    ts = (u, delta, A, B, C, D, z, db)
    outs, grads = [], []
    for fn in (orig, shim):
        for t in ts:
            t.grad = None
        o = fn(u, delta, A, B, C, D, z, db, True)
        outs.append(o.detach().clone())
        o.sum().backward()
        grads.append([t.grad.detach().clone() for t in ts])
    f = (outs[0] - outs[1]).abs().max().item()
    gmax = max((a - b_).abs().max().item() for a, b_ in zip(*grads))
    print(f"  fwd max|diff|={f:.3e}  grad max|diff|={gmax:.3e}", flush=True)
    ok = f < 1e-4 and gmax < 1e-3
    print("  EQUIVALENCE:", "PASS" if ok else "*** FAIL ***", flush=True)
    return ok


class Whole(nn.Module):
    def __init__(self, inner, mode="reduce-overhead"):
        super().__init__()
        self.m = torch.compile(inner, mode=mode)

    def forward(self, x):
        return self.m(x)


def make_vit(hidden=328, layers=10):
    layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=8, dim_feedforward=hidden * 4,
                                       dropout=0.0, activation="gelu", batch_first=True,
                                       norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=layers)


class FairViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(bench.HIDDEN_SIZE, 328)
        self.encoder = torch.compile(make_vit(), mode="reduce-overhead")

    def forward(self, x):
        return self.encoder(self.proj(x)).mean(dim=1)


def measure(model, L):
    x = torch.randn(1, L, bench.HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    for _ in range(WARMUP):
        torch.compiler.cudagraph_mark_step_begin()
        model.zero_grad(set_to_none=True)
        model(x).mean().backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(TIMED):
        torch.compiler.cudagraph_mark_step_begin()
        model.zero_grad(set_to_none=True)
        model(x).mean().backward()
    torch.cuda.synchronize()
    return TIMED / (time.perf_counter() - t0), torch.cuda.max_memory_allocated() / 1e9


def sweep(build, name, grid=SCREEN):
    out = {}
    for L in grid:
        torch.cuda.empty_cache()
        try:
            m = build()
            it, mem = measure(m, L)
            out[L] = (it, mem)
            print(f"  {name:<28} L={L:<6} {it:9.2f} it/s  {mem:5.2f} GB", flush=True)
            del m
            torch._dynamo.reset()
        except Exception as e:  # noqa: BLE001
            print(f"  {name:<28} L={L:<6} FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)
            torch.cuda.empty_cache(); torch._dynamo.reset()
    return out


def main():
    orig, shim = register()
    if not verify(orig, shim):
        print("STOPPING: equivalence failed", flush=True)
        return

    torch._dynamo.reset()
    m = bench.BidirectionalJambaBatched().to(DEVICE, DTYPE)
    exp = torch._dynamo.explain(m)(torch.randn(1, 392, bench.HIDDEN_SIZE, device=DEVICE, dtype=DTYPE))
    print(f"  breaks={exp.graph_break_count} graphs={exp.graph_count}", flush=True)
    del m
    torch._dynamo.reset()

    res = {}
    res["ViT matched"] = sweep(lambda: FairViT().to(DEVICE, DTYPE), "ViT matched")
    res["Jamba surgical (control)"] = sweep(
        lambda: bench.BidirectionalJambaBatchedFullCompiledGlue().to(DEVICE, DTYPE), "Jamba surgical")
    res["Jamba single-graph"] = sweep(
        lambda: Whole(bench.BidirectionalJambaBatched().to(DEVICE, DTYPE)), "Jamba single-graph")
    res["Jamba single-graph default"] = sweep(
        lambda: Whole(bench.BidirectionalJambaBatched().to(DEVICE, DTYPE), mode="default"),
        "Jamba single-graph default")

    print("RESULT_JSON " + json.dumps(
        {k: {str(L): v for L, v in c.items()} for k, c in res.items()}), flush=True)


main()
