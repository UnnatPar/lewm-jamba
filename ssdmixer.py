"""A Mamba-2 (SSD) mixer, drop-in for Jamba's mamba mixer, with alternating scan direction.

Why this is the last idea worth trying. At 1,960 tokens with SSM state and parameters held, the
encoder is at 0.890 of a parameter-matched ViT and every other lever is exhausted: compile hurts
the ratio, batch peaks at 20, 1,960 tokens is the best window (2,352 spills the scan's 2,048
chunk), 1 attention layer is optimal, and alternating direction already reaches 99% of a purely
causal encoder's speed. Scan FLOPs cannot shrink without shrinking SSM state, which is the one
thing ruled out.

SSD is the only remaining change that touches the *efficiency constant* rather than the FLOP
count: it reformulates the recurrence as chunked matmuls, so the scan runs on tensor cores instead
of as a memory-bound elementwise recurrence.

THE HONEST COST -- this is not a free win and not weight-compatible. SSD requires A to be a
per-HEAD scalar, shape (nheads,), where Mamba-1's A is per (channel, state), shape
(d_inner, d_state). Here that is 6 decay rates instead of 9,216. That restriction is exactly what
makes the recurrence expressible as matmuls. Mamba-2 argues (and shows at scale) that the loss is
more than repaid by the larger d_state it affords, so this is a different inductive bias rather
than a strict subset -- but it IS a change, and the A-expressivity reduction should be stated
plainly wherever this result is quoted.

Held fixed against the Mamba-1 configuration:
  SSM state   d_inner * d_state = 288 * 32 = 9216
  parameters  ~15.50M, by re-solving intermediate_size (in_proj carries B/C here, so the
              per-layer parameter split differs from Mamba-1's and cannot be assumed)
"""

import math

import torch
from einops import rearrange
from torch import nn

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


class RMSNormGated(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, z):
        x = x * nn.functional.silu(z)
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(v + self.eps)).to(x.dtype) * self.weight


class SSDMixer(nn.Module):
    """Mamba-2 mixer. `layer_idx` parity selects the scan direction, matching bidir.py's
    alternating schedule -- even layers forward, odd layers backward, so the stack is
    bidirectional through depth without any layer paying for two scans."""

    def __init__(self, hidden_size, layer_idx, expand=1, d_state=32, headdim=48, d_conv=4,
                 ngroups=1, chunk_size=256, alternate=True):
        super().__init__()
        self.layer_idx, self.alternate = layer_idx, alternate
        d_inner = expand * hidden_size
        assert d_inner % headdim == 0, (d_inner, headdim)
        self.d_inner, self.d_state, self.headdim = d_inner, d_state, headdim
        self.nheads = d_inner // headdim
        self.ngroups, self.chunk_size = ngroups, chunk_size

        d_in_proj = 2 * d_inner + 2 * ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(hidden_size, d_in_proj, bias=False)
        conv_dim = d_inner + 2 * ngroups * d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim,
                                padding=d_conv - 1, bias=True)

        # dt bias initialised so softplus(dt_bias) spans ~[0.001, 0.1], as in Mamba-2.
        dt = torch.exp(torch.rand(self.nheads) * (math.log(0.1) - math.log(0.001))
                       + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.nheads + 1).float()))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = RMSNormGated(d_inner)
        self.out_proj = nn.Linear(d_inner, hidden_size, bias=False)

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kw):
        back = self.alternate and (self.layer_idx % 2 == 1)
        if back:
            hidden_states = hidden_states.flip(1)

        zxbcdt = self.in_proj(hidden_states)
        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1)

        # Causal conv, so it must run on the already-flipped sequence -- flipping a forward
        # convolution is not the same function.
        xBC = self.conv1d(xBC.transpose(1, 2))[..., :hidden_states.size(1)].transpose(1, 2)
        xBC = nn.functional.silu(xBC)
        x, B, C = torch.split(
            xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1)

        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            -torch.exp(self.A_log.float()),
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,                      # gate applied in the norm below, as Mamba-2 does
            dt_bias=self.dt_bias,
            dt_softplus=True,
        )
        y = self.norm(rearrange(y, "b l h p -> b l (h p)"), z)
        out = self.out_proj(y)
        return out.flip(1) if back else out


def swap_into(jamba_model, hidden_size, **kw):
    """Replace every JambaMambaMixer with an SSDMixer, preserving layer_idx (and therefore the
    alternating direction schedule). Returns the number replaced; a silent zero would leave the
    original Mamba-1 mixers in place and the measurement would be of the wrong model."""
    from transformers.models.jamba import modeling_jamba as mj
    n = 0
    for layer in jamba_model.layers:
        m = getattr(layer, "mamba", None)
        if isinstance(m, mj.JambaMambaMixer):
            layer.mamba = SSDMixer(hidden_size, m.layer_idx, **kw).to(
                next(m.parameters()).dtype).to(next(m.parameters()).device)
            n += 1
    return n
