"""Standalone benchmark: it/s and peak memory vs. sequence length, Jamba vs. a
bidirectional ViT-style Transformer of matched size.

Bypasses the real patch-embedding/training pipeline entirely — feeds synthetic
`inputs_embeds` of shape (batch, L, hidden_size) directly into each model's stack,
so we can sweep L far beyond the 196 tokens the real JambaEncoder ever sees per
frame. batch_size is fixed at 1 so we can push L as high as possible on the A100
before OOM; the crossover we're hunting for is a property of the *shape* of the
memory/time-vs-L curve, not the absolute numbers at production batch size.

Jamba's forward pass avoids the naive "run the whole 10-layer stack twice"
bidirectional trick. Only the linear-cost Mamba layers are run twice (forward +
reversed, then summed) — cheap, since they're O(L). The single quadratic-cost
attention layer (last in the stack, per attn_layer_offset=9) runs once,
non-causally, over the merged Mamba output. Per the crossover algebra
n* = C_m/C_a (Mamba's per-token kernel constant over attention's per-token-pair
kernel constant), the naive double-pass inflates n* by a constant ~2.25x versus
this single-attention-pass version — this is a free win with no capacity cost,
not a claim that it closes the full ~15000 -> 196-392 gap on its own.

Usage:
    python benchmark_seq_scaling.py
"""

import math

import time

import torch

from einops import rearrange, repeat

from torch import nn

from torch.nn import functional as F

from transformers import JambaConfig

HIDDEN_SIZE = 288

INTERMEDIATE_SIZE = 1152

NUM_HIDDEN_LAYERS = 10

NUM_ATTENTION_HEADS = 8

NUM_KEY_VALUE_HEADS = 4

ATTN_LAYER_PERIOD = 10

ATTN_LAYER_OFFSET = 9

MAMBA_D_STATE = 16

MAMBA_D_CONV = 4

MAMBA_EXPAND = 1  # d_inner = expand * hidden_size = 288; halved from expand=2 to bring params to ~15M

BATCH_SIZE = 1

SEQ_LENGTHS = [196, 392, 784, 1568, 3136, 6272, 12544]

WARMUP_ITERS = 3

TIMED_ITERS = 10

DEVICE = "cuda"

DTYPE = torch.bfloat16

class JambaBiMambaBatchedMixer(nn.Module):
    """Bidirectional Mamba-1 mixer that goes one step past JambaBiMambaMixer:
    instead of calling causal_conv1d_fn/x_proj/dt_proj/selective_scan_fn TWICE
    per layer (once per direction, each its own Python call), the forward and
    reversed sequences are stacked along the BATCH dimension -- (b, d, L) and
    (b, d, L) become one (2b, d, L) tensor -- so each of those ops is called
    ONCE per layer on both directions at once. This requires sharing conv1d/
    x_proj/dt_proj/A_log/D between directions (a single kernel call can't use
    two different weight tensors for two different batch halves), trading
    JambaBiMambaMixer's separate per-direction capacity for a further halving
    of per-layer kernel-launch count: 8 direction-specific op calls/layer
    (2x conv + 2x x_proj + 2x dt_proj + 2x scan) down to 4.

    mamba_inner_fn's fused-kernel experiment made the opposite trade (kept
    direction-specific in_proj/out_proj, lost weight-sharing) and got WORSE
    (14.3-14.6 it/s vs. Fused's 17-18.3) -- see JAMBA_OPTIMIZATION.md. This
    tests whether call-count reduction that preserves (in fact extends)
    weight-sharing, rather than trading it away for kernel fusion, is the
    lever that actually works.
    """

    def __init__(self, d_model, d_state, d_conv, expand, dt_rank="auto"):
        super().__init__()
        from causal_conv1d import causal_conv1d_fn
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

        self._causal_conv1d_fn = causal_conv1d_fn
        self._selective_scan_fn = selective_scan_fn

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n", d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, hidden_states):
        b, seq_len, _ = hidden_states.shape
        xz = rearrange(self.in_proj(hidden_states), "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)

        # Stack [forward input, reversed input] along the batch dim -> one
        # (2b, d, L) tensor. Everything downstream runs as a single call.
        x_cat = torch.cat([x, x.flip(dims=[-1])], dim=0)

        x_cat = self._causal_conv1d_fn(
            x=x_cat, weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
            bias=self.conv1d.bias, activation="silu",
        )
        x_dbl = self.x_proj(rearrange(x_cat, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seq_len)
        B = rearrange(B, "(b l) n -> b n l", l=seq_len).contiguous()
        C = rearrange(C, "(b l) n -> b n l", l=seq_len).contiguous()
        A = -torch.exp(self.A_log.float())
        y_cat = self._selective_scan_fn(
            x_cat, dt, A, B, C, self.D.float(), z=None,
            delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
        )

        y_fwd, y_bwd = y_cat[:b], y_cat[b:].flip(dims=[-1])
        y = rearrange(y_fwd + y_bwd, "b d l -> b l d")
        y = y * F.silu(rearrange(z, "b d l -> b l d"))
        return self.out_proj(y)

class JambaBiMambaBatchedDecoderLayer(nn.Module):
    """Same residual/RMSNorm/MLP wrapper as JambaBiMambaDecoderLayer, wrapping
    JambaBiMambaBatchedMixer instead of JambaBiMambaMixer. Called ONCE per
    layer."""

    def __init__(self, config, layer_idx):
        super().__init__()
        from transformers.models.jamba.modeling_jamba import JambaMLP, JambaRMSNorm

        self.mamba = JambaBiMambaBatchedMixer(
            d_model=config.hidden_size,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
        )
        self.feed_forward = JambaMLP(config)
        self.input_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class BidirectionalJambaBatched(nn.Module):
    """Previous best under eager execution: 9 Mamba layers each called ONCE
    (via JambaBiMambaBatchedDecoderLayer's batch-dim-stacked direction
    calls, sharing conv1d/x_proj/dt_proj/A/D across directions), plus a
    single non-causal attention pass merging the two directions' output --
    see JambaBiMambaBatchedMixer's docstring for the full call-count
    rationale."""

    def __init__(self):
        super().__init__()
        from transformers.models.jamba.modeling_jamba import JambaAttentionDecoderLayer, JambaRMSNorm

        config = JambaConfig(
            vocab_size=8,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            num_attention_heads=NUM_ATTENTION_HEADS,
            num_key_value_heads=NUM_KEY_VALUE_HEADS,
            attn_layer_period=ATTN_LAYER_PERIOD,
            attn_layer_offset=ATTN_LAYER_OFFSET,
            num_experts=1,
            num_experts_per_tok=1,
            mamba_d_state=MAMBA_D_STATE,
            mamba_d_conv=MAMBA_D_CONV,
            mamba_expand=MAMBA_EXPAND,
            max_position_embeddings=max(SEQ_LENGTHS),
        )
        config._attn_implementation = "sdpa"

        num_mamba_layers = NUM_HIDDEN_LAYERS - 1
        self.mamba_layers = nn.ModuleList(
            [JambaBiMambaBatchedDecoderLayer(config, layer_idx=i) for i in range(num_mamba_layers)]
        )
        self.attn_layer = JambaAttentionDecoderLayer(config, layer_idx=NUM_HIDDEN_LAYERS - 1)
        self.attn_layer.self_attn.is_causal = False
        self.final_layernorm = JambaRMSNorm(HIDDEN_SIZE, eps=config.rms_norm_eps)

    def forward(self, x):
        b, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)

        h = x
        for layer in self.mamba_layers:
            h = layer(h)

        out = self.attn_layer(h, attention_mask=None, position_ids=position_ids)
        out = self.final_layernorm(out)
        return out.mean(dim=1)

def benchmark(model, seq_len):
    x = torch.randn(BATCH_SIZE, seq_len, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)

    for _ in range(WARMUP_ITERS):
        model.zero_grad(set_to_none=True)
        out = model(x)
        out.mean().backward()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(TIMED_ITERS):
        model.zero_grad(set_to_none=True)
        out = model(x)
        out.mean().backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    it_per_s = TIMED_ITERS / elapsed
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    return it_per_s, peak_mem_gb


# --------------------------------------------------------------------------- baseline


class BidirectionalViTCompiled(nn.Module):
    """Parameter-matched ViT baseline.

    hidden=328, not 288. At 288 this had 9,990,720 parameters against the Jamba stack's
    12,683,520 -- 21% smaller -- despite the docstring claiming matched size. A smaller
    baseline is a faster baseline, so every crossover number produced before 2026-07-28 was
    biased in Jamba's favour. 328 lands at 12,952,720 (+2.1%), the closest head-divisible
    width.

    Width was widened rather than depth: depth changes the number of sequential kernel
    launches, which is the very thing the Jamba side is being measured on.

    torch.compile is a GENERAL optimization, so it is applied here too. Only SSM-specific
    kernel work (the custom-op registration) is Jamba-only.
    """

    def __init__(self, hidden=328, layers=NUM_HIDDEN_LAYERS, heads=NUM_ATTENTION_HEADS):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.proj = nn.Linear(HIDDEN_SIZE, hidden)
        self.encoder = torch.compile(nn.TransformerEncoder(layer, num_layers=layers),
                                     mode="reduce-overhead")

    def forward(self, x):
        return self.encoder(self.proj(x)).mean(dim=1)


def run_sweep(name, model_fn):
    print(f"\n=== {name} ===")
    results = {}
    for seq_len in SEQ_LENGTHS:
        torch.cuda.empty_cache()
        try:
            model = model_fn().to(device=DEVICE, dtype=DTYPE)
            it_per_s, peak_mem_gb = benchmark(model, seq_len)
            print(f"L={seq_len:6d}  it/s={it_per_s:7.2f}  peak_mem={peak_mem_gb:6.2f} GB")
            results[seq_len] = (it_per_s, peak_mem_gb)
            del model
            torch._dynamo.reset()
        except torch.cuda.OutOfMemoryError:
            print(f"L={seq_len:6d}  OOM")
            torch.cuda.empty_cache()
            torch._dynamo.reset()
            break
    return results


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA not available"

    # This synthetic forward+backward sweep is kept for quick shape-scaling checks only.
    # It is NOT the measurement standard. Every performance claim must come from a full
    # training step -- dataloader, head, loss, optimizer.step -- see realcross.py and
    # unnat-brain/projects/lewm-jamba.md. A batch-1 forward+backward sweep ranks variants by
    # launch overhead rather than compute, and it inverted at least one conclusion here.
    import sscanop2

    orig, shim = sscanop2.register()
    if not sscanop2.verify(orig, shim):
        raise SystemExit("selective_scan equivalence failed -- timings would be meaningless")

    run_sweep(
        "Jamba (batched bidirectional scan, single-pass attention, custom-op registered, "
        "whole-model torch.compile) -- SOTA",
        lambda: torch.compile(BidirectionalJambaBatched(), mode="reduce-overhead"),
    )
    run_sweep(
        "ViT-style Transformer, parameter-matched (torch.compile reduce-overhead)",
        BidirectionalViTCompiled,
    )
