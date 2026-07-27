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

# ---- matches config/train/model/lewm.yaml ----
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


class _MambaPreScanBlock(nn.Module):
    """in_proj -> rearrange -> chunk -> batch-stack directions. No mamba
    CUDA-extension calls -- safe to torch.compile, same rationale as
    _FFNBlock/_AttnTailBlock."""

    def __init__(self, in_proj):
        super().__init__()
        self.in_proj = in_proj

    def forward(self, hidden_states):
        xz = rearrange(self.in_proj(hidden_states), "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        x_cat = torch.cat([x, x.flip(dims=[-1])], dim=0)
        return x_cat, z


class _MambaInterScanBlock(nn.Module):
    """Everything between the conv1d call and the selective_scan_fn call:
    x_proj -> split -> dt_proj matmul -> rearrange/contiguous B,C -> A/D/
    delta_bias fp32 casts. No mamba CUDA-extension calls -- safe to
    torch.compile."""

    def __init__(self, x_proj, dt_proj, A_log, D, dt_rank, d_state):
        super().__init__()
        self.x_proj = x_proj
        self.dt_proj = dt_proj
        self.A_log = A_log
        self.D = D
        self.dt_rank = dt_rank
        self.d_state = d_state

    def forward(self, x_cat, seq_len):
        x_dbl = self.x_proj(rearrange(x_cat, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seq_len)
        B = rearrange(B, "(b l) n -> b n l", l=seq_len).contiguous()
        C = rearrange(C, "(b l) n -> b n l", l=seq_len).contiguous()
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        delta_bias = self.dt_proj.bias.float()
        return dt, A, B, C, D, delta_bias


class _MambaPostScanBlock(nn.Module):
    """Direction merge -> gate -> out_proj. No mamba CUDA-extension calls --
    safe to torch.compile."""

    def __init__(self, out_proj):
        super().__init__()
        self.out_proj = out_proj

    def forward(self, y_cat, z, b):
        y_fwd, y_bwd = y_cat[:b], y_cat[b:].flip(dims=[-1])
        y = rearrange(y_fwd + y_bwd, "b d l -> b l d")
        y = y * F.silu(rearrange(z, "b d l -> b l d"))
        return self.out_proj(y)


class JambaBiMambaBatchedCompiledMixer(nn.Module):
    """Identical math to JambaBiMambaBatchedMixer, but everything EXCEPT the
    two raw mamba CUDA-extension calls (causal_conv1d_fn, selective_scan_fn)
    is split into three torch.compile(mode="reduce-overhead")-wrapped
    blocks: pre-scan, inter-scan (between conv and scan), and post-scan. The
    two opaque kernel calls stay eager, never inside any compiled region --
    the profiler identified exactly this glue (in_proj/out_proj matmuls,
    rearranges, x_proj/dt_proj matmuls, contiguous() copies) as the largest
    remaining chunk of per-call overhead once _FFNBlock/_AttnTailBlock had
    already been compiled, so this is that same low-risk strategy pushed one
    level deeper.

    Uses full Mamba-1 fidelity: A_log is (d_inner, d_state), same as
    JambaBiMambaBatchedMixer -- this variant's speed comes purely from
    compiling the glue around the CUDA kernels, not from any capacity
    tradeoff.
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

        in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True,
        )
        x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n", d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        pre_block = _MambaPreScanBlock(in_proj)
        inter_block = _MambaInterScanBlock(
            x_proj=x_proj, dt_proj=self.dt_proj, A_log=self.A_log, D=self.D,
            dt_rank=self.dt_rank, d_state=self.d_state,
        )
        post_block = _MambaPostScanBlock(out_proj)
        self.pre_block = torch.compile(pre_block, mode="reduce-overhead")
        self.inter_block = torch.compile(inter_block, mode="reduce-overhead")
        self.post_block = torch.compile(post_block, mode="reduce-overhead")

    def forward(self, hidden_states):
        b, seq_len, _ = hidden_states.shape
        x_cat, z = self.pre_block(hidden_states)

        x_cat = self._causal_conv1d_fn(
            x=x_cat, weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
            bias=self.conv1d.bias, activation="silu",
        )
        dt, A, B, C, D, delta_bias = self.inter_block(x_cat, seq_len)
        y_cat = self._selective_scan_fn(
            x_cat, dt, A, B, C, D, z=None,
            delta_bias=delta_bias, delta_softplus=True,
        )
        return self.post_block(y_cat, z, b)


class _FFNBlock(nn.Module):
    """pre_ff_layernorm -> feed_forward (MLP) -> residual add, as one unit.
    Contains NO calls into the mamba CUDA extensions -- those live only in
    JambaBiMambaBatchedMixer, which is never wrapped here -- so torch.compile
    can safely trace and cudagraph-capture this block without hitting the
    FakeTensor crash the earlier whole-model attempt did (Dynamo never has to
    trace anything opaque; every op here is plain PyTorch)."""

    def __init__(self, feed_forward, pre_ff_layernorm):
        super().__init__()
        self.feed_forward = feed_forward
        self.pre_ff_layernorm = pre_ff_layernorm

    def forward(self, hidden_states):
        residual = hidden_states
        h = self.pre_ff_layernorm(hidden_states)
        h = self.feed_forward(h)
        return residual + h


class JambaBiMambaBatchedCompiledDecoderLayer(nn.Module):
    """Same as JambaBiMambaBatchedDecoderLayer, except the FFN half of the
    layer (pre_ff_layernorm + feed_forward + residual) runs as a
    torch.compile(mode="reduce-overhead")-wrapped _FFNBlock. The mamba half
    (input_layernorm + JambaBiMambaBatchedMixer + residual, which contains
    the two opaque CUDA-extension calls) stays fully eager, untouched --
    compilation is scoped to only the pure-PyTorch half of the layer."""

    def __init__(self, config, layer_idx):
        super().__init__()
        from transformers.models.jamba.modeling_jamba import JambaMLP, JambaRMSNorm

        self.mamba = JambaBiMambaBatchedMixer(
            d_model=config.hidden_size,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
        )
        self.input_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        ffn_block = _FFNBlock(
            feed_forward=JambaMLP(config),
            pre_ff_layernorm=JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps),
        )
        self.ffn_block = torch.compile(ffn_block, mode="reduce-overhead")

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.ffn_block(hidden_states)
        return hidden_states


class _AttnTailBlock(nn.Module):
    """attn_layer (non-causal) -> final_layernorm -> mean-pool, as one unit.
    Same rationale as _FFNBlock: no mamba CUDA-extension calls in this path,
    so it's safe to compile on its own."""

    def __init__(self, attn_layer, final_layernorm):
        super().__init__()
        self.attn_layer = attn_layer
        self.final_layernorm = final_layernorm

    def forward(self, hidden_states, position_ids):
        out = self.attn_layer(hidden_states, attention_mask=None, position_ids=position_ids)
        out = self.final_layernorm(out)
        return out.mean(dim=1)


class BidirectionalJambaBatchedCompiledGlue(nn.Module):
    """Same architecture as BidirectionalJambaBatched, but with the non-mamba
    glue (each layer's FFN half, plus the attention+norm+pool tail) wrapped
    in torch.compile(mode="reduce-overhead"). The mamba mixer calls
    themselves (causal_conv1d_fn/selective_scan_fn) are never inside any
    compiled region -- this is a surgical, low-risk retry of torch.compile
    after the earlier whole-model attempt crashed on FakeTensor tracing
    through those opaque CUDA-extension calls."""

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
            [JambaBiMambaBatchedCompiledDecoderLayer(config, layer_idx=i) for i in range(num_mamba_layers)]
        )
        attn_layer = JambaAttentionDecoderLayer(config, layer_idx=NUM_HIDDEN_LAYERS - 1)
        attn_layer.self_attn.is_causal = False
        final_layernorm = JambaRMSNorm(HIDDEN_SIZE, eps=config.rms_norm_eps)
        tail = _AttnTailBlock(attn_layer=attn_layer, final_layernorm=final_layernorm)
        self.tail = torch.compile(tail, mode="reduce-overhead")

    def forward(self, x):
        b, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)

        h = x
        for layer in self.mamba_layers:
            h = layer(h)

        return self.tail(h, position_ids)


class JambaBiMambaBatchedFullCompiledDecoderLayer(nn.Module):
    """Same as JambaBiMambaBatchedCompiledDecoderLayer, but the mamba half
    also uses JambaBiMambaBatchedCompiledMixer (pre/inter/post-scan blocks
    compiled, only the two raw kernel calls eager) instead of the fully-eager
    JambaBiMambaBatchedMixer. This is the "compile everything except the two
    opaque calls" strategy applied to the WHOLE layer, not just its FFN
    half -- the best-measured configuration before this file was trimmed to
    three models; brought back to re-verify that result now that the SSD
    scan's numbers are also in play."""

    def __init__(self, config, layer_idx):
        super().__init__()
        from transformers.models.jamba.modeling_jamba import JambaMLP, JambaRMSNorm

        self.mamba = JambaBiMambaBatchedCompiledMixer(
            d_model=config.hidden_size,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
        )
        self.input_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        ffn_block = _FFNBlock(
            feed_forward=JambaMLP(config),
            pre_ff_layernorm=JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps),
        )
        self.ffn_block = torch.compile(ffn_block, mode="reduce-overhead")

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.ffn_block(hidden_states)
        return hidden_states


class BidirectionalJambaBatchedFullCompiledGlue(nn.Module):
    """Same as BidirectionalJambaBatchedCompiledGlue, but every layer's mamba
    half is also compiled (pre/inter/post-scan blocks) via
    JambaBiMambaBatchedFullCompiledDecoderLayer -- the maximal surgical-
    compile variant: everything except the two raw mamba CUDA-extension
    calls (causal_conv1d_fn, selective_scan_fn) is inside some compiled
    region. Full Mamba-1 fidelity throughout (A_log is (d_inner, d_state)) --
    this is a pure overhead-removal experiment, not a capacity tradeoff."""

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
            [JambaBiMambaBatchedFullCompiledDecoderLayer(config, layer_idx=i) for i in range(num_mamba_layers)]
        )
        attn_layer = JambaAttentionDecoderLayer(config, layer_idx=NUM_HIDDEN_LAYERS - 1)
        attn_layer.self_attn.is_causal = False
        final_layernorm = JambaRMSNorm(HIDDEN_SIZE, eps=config.rms_norm_eps)
        tail = _AttnTailBlock(attn_layer=attn_layer, final_layernorm=final_layernorm)
        self.tail = torch.compile(tail, mode="reduce-overhead")

    def forward(self, x):
        b, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)

        h = x
        for layer in self.mamba_layers:
            h = layer(h)

        return self.tail(h, position_ids)



class BidirectionalViTCompiled(nn.Module):
    """Standard bidirectional (non-causal) Transformer encoder, matched in
    depth/width/mlp-ratio to the Jamba configs above, with the whole encoder
    wrapped in torch.compile(mode="reduce-overhead"). Unlike Jamba's mixer,
    ViT has no CUDA-extension calls to graph-break on -- every op is plain
    PyTorch -- so this needs no surgical splitting; the whole model compiles
    cleanly.

    This exists for fairness, not just completeness: any crossover-point
    claim comparing a compiled Jamba against an eager ViT is the same
    asymmetry that inflated the CUDA-graph crossover estimate earlier this
    session (JAMBA_OPTIMIZATION.md step 4 -- graphing only one side made the
    crossover look like it moved to ~L=196, which reverted to ~8-10k once
    ViT was graphed too). Whatever overhead-removal technique gets applied
    to one side of the comparison must get applied to both before any
    crossover number drawn from it is trustworthy.
    """

    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_SIZE,
            nhead=NUM_ATTENTION_HEADS,
            dim_feedforward=INTERMEDIATE_SIZE,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        encoder = nn.TransformerEncoder(layer, num_layers=NUM_HIDDEN_LAYERS)
        self.encoder = torch.compile(encoder, mode="reduce-overhead")

    def forward(self, x):
        out = self.encoder(x)
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
        except torch.cuda.OutOfMemoryError:
            print(f"L={seq_len:6d}  OOM")
            torch.cuda.empty_cache()
            break
    return results


def run_sweep_compiled_glue(name, model_fn):
    """Like run_sweep, but for models that internally wrap only PART of
    themselves in torch.compile (e.g. BidirectionalJambaBatchedCompiledGlue,
    which compiles the FFN block and attention tail but leaves the mamba
    mixer eager). No outer benchmark_compiled wrapper is needed -- the
    model's own submodules already carry their torch.compile wrapping -- but
    we still reset Dynamo's cache between seq_len values (fresh shapes need
    a fresh compile) and treat a compile failure as a skip, same as
    run_sweep_compiled did for the (riskier) whole-model attempt.
    """
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
        except Exception as e:  # noqa: BLE001 - torch.compile can raise many backend-specific error types
            print(f"L={seq_len:6d}  torch.compile failed: {e}")
            torch.cuda.empty_cache()
            torch._dynamo.reset()
            break
    return results


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA not available"

    # Three models:
    #  1. BidirectionalJambaBatched -- previous best under eager execution
    #     (batch-dim-stacked bidirectional Mamba-1 scan via mamba_ssm's CUDA
    #     kernels, single-pass attention). What real (non-compiled) training
    #     actually uses today.
    #  2. BidirectionalJambaBatchedFullCompiledGlue -- the maximal surgical-
    #     compile variant (mixer's pre/inter/post-scan blocks + FFN + attn
    #     tail all torch.compile'd, only the two raw CUDA kernel calls stay
    #     eager). Full Mamba-1 fidelity (A is (d_inner, d_state)), no
    #     capacity tradeoff -- this was the best-measured configuration
    #     before the file was trimmed down; restored to re-verify that
    #     result. The whole-model-compiled SSD scan (BidirectionalJambaSSDCompiled)
    #     was tried and removed: it underperformed this surgical variant
    #     (~16-17 it/s vs. this) despite having zero opaque CUDA calls, so
    #     full compile coverage of the scan itself isn't worth the loss of
    #     the fused CUDA kernel's throughput.
    #  3. BidirectionalViTCompiled -- fair baseline: same overhead-removal
    #     treatment (whole-model torch.compile) applied to ViT, so any
    #     crossover-point claim isn't the asymmetric-compile mistake from
    #     earlier in this project (JAMBA_OPTIMIZATION.md step 4).
    batched_results = run_sweep(
        "Jamba (Mamba-1, BATCHED bidirectional scan, single-pass attention) -- previous best (eager)",
        BidirectionalJambaBatched,
    )
    full_compiled_glue_results = run_sweep_compiled_glue(
        "Jamba (Mamba-1, BATCHED scan, torch.compile SURGICAL on mixer+FFN+tail) -- restored",
        BidirectionalJambaBatchedFullCompiledGlue,
    )
    vit_compiled_results = run_sweep_compiled_glue(
        "ViT-style Transformer (torch.compile reduce-overhead, whole model)",
        BidirectionalViTCompiled,
    )

    print("\n=== Summary ===")
    print(
        f"{'L':>8} | {'Batched(eager) it/s':>19} {'GB':>8} | "
        f"{'mixer+FFN+tail compiled it/s':>29} {'GB':>8} | "
        f"{'ViT compiled it/s':>18} {'GB':>8}"
    )
    for seq_len in SEQ_LENGTHS:
        ba = batched_results.get(seq_len)
        fc = full_compiled_glue_results.get(seq_len)
        vc = vit_compiled_results.get(seq_len)
        ba_str = f"{ba[0]:19.2f} {ba[1]:8.2f}" if ba else f"{'--':>19} {'--':>8}"
        fc_str = f"{fc[0]:29.2f} {fc[1]:8.2f}" if fc else f"{'--':>29} {'--':>8}"
        vc_str = f"{vc[0]:18.2f} {vc[1]:8.2f}" if vc else f"{'--':>18} {'--':>8}"
        print(f"{seq_len:8d} | {ba_str} | {fc_str} | {vc_str}")
