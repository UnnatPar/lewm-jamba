"""Bidirectionality inside each mamba mixer instead of around the whole stack.

The encoder ran the entire 10-layer Jamba stack twice -- once forward, once on the flipped
sequence -- and summed. That buys bidirectionality at 2x the cost of everything, but only two
components of a mamba layer are direction-sensitive: the causal conv1d and the selective scan.
At hidden=288/expand=1 those are 4.8% of a layer's MACs. The MLP (76%), in_proj (13%) and
out_proj (6%) are position-wise or direction-agnostic and were being computed twice for nothing.

So: run in_proj once, scan both directions with shared weights, sum, run out_proj and the MLP
once. FLOPs drop 1.91x. Zero parameters added or removed -- every weight is shared between the
two directions, so a parameter-matched comparison stays honest.

This is also the standard bidirectional Mamba formulation (Vision Mamba, arXiv 2401.09417) and
is arguably a better function than what it replaces: every layer combines both directions before
feeding the next, rather than two independent unidirectional streams meeting only at the end.

The two directions are additionally batched into ONE set of kernel launches (cat along batch,
split after) rather than two sequential ones, which matters because the scan is launch-heavy.

Note conv1d is CAUSAL, so conv(flip(x)) != flip(conv(x)). The reverse branch has to run the
convolution on the flipped input, not flip a forward convolution -- getting this wrong produces
a model that trains fine and is subtly not bidirectional.

Applied per INSTANCE, not by patching the class. A class-level monkeypatch is global, so an
"outer" encoder built after an "inner" one in the same process would get bidirectionality twice
-- silently, and with a plausible-looking loss curve. That is precisely the situation an
inner-vs-outer comparison creates, so the mechanism must not be able to leak between models.
"""

import types

import torch
from transformers.models.jamba import modeling_jamba as mj


def _scan_both(self, x, gate, use_norms=True):
    """conv -> x_proj -> dt -> selective_scan, on (2B, d_inner, L) holding both directions.

    use_norms=False drops Jamba's dt/B/C RMSNorms, matching what the fused kernel computes.
    That exists so the fused path can be tested for exact equivalence against an unfused
    reference -- otherwise a fusion bug and the intended function change are indistinguishable.
    """
    conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
    x = mj.causal_conv1d_fn(x, conv_weights, self.conv1d.bias, activation=self.activation)

    ssm_parameters = self.x_proj(x.transpose(1, 2))
    time_step, B, C = torch.split(
        ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)
    if use_norms:
        time_step = self.dt_layernorm(time_step)
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)

    # Upstream swaps dt_proj.bias to zero and back under no_grad on every call, purely to reuse
    # nn.Linear.forward for quantization support. That is two extra device ops per layer per
    # step and we do not quantize, so call F.linear directly with the bias left out.
    discrete_time_step = torch.nn.functional.linear(
        time_step, self.dt_proj.weight).transpose(1, 2)

    A = -torch.exp(self.A_log.float())
    # Upstream passes z=gate and return_last_state=True. BOTH are bail-out conditions in
    # sscanop2's shim, so the chunk-padding optimisation -- the one adopted win of the whole
    # optimisation effort -- was dead code in the real encoder and only ever ran in `bench`.
    # We discard the last state anyway (an encoder carries nothing between calls), and the
    # gate is a plain elementwise `out * silu(z)` that the kernel fuses purely for convenience.
    # Applying it outside costs one memory-bound elementwise pass and buys the fast path.
    # `test_bidir.py` asserts manual gating equals the fused gating exactly.
    scan_out = mj.selective_scan_fn(
        x, discrete_time_step, A, B.transpose(1, 2), C.transpose(1, 2), self.D.float(),
        None, self.dt_proj.bias.float(), delta_softplus=True, return_last_state=False)
    return scan_out * torch.nn.functional.silu(gate)


def bidirectional_forward(self, hidden_states, cache_params=None, attention_mask=None, **kw):
    # HF allocates a cache object even in training, so a non-None cache is normal and means
    # nothing on its own -- upstream only takes the cached path when the cache actually holds
    # prior state AND seq_len == 1. That is incremental decode, which cannot be bidirectional,
    # so fail loudly there instead of silently returning a different function. Otherwise
    # ignore the cache: an encoder has nothing to carry between calls.
    if (cache_params is not None and hidden_states.size(1) == 1
            and getattr(cache_params, "has_previous_state", lambda _i: False)(self.layer_idx)):
        raise NotImplementedError(
            "bidir: incremental decode is not supported -- a cached single-step forward "
            "cannot be bidirectional.")

    projected_states = self.in_proj(hidden_states).transpose(1, 2)
    x, gate = projected_states.chunk(2, dim=1)
    if attention_mask is not None:
        x = x * attention_mask.unsqueeze(1)

    n = x.size(0)
    both = _scan_both(self,
                      torch.cat([x, x.flip(-1)], 0).contiguous(),
                      torch.cat([gate, gate.flip(-1)], 0).contiguous())
    fwd, bwd = both[:n], both[n:]
    return self.out_proj((fwd + bwd.flip(-1)).transpose(1, 2))


def bidirectional_fused_forward(self, hidden_states, cache_params=None, attention_mask=None,
                                **kw):
    """Same bidirectional structure, but each direction goes through mamba_inner_fn.

    mamba_inner_fn fuses conv -> x_proj -> dt_proj -> scan -> gate -> out_proj into one kernel.
    Upstream cannot use it because Jamba's dt/B/C RMSNorms sit in the middle of that chain, so
    this path DROPS those three norms (450 parameters across the 9 mixers, 0.003% of the model).
    That strictly enlarges the function class -- an RMSNorm is a constraint, and no setting of
    its weights recovers un-normalised dt/B/C magnitudes -- but it removes the stabiliser Jamba
    added for loss spikes, so watch for NaNs early rather than trusting it.

    out_proj is inside the fused kernel, so it runs per direction rather than once: duplicated
    work goes from ~4.8% to ~11% of a layer. out_proj is linear, so summing after it is exactly
    equivalent to summing before. in_proj and the MLP still run once.

    Note this path cannot use sscanop2's chunk padding -- mamba_inner_fn has its own fused CUDA
    path and never routes through selective_scan_fn. Measured as within noise at 1,960 tokens,
    but it does make the two optimisations mutually exclusive rather than additive.
    """
    if (cache_params is not None and hidden_states.size(1) == 1
            and getattr(cache_params, "has_previous_state", lambda _i: False)(self.layer_idx)):
        raise NotImplementedError("bidir: incremental decode cannot be bidirectional.")

    xz = self.in_proj(hidden_states).transpose(1, 2)
    n = xz.size(0)
    y = mj.mamba_inner_fn(
        torch.cat([xz, xz.flip(-1)], 0).contiguous(),
        self.conv1d.weight, self.conv1d.bias,
        self.x_proj.weight, self.dt_proj.weight,
        self.out_proj.weight, self.out_proj.bias,
        -torch.exp(self.A_log.float()),
        None, None,                       # B and C come from x_proj inside the kernel
        self.D.float(),
        delta_bias=self.dt_proj.bias.float(),
        delta_softplus=True,
    )                                     # (2B, L, hidden_size)
    return y[:n] + y[n:].flip(1)


def alternating_fused_forward(self, hidden_states, cache_params=None, attention_mask=None, **kw):
    """One direction per layer, alternating by depth, instead of both in every layer.

    Every measured lever has now been exhausted except the doubling itself: the fused mixer runs
    at batch 2B so conv, x_proj, dt_proj, scan, gate and out_proj all happen twice per layer, and
    that machinery is ~49% of step time (deleting it entirely reaches 1.258x the matched ViT).

    Alternating direction by depth keeps the STACK bidirectional -- even layers scan forward, odd
    layers backward, so information reaches every position from both sides through depth -- while
    no single layer pays for two scans. Zero parameters change. It is weaker per layer than
    per-layer bidirectionality and stronger than a purely causal encoder.

    Whether the weaker per-layer coupling matters is an empirical question, and one worth asking
    given the measurement that a single mixer at random init has no coupling past ~128 tokens in
    EITHER direction anyway -- cross-frame mixing already has to come from depth, not from one
    layer's scan reaching across a 196-token frame boundary.
    """
    if (cache_params is not None and hidden_states.size(1) == 1
            and getattr(cache_params, "has_previous_state", lambda _i: False)(self.layer_idx)):
        raise NotImplementedError("bidir: incremental decode cannot be bidirectional.")

    xz = self.in_proj(hidden_states).transpose(1, 2)
    backward = (self.layer_idx % 2) == 1
    if backward:
        xz = xz.flip(-1)
    y = mj.mamba_inner_fn(
        xz.contiguous(), self.conv1d.weight, self.conv1d.bias,
        self.x_proj.weight, self.dt_proj.weight,
        self.out_proj.weight, self.out_proj.bias,
        -torch.exp(self.A_log.float()), None, None, self.D.float(),
        delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
    )
    return y.flip(1) if backward else y


def forward_only_fused_forward(self, hidden_states, cache_params=None, attention_mask=None,
                               **kw):
    """Causal encoder. Not a candidate -- it is the bound that says how much the second
    direction costs, and an encoder that cannot see forward is a capability loss."""
    xz = self.in_proj(hidden_states).transpose(1, 2)
    return mj.mamba_inner_fn(
        xz.contiguous(), self.conv1d.weight, self.conv1d.bias,
        self.x_proj.weight, self.dt_proj.weight,
        self.out_proj.weight, self.out_proj.bias,
        -torch.exp(self.A_log.float()), None, None, self.D.float(),
        delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
    )


def apply_to(model, impl="unfused"):
    """Bind the bidirectional mixer forward onto every JambaMambaMixer in `model`.

    Per-instance, so two encoders with different bidir_mode can coexist in one process.
    Returns the number of mixers patched -- callers should assert it is nonzero, since a
    silent zero would mean a unidirectional encoder that still trains and still looks fine.
    """
    fn = {"unfused": bidirectional_forward,
          "fused": bidirectional_fused_forward,
          "alternate": alternating_fused_forward,
          "forward_only": forward_only_fused_forward}[impl]
    n = 0
    for m in model.modules():
        if isinstance(m, mj.JambaMambaMixer):
            m.cuda_kernels_forward = types.MethodType(fn, m)
            n += 1
    return n
