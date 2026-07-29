# Multi-frame Jamba encoder — what changes, and the one thing that would break silently

Measured 2026-07-29. Window size: **10 frames = 1,960 tokens**, the largest multiple of 196 that
fits under the scan's 2,048 chunk boundary.

> **RETRACTED 2026-07-29 (later the same day): "Jamba beats the matched ViT by 11% there" is
> false for the encoder that actually trains.** That figure came from a bare mixer on synthetic
> hidden states under `torch.compile`. Measured with the real `module.JambaEncoder` against a
> parameter-matched ViT (`crossover2.py`), Jamba loses at every window from 784 to 2,352 tokens:
> **0.572x at expand=1, 0.430x at expand=2**, and the ratio is flat in sequence length, meaning
> the crossover is far beyond anything measured. `torch.compile` makes it worse, not better.
> See the ledger entry. **The efficiency claim in §6 is withdrawn.** The 2,048-chunk reasoning
> for choosing 10 over 9 or 11 frames still holds; the "Jamba wins here" reasoning does not.

---

**Status 2026-07-29: §0, §1, §2 are built and committed (`0a30abf`). §3 partly. §4, §5 open.**
The leakage fix in §0 is *not* the block-causal design originally written here — see the
revision note below.

---

## 0. The flaw to decide first: temporal leakage

The encoder today is **per-frame**. `jepa.py:encode()` does
`rearrange(pixels, "b t ... -> (b t) ...")`, so `emb[t]` is a function of frame `t` alone.
`train.py:lejepa_forward` then does:

```python
ctx_emb = emb[:, :ctx_len]      # what the predictor sees
tgt_emb = emb[:, n_preds:]      # what it must predict
pred_emb = self.model.predict(ctx_emb, ctx_act)
output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
```

**If the encoder becomes bidirectional over a 10-frame window, `emb[t]` sees frames `t+1…`, and
`ctx_emb` therefore already contains the answer.** `pred_loss` collapses toward zero without the
model learning any dynamics. Nothing raises. The loss curve looks excellent.

SIGReg does not catch this. It constrains the *marginal* distribution of embeddings to be
isotropic Gaussian; a representation that has leaked the future satisfies that just as well as
one that has not. And there is no EMA target or stop-gradient here to break the shortcut —
`tgt_emb` comes from the same tensor as `ctx_emb`, undetached.

This is the "runs fine, result is meaningless" failure class, and it is the single highest risk
in this change.

### REVISED — the adopted fix is sliding windows, not block-causal

The block-causal design below was **not built**. It buys leak-freedom by forbidding any frame
from seeing a later one, which throws away exactly the cross-frame bidirectionality that
motivates a window encoder at all. The adopted design keeps full bidirectionality inside each
window and gets leak-freedom from the *window layout* instead:

- `T = 14` frames per sample; windows of `W = 10` sliding by `stride = 1` → **5 windows**.
- Window `k` covers frames `k … k+9`. It contributes **one** timestep: its **last slot**, i.e.
  frame `k+9` seen through frames `k … k+9`.
- `ctx = emb[:, :4]` (newest frames 9, 10, 11, 12), `tgt = emb[:, 1:]` (newest frames 10, 11,
  12, 13). The existing slicing in `train.py` is unchanged.
- No context window contains the frame its target predicts: window `k` ends at frame `k+9`, its
  target is frame `k+10`. Verified exhaustively, not argued — `test_multiframe.py` perturbs each
  of the 14 frames and checks all 70 (frame, timestep) dependency pairs.

There is no context-free "frame `t` embedding" anywhere in this design. Frame 10 as seen through
`[1:11]` is a different tensor from frame 10 as seen through `[2:12]`. That is intended: the
axiom being tested is that joint encoding yields a richer representation than encoding ten
frames separately and concatenating.

**Why the last slot and not a mean-pool over the window.** A window mean-pool would make the
target 1/10th about the frame being predicted and 9/10ths about frames the context already
holds. The last slot keeps the target centred on the novel frame while staying in the same
space as the context, which is what makes rollout type-correct: the predictor is supervised
against contextualised window vectors, so it learns to emit them.

<details>
<summary>Original block-causal proposal, kept for the record — not implemented</summary>

### The fix: block-causal, not bidirectional

Bidirectional **within** a frame, causal **across** frames. `emb[t]` may see all 196 tokens of
frames `≤ t` and nothing of `t+1`. Then:

- every frame keeps full spatial bidirectionality — the property that motivated this encoder;
- `tgt_emb = emb[t+1…]` depends on frame `t+1`, which `ctx_emb` never saw, so the existing loss
  in `lejepa_forward` stays correct **unchanged**;
- the encoder gains real temporal context, which is the point of the change.

It is also nearly free to implement, because of how the two scans already work:

| component | today | block-causal |
|---|---|---|
| forward Mamba scan | over 196 tokens | over all 1,960 — causal by construction, carries history across frames |
| reverse Mamba scan | over 196 tokens | **reshape to (B*10, 196, D)**, flip, scan, flip back — stays inside its own frame |
| attention layer (1 of 10) | non-causal over 196 | block-causal mask: frame `f` attends to frames `≤ f`, fully within them |

The reverse scan restriction is a `reshape`, not a kernel change. (It is the same locality idea
as LBMamba, arXiv 2506.15976 — rejected earlier on speed as a kernel optimisation, required here
for correctness.)

</details>

### The residual risk neither design removes

At stride 1, `ctx[k]` and `tgt[k]` share **9 of their 10 frames**. Most of the target is already
inside the context, so a predictor can score well by carrying the shared mass forward and
ignoring the one novel frame. The loss falls, the model is useless, nothing raises.

`train.py` therefore logs `copy_loss` (what "predict no change" scores) and `copy_ratio =
copy_loss / pred_loss` on every step. **`copy_ratio > 1` is the only evidence that any dynamics
were learned.** A falling `pred_loss` on its own is equally consistent with a well-regularised
identity function, and must not be reported without the ratio beside it.

### Deferred experiment: disjoint windows (stride = W)

Stride 1 makes the prediction *easier* than the current per-frame task, since context and target
overlap 90%. Striding by `W` — windows `[0:10], [10:20], [20:30], [30:40]` — makes every
target's input fully disjoint from its context's, turning the task into "given this clip and
these actions, what does the next clip look like." Genuinely harder, no copy path, and it makes
the goal-as-a-clip cost in §4.1 fall out naturally.

Cost: 5 windows either way, but `num_steps` goes 14 → 50, which at `frameskip: 5` is 250 raw
environment steps — longer than a PushT episode (~120 frames). Would need `frameskip` reduced
for the within-window spacing. **Held as the next experiment, not folded into this one**;
`window_stride` is already a config knob, so it costs one line to try.

---

## 1. `module.py:JambaEncoder` — BUILT (`0a30abf`)

All of the below is done. Params went 15,495,330 → 15,497,922 (+0.017%, the frame embedding
alone), so capacity is unchanged. `mamba_expand` was left at the config's **2**, not dropped to
the measured 1 — reducing it would have cut capacity silently. `mfprobe.py` re-measures the
window at expand=2 rather than assuming 1,960 still holds.

- accept `(N, F, C, H, W)`; patch-embed per frame; concatenate to `(N, F*196, D)`
- **two position embeddings**: the existing patch embedding (196) plus a new **frame** embedding
  (F). Without the second, the stack cannot tell frame boundaries apart.
- `_interpolate_pos_encoding` assumes a square grid via `int(num_positions ** 0.5)`. At 1,960
  tokens that is 44.27 and it silently produces a wrong grid. Must operate on the per-frame
  patch grid only.
- `max_position_embeddings=num_patches` → `F * num_patches`.
- **output `(N, F, output_dim)`** — mean-pool within each frame's own 196-token span, not over
  the whole window. Pooling the window would collapse temporal resolution by 10x and change what
  the predictor predicts.
- replace the naive double pass (`self.jamba(...)` twice, lines 446–447) with the batched
  bidirectional mixer from `benchmark_seq_scaling.py`, and call `sscanop2.register()` +
  `PAD_SCAN=True` at construction.

**Config mismatch to reconcile:** `module.py` uses `mamba_expand=2`; every crossover measurement
was taken at `mamba_expand=1` with `hidden_size=288`. At expand=2 the scan cost doubles and 1,960
is no longer the right window. Port at expand=1, or re-measure. Do not assume.

## 2. `jepa.py:encode()` — BUILT (`0a30abf`)

Resolved by keeping the `_EncoderOutput` contract and reading `last_hidden_state[:, -1]` instead
of `[:, 0]`. With `window_size=1` those are the same slot, so `ConvEncoder` and the ViT stay
drop-in and the old path is reproduced byte-identically (test 5).

Cannot flatten `(b t) -> ...` any more — that is exactly what makes the encoder per-frame. It
must hand the encoder a window and receive `(B, F, D)`. `last_hidden_state[:, 0]` as the
"CLS slot" contract goes away; either return `(N*F, 1, D)` to preserve it, or update both call
sites. Preserving it is less invasive and keeps `ConvEncoder` a valid drop-in.

## 3. Data

`T` per sample must be at least `F + ctx_len + n_preds`, contiguous, with `frameskip` respected.
Window sampling has to be added; today the loader only needs enough frames for the predictor.

**Memory is the binding constraint, not speed.** Measured at batch 16, 10 frames: **5.68 GB** for
the encoder alone. Batch 128 extrapolates past 45 GB — over an A100-40GB before the predictor,
projector and targets are counted. Expect real batch 8–16 windows plus gradient accumulation, and
note that 16 windows is already 160 frames, so it is not as small as it sounds.

## 4. The cost function — `jepa.py:criterion()`

Today: MSE against the goal embedding at the **last step only**.

```python
cost = F.mse_loss(pred_emb[..., -1:, :], goal_emb[..., -1:, :].detach(), reduction="none")
```

Multi-frame encoding makes a stricter cost *expressible*, and §0 makes it *necessary*:

1. **Goal as a window, not a still.** With a 10-frame encoder the goal can be a short clip, so
   the cost measures reaching a goal *behaviour* (pose **and** motion) instead of a static pose.
   This is the change multi-frame actually justifies — it is not available to a per-frame encoder.
2. **Whole-trajectory, not endpoint.** Score every step of the rollout against the goal window,
   not just `[-1:]`. Endpoint-only cost is indifferent to how you got there.
3. **Scale-invariant distance.** Raw summed MSE rewards shrinking embedding norms. Cosine or
   per-dimension standardised distance removes that.
4. **Report against the copy baseline.** Always log the cost of the "predict no change"
   trajectory alongside the model's. If the model does not beat it by a clear margin, the
   §0 residual shortcut has happened and the number means nothing.

(4) is not optional decoration. Without it there is no way to distinguish a working world model
from a well-regularised identity function.

## 5. Rollout and eval

`rollout()` encodes the initial observation once and then runs the predictor autoregressively in
embedding space. Block-causal encoding is consistent with this (`emb[t]` needs only frames `≤ t`),
but the initial encode now needs a full `F`-frame window rather than a single frame, and
`history_size=3` interacts with `F`. `eval.py`'s `plan_config.horizon`, `action_block` and
`goal_offset_steps` all assume single-frame goals and need revisiting against §4.1.

## 6. The honesty check on all of this

Joint 10-frame encoding is **not** justified against the current pipeline on speed. Ten separate
196-token encodes are linear in frames; one 1,960-token encode is not cheaper by construction.
The crossover result says Jamba beats **a ViT doing the same joint encoding** — it does not say
joint encoding beats per-frame encoding.

~~What makes the change defensible on cost is a separate measured fact: short sequences waste the
GPU. At batch 16, ViT at L=196 sustains ~362k tokens/s, while Jamba at L=1,960 sustains ~502k —
so joint 10-frame Jamba encoding is roughly 1.4x more throughput per frame than the per-frame
pipeline it replaces.~~

**Withdrawn 2026-07-29.** Those throughput figures are from the bare mixer, not the real
encoder. Measured on the real thing (`mfprobe.py`, full JEPA training step): **0.53 hr/epoch,
53 hours for the configured 100 epochs**, against a per-frame baseline that trained in a few
hours. The change is roughly **15–25x more expensive**, not 1.4x cheaper.

**So the justification is capability alone: temporal context inside the representation.** There
is no efficiency argument for this architecture, and none should be made in the writeup. Whether
the capability is worth 15–25x is an empirical question that `copy_ratio` and downstream planning
performance have to answer — it cannot be argued from the cost side.

Any comparison reported after this change must put the ViT baseline on the same window size.
