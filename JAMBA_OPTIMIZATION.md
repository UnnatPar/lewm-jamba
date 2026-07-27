# Jamba Encoder Optimization — Reference Notes

This documents every architectural/engineering change that took the Jamba-vs-ViT
crossover point (the sequence length past which Jamba's bidirectional encoder
becomes cheaper than an equivalent ViT) from **~15,000-20,000 tokens down to
~4,000-6,000 tokens** — a ~3-4x improvement — measured in
`benchmark_seq_scaling.py`. Hydra (the quasiseparable-matrix-mixer variant) is
excluded here since it didn't beat this configuration; see that file's
`JambaHydraMixer` docstring if you want the details on that attempt.

None of this is free lunch: real training sequences per frame are ~196-392
tokens, well below even the optimized ~4-6k crossover. At L=196, this
optimized Jamba is still ~2-2.5x *slower* than ViT in raw wall-clock terms —
these changes only matter if/when sequences get long (concatenated frames,
video, etc.), not for the current per-frame encoder as configured.

---

## Starting point: naive bidirectional Jamba

`BidirectionalJamba` — bidirectionality achieved by running the **entire**
10-layer decoder stack twice (once forward, once on the reversed sequence,
then summed), including the single attention layer. Per forward pass: 18
Mamba-layer calls + 2 attention-layer calls = 20 full decoder-layer calls.

Crossover vs. ViT: **~15,000-20,000 tokens**.

---

## 1. Merge-before-attention (single-pass attention)

**What changed:** Only the 9 linear-cost Mamba layers are run twice
(forward + reversed, summed). The single quadratic-cost attention layer runs
**once**, non-causally, over the already-merged Mamba output — not once per
direction.

**Why it works:** Per the crossover algebra `n* = C_m/C_a` (Mamba's
per-token kernel constant over attention's per-token-pair constant), running
attention twice doesn't change *where* the crossover is, only how steep the
win is past it — but it does cost real, avoidable compute. Removing that
redundant attention pass is a free win with no capacity cost.

**Implementation detail that mattered:** achieving the non-causal bidirectional
pass requires `attention_mask=None` + `self_attn.is_causal=False`, which
triggers SDPA's maskless bidirectional fast path (`O(L)` memory, flash-attention
style). Passing an *explicit* all-visible `(L, L)` mask tensor instead looks
equivalent but is NOT — it forces the backend off the fast path entirely and
materializes a dense `(L, L)` attention matrix, which is *more* memory than
two causal fast-path passes. This was a real regression we hit and had to
fix (OOM at L=25088 that used to run at L=50176).

**Effect in isolation:** ~2x reduction in the attention layer's contribution
to n* (theoretical ~2.25x, actual measured ~10%, because Jamba only has 1/10
attention layers to begin with — there wasn't much headroom).

---

## 2. Fused-bidirectional-call Mamba layer (halve the call count)

**The insight:** The naive double-pass doesn't just double *compute* — it
doubles the **number of Python/kernel-launch calls**: 18 full decoder-layer
calls (`in_proj`, conv, scan, `out_proj`, gating, residual add, RMSNorm, MLP —
ALL of it — run twice per layer). Diagnostic evidence: Jamba's it/s was
*flat* from L=196 to L=12544 (~8-9 it/s), while ViT's it/s declined smoothly
(genuine O(L²) cost). A flat-regardless-of-L curve is the signature of
**fixed per-call overhead dominating**, not compute cost. A `BidirectionalJambaNoAttn`
control (attention layer removed entirely) confirmed attention wasn't the
bottleneck — the Mamba layers' own call overhead was.

**What changed:** `JambaMamba2BiDecoderLayer` calls the underlying scan
kernel **twice** (forward + reversed) *inside a single layer call*, then
applies residual/RMSNorm/FFN **once** — not by looping the whole decoder
layer twice externally. Total Mamba-layer calls: 9, not 18.

(An earlier Mamba-1 version of this same idea, `JambaBiMambaMixer`, went
further and shared `in_proj`/`out_proj`/gating between directions too,
duplicating only the direction-dependent conv+scan — following Vision
Mamba's "bimamba_v2" pattern. Superseded once Mamba-2's SSD kernel proved
faster overall.)

**Effect in isolation:** ~2x it/s at low-to-mid L, ~2x reduction in peak
memory (half as many duplicate activation buffers to save for backward).
Crossover: ~15-20k → ~8-10k.

---

## 3. Mamba-2 SSD kernel instead of Mamba-1 selective-scan

**What changed:** Swapped the scan kernel itself — `mamba_chunk_scan_combined`
(Mamba-2's structured state-space duality / chunked-matmul formulation)
instead of `selective_scan_fn` (Mamba-1's sequential scan) — while keeping
the same fused-bidirectional-call structure from step 2.

**Why it matters:** This is the one change that alters the actual
**asymptotic slope** (`C_m`, Mamba's per-token cost), not just a constant
multiplier in front of the crossover ratio. The SSD paper's own claim
(beats FlashAttention-2 from L~2000) held up once measured fairly — Mamba-2
loses to Mamba-1 at low L (more per-call overhead) but wins from ~L=1568
onward, with the gap widening as L grows (13.9 → 17.5 it/s at L=12544 for
Mamba-2 vs Mamba-1, under CUDA graphs).

**Important trap avoided:** An earlier, *unfused* attempt at Mamba-2 (double
full-layer-pass, same overhead-bound structure as the original naive Jamba)
looked *worse* than Mamba-1. That result was overhead noise, not a real
reflection of the SSD kernel's cost — it had never been given the fused-call
treatment. Fair comparison required giving Mamba-2 the exact same
call-count-halving treatment Mamba-1 got first.

---

## 4. CUDA graph capture

**What changed:** Wrapped the model's forward+backward in `torch.cuda.graph()`
and replay the captured graph for timed iterations, instead of calling the
model eagerly each iteration.

**Why it matters:** Even after halving the call count (step 2), 9 sequential
layer calls still pay Python dispatch + kernel-launch overhead per call.
CUDA graphs eliminate nearly all of that — replaying pre-captured kernels
skips Python entirely. Effect was dramatic: **~10-17x** it/s improvement at
low L (e.g. Jamba-1-Fused went from 16.4 it/s to 156.9 it/s at L=196).

**Critical methodology point:** graphing *only* one side of a comparison is
not a fair test — ViT has its own per-layer launch overhead too. The first
graphed-Jamba-vs-eager-ViT comparison looked like a total blowout (crossover
"at L=196"), but that was an artifact of comparing an overhead-free Jamba to
an overhead-full ViT. Once ViT was *also* graphed, the crossover reverted to
~8-10k — essentially unchanged from the ungraphed fused-mixer result. CUDA
graphs rescale both curves by roughly the same multiplicative factor; they
don't change the underlying `C_m/C_a` ratio, since they only remove overhead
that both architectures pay.

**Implementation notes:**
- Requires a warm-up pass on a side `torch.cuda.Stream()` before capture (lets
  cuDNN/cuBLAS autotuning and lazy kernel compilation happen outside the graph).
- `zero_grad(set_to_none=True)` right before capture, so the first backward
  call *inside* the capture allocates the `.grad` buffers that become the
  graph's fixed output addresses for every subsequent replay.
- Correctness of accumulated gradient *values* across replays is irrelevant
  for this benchmark (speed/memory only, not a real training loop).

---

## 5. `chunk_size` tuning for the SSD kernel

**What changed:** `mamba_chunk_scan_combined`'s `chunk_size` parameter
(how the sequence gets blocked into matmul-friendly tiles) was left at the
library default (256) through all of the above. Swept
`[64, 128, 256, 512, 1024]` under CUDA-graph capture (the fairest,
least-overhead-contaminated measurement available):

- **128 and 64 tied for best** (chosen 128 as the new default)
- **256 (library default) was consistently ~5-15% slower**
- **512 and 1024 were clearly worse** — oversized chunks waste GPU
  parallelism on padding-heavy blocks at low-to-mid L, where the sequence
  isn't much longer than the chunk itself.

Modest but real further improvement stacked on top of everything above.

---

## Net result

| Stage | Crossover vs. ViT |
|---|---|
| Naive double-pass (`BidirectionalJamba`) | ~15,000-20,000 |
| + merge-before-attention (single-pass attn) | ~13,500-15,000 |
| + fused-bidirectional-call Mamba layers | ~8,000-10,000 |
| + Mamba-2 SSD kernel (fused, fair comparison) | ~4,000-6,000 |
| + CUDA graphs (both sides, fair) | ~4,000-6,000 (unchanged — overhead removal is not asymptotic) |
| + tuned `chunk_size=128` | modest improvement on top of the above |

**~3-4x reduction overall**, entirely from genuine architectural/kernel
changes — the CUDA-graph step proved decisive for *low-L absolute
throughput* (10-17x) but, being pure overhead removal, didn't move the
crossover point itself once both sides were measured fairly.

**Correction (later session, eager-only re-measurement):** the Mamba-2 SSD
row above was never actually validated eagerly — see step 6 below.
Re-running everything under real (non-CUDA-graphed) execution, the only
configuration that consistently beat Mamba-1 `Fused` was the batched-
direction scan (step 7), not Mamba-2. Treat the crossover numbers in this
table as CUDA-graph-only estimates, not what real training sees.

---

## 6. Mamba-2 SSD dropped after eager re-measurement

A later session re-ran the full sweep under eager (real-training) execution
instead of CUDA-graph capture, since real training never runs inside a
captured graph. Result: `BidirectionalJambaMamba2Fused`'s eager it/s
(~7.3-7.5, flat across L) was *worse* than Mamba-1 `Fused` (~17-18.5) at
every tested L, and never even beat eager ViT in the tested range — its
earlier "win" in step 3 only showed up once both were CUDA-graph-captured.
CUDA graphs rescale both curves' overhead constants but don't reflect real
training, which runs eager. `BidirectionalJambaMamba2Fused` and
`JambaMamba2BiDecoderLayer` were removed from `benchmark_seq_scaling.py`.
Mamba-1 `Fused` (`BidirectionalJambaFused` / `JambaBiMambaMixer`) became the
new current-best baseline, evaluated eagerly from here on.

Two further kernel-level ideas were tried against that eager baseline and
both failed to beat it, so both were removed after being tested:

- **`torch.compile(mode="reduce-overhead")`** — eager-training-compatible
  (unlike raw CUDA graph capture), applied fairly to both Jamba `Fused` and
  ViT. ViT benefited (fully Dynamo-traceable). Jamba did not: `Fused`'s
  compiled it/s came back essentially unchanged from eager, just with a
  lower memory footprint. Root cause: `selective_scan_cuda.fwd` /
  `causal_conv1d` are pybind11 C++ extensions, not registered PyTorch custom
  ops, so Dynamo graph-breaks around every call — torch.compile only
  optimizes the non-kernel glue code (residual adds, RMSNorm, FFN, gating)
  for Mamba, never the actual scan/conv kernels.
- **`mamba_inner_fn`** (mamba_ssm's own fully-fused Mamba-1 kernel — conv,
  x_proj, dt_proj, scan, D-skip, z-gating, and out_proj all inside one CUDA
  call chain) — eager it/s came in at 14.3-14.6, *worse* than `Fused`'s
  17-18.3, and `checkpoint_lvl=0` (disabling the backward-pass recompute of
  conv1d_out/delta that `checkpoint_lvl=1` does by default) barely moved
  that (~15.0-15.4). The real cost wasn't the recompute — it was that
  `mamba_inner_fn` bakes `out_proj` inside the fused kernel, so it can't
  share a single `in_proj`/`out_proj` across both directions the way
  `JambaBiMambaMixer` does; each direction pays for its own full linear
  layers. Kernel fusion that trades away weight-sharing lost to hand-chained
  ops that keep it.

---

## 7. Batched-direction bidirectional scan (share conv/scan weights too, one call for both directions)

**The insight from step 6:** `mamba_inner_fn` lost precisely because it
traded weight-sharing for kernel fusion. The lever that actually worked back
in step 2 (`Fused`) was call-count reduction *without* losing weight-sharing
(`in_proj`/`out_proj` shared, only conv+scan duplicated per direction). So
push the same idea further: keep pushing call count down, but do it by
sharing *more* weights, not fewer.

**What changed:** `JambaBiMambaBatchedMixer` shares `conv1d`/`x_proj`/
`dt_proj`/`A_log`/`D` across both directions (`Fused`'s `JambaBiMambaMixer`
only shared `in_proj`/`out_proj`) and stacks the forward and reversed
sequences along the **batch dimension** — `(b, d, L)` + `(b, d, L)` becomes
one `(2b, d, L)` tensor — so `causal_conv1d_fn`, `x_proj`, `dt_proj`, and
`selective_scan_fn` are each called **once** per layer instead of twice.
Direction-specific capacity (separate conv/scan params per direction) is
traded away entirely in exchange for halving the direction-specific op
count from 8 calls/layer (2× conv + 2× x_proj + 2× dt_proj + 2× scan) to 4.

**Why it works where `mamba_inner_fn` didn't:** both approaches reduce
Python/kernel-launch call count, but `JambaBiMambaBatchedMixer` does it by
*extending* weight-sharing (conv/scan now shared too), while `mamba_inner_fn`
did it by *losing* weight-sharing (in_proj/out_proj no longer shared) in
exchange for kernel fusion. Call-count reduction paid off; kernel fusion
alone did not — sharing weights was the load-bearing part of `Fused`'s
original win, not just having fewer calls.

**Effect (eager, measured directly against `Fused`):**

| L | Fused it/s | Fused GB | Batched it/s | Batched GB |
|---|---|---|---|---|
| 196 | 17.22 | 0.09 | 22.77 | 0.35 |
| 392 | 17.39 | 0.13 | 21.21 | 0.39 |
| 784 | 16.81 | 0.21 | 22.50 | 0.48 |
| 1568 | 16.50 | 0.38 | 22.61 | 0.65 |
| 3136 | 16.40 | 0.71 | 22.20 | 0.99 |
| 6272 | 16.86 | 1.39 | 22.22 | 1.68 |
| 12544 | 16.68 | 2.74 | 19.58 | 3.08 |

**~25-33% higher it/s than `Fused`** across nearly the whole tested range
(narrowing at the highest L, where raw scan FLOPs start to dominate over
per-call overhead) — a genuine additional win on top of everything in step
2, achieved purely by removing more calls without giving up any shared
weights. Peak memory is higher (fewer separate buffers to reuse, plus the
`2b` batch expansion), a real tradeoff of this approach, not a free lunch.
Not yet re-measured against ViT to get an updated crossover-point number —
see "Where the best configuration lives" below.

---

## Net result (updated)

| Stage | Eager it/s @ L=196 (approx.) |
|---|---|
| Naive double-pass (`BidirectionalJamba`) | flat, low (~8-9, pre-fused baseline) |
| + fused-bidirectional-call Mamba layers (`Fused`) | ~17-18 |
| + Mamba-2 SSD kernel | dropped — worse eagerly (~7.3-7.5), see step 6 |
| + `torch.compile(reduce-overhead)` | dropped — no real gain on Mamba side, see step 6 |
| + `mamba_inner_fn` fused kernel | dropped — worse than `Fused` (~14.3-15.4), see step 6 |
| + batched-direction scan (`Batched`) | **~22-23**, current best |

Crossover-vs-ViT numbers in the original table above are CUDA-graph-only and
have not been re-validated eagerly for any stage past `Fused` — treat them
as historical context, not current guidance. `Batched`'s ~25-33% eager it/s
gain over `Fused` should shift the eager crossover point lower, but that
needs a direct `Batched`-vs-eager-ViT sweep to quantify.

## Where the "best" (eager) configuration lives

`BidirectionalJambaBatched` in `benchmark_seq_scaling.py` — 9
`JambaBiMambaBatchedDecoderLayer` instances (batch-dim-stacked bidirectional
scan, conv/x_proj/dt_proj/A/D shared across directions) + 1 non-causal
attention layer (merge-before-attention), run eagerly (no CUDA graph
capture) via `run_sweep`. This supersedes `BidirectionalJambaFused` /
`JambaBiMambaMixer` as current-best; `BidirectionalJambaMamba2Fused` /
`JambaMamba2BiDecoderLayer` no longer exist in `benchmark_seq_scaling.py`
(removed per step 6).

---

## Step 7: surgical `torch.compile` (fair, both-sides-compiled) — file trimmed since

Both `Batched` and ViT were wrapped in `torch.compile(mode="reduce-overhead")`
— ViT wholesale (`BidirectionalViTCompiled`, no opaque calls to graph-break
on), Jamba surgically (only the pure-PyTorch glue around
`causal_conv1d_fn`/`selective_scan_fn`, which stayed eager since Dynamo can't
trace into them). This raised both sides' raw it/s substantially over eager,
but — once measured with *both* sides under the same treatment, not just
Jamba — the crossover point landed back at **~6250 tokens**, matching the
old CUDA-graph-fair estimate (~4-6k) from earlier in the project. Confirms
the standing lesson from step 4/6: pure overhead-removal (CUDA graphs,
`torch.compile`) rescales both curves proportionally and does not move the
true asymptotic crossover ratio — only a change to Mamba's actual per-token
compute constant does that.

A specific number from this step — a progression reaching roughly
**~26.5-27 it/s** on the maximal surgical-compile variant
(`BidirectionalJambaBatchedFullCompiledGlue`: mixer's pre/inter/post-scan
glue + FFN + attention tail all compiled, only the two raw
`causal_conv1d_fn`/`selective_scan_fn` calls left eager) — was reported in
conversation but never actually written into this doc, and the classes
implementing it were deleted from `benchmark_seq_scaling.py` in a later
session pass that trimmed the file to three models (SSD experiment below).
Since the file was never committed to git at any point (it's untracked),
there was no history to recover the classes from — they were reconstructed
from the assistant's own conversation context and **restored** to
`benchmark_seq_scaling.py` (`_MambaPreScanBlock`/`_MambaInterScanBlock`/
`_MambaPostScanBlock`, `JambaBiMambaBatchedCompiledMixer`, `_FFNBlock`,
`JambaBiMambaBatchedCompiledDecoderLayer`, `_AttnTailBlock`,
`BidirectionalJambaBatchedCompiledGlue`,
`JambaBiMambaBatchedFullCompiledDecoderLayer`,
`BidirectionalJambaBatchedFullCompiledGlue`) specifically so the ~27 it/s
claim can be re-measured rather than trusted secondhand. **Treat that number
as unverified until the sweep is actually re-run** — the code is back in
`__main__`'s sweep, but no GPU run has confirmed it since restoration.

## Step 8: pure-PyTorch chunked SSD scan (`JambaSSDMixer`) — tried, dropped

Custom-op/meta-kernel registration to make Dynamo trace through
`selective_scan_fn` directly was attempted and abandoned (very difficult,
didn't work) — not revisited. Instead: replace the opaque kernel with an
equivalent computation that has literally no CUDA/Triton extension calls,
so there's nothing left for `torch.compile` to graph-break around.

The linear recurrence `h_t = A_t·h_{t-1} + B_t·x_t` is associative, so it can
be computed via a **chunked scan** written in plain `einsum`/`cumsum`/
`masked_fill` (segsum-style pairwise decay matrix for the intra-chunk term).
Two variants were tried:

**v1 (rejected on capacity grounds): scalar-per-head `A`, fully vectorized
across chunks.** The literal Mamba-2/SSD algorithm — `A` scalar per head,
one small `chunk × chunk` matrix per head, all chunks processed in one
vectorized call. Requires grouping `d_inner` channels into heads sharing one
decay rate (tried `HEAD_DIM=32` → 9 heads, then `HEAD_DIM=1` → 288
heads/fully per-channel). Rejected before ever running on GPU: channels
with genuinely different variance/memory-horizon needs being forced onto a
shared decay rate is a real capacity loss, and with attention already lossy
(single non-causal merge pass), there's no budget to give up more. Full
Mamba-1 fidelity (independent decay per `(channel, state)` pair, 4608 of
them) was made a hard requirement.

**v2 (implemented, measured, then dropped): full `(d_inner × d_state)`
fidelity, chunk-looped instead of vectorized.** `A` is a `(d_inner,
d_state)` matrix — identical granularity to the CUDA-kernel-based mixers.
The pairwise decay matrix scales with 4608 independent rates instead of
9-288 heads, so vectorizing across all chunks simultaneously (v1's
approach) would materialize that matrix for every chunk at once — ~15GB at
L=6272, ~30GB at L=12544, for one intermediate tensor in one layer. Not
viable. Fix: loop over chunks sequentially (`_ssd_full_chunked_scan`),
wrapping each chunk's forward in `torch.utils.checkpoint.checkpoint`
(`_ssd_full_chunk_step`) so autograd doesn't retain every chunk's ~150MB
intermediate simultaneously for backward. Validated forward AND backward
against a naive per-timestep PyTorch reference of Mamba-1's exact
recurrence, max error ~1e-8 (`scratchpad/validate_ssd_full.py`), before
being wired into `BidirectionalJambaSSDCompiled` (whole model under
`torch.compile(mode="reduce-overhead")`, since there was no opaque call
left to graph-break around).

**Measured result: ~16-17 it/s** — worse than `Batched`'s eager ~22-23 and
far worse than the restored surgical-compile variant's (unverified) ~27.
Despite achieving full `torch.compile`/CUDA-graph coverage of the scan
itself (something no CUDA-kernel-based variant can do), the
`L/chunk_size`-step sequential Python loop plus checkpoint recomputation
cost more than the coverage gained back — the fused CUDA kernel, even
though it's an uncompilable black box to Dynamo, is simply faster at doing
the actual scan than a hand-written chunked loop is, compiled or not. This
mixer (`JambaSSDMixer`, `JambaSSDDecoderLayer`, `BidirectionalJambaSSDCompiled`,
`_ssd_full_chunk_step`, `_ssd_full_chunked_scan`) has been **removed** from
`benchmark_seq_scaling.py`. The v1→v2 correctness-validation scripts remain
at `scratchpad/validate_ssd.py` / `scratchpad/validate_ssd_full.py` if this
direction is revisited, but as of this result it's a dead end relative to
CUDA-kernel-based approaches, not a promising unfinished lever.

`benchmark_seq_scaling.py`'s `__main__` sweep now runs three models:
`BidirectionalJambaBatched` (previous best, eager, full Mamba-1 fidelity),
`BidirectionalJambaBatchedFullCompiledGlue` (restored surgical-compile
variant, full fidelity, the still-unverified ~27 it/s claim — this is the
one worth re-measuring next), `BidirectionalViTCompiled` (fair baseline).
Still removed (not restored, not needed for any current question):
`BidirectionalJamba`, `JambaBiMambaMixer`/`JambaBiMambaDecoderLayer`,
`BidirectionalJambaFused`, uncompiled `BidirectionalViT`,
`check_mamba_kernel_generation`, `benchmark_cuda_graph`/`run_sweep_graphed`,
`profile_model`. Since these files are untracked in git, "recover from git
history" is not an option for anything cut from here — recovery has to come
from conversation context while it's still available, same as the surgical-
compile restoration above.
