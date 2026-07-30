"""Real-setting cost of the optimised encoder: full JEPA step, both candidate window sizes.

The encoder benchmarks compare Jamba to a ViT. This measures what actually matters for planning a
run: samples/s and hours/epoch for the whole JEPA stack -- encoder, action encoder, projector,
predictor, backward, optimizer step -- at the optimised config.

Config under test (every capability invariant from the sweep held):
  expand=1, d_state=32, intermediate=1424   -> SSM state 9216/layer, ~15.50M params
  bidir_mode=inner, mixer_impl=alternate    -> bidirectional stack, one scan per layer

Two windows:
  10 frames = 1,960 tokens -- Jamba at ~0.89 of a matched ViT
  20 frames = 3,920 tokens -- Jamba at ~1.10, i.e. past the crossover, and chunk-efficient

The comparison to beat is the ORIGINAL multi-frame measurement: 13.3 samples/s and 0.53 hr/epoch
at 10 frames, expand=2, per-layer-bidirectional, unfused. If 20 frames now matches that, the
window doubles for free and the crossover claim comes with it.

Note the data constraint, which is not a speed question: a W-frame window needs
num_steps = num_preds + history_size + W - 1 frames per sample, so W=20 needs 24 frames, and at
frameskip 5 that is 120 raw environment steps -- the entire length of a PushT episode, leaving no
slack to sample different start points. W=20 therefore requires a lower frameskip, and that is
reported rather than assumed away.
"""

import json
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/content")

import jepa as jepa_mod
import module as mod

DEVICE = "cuda"
IMG, EMB, HIST, NPRED, ACTION_DIM = 224, 192, 4, 1, 2
N_TRAIN = 25200
STEPS, WARM, REPEATS = 12, 4, 2


def build(window):
    enc = mod.JambaEncoder(
        image_size=IMG, output_dim=EMB, hidden_size=288, intermediate_size=1424,
        num_hidden_layers=10, num_attention_heads=8, num_key_value_heads=4,
        attn_layer_period=10, attn_layer_offset=9, mamba_d_state=32, mamba_d_conv=4,
        mamba_expand=1, max_frames=window, bidir_mode="inner", mixer_impl="alternate")
    pred = mod.ARPredictor(num_frames=HIST, input_dim=EMB, hidden_dim=EMB, output_dim=EMB,
                           depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1,
                           emb_dropout=0.0)
    mlp = lambda: mod.MLP(input_dim=EMB, output_dim=EMB, hidden_dim=2048,
                          norm_fn=lambda d: nn.BatchNorm1d(d))
    return jepa_mod.JEPA(
        encoder=enc, predictor=pred,
        action_encoder=mod.Embedder(input_dim=ACTION_DIM, emb_dim=EMB),
        projector=mlp(), pred_proj=mlp(), window_size=window, window_stride=1).to(DEVICE)


def run(window, batch):
    T = NPRED + HIST + window - 1
    best, mem = None, 0.0
    for _ in range(REPEATS):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        n = (STEPS + WARM + 1) * batch
        dl = DataLoader(TensorDataset(torch.randn(n, T, 3, IMG, IMG),
                                      torch.randn(n, T, ACTION_DIM)),
                        batch_size=batch, shuffle=False, pin_memory=True, drop_last=True)
        m = build(window)
        opt = torch.optim.AdamW(m.parameters(), lr=5e-5, weight_decay=1e-3)
        it = iter(dl)
        for i in range(WARM + STEPS):
            if i == WARM:
                torch.cuda.synchronize(); t0 = time.perf_counter()
            px, act = next(it)
            info = {"pixels": px.to(DEVICE, non_blocking=True),
                    "action": act.to(DEVICE, non_blocking=True)}
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = m.encode(info)
                emb, ae = o["emb"], o["act_emb"]
                loss = (m.predict(emb[:, :HIST], ae[:, :HIST]) - emb[:, NPRED:]).pow(2).mean()
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        r = STEPS / (time.perf_counter() - t0)
        best = r if best is None else max(best, r)
        mem = max(mem, torch.cuda.max_memory_allocated() / 1e9)
        del m, opt
        torch.cuda.empty_cache()
    return best, mem, T


def main():
    print("\n  full JEPA training step, expand=1 d_state=32 intermediate=1424, alternate")
    print(f"  reference to beat: 13.3 samp/s, 0.53 hr/epoch "
          f"(10 frames, expand=2, per-layer bidirectional, unfused)\n", flush=True)
    print(f"  {'win':>4} {'tokens':>7} {'T':>4} {'batch':>6} {'it/s':>7} {'samp/s':>7} "
          f"{'hr/epoch':>9} {'GB':>6}", flush=True)
    out = {}
    for window in (10, 20):
        for batch in (2, 3, 4, 6):
            try:
                r, mem, T = run(window, batch)
            except torch.cuda.OutOfMemoryError:
                print(f"  {window:>4} {window*196:>7} {'':>4} {batch:>6}  OOM", flush=True)
                torch.cuda.empty_cache(); continue
            sps = r * batch
            hr = N_TRAIN / sps / 3600
            print(f"  {window:>4} {window*196:>7} {T:>4} {batch:>6} {r:>7.2f} {sps:>7.2f} "
                  f"{hr:>9.2f} {mem:>6.2f}", flush=True)
            out[f"w{window}_b{batch}"] = {"it_s": r, "samp_s": sps, "hr_epoch": hr,
                                          "gb": mem, "T": T}
    print("MF2_JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
