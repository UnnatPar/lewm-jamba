"""How long does one epoch of the multi-frame model take, and what batch size fits?

This is the number to know before spending anything on a real run. It puts the actual JEPA
stack -- real JambaEncoder at the config's mamba_expand=2, real ARPredictor, real projectors,
real optimizer step and backward -- through a full training step on synthetic pixels, and
reports memory and throughput per batch size.

Two things it is specifically checking, because both were assumed rather than measured:

  1. Every crossover measurement so far was at mamba_expand=1. The training config uses 2,
     which doubles the scan's inner width. 1,960 tokens may no longer be the right window.
  2. sscanop2's chunk padding was verified at expand=1 as well. It reports a null control here.

Synthetic pixels, so this measures the model and not the Lance loader. Dataloader cost is
measured separately -- if it dominates, none of these numbers are the bottleneck.
"""

import importlib.util
import json
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/content")

import jepa as jepa_mod
import module as mod
import sscanop2 as co

DEVICE = "cuda"
IMG, EMB, WINDOW, HIST, NPRED = 224, 192, 10, 4, 1
ACTION_DIM = 2
BATCHES = [4, 8, 16, 24]
STEPS, WARM = 20, 6
T = NPRED + HIST + WINDOW - 1  # 14 frames per sample


def build():
    enc = mod.JambaEncoder(
        image_size=IMG, output_dim=EMB, hidden_size=288, intermediate_size=1152,
        num_hidden_layers=10, num_attention_heads=8, num_key_value_heads=4,
        attn_layer_period=10, attn_layer_offset=9, mamba_d_state=16, mamba_d_conv=4,
        mamba_expand=2, max_frames=WINDOW)
    pred = mod.ARPredictor(num_frames=HIST, input_dim=EMB, hidden_dim=EMB, output_dim=EMB,
                           depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1,
                           emb_dropout=0.0)
    mlp = lambda: mod.MLP(input_dim=EMB, output_dim=EMB, hidden_dim=2048,
                          norm_fn=lambda d: nn.BatchNorm1d(d))
    return jepa_mod.JEPA(
        encoder=enc, predictor=pred,
        action_encoder=mod.Embedder(input_dim=ACTION_DIM, emb_dim=EMB),
        projector=mlp(), pred_proj=mlp(), window_size=WINDOW, window_stride=1).to(DEVICE)


def run(batch):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    n = (STEPS + WARM + 1) * batch
    dl = DataLoader(TensorDataset(torch.randn(n, T, 3, IMG, IMG),
                                  torch.randn(n, T, ACTION_DIM)),
                    batch_size=batch, shuffle=False, pin_memory=True, drop_last=True)
    model = build()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    it = iter(dl)
    for i in range(WARM + STEPS):
        if i == WARM:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        px, act = next(it)
        info = {"pixels": px.to(DEVICE, non_blocking=True),
                "action": act.to(DEVICE, non_blocking=True)}
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.encode(info)
            emb, act_emb = out["emb"], out["act_emb"]
            ctx, tgt = emb[:, :HIST], emb[:, NPRED:]
            loss = (model.predict(ctx, act_emb[:, :HIST]) - tgt).pow(2).mean()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    mem = torch.cuda.max_memory_allocated() / 1e9
    del model, opt
    torch.cuda.empty_cache()
    return STEPS / dt, mem


def main():
    orig, shim = co.register()
    co.PAD_SCAN = True
    print(f"pad(1960) -> {co._pad_len(WINDOW * 196)}", flush=True)
    if not co.verify(orig, shim):
        print("STOPPING: scan padding is not equivalent at this config"); return

    # 25200 training samples in pusht_expert_train at 0.9 split, per the loader.
    n_train = 25200
    print(f"\n  T={T} frames/sample, {HIST + NPRED} windows of {WINDOW * 196} tokens\n")
    print(f"  {'batch':>5} {'GB':>7} {'it/s':>8} {'samp/s':>8} {'tok/s':>10} {'hr/epoch':>9}",
          flush=True)
    out = {}
    for b in BATCHES:
        try:
            ips, mem = run(b)
        except torch.cuda.OutOfMemoryError:
            print(f"  {b:>5} {'OOM':>7}", flush=True)
            torch.cuda.empty_cache()
            continue
        sps = ips * b
        tok = sps * (HIST + NPRED) * WINDOW * 196
        hr = n_train / sps / 3600
        print(f"  {b:>5} {mem:>7.2f} {ips:>8.2f} {sps:>8.1f} {tok:>10.0f} {hr:>9.2f}",
              flush=True)
        out[str(b)] = {"gb": mem, "it_s": ips, "samp_s": sps, "hr_epoch": hr}
    print("MFPROBE_JSON " + json.dumps(out), flush=True)


main()
