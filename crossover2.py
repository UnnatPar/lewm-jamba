"""Does Jamba still beat a matched ViT at the window we actually built?

Every crossover number so far (1,946 -> 1,723) came from `bench.BidirectionalJambaBatched` at
mamba_expand=1 against a 12.95M ViT. The model in module.py runs at expand=2 and weighs 15.5M.
Expand=2 doubles the scan's inner width, which raises Jamba's per-token cost, which moves the
crossover UP -- possibly past the 1,960 tokens that expand=1 measurements chose. If so, the
window is on the wrong side of the line and 10 frames is the wrong number.

So this re-measures with the REAL modules: module.JambaEncoder exactly as configured for
training, against a ViT rebuilt to match its parameter count (not the old 328-wide one, which
would be 17% undersized here and would flatter Jamba). Both take a window of F frames and
return one embedding per frame, both run a full training step.

Neither is torch.compile'd, because the training loop does not compile either. The compiled
numbers in realcross.py answer a different question.
"""

import json
import os
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/content")
import module as mod
import sscanop2 as co

DEVICE = "cuda"
IMG, PATCH, EMB = 224, 16, 192
PER_FRAME = (IMG // PATCH) ** 2  # 196
# Batch matters more than it looks. At batch 2 the ViT slowed only 4% for 3x the tokens --
# quadratic attention cannot do that, so both models were kernel-launch-bound and the "ratio"
# was just Jamba's higher launch count (a flat 0.50 at every length). A crossover is a
# statement about compute, so it can only be measured where compute dominates. mfprobe put
# the A100 at saturation around 20 windows per step, so sweep up to there and report the
# regime change rather than a single number.
BATCHES = [int(x) for x in os.environ.get("CROSS2_BATCHES", "2,8,16").split(",")]
FRAMES = [int(x) for x in os.environ.get("CROSS2_FRAMES", "4,6,8,10,12").split(",")]
# Uncompiled is what train.py actually runs. But every historical crossover number came from
# torch.compile(mode="reduce-overhead"), and cudagraphs is precisely the fix for Jamba's many
# small mamba kernel launches -- so the two are different questions and both need answering.
COMPILE = os.environ.get("CROSS2_COMPILE", "0") == "1"
# The one variable that could account for the whole gap. Every historical crossover number was
# at expand=1; module.py defaults to 2, which doubles the scan's inner width and so roughly
# doubles the mamba layers' cost. Sweeping it turns "Jamba loses" into a decision the numbers
# can actually inform: capacity at expand=2 versus speed at expand=1.
EXPANDS = [int(x) for x in os.environ.get("CROSS2_EXPANDS", "2").split(",")]
STEPS, WARM = 15, 5


def jamba(max_frames, expand=2):
    return mod.JambaEncoder(
        image_size=IMG, patch_size=PATCH, output_dim=EMB, hidden_size=288,
        intermediate_size=1152, num_hidden_layers=10, num_attention_heads=8,
        num_key_value_heads=4, attn_layer_period=10, attn_layer_offset=9,
        mamba_d_state=16, mamba_d_conv=4, mamba_expand=expand, max_frames=max_frames)


class ViTEncoder(nn.Module):
    """Same contract as JambaEncoder: (N,F,C,H,W) -> (N,F,EMB), joint attention over the whole
    F*196-token window, pooled within each frame's own span."""

    def __init__(self, hidden, max_frames, layers=10):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, hidden, PATCH, PATCH)
        self.pos_embedding = nn.Parameter(torch.randn(1, PER_FRAME, hidden) * 0.02)
        self.frame_embedding = nn.Parameter(torch.randn(1, max_frames, 1, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=8, dim_feedforward=hidden * 4, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(hidden, EMB)

    def forward(self, pixels):
        n, f = pixels.shape[:2]
        x = self.patch_embed(pixels.flatten(0, 1)).flatten(2).transpose(1, 2)
        x = x + self.pos_embedding
        x = (x.view(n, f, PER_FRAME, -1) + self.frame_embedding[:, :f]).reshape(n, -1, x.size(-1))
        out = self.enc(x).view(n, f, PER_FRAME, -1).mean(dim=2)
        return self.head(out)


def match_hidden(target, max_frames):
    """Pick the ViT width whose parameter count lands closest to Jamba's, in multiples of 8."""
    best = None
    for h in range(256, 481, 8):
        p = sum(q.numel() for q in ViTEncoder(h, max_frames).parameters())
        if best is None or abs(p - target) < abs(best[1] - target):
            best = (h, p)
    return best


def rate(build, f, batch):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    n = (STEPS + WARM + 1) * batch
    dl = DataLoader(TensorDataset(torch.randn(n, f, 3, IMG, IMG), torch.randn(n, f, EMB)),
                    batch_size=batch, shuffle=False, pin_memory=True, drop_last=True)
    model = build().to(DEVICE)
    if COMPILE:
        torch._dynamo.reset()
        model = torch.compile(model, mode="reduce-overhead")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lossf = nn.MSELoss()
    it = iter(dl)
    for i in range(WARM + STEPS):
        if i == WARM:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        x, y = next(it)
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        if COMPILE:
            # Required with cudagraph trees, or the graph replays stale input buffers.
            torch.compiler.cudagraph_mark_step_begin()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            out = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            loss = lossf(out.float(), y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    mem = torch.cuda.max_memory_allocated() / 1e9
    del model, opt
    torch.cuda.empty_cache()
    if COMPILE:
        torch._dynamo.reset()
    return STEPS / dt, mem


def main():
    orig, shim = co.register()
    co.PAD_SCAN = True
    if not co.verify(orig, shim):
        print("STOPPING: equivalence failed"); return

    out, meta = {}, {}
    for expand in EXPANDS:
      # Re-match the ViT per expand. Reusing one baseline across both would put Jamba at
      # expand=1 against a ViT sized for expand=2 and silently flatter the smaller model.
      jp = sum(p.numel() for p in jamba(max(FRAMES), expand).parameters())
      h, vp = match_hidden(jp, max(FRAMES))
      meta[str(expand)] = {"jamba_params": jp, "vit_hidden": h, "vit_params": vp}
      print(f"\n### mamba_expand={expand}: Jamba {jp:,} vs ViT(hidden={h}) {vp:,} "
            f"({100 * (vp - jp) / jp:+.1f}%)  compile={COMPILE}", flush=True)
      for batch in BATCHES:
        print(f"\n=== expand={expand} batch={batch} ({batch} windows/step) ===", flush=True)
        print(f"  {'frames':>6} {'tokens':>7} {'pad':>6} {'ViT it/s':>9} {'Jamba it/s':>11} "
              f"{'ratio':>7} {'J GB':>6}", flush=True)
        first_vit = None
        for f in FRAMES:
            L = f * PER_FRAME
            row = {}
            for name, build in (("vit", lambda f=f, h=h: ViTEncoder(h, f)),
                                ("jamba", lambda f=f, e=expand: jamba(f, e))):
                try:
                    row[name], row[name + "_gb"] = rate(build, f, batch)
                except torch.cuda.OutOfMemoryError:
                    print(f"  {f:>6} {L:>7}  {name} OOM", flush=True)
                    torch.cuda.empty_cache()
            if "vit" in row and "jamba" in row:
                r = row["jamba"] / row["vit"]
                first_vit = first_vit or row["vit"]
                print(f"  {f:>6} {L:>7} {co._pad_len(L):>6} {row['vit']:>9.2f} "
                      f"{row['jamba']:>11.2f} {r:>7.3f} {row['jamba_gb']:>6.2f}"
                      f"{'   JAMBA WINS' if r > 1 else ''}", flush=True)
            out[f"{expand}|{batch}|{f}"] = row
        # Compute-bound or launch-bound? If the ViT barely slows over a 3x token increase, its
        # quadratic term is not what is being timed and no crossover can be read off this.
        last_vit = out[f"{expand}|{batch}|{FRAMES[-1]}"].get("vit")
        if first_vit and last_vit:
            drop = 100 * (1 - last_vit / first_vit)
            print(f"  ViT slowdown {FRAMES[0]}->{FRAMES[-1]} frames: {drop:.0f}%  "
                  f"({'LAUNCH-BOUND, ratios meaningless' if drop < 25 else 'compute-bound'})",
                  flush=True)
    print("CROSS2_JSON " + json.dumps({"meta": meta, "rows": out}), flush=True)


# Guarded so a runner can set CROSS2_* in os.environ first and then
# runpy.run_path(path, run_name="__main__") -- colab-run.sh ships the script to a remote
# kernel, so exporting environment variables on the WSL side never reaches it.
if __name__ == "__main__":
    main()
