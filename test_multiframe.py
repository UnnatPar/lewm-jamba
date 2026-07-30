"""Does the multi-frame window encoder do what the design claims? CPU-only, seconds to run.

The claim the whole design rests on is a dependency claim: emb[k] must depend on frames
k*stride .. k*stride+W-1 and on nothing else. If it depends on a later frame, ctx_emb contains
the answer, pred_loss collapses, no exception is raised, and the loss curve looks excellent.
That is the "runs fine, result is meaningless" failure class, so it gets a test rather than an
argument.

Test 3 is the one that matters: perturb exactly one frame, see which timesteps move.
"""

import torch
from torch import nn

import jepa
import module

torch.manual_seed(0)


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    return ok


class StubEncoder(nn.Module):
    """Maximally leaky within a window: every slot sees every frame of its window. If the
    windowing in jepa.encode is right, the leak still cannot escape the window -- which is
    exactly the property under test. A real bidirectional encoder is no leakier than this."""

    def forward(self, pixels, interpolate_pos_encoding=True):
        if pixels.dim() == 4:
            pixels = pixels.unsqueeze(1)
        n, f = pixels.shape[:2]
        per_frame = pixels.flatten(2).sum(-1)                    # (N, F)
        window_sum = per_frame.sum(1, keepdim=True)              # (N, 1) -- all frames
        feat = (per_frame + window_sum).unsqueeze(-1)            # (N, F, 1)
        return module._EncoderOutput(feat.expand(n, f, 8).contiguous())


def make_jepa(w, s):
    return jepa.JEPA(
        encoder=StubEncoder(),
        predictor=nn.Identity(),
        action_encoder=nn.Identity(),
        window_size=w,
        window_stride=s,
    )


def main():
    ok = True
    W, S, T, B = 10, 1, 14, 2

    print("\n1. encoder shape and window handling")
    enc = module.JambaEncoder(
        image_size=32, patch_size=16, output_dim=6, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        attn_layer_period=2, attn_layer_offset=1, mamba_expand=1, max_frames=4,
    )
    with torch.no_grad():
        out4 = enc(torch.randn(2, 3, 32, 32)).last_hidden_state
        out5 = enc(torch.randn(2, 4, 3, 32, 32)).last_hidden_state
    ok &= check("4D input treated as F=1", tuple(out4.shape) == (2, 1, 6), str(tuple(out4.shape)))
    ok &= check("5D window returns one slot per frame",
                tuple(out5.shape) == (2, 4, 6), str(tuple(out5.shape)))

    print("\n1b. the ViT baseline honours the same contract")
    vit = module.ViTEncoder(
        image_size=32, patch_size=16, output_dim=6, hidden_size=32,
        num_hidden_layers=2, num_attention_heads=4, max_frames=4,
    )
    with torch.no_grad():
        v4 = vit(torch.randn(2, 3, 32, 32)).last_hidden_state
        v5 = vit(torch.randn(2, 4, 3, 32, 32)).last_hidden_state
    ok &= check("ViT 4D input treated as F=1", tuple(v4.shape) == (2, 1, 6), str(tuple(v4.shape)))
    ok &= check("ViT 5D window returns one slot per frame",
                tuple(v5.shape) == (2, 4, 6), str(tuple(v5.shape)))
    # Slot f must be frame f seen through the window, not a copy of a window-wide pooling.
    # If the per-frame pooling span were wrong, every slot would come out identical and the
    # last-slot readout in jepa.encode would silently become a window summary.
    ok &= check("ViT slots differ from one another",
                (v5[:, 0] - v5[:, -1]).abs().max().item() > 1e-5,
                f"max|slot0-slotF|={(v5[:, 0] - v5[:, -1]).abs().max().item():.3e}")

    print("\n2. window count and action alignment")
    m = make_jepa(W, S)
    info = m.encode({"pixels": torch.randn(B, T, 3, 8, 8),
                     "action": torch.arange(T).float().view(1, T, 1).expand(B, T, 1).clone()})
    n_win = (T - W) // S + 1
    ok &= check(f"T={T}, W={W} -> {n_win} timesteps",
                info["emb"].shape[1] == n_win, str(tuple(info["emb"].shape)))
    ok &= check("history_size(4) + num_preds(1) timesteps available", n_win == 5, f"n_win={n_win}")
    acts = info["act_emb"][0, :, 0].tolist()
    ok &= check("action k is the one at window k's newest frame",
                acts == [float(W - 1 + k * S) for k in range(n_win)], str(acts))

    print("\n3. leakage: which frames can each timestep see?")
    base = torch.randn(1, T, 3, 8, 8)
    ref = make_jepa(W, S).encode({"pixels": base.clone()})["emb"][0]
    leaked = []
    for t in range(T):
        bumped = base.clone()
        bumped[0, t] += 100.0
        moved = (make_jepa(W, S).encode({"pixels": bumped})["emb"][0] - ref).abs().sum(-1) > 1e-4
        for k in range(n_win):
            inside = k * S <= t < k * S + W
            if bool(moved[k]) != inside:
                leaked.append((t, k, bool(moved[k]), inside))
    ok &= check("emb[k] moves iff the frame is inside window k",
                not leaked, f"{len(leaked)} violations" if leaked else "all 14x5 pairs correct")

    print("\n4. the pairing train.py actually uses")
    # ctx = emb[:, :4] are windows 0..3 (newest frames 9,10,11,12)
    # tgt = emb[:, 1:] are windows 1..4 (newest frames 10,11,12,13)
    bad = [(k, t) for k in range(4) for t in range(T)
           if (k * S <= t < k * S + W) and t >= (k + 1) * S + W - 1]
    ok &= check("no context window contains the frame its target is predicting",
                not bad, f"{len(bad)} violations" if bad else "4 ctx/tgt pairs clean")

    print("\n5. window_size=1 is byte-identical to the old per-frame path")
    p = torch.randn(2, 4, 3, 8, 8)
    e1 = make_jepa(1, 1).encode({"pixels": p.clone()})["emb"]
    stub = StubEncoder()
    with torch.no_grad():
        legacy = stub(p.reshape(8, 3, 8, 8)).last_hidden_state[:, 0].reshape(2, 4, 8)
    ok &= check("W=1 matches", torch.equal(e1, legacy),
                f"max|diff|={(e1 - legacy).abs().max().item():.3e}")

    print(f"\n{'ALL PASS' if ok else 'FAILURES ABOVE'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
