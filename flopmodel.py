"""Where Jamba's per-token constant actually goes, and what it implies for the crossover.

Costs zero GPU credits. The point is to stop guessing which lever is worth an A100 run:
crossover is n* = C_m/C_a, so only a change to a *large* term in C_m can move it, and this
prints the term sizes.

Counts MACs, not wall-clock, so it will not reproduce the measured 1868 exactly -- the scan
is memory-bound and the GEMMs are not. It is a ranking tool, not a predictor. Anything it
says is <5% of C_m is not worth an experiment regardless of how clever it is.
"""

H, INTER, LAYERS, HEADS = 288, 1152, 10, 8
D_STATE, D_CONV, EXPAND = 16, 4, 1
VIT_H, VIT_LAYERS = 328, 10

D_INNER = EXPAND * H
DT_RANK = -(-H // 16)  # ceil(H/16) = 18
N_MAMBA = LAYERS - 1  # the 10th layer is the single attention layer


def jamba_per_token():
    """MACs per token, per the whole 10-layer stack. Directions marked x2 run on the
    batch-stacked (2b, d, L) tensor and genuinely cost twice."""
    t = {}
    t["mamba in_proj"] = N_MAMBA * H * D_INNER * 2
    t["mamba out_proj"] = N_MAMBA * D_INNER * H
    t["mamba conv1d x2"] = N_MAMBA * 2 * D_INNER * D_CONV
    t["mamba x_proj x2"] = N_MAMBA * 2 * D_INNER * (DT_RANK + 2 * D_STATE)
    t["mamba dt_proj x2"] = N_MAMBA * 2 * DT_RANK * D_INNER
    # selective scan recurrence: per (channel, state) it is ~ dA, dB*x, state update,
    # and the C-contraction. Call it 4 MACs. Doubled for the two directions.
    t["mamba scan x2"] = N_MAMBA * 2 * D_INNER * D_STATE * 4
    t["mamba MLP (gated)"] = N_MAMBA * 3 * H * INTER
    t["attn layer qkvo"] = 4 * H * H
    t["attn layer MLP"] = 3 * H * INTER
    return t


def vit_per_token():
    t = {}
    t["vit qkvo"] = VIT_LAYERS * 4 * VIT_H * VIT_H
    t["vit MLP"] = VIT_LAYERS * 2 * VIT_H * (4 * VIT_H)
    return t


def quadratic_coeff():
    """MACs per token that scale with L: QK^T and AV, both 2*d per token-pair."""
    return {"jamba": 2 * H, "vit": VIT_LAYERS * 2 * VIT_H}


def crossover(jamba_const, vit_const):
    """Solve jamba_const + qj*L = vit_const + qv*L."""
    q = quadratic_coeff()
    denom = q["vit"] - q["jamba"]
    return (jamba_const - vit_const) / denom


def report(label, jt):
    q = quadratic_coeff()
    vt = vit_per_token()
    cj, cv = sum(jt.values()), sum(vt.values())
    n = crossover(cj, cv)
    print(f"\n=== {label} ===")
    for k, v in sorted(jt.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22} {v/1e6:7.3f} M  {100*v/cj:5.1f}% of C_m")
    print(f"  {'-'*22} {cj/1e6:7.3f} M  Jamba per-token")
    print(f"  {'ViT per-token':<22} {cv/1e6:7.3f} M")
    print(f"  quadratic: jamba {q['jamba']}/tok/tok, vit {q['vit']}/tok/tok")
    print(f"  => crossover L = {n:,.0f}")
    return n


def main():
    base = jamba_per_token()
    n0 = report("current", base)

    print("\n\n=== levers: crossover if this term went to zero ===")
    print("(zeroing a term is not a proposal, it is an upper bound on what optimising it "
          "could ever buy)\n")
    vt = sum(vit_per_token().values())
    rows = []
    for k in base:
        cut = {kk: (0 if kk == k else vv) for kk, vv in base.items()}
        rows.append((k, crossover(sum(cut.values()), vt)))
    for k, n in sorted(rows, key=lambda r: r[1]):
        print(f"  drop {k:<22} -> {n:8,.0f}   ({100*(n0-n)/n0:5.1f}% reduction)")

    print("\n\n=== the only structural knob that is not capacity loss ===")
    # The single attention layer is what makes Jamba quadratic at all. Its coefficient is
    # 1/10th of ViT's, so it barely matters -- confirm that rather than assume it.
    q = quadratic_coeff()
    print(f"  jamba quadratic is {100*q['jamba']/q['vit']:.0f}% of ViT's; removing it entirely")
    print(f"  would give crossover {(sum(base.values())-4*H*H-3*H*INTER-vt)/q['vit']:,.0f}"
          f" -- and costs the only global mixing in the stack.")


main()
