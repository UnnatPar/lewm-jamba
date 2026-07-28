"""Chart the padded-scan result. CPU only -- no GPU, no credits.

    python padplot.py outputs/padcross.json --out outputs/
"""

import json, math, sys, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def crossover(j, v):
    shared = sorted(set(j) & set(v))
    d = [(L, j[L] - v[L]) for L in shared]
    for (L0, d0), (L1, d1) in zip(d, d[1:]):
        if d0 <= 0 < d1 or d0 < 0 <= d1:
            t = -d0 / (d1 - d0)
            return math.exp(math.log(L0) + t * (math.log(L1) - math.log(L0)))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    raw = json.load(open(a.json, encoding="utf-8-sig"))
    curves = {k: {int(L): r for L, r in c.items()} for k, c in raw.items()}

    style = {"ViT matched": ("#4C6EF5", "o", "-"),
             "Jamba pad=off": ("#9AA0A6", "s", "--"),
             "Jamba pad=ON": ("#E8590C", "^", "-")}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, c in curves.items():
        col, mk, ls = style.get(name, ("#333", "o", "-"))
        Ls = sorted(c)
        ax.plot(Ls, [c[L] for L in Ls], marker=mk, ls=ls, color=col, label=name, lw=2)

    v = curves["ViT matched"]
    txt = []
    for name, col in (("Jamba pad=off", "#9AA0A6"), ("Jamba pad=ON", "#E8590C")):
        n = crossover(curves[name], v)
        if n:
            ax.axvline(n, color=col, ls=":", lw=1.5, alpha=0.8)
            txt.append(f"{name}: n* = {n:,.0f}")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sequence length L (tokens)")
    ax.set_ylabel("training steps / s  (batch 16, real loop)")
    ax.set_title("Padding the selective scan to its kernel chunk boundary\n"
                 "crossover 1,946 -> 1,723   (exact: forward bitwise identical)", fontsize=11)
    ax.grid(alpha=0.25, which="both")
    ax.legend(title="\n".join(txt), fontsize=9, title_fontsize=9)
    fig.tight_layout()
    p = f"{a.out}/padcross.png"
    fig.savefig(p, dpi=150)
    print("wrote", p)


main()
