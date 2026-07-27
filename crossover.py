#!/usr/bin/env python3
"""Turn a benchmark sweep into the one number this project is actually optimizing.

`benchmark_seq_scaling.py` prints per-model tables and the crossover gets read off them by eye.
The crossover IS the objective, so it should be computed, recorded, and plotted -- not squinted at.

    python benchmark_seq_scaling.py | tee sweep.txt
    python crossover.py sweep.txt --out outputs/

Writes `crossover.json` (machine-readable, for the ledger and for overnight searches to compare
against) and `crossover.png` (the chart to actually look at).

Runs anywhere -- it only processes numbers, so no GPU needed. That matters: the sweep runs on a
remote GPU, and the analysis runs wherever you are.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

HEADER = re.compile(r"^=+\s*(.+?)\s*=+\s*$")
ROW = re.compile(r"^L=\s*(\d+)\s+it/s=\s*([\d.]+)\s+peak_mem=\s*([\d.]+)\s*GB")
OOM = re.compile(r"^L=\s*(\d+)\s+OOM")


def parse(text: str) -> dict[str, dict[int, tuple[float, float]]]:
    """stdout -> {model_name: {seq_len: (it_per_s, peak_mem_gb)}}"""
    models: dict[str, dict[int, tuple[float, float]]] = {}
    current: str | None = None
    for line in text.splitlines():
        line = line.strip()
        m = HEADER.match(line)
        if m:
            current = m.group(1)
            models.setdefault(current, {})
            continue
        if current is None:
            continue
        m = ROW.match(line)
        if m:
            models[current][int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
            continue
        if OOM.match(line):
            continue  # OOM ends a model's curve; absence of the point says enough
    return {k: v for k, v in models.items() if v}


def classify(name: str) -> str:
    low = name.lower()
    if "vit" in low or "transformer" in low:
        return "vit"
    return "jamba"


def find_crossover(jamba: dict[int, float], vit: dict[int, float]) -> dict:
    """Sequence length where Jamba's it/s overtakes ViT's.

    Interpolated in log-L space, because the sweep is geometric (196, 392, 784, ...) and the
    curves are much closer to straight lines against log L than against L. Interpolating
    linearly in L on a geometric grid would bias the crossover toward the upper endpoint.
    """
    shared = sorted(set(jamba) & set(vit))
    if len(shared) < 2:
        return {"crossover": None, "reason": "need at least two shared sequence lengths"}

    diffs = [(L, jamba[L] - vit[L]) for L in shared]

    if all(d > 0 for _, d in diffs):
        return {"crossover": None, "reason": "jamba faster at every measured L -- crossover is "
                                             "below the sweep's lower bound",
                "bound": f"< {shared[0]}"}
    if all(d < 0 for _, d in diffs):
        return {"crossover": None, "reason": "jamba slower at every measured L -- crossover is "
                                             "above the sweep's upper bound",
                "bound": f"> {shared[-1]}"}

    for (L0, d0), (L1, d1) in zip(diffs, diffs[1:]):
        if d0 <= 0 < d1 or d0 < 0 <= d1:
            # linear interpolation of the difference against log L
            x0, x1 = math.log(L0), math.log(L1)
            t = -d0 / (d1 - d0)
            return {
                "crossover": round(math.exp(x0 + t * (x1 - x0))),
                "bracket": [L0, L1],
                "reason": "interpolated in log-L between the bracketing measurements",
            }
    return {"crossover": None, "reason": "curves cross more than once or not cleanly"}


def plot(models, crossovers, out_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, series in models.items():
        Ls = sorted(series)
        ax.plot(Ls, [series[L][0] for L in Ls], marker="o", linewidth=2,
                linestyle="--" if classify(name) == "vit" else "-",
                label=name[:58])

    for label, info in crossovers.items():
        if info.get("crossover"):
            x = info["crossover"]
            ax.axvline(x, color="crimson", linestyle=":", linewidth=1.5)
            ax.annotate(f"crossover ≈ {x}", xy=(x, ax.get_ylim()[1] * 0.94),
                        rotation=90, va="top", ha="right", color="crimson", fontsize=9)

    ax.axvspan(196, 392, alpha=0.12, color="green")
    ax.annotate("real training\nsequences", xy=(275, ax.get_ylim()[1] * 0.5),
                ha="center", fontsize=8, color="darkgreen")
    ax.axvspan(200, 400, alpha=0.0)

    ax.set_xscale("log")
    ax.set_xlabel("sequence length L (tokens, log scale)")
    ax.set_ylabel("iterations / second  (higher is better)")
    ax.set_title("Jamba vs ViT throughput — crossover is the objective")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep", help="file containing benchmark_seq_scaling.py stdout, or - for stdin")
    p.add_argument("--out", default="outputs", help="directory for crossover.json / crossover.png")
    p.add_argument("--target", type=int, default=400,
                   help="crossover we are trying to reach (default 400)")
    args = p.parse_args()

    # utf-8-sig, and an explicit lstrip for the stdin path: on Windows a leading BOM makes the
    # first "=== model ===" header fail to match, which silently drops that model's entire
    # table. Silently -- the tool still succeeds and just reports one fewer curve.
    text = (sys.stdin.read() if args.sweep == "-"
            else Path(args.sweep).read_text(encoding="utf-8-sig", errors="replace"))
    models = parse(text.lstrip("﻿"))
    if not models:
        print("no model tables found -- is this benchmark_seq_scaling.py output?", file=sys.stderr)
        return 1

    vits = {n: s for n, s in models.items() if classify(n) == "vit"}
    jambas = {n: s for n, s in models.items() if classify(n) == "jamba"}
    if not vits:
        print("no ViT baseline in this sweep. A crossover number without the baseline in the "
              "same run is not a result -- see the honesty constraint in the project contract.",
              file=sys.stderr)
        return 1
    if not jambas:
        print(f"parsed {len(models)} table(s), all classified as ViT. Nothing to compare.",
              file=sys.stderr)
        return 1

    crossovers = {}
    for jname, jseries in jambas.items():
        for vname, vseries in vits.items():
            key = f"{jname}  vs  {vname}"
            crossovers[key] = find_crossover(
                {L: v[0] for L, v in jseries.items()},
                {L: v[0] for L, v in vseries.items()},
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    best = min((c["crossover"] for c in crossovers.values() if c.get("crossover")), default=None)
    result = {
        "crossovers": crossovers,
        "best_crossover": best,
        "target": args.target,
        "target_met": bool(best and best <= args.target),
        "models": {n: {str(L): {"it_per_s": v[0], "peak_mem_gb": v[1]}
                       for L, v in s.items()} for n, s in models.items()},
    }
    (out / "crossover.json").write_text(json.dumps(result, indent=2))

    charted = plot(models, crossovers, out / "crossover.png")

    for key, info in crossovers.items():
        val = info.get("crossover")
        print(f"{key}\n    crossover = {val if val else info.get('bound', 'n/a')}"
              f"   ({info['reason']})")
    print()
    if best:
        verdict = "TARGET MET" if best <= args.target else f"still {best / args.target:.1f}x above target"
        print(f"best crossover = {best} tokens   target = {args.target}   -> {verdict}")
    else:
        print("no crossover bracketed within the swept range")
    print(f"wrote {out / 'crossover.json'}" + (f" and {out / 'crossover.png'}" if charted
                                               else "  (no matplotlib -- chart skipped)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
