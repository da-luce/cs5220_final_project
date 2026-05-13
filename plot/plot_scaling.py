"""
Strong-scaling speedup plot.

Reads a manifest produced by bench_perlmutter.sh -- one line per N with
the form `N=<n> wall=<ms>ms log=<path>` -- and emits a single panel:
wall-clock speedup vs N.

Usage:
    python plot/plot_scaling.py logs/bench_<...>_manifest.txt
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


MANIFEST_LINE = re.compile(r"^N=(\d+)\s+wall=(\d+)ms\s+log=(\S+)\s*$")


def parse_manifest(path: Path):
    runs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = MANIFEST_LINE.match(line)
        if not m:
            print(f"warn: skipping unparseable manifest line: {line}", file=sys.stderr)
            continue
        n, wall_ms, log = m.groups()
        runs.append({"n": int(n), "wall_ms": int(wall_ms), "log": log})
    runs.sort(key=lambda r: r["n"])
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="path to logs/bench_..._manifest.txt")
    ap.add_argument("--out", default=None,
                    help="output PNG path (default: alongside manifest)")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        sys.exit(f"manifest not found: {manifest_path}")

    runs = parse_manifest(manifest_path)
    if not runs:
        sys.exit("no runs parsed from manifest")

    base = next((r for r in runs if r["n"] == 1), runs[0])
    base_wall = base["wall_ms"]

    ns = [r["n"] for r in runs]
    speedup = [base_wall / r["wall_ms"] if r["wall_ms"] else float("nan")
               for r in runs]

    print(f"{'N':>3} {'wall_ms':>10} {'speedup':>10}")
    for r, sp in zip(runs, speedup):
        print(f"{r['n']:>3} {r['wall_ms']:>10} {sp:>10.3f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(ns, ns, "k--", alpha=0.4, label="ideal")
    ax.plot(ns, speedup, "o-", label="measured")
    ax.set_xlabel("GPUs (N)")
    ax.set_ylabel("Speedup vs N=1")
    ax.set_xticks(ns)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    out_path = Path(args.out) if args.out else manifest_path.with_suffix(".png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nplot -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
