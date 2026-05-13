"""
Strong-scaling speedup plot (rank/GPU sweep), one curve per variant.

Reads a manifest produced by bench_rank_sweep.sh. Lines look like

    VARIANT=<v> NGPU=<n> BATCH=<b> wall=<ms>ms episodes=<e> num_batches=<nb> log=<path>

Speedup is wall-clock vs N=1.

Usage:
    python plot/plot_rank_sweep.py logs/rank_sweep_<...>_manifest.txt
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from _colors import VARIANT_COLORS, color_for


MANIFEST_LINE = re.compile(
    r"^(?P<prefix>(?:\w+=\S+\s+)*)"
    r"wall=(?P<wall>\d+)ms\s+"
    r"episodes=(?P<ep>\d+)\s+"
    r"num_batches=(?P<nb>\d+)\s+"
    r"log=(?P<log>\S+)\s*$"
)
PREFIX_TOKEN_RE = re.compile(r"(\w+)=(\S+)")
VARIANT_HEADER_RE = re.compile(r"variant=(\S+)")


def parse_manifest(path: Path):
    header_variant = None
    runs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = VARIANT_HEADER_RE.search(line)
            if m:
                header_variant = m.group(1)
            continue
        m = MANIFEST_LINE.match(line)
        if not m:
            print(f"warn: skipping unparseable manifest line: {line}", file=sys.stderr)
            continue
        fields = dict(PREFIX_TOKEN_RE.findall(m.group("prefix")))
        variant = fields.get("VARIANT", header_variant)
        if variant is None or "NGPU" not in fields:
            print(f"warn: skipping line: {line}", file=sys.stderr)
            continue
        runs.append({
            "variant": variant,
            "ngpu":  int(fields["NGPU"]),
            "batch": int(fields["BATCH"]) if "BATCH" in fields else 0,
            "wall_ms":     int(m.group("wall")),
            "episodes":    int(m.group("ep")),
            "num_batches": int(m.group("nb")),
            "log":         m.group("log"),
        })
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="path to logs/rank_sweep_..._manifest.txt")
    ap.add_argument("--out", default=None,
                    help="output PNG path (default: alongside manifest)")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        sys.exit(f"manifest not found: {manifest_path}")

    runs = parse_manifest(manifest_path)
    if not runs:
        sys.exit("no runs parsed from manifest")

    variants = []
    by_variant = {}
    for r in runs:
        if r["variant"] not in by_variant:
            by_variant[r["variant"]] = []
            variants.append(r["variant"])
        by_variant[r["variant"]].append(r)
    for v in variants:
        by_variant[v].sort(key=lambda r: r["ngpu"])

    base_by_variant = {v: next((r for r in vrows if r["ngpu"] == 1), vrows[0])
                       for v, vrows in by_variant.items()}

    print(f"{'variant':>10} {'NGPU':>5} {'wall_ms':>10} {'speedup':>10}")
    for v in variants:
        base = base_by_variant[v]
        for r in by_variant[v]:
            sp = base["wall_ms"] / r["wall_ms"] if r["wall_ms"] else float("nan")
            print(f"{r['variant']:>10} {r['ngpu']:>5} "
                  f"{r['wall_ms']:>10} {sp:>10.3f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    fallback_cycle = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    variant_color = {v: color_for(v, fallback_cycle) for v in variants}
    all_ns = sorted({r["ngpu"] for r in runs})

    ax.plot(all_ns, all_ns, "k--", alpha=0.4, label="ideal")
    for v in variants:
        vrows = by_variant[v]
        base = base_by_variant[v]
        ns = [r["ngpu"] for r in vrows]
        sp = [base["wall_ms"] / r["wall_ms"] if r["wall_ms"] else float("nan")
              for r in vrows]
        ax.plot(ns, sp, "o-", color=variant_color[v], label=v)

    ax.set_xlabel("GPUs (N)")
    ax.set_ylabel("Speedup vs N=1")
    ax.set_xticks(all_ns)
    ax.grid(True, alpha=0.3)
    ax.legend(title="variant")

    plt.tight_layout()

    out_path = Path(args.out) if args.out else manifest_path.with_suffix(".png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nplot -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
