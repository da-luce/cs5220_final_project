"""
Throughput vs env batch size, one curve per variant.

Reads a manifest produced by bench_batch_sweep.sh. Lines look like

    VARIANT=<v> BATCH=<b> wall=<ms>ms episodes=<e> num_batches=<n> log=<path>

Usage:
    python plot/plot_batch_sweep.py logs/batch_sweep_<...>_manifest.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from _colors import VARIANT_COLORS, color_for


MANIFEST_LINE = re.compile(
    r"^(?:VARIANT=(?P<variant>\S+)\s+)?BATCH=(?P<batch>\d+)\s+"
    r"wall=(?P<wall>\d+)ms\s+"
    r"episodes=(?P<ep>\d+)\s+"
    r"num_batches=(?P<nb>\d+)\s+"
    r"log=(?P<log>\S+)\s*$"
)
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
        variant = m.group("variant") or header_variant
        if variant is None:
            print(f"warn: no variant for line: {line}", file=sys.stderr)
            continue
        runs.append({
            "variant": variant,
            "batch": int(m.group("batch")),
            "wall_ms": int(m.group("wall")),
            "episodes": int(m.group("ep")),
            "num_batches": int(m.group("nb")),
            "log": m.group("log"),
        })
    return runs


def load_batch_times_ms(jsonl_path: Path, skip_first: int = 1):
    cumulative = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "meta" in row or "summary" in row:
                continue
            if "time_ms" in row and "batch" in row:
                cumulative.append(row["time_ms"])
    if len(cumulative) < 2:
        return []
    deltas = [cumulative[i] - cumulative[i - 1] for i in range(1, len(cumulative))]
    if len(deltas) > skip_first:
        return deltas[skip_first:]
    return deltas


def median(xs):
    if not xs:
        return float("nan")
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="path to logs/batch_sweep_..._manifest.txt")
    ap.add_argument("--out", default=None,
                    help="output PNG path (default: alongside manifest)")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        sys.exit(f"manifest not found: {manifest_path}")

    runs = parse_manifest(manifest_path)
    if not runs:
        sys.exit("no runs parsed from manifest")

    project_root = manifest_path.resolve().parent.parent

    for r in runs:
        log = Path(r["log"])
        if not log.is_absolute():
            log = project_root / log
        if not log.exists():
            print(f"warn: missing log for variant={r['variant']} B={r['batch']}: {log}",
                  file=sys.stderr)
            r["median_batch_ms"] = float("nan")
            r["throughput"] = float("nan")
            continue
        r["median_batch_ms"] = median(load_batch_times_ms(log))
        if r["median_batch_ms"] == r["median_batch_ms"] and r["median_batch_ms"] > 0:
            r["throughput"] = r["batch"] * 1000.0 / r["median_batch_ms"]
        else:
            r["throughput"] = float("nan")

    variants = []
    by_variant = {}
    for r in runs:
        if r["variant"] not in by_variant:
            by_variant[r["variant"]] = []
            variants.append(r["variant"])
        by_variant[r["variant"]].append(r)
    for v in variants:
        by_variant[v].sort(key=lambda r: r["batch"])

    peaks = {}
    for v, vrows in by_variant.items():
        valid = [r for r in vrows if r["throughput"] == r["throughput"]]
        if valid:
            peaks[v] = max(valid, key=lambda r: r["throughput"])

    print(f"{'variant':>10} {'BATCH':>8} {'med_batch_ms':>14} {'throughput_g/s':>16}")
    for v in variants:
        for r in by_variant[v]:
            peak = peaks.get(v)
            marker = "  <- peak" if peak is not None and r["batch"] == peak["batch"] else ""
            print(f"{r['variant']:>10} {r['batch']:>8} "
                  f"{r['median_batch_ms']:>14.2f} {r['throughput']:>16.1f}{marker}")

    fig, ax = plt.subplots(figsize=(7, 5))
    fallback_cycle = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    variant_color = {v: color_for(v, fallback_cycle) for v in variants}
    all_bs = sorted({r["batch"] for r in runs})

    for v in variants:
        vrows = by_variant[v]
        bs = [r["batch"] for r in vrows]
        thr = [r["throughput"] for r in vrows]
        ax.plot(bs, thr, "o-", color=variant_color[v], label=v)
        peak = peaks.get(v)
        if peak is not None:
            ax.axvline(peak["batch"], color=variant_color[v], linestyle="--",
                       alpha=0.4)
            ax.annotate(f"peak B={peak['batch']}\n{peak['throughput']:.0f} g/s",
                        xy=(peak["batch"], peak["throughput"]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=8, color=variant_color[v])

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Env batch size (games / rank)")
    ax.set_ylabel("Throughput (games / sec)")
    ax.set_xticks(all_bs)
    ax.set_xticklabels([str(b) for b in all_bs])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="variant")

    plt.tight_layout()

    out_path = Path(args.out) if args.out else manifest_path.with_suffix(".png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nplot -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
