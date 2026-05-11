"""
Stacked bar chart of per-stage training time across (variant, batch_size)
combos.

Reads a manifest produced by bench_batch_sweep.sh — lines of the form

    VARIANT=<v> BATCH=<b> wall=<ms>ms episodes=<n> num_batches=<n> log=<path>

For each entry, loads the profiling record from the JSONL log and stacks
the per-stage totals into one bar. Stages collapse into seven buckets:

    rollout_aux, env_step, infer_gpu, ppo_train, allreduce  (called out)
    misc        — everything else the profiler measured (h2d, d2h, kl, ...)
    overhead    — total_time_ms - sum(measured stages); unmeasured time

Bars are grouped by variant on the x-axis and ordered by batch size.

Usage:
    python plot/plot_batch_sweep.py logs/batch_sweep_combined_..._manifest.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Stages we explicitly call out, in stack order (bottom -> top).
HEADLINE_STAGES = ["env_step", "infer_gpu", "rollout_aux", "ppo_train", "allreduce"]

# Category order in the stacked bar (bottom -> top) and their colors.
CATEGORIES = [
    ("env_step",    "tab:orange"),
    ("infer_gpu",   "tab:blue"),
    ("rollout_aux", "tab:cyan"),
    ("ppo_train",   "tab:green"),
    ("allreduce",   "tab:red"),
    ("misc",        "tab:gray"),
    ("overhead",    "lightgray"),
]

# Accepts both the legacy single-variant manifest and the combined one.
LINE_RE = re.compile(
    r"^(?:VARIANT=(?P<variant>\S+)\s+)?BATCH=(?P<batch>\d+)\s+"
    r"wall=(?P<wall>\d+)ms\s+"
    r"episodes=(?P<ep>\d+)\s+"
    r"num_batches=(?P<nb>\d+)\s+"
    r"log=(?P<log>\S+)\s*$"
)

VARIANT_HEADER_RE = re.compile(r"variant=(\S+)")


def parse_manifest(path: Path):
    header_variant = None
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = VARIANT_HEADER_RE.search(line)
            if m:
                header_variant = m.group(1)
            continue
        m = LINE_RE.match(line)
        if not m:
            print(f"warn: skipping unparseable manifest line: {line}", file=sys.stderr)
            continue
        variant = m.group("variant") or header_variant
        if variant is None:
            print(f"warn: no variant for line (and none in header): {line}", file=sys.stderr)
            continue
        rows.append({
            "variant": variant,
            "batch": int(m.group("batch")),
            "wall_ms": int(m.group("wall")),
            "episodes": int(m.group("ep")),
            "num_batches": int(m.group("nb")),
            "log": m.group("log"),
        })
    return rows


def load_profile(jsonl_path: Path):
    """Return (total_time_ms, {stage_name: total_ms}) or (None, None) if absent."""
    if not jsonl_path.exists():
        return None, None
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("profiling"):
                stages = {k: v.get("total_ms", 0.0) for k, v in row.get("stages", {}).items()}
                return float(row.get("total_time_ms", 0.0)), stages
    return None, None


def categorize(total_ms: float, stages: dict):
    """Return dict {category: ms} summing to total_ms."""
    out = {c: 0.0 for c, _ in CATEGORIES}
    measured = 0.0
    for name, ms in stages.items():
        measured += ms
        if name in HEADLINE_STAGES:
            out[name] += ms
        else:
            out["misc"] += ms
    out["overhead"] = max(0.0, total_ms - measured)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="path to logs/batch_sweep_..._manifest.txt")
    ap.add_argument("--out", default=None, help="output PNG path (default: alongside manifest)")
    ap.add_argument("--per-batch", action="store_true",
                    help="divide each stack by num_batches to show per-step time")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        sys.exit(f"manifest not found: {manifest_path}")

    rows = parse_manifest(manifest_path)
    if not rows:
        sys.exit("no runs parsed from manifest")

    project_root = manifest_path.resolve().parent.parent

    # Attach profile data.
    for r in rows:
        log_path = Path(r["log"])
        if not log_path.is_absolute():
            log_path = project_root / log_path
        total_ms, stages = load_profile(log_path)
        if total_ms is None:
            print(f"warn: no profiling record in {log_path}", file=sys.stderr)
            r["categories"] = None
            r["total_ms"] = 0.0
            continue
        r["total_ms"] = total_ms
        r["categories"] = categorize(total_ms, stages)

    rows = [r for r in rows if r["categories"] is not None]
    if not rows:
        sys.exit("no rows with profiling data; rerun bench_batch_sweep.sh to regenerate logs")

    # Order: variant groups (preserving first-seen order), batch ascending within.
    variants_seen = []
    for r in rows:
        if r["variant"] not in variants_seen:
            variants_seen.append(r["variant"])
    rows.sort(key=lambda r: (variants_seen.index(r["variant"]), r["batch"]))

    n = len(rows)
    fig_w = max(8.0, 0.7 * n + 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, 6.5))

    x = list(range(n))
    bottoms = [0.0] * n
    scale = 1.0 / 1000.0  # ms -> s
    for category, color in CATEGORIES:
        heights = []
        for r in rows:
            h = r["categories"][category] * scale
            if args.per_batch and r["num_batches"]:
                h /= r["num_batches"]
            heights.append(h)
        ax.bar(x, heights, bottom=bottoms, color=color, edgecolor="white",
               linewidth=0.5, label=category)
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    # Per-bar batch-size label below x-axis.
    ax.set_xticks(x)
    ax.set_xticklabels([f"B={r['batch']}" for r in rows], rotation=0, fontsize=9)

    # Variant group labels (under the tick labels) and dividers between groups.
    ylim_top = max(bottoms) * 1.18 if bottoms else 1.0
    ax.set_ylim(0, ylim_top)
    group_start = 0
    for i in range(1, n + 1):
        if i == n or rows[i]["variant"] != rows[group_start]["variant"]:
            mid = 0.5 * (group_start + (i - 1))
            ax.text(mid, -ylim_top * 0.085, rows[group_start]["variant"],
                    ha="center", va="top", fontsize=12, fontweight="bold",
                    transform=ax.transData)
            if i < n:
                ax.axvline(i - 0.5, color="black", alpha=0.25, linewidth=0.8)
            group_start = i

    ax.set_ylabel("Time per step (s)" if args.per_batch else "Wall time (s)")
    title = "Training time breakdown — env batch-size sweep"
    if args.per_batch:
        title += " (per gradient step)"
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # Total time annotation atop each bar.
    for xi, total in zip(x, bottoms):
        ax.text(xi, total + ylim_top * 0.01, f"{total:.1f}s",
                ha="center", va="bottom", fontsize=8, color="black")

    # Legend in bottom->top order (matches stack from bottom up).
    handles = [mpatches.Patch(color=c, label=name) for name, c in CATEGORIES]
    ax.legend(handles=list(reversed(handles)), loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0., frameon=False)

    plt.tight_layout()

    out_path = Path(args.out) if args.out else manifest_path.with_suffix(".png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"plot -> {out_path.resolve()}")

    # Also dump a small text summary.
    print()
    hdr = f"{'variant':>8} {'batch':>6} {'NB':>4} {'total_s':>8}"
    for name, _ in CATEGORIES:
        hdr += f" {name:>11}"
    print(hdr)
    for r in rows:
        total_s = r["total_ms"] / 1000.0
        line = f"{r['variant']:>8} {r['batch']:>6} {r['num_batches']:>4} {total_s:>8.2f}"
        for name, _ in CATEGORIES:
            line += f" {r['categories'][name]/1000.0:>11.2f}"
        print(line)


if __name__ == "__main__":
    main()
