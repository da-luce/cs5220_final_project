"""
Stacked bar chart of per-stage training time across (variant, sweep_value)
combos. Shared between the env batch-size sweep (bench_batch_sweep.sh), the
OMP thread sweep (bench_thread_sweep.sh), and the rank/GPU sweep
(bench_rank_sweep.sh) — the manifest line format overlaps and we
auto-detect which dimension varies.

Manifest lines look like

    VARIANT=<v> [THREADS=<t>] [BATCH=<b>] [NGPU=<n>] wall=<ms>ms episodes=<n> num_batches=<n> log=<path>

The sweep dimension is whichever of {BATCH, THREADS, NGPU} varies within
at least one variant (override with --x BATCH|THREADS|NGPU). Stages
collapse into eight buckets:

    rollout_aux, env_step, infer_h2d, infer_gpu, ppo_train, allreduce  (called out)
    misc        — everything else the profiler measured (d2h, kl, ...)
    overhead    — total_time_ms - sum(measured stages); unmeasured time

Bars are grouped by variant on the x-axis and ordered by the sweep value.

Usage:
    python plot/plot_breakdown.py logs/batch_sweep_..._manifest.txt
    python plot/plot_breakdown.py logs/thread_sweep_..._manifest.txt
    python plot/plot_breakdown.py logs/rank_sweep_..._manifest.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


HEADLINE_STAGES = ["env_step", "infer_h2d", "infer_gpu", "rollout_aux", "ppo_train", "allreduce"]

CATEGORIES = [
    ("env_step",    "tab:orange"),
    ("infer_h2d",   "tab:purple"),
    ("infer_gpu",   "tab:blue"),
    ("rollout_aux", "tab:cyan"),
    ("ppo_train",   "tab:green"),
    ("allreduce",   "tab:red"),
    ("misc",        "tab:gray"),
    ("overhead",    "lightgray"),
]

# Match: any order of `KEY=value` tokens, then the fixed wall=...log= tail.
# Lets us add new sweep dimensions (NGPU, ...) without touching the regex.
LINE_RE = re.compile(
    r"^(?P<prefix>(?:\w+=\S+\s+)*)"
    r"wall=(?P<wall>\d+)ms\s+"
    r"episodes=(?P<ep>\d+)\s+"
    r"num_batches=(?P<nb>\d+)\s+"
    r"log=(?P<log>\S+)\s*$"
)
PREFIX_TOKEN_RE = re.compile(r"(\w+)=(\S+)")

VARIANT_HEADER_RE = re.compile(r"variant=(\S+)")

X_LABELS = {"BATCH": "B", "THREADS": "T", "NGPU": "N"}
# Mapping from sweep-dim name to the row key holding the integer value.
DIM_KEYS = {"BATCH": "batch", "THREADS": "threads", "NGPU": "ngpu"}


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
        fields = dict(PREFIX_TOKEN_RE.findall(m.group("prefix")))
        variant = fields.get("VARIANT", header_variant)
        if variant is None:
            print(f"warn: no variant for line (and none in header): {line}", file=sys.stderr)
            continue
        rows.append({
            "variant": variant,
            "batch":   int(fields["BATCH"])   if "BATCH"   in fields else None,
            "threads": int(fields["THREADS"]) if "THREADS" in fields else None,
            "ngpu":    int(fields["NGPU"])    if "NGPU"    in fields else None,
            "wall_ms": int(m.group("wall")),
            "episodes": int(m.group("ep")),
            "num_batches": int(m.group("nb")),
            "log": m.group("log"),
        })
    return rows


def detect_sweep_dim(rows):
    """Return 'BATCH', 'THREADS', or 'NGPU' — whichever varies within any variant."""
    by_variant = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)
    for dim, key in (("NGPU", "ngpu"), ("THREADS", "threads"), ("BATCH", "batch")):
        for vrows in by_variant.values():
            vals = {r[key] for r in vrows if r[key] is not None}
            if len(vals) > 1:
                return dim
    # Nothing varies — fall back to whichever field is present.
    for dim, key in (("NGPU", "ngpu"), ("THREADS", "threads"), ("BATCH", "batch")):
        for r in rows:
            if r[key] is not None:
                return dim
    sys.exit("manifest has no BATCH, THREADS, or NGPU field to sweep over")


def load_profile(jsonl_path: Path):
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
    ap.add_argument("manifest", help="path to logs/{batch,thread}_sweep_..._manifest.txt")
    ap.add_argument("--out", default=None, help="output PNG path (default: alongside manifest)")
    ap.add_argument("--per-batch", action="store_true",
                    help="divide each stack by num_batches to show per-step time")
    ap.add_argument("--x", choices=["BATCH", "THREADS", "NGPU"], default=None,
                    help="sweep dimension (default: auto-detect from manifest)")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        sys.exit(f"manifest not found: {manifest_path}")

    rows = parse_manifest(manifest_path)
    if not rows:
        sys.exit("no runs parsed from manifest")

    sweep_dim = args.x or detect_sweep_dim(rows)
    sweep_key = DIM_KEYS[sweep_dim]
    x_prefix = X_LABELS[sweep_dim]

    project_root = manifest_path.resolve().parent.parent

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

    rows = [r for r in rows if r["categories"] is not None and r[sweep_key] is not None]
    if not rows:
        sys.exit("no rows with profiling data; rerun the sweep to regenerate logs")

    # Order: variant groups (preserving first-seen order), sweep value ascending within.
    variants_seen = []
    for r in rows:
        if r["variant"] not in variants_seen:
            variants_seen.append(r["variant"])
    rows.sort(key=lambda r: (variants_seen.index(r["variant"]), r[sweep_key]))

    n = len(rows)
    fig_w = max(8.0, 0.7 * n + 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, 6.5))

    x = list(range(n))
    bottoms = [0.0] * n
    scale = 1.0 / 1000.0
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

    ax.set_xticks(x)
    ax.set_xticklabels([f"{x_prefix}={r[sweep_key]}" for r in rows], rotation=0, fontsize=9)

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
    sweep_name = {"BATCH": "env batch-size", "THREADS": "OMP thread",
                  "NGPU": "rank (GPU)"}[sweep_dim]
    title = f"Training time breakdown — {sweep_name} sweep"
    if args.per_batch:
        title += " (per gradient step)"
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    for xi, total in zip(x, bottoms):
        ax.text(xi, total + ylim_top * 0.01, f"{total:.1f}s",
                ha="center", va="bottom", fontsize=8, color="black")

    handles = [mpatches.Patch(color=c, label=name) for name, c in CATEGORIES]
    ax.legend(handles=list(reversed(handles)), loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0., frameon=False)

    plt.tight_layout()

    out_path = Path(args.out) if args.out else manifest_path.with_suffix(".png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"plot -> {out_path.resolve()}")

    print()
    hdr = f"{'variant':>8} {x_prefix:>6} {'NB':>4} {'total_s':>8}"
    for name, _ in CATEGORIES:
        hdr += f" {name:>11}"
    print(hdr)
    for r in rows:
        total_s = r["total_ms"] / 1000.0
        line = f"{r['variant']:>8} {r[sweep_key]:>6} {r['num_batches']:>4} {total_s:>8.2f}"
        for name, _ in CATEGORIES:
            line += f" {r['categories'][name]/1000.0:>11.2f}"
        print(line)


if __name__ == "__main__":
    main()
