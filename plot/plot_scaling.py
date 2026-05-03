"""
Strong-scaling speedup plot for the Stratego RL trainer.

Reads a manifest produced by bench_perlmutter.sh -- one line per N with
the form `N=<n> wall=<ms>ms log=<path>` -- and emits a 2-panel figure:

  (left)  Speedup vs N, with ideal line.
  (right) Median per-batch wall time vs N (lower is better).

Two speedup curves are computed:
  - "wall":  from the bash-measured wall-clock per run (includes
             startup, NCCL init, model save, etc.).
  - "median per-batch": from the metrics jsonl `time_ms` column,
             taking diffs between consecutive batches and using the
             median. Skips the first batch to avoid warm-up bias.

Median per-batch is the more honest measurement of the steady-state
training step cost; wall is what you'd quote in a paper.

Usage:
    python plot/plot_scaling.py logs/bench_<...>_manifest.txt
"""

import argparse
import json
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


def load_batch_times_ms(jsonl_path: Path):
    """Return per-batch wall-clock deltas in ms, skipping the first batch."""
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
    return deltas


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

    project_root = manifest_path.resolve().parent.parent

    for r in runs:
        log_path = Path(r["log"])
        if not log_path.is_absolute():
            log_path = project_root / log_path
        if not log_path.exists():
            print(f"warn: missing log for N={r['n']}: {log_path}", file=sys.stderr)
            r["batch_times_ms"] = []
            r["median_batch_ms"] = float("nan")
            continue
        deltas = load_batch_times_ms(log_path)
        r["batch_times_ms"] = deltas
        if deltas:
            sorted_d = sorted(deltas)
            mid = len(sorted_d) // 2
            r["median_batch_ms"] = (sorted_d[mid] if len(sorted_d) % 2
                                    else 0.5 * (sorted_d[mid - 1] + sorted_d[mid]))
        else:
            r["median_batch_ms"] = float("nan")

    base = next((r for r in runs if r["n"] == 1), runs[0])
    base_wall = base["wall_ms"]
    base_med = base["median_batch_ms"]

    ns = [r["n"] for r in runs]
    wall_speedup = [base_wall / r["wall_ms"] if r["wall_ms"] else float("nan")
                    for r in runs]
    med_speedup = [base_med / r["median_batch_ms"]
                   if r["median_batch_ms"] and r["median_batch_ms"] == r["median_batch_ms"]
                   else float("nan")
                   for r in runs]

    print(f"{'N':>3} {'wall_ms':>10} {'speedup_wall':>14} "
          f"{'med_batch_ms':>14} {'speedup_med':>13} {'efficiency':>11}")
    for r, sw, sm in zip(runs, wall_speedup, med_speedup):
        eff = (sm / r["n"]) if sm == sm else float("nan")
        print(f"{r['n']:>3} {r['wall_ms']:>10} {sw:>14.3f} "
              f"{r['median_batch_ms']:>14.2f} {sm:>13.3f} {eff:>11.3f}")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Stratego RL: Strong-Scaling on Perlmutter ({manifest_path.stem})",
                 fontsize=13)

    ax = axs[0]
    ax.plot(ns, ns, "k--", alpha=0.4, label="ideal")
    ax.plot(ns, wall_speedup, "o-", color="tab:blue", label="wall-clock")
    ax.plot(ns, med_speedup, "s-", color="tab:green", label="median per-batch")
    ax.set_xlabel("GPUs (N)")
    ax.set_ylabel("Speedup vs N=1")
    ax.set_title("Parallel speedup")
    ax.set_xticks(ns)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axs[1]
    med_times = [r["median_batch_ms"] for r in runs]
    ax.plot(ns, med_times, "s-", color="tab:green", label="median per-batch")
    wall_per_batch = []
    for r in runs:
        n_batches_per_rank = len(r["batch_times_ms"]) + 1 if r["batch_times_ms"] else 0
        wall_per_batch.append(r["wall_ms"] / n_batches_per_rank
                              if n_batches_per_rank else float("nan"))
    ax.plot(ns, wall_per_batch, "o--", color="tab:blue", alpha=0.6,
            label="wall / batches-per-rank")
    ax.set_xlabel("GPUs (N)")
    ax.set_ylabel("Time per batch (ms)")
    ax.set_title("Per-batch wall time (lower is better)")
    ax.set_xticks(ns)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    out_path = Path(args.out) if args.out else manifest_path.with_suffix(".png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nplot -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
