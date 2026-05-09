"""
Environment batch-size sweep plot for the Stratego RL trainer.

Reads a manifest produced by bench_batch_sweep.sh -- one line per batch size
with the form `BATCH=<b> wall=<ms>ms episodes=<e> num_batches=<n> log=<path>`
-- and emits a 3-panel figure characterizing the CPU-vs-GPU tradeoff:

  (left)   Throughput (games/sec) vs batch size. Marks the peak.
  (center) Median per-batch wall time vs batch size (log-log).
  (right)  Wall time per game vs batch size (lower = better utilization).

Throughput is computed as BATCH / median_per_batch_time -- one PPO "batch"
in the metrics jsonl corresponds to BATCH games completed (sync_freq=batch).
We use the median across batches in the run (skipping the first as warm-up)
because the first iteration includes lazy CUDA context init, cuDNN algo
selection, and caching-allocator warm-up.

Usage:
    python plot/plot_batch_sweep.py logs/batch_sweep_<...>_manifest.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


MANIFEST_LINE = re.compile(
    r"^BATCH=(\d+)\s+wall=(\d+)ms\s+episodes=(\d+)\s+num_batches=(\d+)\s+log=(\S+)\s*$"
)


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
        b, wall_ms, ep, nb, log = m.groups()
        runs.append({
            "batch": int(b),
            "wall_ms": int(wall_ms),
            "episodes": int(ep),
            "num_batches": int(nb),
            "log": log,
        })
    runs.sort(key=lambda r: r["batch"])
    return runs


def load_batch_times_ms(jsonl_path: Path, skip_first: int = 1):
    """Return per-batch wall-clock deltas (ms), skipping up to `skip_first`
    warm-up deltas. If the run has too few PPO updates to spare a warm-up
    delta (large-BATCH runs may have only 2 updates total = 1 delta), we keep
    whatever we have rather than returning empty -- noisy is better than nothing."""
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
            print(f"warn: missing log for B={r['batch']}: {log}", file=sys.stderr)
            r["batch_times_ms"] = []
            r["median_batch_ms"] = float("nan")
            r["throughput"] = float("nan")
            r["ms_per_game"] = float("nan")
            continue

        deltas = load_batch_times_ms(log)
        r["batch_times_ms"] = deltas
        r["median_batch_ms"] = median(deltas)
        # One "batch" in the log corresponds to BATCH games completed
        # (sync_freq=batch in main_train.cpp), so throughput is straightforward.
        if r["median_batch_ms"] == r["median_batch_ms"] and r["median_batch_ms"] > 0:
            r["throughput"]  = r["batch"] * 1000.0 / r["median_batch_ms"]
            r["ms_per_game"] = r["median_batch_ms"] / r["batch"]
        else:
            r["throughput"]  = float("nan")
            r["ms_per_game"] = float("nan")

    # Find peak throughput row
    valid = [r for r in runs if r["throughput"] == r["throughput"]]
    peak = max(valid, key=lambda r: r["throughput"], default=None)

    print(f"{'BATCH':>8} {'med_batch_ms':>14} {'throughput_g/s':>16} "
          f"{'ms/game':>10} {'wall_ms':>10}")
    for r in runs:
        marker = "  <- peak" if peak is not None and r["batch"] == peak["batch"] else ""
        print(f"{r['batch']:>8} {r['median_batch_ms']:>14.2f} "
              f"{r['throughput']:>16.1f} {r['ms_per_game']:>10.3f} "
              f"{r['wall_ms']:>10}{marker}")

    bs    = [r["batch"] for r in runs]
    med_t = [r["median_batch_ms"] for r in runs]
    thr   = [r["throughput"] for r in runs]
    mpg   = [r["ms_per_game"] for r in runs]

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Stratego RL: Env Batch-Size Sweep ({manifest_path.stem})",
                 fontsize=13)

    # -- Throughput --
    ax = axs[0]
    ax.plot(bs, thr, "o-", color="tab:blue")
    if peak is not None:
        ax.axvline(peak["batch"], color="tab:red", linestyle="--", alpha=0.5,
                   label=f"peak: B={peak['batch']} ({peak['throughput']:.0f} g/s)")
        ax.legend(loc="lower right")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Env batch size (games / rank)")
    ax.set_ylabel("Throughput (games / sec)")
    ax.set_title("Throughput vs batch size")
    ax.set_xticks(bs)
    ax.set_xticklabels([str(b) for b in bs])
    ax.grid(True, which="both", alpha=0.3)

    # -- Per-batch wall time --
    ax = axs[1]
    ax.plot(bs, med_t, "s-", color="tab:green")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Env batch size (games / rank)")
    ax.set_ylabel("Median per-batch time (ms)")
    ax.set_title("Per-batch wall time")
    ax.set_xticks(bs)
    ax.set_xticklabels([str(b) for b in bs])
    ax.grid(True, which="both", alpha=0.3)

    # -- ms / game --
    ax = axs[2]
    ax.plot(bs, mpg, "^-", color="tab:purple")
    if peak is not None:
        ax.axvline(peak["batch"], color="tab:red", linestyle="--", alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Env batch size (games / rank)")
    ax.set_ylabel("Wall time per game (ms)")
    ax.set_title("Per-game cost (lower = better)")
    ax.set_xticks(bs)
    ax.set_xticklabels([str(b) for b in bs])
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    out_path = Path(args.out) if args.out else manifest_path.with_suffix(".png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nplot -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
