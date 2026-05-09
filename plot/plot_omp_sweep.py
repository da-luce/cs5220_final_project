"""
OpenMP thread-count sweep plot for the Stratego RL trainer.

Reads a manifest produced by bench_omp_sweep.sh -- one line per thread count
with the form
    `OMP=<t> wall=<ms>ms batch=<b> episodes=<e> num_batches=<n> log=<path>`
-- and emits a 3-panel figure characterizing CPU thread scaling for batched
env stepping:

  (left)   Throughput (games/sec) vs OMP_NUM_THREADS. Marks the peak.
  (center) Speedup vs OMP_NUM_THREADS, with ideal linear line.
  (right)  Parallel efficiency (speedup / threads).

Like plot_batch_sweep.py, throughput is computed from the median per-batch
delta in the jsonl `time_ms` column, skipping the first batch as warm-up.
Batch size is fixed across all runs so throughput differences are entirely
attributable to thread scaling on env stepping (and any libtorch/allocator
contention from the shared pinned `obs_buf`/`mask_buf`).

Usage:
    python plot/plot_omp_sweep.py logs/omp_sweep_<...>_manifest.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


MANIFEST_LINE = re.compile(
    r"^OMP=(\d+)\s+wall=(\d+)ms\s+batch=(\d+)\s+episodes=(\d+)\s+num_batches=(\d+)\s+log=(\S+)\s*$"
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
        t, wall_ms, b, ep, nb, log = m.groups()
        runs.append({
            "threads": int(t),
            "wall_ms": int(wall_ms),
            "batch": int(b),
            "episodes": int(ep),
            "num_batches": int(nb),
            "log": log,
        })
    runs.sort(key=lambda r: r["threads"])
    return runs


def load_batch_times_ms(jsonl_path: Path, skip_first: int = 1):
    """Return per-batch wall-clock deltas (ms), skipping `skip_first` warm-up batches."""
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
    if len(cumulative) < skip_first + 2:
        return []
    deltas = [cumulative[i] - cumulative[i - 1] for i in range(1, len(cumulative))]
    return deltas[skip_first:]


def median(xs):
    if not xs:
        return float("nan")
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="path to logs/omp_sweep_..._manifest.txt")
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
            print(f"warn: missing log for OMP={r['threads']}: {log}", file=sys.stderr)
            r["batch_times_ms"] = []
            r["median_batch_ms"] = float("nan")
            r["throughput"] = float("nan")
            continue

        deltas = load_batch_times_ms(log)
        r["batch_times_ms"] = deltas
        r["median_batch_ms"] = median(deltas)
        if r["median_batch_ms"] == r["median_batch_ms"] and r["median_batch_ms"] > 0:
            # Batch is fixed across runs; throughput = batch / time
            r["throughput"] = r["batch"] * 1000.0 / r["median_batch_ms"]
        else:
            r["throughput"] = float("nan")

    # Baseline = single-threaded run (or smallest threads if 1 wasn't measured)
    base = next((r for r in runs if r["threads"] == 1), runs[0])
    base_t = base["median_batch_ms"]
    for r in runs:
        if r["median_batch_ms"] == r["median_batch_ms"] and r["median_batch_ms"] > 0:
            r["speedup"] = base_t / r["median_batch_ms"]
            r["efficiency"] = r["speedup"] / r["threads"]
        else:
            r["speedup"] = float("nan")
            r["efficiency"] = float("nan")

    valid = [r for r in runs if r["throughput"] == r["throughput"]]
    peak = max(valid, key=lambda r: r["throughput"], default=None)

    print(f"{'OMP':>4} {'med_batch_ms':>14} {'throughput_g/s':>16} "
          f"{'speedup':>9} {'efficiency':>11}")
    for r in runs:
        marker = "  <- peak" if peak is not None and r["threads"] == peak["threads"] else ""
        print(f"{r['threads']:>4} {r['median_batch_ms']:>14.2f} "
              f"{r['throughput']:>16.1f} {r['speedup']:>9.3f} "
              f"{r['efficiency']:>11.3f}{marker}")

    ts  = [r["threads"] for r in runs]
    thr = [r["throughput"] for r in runs]
    sp  = [r["speedup"] for r in runs]
    eff = [r["efficiency"] for r in runs]

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Stratego RL: OMP-Thread Sweep ({manifest_path.stem})",
                 fontsize=13)

    # -- Throughput --
    ax = axs[0]
    ax.plot(ts, thr, "o-", color="tab:blue")
    if peak is not None:
        ax.axvline(peak["threads"], color="tab:red", linestyle="--", alpha=0.5,
                   label=f"peak: T={peak['threads']} ({peak['throughput']:.0f} g/s)")
        ax.legend(loc="lower right")
    ax.set_xlabel("OMP_NUM_THREADS")
    ax.set_ylabel("Throughput (games / sec)")
    ax.set_title(f"Throughput (BATCH={runs[0]['batch']}, fixed)")
    ax.set_xticks(ts)
    ax.grid(True, alpha=0.3)

    # -- Speedup --
    ax = axs[1]
    ax.plot(ts, ts, "k--", alpha=0.4, label="ideal")
    ax.plot(ts, sp, "s-", color="tab:green", label="measured")
    ax.set_xlabel("OMP_NUM_THREADS")
    ax.set_ylabel(f"Speedup vs OMP={base['threads']}")
    ax.set_title("Parallel speedup")
    ax.set_xticks(ts)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # -- Efficiency --
    ax = axs[2]
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.4, label="ideal")
    ax.plot(ts, eff, "^-", color="tab:purple", label="measured")
    ax.set_xlabel("OMP_NUM_THREADS")
    ax.set_ylabel("Parallel efficiency (speedup / T)")
    ax.set_title("Efficiency")
    ax.set_xticks(ts)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    out_path = Path(args.out) if args.out else manifest_path.with_suffix(".png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nplot -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
