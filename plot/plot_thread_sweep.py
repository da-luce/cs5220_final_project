"""
OpenMP thread sweep plot for the Stratego RL trainer.

Reads a manifest produced by bench_thread_sweep.sh. Lines look like

    VARIANT=<v> THREADS=<t> BATCH=<b> wall=<ms>ms episodes=<e> num_batches=<n> log=<path>

Emits a 3-panel figure characterizing how OpenMP thread count scales the
CPU-side work, with one curve per variant:

  (left)   Throughput (games/sec) vs thread count. Marks per-variant peak.
  (center) Median per-batch wall time vs thread count (log-log).
  (right)  Wall time per game vs thread count (lower = better).

Throughput is computed as BATCH / median_per_batch_time. We use the median
across batches in the run (skipping the first as warm-up) because the first
iteration includes lazy CUDA context init, cuDNN algo selection, and the
caching-allocator first-growth costs.

Usage:
    python plot/plot_thread_sweep.py logs/thread_sweep_<...>_manifest.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


MANIFEST_LINE = re.compile(
    r"^(?:VARIANT=(?P<variant>\S+)\s+)?"
    r"THREADS=(?P<threads>\d+)\s+"
    r"BATCH=(?P<batch>\d+)\s+"
    r"wall=(?P<wall>\d+)ms\s+"
    r"episodes=(?P<ep>\d+)\s+"
    r"num_batches=(?P<nb>\d+)\s+"
    r"log=(?P<log>\S+)\s*$"
)

VARIANT_HEADER_RE = re.compile(r"variant=(\S+)")

# Same palette as plot_batch_sweep.py so a variant looks identical across plots.
VARIANT_COLORS = {
    "tiny":    "tab:blue",
    "quick":   "tab:purple",
    "barrage": "tab:orange",
    "classic": "tab:green",
}


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
            print(f"warn: no variant for line (and none in header): {line}", file=sys.stderr)
            continue
        runs.append({
            "variant": variant,
            "threads": int(m.group("threads")),
            "batch": int(m.group("batch")),
            "wall_ms": int(m.group("wall")),
            "episodes": int(m.group("ep")),
            "num_batches": int(m.group("nb")),
            "log": m.group("log"),
        })
    return runs


def load_batch_times_ms(jsonl_path: Path, skip_first: int = 1):
    """Per-batch wall-clock deltas (ms), dropping `skip_first` warm-up deltas
    when the run has enough updates to spare them."""
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


def color_for(variant: str, fallback_cycle):
    if variant in VARIANT_COLORS:
        return VARIANT_COLORS[variant]
    return next(fallback_cycle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="path to logs/thread_sweep_..._manifest.txt")
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
            print(f"warn: missing log for variant={r['variant']} T={r['threads']}: {log}",
                  file=sys.stderr)
            r["batch_times_ms"] = []
            r["median_batch_ms"] = float("nan")
            r["throughput"] = float("nan")
            r["ms_per_game"] = float("nan")
            continue

        deltas = load_batch_times_ms(log)
        r["batch_times_ms"] = deltas
        r["median_batch_ms"] = median(deltas)
        if r["median_batch_ms"] == r["median_batch_ms"] and r["median_batch_ms"] > 0:
            r["throughput"]  = r["batch"] * 1000.0 / r["median_batch_ms"]
            r["ms_per_game"] = r["median_batch_ms"] / r["batch"]
        else:
            r["throughput"]  = float("nan")
            r["ms_per_game"] = float("nan")

    variants = []
    by_variant = {}
    for r in runs:
        if r["variant"] not in by_variant:
            by_variant[r["variant"]] = []
            variants.append(r["variant"])
        by_variant[r["variant"]].append(r)
    for v in variants:
        by_variant[v].sort(key=lambda r: r["threads"])

    peaks = {}
    for v, vrows in by_variant.items():
        valid = [r for r in vrows if r["throughput"] == r["throughput"]]
        if valid:
            peaks[v] = max(valid, key=lambda r: r["throughput"])

    print(f"{'variant':>10} {'THREADS':>8} {'BATCH':>6} {'med_batch_ms':>14} "
          f"{'throughput_g/s':>16} {'ms/game':>10} {'wall_ms':>10}")
    for v in variants:
        for r in by_variant[v]:
            peak = peaks.get(v)
            marker = "  <- peak" if peak is not None and r["threads"] == peak["threads"] else ""
            print(f"{r['variant']:>10} {r['threads']:>8} {r['batch']:>6} "
                  f"{r['median_batch_ms']:>14.2f} {r['throughput']:>16.1f} "
                  f"{r['ms_per_game']:>10.3f} {r['wall_ms']:>10}{marker}")

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Stratego RL: OMP Thread Sweep ({manifest_path.stem})",
                 fontsize=13)

    fallback_cycle = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    variant_color = {v: color_for(v, fallback_cycle) for v in variants}

    all_ts = sorted({r["threads"] for r in runs})

    # -- Throughput --
    ax = axs[0]
    for v in variants:
        vrows = by_variant[v]
        ts = [r["threads"] for r in vrows]
        thr = [r["throughput"] for r in vrows]
        # Include the fixed batch in the legend so the curves are unambiguous.
        label = f"{v} (B={vrows[0]['batch']})"
        ax.plot(ts, thr, "o-", color=variant_color[v], label=label)
        peak = peaks.get(v)
        if peak is not None:
            ax.axvline(peak["threads"], color=variant_color[v], linestyle="--",
                       alpha=0.4)
            ax.annotate(f"peak T={peak['threads']}\n{peak['throughput']:.0f} g/s",
                        xy=(peak["threads"], peak["throughput"]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=8, color=variant_color[v])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("OMP_NUM_THREADS (per rank)")
    ax.set_ylabel("Throughput (games / sec)")
    ax.set_title("Throughput vs thread count")
    ax.set_xticks(all_ts)
    ax.set_xticklabels([str(t) for t in all_ts])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="variant")

    # -- Per-batch wall time --
    ax = axs[1]
    for v in variants:
        vrows = by_variant[v]
        ts = [r["threads"] for r in vrows]
        med_t = [r["median_batch_ms"] for r in vrows]
        ax.plot(ts, med_t, "s-", color=variant_color[v], label=v)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("OMP_NUM_THREADS (per rank)")
    ax.set_ylabel("Median per-batch time (ms)")
    ax.set_title("Per-batch wall time")
    ax.set_xticks(all_ts)
    ax.set_xticklabels([str(t) for t in all_ts])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="variant")

    # -- ms / game --
    ax = axs[2]
    for v in variants:
        vrows = by_variant[v]
        ts = [r["threads"] for r in vrows]
        mpg = [r["ms_per_game"] for r in vrows]
        ax.plot(ts, mpg, "^-", color=variant_color[v], label=v)
        peak = peaks.get(v)
        if peak is not None:
            ax.axvline(peak["threads"], color=variant_color[v], linestyle="--",
                       alpha=0.4)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("OMP_NUM_THREADS (per rank)")
    ax.set_ylabel("Wall time per game (ms)")
    ax.set_title("Per-game cost (lower = better)")
    ax.set_xticks(all_ts)
    ax.set_xticklabels([str(t) for t in all_ts])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="variant")

    plt.tight_layout()

    out_path = Path(args.out) if args.out else manifest_path.with_suffix(".png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nplot -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
