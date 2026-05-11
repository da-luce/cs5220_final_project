"""
Environment batch-size sweep plot for the Stratego RL trainer.

Reads a manifest produced by bench_batch_sweep.sh. Lines look like

    VARIANT=<v> BATCH=<b> wall=<ms>ms episodes=<e> num_batches=<n> log=<path>

(the legacy single-variant manifest form -- no VARIANT= field, variant in
the header `# variant=<v>` comment -- still works). Emits a 3-panel
figure characterizing the CPU-vs-GPU tradeoff, with one curve per variant:

  (left)   Throughput (games/sec) vs batch size. Marks per-variant peak.
  (center) Median per-batch wall time vs batch size (log-log).
  (right)  Wall time per game vs batch size (lower = better utilization).

Throughput is computed as BATCH / median_per_batch_time -- one PPO "batch"
in the metrics jsonl corresponds to BATCH games completed per rank
(sync_freq=batch). We use the median across batches in the run (skipping
the first as warm-up) because the first iteration includes lazy CUDA
context init, cuDNN algo selection, and caching-allocator warm-up.

Usage:
    python plot/plot_batch_sweep.py logs/batch_sweep_<...>_manifest.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


# Accepts both new (VARIANT=... BATCH=...) and legacy (BATCH=... only) lines.
MANIFEST_LINE = re.compile(
    r"^(?:VARIANT=(?P<variant>\S+)\s+)?BATCH=(?P<batch>\d+)\s+"
    r"wall=(?P<wall>\d+)ms\s+"
    r"episodes=(?P<ep>\d+)\s+"
    r"num_batches=(?P<nb>\d+)\s+"
    r"log=(?P<log>\S+)\s*$"
)

VARIANT_HEADER_RE = re.compile(r"variant=(\S+)")

# Deterministic per-variant color so the same variant looks the same
# across the breakdown plot and this one. Extras fall back to matplotlib's
# default cycle.
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
            "batch": int(m.group("batch")),
            "wall_ms": int(m.group("wall")),
            "episodes": int(m.group("ep")),
            "num_batches": int(m.group("nb")),
            "log": m.group("log"),
        })
    return runs


def load_batch_times_ms(jsonl_path: Path, skip_first: int = 1):
    """Return per-batch wall-clock deltas (ms), skipping up to `skip_first`
    warm-up deltas. If the run has too few PPO updates to spare a warm-up
    delta (large-BATCH runs may have only 2 updates total = 1 delta), we
    keep whatever we have -- noisy is better than nothing."""
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
    # Drop the first delta: batch 1 wall-time includes one-time costs that
    # never recur -- lazy CUDA context init, cuDNN algo auto-tune, the
    # caching allocator's first growth, NCCL group construction on the first
    # allreduce. Keeping it would bias the median upward, especially at small
    # BATCH where steady-state batches are short.
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

    # Group by variant (preserve first-seen order), sort each group by batch.
    variants = []
    by_variant = {}
    for r in runs:
        if r["variant"] not in by_variant:
            by_variant[r["variant"]] = []
            variants.append(r["variant"])
        by_variant[r["variant"]].append(r)
    for v in variants:
        by_variant[v].sort(key=lambda r: r["batch"])

    # Per-variant peak (max throughput).
    peaks = {}
    for v, vrows in by_variant.items():
        valid = [r for r in vrows if r["throughput"] == r["throughput"]]
        if valid:
            peaks[v] = max(valid, key=lambda r: r["throughput"])

    # Pretty-print summary.
    print(f"{'variant':>10} {'BATCH':>8} {'med_batch_ms':>14} "
          f"{'throughput_g/s':>16} {'ms/game':>10} {'wall_ms':>10}")
    for v in variants:
        for r in by_variant[v]:
            peak = peaks.get(v)
            marker = "  <- peak" if peak is not None and r["batch"] == peak["batch"] else ""
            print(f"{r['variant']:>10} {r['batch']:>8} {r['median_batch_ms']:>14.2f} "
                  f"{r['throughput']:>16.1f} {r['ms_per_game']:>10.3f} "
                  f"{r['wall_ms']:>10}{marker}")

    # Build the figure.
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Stratego RL: Env Batch-Size Sweep ({manifest_path.stem})",
                 fontsize=13)

    fallback_cycle = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    variant_color = {v: color_for(v, fallback_cycle) for v in variants}

    # Union of batch sizes seen across variants, for x-ticks.
    all_bs = sorted({r["batch"] for r in runs})

    # -- Throughput --
    ax = axs[0]
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
    ax.set_title("Throughput vs batch size")
    ax.set_xticks(all_bs)
    ax.set_xticklabels([str(b) for b in all_bs])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="variant")

    # -- Per-batch wall time --
    ax = axs[1]
    for v in variants:
        vrows = by_variant[v]
        bs = [r["batch"] for r in vrows]
        med_t = [r["median_batch_ms"] for r in vrows]
        ax.plot(bs, med_t, "s-", color=variant_color[v], label=v)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Env batch size (games / rank)")
    ax.set_ylabel("Median per-batch time (ms)")
    ax.set_title("Per-batch wall time")
    ax.set_xticks(all_bs)
    ax.set_xticklabels([str(b) for b in all_bs])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="variant")

    # -- ms / game --
    ax = axs[2]
    for v in variants:
        vrows = by_variant[v]
        bs = [r["batch"] for r in vrows]
        mpg = [r["ms_per_game"] for r in vrows]
        ax.plot(bs, mpg, "^-", color=variant_color[v], label=v)
        peak = peaks.get(v)
        if peak is not None:
            ax.axvline(peak["batch"], color=variant_color[v], linestyle="--",
                       alpha=0.4)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Env batch size (games / rank)")
    ax.set_ylabel("Wall time per game (ms)")
    ax.set_title("Per-game cost (lower = better)")
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
