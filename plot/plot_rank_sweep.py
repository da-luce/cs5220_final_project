"""
Rank (GPU) sweep plot for the Stratego RL trainer.

Reads a manifest produced by bench_rank_sweep.sh. Lines look like

    VARIANT=<v> NGPU=<n> BATCH=<b> wall=<ms>ms episodes=<e> num_batches=<nb> log=<path>

Emits a 3-panel figure characterizing strong scaling, one curve per variant:

  (left)   Speedup vs N, with ideal y=x line. Two curves per variant: the
           bash-measured wall-clock speedup (includes NCCL init, model save,
           etc.) and the median per-batch speedup (steady-state PPO step).
  (center) Aggregate throughput (games/sec across the cluster) vs N. Each
           PPO step plays BATCH games on each of N ranks, so aggregate
           throughput = N * BATCH / median_per_batch_time.
  (right)  Median per-batch wall time vs N. Flat = ideal weak scaling per
           step; rising = allreduce / sync overhead dominating.

Median per-batch is taken across PPO updates in the run (dropping the first
as warm-up to skip lazy CUDA init / cuDNN auto-tune / first-allreduce
group construction). Median per-batch is the honest measurement of
steady-state cost; wall is what you'd quote in a paper.

Usage:
    python plot/plot_rank_sweep.py logs/rank_sweep_<...>_manifest.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


# Order-agnostic prefix parse (matches plot_breakdown.py).
MANIFEST_LINE = re.compile(
    r"^(?P<prefix>(?:\w+=\S+\s+)*)"
    r"wall=(?P<wall>\d+)ms\s+"
    r"episodes=(?P<ep>\d+)\s+"
    r"num_batches=(?P<nb>\d+)\s+"
    r"log=(?P<log>\S+)\s*$"
)
PREFIX_TOKEN_RE = re.compile(r"(\w+)=(\S+)")

VARIANT_HEADER_RE = re.compile(r"variant=(\S+)")

# Same palette as the batch/thread sweep plots so a variant looks identical
# across the report figures.
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
        fields = dict(PREFIX_TOKEN_RE.findall(m.group("prefix")))
        variant = fields.get("VARIANT", header_variant)
        if variant is None:
            print(f"warn: no variant for line (and none in header): {line}", file=sys.stderr)
            continue
        if "NGPU" not in fields:
            print(f"warn: skipping line without NGPU: {line}", file=sys.stderr)
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


def load_batch_times_ms(jsonl_path: Path, skip_first: int = 1):
    """Per-batch wall-clock deltas (ms), dropping `skip_first` warm-up deltas
    when the run has enough updates to spare them.

    Batch 1 includes one-time costs that never recur (lazy CUDA context
    init, cuDNN algo auto-tune, caching-allocator first growth, NCCL group
    construction on the first allreduce). Keeping it would bias the median
    upward, especially at large N where the run is short."""
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

    project_root = manifest_path.resolve().parent.parent

    for r in runs:
        log = Path(r["log"])
        if not log.is_absolute():
            log = project_root / log
        if not log.exists():
            print(f"warn: missing log for variant={r['variant']} N={r['ngpu']}: {log}",
                  file=sys.stderr)
            r["batch_times_ms"] = []
            r["median_batch_ms"] = float("nan")
            r["throughput"] = float("nan")
            r["ms_per_game"] = float("nan")
            continue

        deltas = load_batch_times_ms(log)
        r["batch_times_ms"] = deltas
        r["median_batch_ms"] = median(deltas)
        # Aggregate throughput: each PPO step plays BATCH games on each of
        # N ranks, so cluster-wide games/sec = N * BATCH * 1000 / med_ms.
        if r["median_batch_ms"] == r["median_batch_ms"] and r["median_batch_ms"] > 0:
            games_per_step  = r["ngpu"] * r["batch"]
            r["throughput"] = games_per_step * 1000.0 / r["median_batch_ms"]
            r["ms_per_game"] = r["median_batch_ms"] / games_per_step if games_per_step else float("nan")
        else:
            r["throughput"]  = float("nan")
            r["ms_per_game"] = float("nan")

    # Group by variant (preserve first-seen order), sort each group by N.
    variants = []
    by_variant = {}
    for r in runs:
        if r["variant"] not in by_variant:
            by_variant[r["variant"]] = []
            variants.append(r["variant"])
        by_variant[r["variant"]].append(r)
    for v in variants:
        by_variant[v].sort(key=lambda r: r["ngpu"])

    # Per-variant baseline (N=1 if present, else the smallest N).
    base_by_variant = {}
    for v, vrows in by_variant.items():
        base = next((r for r in vrows if r["ngpu"] == 1), vrows[0])
        base_by_variant[v] = base

    print(f"{'variant':>10} {'NGPU':>5} {'BATCH':>6} {'med_batch_ms':>14} "
          f"{'throughput_g/s':>16} {'speedup_wall':>14} {'speedup_med':>13} "
          f"{'eff':>6} {'wall_ms':>10}")
    for v in variants:
        base = base_by_variant[v]
        for r in by_variant[v]:
            sw = base["wall_ms"] / r["wall_ms"] if r["wall_ms"] else float("nan")
            sm = (base["median_batch_ms"] / r["median_batch_ms"]
                  if r["median_batch_ms"] == r["median_batch_ms"]
                  and r["median_batch_ms"] > 0
                  and base["median_batch_ms"] == base["median_batch_ms"]
                  else float("nan"))
            eff = sw / r["ngpu"] if sw == sw else float("nan")
            print(f"{r['variant']:>10} {r['ngpu']:>5} {r['batch']:>6} "
                  f"{r['median_batch_ms']:>14.2f} {r['throughput']:>16.1f} "
                  f"{sw:>14.3f} {sm:>13.3f} {eff:>6.2f} {r['wall_ms']:>10}")

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Stratego RL: Rank (GPU) Sweep ({manifest_path.stem})",
                 fontsize=13)

    fallback_cycle = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    variant_color = {v: color_for(v, fallback_cycle) for v in variants}

    all_ns = sorted({r["ngpu"] for r in runs})

    # -- Speedup --
    ax = axs[0]
    ax.plot(all_ns, all_ns, "k--", alpha=0.4, label="ideal")
    for v in variants:
        vrows = by_variant[v]
        base = base_by_variant[v]
        ns = [r["ngpu"] for r in vrows]
        sw = [base["wall_ms"] / r["wall_ms"] if r["wall_ms"] else float("nan")
              for r in vrows]
        sm = [(base["median_batch_ms"] / r["median_batch_ms"]
               if r["median_batch_ms"] == r["median_batch_ms"]
               and r["median_batch_ms"] > 0
               and base["median_batch_ms"] == base["median_batch_ms"]
               else float("nan"))
              for r in vrows]
        ax.plot(ns, sw, "o-", color=variant_color[v], label=f"{v} wall")
        ax.plot(ns, sm, "s--", color=variant_color[v], alpha=0.7,
                label=f"{v} median per-batch")
    ax.set_xlabel("GPUs (N)")
    ax.set_ylabel("Speedup vs N=1")
    ax.set_title("Strong-scaling speedup")
    ax.set_xticks(all_ns)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # -- Aggregate throughput --
    ax = axs[1]
    for v in variants:
        vrows = by_variant[v]
        ns = [r["ngpu"] for r in vrows]
        thr = [r["throughput"] for r in vrows]
        label = f"{v} (B={vrows[0]['batch']})"
        ax.plot(ns, thr, "o-", color=variant_color[v], label=label)
    ax.set_xlabel("GPUs (N)")
    ax.set_ylabel("Aggregate throughput (games / sec)")
    ax.set_title("Cluster throughput vs N")
    ax.set_xticks(all_ns)
    ax.grid(True, alpha=0.3)
    ax.legend(title="variant")

    # -- Per-batch wall time --
    ax = axs[2]
    for v in variants:
        vrows = by_variant[v]
        ns = [r["ngpu"] for r in vrows]
        med_t = [r["median_batch_ms"] for r in vrows]
        ax.plot(ns, med_t, "s-", color=variant_color[v], label=v)
    ax.set_xlabel("GPUs (N)")
    ax.set_ylabel("Median per-batch time (ms)")
    ax.set_title("Per-batch wall time (flat = ideal weak scaling)")
    ax.set_xticks(all_ns)
    ax.grid(True, alpha=0.3)
    ax.legend(title="variant")

    plt.tight_layout()

    out_path = Path(args.out) if args.out else manifest_path.with_suffix(".png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nplot -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
