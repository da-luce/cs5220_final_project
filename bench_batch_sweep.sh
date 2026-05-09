#!/bin/bash
#SBATCH --job-name=stratego_batch_sweep
#SBATCH --account=m4341_g
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/batch_sweep_%j.out
#SBATCH --error=logs/batch_sweep_%j.err
#SBATCH --network=single_node_vni

# Sweep per-rank environment batch size at fixed N=1 GPU to characterize the
# CPU-vs-GPU tradeoff. Holds the number of PPO updates (NUM_BATCHES) fixed so
# every run produces the same amount of statistics; total games therefore
# scales linearly with batch size. Throughput = BATCH / median_batch_time
# gives the steady-state games/sec at each setting.
#
# Expected curve shape:
#   - Small BATCH: GPU starved by kernel-launch + H2D transfer overhead;
#     PPO update fixed cost dominates per-game time.
#   - Mid   BATCH: forward pass amortizes overhead; OpenMP env stepping fits in
#     cpus-per-task cores. This is the throughput peak.
#   - Large BATCH: env stepping serializes over the limited CPU budget while
#     GPU sits idle waiting for the next rollout. Throughput plateaus or drops.
#
# We run on N=1 GPU on purpose: at multi-GPU we'd also be measuring NCCL
# all-reduce latency, which obscures the env/inference balance we're tuning
# here. cpus-per-task=8 matches what each rank gets in bench_perlmutter.sh,
# so the curve characterizes the same per-rank regime distributed runs see.
#
# Per-run NUM_BATCHES is *not* fixed across the sweep: at the largest BATCH
# we'd otherwise simulate millions of games, so we instead aim for a fixed
# total-episodes budget per run. Specifically:
#
#   target = MIN_BATCHES * max(BATCHES)        # ensures the largest batch hits the floor
#   NUM_BATCHES(B) = clamp(target / B, MIN_BATCHES, MAX_BATCHES_PER_RUN)
#
# So the largest BATCH does exactly MIN_BATCHES PPO updates (default 2),
# while small BATCHes scale up to the MAX cap to give clean median timing
# without spending forever in the tail.
#
# Args:
#   $1  MIN_BATCHES         PPO updates at the largest BATCH (default 2).
#                           Bumps the per-run floor; with skip-first-as-warmup
#                           the largest gives one usable timing sample.
#   $2  VARIANT             tiny|quick|barrage|classic (default tiny).
#   $3  SETUP               random|default|probabilistic (default random).
#   $4  BATCHES             comma-separated batch sizes
#                           (default "32,64,128,256,512,1024,2048,4096,8192,16384").
#   $5  MAX_BATCHES_PER_RUN cap on PPO updates per run (default 30) so small
#                           BATCHes don't bloat into thousands of updates.
#
# Eval is disabled (5th positional arg = 0) so wall-time reflects pure
# training, not single-threaded eval games on rank 0.

module purge
module load PrgEnv-gnu
module load gcc/12.2.0
module load cudatoolkit
module load craype-accel-nvidia80
module load cray-mpich/8.1.25

# Match bench_perlmutter.sh: keep MPI off the GPU path so NCCL owns comms.
export MPICH_GPU_SUPPORT_ENABLED=0
export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=PHB

MIN_BATCHES=${1:-2}
VARIANT=${2:-classic}
SETUP=${3:-random}
BATCHES_CSV=${4:-"32,64,128,256,512,1024,2048,4096,8192,16384"}
MAX_BATCHES_PER_RUN=${5:-30}
IFS=',' read -ra BATCHES <<< "$BATCHES_CSV"

# Find the largest BATCH so we can size the total-episode target around it.
MAX_BATCH=0
for B in "${BATCHES[@]}"; do
    [ "$B" -gt "$MAX_BATCH" ] && MAX_BATCH=$B
done
TARGET_EPISODES=$(( MIN_BATCHES * MAX_BATCH ))

mkdir -p logs models
STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="logs/batch_sweep_${VARIANT}_${SETUP}_${STAMP}_manifest.txt"

echo "Batch-size sweep: VARIANT=$VARIANT SETUP=$SETUP"
echo "MIN_BATCHES=$MIN_BATCHES MAX_BATCHES_PER_RUN=$MAX_BATCHES_PER_RUN target_episodes=$TARGET_EPISODES"
echo "Sweep:    ${BATCHES[*]}"
echo "Manifest: $MANIFEST"
{
    echo "# stratego env batch-size sweep ($STAMP)"
    echo "# variant=$VARIANT setup=$SETUP min_batches=$MIN_BATCHES max_batches=$MAX_BATCHES_PER_RUN target_episodes=$TARGET_EPISODES gpus=1"
} > "$MANIFEST"

for B in "${BATCHES[@]}"; do
    NUM_BATCHES=$(( TARGET_EPISODES / B ))
    [ "$NUM_BATCHES" -lt "$MIN_BATCHES" ] && NUM_BATCHES=$MIN_BATCHES
    [ "$NUM_BATCHES" -gt "$MAX_BATCHES_PER_RUN" ] && NUM_BATCHES=$MAX_BATCHES_PER_RUN
    EPISODES=$(( NUM_BATCHES * B ))

    echo
    echo "========== BATCH=$B (episodes=$EPISODES, batches=$NUM_BATCHES) =========="

    BEFORE_MS=$(date +%s%3N)
    # Same disambiguation trick bench_perlmutter.sh uses: train_stratego writes
    # bench_<variant>_<setup>_n1_<stamp>.jsonl, so we diff the dir before/after
    # to find the file this iteration produced.
    BEFORE_LOGS=$(ls -1 logs/bench_${VARIANT}_${SETUP}_n1_*.jsonl 2>/dev/null | sort)

    # No --network flag here: inherit whatever VNI the parent allocation has
    # (the SBATCH --network=single_node_vni above when launched via sbatch, or
    # whatever the user's salloc gave them in interactive mode).
    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
         --gpus-per-node=1 --cpus-per-task=8 --mpi=cray_shasta \
         ./build/rl/train_stratego "$VARIANT" "$SETUP" "$EPISODES" "$B" 0
    SRUN_RC=$?

    AFTER_MS=$(date +%s%3N)

    if [ $SRUN_RC -ne 0 ]; then
        echo "srun failed (rc=$SRUN_RC) for BATCH=$B; aborting sweep" | tee -a "$MANIFEST"
        exit $SRUN_RC
    fi

    AFTER_LOGS=$(ls -1 logs/bench_${VARIANT}_${SETUP}_n1_*.jsonl 2>/dev/null | sort)
    NEW_LOG=$(comm -13 <(echo "$BEFORE_LOGS") <(echo "$AFTER_LOGS") | tail -n 1)

    WALL_MS=$((AFTER_MS - BEFORE_MS))

    echo "BATCH=$B wall=${WALL_MS}ms episodes=${EPISODES} num_batches=${NUM_BATCHES} log=$NEW_LOG" \
        | tee -a "$MANIFEST"
done

echo
echo "Done. To plot:"
echo "  python plot/plot_batch_sweep.py $MANIFEST"
