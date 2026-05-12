#!/bin/bash
#SBATCH --job-name=stratego_thread_sweep
#SBATCH --account=m4341_g
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=logs/thread_sweep_%j.out
#SBATCH --error=logs/thread_sweep_%j.err

# OpenMP thread sweep across both variants. BATCH and total episodes are
# held fixed per variant; only OMP_NUM_THREADS varies. Pairs with
# bench_batch_sweep.sh and shares the manifest line format so
# plot/plot_breakdown.py works against either output.
#
# Perlmutter GPU nodes have 64 physical cores / 128 logical CPUs. With 4
# tasks/node, cpus-per-task=32 is the max — enough to host
# OMP_NUM_THREADS=32 without oversubscribing the rank's allocation.

module load PrgEnv-gnu
module load cudatoolkit

export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_IB_HCA=mlx5

# Arguments
TARGET_EPISODES=${1:-65536}
SETUP=${2:-random}
NGPU=${3:-4}

VARIANTS=(tiny classic)
# Per-variant batch size, held fixed across the thread sweep. Picked near
# (but conservatively below) the throughput peak from bench_batch_sweep.sh
# so the CPU stages still have meaningful work — at very large BATCH the
# GPU dominates and the thread count barely registers.
BATCH_tiny=256
BATCH_classic=256

THREADS_LIST=(1 2 4 8 16 32)

mkdir -p logs models
STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="logs/thread_sweep_${SETUP}_${STAMP}_manifest.txt"

echo "# stratego OMP thread sweep across variants (${STAMP})" > "$MANIFEST"
echo "# setup=$SETUP target_episodes=$TARGET_EPISODES gpus=$NGPU" >> "$MANIFEST"
echo "Manifest: $MANIFEST"

for VARIANT in "${VARIANTS[@]}"; do
    declare -n BATCH_VAR="BATCH_${VARIANT}"
    BATCH=$BATCH_VAR
    for THREADS in "${THREADS_LIST[@]}"; do
        EPISODES=$TARGET_EPISODES
        NB=$((EPISODES / (BATCH * NGPU)))
        if [ "$NB" -lt 2 ]; then
            echo "WARN: VARIANT=$VARIANT BATCH=$BATCH NGPU=$NGPU only yields $NB batch(es) per rank" \
                 "from TARGET_EPISODES=$EPISODES -- throughput plot will be NaN. Increase TARGET_EPISODES" \
                 "to at least $((2 * BATCH * NGPU)) to get usable timing." >&2
        fi

        echo
        echo "========== VARIANT=$VARIANT THREADS=$THREADS BATCH=$BATCH NB=$NB EPISODES=$EPISODES NGPU=$NGPU =========="

        BEFORE_MS=$(date +%s%3N)
        BEFORE_LOGS=$(ls -1 logs/bench_${VARIANT}_${SETUP}_n${NGPU}_*.jsonl 2>/dev/null | sort)

        export OMP_NUM_THREADS=$THREADS
        srun --ntasks=$NGPU --gpus-per-node=4 --cpus-per-task=32 --mpi=cray_shasta \
             ./build/src/rl/train_stratego "$VARIANT" "$SETUP" "$EPISODES" "$BATCH" 0

        AFTER_MS=$(date +%s%3N)
        AFTER_LOGS=$(ls -1 logs/bench_${VARIANT}_${SETUP}_n${NGPU}_*.jsonl 2>/dev/null | sort)
        NEW_LOG=$(comm -13 <(echo "$BEFORE_LOGS") <(echo "$AFTER_LOGS") | tail -n 1)

        WALL_MS=$((AFTER_MS - BEFORE_MS))

        echo "VARIANT=$VARIANT THREADS=$THREADS BATCH=$BATCH wall=${WALL_MS}ms episodes=$EPISODES num_batches=$NB log=$NEW_LOG" \
            | tee -a "$MANIFEST"
    done
done

echo
echo "Done. To plot:"
echo "  python plot/plot_breakdown.py    $MANIFEST    # stacked stage breakdown (shared with batch sweep)"
echo "  python plot/plot_thread_sweep.py $MANIFEST    # throughput / per-batch / ms-per-game vs threads"
