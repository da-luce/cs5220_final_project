#!/bin/bash
#SBATCH --job-name=stratego_batch_sweep
#SBATCH --account=m4341_g
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/batch_sweep_%j.out
#SBATCH --error=logs/batch_sweep_%j.err

module load PrgEnv-gnu
module load cudatoolkit

export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_IB_HCA=mlx5

export OMP_NUM_THREADS=8 # Empirically found this to be the best for both tiny and classic

# Arguments
TARGET_EPISODES=${1:-65536} # Allows us to go up to B=8192 for tiny
SETUP=${2:-random}
NGPU=${3:-4}

VARIANTS=(tiny classic)
# Per-variant batch grids. classic OOMs / breaks past 1024; tiny is small
# enough to keep climbing.
BATCHES_tiny=(32 64 128 256 512 1024 2048 4096 8192)
BATCHES_classic=(32 64 128 256 512 1024)

mkdir -p logs models
STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="logs/batch_sweep_fixed_${SETUP}_${STAMP}_manifest.txt"

echo "# stratego batch-size sweep across variants (${STAMP})" > "$MANIFEST"
echo "# setup=$SETUP target_episodes=$TARGET_EPISODES gpus=$NGPU" >> "$MANIFEST"
echo "Manifest: $MANIFEST"

for VARIANT in "${VARIANTS[@]}"; do
    # Resolve the per-variant batch list via nameref.
    declare -n BATCH_LIST="BATCHES_${VARIANT}"
    for BATCH in "${BATCH_LIST[@]}"; do
        EPISODES=$TARGET_EPISODES
        # The trainer divides num_episodes by world_size (main_train.cpp:303),
        # so per-rank batches = EPISODES / (BATCH * NGPU). Warn when that
        # leaves fewer than 2 gradient steps per rank -- the throughput plot
        # needs >=2 batch rows to compute a delta.
        NB=$((EPISODES / (BATCH * NGPU)))
        if [ "$NB" -lt 2 ]; then
            echo "WARN: VARIANT=$VARIANT BATCH=$BATCH NGPU=$NGPU only yields $NB batch(es) per rank" \
                 "from TARGET_EPISODES=$EPISODES -- throughput plot will be NaN. Increase TARGET_EPISODES" \
                 "to at least $((2 * BATCH * NGPU)) to get usable timing." >&2
        fi

        echo
        echo "========== VARIANT=$VARIANT BATCH=$BATCH NB=$NB EPISODES=$EPISODES NGPU=$NGPU =========="

        BEFORE_MS=$(date +%s%3N)
        BEFORE_LOGS=$(ls -1 logs/bench_${VARIANT}_${SETUP}_n${NGPU}_*.jsonl 2>/dev/null | sort)

        srun --ntasks=$NGPU --gpus-per-node=4 --cpus-per-task=8 --mpi=cray_shasta \
             ./build/src/rl/train_stratego "$VARIANT" "$SETUP" "$EPISODES" "$BATCH" 0

        AFTER_MS=$(date +%s%3N)
        AFTER_LOGS=$(ls -1 logs/bench_${VARIANT}_${SETUP}_n${NGPU}_*.jsonl 2>/dev/null | sort)
        NEW_LOG=$(comm -13 <(echo "$BEFORE_LOGS") <(echo "$AFTER_LOGS") | tail -n 1)

        WALL_MS=$((AFTER_MS - BEFORE_MS))

        # THREADS is fixed here but logged for parity with bench_thread_sweep.sh
        # so plot/plot_breakdown.py can ingest either manifest.
        echo "VARIANT=$VARIANT THREADS=$OMP_NUM_THREADS BATCH=$BATCH wall=${WALL_MS}ms episodes=$EPISODES num_batches=$NB log=$NEW_LOG" \
            | tee -a "$MANIFEST"
    done
done

echo
echo "Done. To plot:"
echo "  python plot/plot_breakdown.py   $MANIFEST    # stacked stage breakdown (shared with thread sweep)"
echo "  python plot/plot_batch_sweep.py $MANIFEST    # throughput / per-batch / ms-per-game"