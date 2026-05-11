#!/bin/bash
#SBATCH --job-name=stratego_rank_sweep
#SBATCH --account=m4341_g
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/rank_sweep_%j.out
#SBATCH --error=logs/rank_sweep_%j.err

# Strong-scaling rank (GPU) sweep across variants. Total episodes are held
# fixed per variant; only NGPU varies. Pairs with bench_batch_sweep.sh /
# bench_thread_sweep.sh and shares the manifest line format so
# plot/plot_breakdown.py works against this manifest too.
#
# Args:
#   $1  TARGET_EPISODES  total games held fixed across N (default 65536).
#                        per-rank batches = TARGET_EPISODES / (BATCH * NGPU)
#                        since main_train divides num_episodes by world_size.
#   $2  SETUP            random|default|probabilistic (default random).
#
# Each sub-run writes logs/bench_<variant>_<setup>_n<N>_<stamp>.jsonl. We
# diff the directory listing before/after each srun to recover which jsonl
# came from which (variant, N) and record it in the manifest.

module load PrgEnv-gnu
module load gcc/12.2.0
module load cray-mpich/8.1.25
module load cudatoolkit

export MPICH_GPU_SUPPORT_ENABLED=0
export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=PHB

# Held fixed across the rank sweep so only NGPU varies. Picked near the
# throughput plateau from bench_thread_sweep.sh -- enough CPU help to keep
# rollouts from starving the GPU, not so many that we contend with the
# trainer's other threads.
export OMP_NUM_THREADS=8

TARGET_EPISODES=${1:-65536}
SETUP=${2:-random}

VARIANTS=(tiny classic)
# Per-variant batch size, held fixed across the rank sweep. Same defaults as
# bench_thread_sweep.sh so the breakdown plot is comparable side-by-side.
BATCH_tiny=256
BATCH_classic=256

NGPU_LIST=(1 2 3 4)

mkdir -p logs models
STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="logs/rank_sweep_${SETUP}_${STAMP}_manifest.txt"

echo "# stratego rank (GPU) sweep across variants (${STAMP})" > "$MANIFEST"
echo "# setup=$SETUP target_episodes=$TARGET_EPISODES omp_threads=$OMP_NUM_THREADS" >> "$MANIFEST"
echo "Manifest: $MANIFEST"

for VARIANT in "${VARIANTS[@]}"; do
    declare -n BATCH_VAR="BATCH_${VARIANT}"
    BATCH=$BATCH_VAR
    for NGPU in "${NGPU_LIST[@]}"; do
        EPISODES=$TARGET_EPISODES
        # main_train.cpp:303 divides num_episodes by world_size, so per-rank
        # batches = EPISODES / (BATCH * NGPU). Warn when that leaves fewer
        # than 2 gradient steps per rank -- the throughput / per-batch plots
        # need >=2 batch rows to compute a delta.
        NB=$((EPISODES / (BATCH * NGPU)))
        if [ "$NB" -lt 2 ]; then
            echo "WARN: VARIANT=$VARIANT BATCH=$BATCH NGPU=$NGPU only yields $NB batch(es) per rank" \
                 "from TARGET_EPISODES=$EPISODES -- throughput plot will be NaN. Increase TARGET_EPISODES" \
                 "to at least $((2 * BATCH * NGPU)) to get usable timing." >&2
        fi

        echo
        echo "========== VARIANT=$VARIANT NGPU=$NGPU BATCH=$BATCH NB=$NB EPISODES=$EPISODES =========="

        BEFORE_MS=$(date +%s%3N)
        BEFORE_LOGS=$(ls -1 logs/bench_${VARIANT}_${SETUP}_n${NGPU}_*.jsonl 2>/dev/null | sort)

        srun --ntasks=$NGPU --gpus-per-node=4 --cpus-per-task=8 --mpi=cray_shasta \
             ./build/src/rl/train_stratego "$VARIANT" "$SETUP" "$EPISODES" "$BATCH" 0

        AFTER_MS=$(date +%s%3N)
        AFTER_LOGS=$(ls -1 logs/bench_${VARIANT}_${SETUP}_n${NGPU}_*.jsonl 2>/dev/null | sort)
        NEW_LOG=$(comm -13 <(echo "$BEFORE_LOGS") <(echo "$AFTER_LOGS") | tail -n 1)

        WALL_MS=$((AFTER_MS - BEFORE_MS))

        echo "VARIANT=$VARIANT NGPU=$NGPU BATCH=$BATCH wall=${WALL_MS}ms episodes=$EPISODES num_batches=$NB log=$NEW_LOG" \
            | tee -a "$MANIFEST"
    done
done

echo
echo "Done. To plot:"
echo "  python plot/plot_breakdown.py  $MANIFEST    # stacked stage breakdown (shared with batch/thread sweep)"
echo "  python plot/plot_rank_sweep.py $MANIFEST    # speedup / throughput / per-batch vs N"
