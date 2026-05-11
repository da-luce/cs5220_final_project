#!/bin/bash
#SBATCH --job-name=stratego_bench
#SBATCH --account=m4341_g
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err

# Strong-scaling sweep on a single node: run with 1, 2, 3, 4 GPUs back-to-back,
# holding total games-played fixed so wall-time speedup ~= T(1)/T(N).
#
# Args:
#   $1  NUM_BATCHES   total gradient steps when run on 1 GPU (default 80).
#                     With N GPUs the per-rank batches = NUM_BATCHES / N
#                     (total games stays fixed for strong scaling).
#   $2  BATCH         per-rank batch size (default 256).
#   $3  VARIANT       tiny|quick|barrage|classic (default tiny).
#   $4  SETUP         random|default|probabilistic (default random).
#
# Eval is disabled (5th positional arg = 0) so the wall-time reflects
# pure training cost, not the single-threaded eval games on rank 0.
#
# Each sub-run writes logs/bench_<variant>_<setup>_n<N>_<stamp>.jsonl. The
# manifest written below records exactly which file came from which N so
# plot/plot_scaling.py can correlate them.

module load PrgEnv-gnu
module load gcc/12.2.0
module load cray-mpich/8.1.25
module load cudatoolkit

export MPICH_GPU_SUPPORT_ENABLED=0
export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=PHB

NUM_BATCHES=${1:-80}
BATCH=${2:-256}
VARIANT=${3:-tiny}
SETUP=${4:-random}

EPISODES=$((NUM_BATCHES * BATCH))   # strong scaling: total games fixed across N

mkdir -p logs models
STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="logs/bench_${VARIANT}_${SETUP}_${STAMP}_manifest.txt"

echo "Strong-scaling sweep: NUM_BATCHES=$NUM_BATCHES BATCH=$BATCH EPISODES=$EPISODES"
echo "Manifest: $MANIFEST"
echo "# stratego strong-scaling sweep ($STAMP)" >  "$MANIFEST"
echo "# variant=$VARIANT setup=$SETUP num_batches=$NUM_BATCHES batch=$BATCH episodes=$EPISODES" >> "$MANIFEST"

for N in 1 2 3 4; do
    echo
    echo "========== N=$N GPU(s) =========="
    
    # Get start time in milliseconds
    BEFORE_MS=$(date +%s%3N)

    BEFORE_LOGS=$(ls -1 logs/bench_${VARIANT}_${SETUP}_n${N}_*.jsonl 2>/dev/null | sort)

    srun --ntasks=$N --gpus-per-node=4 --cpus-per-task=8 --mpi=cray_shasta \
         ./build/rl/train_stratego "$VARIANT" "$SETUP" "$EPISODES" "$BATCH" 0

    # Get end time in milliseconds
    AFTER_MS=$(date +%s%3N)
    
    AFTER_LOGS=$(ls -1 logs/bench_${VARIANT}_${SETUP}_n${N}_*.jsonl 2>/dev/null | sort)
    NEW_LOG=$(comm -13 <(echo "$BEFORE_LOGS") <(echo "$AFTER_LOGS") | tail -n 1)

    # Calculate difference in ms
    WALL_MS=$((AFTER_MS - BEFORE_MS))

    # Update the echo/tee command to record ms
    echo "N=$N wall=${WALL_MS}ms log=$NEW_LOG" | tee -a "$MANIFEST"
done

echo
echo "Done. To plot speedup:"
echo "  python plot/plot_scaling.py $MANIFEST"
