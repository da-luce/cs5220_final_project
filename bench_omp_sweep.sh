#!/bin/bash
#SBATCH --job-name=stratego_omp_sweep
#SBATCH --account=m4341_g
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=logs/omp_sweep_%j.out
#SBATCH --error=logs/omp_sweep_%j.err
#SBATCH --network=single_node_vni

# Sweep OMP_NUM_THREADS at fixed N=1 GPU and fixed batch size to find the
# optimal per-rank thread count for batched env stepping. The Stratego rules
# engine is too branchy to vectorize on GPU, so BatchedStrategoEnv's
# `#pragma omp parallel for` over the batch is the load-bearing CPU
# parallelism in the rollout phase.
#
# Each Perlmutter GPU node has dual-socket EPYC 7763 (128 physical cores
# spread over 8 NUMA domains, 4 per socket). With 4 ranks/node in
# production, each rank's natural budget is 128/4 = 32 cores -- one CCX
# group's worth, NUMA-affine. We request --cpus-per-task=32 to mirror that
# and sweep OMP threads up to the full per-rank budget.
#
# OMP_PROC_BIND=close + OMP_PLACES=cores keep threads pinned to specific
# physical cores within the SLURM cgroup, so low-T runs aren't penalized
# by thread migration and we get a clean scaling curve.
#
# Expected curve:
#   - 1-8 threads: near-linear speedup; env stepping is embarrassingly
#     parallel across the batch.
#   - 8-16 threads: diminishing returns from libtorch's caching allocator,
#     `obs_buf[e].copy_(...)` writes into the shared pinned tensor, and
#     contention on std::random_device inside env reset.
#   - 16-32 threads: plateau or regression as threads cross L3/CCX
#     boundaries on the EPYC chiplet layout.
#
# Args:
#   $1  NUM_BATCHES   PPO updates per run (default 20). First few warm-up.
#   $2  VARIANT       tiny|quick|barrage|classic (default tiny).
#   $3  SETUP         random|default|probabilistic (default random).
#   $4  BATCH         per-rank env batch size (default 512). Must be much
#                     larger than max thread count so even 32 threads have
#                     plenty of envs per thread (512/32 = 16).
#   $5  THREADS       comma-separated OMP_NUM_THREADS values
#                     (default "1,2,4,8,16,24,32").

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

# Pin OMP threads to physical cores in our SLURM cgroup; close binding
# keeps them packed onto the same NUMA domain when possible.
export OMP_PROC_BIND=close
export OMP_PLACES=cores

NUM_BATCHES=${1:-20}
VARIANT=${2:-classic}
SETUP=${3:-random}
BATCH=${4:-512}
THREADS_CSV=${5:-"1,2,4,8,16,24,32"}
IFS=',' read -ra THREADS <<< "$THREADS_CSV"

EPISODES=$(( NUM_BATCHES * BATCH ))

mkdir -p logs models
STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="logs/omp_sweep_${VARIANT}_${SETUP}_${STAMP}_manifest.txt"

echo "OMP-thread sweep: NUM_BATCHES=$NUM_BATCHES VARIANT=$VARIANT SETUP=$SETUP BATCH=$BATCH"
echo "Sweep:    ${THREADS[*]}"
echo "Manifest: $MANIFEST"
{
    echo "# stratego OMP-thread sweep ($STAMP)"
    echo "# variant=$VARIANT setup=$SETUP num_batches=$NUM_BATCHES batch=$BATCH gpus=1"
} > "$MANIFEST"

for T in "${THREADS[@]}"; do
    echo
    echo "========== OMP_NUM_THREADS=$T (batch=$BATCH episodes=$EPISODES) =========="

    BEFORE_MS=$(date +%s%3N)
    # train_stratego writes bench_<variant>_<setup>_n1_<stamp>.jsonl, so we
    # diff the dir before/after to find the file produced this iteration.
    BEFORE_LOGS=$(ls -1 logs/bench_${VARIANT}_${SETUP}_n1_*.jsonl 2>/dev/null | sort)

    # Keep --cpus-per-task=32 (= the full per-rank cgroup) regardless of T,
    # and let OMP_PROC_BIND/PLACES restrict actual thread placement to T
    # physical cores. This way the OS scheduler has the same context across
    # all sweep points, and only the OMP-visible parallelism varies.
    # No --network flag here: inherit whatever VNI the parent allocation has
    # (SBATCH --network when launched via sbatch, or salloc's network in
    # interactive mode).
    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
         --gpus-per-node=1 --cpus-per-task=32 --mpi=cray_shasta \
         --export=ALL,OMP_NUM_THREADS=$T \
         ./build/rl/train_stratego "$VARIANT" "$SETUP" "$EPISODES" "$BATCH" 0
    SRUN_RC=$?

    AFTER_MS=$(date +%s%3N)

    if [ $SRUN_RC -ne 0 ]; then
        echo "srun failed (rc=$SRUN_RC) for OMP=$T; aborting sweep" | tee -a "$MANIFEST"
        exit $SRUN_RC
    fi

    AFTER_LOGS=$(ls -1 logs/bench_${VARIANT}_${SETUP}_n1_*.jsonl 2>/dev/null | sort)
    NEW_LOG=$(comm -13 <(echo "$BEFORE_LOGS") <(echo "$AFTER_LOGS") | tail -n 1)

    WALL_MS=$((AFTER_MS - BEFORE_MS))

    echo "OMP=$T wall=${WALL_MS}ms batch=${BATCH} episodes=${EPISODES} num_batches=${NUM_BATCHES} log=$NEW_LOG" \
        | tee -a "$MANIFEST"
done

echo
echo "Done. To plot:"
echo "  python plot/plot_omp_sweep.py $MANIFEST"
