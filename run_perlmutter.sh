#!/bin/bash
#SBATCH --job-name=stratego_train
#SBATCH --account=m4341_g
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module swap PrgEnv-cray PrgEnv-gnu
module load gcc/12.2.0
module load cray-mpich
module load cudatoolkit

# Make the NCCL bundled with PyTorch available at runtime so libtorch_cuda.so can
# resolve ncclCommWindowRegister and other symbols not in the system NCCL.
PYTORCH_SITE=/global/common/software/nersc9/pytorch/2.8.0/lib/python3.12/site-packages
export LD_LIBRARY_PATH=${PYTORCH_SITE}/nvidia/nccl/lib:${PYTORCH_SITE}/torch/lib:${LD_LIBRARY_PATH}

# NCCL handles GPU transfers; MPI only does control-plane work here
export MPICH_GPU_SUPPORT_ENABLED=0

# Tune NCCL for Perlmutter's NVLink/Infiniband topology
export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=PHB

VARIANT=${1:-tiny}
SETUP=${2:-random}
EPISODES=${3:-20000}
BATCH=${4:-256}

mkdir -p logs models

srun --mpi=pmi2 ./build/rl/train_stratego "$VARIANT" "$SETUP" "$EPISODES" "$BATCH"
