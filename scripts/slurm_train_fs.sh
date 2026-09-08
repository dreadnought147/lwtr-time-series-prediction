#!/bin/bash
# SLURM job — PPO + Frame Stacking sweep (RQ2 baseline arm)
# Submit from repo root:  sbatch scripts/slurm_train_fs.sh
#
# Adjust partition/time/cpus to Wits mscluster's actual CPU partitions.
# stampede / bigbatch / biggpu are GPU-heavy; for MLP-PPO on CPU pick
# whichever CPU-friendly partition your admin recommends.

#SBATCH --job-name=rq_rl_fs
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err

set -euo pipefail

# --- environment setup ---
mkdir -p slurm-logs
module load python/3.10 || true       # ignore if module system is different
source .venv/bin/activate

# --- pin thread counts so multi-CPU workers don't fight ---
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Running on $(hostname), partition=$SLURM_JOB_PARTITION, cpus=$SLURM_CPUS_PER_TASK"

# --- the actual training call — resume-safe ---
python scripts/train.py --config configs/rq_rl_fs_sweep.yaml

echo "Job $SLURM_JOB_ID finished at $(date)"
