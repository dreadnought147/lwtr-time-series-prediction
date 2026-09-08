#!/bin/bash
# SLURM job — PPO + Adaptive Stacking sweep (RQ2 primary contribution arm)
# Submit from repo root:  sbatch scripts/slurm_train_as.sh
#
# Twin of slurm_train_fs.sh — identical resources, only the config differs.

#SBATCH --job-name=rq_rl_as
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err

set -euo pipefail

mkdir -p slurm-logs
module load python/3.10 || true
source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Running on $(hostname), partition=$SLURM_JOB_PARTITION, cpus=$SLURM_CPUS_PER_TASK"

python scripts/train.py --config configs/rq_rl_as_sweep.yaml

echo "Job $SLURM_JOB_ID finished at $(date)"
