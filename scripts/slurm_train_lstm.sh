#!/bin/bash
# SLURM job — LSTM SL baseline sweep (RQ1)
# Submit from repo root:  sbatch scripts/slurm_train_lstm.sh

#SBATCH --job-name=rq1_lstm
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
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

python scripts/train_lstm.py --config configs/rq1_lstm_sweep.yaml

echo "Job $SLURM_JOB_ID finished at $(date)"
