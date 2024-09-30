#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output /network/scratch/b/ben.hudson/slurm/%j.out
#SBATCH --error /network/scratch/b/ben.hudson/slurm/%j.err

set -e  # exit on error
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

module --quiet purge
module --quiet load python/3.10
module --quiet load cuda/11.8

source .venv/bin/activate

EXTRA_ARGS=${@}

python scripts/train.py toy --wandb_project toy_example --workers 4 $EXTRA_ARGS
