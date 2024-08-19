#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output /network/scratch/b/ben.hudson/slurm/%j.out
#SBATCH --error /network/scratch/b/ben.hudson/slurm/%j.err

set -e  # exit on error
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

module --quiet purge
module --quiet load python/3.10
module --quiet load cuda/11.8

source .venv/bin/activate

if [ $# -lt 1 ]; then
    echo "Pass an dataset"
    exit 1
fi
DATASET=${1}
EXTRA_ARGS=${@:2}

python scripts/pyepo_experiment_2.py $DATASET --wandb_project pyepo --workers 4 $EXTRA_ARGS
