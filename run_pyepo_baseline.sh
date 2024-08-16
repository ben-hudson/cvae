#!/bin/bash
#SBATCH --cpus-per-task=2
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

if [ $# -lt 2 ]; then
    echo "Pass a dataset and baseline"
    exit 1
fi
DATASET=${1}
BASELINE=${2}
EXTRA_ARGS=${@:3}

python scripts/pyepo_baseline.py $DATASET $BASELINE --wandb_project pyepo $EXTRA_ARGS
