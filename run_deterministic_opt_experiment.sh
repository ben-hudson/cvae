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

DATAROOT=$SCRATCH/synthetic_lp_dataset
DATASET=2var8cons_InvPLP0.tar.gz

EXTRA_ARGS=${@}

cp $DATAROOT/$DATASET $SLURM_TMPDIR

python scripts/deterministic_opt_test.py $SLURM_TMPDIR/$DATASET --wandb_project deterministic_opt_test $EXTRA_ARGS
