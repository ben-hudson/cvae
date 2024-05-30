#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=06:15:00
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
    echo "Pass an experiment name"
    exit 1
fi
EXP_NAME=${1}
DATASET=taxi_4_1000000_3dimsintervention_2.nc
EXTRA_ARGS=${@:2}

RESULTS_DIR=$HOME/cvae/results/$DATASET/$EXP_NAME
# DATAROOT=$SCRATCH/taxi_dataset

# cp $DATAROOT/$DATASET $SLURM_TMPDIR
mkdir -p $SLURM_TMPDIR/results

python scripts/train.py --model solver_vae --latent_dim_to_gt --workers 4 --max_hours 6 --wandb_project learn_2ssps --wandb_exp $EXP_NAME $EXTRA_ARGS

mkdir -p $RESULTS_DIR
mv $SLURM_TMPDIR/results/* $RESULTS_DIR
echo "$EXTRA_ARGS" > $RESULTS_DIR/extra_args.txt
