#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=03:15:00
#SBATCH --output /network/scratch/b/ben.hudson/slurm-%j.out
#SBATCH --error /network/scratch/b/ben.hudson/slurm-%j.err

set -e  # exit on error
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

module --quiet purge
module --quiet load miniconda/3
module --quiet load cuda/11.8

# Load environment variables
source .env

# Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
unset CUDA_VISIBLE_DEVICES

conda activate cvae

if [ $# -lt 1 ]; then
    echo "Pass an experiment name"
    exit 1
fi
EXP_NAME=${1}
DATASET=taxi_4_1000000_3dimsintervention.nc
EXTRA_ARGS=${@:3}

RESULTS_DIR=$HOME/cvae/results/$DATASET/$EXP_NAME
DATAROOT=$SCRATCH/taxi_dataset

cp $DATAROOT/$DATASET $SLURM_TMPDIR
mkdir -p $SLURM_TMPDIR/results

python scripts/train.py $SLURM_TMPDIR/$DATASET --latent_dim_to_gt --kld_weight 0.001  --wandb_project learn_2ssps --wandb_exp $EXP_NAME $EXTRA_ARGS

mkdir -p $RESULTS_DIR
mv $SLURM_TMPDIR/results/* $RESULTS_DIR
echo "$EXTRA_ARGS" > $RESULTS_DIR/extra_args.txt
