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
module --quiet load anaconda/3
module --quiet load cuda/11.7

# Load environment variables
source .env

# Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
unset CUDA_VISIBLE_DEVICES

conda activate causality-course-project

if [ $# -lt 2 ]; then
    echo "Pass an experiment, dataset name"
    exit 1
fi
EXP_NAME=${1}
DATASET=${2}
EXTRA_ARGS=${@:3}

RESULTS_DIR=$HOME/causality-course-project/results/$DATASET/$EXP_NAME
DATAROOT=$SCRATCH/causality-course-project/datasets

cp $DATAROOT/$DATASET $SLURM_TMPDIR
mkdir -p $SLURM_TMPDIR/results

python disentanglement_via_mechanism_sparsity/baseline_models/cvae/train.py --dataroot $SLURM_TMPDIR --dataset $DATASET --output_dir $SLURM_TMPDIR/results \
    --z_dim 8 --hidden_dim 64 --time_limit 3 \
    --comet_key $COMET_API_KEY --comet_workspace $COMET_WORKSPACE --comet_project_name $COMET_PROJECT --comet_tag $EXP_NAME $EXTRA_ARGS

mkdir -p $RESULTS_DIR
mv $SLURM_TMPDIR/results/* $RESULTS_DIR
echo "$EXTRA_ARGS" > $RESULTS_DIR/extra_args.txt
