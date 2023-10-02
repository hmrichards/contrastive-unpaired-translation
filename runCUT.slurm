#!/bin/bash
#SBATCH -c 8
#SBATCH -e runCUT.err
#SBATCH -o runCUT.out
#SBATCH -t 10-10:00
#SBATCH --gpus=1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate CUT

python train.py --dataroot '/fh/scratch/delete90/haffner_m/user/hrichards/CUT_data/CUT_IHC_norm' --preprocess 'none' --name 'CUT_IHC_norm' --display_id -1 --n_epochs 50 --n_epochs_decay 50 