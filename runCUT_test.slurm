#!/bin/bash
#SBATCH -c 8
#SBATCH -e runCUT_test.err
#SBATCH -o runCUT_test.out
#SBATCH -t 3-10:00
#SBATCH --gpus=1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate CUT

python test.py --dataroot '/fh/scratch/delete90/haffner_m/user/hrichards/CUT_data/CUT_IHC_tiles' --preprocess 'none' --name 'CUT_IHC' --phase test --gpu_ids 0 