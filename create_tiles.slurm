#!/bin/bash
#SBATCH -c 8
#SBATCH -e create_tiles.err
#SBATCH -o create_tiles.out
#SBATCH -t 0-10:00
#SBATCH --gpus=1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate training

python create_tiles_CUT.py --image_location '/fh/scratch/delete90/haffner_m/user/hrichards/CUT_data/IHC_cores_to_tile' --save_location '/fh/scratch/delete90/haffner_m/user/hrichards/CUT_data/CUT_IHC_norm/trainA' --save_image_size 256 --resamp 2 --stain_norm False 