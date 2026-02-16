#!/bin/bash -l

## Slurm job script to be adapted as required ## 

#SBATCH --job-name="bdiff_transformer%J.out"
#SBATCH --output=res_bdiff_transformer.txt%J.out
#SBATCH --error=err_bdiff_transformer.txt%J.out
#SBATCH --time=180:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=50G

## view allocated GPU cards
nvidia-smi

module purge
module load miniconda3@4.10.3/gcc-12.1.0


conda activate meso_test


python3.9 train_model_from_config.py -c config/config_transformer.json
