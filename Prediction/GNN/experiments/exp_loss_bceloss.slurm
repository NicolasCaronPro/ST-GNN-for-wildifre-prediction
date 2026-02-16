#!/bin/bash -l
# Fichier submission.SBATCH

#SBATCH --job-name="bceloss"
#SBATCH --output=res_bceloss.txt%J.out
#SBATCH --error=err_bceloss.txt%J.out
 # walltime (hh:mm::ss) max is 8 days
#SBATCH -t 180:00:00
#SBATCH --partition=bigmem
#SBATCH -c 8  ## request 16 cores (MAX is 32)
#SBATCH --mem=100G ## (MAX is 250G)

module purge
module load miniconda3@4.10.3/gcc-12.1.0

conda activate meso_test

python3.9 train_model_from_config.py -c config/config_loss_bceloss.json
