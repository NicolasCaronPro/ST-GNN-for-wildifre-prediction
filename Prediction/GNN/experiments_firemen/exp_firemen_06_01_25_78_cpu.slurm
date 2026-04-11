#!/bin/bash -l
# Fichier submission.SBATCH

#SBATCH --job-name="01_06_25_78%J.out"
#SBATCH --output=res_01_06_25_78t%J.out
#SBATCH --error=err_01_25_78%J.out
 # walltime (hh:mm::ss) max is 8 days
#SBATCH -t 180:00:00
#SBATCH --partition=bigmem
#SBATCH -c 8  ## request 16 cores (MAX is 32)
#SBATCH --mem=50G ## (MAX is 250G)

## view allocated GPU cards
module purge
module load miniconda3@4.10.3/gcc-12.1.0

conda activate meso_test

#python3.9 train_model_from_config.py -c config_firemen/config_06_01_25_78.json
python3.9 train_model_from_config.py -c config_firemen/config_06_01_25_78_ccllt.json
