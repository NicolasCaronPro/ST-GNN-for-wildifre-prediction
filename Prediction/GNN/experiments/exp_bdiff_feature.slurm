#!/bin/bash -l
# Fichier submission.SBATCH

#SBATCH --job-name="bdiff_feature%J.out"
#SBATCH --output=res_bdiff_feature.txt%J.out
#SBATCH --error=err_bdiff_feature.txt%J.out
 # walltime (hh:mm::ss) max is 8 days
#SBATCH -t 180:00:00
#SBATCH --partition=bigmem
#SBATCH -c 8  ## request 16 cores (MAX is 32)
#SBATCH --mem=75G ## (MAX is 250G)

module purge
module load miniconda3@4.10.3/gcc-12.1.0

conda activate meso_test

#python3.9 train_model_from_config.py -c config/config_features.json
#python3.9 train_model_from_config.py -c config/config_nometeo.json
#python3.9 train_model_from_config.py -c config/config_nohisto.json
#python3.9 train_model_from_config.py -c config/config_nosocio.json
python3.9 train_model_from_config.py -c config/config_notopo.json
#python3.9 train_model_from_config.py -c config/config_nopop.json
#python3.9 train_model_from_config.py -c config/config_nocalendar.json
#python3.9 train_model_from_config.py -c config/config_nohighway.json
