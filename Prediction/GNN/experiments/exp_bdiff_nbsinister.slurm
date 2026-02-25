#!/bin/bash -l
# Fichier submission.SBATCH

#SBATCH --job-name="bdiff_nbsinister%J.out"
#SBATCH --output=res_bdiff_nbsinister.txt%J.out
#SBATCH --error=err_bdiff_nbsinister.txt%J.out
 # walltime (hh:mm::ss) max is 8 days
#SBATCH -t 180:00:00
#SBATCH --partition=bigmem
#SBATCH -c 8  ## request 16 cores (MAX is 32)
#SBATCH --mem=200G ## (MAX is 250G)

module purge
module load miniconda3@4.10.3/gcc-12.1.0

conda activate meso_test


python3.9 train_model_from_config.py -c config/config_06_nbsinister_bceloss.json
python3.9 train_model_from_config.py -c config/config_06_nbsinister_cornfl.json
python3.9 train_model_from_config.py -c config/config_06_nbsinister_cornloss.json
python3.9 train_model_from_config.py -c config/config_06_nbsinister_fl.json
python3.9 train_model_from_config.py -c config/config_06_nbsinister_flwk.json
python3.9 train_model_from_config.py -c config/config_06_nbsinister_flwki-id_departement.json
python3.9 train_model_from_config.py -c config/config_06_nbsinister_pdegpd.json
python3.9 train_model_from_config.py -c config/config_06_nbsinister_weightedcrossentropy.json
python3.9 train_model_from_config.py -c config/config_06_nbsinister_wkloss.json
python3.9 train_model_from_config.py -c config/config_06_nbsinister_cll.json
#python3.9 train_model_from_config.py -c config/config_06_nbsinister_lmol.json
#python3.9 train_model_from_config.py -c config/config_06_nbsinister_prls.json
python3.9 train_model_from_config.py -c config/config_06_nbsinister_cllt.json