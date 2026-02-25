#!/bin/bash -l
# Fichier submission.SBATCH

#SBATCH --job-name="06_ressource%J.out"
#SBATCH --output=res_06_ressource.txt%J.out
#SBATCH --error=err_06_ressource.txt%J.out
 # walltime (hh:mm::ss) max is 8 days
#SBATCH -t 180:00:00
#SBATCH --partition=bigmem
#SBATCH -c 8  ## request 16 cores (MAX is 32)
#SBATCH --mem=100G ## (MAX is 250G)

module purge
module load miniconda3@4.10.3/gcc-12.1.0

conda activate meso_test


#python3.9 train_model_from_config.py -c config_firemen/config_06_ressource_bceloss.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_ressource_cornfl.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_ressource_cornloss.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_ressource_fl.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_ressource_flwk.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_ressource_flwki-id_departement.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_ressource_pdegpd.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_ressource_weightedcrossentropy.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_ressource_wkloss.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_ressource_cll.json
python3.9 train_model_from_config.py -c config_firemen/config_06_ressource_cllt.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_ressource_lmol.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_ressource_prls.json
