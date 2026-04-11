#!/bin/bash -l
# Fichier submission.SBATCH

#SBATCH --job-name="06_timeintervention%J.out"
#SBATCH --output=res_06_timeintervention.txt%J.out
#SBATCH --error=err_06_timeintervention.txt%J.out
 # walltime (hh:mm::ss) max is 8 days
#SBATCH -t 180:00:00
#SBATCH --partition=bigmem
#SBATCH -c 8  ## request 16 cores (MAX is 32)
#SBATCH --mem=50G ## (MAX is 250G)

module purge
module load miniconda3@4.10.3/gcc-12.1.0

conda activate meso_test


#python3.9 train_model_from_config.py -c config_firemen/config_06_timeintervention_bceloss.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_timeintervention_cornfl.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_timeintervention_cornloss.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_timeintervention_fl.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_timeintervention_flwk.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_timeintervention_flwki-id_departement.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_timeintervention_pdegpd.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_timeintervention_weightedcrossentropy.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_timeintervention_wkloss.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_timeintervention_cll.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_timeintervention_cllt.json
python3.9 train_model_from_config.py -c config_firemen/config_06_ressource_ranknet.json

#python3.9 train_model_from_config.py -c config_firemen/config_06_timeintervention_lmol.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_timeintervention_prls.json
