#!/bin/bash -l
# Fichier submission.SBATCH

#SBATCH --job-name="06_nbsinister%J.out"
#SBATCH --output=res_06_nbsinister.txt%J.out
#SBATCH --error=err_06_nbsinister.txt%J.out
#SBATCH --time=180:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=50G

module purge
module load miniconda3@4.10.3/gcc-12.1.0

conda activate meso_test


#python3.9 train_model_from_config.py -c config_firemen/config_06_nbsinister_bceloss.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_nbsinister_cornfl.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_nbsinister_cornloss.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_nbsinister_fl.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_nbsinister_flwk.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_nbsinister_flwki-id_departement.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_nbsinister_pdegpd.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_nbsinister_weightedcrossentropy.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_nbsinister_wkloss.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_nbsinister_cll.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_nbsinister_cllt.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_nbsinister_ranknet.json
python3.9 train_model_from_config.py -c config_firemen/config_06_nbsinister_msetheta.json

#python3.9 train_model_from_config.py -c config_firemen/config_06_nbsinister_lmol.json
#python3.9 train_model_from_config.py -c config_firemen/config_06_nbsinister_prls.json
