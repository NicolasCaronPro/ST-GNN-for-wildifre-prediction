#!/bin/bash -l
# Fichier submission.SBATCH

#SBATCH --job-name="06_nbsinister%J.out"
#SBATCH --output=res_06_nbsinister.txt%J.out
#SBATCH --error=err_06_nbsinister.txt%J.out
 # walltime (hh:mm::ss) max is 8 days
#SBATCH -t 180:00:00
#SBATCH --partition=bigmem
#SBATCH -c 8  ## request 16 cores (MAX is 32)
#SBATCH --mem=75G ## (MAX is 250G)

module purge
module load miniconda3@4.10.3/gcc-12.1.0

conda activate meso_test


#python3.9 train_model_from_config.py -c config/config_06_burnedareaRoot_bceloss.json
#python3.9 train_model_from_config.py -c config/config_06_burnedareaRoot_cornfl.json
#python3.9 train_model_from_config.py -c config/config_06_burnedareaRoot_cornloss.json
#python3.9 train_model_from_config.py -c config/config_06_burnedareaRoot_fl.json
#python3.9 train_model_from_config.py -c config/config_06_burnedareaRoot_flwk.json
#python3.9 train_model_from_config.py -c config/config_06_burnedareaRoot_flwki-id_departement.json
#python3.9 train_model_from_config.py -c config/config_06_burnedareaRoot_pdegpd.json
#python3.9 train_model_from_config.py -c config/config_06_burnedareaRoot_weightedcrossentropy.json
#python3.9 train_model_from_config.py -c config/config_06_burnedareaRoot_wkloss.json
#python3.9 train_model_from_config.py -c config/config_06_burnedareaRoot_cll.json
#python3.9 train_model_from_config.py -c config/config_06_burnedareaRoot_cllt.json
python3.9 train_model_from_config.py -c config/config_06_burnedareaRoot_ranknet.json

#python3.9 train_model_from_config.py -c config/config_06_nbsinister_lmol.json
#python3.9 train_model_from_config.py -c config/config_06_nbsinister_prls.json
