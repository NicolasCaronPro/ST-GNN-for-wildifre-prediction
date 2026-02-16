#!/bin/bash -l

## Slurm job script to be adapted as required ## 

#SBATCH --job-name="bdiff_feature%J.out"
#SBATCH --output=res_bdiff_feature.txt%J.out
#SBATCH --error=err_bdiff_feature.txt%J.out
#SBATCH --time=180:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=50G

## view allocated GPU cards
nvidia-smi

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
