#!/bin/bash -l

## Slurm job script to be adapted as required ## 

#SBATCH --job-name="GPU_CNN" ## <---(modify me) 
#SBATCH --output=%x.%J.out       ## %x=job name, %J=job id
#SBATCH --error=%x.%J.out
#SBATCH --time=180:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=50G

## view allocated GPU cards
nvidia-smi

module purge
module load miniconda3@4.10.3/gcc-12.1.0

if [ "$1" == "bdiff_FLTGfederated_transformer" ]; then
    echo "Running bdiff_FLTGfederated_transformer experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_FLTGfederated_transformer.json
fi

if [ "$1" == "bdiff_student" ]; then
    echo "Running bdiff_student experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_student.json
fi


if [ "$1" == "bdiff_gru_relation_distillilation_MATTKD" ]; then
    echo "Running bdiff_gru_relation_distillilation_MATTKD experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_voting_MATTKD_distillation_GRU.json
fi


if [ "$1" == "bdiff_distillation" ]; then
    echo "Running bdiff_distillation experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_distillation.json
fi


if [ "$1" == "bdiff_degpd" ]; then
    echo "Running bdiff_degpd experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_degpd.json
fi

if [ "$1" == "bdiff_pdegpd" ]; then
    echo "Running bdiff_degpd experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_pdegpd.json
fi

if [ "$1" == "bdiff_degpd_cluster" ]; then
    echo "Running bdiff_degpd experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_degpd_cluster.json
fi

if [ "$1" == "bdiff_gamma_log" ]; then
    echo "Running bdiff_gamma_log experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_gamma_log.json
fi

if [ "$1" == "bdiff_test_FXA" ]; then
    echo "Running bdiff_test experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config_fxa/config_25.json
fi

if [ "$1" == "bdiff_cluster" ]; then
    echo "Running bdiff_cluster experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_cluster.json
fi

if [ "$1" == "bdiff_graphcast" ]; then
    echo "Running bdiff_graphcast experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_graphcast.json
fi

if [ "$1" == "bdiff_graphcast_with_attention" ]; then
    echo "Running bdiff_graphcast experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_graphcast_test.json
fi

if [ "$1" == "bdiff_graphcast_test" ]; then
    echo "Running bdiff_graphcast experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_graphcast_test.json
fi

if [ "$1" == "bdiff_voting_gru" ]; then
    echo "Running bdiff_voting_gru experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 -u train_model_from_config.py -c config/config_voting_GRU.json
fi

if [ "$1" == "bdiff_BA" ]; then
    echo "Running bdiff_BA experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_BA.json
fi

if [ "$1" == "bdiff_cnn" ]; then
    echo "Running bdiff_cnn experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_CNN.json
fi

if [ "$1" == "bdiff_2024_test" ]; then
    echo "Running bdiff_test experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_2024.json
fi

if [ "$1" == "bdiff_test" ]; then
    echo "Running bdiff_test experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_2.json
fi

if [ "$1" == "bdiff_area_learnable" ]; then
    echo "Running bdiff_test experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_area_loss_learnable.json
fi

if [ "$1" == "bdiff_area" ]; then
    echo "Running bdiff_test experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_area_loss.json
fi


if [ "$1" == "bdiff_egpd" ]; then
    echo "Running bdiff_egpd experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_egpd.json
fi

if [ "$1" == "bdiff_gwdl" ]; then
    echo "Running bdiff_gwdl experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_2_gwdl.json
fi

if [ "$1" == "bdiff_test_C" ]; then
    echo "Running bdiff_test_C experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_C_explore.json
fi

if [ "$1" == "bdiff_test_dice" ]; then
    echo "Running bdiff_test_dice experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_2_ordinal_dice.json
fi