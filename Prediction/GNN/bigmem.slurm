#!/bin/bash -l
# Fichier submission.SBATCH

#SBATCH --job-name="BIGMEM"
#SBATCH --output=%x.%J.out   ## %x=job name, %J=job id
#SBATCH --error=%x.%J.out
 # walltime (hh:mm::ss) max is 8 days
#SBATCH -t 180:00:00
#SBATCH --partition=bigmem
#SBATCH -c 8  ## request 16 cores (MAX is 32)
#SBATCH --mem=100G ## (MAX is 250G)

module purge
module load miniconda3@4.10.3/gcc-12.1.0

conda activate meso_test

#!/bin/bash

if [ "$1" == "bdiff_all_BA_loss" ]; then
    echo "Running all_BA_loss experiments"
    python3.9 train_model_from_config.py -c config/config_all_BA_loss.json
fi

if [ "$1" == "bdiff_all_loss" ]; then
    echo "Running all_loss experiments"
    python3.9 train_model_from_config.py -c config/config_all_loss.json
fi

if [ "$1" == "bdiff_esn" ]; then
    echo "Running bdiff_esn experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_esn.json
fi


if [ "$1" == "bdiff_pdegpd2" ]; then
    echo "Running bdiff_degpd experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_pdegpd2.json
fi

if [ "$1" == "bdiff_pdegpd_cluster" ]; then
    echo "Running bdiff_degpd experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_pdegpd2cluster.json
fi

if [ "$1" == "bdiff_gwdl" ]; then
    echo "Running bdiff_gwdl experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_2_gwdl.json
fi

if [ "$1" == "bdiff_test_FXA" ]; then
    echo "Running bdiff_test_FXA experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config_fxa/config_25.json
fi

if [ "$1" == "bdiff_catboost" ]; then
    echo "Running bdiff_catboost experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_catboost.json
fi

if [ "$1" == "bdiff_test_2" ]; then
    echo "Running bdiff_test_2 experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_2.json
fi

if [ "$1" == "bdiff_test_C" ]; then
    echo "Running bdiff_test_C experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_C_explore_05.json
fi

if [ "$1" == "bdiff_test_C2" ]; then
    echo "Running bdiff_test_C experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_C_explore_02.json
fi

if [ "$1" == "bdiff_test_C7" ]; then
    echo "Running bdiff_test_C experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_C_explore_07.json
fi

if [ "$1" == "bdiff_test_C9" ]; then
    echo "Running bdiff_test_C experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_C_explore_09.json
fi

if [ "$1" == "bdiff_test_dice" ]; then
    echo "Running bdiff_test_dice experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_2_ordinal_dice.json
fi

if [ "$1" == "bdiff_base" ]; then
    echo "Running bdiff_base experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_base.json
fi

if [ "$1" == "bdiff_tree_BA" ]; then
    echo "Running bdiff_BA experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_BA_tree.json
fi

if [ "$1" == "bdiff_BA" ]; then
    echo "Running bdiff_BA experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_BA.json
fi

if [ "$1" == "bdiff_BA_root" ]; then
    echo "Running bdiff_BA experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_BA_root.json
fi

if [ "$1" == "bdiff_binary" ]; then
    echo "Running bdiff_binary experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_binary.json
fi

if [ "$1" == "bdiff_fire_indice" ]; then
    echo "Running bdiff_fire_indice experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_fire_indice.json
fi

if [ "$1" == "bdiff_graphcast" ]; then
    echo "Running bdiff_graphcast experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_graphcast.json
fi

if [ "$1" == "bdiff_graphcast_test" ]; then
    echo "Running bdiff_graphcast experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_graphcast_test.json
fi

if [ "$1" == "bdiff_graphcast_4" ]; then
    echo "Running bdiff_graphcast_4 experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_graphcast_4.json
fi

if [ "$1" == "bdiff_graphcast_BA" ]; then
    echo "Running bdiff_graphcast_BA experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_graphcast_BA.json
fi

if [ "$1" == "bdiff_transformer" ]; then
    echo "Running bdiff_transformer experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_transformer.json
fi

if [ "$1" == "bdiff_cluster" ]; then
    echo "Running bdiff_cluster experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_cluster.json
fi

if [ "$1" == "bdiff_distillation" ]; then
    echo "Running bdiff_distillation experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_distillation.json
fi

if [ "$1" == "bdiff_test" ]; then
    echo "Running bdiff_test experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_2.json
fi

if [ "$1" == "bdiff_tail_cluster" ]; then
    echo "Running bdiff_test experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_bdiff_cluster_tail.json
fi

if [ "$1" == "bdiff_bulktail_cluster" ]; then
    echo "Running bdiff_test experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_bdiff_cluster_bulk_tail.json
fi

if [ "$1" == "bdiff_egpd" ]; then
    echo "Running bdiff_egpd experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_egpd.json
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

if [ "$1" == "bdiff_distrib" ]; then
    echo "Running bdiff_distrib experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_distrib.json
fi

if [ "$1" == "bdiff_egpd_ordinal" ]; then
    echo "Running bdiff_egpd_ordinal experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_egpd_ordinal.json
fi

if [ "$1" == "bdiff_bceloss" ]; then
    echo "Running bdiff_bceloss experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_egpd_ordinal_bceloss.json
fi

if [ "$1" == "bdiff_bulktail" ]; then
    echo "Running bdiff_gamma_log experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_bulktail.json
fi

if [ "$1" == "bdiff_fdice" ]; then
    echo "Running bdiff_fdice experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_fdice.json
fi

if [ "$1" == "bdiff_wkloss" ]; then
    echo "Running bdiff_wkloss experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_wk.json
fi

if [ "$1" == "bdiff_gru_distillilation" ]; then
    echo "Running bdiff_gru_distillilation experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_voting_distillation_GRU.json
fi

if [ "$1" == "bdiff_gru_relation_distillilation_MLP" ]; then
    echo "Running bdiff_gru_relation_distillilation_MLP experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_voting_relationMLP_distillation_GRU.json
fi

if [ "$1" == "bdiff_gru_relation_distillilation_MATTKD" ]; then
    echo "Running bdiff_gru_relation_distillilation_MATTKD experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_voting_MATTKD_distillation_GRU.json
fi

if [ "$1" == "bdiff_gru_relation_distillilation_ATT" ]; then
    echo "Running bdiff_gru_relation_distillilation_ATT experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_voting_relationATT_distillation_GRU.json
fi

if [ "$1" == "bdiff_student" ]; then
    echo "Running bdiff_student experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_student.json
fi


if [ "$1" == "bdiff_gru_confidence_distillilation" ]; then
    echo "Running bdiff_gru_confidence_distillilation experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_voting_confidence_distillation_GRU.json
fi

if [ "$1" == "bdiff_gru_normal_distillilation" ]; then
    echo "Running bdiff_gru_normal_distillilation experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_voting_normal_distillation_GRU.json
fi

if [ "$1" == "bdiff_gru_adaptativeMLP_distillilation" ]; then
    echo "Running bdiff_gru_adaptativeMLP_distillilation experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_voting_AdaptativeMLP_distillation_GRU.json
fi

if [ "$1" == "bdiff_cdw" ]; then
    echo "Running bdiff_cdw experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_cdw.json
fi

if [ "$1" == "bdiff_cewk" ]; then
    echo "Running bdiff_wkloss experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_cewk.json
fi

if [ "$1" == "bdiff_bceloss" ]; then
    echo "Running bdiff_wkloss experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_bce.json
fi


if [ "$1" == "bdiff_test_area" ]; then
    echo "Running bdiff_test experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_area_loss.json
fi

if [ "$1" == "bdiff_2024_test" ]; then
    echo "Running bdiff_test experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_2024.json
fi

if [ "$1" == "bdiff_voting_catboost" ]; then
    echo "Running bdiff_voting_catboost experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 -u train_model_from_config.py -c config/config_voting_catboost.json
fi

if [ "$1" == "bdiff_voting_gru" ]; then
    echo "Running bdiff_voting_gru experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 -u train_model_from_config.py -c config/config_voting_GRU.json
fi

if [ "$1" == "bdiff_area_learnable" ]; then
    echo "Running bdiff_test experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_area_loss_learnable.json
fi

if [ "$1" == "bdiff_voting_lstm" ]; then
    echo "Running bdiff_voting_lstm experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 -u train_model_from_config.py -c config/config_voting_LSTM.json
fi

if [ "$1" == "bdiff_voting_mlp" ]; then
    echo "Running bdiff_voting_lstm experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 -u train_model_from_config.py -c config/config_voting_MLP.json
fi

if [ "$1" == "bdiff_cnn" ]; then
    echo "Running bdiff_cnn experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_CNN.json
fi

if [ "$1" == "bdiff_06" ]; then
    echo "Running bdiff_06 experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_06.json
fi

if [ "$1" == "bdiff_corine_route" ]; then
    echo "Running bdiff_corine_route experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_features_corine_route.json
fi

if [ "$1" == "bdiff_route" ]; then
    echo "Running bdiff_corine_route experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_features_route.json
fi

if [ "$1" == "bdiff_new_calendar" ]; then
    echo "Running bdiff_new_calendar experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_features_new_calendar.json
fi


if [ "$1" == "bdiff_bayesian" ]; then
    echo "Running bdiff_test experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_bayesian.json
fi

if [ "$1" == "bdiff_feature" ]; then
    echo "Running bdiff_feature experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    #python3.9 train_model_from_config.py -c config/config_features.json
    #python3.9 train_model_from_config.py -c config/config_nometeo.json
    #python3.9 train_model_from_config.py -c config/config_nohisto.json
    #python3.9 train_model_from_config.py -c config/config_nosocio.json
    python3.9 train_model_from_config.py -c config/config_notopo.json
    #python3.9 train_model_from_config.py -c config/config_nopop.json
    #python3.9 train_model_from_config.py -c config/config_nocalendar.json
    #python3.9 train_model_from_config.py -c config/config_nohighway.json
fi

if [ "$1" == "bdiff_tree" ]; then
    echo "Running bdiff_tree experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_tree.json
fi

if [ "$1" == "bdiff_TFN" ]; then
    echo "Running bdiff_TFN experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_TFN.json
fi


if [ "$1" == "bdiff_federated_transformer" ]; then
    echo "Running bdiff_federated_transformer experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_federated_transformer.json
fi

if [ "$1" == "bdiff_splittraining_transformer" ]; then
    echo "Running bdiff_splittraining_transformer experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_splittraining_transformer.json
fi

if [ "$1" == "bdiff_moonfederated_transformer" ]; then
    echo "Running bdiff_moonfederated_transformer experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_moonfederated_transformer.json
fi

if [ "$1" == "bdiff_FLTGfederated_transformer" ]; then
    echo "Running bdiff_FLTGfederated_transformer experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_FLTGfederated_transformer.json
fi

if [ "$1" == "bdiff_alafederated_transformer" ]; then
    echo "Running bdiff_alafederated_transformer experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_alafederated_transformer.json
fi

if [ "$1" == "bdiff_proxfederated_transformer" ]; then
    echo "Running bdiff_proxfederated_transformer experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_proxfederated_transformer.json
fi

if [ "$1" == "bdiff_protofederated_transformer" ]; then
    echo "Running bdiff_protofederated_transformer experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_protofederated_transformer.json
fi

if [ "$1" == "bdiff_ordinality" ]; then
    echo "Running bdiff_ordinality experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)
    
    python3.9 train_model_from_config.py -c config/config_ordinality.json
fi

if [ "$1" == "bdiff_trainonhighdepartment" ]; then
    echo "Running bdiff_trainonhighdepartment experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_trainonhighdepartment.json
fi

if [ "$1" == "bdiff_loss" ]; then
    echo "Running bdiff_loss experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)
    
    python3.9 train_model_from_config.py -c config/config_loss.json
fi

if [ "$1" == "bdiff_dualtraining" ]; then
    echo "Running bdiff_dualtraining experiments..."
    #SBATCH --job-name="LSTM" ## <---(modify me)

    python3.9 train_model_from_config.py -c config/config_test_dualtraining.json
fi