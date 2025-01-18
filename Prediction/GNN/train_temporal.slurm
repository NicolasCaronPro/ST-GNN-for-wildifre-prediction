#!/bin/bash -l

## Slurm job script to be adapted as required ## 

#SBATCH --job-name="train_temporal" ## <---(modify me) 
#SBATCH --output=%x.%J.out       ## %x=job name, %J=job id
#SBATCH --error=%x.%J.out
#SBATCH --time=24:00:00 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=128G

module purge
module load miniconda3@4.10.3/gcc-12.1.0

conda activate meso_test

#!/bin/bash

########################### DATABASE ###############################
if [ "$1" == "database" ]; then
    echo "Running DEFAULT experiments..."

    #python3.9 train_trees.py -n default -dataset firemen -e False -s firepoint -p False -np binary -d True -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc 4 -nf all -of False -test True -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_outlier_2
    #python3.9 train_trees.py -n default -dataset firemen -e False -s firepoint -p False -np binary -d True -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc 5 -nf all -of False -test True -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_outlier_2
    python3.9 train_trees.py -n default -dataset firemen -e False -s firepoint -p False -np binary -d True -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc 6 -nf all -of False -test False -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one
    #python3.9 train_trees.py -n default -dataset firemen -e False -s firepoint -p False -np binary -d True -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc 7 -nf all -of False -test True -train True -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one
fi

########################### DEFAULT ###############################
if [ "$1" == "default" ]; then
    echo "Running DEFAULT experiments..."

    #python3.9 train_1D_temporal.py -n "$1" -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc 4 -nf all -of False -test True -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one    
    #python3.9 train_1D_temporal.py -n "$1" -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc 5 -nf all -of False -test True -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one
    #python3.9 train_1D_temporal.py -n "$1" -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc 6 -nf all -of False -test True -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one
    #python3.9 train_1D_temporal.py -n "$1" -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc 7 -nf all -of False -test True -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one
    python3.9 train_1D_temporal.py -n "$1" -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc departement -nf all -of False -test True -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct None -sinisterEncoding occurence -weights weight_one

fi

if [ "$1" == "default_graph_temp" ]; then
    echo "Running DEFAULT experiments..."

    #python3.9 train_1D_temporal.py -n "$1" -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc 4 -nf all -of False -test True -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one -graph_method 'graph'    
    python3.9 train_1D_temporal.py -n "$1" -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc 5 -nf all -of False -test True -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one -graph_method 'graph'
    python3.9 train_1D_temporal.py -n "$1" -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc 6 -nf all -of False -test True -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one -graph_method 'graph'
    python3.9 train_1D_temporal.py -n "$1" -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc 7 -nf all -of False -test True -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one -graph_method 'graph'
    python3.9 train_1D_temporal.py -n "$1" -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc departement -nf all -of False -test True -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct None -sinisterEncoding occurence -weights weight_one -graph_method 'graph'

fi

########################### DEFAULT ###############################
if [ "$1" == "default_graph" ]; then
    echo "Running default_graph experiments..."

    python3.9 train_1D_daily.py -n default -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f True -dd False -sc 4 -nf all -of False -test True -train True -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 0 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one -graph_method 'graph'    
    python3.9 train_1D_daily.py -n default -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc 5 -nf all -of False -test True -train True -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 0 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one -graph_method 'graph'
    python3.9 train_1D_daily.py -n default -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc 6 -nf all -of False -test True -train True -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 0 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one -graph_method 'graph'
    python3.9 train_1D_daily.py -n default -dataset firemen -e True -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc 7 -nf all -of False -test True -train True -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 0 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one -graph_method 'graph'
    python3.9 train_1D_daily.py -n default -dataset firemen -e True -s firepoint -p False -np full -d False -g True -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc departement -nf all -of False -test True -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 0 -days_in_futur 0 -scaling z-score -graphConstruct None -sinisterEncoding occurence -weights weight_one -graph_method 'graph'

fi

########################### DEFAULT ###############################
if [ "$1" == "default_cnn" ]; then
    echo "Running default_graph experiments..."

    python3.9 train_cnn.py -n default -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f True -dd False -sc 4 -nf all -of False -test True -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one -graph_method 'node'    
    python3.9 train_cnn.py -n default -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc 5 -nf all -of False -test True -train True -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one -graph_method 'node'
    python3.9 train_cnn.py -n default -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc 6 -nf all -of False -test True -train True -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one -graph_method 'node'
    python3.9 train_cnn.py -n default -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc 7 -nf all -of False -test True -train True -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one -graph_method 'node'
    python3.9 train_cnn.py -n default -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc departement -nf all -of False -test True -train True -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct None -sinisterEncoding occurence -weights weight_one -graph_method 'node'

fi