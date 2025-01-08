#!/bin/bash

########################### default ###############################
if [ "$1" == "default" ]; then
    echo "Running "$1" experiments..."

    python train_1D_temporal.py -n "$1" -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc 4 -nf all -of False -test True -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one -graph_method 'node'    
    python train_1D_temporal.py -n "$1" -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc 5 -nf all -of False -test True -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one -graph_method 'node'
    python train_1D_temporal.py -n "$1" -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc 6 -nf all -of False -test True -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one -graph_method 'node'
    python train_1D_temporal.py -n "$1" -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc 7 -nf all -of False -test True -train True -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one -graph_method 'node'
    python train_1D_temporal.py -n "$1" -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd False -sc departement -nf all -of False -test True -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 7 -days_in_futur 0 -scaling z-score -graphConstruct None -sinisterEncoding occurence -weights weight_one -graph_method 'node'

fi