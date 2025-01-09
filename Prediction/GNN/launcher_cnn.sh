#!/bin/bash

########################### default ###############################
if [ "$1" == "default" ]; then
    echo "Running "$1" experiments..."

    #python train_cnn.py -n "$1" -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd True -sc 4 -nf all -of False -test False -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 0 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one -graph_method 'node'    
    #python train_cnn.py -n "$1" -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd True -sc 5 -nf all -of False -test False -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 0 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one -graph_method 'node'
    #python train_cnn.py -n "$1" -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd True -sc 6 -nf all -of False -test False -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 0 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one -graph_method 'node'
    #python train_cnn.py -n "$1" -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd True -sc 7 -nf all -of False -test False -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 0 -days_in_futur 0 -scaling z-score -graphConstruct risk-size-watershed -sinisterEncoding occurence -weights weight_one -graph_method 'node'
    python train_cnn.py -n "$1" -dataset firemen -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -dd True -sc departement -nf all -of False -test False -train False -r 2x2 -pca False -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 0 -days_in_futur 0 -scaling z-score -graphConstruct None -sinisterEncoding occurence -weights weight_one -graph_method 'node'

fi