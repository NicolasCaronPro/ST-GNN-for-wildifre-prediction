#!/bin/bash

python train_susceptibility_map.py -n susecptibility -dataset bdiff -e False -s firepoint -p False -np full -d False -g False -mxd 2023-01-01 -mxdv 2022-01-01 -f False -test True -train True -r 2x2 -kmeans False -ncluster 5 -shift 5 -thresh_kmeans 0.5 -k_days 0 -days_in_futur 0 -scaling z-score -sinisterEncoding occurence    
