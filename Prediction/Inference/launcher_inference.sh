#!/bin/bash

python sinister_prediction.py -d 0 -l softmax -t classification -ta nbsinister-kmeans-5-Class-Dept -dept departement-25-doubs -m xgboost -np full -nf all -dh False -r 2x2 -dataset firemen -doKMEANS False -graphConstruct risk-size-watershed -graph_method node -thresh_kmeans 0.15 -shift_kmeans 5 -sinis firepoint