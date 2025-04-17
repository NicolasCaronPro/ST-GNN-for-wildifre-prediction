import sys
import os

from zmq import device

import loader
from pytorch_model import get_model_from_string

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory
parent_dir = os.path.dirname(current_dir)

# Insert the parent directory into sys.path
sys.path.insert(0, parent_dir)
from loader import *
from pytorch_model import *
import geopandas as gpd
import argparse

########################### Input Arg ######################################

parser = argparse.ArgumentParser(
    prog='Train',
    description='Create graph and database according to config.py and tained model',
)
parser.add_argument('-n', '--name', type=str, help='Name of the experiment')
parser.add_argument('-s', '--sinister', type=str, help='Sinister type')
parser.add_argument('-se', '--sinisterEncoding', type=str, help='Sinister type')
parser.add_argument('-d', '--database', type=str, help='Do database')
parser.add_argument('-dataset', '--dataset', type=str, help='Dataset to use')
parser.add_argument('-test', '--doTest', type=str, help='Launch test')
parser.add_argument('-train', '--doTrain', type=str, help='Launch train')
parser.add_argument('-r', '--resolution', type=str, help='Resolution of image')
parser.add_argument('-a', '--aggregation', type=str, help='Aggregation of input fire raster')

args = parser.parse_args()

QUICK = False

# Input config
name_exp = args.name
sinister = args.sinister
sinister_encoding = args.sinisterEncoding
doDatabase = args.database == 'True'
doTest = args.doTest == 'True'
doTrain = args.doTrain == 'True'
resolution = args.resolution
aggregation = args.aggregation
dataset = args.dataset

############################# LOADER ######################################

Loader = Myloader(args)

# Load target
Loader.load_target()

# Aggregate
Loader.apply_aggregation()

# Compute target
y = Loader.compute_target_rules()

# Compute features
departement_mask = read_object('departement.pkl', root_target / sinister / dataset / sinister_encoding)[0]
departement_mask = interpolate_image_2d(departement_mask, np.isnan(Loader.images_aggregated[0]))
X, features_name = Loader.compute_features(departement_mask, features_list=default_features)

models = [#('ResNet_spatial_classification_supervised_weightedcrossentropy', 'cnn', None),
          #('ResNet_spatial_classification_reinforcement-POO_mse', 'cnn', None),
          #('ConvLSTM_spatial_classification_supervised_weightedcrossentropy', 'cnn', None),
          #('LSTM_spatial_classification_supervised_weightedcrossentropy', 'normal', None),
          #('LSTM_spatial_regression_reinforcement-POO_mse', 'normal', None),
          ('NetMLP_spatial_regression_reinforcement-POO_mse', 'normal', None),
          #('NetGCN_classic_classification_weightedcrossentropy', 'gnn', None),
          ]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')

#years = ['2018', '2019', '2020', '2021', '2022', '2023']

years = ['2020']
X = [X[2]]
y = [y[2]]

ks = 0

other_params = {'features_name' : features_name,
                'ks' : ks,
                'epochs' : 1,
                'dir_output' : Loader.dir_output,
                'device': device,
                'spatial_mask' : departement_mask,
                'spatial_mask_name' : 'departement',
                'temporal_mask' : {'train' : [0, 1, 2, 3, 4], 'val': [5], 'test' : [6]},
                'temporal_mask_name' : '2023-test',
                'under_sampling' : 'full',
                'lr' : 1e-5,
                'valid_transitions_dict' : Loader.valid_transitions_dict
                }

for model in models:
    model_torch = get_model_from_string(model[1], model[0], **other_params)
    model_torch.train(X, y, True, {'early_stopping_rounds': 15})
    model_torch.test(model_torch.train_loader, 'full_train')
    model_torch.test(model_torch.val_loader, 'full_val')
    model_torch.test(model_torch.test_loader, 'full_test')
    
    for i in range(ks, len(X)):
        model_torch.plot_test(X, y[i], i, years[i])