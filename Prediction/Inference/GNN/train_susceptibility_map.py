from operator import index
import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory
parent_dir = os.path.dirname(current_dir)

# Insert the parent directory into sys.path
sys.path.insert(0, parent_dir)

from GNN.construct import *
import geopandas as gpd
import argparse

########################### Input Arg ######################################

parser = argparse.ArgumentParser(
    prog='Train',
    description='Create graph and database according to config.py and tained model',
)
parser.add_argument('-n', '--name', type=str, help='Name of the experiment')
parser.add_argument('-e', '--encoder', type=str, help='Create encoder model')
parser.add_argument('-s', '--sinister', type=str, help='Sinister type')
parser.add_argument('-p', '--point', type=str, help='Construct points')
parser.add_argument('-np', '--nbpoint', type=str, help='Number of points')
parser.add_argument('-d', '--database', type=str, help='Do database')
parser.add_argument('-g', '--graph', type=str, help='Construct graph')
parser.add_argument('-mxd', '--maxDate', type=str, help='Limit train and validation date')
parser.add_argument('-mxdv', '--trainDate', type=str, help='Limit training date')
parser.add_argument('-f', '--featuresSelection', type=str, help='Do features selection')
parser.add_argument('-sc', '--scale', type=str, help='Scale')
parser.add_argument('-dataset', '--dataset', type=str, help='Dataset to use')
parser.add_argument('-test', '--doTest', type=str, help='Launch test')
parser.add_argument('-train', '--doTrain', type=str, help='Launch train')
parser.add_argument('-r', '--resolution', type=str, help='Resolution of image')
parser.add_argument('-kmeans', '--KMEANS', type=str, help='Apply kmeans preprocessing')
parser.add_argument('-ncluster', '--ncluster', type=str, help='Number of cluster for kmeans')
parser.add_argument('-shift', '--shift', type=str, help='Shift of kmeans', default='0')
parser.add_argument('-thresh_kmeans', '--thresh_kmeans', type=str, help='Thresh of fr to remove sinister', default='0')
parser.add_argument('-k_days', '--k_days', type=str, help='k_days')
parser.add_argument('-days_in_futur', '--days_in_futur', type=str, help='days_in_futur')
parser.add_argument('-scaling', '--scaling', type=str, help='scaling methods')
parser.add_argument('-sinisterEncoding', '--sinisterEncoding', type=str, help='')
parser.add_argument('-weights', '--weights', type=str, help='Type of weights')
parser.add_argument('-top_cluster', '--top_cluster', type=str, help='Top x cluster (on 5)')
parser.add_argument('-training_mode', '--training_mode', type=str, help='training_mode', default='normal')

args = parser.parse_args()

###########################################################################

QUICK = False

# Input config
dataset_name = args.dataset
name_exp = args.name
maxDate = args.maxDate
trainDate = args.trainDate
doEncoder = args.encoder == "True"
doPoint = args.point == "True"
doGraph = args.graph == "True"
doDatabase = args.database == "True"
doFet = args.featuresSelection == "True"
doTest = args.doTest == "True"
doTrain = args.doTrain == "True"
sinister = args.sinister
values_per_class = args.nbpoint
resolution = args.resolution
doKMEANS = args.KMEANS == 'True'
ncluster = int(args.ncluster)
k_days = int(args.k_days) # Size of the time series sequence use by DL models
days_in_futur = int(args.days_in_futur) # The target time validation
scaling = args.scaling
sinister_encoding = args.sinisterEncoding
shift = args.shift
thresh_kmeans = args.thresh_kmeans

assert values_per_class == 'full'

######################## Get features and train features list ######################

isInference = name_exp == 'inference'

features, train_features, kmeans_features = get_features_for_sinister_prediction(dataset_name, sinister, isInference)

######################## Get departments and train departments #######################

departements, train_departements = select_departments(dataset_name, sinister)

############################# GLOBAL VARIABLES #############################

name_dir = dataset_name + '/' + sinister + '/' + resolution + '/' + 'train' +  '/'
dir_output = Path(name_dir)
check_and_create_path(dir_output)

if dataset_name == 'firemen2':
    two = True
    dataset_name = 'firemen'
else:
    two = False

name_exp = f'{sinister_encoding}_{name_exp}'

dir_target = root_target / sinister / dataset_name / sinister_encoding / 'log' / resolution

minDate = '2017-06-12' # Starting point
allDates = find_dates_between('2017-06-12', '2023-12-31')

if dataset_name == '':
    dataset_name = 'firemen'

sucseptibility_map_model_config = {}
variables_for_susecptibilty_and_clustering = []

train_features = ['population', 'foret', 'osmnx', 'elevation',
                                                  #'vigicrues',
                                                  #'nappes',
                                                  'temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
                                            'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                                            'isi', 'angstroem', 'bui', 'fwi', 'dailySeverityRating',
                                            'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
                                            'days_since_rain', 'sum_consecutive_rainfall',
                                            'sum_rain_last_7_days',
                                            'sum_snow_last_7_days', 'snow24h', 'snow24h16']

model_name = f'susceptibility_mapper'
target = 'risk'
out_channels = 1
resolution = resolution
in_dim = 61
task_type = 'regression'
model_config = {
        'type': 'UNet',
        'device': 'cuda',
        'task': 'regression',
        'loss' : 'rmse',
        'name': 'mapper',
        'scaling': 'z-score',
        'infos': 'all_one_risk_regression_rmse',
        'params' : {'in_channels': in_dim,
        'out_channels': out_channels,  # Use out_channels for the final layer
        'conv_channels' : [in_dim * 2, in_dim * 4],
        'dropout': dropout,
        'device': device,
        'lr':0.00001,
        'k_days' : k_days,
        'PATIENCE_CNT' : PATIENCE_CNT,
        'CHECKPOINT' : CHECKPOINT,
        'epoch' : epochs,
        'act_func': 'relu',
        'task_type': task_type},
    }
ks = k_days
departements = departements
train_departements = train_departements
train_date = trainDate
val_date = maxDate
dir_log = dir_output
susceptibility_mapper = Model_susceptibility(model_name=model_name, target=target,\
                                                resolution=resolution, model_config=model_config,\
                                            features_name=train_features, out_channels=out_channels,
                                            task_type=task_type, ks=k_days, departements=departements,
                                            train_departements=train_departements, train_date=train_date,
                                            val_date=val_date, dir_log=dir_log)

susceptibility_mapper.create_model_and_train(root_target / sinister / dataset_name / sinister_encoding)
susceptibility_mapper.test_model(root_target / sinister / dataset_name / sinister_encoding)