from dataloader import *
from config import *
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
parser.add_argument('-dd', '--database2D', type=str, help='Do 2D database')
parser.add_argument('-sc', '--scale', type=str, help='Scale')
parser.add_argument('-sp', '--spec', type=str, help='spec')
parser.add_argument('-nf', '--NbFeatures', type=str, help='Number de Features')
parser.add_argument('-of', '--optimizeFeature', type=str, help='Launch test')
parser.add_argument('-gs', '--GridSearch', type=str, help='GridSearch')
parser.add_argument('-bs', '--BayesSearch', type=str, help='BayesSearch')
parser.add_argument('-test', '--doTest', type=str, help='Launch test')
parser.add_argument('-train', '--doTrain', type=str, help='Launch train')
parser.add_argument('-r', '--resolution', type=str, help='Resolution of image')
parser.add_argument('-pca', '--pca', type=str, help='Apply PCA')
parser.add_argument('-kmeans', '--KMEANS', type=str, help='Apply kmeans preprocessing')
parser.add_argument('-ncluster', '--ncluster', type=str, help='Number of cluster for kmeans')
parser.add_argument('-k_days', '--k_days', type=str, help='k_days')
parser.add_argument('-days_in_futur', '--days_in_futur', type=str, help='days_in_futur')

args = parser.parse_args()

# Input config
name_exp = args.name
maxDate = args.maxDate
trainDate = args.trainDate
doEncoder = args.encoder == "True"
doPoint = args.point == "True"
doGraph = args.graph == "True"
doDatabase = args.database == "True"
do2D = args.database2D == "True"
doFet = args.featuresSelection == "True"
optimize_feature = args.optimizeFeature == "True"
doTest = args.doTest == "True"
doTrain = args.doTrain == "True"
nbfeatures = int(args.NbFeatures)
sinister = args.sinister
values_per_class = args.nbpoint
scale = int(args.scale)
spec = args.spec
resolution = args.resolution
doPCA = args.pca == 'True'
doKMEANS = args.KMEANS == 'True'
ncluster = int(args.ncluster)
do_grid_search = args.GridSearch ==  'True'
do_bayes_search = args.BayesSearch == 'True'
k_days = int(args.k_days) # Size of the time series sequence use by DL models
days_in_futur = int(args.days_in_futur) # The target time validation
scaling = args.scaling

############################# GLOBAL VARIABLES #############################

dir_target = root_target / sinister / 'log' / resolution

geo = gpd.read_file('regions/regions.geojson')
geo = geo[geo['departement'].isin(departements)].reset_index(drop=True)

name_dir = name_exp + '/' + sinister + '/' + resolution + '/' + 'train' +  '/'
dir_output = Path(name_dir)
check_and_create_path(dir_output)

minDate = '2017-06-12' # Starting point

if spec == '':
    spec = 'default'

if dummy:
    spec += '_dummy'

autoRegression = 'AutoRegressionReg' in train_features
if autoRegression:
    spec = '_AutoRegressionReg'

####################### INIT ################################

df, graphScale, prefix, _ = init(args, dir_output, True)
save_object(df.columns, 'features_name.pkl', dir_output)

############################# Train, Val, test ###########################

train_dataset, val_dataset, test_dataset, features_selected = get_train_val_test_set(graphScale, df,
                                                                                    train_features, train_departements,
                                                                                    prefix,
                                                                                    dir_output, args)

# Base models
train_dataset_list, val_dataset_list, test_dataset_list, features_selected_list, dir_model_list, model_list, target_list = [], [], [], [], [], [], []

############################ Models ####################################

# Model 1
values_per_class = 'full'
k_days = 7
scale = 30
nbfeatures = 700
model_type = 'xgboost'
target = 'risk'
task_type = 'regression'
loss = 'rmse'
name_exp_model = 'default'
sinister_model = 'firepoint'
resolution_model = '2x2'
prefix_model = f'{values_per_class}_{k_days}_{scale}_{nbfeatures}'
model_name = f'{model_type}_{target}_{task_type}_{loss}'
dir_model =  Path(name_exp + '/' + sinister + '/' + resolution + '/' + 'train' +  '/')

df_model = read_object(f'df_{prefix_model}.pkl', dir_model)
features_name_model = read_object('train_features.pkl', dir_model)
graph_model = read_object(f'graph_{scale}.pkl', dir_model)

dir_model = dir_model / prefix_model

train_dataset, val_dataset, test_dataset, features_selected = get_train_val_test_set(graph_model, df_model, train_features, train_departements,
                                                                                    prefix,
                                                                                    dir_output, args)

train_dataset_list.append(train_dataset)
val_dataset_list.append(val_dataset)
test_dataset_list.append(test_dataset)
features_selected_list.append(features_selected)
dir_model_list.append(dir_model)
model_list.append(model_name)
target_list.append(target)


# Model 2
values_per_class = 'full'
k_days = 7
scale = 10
nbfeatures = 700
model_type = 'xgboost'
target = 'risk'
task_type = 'regression'
loss = 'rmse'
name_exp_model = 'default'
sinister_model = 'firepoint'
resolution_model = '2x2'
prefix_model = f'{values_per_class}_{k_days}_{scale}_{nbfeatures}'
model_name = f'{model_type}_{target}_{task_type}_{loss}'
dir_model =  Path(name_exp + '/' + sinister + '/' + resolution + '/' + 'train' +  '/')

df_model = read_object(f'df_{prefix_model}.pkl', dir_model)
features_name_model = read_object('train_features.pkl', dir_model)
graph_model = read_object(f'graph_{scale}.pkl', dir_model)

dir_model = dir_model / prefix_model

train_dataset, val_dataset, test_dataset, features_selected = get_train_val_test_set(graph_model, df_model, train_features, train_departements,
                                                                                    prefix,
                                                                                    dir_output, args)

train_dataset_list.append(train_dataset)
val_dataset_list.append(val_dataset)
test_dataset_list.append(test_dataset)
features_selected_list.append(features_selected)
dir_model_list.append(dir_model)
model_list.append(model_name)
target_list.append(target)

############################# Training ###########################

if doTrain:
    name = 'check_'+scaling + '/' + prefix + '/' + '/baseline'
    wrapped_train_sklearn_api_fusion_model(train_dataset, val_dataset,
                                           test_dataset,
                                           train_dataset_list=train_dataset_list,
                                           val_dataset_list=val_dataset_list,
                                           test_dataset_list=test_dataset_list,
                                           target_list=target_list,
                                           model_list=model_list,
                                           dir_model_list=dir_model_list,
                            dir_output=dir_output,
                            device='cpu',
                            autoRegression=autoRegression,
                            optimize_feature=optimize_feature,
                            do_grid_search=do_grid_search,
                            do_bayes_search=do_bayes_search,
                            deep=False)

if doTest:
    x_test = test_dataset[0]
    y_test = test_dataset[1]

    name_dir = name_exp + '/' + sinister + '/' + resolution + '/train' + '/'
    dir_train = Path(name_dir)

    name_dir = name_exp + '/' + sinister + '/' + resolution + '/test' + '/'
    dir_output = Path(name_dir)

    testd_departement = [int2name[d] for d in np.unique(y_test[:, 3])]

    mae = my_mean_absolute_error
    rmse = weighted_rmse_loss
    std = standard_deviation
    cal = quantile_prediction_error
    f1 = my_f1_score
    ca = class_accuracy
    bca = balanced_class_accuracy
    ck = class_risk
    po = poisson_loss
    meac = mean_absolute_error_class

    methods = [
            ('mae', mae, 'proba'),
            ('rmse', rmse, 'proba'),
            ('std', std, 'proba'),
            ('cal', cal, 'bin'),
            ('f1', f1, 'bin'),
            ('class', ck, 'class'),
            ('poisson', po, 'proba'),
            ('ca', ca, 'class'),
            ('bca', bca, 'class'),
            ('maec', meac, 'class'),
            ]
    
    models = [
        ('linear', 'risk', autoRegression),
        ('mean', 'risk', autoRegression),
        ('xgboost_risk', 'risk', autoRegression),

        ('linear_binary', 'binary', autoRegression),
        ('mean_binary', 'binary', autoRegression),
        ('xgboost_binary', 'binary', autoRegression),

        ('linear_nbsinister', 'nbsinister', autoRegression),
        ('mean_nbsinister', 'nbsinister', autoRegression),
        ('xgboost_nbsinister', 'nbsinister', autoRegression),
        ]

    for dept in departements:
        Xset = x_test[(x_test[:, 3] == name2int[dept])]
        Yset = y_test[(x_test[:, 3] == name2int[dept])]

        if Xset.shape[0] == 0:
            continue

        save_object(Xset, 'X'+'_'+dept+'_'+prefix+'.pkl', dir_output)
        save_object(Xset, 'Y_'+dept+'_'+prefix+'.pkl', dir_output)

        logger.info(f'{dept} test : {Xset.shape}, {allDates[int(np.min(Xset[:,4]))], allDates[int(np.max(Yset[:,4]))]}')

        test_sklearn_api_model(graphScale, Xset, Yset,
                        methods,
                        dept,
                        prefix,
                        models,
                        dir_output / dept / prefix,
                        device,
                        encoding,
                        scaling,
                        [dept],
                        dir_train,
                        features_name)