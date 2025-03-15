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
parser.add_argument('-dd', '--database2D', type=str, help='Do 2D database')
parser.add_argument('-sc', '--scale', type=str, help='Scale')
parser.add_argument('-dataset', '--dataset', type=str, help='spec')
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
parser.add_argument('-scaling', '--scaling', type=str, help='scaling methods')
parser.add_argument('-graphConstruct', '--graphConstruct', type=str, help='')
parser.add_argument('-sinisterEncoding', '--sinisterEncoding', type=str, help='')
parser.add_argument('-weights', '--weights', type=str, help='Type of weights')

args = parser.parse_args()

# Input config
dataset_name = args.dataset
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
nbfeatures = args.NbFeatures
sinister = args.sinister
values_per_class = args.nbpoint
scale = int(args.scale) if args.scale != 'departement' else args.scale
resolution = args.resolution
doPCA = args.pca == 'True'
doKMEANS = args.KMEANS == 'True'
ncluster = int(args.ncluster)
do_grid_search = args.GridSearch ==  'True'
do_bayes_search = args.BayesSearch == 'True'
k_days = int(args.k_days) # Size of the time series sequence use by DL models
days_in_futur = args.days_in_futur # The target time validation
scaling = args.scaling
graph_construct = args.graphConstruct
sinister_encoding = args.sinisterEncoding
weights_version = args.weights

######################## Get features and train features list ######################

isInference = name_exp == 'inference'

features, train_features, kmeans_features = get_features_for_sinister_prediction(dataset_name, sinister, isInference)

######################## Get departments and train departments #######################

departements, train_departements = select_departments(dataset_name, sinister)

############################# GLOBAL VARIABLES #############################

name_exp = f'{sinister_encoding}_{name_exp}'

dir_target = root_target / sinister / dataset_name / sinister_encoding / 'log' / resolution

geo = gpd.read_file(f'regions/{sinister}/{dataset_name}/regions.geojson')
geo = geo[geo['departement'].isin(departements)].reset_index(drop=True)

name_dir = dataset_name + '/' + sinister + '/' + resolution + '/' + 'train' +  '/'
dir_output = Path(name_dir)
check_and_create_path(dir_output)

minDate = '2017-06-12' # Starting point

if dataset_name == '':
    dataset_name = 'firemen'

if dummy:
    dataset_name+= '_dummy'

autoRegression = 'AutoRegressionReg' in train_features
if autoRegression:
    dataset_name += '_AutoRegressionReg'

####################### INIT ################################

df, graphScale, prefix, fp, features_name = init(args, dir_output, 'train_cnn')
save_object(df.columns, 'features_name.pkl', dir_output)

if MLFLOW:
    exp_name = f"{dataset_name}_train"
    experiments = client.search_experiments()

    if exp_name not in list(map(lambda x: x.name, experiments)):
        tags = {
            "mlflow.note.content": "This experiment is an example of how to use mlflow. The project allows to predict housing prices in california.",
        }

        client.create_experiment(name=exp_name, tags=tags)
        
    mlflow.set_experiment(exp_name)
    existing_run = get_existing_run('Preprocessing')
    if existing_run:
        mlflow.start_run(run_id=existing_run.info.run_id)
    else:
        mlflow.start_run(run_name='Preprocessing')

############################# Train, Val, test ###########################
dir_output = dir_output / name_exp

train_dataset, val_dataset, test_dataset, train_dataset_unscale, test_dataset_unscale, prefix, features_selected = get_train_val_test_set(graphScale, df,
                                                                                    features_name, train_departements,
                                                                                    prefix,
                                                                                    dir_output,
                                                                                    ['mean'], args)

if nbfeatures == 'all':
    nbfeatures = len(features_selected)

for k in range(1, k_days + 1):
    new_fet = [f'{v}_{str(k)}' for v in varying_time_variables]
    new_fet = [nf for nf in new_fet if nf.split('_')[0] in train_features]
    new_fet_features_name, _ = get_features_name_list(graphScale.scale, new_fet, METHODS_TEMPORAL)
    df.drop(new_fet_features_name, axis=1, inplace=True)
    for nf in new_fet:
        train_features.pop(train_features.index(nf))
        features.pop(features.index(nf))

features_name, newShape = get_features_name_list(6, train_features, ['mean'])
features_name_2D, newShape2D = get_features_name_lists_2D(6, train_features)

features_selected_str_2D = get_features_selected_for_time_series_for_2D(features_selected, features_name_2D, varying_time_variables, nbfeatures)
features_selected_str_2D = list(np.unique(features_selected_str_2D))
features_selected_2D = np.arange(0, len(features_selected_str_2D))

if k_days > 0:
    features_name, newshape = get_features_name_list(graphScale.scale, train_features, METHODS_SPATIAL_TRAIN)
    features_selected_str_1D = get_features_selected_for_time_series(features_selected, features_name, varying_time_variables, nbfeatures)
    features_selected_str_1D = list(features_selected_str_1D)
    features_selected_1D = np.arange(0, len(features_selected_str_1D))
else:
    features_selected_str_1D = list(features_selected)[:nbfeatures]
    features_selected_1D = np.arange(0, len(features_selected_str_1D))

features_selected = (features_selected_1D, features_selected_2D)

if MLFLOW:
    train_dataset_ml_flow = mlflow.data.from_pandas(train_dataset)
    val_dataset_ml_flow = mlflow.data.from_pandas(val_dataset)
    test_dataset_ml_flow = mlflow.data.from_pandas(test_dataset)

    mlflow.log_input(train_dataset_ml_flow, context='trainig')
    mlflow.log_input(val_dataset_ml_flow, context='validation')
    mlflow.log_input(test_dataset_ml_flow, context='testing')
    mlflow.log_param('train_features', train_features)
    mlflow.log_param('features', features)
    mlflow.log_param('features_selected', features_selected)
    mlflow.log_param('features_name_2D', features_name_2D)
    mlflow.end_run()

############################# Training ##################################

prefix += f'_{weights_version}'
train_dataset['weight'] = train_dataset[weights_version]
val_dataset['weight'] = val_dataset[weights_version]

cnn_models = [
             #('HybridConvGraphNet', False, 'risk_regression_rmse', True, autoRegression),
              ('ConvGraphNet', False, 'risk_regression_rmse', True, autoRegression),
            ]

train_loader = None
val_loader = None
last_bool = None

params = {
    "graphScale": graphScale,
    "train_dataset": train_dataset,
    "val_dataset": val_dataset,
    "test_dataset": test_dataset,
    "k_days": k_days,
    "device": device,
    "optimize_feature": optimize_feature,
    "PATIENCE_CNT": PATIENCE_CNT,
    "CHECKPOINT": CHECKPOINT,
    "epochs": epochs,
    "lr": lr,
    "scaling": scaling,
    "encoding": encoding,
    "prefix": prefix,
    "Rewrite": Rewrite,
    "dir_output": dir_output,
    "train_dataset_unscale": train_dataset_unscale,
    "scale": scale,
    'features': features,
    "features_name": features_name,
    "features_selected": features_selected,
    "features_selected_str_2D": features_selected_str_2D,
    "features_selected_str_1D": features_selected_str_1D,
    "name_dir": name_dir,
    'k_days' : k_days,
}

if doTrain:
    for cnn_model in cnn_models:
        params['model'] = cnn_model[0]
        params['use_temporal_as_edges'] = cnn_model[1]
        params['infos'] = cnn_model[2]
        params['image_per_node'] = cnn_model[3]
        params['autoRegression'] = cnn_model[4]

        wrapped_train_deep_learning_hybrid(params)

if doTest:
    
    if MLFLOW:
        exp_name = f"{dataset_name}_{sinister}_{sinister_encoding}_test"
        experiments = client.search_experiments()

        if exp_name not in list(map(lambda x: x.name, experiments)):
            tags = {
                "mlflow.note.content": "This experiment is an example of how to use mlflow. The project allows to predict housing prices in california.",
            }

            client.create_experiment(name=exp_name, tags=tags)

        mlflow.set_experiment(exp_name)
    
    prefix_kmeans = f'{values_per_class}_{k_days}_{scale}_{graph_construct}'

    trainCode = [name2int[dept] for dept in train_departements]

    name_dir = dataset_name + '/' + sinister + '/' + resolution + '/train' + '/'
    dir_train = Path(name_dir)

    name_dir = dataset_name + '/' + sinister + '/' + resolution + '/test' + '/' + name_exp
    dir_output = Path(name_dir)

    mae = my_mean_absolute_error
    rmse = weighted_rmse_loss
    std = standard_deviation
    cal = quantile_prediction_error
    fre = frequency_class_error
    f1 = my_f1_score
    ca = class_accuracy
    bca = balanced_class_accuracy
    acc = class_accuracy
    ck = class_risk
    po = poisson_loss
    meac = mean_absolute_error_class
    c_i_class = c_index_class
    c_i = c_index
    bacc = binary_accuracy

    methods = [
            ('mae', mae, 'proba'),
            ('rmse', rmse, 'proba'),
            ('std', std, 'proba'),
            ('cal', cal, 'cal'),
            ('fre', fre, 'cal'),
            ('binary', f1, 'bin'),
            ('binary', bacc, 'bin'),
            ('class', ck, 'class'),
            #('poisson', po, 'proba'),
            ('ca', ca, 'class'),
            ('bca', bca, 'class'),
            ('maec', meac, 'class'),
            ('acc', acc, 'class'),
            ('c_index', c_i, 'class'),
            ('c_index_class', c_i_class, 'class'),
            ('kendall', kendall_coefficient, 'correlation'),
            ('pearson', pearson_coefficient, 'correlation'),
            ('spearman', spearman_coefficient, 'correlation')
            ]

    cnn_models = [('HybridConvGraphNet_risk_regression_rmse', False, 'risk', autoRegression),
                ]

    for dept in departements:
        test_dataset_dept = test_dataset[(test_dataset['departement'] == name2int[dept])].reset_index(drop=True)
        if test_dataset_unscale is not None:
            test_dataset_unscale_dept = test_dataset_unscale[(test_dataset_unscale['departement'] == name2int[dept])].reset_index(drop=True)
        else:
            test_dataset_unscale_dept = None

        if test_dataset_dept.shape[0] < 5:
            continue

        logger.info(f'{dept} test : {test_dataset_dept.shape}')

        test_dl_model(args=vars(args), graphScale=graphScale, test_dataset_dept=test_dataset_dept,
                          test_dataset_unscale_dept=test_dataset_unscale_dept,
                          train_dataset=train_dataset_unscale,
                           methods=methods,
                           test_name=dept,
                           features_name=features_name,
                           prefix=prefix,
                           prefix_train=prefix,
                           models=cnn_models,
                           dir_output=dir_output / dept / prefix,
                           device=device,
                           k_days=k_days,
                           encoding=encoding,
                           features=features,
                           scaling=scaling,
                           test_departement=[dept],
                           dir_train=dir_train,
                           dir_break_point=dir_train / 'check_none' / prefix_kmeans / 'kmeans',
                           name_exp=name_exp,
                           doKMEANS=doKMEANS)