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
parser.add_argument('-dataset', '--dataset', type=str, help='Dataset to use')
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
parser.add_argument('-shift', '--shift', type=str, help='Shift of kmeans', default='0')
parser.add_argument('-thresh_kmeans', '--thresh_kmeans', type=str, help='Thresh of fr to remove sinister', default='0')
parser.add_argument('-k_days', '--k_days', type=str, help='k_days')
parser.add_argument('-days_in_futur', '--days_in_futur', type=str, help='days_in_futur')
parser.add_argument('-scaling', '--scaling', type=str, help='scaling methods')
parser.add_argument('-graphConstruct', '--graphConstruct', type=str, help='')
parser.add_argument('-sinisterEncoding', '--sinisterEncoding', type=str, help='')
parser.add_argument('-weights', '--weights', type=str, help='Type of weights')
parser.add_argument('-top_cluster', '--top_cluster', type=str, help='Top x cluster (on 5)')
parser.add_argument('-graph_method', '--graph_method', type=str, help='Top x cluster (on 5)', default='node')
parser.add_argument('-training_mode', '--training_mode', type=str, help='training_mode', default='normal')

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
top_cluster = args.top_cluster
graph_method = args.graph_method
training_mode = args.training_mode

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

if dataset_name== '':
    dataset_name= 'firemen'

if dummy:
    dataset_name+= '_dummy'

autoRegression = 'AutoRegressionReg' in train_features
if autoRegression:
    dataset_name= '_AutoRegressionReg'

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

prefix_config = deepcopy(prefix)

train_dataset, val_dataset, test_dataset, train_dataset_unscale, val_dataset_unscale, test_dataset_unscale, prefix, features_selected = get_train_val_test_set(graphScale, df,
                                                                                    features_name, train_departements,
                                                                                    prefix,
                                                                                    dir_output,
                                                                                    ['mean'], args)
if nbfeatures == 'all':
    nbfeatures = len(features_selected)
else:
    nbfeatures = int(nbfeatures)
    
varying_time_variables_2 = get_time_columns(varying_time_variables, k_days, train_dataset.copy(), train_features)

features_name, newShape = get_features_name_list(6, train_features, ['mean'])
features_name = [fet for fet in features_name if fet in train_dataset.columns]
features_name_2D, newShape2D = get_features_name_lists_2D(6, train_features)
features_selected_str = get_features_selected_for_time_series_for_2D(features_name, features_name_2D, varying_time_variables_2, nbfeatures)
features_selected_str = list(np.unique(features_selected_str))
features_selected = np.arange(0, len(features_selected_str))

logger.info(f'features_name_2D shape : {len(features_name_2D)}')
logger.info(features_selected_str) 

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

###################### Defined ClassRisk model ######################

if is_pc:
    temp_dir = 'Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN'
else:
    temp_dir = 'GNN/'


dir_post_process = dir_output / 'post_process'

post_process_model_dico, train_dataset, val_dataset, test_dataset, new_cols = post_process_model(train_dataset, val_dataset, test_dataset, dir_post_process, graphScale)

features_name.append('Past_risk')
features_selected_str.append('Past_risk')
features_selected = np.arange(0, len(features_selected_str))
features.append('Past_risk')

train_dataset = add_past_risk(train_dataset, 'nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized-Past')
test_dataset = add_past_risk(test_dataset, 'nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized-Past')
val_dataset = add_past_risk(val_dataset, 'nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized-Past')

raster_past_risk(train_dataset, f'rasterScale{graphScale.scale}_{graphScale.base}_{graphScale.graph_method}', \
                  dir_output / '../raster', \
                rootDisk / temp_dir / name_dir / f'2D_database_{graphScale.scale}_{graphScale.base}_{graphScale.graph_method}')

raster_past_risk(test_dataset, f'rasterScale{graphScale.scale}_{graphScale.base}_{graphScale.graph_method}', \
                  dir_output / '../raster', \
                rootDisk / temp_dir / name_dir / f'2D_database_{graphScale.scale}_{graphScale.base}_{graphScale.graph_method}')

raster_past_risk(val_dataset, f'rasterScale{graphScale.scale}_{graphScale.base}_{graphScale.graph_method}', \
                  dir_output / '../raster', \
                rootDisk / temp_dir / name_dir / f'2D_database_{graphScale.scale}_{graphScale.base}_{graphScale.graph_method}')

save_object(train_dataset, 'df_train_'+prefix+'.pkl', dir_output)
save_object(val_dataset, 'df_test_'+prefix+'.pkl', dir_output)
save_object(test_dataset, 'df_val_'+prefix+'.pkl', dir_output)
############################# Training ##################################

cnn_models = [
             #('ConvLSTM', None, 'ConvLSTM_binary_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', True, 5),
             ('Zhang', None, 'search_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', True, 5),
             #('Zhang', None, 'binary_one_nbsinister-kmeans-5-Class-Dept-both_classification_weightedcrossentropy', True, 5),
             #('Zhang',  None, 'percentage-0.3_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', True, 5),
             #('Zhang', None, 'nbsinister_regression_poisson', True),
             #('Unet', FalsNonee, 'risk_regression_rmse', False),
            ]

train_loader = None
val_loader = None
last_bool = None

params = {
    "graph": graphScale,
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
    "features_name_1D": features_name,
    "features_selected_2D": features_selected,
    "features_name_2D": features_selected_str,
    'features' : features,
    "name_dir": rootDisk / temp_dir / name_dir,
    'k_days' : k_days,
    'graph_method' : graph_method,
}

if doTrain:
    for cnn_model in cnn_models:
        params['model'] = cnn_model[0]
        params['use_temporal_as_edges'] = cnn_model[1]
        params['infos'] = cnn_model[2]
        params['image_per_node'] = cnn_model[3]
        params['out_channels'] = cnn_model[-1]
        params['autoRegression'] = False
        params['torch_structure'] = 'Model_CNN'

        wrapped_train_deep_learning_2D(params)

if doTest:
    
    host = 'pc'
    
    prefix_kmeans = f'{values_per_class}_{k_days}_{scale}_{graph_construct}_{top_cluster}'

    trainCode = [name2int[dept] for dept in train_departements]

    name_dir = dataset_name + '/' + sinister + '/' + resolution + '/train' + '/'
    dir_train = Path(name_dir)

    name_dir = dataset_name + '/' + sinister + '/' + resolution + '/test' + '/' + name_exp
    dir_output = Path(name_dir)
    aggregated_prediction = []
    
    models = [
                ('Zhang_search_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy'),
                #('Zhang_percentage-0.3_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy'),
                #('UNET_risk_regression_rmse', None, 'risk', autoRegression)
                ]

    for dept in departements:

        if MLFLOW:
            exp_name = f"{name_exp}_{dataset_name}_{dept}_{sinister}_{sinister_encoding}_test"
            experiments = client.search_experiments()

            if exp_name not in list(map(lambda x: x.name, experiments)):
                tags = {
                    "mlflow.note.content": "This experiment is an example of how to use mlflow. The project allows to predict housing prices in california.",
                }

                client.create_experiment(name=exp_name, tags=tags)

            mlflow.set_experiment(exp_name)
        test_dataset_dept = test_dataset[(test_dataset['departement'] == name2int[dept])].reset_index(drop=True)
        if test_dataset_unscale is not None:
            test_dataset_unscale_dept = test_dataset_unscale[(test_dataset_unscale['departement'] == name2int[dept])].reset_index(drop=True)
        else:
            test_dataset_unscale_dept = None

        if test_dataset_dept.shape[0] < 5:
            continue

        logger.info(f'{dept} test : {test_dataset_dept.shape}')

        if host == 'pc':
            metrics, metrics_dept, res, res_dept = test_dl_model(args=vars(args), graphScale=graphScale, test_dataset_dept=test_dataset_dept, train_dataset=train_dataset_unscale,
                            test_dataset_unscale_dept=test_dataset_unscale,
                            test_name=dept,
                            features_name=features_selected_str,
                            prefix_train=prefix,
                            prefix_config=prefix_config,
                            models=models,
                            dir_output=dir_output / dept / prefix,
                            device=device,
                            k_days=k_days,
                            encoding=encoding,
                            scaling=scaling,
                            test_departement=[dept],
                            dir_train=dir_train,
                            features=features,
                            dir_break_point=dir_train / 'check_none' / prefix_kmeans / 'kmeans',
                            name_exp=name_exp,
                            doKMEANS=doKMEANS,
                            suffix='cnn')
        else:
            metrics = read_object('metrics'+'_'+prefix+'_'+scaling+'_'+encoding+'_'+dept+'_cnn.pkl', dir_output / dept / prefix)
            assert metrics is not None
            run = f'{dept}_{name}_{prefix}'
            for name, _, target_name, _ in models:
                res = read_object(name+'_'+prefix+'_'+scaling+'_'+encoding+'_'+dept+'_pred.pkl', dir_output / dept / prefix / name)
                if MLFLOW:
                    existing_run = get_existing_run(f'{run}')
                    if existing_run:
                        mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
                    else:
                        mlflow.start_run(run_name=f'{run}', nested=True)
                    if name in metrics.keys():
                        log_metrics_recursively(metrics[run], prefix='')
                    else:
                        logger.info(f'{name} not found')
                    mlflow.end_run()

        if 'df_metrics' not in locals():
            df_metrics = pd.DataFrame.from_dict(metrics, orient='index').reset_index()
        else:
            df_metrics = pd.concat((df_metrics, pd.DataFrame.from_dict(metrics, orient='index').reset_index()))

        aggregated_prediction.append(res)

    df_metrics.rename({'index': 'Run'}, inplace=True, axis=1)
    df_metrics.reset_index(drop=True, inplace=True)
    
    check_and_create_path(dir_output / prefix)
    df_metrics.to_csv(dir_output / prefix / 'df_metrics_cnn.csv')