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

dir_post_process = dir_output / 'post_process'

post_process_model_dico = post_process_model(train_dataset, val_dataset, test_dataset, dir_post_process)

############################# Training ##################################

cnn_models = [
             ('Zhang', None, 'binary_one_nbsinister-sum-3-kmeans-5-Class-Dept_classification_weightedcrossentropy', True),
             #('ConvLSTM', None, 'nbsinister_regression_poisson', True),
            #('Zhang', None, 'nbsinister_regression_poisson', True),
             #('Unet', FalsNonee, 'risk_regression_rmse', False)
            ]

train_loader = None
val_loader = None
last_bool = None

if is_pc:
    temp_dir = 'Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN'
else:
    temp_dir = '.'

params = {
    "graph": graphScale,
    "val_dataset": val_dataset_unscale,
    "test_dataset": test_dataset_unscale,
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
}

if doTrain:
    for cnn_model in cnn_models:
        params['model'] = cnn_model[0]
        params['use_temporal_as_edges'] = cnn_model[1]
        params['infos'] = cnn_model[2]
        params['image_per_node'] = cnn_model[3]
        params['autoRegression'] = False
        params['torch_structure'] = 'Model_CNN'

        wrapped_train_deep_learning_2D(params)

if doTest:
    
    host = 'server'
    
    prefix_kmeans = f'{values_per_class}_{k_days}_{scale}_{graph_construct}_{top_cluster}'

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
            ('accuracy', bacc, 'bin'),
            ('class', ck, 'class'),
            #('poisson', po, 'proba'),
            ('ca', ca, 'class'),
            ('bca', bca, 'class'),
            ('maec', meac, 'class'),
            ('acc', acc, 'class'),
            #('c_index', c_i, 'class'),
            #('c_index_class', c_i_class, 'class'),
            ('kendall', kendall_coefficient, 'correlation'),
            ('pearson', pearson_coefficient, 'correlation'),
            ('spearman', spearman_coefficient, 'correlation'),
            ('auc_roc', my_roc_auc, 'bin')
            ]

    cnn_models = [('Zhang_risk_regression_rmse', None, 'risk', autoRegression),
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
            test_dl_model(args=vars(args), graphScale=graphScale, test_dataset_dept=test_dataset_dept,
                            test_dataset_unscale_dept=test_dataset_unscale_dept,
                            train_dataset=train_dataset_unscale,
                            methods=methods,
                            test_name=dept,
                            features_name=features_name,
                            prefix_train=prefix,
                            prefix_config=prefix_config,
                            models=cnn_models,
                            dir_output=dir_output / dept / prefix,
                            device=device,
                            k_days=k_days,
                            encoding=encoding,
                            scaling=scaling,
                            features=features,
                            test_departement=[dept],
                            dir_train=dir_train,
                            dir_break_point=dir_train / 'check_none' / prefix_kmeans / 'kmeans',
                            name_exp=name_exp,
                            doKMEANS=doKMEANS)
        
        else:
            #for name, use_temporal_as_edges, target_name, autoRegression in cnn_model:
            #    res = read_object(name+'_'+prefix+'_'+scaling+'_'+encoding+'_'+dept+'_pred.pkl', dir_output / dept / prefix / name)
            #    res_dept = read_object(name+'_'+prefix+'_'+scaling+'_'+encoding+'_'+dept+'_dept_pred.pkl', dir_output / dept / prefix / name)
            
            metrics = read_object('metrics'+'_'+prefix+'_'+'_'+scaling+'_'+encoding+'_'+dept+'_dl.pkl', dir_output / dept / prefix)
            #metrics_dept = read_object('metrics'+'_'+prefix+'_'+'_'+scaling+'_'+encoding+'_'+dept+'_dept_dl.pkl', dir_output / dept / prefix)
            if MLFLOW:
                for name, use_temporal_as_edges, target_name, autoRegression in cnn_models:
                    existing_run = get_existing_run(f'{dept}_{name}_{prefix}')
                    if existing_run:
                        mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
                    else:
                        mlflow.start_run(run_name=f'{dept}_{name}_{prefix}', nested=True)
                    if name in metrics.keys():
                        log_metrics_recursively(metrics[name], prefix='')
                    else:
                        logger.info(f'{name} not found')
                    mlflow.end_run()