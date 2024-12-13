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

assert k_days == 0

######################## Get features and train features list ######################

isInference = name_exp == 'inference'

features, train_features, kmeasn_features = get_features_for_sinister_prediction(dataset_name, sinister, isInference)

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
if dataset_name == 'firemen2':
    two = True
    dataset_name = 'firemen'
else:
    two = False

minDate = '2017-06-12' # Starting point

if dataset_name== '':
    dataset_name= 'default'

if dummy:
    dataset_name+= '_dummy'

autoRegression = 'AutoRegressionReg' in train_features
if autoRegression:
    name_exp += '_AutoRegressionReg'

####################### INIT ################################

df, graphScale, prefix, fp, features_name = init(args, dir_output, 'train_gnn')
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
features_selected = list(features_selected)
if nbfeatures == 'all':
    nbfeatures = len(features_selected)
else:
    nbfeatures = int(nbfeatures)

features_selected_str = features_selected
print(features_selected_str)
features_selected = np.arange(0, len(features_selected_str))

if MLFLOW:
    train_dataset_ml_flow = mlflow.data.from_pandas(train_dataset)
    val_dataset_ml_flow = mlflow.data.from_pandas(val_dataset)
    test_dataset_ml_flow = mlflow.data.from_pandas(test_dataset)
    
    mlflow.log_param('train_features', train_features)
    mlflow.log_param('features', features)
    mlflow.log_param('features_selected_str', features_selected_str)
    mlflow.log_input(train_dataset_ml_flow, context='trainig')
    mlflow.log_input(val_dataset_ml_flow, context='validation')
    mlflow.log_input(test_dataset_ml_flow, context='testing')

    mlflow.end_run()

############################# Training ##################################

prefix += f'_{weights_version}'

train_dataset['weight'] = train_dataset[f'{weights_version}_nbsinister']
val_dataset['weight'] = val_dataset[f'{weights_version}_nbsinister']

train_dataset['weight_nbsinister'] = train_dataset[f'{weights_version}_nbsinister']
val_dataset['weight_nbsinister'] = val_dataset[f'{weights_version}_nbsinister']

test_dataset_unscale['weight_nbsinister'] = test_dataset_unscale[f'weight_outlier_2_nbsinister']
test_dataset['weight'] = test_dataset[f'weight_outlier_2_nbsinister']
#test_dataset = val_dataset

models = [
    #('KAN', 'risk_regression_rmse', autoRegression),
    #('KAN', 'nbsinister_regression_rmse', autoRegression),
    #('KAN', 'binary_classification_weightedcrossentropy', autoRegression),
            ]

gnn_models = [
    ('GAT', True, 'nbsinister_regression_rmse', autoRegression),
    #('GAT', True, 'nbsinister_regression_rmse', autoRegression),
    #('GAT', True, 'binary_classification_weightedcrossentropy', autoRegression),

    #('GCN', True, 'nbsinister_regression_rmse', autoRegression),
    #('GCN', True, 'risk_regression_rmse', autoRegression),
    #('GCN', True, 'binary_classification_weightedcrossentropy', autoRegression),
]

train_loader = None
val_loader = None
last_bool = None

params = {
    "graphScale": graphScale,
    "train_dataset": train_dataset,
    "val_dataset": val_dataset,
    "test_dataset": test_dataset,
    "features_selected": features_selected,
    "features_selected_str": features_selected_str,
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
    "graph": graphScale,
    "name_dir": name_dir,
    'k_days' : k_days,
}

if doTrain:
    for gnn_model in gnn_models:
        params['model'] = gnn_model[0]
        params['use_temporal_as_edges'] = gnn_model[1]
        params['infos'] = gnn_model[2]
        params['autoRegression'] = gnn_model[3]

        wrapped_train_deep_learning_1D(params)

    for model in models:
        params['model'] = model[0]
        params['infos'] = model[1]
        params['autoRegression'] = model[2]
        params['use_temporal_as_edges'] = None
        
        wrapped_train_deep_learning_1D(params)

if doTest:

    host = 'server'
    
    prefix_kmeans = f'{values_per_class}_{k_days}_{scale}_{graph_construct}_{graph_method}_{top_cluster}'

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

    models = [
                #('KAN_nbsinister_regression_rmse', None, 'nbsinister', autoRegression),
                #('KAN_risk_regression_rmse', None, 'risk', autoRegression),
                #('KAN_nbsinister_regression_rmse', False, 'nbsinister', autoRegression),
              #('KAN_binary_classification_weightedcrossentropy', False, 'binary', autoRegression),
    ]

    gnn_models = [
        ('GAT_nbsinister_regression_rmse', True, 'nbsinister', autoRegression),
        #('GAT_nbsinister_regression_rmse', True, 'nbsinister', autoRegression),
        #('GAT_binary_classification_weightedcrossentropy', True, 'risk', autoRegression),

        #('GCN_nbsinister_regression_rmse', True, 'nbsinister', autoRegression),
        #('GCN_risk_regression_rmse', True, 'risk', autoRegression),
        #('GCN_binary_classification_weightedcrossentropy', True, 'risk', autoRegression),
    ]

    for dept in departements:
        if MLFLOW:
            dn = dataset_name
            if two:
                dn += '2'
            exp_name = f"{name_exp}_{dn}_{dept}_{sinister}_{sinister_encoding}_test"
            experiments = client.search_experiments()

            if exp_name not in list(map(lambda x: x.name, experiments)):
                tags = {
                    "",
                }

                client.create_experiment(name=exp_name, tags=tags)

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
                            methods=methods,
                            test_name=dept,
                            features_name=features_selected_str,
                            prefix_train=prefix,
                            prefix_config=prefix_config,
                            models=gnn_models,
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
                            doKMEANS=doKMEANS)
        else:
            #for name, use_temporal_as_edges, target_name, autoRegression in cnn_model:
            #    res = read_object(name+'_'+prefix+'_'+scaling+'_'+encoding+'_'+dept+'_pred.pkl', dir_output / dept / prefix / name)
            #    res_dept = read_object(name+'_'+prefix+'_'+scaling+'_'+encoding+'_'+dept+'_dept_pred.pkl', dir_output / dept / prefix / name)
            
            metrics = read_object('metrics'+'_'+prefix+'_'+scaling+'_'+encoding+'_'+dept+'_dl.pkl', dir_output / dept / prefix)
            print(metrics)
            #metrics_dept = read_object('metrics'+'_'+prefix+'_'+'_'+scaling+'_'+encoding+'_'+dept+'_dept_dl.pkl', dir_output / dept / prefix)
            if MLFLOW:
                for name, use_temporal_as_edges, target_name, autoRegression in gnn_models:
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

        """if len(res) > 0:
            res = pd.concat(res).reset_index(drop=True)
            res_dept = pd.concat(res_dept).reset_index(drop=True)

            ############## Plot ######################
            for name, _, target_name, _ in gnn_models:
                if name not in res.model.unique():
                    continue
                if target_name == 'risk':
                    predictor = read_object(f'{dept}Predictor{scale}_{graphScale.base}.pkl', dir_train / 'influenceClustering')
                else:
                    predictor = read_object(f'{dept}Predictor{name}{scale}_{graphScale.base}.pkl', dir_train / 'influenceClustering')

                if target_name == 'binary':
                    vmax_band = 1
                else:
                    vmax_band = np.nanmax(train_dataset[train_dataset['departement'] == name2int[dept]][target_name].values)
                res_test = res[res['model'] == name]
                res_test_dept = res_dept[res_dept['model'] == name]
                regions_test = geo[geo['departement'] == dept]
                #dates = res_test_dept[res_test_dept['nbsinister'] > 1]['date'].values
                dates = res[res['nbsinister'] == res['nbsinister'].max()]['date']
                
                susectibility_map_france_daily_geojson(res_test, regions_test, graphScale, np.unique(dates).astype(int), 'prediction',
                                            scale, dir_output / dept / prefix / name, vmax_band=vmax_band,
                                            dept_reg=False, sinister=sinister, sinister_point=fp)

                susectibility_map_france_daily_image(df=res_test, vmax=vmax_band, graph=graphScale, departement=dept, dates=dates,
                                resolution='0.03x0.03', region_dept=regions_test, column='prediction',
                                dir_output=dir_output / dept / prefix / name, predictor=predictor, sinister_point=fp)"""

        metrics, metrics_dept, res, res_dept = test_dl_model(args=vars(args), graphScale=graphScale, test_dataset_dept=test_dataset_dept, train_dataset=train_dataset_unscale,
                        test_dataset_unscale_dept=test_dataset_unscale,
                           methods=methods,
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
                           doKMEANS=doKMEANS)
        
        """if len(res) > 0:
            res = pd.concat(res).reset_index(drop=True)
            res_dept = pd.concat(res_dept).reset_index(drop=True)

            ############## Plot ######################
            for name, _, target_name, _ in models:
                if name not in res.model.unique():
                    continue
                if target_name == 'risk':
                    predictor = read_object(f'{dept}Predictor{scale}_{graphScale.base}.pkl', dir_train / 'influenceClustering')
                else:
                    predictor = read_object(f'{dept}Predictor{name}{scale}_{graphScale.base}.pkl', dir_train / 'influenceClustering')

                if target_name == 'binary':
                    vmax_band = 1
                else:
                    vmax_band = np.nanmax(train_dataset[train_dataset['departement'] == name2int[dept]][target_name].values)
                    
                res_test = res[res['model'] == name]
                res_test_dept = res_dept[res_dept['model'] == name]
                regions_test = geo[geo['departement'] == dept]
                #dates = res[res['risk'] == res['risk'].max()]['date'].values
                dates = res_test_dept[res_test_dept['nbsinister'] > 0]['date'].values
                
            susectibility_map_france_daily_geojson(res_test, regions_test, graphScale, np.unique(dates).astype(int), 'prediction',
                                            scale, dir_output / dept / prefix / name, vmax_band=vmax_band,
                                            dept_reg=False, sinister=sinister, sinister_point=fp)

                susectibility_map_france_daily_image(df=res_test, vmax=vmax_band, graph=graphScale, departement=dept, dates=dates,
                                resolution='0.03x0.03', region_dept=regions_test, column='prediction',
                                dir_output=dir_output / dept / prefix / name, predictor=predictor, sinister_point=fp)

                train_dataset_dept = train_dataset[train_dataset['departement'] == name2int[dept]].groupby(['departement', 'date'])[target_name].sum().reset_index()
                vmax_band = np.nanmax(train_dataset_dept[target_name].values)

                susectibility_map_france_daily_geojson(res_test_dept,
                                            regions_test, graphScale, np.unique(dates).astype(int),
                                            target_name, 'departement',
                                            dir_output / dept / prefix / name, vmax_band=vmax_band,
                                            dept_reg=True, sinister=sinister, sinister_point=fp)"""