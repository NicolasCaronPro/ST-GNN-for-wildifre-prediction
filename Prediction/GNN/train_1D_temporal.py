import sys
import os

from itsdangerous import NoneAlgorithm

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
parser.add_argument('-k_days', '--k_days', type=str, help='k_days')
parser.add_argument('-days_in_futur', '--days_in_futur', type=str, help='days_in_futur')
parser.add_argument('-scaling', '--scaling', type=str, help='scaling methods')
parser.add_argument('-graphConstruct', '--graphConstruct', type=str, help='')
parser.add_argument('-sinisterEncoding', '--sinisterEncoding', type=str, help='')
parser.add_argument('-weights', '--weights', type=str, help='Type of weights')
parser.add_argument('-top_cluster', '--top_cluster', type=str, help='Top x cluster (on 5)')

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

minDate = '2017-06-12' # Starting point

if dataset_name== '':
    dataset_name= 'firemen'

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
train_dataset, val_dataset, test_dataset, train_dataset_unscale, val_dataset_unscale, test_dataset_unscale, prefix, features_selected = get_train_val_test_set(graphScale, df,
                                                                                    features_name, train_departements,
                                                                                    prefix,
                                                                                    dir_output,
                                                                                    ['mean'], args)

if nbfeatures == 'all':
    nbfeatures = len(features_selected)

_, varying_time_variables_2 = add_time_columns(varying_time_variables, k_days, train_dataset, train_features)

features_name, newshape = get_features_name_list(graphScale.scale, train_features, METHODS_SPATIAL_TRAIN)
features_selected_str = get_features_selected_for_time_series(features_selected, features_name, varying_time_variables_2, nbfeatures)
features_selected_str = list(features_selected_str)
features_selected = np.arange(0, len(features_selected_str))
logger.info((features_selected_str, len(features_selected_str)))

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

train_dataset['weight'] = train_dataset[weights_version]
val_dataset['weight'] = val_dataset[weights_version]
train_dataset['weight_nbsinister'] = train_dataset[f'{weights_version}_nbsinister']
val_dataset['weight_nbsinister'] = val_dataset[f'{weights_version}_nbsinister']

test_dataset['weight'] = val_dataset['weight_class_nbsinister']
test_dataset['weight_nbsinister'] = val_dataset['weight_class_nbsinister']

models = [
        ('LSTM', 'risk_regression_rmse', autoRegression),
        ('LSTM', 'nbsinister_regression_rmse', autoRegression),
          #('LSTM', 'binary_classification_weightedcrossentropy', autoRegression),
           # ('LSTM', 'risk_regression_rmse', autoRegression),
           # ('LSTM', 'risk_regression_rmsle', autoRegression),
           # ('LSTM', 'risk_regression_mse', autoRegression),
           # ('LSTM', 'risk_regression_huber', autoRegression),
           # ('LSTM', 'risk_regression_logcosh', autoRegression),
           # ('LSTM', 'risk_regression_tukeybiweight', autoRegression),
           # ('LSTM', 'risk_regression_exponential', autoRegression),
            ]

gnn_models = [
    #('DST-GCN', False, 'risk_regression_rmse', autoRegression),
    ('LSTMGCN', False, 'risk_regression_rmse', autoRegression),
    ('LSTMGCN', False, 'nbsinister_regression_rmse', autoRegression),
    ('LSTMGAT', False, 'risk_regression_rmse', autoRegression),
    ('LSTMGAT', False, 'nbsinister_regression_rmse', autoRegression),
    #('ST-GAT', False, 'risk_regression_rmse', autoRegression),
    #('ST-GCN', False, 'risk_regression_rmse', autoRegression),
    #('SDT-GCN', False, 'risk_regression_rmse', autoRegression),
    #('ATGN', False, 'risk_regression_rmse', autoRegression),
    #('ST-GATLSTM', False, 'risk_regression_rmse', autoRegression),
    #('GAT', False, 'nbsinister_regression_rmse', autoRegression),
    #('DST-GCN', False, 'nbsinister_regression_rmse', autoRegression),
    #('ST-GAT', False, 'nbsinister_regression_rmse', autoRegression),
    #('ST-GCN', False, 'nbsinister_regression_rmse', autoRegression),
    #('SDT-GCN', False, 'nbsinister_regression_rmse', autoRegression),
    #('ATGN', False, 'nbsinister_regression_rmse', autoRegression),
    #('ST-GATLTSM', False, 'nbsinister_regression_rmse', autoRegression)
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
    "scale": scale,
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

    host = 'pc'
    
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
            ('c_index', c_i, 'class'),
            ('c_index_class', c_i_class, 'class'),
            ('kendall', kendall_coefficient, 'correlation'),
            ('pearson', pearson_coefficient, 'correlation'),
            ('spearman', spearman_coefficient, 'correlation')
            ]

    models = [
            ('LSTM_risk_regression_rmse', None, 'risk', autoRegression),
            ('LSTM_nbsinister_regression_rmse', None, 'nbsinister', autoRegression),
            #('LSTM_binary_classification_weightedcrossentropy', False, 'binary', autoRegression),
            #('LSTM_risk_regression_poisson', False, 'risk', autoRegression),
            #('LSTM_risk_regression_rmsle', False, 'risk', autoRegression),
            #('LSTM_risk_regression_mse', False, 'risk', autoRegression),
            #('LSTM_risk_regression_huber', False, 'risk', autoRegression),
            #('LSTM_risk_regression_logcosh', False, 'risk', autoRegression),
            #('LSTM_risk_regression_tukeybiweight', False, 'risk', autoRegression),
            #('LSTM_risk_regression_exponential', False, 'risk', autoRegression),
    ]

    gnn_models = [
        #('DST-GCN_risk_regression_rmse', False, 'risk', autoRegression),
        ('ST-GAT_risk_regression_rmse', False, 'risk', autoRegression),
        ('ST-GCN_risk_regression_rmse', False, 'risk', autoRegression),
        ('LSTMGCN_risk_regression_rmse', False, 'risk', autoRegression),
        ('LSTMGCN_nbsinister_regression_rmse', False, 'nbsinister', autoRegression),
        ('LSTMGAT_risk_regression_rmse', False, 'risk', autoRegression),
        ('LSTMGAT_nbsinister_regression_rmse', False, 'nbsinister', autoRegression),
        #('SDT-GCN_risk_regression_rmse', False, 'risk', autoRegression),
        #('ATGN_risk_regression_rmse', False, 'risk', autoRegression),
        #('ST-GATLSTM_risk_regression_rmse', False, 'risk', autoRegression),
        #('DST-GCN', False, 'nbsinister_regression_rmse', autoRegression),
        #('ST-GAT', False, 'nbsinister_regression_rmse', autoRegression),
        #('ST-GCN', False, 'nbsinister_regression_rmse', autoRegression),
        #('SDT-GCN', False, 'nbsinister_regression_rmse', autoRegression),
        #('ATGN', False, 'nbsinister_regression_rmse', autoRegression),
        #('ST-GATLTSM', False, 'nbsinister_regression_rmse', autoRegression)
    ]

    for dept in departements:
        if MLFLOW:
            exp_name = f"{dataset_name}_{dept}_{sinister}_{sinister_encoding}_test"
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
            #test_dataset_unscale_dept = train_dataset_unscale[(train_dataset_unscale['departement'] == name2int[dept]) & (train_dataset['weight_one'] > 0) ].reset_index(drop=True)
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
                            prefix=prefix,
                            prefix_train=prefix,
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
            for name, use_temporal_as_edges, target_name, autoRegression in models:
                res = read_object(name+'_'+prefix+'_'+scaling+'_'+encoding+'_'+dept+'_pred.pkl', dir_output / dept / prefix / name)
                res_dept = read_object(name+'_'+prefix+'_'+scaling+'_'+encoding+'_'+dept+'_dept_pred.pkl', dir_output / dept / prefix / name)
            
            metrics = read_object('metrics'+'_'+prefix+'_'+'_'+scaling+'_'+encoding+'_'+dept+'_dl.pkl', dir_output / dept / prefix)
            metrics_dept = read_object('metrics'+'_'+prefix+'_'+'_'+scaling+'_'+encoding+'_'+dept+'_dept_dl.pkl', dir_output / dept / prefix)

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
                           prefix=prefix,
                           prefix_train=prefix,
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