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
days_in_futur = int(args.days_in_futur) # The target time validation
scaling = args.scaling
graph_construct = args.graphConstruct
sinister_encoding = args.sinisterEncoding
weights_version = args.weights
top_cluster = args.top_cluster
graph_method = args.graph_method
shift = args.shift
thresh_kmeans = args.thresh_kmeans

assert values_per_class == 'full'
assert graph_method == 'node'

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

geo = gpd.read_file(f'regions/{sinister}/{dataset_name}/regions.geojson')
geo = geo[geo['departement'].isin(departements)].reset_index(drop=True)

minDate = '2017-06-12' # Starting point

if dataset_name == '':
    dataset_name = 'firemen'

autoRegression = 'AutoRegressionReg' in train_features
if autoRegression:
    name_exp += '_AutoRegressionReg'

####################### INIT ################################

df, graphScale, prefix, fp, features_selected = init(args, dir_output, 'train_trees')

if MLFLOW:
    exp_name = f"{dataset_name}_train"
    experiments = client.search_experiments()

    if exp_name not in list(map(lambda x: x.name, experiments)):
        tags = {
            "",
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
                                                                                    features_selected, train_departements,
                                                                                    prefix,
                                                                                    dir_output,
                                                                                    ['mean'], args)
if MLFLOW:
    train_dataset_ml_flow = mlflow.data.from_pandas(train_dataset)
    val_dataset_ml_flow = mlflow.data.from_pandas(val_dataset)
    test_dataset_ml_flow = mlflow.data.from_pandas(test_dataset)

    mlflow.log_param('train_features', train_features)
    mlflow.log_param('features', features)
    mlflow.log_param('features_selected', features_selected)
    mlflow.log_input(train_dataset_ml_flow, context='trainig')
    mlflow.log_input(val_dataset_ml_flow, context='validation')
    mlflow.log_input(test_dataset_ml_flow, context='testing')
    mlflow.end_run()

######################## Training ##############################

prefix += f'_{weights_version}'

test_dataset_unscale['weight_nbsinister'] = 1
test_dataset['weight'] = 1

name = 'check_'+scaling + '/' + prefix + '/' + 'baseline'

###################### Defined ClassRisk model ######################

# Min Max
minmaxclass = MinMaxClass(thresholds=[0.25, 0.5, 0.75, 1.0])

minmaxclass.fit(train_dataset['nbsinister'].values)

train_dataset['nbsinister-MinMaxClass'] = minmaxclass.predict_risk(train_dataset['nbsinister'].values)     
val_dataset['nbsinister-MinMaxClass'] = minmaxclass.predict_risk(val_dataset['nbsinister'].values)
test_dataset['nbsinister-MinMaxClass'] = minmaxclass.predict_risk(test_dataset['nbsinister'].values)

train_dataset['nbsinister-MinMax'] = minmaxclass.predict(train_dataset['nbsinister'].values)     
val_dataset['nbsinister-MinMax'] = minmaxclass.predict(val_dataset['nbsinister'].values)        
test_dataset['nbsinister-MinMax'] = minmaxclass.predict(test_dataset['nbsinister'].values)

# Other...

models = [
        #('xgboost_risk_regression_eae-1', 'risk', None),
        #('xgboost_risk_regression_rmse', 'risk', MinMaxClass(thresholds=[0.25, 0.5, 0.75, 1.0])),
        #('xgboost_nbsinister_regression_rmsle', 'nbsinister', None),
        #('xgboost_nbsinister-MinMax_regression_rmse', 'nbsinister-robust', MinMaxClass(thresholds=[0.25, 0.5, 0.75, 1.0])),
        #('xgboost_nbsinister-MinMaxClass_regression_rmse', 'nbsinister-MinMax', None),
        ('xgboost_binary-1_one_nbsinister-MinMaxClass_classification_softmax', 'nbsinister-MinMaxClass', minmaxclass),
        ('xgboost_binary-2_one_nbsinister-MinMaxClass_classification_softmax', 'nbsinister-MinMaxClass', minmaxclass),
        ('xgboost_binary-3_one_nbsinister-MinMaxClass_classification_softmax', 'nbsinister-MinMaxClass', minmaxclass),
        ('xgboost_binary-4_one_nbsinister-MinMaxClass_classification_softmax', 'nbsinister-MinMaxClass', minmaxclass),
        #('xgboost_nbsinister_regression_sig2', 'nbsinister', None),
        #('xgboost_nbsinister_regression_sig', 'nbsinister', None),
        #('xgboost_nbsinister_regression_rmse', 'nbsinister', None),
        #('xgboost_nbsinister_classification_softmax', 'nbsinister', None),
        #('xgboost_nbsinister_regression_quantile', 'nbsinister', None),
        #('xgboost_nbsinister_regression_mse', 'nbsinister', None),
        #('xgboost_nbsinister_regression_eae-1', 'nbsinister', None),
        #('xgboost_binary_classification_logloss', 'binary', None),
        ]

gam_models = [
        #('gam_binary_classification_logloss', 'binary'),
        #('gam_risk_regression_rmse', 'risk'),
        #('gam_nbsinister_regression_rmse', 'nbsinister'),
]

voting_models = [
                #('xgboost_risk_regression_rmse', 'risk'),
                #('xgboost_binary_classification_logloss', 'binary')
                #('xgboost-xgboost-xgboost_nbsinister_regression_rmse', 'nbsinister')
                ]

one_by_id_model = [
       #('xgboost_risk_regression_rmse', 'risk', 'unique', 'unique', 'departement'),
       #('xgboost_nbsinister_regression_rmse', 'nbsinister', 'unique', 'departement'),
       #('xgboost_nbsinister_regression_rmse', 'nbsinister', 'unique', 'graph_id'),
       #('xgboost_nbsinister_regression_rmse', 'nbsinister', 'unique', 'cluster_encoder'),
       #('xgboost_nbsinister_regression_rmse', 'nbsinister', 'unique', 'month'),
       #('xgboost_nbsinister_regression_rmse', 'nbsinister', 'unique', 'dayofweek'),
       #('xgboost_nbsinister_regression_rmse', 'nbsinister', 'unique', 'isweekend'),
       #('xgboost_binary_classification_logloss', 'binary', 'unique', 'departement'),
]

federate_model_by_id_model = [
]

voting_models_list = [
                    #[('xgboost_risk_regression_rmse', 'risk'),
                    # ('svm_risk_regression_rmse', 'risk'),
                    # ('dt_risk_regresion_rmse', 'risk')]
                    ]


if doTrain:

    for model in models:
        wrapped_train_sklearn_api_model(train_dataset=train_dataset.copy(deep=True),
                                val_dataset=val_dataset.copy(deep=True),
                                test_dataset=test_dataset.copy(deep=True),
                                graph_method=graph_method,
                                dir_output=dir_output / name,
                                device='gpu',
                                autoRegression=autoRegression,
                                features=features_selected,
                                optimize_feature=optimize_feature,
                                do_grid_search=do_grid_search,
                                do_bayes_search=do_bayes_search,
                                model=model)
        
    for model in voting_models:
        wrapped_train_sklearn_api_voting_model(train_dataset=train_dataset.copy(deep=True),
                                val_dataset=val_dataset.copy(deep=True),
                                test_dataset=test_dataset.copy(deep=True),
                                graph_method=graph_method,
                                dir_output=dir_output / name,
                                device='cpu',
                                autoRegression=autoRegression,
                                features=features_selected,
                                optimize_feature=optimize_feature,
                                do_grid_search=do_grid_search,
                                do_bayes_search=do_bayes_search,
                                model_name=model)
        
    for model in gam_models:
         wrapped_train_sklearn_api_model(train_dataset=train_dataset.copy(deep=True),
                                val_dataset=val_dataset.copy(deep=True),
                                test_dataset=test_dataset.copy(deep=True),
                                graph_method=graph_method,
                                dir_output=dir_output / name,
                                device='cpu',
                                autoRegression=autoRegression,
                                features=features_selected,
                                optimize_feature=optimize_feature,
                                do_grid_search=do_grid_search,
                                do_bayes_search=do_bayes_search,
                                model=model)
        
    for model in one_by_id_model:
        wrapped_train_sklearn_api_model(train_dataset=train_dataset.copy(deep=True),
                                val_dataset=val_dataset.copy(deep=True),
                                test_dataset=test_dataset.copy(deep=True),
                                graph_method=graph_method,
                                dir_output=dir_output / name,
                                device='cpu',
                                autoRegression=autoRegression,
                                features=features_selected,
                                optimize_feature=optimize_feature,
                                do_grid_search=do_grid_search,
                                do_bayes_search=do_bayes_search,
                                model=model)
        
    for model in federate_model_by_id_model:
        wrapped_train_sklearn_api_model(train_dataset=train_dataset.copy(deep=True),
                                val_dataset=val_dataset.copy(deep=True),
                                test_dataset=test_dataset.copy(deep=True),
                                graph_method=graph_method,
                                dir_output=dir_output / name,
                                device='cpu',
                                autoRegression=autoRegression,
                                features=features_selected,
                                optimize_feature=optimize_feature,
                                do_grid_search=do_grid_search,
                                do_bayes_search=do_bayes_search,
                                model=model)

if doTest:

    host = 'pc'

    logger.info('############################# TEST ###############################')

    dn = dataset_name
    if two:
        dn += '2'

    name_dir = dn + '/' + sinister + '/' + resolution + '/train' + '/'
    dir_train = Path(name_dir)

    name_dir = dn + '/' + sinister + '/' + resolution + '/test' + '/' + name_exp
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

    models = [
        #('xgboost_nbsinister-MinMax_regression_rmse', 'nbsinister', autoRegression),
        #('xgboost_nbsinister-MinMaxClass_regression_rmse', 'nbsinister', autoRegression),
        ('xgboost_binary-1_one_nbsinister-MinMaxClass_classification_softmax', 'nbsinister-MinMaxClass', autoRegression),
        ('xgboost_binary-2_one_nbsinister-MinMaxClass_classification_softmax', 'nbsinister-MinMaxClass', autoRegression),
        ('xgboost_binary-3_one_nbsinister-MinMaxClass_classification_softmax', 'nbsinister-MinMaxClass', autoRegression),
        ('xgboost_binary-4_one_nbsinister-MinMaxClass_classification_softmax', 'nbsinister-MinMaxClass', autoRegression),
        #('xgboost_nbsinister_regression_rmse', 'nbsinister', autoRegression),
        #('xgboost_risk_regression_rmse', 'risk', autoRegression),
        #('xgboost_nbsinister_regression_quantile', 'nbsinister', autoRegression),
        #('unique-departement-xgboost_nbsinister_regression_rmse', 'nbsinister', autoRegression),
        #('unique-month-xgboost_nbsinister_regression_rmse', 'nbsinister', autoRegression),
        #('unique-dayofweek-xgboost_nbsinister_regression_rmse', 'nbsinister', autoRegression),
        #('unique-isweekend-xgboost_nbsinister_regression_rmse', 'nbsinister', autoRegression),
        #('xgboost-xgboost-xgboost_nbsinister_nbsinister_regression_rmse', 'nbsinister', autoRegression),
        ]

    prefix_kmeans = f'{values_per_class}_{k_days}_{scale}_{graph_construct}_{top_cluster}'

    if days_in_futur > 0:
        prefix_kmeans += f'_{days_in_futur}_{futur_met}'

    aggregated_prediction = []
    aggregated_prediction_dept = []

    for dept in departements:
        if MLFLOW:
            dn = dataset_name
            if two:
                dn += '2'
            exp_name = f"{name_exp}_{dn}_{dept}_{sinister}_{sinister_encoding}_test"
            experiments = client.search_experiments()

            if exp_name not in list(map(lambda x: x.name, experiments)):

                client.create_experiment(name=exp_name)

            mlflow.set_experiment(exp_name)

        test_dataset_dept = test_dataset[(test_dataset['departement'] == name2int[dept])].reset_index(drop=True)
        if test_dataset_unscale is not None:
            test_dataset_unscale_dept = test_dataset_unscale[(test_dataset_unscale['departement'] == name2int[dept])].reset_index(drop=True)
        else:
            test_dataset_unscale_dept = None
        
        if test_dataset_dept.shape[0] < 5:
            continue

        logger.info(f'{dept} test : {test_dataset_dept.shape}, {np.unique(test_dataset_dept["id"].values)}')

        if host == 'pc':
            metrics, metrics_dept, res, res_dept = test_sklearn_api_model(vars(args), graphScale, test_dataset_dept,
                                test_dataset_unscale_dept,
                                    dept,
                                    prefix,
                                    prefix_config,
                                    models,
                                    dir_output / dept / prefix,
                                    device,
                                    encoding,
                                    scaling,
                                    [dept],
                                    dir_train,
                                    name_exp,
                                    dir_train / 'check_none' / prefix_kmeans / 'kmeans',
                                    doKMEANS
                                    )
        
            res = pd.concat(res).reset_index(drop=True)
        else:
            metrics = read_object('metrics'+'_'+prefix+'_'+scaling+'_'+encoding+'_'+dept+'_tree.pkl', dir_output / dept / prefix)
            res = read_object(name+'_'+prefix+'_'+scaling+'_'+encoding+'_'+dept+'_pred.pkl', dir_output / name)
            assert metrics is not None
            run = f'{dept}_{name}_{prefix}'
            if MLFLOW:
                for name, use_temporal_as_edges, target_name, autoRegression in models:
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
        #aggregated_prediction_dept.append(res_dept)

    df_metrics.rename({'index': 'Run'}, inplace=True, axis=1)
    df_metrics.reset_index(drop=True, inplace=True)
    
    check_and_create_path(dir_output / prefix)
    df_metrics.to_csv(dir_output / prefix / 'df_metrics.csv')

    wrapped_compare(df_metrics,  dir_output / prefix, 'Model')
    wrapped_compare(df_metrics,  dir_output / prefix, 'Loss_function')
    wrapped_compare(df_metrics,  dir_output / prefix, 'Target')
    wrapped_compare(df_metrics,  dir_output / prefix, 'Days_in_futur')

    """############## Prediction ######################
    for name, target_name, _ in models:
        if name not in res.model.unique():
            continue
        if target_name == 'risk':
            predictor = read_object(f'{dept}Predictor{scale}_{graphScale.base}.pkl', dir_train / 'influenceClustering')
        else:
            predictor = read_object(f'{dept}Predictor{name}{scale}_{graphScale.base}.pkl', dir_train / 'influenceClustering')

        if target_name == 'binary':
            vmax_band = 1
        else:
            #vmax_band = np.nanmax(train_dataset[train_dataset['departement'] == name2int[dept]][target_name].values)
            vmax_band = np.nanmax(train_dataset[target_name].values)

        res_test = res[res['model'] == name]
        res_test_dept = res_dept[res_dept['model'] == name]
        regions_test = geo[geo['departement'] == dept]
        #dates = res_test[(res_test['risk'] == res_test['risk'].max())]['date'].values
        dates = allDates.index('2023-07-14')

        susectibility_map_france_daily_geojson(res_test, regions_test, graphScale, np.unique(dates).astype(int), 'prediction',
                                    scale, dir_output / dept / prefix / name, vmax_band=vmax_band,
                                    dept_reg=False, sinister=sinister, sinister_point=fp)

        #susectibility_map_france_daily_image(df=res_test, vmax=vmax_band, graph=graphScale, departement=dept, dates=dates,
        #                resolution='2x2', region_dept=regions_test, column='prediction',
        #                dir_output=dir_output / dept / prefix / name, predictor=predictor, sinister_point=fp.copy(deep=True))

        train_dataset_dept = train_dataset[train_dataset['departement'] == name2int[dept]].groupby(['departement', 'date'])[target_name].sum().reset_index()
        vmax_band = np.nanmax(train_dataset_dept[target_name].values)

        susectibility_map_france_daily_geojson(res_test_dept,
                                    regions_test, graphScale, np.unique(dates).astype(int),
                                    target_name, 'departement',
                                    dir_output / dept / prefix / name, vmax_band=vmax_band,
                                    dept_reg=True, sinister=sinister, sinister_point=fp)

        ################# Ground Truth #####################
        name = 'GT'
        susectibility_map_france_daily_geojson(res_test, regions_test, graphScale, np.unique(dates).astype(int), target_name,
                                    f'{scale}_gt', dir_output / dept / prefix / name, vmax_band=vmax_band,
                                    dept_reg=False, sinister=sinister, sinister_point=fp)
        
        susectibility_map_france_daily_geojson(res_test_dept,
                                    regions_test, graphScale, np.unique(dates).astype(int),
                                    target_name, 'departement_gt',
                                    dir_output / dept / prefix / name, vmax_band=vmax_band,
                                    dept_reg=True, sinister=sinister, sinister_point=fp)"""

    """aggregated_prediction = pd.concat(aggregated_prediction).reset_index(drop=True)
    #aggregated_prediction_dept = pd.concat(aggregated_prediction_dept).reset_index(drop=True)

    aggregated_prediction.to_csv(dir_output / 'aggregated_prediction.csv', index=False)

    for name, target_name, _ in models:
        band = 'prediction'
        if target_name == 'binary':
            vmax_band = 1
        else:
            vmax_band = np.nanmax(aggregated_prediction[target_name].values)

        aggregated_prediction_model = aggregated_prediction[aggregated_prediction['model'] == name]
        #aggregated_prediction_dept_model = aggregated_prediction_dept[aggregated_prediction_dept['model'] == name]

        #dates = aggregated_prediction[aggregated_prediction['nbsinister'] == aggregated_prediction['nbsinister'].max()]['date'].values
        dates = [
            allDates.index('2023-04-26'),
            allDates.index('2023-05-25'),
            allDates.index('2023-03-06'),
            allDates.index('2023-04-17'),
            allDates.index('2023-06-15'),
            allDates.index('2023-09-03'),
            allDates.index('2023-09-17'),
            allDates.index('2023-09-13'),
            allDates.index('2023-08-15'),
            allDates.index('2023-05-08')
        ]

        region_france = gpd.read_file('/home/caron/Bureau/csv/france/data/geo/hexagones_france.gpkg')
        region_france['latitude'] = region_france['geometry'].apply(lambda x : float(x.centroid.y))
        region_france['longitude'] = region_france['geometry'].apply(lambda x : float(x.centroid.x))

        ####################### Prediction #####################
        #check_and_create_path(dir_output / name)
        susectibility_map_france_daily_geojson(aggregated_prediction_model, region_france, graphScale, np.unique(dates).astype(int), 'prediction', f'{scale}_france',
                                    dir_output / name, vmax_band, dept_reg=False, sinister=sinister, sinister_point=fp)
        
        train_dataset_dept = train_dataset.groupby(['departement', 'date'])[target_name].sum().reset_index()
        vmax_band = np.nanmax(train_dataset_dept[target_name].values)
        
        #susectibility_map_france_daily_geojson(aggregated_prediction_dept, region_france.copy(deep=True), graphScale, np.unique(dates).astype(int), 'prediction', f'departemnnt_france',
        #                            dir_output / name, vmax_band, dept_reg=True, sinister=sinister, sinister_point=fp)
    
    ######################## Ground Truth ####################
    name = 'GT'
    check_and_create_path(dir_output / name)
    susectibility_map_france_daily_geojson(aggregated_prediction, region_france.copy(deep=True), graphScale, np.unique(dates).astype(int), target_name,
                                       f'{scale}_gt', dir_output / name, vmax_band=vmax_band,
                                       dept_reg=False, sinister=sinister, sinister_point=fp)

    susectibility_map_france_daily_geojson(aggregated_prediction_dept,
                                       region_france.copy(deep=True), graphScale, np.unique(dates).astype(int),
                                       target_name, 'departement_gt',
                                       dir_output / name, vmax_band=vmax_band,
                                       dept_reg=True, sinister=sinister, sinister_point=fp)"""