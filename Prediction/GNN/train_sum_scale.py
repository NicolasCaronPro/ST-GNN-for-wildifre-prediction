import sys
import os

from sqlalchemy import false

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
parser.add_argument('-dataset', '--dataset', type=str, help='Dataset')
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
nbfeatures = int(args.NbFeatures)
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

df, graphScale, prefix, fp, features_name = init(args, dir_output, 'train_trees')

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

prefix += f'_{weights_version}'

train_dataset['weight'] = train_dataset[weights_version]
val_dataset['weight'] = val_dataset[weights_version]
train_dataset['weight_nbsinister'] = train_dataset[f'{weights_version}_nbsinister']
val_dataset['weight_nbsinister'] = val_dataset[f'{weights_version}_nbsinister']

test_dataset['weight'] = test_dataset[f'weight_normalize_nbsinister']
test_dataset['weight_nbsinister'] = test_dataset['weight_normalize_nbsinister']

if doTest:

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
            ('spearman', spearman_coefficient, 'correlation')
            ]
    
    name_models = 'xgboost_5_7_risk'
    models = [
        #('full_0_100_5_risk-watershed_None_weight_normalize', 'xgboost_risk_regression_rmse_features', 'risk', False),
        ('full_0_100_7_risk-watershed_None_weight_normalize', 'xgboost_risk_regression_rmse_features', 'risk', False),
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
            exp_name = f"{dn}_{dept}_{sinister}_{sinister_encoding}_test"
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

        logger.info(f'{dept} test : {test_dataset_dept.shape}, {np.unique(test_dataset_dept["id"].values)}')

        metrics, metrics_dept, res, res_dept = test_sum_scale_model(vars(args),
                               name_models,
                                graphScale, test_dataset_dept,
                               test_dataset_unscale_dept,
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
                                name_exp,
                                dir_train / 'check_none' / prefix_kmeans / 'kmeans',
                                doKMEANS
                                )
        
        res = pd.concat(res).reset_index(drop=True)
        #res_dept = pd.concat(res_dept).reset_index(drop=True)

        ############## Prediction ######################
        for prefix_model, name, target_name, autoRegression in models:
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
            #res_test_dept = res_dept[res_dept['model'] == name]
            regions_test = geo[geo['departement'] == dept]
            #dates = res_test[(res_test['risk'] == res_test['risk'].max())]['date'].values
            dates = allDates.index('2023-07-14')

            susectibility_map_france_daily_geojson(res_test, regions_test, graphScale, np.unique(dates).astype(int), 'prediction',
                                        scale, dir_output / dept / prefix / name, vmax_band=vmax_band,
                                        dept_reg=False, sinister=sinister, sinister_point=fp)

            """susectibility_map_france_daily_image(df=res_test, vmax=vmax_band, graph=graphScale, departement=dept, dates=dates,
                            resolution='2x2', region_dept=regions_test, column='prediction',
                            dir_output=dir_output / dept / prefix / name, predictor=predictor, sinister_point=fp.copy(deep=True))"""

        """train_dataset_dept = train_dataset[train_dataset['departement'] == name2int[dept]].groupby(['departement', 'date'])[target_name].sum().reset_index()
            vmax_band = np.nanmax(train_dataset_dept[target_name].values)

            susectibility_map_france_daily_geojson(res_test_dept,
                                        regions_test, graphScale, np.unique(dates).astype(int),
                                        target_name, 'departement',
                                        dir_output / dept / prefix / name, vmax_band=vmax_band,
                                        dept_reg=True, sinister=sinister, sinister_point=fp)"""

        """################# Ground Truth #####################
        name = 'GT'
        susectibility_map_france_daily_geojson(res_test, regions_test, graphScale, np.unique(dates).astype(int), 'risk',
                                    f'{scale}_gt', dir_output / dept / prefix / name, vmax_band=vmax_band,
                                    dept_reg=False, sinister=sinister, sinister_point=fp)
        
        susectibility_map_france_daily_geojson(res_test_dept,
                                    regions_test, graphScale, np.unique(dates).astype(int),
                                    target_name, 'departement_gt',
                                    dir_output / dept / prefix / name, vmax_band=vmax_band,
                                    dept_reg=True, sinister=sinister, sinister_point=fp)"""
        
        aggregated_prediction.append(res)
        #aggregated_prediction_dept.append(res_dept)

    aggregated_prediction = pd.concat(aggregated_prediction).reset_index(drop=True)
    #aggregated_prediction_dept = pd.concat(aggregated_prediction_dept).reset_index(drop=True)

    aggregated_prediction.to_csv(dir_output / 'aggregated_prediction.csv', index=False)

    for prefix_model, name, target_name, autoRegression in models:
        band = 'prediction'
        if target_name == 'binary':
            vmax_band = 1
        else:
            vmax_band = np.nanmax(aggregated_prediction['prediction'].values)

        aggregated_prediction = aggregated_prediction[aggregated_prediction['model'] == name]
        #aggregated_prediction_dept = aggregated_prediction_dept[aggregated_prediction_dept['model'] == name]

        #dates = aggregated_prediction[aggregated_prediction['nbsinister'] == aggregated_prediction['nbsinister'].max()]['date'].values
        dates = [allDates.index('2023-06-14')]

        region_france = gpd.read_file('/home/caron/Bureau/csv/france/data/geo/hexagones_france.gpkg')
        region_france['latitude'] = region_france['geometry'].apply(lambda x : float(x.centroid.y))
        region_france['longitude'] = region_france['geometry'].apply(lambda x : float(x.centroid.x))

        ####################### Prediction #####################
        check_and_create_path(dir_output / name)
        susectibility_map_france_daily_geojson(aggregated_prediction, region_france, graphScale, np.unique(dates).astype(int), 'prediction', f'{scale}_france',
                                    dir_output / name, vmax_band, dept_reg=False, sinister=sinister, sinister_point=fp)
        
        """train_dataset_dept = train_dataset.groupby(['departement', 'date'])[target_name].sum().reset_index()
        vmax_band = np.nanmax(train_dataset_dept[target_name].values)
        
        susectibility_map_france_daily_geojson(aggregated_prediction_dept, region_france.copy(deep=True), graphScale, np.unique(dates).astype(int), 'prediction', f'departemnnt_france',
                                    dir_output / name, vmax_band, dept_reg=True, sinister=sinister, sinister_point=fp)"""
    
    """######################## Ground Truth ####################
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