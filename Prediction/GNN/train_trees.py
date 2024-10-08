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
parser.add_argument('-scaling', '--scaling', type=str, help='scaling methods')
parser.add_argument('-graphConstruct', '--graphConstruct', type=str, help='')

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
graph_construct = args.graphConstruct

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

df, graphScale, prefix = init(args, dir_output, True)

if MLFLOW:
    mlflow.set_experiment(f"{dir_output}_{prefix}")
    existing_run = get_existing_run('Preprocessing')
    if existing_run:
        mlflow.start_run(run_id=existing_run.info.run_id)
    else:
        mlflow.start_run(run_name='Preprocessing')
    
############################# Train, Val, test ###########################
dir_output = dir_output / spec
train_dataset, val_dataset, test_dataset, train_dataset_unscale, test_dataset_unscale, prefix, features_selected = get_train_val_test_set(graphScale, df,
                                                                                    train_features, train_departements,
                                                                                    prefix,
                                                                                    dir_output, METHODS_SPATIAL_TRAIN, args)

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

######################## Training ##############################

name = 'check_'+scaling + '/' + prefix + '/' + '/baseline'

if doTrain:
    wrapped_train_sklearn_api_model(train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            test_dataset=test_dataset,
                            dir_output=dir_output / name,
                            device='cpu',
                            autoRegression=autoRegression,
                            features=features_selected,
                            optimize_feature=optimize_feature,
                            do_grid_search = do_grid_search,
                            do_bayes_search = do_bayes_search)
    
    if MLFLOW:
        mlflow.end_run()

if doTest:
    if MLFLOW:
        mlflow.set_experiment(f"{dir_output}_{prefix}")
        existing_run = get_existing_run('Preprocessing')
        if existing_run:
            mlflow.start_run(run_id=existing_run.info.run_id)
        else:
            mlflow.start_run(run_name='Preprocessing')
    logger.info('############################# TEST ###############################')

    dir_train = copy(dir_output)

    name_dir = name_exp + '/' + sinister + '/' + resolution + '/test' + '/' + spec
    dir_output = Path(name_dir)

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
            ('cal', cal, 'cal'),
            ('f1', f1, 'bin'),
            ('class', ck, 'class'),
            #('poisson', po, 'proba'),
            ('ca', ca, 'class'),
            ('bca', bca, 'class'),
            ('maec', meac, 'class'),
            ]

    models = [
        ('xgboost_risk_regression_rmse_features', 'risk', autoRegression),
        ('xgboost_nbsinister_regression_rmse', 'nbsinister', autoRegression),
        ('xgboost_binary_classification_log_loss', 'binary', autoRegression),
        ]
    
    prefix_kmeans = f'{str(values_per_class)}_{str(k_days)}_{str(scale)}_{str(nbfeatures)}'
    if days_in_futur > 0:
        prefix_kmeans += f'_{days_in_futur}_{futur_met}'
    
    if graph_construct == 'geometry':
        prefix_kmeans += f'_{scale}_{graph_construct}'
    else:
        prefix_kmeans += f'_{graph_construct}'

    for dept in departements:
        test_dataset_dept = test_dataset[(test_dataset['departement'] == name2int[dept])].reset_index(drop=True)
        if test_dataset_unscale is not None:
            test_dataset_unscale_dept = test_dataset_unscale[(test_dataset_unscale['departement'] == name2int[dept])].reset_index(drop=True)
        else:
            test_dataset_unscale_dept = None

        if test_dataset_dept.shape[0] < 5:
            continue

        logger.info(f'{dept} test : {test_dataset_dept.shape}')


        test_sklearn_api_model(graphScale, test_dataset_dept,
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
                                dir_train / 'check_none' / prefix_kmeans / 'kmeans',
                                doKMEANS
                                )
        if MLFLOW:
            mlflow.end_run()