from dataloader import *
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
days_in_futur = args.days_in_futur # The target time validation
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

df, graphScale, prefix = init(args, dir_output, True)
save_object(df.columns, 'features_name.pkl', dir_output)

if MLFLOW:
    mlflow.set_experiment(f"{dir_output}_{prefix}")

    existing_run = get_existing_run('Preprocessing')
    if existing_run:
        mlflow.start_run(run_id=existing_run.info.run_id)
    else:
        mlflow.start_run(run_name='Preprocessing', nested=True)
    

############################# Train, Val, test ###########################

train_dataset, val_dataset, test_dataset, train_dataset_unscale, test_dataset_unscale, prefix, features_selected = get_train_val_test_set(graphScale, df,
                                                                                    train_features, train_departements,
                                                                                    prefix,
                                                                                    dir_output, ['mean'], args)


features_name, newShape = get_features_name_list(6, features, ['mean'])
features_name_2D, newShape2D = get_features_name_lists_2D(6, train_features)
logger.info(f'features_name_2D shape : {len(features_name_2D)}')

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
dir_output = dir_output / spec

epochs = 10000
lr = 0.01
PATIENCE_CNT = 200
CHECKPOINT = 100

cnn_models = [('Zhang', False, 'risk_regression_rmse', True, autoRegression),
             ('UNET', False, 'risk_regression_rmse', False, autoRegression)
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
    "features_name": features_name,
    "train_dataset_unscale": train_dataset_unscale,
    "scale": scale,
    "features_name_2D": features_name_2D,
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

        wrapped_train_deep_learning_2D(params)

if doTest:

    trainCode = [name2int[dept] for dept in train_departements]

    x_train, y_train = train_dataset

    x_test, y_test  = test_dataset

    name_dir = name_exp + '/' + sinister + '/' + resolution + '/train' + '/'
    dir_train = Path(name_dir)

    name_dir = name_exp + '/' + sinister + '/' + resolution + '/test' + '/'
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
            ('cal', cal, 'bin'),
            ('f1', f1, 'bin'),
            ('class', ck, 'class'),
            ('poisson', po, 'proba'),
            ('ca', ca, 'class'),
            ('bca', bca, 'class'),
            ('maec', meac, 'class'),
            ]

    cnn_models = [('Zhang', False, 'risk_regression_rmse', True, autoRegression),
                ('UNET', False, 'risk_regression_rmse', False, autoRegression)
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

        test_dl_model(graphScale, test_dataset_dept, train_dataset_unscale,
                           methods,
                           dept,
                           features_name,
                           prefix,
                           prefix,
                           dummy,
                           cnn_models,
                           dir_output / spec / dept / prefix,
                           device,
                           k_days,
                           Rewrite, 
                           encoding,
                           scaling,
                           [dept],
                           dir_train,
                           features_name_2D,
                           sinister,
                           resolution,
                           spec)