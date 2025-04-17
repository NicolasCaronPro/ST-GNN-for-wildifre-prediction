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
parser.add_argument('-sp', '--dataset', type=str, help='spec')
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
dataset_name= args.dataset
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

dir_target = root_target / dataset_name / sinister / 'log' / resolution

geo = gpd.read_file('regions/regions.geojson')
geo = geo[geo['departement'].isin(departements)].reset_index(drop=True)

name_dir = name_exp + '/' + sinister + '/' + resolution + '/' + 'train' +  '/'
dir_output = Path(name_dir)
check_and_create_path(dir_output)

minDate = '2017-06-12' # Starting point

if dataset_name== '':
    dataset_name= 'default'

if dummy:
    dataset_name+= '_dummy'

autoRegression = 'AutoRegressionReg' in train_features
if autoRegression:
    dataset_name= '_AutoRegressionReg'

####################### INIT ################################

df, graphScale, prefix = init(args, dir_output, True)
dir_output = dir_output / spec

############################# Train, Val, test ###########################
features_selected, newshape = get_features_name_list(graphScale.scale, kmeans_features, METHODS_SPATIAL_TRAIN)

df = df[ids_columns + features_selected + targets_columns]

train_dataset, val_dataset, test_dataset, test_dataset_unscale = preprocess(df=df, scaling=scaling, maxDate=maxDate,
                                                trainDate=trainDate, train_departements=train_departements,
                                                departements = departements,
                                                ks=k_days, dir_output=dir_output, prefix=prefix, features_name=features_selected,
                                                days_in_futur=days_in_futur,
                                                futur_met=futur_met, doKMEANS=doKMEANS | doTrain, ncluster=ncluster, scale=scale, save=False)

if days_in_futur > 1:
    prefix += '_'+str(days_in_futur)
    prefix += '_'+futur_met

if doTest:
    logger.info('############################# TEST ###############################')

    dir_train = dir_output

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
            ('cal', cal, 'cal'),
            ('f1', f1, 'bin'),
            ('class', ck, 'class'),
            ('poisson', po, 'proba'),
            ('ca', ca, 'class'),
            ('bca', bca, 'class'),
            ('maec', meac, 'class'),
            ]
    
    for dept in departements:
        test_dataset_dept = test_dataset[(test_dataset['departement'] == name2int[dept])]

        if test_dataset_dept.shape[0] == 0:
            continue

        logger.info(f'{dept} test : {test_dataset_dept.shape}')

        test_break_points_model(test_dataset_dept,
                           methods,
                           dept,
                           prefix,
                           dir_output / dataset_name/ dept / prefix,
                           encoding,
                           scaling,
                           [dept],
                           scale,
                           dir_train,
                           features_selected)