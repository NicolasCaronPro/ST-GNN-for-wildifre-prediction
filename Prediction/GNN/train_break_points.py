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

X, Y, graphScale, prefix, features_name, _ = init(args, dir_output, True)

############################# Train, Val, test ###########################

train_dataset, val_dataset, test_dataset, features_selected = get_train_val_test_set(graphScale, X, Y, features_name, train_departements, dir_output, args)

############################# Training ###########################

trainCode = [name2int[dept] for dept in train_departements]

# Select train features
train_fet_num = select_train_features(train_features, scale, features_name)

dir_output =  dir_output / spec

save_object(features, 'features.pkl', dir_output)
save_object(train_features, 'train_features.pkl', dir_output)
save_object(features_name, 'features_name.pkl', dir_output)

if days_in_futur > 1:
    prefix += '_'+str(days_in_futur)
    prefix += '_'+futur_met

features_name, newshape = get_features_name_list(graphScale.scale, 6, train_features)

X = X[:, np.asarray(train_fet_num)]
logger.info(features_name)
logger.info(np.max(X[:,4]))
logger.info(np.unique(X[:,0]))
logger.info(np.nanmax(X))
logger.info(np.unravel_index(np.nanargmax(X), X.shape))

# Preprocess
train_dataset, val_dataset, test_dataset = preprocess(X=X, Y=Y, scaling=scaling, maxDate=maxDate,
                                                   trainDate=trainDate, train_departements=train_departements,
                                                   departements = departements,
                                                   ks=k_days, dir_output=dir_output, prefix=prefix, features_name=features_name,
                                                   days_in_futur=days_in_futur,
                                                   futur_met=futur_met)

if doPCA:
    prefix += '_pca'
    Xtrain = train_dataset[0]
    pca, components = train_pca(Xtrain, 0.99, dir_output / prefix, train_fet_num)
    train_dataset = (apply_pca(train_dataset[0], pca, components), train_dataset[1], train_fet_num)
    val_dataset = (apply_pca(val_dataset[0], pca, components), val_dataset[1], train_fet_num)
    test_dataset = (apply_pca(test_dataset[0], pca, components), test_dataset[1], train_fet_num)
    features_selected = np.arange(6, components + 6)
    train_features = ['pca_'+str(i) for i in range(components)]
    features_name, _ = get_features_name_list(0, 6, train_features)
    kmeansFeatures = train_features
else:
    features_selected = np.arange(6, X.shape[1])

prefix = f'{str(values_per_class)}_{str(k_days)}_{str(scale)}_{str(nbfeatures)}'

if days_in_futur > 1:
    prefix += '_'+str(days_in_futur)
    prefix += '_'+futur_met

if doTrain:
    train_break_point(train_dataset[0], train_dataset[1], kmeansFeatures, dir_output / 'varOnValue' / prefix, features_name, scale, ncluster)

    y_train = train_dataset[1]
    y_val = val_dataset[1]
    y_test = test_dataset[1]

    train_dataset = (train_dataset[0], apply_kmeans_class_on_target(train_dataset[0], y_train, dir_output / 'varOnValue' / prefix, True))
    val_dataset = (val_dataset[0], apply_kmeans_class_on_target(val_dataset[0], y_val, dir_output / 'varOnValue' / prefix, True))
    test_dataset = (test_dataset[0], apply_kmeans_class_on_target(test_dataset[0], y_test, dir_output / 'varOnValue' / prefix, True))

    realVspredict(Y[:, -1], Y, -1, dir_output / prefix, 'raw')
    realVspredict(Y[:, -3], Y, -3, dir_output / prefix, 'class')

    sinister_distribution_in_class(Y[:, -3], Y, dir_output / prefix)

logger.info(f'Train dates are between : {allDates[int(np.min(train_dataset[0][:,4]))], allDates[int(np.max(train_dataset[0][:,4]))]}')
logger.info(f'Val dates are bewteen : {allDates[int(np.min(val_dataset[0][:,4]))], allDates[int(np.max(val_dataset[0][:,4]))]}')

if doTest:
    logger.info('############################# TEST ###############################')

    Xtrain = train_dataset[0]
    y_train = train_dataset[1]

    x_test = test_dataset[0]
    y_test = testDataset[1]

    train_dir = copy(dir_output)

    name_dir = nameExp + '/' + sinister + '/' + resolution + '/test' + '/'
    dir_output = Path(name_dir)

    rmse = weighted_rmse_loss
    std = standard_deviation
    cal = quantile_prediction_error
    f1 = my_f1_score
    ca = class_accuracy
    bca = balanced_class_accuracy
    ck = class_risk
    po = poisson_loss
    meac = mean_absolute_error_class

    methods = [('rmse', rmse, 'proba'),
            ('std', std, 'proba'),
            #('cal', cal, 'bin'),
            ('f1', f1, 'bin'),
            #('class', ck, 'class'),
            #('poisson', po, 'proba'),
            #('ca', ca, 'class'),
            #('bca', bca, 'class'),
            #('meac', meac, 'class'),
            ]
    
    for dept in departements:
        Xset = x_test[(x_test[:, 3] == name2int[dept])]
        Yset = y_test[(x_test[:, 3] == name2int[dept])]

        if Xset.shape[0] == 0:
            continue

        save_object(Xset, 'X'+'_'+dept+'_'+prefix+'.pkl', dir_output)
        save_object(Xset, 'Y_'+dept+'_'+prefix+'.pkl', dir_output)

        logger.info(f'{dept} test : {Xset.shape}, {allDates[int(np.min(Xset[:,4]))], allDates[int(np.max(Yset[:,4]))]}')

        test_break_points_model(Xset, Yset,
                           methods,
                           dept,
                           prefix,
                           dir_output / dept / prefix,
                           encoding,
                           scaling,
                           [dept],
                           scale,
                           train_dir / dept,
                           sinister,
                           resolution)