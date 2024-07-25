from dataloader import *
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
parser.add_argument('-scS', '--scaleSpatial', type=str, help='scaleSpatial')
parser.add_argument('-scT', '--scaleTemporal', type=str, help='scaleTemporal')
parser.add_argument('-tn', '--target_name', type=str, help='target_name')

args = parser.parse_args()

# Input config
nameExp = args.name
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
doGridSearch = args.GridSearch ==  'True'
doBayesSearch = args.BayesSearch == 'True'
k_days = int(args.k_days) # Size of the time series sequence use by DL models
days_in_futur = int(args.days_in_futur) # The target time validation
scaleSpatial = int(args.scaleSpatial)
scaleTemporal = int(args.scaleTemporal)
target_name = int(args.target_name)

############################# GLOBAL VARIABLES #############################

dir_target = root_target / sinister / 'log' / resolution

geo = gpd.read_file('regions/regions.geojson')
geo = geo[geo['departement'].isin(departements)].reset_index(drop=True)

name_dir = nameExp + '/' + sinister + '/' + resolution + '/' + 'train' +  '/'
dir_output = Path(name_dir)
check_and_create_path(dir_output)

minDate = '2017-06-12' # Starting point

if spec == '':
    spec = 'default'

if dummy:
    spec += '_dummy'

autoRegression = 'AutoRegressionReg' in trainFeatures
if autoRegression:
    spec = '_AutoRegressionReg'

############################# INIT ########################################

X, Y, graphScale, prefix, features_name, _ = init(args)

dir_output = dir_output / spec

############################# Training ###########################
    
# Daily model
dir_train = Path('final/firepoint/2x2/train/')
X_daily = read_object(f'X_full_{scaleSpatial}.pkl', dir_train)
Y_daily = read_object(f'Y_full_{scaleSpatial}.pkl', dir_train)
graph1 = read_object('graph_'+str(scaleSpatial)+'.pkl', dir_train)

trainFeatures = read_object('trainFeatures.pkl', dir_train / spec)

args1 = {'dir_train' : dir_train,
            'dir_model':  dir_train / spec / f'check_z-score/full_{k_days}_{scaleSpatial}_100' / 'baseline',
            'autoRegression' : autoRegression,
            'features_name' : get_features_name_list(scaleSpatial, 6, trainFeatures)[0],
            'model_name': f'xgboost_{target_name}',
            'model_type' : 'traditionnal',
            'target_name' : {target_name},
            'scale' : scaleSpatial,
            'isBin': False,
        'graph' : 'graph_'+str(scaleSpatial)+'.pkl',
        }

# Localized model
dir_train = Path('final/firepoint/2x2/train/')
X_localized = read_object(f'X_full_{scale}.pkl', dir_train)
Y_localized = read_object(f'Y_full_{scale}.pkl', dir_train)

trainFeatures = read_object('trainFeatures.pkl', dir_train / spec)

args2 = {'train_dir' : dir_train,
            'dir_model': dir_train / spec / f'check_z-score/full_{k_days}_{scale}_100_{scaleTemporal}_mean' / 'baseline',
            'autoRegression' : autoRegression,
            'features_name' : get_features_name_list(scale, 6, trainFeatures)[0],
            'model_name': f'xgboost_{target_name}',
            'model_type' : 'traditionnal',
            'scale' : scale,
            'target_name' : {target_name},
            'isBin': False,
        'graph' : 'graph_'+str(scale)+'.pkl',
        }

trainCode = [name2int[dept] for dept in trainDepartements]

train_fet_num = select_train_features(trainFeatures, features, scaleSpatial, get_features_name_list(scaleSpatial, 6, features)[0])
X_daily = X_daily[:, np.asarray(train_fet_num)]

train_fet_num = select_train_features(trainFeatures, features, scale, get_features_name_list(scale, 6, features)[0])
X_localized = X_localized[:, np.asarray(train_fet_num)]

# Add spatial and temporal prediction in X
trainFeatures += ['temporal_prediction', 'spatial_prediction']
features += ['temporal_prediction', 'spatial_prediction']
features_name, newshape = get_features_name_list(scale, 6, features)
X = add_temporal_spatial_prediction(X, Y_localized, Y_daily, graph1, features_name, target_name)
train_fet_num = select_train_features(trainFeatures, features, scale, features_name)
features_name, newshape = get_features_name_list(scale, 6, trainFeatures)
X = X[:, np.asarray(train_fet_num)]

# Preprocess
trainDataset, valDataset, testDataset = preprocess(X=X, Y=Y, scaling=scaling, maxDate=maxDate,
                                                trainDate=trainDate, trainDepartements=trainDepartements,
                                                departements = departements,
                                                ks=k_days, dir_output=dir_output, prefix=prefix, features_name=features_name,
                                                days_in_futur=0,
                                                futur_met=futur_met)

_, _, testDatasetDaily = preprocess(X=X_daily, Y=Y_daily, scaling=scaling, maxDate=maxDate,
                                                trainDate=trainDate, trainDepartements=trainDepartements,
                                                departements = departements,
                                                ks=k_days, dir_output=dir_output, prefix=prefix, features_name=args1['features_name'],
                                                days_in_futur=0,
                                                futur_met=futur_met)

_, _, testDatasetLocalized = preprocess(X=X_localized, Y=Y_localized, scaling=scaling, maxDate=maxDate,
                                                trainDate=trainDate, trainDepartements=trainDepartements,
                                                departements = departements,
                                                ks=k_days, dir_output=dir_output, prefix=prefix, features_name=args2['features_name'],
                                                days_in_futur=scaleTemporal,
                                                futur_met=futur_met)

prefix += '_fusion'

features_selected = features_selection(doFet, trainDataset[0], trainDataset[1], dir_output / prefix, features_name, nbfeatures, scale)

prefix = f'{str(values_per_class)}_{str(k_days)}_{str(scale)}_{str(nbfeatures)}'

if days_in_futur > 1:
    prefix += '_'+str(days_in_futur)
    prefix += '_'+futur_met

prefix += '_fusion'

if doTrain:
    name = 'check_'+scaling + '/' + prefix + '/' + '/baseline'
    # Linear combination
    train_linear(train_dataset=trainDataset, test_dataset=testDataset,
                 features=np.asarray([features_name.index('temporal_prediction'), features_name.index('spatial_prediction')]),
                 scale=scale, features_name=features_name, dir_output=dir_output / name)

    # Simple combination
    train_mean(train_dataset=trainDataset, test_dataset=testDataset,
                 features=np.asarray([features_name.index('temporal_prediction'), features_name.index('spatial_prediction')]),
                 scale=scale, features_name=features_name, dir_output=dir_output / name)

    # Add features
    train_sklearn_api_model(trainDataset=trainDataset, testDataset=testDataset, valDataset=valDataset,
                            dir_output=dir_output / name, device=device, features=features_selected,
                            scale=scale, features_name=features_name, autoRegression=autoRegression,
                            doGridSearch=doGridSearch, doBayesSearch=doBayesSearch, optimize_feature=optimize_feature)

if doTest:
    Xtest = testDataset[0]
    Ytest = testDataset[1]

    XtestDaily = testDatasetDaily[0]
    YtestDaily = testDatasetDaily[1]

    XtestLocalized = testDatasetLocalized[0]
    YtestLocalized = testDatasetLocalized[1]

    name_dir = nameExp + '/' + sinister + '/' + resolution + '/train' + '/'
    train_dir = Path(name_dir)

    name_dir = nameExp + '/' + sinister + '/' + resolution + '/test' + '/'
    dir_output = Path(name_dir)

    testDepartement = [int2name[d] for d in np.unique(Ytest[:, 3])]

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
    
    models = [
        ('linear', 'risk', autoRegression),
        ('mean', 'risk', autoRegression),
        ('xgboost_risk', 'risk', autoRegression),

        ('linear_binary', 'binary', autoRegression),
        ('mean_binary', 'binary', autoRegression),
        ('xgboost_binary', 'binary', autoRegression),

        ('linear_nbsinister', 'nbsinister', autoRegression),
        ('mean_nbsinister', 'nbsinister', autoRegression),
        ('xgboost_nbsinister', 'nbsinister', autoRegression),
        ]

    for dept in departements:
        Xset = Xtest[(Xtest[:, 3] == name2int[dept])]
        Yset = Ytest[(Xtest[:, 3] == name2int[dept])]

        XsetDaily = XtestDaily[(XtestDaily[:, 3] == name2int[dept])]
        XsetLocalized = XtestLocalized[(XtestLocalized[:, 3] == name2int[dept])]

        if Xset.shape[0] == 0:
            continue

        save_object(Xset, 'X'+'_'+dept+'_'+prefix+'.pkl', dir_output)
        save_object(Xset, 'Y_'+dept+'_'+prefix+'.pkl', dir_output)

        logger.info(f'{dept} test : {Xset.shape}, {allDates[int(np.min(Xset[:,4]))], allDates[int(np.max(Yset[:,4]))]}')

        test_fusion_prediction(models,
                           args1,
                           args2,
                           XsetDaily,
                           XsetLocalized,
                           Xset,
                           Yset,
                           dir_output / spec / dept / prefix,
                           dir_train / spec,
                           graphScale,
                        testDepartement,
                        dept,
                        methods,
                        prefix,
                        features_name)