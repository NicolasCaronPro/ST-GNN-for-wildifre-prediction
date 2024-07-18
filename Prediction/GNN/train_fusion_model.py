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
parser.add_argument('-scT', '--scaleTemporal', type=str, help='Scale of temporal')
parser.add_argument('-scS', '--scaleSpatial', type=str, help='Scale of spatial')
parser.add_argument('-sp', '--spec', type=str, help='spec')
parser.add_argument('-nf', '--NbFeatures', type=str, help='Number de Features')
parser.add_argument('-of', '--optimizeFeature', type=str, help='Launch test')
parser.add_argument('-test', '--doTest', type=str, help='Launch test')
parser.add_argument('-train', '--doTrain', type=str, help='Launch train')
parser.add_argument('-r', '--resolution', type=str, help='Resolution of image')
parser.add_argument('-pca', '--pca', type=str, help='Apply PCA')
parser.add_argument('-kmeans', '--KMEANS', type=str, help='Apply kmeans preprocessing')
parser.add_argument('-ncluster', '--ncluster', type=str, help='Number of cluster for kmeans')

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
scaleTemporal = int(args.scaleTemporal)
scaleSpatial = int(args.scaleSpatial)
spec = args.spec
resolution = args.resolution
doPCA = args.pca == 'True'
doKMEANS = args.KMEANS == 'True'
ncluster = int(args.ncluster)

############################# GLOBAL VARIABLES #############################

dir_target = root_target / sinister / 'log' / resolution

geo = gpd.read_file('regions/regions.geojson')
geo = geo[geo['departement'].isin(departements)].reset_index(drop=True)

name_dir = nameExp + '/' + sinister + '/' + resolution + '/' + 'train' +  '/'
dir_output = Path(name_dir)
check_and_create_path(dir_output)

minDate = '2017-06-12' # Starting point

if values_per_class == 'full':
    prefix = str(values_per_class)+'_'+str(scale)
else:
    prefix = str(values_per_class)+'_'+str(k_days)+'_'+str(scale)+'_'+str(nbfeatures)

if dummy:
    prefix += '_dummy'

autoRegression = 'AutoRegressionReg' in trainFeatures
if spec == '' and autoRegression:
    spec = 'AutoRegressionReg'

############################# INIT ########################################

X, Y, graphScale, prefix, prefix_train, pos_feature = init(args)

############################# Training ###########################
    
# Daily model
dir_train = Path('final/firepoint/2x2/train')
trainFeatures = read_object('_trainFeatures.pkl', dir_train)
X_daily = read_object(f'X_full_{scaleSpatial}.pkl', dir_train)
Y_daily = read_object(f'Y_full_{scaleSpatial}.pkl', dir_train)
args1 = {'dir_train' : dir_train,
            'dir_model':  dir_train / f'check_z-score/full_0_{scaleSpatial}_100' / 'baseline',
            'autoRegression' : autoRegression,
            'pos_feature' : create_pos_feature(scaleSpatial, 6, trainFeatures)[0],
            'model_name': 'xgboost_features',
            'model_type' : 'traditionnal',
            'scale' : scaleSpatial,
            'isBin': False,
        'graph' : 'graph_'+str(scaleSpatial)+'.pkl',
        }
graph1 = read_object(args1['graph'], dir_train)

# Localized model
dir_train = Path('final/firepoint/2x2/train')
trainFeatures = read_object('_trainFeatures.pkl', dir_train)
X_localized = read_object(f'X_full_{scale}.pkl', dir_train)
Y_localized = read_object(f'Y_full_{scale}.pkl', dir_train)
args2 = {'train_dir' : dir_train,
            'dir_model': dir_train / f'check_z-score/full_0_{scale}_100_{scaleTemporal}_mean' / 'baseline',
            'autoRegression' : autoRegression,
            'pos_feature' : create_pos_feature(scale, 6, trainFeatures)[0],
            'model_name': 'xgboost_features',
            'model_type' : 'traditionnal',
            'scale' : scale,
            'isBin': False,
        'graph' : 'graph_'+str(scale)+'.pkl',
        }

trainCode = [name2int[dept] for dept in trainDepartements]

train_fet_num = select_train_features(trainFeatures, features, scaleSpatial, create_pos_feature(scaleSpatial, 6, features)[0])
X_daily = X_daily[:, np.asarray(train_fet_num)]

train_fet_num = select_train_features(trainFeatures, features, scale, create_pos_feature(scale, 6, features)[0])
X_localized = X_localized[:, np.asarray(train_fet_num)]

# Add spatial and temporal prediction in X
trainFeatures += ['temporal_prediction', 'spatial_prediction']
features += ['temporal_prediction', 'spatial_prediction']
pos_feature, newshape = create_pos_feature(scale, 6, features)
X = add_temporal_spatial_prediction(X, Y_localized, Y_daily, graph1, pos_feature)
train_fet_num = select_train_features(trainFeatures, features, scale, pos_feature)
pos_feature, newshape = create_pos_feature(scale, 6, trainFeatures)
X = X[:, np.asarray(train_fet_num)]

if spec != '':
    prefix += '_'+spec

# Preprocess
trainDataset, valDataset, testDataset = preprocess(X=X, Y=Y, scaling=scaling, maxDate=maxDate,
                                                trainDate=trainDate, trainDepartements=trainDepartements,
                                                departements = departements,
                                                ks=k_days, dir_output=dir_output, prefix=prefix, pos_feature=pos_feature,
                                                days_in_futur=0,
                                                futur_met=futur_met)

_, _, testDatasetDaily = preprocess(X=X_daily, Y=Y_daily, scaling=scaling, maxDate=maxDate,
                                                trainDate=trainDate, trainDepartements=trainDepartements,
                                                departements = departements,
                                                ks=k_days, dir_output=dir_output, prefix=prefix, pos_feature=args1['pos_feature'],
                                                days_in_futur=0,
                                                futur_met=futur_met)

_, _, testDatasetLocalized = preprocess(X=X_localized, Y=Y_localized, scaling=scaling, maxDate=maxDate,
                                                trainDate=trainDate, trainDepartements=trainDepartements,
                                                departements = departements,
                                                ks=k_days, dir_output=dir_output, prefix=prefix, pos_feature=args2['pos_feature'],
                                                days_in_futur=scaleTemporal,
                                                futur_met=futur_met)

prefix += '_fusion'

features_selected = features_selection(doFet, trainDataset[0], trainDataset[1], dir_output, pos_feature, spec, nbfeatures, scale)

if doTrain:
    name = 'check_'+scaling + '/' + prefix + '/' + '/baseline'
    # Linear combination
    train_linear(train_dataset=trainDataset, test_dataset=testDataset,
                 features=np.asarray([pos_feature['temporal_prediction'], pos_feature['spatial_prediction']]),
                 scale=scale, pos_feature=pos_feature, dir_output=dir_output / name)

    # Simple combination
    train_mean(train_dataset=trainDataset, test_dataset=testDataset,
                 features=np.asarray([pos_feature['temporal_prediction'], pos_feature['spatial_prediction']]),
                 scale=scale, pos_feature=pos_feature, dir_output=dir_output / name)

    # Add features
    train_sklearn_api_model(trainDataset=trainDataset, testDataset=testDataset, valDataset=valDataset,
                            dir_output=dir_output / name, device=device, features=features_selected,
                            scale=scale, pos_feature=pos_feature, autoRegression=autoRegression)

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
            ('cal', cal, 'bin'),
            ('f1', f1, 'bin'),
            ('class', ck, 'class'),
            ('poisson', po, 'proba'),
            ('ca', ca, 'class'),
            ('bca', bca, 'class'),
            ('meac', meac, 'class'),
            ]
    
    models = [
        ('linear', False, autoRegression),
        ('mean', False, autoRegression),
        ('xgboost_features', False, autoRegression),
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
                           dir_output / dept / prefix,
                           graphScale,
                        testDepartement,
                        dept,
                        methods,
                        prefix,
                        pos_feature,)