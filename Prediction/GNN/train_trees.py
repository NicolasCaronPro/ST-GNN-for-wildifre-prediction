from dataloader import *
from config import *
import geopandas as gpd
import argparse
from encoding import encode

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
parser.add_argument('-test', '--doTest', type=str, help='Launch test')
parser.add_argument('-train', '--doTrain', type=str, help='Launch train')
parser.add_argument('-r', '--resolution', type=str, help='Resolution of image')
parser.add_argument('-pca', '--pca', type=str, help='Apply PCA')

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

######################### Incorporate new features ##########################

newFeatures = []

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
if doPCA:
    prefix += '_pca'

if dummy:
    prefix += '_dummy'

########################### Create points ################################
if doPoint:

    check_and_create_path(dir_output)

    fp = pd.read_csv('sinister/'+sinister+'.csv')

    construct_non_point(fp, geo, maxDate, sinister, Path(nameExp))
    look_for_information(dir_target,
                         departements,
                         geo,
                         maxDate,
                         sinister,
                         Path(nameExp))

######################### Encoding ######################################

if doEncoder:
    encode(dir_target, trainDate, trainDepartements, dir_output / 'Encoder', resolution)

########################## Do Graph ######################################

if doGraph:
    logger.info('#################################')
    logger.info('#      Construct   Graph        #')
    logger.info('#################################')
    graphScale = construct_graph(scale,
                                 maxDist[scale],
                                 sinister,
                                 geo, nmax, k_days,
                                 dir_output,
                                 True,
                                 False,
                                 resolution)
    graphScale._create_predictor(minDate, maxDate, dir_output, sinister, resolution)
else:
    graphScale = read_object('graph_'+str(scale)+'.pkl', dir_output)

########################## Do Database ####################################

if doDatabase:
    logger.info('#####################################')
    logger.info('#      Construct   Database         #')
    logger.info('#####################################')
    if not dummy:
        name = 'points.csv'
        ps = pd.read_csv(Path(nameExp) / sinister / name)
        logger.info(f'{ps.date.min(), allDates[ps.date.min()]}')
        logger.info(f'{ps.date.max(), allDates[ps.date.max()]}')
        depts = [name2int[dept] for dept in departements]
        logger.info(ps.departement.unique())
        ps = ps[ps['departement'].isin(depts)].reset_index(drop=True)
        logger.info(ps.departement.unique())
        #ps['date'] = ps['date'] - 1
        ps = ps[ps['date'] > 0]
        logger.info(ps[ps['departement'] == 1].date.min())
    else:
        name = sinister+'.csv'
        fp = pd.read_csv(Path(nameExp) / sinister / name)
        name = 'non'+sinister+'csv'
        nfp = pd.read_csv(Path(nameExp) / sinister / name)
        ps = pd.concat((fp, nfp)).reset_index(drop=True)
        ps = ps.copy(deep=True)
        ps = ps[(ps['date'].isin(allDates)) & (ps['date'] >= '2017-06-12')]
        ps['date'] = [allDates.index(date) for date in ps.date if date in allDates]

    X, Y, pos_feature = construct_database(graphScale,
                                           ps, scale, k_days,
                                           departements,
                                            features, sinister,
                                           dir_output,
                                           dir_output,
                                           prefix,
                                           values_per_class,
                                           'train',
                                           resolution,
                                           trainDate)
    
    save_object(X, 'X_'+prefix+'.pkl', dir_output)
    save_object(Y, 'Y_'+prefix+'.pkl', dir_output)
else:
    X = read_object('X_'+prefix+'.pkl', dir_output)
    Y = read_object('Y_'+prefix+'.pkl', dir_output)
    pos_feature, newshape = create_pos_feature(graphScale.scale, 6, features)

    
realVspredict(Y[:, -1], Y, -1,
                    dir_output / prefix, 'raw')

realVspredict(Y[:, -3], Y, -3,
                    dir_output / prefix, 'class')

sinister_distribution_in_class(Y[:, -3], Y, dir_output / prefix)

if do2D:
    subnode = X[:,:6]
    pos_feature_2D, newShape2D = get_sub_nodes_feature_2D(graphScale,
                                                 subnode.shape[1],
                                                 X,
                                                 scaling,
                                                 pos_feature,
                                                 departements,
                                                 features,
                                                 sinister,
                                                 dir_output,
                                                 prefix)
else:
    subnode = X[:,:6]
    pos_feature_2D, newShape2D = create_pos_features_2D(subnode.shape[1], features)

prefix = str(values_per_class)+'_'+str(k_days)+'_'+str(scale)+'_'+str(nbfeatures)
prefix_train = str(values_per_class)+'_'+str(k_days)+'_'+str(scale)+'_'+str(nbfeatures)

# Add varying time features
if k_days > 0:
    trainFeatures += varying_time_variables
    features += varying_time_variables
    pos_feature, newShape = create_pos_feature(graphScale.scale, 6, features)
    if doDatabase:
        X = add_varying_time_features(X=X, features=varying_time_variables, newShape=newShape, pos_feature=pos_feature, ks=k_days)
        save_object(X, 'X_'+prefix+'.pkl', dir_output)

############################# Training ###########################
trainCode = [name2int[dept] for dept in trainDepartements]

# Select train features
train_fet_num = select_train_features(trainFeatures, features, scale, pos_feature)

save_object(features, spec+'_features.pkl', dir_output)
save_object(trainFeatures, spec+'_trainFeatures.pkl', dir_output)

if spec != '':
    prefix += '_'+spec
logger.info(pos_feature)
pos_feature, newshape = create_pos_feature(graphScale.scale, 6, trainFeatures)
X = X[:, np.asarray(train_fet_num)]
logger.info(pos_feature)
logger.info(np.max(X[:,4]))
logger.info(np.unique(X[:,0]))
logger.info(np.nanmax(X))
logger.info(np.unravel_index(np.nanargmax(X), X.shape))

# Preprocess
trainDataset, valDataset, testDataset = preprocess(X=X, Y=Y, scaling=scaling, maxDate=maxDate,
                                                   trainDate=trainDate, trainDepartements=trainDepartements,
                                                   ks=k_days, dir_output=dir_output, prefix=prefix)

features_selected = features_selection(doFet, trainDataset[0], trainDataset[1], dir_output, pos_feature, prefix, True, nbfeatures, scale)

if doPCA:
    Xtrain = trainDataset[0]
    pca, components = train_pca(Xtrain, 0.95, dir_output / prefix)
    trainDataset[0] = apply_pca(trainDataset[0], pca, components)
    valDataset[0] = apply_pca(valDataset[0], pca, components)
    testDataset[0] = apply_pca(testDataset[0], pca, components)
    features_selected = np.arange(6, components + 6)
    trainFeatures = ['pca_'+str(i) for i in range(components)] 
    pos_feature, _ = create_pos_feature(0, 6, trainFeatures)

logger.info(f'Train dates are between : {allDates[int(np.min(trainDataset[0][:,4]))], allDates[int(np.max(trainDataset[0][:,4]))]}')
logger.info(f'Val dates are bewteen : {allDates[int(np.min(valDataset[0][:,4]))], allDates[int(np.max(valDataset[0][:,4]))]}')

# Features selection

######################## Baseline ##############################
name = 'check_'+scaling + '/' + prefix + '/' + '/baseline'
trainCode = [name2int[dept] for dept in trainDepartements]

if doTrain:
    train_sklearn_api_model(trainDataset=trainDataset,
                            valDataset=valDataset,
                            testDataset=testDataset,
                            pos_feature=pos_feature,
                            graph=graphScale,
                            dir_output=dir_output / name,
                            dir_train = dir_output,
                            device='cpu',
                            binary=False,
                            scaling=scaling,
                            features=features_selected,
                            optimize_feature=optimize_feature,
                            departements=departements,
                            weight=True)

    train_sklearn_api_model(trainDataset=trainDataset,
                            valDataset=valDataset,
                            testDataset=testDataset,
                            graph=graphScale,
                            pos_feature=pos_feature,
                            dir_train = dir_output,
                            dir_output=dir_output / name,
                            device='cpu',
                            binary=True,
                            scaling=scaling,
                            optimize_feature=optimize_feature,
                            features=features_selected,
                            departements=departements,
                            weight=True)

    train_sklearn_api_model(trainDataset=trainDataset,
                            valDataset=valDataset,
                            testDataset=testDataset,
                            graph=graphScale,
                            pos_feature=pos_feature,
                            dir_train = dir_output,
                            dir_output=dir_output / name,
                            device='cpu',
                            binary=True,
                            scaling=scaling,
                            optimize_feature=optimize_feature,
                            features=features_selected,
                            departements=departements,
                            weight=False)

    train_sklearn_api_model(trainDataset=trainDataset,
                            valDataset=valDataset,
                            testDataset=testDataset,
                            graph=graphScale,
                            pos_feature=pos_feature,
                            dir_train = dir_output,
                            dir_output=dir_output / name,
                            device='cpu',
                            binary=False,
                            scaling=scaling,
                            optimize_feature=optimize_feature,
                            features=features_selected,
                            departements=departements,
                            weight=False)

if doTest:
    logger.info('############################# TEST ###############################')

    Xtrain = trainDataset[0]
    Ytrain = trainDataset[1]

    Xtest = testDataset[0]
    Ytest = testDataset[1]

    name_dir = nameExp + '/' + sinister + '/' + resolution + '/train' + '/'
    train_dir = Path(name_dir)

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
            ('cal', cal, 'bin'),
            ('f1', f1, 'bin'),
            ('class', ck, 'class'),
            ('poisson', po, 'proba'),
            ('ca', ca, 'class'),
            ('bca', bca, 'class'),
            ('meac', meac, 'class'),
            ]

    models = [
        ('xgboost', False),
        #('xgboost_optimize_feature', False),
        ('lightgbm', False),
        ('ngboost', False),
        ('xgboost_bin', True),
        ('lightgbm_bin', True),
        ('xgboost_bin_unweighted', True),
        ('lightgbm_bin_unweighted', True),
        ('xgboost_unweighted', False),
        ('lightgbm_unweighted', False),
        ('ngboost_unweighted', False),
        #('ngboost_bin', True)
        ]
    
    if sinister == 'firepoint':
        # 69 test
        testDepartement = ['departement-69-rhone']
        Xset = Xtest[(Xtest[:, 3] == 69) & (Xtest[:,4] >= allDates.index('2018-01-01')) & (Xtest[:,4] < allDates.index('2023-01-01'))]
        Yset = Ytest[(Xtest[:, 3] == 69) & (Xtest[:,4] >= allDates.index('2018-01-01')) & (Xtest[:,4] < allDates.index('2023-01-01'))]
        save_object(Xset, 'X69'+prefix+'.pkl', dir_output)
        save_object(Xset, 'Y69'+prefix+'.pkl', dir_output)
        logger.info(f'69 test {Xset.shape}')
        logger.info(f'{allDates[int(np.min(Xset[:,4]))], allDates[int(np.max(Yset[:,4]))]}')
        logger.info(f'{np.nanmax(Xset[:, 6:]), np.nanmin(Xset[:, 6:])}')

        test_sklearn_api_model(graphScale, Xset, Yset,
                           methods,
                           '69',
                           prefix,
                           dummy,
                           models,
                           dir_output / '69',
                           device,
                           k_days,
                           encoding,
                           scaling,
                           testDepartement,
                           train_dir,
                           sinister,
                           resolution)

        # 2023 test
        testDepartement = ['departement-01-ain', 'departement-25-doubs', 'departement-78-yvelines']
        Xset = Xtest[(Xtest[:, 3] != 69)]
        Yset = Ytest[(Xtest[:, 3] != 69)]
        save_object(Xset, 'X2023'+prefix+'.pkl', dir_output)
        save_object(Xset, 'Y2023'+prefix+'.pkl', dir_output)
        logger.info(f'2023 test {Xset.shape}')
        logger.info(f'{allDates[int(np.min(Xset[:,4]))], allDates[int(np.max(Yset[:,4]))]}')
        logger.info(f'{np.nanmax(Xset[:, 6:]), np.nanmin(Xset[:, 6:])}')

        test_sklearn_api_model(graphScale, Xset, Yset,
                           methods,
                           '2023',
                           prefix,
                           dummy,
                           models,
                           dir_output / '2023',
                           device,
                           k_days,
                           encoding,
                           scaling,
                           testDepartement,
                           train_dir,
                           sinister,
                           resolution)
         
    if sinister == 'inondation':
        # 69 test
        testDepartement = ['departement-69-rhone', 'departement-01-ain', 'departement-78-yvelines']
        Xset = Xtest[(Xtest[:, 3] != 25)]
        Yset = Ytest[(Xtest[:, 3] != 25)]

        test_sklearn_api_model(graphScale, Xset, Yset,
                           methods,
                           '69',
                           prefix,
                           dummy,
                           models,
                           dir_output / '69',
                           device,
                           k_days,
                           encoding,
                           scaling,
                           testDepartement,
                           train_dir,
                           sinister,
                           resolution)
        # 2023 test
        testDepartement = ['departement-25-doubs']
        Xset = Xtest[(Xtest[:, 3] == 25)]
        Yset = Ytest[(Xtest[:, 3] == 25)]

        test_sklearn_api_model(graphScale, Xset, Yset,
                           methods,
                           '2023',
                           prefix,
                           dummy,
                           models,
                           dir_output / '2023',
                           device,
                           k_days,
                           encoding,
                           scaling,
                           testDepartement,
                           train_dir,
                           sinister,
                           resolution)