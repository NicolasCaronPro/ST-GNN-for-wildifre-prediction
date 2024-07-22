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
parser.add_argument('-gs', '--GridSearch', type=str, help='GridSearch')
parser.add_argument('-bs', '--BayesSearch', type=str, help='BayesSearch')
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
spec = args.spec
resolution = args.resolution
doPCA = args.pca == 'True'
doKMEANS = args.KMEANS == 'True'
ncluster = int(args.ncluster)
doGridSearch = args.GridSearch ==  'True'
doBayesSearch = args.BayesSearch == 'True'

######################### Incorporate new features ##########################

dir_target = root_target / sinister / 'log'

geo = gpd.read_file('regions/regions.geojson')
geo = geo[geo['departement'].isin([name2str[dept] for dept in departements])].reset_index(drop=True)

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

############################ INIT ################################

X, Y, graphScale, prefix, prefix_train, pos_feature, _ = init(args, True)

############################# Training ###########################
trainCode = [name2int[dept] for dept in trainDepartements]

train_fet_num = select_train_features(trainFeatures, features, scale, pos_feature)

save_object(features, spec+'_features.pkl', dir_output)
save_object(trainFeatures, spec+'_trainFeatures.pkl', dir_output)

logger.info(pos_feature)
pos_feature, newshape = create_pos_feature(graphScale.scale, 6, trainFeatures)

X = X[:, np.asarray(train_fet_num)]
logger.info(np.unravel_index(np.nanargmax(X), X.shape))

if days_in_futur > 1:
    prefix += '_'+str(days_in_futur)
    prefix += '_'+futur_met

# Preprocess
trainDataset, valDataset, testDataset = preprocess(X=X, Y=Y, scaling=scaling, maxDate=maxDate,
                                                   trainDate=trainDate, trainDepartements=trainDepartements,
                                                   departements = departements,
                                                   ks=k_days, dir_output=dir_output, prefix=prefix, pos_feature=pos_feature,
                                                   days_in_futur=days_in_futur,
                                                   futur_met=futur_met)

if doPCA:
    prefix += '_pca'
    Xtrain = trainDataset[0]
    pca, components = train_pca(Xtrain, 0.99, dir_output / prefix, train_fet_num)
    trainDataset = (apply_pca(trainDataset[0], pca, components), trainDataset[1], train_fet_num)
    valDataset = (apply_pca(valDataset[0], pca, components), valDataset[1], train_fet_num)
    testDataset = (apply_pca(testDataset[0], pca, components), testDataset[1], train_fet_num)
    features_selected = np.arange(6, components + 6)
    trainFeatures = ['pca_'+str(i) for i in range(components)]
    pos_feature, _ = create_pos_feature(0, 6, trainFeatures)
    kmeansFeatures = trainFeatures
else:
    features_selected = np.arange(6, X.shape[1])

prefix = str(values_per_class)+'_'+str(k_days)+'_'+str(scale)+'_'+str(len(features_selected))+'_'+str(ncluster)
prefix_train = str(values_per_class)+'_'+str(k_days)+'_'+str(scale)+'_'+str(len(features_selected))+'_'+str(ncluster)

if doTrain:
    train_break_point(trainDataset[0], trainDataset[1], kmeansFeatures, dir_output / 'varOnValue' / prefix, pos_feature, scale, ncluster)

    Ytrain = trainDataset[1]
    Yval = valDataset[1]
    Ytest = testDataset[1]

    trainDataset = (trainDataset[0], apply_kmeans_class_on_target(trainDataset[0], Ytrain, dir_output / 'varOnValue' / prefix, True))
    valDataset = (valDataset[0], apply_kmeans_class_on_target(valDataset[0], Yval, dir_output / 'varOnValue' / prefix, True))
    testDataset = (testDataset[0], apply_kmeans_class_on_target(testDataset[0], Ytest, dir_output / 'varOnValue' / prefix, True))

    realVspredict(Y[:, -1], Y, -1, dir_output / prefix, 'raw')
    realVspredict(Y[:, -3], Y, -3, dir_output / prefix, 'class')

    sinister_distribution_in_class(Y[:, -3], Y, dir_output / prefix)

logger.info(f'Train dates are between : {allDates[int(np.min(trainDataset[0][:,4]))], allDates[int(np.max(trainDataset[0][:,4]))]}')
logger.info(f'Val dates are bewteen : {allDates[int(np.min(valDataset[0][:,4]))], allDates[int(np.max(valDataset[0][:,4]))]}')

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
            #('cal', cal, 'bin'),
            ('f1', f1, 'bin'),
            #('class', ck, 'class'),
            #('poisson', po, 'proba'),
            #('ca', ca, 'class'),
            #('bca', bca, 'class'),
            #('meac', meac, 'class'),
            ]
    
    for dept in departements:
        Xset = Xtest[(Xtest[:, 3] == name2int[dept])]
        Yset = Ytest[(Xtest[:, 3] == name2int[dept])]

        if Xset.shape[0] == 0:
            continue

        save_object(Xset, 'X'+'_'+dept+'_'+prefix+'.pkl', dir_output)
        save_object(Xset, 'Y_'+dept+'_'+prefix+'.pkl', dir_output)

        logger.info(f'{dept} test : {Xset.shape}, {allDates[int(np.min(Xset[:,4]))], allDates[int(np.max(Yset[:,4]))]}')

        test_break_points_model(Xset, Yset,
                           methods,
                           dept,
                           prefix_train,
                           dir_output / dept / prefix,
                           encoding,
                           scaling,
                           [dept],
                           scale,
                           train_dir,
                           sinister,
                           resolution)