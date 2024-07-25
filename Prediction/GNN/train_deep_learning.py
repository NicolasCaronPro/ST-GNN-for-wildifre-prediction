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
k_days = args.k_days # Size of the time series sequence use by DL models
days_in_futur = args.days_in_futur # The target time validation

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

####################### INIT ################################

X, Y, graphScale, prefix, features_name, features_name_2D = init(args, False)

############################# Training ###########################
trainCode = [name2int[dept] for dept in trainDepartements]

# Select train features
train_fet_num = select_train_features(trainFeatures, scale, features_name)

dir_output =  dir_output / spec

save_object(features, 'features.pkl', dir_output)
save_object(trainFeatures, 'trainFeatures.pkl', dir_output)
save_object(features_name, 'features_name.pkl', dir_output)

if days_in_futur > 1:
    prefix += '_'+str(days_in_futur)
    prefix += '_'+futur_met

logger.info(features_name)
features_name, newshape = get_features_name_list(graphScale.scale, 6, trainFeatures)
X = X[:, np.asarray(train_fet_num)]
logger.info(features_name)
logger.info(np.max(X[:,4]))
logger.info(np.unique(X[:,0]))
logger.info(np.nanmax(X))
logger.info(np.unravel_index(np.nanargmax(X), X.shape))

# Preprocess
trainDataset, valDataset, testDataset = preprocess(X=X, Y=Y, scaling=scaling, maxDate=maxDate,
                                                   trainDate=trainDate, trainDepartements=trainDepartements,
                                                   departements = departements,
                                                   ks=k_days, dir_output=dir_output, prefix=prefix, features_name=features_name,
                                                   days_in_futur=days_in_futur,
                                                   futur_met=futur_met)

realVspredict(testDataset[1][:, -1], testDataset[1], -1, dir_output / prefix, 'raw')

realVspredict(testDataset[1][:, -3], testDataset[1], -3, dir_output / prefix, 'class')

sinister_distribution_in_class(testDataset[1][:, -3], testDataset[1], dir_output / prefix)

features_selected = features_selection(doFet, trainDataset[0], trainDataset[1], dir_output / prefix, features_name, nbfeatures, scale)

prefix = f'{str(values_per_class)}_{str(k_days)}_{str(scale)}_{str(nbfeatures)}'

if days_in_futur > 1:
    prefix += '_'+str(days_in_futur)
    prefix += '_'+futur_met

if doPCA:
    prefix += '_pca'
    Xtrain = trainDataset[0]
    pca, components = train_pca(Xtrain, 0.99, dir_output / prefix, features_selected)
    trainDataset = (apply_pca(trainDataset[0], pca, components, features_selected), trainDataset[1])
    valDataset = (apply_pca(valDataset[0], pca, components, features_selected), valDataset[1])
    testDataset = (apply_pca(testDataset[0], pca, components, features_selected), testDataset[1])
    features_selected = np.arange(6, components + 6)
    trainFeatures = ['pca_'+str(i) for i in range(components)]
    features_name, _ = get_features_name_list(0, 6, trainFeatures)
    kmeansFeatures = trainFeatures

if doKMEANS:
    prefix += '_kmeans_'+str(ncluster)
    train_break_point(trainDataset[0], trainDataset[1], kmeansFeatures, dir_output / 'varOnValue' / prefix, features_name, scale, ncluster)
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

# Train
epochs = 10000
lr = 0.01
PATIENCE_CNT = 200
CHECKPOINT = 100

gnnModels = [
        ('GAT_risk', False, 'risk', False, autoRegression),
        #('DST-GCNCONV_risk', 'risk', False, False, autoRegression),
        #('ST-GATTCN_risk', 'risk', False, False, autoRegression),
        #('ST-GATCONV_risk', 'risk', False, False, autoRegression),
        #('ST-GCNCONV_risk', 'risk', False, False, autoRegression),
        #('ATGN_risk', False, 'risk', False, autoRegression),
        #('ST-GATLTSM_risk', 'risk', False, False, autoRegression),
        ('LTSM_risk', False, 'risk', False, autoRegression),

        #('GAT_nbsinister', False, 'nbsinister', False, autoRegression),
        #('DST-GCNCONV_nbsinister', 'nbsinister', False, False, autoRegression),
        #('ST-GATTCN_nbsinister', 'nbsinister', False, False, autoRegression),
        #('ST-GATCONV_nbsinister', 'nbsinister', False, False, autoRegression),
        #('ST-GCNCONV_nbsinister', 'nbsinister', False, False, autoRegression),
        #('ATGN_nbsinister', False, 'nbsinister', False, autoRegression),
        #('ST-GATLTSM_nbsinister', 'nbsinister', False, False, autoRegression),
        #('LTSM_nbsinister', False, 'nbsinister', False, autoRegression),

        #('DST-GCNCONV_bin', False, 'binary', False, autoRegression),
        #('ST-GATTCN_bin', False, 'binary', False, autoRegression),
        #('ST-GATCONV_bin', False, 'binary', False, autoRegression),
        #('ST-GCNCONV_bin', False, 'binary', False, autoRegression),
        #('ATGN_bin', False, 'binary', False, autoRegression),
        #('ST-GATLTSM_bin', False, 'binary', False, autoRegression),
        #('Zhang', False, 'binary', autoRegression)
        #('ConvLSTM', False, 'binary', autoRegression)
        ]

trainLoader = None
valLoader = None
last_bool = None

if doTrain:
    for model, use_temporal_as_edges, target_name, is_2D_model, autoRegression in gnnModels:
        if target_name != 'binary':
            criterion = weighted_rmse_loss
        else:
            criterion = weighted_cross_entropy
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        logger.info(f'Fitting model {model}')
        logger.info('Try loading loader')
        if not is_2D_model:
            trainLoader = read_object('trainloader_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
            valLoader = read_object('valLoader_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
        else:
            trainLoader = read_object('trainloader_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'_2D.pkl', dir_output)
            valLoader = read_object('valLoader_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'_2D.pkl', dir_output)

        if trainLoader is None or Rewrite:
            logger.info(f'Building loader, loading loader_{str(use_temporal_as_edges)}.pkl')
            last_bool = use_temporal_as_edges
            if not is_2D_model:
                trainLoader, valLoader, testLoader = train_val_data_loader(graph=graphScale, train=trainDataset,
                                                                        val=valDataset,
                                                                        test=testDataset,
                                                                batch_size=64, device=device,
                                                                use_temporal_as_edges = use_temporal_as_edges,
                                                                ks=k_days)
            
                save_object(trainLoader, 'trainloader_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
                save_object(valLoader, 'valLoader_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
            else:
                trainLoader, valLoader, testLoader = train_val_data_loader_2D(graph=graphScale,
                                                                train=trainDataset,
                                                                val=valDataset,
                                                                test=testDataset,
                                                                use_temporal_as_edges=use_temporal_as_edges,
                                                                batch_size=64,
                                                                device=device,
                                                                scaling=scaling,
                                                                features_name=features_name,
                                                                features_name_2D=features_name_2D,
                                                                ks=k_days,
                                                                shape=(shape2D[0], shape2D[1], newShape2D),
                                                                path=dir_output / '2D' / prefix / 'data')

                save_object(trainLoader, 'trainloader_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'_2D.pkl', dir_output)
                save_object(valLoader, 'valLoader_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'_2D.pkl', dir_output)

        train(trainLoader=trainLoader, valLoader=valLoader,
            testLoader=testLoader,
            scale=scale,
            optmize_feature = optimize_feature,
            PATIENCE_CNT=PATIENCE_CNT,
            CHECKPOINT=CHECKPOINT,
            epochs=epochs,
            features=features_selected,
            lr=lr,
            features_name=features_name,
            criterion=criterion,
            modelname=model,
            dir_output=dir_output / Path('check_'+scaling + '/' + prefix + '/' + model),
            target_name=target_name,
            autoRegression=autoRegression)

if doTest:

    dico_model = make_models(len(features_selected), 52, 0.03, 'relu')

    trainCode = [name2int[dept] for dept in trainDepartements]

    Xtrain, Ytrain = trainDataset

    Xtest, Ytest  = testDataset

    name_dir = nameExp + '/' + sinister + '/' + resolution + '/train' + '/'
    train_dir = Path(name_dir)

    name_dir = nameExp + '/' + sinister + '/' + resolution + '/test' + '/'
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

    gnnModels = [
        ('GAT_risk', False, 'risk', False, autoRegression),
        #('DST-GCNCONV_risk', False, 'risk', False, autoRegression),
        #('ST-GATTCN_risk', False, 'risk', False, autoRegression),
        #('ST-GATCONV_risk', False, 'risk', False, autoRegression),
        #('ST-GCNCONV_risk', False, 'risk', False, autoRegression),
        #('ATGN_risk', False, 'risk', False, autoRegression),
        #('ST-GATLTSM_risk', 'risk', False, False, autoRegression),
        ('LTSM_risk', False, 'risk', False, autoRegression),

        #('GAT_nbsinister', False, 'nbsinister', False, autoRegression),
        #('DST-GCNCONV_nbsinister', 'nbsinister', False, False, autoRegression),
        #('ST-GATTCN_nbsinister', 'nbsinister', False, False, autoRegression),
        #('ST-GATCONV_nbsinister', 'nbsinister', False, False, autoRegression),
        #('ST-GCNCONV_nbsinister', 'nbsinister', False, False, autoRegression),
        #('ATGN_nbsinister', False, 'nbsinister', False, autoRegression),
        #('ST-GATLTSM_nbsinister', 'nbsinister', False, False, autoRegression),
        #('LTSM_nbsinister', False, 'nbsinister', False, autoRegression),

        #('DST-GCNCONV_binary', False, 'binary', autoRegression),
        #('ST-GATTCN_binary', False, 'binary', autoRegression),
        #('ST-GATCONV_binary', False, 'binary', autoRegression),
        #('ST-GCNCONV_binary', False, 'binary', autoRegression),
        #('ATGN_binary', False, 'binary', autoRegression),
        #('ST-GATLTSM_binary', False, 'binary', autoRegression),
        #('Zhang_binary', False, 'binary', autoRegression)
        #('ConvLSTM_binary', False, 'binary', autoRegression)
    ]

    for dept in departements:
        Xset = Xtest[(Xtest[:, 3] == name2int[dept])]
        Yset = Ytest[(Xtest[:, 3] == name2int[dept])]

        if Xset.shape[0] == 0:
            continue

        save_object(Xset, 'X'+'_'+dept+'_'+prefix+'.pkl', dir_output)
        save_object(Xset, 'Y_'+dept+'_'+prefix+'.pkl', dir_output)

        logger.info(f'{dept} test : {Xset.shape}, {allDates[int(np.min(Xset[:,4]))], allDates[int(np.max(Yset[:,4]))]}')

        test_dl_model(graphScale, Xset, Yset, Xtrain, Ytrain,
                           methods,
                           dept,
                           features_name,
                           prefix,
                           prefix,
                           dummy,
                           gnnModels,
                           dir_output / spec / dept / prefix,
                           device,
                           k_days,
                           Rewrite, 
                           encoding,
                           scaling,
                           [dept],
                           train_dir / spec,
                           dico_model,
                           features_name_2D,
                           shape2D,
                           sinister,
                           resolution)