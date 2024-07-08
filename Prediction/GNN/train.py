from dataloader import *
from graph_structure import *
import geopandas as gpd
from construct import *
from config import *
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
parser.add_argument('-p', '--point', type=str, help='Construct n point')
parser.add_argument('-np', '--nbpoint', type=str, help='Number of point')
parser.add_argument('-d', '--database', type=str, help='Do database')
parser.add_argument('-g', '--graph', type=str, help='Construct graph')
parser.add_argument('-mxd', '--maxDate', type=str, help='Limit train and validation date')
parser.add_argument('-mxdv', '--trainDate', type=str, help='Limit training date')
parser.add_argument('-f', '--featuresSelection', type=str, help='Do features selection')
parser.add_argument('-dd', '--database2D', type=str, help='Do 2D database')
parser.add_argument('-sc', '--scale', type=str, help='Scale')
parser.add_argument('-sp', '--spec', type=str, help='spec')
parser.add_argument('-nf', '--NbFeatures', type=str, help='Nombur de Features')
parser.add_argument('-test', '--doTest', type=str, help='Launch test')
parser.add_argument('-train', '--doTrain', type=str, help='Launch train')
parser.add_argument('-of', '--optimizeFeature', type=str, help='Number de Features')
parser.add_argument('-r', '--resolution', type=str, help='Resolution')
parser.add_argument('-pca', '--pca', type=str, help='Apply PCA')
parser.add_argument('-ncluster', '--ncluster', type=str, help='Number of cluster for kmeans')
parser.add_argument('-kmeansTest', '--kmeansTest', type=str, help='Apply kmeans on test set')

args = parser.parse_args()

# Input config
nameExp = args.name
maxDate = args.maxDate
trainDate = args.trainDate
doEncoder = args.encoder == 'True'
doPoint = args.point == "True"
doGraph = args.graph == "True"
doDatabase = args.database == "True"
do2D = args.database2D == "True"
doFet = args.featuresSelection == "True"
doTest = args.doTest == "True"
doTrain = args.doTrain == "True"
optimize_feature = args.optimizeFeature == "True"
sinister = args.sinister
values_per_class = args.nbpoint
nbfeatures = int(args.NbFeatures) if args.NbFeatures != 'all' else args.NbFeatures 
scale = int(args.scale)
spec = args.spec
doPCA = args.pca == 'True'
resolution = args.resolution
doKMEANS = args.KMEANS == 'True'
doTestKMEANS = args.KMEANS == 'True'
ncluster = int(args.ncluster)

######################### Incorporate new features ##########################

newFeatures = []

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
if doPCA:
    prefix += '_pca'

if dummy:
    prefix += '_dummy'

########################### Create points ################################
if doPoint:

    check_and_create_path(dir_output)

    regions = gpd.read_file('regions/regions.geojson')
    fp = pd.read_csv('sinister/'+sinister+'.csv')

    construct_non_point(fp, regions, maxDate, sinister, Path(nameExp))
    look_for_information(dir_target,
                         departements,
                         regions,
                         maxDate,
                         sinister,
                         Path(nameExp))

######################### Encoding ######################################

if doEncoder:
    encode(dir_target, trainDate, trainDepartements, dir_output / 'Encoder')

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
        name = 'points'+str(values_per_class)+'.csv'
        ps = pd.read_csv(Path(nameExp) / sinister / name)
        logger.info(ps.date.min())
        logger.info(ps.date.max())
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

if len(newFeatures) > 0:
    X = X[:, :pos_feature['Geo'] + 1]
    subnode = X[:,:6]
    XnewFeatures, _ = get_sub_nodes_feature(graphScale, subnode, trainDepartements, newFeatures, dir_output)
    newX = np.empty((X.shape[0], X.shape[1] + XnewFeatures.shape[1] - 6))
    newX[:,:X.shape[1]] = X
    newX[:, X.shape[1]:] = XnewFeatures[:,6:]
    X = newX
    features += newFeatures
    logger.info(X.shape)
    save_object(X, 'X_'+prefix+'.pkl', dir_output)

if do2D:
    subnode = X[:,:6]
    pos_feature_2D, newShape2D = get_sub_nodes_feature_2D(graphScale,
                                                 subnode.shape[1],
                                                 X,
                                                 scaling,
                                                 pos_feature,
                                                 trainDepartements,
                                                 features,
                                                 sinister,
                                                 dir_output,
                                                 prefix)
    
else:
    subnode = X[:,:6]
    pos_feature_2D, newShape2D = create_pos_features_2D(subnode.shape[1], features)

############################# Training ###########################

trainCode = [name2int[dept] for dept in trainDepartements]

Xtrain, Ytrain = (X[(X[:,4] < allDates.index(trainDate)) & (np.isin(X[:,0], trainCode))],
                      Y[(X[:,4] < allDates.index(trainDate)) & (np.isin(X[:,0], trainCode))])

save_object(Xtrain, 'Xtrain_'+prefix+'.pkl', dir_output)
save_object(Ytrain, 'Ytrain_'+prefix+'.pkl', dir_output)

train_fet_num = select_train_features(trainFeatures, features, scale, pos_feature)

save_object(features, spec+'_features.pkl', dir_output)        
save_object(trainFeatures, spec+'_trainFeatures.pkl', dir_output)        
prefix = str(values_per_class)+'_'+str(k_days)+'_'+str(scale)+'_'+str(nbfeatures)
if spec != '':
    prefix += '_'+spec
pos_feature, newshape = create_pos_feature(graphScale.scale, 6, trainFeatures)
X = X[:, np.asarray(train_fet_num)]

logger.info(X.shape)

# Preprocess
trainDataset, valDataset, testDataset = preprocess(X=X, Y=Y, scaling=scaling, maxDate=maxDate,
                                                   trainDate=trainDate, trainDepartements=trainDepartements,
                                                   ks=k_days, dir_output=dir_output, prefix=prefix)

# Features selection
features_selected = features_selection(doFet, trainDataset[0], trainDataset[1], dir_output, pos_feature, prefix, False, nbfeatures, scale)

prefix = str(values_per_class)+'_'+str(k_days)+'_'+str(scale)+'_'+str(nbfeatures)
prefix_train = str(values_per_class)+'_'+str(k_days)+'_'+str(scale)+'_'+str(nbfeatures)

if doPCA:
    prefix += '_pca'
    prefix_train += '_pca'
    Xtrain = trainDataset[0]
    pca, components = train_pca(Xtrain, 0.99, dir_output / prefix, features_selected)
    trainDataset = (apply_pca(trainDataset[0], pca, components, features_selected), trainDataset[1])
    valDataset = (apply_pca(valDataset[0], pca, components, features_selected), valDataset[1])
    testDataset = (apply_pca(testDataset[0], pca, components, features_selected), testDataset[1])
    features_selected = np.arange(6, components + 6)
    trainFeatures = ['pca_'+str(i) for i in range(components)]
    pos_feature, _ = create_pos_feature(0, 6, trainFeatures)
    kmeansFeatures = trainFeatures

if doKMEANS:
    prefix += '_kmeans_'+str(ncluster)
    prefix_train += '_kmeans_'+str(ncluster)
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

# Train
epochs = 10000
lr = 0.01
PATIENCE_CNT = 200
CHECKPOINT = 100

gnnModels = [
        #('GAT', False, False),
        #('DST-GCNCONV', False, False, False),
        #('ST-GATTCN', False, False, False),
        #('ST-GATCONV', False, False, False),
        #('ST-GCNCONV', False, False, False),
        #('ATGN', False, False, False),
        #('ST-GATLTSM', False, False, False),
        ('DST-GCNCONV_bin', False, True, False),
        ('ST-GATTCN_bin', False, True, False),
        ('ST-GATCONV_bin', False, True, False),
        ('ST-GCNCONV_bin', False, True, False),
        ('ATGN_bin', False, True, False),
        ('ST-GATLTSM_bin', False, True, False),
        #('Zhang', False, True)
        #('ConvLSTM', False, True)
        ]

trainLoader = None
valLoader = None
last_bool = None

if doTrain:
    for model, use_temporal_as_edges, isBin, is_2D_model in gnnModels:
        if not isBin:
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
                                                                pos_feature=pos_feature,
                                                                pos_feature_2D=pos_feature_2D,
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
            pos_feature=pos_feature,
            criterion=criterion,
            modelname=model,
            dir_output=dir_output / Path('check_'+scaling + '/' + prefix + '/' + model),
            binary=isBin)

if doTest:

    dico_model = make_models(len(features_selected), 52, 0.03, 'relu')

    trainCode = [name2int[dept] for dept in trainDepartements]

    Xtrain, Ytrain = trainDataset

    Xtest, Ytest  = testDataset

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

    gnnModels = [
        #('GAT', False, False, False),
        ('DST-GCNCONV', False, False, False),
        ('ST-GATTCN', False, False, False),
        ('ST-GATCONV', False, False, False),
        ('ATGN', False, False, False),
        ('ST-GATLTSM', False, False, False),
        ('ST-GCNCONV', False, False, False),
        ('DST-GCNCONV_bin', False, False, False),
        ('ST-GATTCN_bin', False, False, False),
        ('ST-GATCONV_bin', False, False, False),
        ('ATGN_bin', False, False, False),
        ('ST-GATLTSM_bin', False, False, False),
        ('ST-GCNCONV_bin', False, False, False),
        #('Zhang', False, False, True),
    ]

    # 69 test
    testDepartement = ['departement-69-rhone']
    Xset = Xtest[(Xtest[:, 3] == 69)]
    Yset = Ytest[(Xtest[:, 3] == 69)]

    test_dl_model(graphScale, Xset, Yset, Xtrain, Ytrain,
                           methods,
                           '69',
                           pos_feature,
                           prefix,
                           prefix,
                           dummy,
                           gnnModels,
                           dir_output / '69',
                           device,
                           k_days,
                           Rewrite, 
                           encoding,
                           scaling,
                           testDepartement,
                           train_dir,
                           dico_model,
                           pos_feature_2D,
                           shape2D,
                           sinister,
                           resolution)
    
    # 2023 test
    testDepartement = ['departement-01-ain', 'departement-25-doubs', 'departement-78-yvelines']
    Xset = Xtest[(Xtest[:, 3] != 69)]
    Yset = Ytest[(Xtest[:, 3] != 69)]
    
    test_dl_model(graphScale, Xset, Yset, Xtrain, Ytrain,
                           methods,
                           '2023',
                           pos_feature,
                           prefix,
                           prefix,
                           dummy,
                           gnnModels,
                           dir_output / '2023',
                           device,
                           k_days,
                           Rewrite, 
                           encoding,
                           scaling,
                           testDepartement,
                           train_dir,
                           dico_model,
                           pos_feature_2D,
                           shape2D,
                           sinister,
                           resolution)