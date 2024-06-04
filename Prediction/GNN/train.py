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
parser.add_argument('-t', '--doTest', type=str, help='Launch test')
parser.add_argument('-of', '--optimizeFeature', type=str, help='Number de Features')

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
optimize_feature = args.optimizeFeature == "True"
sinister = args.sinister
minPoint = args.nbpoint
nbfeatures = int(args.NbFeatures) if args.NbFeatures != 'all' else args.NbFeatures 
scale = int(args.scale)
spec = args.spec

######################### Incorporate new features ##########################

newFeatures = []

############################# GLOBAL VARIABLES #############################

dir_target = root_target / sinister / 'log'

geo = gpd.read_file('regions/regions.geojson')
geo = geo[geo['departement'].isin([name2str[dept] for dept in departements])].reset_index(drop=True)

doBaseline = True
name_dir = nameExp + '/' + sinister + '/' + 'train'
dir_output = Path(name_dir)
check_and_create_path(dir_output)

minDate = '2017-06-12' # Starting point

if minPoint == 'full':
    prefix = str(minPoint)+'_'+str(scale)
else:
    prefix = str(minPoint)+'_'+str(k_days)+'_'+str(scale)

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
                         fp,
                         maxDate,
                         sinister,
                         Path(nameExp),
                         minPoint)

######################### Encoding ######################################

if doEncoder:
    encode(dir_target, trainDate, trainDepartements, dir_output / 'Encoder')

########################## Do Graph ######################################

if doGraph:
    graphScale = construct_graph(scale,
                                 maxDist[scale],
                                 sinister,
                                 geo, nmax, k_days,
                                 dir_output,
                                 True)
    graphScale._create_predictor(minDate, maxDate, dir_output)
    doDtabase = True
else:
    graphScale = read_object('graph_'+str(scale)+'.pkl', dir_output)

########################## Do Database ####################################

if doDatabase:
    if not dummy:
        name = 'points'+str(minPoint)+'.csv'
        ps = pd.read_csv(Path(nameExp) / sinister / name)
    else:
        name = sinister+'.csv'
        fp = pd.read_csv(Path(nameExp) / sinister / name)
        name = 'non'+sinister+'csv'
        nfp = pd.read_csv(Path(nameExp) / sinister / name)
        ps = pd.concat((fp, nfp)).reset_index(drop=True)
        ps = ps.copy(deep=True)
        ps = ps[ps['date'].isin(allDates)]
        ps = ps[ps['date'] < maxDate]
        ps['date'] = [allDates.index(date) - 1 for date in ps.date]

    X, Y, pos_feature = construct_database(graphScale,
                                           ps, scale, k_days,
                                           trainDepartements,
                                           features, sinister,
                                           dir_output,
                                           dir_output,
                                           prefix,
                                           'train')
    
    save_object(X, 'X_'+prefix+'.pkl', dir_output)
    save_object(Y, 'Y_'+prefix+'.pkl', dir_output)

else:
    X = read_object('X_'+prefix+'.pkl', dir_output)
    Y = read_object('Y_'+prefix+'.pkl', dir_output)
    pos_feature, newshape = create_pos_feature(graphScale, 6, features)

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

# Select train features
train_fet_num = [0,1,2,3,4,5]
for fet in features:
    if fet in trainFeatures:
        coef = 4 if scale > 0 else 1
        if fet == 'Calendar' or fet == 'Calendar_mean':
            maxi = len(calendar_variables)
        elif fet == 'air' or fet == 'air_mean':
            maxi = len(air_variables)
        elif fet == 'sentinel':
            maxi = coef * len(sentinel_variables)
        elif fet == 'Geo':
            maxi = len(geo_variables)
        elif fet == 'foret':
            maxi = coef * len(foret_variables)
        elif fet == 'highway':
            maxi = coef * len(osmnx_variables)
        elif fet == 'landcover':
            maxi = coef * len(landcover_variables)
        elif fet == 'dynamicWorld':
            maxi = coef * len(dynamic_world_variables)
        elif fet == 'vigicrues':
            maxi = coef * len(vigicrues_variables)
        elif fet == 'nappes':
            maxi = coef * len(nappes_variables)
        elif fet == 'AutoRegressionReg':
            maxi = len(auto_regression_variable_reg)
        elif fet == 'AutoRegressionBin':
            maxi = len(auto_regression_variable_bin)
        elif fet in varying_time_variables:
            maxi = 1
        else:
            maxi = coef
        train_fet_num += list(np.arange(pos_feature[fet], pos_feature[fet] + maxi))

save_object(features, 'features.pkl', dir_output)        
save_object(trainFeatures, 'trainFeatures.pkl', dir_output)        
prefix = str(minPoint)+'_'+str(k_days)+'_'+str(scale)+'_'+str(nbfeatures)
if spec != '':
    prefix += '_'+spec
pos_feature, newshape = create_pos_feature(graphScale, 6, trainFeatures)
X = X[:, np.asarray(train_fet_num)]

logger.info(X.shape)

############################# Training ###########################

# Preprocess
Xset, Yset = preprocess(X=X, Y=Y, scaling=scaling, maxDate=trainDate, ks=k_days)

# Features selection
features_selected = features_selection(doFet, Xset, Yset, dir_output, pos_feature, prefix, False, nbfeatures)

# Make models
dico_model = make_models(features_selected.shape[0], 0.03, 'relu')

# Train
criterion = weighted_rmse_loss
epochs = 10000
lr = 0.01
PATIENCE_CNT = 200
CHECKPOINT = 100

gnnModels = [
        #('GAT', False, False),
        ('ST-GATTCN', False, False),
        ('ST-GATCONV', False, False),
        ('ST-GCNCONV', False, False),
        ('ATGN', False, False),
        ('ST-GATLTSM', False, False),
        #('Zhang', False, True)
        #('ConvLSTM', False, True)
        ]

trainLoader = None
valLoader = None
last_bool = None

for model, use_temporal_as_edges, is_2D_model in gnnModels:
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
            trainLoader, valLoader, testLoader = train_val_data_loader(graph=graphScale, Xset=Xset, Yset=Yset,
                                                            trainDate=trainDate,
                                                            maxDate=maxDate,
                                                            batch_size=64, device=device,
                                                            use_temporal_as_edges = use_temporal_as_edges,
                                                            ks=k_days)
        
            save_object(trainLoader, 'trainloader_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
            save_object(valLoader, 'valLoader_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
        else:
            trainLoader, valLoader, testLoader = train_val_data_loader_2D(graph=graphScale,
                                                              Xset=Xset, Yset=Yset,
                                                              trainDate=trainDate,
                                                              maxDate=maxDate,
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

    optimize_str = 'optimize_features' if optimize_feature else 'no_optimize_features'
    logger.info(dico_model[model])
    train(trainLoader=trainLoader, valLoader=valLoader,
          testLoader=testLoader,
          optmize_feature = optimize_feature,
        PATIENCE_CNT=PATIENCE_CNT,
        CHECKPOINT=CHECKPOINT,
        epochs=epochs,
        features=features_selected,
        lr=lr,
        criterion=criterion,
        model=dico_model[model],
        dir_output=dir_output / Path('check_'+scaling + '_' + optimize_str + '/' + prefix + '/' + model))

if doTest:
    trainCode = [name2int[dept] for dept in trainDepartements]

    Xtrain = Xset[(Xset[:,4] < allDates.index(trainDate)) & (np.isin(Xset[:, 3], trainCode))]
    Ytrain = Yset[(Xset[:,4] < allDates.index(trainDate)) & (np.isin(Xset[:, 3], trainCode))]

    Xtest = Xset[(Xset[:,4] >= allDates.index(maxDate)) | ((~np.isin(Xset[:, 3], trainCode)) & (Xset[:,4] < allDates.index(maxDate)))]
    Ytest = Yset[(Xset[:,4] >= allDates.index(maxDate)) | ((~np.isin(Xset[:, 3], trainCode)) & (Xset[:,4] < allDates.index(maxDate)))]

    name_dir = nameExp + '/' + sinister + '/' + 'train'
    train_dir = Path(name_dir)

    name_dir = nameExp + '/' + sinister + '/' + 'test'
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
        #('GAT', False, False, False),
        ('ST-GATTCN', False, False, False),
        ('ST-GATCONV', False, False, False),
        ('ATGN', False, False, False),
        ('ST-GATLTSM', False, False, False),
        ('ST-GCNCONV', False, False, False),
        #('Zhang', False, False, True),
        #('ConvLSTM', False, False, True)
        ]
    
    # 69 test
    testDepartement = ['departement-69-rhone']
    Xset = Xtest[(Xtest[:, 3] == 69) & (Xtest[:, 4] < allDates.index(maxDate))]
    Yset = Ytest[(Xtest[:, 3] == 69) & (Xtest[:, 4] < allDates.index(maxDate))]

    test_sklearn_api_model(graphScale, Xset, Yset, Xtrain, Ytrain,
                        methods,
                        '69',
                        pos_feature,
                        prefix,
                        prefix,
                        dummy,
                        models,
                        dir_output,
                        device,
                        k_days,
                        Rewrite, 
                        encoding,
                        scaling,
                        testDepartement,
                        train_dir)
    
    # 2023 test
    testDepartement = ['departement-01-ain', 'departement-25-doubs', 'departement-78-yvelines']
    Xset = Xtest[(Xtest[:, 3] != 69) & (Xtest[:, 4] >= allDates.index(maxDate))]
    Yset = Ytest[(Xtest[:, 3] != 69) & (Xtest[:, 4] >= allDates.index(maxDate))]
    
    test_sklearn_api_model(graphScale, Xset, Yset, Xtrain, Ytrain,
                        methods,
                        '2023',
                        pos_feature,
                        prefix,
                        prefix,
                        dummy,
                        models,
                        dir_output,
                        device,
                        k_days,
                        Rewrite, 
                        encoding,
                        scaling,
                        testDepartement,
                        train_dir)