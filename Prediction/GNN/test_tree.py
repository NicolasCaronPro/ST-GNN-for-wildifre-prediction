from graph_structure import *
from features import *
from dataloader import *
from construct import *
from models.loss import *
import argparse

########################### Input Arg ######################################

parser = argparse.ArgumentParser(
    prog='Test',
)

parser.add_argument('-n', '--name', type=str, help='Name of the experiment')
parser.add_argument('-s', '--sinister', type=str, help='Sinister type')
parser.add_argument('-d', '--database', type=str, help='Do database')
parser.add_argument('-g', '--graph', type=str, help='Construct graph')
parser.add_argument('-sc', '--scale', type=str, help='Scale')
parser.add_argument('-dd', '--database2D', type=str, help='Do 2D database')
parser.add_argument('-sp', '--spec', type=str, help='spec')
parser.add_argument('-np', '--nbpoint', type=str, help='Number of point')

args = parser.parse_args()

# Input config
nameExp = args.name
doGraph = args.graph == "True"
doDatabase = args.database == "True"
do2D = args.database2D == "True"
"""doGraph = True
doDatabase = True
do2D = False"""
scale = int(args.scale)
sinister = args.sinister
spec = args.spec
minPoint = args.nbpoint

newFeatures = []

############### Global variable ###############

scaleTrain = scale

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
            ('meac', meac, 'class')
           ]

geo = gpd.read_file('regions/regions.geojson')

models = [
        #('GAT', False, False, False),
        #('ST-GATTCN', False, False, False),
        #('ST-GATCONV', False, False, False),
        #('ATGN', False, False, False),
        #('ST-GATLTSM', False, False, False),
        #('ST-GCNCONV', False, False, False),
        #('Zhang', False, False, True),
        #('ConvLSTM', False, False, True)
        ]

tradiModels = [
    ('xgboost', False),
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

metrics = {}

############################# Pipelines ##############################

def test(testname, testDate, pss, geo, testDepartement, dir_output, features, doDatabase, trainFeatures):
    logger.info('##############################################################')
    logger.info(f'                        {testname}                            ')
    logger.info('##############################################################')
    
    if minPoint == 'full':
        prefix_train = str(minPoint)+'_'+str(scaleTrain)
    else:
        prefix_train = str(minPoint)+'_'+str(k_days)+'_'+str(scaleTrain)

    prefix = str(scale)

    if dummy:
        prefix_train += '_dummy'

    ##################### Construct graph test ##############################
    dir_output = dir_output / testname
    if doGraph:
        graphScale = construct_graph(scale, maxDist[scale], sinister, geo, nmax, k_days, dir_output, False)
        #graphScale._create_predictor('2017-06-12', '2023-09-10', dir_output, sinister)
        doDatabase = True
    else:
        graphScale = read_object('graph_'+str(scale)+'.pkl', dir_output)

    ########################## Construct database #############################
    if doDatabase:
        pss = pss[pss['departement'].isin([name2str[dep] for dep in testDepartement])].reset_index(drop=True)
        if testDate is not None:
            iterables = [testDate, list(pss.latitude)]
            ps = pd.DataFrame(index=pd.MultiIndex.from_product(iterables=iterables,  names=['date', 'latitude'])).reset_index()

            iterables = [testDate, list(pss.longitude)]
            ps2 = pd.DataFrame(index=pd.MultiIndex.from_product(iterables=iterables,  names=['date', 'longitude'])).reset_index()
            ps['longitude'] = ps2['longitude']

            del ps2
        else:
            ps = pss

        ps['date'] = [allDates.index(date) - 1 for date in ps.date]
        ps = ps.sort_values(by='date')

        X, Y, pos_feature = construct_database(graphScale, ps, scale, k_days, departements,
                                               features, sinister, dir_output, train_dir, prefix, 'test')
        
        save_object(Y, 'Y_'+prefix+'.pkl', dir_output)
        save_object(X, 'X_'+prefix+'.pkl', dir_output)
    else:
        X = read_object('X_'+prefix+'.pkl', dir_output)
        Y = read_object('Y_'+prefix+'.pkl', dir_output)

        graphScale._info_on_graph(X, dir_output)

    if len(newFeatures) > 0:
        logger.info(X.shape)
        subnode = X[:,:6]
        XnewFeatures, _ = get_sub_nodes_feature(graphScale, subnode, testDepartement, newFeatures, sinister, dir_output)
        newX = np.empty((X.shape[0], X.shape[1] + XnewFeatures.shape[1] - 6))
        newX[:,:X.shape[1]] = X
        newX[:, X.shape[1]:] = XnewFeatures[:,6:]
        X = newX
        features += newFeatures
        logger.info(X.shape)
        save_object(X, 'X_'+prefix+'.pkl', dir_output)

    pos_feature, _ = create_pos_feature(graphScale, 6, features)

    if do2D:
        subnode = X[:,:6]
        pos_feature_2D, newShape2D = get_sub_nodes_feature_2D(graphScale,
                                                 subnode.shape[1],
                                                 X,
                                                 scaling,
                                                 pos_feature,
                                                 testDepartement,
                                                 features,
                                                 dir_output,
                                                 prefix)
    else:
        subnode = X[:,:6]
        pos_feature_2D, newShape2D = create_pos_features_2D(subnode.shape[1], features)

    Xtrain = read_object('X_'+prefix_train+'.pkl', train_dir)
    Ytrain = read_object('Y_'+prefix_train+'.pkl', train_dir)
    if Xtrain is None:
        logger.info(f'Fuck it')
        return

    # Add varying time features
    if k_days > 0:
        trainFeatures_ = trainFeatures + varying_time_variables
        features_ = features + varying_time_variables
        pos_feature, newShape = create_pos_feature(graphScale, 6, features_)
        if doDatabase:
            X = add_varying_time_features(X=X, features=varying_time_variables, newShape=newShape, pos_feature=pos_feature, ks=k_days)
            save_object(X, 'X_'+prefix+'.pkl', dir_output)
    else:
        trainFeatures_ = trainFeatures
        features_ =  features

    # Select train features
    pos_train_feature, newshape = create_pos_feature(graphScale, 6, trainFeatures_)
    train_fet_num = [0,1,2,3,4,5]
    for fet in trainFeatures_:
        if fet in features_:
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
            elif fet == 'landcover':
                maxi = coef * len(landcover_variables)
            elif fet == 'dynamicWorld':
                maxi += coef * len(dynamic_world_variables)
            elif fet == 'vigicrues':
                maxi = len(vigicrues_variables)
            elif fet in varying_time_variables:
                maxi = 1
            else:
                maxi = coef
            train_fet_num += list(np.arange(pos_feature[fet], pos_feature[fet] + maxi))

    logger.info(train_fet_num)
    pos_feature = pos_train_feature
    X = X[:, np.asarray(train_fet_num)]

    prefix = str(minPoint)+'_'+str(k_days)+'_'+str(scale)
    prefix_train = str(minPoint)+'_'+str(k_days)+'_'+str(scale)
    if spec != '':
        prefix_train += '_'+spec
        prefix += '_'+spec

    # Preprocess
    Xset, Yset = preprocess_test(X=X, Y=Y, Xtrain=Xtrain, scaling=scaling)

    # Features selection
    features_importance = read_object('features_importance_tree_'+prefix_train+'.pkl', train_dir)

    log_features(features_importance, pos_feature, ['min', 'mean', 'max', 'std'])
    features_selected = np.unique(features_importance[:,0]).astype(int)

    ################################ Ground Truth ############################################
    logger.info('#########################')
    logger.info(f'       GT              ')
    logger.info('#########################')

    if Rewrite:
        XTensor, YTensor, ETensor = load_tensor_test(use_temporal_as_edges=True, graphScale=graphScale, dir_output=dir_output,
                                                            X=X, Y=Y, Xtrain=Xtrain, Ytrain=Ytrain, device=device, k_days=k_days,
                                                            test=testname, pos_feature=pos_feature,
                                                            scaling=scaling, prefix=prefix, encoding=encoding,
                                                            Rewrite=False)
        
        _ = load_loader_test(use_temporal_as_edges=False, graphScale=graphScale, dir_output=dir_output,
                                        X=Xset, Y=Yset, Xtrain=Xtrain, Ytrain=Ytrain,
                                        device=device, k_days=k_days, test=testname, pos_feature=pos_feature,
                                        scaling=scaling, encoding=encoding, prefix=prefix, Rewrite=Rewrite)
        
    ITensor = YTensor[:,-3]
    Ypred = torch.masked_select(YTensor[:,-1], ITensor.gt(0))
    YTensor = YTensor[ITensor.gt(0)]
    
    metrics['GT'] = add_metrics(methods, 0, Ypred, YTensor, testDepartement, False, scale, 'gt', train_dir)
    i = 1

    #################################### Traditionnal ##################################################

    for name, isBin in tradiModels:
        logger.info('#########################')
        logger.info(f'      {name}            ')
        logger.info('#########################')
        XTensor, YTensor, _ = load_tensor_test(use_temporal_as_edges=False, graphScale=graphScale, dir_output=dir_output,
                                                        X=X, Y=Y, Xtrain=Xtrain, Ytrain=Ytrain, device=device, k_days=k_days,
                                                        test=testname, pos_feature=pos_feature,
                                                        scaling=scaling, prefix=prefix, encoding=encoding,
                                                        Rewrite=True)
        print(XTensor.shape)
        XTensor = XTensor.detach().cpu().numpy()
        XTensor = XTensor[:, features_selected, -1]

        YTensor = YTensor[:,:, -1]
        ITensor = YTensor[:,-3]

        model = read_object(name+'.pkl', train_dir / Path('check_'+scaling+'/' + prefix_train + '/baseline/' + name + '/'))
        graphScale._set_model(model)
        Ypred = graphScale.predict_model_api_sklearn(XTensor, isBin)

        Ypred = torch.tensor(Ypred, dtype=torch.float32, device=device)
        YTensor = YTensor[ITensor.gt(0)]
        Ypred = torch.masked_select(Ypred, ITensor.gt(0))

        if not dummy:
            metrics[name] = add_metrics(methods, i, Ypred, YTensor, testDepartement, isBin, scale, name, train_dir)
        else:
            metrics[name +'_dummy'] = add_metrics(methods, i, Ypred, YTensor, testDepartement, isBin, scale, name, train_dir)

        pred = Ypred.detach().cpu().numpy()
        y = YTensor.detach().cpu().numpy()

        realVspredict(pred, y,
                      dir_output, name+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'.png')
        
        res = np.empty((pred.shape[0], y.shape[1] + 3))
        res[:, :y.shape[1]] = y
        res[:,y.shape[1]] = pred
        res[:,y.shape[1]+1] = np.sqrt((y[:,-3] * (pred - y[:,-1]) ** 2))
        res[:,y.shape[1]+2] = np.sqrt(((pred - y[:,-1]) ** 2))

        save_object(res, name+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output)
        
        """n = name+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname
        realVspredict2d(pred,
            y,
            isBin,
            name,
            scale,
            dir_output / n,
            train_dir,
            geo,
            testDepartement,
            graphScale)"""

        i += 1

    ######################################## Simple model ##################################
    if not dummy:
        XTensor, YTensor, ETensor = load_tensor_test(use_temporal_as_edges=True, graphScale=graphScale, dir_output=dir_output,
                                                            X=X, Y=Y, Xtrain=Xtrain, Ytrain=Ytrain, device=device, k_days=k_days,
                                                            test=testname, pos_feature=pos_feature,
                                                            scaling=scaling, prefix=prefix, encoding=encoding,
                                                            Rewrite=False)
        ITensor = YTensor[:,-3]
        if testname != 'dummy':
            """y2 = YTensor.detach().cpu().numpy()

            logger.info('#########################')
            logger.info(f'      perf_Y            ')
            logger.info('#########################')
            name = 'perf_Y'
            Ypred = graphScale._predict_perference_with_Y(y2, False)
            Ypred = torch.tensor(Ypred, dtype=torch.float32, device=device)
            YTensor = YTensor[ITensor.gt(0)]
            Ypred = torch.masked_select(Ypred, ITensor.gt(0))
            metrics[name] = add_metrics(methods, 0, Ypred, YTensor, testDepartement, False, scale, name)

            pred = Ypred.detach().cpu().numpy()
            y = YTensor.detach().cpu().numpy()

            res = np.empty((pred.shape[0], y.shape[1] + 3))
            res[:, :y.shape[1]] = y
            res[:,y.shape[1]] = pred
            res[:,y.shape[1]+1] = np.sqrt((y[:,-3] * (pred - y[:,-1]) ** 2))
            res[:,y.shape[1]+2] = np.sqrt(((pred - y[:,-1]) ** 2))

            save_object(res, name+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output)

            logger.info('#########################')
            logger.info(f'      perf_Y_bin         ')
            logger.info('#########################')
            name = 'perf_Y_bin'
            Ypred = graphScale._predict_perference_with_Y(y2, True)
            Ypred = torch.tensor(Ypred, dtype=torch.float32, device=device)
            Ypred = torch.masked_select(Ypred, ITensor.gt(0))
            metrics[name] = add_metrics(methods, 0, Ypred, YTensor, testDepartement, True, scale, name, path(nameExp))

            pred = Ypred.detach().cpu().numpy()
            y = YTensor.detach().cpu().numpy()

            res = np.empty((pred.shape[0], y.shape[1] + 3))
            res[:, :y.shape[1]] = y
            res[:,y.shape[1]] = pred
            res[:,y.shape[1]+1] = np.sqrt((y[:,-3] * (pred - y[:,-1]) ** 2))
            res[:,y.shape[1]+2] = np.sqrt(((pred - y[:,-1]) ** 2))

            save_object(res, name+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output)"""

        """logger.info('#########################')
        logger.info(f'      perf_X         ')
        logger.info('#########################')
        name = 'perf_X'
        XTensor, YTensor, ETensor = load_tensor_test(use_temporal_as_edges=True, graphScale=graphScale, dir_output=dir_output,
                                                            X=X, Y=Y, Xtrain=Xtrain, Ytrain=Ytrain, device=device, k_days=k_days,
                                                            test=testname, pos_feature=pos_feature,
                                                            scaling=scaling, prefix=prefix, encoding=encoding)
                                                            
        x = XTensor.detach().cpu().numpy()
        
        Ypred = graphScale._predict_perference_with_X(x, pos_feature)
        Ypred = torch.tensor(Ypred, dtype=torch.float32, device=device)
        YTensor = YTensor[ITensor.gt(0)]
        Ypred = torch.masked_select(Ypred, ITensor.gt(0))
        metrics['perf_X'] = add_metrics(methods, 0, Ypred, YTensor, testDepartement, False, scale, name, path(nameExp))

        pred = Ypred.detach().cpu().numpy()
        y = YTensor.detach().cpu().numpy()

        res = np.empty((pred.shape[0], y.shape[1] + 3))
        res[:, :y.shape[1]] = y
        res[:,y.shape[1]] = pred
        res[:,y.shape[1]+1] = np.sqrt((y[:,-3] * (pred - y[:,-1]) ** 2))
        res[:,y.shape[1]+2] = np.sqrt(((pred - y[:,-1]) ** 2))

        save_object(res, name+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output)"""

    ########################################## Save metrics ################################ 
    outname = 'metrics'+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_tree.pkl'
    save_object(metrics, outname, dir_output)

def dummy_test(dates):
    name = sinister+".csv"
    fp = pd.read_csv(train_dir / ".." / name)
    name = "non"+sinister+".csv"
    nfp = pd.read_csv(train_dir / ".." / name)
    fp = fp[fp['date'].isin(dates)]
    nfp = nfp[nfp['date'].isin(dates)]
    logger.info(f'Dummy test {len(fp)}, {len(nfp)}')
    res = pd.concat((fp, nfp)).reset_index(drop=True)
    res['departement'] = res['departement'].apply(lambda x : int2strMaj[x])
    return res

# 69 test
testDates = find_dates_between('2018-01-01', '2023-01-01')
testDepartement = ['departement-69-rhone']

test('69', testDates, geo, geo, testDepartement, dir_output, features, doDatabase, trainFeatures)

testDates = find_dates_between('2023-01-01', '2023-09-11')
testDepartement = ['departement-01-ain', 'departement-25-doubs', 'departement-78-yvelines']

# Dummy
dummyGeo = dummy_test(testDates)
#test('dummy', None, dummyGeo, geo, testDepartement, dir_output, features, doDatabase, trainFeatures)

# 2023 test
test('2023', testDates, geo, geo, testDepartement, dir_output, features, doDatabase, trainFeatures)

# 2023 test Ain
testDepartement = ['departement-01-ain']
#test('Ain', testDates, geo, geo, testDepartement, dir_output, features, doDatabase, trainFeatures)

# 2023 test Doubs
testDepartement = ['departement-25-doubs']
#test('Doubs', testDates, geo, geo, testDepartement, dir_output, features, doDatabase, trainFeatures)

# 2023 test Yvelines
testDepartement = ['departement-78-yvelines']
#test('Yvelines', testDates, geo, geo, testDepartement, dir_output, features, doDatabase, trainFeatures)