from graph_structure import *
from features import *
from dataloader import *
from construct import *
from forecasting_models.loss import *
import argparse

########################### Input Arg ######################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='Test',
    )

    parser.add_argument('-n', '--name', type=str, help='Name of the experiment')
    parser.add_argument('-s', '--sinister', type=str, help='Sinister type')
    parser.add_argument('-d', '--database', type=str, help='Do database')
    parser.add_argument('-g', '--graph', type=str, help='Construct graph')
    parser.add_argument('-sc', '--scale', type=str, help='Scale')
    parser.add_argument('-sp', '--spec', type=str, help='spec')
    parser.add_argument('-np', '--nbpoint', type=str, help='Number of point')
    parser.add_argument('-nf', '--NbFeatures', type=str, help='Nombur de Features')

    args = parser.parse_args()

    # Input config
    name_exp = args.name
    doGraph = args.graph == "True"
    doDatabase = args.database == "True"
    scale = int(args.scale)
    sinister = args.sinister
    nbfeatures = args.NbFeatures
    datatset_name= args.spec
    minPoint = args.nbpoint

    geo = gpd.read_file('regions/regions.geojson')

############### Global variable ###############

scaleTrain = scale

name_dir = name_exp + '/' + sinister + '/' + 'train'
dir_train = Path(name_dir)

name_dir = name_exp + '/' + sinister + '/' + 'test'
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

def process_test(testname, testDate, pss, geo, testd_departement, dir_output, features, doDatabase):
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
        graphScale = construct_graph(scale, maxDist[scale], sinister, geo, nmax, k_days, dir_output, True, False)
        doDatabase = True
    else:
        graphScale = read_object('graph_'+str(scale)+'.pkl', dir_output)

    ########################## Construct database #############################
    if doDatabase:
        pss = pss[pss['departement'].isin([name2str[dep] for dep in testd_departement])].reset_index(drop=True)
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

        X, Y, features_name = construct_database(graphScale, ps, scale, k_days, departements,
                                               features, sinister, dir_output, dir_train, prefix, 'test')
        
        save_object(Y, 'Y_'+prefix+'.pkl', dir_output)
        save_object(X, 'X_'+prefix+'.pkl', dir_output)
    else:
        X = read_object('X_'+prefix+'.pkl', dir_output)
        Y = read_object('Y_'+prefix+'.pkl', dir_output)

        graphScale._info_on_graph(X, dir_output)

    features_name, _ = get_features_name_list(graphScale.scale, 6, features)

    Xtrain = read_object('Xtrain_'+prefix_train+'.pkl', dir_train)
    y_train = read_object('y_train_'+prefix_train+'.pkl', dir_train)
    if Xtrain is None:
        logger.info(f'Fuck it')
        return

    train_features = read_object(spec+'_train_features.pkl', dir_train)
    if train_features is None:
        logger.info('Train Feature not found')
        exit(1)

    # Add varying time features
    if k_days > 0:
        train_features_ = deepcopy(train_features) + varying_time_variables
        features_ = deepcopy(features) + varying_time_variables
        features_name, newShape = get_features_name_list(graphScale.scale, 6, features_)
        if doDatabase:
            X = add_varying_time_features(X=X, features=varying_time_variables, newShape=newShape, features_name=features_name, ks=k_days)
            save_object(X, 'X_'+prefix+'.pkl', dir_output)
    else:
        train_features_ = deepcopy(train_features)
        features_ =  deepcopy(features)

    # Remove some bad nodes that goes to wrong departement
    logger.info(f'Orignal dept {np.unique(X[:, 3]), X.shape}')
    X = X[np.isin(X[:, 3], [name2int[dep] for dep in testd_departement])]
    Y = Y[np.isin(Y[:, 3], [name2int[dep] for dep in testd_departement])]
    logger.info(f'Test dept {np.unique(X[:, 3]), X.shape}')

    # Select train features
    train_fet_num = [0,1,2,3,4,5]
    for fet in features_:
        if fet in train_features_:
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
            train_fet_num += list(np.arange(features_name.index(fet), features_name.index(fet) + maxi))

    features_name, newshape = get_features_name_list(graphScale.scale, 6, train_features_)
    X = X[:, np.asarray(train_fet_num)]
    Xtrain = Xtrain[:, np.asarray(train_fet_num)]
    logger.info(features_name)
    prefix = str(minPoint)+'_'+str(k_days)+'_'+str(scale)+'_'+str(nbfeatures)
    prefix_train = str(minPoint)+'_'+str(k_days)+'_'+str(scale)+'_'+str(nbfeatures)
    if datatset_name!= '':
        prefix_train += '_'+spec
        prefix += '_'+spec

    Xset, Yset = preprocess_test(X, Y , Xtrain, scaling)
    logger.info(f'{testname} test {Xset.shape}')
    logger.info(f'{allDates[int(np.min(X[:,4]))], allDates[int(np.max(Y[:,4]))]}')

    test_sklearn_api_model(graphScale, Xset, Yset, Xtrain, y_train,
                           methods,
                           testname,
                           features_name,
                           prefix,
                           prefix_train,
                           dummy,
                           models,
                           dir_output,
                           device,
                           k_days,
                           Rewrite, 
                           encoding,
                           scaling,
                           testd_departement,
                           dir_train)

def dummy_test(dates):
    name = sinister+".csv"
    fp = pd.read_csv(dir_train / ".." / name)
    name = "non"+sinister+".csv"
    nfp = pd.read_csv(dir_train / ".." / name)
    fp = fp[fp['date'].isin(dates)]
    nfp = nfp[nfp['date'].isin(dates)]
    logger.info(f'Dummy test {len(fp)}, {len(nfp)}')
    res = pd.concat((fp, nfp)).reset_index(drop=True)
    res['departement'] = res['departement'].apply(lambda x : int2strMaj[x])
    return res

if sinister == 'firepoint':
    # 69 test
    testDates = find_dates_between('2018-01-01', '2023-01-01')
    testd_departement = ['departement-69-rhone']
    process_test('69', testDates, geo, geo, testd_departement, dir_output, features, doDatabase)

    # 2023 test
    testDates = find_dates_between('2023-01-01', '2023-09-11')
    testd_departement = ['departement-01-ain', 'departement-25-doubs', 'departement-78-yvelines']
    process_test('2023', testDates, geo, geo, testd_departement, dir_output, features, doDatabase)

if sinister == 'inondation':
    # 69 test
    testDates = find_dates_between('2018-01-01', '2023-09-11')
    testd_departement = ['departement-69-rhone', 'departement-01-ain', 'departement-78-yvelines']
    process_test('69', testDates, geo, geo, testd_departement, dir_output, features, doDatabase)

    # 2023 test
    testDates = find_dates_between('2023-01-01', '2023-09-11')
    testd_departement = ['departement-25-doubs']
    process_test('2023', testDates, geo, geo, testd_departement, dir_output, features, doDatabase)

# 2023 test Ain
testd_departement = ['departement-01-ain']
#test('Ain', testDates, geo, geo, testd_departement, dir_output, features, doDatabase, train_features)

# 2023 test Doubs
testd_departement = ['departement-25-doubs']
#test('Doubs', testDates, geo, geo, testd_departement, dir_output, features, doDatabase, train_features)

# 2023 test Yvelines
testd_departement = ['departement-78-yvelines']
#test('Yvelines', testDates, geo, geo, testd_departement, dir_output, features, doDatabase, train_features)