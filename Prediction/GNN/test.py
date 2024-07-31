from graph_structure import *
from features import *
from dataloader import *
from construct import *
from forecasting_models.loss import *
import argparse

########################### Input Arg ######################################

parser = argparse.ArgumentParser(
    prog='Test',
)

parser.add_argument('-n', '--name', type=str, help='Name of the experiment')
parser.add_argument('-s', '--sinister', type=str, help='Sinister type')
parser.add_argument('-g', '--graph', type=str, help='Construct graph')
parser.add_argument('-d', '--database', type=str, help='Do database')
parser.add_argument('-sc', '--scale', type=str, help='Scale')
parser.add_argument('-dd', '--database2D', type=str, help='Do 2D database')
parser.add_argument('-sp', '--spec', type=str, help='spec')
parser.add_argument('-np', '--nbpoint', type=str, help='Number of point')
parser.add_argument('-nf', '--NbFeatures', type=str, help='Nombur de Features')

args = parser.parse_args()

# Input config
name_exp = args.name
doDatabase = args.database == "True"
doGraph = args.graph == "True"
do2D = args.database2D == "True"
scale = int(args.scale)
nbfeatures = int(args.NbFeatures) if args.NbFeatures != 'all' else args.NbFeatures
sinister = args.sinister
spec = args.spec
minPoint = args.nbpoint

newFeatures = []

############### Global variable ###############

scaleTrain = scale

name_dir = name_exp + '/' + sinister + '/' + 'train'
train_dir = Path(name_dir)

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
            ('meac', meac, 'class')
           ]

geo = gpd.read_file('regions/regions.geojson')

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

metrics = {}

############################# Pipelines ##############################

def process_test(testname, testDate, pss, geo, testDepartement, dir_output, features, doDatabase):
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

        X, Y, features_name = construct_database(graphScale, ps, scale, k_days, departements,
                                               features, sinister, dir_output, train_dir, prefix, 'test')
        
        save_object(Y, 'Y_'+prefix+'.pkl', dir_output)
        save_object(X, 'X_'+prefix+'.pkl', dir_output)
    else:
        X = read_object('X_'+prefix+'.pkl', dir_output)
        Y = read_object('Y_'+prefix+'.pkl', dir_output)

        graphScale._info_on_graph(X, dir_output)

    features_name, _ = get_features_name_list(graphScale.scale, 6, features)
    if do2D:
        subnode = X[:,:6]
        features_name_2D, newShape2D = get_sub_nodes_feature_2D(graphScale,
                                                 subnode.shape[1],
                                                 X,
                                                 scaling,
                                                 features_name,
                                                 testDepartement,
                                                 features,
                                                 dir_output,
                                                 prefix)
    else:
        subnode = X[:,:6]
        features_name_2D, newShape2D = get_features_name_lists_2D(subnode.shape[1], features)

    x_train = read_object('x_train_'+prefix_train+'.pkl', train_dir)
    y_train = read_object('y_train_'+prefix_train+'.pkl', train_dir)
    if x_train is None:
        logger.info(f'Fuck it')
        return
    
    train_features = read_object(spec+'_train_features.pkl', train_dir)
    if train_features is None:
        logger.info('Train Feature not found')
        exit(1)

    # Select train features
    train_fet_num = [0,1,2,3,4,5]
    for fet in features:
        if fet in train_features:
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

    features_name, newshape = get_features_name_list(graphScale.scale, 6, train_features)
    X = X[:, np.asarray(train_fet_num)]
    x_train = x_train[:, np.asarray(train_fet_num)]
    logger.info(features_name)
    prefix = str(minPoint)+'_'+str(k_days)+'_'+str(scale)+'_'+str(nbfeatures)
    prefix_train = str(minPoint)+'_'+str(k_days)+'_'+str(scale)+'_'+str(nbfeatures)
    if spec != '':
        prefix_train += '_'+spec
        prefix += '_'+spec

    Xset, Yset = preprocess_test(X, Y , x_train, scaling)

    test_dl_model(graphScale, Xset, Yset, x_train, y_train,
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
                           testDepartement,
                           train_dir,
                           features_name_2D,
                           shape2D)

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

if sinister == 'firepoint':
    # 69 test
    testDates = find_dates_between('2018-01-01', '2023-01-01')
    testDepartement = ['departement-69-rhone']
    process_test('69', testDates, geo, geo, testDepartement, dir_output, features, doDatabase)
    
    # 2023 test
    testDates = find_dates_between('2023-01-01', '2023-09-11')
    testDepartement = ['departement-01-ain', 'departement-25-doubs', 'departement-78-yvelines']
    process_test('2023', testDates, geo, geo, testDepartement, dir_output, features, doDatabase)

if sinister == 'inondation':
    # 69 test
    testDates = find_dates_between('2018-01-01', '2023-09-11')
    testDepartement = ['departement-69-rhone', 'departement-01-ain', 'departement-78-yvelines']
    process_test('69', testDates, geo, geo, testDepartement, dir_output, features, doDatabase)

    # 2023 test
    testDates = find_dates_between('2023-01-01', '2023-09-11')
    testDepartement = ['departement-25-doubs']
    process_test('2023', testDates, geo, geo, testDepartement, dir_output, features, doDatabase)

# 2023 test Ain
testDepartement = ['departement-01-ain']
#test('Ain', testDates, geo, geo, testDepartement, dir_output, features, doDatabase)

# 2023 test Doubs
testDepartement = ['departement-25-doubs']
#test('Doubs', testDates, geo, geo, testDepartement, dir_output, features, doDatabase)

# 2023 test Yvelines
testDepartement = ['departement-78-yvelines']
#test('Yvelines', testDates, geo, geo, testDepartement, dir_output, features, doDatabase)