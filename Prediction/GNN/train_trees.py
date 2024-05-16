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
#parser.add_argument('-ks', '--k_days', type=str, help='Number of days of timeseries')
parser.add_argument('-sc', '--scale', type=str, help='Scale')
parser.add_argument('-sp', '--spec', type=str, help='spec')

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
sinister = args.sinister
minPoint = args.nbpoint
#ks = int(args.k_days)
scale = int(args.scale)
spec = args.spec

######################### Incorporate new features ##########################

newFeatures = []

############################# GLOBAL VARIABLES #############################

dir_target = root_target / sinister / 'log'

geo = gpd.read_file('regions/regions.geojson')
geo = geo[geo['departement'].isin([name2str[dept] for dept in departements])].reset_index(drop=True)

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
        depts = [name2int[dept] for dept in trainDepartements]
        logger.info(ps.departement.unique())
        ps = ps[ps['departement'].isin(depts)].reset_index(drop=True)
        logger.info(ps.departement.unique())
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

# Add varying time features
if k_days > 0:
    trainFeatures += varying_time_variables
    features += varying_time_variables
    pos_feature, newShape = create_pos_feature(graphScale, 6, features)
    if doDatabase:
        X = add_varying_time_features(X=X, features=varying_time_variables, newShape=newShape, pos_feature=pos_feature, ks=k_days)
        save_object(X, 'X_'+prefix+'.pkl', dir_output)

############################# Training ###########################

# Select train features
pos_train_feature, newshape = create_pos_feature(graphScale, 6, trainFeatures)
train_fet_num = [0,1,2,3,4,5]
for fet in trainFeatures:
    if fet in features:
        coef = 4 if scale > 0 else 1
        if fet == 'Calendar' or fet == 'Calendar_mean':
            maxi = len(calendar_variables)
        elif fet == 'air' or fet == 'air_mean':
            maxi = len(air_variables)
        elif fet == 'sentinel':
            maxi = coef * len(sentinel_variables)
        elif fet == 'Geo':
            maxi = len(geo_variables)
        elif fet in varying_time_variables:
            maxi = 1
        else:
            maxi = coef
        train_fet_num += list(np.arange(pos_feature[fet], pos_feature[fet] + maxi))

logger.info(train_fet_num)
prefix = str(minPoint)+'_'+str(k_days)+'_'+str(scale)
if spec != '':
    prefix += '_'+spec
pos_feature = pos_train_feature
X = X[:, np.asarray(train_fet_num)]

logger.info(allDates.index('2023-01-01'))
logger.info(np.max(X[:,4]))
logger.info(np.unique(X[:,0]))

# Preprocess
Xset, Yset = preprocess(X=X, Y=Y, scaling=scaling, maxDate=trainDate, ks=k_days)

logger.info(pos_feature)
# Features selection
features_selected = features_selection(doFet, Xset, Yset, dir_output, pos_feature, prefix, True)

######################## Baseline ##############################

name = 'check_'+scaling + '/' + prefix + '/' + '/baseline'

trainDataset = (Xset[Xset[:,4] < allDates.index(trainDate)], Yset[Xset[:,4] < allDates.index(trainDate)])
valDataset = (Xset[Xset[:,4] >= allDates.index(trainDate)], Yset[Xset[:,4] >= allDates.index(trainDate)])

train_sklearn_api_model(trainDataset, valDataset,
dir_output / name,
device='cpu',
binary=False,
scaling=scaling,
features=features_selected,
weight=True)

train_sklearn_api_model(trainDataset, valDataset,
dir_output / name,
device='cpu',
binary=True,
scaling=scaling,
features=features_selected,
weight=True)

train_sklearn_api_model(trainDataset, valDataset,
dir_output / name,
device='cpu',
binary=True,
scaling=scaling,
features=features_selected,
weight=False)