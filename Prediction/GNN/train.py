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
parser.add_argument('-d', '--database', type=str, help='Do database')
parser.add_argument('-g', '--graph', type=str, help='Construct graph')
parser.add_argument('-mxd', '--maxDate', type=str, help='Limit train and validation date')
parser.add_argument('-mxdv', '--trainDate', type=str, help='Limit training date')
parser.add_argument('-f', '--featuresSelection', type=str, help='Do features selection')
parser.add_argument('-dd', '--database2D', type=str, help='Do 2D database')

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
sinister = args.sinister

######################### Incorporate new features ##########################

newFeatures = []

############################# GLOBAL VARIABLES #############################

dir_target = root_target / sinister / 'log'

geo = gpd.read_file('regions.geojson')
geo = geo[geo['departement'].isin(['Ain', 'Doubs', 'Yvelines', 'Rhone'])].reset_index(drop=True)

doBaseline = True
name_dir = nameExp + '/' + sinister + '/' + 'train'
dir_output = Path(name_dir)
check_and_create_path(dir_output)

minDate = '2017-06-12' # Starting point

prefix = str(minPoint)+'_'+str(k_days)+'_'+str(scale)
if dummy:
    prefix += '_dummy'

########################### Create points ################################
if doPoint:
    minPoint = 100

    check_and_create_path(dir_output)

    regions = gpd.read_file('regions.geojson')
    fp = pd.read_csv('sinister/'+sinister+'.csv')

    construct_non_point(fp, regions, maxDate, sinister, Path(nameExp))
    look_for_information(dir_target,
                         departements,
                         fp,
                         maxDate,
                         sinister,
                         Path(nameExp))

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
    print(X.shape)
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

# Preprocess
Xset, Yset = preprocess(X=X, Y=Y, scaling=scaling, maxDate=maxDate)

# Features selection
features_selected = features_selection(doFet, Xset, Yset, dir_output, pos_feature)

# Train
criterion = weighted_rmse_loss
epochs = 10000
lr = 0.01
PATIENCE_CNT = 200
CHECKPOINT = 100

gnnModels = [
        #('GAT', False, False),
        #('ST-GATTCN', False, False),
        #('ST-GATCONV', False, False),
        #('ST-GCNCONV', False, False),
        #('ATGN', False, False),
        #('ST-GATLTSM', False, False),
        #('Zhang', False, True)
        #('ConvLSTM', False, True)
        ]

trainLoader = None
valLoader = None
last_bool = None

for model, use_temporal_as_edges, is_2D_model in gnnModels:
    print(f'Fitting model {model}')
    try:
        print('Try loading loader')

        if not is_2D_model:
            trainLoader = read_object('trainloader_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
            valLoader = read_object('valLoader_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
        else:
            trainLoader = read_object('trainloader_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'_2D.pkl', dir_output)
            valLoader = read_object('valLoader_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'_2D.pkl', dir_output)
    except:
        print(f'Reading failed, computing, loading loader_{str(use_temporal_as_edges)}.pkl')

        last_bool = use_temporal_as_edges
        if not is_2D_model:
            trainLoader, valLoader = train_val_data_loader(graph=graphScale, Xset=Xset, Yset=Yset,
                                                            maxDate=trainDate,
                                                            batch_size=64, device=device,
                                                            use_temporal_as_edges = use_temporal_as_edges,
                                                            ks=k_days)
        
            save_object(trainLoader, 'trainloader_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
            save_object(valLoader, 'valLoader_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
        else:
            trainLoader, valLoader = train_val_data_loader_2D(graph=graphScale,
                                                              Xset=Xset, Yset=Yset,
                                                              maxDate=trainDate,
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

    print(dico_model[model])
    train(trainLoader=trainLoader, valLoader=valLoader,
        PATIENCE_CNT=PATIENCE_CNT,
        CHECKPOINT=CHECKPOINT,
        epochs=epochs,
        features=features_selected,
        lr=lr,
        criterion=criterion,
        model=dico_model[model],
        dir_output=dir_output / Path('check_'+scaling + '/' + prefix + '/' + model))

######################## Baseline ##############################

trainDataset, valDataset = create_dataset(graphScale,
                                Xset, Yset, trainDate, False,
                                device, k_days)

if doBaseline:
    name = 'check_'+scaling + '/' + prefix + '/' + '/baseline'

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