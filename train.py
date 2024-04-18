from dataloader import *
from graph_structure import *
import geopandas as gpd
from construct import *
from config import *
from models.loss import *

# Input config
scale = int(sys.argv[1])
readGraph = sys.argv[2] == 'read'
doDatabase = sys.argv[3] == 'doDatabase'
do2D = sys.argv[4] == 'do2D'
newFeatures = []

############################# GLOBAL VARIABLES #############################

geo = gpd.read_file('regions.geojson')
geo = geo[geo['departement'].isin(['Ain', 'Doubs', 'Yvelines'])].reset_index(drop=True)

doBaseline = True
name_dir = 'train'
dir_output = Path(name_dir)

prefix = str(minPoint)+'_'+str(k_days)+'_'+str(scale)
if dummy:
    prefix += '_dummy'

########################## Construct database #######################################

if readGraph:
    graphScale = read_object('graph_'+str(scale)+'.pkl', dir_output)
else:
    graphScale = construct_graph(scale, maxDist, geo, nmax, k_days, dir_output)
    doDtabase = True

if doDatabase:

    # Base node and geometry
    if not dummy and not binary:
        ps = pd.read_csv('points'+str(minPoint)+'.csv')
    else:
        fp = pd.read_csv('../CNN/STA/firepoints.csv')
        nfp = pd.read_csv('../CNN/STA/nonfirepoints.csv')
        ps = pd.concat((fp, nfp)).reset_index(drop=True)
        ps = fp.copy(deep=True)
        ps = ps[ps['date'].isin(allDates)]
        ps = ps[ps['date'] < '2023-01-01']
        ps['date'] = [allDates.index(date) - 1 for date in ps.date]

    X, Y, pos_feature = construct_database(graphScale, ps, scale, k_days, trainDepartements, features, dir_output, prefix)
else:
    X = read_object('X_'+prefix+'.pkl', dir_output)
    Y = read_object('Y_'+prefix+'.pkl', dir_output)
    pos_feature, newshape = create_pos_feature(graphScale, 6, features)

if len(newFeatures) > 0:
    print(X.shape)
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
                                                 dir_output,
                                                 prefix)
    
else:
    subnode = X[:,:6]
    pos_feature_2D, newShape2D = create_pos_features_2D(subnode.shape[1], features)

save_object(graphScale, 'scale'+str(scale)+'.pkl', dir_output)

############################# Training ###########################

# Preprocess
Xset, Yset, dateTrain, dateVal, trainSet, valSet = preprocess(X=X, Y=Y, test_size=0.3, pos_feature=pos_feature, scaling=scaling, encoding=encoding)

criterion = weighted_rmse_loss
#criterion = poisson_loss
#criterion = WeightedMSELoss()
#criterion = WeightedPoissonLoss(log_input=True, full=False, size_average=None, eps=1e-8, reduce=None, reduction='None')

epochs = 10000
lr = 0.01
PATIENCE_CNT = 200
CHECKPOINT = 100

gnnModels = [
        #('GAT', False, False),
        #('TGAT', True, False),
        #('ST-GATTCN', False, False),
        #('ST-GATCONV', False, False),
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

        print(f'Try loading loader')

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
                                                            dateTrain=dateTrain, dateVal=dateVal,
                                                            batch_size=64, device=device,
                                                            use_temporal_as_edges = use_temporal_as_edges,
                                                            ks=k_days)
        
            save_object(trainLoader, 'trainloader_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
            save_object(valLoader, 'valLoader_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
            
        else:
            trainLoader, valLoader = train_val_data_loader_2D(graph=graphScale,
                                                              Xset=Xset, Yset=Yset,
                                                              Xtrain=trainSet[0], Ytrain=trainSet[1],
                                                              dateTrain=dateTrain, dateVal=dateVal, 
                                                              use_temporal_as_edges=use_temporal_as_edges, batch_size=64,
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
        lr=lr,
        criterion=criterion,
        model=dico_model[model],
        dir_output=Path('check_'+scaling + '/' +prefix + '/' + model))

trainLoader, valLoader = train_val_data_loader(graphScale, Xset, Yset, dateTrain, dateVal, batch_size=64, device=device,
                                                                    use_temporal_as_edges = False, ks=k_days)

######################## Baseline ##############################

if doBaseline:
    train_sklearn_api_model(trainSet, valSet,
    Path('check_'+scaling + '/' + str(minPoint)+'_'+str(k_days)+'_'+str(scale) + '/baseline'),
    device='cuda',
    binary=binary,
    scaling=scaling,
    encoding=encoding,
    pos_feature=pos_feature)