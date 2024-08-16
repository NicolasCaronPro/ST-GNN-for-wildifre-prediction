import pickle
from sklearn.cluster import KMeans
from copy import copy
from weigh_predictor import Predictor
from encoding import *

def look_for_information(path : Path, departements : list,
                         regions : gpd.GeoDataFrame, maxDate : str,
                         sinister : str, dir : Path,
                         ):

    points = []
    for departement in departements:
        print(departement)
        name = departement+'Influence.pkl'
        influence = read_object(name, path)

        if influence is None:
            continue
        
        if departement == 'departement-69-rhone':
            start = allDates.index('2018-01-01')
            md = '2023-01-01' if maxDate > '2023-01-01' else maxDate
            end = allDates.index(md) 
            datesForDepartement = find_dates_between('2018-01-01', md)
        elif departement == 'departement-01-ain':
            start = allDates.index('2018-01-01')
            end = allDates.index(maxDate)
            datesForDepartement = find_dates_between('2018-01-01', maxDate)
        else:
            end = allDates.index(maxDate)
            start = 0
            datesForDepartement = find_dates_between('2017-06-12', maxDate)

        dis = list(np.arange(start, end))

        if departement != 'departement-69-rhone':
            logger.info(f"{allDates.index(maxDate), influence.shape[-1]}")
            dis += list(np.arange(allDates.index(maxDate), influence.shape[-1]))
        else:
            logger.info(f"{allDates.index(maxDate), influence.shape[-1]}")
            dis += list(np.arange(allDates.index(maxDate), allDates.index('2023-01-01')))
        
        fp = regions[regions['departement'] == departement]

        iterables = [dis, list(fp.latitude)]
        ps = pd.DataFrame(index=pd.MultiIndex.from_product(iterables, names=('date', 'latitude'))).reset_index()

        iterables = [dis, list(fp.longitude)]
        ps2 = pd.DataFrame(index=pd.MultiIndex.from_product(iterables, names=('date', 'longitude'))).reset_index()
        ps['longitude'] = ps2['longitude']
        del ps2

        iterables = [dis, list(fp.departement)]
        ps2 = pd.DataFrame(index=pd.MultiIndex.from_product(iterables, names=('date', 'departement'))).reset_index()
        ps['departement'] = ps2['departement']
        del ps2

        iterables = [dis, list(fp.scale0)]
        ps2 = pd.DataFrame(index=pd.MultiIndex.from_product(iterables, names=('date', 'scale0'))).reset_index()
        ps['scale0'] = ps2['scale0']
        del ps2

        points.append(ps)

    points = pd.concat(points)
    points.drop_duplicates(inplace=True)
    points['departement'] = points['departement'].apply(lambda x : name2int[x])
    check_and_create_path(dir/sinister)
    name = 'points.csv'
    points.to_csv(dir / sinister / name, index=False)

def construct_graph(scale, maxDist, sinister, geo, nmax, k_days, dir_output, doRaster, doEdgesFeatures, resolution, graph_construct):
    graphScale = GraphStructure(scale, geo, maxDist, nmax)
    graphScale._train_clustering(base=graph_construct, doRaster=doRaster, path=dir_output, sinister=sinister, resolution=resolution)
    graphScale._create_nodes_list()
    graphScale._create_edges_list()
    graphScale._create_temporal_edges_list(allDates, k_days=k_days)
    graphScale.nodes = graphScale._assign_department(graphScale.nodes)
    if doEdgesFeatures:
        graphScale.edges = edges_feature(graphScale, ['slope', 'highway'], dir_output, geo)
    save_object(graphScale, f'graph_{scale}_{graph_construct}.pkl', dir_output)
    return graphScale

def construct_database(graphScale : GraphStructure,
                       ps : pd.DataFrame,
                       scale : int,
                       k_days : int,
                       departements : list,
                       features  : list,
                       sinister : str,
                       dir_output : Path,
                       dir_train : Path,
                       prefix : str,
                       values_per_class : int,
                       mode : str,
                       resolution : str,
                       maxDate : str,
                       graph_construct : str):
    
    trainCode = [name2int[d] for d in train_departements]
    X_kmeans = list(zip(ps.longitude, ps.latitude))
    ps['scale'+str(scale)] = graphScale._predict_node(X_kmeans, ps.departement)
    name = prefix+'.csv'
    ps.to_csv(dir_output / name, index=False)
    print(np.unique(ps['scale'+str(scale)].values))
    ps.drop_duplicates(subset=('scale'+str(scale), 'date'), inplace=True)
    print(f'{len(ps)} point in the dataset. Constructing database')

    orinode = np.full((len(ps), 5), -1.0, dtype=float)
    orinode[:,0] = ps['scale'+str(scale)].values
    orinode[:,4] = ps['date']

    orinode = generate_subgraph(graphScale, 0, 0, orinode)

    subNode = add_k_temporal_node(k_days=k_days, nodes=orinode)

    subNode = graphScale._assign_latitude_longitude(subNode)

    subNode = graphScale._assign_department(subNode)

    graphScale._info_on_graph(subNode, Path('log'))

    n = f'Y_full_{scale}_{graph_construct}.pkl'
    if not (dir_output / n).is_file():
    #if True:
        Y = get_sub_nodes_ground_truth(graphScale, subNode, departements, orinode, dir_output, dir_train, resolution, graph_construct)
        save_object(Y, n, dir_output)
    else:
        Y = read_object(n, dir_output)

    n = f'X_full_{scale}_{graph_construct}.pkl'
    #if False:
    if (dir_output / n).is_file():
        features_name, _ = get_features_name_list(scale, features, METHODS_SPATIAL)
        X = read_object(n, dir_output)
    else:
        X = None

    if values_per_class != 'full':
        y_train = Y[(Y[:,4] < allDates.index(maxDate)) & (np.isin(Y[:,3], trainCode))]
        y_test = Y[((Y[:,4] >= allDates.index(maxDate)) & (np.isin(Y[:,3], trainCode))) |
                ((Y[:,4] > allDates.index('2017-12-31')) & (Y[:,4] < allDates.index('2022-12-31')) & (~np.isin(Y[:,3], trainCode)))]
        if X is not None:
            x_test = X[((Y[:,4] >= allDates.index(maxDate)) & (np.isin(Y[:,3], trainCode))) |
                ((Y[:,4] > allDates.index('2017-12-31')) & (Y[:,4] < allDates.index('2022-12-31')) & (~np.isin(Y[:,3], trainCode)))]
    
            x_train = X[(Y[:,4] < allDates.index(maxDate)) & (np.isin(Y[:,3], trainCode))]

        # binary selection (same number of non fire than fire)
        if values_per_class == 'binary':
            new_Y = []
            new_X = []
            for dept in departements:
                maskdept = np.argwhere(y_train[:, 3] == name2int[dept])
                if maskdept.shape[0] == 0:
                    continue
                mask_non_sinister = np.argwhere(y_train[maskdept, -2] == 0)[:, 0]
                mask_sinister = np.argwhere(y_train[maskdept, -2] > 0)[:, 0]
                number_of_samples = np.argwhere(y_train[maskdept, -2] > 0).shape[0]
                new_Y_index = []
                choices = np.random.choice(mask_non_sinister, min(mask_non_sinister.shape[0], number_of_samples), replace=False)
                new_Y_index += list(choices)
                new_Y_index += list(mask_sinister)
                new_Y += list(y_train[maskdept][new_Y_index])
                if X is not None:
                    new_X += list(x_train[maskdept][new_Y_index])

        # Class selction sample (specify of number maximum of samples per class)
        else:
            new_Y = []
            new_X = []
            for dept in departements:
                maskdept = np.argwhere(y_train[:, 3] == name2int[dept])
                if maskdept.shape[0] == 0:
                    continue
                classs = np.unique(y_train[maskdept, -3])
                new_Y_index = []
                for cls in classs:
                    Y_class_index = np.argwhere(y_train[maskdept, -3] == cls)[:, 0]
                    choices = np.random.choice(Y_class_index, min(Y_class_index.shape[0], int(values_per_class)), replace=False)
                    new_Y_index += list(choices)
                new_Y += list(y_train[maskdept][new_Y_index])
                if X is not None:
                    new_X += list(x_train[maskdept][new_Y_index])

        new_Y = np.asarray(new_Y).reshape(-1, y_train.shape[-1])
        mask_days = np.zeros(y_train.shape[0], dtype=bool)
        for k in range(k_days + 1):
            mask_days = mask_days | np.isin(y_train[:, 4], new_Y[:, 4] - k)
        new_Y_neighboor = y_train[mask_days]
        new_Y_neighboor[:, 5] = 0

        if X is not None:
            new_X = np.asarray(new_X).reshape(-1, x_train.shape[-1])
            new_X_neighboor = x_train[mask_days]
            X = np.concatenate((new_X, new_X_neighboor, x_test), casting='no')
            X[:, features_name.index(f'{vigicrues_variables[0]}_mean'): features_name.index(f'{vigicrues_variables[-1]}_std') + 1] = -1 # To much nan values

        Y = np.concatenate((new_Y, new_Y_neighboor, y_test), casting='no')
        """X, Y = remove_nan_nodes_np(X, Y)
        X, Y = remove_none_target_np(X, Y)
        if X is not None:
            X = np.unique(X, axis=0)
            print(X.shape, Y.shape)

        Y = np.unique(Y, axis=0)
        print(X.shape, Y.shape)"""

    if X is None:
        print(np.unique(Y[:, 0]))
        X, features_name = get_sub_nodes_feature(graphScale, Y[:, :6], departements, features, sinister, dir_output, dir_train, resolution, graph_construct)

    ind = np.lexsort([Y[:, 4], Y[:, 0]])
    Y = Y[ind]
    if X is not None:
        X = X[ind]

    logger.info(f'{X.shape, Y.shape}')
    Y[Y[:, 4] < k_days, 5] = 0

    return X, Y, features_name

def construct_non_point(firepoints, regions, maxDate, sinister, dir):
    nfps = []
    for dept in firepoints.departement.unique():
        if dept == 'departelent-01-ain':
            ad = find_dates_between('2018-01-01', maxDate)
        else:
            ad = allDates
            
        reg = regions[regions['departement'] == dept]
        fp = firepoints[firepoints['departement'] == dept]

        non_fire_date = [date for date in ad if date not in fp.date]
        non_fire_h3 = reg[~np.isin(reg['hex_id'], fp.h3)][['latitude', 'longitude']]
     
        # Temporal
        iterables = [non_fire_date, list(fp.latitude)]
        nfp = pd.DataFrame(index=pd.MultiIndex.from_product(iterables, names=('date', 'latitude'))).reset_index()

        iterables = [non_fire_date, list(fp.longitude)]
        ps2 = pd.DataFrame(index=pd.MultiIndex.from_product(iterables, names=('date', 'longitude'))).reset_index()
        nfp['longitude'] = ps2['longitude']
        del ps2
        
        # Spatial
        iterables = [fp.date, list(non_fire_h3.latitude)]
        ps3 = pd.DataFrame(index=pd.MultiIndex.from_product(iterables, names=('date', 'latitude'))).reset_index()

        iterables = [fp.date, list(non_fire_h3.longitude)]
        ps2 = pd.DataFrame(index=pd.MultiIndex.from_product(iterables, names=('date', 'longitude'))).reset_index()
        ps3['longitude'] = ps2['longitude']
        del ps2
        
        # Final
        nfp = pd.concat((nfp, ps3)).reset_index(drop=True)
        nfp['departement'] = dept
        nfps.append(nfp)

    nfps = pd.concat(nfps).reset_index(drop=True).sample(2 * len(firepoints), random_state=42)
    print(f'Size of non {sinister} points : {len(nfps)}')

    name = 'non'+sinister+'.csv'
    nfps.to_csv(dir / sinister / name, index=False)

    name = sinister+'.csv'
    firepoints.to_csv(dir / sinister / name, index=False)

def init(args, dir_output, add_time_varying):
    
    global train_features
    global features

    ######################### Input config #############################
    name_exp = args.name
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
    k_days = int(args.k_days) # Size of the time series sequence use by DL models
    days_in_futur = int(args.days_in_futur) # The target time validation
    dir_output = dir_output
    scaling = args.scaling
    graph_construct = args.graphConstruct

    ######################## CONFIG ################################

    dir_target = root_target / sinister / 'log' / resolution

    geo = gpd.read_file('regions/regions.geojson')
    geo = geo[geo['departement'].isin(departements)].reset_index(drop=True)

    minDate = '2017-06-12' # Starting point

    if values_per_class == 'full':
        prefix = f'{values_per_class}'
    else:
        prefix = f'{values_per_class}_{k_days}'

    prefix += f'_{scale}_{graph_construct}'

    autoRegression = 'AutoRegressionReg' in train_features
    if spec == '' and autoRegression:
        spec = 'AutoRegressionReg'

    ########################### Create points ################################
    if doPoint:

        check_and_create_path(dir_output)

        fp = pd.read_csv('sinister/'+sinister+'.csv')

        construct_non_point(fp, geo, maxDate, sinister, Path(name_exp))
        look_for_information(dir_target,
                            departements,
                            geo,
                            maxDate,
                            sinister,
                            Path(name_exp))

    ######################### Encoding ######################################

    if doEncoder:
        encode(dir_target, trainDate, train_departements, dir_output / 'Encoder', resolution)

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
                                    resolution,
                                    graph_construct)
        
        graphScale._create_predictor(minDate, maxDate, dir_output, sinister, resolution)
    else:
        graphScale = read_object(f'graph_{scale}_{graph_construct}.pkl', dir_output)

    graphScale._plot(graphScale.nodes, dir_output=dir_output)

    ########################## Do Database ####################################

    if doDatabase:
        logger.info('#####################################')
        logger.info('#      Construct   Database         #')
        logger.info('#####################################')
        if not dummy:
            name = 'points.csv'
            ps = pd.read_csv(Path(name_exp) / sinister / name)
            logger.info(f'{ps.date.min(), allDates[ps.date.min()]}')
            logger.info(f'{ps.date.max(), allDates[ps.date.max()]}')
            depts = [name2int[dept] for dept in departements]
            logger.info(ps.departement.unique())
            ps = ps[ps['departement'].isin(depts)].reset_index(drop=True)
            logger.info(ps.departement.unique())
            ps = ps[ps['date'] > 0]
            logger.info(ps[ps['departement'] == 1].date.min())
        else:
            name = sinister+'.csv'
            fp = pd.read_csv(Path(name_exp) / sinister / name)
            name = 'non'+sinister+'csv'
            nfp = pd.read_csv(Path(name_exp) / sinister / name)
            ps = pd.concat((fp, nfp)).reset_index(drop=True)
            ps = ps.copy(deep=True)
            ps = ps[(ps['date'].isin(allDates)) & (ps['date'] >= '2017-06-12')]
            ps['date'] = [allDates.index(date) for date in ps.date if date in allDates]

        X, Y, features_name = construct_database(graphScale,
                                            ps, scale, k_days,
                                            departements,
                                            features, sinister,
                                            dir_output,
                                            dir_output,
                                            prefix,
                                            values_per_class,
                                            'train',
                                            resolution,
                                            trainDate,
                                            graph_construct)
        
        save_object(X, 'X_'+prefix+'.pkl', dir_output)
        save_object(Y, 'Y_'+prefix+'.pkl', dir_output)
    else:
        X = read_object('X_'+prefix+'.pkl', dir_output)
        Y = read_object('Y_'+prefix+'.pkl', dir_output)
        features_name, newshape = get_features_name_list(graphScale.scale, features, METHODS_SPATIAL)

    if newFeatures != []:
        X2, features_name_ = get_sub_nodes_feature(graphScale, Y[:, :6], departements, newFeatures, sinister, dir_output, dir_output, resolution, graph_construct)
        for fet in newFeatures:
            start , maxi, _, _ = calculate_feature_range(fet, scale, METHODS_SPATIAL)
            X[:, features_name.index(start): features_name.index(start) + maxi] = X2[:, features_name_.index(start): features_name_.index(start) + maxi]

    ################################# 2D Database ###################################
    
    if do2D:
        subnode = Y[:,:6]
        features_name_2D, newShape2D = get_sub_nodes_feature_2D(graphScale, subnode, departements, features, sinister, dir_output, dir_output, resolution, graph_construct)
    else:
        subnode = Y[:,:6]
        features_name_2D, newShape2D = get_features_name_lists_2D(subnode.shape[1], features)

    prefix = f'{values_per_class}_{k_days}'

    if graph_construct == 'geometry':
        prefix += f'_{scale}_{graph_construct}'
    else:
        prefix += f'_{graph_construct}'

    ############################## Add varying time features #############################
    if k_days > 0 and add_time_varying:
        name = f'X_{prefix}.pkl'
        if (dir_output / name).is_file() and not doDatabase:
            for k in range(1, k_days + 1):
                new_fet = [v+'_'+str(k) for v in varying_time_variables]
                train_features += [nf for nf in new_fet if nf.split('_')[0] in train_features]
                features += new_fet
                features_name, newShape = get_features_name_list(graphScale.scale, features, METHODS_TEMPORAL)
            X = read_object('X_'+prefix+'.pkl', dir_output)
        else:
            for k in range(1, k_days + 1):
                new_fet = [f'{v}_{str(k)}' for v in varying_time_variables]
                train_features += [nf for nf in new_fet if nf.split('_')[0] in train_features]
                features += new_fet
                features_name, newShape = get_features_name_list(graphScale.scale, features, METHODS_TEMPORAL)
                X = add_varying_time_features(X=X, Y=Y, features=new_fet, newShape=newShape, features_name=features_name, ks=k, scale=scale, methods=METHODS_TEMPORAL)
            save_object(X, 'X_'+prefix+'.pkl', dir_output)
    
    df = pd.DataFrame(columns=features_name + ids_columns + targets_columns, index=np.arange(0, X.shape[0]))
    df[features_name] = X
    df[ids_columns + targets_columns] = Y

    df.drop_duplicates(inplace=True)

    save_object(df, 'df_'+prefix+'.pkl', dir_output)

    return df, graphScale, prefix