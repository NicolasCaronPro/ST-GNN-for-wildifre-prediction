import pickle
from sklearn.cluster import KMeans
from graph_structure import *
from copy import copy
from features import *
from features_2D import *
from weigh_predictor import Predictor
from config import *

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

def construct_graph(scale, maxDist, sinister, geo, nmax, k_days, dir_output, doRaster, doEdgesFeatures, resolution):
    graphScale = GraphStructure(scale, geo, maxDist, nmax)
    graphScale._train_kmeans(doRaster=doRaster, path=dir_output, sinister=sinister, resolution=resolution)
    graphScale._create_nodes_list()
    graphScale._create_edges_list()
    graphScale._create_temporal_edges_list(allDates, k_days=k_days)
    graphScale.nodes = graphScale._assign_department(graphScale.nodes)
    if doEdgesFeatures:
        graphScale.edges = get_edges_feature(graphScale, ['slope', 'highway'], dir_output, geo)
    save_object(graphScale, 'graph_'+str(scale)+'.pkl', dir_output)
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
                       maxDate : str):
    
    trainCode = [name2int[d] for d in trainDepartements]
    X_kmeans = list(zip(ps.longitude, ps.latitude))
    ps['scale'+str(scale)] = graphScale._predict_node(X_kmeans)
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

    print(np.unique(subNode[:,3]))

    graphScale._info_on_graph(subNode, Path('log'))

    n = 'Y_full_'+str(scale)+'.pkl'
    if not (dir_output / n).is_file():
    #if True:
        Y = get_sub_nodes_ground_truth(graphScale, subNode, departements, orinode, dir_output, dir_train, resolution)
        save_object(Y, 'Y_full_'+str(scale)+'.pkl', dir_output)
    else:
        Y = read_object('Y_full_'+str(scale)+'.pkl', dir_output)

    n = 'X_full_'+str(scale)+'.pkl'
    if (dir_output / n).is_file():
        pos_feature, _ = create_pos_feature(scale, 6, features)
        X = read_object('X_full_'+str(scale)+'.pkl', dir_output)
    else:
        X = None

    if values_per_class != 'full':
        Ytrain = Y[(Y[:,4] < allDates.index(maxDate)) & (np.isin(Y[:,3], trainCode))]
        Ytest = Y[((Y[:,4] >= allDates.index(maxDate)) & (np.isin(Y[:,3], trainCode))) |
                ((Y[:,4] > allDates.index('2017-12-31')) & (Y[:,4] < allDates.index('2022-12-31')) & (~np.isin(Y[:,3], trainCode)))]
        if X is not None:
            Xtest = X[((Y[:,4] >= allDates.index(maxDate)) & (np.isin(Y[:,3], trainCode))) |
                ((Y[:,4] > allDates.index('2017-12-31')) & (Y[:,4] < allDates.index('2022-12-31')) & (~np.isin(Y[:,3], trainCode)))]

        values_per_class = int(values_per_class)
        new_Y = []
        new_X = []
        for dept in departements:
            maskdept = np.argwhere(Ytrain[:, 3] == name2int[dept])
            if maskdept.shape[0] == 0:
                continue
            classs = np.unique(Ytrain[maskdept, -3])
            new_Y_index = []
            for cls in classs:
                Y_class_index = np.where(Ytrain[maskdept, -3] == cls)[0]
                new_Y_index += list(np.random.choice(Y_class_index, min(Y_class_index.shape[0], values_per_class)))
            new_Y += list(Y[np.asarray(new_Y_index), :])
            if X is not None:
                new_X += list(X[np.asarray(new_Y_index), :])

        new_Y = np.asarray(new_Y)
        new_Y_neighboor = Y[np.isin(Y[:, 4], new_Y[:, 4])]
        mask_remove = np.argwhere(~np.isin(new_Y_neighboor, new_Y))[:,0]
        new_Y_neighboor = new_Y_neighboor[mask_remove]
        new_Y_neighboor[:, 5] = 0

        if X is not None:
            new_X_neighboor = X[np.isin(Y[:, 4], new_Y[:, 4])]
            new_X_neighboor = new_X_neighboor[mask_remove]
            X = np.concatenate((new_X, new_X_neighboor, Xtest))

        Y = np.concatenate((new_Y, new_Y_neighboor, Ytest))

    if X is None:
        if graphScale.scale == 0:
            X, pos_feature = get_sub_nodes_feature_2(graphScale, Y[:, :6], departements, features, sinister, dir_output, dir_train, resolution)
        else:
            X, pos_feature = get_sub_nodes_feature(graphScale, Y[:, :6], departements, features, sinister, dir_output, dir_train, resolution)

    logger.info(f'{X.shape, Y.shape}')
    X[:, 5] = Y[:, 5]

    return X, Y, pos_feature

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