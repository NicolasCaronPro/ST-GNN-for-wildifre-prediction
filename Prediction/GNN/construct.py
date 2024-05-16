import pickle
from sklearn.cluster import KMeans
from graph_structure import GraphStructure
from copy import copy
from features import *
from features_2D import *
from weigh_predictor import Predictor
from config import *

def look_for_information(path : Path, departements : list,
                         firepoint : pd.DataFrame, maxDate : str,
                         sinister : str, dir : Path,
                         minPoint):

    points = []
    all_dep_predictor = Predictor(5)
    all_maxi_influence = []
    for departement in departements:
        print(departement)
        maxInfluence = []
        name = departement+'Influence.pkl'
        influence = read_object(name, path)

        if influence is None:
            continue

        influence = influence[:,:,allDates.index('2017-06-17'):allDates.index(maxDate)]
        print(influence.shape)
        datesForDepartement = []
        for di in range(influence.shape[-1]):
            maxInfluence.append(np.nanmax(influence[:,:,di]))
        
        all_maxi_influence += maxInfluence

        predictor = Predictor(5)
        predictor.fit(np.asarray(influence[~np.isnan(influence)]))
        save_object(predictor, departement+'PredictorFull.pkl', path=dir / sinister / 'influenceClustering')

        predictor = Predictor(5)
        predictor.fit(np.asarray(maxInfluence))
        save_object(predictor, departement+'Predictor.pkl', path=dir / sinister / 'influenceClustering')

        clusterInfluence = predictor.predict(np.asarray(maxInfluence))
        predictor.log()
        classes = np.unique(clusterInfluence)
        histogram = np.bincount(clusterInfluence)
        histogram = histogram.astype(float)
        histogram = (histogram * 0.8).astype(int)
        for c in classes:
            mask = np.argwhere(clusterInfluence == c)[:,0]
            if departement == 'departement-01-ain':
                mask = mask[mask > allDates.index('2018-01-01')]
            if minPoint == 'full':
                minValue = mask.shape[0]
            else:
                minValue = min(minPoint, mask.shape[0])
            m = np.random.choice(mask, minValue, replace=False).ravel()
            datesForDepartement += list(m)
        datesForDepartement = np.asarray(datesForDepartement)
        
        fp = firepoint[firepoint['departement'] == name2int[departement]].copy(deep=True)

        iterables = [datesForDepartement, list(fp.latitude)]
        ps = pd.DataFrame(index=pd.MultiIndex.from_product(iterables, names=('date', 'latitude'))).reset_index()

        iterables = [datesForDepartement, list(fp.longitude)]
        ps2 = pd.DataFrame(index=pd.MultiIndex.from_product(iterables, names=('date', 'longitude'))).reset_index()
        ps['longitude'] = ps2['longitude']
        del ps2

        iterables = [datesForDepartement, list(fp.departement)]
        ps2 = pd.DataFrame(index=pd.MultiIndex.from_product(iterables, names=('date', 'departement'))).reset_index()
        ps['departement'] = ps2['departement']
        del ps2

        points.append(ps)

    all_dep_predictor.fit(np.asarray(all_maxi_influence))
    save_object(all_dep_predictor, 'PredictorAll.pkl', path=dir / sinister / 'influenceClustering')

    points = pd.concat(points)
    check_and_create_path(dir/sinister)
    name = 'points'+str(minPoint)+'.csv'
    points.to_csv(dir / sinister / name, index=False)

def construct_graph(scale, maxDist, sinister, geo, nmax, k_days, dir_output, doRaster):
    graphScale = GraphStructure(scale, geo, maxDist, nmax)
    graphScale._train_kmeans(doRaster=doRaster, path=dir_output, sinister=sinister)
    graphScale._create_nodes_list()
    graphScale._create_edges_list()
    graphScale._create_temporal_edges_list(allDates, k_days=k_days)
    graphScale.nodes = graphScale._assign_department(graphScale.nodes)
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
                       mode : str):
    
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

    graphScale._info_on_graph(subNode, Path('log'))

    Y = get_sub_nodes_ground_truth(graphScale, subNode, departements, orinode, dir_output)

    subNode[:,-1] = Y[:,-3]

    X, pos_feature = get_sub_nodes_feature(graphScale, subNode, departements, features, sinister, dir_output, dir_train)

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