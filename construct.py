import pickle
from sklearn.cluster import KMeans
from graph_structure import GraphStructure
from copy import copy
from features import *
from features_2D import *
from weigh_predictor import Predictor
from config import *

def look_for_information(path : Path, departements : list, firepoint : pd.DataFrame):
    points = []
    all_dep_predictor = Predictor(5)
    all_maxi_influence = []
    for departement in departements:
        print(departement)
        maxInfluence = []
        name = departement+'Influence.pkl'
        influence = pickle.load(open(path / name, 'rb'))
        influence = influence[:,:,allDates.index('2017-06-17'):allDates.index('2023-01-01')]
        print(influence.shape)
        datesForDepartement = []
        for di in range(influence.shape[-1]):
            maxInfluence.append(np.nanmax(influence[:,:,di]))
        
        all_maxi_influence += maxInfluence

        predictor = Predictor(5)
        predictor.fit(np.asarray(influence[~np.isnan(influence)]))
        save_object(predictor, departement+'PredictorFull.pkl', path=Path('influenceClustering'))

        predictor = Predictor(5)
        predictor.fit(np.asarray(maxInfluence))
        save_object(predictor, departement+'Predictor.pkl', path=Path('influenceClustering'))

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
            minValue = min(minPoint, mask.shape[0])
            m = np.random.choice(mask, minValue, replace=False).ravel()
            datesForDepartement += list(m)
        datesForDepartement = np.asarray(datesForDepartement)
        print(np.argwhere(datesForDepartement >= allDates.index('2023-01-01')).shape)
        fp = firepoint[firepoint['departement'] == name2int[departement]].copy(deep=True)
        print(fp.departement.unique())
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
    save_object(all_dep_predictor, 'PredictorAll.pkl', path=Path('influenceClustering'))

    points = pd.concat(points)
    print(len(points))
    points.to_csv('points'+str(minPoint)+'.csv', index=False)

def construct_graph(scale, maxDist, geo, nmax, k_days, dir_output):
    graphScale = GraphStructure(scale, geo, maxDist, nmax)
    graphScale._train_kmeans(doRaster=True, path=dir_output)
    graphScale._create_nodes_list()
    graphScale._create_edges_list()
    graphScale._create_temporal_edges_list(allDates, k_days=k_days)
    graphScale.nodes = graphScale._assign_department(graphScale.nodes)
    save_object(graphScale, 'graph_'+str(scale)+'.pkl', dir_output)
    return graphScale

def construct_database(graphScale, ps, scale, k_days,
                       departements, features,
                       dir_output, prefix):
    

    X_kmeans = list(zip(ps.longitude, ps.latitude))
    ps['scale'+str(scale)] = graphScale._predict_node(X_kmeans)
    ps.to_csv('test.csv', index=False)
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
    save_object(Y, 'Y_'+prefix+'.pkl', dir_output)

    subNode[:,-1] = Y[:,-3]

    pos_feature, _ = create_pos_feature(graphScale, subNode.shape[1], features)
    X, pos_feature = get_sub_nodes_feature(graphScale, subNode, departements, features, dir_output)
    save_object(X, 'X_'+prefix+'.pkl', dir_output)

    
    return X, Y, pos_feature

if __name__ == '__main__':
    allDates = find_dates_between('2017-06-12', '2023-09-11')
    minPoint = 100

    path = Path('/home/caron/Bureau/Model/HexagonalScale/log')
    fp = pd.read_csv('../CNN/STA/firepoints.csv')
    #fp = fp[fp['date'] < '2023-09-11']

    look_for_information(path, ['departement-01-ain', 'departement-25-doubs', 'departement-78-yvelines', 'departement-69-rhone'], fp)