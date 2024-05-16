from pathlib import Path
from shapely import unary_union
from tools import *
from arborescence import *
from weigh_predictor import Predictor
from config import *
import itertools

def get_sub_nodes_ground_truth(graph, subNode: np.array,
                           departements : list,
                           oriNode : np.array, path : Path) -> np.array:

        assert graph.nodes is not None
        
        logger.info('Ground truth')

        newShape = subNode.shape[1] + 2
        Y = np.full((subNode.shape[0], newShape), -1, dtype=float)
        Y[:, :subNode.shape[1]] = subNode
        codes = []
            
        dir_mask = path / 'raster' / '2x2'
        dir_bin = path / 'bin' / '2x2'
        dir_proba = path / 'influence' / '2x2'
        dir_predictor = path / 'influenceClustering'

        for departement in departements:

            nodeDepartementMask = np.argwhere(subNode[:,3] == str2int[departement.split('-')[-1]])
            nodeDepartement = subNode[nodeDepartementMask].reshape(-1, subNode.shape[1])
            codes.append(str2int[departement.split('-')[-1]])

            mask = read_object(departement+'rasterScale'+str(graph.scale)+'.pkl', dir_mask)
            array = read_object(departement+'InfluenceScale'+str(graph.scale)+'.pkl', dir_proba)
            arrayBin = read_object(departement+'binScale'+str(graph.scale)+'.pkl', dir_bin)

            for node in nodeDepartement:
                index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                indexOri = np.argwhere((oriNode[:,0] == node[0]) & (oriNode[:,4] == node[4]))
                maskNode = mask == node[0]
                if node[4] + 1 >= array.shape[-1]:
                #################
                #               #
                #               #
                #               #
                    continue    #  <----------- SALOPERIE DE CONTINUE QUI M'A FAIT PERDRE 2 JOURS
                #               #
                #               #
                #               #
                #################

                if array[:, :, int(node[4] + 1)][maskNode].shape[0] != 0:
                    Y[index, -3] = 1

                    # Influence
                    values = np.unique(array[maskNode, int(node[4] + 1)])[0]
                    Y[index, -1] = round(values, 3)

                    # Nb events
                    values = (np.unique(arrayBin[maskNode, int(node[4] + 1)]))[0]
                    Y[index, -2] = values
            
            predictor = read_object(departement+'Predictor'+str(graph.scale)+'.pkl', dir_predictor)
            mask = np.argwhere((Y[:,3] == str2int[departement.split('-')[-1]]) & (Y[:,-3] == 1))[:,0]
            if mask.shape[0] != 0:
                influence = Y[mask, -1]
                Y[mask, -3] = predictor.weight_array(predictor.predict(influence)).reshape(-1)

        Y[~np.isin(Y[:,3], codes)][:,-3] = 0

        return Y

def get_sub_nodes_feature(graph, subNode: np.array,
                        departements : list,
                        features : list, sinister : str,
                        path : Path,
                        dir_train : Path) -> np.array:
    
    assert graph.nodes is not None

    logger.info('Load nodes features')

    pos_feature, newShape = create_pos_feature(graph, subNode.shape[1], features)

    def save_values(array, indexVvar, indexNode, mask):
        if False not in np.unique(np.isnan(array[mask])):
            return
        if graph.scale > -1:
            X[indexNode, indexVvar] = round(np.nanmean(array[mask]), 3)
            X[indexNode, indexVvar+1] = round(np.nanmin(array[mask]), 3)
            X[indexNode, indexVvar+2] = round(np.nanmax(array[mask]), 3)
            X[indexNode, indexVvar+3] = round(np.nanstd(array[mask]), 3)
        else:
            X[indexNode, indexVvar] = np.nanmean(array[mask])

    def save_value(array, indexVvar, indexNode, mask):
        if False not in np.unique(np.isnan(array[mask])):
            return
        X[indexNode, indexVvar] = np.nanmean(array[mask])

    def save_values_count(array, lc, indexVvar, indexNode, mask):
        X[indexNode, indexVvar] = np.argwhere(array[mask] == float(lc)).shape[0]

    def save_value_with_encoding(array, indexVvar, indexNode, mask, encoder):
        values = array[mask].reshape(-1,1)
        encode_values = encoder.transform(values).values
        if graph.scale > -1:
            X[indexNode, indexVvar] = round(np.nanmean(encode_values), 3)
            X[indexNode, indexVvar+1] = round(np.nanmin(encode_values), 3)
            X[indexNode, indexVvar+2] = round(np.nanmax(encode_values), 3)
            X[indexNode, indexVvar+3] = round(np.nanstd(encode_values), 3)
        else:
            X[indexNode, indexVvar] = np.nanmean(encode_values)

    X = np.full((subNode.shape[0], newShape), np.nan, dtype=float)

    X[:,:subNode.shape[1]] = subNode

    dir_encoder = dir_train / 'Encoder'

    if 'landcover' in features:
        encoder_landcover = read_object('encoder_landcover.pkl', dir_encoder)

    if 'foret' in features:
        encoder_foret = read_object('encoder_foret.pkl', dir_encoder)

    if 'Calendar' in features:
        size_calendar = len(calendar_variables)
        encoder_calendar = read_object('encoder_calendar.pkl', dir_encoder)

    if 'Geo' in features:
        encoder_geo = read_object('encoder_geo.pkl', dir_encoder)

    dir_mask = path / 'raster' / '2x2'

    for departement in departements:
        logger.info(departement)

        dir_data = root / 'csv' / departement / 'raster'
        
        name = departement+'rasterScale'+str(graph.scale)+'.pkl'
        mask = pickle.load(open(dir_mask / name, 'rb'))
        nodeDepartementMask = np.argwhere(subNode[:,3] == str2int[departement.split('-')[-1]])
        nodeDepartement = subNode[nodeDepartementMask].reshape(-1, subNode.shape[1])
        logger.info(nodeDepartement.shape)
        if nodeDepartement.shape[0] == 0:
            continue

        logger.info('Calendar')
        if 'Calendar' in features:
            unDate = np.unique(nodeDepartement[:,4]).astype(int)
            for unDate in unDate:
                date = allDates[unDate]
                ddate = dt.datetime.strptime(date, '%Y-%m-%d')
                index = np.argwhere((subNode[:,4] == unDate))

                X[index, pos_feature['Calendar']] = int(date.split('-')[1]) # month
                X[index, pos_feature['Calendar'] + 1] = ddate.timetuple().tm_yday # dayofweek
                X[index, pos_feature['Calendar'] + 2] = ddate.weekday() # dayofyear
                X[index, pos_feature['Calendar'] + 3] = ddate.weekday() >= 5 # isweekend
                X[index, pos_feature['Calendar'] + 4] = pendant_couvrefeux(ddate) # couvrefeux
                X[index, pos_feature['Calendar'] + 5] = (1 if dt.datetime(2020, 3, 17, 12) <= ddate <= dt.datetime(2020, 5, 11) else 0) or 1 if dt.datetime(2020, 10, 30) <= ddate <= dt.datetime(2020, 12, 15) else 0# confinement
                X[index, pos_feature['Calendar'] + 6] = 1 if convertdate.islamic.from_gregorian(ddate.year, ddate.month, ddate.day)[1] == 9 else 0 # ramadan
                X[index, pos_feature['Calendar'] + 7] = 1 if ddate in jours_feries else 0 # bankHolidays
                X[index, pos_feature['Calendar'] + 8] = 1 if ddate in veille_jours_feries else 0 # bankHolidaysEve
                X[index, pos_feature['Calendar'] + 9] = 1 if vacances_scolaire.is_holiday_for_zone(ddate.date(), get_academic_zone(ACADEMIES[str(name2int[departement])], ddate)) else 0 # holidays
                X[index, pos_feature['Calendar'] + 10] = (1 if vacances_scolaire.is_holiday_for_zone(ddate.date() + dt.timedelta(days=1), get_academic_zone(ACADEMIES[str(name2int[departement])], ddate)) else 0 ) \
                    or (1 if vacances_scolaire.is_holiday_for_zone(ddate.date() - dt.timedelta(days=1), get_academic_zone(ACADEMIES[str(name2int[departement])], ddate)) else 0) # holidaysBorder

                #logger.info(X[index, pos_feature['Calendar']])
                X[index, pos_feature['Calendar'] : pos_feature['Calendar'] + size_calendar] = \
                        encoder_calendar.transform(np.moveaxis(X[index, pos_feature['Calendar'] : pos_feature['Calendar'] + size_calendar], 1, 2).reshape(-1, size_calendar)).values.reshape(-1, 1, size_calendar)
                #logger.info(X[index, pos_feature['Calendar']])
                #logger.info('#######################################')

        ### Geo spatial
        logger.info('Geo')
        if 'Geo' in features:
            X[nodeDepartementMask, pos_feature['Geo']] = encoder_geo.transform([name2int[departement]]).values[0] # departement

        logger.info('Meteorological')
        array = None
        ### Meteo
        for i, var in enumerate(cems_variables):
            if var not in features:
                continue
            logger.info(var)
            name = var +'raw.pkl'
            array = read_object(name, dir_data)
            for node in nodeDepartement:
                maskNode = mask == node[0]
                if True not in maskNode:
                    continue
                index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                save_values(array[:,:,int(node[4])], pos_feature[var], index, maskNode)

        del array

        logger.info('Air Quality')
        if 'air' in features:
            for i, var in enumerate(air_variables):
                logger.info(var)
                name = var +'raw.pkl'
                array = read_object(name, dir_data)
                for node in nodeDepartement:
                    maskNode = mask == node[0]
                    if True not in maskNode:
                        continue
                    index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                    save_value(array[:,:,int(node[4])], pos_feature['air'] + i, index, maskNode)

        logger.info('Population elevation Highway Sentinel')

        arrayPop = None
        arrayEl = None
        arrayOS = None
        arrayLand = None
        arraySent = None
        arrayForet = None

        ### Population
        if 'population' in features:
            name = 'population.pkl'
            arrayPop = read_object(name, dir_data)

        ### Elevation
        if 'elevation' in features: 
            name = 'elevation.pkl'
            arrayEl = read_object(name, dir_data)

        ### OSMNX
        if 'highway' in features:
            name = 'osmnx.pkl'
            arrayOS = read_object(name, dir_data)

        if 'foret' in features:
            logger.info('Foret')
            name = 'foret.pkl'
            arrayForet = read_object(name, dir_data)
            arrayForet = resize_no_dim(arrayForet, mask.shape[0], mask.shape[1])

        if arrayPop is not None or arrayEl is not None or arrayOS is not None or arrayForet is not None:
            unode = np.unique(nodeDepartement[:,0])
            for node in unode:
                maskNode = mask == node
                index = np.argwhere(subNode[:,0] == node)
                if 'population' in features:
                    save_values(arrayPop, pos_feature['population'], index, maskNode)
                if 'elevation' in features:
                    save_values(arrayEl, pos_feature['elevation'], index, maskNode)
                if 'highway' in features:
                    save_values(arrayOS, pos_feature['highway'], index, maskNode)
                if 'foret' in features: 
                    save_value_with_encoding(arrayForet, pos_feature['foret'], index, maskNode, encoder_foret)

        ### Sentinel
        if 'sentinel' in features:
            logger.info('Sentinel')
            name = 'sentinel.pkl'
            arraySent = read_object(name, dir_data)

        ### Landcover
        if 'landcover' in features:
            logger.info('landcover')
            name = 'landcover.pkl'
            arrayLand = read_object(name, dir_data)
            arrayLand[~np.isnan(arrayLand)] = np.round(arrayLand[~np.isnan(arrayLand)])

        coef = 4 if graph.scale > -1 else 1
        if arraySent is not None or arrayLand is not None:
            for node in nodeDepartement:
                maskNode = mask == node[0]
                index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                if 'sentinel' in features:
                    for band, var in enumerate(sentinel_variables):
                        save_values(arraySent[band,:, :,int(node[4])], pos_feature['sentinel'] + (sentinel_variables.index(var) * coef) , index, maskNode)

                if 'landcover' in features: 
                    save_value_with_encoding(arrayLand[:,:,int(node[4])], pos_feature['landcover'], index, maskNode, encoder_landcover)

        del arrayPop
        del arrayEl
        del arrayOS
        del arraySent
        del arrayLand

        logger.info('Historical')
        if 'Historical' in features:
            dir_target = root_target / sinister / 'log'
            name = departement+'pastInfluence.pkl'
            arrayInfluence = pickle.load(open(dir_target / name, 'rb'))
            for node in nodeDepartement:
                index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                maskNode = mask == node[0]
                if node[4] - 1 < 0:
                    continue

                save_values(arrayInfluence[:,:, int(node[4] - 1)], pos_feature['Historical'], index, maskNode)

            del arrayInfluence

    return X, pos_feature

def add_varying_time_features(X : np.array, features : list, newShape : int, pos_feature : dict, ks : int):
    res = np.empty((X.shape[0], newShape))
    res[:, :X.shape[1]] = X
    for feat in features:
        logger.info(feat)
        vec = feat.split('_')
        name = vec[0]
        methods = vec[1:]
        index_feat = pos_feature[feat]
        index_feat_X = pos_feature[name]
        for met in methods:
            if met == "mean":
                for node in X:
                    index_node = np.argwhere((X[:,0] == node[0]) & (X[:,4] == node[4]))
                    index_nodes_ks = np.argwhere((X[:,0] == node[0]) & (X[:,4] < node[4]) & (X[:,4] >= node[4] - ks))
                    if name == 'Calendar':
                        for i, _ in enumerate(calendar_variables):
                            res[index_node, index_feat + i] = np.mean(X[index_nodes_ks, index_feat_X + i])
                    elif name == 'Historical':
                        for i, _ in enumerate(historical_variables):
                            res[index_node, index_feat + i] = np.mean(X[index_nodes_ks, index_feat_X + i])
                    elif name == "air":
                        for i, _ in enumerate(air_variables):
                            res[index_node, index_feat + i] = np.mean(X[index_nodes_ks, index_feat_X + i])
                    else:
                        res[index_node, index_feat] = np.mean(X[index_nodes_ks, index_feat_X])
            elif met == "sum":
                    index_node = np.argwhere((X[:,0] == node[0]) & (X[:,4] == node[4]))
                    index_nodes_ks = np.argwhere((X[:,0] == node[0]) & (X[:,4] < node[4]) & (X[:,4] >= node[4] - ks))
                    res[index_node, index_feat] = np.sum(X[index_nodes_ks, index_feat_X])
            else:
                logger.info(f'{met} unknow method')
                exit(1)
    return res

def get_edges_feature(graph, subNode: np.array,
                           newAxis : list) -> np.array:
        """ 
        """
        assert graph.edges is not None
        
        if newAxis == []:
            return

        edges = np.repeat(graph.edges[:, np.newaxis], k = len(newAxis))

        n_pixel_x = 0.02875215641173088
        n_pixel_y = 0.020721094073767096

        if 'slope' in newAxis:
            for node in subNode:
                edgesNode = graph.edges[graph.edges[0] == node[0]]
                altitude = graph.dataset[graph.dataset['id'] == node[0]]['elevation_mean']
                for ed in edgesNode:
                    targetAltitude = graph.dataset[graph.dataset['id'] == ed[1]]['elevation_mean']
                    slope = funcSlope(altitude, targetAltitude, (node[1], node[2]), (graph.nodes[ed[1]][1], graph.nodes[ed[1]][2]))
                    edges[:,2] = slope

        if 'highway' in newAxis:
            for node in subNode:
                edgesNode = graph.edges[graph.edges[0] == node[0]]
                maskNode = np.argwhere(graph.ids == node[0])
                maskTarget = np.argwhere(graph.ids in edgesNode[1])
                geom = unary_union(np.concatenate(graph.oriGeometry[maskNode[:,0]], graph.oriGeometry[maskTarget[:,0]]).ravel())
                osmnx = ox.graph_from_polygon(geom, network_type='drive')
                osmnx = osmnx[osmnx['highway'].isin(['motorway', 'primary', 'secondary', 'tertiary'])]
                osmnx['label'] = 0
                for i, hw in enumerate(['motorway', 'primary', 'secondary', 'tertiary']):
                    osmnx.loc[osmnx['highway'] == hw, 'label'] = i +1
                mask = rasterization(geom, n_pixel_y, n_pixel_x, 'id', Path('.'), 'geo')
                osmnx = rasterization(osmnx, n_pixel_y, n_pixel_x, 'label', Path('.'), 'highway')

            for ed in edgesNode:
                edges[:,3:6] = stat(node[0], ed[1], mask, osmnx)
        return edges