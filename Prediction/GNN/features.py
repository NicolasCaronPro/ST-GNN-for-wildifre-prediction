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

    logger.info(pos_feature)

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
        encoder_osmnx = read_object('encoder_osmnx.pkl', dir_encoder)
        encoder_foret = read_object('encoder_foret.pkl', dir_encoder)

    if 'Calendar' in features:
        size_calendar = len(calendar_variables)
        encoder_calendar = read_object('encoder_calendar.pkl', dir_encoder)

    if 'Geo' in features:
        encoder_geo = read_object('encoder_geo.pkl', dir_encoder)

    dir_mask = dir_train / 'raster' / '2x2'
    logging.info(f'Shape of X {X.shape}')
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

                stop_calendar = 11
                
                X[index, pos_feature['Calendar'] : pos_feature['Calendar'] + stop_calendar] = \
                        encoder_calendar.transform(np.moveaxis(X[index, pos_feature['Calendar'] : pos_feature['Calendar'] + stop_calendar], 1, 2).reshape(-1, stop_calendar)).values.reshape(-1, 1, stop_calendar)
                
                for ir in range(stop_calendar, size_calendar):
                    var_ir = calendar_variables[ir]
                    if var_ir == 'mean':
                        X[index, pos_feature['Calendar'] + ir] = np.mean(X[index, pos_feature['Calendar'] : pos_feature['Calendar'] + stop_calendar])
                    elif var_ir == 'max':
                        X[index, pos_feature['Calendar'] + ir] = np.max(X[index, pos_feature['Calendar'] : pos_feature['Calendar'] + stop_calendar])
                    elif var_ir == 'min':
                        X[index, pos_feature['Calendar'] + ir] = np.min(X[index, pos_feature['Calendar'] : pos_feature['Calendar'] + stop_calendar])
                    elif var_ir == 'sum':
                        X[index, pos_feature['Calendar'] + ir] = np.sum(X[index, pos_feature['Calendar'] : pos_feature['Calendar'] + stop_calendar])
                    else:
                        logger.info(f'Unknow operation {var_ir}')
                        exit(1)
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

        if 'dynamicWorld' in features:
            arrayDW = read_object('dynamic_world.pkl', dir_data)

        ### OSMNX
        if 'highway' in features:
            name = 'osmnx.pkl'
            arrayOS = read_object(name, dir_data)

        if 'foret' in features:
            logger.info('Foret')
            name = 'foret.pkl'
            arrayForet = read_object(name, dir_data)

        if 'landcover' in features:
            if 'foret' in landcover_variables:
                logger.info('Foret landcover')
                name = 'foret_landcover.pkl'
                arrayForetLandcover = read_object(name, dir_data)

            if 'landcover' in landcover_variables:
                logger.info('Dynamic World landcover')
                name = 'dynamic_world_landcover.pkl'
                arrayLand = read_object(name, dir_data)

            if 'highway' in landcover_variables:
                logger.info('OSMNX landcover')
                name = 'osmnx_landcover.pkl'
                arrayOSLand = read_object(name, dir_data)

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
                    for band in osmnx_variables:
                        save_values(arrayOS[int(band), :, :], pos_feature['highway'] + (osmnx_variables.index(band) * 4), index, maskNode)

                if 'foret' in features:
                    for band in foret_variables:
                        save_values(arrayForet[int(band), :, :], pos_feature['foret'] + (foret_variables.index(band) * 4), index, maskNode)

                if 'landcover' in features:
                    if 'foret' in landcover_variables:
                        save_value_with_encoding(arrayForetLandcover, pos_feature['landcover'] + (landcover_variables.index('foret') * 4), index, maskNode, encoder_foret)
                    if 'highway' in landcover_variables:
                        save_value_with_encoding(arrayOSLand, pos_feature['landcover'] + (landcover_variables.index('highway') * 4), index, maskNode, encoder_osmnx)

        ### Sentinel
        if 'sentinel' in features:
            logger.info('Sentinel')
            name = 'sentinel.pkl'
            arraySent = read_object(name, dir_data)

        coef = 4 if graph.scale > -1 else 1
        if arraySent is not None or arrayLand is not None:
            for node in nodeDepartement:
                maskNode = mask == node[0]
                index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))

                if 'sentinel' in features:
                    for band, var in enumerate(sentinel_variables):
                        save_values(arraySent[band,:, :,int(node[4])], pos_feature['sentinel'] + (sentinel_variables.index(var) * coef) , index, maskNode)
                
                if 'landcover' in features:
                    if 'landcover' in landcover_variables:
                        save_value_with_encoding(arrayLand[:,:], pos_feature['landcover'] + (landcover_variables.index('landcover') * 4), index, maskNode, encoder_landcover)

                if 'dynamicWorld' in features:
                    for band in dynamic_world_variables:
                        save_values(arrayDW[i], pos_feature['dynamicWorld'] + (dynamic_world_variables.index(band) * 4), index, maskNode)

        del arrayPop
        del arrayEl
        del arrayOS
        del arraySent
        del arrayLand

        logger.info('Historical')
        if 'Historical' in features:
            dir_target = root_target / sinister / 'log'
            name = departement+'pastInfluence.pkl'
            arrayInfluence = read_object(name, dir_target)
            for node in nodeDepartement:
                index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                maskNode = mask == node[0]
                if node[4] - 1 < 0:
                    continue

                save_values(arrayInfluence[:,:, int(node[4])], pos_feature['Historical'], index, maskNode)
            del arrayInfluence

        logger.info('AutoRegression')
        if 'AutoRegression' in features:
            dir_target = path / 'influence' / '2x2'
            dir_bin = path / 'bin' / '2x2'
            arrayInfluence = read_object(departement+'InfluenceScale'+str(graph.scale)+'.pkl', dir_target)
            arrayBin = read_object(departement+'binScale'+str(graph.scale)+'.pkl', dir_bin)
            for node in nodeDepartement:
                index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                maskNode = mask == node[0]
                if node[4] - 1 < 0:
                    continue
                
                for band in auto_regression_variable_reg:
                    step = int(band.split('-')[-1])
                    save_value(arrayInfluence[:,:, int(node[4] - step)], pos_feature['AutoRegression'] + auto_regression_variable_reg.index(band), index, maskNode)
                
                for band in auto_regression_variable_bin:
                    step = int(band.split('-')[-1])
                    save_value(arrayBin[:,:, int(node[4] - step)], pos_feature['AutoRegression'] + auto_regression_variable_bin.index(band), index, maskNode)

            del arrayInfluence
            del arrayBin

        logger.info('Vigicrues')
        if 'vigicrues' in features:
            for var in vigicrues_variables:
                array = read_object('vigicrues'+var+'.pkl', dir_data)
                for node in nodeDepartement:
                    index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                    maskNode = mask == node[0]
                    if node[4] - 1 < 0:
                        continue

                    save_values(array[:,:, int(node[4])], pos_feature['vigicrues'] + (4 * vigicrues_variables.index(var)), index, maskNode)
                del array

        logger.info('nappes')
        if 'nappes' in features:
            for var in nappes_variables:
                array = read_object(var+'.pkl', dir_data)
                for node in nodeDepartement:
                    index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                    maskNode = mask == node[0]
                    if node[4] - 1 < 0:
                        continue

                    save_values(array[:,:, int(node[4])], pos_feature['nappes'] + (4 * nappes_variables.index(var)), index, maskNode)
                del array

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