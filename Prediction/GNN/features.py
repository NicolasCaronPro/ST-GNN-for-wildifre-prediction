from pathlib import Path
from shapely import unary_union
from arborescence import *
from weigh_predictor import Predictor
import itertools
from config import *
from tools import *

def get_sub_nodes_ground_truth(graph, subNode: np.array,
                           departements : list,
                           oriNode : np.array, path : Path,
                           dir_train : Path,
                           resolution  : str,
                           graph_construct : str) -> np.array:

        assert graph.nodes is not None
        
        logger.info('Ground truth')

        newShape = subNode.shape[1] + 3
        Y = np.full((subNode.shape[0], newShape), -1, dtype=float)
        Y[:, :subNode.shape[1]] = subNode
        codes = []
            
        dir_mask = path / 'raster'
        dir_bin = path / 'bin'
        dir_proba = path / 'influence'
        dir_predictor = dir_train / 'influenceClustering'

        for departement in departements:

            nodeDepartementMask = np.argwhere(subNode[:,3] == str2int[departement.split('-')[-1]])
            nodeDepartement = subNode[nodeDepartementMask].reshape(-1, subNode.shape[1])
            codes.append(str2int[departement.split('-')[-1]])

            mask = read_object(f'{departement}rasterScale{graph.scale}_{graph_construct}.pkl', dir_mask)
            array = read_object(f'{departement}InfluenceScale{graph.scale}_{graph_construct}.pkl', dir_proba)
            arrayBin = read_object(f'{departement}binScale{graph.scale}_{graph_construct}.pkl', dir_bin)

            if array is None:
                Y[nodeDepartementMask, -4:] = -1
                continue

            predictor = read_object(f'{departement}Predictor{graph.scale}_{graph_construct}.pkl', dir_predictor)

            for node in nodeDepartement:
                index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                maskNode = mask == node[0]
                if node[4] >= array.shape[-1]:
                #################
                #               #
                #               #
                #               #
                    continue    #  <----------- SALOPERIE DE CONTINUE QUI M'A FAIT PERDRE 2 JOURS
                #               #
                #               #
                #               #
                #################

                if array[:, :, int(node[4])][maskNode].shape[0] != 0:
                    # Influence
                    values = np.unique(array[maskNode, int(node[4])])[0]
                    Y[index, -1] = values

                    # Nb events
                    values = (np.unique(arrayBin[maskNode, int(node[4])]))[0]
                    Y[index, -2] = values

                    # Class
                    Y[index, -3] = predictor.predict(np.asarray(Y[index, -1])).reshape(-1)

                    # Weights
                    Y[index, -4] = predictor.weight_array(np.asarray(Y[index, -3], dtype=int)).reshape(-1)

            Y[nodeDepartementMask, -3] = order_class(predictor, Y[nodeDepartementMask, -3]).reshape(-1, 1)

        Y[~np.isin(Y[:,4], codes)][:,-4] = 0
        Y = Y[Y[:, -4] > 0]

        return Y

def get_sub_nodes_feature(graph, subNode: np.array,
                        departements : list,
                        features : list, sinister : str,
                        path : Path,
                        dir_train : Path,
                        resolution : str, 
                        graph_construct : str) -> np.array:
    
    assert graph.nodes is not None

    logger.info('Load nodes features')

    methods = METHODS_SPATIAL

    features_name, newShape = get_features_name_list(graph.scale, features, methods)

    def save_values(array, band, indexNode, mask):

        indexVar = features_name.index(f'{band}_mean')

        if False not in np.unique(np.isnan(array[mask])):
            return

        values = array[mask].reshape(-1,1)
        if graph.scale > 0:
            for imet, metstr in enumerate(methods):
                if metstr == 'mean':
                    X[indexNode[:, 0], indexVar+imet] = round(np.nanmean(values), 3)
                elif metstr == 'min':
                    X[indexNode[:, 0], indexVar+imet] = round(np.nanmin(values), 3)
                elif metstr == 'max':
                    X[indexNode[:, 0], indexVar+imet] = round(np.nanmax(values), 3)
                elif metstr == 'std':
                    X[indexNode[:, 0], indexVar+imet] = round(np.nanstd(values), 3)
                elif metstr == 'sum':
                    X[indexNode[:, 0], indexVar+imet] = round(np.nansum(values), 3)
                elif metstr == 'grad':
                    X[indexNode[:, 0], indexVar+imet] = round(np.gradient(values), 3)
                else:
                    raise ValueError(f'Unknow {metstr}')
        else:
            X[indexNode[:, 0], indexVar] = np.nanmean(values)

    def save_value(array, band, indexNode, mask):
        if False not in np.unique(np.isnan(array[mask])):
            return

        indexVar = features_name.index(f'{band}')
        
        X[indexNode[:, 0], indexVar] = np.nanmean(array[mask])

    def save_value_with_encoding(array, band, indexNode, mask, encoder):
        values = array[mask].reshape(-1,1)
        encode_values = encoder.transform(values).values
        indexVar = features_name.index(f'{band}_mean')
        if graph.scale > 0:
            for imet, metstr in enumerate(methods):
                if metstr == 'mean':
                    X[indexNode[:, 0], indexVar+imet] = round(np.nanmean(encode_values), 3)
                elif metstr == 'min':
                    X[indexNode[:, 0], indexVar+imet] = round(np.nanmin(encode_values), 3)
                elif metstr == 'max':
                    X[indexNode[:, 0], indexVar+imet] = round(np.nanmax(encode_values), 3)
                elif metstr == 'std':
                    X[indexNode[:, 0], indexVar+imet] = round(np.nanstd(encode_values), 3)
                elif metstr == 'sum':
                    X[indexNode[:, 0], indexVar+imet] = round(np.nansum(encode_values), 3)
                elif metstr == 'grad':
                    X[indexNode[:, 0], indexVar+imet] = round(np.gradient(encode_values), 3)
                else:
                    raise ValueError(f'Unknow {metstr}')
        else:
            X[indexNode[:, 0], indexVar] = np.nanmean(encode_values)

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

    dir_mask = path / 'raster'
    logging.info(f'Shape of X {X.shape}, {np.unique(X[:,3])}')
    for departement in departements:
        logger.info(departement)

        dir_data = root / 'csv' / departement / 'raster' / resolution
        dir_target = root_target / sinister / 'log' / resolution

        name = f'{departement}rasterScale{graph.scale}_{graph_construct}.pkl'
        mask = read_object(name, dir_mask)
        print(np.unique(mask))
        
        nodeDepartementMask = np.argwhere(subNode[:,3] == str2int[departement.split('-')[-1]])
        nodeDepartement = subNode[nodeDepartementMask].reshape(-1, subNode.shape[1])
        logger.info(nodeDepartement.shape)
        if nodeDepartement.shape[0] == 0:
            continue
        logger.info('Calendar')
        if 'Calendar' in features:
            unDate = np.unique(nodeDepartement[:,4]).astype(int)
            band = calendar_variables[0]
            for unDate in unDate:
                date = allDates[unDate]
                ddate = dt.datetime.strptime(date, '%Y-%m-%d')
                index = np.argwhere((subNode[:,4] == unDate))
                X[index, features_name.index(band)] = int(date.split('-')[1]) # month
                X[index, features_name.index(band) + 1] = ajuster_jour_annee(ddate, ddate.timetuple().tm_yday) # dayofyear
                X[index, features_name.index(band) + 2] = ddate.weekday() # dayofweek
                X[index, features_name.index(band) + 3] = ddate.weekday() >= 5 # isweekend
                X[index, features_name.index(band) + 4] = pendant_couvrefeux(ddate) # couvrefeux
                X[index, features_name.index(band) + 5] = (1 if dt.datetime(2020, 3, 17, 12) <= ddate <= dt.datetime(2020, 5, 11) else 0) or 1 if dt.datetime(2020, 10, 30) <= ddate <= dt.datetime(2020, 12, 15) else 0# confinement
                X[index, features_name.index(band) + 6] = 1 if convertdate.islamic.from_gregorian(ddate.year, ddate.month, ddate.day)[1] == 9 else 0 # ramadan
                X[index, features_name.index(band) + 7] = 1 if ddate in jours_feries else 0 # bankHolidays
                X[index, features_name.index(band) + 8] = 1 if ddate in veille_jours_feries else 0 # bankHolidaysEve
                X[index, features_name.index(band) + 9] = 1 if vacances_scolaire.is_holiday_for_zone(ddate.date(), get_academic_zone(ACADEMIES[str(name2int[departement])], ddate)) else 0 # holidays
                X[index, features_name.index(band) + 10] = (1 if vacances_scolaire.is_holiday_for_zone(ddate.date() + dt.timedelta(days=1), get_academic_zone(ACADEMIES[str(name2int[departement])], ddate)) else 0 ) \
                    or (1 if vacances_scolaire.is_holiday_for_zone(ddate.date() - dt.timedelta(days=1), get_academic_zone(ACADEMIES[str(name2int[departement])], ddate)) else 0) # holidaysBorder

                stop_calendar = 11

                X[index, features_name.index(band) : features_name.index(band) + stop_calendar] = \
                        np.round(encoder_calendar.transform(np.moveaxis(X[index, features_name.index(band) : features_name.index(band) + stop_calendar], 1, 2).reshape(-1, stop_calendar)).values.reshape(-1, 1, stop_calendar), 3)
                
                
                for ir in range(stop_calendar, size_calendar):
                    var_ir = calendar_variables[ir]
                    if var_ir == 'calendar_mean':
                        X[index, features_name.index(band) + ir] = round(np.mean(X[index, features_name.index(band) : features_name.index(band) + stop_calendar]), 3)
                    elif var_ir == 'calendar_max':
                        X[index, features_name.index(band) + ir] = round(np.max(X[index, features_name.index(band) : features_name.index(band) + stop_calendar]), 3)
                    elif var_ir == 'calendar_min':
                        X[index, features_name.index(band) + ir] = round(np.min(X[index, features_name.index(band) : features_name.index(band) + stop_calendar]), 3)
                    elif var_ir == 'calendar_sum':
                        X[index, features_name.index(band) + ir] = round(np.sum(X[index, features_name.index(band) : features_name.index(band) + stop_calendar]), 3)
                    else:
                        logger.info(f'Unknow operation {var_ir}')
                        exit(1)
        ### Geo spatial
        logger.info('Geo')
        if 'Geo' in features:
            X[nodeDepartementMask, features_name.index(geo_variables[0])] = encoder_geo.transform([name2int[departement]]).values[0] # departement

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
                index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                save_values(array[:,:,int(node[4])], var, index, maskNode)

        del array

        logger.info('Air Quality')
        if 'air' in features:
            for i, var in enumerate(air_variables):
                logger.info(var)
                name = var +'raw.pkl'
                array = read_object(name, dir_data)
                for node in nodeDepartement:
                    maskNode = mask == node[0]
                    index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                    save_value(array[:,:,int(node[4])], var, index, maskNode)

        logger.info('Population elevation Highway Sentinel Foret')

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

        if 'landcover' in features:
            if 'foret_encoder' in landcover_variables:
                logger.info('Foret landcover')
                name = 'foret_landcover.pkl'
                arrayForetLandcover = read_object(name, dir_data)

            if 'landcover_encoder' in landcover_variables:
                logger.info('Dynamic World landcover')
                name = 'dynamic_world_landcover.pkl'
                arrayLand = read_object(name, dir_data)

            if 'highway_encoder' in landcover_variables:
                logger.info('OSMNX landcover')
                name = 'osmnx_landcover.pkl'
                arrayOSLand = read_object(name, dir_data)

        if arrayPop is not None or arrayEl is not None or arrayOS is not None or arrayForet is not None:
            unode = np.unique(nodeDepartement[:,0])
            for node in unode:
                maskNode = mask == node
                index = np.argwhere(subNode[:,0] == node)
                if 'population' in features:
                    save_values(arrayPop, 'population', index, maskNode)

                if 'elevation' in features:
                    save_values(arrayEl, 'elevation', index, maskNode)

                if 'highway' in features:
                    for var in osmnx_variables:
                        save_values(arrayOS[int(var), :, :], osmnxint2str[var], index, maskNode)

                if 'foret' in features:
                    for var in foret_variables:
                        save_values(arrayForet[int(var), :, :], foretint2str[var], index, maskNode)

                if 'landcover' in features:
                    if 'foret_encoder' in landcover_variables:
                        save_value_with_encoding(arrayForetLandcover, 'foret_encoder', index, maskNode, encoder_foret)
                    if 'highway_encoder' in landcover_variables:
                        save_value_with_encoding(arrayOSLand, 'highway_encoder', index, maskNode, encoder_osmnx)

        logger.info('Sentinel Dynamic World')
        ### Sentinel
        if 'sentinel' in features:
            logger.info('Sentinel')
            name = 'sentinel.pkl'
            arraySent = read_object(name, dir_data)

        if 'dynamicWorld' in features:
            arrayDW = read_object('dynamic_world.pkl', dir_data)
        else:
            arrayDW = None

        if arraySent is not None or arrayLand is not None or arrayDW is not None:
            for node in nodeDepartement:
                maskNode = mask == node[0]
                if True not in np.unique(maskNode):
                    continue
                index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))

                if 'sentinel' in features:
                    for band, var in enumerate(sentinel_variables):
                        save_values(arraySent[band,:, :,int(node[4])], var , index, maskNode)
                
                if 'landcover' in features:
                    if 'landcover_encoder' in landcover_variables:
                        save_value_with_encoding(arrayLand[:,:], 'landcover_encoder', index, maskNode, encoder_landcover)

                if 'dynamicWorld' in features:
                    for band, var in enumerate(dynamic_world_variables):
                        save_values(arrayDW[band, :, :, int(node[4])], var, index, maskNode)

        del arrayPop
        del arrayEl
        del arrayOS
        del arraySent
        del arrayLand

        logger.info('Historical')
        if 'Historical' in features:
            name = departement+'pastInfluence.pkl'
            arrayInfluence = read_object(name, dir_target)
            if arrayInfluence is not None:
                for node in nodeDepartement:
                    index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                    maskNode = mask == node[0]

                    save_values(arrayInfluence[:,:, int(node[4] - 1)], historical_variables[0], index, maskNode)
                del arrayInfluence

        logger.info('AutoRegressionReg')
        if 'AutoRegressionReg' in features:
            dir_target = dir_train / 'influence'
            arrayInfluence = read_object(f'{departement}InfluenceScale{graph.scale}_{graph_construct}.pkl', dir_target)
            if arrayInfluence is not None:
                for node in nodeDepartement:
                    index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                    maskNode = mask == node[0]
                    if node[4] - 1 < 0:
                        continue

                    for var in auto_regression_variable_reg:
                        step = int(var.split('-')[-1])
                        save_value(arrayInfluence[:,:, int(node[4] - step)], f'AutoRegressionReg_{var}', index, maskNode)

                del arrayInfluence

        logger.info('AutoRegressionBin')
        if 'AutoRegressionBin' in features:
            dir_bin = dir_train / 'bin'
            arrayBin = read_object(f'{departement}binScale{graph.scale}_{graph_construct}.pkl', dir_bin)
            if arrayBin is not None:
                for node in nodeDepartement:
                    index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                    maskNode = mask == node[0]
                    if node[4] - 1 < 0:
                        continue

                    for var in auto_regression_variable_bin:
                        step = int(var.split('-')[-1])
                        save_value(arrayBin[:,:, int(node[4] - step)], f'AutoRegressionBin_{var}', index, maskNode)
                del arrayBin

        logger.info('Vigicrues')
        if 'vigicrues' in features:
            for var in vigicrues_variables:
                array = read_object('vigicrues'+var+'.pkl', dir_data)
                for node in nodeDepartement:
                    index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                    maskNode = mask == node[0]
                    if node[4] >= array.shape[-1]:
                        continue
                    save_values(array[:,:, int(node[4])], var, index, maskNode)
                del array

        logger.info('nappes')
        if 'nappes' in features:
            for var in nappes_variables:
                array = read_object(var+'.pkl', dir_data)
                for node in nodeDepartement:
                    index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                    maskNode = mask == node[0]
                    
                    save_values(array[:,:, int(node[4])], var, index, maskNode)
                del array

    return X, features_name

def add_varying_time_features(X : np.array, Y : np.array, newShape : int, features : list, features_name : list, ks : int, scale : int, methods : list):
    res = np.empty((X.shape[0], newShape))
    res[:, :X.shape[1]] = X

    methods_dict = {'mean': np.nanmean,
                    'max' : np.nanmax,
                    'min' : np.nanmin,
                    'std': np.nanstd,
                    'sum' : np.nansum}
    
    unodes = np.unique(Y[: , 0])

    for node in unodes:
        index_node = np.argwhere((Y[:,0] == node))[:, 0]
        res[index_node, X.shape[1]:] = 0

        X_node_copy = []

        for k in range(ks + 1):
            roll_X = np.roll(res[index_node], shift=k, axis=0)
            roll_X[:k] = np.nan
            X_node_copy.append(roll_X)

        X_node_copy = np.asarray(X_node_copy, dtype=float)

        for feat in features:
            vec = feat.split('_')
            name = vec[0]

            if feat.find('Calendar') != -1:
                _, limit, _, _ = calculate_feature_range('Calendar', scale, methods)
                index_feat = features_name.index(f'{calendar_variables[0]}')
            elif feat.find('air') != -1:
                _, limit, _, _ = calculate_feature_range('air', scale, methods)
                index_feat = features_name.index(f'{air_variables[0]}')
            elif feat.find('Historical') != -1:
                _, limit, _, _ = calculate_feature_range('Historical', scale, methods)
                index_feat = features_name.index(f'{historical_variables[0]}_mean')
            else:
                index_feat = features_name.index(f'{name}_mean')
                limit = 1

            for method in methods:
                try:
                    func = methods_dict[method]
                except:
                    raise ValueError(f'{method} unknow method, {feat}')
                
                if feat.find('Calendar') != -1:
                    index_new_feat = features_name.index(f'{calendar_variables[0]}_{method}_{ks}')
                elif feat.find('air') != -1:
                    index_new_feat = features_name.index(f'{air_variables[0]}_{method}_{ks}')
                elif feat.find('Historical') != -1:
                    index_new_feat = features_name.index(f'{historical_variables[0]}_{method}_{ks}')
                else:
                    index_new_feat = features_name.index(feat)
                
                res[index_node, index_new_feat:index_new_feat+limit] = \
                            func(X_node_copy[:, :, index_feat:index_feat+limit], axis=0)

    return res

def get_edges_feature(graph, newAxis : list, path: Path, regions : gpd.GeoDataFrame, graph_structure : str) -> np.array:

        assert graph.edges is not None
        assert graph.nodes is not None

        subNode = graph.nodes

        if newAxis == []:
            return

        n_pixel_x = 0.02875215641173088
        n_pixel_y = 0.020721094073767096

        edges = np.copy(graph.edges)

        X = list(zip(regions.longitude, regions.latitude))
        regions['id'] = graph._predict_node(X)

        if 'slope' in newAxis:
            originalShape = edges.shape[0]
            res = np.full((originalShape + 1, edges.shape[1]), fill_value=np.nan)
            res[:originalShape,:] = edges

            for node in subNode:
                dept = node[3]
                departement = int2name[int(dept)]
                dir_data = root / 'csv' / departement / 'raster'
                dir_mask = root_graph / path / 'raster' / '2x2'

                mask = read_object(f'{departement}rasterScale{graph.scale}_{graph_structure}.pkl', dir_mask)
                elevation = read_object('elevation.pkl', dir_data)

                edgesNode = subNode[np.isin(subNode[:, 0], graph.edges[1][graph.edges[0] == node[0]])]
                altitude = np.nanmean(elevation[mask == node[0]])

                for ed in edgesNode:
                    index = np.argwhere((edges[:, 0] == node[0]) & (edges[:, 1] == ed[0]))
                    targetAltitude = np.nanmean(elevation[mask == ed[0]])
                    if node[0] == ed[0]:
                        slope = 0
                    else:
                        slope = funcSlope(altitude, targetAltitude, (node[1], node[2]), (ed[1], ed[2]))
                    res[originalShape, index] = slope

            edges = np.copy(res)

        if 'highway' in newAxis:

            dir_data = root / 'csv' / departement / 'data'
            n_pixel_x = 0.0002694945852326214
            n_pixel_y = 0.0002694945852352859

            originalShape = edges.shape[0]
            numNewFet = len(osmnx_variables) - 1
            res = np.full((originalShape + numNewFet, edges.shape[1]), fill_value=np.nan)

            res[:originalShape,:] = edges
            udepts = np.unique(subNode[:,3]).astype(int)

            for udept in udepts:
                mask, _, _  = rasterization(regions[regions['departement'] == int2strMaj[udept]], n_pixel_y, n_pixel_x, 'id', Path('log'), int2strMaj[udept])
                osmnx, _, _ = read_tif(dir_data / 'osmnx' / 'osmnx.tif')
                nodeDept = subNode[subNode[:,3] == udept]
                for node in nodeDept:
                    edgesNode = subNode[np.isin(subNode[:, 0], graph.edges[1][graph.edges[0] == node[0]])]
                    for ed in edgesNode:
                        index = np.argwhere((edges[:, 0] == node[0]) & (edges[:, 1] == ed[0]))
                        val = stat(node[0], ed[0], mask[0], osmnx[0], osmnx_variables[1:])
                        print(val)
                        res[originalShape: originalShape + numNewFet, index] = val

            edges = np.copy(res)

        print(edges)
        return edges