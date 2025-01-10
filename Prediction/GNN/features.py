from datetime import date
from pathlib import Path
from shapely import unary_union
from arborescence import *
from weigh_predictor import Predictor
import itertools
from GNN.tools import *

def get_features_for_sinister_prediction(dataset_name, sinister, isInference):
    # mean max min and std for scale > 0 else value
    features = [
                'temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
                'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                'isi', 'angstroem', 'bui', 'fwi', 'dailySeverityRating',
                'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
                'days_since_rain', 'sum_consecutive_rainfall',
                'sum_rain_last_7_days',
                'sum_snow_last_7_days', 'snow24h', 'snow24h16',
                'elevation',
                'population',
                'sentinel',
                'foret_encoder',
                'argile_encoder',
                'id_encoder',
                'cluster_encoder',
                'cosia_encoder',
                'vigicrues',
                'foret',
                'highway',
                'cosia',
                'Calendar',
                'Historical',
                'Geo',
                'air',
                'nappes',
                'AutoRegressionReg',
                'AutoRegressionBin',
                ]
    
    kmeans_features = [
            #'rhum',
            'prec24h',
            #'rhum16',
            'prec24h16',
            'sum_consecutive_rainfall',
            'sum_rain_last_7_days',
            'sum_snow_last_7_days',
            'snow24h', 'snow24h16',
            ]
    
    if dataset_name == 'bdiff' or dataset_name == 'georisques':
        train_features = [
                    'temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
                    'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                    'isi', 'angstroem', 'bui', 'fwi', 'dailySeverityRating',
                    'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
                    'days_since_rain', 'sum_consecutive_rainfall',
                    'sum_rain_last_7_days',
                    'sum_snow_last_7_days', 'snow24h', 'snow24h16',
                    'elevation',
                    'population',
                    #'sentinel',
                    'foret_encoder',
                    'argile_encoder',
                    'id_encoder',
                    #'cluster_encoder',
                    #'cosia_encoder',
                    'vigicrues',
                    'foret',
                    'highway',
                    #'cosia',
                    'Calendar',
                    'Historical',
                    'Geo',
                    #'air',
                    'nappes',
                    #'AutoRegressionReg',
                    'AutoRegressionBin',
                ]

    elif dataset_name == 'firemen':
        if sinister == 'firepoint':
            train_features = [
                    'temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
                    'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                    'isi', 'angstroem', 'bui',
                    'fwi',
                    'dailySeverityRating',
                    'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
                    'days_since_rain', 'sum_consecutive_rainfall',
                    'sum_rain_last_7_days',
                    'sum_snow_last_7_days', 'snow24h', 'snow24h16',
                    'elevation',
                    'population',
                    'sentinel',
                    'foret_encoder',
                    'argile_encoder',
                    'id_encoder',
                    'cluster_encoder',
                    'cosia_encoder',
                    'vigicrues',
                    'foret',
                    'highway',
                    'cosia',
                    'Calendar',
                    'Historical',
                    'Geo',
                    'air',
                    'nappes',
                    #'AutoRegressionReg',
                   'AutoRegressionBin',
                ]
          
        elif sinister == 'inondation':
             train_features = [
                    'temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
                    'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                    'isi', 'angstroem', 'bui', 'fwi', 'dailySeverityRating',
                    'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
                    'days_since_rain', 'sum_consecutive_rainfall',
                    'sum_rain_last_7_days',
                    'sum_snow_last_7_days', 'snow24h', 'snow24h16',
                    'elevation',
                    'population',
                    'sentinel',
                    'landcover',
                    'vigicrues',
                    'foret',
                    'highway',
                    'cosia',
                    'Calendar',
                    'Historical',
                    'Geo',
                    'air',
                    'nappes',
                    #'AutoRegressionReg',
                    'AutoRegressionBin',
                ]
         
        else:
            pass
    elif dataset_name == 'vigicrues':
         train_features = [
                    'temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
                    'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                    'isi', 'angstroem', 'bui', 'fwi', 'dailySeverityRating',
                    'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
                    'days_since_rain', 'sum_consecutive_rainfall',
                    'sum_rain_last_7_days',
                    'sum_snow_last_7_days', 'snow24h', 'snow24h16',
                    'elevation',
                    'population',
                    #'sentinel',
                    'landcover',
                    #'vigicrues',
                    'foret',
                    'highway',
                    #'cosia',
                    'Calendar',
                    'Historical',
                    'Geo',
                    #'air',
                    'nappes',
                    #'AutoRegressionReg',
                    'AutoRegressionBin',
                ]
         
    elif dataset_name == 'firemen2':
            train_features = [
                    'temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
                    'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                    'isi', 'angstroem', 'bui', 'fwi', 'dailySeverityRating',
                    'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
                    'days_since_rain', 'sum_consecutive_rainfall',
                    'sum_rain_last_7_days',
                    'sum_snow_last_7_days', 'snow24h', 'snow24h16',
                    'elevation',
                    'population',
                    #'sentinel',
                    'foret_encoder',
                    'argile_encoder',
                    #'id_encoder',
                    'cluster_encoder',
                    #'cosia_encoder',
                    #'vigicrues',
                    'foret',
                    'highway',
                    #'cosia',
                    'Calendar',
                    #'Historical',
                    'Geo',
                    #'air',
                    'nappes',
                    #'AutoRegressionReg',
                    #'AutoRegressionBin',
                ]
    else:
        pass

    if isInference:
        train_features.pop(train_features.index('sentinel'))
        train_features.pop(train_features.index('dynamicWorld'))

    return features, train_features, kmeans_features

def get_sub_nodes_ground_truth(graph, subNode: np.array,
                           departements : list,
                           oriNode : np.array, path : Path,
                           dir_train : Path,
                           resolution  : str) -> np.array:

        assert graph.nodes is not None
        
        logger.info('Ground truth')

        newShape = subNode.shape[1] + 4
        Y = np.full((subNode.shape[0], newShape), 0, dtype=float)
        Y[:, :subNode.shape[1]] = subNode
        codes = []
            
        dir_mask = path / 'raster'
        dir_bin = path / 'bin'
        dir_proba = path / 'influence'
        dir_predictor = dir_train / 'influenceClustering'

        for departement in departements:

            if departement in graph.drop_department:
                continue

            nodeDepartementMask = np.argwhere(subNode[:,departement_index] == name2int[departement])
            nodeDepartement = subNode[nodeDepartementMask].reshape(-1, subNode.shape[1])
            nodeDepartement = np.unique(nodeDepartement, axis=0)
            codes.append(name2int[departement])

            mask = read_object(f'{departement}rasterScale{graph.scale}_{graph.base}_{graph.graph_method}.pkl', dir_mask)
            array = read_object(f'{departement}InfluenceScale{graph.scale}_{graph.base}_{graph.graph_method}.pkl', dir_proba)
            arrayBin = read_object(f'{departement}binScale{graph.scale}_{graph.base}_{graph.graph_method}.pkl', dir_bin)
            arrayTime = read_object(f'{departement}TimeScale{graph.scale}_{graph.base}_{graph.graph_method}.pkl', dir_bin)
                
            mask_node = read_object(f'{departement}rasterScale{graph.scale}_{graph.base}_{graph.graph_method}_node.pkl', dir_mask)
            arrayBin_node = read_object(f'{departement}binScale{graph.scale}_{graph.base}_{graph.graph_method}_node.pkl', dir_bin)

            if array is None:
                Y[nodeDepartementMask, :] = 0
                continue

            #predictor = read_object(f'{departement}Predictor{graph.scale}_{graph_construct}.pkl', dir_predictor)

            for node in nodeDepartement:
                index = np.argwhere((subNode[:,graph_id_index] == node[graph_id_index]) & (subNode[:, date_index] == node[date_index]))
                maskgraph = mask == node[graph_id_index]
                masknode = mask_node == node[id_index]
                    
                if node[4] >= array.shape[-1]:
                    continue
                try:
                    # Influence
                    #values = np.unique(array[maskNode, int(node[4])])[0]
                    #Y[index, -1] = values
                    Y[index, -1] = 0

                    # Nb events
                    values = (np.unique(arrayBin[maskgraph, int(node[date_index])]))[0]
                    Y[index, -2] = values

                    # Nb events
                    values = (np.unique(arrayBin_node[masknode, int(node[date_index])]))[0]
                    Y[index, -4] = values

                    # Class
                    #Y[index, -3] = predictor.predict(np.asarray(Y[index, -1])).reshape(-1)
                    Y[index, -3] = 0

                    # Time intervention
                    #Y[index, -4] = (np.unique(arrayTime[maskgraph, int(node[date_index])]))[0]

                except Exception as e:
                    logger.info(f'{departement}, {node}, {np.unique(mask)}, {e}')
                    continue
                
            #Y[nodeDepartementMask, -3] = order_class(predictor, Y[nodeDepartementMask, -3]).reshape(-1, 1)
            
        Y[:, weight_index] = Y[:, -3] + 1
        Y = Y[np.isin(Y[:, departement_index], codes)]
        #Y = Y[Y[:, ids_columns.index('weight')] > 0]
        return Y

def get_sub_nodes_feature(graph, subNode: np.array,
                        departements : list,
                        features : list, sinister : str, dataset_name: str, sinister_encoding :str,
                        path : Path,
                        dir_train : Path,
                        resolution : str) -> np.array:
    
    assert graph.nodes is not None
    graph_construct = graph.base

    logger.info('Load nodes features')

    methods = METHODS_SPATIAL

    features_name, newShape = get_features_name_list(graph.scale, features, methods)
    features_name = ids_columns[:-1] + features_name
    newShape += subNode.shape[1]

    def save_values(array, band, indexNode, mask):

        indexVar = features_name.index(f'{band}_mean')

        if False not in np.unique(np.isnan(array[mask])):
            return

        values = array[mask].reshape(-1,1)
        if isinstance(graph.scale, str) or graph.scale > 0:
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

    def save_value_sum(array, band, indexNode, mask):
        if False not in np.unique(np.isnan(array[mask])):
            return

        indexVar = features_name.index(f'{band}')
        
        X[indexNode[:, 0], indexVar] = np.nansum(array[mask])

    def save_value_with_encoding(array, band, indexNode, mask, encoder):
        values = array[mask].reshape(-1,1)
        encode_values = encoder.transform(values).values
        indexVar = features_name.index(f'{band}_mean')
        if isinstance(graph.scale, str) or graph.scale > 0:
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

    encoder_landcover = read_object('encoder_landcover.pkl', dir_encoder)
    encoder_osmnx = read_object('encoder_osmnx.pkl', dir_encoder)
    encoder_foret = read_object('encoder_foret.pkl', dir_encoder)
    encoder_argile = read_object('encoder_argile.pkl', dir_encoder)
    encoder_id = read_object(f'encoder_ids_{graph.scale}_{graph.base}_{graph.graph_method}.pkl', dir_encoder)
    encoder_cosia = read_object('encoder_cosia.pkl', dir_encoder)
    encoder_cluster = read_object(f'encoder_cluster_{graph.scale}_{graph.base}_{graph.graph_method}.pkl', dir_encoder)

    if 'Calendar' in features:
        size_calendar = len(calendar_variables)
        encoder_calendar = read_object('encoder_calendar.pkl', dir_encoder)
        
    if 'Geo' in features:
        encoder_geo = read_object('encoder_geo.pkl', dir_encoder)

    dir_mask = path / 'raster'
    logging.info(f'Shape of X {X.shape}, {np.unique(X[:,departement_index])}')
    for departement in departements:
        if departement in graph.drop_department:
            continue
        logger.info(departement)

        #dir_data = root / 'csv' / departement / 'raster' / resolution
        dir_data = rootDisk / 'csv' / departement / 'raster' / resolution

        name = f'{departement}rasterScale{graph.scale}_{graph_construct}_{graph.graph_method}_node.pkl'
        mask = read_object(name, dir_mask)

        name_graph = f'{departement}rasterScale{graph.scale}_{graph_construct}_{graph.graph_method}.pkl'
        mask_graph = read_object(name_graph, dir_mask)
        
        nodeDepartementMask = np.argwhere(subNode[:,departement_index] == name2int[departement])
        nodeDepartement = subNode[nodeDepartementMask].reshape(-1, subNode.shape[1])
        print(np.unique(mask), np.unique(nodeDepartement[:, id_index]))
        print(np.unique(mask_graph), np.unique(nodeDepartement[:, graph_id_index]))

        if (path / 'log' / f'X_{departement}_{graph.scale}_{graph.base}_{graph.graph_method}_log.pkl').is_file():
            try:
                X_dept = read_object(f'X_{departement}_{graph.scale}_{graph.base}_{graph.graph_method}_log.pkl', path / 'log')
                assert X_dept is not None
                X[nodeDepartementMask] = X_dept.reshape(X[nodeDepartementMask].shape)
                continue
            except Exception as e:
                logger.info(f'{e}')
                pass

        logger.info(nodeDepartement.shape)
        if nodeDepartement.shape[0] == 0:
            continue
        logger.info('Calendar')
        if 'Calendar' in features:
            unDate = np.unique(nodeDepartement[:, date_index]).astype(int)
            band = calendar_variables[0]
            for unDate in unDate:
                date = allDates[unDate]
                ddate = dt.datetime.strptime(date, '%Y-%m-%d')
                index = np.argwhere((subNode[:, ids_columns.index('date')] == unDate))
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
            if array is None:
                continue
            for node in nodeDepartement:
                maskNode = mask == node[id_index]
                index = np.argwhere((subNode[:,id_index] == node[id_index]) & (subNode[:, date_index] == node[date_index]))
                save_values(array[:,:,int(node[date_index])], var, index, maskNode)

        del array

        logger.info('Air Quality')
        if 'air' in features:
            for i, var in enumerate(air_variables):
                logger.info(var)
                name = var +'raw.pkl'
                array = read_object(name, dir_data)
                if array is None:
                    continue
                for node in nodeDepartement:
                    maskNode = mask == node[id_index]
                    index = np.argwhere((subNode[:,id_index] == node[id_index]) & (subNode[:, date_index] == node[date_index]))
                    save_value(array[:,:,int(node[date_index])], var, index, maskNode)

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

        if 'cosia' in features:
            logger.info('Foret')
            name = 'cosia.pkl'
            arrayCosia = read_object(name, dir_data)

        if 'foret_encoder' in features:
            logger.info('Foret landcover')
            name = 'foret_landcover.pkl'
            arrayForetLandcover = read_object(name, dir_data)

        if 'landcover_encoder' in features:
            logger.info('Dynamic World landcover')
            name = 'dynamic_world_landcover.pkl'
            arrayLand = read_object(name, dir_data)

        if 'highway_encoder' in features:
            logger.info('OSMNX landcover')
            name = 'osmnx_landcover.pkl'
            arrayOSLand = read_object(name, dir_data)

        if 'argile_encoder' in features:
            logger.info('ARGILE')
            name = 'argile.pkl'
            arrayARLand = read_object(name, dir_data)

        if 'id_encoder' in features:
            logger.info('ID')

        if 'cosia_encoder' in features:
            logger.info('COSIA')
            name = "cosia_landcover.pkl"
            arrayCOSIALandcover = read_object(name, dir_data)

        if arrayPop is not None or arrayEl is not None or arrayOS is not None or arrayForet is not None:
            unode = np.unique(nodeDepartement[:,id_index])
            ugraph = np.unique(nodeDepartement[:,graph_id_index])
            for node in unode:
                maskNode = mask == node
                index = np.argwhere(subNode[:,id_index] == node)
                if 'population' in features:
                    if arrayPop is not None:
                        save_values(arrayPop, 'population', index, maskNode)

                if 'elevation' in features:
                    if arrayEl is not None:
                        save_values(arrayEl, 'elevation', index, maskNode)

                if 'highway' in features:
                    if arrayOS is not None:
                        for var in osmnx_variables:
                            save_values(arrayOS[int(var), :, :], osmnxint2str[var], index, maskNode)

                if 'foret' in features:
                    if arrayForet is not None:
                        for var in foret_variables:
                            save_values(arrayForet[int(var), :, :], foretint2str[var], index, maskNode)

                if 'cosia' in features:
                    if arrayCosia is not None:
                        for i, var in enumerate(cosia_variables):
                            save_values(arrayCosia[i, :, :], var, index, maskNode)

                if 'foret_encoder' in features:
                    if arrayForetLandcover is not None:
                        try:
                            save_value_with_encoding(arrayForetLandcover, 'foret_encoder', index, maskNode, encoder_foret)
                        except Exception as e:
                            exit(1)

                if 'highway_encoder' in features:
                    if arrayOSLand is not None:
                        save_value_with_encoding(arrayOSLand, 'highway_encoder', index, maskNode, encoder_osmnx)
                
                if 'argile_encoder' in features:
                    if arrayARLand is not None:
                        save_value_with_encoding(arrayARLand, 'argile_encoder', index, maskNode, encoder_argile)

                if 'cosia_encoder' in features:
                    if arrayCOSIALandcover is not None:
                        save_value_with_encoding(arrayCOSIALandcover, 'cosia_encoder', index, maskNode, encoder_cosia)

                if 'id_encoder' in features:
                    save_value_with_encoding(mask, 'id_encoder', index, maskNode, encoder_id)

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
                maskNode = mask == node[id_index]
                if True not in np.unique(maskNode):
                    continue
                index = np.argwhere((subNode[:, id_index] == node[id_index]) & (subNode[:, date_index] == node[date_index]))

                if 'sentinel' in features:
                    if arraySent is not None:
                        for band, var in enumerate(sentinel_variables):
                            save_values(arraySent[band,:, :,int(node[date_index])], var , index, maskNode)
                
                if 'landcover' in features:
                    if arrayLand is not None:
                        if 'landcover_encoder' in landcover_variables:
                            save_value_with_encoding(arrayLand[:,:], 'landcover_encoder', index, maskNode, encoder_landcover)

                if 'dynamicWorld' in features:
                    if arrayDW is not None:
                        for band, var in enumerate(dynamic_world_variables):
                            save_values(arrayDW[band, :, :, int(node[date_index])], var, index, maskNode)

        del arrayPop
        del arrayEl
        del arrayOS
        del arraySent
        del arrayLand

        logger.info('Historical')
        if 'Historical' in features:
            #name = departement+'pastInfluence.pkl'
            #arrayInfluence = read_object(name, dir_target)

            #if arrayInfluence is not None:

                for node in nodeDepartement:
                    index = np.argwhere((subNode[:,id_index] == node[id_index]) & (subNode[:, date_index] == node[date_index]))
                    maskNode = mask == node[id_index]
                    X[index, features_name.index('pastinfluence')] = 0

                    #save_value_sum(arrayInfluence[:,:, int(node[4] - 1)], historical_variables[0], index, maskNode)
                #del arrayInfluence

        logger.info('AutoRegressionReg')
        if 'AutoRegressionReg' in features:
            #dir_target_2 = dir_train / 'influence'
            #arrayInfluence = read_object(f'{departement}InfluenceScale{graph.scale}_{graph_construct}.pkl', dir_target_2)
            #if arrayInfluence is not None:

                for node in nodeDepartement:
                    index = np.argwhere((subNode[:,id_index] == node[id_index]) & (subNode[:, date_index] == node[date_index]))
                    maskNode = mask == node[id_index]
                    if node[date_index] - 1 < 0:
                        continue

                    for var in auto_regression_variable_reg:
                        step = int(var.split('-')[-1])
                        X[index, features_name.index(f'AutoRegressionReg-{var}')] = 0
                        #save_value(arrayInfluence[:,:, int(node[4] - step)], f'AutoRegressionReg-{var}', index, maskNode)

                #del arrayInfluence

        logger.info('AutoRegressionBin')
        if 'AutoRegressionBin' in features:
            dir_bin = dir_train / 'bin'
            arrayBin = read_object(f'{departement}binScale{graph.scale}_{graph_construct}_{graph.graph_method}_node.pkl', dir_bin)
            if arrayBin is not None:
                for node in nodeDepartement:
                    index = np.argwhere((subNode[:, id_index] == node[id_index]) & (subNode[:, date_index] == node[date_index]))
                    maskNode = mask == node[id_index]
                    if node[date_index] - 1 < 0:
                        continue
                    
                    for var in auto_regression_variable_bin:
                        step = int(var.split('-')[-1])
                        save_value(arrayBin[:,:, int(node[date_index] - step)], f'AutoRegressionBin-{var}', index, maskNode)
                del arrayBin

        logger.info('Vigicrues')
        if 'vigicrues' in features:
            for var in vigicrues_variables:
                array = read_object('vigicrues'+var+'.pkl', dir_data)
                if array is not None:
                    for node in nodeDepartement:
                        index = np.argwhere((subNode[:,id_index] == node[id_index]) & (subNode[:, date_index] == node[date_index]))
                        maskNode = mask == node[id_index]
                        if node[date_index] >= array.shape[-1]:
                            continue
                        save_values(array[:,:, int(node[date_index])], var, index, maskNode)
                    del array

        logger.info('nappes')
        if 'nappes' in features:
            for var in nappes_variables:
                array = read_object(var+'.pkl', dir_data)
                if array is not None:
                    for node in nodeDepartement:
                        index = np.argwhere((subNode[:,id_index] == node[id_index]) & (subNode[:, date_index] == node[date_index]))
                        maskNode = mask == node[id_index]
                        
                        save_values(array[:,:, int(node[date_index])], var, index, maskNode)
                del array

        logger.info('Cluster encoder')
        assert encoder_cluster is not None
        if 'cluster_encoder' in features:
            ugraph = np.unique(nodeDepartement[:, graph_id_index])
            for graphid in ugraph:
                index = np.argwhere((subNode[:,graph_id_index] == graphid))
                X[index, features_name.index('cluster_encoder')] = encoder_cluster.transform([graph.node_cluster[graphid]]).values[0]
                #X[index, features_name.index('cluster_encoder')] = 0

        check_and_create_path(path / 'log')
        save_object(X[nodeDepartementMask], f'X_{departement}_{graph.scale}_{graph.base}_{graph.graph_method}_log.pkl', path / 'log')

    return X, features_name[len(ids_columns)-1:]

def add_varying_time_features(X: np.ndarray, Y: np.ndarray, new_shape: int, features: list,
                              feature_names: list, ks: int, scale: int, methods: list):
    """
    Adds varying time features to the dataset X based on past ks time steps.

    Parameters:
    - X: Input feature matrix (numpy array of shape (n_samples, n_features))
    - Y: Output variable or additional data (numpy array)
    - new_shape: New number of features after adding the varying time features
    - features: List of feature names to process
    - feature_names: List of all feature names in X
    - ks: Number of past time steps to consider
    - scale: Used in calculate_feature_range
    - methods: List of aggregation methods to apply (e.g., ['mean', 'max'])

    Returns:
    - res: Feature matrix with added varying time features
    """

    # Initialize result matrix with NaNs
    res = np.full((X.shape[0], new_shape), fill_value=np.nan)

    # Copy original X into res
    res[:, :X.shape[1]] = X

    # Dictionary mapping method names to numpy functions
    methods_dict = {
        'mean': np.nanmean,
        'max': np.nanmax,
        'min': np.nanmin,
        'std': np.nanstd,
        'sum': np.nansum
    }

    # Unique nodes in Y[:, 0]
    unique_nodes = np.unique(Y[:, 0])

    for node in unique_nodes:
        # Indices where Y[:, 0] == node
        node_indices = np.argwhere(Y[:, 0] == node).flatten()

        # Initialize the additional features to zero for this node
        res[node_indices, X.shape[1]:] = 0

        # List to store shifted copies of X for this node
        X_node_copy = []

        for k in range(ks + 1):
            # Shift the node's data by k time steps
            rolled_X = np.roll(res[node_indices], shift=k, axis=0)
            # Set the first k elements to NaN since they don't have enough history
            rolled_X[:k] = np.nan
            X_node_copy.append(rolled_X)

        # Stack shifted data along a new axis (shape: (ks + 1, num_samples_for_node, num_features))
        X_node_copy = np.array(X_node_copy, dtype=float)

        # Process each feature
        for feat in features:
            # Extract base feature name
            name = feat.split('_')[0]

            # Determine index and limit based on feature type
            if 'Calendar' in feat:
                _, limit, _, _ = calculate_feature_range('Calendar', scale, methods)
                index_feat = feature_names.index(f'{calendar_variables[0]}')
            elif 'air' in feat:
                _, limit, _, _ = calculate_feature_range('air', scale, methods)
                index_feat = feature_names.index(f'{air_variables[0]}')
            elif 'Historical' in feat:
                _, limit, _, _ = calculate_feature_range('Historical', scale, methods)
                index_feat = feature_names.index(f'{historical_variables[0]}_mean')
            else:
                index_feat = feature_names.index(f'{name}_mean')
                limit = 1  # Default limit for single features

            for method in methods:
                # Get the aggregation function
                func = methods_dict.get(method)
                if func is None:
                    raise ValueError(f'Unknown method "{method}" for feature "{feat}"')

                # Determine index for new feature in res
                if 'Calendar' in feat:
                    index_new_feat = feature_names.index(f'{calendar_variables[0]}_{method}_{ks}')
                elif 'air' in feat:
                    index_new_feat = feature_names.index(f'{air_variables[0]}_{method}_{ks}')
                elif 'Historical' in feat:
                    index_new_feat = feature_names.index(f'{historical_variables[0]}_{method}_{ks}')
                else:
                    index_new_feat = feature_names.index(feat)

                # Check if all values are NaN; if so, skip computation
                if np.isnan(X_node_copy[:, :, index_feat:index_feat + limit]).all():
                    continue

                # Compute the aggregation over ks time steps
                aggregated_values = func(
                    X_node_copy[:, :, index_feat:index_feat + limit],
                    axis=0
                )

                # Assign the aggregated values to res
                res[node_indices, index_new_feat:index_new_feat + limit] = aggregated_values
                
    return res

def nan_gradient(arr, *args, **kwargs):
    """
    Calcule le gradient d'un tableau en gérant les NaN.
    Les NaN sont interpolés linéairement avant le calcul du gradient.
    """
    if len(arr) == 1:
        return np.asarray([0])
    arr = np.asarray(arr, dtype=np.float64)
    nan_mask = np.isnan(arr)

    # Si le tableau ne contient pas de NaN, utiliser np.gradient directement
    if not nan_mask.any():
        return np.gradient(arr, *args, **kwargs)

    # Interpolation des NaN
    arr_filled = arr.copy()

    # Obtenir les indices des valeurs non-NaN
    not_nan_indices = np.nonzero(~nan_mask)[0]
    not_nan_values = arr[not_nan_indices]

    # Si toutes les valeurs sont NaN, retourner un tableau de NaN
    if not_nan_indices.size == 0:
        return np.full_like(arr, np.nan)

    # Interpoler les NaN
    interpolated_values = np.interp(
        x=np.arange(arr.size),
        xp=not_nan_indices,
        fp=not_nan_values
    )
    arr_filled[nan_mask] = interpolated_values[nan_mask]

    # Calculer le gradient sur le tableau rempli
    grad = np.gradient(arr_filled, *args, **kwargs)

    # Remettre les NaN aux positions où le gradient n'est pas défini
    grad[nan_mask] = np.nan

    return grad

def add_time_columns(array, integer_param, dataframe, train_features, features_name):
    """
    For each integer between 1 and the integer parameter,
    the function iterates through the array of column names and adds to the dataframe
    the values of the method applied to each element of the array.
    The elements are created as {column}_{method}.
    The final column name is {column}_{method}_{integer}.

    :param array: List of column names to process
    :param integer_param: Integer specifying the range of integers to iterate over
    :param dataframe: The pandas DataFrame to which new columns will be added
    :return: The updated DataFrame with new columns added
    """
    dataframe = dataframe.copy()
    unode = dataframe['id'].unique()

    # Dictionary of available methods
    methods_dict = {
        'mean': lambda x: np.nanmean(x),
        'sum': lambda x: np.nansum(x),
        'max': lambda x: np.nanmax(x),
        'min': lambda x: np.nanmin(x),
        'std': lambda x: np.nanstd(x),
        'grad': lambda x : nan_gradient(x)
    }
    new_fet = []
    # List of methods you want to apply
    for i in range(1, integer_param + 1):
        for column in array:
            vec = column.split('_')
            if len(vec) == 2:
                column_name, method_name = vec[0], vec[1]
            else:
                column_name, method_name = vec[0] + '_' + vec[1] + '_' + vec[1], vec[2]

            if 'Calendar' in vec[0] and 'Calendar' in train_features:
                columns = calendar_variables
            elif 'air' in vec[0] and 'air' in train_features:
                columns = air_variables
            elif 'Historical' in column_name and 'Historical' in train_features:
                columns = historical_variables
            elif 'AutoRegressionBin' in vec[0] and 'AutoRegressionBin' in train_features:
                columns = [f'AutoRegressionBin-{bin_fet}' for bin_fet in auto_regression_variable_bin]
            elif 'AutoRegressionReg' in vec[0] and 'AutoRegressionReg' in train_features:
                columns = [f'AutoRegressionReg-{reg_fet}' for reg_fet in auto_regression_variable_reg]
            elif vec[0] in train_features:
                columns = [column_name]
            else:
                continue

            for col in columns:
                for node in unode:
                    index = dataframe[dataframe['id'] == node].index
                    if method_name in methods_dict:
                        if col in features_name:
                            new_column_name = f"{col}_{method_name}_{i}"
                            if new_column_name not in list(dataframe.columns):
                                logger.info(f'{new_column_name}')
                                dataframe[new_column_name] = np.nan
                            dataframe.loc[index, new_column_name] = dataframe.loc[index, col].rolling(window=i+1).apply(methods_dict[method_name], raw=True)
                            if new_column_name not in new_fet:
                                new_fet.append(new_column_name)
                        else:
                            for spatial_method in METHODS_SPATIAL:
                                if f'{col}_{spatial_method}' in features_name:
                                    new_column_name = f"{col}_{spatial_method}_{method_name}_{i}"
                                    if new_column_name not in list(dataframe.columns):
                                        logger.info(f'{new_column_name}')
                                        dataframe[new_column_name] = np.nan
                                    dataframe.loc[index, new_column_name] = dataframe.loc[index, f'{col}_{spatial_method}'].rolling(window=i+1).apply(methods_dict[method_name], raw=True)
                                    if new_column_name not in new_fet:
                                        new_fet.append(new_column_name)
                    else:
                        raise ValueError(f"Unknown method '{method_name}'")
                
                    if col in features_name:
                        new_column_name_shift = f"{col}_-{i}"
                        if new_column_name_shift not in list(dataframe.columns):
                            logger.info(f'{new_column_name_shift}')
                            dataframe[new_column_name_shift] = np.nan
                        dataframe.loc[index, new_column_name_shift] = dataframe.loc[index, col].shift(i)
                        if new_column_name not in new_fet:
                            new_fet.append(new_column_name_shift)
                    else:
                        for spatial_method in METHODS_SPATIAL:
                            if f'{col}_{spatial_method}' in features_name:
                                new_column_name_shift = f"{col}_{spatial_method}_-{i}"
                                if new_column_name_shift not in list(dataframe.columns):
                                    dataframe[new_column_name_shift] = np.nan
                                    logger.info(f'{new_column_name_shift}')
                                dataframe.loc[index, new_column_name_shift] = dataframe.loc[index, f'{col}_{spatial_method}'].shift(i)
                                if new_column_name not in new_fet:
                                    new_fet.append(new_column_name_shift)
                    
    return dataframe, new_fet

def get_time_columns(array, integer_param, train_features, features_name):
    """
    For each integer between 1 and the integer parameter,
    the function iterates through the array of column names and returns
    the list of new column names based on the method applied to each element of the array.

    :param array: List of column names to process
    :param integer_param: Integer specifying the range of integers to iterate over
    :param dataframe: The pandas DataFrame (not modified)
    :param train_features: List of features to check for the corresponding categories
    :param features_name: List of features present in the DataFrame
    :return: List of new feature names (new_fet)
    """

    # Dictionary of available methods
    methods_dict = {
        'mean': lambda x: np.nanmean(x),
        'sum': lambda x: np.nansum(x),
        'max': lambda x: np.nanmax(x),
        'min': lambda x: np.nanmin(x),
        'std': lambda x: np.nanstd(x),
        'grad': lambda x: nan_gradient(x)
    }

    new_fet = []  # List to store new feature names

    for i in range(1, integer_param + 1):
        for column in array:
            vec = column.split('_')
            if len(vec) == 2:
                column_name, method_name = vec[0], vec[1]
            else:
                column_name, method_name = vec[0] + '_' + vec[1] + '_' + vec[1], vec[2]

            # Determine which columns to use based on the feature categories
            if 'Calendar' in vec[0] and 'Calendar' in train_features:
                columns = calendar_variables
            elif 'air' in vec[0] and 'air' in train_features:
                columns = air_variables
            elif 'Historical' in column_name and 'Historical' in train_features:
                columns = historical_variables
            elif 'AutoRegressionBin' in vec[0] and 'AutoRegressionBin' in train_features:
                columns = [f'AutoRegressionBin-{bin_fet}' for bin_fet in auto_regression_variable_bin]
            elif 'AutoRegressionReg' in vec[0] and 'AutoRegressionReg' in train_features:
                columns = [f'AutoRegressionReg-{reg_fet}' for reg_fet in auto_regression_variable_reg]
            elif vec[0] in train_features:
                columns = [column_name]
            else:
                continue

            for col in columns:
                if method_name in methods_dict:
                    if col in features_name:
                        new_column_name = f"{col}_{method_name}_{i}"
                        if new_column_name not in new_fet:
                            new_fet.append(new_column_name)
                    else:
                        for spatial_method in METHODS_SPATIAL:
                            if f'{col}_{spatial_method}' in features_name:
                                new_column_name = f"{col}_{spatial_method}_{method_name}_{i}"
                                if new_column_name not in new_fet:
                                    new_fet.append(new_column_name)
                else:
                    raise ValueError(f"Unknown method '{method_name}'")

                # Add shifted versions of the column
                if col in features_name:
                    new_column_name_shift = f"{col}_-{i}"
                    if new_column_name_shift not in new_fet:
                        new_fet.append(new_column_name_shift)
                else:
                    for spatial_method in METHODS_SPATIAL:
                        if f'{col}_{spatial_method}' in features_name:
                            new_column_name_shift = f"{col}_{spatial_method}_-{i}"
                            if new_column_name_shift not in new_fet:
                                new_fet.append(new_column_name_shift)

    return new_fet

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
        regions['id'] = graph._predict_node_with_position(X)

        if 'slope' in newAxis:
            originalShape = edges.shape[0]
            res = np.full((originalShape + 1, edges.shape[1]), fill_value=np.nan)
            res[:originalShape,:] = edges

            for node in subNode:
                dept = node[3]
                departement = int2name[int(dept)]
                dir_data = root / 'csv' / departement / 'raster'
                dir_mask = root_graph / path / 'raster' / '2x2'

                mask = read_object(f'{departement}rasterScale{graph.scale}_{graph_structure}_{graph.graph_method}.pkl', dir_mask)
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

def raster2geojson(data, mask, h3, var, encoder):
    if var == 'sentinel':
        h3[sentinel_variables] = np.nan
    else:
        h3[var] = np.nan
    if data is None:
        return h3
    else:
        unode = np.unique(mask)
        for node in unode:
            if var == 'sentinel':
                for band, v2 in enumerate(sentinel_variables):
                    dataNode = data[band, (mask == node) & ~(np.isnan(data[band]))]
                    h3.loc[h3[h3['scale0'] == node].index, v2] = np.mean(dataNode) 
            elif var == 'foret':
                for band, v2 in enumerate(foret_variables):
                    dataNode = data[band, (mask == node) & ~(np.isnan(data[band]))]
                    h3.loc[h3[h3['scale0'] == node].index, foretint2str[v2]] = np.mean(dataNode) 
            elif var == "osmnx":
                for band, v2 in enumerate(osmnx_variables):
                    dataNode = data[band, (mask == node) & ~(np.isnan(data[band]))]
                    h3.loc[h3[h3['scale0'] == node].index, osmnxint2str[v2]] = np.mean(dataNode)
            elif var == 'dynamic_world':
                for band, v2 in enumerate(dynamic_world_variables):
                    dataNode = data[band, (mask == node) & ~(np.isnan(data[band]))]
                    h3.loc[h3[h3['scale0'] == node].index, v2] = np.mean(dataNode)
            else:
                dataNode = data[(mask == node) & ~(np.isnan(data))]
                if encoder is not None:
                    dataNode = encoder.transform(dataNode.reshape(-1,1))
                h3.loc[h3['scale0'] == node, var] = np.mean(dataNode)

    return h3

def process_departments(root_data: Path, root_raster: Path, dir_mask: Path, dir_encoder: Path, 
                        regions: gpd.GeoDataFrame, sinister: str, departements: list, resolution: str, variables: list):
    """
    Processes data for each department and generates a GeoJSON file with hexagonal grid features.

    :param root_data: Path to the root directory where department data is stored.
    :param root_raster: Path to the root directory where raster data is stored.
    :param dir_mask: Path to the directory containing mask data.
    :param dir_encoder: Path to the directory containing encoder data.
    :param regions: GeoDataFrame containing region geometries and associated data.
    :param sinister: The name of the disaster or event being processed.
    :param departements: List of departments to process.
    :param resolution: The resolution level of the raster data.
    :param variables: List of variables to process for each department.
    """
    # Prepare the regions data
    regions['scale0'] = regions.index
    regions.index = regions['hex_id']
    dico = regions['scale0'].to_dict()
    regions.reset_index(drop=True, inplace=True)

    allDates = find_dates_between('2017-06-12', '2023-09-11')
    date_to_choose = allDates[-1]
    index = allDates.index(date_to_choose)

    for departement in departements:
        print(departement)
        dir_data = root_data / departement / 'data'
        dir_raster = root_raster / departement / 'raster' / resolution
        h3 = regions[regions['departement'] == departement][['geometry', 'hex_id', 'scale0']]
        mask = read_object(departement + 'rasterScale0.pkl', dir_mask)[0]

        for var in variables:
            print(var)
            data = read_object(var + '.pkl', dir_raster)
            
            # Determine the correct encoder if needed
            encoder = None
            if var in ['dynamic_world_landcover', 'foret_landcover', 'osmnx_landcover']:
                vv = var.split('_')[1]
                encoder = read_object('encoder_' + vv + '.pkl', dir_encoder)
            
            # Select the correct data slice based on variable
            if var == 'sentinel' or var == 'dynamic_world':
                data = data[:, :, :, index]

            # Convert raster data to GeoJSON and add it to the h3 dataframe
            h3 = raster2geojson(data, mask, h3, var, encoder)

        # Save the results as a GeoJSON file
        outname = 'hexagones_' + sinister + '.geojson'
        h3.rename({'osmnx_landcover': 'highway_encoder', 'foret_landcover': 'foret_encoder'}, axis=1, inplace=True)
        h3.to_file(dir_data / 'spatial' / outname, driver='GeoJSON')