from raster_utils import *
from features import *

def find_spatial_date(x, firepoint):
    return firepoint[firepoint['h3'] != x].sample(1)['date'].values[0]

def find_temporal_date(dates):
    return sample(dates, k = 1)[0]

def get_sub_nodes_feature_2D(graph, subNode: np.array,
                        departements : list,
                        features : list, sinister : str,
                        path : Path,
                        dir_train : Path,
                        resolution : str, 
                        graph_construct : str) -> tuple:
    
    # Assert that graph structure is build (not really important for that part BUT it is preferable to follow the api)

    dir_output = dir_train / '2D_database'

    assert graph.nodes is not None

    logger.info('Load 2D nodes features')

    features_name, newShape = get_features_name_lists_2D(graph.scale, features)

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

    for departement in departements:
        check_and_create_path(dir_output / departement)
        check_and_create_path(dir_output / departement / str(graph.scale))
        logger.info(departement)

        dir_data = root / 'csv' / departement / 'raster' / resolution
        dir_target =  dir_train / 'influence'
        dir_target_bin =  dir_train / 'bin'

        name = f'{departement}rasterScale{graph.scale}_{graph_construct}.pkl'
        mask = read_object(name, dir_mask)

        Y_dept = read_object(f'{departement}InfluenceScale{graph.scale}.pkl', dir_target)
        Y_dept_bin = read_object(f'{departement}binScale{graph.scale}.pkl', dir_target_bin)

        nodeDepartementMask = np.argwhere(subNode[:,3] == str2int[departement.split('-')[-1]])
        nodeDepartement = subNode[nodeDepartementMask].reshape(-1, subNode.shape[1])

        logger.info(nodeDepartement.shape)

        if nodeDepartement.shape[0] == 0:
            continue

        unDates = np.unique(nodeDepartement[:,4]).astype(int)

        if 'population' in features:
            name = 'population.pkl'
            arrayPop = read_object(name, dir_data)

        if 'elevation' in features:
            name = 'elevation.pkl'
            arrayEl = read_object(name, dir_data)

        if 'highway' in features:
            name = 'osmnx.pkl'
            arrayOS = read_object(name, dir_data)

        if 'foret' in features:
            name = 'foret.pkl'
            arrayForet = read_object(name, dir_data)

        if 'landcover' in features:
            if 'foret_encoder' in landcover_variables:
                #logger.info('Foret landcover')
                name = 'foret_landcover.pkl'
                arrayForetLandcover = read_object(name, dir_data)

            if 'landcover_encoder' in landcover_variables:
                #logger.info('Dynamic World landcover')
                name = 'dynamic_world_landcover.pkl'
                arrayLand = read_object(name, dir_data)

            if 'highway_encoder' in landcover_variables:
                #logger.info('OSMNX landcover')
                name = 'osmnx_landcover.pkl'
                arrayOSLand = read_object(name, dir_data)
        
        for unDate in unDates:
            if unDate % 100 == 0:
                logger.info(f'{allDates[unDate]}')

            Y = Y_dept[:, :, unDate]
            save_object(Y, f'Y_{unDate}.pkl', dir_output / departement / str(graph.scale) / 'influence')

            Y = Y_dept_bin[:, :, unDate]
            save_object(Y, f'Y_{unDate}.pkl', dir_output / departement / str(graph.scale) / 'binary')

            X = np.empty((newShape, *mask.shape))

            if (dir_output / departement / f'X_{unDate}.pkl').is_file():
                continue

            if 'population' in features:
                X[features_name.index('population'), :, :] = arrayPop

            if 'elevation' in features:
                X[features_name.index('elevation'), :, :] = arrayEl

            if 'highway' in features:
                X[features_name.index(osmnxint2str[osmnx_variables[0]]) : features_name.index(osmnxint2str[osmnx_variables[-1]]) + 1, :, :] = arrayOS

            if 'foret' in features:
                X[features_name.index(foretint2str[foret_variables[0]]) : features_name.index(foretint2str[foret_variables[-1]]) + 1, :, :] = arrayForet

            if 'landcover' in features:
                if 'foret_encoder' in landcover_variables:
                    #logger.info('Foret landcover')
                    X[features_name.index('foret_encoder'), :, :] = encoder_foret.transform(arrayForetLandcover.reshape(-1,1)).values.reshape(arrayForetLandcover.shape)

                if 'landcover_encoder' in landcover_variables:
                    #logger.info('Dynamic World landcover')
                    X[features_name.index('landcover_encoder'), :, :] = encoder_landcover.transform(arrayLand.reshape(-1,1)).values.reshape(arrayLand.shape)

                if 'highway_encoder' in landcover_variables:
                    #logger.info('OSMNX landcover')
                    X[features_name.index('highway_encoder'), :, :] = encoder_osmnx.transform(arrayOSLand.reshape(-1,1)).values.reshape(arrayOSLand.shape)

            #logger.info('Calendar')
            if 'Calendar' in features:
                band = calendar_variables[0]
                date = allDates[unDate]
                ddate = dt.datetime.strptime(date, '%Y-%m-%d')
                stop_calendar = 11
                x = np.empty(stop_calendar + 4)
                x[0] = int(date.split('-')[1]) # month
                x[1] = ajuster_jour_annee(ddate, ddate.timetuple().tm_yday) # dayofyear
                x[2] = ddate.weekday() # dayofweek
                x[3] = ddate.weekday() >= 5 # isweekend
                x[4] = pendant_couvrefeux(ddate) # couvrefeux
                x[5] = (1 if dt.datetime(2020, 3, 17, 12) <= ddate <= dt.datetime(2020, 5, 11) else 0) or 1 if dt.datetime(2020, 10, 30) <= ddate <= dt.datetime(2020, 12, 15) else 0# confinement
                x[6] = 1 if convertdate.islamic.from_gregorian(ddate.year, ddate.month, ddate.day)[1] == 9 else 0 # ramadan
                x[7] = 1 if ddate in jours_feries else 0 # bankHolidays
                x[8] = 1 if ddate in veille_jours_feries else 0 # bankHolidaysEve
                x[9] = 1 if vacances_scolaire.is_holiday_for_zone(ddate.date(), get_academic_zone(ACADEMIES[str(name2int[departement])], ddate)) else 0 # holidays
                x[10] = (1 if vacances_scolaire.is_holiday_for_zone(ddate.date() + dt.timedelta(days=1), get_academic_zone(ACADEMIES[str(name2int[departement])], ddate)) else 0 ) \
                    or (1 if vacances_scolaire.is_holiday_for_zone(ddate.date() - dt.timedelta(days=1), get_academic_zone(ACADEMIES[str(name2int[departement])], ddate)) else 0) # holidaysBorder

                x[:stop_calendar] = np.round(encoder_calendar.transform(x[:stop_calendar].reshape(-1, stop_calendar))).values
                
                for ir in range(stop_calendar, size_calendar):
                    var_ir = calendar_variables[ir]
                    if var_ir == 'calendar_mean':
                        x[ir] = round(np.mean(x), 3)
                    elif var_ir == 'calendar_max':
                        x[ir] = round(np.max(x), 3)
                    elif var_ir == 'calendar_min':
                        x[ir] = round(np.min(x), 3)
                    elif var_ir == 'calendar_sum':
                        x[ir] = round(np.sum(x), 3)
                    else:
                        logger.info(f'Unknow operation {var_ir}')
                        exit(1)
                        
                for band in range(x.shape[0]):
                    X[features_name.index(calendar_variables[0]) + band, :, :] = x[band]

            ### Geo spatial
            #logger.info('Geo')
            if 'Geo' in features:
                X[features_name.index(geo_variables[0])] = encoder_geo.transform([name2int[departement]]).values[0] # departement

            #logger.info('Meteorological')
            ### Meteo
            for i, var in enumerate(cems_variables):
                if var not in features:
                    continue
                #logger.info(var)
                name = var +'raw.pkl'
                array = read_object(name, dir_data)
                X[features_name.index(var), :, :] = array[:, :, unDate]
                del array

            #logger.info('Air Quality')
            if 'air' in features:
                for i, var in enumerate(air_variables):
                    #logger.info(var)
                    name = var +'raw.pkl'
                    array = read_object(name, dir_data)
                    X[features_name.index(var), :, :] = array[:, :, unDate]

            #logger.info('Sentinel Dynamic World')
            ### Sentinel
            if 'sentinel' in features:
                #logger.info('Sentinel')
                name = 'sentinel.pkl'
                arraySent = read_object(name, dir_data)
                X[features_name.index(sentinel_variables[0]) : features_name.index(sentinel_variables[-1]) + 1, :, :] = arraySent[:, :, :, unDate]
                del arraySent

            if 'dynamicWorld' in features:
                arrayDW = read_object('dynamic_world.pkl', dir_data)
                X[features_name.index(dynamic_world_variables[0]) : features_name.index(dynamic_world_variables[-1]) + 1, :, :] = arrayDW[:, :, :, unDate]
                del arrayDW

            #logger.info('Historical')
            if 'Historical' in features:
                name = departement+'pastInfluence.pkl'
                arrayInfluence = read_object(name, dir_target)
                if arrayInfluence is not None:
                    X[features_name.index(historical_variables[0]), :, :] = arrayInfluence[:, :, unDate]
                    del arrayInfluence

            #logger.info('AutoRegressionReg')
            if 'AutoRegressionReg' in features:
                dir_target_2 = dir_train / 'influence'
                arrayInfluence = read_object(departement+'InfluenceScale'+str(graph.scale)+'.pkl', dir_target_2)
                if arrayInfluence is not None:
                    for var in auto_regression_variable_reg:
                        step = int(var.split('-')[-1])
                        X[features_name.index(f'AutoRegressionReg_{var}'), :, :] = arrayInfluence[:, :, unDate - step]
                    del arrayInfluence

            #logger.info('AutoRegressionBin')
            if 'AutoRegressionBin' in features:
                dir_bin = dir_train / 'bin'
                arrayBin = read_object(departement+'binScale'+str(graph.scale)+'.pkl', dir_bin)
                if arrayBin is not None:
                    for var in auto_regression_variable_reg:
                        step = int(var.split('-')[-1])
                        X[features_name.index(f'AutoRegressionBin_{var}'), :, :] = arrayBin[:, :, unDate - step]
                    del arrayBin

            #logger.info('Vigicrues')
            if 'vigicrues' in features:
                for var in vigicrues_variables:
                    array = read_object('vigicrues'+var+'.pkl', dir_data)
                    X[features_name.index(var), :, :] = array[:, :, unDate]
                    del array

            #logger.info('nappes')
            if 'nappes' in features:
                for var in nappes_variables:
                    array = read_object(var+'.pkl', dir_data)
                    X[features_name.index(var), :, :] = array[:, :, unDate]
                    del array
        
            save_object(X, f'X_{unDate}.pkl', dir_output / departement)

    return X, features_name