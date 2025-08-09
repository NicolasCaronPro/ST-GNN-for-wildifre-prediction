from GNN.raster_utils import *
from GNN.features import *

def find_spatial_date(x, firepoint):
    return firepoint[firepoint['h3'] != x].sample(1)['date'].values[0]

def find_temporal_date(dates):
    return sample(dates, k = 1)[0]

def get_sub_nodes_feature_2D(graph, df: pd.DataFrame,
                        departements : list,
                        features : list, sinister : str, dataset_name : str,
                        path : Path,
                        dir_train : Path,
                        resolution : str, 
                        graph_construct : str,
                        sinister_encoding : str,
                        name_exp: str,
                        newFeatures: list=[],
                        changeFeature: list=[],
                        save : bool = True,
                        use_log=True) -> tuple:
    
    # Assert that graph structure is build (not really important for that part BUT it is preferable to follow the api)

    dir_output = dir_train / f'2D_database'

    assert graph.nodes is not None

    if use_log:
        logger.info('Load 2D nodes features')

    features_name, newShape = get_features_name_lists_2D(graph.scale, features)

    dir_encoder = dir_train / 'Encoder'

    if dataset_name != 'bdiff':
        encoder_landcover = read_object(f'encoder_landcover_{name_exp}.pkl', dir_encoder)
    encoder_osmnx = read_object(f'encoder_osmnx.pkl_{name_exp}', dir_encoder)
    encoder_foret = read_object(f'encoder_foret_{name_exp}.pkl', dir_encoder)
    encoder_ids = read_object(f'encoder_ids_{graph.scale}_{graph.base}_{graph.graph_method}_{name_exp}.pkl', dir_encoder)
    encoder_argile = read_object(f'encoder_argile_{name_exp}.pkl', dir_encoder)
    encoder_cosia = read_object(f'encoder_cosia_{name_exp}.pkl', dir_encoder)

    if 'Calendar' in features:
        size_calendar = len(calendar_variables)
        encoder_calendar = read_object(f'encoder_calendar_{name_exp}.pkl', dir_encoder)

    if 'Geo' in features:
        encoder_geo = read_object(f'encoder_geo_{name_exp}.pkl', dir_encoder)

    dir_mask = path / 'raster'

    for departement in departements:
        check_and_create_path(dir_output / departement)
        check_and_create_path(dir_output / departement / str(graph.scale))
        if use_log:
            logger.info(departement)
        
        dir_data = rootDisk / 'csv' / departement / 'raster' / resolution
        dir_target = dir_train / 'influence'
        dir_target_bin =  dir_train / 'bin'

        name = f'{departement}rasterScale{graph.scale}_{graph_construct}_{graph.graph_method}.pkl'
        mask = read_object(name, dir_mask)
        assert mask is not None

        name_node = f'{departement}rasterScale{graph.scale}_{graph_construct}_{graph.graph_method}_node.pkl'
        mask_node = read_object(name_node, dir_mask)
        assert mask_node is not None

        Y_dept = read_object(f'{departement}InfluenceScale{graph.scale}_{graph_construct}_{graph.graph_method}.pkl', dir_target)
        Y_dept_bin = read_object(f'{departement}binScale{graph.scale}_{graph_construct}_{graph.graph_method}.pkl', dir_target_bin)

        nodeDepartement = df[df['departement'] == name2int[departement]][ids_columns].values

        if nodeDepartement.shape[0] == 0:
            continue

        unDates = np.unique(nodeDepartement[:,date_index]).astype(int)

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

        if 'cosia' in features:
            name = 'cosia.pkl'
            arrayCosia = read_object(name, dir_data)

        if 'foret_encoder' in features:
            #logger.info('Foret landcover')
            name = 'foret_landcover.pkl'
            arrayForetLandcover = read_object(name, dir_data)

        if 'cosia_encoder' in features:
            #logger.info('Foret landcover')
            name = 'cosia_landcover.pkl'
            arrayCosiaLandcover = read_object(name, dir_data)

        if 'landcover_encoder' in features and dataset_name != 'bdiff':
            #logger.info('Dynamic World landcover')
            name = 'dynamic_world_landcover.pkl'
            arrayLand = read_object(name, dir_data)

        if 'highway_encoder' in features:
            #logger.info('OSMNX landcover')
            name = 'osmnx_landcover.pkl'
            arrayOSLand = read_object(name, dir_data)

        if 'argile_encoder' in features:
            #logger.info('OSMNX landcover')
            name = 'argile.pkl'
            arrayARLand = read_object(name, dir_data)

        if 'ids_encoder' in features:
            logger.info('IDS')
    
        for unDate in unDates:
            if unDate % 100 == 0 and use_log:
                logger.info(f'{allDates[unDate]}')

            Y = Y_dept[:, :, unDate]
            #save_object(Y, f'Y_{unDate}.pkl', dir_output / departement / f'{graph.scale}_{graph_construct}' / 'influence')

            Y = Y_dept_bin[:, :, unDate]
            #save_object(Y, f'Y_{unDate}.pkl', dir_output / departement / f'{graph.scale}_{graph_construct}' / 'binary')

            X = np.empty((newShape, *mask.shape))
            
            if (dir_output / departement / f'X_{unDate}.pkl').is_file() and use_log:
                X = read_object(f'X_{unDate}.pkl', dir_output / departement)
                if newFeatures != []:
                    if X.shape[0] != len(features_name):
                        X2, features_name_2 = get_sub_nodes_feature_2D(graph, df[(df['departement'] == name2int[departement]) & (df['date'] == unDate)], [departement], newFeatures,
                                                                        sinister, dataset_name, dir_train, dir_train,
                                                                        resolution, graph_construct, sinister_encoding, newFeatures=[], chanegFeatures=[], save=False, use_log=False)
                        
                        features_name_ori, newShape = get_features_name_lists_2D(graph.scale, [fet for fet in features if fet not in newFeatures])

                        new_X = np.empty((X.shape[0] + X2.shape[0], X.shape[1], X.shape[2]))

                        for fet in features_name:
                            if fet in features_name_2:
                                new_X[features_name.index(fet)] = X2[features_name_2.index(fet)]
                            else:
                                new_X[features_name.index(fet)] = X[features_name_ori.index(fet)]
                        
                        X = new_X
                        if save:
                            save_object(X, f'X_{unDate}.pkl', dir_output / departement)
                                
                if changeFeature != []:
                    X2, features_name_2 = get_sub_nodes_feature_2D(graph, df[(df['departement'] == name2int[departement]) & (df['date'] == unDate)], [departement], changeFeature,
                                                                    sinister, dataset_name, dir_train, dir_train,
                                                                    resolution, graph_construct, sinister_encoding, newFeatures=[], changeFeature=[], save=False, use_log=False)
                    plt.imshow(X2[3])
                    plt.colorbar()
                    plt.savefig(f'NDSI.png')
                    plt.close('all')
                    new_X = np.empty((X.shape[0], X.shape[1], X.shape[2]))

                    for fet in features_name:
                        if fet in features_name_2:
                            new_X[features_name.index(fet)] = X2[features_name_2.index(fet)]
                        else:
                            new_X[features_name.index(fet)] = X[features_name.index(fet)]

                    plt.imshow(new_X[features_name.index('NDSI')] )
                    plt.colorbar()
                    plt.savefig(f'NDSI_2.png')
                    plt.close('all')
                    
                    X = new_X
                    if save:
                        save_object(X, f'X_{unDate}.pkl', dir_output / departement)

                continue

            if 'population' in features:
                X[features_name.index('population'), :, :] = arrayPop

            if 'elevation' in features:
                X[features_name.index('elevation'), :, :] = arrayEl

            if 'highway' in features:
                X[features_name.index(osmnxint2str[osmnx_variables[0]]) : features_name.index(osmnxint2str[osmnx_variables[-1]]) + 1, :, :] = arrayOS

            if 'foret' in features:
                X[features_name.index(foretint2str[foret_variables[0]]) : features_name.index(foretint2str[foret_variables[-1]]) + 1, :, :] = arrayForet

            if 'cosia' in features:
                X[features_name.index(cosia_variables[0]) : features_name.index(cosia_variables[-1]) + 1, :, :] = arrayCosia

            if 'foret_encoder' in features:
                #logger.info('Foret landcover')
                assert encoder_foret is not None
                assert arrayForetLandcover is not None
                #X[features_name.index('foret_encoder'), :, :] = encoder_foret.transform(arrayForetLandcover.reshape(-1,1)).values.reshape(arrayForetLandcover.shape)
                X[features_name.index('foret_encoder'), :, :] = arrayForetLandcover

            if 'landcover_encoder' in features:
                #logger.info('Dynamic World landcover')
                assert encoder_landcover is not None
                assert arrayLand is not None
                #X[features_name.index('landcover_encoder'), :, :] = encoder_landcover.transform(arrayLand.reshape(-1,1)).values.reshape(arrayLand.shape)
                X[features_name.index('landcover_encoder'), :, :] = arrayLand

            if 'highway_encoder' in features:
                #logger.info('OSMNX landcover')
                assert arrayForetLandcover is not None
                assert arrayOSLand is not None
                #X[features_name.index('highway_encoder'), :, :] = encoder_osmnx.transform(arrayOSLand.reshape(-1,1)).values.reshape(arrayOSLand.shape)
                X[features_name.index('highway_encoder'), :, :] = arrayOSLand

            if 'argile_encoder' in features:
                #logger.info('OSMNX landcover')
                assert encoder_argile is not None
                assert arrayARLand is not None
                #X[features_name.index('argile_encoder'), :, :] = encoder_argile.transform(arrayARLand.reshape(-1,1)).values.reshape(arrayARLand.shape)
                X[features_name.index('argile_encoder'), :, :] = arrayARLand
                
            if 'id_encoder' in features:
                #logger.info('OSMNX landcover')
                assert encoder_ids is not None
                assert mask_node is not None
                #X[features_name.index('id_encoder'), :, :] = encoder_ids.transform(mask_node.reshape(-1,1)).values.reshape(mask_node.shape)
                X[features_name.index('id_encoder'), :, :] = 0

            if 'cosia_encoder' in features:
                #logger.info('OSMNX landcover')
                assert encoder_cosia is not None
                assert arrayCosiaLandcover is not None
                #X[features_name.index('cosia_encoder'), :, :] = encoder_cosia.transform(arrayCosiaLandcover.reshape(-1,1)).values.reshape(mask.shape)
                X[features_name.index('cosia_encoder'), :, :] = arrayCosiaLandcover

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
                        x[ir] = round(np.nanmean(x), 3)
                    elif var_ir == 'calendar_max':
                        x[ir] = round(np.nanmax(x), 3)
                    elif var_ir == 'calendar_min':
                        x[ir] = round(np.nanmin(x), 3)
                    elif var_ir == 'calendar_sum':
                        x[ir] = round(np.nansum(x), 3)
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
                    if array is not None:
                        X[features_name.index(var), :, :] = array[:, :, unDate]

            #logger.info('Sentinel Dynamic World')
            ### Sentinel
            if 'sentinel' in features:
                #logger.info('Sentinel')
                name = 'sentinel.pkl'
                arraySent = read_object(name, dir_data)
                if arraySent is not None:
                    X[features_name.index(sentinel_variables[0]) : features_name.index(sentinel_variables[-1]) + 1, :, :] = arraySent[:, :, :, unDate]
                del arraySent

            if 'dynamicWorld' in features:
                arrayDW = read_object('dynamic_world.pkl', dir_data)
                if arrayDW is not None:
                    X[features_name.index(dynamic_world_variables[0]) : features_name.index(dynamic_world_variables[-1]) + 1, :, :] = arrayDW[:, :, :, unDate]
                del arrayDW

            #logger.info('Historical')
            if 'Historical' in features:
                #dir_target_2 = root_target / sinister / dataset_name / sinister_encoding / 'log' / resolution
                #name = departement+'pastInfluence.pkl'
                #arrayInfluence = read_object(name, dir_target_2)
                #if arrayInfluence is not None:
                    #X[features_name.index(historical_variables[0]), :, :] = arrayInfluence[:, :, unDate - 1]
                unode = np.unique(nodeDepartement[:, id_index])
                for node in unode:
                    if len(df[(df['id'] == node) & (df['date'] == unDate)]) == 0:
                        X[features_name.index(historical_variables[0]), mask == node] = 0
                    else:
                        X[features_name.index(historical_variables[0]), mask == node] = df[(df['id'] == node) & (df['date'] == unDate)]['pastinfluence'].values[0]
                    #del arrayInfluence
                    
            #logger.info('AutoRegressionReg')
            if 'AutoRegressionReg' in features:
                X[features_name.index(f'AutoRegressionReg-J-1'), :] = 0
                #dir_target_2 = dir_train / 'influence'
                #arrayInfluence = read_object(f'{departement}InfluenceScale{graph.scale}_{graph_construct}.pkl', dir_target_2)
                #if arrayInfluence is not None:
                """unode = np.unique(nodeDepartement[:, 0])
                for var in auto_regression_variable_reg:
                    step = int(var.split('-')[-1])
                    #X[features_name.index(f'AutoRegressionReg_{var}'), :, :] = arrayInfluence[:, :, unDate - step]
                    for node in unode:
                        if len(df[(df['id'] == node) & (df['date'] == unDate)]) == 0:
                            X[features_name.index(f'AutoRegressionReg-{var}'), mask == node] = 0
                        else:
                            X[features_name.index(f'AutoRegressionReg-{var}'), mask == node] = df[(df['id'] == node) & (df['date'] == unDate)][f'AutoRegressionReg-{var}'].values[0]"""
                            #del arrayInfluence
                    #del arrayInfluence

            #logger.info('AutoRegressionBin')
            if 'AutoRegressionBin' in features:
                dir_bin = dir_train / 'bin'
                arrayBin = read_object(f'{departement}binScale{graph.scale}_{graph_construct}_{graph.graph_method}_node.pkl', dir_bin)
                if arrayBin is not None:
                    for var in auto_regression_variable_bin:
                        step = int(var.split('-')[-1])
                        X[features_name.index(f'AutoRegressionBin-{var}'), :, :] = arrayBin[:, :, unDate - step]
                    del arrayBin

            #logger.info('Vigicrues')
            if 'vigicrues' in features:
                for var in vigicrues_variables:
                    array = read_object('vigicrues'+var+'.pkl', dir_data)
                    if array is not None:
                        X[features_name.index(var), :, :] = array[:, :, unDate]
                        del array

            #logger.info('nappes')
            if 'nappes' in features:
                for var in nappes_variables:
                    array = read_object(var+'.pkl', dir_data)
                    if array is None:
                        continue
                    X[features_name.index(var), :, :] = array[:, :, unDate]
                    del array

            #logger.info('region_class')
            if 'cluster_encoder' in features:
                #array = read_object(f'{departement}_{graph.scale}.pkl', dir_train / 'time_series_clustering')
                #X[features_name.index('region_class'), :, :] = array
                unode = np.unique(nodeDepartement[:, 0])
                X[features_name.index('cluster_encoder')] = 0
                #for node in unode:
                #    masknode = np.argwhere(mask == node)
                #    X[features_name.index('cluster_encoder'), masknode[:, 0], masknode[:, 1]] = df[(df['id'] == node)][f'cluster_encoder'].values[0]
                #del array
            
            if save:
                save_object(X, f'X_{unDate}.pkl', dir_output / departement)

    return X, features_name

################################################################################################################################################

def get_sub_nodes_feature_2D_from_xarray(graph,
                        departements : list,
                        features : list,
                        path : Path,
                        dir_train : Path,
                        name_expe: str,
                        use_log=True) -> tuple:
    
    dir_output = dir_train / f'2D_database'

    assert graph.nodes is not None

    res = {}

    if use_log:
        logger.info('Load 2D nodes features')

    features_name, newShape = get_features_name_lists_2D(graph.scale, features)

    dir_encoder = dir_train / 'Encoder'

    encoder_landcover = read_object(f'encoder_landcover_{name_expe}.pkl', dir_encoder)
    encoder_osmnx = read_object(f'encoder_osmnx_{name_expe}.pkl', dir_encoder)
    encoder_foret = read_object(f'encoder_foret_{name_expe}.pkl', dir_encoder)
    encoder_argile = read_object(f'encoder_argile_{name_expe}.pkl', dir_encoder)
    encoder_id = read_object(f'encoder_ids_{graph.scale}_{graph.base}_{graph.graph_method}_{name_expe}.pkl', dir_encoder)
    encoder_cosia = read_object(f'encoder_cosia_{name_expe}.pkl', dir_encoder)
    encoder_corine = read_object(f'encoder_corine_{name_expe}.pkl', dir_encoder)
    encoder_bdroute = read_object(f'encoder_route_{name_expe}.pkl', dir_encoder)
    encoder_cluster = read_object(f'encoder_cluster_{graph.scale}_{graph.base}_{graph.graph_method}_{name_expe}.pkl', dir_encoder)
    encoder_calendar = read_object(f'encoder_calendar_{name_expe}.pkl', dir_encoder)
    encoder_geo = read_object(f'encoder_geo_{name_expe}.pkl', dir_encoder)
    
    dir_mask = path / 'raster'

    for departement in departements:
        check_and_create_path(dir_output / departement)
        check_and_create_path(dir_output / departement / str(graph.scale))
        if use_log:
            logger.info(departement)
        
        dir_datacube = rootDisk / 'csv' / departement / 'raster' / '2x2'

        dir_datacube_mask = path / 'datacube'

        datacube_feature = read_object('datacube.pkl', dir_datacube)

        datacube_mask = read_object(f'datacube_target_{departement}_{graph.scale}_{graph.base}_{graph.graph_method}.pkl', dir_datacube_mask)
        
        #datacube_feature = datacube_feature.sel(date=datacube.sel(departement=departement)['date'].values)

        if 'ids_encoder' in features:
            logger.info('IDS')
    
        if 'population' in features:
            datacube_mask['population'] = (('latitude', 'longitude'), datacube_feature['population'].values[:, :, -1])
        
        if 'elevation' in features:
            datacube_mask['elevation'] = (('latitude', 'longitude'), datacube_feature['elevation'].values[:, :, -1])

        #if 'highway' in features:
        #    datacube_mask['highway_encoder'] = encoder_p

        if 'foret' in features:
            for var in foret_variables:
                datacube_mask[foretint2str[var]] = datacube_feature[foretint2str[var]]

        if 'cosia' in features:
            for var in corine_variable:
                datacube_mask[var] = datacube_feature[var]

        if 'foret_encoder' in features:
            shape = datacube_feature['forest_landcover'].values[:, :, -1].shape
            datacube_mask['foret_encoder'] = (('latitude', 'longitude'), encoder_foret.transform(datacube_feature['forest_landcover'].values[:, :, -1].reshape(-1,1)).values.reshape(shape))

        #if 'landcover_encoder' in features:
        #    datacube_mask['foret_encoder'] = encoder_foret.transform(datacube_feature['foret_landcover'].reshape(-1,1)).values.reshape(shape)
        #    pass

        if 'highway_encoder' in features:
            pass

        if 'argile_encoder' in features:
            pass

        if 'id_encoder' in features:
            shape = datacube_mask['area'].values[0].shape
            datacube_mask['id_encoder'] = (('latitude', 'longitude'), encoder_id.transform(datacube_mask['area'].values[0].reshape(-1,1)).values.reshape(shape))

        if 'corine_encoder' in features:
            shape = datacube_feature['corine_landcover'].values[:, :, -1].shape
            datacube_mask['corine_encoder'] = (('latitude', 'longitude'), encoder_corine.transform(datacube_feature['corine_landcover'].values[:, :, -1].reshape(-1,1)).values.reshape(shape))

        #logger.info('Calendar')
        if 'Calendar' in features:

            def expand_to_3d(arr_1d):
                """
                Étend un tableau 1D (par date) en 3D (lat x lon x date)
                
                Paramètres :
                    arr_1d : np.ndarray de forme (n_date,)
                    n_lat : int, nombre de latitudes
                    n_lon : int, nombre de longitudes
                
                Retour :
                    np.ndarray de forme (n_lat, n_lon, n_date)
                """
                return np.tile(arr_1d, (n_lat, n_longs, 1)).transpose(0, 1, 2)

            
            dates = pd.to_datetime(datacube_mask.loc[dict(departement=departement)]['date'].values)
            lats = datacube_mask.loc[dict(departement=departement)]['latitude'].values
            longs = datacube_mask.loc[dict(departement=departement)]['longitude'].values
            n_lat = len(lats)
            n_longs = len(longs)
            n_date = len(dates)

            # --- Étape 1 : Construction des variables calendaires dans un tableau numpy
            calendar_data = {
                'month': dates.month,
                'dayofyear': dates.dayofyear,
                'dayofweek': dates.dayofweek,
                'isweekend': (dates.dayofweek >= 5).astype(int),
                'couvrefeux': np.array([pendant_couvrefeux(d) for d in dates], dtype=int),
                'confinement': np.array([
                    1 if (
                        dt.datetime(2020, 3, 17, 12) <= d <= dt.datetime(2020, 5, 11)
                        or dt.datetime(2020, 10, 30) <= d <= dt.datetime(2020, 12, 15)
                    ) else 0 for d in dates
                ], dtype=int),
                'ramadan': np.array([
                    1 if convertdate.islamic.from_gregorian(d.year, d.month, d.day)[1] == 9 else 0 for d in dates
                ], dtype=int),
                'bankHolidays': np.array([1 if d in jours_feries else 0 for d in dates], dtype=int),
                'bankHolidaysEve': np.array([1 if d in veille_jours_feries else 0 for d in dates], dtype=int),
                'holidays': np.array([
                    1 if vacances_scolaire.is_holiday_for_zone(
                        d.date(), get_academic_zone(ACADEMIES[str(name2int[departement])], d)
                    ) else 0 for d in dates
                ], dtype=int),
                'holidaysBorder': np.array([
                    int(
                        vacances_scolaire.is_holiday_for_zone((d + dt.timedelta(days=1)).date(), get_academic_zone(ACADEMIES[str(name2int[departement])], d)) or
                        vacances_scolaire.is_holiday_for_zone((d - dt.timedelta(days=1)).date(), get_academic_zone(ACADEMIES[str(name2int[departement])], d))
                    )
                    for d in dates
                ], dtype=int),
            }

            # --- Étape 2 : Expand 1D → 2D (id, date), puis stack
            calendar_vars_raw = list(calendar_data.keys())
            calendar_array = np.stack([calendar_data[var] for var in calendar_vars_raw], axis=-1)  # shape (date, nb_vars)

            # --- Étape 3 : Encodage
            calendar_flat = calendar_array.reshape(-1, len(calendar_vars_raw))  # shape (id*date, nb_vars)
            calendar_encoded = encoder_calendar.transform(calendar_flat).values.reshape(n_date, -1)

            # --- Étape 4 : Injection dans le datacube
            for i, var in enumerate(calendar_vars_raw):
                datacube_mask[var] = (('latitude', 'longitude', 'date'), expand_to_3d(calendar_encoded[:, i]))

            datacube_mask['calendar_mean'] = (('latitude', 'longitude', 'date'), expand_to_3d(np.round(np.mean(calendar_encoded, axis=1), 3)))
            datacube_mask['calendar_min'] = (('latitude', 'longitude', 'date'), expand_to_3d(np.round(np.min(calendar_encoded, axis=1), 3)))
            datacube_mask['calendar_max'] = (('latitude', 'longitude', 'date'), expand_to_3d(np.round(np.max(calendar_encoded, axis=1), 3)))
            datacube_mask['calendar_sum'] = (('latitude', 'longitude', 'date'), expand_to_3d(np.round(np.sum(calendar_encoded, axis=1), 3)))

        ### Geo spatial
        #logger.info('Geo')
        if 'Geo' in features:
            datacube_mask['Geo'] = encoder_geo.transform([name2int[departement]]).values[0]

        logger.info('Meteorological')
        ### Meteo
        for i, var in enumerate(cems_variables):
            if var not in features:
                continue
            if 'precipitationIndex' in var:
                n = int(var[-1])
                array = calculate_precipitation_index_image_full(datacube_feature['prec24h'].values, A=0.1657, n=n)
                datacube_mask[var] = (('latitude', 'longitude', 'date'), array)
            else:    
                datacube_mask[var] = datacube_feature[var]

        #logger.info('Air Quality')
        if 'air' in features:
            pass

        ### Sentinel
        if 'sentinel' in features:
            for var in sentinel_variables:
                datacube_mask[var] = datacube_feature[var]

        if 'dynamicWorld' in features:
            pass

        #logger.info('Historical')
        if 'Historical' in features:
            pass

        if 'AutoRegressionReg' in features:
            pass

        #logger.info('AutoRegressionBin')
        if 'AutoRegressionBin' in features:
            pass

        #logger.info('Vigicrues')
        if 'vigicrues' in features:
            pass

        #logger.info('nappes')
        if 'nappes' in features:
            pass

        #logger.info('region_class')
        if 'cluster_encoder' in features:
            shape = datacube_mask['time_series_clustering'].values.shape
            datacube_mask['cluster_encoder'] = (('latitude', 'longitude'), encoder_cluster.transform(datacube_mask['time_series_clustering'].values.reshape(-1,1)).values.reshape(shape))

        save_object(datacube_mask, f'datacube_target_{departement}_{graph.scale}_{graph.base}_{graph.graph_method}.pkl', dir_datacube_mask)
        
        #if save:
        #    save_object(X, f'X_{unDate}.pkl', dir_output / departement)

    #save_object(dir_output / 'database.pkl')
    return res