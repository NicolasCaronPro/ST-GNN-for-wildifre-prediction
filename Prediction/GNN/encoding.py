from ast import arg
from GNN.features_2D import *

def encode_from_xarray(path_to_target, trainDates, expe, train_departements, dir_output, resolution, graph):
    check_and_create_path(dir_output)
    print(f'Create encoder for categorical features using {train_departements}, at expe {expe}')
    stop_calendar = 11
    trainDate = np.asarray([allDates.index(date) for date in trainDates])
    foret = []
    cosia = []
    corine = []
    route = []
    gt = []
    landcover = []
    argile_value = []
    temporalValues = []
    spatialValues = []
    ids_value = []
    cluster_value = []
    osmnx = []
    calendar_array = [[] for j in range(stop_calendar)]
    geo_array = []
    for dep in train_departements:

        if dep in graph.drop_department:
            continue

        dir_data = rootDisk / 'csv' / dep /  'raster' / resolution

        datacube_target = read_object(f'datacube_target_{dep}_{graph.scale}_{graph.base}_{graph.graph_method}.pkl', dir_output / '..' / 'datacube')
        datacube_feature = read_object(f'datacube.pkl', dir_data)

        tar = datacube_target['occurence'].values[0]
        tar = tar[:,:,trainDate]
        gt += list(tar[~np.isnan(tar)])

        temporalValues.append(np.nansum(tar.reshape(-1, tar.shape[2]), axis=0))
        spatialValues += list(np.nansum(tar, axis=2)[~np.isnan(tar[:,:,0])].reshape(-1))

        fore = datacube_feature['forest_landcover'].values[:, :, 0]
        if fore is not None:
            fore = resize_no_dim(fore, tar.shape[0], tar.shape[1])
            foret += list(fore[~np.isnan(tar[:,:,0])])

        """os = datacube_feature['osmnx_landcover'].values[:, :, 0]
        if os is not None:
            os = resize_no_dim(os, tar.shape[0], tar.shape[1])
            osmnx += list(os[~np.isnan(tar[:,:,0])])"""

        """land = read_object('dynamic_world_landcover.pkl', dir_data)
        if land is not None:
            try:
                land = land[:,trainDate]
                landcover += list(land[~np.isnan(tar[:,:,0])])
            except:
                pass"""
        
        #argile = datacube_feature['argile'].values[:, :, 0]
        #if argile is not None:
        #    argile = resize_no_dim(argile, tar.shape[0], tar.shape[1])
        #    argile_value += list(argile[~np.isnan(tar[:, :, 0])])

        id_mask = datacube_target['area'].values[0]
        if id_mask is not None:
            ids_value += list(id_mask[~np.isnan(tar[:, :, 0])])

        cluster_mask = datacube_target['time_series_clustering'].values
        if cluster_mask is not None:
            cluster_value += list(cluster_mask[~np.isnan(tar[:, :, 0])])

        """os = datacube_feature['argile_landcover'].values[:, :, 0]
        cosia_image = read_object('cosia_landcover.pkl', dir_data)
        if cosia_image is not None:
            cosia_image = resize_no_dim(cosia_image, tar.shape[0], tar.shape[1])
            cosia += list(cosia_image[~np.isnan(tar[:, :, 0])])"""
        
        corine_image = datacube_feature['corine_landcover'].values[:, :, 0]
        if corine_image is not None:
            corine_image = resize_no_dim(corine_image, tar.shape[0], tar.shape[1])
            #plt.figure(figsize=(15,5))
            #plt.imshow(corine_image)
            #plt.savefig(f'{dep}_test_corine.png')
            corine += list(corine_image[~np.isnan(tar[:, :, 0])])
        
        route_image = datacube_feature['route_landcover'].values[:, :, 0]
        if route_image is not None:
            route_image = resize_no_dim(route_image, tar.shape[0], tar.shape[1])
            route += list(route_image[~np.isnan(tar[:, :, 0])])
        
        calendar = np.empty((tar.shape[2], stop_calendar))
        i = 0
        for idate, date in enumerate(allDates):
            
            if idate not in trainDate:
                continue
            
            ddate = dt.datetime.strptime(date, '%Y-%m-%d')
            calendar[i, 0] = int(date.split('-')[1]) # month
            calendar[i, 1] = ajuster_jour_annee(ddate, ddate.timetuple().tm_yday) # dayofyear
            calendar[i, 2] = ddate.weekday() # dayofweek
            calendar[i, 3] = ddate.weekday() >= 5 # isweekend
            calendar[i, 4] = pendant_couvrefeux(ddate) # couvrefeux

            calendar[i, 5] = 1 if (
                dt.datetime(2020, 3, 17, 12) <= ddate <= dt.datetime(2020, 5, 11)
                or dt.datetime(2020, 10, 30) <= ddate <= dt.datetime(2020, 12, 15)
            ) else 0
            
            calendar[i, 6] = 1 if convertdate.islamic.from_gregorian(ddate.year, ddate.month, ddate.day)[1] == 9 else 0 # ramadan
            calendar[i, 7] = 1 if ddate in jours_feries else 0 # bankHolidays
            calendar[i, 8] = 1 if ddate in veille_jours_feries else 0 # bankHolidaysEve
            calendar[i, 9] = 1 if vacances_scolaire.is_holiday_for_zone(ddate.date(), get_academic_zone(ACADEMIES[str(name2int[dep])], ddate)) else 0 # holidays
            calendar[i, 10] = (1 if vacances_scolaire.is_holiday_for_zone(ddate.date() + dt.timedelta(days=1), get_academic_zone(ACADEMIES[str(name2int[dep])], ddate)) else 0 ) \
                or (1 if vacances_scolaire.is_holiday_for_zone(ddate.date() - dt.timedelta(days=1), get_academic_zone(ACADEMIES[str(name2int[dep])], ddate)) else 0) # holidaysBorder
            
            i += 1

        for j in range(stop_calendar):
            calendar_array[j] += list(calendar[:, j])

        geo = np.empty((tar.shape[2], len(geo_variables)))
        geo[:, :] = name2int[dep]
        geo_array += list(geo)
        
    gt = np.asarray(gt)
    temporalValues = np.asarray(temporalValues)
    spatialValues = np.asarray(spatialValues)
    foret = np.asarray(foret)
    osmnx = np.asarray(osmnx)
    ids_value = np.asarray(ids_value)
    cluster_value = np.asarray(cluster_value)
    landcover = np.asarray(landcover)
    argile_value = np.asarray(argile_value)
    cosia = np.asarray(cosia)
    corine = np.asarray(corine)
    route = np.asarray(route)
    calendar_array = np.asarray(calendar_array)
    geo_array = np.asarray(geo_array)

    logger.info(f'{spatialValues.shape, temporalValues.shape, foret.shape, landcover.shape, calendar_array.shape, geo_array.shape, argile_value.shape, cluster_value.shape, cosia.shape}')

    spatialValues = spatialValues.reshape(-1,1)
    foret = foret.reshape(-1,1)
    landcover = landcover.reshape(-1,1)
    calendar_array = np.moveaxis(calendar_array, 0, 1)
    osmnx = osmnx.reshape(-1,1)
    argile_value = argile_value.reshape(-1,1)
    geo_array = geo_array.reshape(-1,1)
    ids_value = ids_value.reshape(-1,1)
    cluster_value = cluster_value.reshape(-1,1)
    temporalValues = temporalValues.reshape(-1,1)
    cosia = cosia.reshape(-1,1)
    corine = corine.reshape(-1,1)
    route = route.reshape(-1,1)

    logger.info(f'{spatialValues.shape, temporalValues.shape, foret.shape, landcover.shape, calendar_array.shape, geo_array.shape}')

    # Calendar
    encoder = CatBoostEncoder(cols=np.arange(0, stop_calendar))
    encoder.fit(calendar_array, temporalValues)
    save_object(encoder, f'encoder_calendar_{expe}.pkl', dir_output)

    if landcover.shape == spatialValues.shape:
        # Landcover
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(landcover, spatialValues)
        save_object(encoder, f'encoder_landcover_{expe}.pkl', dir_output)

    if foret.shape == spatialValues.shape:
        # Foret
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(foret, spatialValues)
        save_object(encoder, f'encoder_foret_{expe}.pkl', dir_output)

    if osmnx.shape == spatialValues.shape:
        # OSMNX
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(osmnx, spatialValues)
        save_object(encoder, f'encoder_osmnx_{expe}.pkl', dir_output)

    if argile_value.shape == spatialValues.shape:
        # OSMNX
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(argile_value, spatialValues)
        save_object(encoder, f'encoder_argile_{expe}.pkl', dir_output)

    if ids_value.shape == spatialValues.shape:
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(ids_value, spatialValues)
        save_object(encoder, f'encoder_ids_{graph.scale}_{graph.base}_{graph.graph_method}_{expe}.pkl', dir_output)

    if cluster_value.shape == spatialValues.shape:
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(cluster_value, spatialValues)
        save_object(encoder, f'encoder_cluster_{graph.scale}_{graph.base}_{graph.graph_method}_{expe}.pkl', dir_output)

    if cosia.shape == spatialValues.shape:
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(cosia, spatialValues)
        save_object(encoder, f'encoder_cosia_{expe}.pkl', dir_output)

    if corine.shape == spatialValues.shape:
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(corine, spatialValues)
        save_object(encoder, f'encoder_corine_{expe}.pkl', dir_output)

    if route.shape == spatialValues.shape:
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(route, spatialValues)
        save_object(encoder, f'encoder_route_{expe}.pkl', dir_output)
    
    # Geo
    encoder = CatBoostEncoder(cols=np.arange(0, 1))
    encoder.fit(geo_array, temporalValues)
    save_object(encoder, f'encoder_geo_{expe}.pkl', dir_output)

from ast import arg
from GNN.features_2D import *

def encode(path_to_target, trainDates, expe, train_departements, dir_output, resolution, graph):

    print(f'Create encoder for categorical features using {train_departements}, at expe {expe}')
    stop_calendar = 11
    trainDate = np.asarray([allDates.index(date) for date in trainDates])
    foret = []
    cosia = []
    corine = []
    route = []
    gt = []
    landcover = []
    argile_value = []
    temporalValues = []
    spatialValues = []
    ids_value = []
    cluster_value = []
    osmnx = []
    calendar_array = [[] for j in range(stop_calendar)]
    geo_array = []
    for dep in train_departements:

        if dep in graph.drop_department:
            continue

        dir_data = rootDisk / 'csv' / dep /  'raster' / resolution
        #dir_data = root / 'csv' / dep /  'raster' / resolution

        tar = read_object(dep+'binScale0.pkl', path_to_target)
        if tar is None:
            continue
        tar = tar[:,:,trainDate]
        gt += list(tar[~np.isnan(tar)])

        temporalValues.append(np.nansum(tar.reshape(-1, tar.shape[2]), axis=0))
        spatialValues += list(np.nansum(tar, axis=2)[~np.isnan(tar[:,:,0])].reshape(-1))

        fore = read_object('foret_landcover.pkl', dir_data)
        if fore is not None:
            fore = resize_no_dim(fore, tar.shape[0], tar.shape[1])
            foret += list(fore[~np.isnan(tar[:,:,0])])
        
        os = read_object('osmnx_landcover.pkl', dir_data)
        if os is not None:
            os = resize_no_dim(os, tar.shape[0], tar.shape[1])
            osmnx += list(os[~np.isnan(tar[:,:,0])])

        land = read_object('dynamic_world_landcover.pkl', dir_data)
        if land is not None:
            try:
                land = land[:,trainDate]
                landcover += list(land[~np.isnan(tar[:,:,0])])
            except:
                pass

        argile = read_object('argile.pkl', dir_data)
        if argile is not None:
            argile = resize_no_dim(argile, tar.shape[0], tar.shape[1])
            argile_value += list(argile[~np.isnan(tar[:, :, 0])])

        dir_mask = Path(dir_output / '..' /  'raster')
        id_mask = read_object(f'{dep}rasterScale{graph.scale}_{graph.base}_{graph.graph_method}_node.pkl', dir_mask)
        if id_mask is not None:
            ids_value += list(id_mask[~np.isnan(tar[:, :, 0])])

        dir_clustering = Path(dir_output / '..' /  'time_series_clustering')
        cluster_mask = read_object(f'{dep}_{graph.scale}_{graph.base}_{graph.graph_method}.pkl', dir_clustering)
        if cluster_mask is not None:
            cluster_value += list(cluster_mask[~np.isnan(tar[:, :, 0])])

        cosia_image = read_object('cosia_landcover.pkl', dir_data)
        if cosia_image is not None:
            cosia_image = resize_no_dim(cosia_image, tar.shape[0], tar.shape[1])
            cosia += list(cosia_image[~np.isnan(tar[:, :, 0])])

        corine_image = read_object('corine_landcover.pkl', dir_data)
        if corine_image is not None:
            corine_image = resize_no_dim(corine_image, tar.shape[0], tar.shape[1])
            #plt.figure(figsize=(15,5))
            #plt.imshow(corine_image)
            #plt.savefig(f'{dep}_test_corine.png')
            corine += list(corine_image[~np.isnan(tar[:, :, 0])])
        
        route_image = read_object('route_landcover.pkl', dir_data)
        if route_image is not None:
            route_image = resize_no_dim(route_image, tar.shape[0], tar.shape[1])
            route += list(route_image[~np.isnan(tar[:, :, 0])])
        
        calendar = np.empty((tar.shape[2], stop_calendar))
        for i, date in enumerate(allDates):
            
            if i not in trainDate:
                break
            
            ddate = dt.datetime.strptime(date, '%Y-%m-%d')
            calendar[i, 0] = int(date.split('-')[1]) # month
            calendar[i, 1] = ajuster_jour_annee(ddate, ddate.timetuple().tm_yday) # dayofyear
            calendar[i, 2] = ddate.weekday() # dayofweek
            calendar[i, 3] = ddate.weekday() >= 5 # isweekend
            calendar[i, 4] = pendant_couvrefeux(ddate) # couvrefeux
            
            calendar[i, 5] = 1 if (
                dt.datetime(2020, 3, 17, 12) <= ddate <= dt.datetime(2020, 5, 11)
                or dt.datetime(2020, 10, 30) <= ddate <= dt.datetime(2020, 12, 15)
            ) else 0
            
            calendar[i, 6] = 1 if convertdate.islamic.from_gregorian(ddate.year, ddate.month, ddate.day)[1] == 9 else 0 # ramadan
            calendar[i, 7] = 1 if ddate in jours_feries else 0 # bankHolidays
            calendar[i, 8] = 1 if ddate in veille_jours_feries else 0 # bankHolidaysEve
            calendar[i, 9] = 1 if vacances_scolaire.is_holiday_for_zone(ddate.date(), get_academic_zone(ACADEMIES[str(name2int[dep])], ddate)) else 0 # holidays
            calendar[i, 10] = (1 if vacances_scolaire.is_holiday_for_zone(ddate.date() + dt.timedelta(days=1), get_academic_zone(ACADEMIES[str(name2int[dep])], ddate)) else 0 ) \
                or (1 if vacances_scolaire.is_holiday_for_zone(ddate.date() - dt.timedelta(days=1), get_academic_zone(ACADEMIES[str(name2int[dep])], ddate)) else 0) # holidaysBorder
                
        for j in range(stop_calendar):
            calendar_array[j] += list(calendar[:, j])

        geo = np.empty((tar.shape[2], len(geo_variables)))
        geo[:, :] = name2int[dep]
        geo_array += list(geo)

    gt = np.asarray(gt)
    temporalValues = np.asarray(temporalValues)
    spatialValues = np.asarray(spatialValues)
    foret = np.asarray(foret)
    osmnx = np.asarray(osmnx)
    ids_value = np.asarray(ids_value)
    cluster_value = np.asarray(cluster_value)
    landcover = np.asarray(landcover)
    argile_value = np.asarray(argile_value)
    cosia = np.asarray(cosia)
    corine = np.asarray(corine)
    route = np.asarray(route)
    calendar_array = np.asarray(calendar_array)
    geo_array = np.asarray(geo_array)

    logger.info(f'{spatialValues.shape, temporalValues.shape, foret.shape, landcover.shape, calendar_array.shape, geo_array.shape, argile_value.shape, cluster_value.shape, cosia.shape}')

    spatialValues = spatialValues.reshape(-1,1)
    foret = foret.reshape(-1,1)
    landcover = landcover.reshape(-1,1)
    calendar_array = np.moveaxis(calendar_array, 0, 1)
    osmnx = osmnx.reshape(-1,1)
    argile_value = argile_value.reshape(-1,1)
    geo_array = geo_array.reshape(-1,1)
    ids_value = ids_value.reshape(-1,1)
    cluster_value = cluster_value.reshape(-1,1)
    temporalValues = temporalValues.reshape(-1,1)
    cosia = cosia.reshape(-1,1)
    corine = corine.reshape(-1,1)
    route = route.reshape(-1,1)

    logger.info(f'{spatialValues.shape, temporalValues.shape, foret.shape, landcover.shape, calendar_array.shape, geo_array.shape}')

    # Calendar
    encoder = CatBoostEncoder(cols=np.arange(0, stop_calendar))
    encoder.fit(calendar_array, temporalValues)
    save_object(encoder, f'encoder_calendar_{expe}.pkl', dir_output)

    if landcover.shape == spatialValues.shape:
        # Landcover
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(landcover, spatialValues)
        save_object(encoder, f'encoder_landcover_{expe}.pkl', dir_output)

    if foret.shape == spatialValues.shape:
        # Foret
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(foret, spatialValues)
        save_object(encoder, f'encoder_foret_{expe}.pkl', dir_output)

    if osmnx.shape == spatialValues.shape:
        # OSMNX
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(osmnx, spatialValues)
        save_object(encoder, f'encoder_osmnx_{expe}.pkl', dir_output)

    if argile_value.shape == spatialValues.shape:
        # OSMNX
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(argile_value, spatialValues)
        save_object(encoder, f'encoder_argile_{expe}.pkl', dir_output)

    if ids_value.shape == spatialValues.shape:
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(ids_value, spatialValues)
        save_object(encoder, f'encoder_ids_{graph.scale}_{graph.base}_{graph.graph_method}_{expe}.pkl', dir_output)

    if cluster_value.shape == spatialValues.shape:
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(cluster_value, spatialValues)
        save_object(encoder, f'encoder_cluster_{graph.scale}_{graph.base}_{graph.graph_method}_{expe}.pkl', dir_output)

    if cosia.shape == spatialValues.shape:
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(cosia, spatialValues)
        save_object(encoder, f'encoder_cosia_{expe}.pkl', dir_output)

    if corine.shape == spatialValues.shape:
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(corine, spatialValues)
        save_object(encoder, f'encoder_corine_{expe}.pkl', dir_output)

    if route.shape == spatialValues.shape:
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(route, spatialValues)
        save_object(encoder, f'encoder_route_{expe}.pkl', dir_output)
    
    # Geo
    encoder = CatBoostEncoder(cols=np.arange(0, 1))
    encoder.fit(geo_array, temporalValues)
    save_object(encoder, f'encoder_geo_{expe}.pkl', dir_output)

