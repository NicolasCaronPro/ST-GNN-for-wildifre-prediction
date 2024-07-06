from tools import *
from dataloader import *
from models.models import *
import geopandas as gpd
import convertdate
import copy
from shapely.geometry import shape, Point
import meteostat
from scipy.interpolate import griddata, interp1d
from astropy.convolution import convolve, convolve_fft
import firedanger
import jours_feries_france
import vacances_scolaires_france

def get_sub_nodes_feature(graph, subNode: np.array,
                        features : list,
                        path : Path,
                        mask : np.array,
                        dept : str,
                        geo : gpd.GeoDataFrame,
                        dates : np.array,
                        logger) -> np.array:
    
    logger.info('Load nodes features')

    pos_feature, newShape = create_pos_feature(graph.scale, subNode.shape[1], features)
    logger.info(pos_feature)

    def save_values(array, indexVvar, indexNode, mask):
        if False not in np.unique(np.isnan(array[mask])):
            return
        if graph.scale > 0:
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
        if graph.scale > 0:
            X[indexNode, indexVvar] = round(np.nanmean(encode_values), 3)
            X[indexNode, indexVvar+1] = round(np.nanmin(encode_values), 3)
            X[indexNode, indexVvar+2] = round(np.nanmax(encode_values), 3)
            X[indexNode, indexVvar+3] = round(np.nanstd(encode_values), 3)
        else:
            X[indexNode, indexVvar] = np.nanmean(encode_values)

    X = np.full((subNode.shape[0], newShape), np.nan, dtype=float)
    X[:,:subNode.shape[1]] = subNode

    dir_encoder = path / 'Encoder'

    if 'landcover' in features:
        encoder_landcover = read_object('encoder_landcover.pkl', dir_encoder)
        encoder_osmnx = read_object('encoder_osmnx.pkl', dir_encoder)
        encoder_foret = read_object('encoder_foret.pkl', dir_encoder)   

    if 'Calendar' in features:
        size_calendar = len(calendar_variables)
        encoder_calendar = read_object('encoder_calendar.pkl', dir_encoder)

    if 'Geo' in features:
        encoder_geo = read_object('encoder_geo.pkl', dir_encoder)

    logger.info('Calendar')
    if 'Calendar' in features:
        for node in subNode:
            index = np.argwhere(subNode[:,4] == node[4])
            ddate = dates[int(node[4])]
            ddate = dt.datetime(ddate.year, ddate.month, ddate.day)
            X[index, pos_feature['Calendar']] = ddate.month # month
            X[index, pos_feature['Calendar'] + 1] = ddate.timetuple().tm_yday # dayofweek
            X[index, pos_feature['Calendar'] + 2] = ddate.weekday() # dayofyear
            X[index, pos_feature['Calendar'] + 3] = ddate.weekday() >= 5 # isweekend
            X[index, pos_feature['Calendar'] + 4] = pendant_couvrefeux(ddate) # couvrefeux
            X[index, pos_feature['Calendar'] + 5] = (1 if dt.datetime(2020, 3, 17, 12) <= ddate <= dt.datetime(2020, 5, 11) else 0) or 1 if dt.datetime(2020, 10, 30) <= ddate <= dt.datetime(2020, 12, 15) else 0# confinement
            X[index, pos_feature['Calendar'] + 6] = 1 if convertdate.islamic.from_gregorian(ddate.year, ddate.month, ddate.day)[1] == 9 else 0 # ramadan
            X[index, pos_feature['Calendar'] + 7] = 1 if ddate in jours_feries else 0 # bankHolidays
            X[index, pos_feature['Calendar'] + 8] = 1 if ddate in veille_jours_feries else 0 # bankHolidaysEve
            X[index, pos_feature['Calendar'] + 9] = 1 if vacances_scolaire.is_holiday_for_zone(ddate.date(), get_academic_zone(ACADEMIES[str(name2int[dept])], ddate)) else 0 # holidays
            X[index, pos_feature['Calendar'] + 10] = (1 if vacances_scolaire.is_holiday_for_zone(ddate.date() + dt.timedelta(days=1), get_academic_zone(ACADEMIES[str(name2int[dept])], ddate)) else 0 ) \
                or (1 if vacances_scolaire.is_holiday_for_zone(ddate.date() - dt.timedelta(days=1), get_academic_zone(ACADEMIES[str(name2int[dept])], ddate)) else 0) # holidaysBorder
        
            stop_calendar = 11
            
            X[:, pos_feature['Calendar'] : pos_feature['Calendar'] + stop_calendar] = \
                    encoder_calendar.transform((X[:, pos_feature['Calendar'] : \
                            pos_feature['Calendar'] + stop_calendar]).reshape(-1, stop_calendar)).values.reshape(-1, stop_calendar)

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
        X[:, pos_feature['Geo']] = encoder_geo.transform(geo['departement'].values).values[0] # departement
  
    logger.info('Meteorological')
    ### Meteo
    for i, var in enumerate(cems_variables):
        logger.info(var)
        for node in subNode:
            maskNode = geo[(geo['id'] == node[0]) & (geo['date'] == node[4])].index
            if maskNode.shape[0] == 0:
                continue
            index = np.argwhere((subNode[:,0] == node[0])
                                & (subNode[:,4] == node[4]))
            
            save_values(geo[var].values, pos_feature[var], index, maskNode)

    logger.info('Air Quality')
    for i, var in enumerate(air_variables):
        logger.info(var)
        for node in subNode:
            maskNode = geo[(geo['id'] == node[0]) & (geo['date'] == node[4])].index
            if maskNode.shape[0] == 0:
                continue
            index = np.argwhere((subNode[:,0] == node[0])
                                & (subNode[:,4] == node[4]))
            
            save_value(geo[var].values, pos_feature['air'] + i, index, maskNode)

    logger.info('Population elevation Highway Sentinel')
    for node in subNode:
        maskNode = geo[geo['id'] == node[0]].index
        index = np.argwhere(subNode[:,0] == node[0])
        if 'population' in features:
            save_values(geo['population'].values, pos_feature['population'], index, maskNode)

        if 'elevation' in features:
            save_values(geo['elevation'].values, pos_feature['elevation'], index, maskNode)

        if 'highway' in features:
            for band in osmnx_variables:
                save_values(geo[band].values, pos_feature['highway'] + (osmnx_variables.index(band) * 4), index, maskNode)

        if 'foret' in features:
            for band in foret_variables:
                save_values(geo[band].values, pos_feature['foret'] + (foret_variables.index(band) * 4), index, maskNode)

        if 'landcover' in features:
            if 'foret' in landcover_variables:
                save_value_with_encoding(geo['foret'].values, pos_feature['landcover'] + (landcover_variables.index('foret') * 4), index, maskNode, encoder_foret)
            if 'highway' in landcover_variables:
                save_value_with_encoding(geo['highway'].values, pos_feature['landcover'] + (landcover_variables.index('highway') * 4), index, maskNode, encoder_osmnx)

    logger.info('Sentinel Dynamic World')
    coef = 4 if graph.scale > -1 else 1
    for node in subNode:
        maskNode = geo[(geo['id'] == node[0]) & (geo['date'] == node[4])].index
        index = np.argwhere((subNode[:,0] == node[0])
                            & (subNode[:,4] == node[4])
                            )

        if 'sentinel' in features:
            for band, var in enumerate(sentinel_variables):
                save_values(geo[var].values, pos_feature['sentinel'] + (sentinel_variables.index(var) * coef) , index, maskNode)
        
        if 'landcover' in features:
            if 'landcover' in landcover_variables:
                save_value_with_encoding(geo['landcover'].values, pos_feature['landcover'] + (landcover_variables.index('landcover') * 4), index, maskNode, encoder_landcover)

        if 'dynamicWorld' in features:
            for band in dynamic_world_variables:
                save_values(geo[band].values, pos_feature['dynamicWorld'] + (dynamic_world_variables.index(band) * 4), index, maskNode)

    logger.info('Historical')
    if 'Historical' in features:
        name = dept+'pastInfluence.pkl'
        arrayInfluence = pickle.load(open(Path('.') / 'log' / name, 'rb'))
        for node in subNode:
            index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
            maskNode = mask == node[0]

            save_values(arrayInfluence[:,:, int(node[4] - 1)], pos_feature['Historical'], index, maskNode)

    logger.info('AutoRegressionReg')
    if 'AutoRegressionReg' in features:
        for node in subNode:
            index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
            
            for band in auto_regression_variable_reg:
                step = int(band.split('-')[-1])
                maskNode = geo[(geo['id'] == node[0]) & (geo['date'] == node[4] - step)].index
                save_value(geo[band].values, pos_feature['AutoRegressionReg'] + auto_regression_variable_reg.index(band), index, maskNode)

    logger.info('AutoRegressionBin')
    if 'AutoRegressionBin' in features:
        for node in subNode:
            index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))

            for band in auto_regression_variable_bin:
                step = int(band.split('-')[-1])
                maskNode = geo[(geo['id'] == node[0]) & (geo['date'] == node[4] - step)].index
                save_value(geo[band].values, pos_feature['AutoRegressionBin'] + auto_regression_variable_bin.index(band), index, maskNode)

    logger.info('Vigicrues')
    if 'vigicrues' in features:
        for band in vigicrues_variables:
            for node in subNode:
                index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                maskNode = geo[(geo['id'] == node[0]) & (geo['date'] == node[4])].index

                save_values(geo[band].values, pos_feature['vigicrues'] + (4 * vigicrues_variables.index(band)), index, maskNode)

    logger.info('nappes')
    if 'nappes' in features:
        for band in nappes_variables:
            for node in subNode:
                index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                maskNode = geo[(geo['id'] == node[0]) & (geo['date'] == node[4])].index

                save_values(geo[band].values, pos_feature['nappes'] + (4 * nappes_variables.index(band)), index, maskNode)

    return X, pos_feature

def myFunctionDistanceDugrandCercle3D(outputShape, earth_radius=6371.0,
                                      resolution_lon=0.0002694945852352859, resolution_lat=0.0002694945852326214, resolution_altitude=10):
    half_rows = outputShape[0] // 2
    half_cols = outputShape[1] // 2
    half_altitude = outputShape[2] // 2

    # Créer une grille de coordonnées géographiques avec les résolutions souhaitées
    latitudes = np.linspace(-half_rows * resolution_lat, half_rows * resolution_lat, outputShape[0])
    longitudes = np.linspace(-half_cols * resolution_lon, half_cols * resolution_lon, outputShape[1])
    altitudes = np.linspace(-half_altitude * resolution_altitude, half_altitude * resolution_altitude, outputShape[2])
    
    latitudes, longitudes, altitudes = np.meshgrid(latitudes, longitudes, altitudes, indexing='ij')

    # Coordonnées du point central
    center_lat = latitudes[outputShape[0] // 2, outputShape[1] // 2, outputShape[2] // 2]
    center_lon = longitudes[outputShape[0] // 2, outputShape[1] // 2, outputShape[2] // 2]
    center_altitude = altitudes[outputShape[0] // 2, outputShape[1] // 2, outputShape[2] // 2]
    
    # Convertir les coordonnées géographiques en radians
    latitudes_rad = np.radians(latitudes)
    longitudes_rad = np.radians(longitudes)

    # Calculer la distance du grand cercle entre chaque point et le point central
    delta_lon = longitudes_rad - np.radians(center_lon)
    delta_lat = latitudes_rad - np.radians(center_lat)
    delta_altitude = abs(altitudes - center_altitude)
    
    a = np.sin(delta_lat/2)**2 + np.cos(latitudes_rad) * np.cos(np.radians(center_lat)) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distances = earth_radius * c * np.cos(np.radians(center_lat)) + delta_altitude
    
    return distances

def influence_index3D(raster, mask, dimS, mode, dim=(90, 150, 3), semi=False):
    dimX, dimY, dimZ = dimS
    if mode == "laplace":
        kernel = myFunctionDistanceDugrandCercle3D(dim, resolution_lon=dimY, resolution_lat=dimX, resolution_altitude=dimZ) + 1
        kernel = 1 / kernel 
    else:
        kernel = np.full(dim, 1/(dim[0]*dim[1]*dim[2]), dtype=float)

    if semi:
        kernel[:,:,(dim[2]//2) + 1:] = 0.0

    res = convolve_fft(raster, kernel, normalize_kernel=False, mask=mask)

    return res

def construct_historical_meteo(acq_date, region, dir_meteostat):
    MAX_NAN = 5000
    print(dir_meteostat)
    if not (dir_meteostat / 'liste_de_points.pkl').is_file():
        print("Pas de listes de points trouvés pour Meteostat, on les calcule")

        #N = config.getint('PARAMETERS', 'cote_meteostat')
        N = 11
        range_x = np.linspace(
            *region.iloc[0].geometry.buffer(0.15).envelope.boundary.xy[0][:2], N)
        range_y = np.linspace(
            *region.iloc[0].geometry.buffer(0.15).envelope.boundary.xy[1][1:3], N)
        points = []
        for point_y in range_y:
            for point_x in range_x:
                if region.iloc[0].geometry.buffer(0.15).contains(Point((point_x, point_y))):
                    points.append((point_y, point_x))

        print(f"Nombre de points de surveillance pour Meteostat : {len(points)}")
        print(f"On sauvegarde ces points")
        with open(dir_meteostat / 'liste_de_points.pkl', 'wb') as f:
            pickle.dump(points, f)
    else:
        print("On relit les points de Meteostat")
        with open(dir_meteostat / 'liste_de_points.pkl', 'rb') as f:
            points = pickle.load(f)
    print(points)
    print(f"Nombre de points de surveillance pour Meteostat : {len(points)}")
    print("On récupère les archives Meteostat")

    START = dt.datetime.strptime(acq_date, '%Y-%m-%d') - dt.timedelta(days=7)
    END = dt.datetime.strptime(acq_date, '%Y-%m-%d') + dt.timedelta(days=7)

    END += dt.timedelta(hours=1)

    data, data24h, liste, datakbdi = {}, {}, [], {}
    last_point = None
    #meteostat.Stations.max_age = 0
    meteostat.Point.radius = 2000000
    #meteostat.Point.method = "weighted"
    meteostat.Point.alt_range = 10000
    meteostat.Point.max_count = 6
    res = []
    for index, point in enumerate(sorted(points)):
        location = meteostat.Point(point[0], point[1])
        print(f"Intégration du point de coordonnées {point}")
        try:
            data[point] = meteostat.Hourly(location, START, END + dt.timedelta(days=1))
            # on comble les heures manquantes (index) dans les données collectées
            data[point] = data[point].normalize()
            # On complète les Nan, quand il n'y en a pas plus de 3 consécutifs
            data[point] = data[point].interpolate()
            data[point] = data[point].fetch()
            assert len(data[point]) > 0
            data[point]['snow'].fillna(0, inplace=True)
            data[point].drop(['tsun', 'coco', 'wpgt'], axis=1, inplace=True)
            #data[point].ffill(inplace=True)
            #data[point].bfill(inplace=True)
            data[point].reset_index(inplace=True)
            
            for k in sorted(data[point].columns):
                if data[point][k].isna().sum() > MAX_NAN:
                    print(f"{k} du point {point} possède trop de NaN ({data[point][k].isna().sum()}")
                    if last_point is None:
                        assert False
                    data[point][k] = last_point[k].values

                elif data[point][k].isna().sum() > 0:
                    #print(f"{k} du point {point} possède des NaN ({data[point][k].isna().sum()}, on interpole")
                    x = data[point][data[point][k].notna()].index
                    y = data[point][data[point][k].notna()][k]

                    f2 = interp1d(x, y, kind='linear', fill_value='extrapolate')
                    
                    newx = data[point][data[point][k].isna()].index
                    newy = f2(newx)

                    data[point].loc[newx, k] = newy

            #data[point].fillna(0, inplace=True)
       
            data[point].rename({'time': 'creneau'}, axis=1, inplace=True)
            data[point].set_index('creneau', inplace=True)
            last_point = data[point]
        except Exception as e:
            print(e)
            if last_point is None:
                print(f"Pas de données pour {point}, et pas de voisin")
                del data[point]
                continue
            else:
                print(f"Pas de données pour {point}, on prend ceux des voisins")
                data[point] = copy.deepcopy(last_point)

        fill_secheresse = False
        #if DEBUT_SECHERESSE <= data[point].index[-1].month <= FIN_SECHERESSE:
        fill_secheresse = True
        data24h[point] = copy.deepcopy(data[point])
        datakbdi[point] = copy.deepcopy(data[point])

        for k in range(0, 24):
            data24h[point][f"prcp+{k}"] = data24h[point].prcp.shift(k)
            datakbdi[point][f"prcp+{k}"] = datakbdi[point].prcp.shift(k)

        data24h[point]['prec24h'] = data24h[point][[f"prcp+{k}" for k in range(0, 24)]].sum(axis=1)
        datakbdi[point]['prec24h'] = datakbdi[point][[f"prcp+{k}" for k in range(0, 24)]].sum(axis=1)

        data24h[point] = data24h[point][['temp',
                                'dwpt',
                                'rhum',
                                'prcp',
                                'wdir',
                                'wspd',
                                'prec24h']]
        datakbdi[point] = datakbdi[point][['temp',
                                'dwpt',
                                'rhum',
                                'prcp',
                                'wdir',
                                'wspd',
                                'prec24h']]
        
        data24h[point]['wspd'] =  data24h[point]['wspd'] * 1000 / 3600
        datakbdi[point]['wspd'] =  datakbdi[point]['wspd'] * 1000 / 3600

        data16h = data24h[point].loc[data24h[point].index.hour == 16].reset_index()
        data15h = data24h[point].loc[data24h[point].index.hour == 15].reset_index()
        data24h[point] = data24h[point].loc[data24h[point].index.hour == 12]

        treshPrec24 = 1.8

        #### Data24h
        data24h[point].reset_index(inplace=True)
        data24h[point].loc[0, 'dc'] = 15
        data24h[point].loc[0, 'ffmc'] = 85
        data24h[point].loc[0, 'dmc'] = 6
        data24h[point].loc[0, 'nesterov'] = 0
        data24h[point].loc[0, 'munger'] = 0
        data24h[point].loc[0, 'kbdi'] = 0
        data24h[point].loc[0, 'days_since_rain'] = int(data24h[point].loc[0, 'prec24h'] > treshPrec24)
        data24h[point]['rhum'] = data24h[point]['rhum'].apply(lambda x: min(x, 100))

        datakbdi[point].reset_index(inplace=True)
        datakbdi[point]['day'] = datakbdi[point]['creneau'].apply(lambda x : x.date())

        tmax = datakbdi[point].groupby('day')['temp'].max()
        prec = datakbdi[point].groupby('day')['prcp'].sum()
        leni = len(data24h[point])

        consecutive = 0
        print(leni)
        for i in range(1, len(data24h[point])):
            #print(i, '/', leni, data24h[point].loc[i, 'creneau'], consecutive)
            if consecutive == 3:
                data24h[point].loc[i, 'dc'] = firedanger.indices.dc(data24h[point].loc[i, 'temp'],
                                                    data24h[point].loc[i, 'prec24h'],
                                                    data24h[point].loc[i, 'creneau'].month,
                                                    point[0],
                                                    data24h[point].loc[i - 1, 'dc'])
                if data24h[point].loc[i, 'dc'] > 1000:
                    print(data24h[point].loc[i - 1, 'dc'], data24h[point].loc[i, 'dc'], data24h[point].loc[i, 'prec24h'], data24h[point].loc[i-1, 'days_since_rain'])

                data24h[point].loc[i, 'ffmc'] = firedanger.indices.ffmc(data24h[point].loc[i, 'temp'],
                                                    data24h[point].loc[i, 'prec24h'],
                                                    data24h[point].loc[i, 'wspd'],
                                                    data24h[point].loc[i, 'rhum'],
                                                    data24h[point].loc[i - 1, 'ffmc'])
                data24h[point].loc[i, 'dmc'] = firedanger.indices.dmc(data24h[point].loc[i, 'temp'],
                                                    data24h[point].loc[i, 'prec24h'],
                                                    data24h[point].loc[i, 'rhum'],
                                                    data24h[point].loc[i, 'creneau'].month,
                                                    point[0],
                                                    data24h[point].loc[i - 1, 'dmc'])
            else:
                data24h[point].loc[i, 'dc'] = data24h[point].loc[i - 1, 'dc']
                data24h[point].loc[i, 'ffmc'] = data24h[point].loc[i - 1, 'ffmc']
                data24h[point].loc[i, 'dmc'] = data24h[point].loc[i - 1, 'dmc']

                if data24h[point].loc[i, 'temp'] >= 12:
                    consecutive += 1
                else:
                    consecutive = 0
            
            data24h[point].loc[i, 'nesterov'] = firedanger.indices.nesterov(data15h.loc[i, 'temp'], data15h.loc[i, 'rhum'], data15h.loc[i, 'prec24h'],
                                                                   data24h[point].loc[i - 1, 'nesterov'])
            
            data24h[point].loc[i, 'munger'] = firedanger.indices.munger(data24h[point].loc[i, 'prec24h'], data24h[point].loc[i - 1, 'munger'])
            data24h[point].loc[i, 'days_since_rain'] = data24h[point].loc[i - 1, 'days_since_rain'] + 1 if data24h[point].loc[i, 'prec24h'] < treshPrec24 else 0
            
            creneau = data24h[point].loc[i, 'creneau'].date()
            prec2 = prec[(prec.index >= creneau - dt.timedelta(days=data24h[point].loc[i, 'days_since_rain'])) & (prec.index <= creneau)].sum()
            precweek = prec[(prec.index >= creneau - dt.timedelta(7)) & (prec.index <= creneau)].sum()
            preci = prec[prec.index == creneau].values[0]
            tmaxi = tmax[tmax.index == creneau].values[0]
            data24h[point].loc[i, 'kbdi'] = firedanger.indices.kbdi(tmaxi, preci, data24h[point].loc[i - 1, 'kbdi'], prec2, precweek, 30, 3.467326)
        
        data24h[point]['isi'] = data24h[point].apply(lambda x: firedanger.indices.isi(x.wspd, x.ffmc), axis=1)
        data24h[point]['angstroem'] = data24h[point].apply(lambda x: firedanger.indices.angstroem(x.temp, x.rhum), axis=1)
        data24h[point]['bui'] = firedanger.indices.bui(data24h[point]['dmc'], data24h[point]['dc'])
        data24h[point]['fwi'] = firedanger.indices.bui(data24h[point]['isi'], data24h[point]['bui'])
        data24h[point]['daily_severity_rating'] = data24h[point].apply(lambda x: firedanger.indices.daily_severity_rating(x.fwi), axis=1)

        var16 = ['temp','dwpt', 'rhum','prcp','wdir','wspd','prec24h']
        for v in var16:
            data24h[point][f"{v}16"] = data16h[v].values

        for k in sorted(data24h[point].columns):
            if data24h[point][k].isna().sum() > MAX_NAN:
                print(f"{k} du point {point} possède trop de NaN ({data24h[point][k].isna().sum()}")
                if last_point is None:
                    assert False
                data24h[point][k] = last_point[k].values

            elif data24h[point][k].isna().sum() > 0:
                #print(f"{k} du point {point} possède des NaN ({data[point][k].isna().sum()}, on interpole")
                x = data24h[point][data24h[point][k].notna()].index
                y = data24h[point][data24h[point][k].notna()][k]

                f2 = interp1d(x, y, kind='linear', fill_value='extrapolate')
                
                newx = data24h[point][data24h[point][k].isna()].index
                newy = f2(newx)

                data24h[point].loc[newx, k] = newy

        data24h[point]['latitude'] = point[0]
        data24h[point]['longitude'] = point[1]
        res.append(data24h[point].reset_index())

    def get_date(x):
        return x.strftime('%Y-%m-%d')

    res = pd.concat(res)

    res['creneau'] =  res['creneau'].apply(get_date)
    
    #for var in cems_variables:
    #    res.rename({var:var+'_0'}, axis=1, inplace=True)
    #res.to_csv(dir_incendie / 'CDS/cems_v2.csv', index=False)
    return res

def pendant_couvrefeux(date):
    # Fonction testant is une date tombe dans une période de confinement
    if ((dt.datetime(2020, 12, 15) <= date <= dt.datetime(2021, 1, 2)) 
        and (date.hour >= 20 or date.hour <= 6)):
        return 1
    elif ((dt.datetime(2021, 1, 2) <= date <= dt.datetime(2021, 3, 20))
        and (date.hour >= 18 or date.hour <= 6)):
            return 1
    elif ((dt.datetime(2021, 3, 20) <= date <= dt.datetime(2021, 5, 19))
        and (date.hour >= 19 or date.hour <= 6)):
            return 1
    elif ((dt.datetime(2021, 5, 19) <= date <= dt.datetime(2021, 6, 9))
        and (date.hour >= 21 or date.hour <= 6)):
            return 1
    elif ((dt.datetime(2021, 6, 9) <= date <= dt.datetime(2021, 6, 30))
        and (date.hour >= 23 or date.hour <= 6)):
            return 1
    return 0

def create_grid_cems(df, time, delta, variables):
    interdate = dt.datetime.strptime(time, '%Y-%m-%d') - dt.timedelta(days=delta)
    interdate = interdate.strftime('%Y-%m-%d')
    dff = df[(df["creneau"] >= interdate) & (df["creneau"] <= time)]
    dff[variables] = dff[variables].astype(float)
    if len(dff) == 0:
        return None
    return dff.reset_index(drop=True)

def interpolate_gridd(var, grid, newx, newy):
    x = grid['longitude'].values
    y = grid["latitude"].values
    points = np.zeros((y.shape[0], 2))
    points[:,0] = x
    points[:,1] = y
    return griddata(points, grid[var].values, (newx, newy), method='linear')

def create_spatio_temporal_sinister_image(regions : gpd.GeoDataFrame,
                                          mask : np.array,
                                          sinisterType : str,
                                          n_pixel_y : float, 
                                          n_pixel_x : float,
                                          dir_output : Path,
                                          dept : str):
    
    nonNanMask = np.argwhere(~np.isnan(mask))
    dates = regions['date'].unique()
    spatioTemporalRaster = np.full((mask.shape[0], mask.shape[1], len(dates)), np.nan)
    for i, date in enumerate(dates):

        if regions['nb'+sinisterType].sum() == 0:
            spatioTemporalRaster[nonNanMask[:,0], nonNanMask[:,1], i] = 0
            continue

        hexaFire = regions[regions['date'] == date].copy(deep=True)

        rasterVar, _, _ = rasterization(hexaFire, n_pixel_y, n_pixel_x, 'nb'+sinisterType, dir_output, dept+'_bin0')
        spatioTemporalRaster[nonNanMask[:, 0], nonNanMask[:, 1], i] = rasterVar[0][nonNanMask[:, 0], nonNanMask[:, 1]].astype(int)

    return spatioTemporalRaster

jours_feries = sum([list(jours_feries_france.JoursFeries.for_year(k).values()) for k in range(2017,2023)],[]) # French Jours fériés, used in features_*.py 
veille_jours_feries = sum([[l-dt.timedelta(days=1) for l \
            in jours_feries_france.JoursFeries.for_year(k).values()] for k in range(2017,2023)],[]) # French Veille Jours fériés, used in features_*.py 
vacances_scolaire = vacances_scolaires_france.SchoolHolidayDates() # French Holidays used in features_*.py

def get_academic_zone(name, date):
        dict_zones = {
            'Aix-Marseille': ('B', 'B'),
            'Amiens': ('B', 'B'),
            'Besançon': ('B', 'A'),
            'Bordeaux': ('C', 'A'),
            'Caen': ('A', 'B'),
            'Clermont-Ferrand': ('A', 'A'),
            'Créteil': ('C', 'C'),
            'Dijon': ('B', 'A'),
            'Grenoble': ('A', 'A'),
            'Lille': ('B', 'B'),
            'Limoges': ('B', 'A'),
            'Lyon': ('A', 'A'),
            'Montpellier': ('A', 'C'),
            'Nancy-Metz': ('A', 'B'),
            'Nantes': ('A', 'B'),
            'Nice': ('B', 'B'),
            'Orléans-Tours': ('B', 'B'),
            'Paris': ('C', 'C'),
            'Poitiers': ('B', 'A'),
            'Reims': ('B', 'B'),
            'Rennes': ('A', 'B'),
            'Rouen ': ('B', 'B'),
            'Strasbourg': ('B', 'B'),
            'Toulouse': ('A', 'C'),
            'Versailles': ('C', 'C')
        }
        if date < dt.datetime(2016, 1, 1):
            return dict_zones[name][0]
        return dict_zones[name][1]

ACADEMIES = {
    '1': 'Lyon',
    '69': 'Lyon',
    '25': 'Besançon',
    '78': 'Versailles',
}