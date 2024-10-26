from tools import *
#from forecasting_models.models import *
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

def get_sub_nodes_feature_with_geodataframe(graph, subNode: np.array,
                        features : list,
                        path : Path,
                        geo : gpd.GeoDataFrame,
                        dates : np.array,
                        departement : str,
                        resolution : str) -> np.array:
    
    assert graph.nodes is not None

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
            X[indexNode, indexVar] = np.nanmean(values)

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

    dir_mask = path / 'raster'
    logger.info(f'Shape of X {X.shape}, {np.unique(X[:,3])}')

    name = f'{departement}rasterScale{graph.scale}_{graph.base}.pkl'
    mask = read_object(name, dir_mask)
    print(np.unique(mask))
    logger.info('Calendar')
    if 'Calendar' in features:
        unDate = np.unique(subNode[:,4]).astype(int)
        band = calendar_variables[0]
        for unDate in unDate:
            date = dates[unDate]
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
        X[:, features_name.index(geo_variables[0])] = encoder_geo.transform([name2int[departement]]).values[0] # departement

    logger.info('Meteorological')
    array = None
    ### Meteo
    for i, var in enumerate(cems_variables):
        if var not in features:
            continue
        logger.info(var)
        name = var +'raw.pkl'
        for node in subNode:
            maskNode = geo[(geo['id'] == node[0]) & (geo['date'] == node[4])].index
            index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
            save_values(geo[var].values, var, index, maskNode)

    del array

    logger.info('Air Quality')
    if 'air' in features:
        for i, var in enumerate(air_variables):
            logger.info(var)
            name = var +'raw.pkl'
            for node in subNode:
                maskNode = geo[(geo['id'] == node[0]) & (geo['date'] == node[4])].index
                index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                save_value(geo[var].values, var, index, maskNode)

    logger.info('Population elevation Highway Sentinel Foret')

    unode = np.unique(subNode[:,0])
    for node in unode:
        maskNode = geo[(geo['id'] == node) & (geo['date'] == 0)].index
        index = np.argwhere(subNode[:,0] == node)
        if 'population' in features:
            save_values(geo['population'].values, 'population', index, maskNode)

        if 'elevation' in features:
            save_values(geo['elevation'].values, 'elevation', index, maskNode)

        if 'highway' in features:   
            for var in osmnx_variables:
                save_values(geo[osmnxint2str[var]].values, osmnxint2str[var], index, maskNode)

        if 'foret' in features:
            for var in foret_variables:
                save_values(geo[foretint2str[var]].values, foretint2str[var], index, maskNode)

        if 'landcover' in features:
            if 'foret_encoder' in landcover_variables:
                save_value_with_encoding(geo['foret_encoder'].values, 'foret_encoder', index, maskNode, encoder_foret)
            if 'highway_encoder' in landcover_variables:
                save_value_with_encoding(geo['highway_encoder'].values, 'highway_encoder', index, maskNode, encoder_osmnx)

    logger.info('Sentinel Dynamic World')
    for node in subNode:
        maskNode = geo[(geo['id'] == node[0]) & (geo['date'] == node[4])].index

        index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))

        if 'sentinel' in features:
            for band, var in enumerate(sentinel_variables):
                save_values(geo[var].values, var, index, maskNode)
        
        if 'landcover_encoder' in features:
            if 'landcover' in landcover_variables:
                save_value_with_encoding(geo['landcover_encoder'].values, 'landcover_encoder', index, maskNode, encoder_landcover)

        if 'dynamicWorld' in features:
            for band, var in enumerate(dynamic_world_variables):
                save_values(geo[var].values, var, index, maskNode)

    logger.info('Historical')
    if 'Historical' in features:
        name = departement+'pastInfluence.pkl'
        arrayInfluence = read_object(name, Path(__file__).absolute().parent.resolve()/ 'log' / resolution)
        if arrayInfluence is not None:
            for node in subNode:
                index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                maskNode = mask == node[0]

                save_values(arrayInfluence[:,:, int(node[4] - 1)], historical_variables[0], index, maskNode)
            del arrayInfluence

    logger.info('AutoRegressionReg')
    if 'AutoRegressionReg' in features:
        for node in subNode:
            index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
            if node[4] - 1 < 0:
                continue

            for var in auto_regression_variable_reg:
                step = int(var.split('-')[-1])
                maskNode = geo[(geo['id'] == node[0]) & (geo['date'] == node[4] - step)].index

                save_value(geo[f'AutoRegressionReg'].values, f'AutoRegressionReg_{var}', index, maskNode)

    logger.info('AutoRegressionBin')
    if 'AutoRegressionBin' in features:
        for node in subNode:
            index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
            maskNode = mask == node[0]
            if node[4] - 1 < 0:
                continue
            
            for var in auto_regression_variable_bin:
                step = int(var.split('-')[-1])
                maskNode = geo[(geo['id'] == node[0]) & (geo['date'] == node[4] - step)].index
                save_value(geo[f'AutoRegressionBin'].values, f'AutoRegressionBin_{var}', index, maskNode)
 
    logger.info('Vigicrues')
    if 'vigicrues' in features:
        for var in vigicrues_variables:
            for node in subNode:
                index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                maskNode = geo[(geo['id'] == node[0]) & (geo['date'] == node[4])].index
                save_values(geo[var].values, var, index, maskNode)

    logger.info('nappes')
    if 'nappes' in features:
        for var in nappes_variables:
            for node in subNode:
                index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                maskNode = geo[(geo['id'] == node[0]) & (geo['date'] == node[4])].index
                save_values(geo[var].values, var, index, maskNode)

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
    if dim[-1] == 1:
        dimZ = 1
    else:
        dimZ = np.linspace(dim[-1]/2, 0, num=(dim[-1] // 2) + 1)[0] - np.linspace(dim[-1]//2, 0, num=(dim[-1] // 2) + 1)[1]
    if mode == "laplace":
        kernel = myFunctionDistanceDugrandCercle3D(dim, resolution_lon=dimX, resolution_lat=dimY, resolution_altitude=dimZ) + dimZ
        kernel = dimZ / kernel
    else:
        kernel = np.full(dim, 1/(dim[0]*dim[1]*dim[2]), dtype=float)
        #kernel[:, :, dim[2]//2] = 0

    if semi:
        kernel[:,:,:(dim[2]//2)] = 0.0

    logger.info(f'{dim[-1]}, {np.linspace(dim[-1]/2, 0, num=(dim[-1] // 2) + 1)}, {kernel[1,1]}')

    res = convolve_fft(raster, kernel, normalize_kernel=False, mask=mask)
    #res = scipy_fft_conv(raster, kernel, mode='same')
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

def get_sub_nodes_feature_with_images(graph, subNode: np.array,
                        departements : list,
                        features : list, sinister : str,
                        dir_mask : Path,
                        dir_data : Path,
                        dir_train : Path,
                        resolution : str,
                        dates : np.array,
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

    logging.info(f'Shape of X {X.shape}, {np.unique(X[:,3])}')
    for departement in departements:
        logger.info(departement)

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
                date = dates[unDate]
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
            if arrayPop is None:
                arrayPop = np.full(mask.shape, fill_value=-1)

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
                    if arrayPop is None:
                        continue
                    save_values(arrayPop, 'population', index, maskNode)

                if 'elevation' in features:
                    if arrayEl is None:
                        continue
                    save_values(arrayEl, 'elevation', index, maskNode)

                if 'highway' in features:
                    if arrayOS is None:
                        continue
                    for var in osmnx_variables:
                        save_values(arrayOS[int(var), :, :], osmnxint2str[var], index, maskNode)

                if 'foret' in features:
                    if arrayForet is None:
                        continue
                    for var in foret_variables:
                        save_values(arrayForet[int(var), :, :], foretint2str[var], index, maskNode)

                if 'landcover' in features:
                    if 'foret_encoder' in landcover_variables:
                        if arrayForetLandcover is None:
                            continue
                        save_value_with_encoding(arrayForetLandcover, 'foret_encoder', index, maskNode, encoder_foret)
                    if 'highway_encoder' in landcover_variables:
                        if arrayOSLand is None:
                            continue
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
                    if arraySent is None:
                        continue
                    for band, var in enumerate(sentinel_variables):
                        save_values(arraySent[band,:, :,int(node[4])], var , index, maskNode)
                
                if 'landcover' in features:
                    if arrayLand is None:
                        continue
                    if 'landcover_encoder' in landcover_variables:
                        save_value_with_encoding(arrayLand[:,:], 'landcover_encoder', index, maskNode, encoder_landcover)

                if 'dynamicWorld' in features:
                    if arrayDW is None:
                        continue
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
            arrayInfluence = read_object(name, Path(__file__).absolute().parent.resolve()/ 'log' / resolution)
            if arrayInfluence is not None:
                for node in nodeDepartement:
                    index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                    maskNode = mask == node[0]

                    save_values(arrayInfluence[:,:, int(node[4] - 1)], historical_variables[0], index, maskNode)
                del arrayInfluence

        logger.info('AutoRegressionReg')
        if 'AutoRegressionReg' in features:
            dir_target_2 = dir_train / 'influence'
            arrayInfluence = read_object(f'{departement}InfluenceScale{graph.scale}_{graph_construct}.pkl', dir_target_2)
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
                if array is None:
                    continue
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
                if array is None:
                    continue
                for node in nodeDepartement:
                    index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                    maskNode = mask == node[0]
                    
                    save_values(array[:,:, int(node[4])], var, index, maskNode)
                del array

    return X, features_name

def add_varying_time_features(X : np.array, Y : np.array, newShape : int, features : list, features_name : list, ks : int, scale : int, methods : list):
    res = np.full((X.shape[0], newShape), fill_value=np.nan)
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
                
                
                if False not in np.isnan(X_node_copy[:, :, index_feat:index_feat+limit]):
                    continue

                res[index_node, index_new_feat:index_new_feat+limit] = \
                            func(X_node_copy[:, :, index_feat:index_feat+limit], axis=0)

    return res

def process_department(root_data: Path, root_raster: Path, dir_mask: Path, dir_encoder: Path, 
                        h3: gpd.GeoDataFrame, sinister: str, departement : str, resolution: str, variables: list):
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
    h3['scale0'] = h3.index
    h3.index = h3['hex_id']
    dico = h3['scale0'].to_dict()
    h3.reset_index(drop=True, inplace=True)

    allDates = find_dates_between('2017-06-12', '2023-09-11')
    date_to_choose = allDates[-1]
    index = allDates.index(date_to_choose)

    dir_data = root_data / departement / 'data'
    dir_raster = root_raster / departement / 'raster' / resolution
    mask = read_object(f'{departement}rasterScale0.pkl', dir_mask)
    if mask is None:
        mask,_,_  = rasterization(h3, resolutions[resolution]['y'], resolutions[resolution]['x'], f'scale0', Path('log'), 'ori')
    mask = mask[0]

    for var in variables:
        logger.info(var)
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

    return h3