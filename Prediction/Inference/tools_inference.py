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
        kernel[:,:,dim[2]//2:] = 0.0

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
        hexaFire['is'+sinisterType] = 0
        hexaFire['nb'+sinisterType] = 0

        rasterVar, _, _ = rasterization(hexaFire, n_pixel_y, n_pixel_x, 'nb'+sinisterType, dir_output, dept+'_bin0')

        spatioTemporalRaster[nonNanMask[:, 0], nonNanMask[:, 1], i] = rasterVar[0][nonNanMask[:, 0], nonNanMask[:, 1]].astype(int)

    return spatioTemporalRaster