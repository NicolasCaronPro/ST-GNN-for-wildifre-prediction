import firedanger
import copy
import meteostat
import datetime as dt
from pathlib import Path
import random
random.seed(0)
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import shape, Point
import math
from osgeo import gdal, ogr
from scipy.signal import fftconvolve as scipy_fft_conv
from astropy.convolution import convolve_fft
import rasterio
import rasterio.features
import rasterio.warp
from skimage import img_as_float
from skimage import transform
from sklearn.linear_model import LinearRegression
import pickle
import json
import warnings
from skimage import morphology
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, normalize
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.geometry import box
from skimage import morphology
from skimage.segmentation import watershed
from scipy.interpolate import interp1d
from scipy import ndimage as ndi
from skimage.filters import rank
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from scipy.interpolate import griddata
import sys
import time
import requests

##################################################################################################
#                                       Meteostat
##################################################################################################

def construct_historical_meteo(start, end, region, dir_meteostat):
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

    START = dt.datetime.strptime(start, '%Y-%m-%d') #- dt.timedelta(days=10)
    END = dt.datetime.strptime(end, '%Y-%m-%d')

    END += dt.timedelta(hours=1)

    data, data12h, liste, datakbdi = {}, {}, [], {}
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
            data[point] = data[point].normalize()
            data[point] = data[point].interpolate()
            data[point] = data[point].fetch()
            assert len(data[point]) > 0
            data[point]['snow'].fillna(0, inplace=True)
            data[point].drop(['tsun', 'coco', 'wpgt'], axis=1, inplace=True)
            data[point].reset_index(inplace=True)
            
            for k in sorted(data[point].columns):
                if data[point][k].isna().sum() > MAX_NAN:
                    print(f"{k} du point {point} possède trop de NaN ({data[point][k].isna().sum()}")
                    if last_point is None:
                        assert False
                    data[point][k] = last_point[k].values

                elif data[point][k].isna().sum() > 0:
                    x = data[point][data[point][k].notna()].index
                    y = data[point][data[point][k].notna()][k]

                    f2 = interp1d(x, y, kind='linear', fill_value='extrapolate')
                    
                    newx = data[point][data[point][k].isna()].index
                    newy = f2(newx)

                    data[point].loc[newx, k] = newy

       
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

        data12h[point] = copy.deepcopy(data[point])
        datakbdi[point] = copy.deepcopy(data[point])

        for k in range(0, 24):
            data12h[point][f"prcp+{k}"] = data12h[point].prcp.shift(k)
            datakbdi[point][f"prcp+{k}"] = datakbdi[point].prcp.shift(k)

        data12h[point]['prec24h'] = data12h[point][[f"prcp+{k}" for k in range(0, 24)]].sum(axis=1)
        datakbdi[point]['prec24h'] = datakbdi[point][[f"prcp+{k}" for k in range(0, 24)]].sum(axis=1)

        data12h[point] = data12h[point][['temp',
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
        
        data12h[point]['wspd'] =  data12h[point]['wspd'] * 1000 / 3600
        datakbdi[point]['wspd'] =  datakbdi[point]['wspd'] * 1000 / 3600

        data16h = data12h[point].loc[data12h[point].index.hour == 16].reset_index()
        data15h = data12h[point].loc[data12h[point].index.hour == 15].reset_index()
        data12h[point] = data12h[point].loc[data12h[point].index.hour == 12]

        treshPrec24 = 1.8

        #### data12h
        data12h[point].reset_index(inplace=True)
        data12h[point].loc[0, 'dc'] = 15
        data12h[point].loc[0, 'ffmc'] = 85
        data12h[point].loc[0, 'dmc'] = 6
        data12h[point].loc[0, 'nesterov'] = 0
        data12h[point].loc[0, 'munger'] = 0
        data12h[point].loc[0, 'kbdi'] = 0
        data12h[point].loc[0, 'days_since_rain'] = int(data12h[point].loc[0, 'prec24h'] > treshPrec24)
        data12h[point].loc[0, 'sum_consecutive_rainfall'] = data12h[point].loc[0, 'prec24h']
        data12h[point]['rhum'] = data12h[point]['rhum'].apply(lambda x: min(x, 100))

        datakbdi[point].reset_index(inplace=True)
        datakbdi[point]['day'] = datakbdi[point]['creneau'].apply(lambda x : x.date())

        tmax = datakbdi[point].groupby('day')['temp'].max()
        prec = datakbdi[point].groupby('day')['prcp'].sum()
        leni = len(data12h[point])

        consecutive = 0
        print(leni)
        for i in range(1, len(data12h[point])):
            #print(i, '/', leni, data12h[point].loc[i, 'creneau'], consecutive)
            if consecutive == 3:
                data12h[point].loc[i, 'dc'] = firedanger.indices.dc(data12h[point].loc[i, 'temp'],
                                                    data12h[point].loc[i, 'prec24h'],
                                                    data12h[point].loc[i, 'creneau'].month,
                                                    point[0],
                                                    data12h[point].loc[i - 1, 'dc'])
                if data12h[point].loc[i, 'dc'] > 1000:
                    print(data12h[point].loc[i - 1, 'dc'], data12h[point].loc[i, 'dc'], data12h[point].loc[i, 'prec24h'], data12h[point].loc[i-1, 'days_since_rain'])

                data12h[point].loc[i, 'ffmc'] = firedanger.indices.ffmc(data12h[point].loc[i, 'temp'],
                                                    data12h[point].loc[i, 'prec24h'],
                                                    data12h[point].loc[i, 'wspd'],
                                                    data12h[point].loc[i, 'rhum'],
                                                    data12h[point].loc[i - 1, 'ffmc'])
                data12h[point].loc[i, 'dmc'] = firedanger.indices.dmc(data12h[point].loc[i, 'temp'],
                                                    data12h[point].loc[i, 'prec24h'],
                                                    data12h[point].loc[i, 'rhum'],
                                                    data12h[point].loc[i, 'creneau'].month,
                                                    point[0],
                                                    data12h[point].loc[i - 1, 'dmc'])
            else:
                data12h[point].loc[i, 'dc'] = data12h[point].loc[i - 1, 'dc']
                data12h[point].loc[i, 'ffmc'] = data12h[point].loc[i - 1, 'ffmc']
                data12h[point].loc[i, 'dmc'] = data12h[point].loc[i - 1, 'dmc']

                if data12h[point].loc[i, 'temp'] >= 12:
                    consecutive += 1
                else:
                    consecutive = 0
            
            data12h[point].loc[i, 'nesterov'] = firedanger.indices.nesterov(data15h.loc[i, 'temp'], data15h.loc[i, 'rhum'], data15h.loc[i, 'prec24h'],
                                                                   data12h[point].loc[i - 1, 'nesterov'])
            
            data12h[point].loc[i, 'munger'] = firedanger.indices.munger(data12h[point].loc[i, 'prec24h'], data12h[point].loc[i - 1, 'munger'])
            data12h[point].loc[i, 'days_since_rain'] = data12h[point].loc[i - 1, 'days_since_rain'] + 1 if data12h[point].loc[i, 'prec24h'] < treshPrec24 else 0
            
            creneau = data12h[point].loc[i, 'creneau'].date()
            sum_consecutive_rainfall = prec[(prec.index >= creneau - dt.timedelta(days=data12h[point].loc[i, 'days_since_rain'])) & (prec.index <= creneau)].sum()
            data12h[point].loc[i, 'sum_consecutive_rainfall'] = sum_consecutive_rainfall
            sum_last_7_days = prec[(prec.index >= creneau - dt.timedelta(7)) & (prec.index <= creneau)].sum()
            data12h[point].loc[i, 'sum_last_7_days'] = sum_last_7_days
            preci = prec[prec.index == creneau].values[0]
            tmaxi = tmax[tmax.index == creneau].values[0]
            data12h[point].loc[i, 'kbdi'] = firedanger.indices.kbdi(tmaxi, preci, data12h[point].loc[i - 1, 'kbdi'], sum_consecutive_rainfall, sum_last_7_days, 30, 3.467326)
        
        data12h[point]['isi'] = data12h[point].apply(lambda x: firedanger.indices.isi(x.wspd, x.ffmc), axis=1)
        data12h[point]['angstroem'] = data12h[point].apply(lambda x: firedanger.indices.angstroem(x.temp, x.rhum), axis=1)
        data12h[point]['bui'] = firedanger.indices.bui(data12h[point]['dmc'], data12h[point]['dc'])
        data12h[point]['fwi'] = firedanger.indices.bui(data12h[point]['isi'], data12h[point]['bui'])
        data12h[point]['daily_severity_rating'] = data12h[point].apply(lambda x: firedanger.indices.daily_severity_rating(x.fwi), axis=1)

        var16 = ['temp','dwpt', 'rhum','prcp','wdir','wspd','prec24h']
        for v in var16:
            data12h[point][f"{v}16"] = data16h[v].values

        for k in sorted(data12h[point].columns):
            if data12h[point][k].isna().sum() > MAX_NAN:
                print(f"{k} du point {point} possède trop de NaN ({data12h[point][k].isna().sum()}")
                if last_point is None:
                    assert False
                data12h[point][k] = last_point[k].values

            elif data12h[point][k].isna().sum() > 0:
                x = data12h[point][data12h[point][k].notna()].index
                y = data12h[point][data12h[point][k].notna()][k]

                f2 = interp1d(x, y, kind='linear', fill_value='extrapolate')
                
                newx = data12h[point][data12h[point][k].isna()].index
                newy = f2(newx)

                data12h[point].loc[newx, k] = newy

        data12h[point]['latitude'] = point[0]
        data12h[point]['longitude'] = point[1]
        res.append(data12h[point].reset_index())

    def get_date(x):
        return x.strftime('%Y-%m-%d')

    res = pd.concat(res)

    res['creneau'] =  res['creneau'].apply(get_date)
    
    return res

def check_and_create_path(path: Path):
    """
    Creer un dossier s'il n'existe pas
    """
    path_way = path.parent if path.is_file() else path

    path_way.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()

def create_grid_cems(df, time, delta, variables):
    interdate = dt.datetime.strptime(time, '%Y-%m-%d') - dt.timedelta(days=delta)
    interdate = interdate.strftime('%Y-%m-%d')
    dff = df[(df["creneau"] >= interdate) & (df["creneau"] <= time)]
    dff[variables] = dff[variables].astype(float)
    if len(dff) == 0:
        return None
    return dff.reset_index(drop=True)

def interpolate_gridd(var, grid, newx, newy, met):
    x = grid['longitude'].values
    y = grid["latitude"].values
    points = np.zeros((y.shape[0], 2))
    points[:,0] = x
    points[:,1] = y
    return griddata(points, grid[var].values, (newx, newy), method=met)

def create_dict_from_arry(array):
    res = {}
    for var in array:
        res[var] = 0
    return res

def myRasterization(geo, tif, maskNan, sh, column):

    res = np.full(sh, np.nan, dtype=float)
    
    if maskNan is not None:
        res[maskNan[:,0], maskNan[:,1]] = np.nan

    for index, row in geo.iterrows():
        #inds = indss[row['hex_id']]
        """ mask = np.zeros((sh[0], sh[1]))
        cv2.fillConvexPoly(mask, inds, 1)
        mask = mask > 0"""
        mask = tif == row['cluster']
        res[mask] = row[column]

    return res

def rasterise_meteo_data(h3, maskh3, cems, sh, dates, dir_output):
    cems_variables = ['temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
                    'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                    'isi', 'angstroem', 'bui', 'fwi', 'daily_severity_rating',
                    'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
                    'days_since_rain', 'sum_consecutive_rainfall',
                    'sum_last_7_days'
                    ]
    
    lenDates = len(dates)
    print(lenDates)
    for var in cems_variables:
        print(var)
        spatioTemporalRaster = np.full((sh[0], sh[1], lenDates), np.nan)

        for i, date in enumerate(dates):
            if i % 200 == 0:
                print(date)
            #ddate = dt.datetime.strptime(date, "%Y-%m-%d")
            cems_grid = create_grid_cems(cems, date, 0, var)

            h3[var] = interpolate_gridd(var, cems_grid, h3.longitude.values, h3.latitude.values, 'cubic')

            rasterVar = myRasterization(h3, maskh3, None, maskh3.shape, var)

            if rasterVar.shape != sh:
                rasterVar = resize(rasterVar, sh[0], sh[1], 1)

            spatioTemporalRaster[:,:, i] = rasterVar

        outputName = var+'raw.pkl'
        f = open(dir_output / outputName,"wb")
        pickle.dump(spatioTemporalRaster, f)

def rasterise_vigicrues(h3, maskh3, cems, sh, dates, dir_output, name):
    
    lenDates = len(dates)
    print(lenDates)
    spatioTemporalRaster = np.full((sh[0], sh[1], lenDates), np.nan)

    for i, date in enumerate(dates):
        if i % 200 == 0:
            print(date)
        #ddate = dt.datetime.strptime(date, "%Y-%m-%d")
        cems_grid = create_grid_cems(cems, date, 0, 'hauteur')

        h3['hauteur'] = interpolate_gridd('hauteur', cems_grid, h3.longitude.values, h3.latitude.values, 'linear')

        rasterVar = myRasterization(h3, maskh3, None, maskh3.shape, 'hauteur')

        if rasterVar.shape != sh:
            rasterVar = resize(rasterVar, sh[0], sh[1], 1)

        spatioTemporalRaster[:,:, i] = rasterVar

        outputName = name+'.pkl'
        f = open(dir_output / outputName,"wb")
        pickle.dump(spatioTemporalRaster, f)

def rasterise_air_qualite(h3, maskh3, air, sh, dates, dir_output):
    air_variables = ['NO2', 'O3', 'PM10', 'PM25',
                     'NO216', 'O316', 'PM1016', 'PM2516']
    
    lenDates = len(dates)
    print(lenDates)
    for var in air_variables:
        print(var)
        spatioTemporalRaster = np.full((sh[0], sh[1], lenDates), np.nan)

        for i, date in enumerate(dates):
            if i % 200 == 0:
                print(date)
            #ddate = dt.datetime.strptime(date, "%Y-%m-%d")
            air_grid = create_grid_cems(air, date, 0, var)
            air_grid.dropna(subset=var, inplace=True)
            if np.unique(air_grid[var].shape[0] != 0):
                #try:
                #    h3[var] = interpolate_gridd(var, air_grid, h3.longitude.values, h3.latitude.values, 'linear')
                #except:
                h3[var] = interpolate_gridd(var, air_grid, h3.longitude.values, h3.latitude.values, 'nearest')

                rasterVar = myRasterization(h3, maskh3, None, maskh3.shape, var)

                if rasterVar.shape != sh:
                    rasterVar = resize(rasterVar, sh[0], sh[1], 1)

                spatioTemporalRaster[:,:, i] = rasterVar

        outputName = var+'raw.pkl'
        f = open(dir_output / outputName,"wb")
        pickle.dump(spatioTemporalRaster, f)


def find_dates_between(start, end):
    start_date = dt.datetime.strptime(start, '%Y-%m-%d').date()
    end_date = dt.datetime.strptime(end, '%Y-%m-%d').date()

    delta = dt.timedelta(days=1)
    date = start_date
    res = []
    while date < end_date:
            res.append(date.strftime("%Y-%m-%d"))
            date += delta
    return res

def get_hourly(x):
    return x.hour

def get_date(x):
    return x.date().strftime('%Y-%m-%d')


warnings.filterwarnings("ignore")


##################################################################################################
#                                       Spatial
##################################################################################################

def read_tif(name):
    """
    Open a satellite images and return bands, latitude and longitude of each pixel.
    """
    with rasterio.open(name) as src:
        dt = src.read()
        height = dt.shape[1]
        width = dt.shape[2]
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        lons = np.array(xs)
        lats = np.array(ys)
        src.close()
    return dt, lons, lats

def find_pixel(lats, lons, lat, lon):

    lonIndex = (np.abs(lons - lon)).argmin()
    latIndex = (np.abs(lats - lat)).argmin()

    lonValues = lons.reshape(-1,1)[lonIndex]
    latValues = lats.reshape(-1,1)[latIndex]
    #print(lonValues, latValues)
    return np.where((lons == lonValues) & (lats == latValues))

def resize(input_image, height, width, dim):
    """
    Resize the input_image into heigh, with, dim
    """
    img = img_as_float(input_image)
    img = transform.resize(img, (dim, height, width), mode='constant', order=0,
                 preserve_range=True, anti_aliasing=True)
    return np.asarray(img)

def resize_no_dim(input_image, height, width, mode='constant', order=0,
                 preserve_range=True, anti_aliasing=True):
    """
    Resize the input_image into heigh, with, dim
    """
    img = img_as_float(input_image)
    img = transform.resize(img, (height, width), mode=mode, order=order,
                 preserve_range=preserve_range, anti_aliasing=anti_aliasing)
    return np.asarray(img)

def create_geocube(df, variables, reslons, reslats):
    """
    Create a image representing variables with the corresponding resolution from df
    """
    geo_grid = make_geocube(
        vector_data=df,
        measurements=variables,
        resolution=(reslons, reslats),
        rasterize_function=rasterize_points_griddata,
        fill = 0
    )
    return geo_grid

def raster_population(tifFile, tifFile_high, dir_output, reslon, reslat, dir_data):
    population = pd.read_csv(dir_data / 'population' / 'population.csv')
    population = gpd.GeoDataFrame(population, geometry=gpd.points_from_xy(population.longitude, population.latitude))

    population = create_geocube(population, ['population'], -reslon, reslat)
    population = population.to_array().values[0]
    population = resize_no_dim(population, tifFile.shape[0], tifFile.shape[1])

    mask = np.argwhere(np.isnan(tifFile))
    population[mask[:,0], mask[:,1]] = np.nan

    """res = np.zeros((tifFile.shape[0], tifFile.shape[1]))

    unodes = np.unique(tifFile)
    for node in unodes:
        mask1 = tifFile == node
        mask2 = tifFile_high == node
        if True not in np.unique(population[mask2]):
            continue
        res[mask1] = np.nanmax(population[mask2])

    outputName = 'population22.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(population,f)

    outputName = 'population2.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res,f)"""

    outputName = 'population.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(population,f)

def raster_elevation(tifFile, dir_output, elevation):
    elevation = resize(elevation, tifFile.shape[0], tifFile.shape[1], 1)[0]
    minusMask = np.argwhere(tifFile == -1)
    minusMask = np.argwhere(np.isnan(tifFile))
    elevation[minusMask[:,0], minusMask[:,1]] = np.nan
    outputName = 'elevation.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(elevation,f)

def raster_foret(tifFile, tifFile_high, dir_output, reslon, reslat, dir_data, dept):
    foret = gpd.read_file(dir_data / 'BDFORET' / 'foret.geojson')

    foret = rasterisation(foret, reslat, reslon, 'code', defval=0, name=dept)
    foret = resize_no_dim(foret, tifFile_high.shape[0], tifFile_high.shape[1])
    bands = np.unique(foret).astype(int)
    res = np.full((np.max(bands) + 1, tifFile.shape[0], tifFile.shape[1]), fill_value=0.0)
    res2 = np.full((tifFile.shape[0], tifFile.shape[1]), fill_value=np.nan)
    unodes = np.unique(tifFile)
    for node in unodes:

        if node not in tifFile_high:
            continue

        mask1 = tifFile == node
        mask2 = tifFile_high == node

        for band in bands:
            res[band, mask1] = (np.argwhere(foret[mask2] == band).shape[0] / foret[mask2].shape[0]) * 100

        if res[:, mask1].shape[1] == 1:
            res2[mask1] = np.nanargmax(res[:, mask1])
        else:
            res2[mask1] = np.nanargmax(res[:, mask1][:,0])

    res[:, np.isnan(tifFile)] = np.nan

    outputName = 'foret.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res,f)

    outputName = 'foret_landcover.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res2,f)


def raster_osmnx(tifFile, tifFile_high, dir_output, reslon, reslat, dir_data):
    osmnx_, _, _ = read_tif(dir_data / 'osmnx' / 'osmnx.tif')
    osmnx_ = osmnx_[0]
    osmnx = osmnx_ > 0
    mask = np.isnan(tifFile_high)
    osmnx = resize_no_dim(osmnx, tifFile_high.shape[0], tifFile_high.shape[1])
    osmnx = influence_index(osmnx, mask)
    mask = np.isnan(tifFile_high)
    osmnx[mask] = np.nan
    res = np.zeros((tifFile.shape[0], tifFile.shape[1]))

    #osmnx = gpd.read_file(dir_data / 'spatial' / 'hexagones.geojson')
    #osmnx = rasterisation(osmnx, reslat, reslon, 'osmnx')

    unodes = np.unique(tifFile)
    for node in unodes:
        mask1 = tifFile == node
        mask2 = tifFile_high == node
        if True not in np.unique(mask2):
            continue
        res[mask1] = np.nanmean(osmnx[mask2])

    """outputName = 'osmnx22.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(osmnx,f)"""

    outputName = 'osmnx.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res,f)

def raster_water(tifFile, tifFile_high, dir_output, reslon, reslat, dir_data):
    dir_sat = dir_data / 'GEE' / 'landcover' 
    water, _, _ = read_tif(dir_sat / 'summer.tif')
    water = water[-1]
    water = water == 0
    mask = np.isnan(tifFile_high)
    water = resize_no_dim(water, tifFile_high.shape[0], tifFile_high.shape[1])
    water[mask] = np.nan
    water = influence_index(water, mask)
    mask = np.isnan(tifFile_high)
    res = np.zeros((tifFile.shape[0], tifFile.shape[1]))

    #osmnx = gpd.read_file(dir_data / 'spatial' / 'hexagones.geojson')
    #osmnx = rasterisation(osmnx, reslat, reslon, 'osmnx')

    unodes = np.unique(tifFile)
    for node in unodes:
        mask1 = tifFile == node
        mask2 = tifFile_high == node
        if True not in np.unique(mask2):
            continue
        res[mask1] = np.nanmean(water[mask2])

    """outputName = 'osmnx22.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(osmnx,f)"""

    outputName = 'water.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res,f)

def raster_sat(base, dir_reg, dir_output, dates):
    size = '30m'
    dir_sat = dir_reg / 'GEE' / size
    res = np.full((5, base.shape[0], base.shape[1],  len(dates)), np.nan)
    minusMask = np.argwhere(np.isnan(base))
    
    for tifFile in dir_sat.glob('sentinel/*.tif'):
        
        tifFile = tifFile.as_posix()
        dateFile = tifFile.split('/')[-1]
        date = dateFile.split('.')[0]

        if date not in dates:
            continue
        
        i = dates.index(date)
        print(dateFile, i)

        sentinel, _, _ = read_tif(tifFile)
        
        for band in range(sentinel.shape[0]):
            values = resize_no_dim(sentinel[band], base.shape[0], base.shape[1])
            values[np.isnan(values)] = np.nanmean(values)
            if i + 9 > res.shape[3]:
                lmin = i-7
                lmax = res.shape[3]
            elif i == 0:
                lmin = 0
                lmax = i + 10
            else:
                lmin = i-7
                lmax = i + 10

            k = lmax - lmin
            res[band,:,:, lmin:lmax] = np.repeat(values[:,:, np.newaxis], k, axis=2)

            res[:, minusMask[:,0], minusMask[:,1], :] = np.nan

    
    res[:, minusMask[:,0], minusMask[:,1], :] = np.nan

    outputName = 'sentinel.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res,f)
    res = np.full((base.shape[0], base.shape[1], len(dates)), np.nan)

def raster_land(base, base_high, dir_reg, dir_output, dates):

    size = '30m'
    dir_sat = dir_reg / 'GEE' / size
    bands = [0,1,2,3,4,5,6,7,8]
    res = np.full((len(bands), base.shape[0], base.shape[1],  len(dates)), np.nan)
    unodes = np.unique(base)
    minusMask = np.argwhere(np.isnan(base))
    for tifFile in dir_sat.glob('dynamic_world/*.tif'):

        tifFile = tifFile.as_posix()
        dateFile = tifFile.split('/')[-1]
        print(dateFile)

        date = dateFile.split('.')[0]
        if date not in dates:
            continue

        i = dates.index(date)

        dynamicWorld, _, _ = read_tif(dir_sat / 'dynamic_world' / dateFile)
        dynamicWorld = dynamicWorld.astype(np.float64)
        dynamicWorld[np.isnan(dynamicWorld)] = -1
        dynamicWorld = dynamicWorld[-1]
        dynamicWorld = resize_no_dim(dynamicWorld, base_high.shape[0], base_high.shape[1])

        if i + 9 > res.shape[2]:
            lmin = i - 7
            lmax = res.shape[2]
        elif i == 0:
            lmin = 0
            lmax = i + 10
        else:
            lmin = i - 7
            lmax = i + 10

        for node in unodes: 
            mask1 = res == node
            mask2 = base_high == node
            if mask2.shape[0] == 0:
                continue
            for band in bands:
                res[band,mask1,lmin:lmax] = np.argwhere(dynamicWorld[mask2] == band).shape[0]

        i += 15

    res[minusMask[:,0], minusMask[:,1], :] = np.nan
    outputName = 'landcover.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res,f)

def rasterisation(h3, lats, longs, column='cluster', defval = np.nan, name='default'):
    #h3['cluster'] = h3.index

    h3.to_file(name+'.geojson', driver='GeoJSON')

    input_geojson = name+'.geojson'
    output_raster = name+'.tif'

    # Si on veut rasteriser en fonction de la valeur d'un attribut du vecteur, mettre son nom ici 
    attribute_name = column

    # Taille des pixels
    if isinstance(lats, float):
        pixel_size_y = lats
        pixel_size_x = longs
    else:
        pixel_size_x = abs(longs[0][0] - longs[0][1])
        pixel_size_y = abs(lats[0][0] - lats[1][0])
    print(f'px {pixel_size_x}, py {pixel_size_y}')
    #pixel_size_x = res[dim][0]
    #pixel_size_y = res[dim][1]

    source_ds = ogr.Open(input_geojson)
    source_layer = source_ds.GetLayer()

    # On obtient l'étendue du raster
    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    # On calcule le nombre de pixels
    width = int((x_max - x_min) / pixel_size_x)
    height = int((y_max - y_min) / pixel_size_y)

    # Oncrée un nouveau raster dataset et on passe de "coordonnées image" (pixels) à des coordonnées goréférencées
    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(output_raster, width, height, 1, gdal.GDT_Float32)
    output_ds.GetRasterBand(1).Fill(defval)
    output_ds.SetGeoTransform([x_min, pixel_size_x, 0, y_max, 0, -pixel_size_y])
    output_ds.SetProjection(source_layer.GetSpatialRef().ExportToWkt())

    if attribute_name != '' :
        # On  rasterise en fonction de l'attribut donné
        gdal.RasterizeLayer(output_ds, [1], source_layer, options=["ATTRIBUTE=" + attribute_name])
    else :
        # On  rasterise. Le raster prend la valeur 1 là où il y a un vecteur
        gdal.RasterizeLayer(output_ds, [1], source_layer)

    output_ds = None
    source_ds = None

    res, _, _ = read_tif(name+'.tif')
    return res[0]

def myFunctionDistanceDugrandCercle(outputShape, earth_radius=6371.0, resolution_lon=0.0002694945852352859, resolution_lat=0.0002694945852326214):
    half_rows = outputShape[0] // 2
    half_cols = outputShape[1] // 2

    # Créer une grille de coordonnées géographiques avec les résolutions souhaitées
    latitudes = np.linspace(-half_rows * resolution_lat, half_rows * resolution_lat, outputShape[0])
    longitudes = np.linspace(-half_cols * resolution_lon, half_cols * resolution_lon, outputShape[1])
    latitudes, longitudes = np.meshgrid(latitudes, longitudes, indexing='ij')

    # Coordonnées du point central
    center_lat = latitudes[outputShape[0] // 2, outputShape[1] // 2]
    center_lon = longitudes[outputShape[0] // 2, outputShape[1] // 2]
    print(center_lat, center_lon)
    
    # Convertir les coordonnées géographiques en radians
    latitudes_rad = np.radians(latitudes)
    longitudes_rad = np.radians(longitudes)

    # Calculer la distance du grand cercle entre chaque point et le point central
    delta_lon = longitudes_rad - np.radians(center_lon)
    delta_lat = latitudes_rad - np.radians(center_lat)
    a = np.sin(delta_lat/2)**2 + np.cos(latitudes_rad) * np.cos(np.radians(center_lat)) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distances = earth_radius * c
    
    return distances

def influence_index(categorical_array, mask):
    res = np.full(categorical_array.shape, np.nan)

    kernel = myFunctionDistanceDugrandCercle((30,30))
    #kernel = normalize(kernel, norm='l2')
    #res[mask[:,0], mask[:,1]] = (scipy_fft_conv(categorical_array, kernel, mode='same')[mask[:,0], mask[:,1]])

    res = convolve_fft(categorical_array, kernel, normalize_kernel=True, mask=mask)

    return res

def add_osmnx(clusterSum, dataset, dir_reg, kmeans, tifFile):
    print('Add osmnx')
    dir_data = dir_reg / 'osmnx'

    tif, _, _ = read_tif(dir_data / 'osmnx.tif')
    
    sentinel, lons, lats = read_tif(tifFile)
    nanmask = np.argwhere(np.isnan(sentinel[0]))
    sentinel = sentinel[0]

    coord = np.empty((sentinel.shape[0], sentinel.shape[1], 2))
    tif = resize(tif, sentinel.shape[0], sentinel.shape[1], 1).astype(int)[0]
    coord[:,:,0] = lats
    coord[:,:,1] = lons

    clustered = kmeans.predict(np.reshape(coord, (-1, coord.shape[-1])))
    clustered = clustered.reshape(sentinel.shape[0], sentinel.shape[1]).astype(float)
    clustered[nanmask[:,0], nanmask[:,1]] = np.nan

    mask = np.argwhere(~np.isnan(tif))
    density = influence_index(tif.astype(int), mask)

    dataset['highway_min'] = 0
    dataset['highway_max'] = 0
    dataset['highway_std'] = 0
    dataset['highway_mean'] = 0
    dataset['highway_sum'] = 0

    for cluster in clusterSum['cluster'].unique():
        clusterMask = clustered == cluster
        indexDataset = dataset[dataset['cluster'] == cluster].index.values

        dataset.loc[indexDataset, 'highway_sum'] = np.nansum(density[clusterMask])

        dataset.loc[indexDataset, 'highway_max'] = np.nanmax(density[clusterMask])

        dataset.loc[indexDataset, 'highway_min'] = np.nanmin(density[clusterMask])

        dataset.loc[indexDataset, 'highway_mean'] = np.nanmean(density[clusterMask])

        dataset.loc[indexDataset, 'highway_std'] =  np.nanstd(density[clusterMask])
    
    return dataset

##################################################################################################
#                                       Cluster
##################################################################################################

def get_cluster(x, y, z, cluster, scaler):
    #X = scaler.transform([[x, y]])
    X = [[x, y]]
    return cluster.predict(X)[0]

def get_cluster_polygon(polygon, z, scaler, cluster):
    center = polygon.centroid
    x = float(center.y)
    y = float(center.x)
    #X = scaler.transform([[x, y]])
    X = [[x, y]]
    return cluster.predict(X)[0]

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)

def get_cluster_latitude(x, cluster):
    x = int(x)
    return cluster.cluster_centers_[x][0]

def get_cluster_longitude(x, cluster):
    x = int(x)
    return cluster.cluster_centers_[x][1]

def get_cluster_altitude(x, cluster):
    x = int(x)
    return cluster.cluster_centers_[x][2]

def read_object(filename: str, path : Path):
    if not (path / filename).is_file():
        print(f'{path / filename} not found')
        return None
    return pickle.load(open(path / filename, 'rb'))