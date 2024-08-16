import pickle
from pathlib import Path
import datetime as dt
import pandas as pd
import rasterio
import rasterio.features
import rasterio.warp
import osmnx as ox
from osgeo import gdal, ogr
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from skimage import img_as_float
from sklearn.preprocessing import normalize
from scipy.signal import fftconvolve as scipy_fft_conv
import random
import geopandas as gpd
from torch import optim
from tqdm import tqdm
from copy import copy
import torch
import math
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from ngboost import NGBClassifier, NGBRegressor
from sklearn.decomposition import PCA
from sklearn.svm import SVR
import sys
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, balanced_accuracy_score, \
mean_absolute_error
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from scipy.interpolate import griddata
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata
from skimage import img_as_float
from category_encoders import TargetEncoder, CatBoostEncoder
import convertdate
from skimage import transform
from arborescence import *
from array_fet import *
from weigh_predictor import *
from features_selection import *
from config import *
from sklearn.cluster import SpectralClustering
import scipy.stats
import plotly.express as px
import plotly.io as pio
from forecasting_models.sklearn_api_models_config import *
import warnings
import mlflow
from mlflow.models import infer_signature

if MLFLOW:
    try:
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    except Exception as e:
        logger.info(f'{e}')
        exit(1)

random.seed(42)

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


# Dictionnaire de gestion des départements
int2str = {
    1 : 'ain',
    25 : 'doubs',
    69 : 'rhone',
    78 : 'yvelines'
}

int2strMaj = {
    1 : 'Ain',
    25 : 'Doubs',
    69 : 'Rhone',
    78 : 'Yvelines'
}

int2name = {
    1 : 'departement-01-ain',
    25 : 'departement-25-doubs',
    69 : 'departement-69-rhone',
    78 : 'departement-78-yvelines'
}

str2int = {
    'ain': 1,
    'doubs' : 25,
    'rhone' : 69,
    'yvelines' : 78
}

str2intMaj = {
    'Ain': 1,
    'Doubs' : 25,
    'Rhone' : 69,
    'Yvelines' : 78
}

str2name = {
    'Ain': 'departement-01-ain',
    'Doubs' : 'departement-25-doubs',
    'Rhone' : 'departement-69-rhone',
    'Yvelines' : 'departement-78-yvelines'
}

name2str = {
    'departement-01-ain': 'Ain',
    'departement-25-doubs' : 'Doubs',
    'departement-69-rhone' : 'Rhone',
    'departement-78-yvelines' : 'Yvelines'
}

name2int = {
    'departement-01-ain': 1,
    'departement-25-doubs' : 25,
    'departement-69-rhone' : 69,
    'departement-78-yvelines' : 78
}

def create_larger_scale_image(input, proba, bin, raster):
    probaImageScale = np.full(proba.shape, np.nan)
    binImageScale = np.full(proba.shape, np.nan)
    
    clusterID = np.unique(input)
    for di in range(bin.shape[-1]):
        for id in clusterID:
            mask = np.argwhere(input == id)
            ones = np.ones(proba[mask[:,0], mask[:,1], di].shape)
            probaImageScale[mask[:,0], mask[:,1], di] = 1 - np.prod(ones - proba[mask[:,0], mask[:,1], di])
            unique_ids_in_mask = np.unique(raster[bin[mask[:,0], mask[:,1], di]])
            binImageScale[mask[:,0], mask[:,1], di] = np.sum()

    return None, binImageScale

def create_larger_scale_bin(input, bin, influence, raster):
    binImageScale = np.full(bin.shape, np.nan)
    influenceImageScale = np.full(influence.shape, np.nan)

    clusterID = np.unique(input)
    for di in range(bin.shape[-1]):
        for id in clusterID:
            mask = (input == id)
            if np.any(influence[mask, di] > 0):
                influence_mask = (influence[:, :, di] > 0) & mask
                unique_ids_in_mask = np.unique(raster[influence_mask])
                
                binval = 0
                influenceval = 0

                for uim in unique_ids_in_mask:
                    mask3 = (raster == uim)
                    binval += (np.unique(bin[mask3, di]))[0]
                    influenceval += (np.unique(influence[mask3, di]))[0]
                binImageScale[mask, di] = binval
                influenceImageScale[mask, di] = influenceval
            else:
                binImageScale[mask, di] = 0
                influenceImageScale[mask, di] = 0

    return binImageScale, influenceImageScale

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

allDates = find_dates_between('2017-06-12', '2024-06-29')
years = list(np.unique([d.split('-')[0] for d in allDates]))
#allDates = find_dates_between('2017-06-12', dt.datetime.now().date().strftime('%Y-%m-%d'))

def save_object(obj, filename: str, path : Path):
    check_and_create_path(path)
    with open(path / filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def save_object_torch(obj, filename : str, path : Path):
    check_and_create_path(path)
    torch.save(obj, path/filename)

def read_object(filename: str, path : Path):
    if not (path / filename).is_file():
        logger.info(f'{path / filename} not found')
        return None
    return pickle.load(open(path / filename, 'rb'))

def read_object_torch(filename: str, path : Path):
    return torch.load(open(path / filename, 'rb'))
        
def check_and_create_path(path: Path):
    """
    Creer un dossier s'il n'existe pas
    """
    path_way = path.parent if path.is_file() else path

    path_way.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()

def haversine(p1, p2, unit = 'kilometer'):
    import math
    # Coordinates in decimal degrees (e.g. 2.89078, 12.79797)
    lon1 = p1[0]
    lat1 = p1[1]
    lon2 = p2[0]
    lat2 = p2[1]

    R = 6371000  # radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    meters = R * c  # output distance in meters
    km = meters / 1000.0  # output distance in kilometers
    print(p1, p2, meters)
    meters = round(meters)
    km = round(km, 3)

    if unit == 'kilometer':
        return km
    elif unit == 'meters':
        return meters
    else:
        return math.inf

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

def funcSlope(e1, e2, p1, p2):
    distance = haversine(p1, p2) * 1000
    return ((e1 - e2) * 100) / distance

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
    logger.info(center_lat, center_lon)
    
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

def influence_index(raster, mask, dimS, mode, dim=(90,150)):

    res = {}
    dimX, dimY = dimS
    res = np.full(raster.shape, np.nan)
    if mode == 'laplace':
        kernel = myFunctionDistanceDugrandCercle(dim, resolution_lon=dimY, resolution_lat=dimX) + 1
        kernel = 1 / kernel
        kernel = normalize(kernel, norm='l2')
    else:
        kernel = np.full(dim, 1/(dim[0]*dim[1]), dtype=float)

    res[mask[:,0], mask[:,1]] = (scipy_fft_conv(raster, kernel, mode='same')[mask[:,0], mask[:,1]])
    return res

def stat(c1, c2, clustered, osmnx, bands):
    mask1 = clustered == c1
    mask2 = clustered == c2

    clusterMask1 = clustered.copy()
    clusterMask1[~mask1] = 0
    clusterMask1[mask1] = 1
    clusterMask1 = clusterMask1.astype(int)

    clusterMask2 = clustered.copy()
    clusterMask2[~mask2] = 0
    clusterMask2[mask2] = 1
    clusterMask2 = clusterMask2.astype(int)

    morpho1 = morphology.dilation(clusterMask1, morphology.disk(10))
    morpho2 = morphology.dilation(clusterMask2, morphology.disk(10))
    mask13 = (morpho1 == c1) & (morpho2 == c2)

    box = np.argwhere(mask13 == 1)

    if box.shape[0] == 0:
        return np.zeros(len(bands))
 
    indexXmin = np.argwhere(mask13 == 1)[:,0].min()
    indexXmax = np.argwhere(mask13 == 1)[:,0].max()
    indexYmin = np.argwhere(mask13 == 1)[:,1].min()
    indexYmax = np.argwhere(mask13 == 1)[:,1].max()

    cropOsmnx = osmnx[indexXmin: indexXmax, indexYmin:indexYmax].astype(int)

    res = []

    for band in bands:
        val = (np.argwhere(cropOsmnx == int(band)).shape[0] / cropOsmnx.shape[0]) * 100
        res.append(val)
    return np.asarray(res)

def rasterization(ori, lats, longs, column, dir_output, outputname='ori', defVal = np.nan):
    check_and_create_path(dir_output)

    ori.to_file(dir_output.as_posix() + '/' + outputname+'.geojson', driver="GeoJSON")
    
    # paths du geojson d'entree et du raster tif de sortie
    input_geojson = dir_output.as_posix() + '/' + outputname+'.geojson'
    output_raster = dir_output.as_posix() + '/' + outputname+'.tif'

    # Si on veut rasteriser en fonction de la valeur d'un attribut du vecteur, mettre son nom ici
    attribute_name = column

    # Taille des pixels
    if isinstance(lats, float):
        pixel_size_y = lats
        pixel_size_x = longs
    else:
        pixel_size_x = abs(longs[0][0] - longs[0][1])
        pixel_size_y = abs(lats[0][0] - lats[1][0])
        logger.info(f'px {pixel_size_x}, py {pixel_size_y}')
        
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
    output_ds.GetRasterBand(1).Fill(defVal)
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
    return read_tif(output_raster)

import pandas as pd
import numpy as np

def remove_nan_nodes_np(arr: np.array, target: np.array) -> np.ndarray:
    """
    Remove rows where any NaN values are present in the array.
    """
    if target is None:
        return arr[~np.isnan(arr).any(axis=1)], None
    return arr[~np.isnan(arr).any(axis=1)], target[~np.isnan(arr).any(axis=1)]

def remove_none_target_np(arr: np.array, target: int) -> np.ndarray:
    """
    Remove rows where the target column has -1 or NaN values.
    """
    target_column = target[:, -1]
    mask = (target_column != -1) & (~np.isnan(target_column))
    if arr is None:
        return None, target[mask]
    return arr[mask], target[mask]

def remove_nan_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where any NaN values are present in 'nodes' columns.
    """
    return df.dropna().reset_index(drop=True)

def remove_none_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where the target column has -1 or NaN values.
    """
    return df[(df['risk'] != -1) & (~df['risk'].isna())].reset_index(drop=True)

def remove_bad_period(df: pd.DataFrame, period2ignore: dict, departements: list) -> pd.DataFrame:
    global allDates, name2int

    bad_dates = np.array([], dtype=int)
    for dept in departements:
        period = period2ignore[name2int[dept]]['interventions']
        if period != []:
            for per in period:
                ds = per[0].strftime('%Y-%m-%d')
                de = per[1].strftime('%Y-%m-%d')

                if ds < allDates[0]:
                    ds = allDates[0]
                if de > allDates[-1]:
                    de = allDates[-1]

                if de not in allDates or ds not in allDates:
                    continue

                ds_idx = allDates.index(ds)
                de_idx = allDates.index(de)
                bad_dates = np.concatenate((bad_dates, np.arange(start=ds_idx, stop=de_idx)))

    for dept in departements:
        df = df[~((df['departement'] == name2int[dept]) & (df['date'].isin(bad_dates)))]

    return df.reset_index(drop=True)

def remove_non_fire_season(df: pd.DataFrame, SAISON_FEUX: dict, departements: list) -> pd.DataFrame:
    global allDates, name2int, years

    valid_indices = np.array([], dtype=int)
    for dept in departements:
        jour_debut = SAISON_FEUX[name2int[dept]]['jour_debut']
        mois_debut = SAISON_FEUX[name2int[dept]]['mois_debut']
        jour_fin = SAISON_FEUX[name2int[dept]]['jour_fin']
        mois_fin = SAISON_FEUX[name2int[dept]]['mois_fin']

        for year in years:
            date_debut = f'{year}-{mois_debut}-{jour_debut}'
            date_fin = f'{year}-{mois_fin}-{jour_fin}'

            if date_debut < allDates[0]:
                date_debut = allDates[0]
            if date_fin > allDates[-1]:
                date_fin = allDates[-1]

            if date_debut not in allDates or date_fin not in allDates:
                continue

            start_idx = allDates.index(date_debut)
            end_idx = allDates.index(date_fin)

            valid_indices = np.concatenate((valid_indices, df[(df['departement'] == name2int[dept]) & (df['date'] >= start_idx) & (df['date'] < end_idx)].index))

    df = df.loc[valid_indices]
    return df.reset_index(drop=True)

def add_k_temporal_node(k_days: int, nodes: np.array) -> np.array:

    logger.info(f'Add {k_days} temporal nodes')

    ori = np.empty((nodes.shape[0], 2))
    ori[:,0] = nodes[:,0]
    ori[:,1] = nodes[:,4]

    newSubNode = np.full((nodes.shape[0], nodes.shape[1] + 1), -1, dtype=nodes.dtype)
    newSubNode[:,:nodes.shape[1]] = nodes
    newSubNode[:,-1] = 1

    if k_days == 0:
        return newSubNode

    array = np.full((nodes.shape[0], nodes.shape[1] + 1), -1, dtype=nodes.dtype)
    array[:,:nodes.shape[1]] = nodes
    array[:,-1] = 1
    for k in range(1, k_days+1):
        array[:,4] = nodes[:,4] - k
        newSubNode = np.concatenate((newSubNode, array))

    indexToDel = np.argwhere(newSubNode[:,4] < 0)
    newSubNode = np.delete(newSubNode, indexToDel, axis=0)

    return np.unique(newSubNode, axis=0)

def add_new_random_nodes_from_nei(nei : np.array,
                  newSubNode : np.array,
                  node : np.array,
                  minNumber : int,
                  maxNumber : int,
                  graph_nodes : np.array,
                  nodes : np.array):
           
    maskNode = np.argwhere((np.isin(newSubNode[:,0], nei)) & (newSubNode[:,4] == node[4]))
    # Check for current nei in dataset
    if maskNode.shape[0] > 0:
        nei = np.delete(nei, np.argwhere(np.isin(nei, newSubNode[maskNode][:, 0])))

    maxNumber = min(nei.shape[0], maxNumber)
    minNumber = min(nei.shape[0], minNumber)
    if maxNumber == 0:
        return newSubNode
    if minNumber == maxNumber:
        number_of_new_nodes = 1
    else:
        number_of_new_nodes = random.randrange(minNumber, maxNumber)

    nei = np.random.choice(nei, number_of_new_nodes, replace=False)
    new_nodes = np.full((number_of_new_nodes, nodes.shape[1]), -1.0)
    new_nodes[:,:5] = graph_nodes[np.isin(graph_nodes[:,0], nei)]
    new_nodes[:,4] = node[4]

    newSubNode = np.concatenate((newSubNode, new_nodes))
    return newSubNode

def assign_graphID(nei : np.array,
                   newSubNode : np.array,
                   date : int,
                   graphId : int,
                   edges : np.array) -> np.array:
    
    maskNode = np.argwhere((newSubNode[:,0] == nei) & (newSubNode[:,4] == date) & (newSubNode[:,-1] != graphId))
   
    if maskNode.shape[0] == 0:
        #logger.info(nei, date, graphId)
        return newSubNode

    newSubNode[maskNode, -1] = graphId
    newNei = np.unique(edges[1][np.argwhere((edges[0] == nei) &
                                             (np.isin(edges[1], np.unique(newSubNode[:,0])))
                                             )].reshape(-1))
    
    newNei = np.delete(newNei, np.argwhere(np.isin(newNei, nei)))

    for ne in newNei:
        newSubNode = assign_graphID(ne, newSubNode, date, graphId, edges)
    return newSubNode

def generate_subgraph(graph, minNumber : int, maxNumber : int, nodes : np.array) -> np.array:
        """
        Generate a sub graph dataset from the corresponding nodes using self.edges. 
        nodes is a numpy array [id, lat, lon, dep, date]
        output a numpy arrat [id, lat, lon, dep, date, graphId] with graphId a unique ID for each graph.
        """
        assert maxNumber <= graph.numNei
        if maxNumber == 0:
            return nodes

        logger.info('Generate sub graph')
        newSubNode = np.full((nodes.shape[0], nodes.shape[1]), -1, dtype=nodes.dtype)
        newSubNode[:,:5] = nodes

        for i, node in enumerate(nodes):
            nei = graph.edges[1][np.argwhere(graph.edges[0] == node[0])].reshape(-1)
            if nei.shape[0] != 0:
                if maxNumber != 0:
                    newSubNode = add_new_random_nodes_from_nei(nei, newSubNode, node, minNumber, maxNumber, graph.nodes, nodes)
                    
        return newSubNode

def construct_graph_set(graph, date, X, Y, ks):
    """
    Construct indexing graph with nodes sort by their id and date and corresponding edges.
    We consider spatial edges and temporal edges
    date : date
    X : train or val nodes
    Y : train or val target
    ks : size of the time series
    """

    maskgraph = np.argwhere((X[:,4] == date))
    x = X[maskgraph[:,0]]
    if ks != 0:
        maskts = np.argwhere((np.isin(X[:,0], x[:,0]) & (X[:,4] < date) & (X[:,4] >= date - ks)))
        if maskts.shape[0] == 0:
            return None, None, None
        xts = X[maskts[:,0]]
        #xts[:,5] = 0
        x = np.concatenate((x, xts))

    if Y is not None:
        y = Y[maskgraph[:,0]]
        if ks != 0:
            yts = Y[maskts[:,0]]
            yts[:,-4] = 0
            y = np.concatenate((y, yts))

    # Graph indexing
    ind = np.lexsort((x[:,0], x[:,4]))
    x = x[ind]
    if Y is not None:
        y = y[ind]

    # Get graph specific spatial and temporal edges 
    mask = np.argwhere((np.isin(graph.edges[0], np.unique(x[:,0]))) & (np.isin(graph.edges[1], np.unique(x[:,0]))))
    maskTemp = np.argwhere((np.isin(graph.temporalEdges[0], np.unique(x[:,4]))) & (np.isin(graph.temporalEdges[1], np.unique(x[:,4]))))

    spatialEdges = np.asarray([graph.edges[0][mask[:,0]], graph.edges[1][mask[:,0]]])
    temporalEdges = np.asarray([graph.temporalEdges[0][maskTemp[:,0]], graph.temporalEdges[1][maskTemp[:,0]]])

    edges = []
    target = []
    src = []
    time_delta = []

    if spatialEdges.shape[1] != 0  or temporalEdges.shape[1] != 0:
        for i, node in enumerate(x):
            # Spatial edges
            if spatialEdges.shape[1] != 0:
                spatialNodes = x[np.argwhere((x[:,4] == node[4]))]
                spatialNodes = spatialNodes.reshape(-1, spatialNodes.shape[-1])
                spatial = spatialEdges[1][(np.isin(spatialEdges[1], spatialNodes[:,0])) & (spatialEdges[0] == node[0])]
                for sp in spatial:
                    src.append(i)
                    target.append(np.argwhere((x[:,4] == node[4]) & (x[:,0] == sp))[0][0])
                    time_delta.append(0)

            # temporal edges
            if temporalEdges.shape[1] != 0:
                temporalNodes = x[np.argwhere((x[:,0] == node[0]))]
                temporalNodes = temporalNodes.reshape(-1, temporalNodes.shape[-1])
                temporal = temporalEdges[1][(np.isin(temporalEdges[1], temporalNodes[:,4])) & (temporalEdges[0] == node[4])]
                for tm in temporal:
                    src.append(i)
                    target.append(np.argwhere((x[:,4] == tm) & (x[:,0] == node[0]))[0][0])
                    time_delta.append(abs(node[4] - tm))

            # Spatio-temporal edges
            if temporalEdges.shape[1] != 0  and spatialEdges.shape[1] != 0:
                nodes = x[(np.argwhere((x[:,4] != node[4]) & (x[:,0]  != node[0])))]
                nodes = nodes.reshape(-1, nodes.shape[-1])
                spatial = spatialEdges[1][(np.isin(spatialEdges[1], nodes[:,0])) & (spatialEdges[0] == node[0])]
                temporal = temporalEdges[1][(np.isin(temporalEdges[1], nodes[:,4])) & (temporalEdges[0] == node[4])]
                for sp in spatial:
                    for tm in temporal:
                        arg = np.argwhere((x[:,0] == sp) & (x[:,4] == tm))
                        if arg.shape[0] == 0:
                            continue
                        src.append(i)
                        target.append(arg[0][0])
                        time_delta.append(abs(node[4] - tm))

        edges = np.row_stack((src, target, time_delta)).astype(int)

    return x, y, edges

def concat_temporal_graph_into_time_series(array : np.array, ks : int, isY=False) -> np.array:
    uniqueNodes = np.unique(array[:,0])
    res = []

    for uNode in uniqueNodes:
        arrayNode = array[array[:,0] == uNode]
        ind = np.lexsort([arrayNode[:,4]])

        #ind = np.flip(ind)
        arrayNode = arrayNode[ind]

        if ks > arrayNode.shape[0]:
            return None
        res.append(arrayNode[:ks+1])

    #print(res)
    res = np.asarray(res)
    res = np.moveaxis(res, 1,2)
    return res

def construct_graph_with_time_series(graph, date : int,
                                     X : np.array, Y : np.array,
                                     ks :int) -> np.array:
    """
    Construct indexing graph with nodes sort by their id and date and corresponding edges.
    We consider spatial edges and time series X
    id : id of the subgraph in X
    X : train or val nodes
    Y : train or val target
    ks : size of the time series
    """

    #maskgraph = np.argwhere((X[:,4] == date) & (X[:, 5] > 0))
    maskgraph = np.argwhere((X[:,4] == date))
    x = X[maskgraph[:,0]]
    if ks != 0:
        maskts = np.argwhere((np.isin(X[:,0], x[:,0]) & (X[:,4] < date) & (X[:,4] >= date - ks)))
        
        if maskts.shape[0] == 0:
            return None, None, None
    
        xts = X[maskts[:,0]]
        x = np.concatenate((x, xts))

    if Y is not None:
        y = Y[maskgraph[:,0]]
        if ks != 0:
            yts = Y[maskts[:,0]]
            yts[:,-4] = 0
            y = np.concatenate((y, yts))

    # Graph indexing
    x = concat_temporal_graph_into_time_series(x, ks)
    if x is None:
        return None, None, None
    if Y is not None:
        y = concat_temporal_graph_into_time_series(y, ks)

    # Get graph specific spatial
    mask = np.argwhere((np.isin(graph.edges[0], np.unique(x[:,0]))) & (np.isin(graph.edges[1], np.unique(x[:,0]))))
    spatialEdges = np.asarray([graph.edges[0][mask[:,0]], graph.edges[1][mask[:,0]]])
    edges = []
    target = []
    src = []

    for i, node in enumerate(x):
        spatialNodes = x[np.argwhere((x[:,4,-1] == node[4][-1]))][:,:, 0, 0]
        if spatialEdges.shape[1] != 0:
            spatial = spatialEdges[1][(np.isin(spatialEdges[1], spatialNodes[:,0])) & (spatialEdges[0] == node[0][0])]
            for sp in spatial:
                src.append(i)
                target.append(np.argwhere((x[:,4,-1] == node[4][-1]) & (x[:,0,0] == sp))[0][0])
        src.append(i)
        target.append(i)

    edges = np.row_stack((src, target)).astype(int)
    return x, y, edges

def order_class(predictor, pred, min_values=0):
    res = np.zeros(pred[~np.isnan(pred)].shape[0], dtype=int)
    cc = predictor.cluster_centers.reshape(-1)
    classes = np.arange(cc.shape[0])
    ind = np.lexsort([cc])
    cc = cc[ind]
    classes = classes[ind]
    for c in range(cc.shape[0]):
        mask = np.argwhere(pred == classes[c])
        res[mask] = c
    return res + min_values

def array2image(X : np.array, dir_mask : Path, scale : int, departement: str, method : str, band : int, dir_output : Path, name : str):

    if method not in METHODS:
        logger.info(f'Methods must be {METHODS}, {method} isn t')
        exit(1)
    name_mask = departement+'rasterScale'+str(scale)+'.pkl'
    mask = read_object(name_mask, dir_mask)
    res = np.full(mask.shape, fill_value=np.nan)

    unodes = np.unique(X[:,0])
    for node in unodes:
        index = mask == node
        if method == 'sum':
            func = np.nansum
        elif method == 'mean':
            func = np.nanmean
        elif method == 'max':
            func = np.nanmax
        elif method == 'min':
            func = np.nanmin
        res[index] = func(X[X[:, 0] == node][:,band])

    save_object(res, name, dir_output)
    return res

def weighted_rmse_loss(input, target, weights = None):
    if not torch.is_tensor(input):
        input = torch.tensor(input, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)

    if weights is None:
        return torch.sqrt(((input - target) ** 2).mean())
    if not torch.is_tensor(weights):
        weights = torch.tensor(weights, dtype=torch.float32)

    return torch.sqrt((weights * (input - target) ** 2).sum() / weights.sum())

def log_sum_exp(x):
    if not torch.is_tensor(input):
        input = torch.tensor(input, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)

    if weights is None:
        return torch.sqrt(((input - target) ** 2).mean())
    if not torch.is_tensor(weights):
        weights = torch.tensor(weights, dtype=torch.float32)
    b, _ = torch.max(x, 1)
    # b.size() = [N, ], unsqueeze() required
    y = b + torch.log(torch.exp(x - b.unsqueeze(dim=1).expand_as(x)).sum(1))
    # y.size() = [N, ], no need to squeeze()
    return y

def class_select(logits, target):
    # in numpy, this would be logits[:, target].
    batch_size, num_classes = logits.size()
    if target.is_cuda:
        device = target.data.get_device()
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .cuda(device)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    else:
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    return logits.masked_select(one_hot_mask)

def weighted_cross_entropy(logits, target, weight=None):
    assert logits.dim() == 2
    assert not target.requires_grad
    target = target.squeeze(1) if target.dim() == 2 else target
    assert target.dim() == 1
    loss = log_sum_exp(logits) - class_select(logits, target)
    loss = loss.view((target.shape[0], 1))
    if weight is not None:
        # loss.size() = [N]. Assert weights has the same shape
        assert list(loss.size()) == list(weight.size())
        # Weight the loss
        loss = loss * weight
        return loss.sum() / weight.sum()
    return loss.mean()

def np_groupby(x, index):
    return np.split(x, np.where(np.diff(x[:,index]))[0]+1)

def add_metrics(methods : list, i : int,
                ypred : torch.tensor,
                ytrue : torch.tensor,
                testd_departement : list,
                target : str,
                scale : int,
                model : Model,
                dir : Path) -> dict:
    
    datesoftest = np.unique(ytrue[:, 4]).astype(int)
    years = np.unique([allDates[di].split('-')[0] for di in datesoftest])
    seasons = generate_season_dict(years)

    top = 5
    uids = np.unique(ytrue[:, 0])
    ysum = np.empty((uids.shape[0], 2))
    ysum[:, 0] = uids
    for id in uids:
        ysum[np.argwhere(ysum[:, 0] == id)[:, 0], 1] = np.sum(ytrue[np.argwhere(ytrue[:, 0] == id)[:, 0], -2])
    
    ind = np.lexsort([ysum[:,1]])
    ymax = np.flip(ysum[ind, 0])[:top]
    mask_top = np.argwhere(np.isin(ytrue[:, 0], ymax))[:, 0]

    res = {}
    for name, met, met_type in methods:

        if met_type == 'proba':
            band = -1
            if target == 'nbsinister' or target == 'binary':
                band = -2
                if target == 'binary':
                    ytrue[:,band] = (ytrue[:,band] > 0).astype(int)
           
            mett = met(ypred[:,0], ytrue[:,band], ytrue[:,-4])
            
            if torch.is_tensor(mett):
                mett = np.mean(mett.detach().cpu().numpy())

            res[name] = mett

            for season, datesIndex in seasons.items():
                mask = np.argwhere(np.isin(ytrue[:, 4], datesIndex))[:,0]
                if mask.shape[0] == 0:
                    continue
                mett = met(ypred[mask, 0], ytrue[mask,band], ytrue[mask,-4])
                if torch.is_tensor(mett):
                    mett = np.mean(mett.detach().cpu().numpy())

                res[name+'_'+season] = mett

            mett = met(ypred[mask_top, 0], ytrue[mask_top,band], ytrue[mask_top,-4])
            if torch.is_tensor(mett):
                mett = mett.detach().cpu().numpy()
            res[name+'_top_'+str(top)+'_cluster'] = mett

        elif met_type == 'cal':
            mett = met(ytrue, ypred[:,1], target, ytrue[:,-4])

            if torch.is_tensor(mett):
                mett = mett.detach().cpu().numpy()
            res[name] = mett

            for season, datesIndex in seasons.items():
                mask = np.argwhere(np.isin(ytrue[:, 4], datesIndex))[:,0]
                if mask.shape[0] == 0:
                    continue
                mett = met(ytrue[:, :], ypred[:,1], target, ytrue[:,-4], mask)
                if torch.is_tensor(mett):
                    mett = mett.detach().cpu().numpy()

                res[name+'_'+season] = mett

            mett = met(ytrue, ypred[:,1], target, None, None)
            if torch.is_tensor(mett):
                mett = mett.detach().cpu().numpy()
            
            res[name+'_unweighted'] = mett

            for season, datesIndex in seasons.items():
                mask = np.argwhere(np.isin(ytrue[:, 4], datesIndex))[:,0]
                if mask.shape[0] == 0:
                    continue
                mett = met(ytrue[:], ypred[:, 1], target, None, mask)

                if torch.is_tensor(mett):
                    mett = mett.detach().cpu().numpy()

                res[name+'_unweighted_'+season] = mett

            mett = met(ytrue[:], ypred[:, 1], target, None, mask_top)
            if torch.is_tensor(mett):
                mett = mett.detach().cpu().numpy()
            res[name+'_top_'+str(top)+'_cluster_unweighted'] = mett

            mett = met(ytrue[:], ypred[:, 1], target, ytrue[:,-4], mask_top)
            if torch.is_tensor(mett):
                mett = mett.detach().cpu().numpy()
            res[name+'_top_'+str(top)+'_cluster'] = mett

        elif met_type == 'bin':

            mett = met(ytrue, ypred[:,0], target, ytrue[:,-4])

            if torch.is_tensor(mett):
                mett = mett.detach().cpu().numpy()
            res[name] = mett

            for season, datesIndex in seasons.items():
                mask = np.argwhere(np.isin(ytrue[:, 4], datesIndex))[:,0]
                if mask.shape[0] == 0:
                    continue
                mett = met(ytrue[:, :], ypred[:,0], target, ytrue[:,-4], mask)
                if torch.is_tensor(mett):
                    mett = mett.detach().cpu().numpy()

                res[name+'_'+season] = mett

            mett = met(ytrue, ypred[:,0], target, None, None)
            if torch.is_tensor(mett):
                mett = mett.detach().cpu().numpy()
            
            res[name+'_unweighted'] = mett

            for season, datesIndex in seasons.items():
                mask = np.argwhere(np.isin(ytrue[:, 4], datesIndex))[:,0]
                if mask.shape[0] == 0:
                    continue
                mett = met(ytrue[:], ypred[:, 0], target, None, mask)

                if torch.is_tensor(mett):
                    mett = mett.detach().cpu().numpy()

                res[name+'_unweighted_'+season] = mett

            mett = met(ytrue[:], ypred[:, 0], target, None, mask_top)
            if torch.is_tensor(mett):
                mett = mett.detach().cpu().numpy()
            res[name+'_top_'+str(top)+'_cluster_unweighted'] = mett

            mett = met(ytrue[:], ypred[:, 0], target, ytrue[:,-4], mask_top)
            if torch.is_tensor(mett):
                mett = mett.detach().cpu().numpy()
            res[name+'_top_'+str(top)+'_cluster'] = mett

        elif met_type == 'class':
            mett = met(ypred, ytrue, testd_departement, dir, weights=ytrue[:,-4], top=None)
            res[name] = mett

            for season, datesIndex in seasons.items():
                mask = np.argwhere(np.isin(ytrue[:, 4], datesIndex))[:,0]
                if mask.shape[0] == 0:
                    continue
                mett = met(ypred[mask], ytrue[mask], testd_departement, dir, weights=ytrue[mask,-4], top=None)
                res[name+'_'+season] = mett

            mett = met(ypred, ytrue, testd_departement, dir, weights=None, top=10)
            res[name+'top10'] = mett

            mett = met(ypred[mask_top], ytrue[mask_top], testd_departement, dir, weights=ytrue[mask_top,-4], top=None)
            if torch.is_tensor(mett):
                mett = mett.detach().cpu().numpy()
            res[name+'_top_'+str(top)+'_cluster'] = mett

            mett = met(ypred[mask_top], ytrue[mask_top], testd_departement, dir, weights=None, top=None)
            if torch.is_tensor(mett):
                mett = mett.detach().cpu().numpy()
            res[name+'_top_'+str(top)+'_cluster_unweighted'] = mett

        logger.info(f'{name, res[name]}')

    return res

def create_predictor(ypred : np.array, modelName : str, nameDep : str, dir_predictor, scale : int, isBin : bool, graph_construct):
    predictor = read_object(nameDep+'Predictor'+modelName+str(scale)+'.pkl', dir_predictor)
    logger.info(f'Create {nameDep}Predictor{modelName}{str(scale)}.pkl')
    predictor = Predictor(min(5, np.unique(ypred[:,0]).shape[0]), name=nameDep+'Binary', binary=isBin)
    predictor.fit(np.unique(ypred[:,0]))

    if scale == 'Departement':
        save_object(predictor, f'{nameDep}Predictor{modelName}{scale}.pkl', dir_predictor)
    else:
        save_object(predictor, f'{nameDep}Predictor{modelName}{scale}_{graph_construct}.pkl', dir_predictor)

def my_mean_absolute_error(input, target, weights = None):
    return mean_absolute_error(y_true=target, y_pred=input, sample_weight=weights)

def standard_deviation(inputs, target, weights = None):
    return np.std(inputs)

def log_factorial(x):
  return torch.lgamma(x + 1)

def factorial(x):
    if not torch.is_tensor(x):
        input = torch.tensor(x, dtype=torch.float32)
    return torch.exp(log_factorial(x))

def poisson_loss(input, target, weights = None):
    if not torch.is_tensor(input):
        input = torch.tensor(input, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)

    if weights is None:
        return torch.sqrt(((input - target) ** 2).mean())
    if not torch.is_tensor(weights):
        weights = torch.tensor(weights, dtype=torch.float32)
    if weights is None:
        return torch.exp(input) - target*torch.log(input) + torch.log(factorial(target))
    else:
        return ((torch.exp(input) - target*torch.log(input) + torch.log(factorial(target))) * weights).sum() / weights.sum()
    
def quantile_prediction_error(Y : np.array, ypred : np.array, target : str, weights = None, mask : np.array = None):

    if mask is None:
        mask = np.arange(ypred.shape[0])

    ytrue = Y[mask,-2]
    ypred = ypred[mask]

    quantiles = [(0, 0.0),
                 (1, 0.2),
                 (2, 0.4),
                 (3, 0.6),
                 (4, 0.8),
                 (5, 1.0)
                 ]
    
    error = []
    for (cl, expected) in quantiles:
        nf = (ytrue[ypred == cl] > 0).astype(int)
        number_of_fire = np.mean(nf)
        error.append(abs((expected - number_of_fire)))
    if len(error) == 0:
        return math.inf
    
    return np.nanmean(error)

def my_f1_score(Y : np.array, ypred : np.array, target : str, weights : np.array = None, mask : np.array = None):

    if mask is None:
        mask = np.arange(ypred.shape[0])
    
    ytrue = Y[:,-2] > 0
    ytrueReg = Y[:,-1]

    bounds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    bestScores = []
    precs = []
    recs = []
    bestBounds = []
    values = []

    ydep = np.unique(Y[:, 3])

    res = {}
    for d in ydep:
        res[int2name[d]] = {}
        mask_dep = np.argwhere(Y[:, 3] == d)[:, 0]

        if target == 'binary':
            maxi = 1.0
        elif target == 'nbsinister':
            maxi = 1.0
        else:
            maxi = np.nanmax(ytrueReg[mask_dep])

        mask_dep = np.intersect1d(mask_dep, mask)
        if mask_dep.shape[0] == 0:
            continue
        ypredNumpy = ypred[mask_dep]
        ytrueNumpy = ytrue[mask_dep]
        ytrueRegNumpy = ytrueReg[mask_dep]

        if weights is not None:
            weightsNumpy = weights[mask_dep]
            weightsNumpy[np.argwhere(ytrue[mask_dep] == 0)[:,0]] = 1
        else:
            weightsNumpy = np.ones(ytrueNumpy.shape[0])

        bestScore = 0.0
        prec = 0.0
        rec = 0.0
        bestBound = 0.0

        for bound in bounds:
            if target == 'binary' or target == 'nbsinister':
                yBinPred = (ypredNumpy > bound * maxi).astype(int)
            else:
                yBinPred = (ytrueRegNumpy > bound * maxi).astype(int)

            f1 = f1_score(ytrueNumpy, yBinPred, sample_weight=weightsNumpy)
            if f1 > bestScore:
                bestScore = f1
                bestBound =  bound
                prec = precision_score(ytrueNumpy, yBinPred, sample_weight=weightsNumpy)
                rec = recall_score(ytrueNumpy, yBinPred, sample_weight=weightsNumpy)
        if target != 'binary' and target != 'nbsinister':
            yBinPred = (ypredNumpy > bestBound * maxi).astype(int)
            f1 = f1_score(ytrueNumpy, yBinPred, sample_weight=weightsNumpy)
            bestScore = f1
            prec = precision_score(ytrueNumpy, yBinPred, sample_weight=weightsNumpy)
            rec = recall_score(ytrueNumpy, yBinPred, sample_weight=weightsNumpy)

        res[int2name[d]]['f1'] = bestScore
        res[int2name[d]]['precision'] = prec
        res[int2name[d]]['recall'] = rec
        res[int2name[d]]['threshold'] = bestScore
        res[int2name[d]]['value'] = bestBound * maxi

    return res

def class_risk(ypred, ytrue, departements : list, dir : Path, weights = None, top=None) -> dict:
    if torch.is_tensor(ypred):
        ypred = ypred.detach().cpu().numpy().astype(float)
    if torch.is_tensor(ytrue):
        ytrue = ytrue.detach().cpu().numpy().astype(float)
    
    ydep = ytrue[:,3]

    res = {}
    for nameDep in departements:
        mask = np.argwhere(ydep == name2int[nameDep])
        if mask.shape[0] == 0:
            continue
        
        uniqueClass = [0,1,2,3,4]
        res[nameDep] = {}

        nbsinisteriny = []
        nbsinisterinpred = []

        for cls in uniqueClass:
            mask2 = np.argwhere(ytrue[mask, -3] == cls)[:, 0]
            if mask.shape[0] == 0:
                nbsinisteriny.append(0)
            nbsinisteriny.append(frequency_ratio(ytrue[:, -2], mask2))
            
            mask2 = np.argwhere(ypred == cls)[:, 0]
            nbsinisterinpred.append(frequency_ratio(ytrue[:, -2], mask2))

            res[nameDep][cls] = nbsinisteriny[-1] - nbsinisterinpred[-1]

        res[nameDep]['mean'] = np.mean(np.asarray(nbsinisteriny) - np.asarray(nbsinisterinpred))

    return res

def class_accuracy(ypred, ytrue, departements : list, dir : Path, weights = None, top=None) -> dict:
    if torch.is_tensor(ypred):
        ypred = ypred.detach().cpu().numpy().astype(float)
    if torch.is_tensor(ytrue):
        ytrue = ytrue.detach().cpu().numpy().astype(float)

    if weights is not None:
        if torch.is_tensor(weights):
            weightsNumpy = weights.detach().cpu().numpy()
        else:
            weightsNumpy = weights
    else:
        weightsNumpy = np.ones(ytrue.shape[0])

    res = {}
    for nameDep in departements:
        mask = np.argwhere(ytrue[:, 3] == name2int[nameDep])
        if mask.shape[0] == 0:
            continue

        yweights = weightsNumpy[mask].reshape(-1)
        res[nameDep] = accuracy_score(ytrue[mask, -3], ypred[mask, -1], sample_weight=yweights)
            
    return res

def balanced_class_accuracy(ypred, ytrue, departements : list, dir : Path, weights = None, top=None) -> dict:
    if torch.is_tensor(ypred):
        ypred = ypred.detach().cpu().numpy().astype(float)
    if torch.is_tensor(ytrue):
        ytrue = ytrue.detach().cpu().numpy().astype(float)

    if weights is not None:
        if torch.is_tensor(weights):
            weightsNumpy = weights.detach().cpu().numpy()
        else:
            weightsNumpy = weights
    else:
        weightsNumpy = np.ones(ytrue.shape[0])


    res = {}
    for nameDep in departements:
        mask = np.argwhere(ytrue[:, 3] == name2int[nameDep])
        if mask.shape[0] == 0:
            continue

        yweights = weightsNumpy[mask].reshape(-1)

        res[nameDep] = balanced_accuracy_score(ytrue[mask, -3], ypred[mask, -1], sample_weight=yweights)
            
    return res

def mean_absolute_error_class(ypred, ytrue, departements : list,
                              dir : Path, weights = None, top : int = None) -> dict:
    
    if torch.is_tensor(ypred):
        ypred = ypred.detach().cpu().numpy().astype(float)
    if torch.is_tensor(ytrue):
        ytrue = ytrue.detach().cpu().numpy().astype(float)

    if weights is not None:
        if torch.is_tensor(weights):
            weightsNumpy = weights.detach().cpu().numpy()
        else:
            weightsNumpy = weights
    else:
        weightsNumpy = np.ones(ytrue.shape[0])

    ydep = ytrue[:,3]
    res = {}
    for nameDep in departements:
        mask = np.argwhere(ydep == name2int[nameDep])
        if mask.shape[0] == 0:
            continue

        yweights = weightsNumpy[mask].reshape(-1)

        if top is not None:
            minBound = np.nanmax(ytrue[mask,-1]) * (1 - top/100)
            mask2 = (ytrue[mask,-1] > minBound)
            ytrueclass = ytrue[mask, -3][mask2]
            ypredclass = ypred[mask, -1][mask2]
            yweights = None
        else:
            ytrueclass = ytrue[mask, -3]
            ypredclass = ypred[mask, -1]

        if ytrueclass.shape[0] == 0:
            continue
        res[nameDep] = mean_absolute_error(ytrueclass, ypredclass, sample_weight=yweights)

    return res

def get_features_name_list(scale, features, methods):

    features_name = []
    if scale == 0:
        methods = ['mean']
    for var in features:
        if var == 'Calendar':
            features_name += calendar_variables
        elif var == 'air':
            features_name += air_variables
        elif var == 'landcover':
            features_name += [f'{v}_{met}' for v in landcover_variables for met in methods]
        elif var == 'sentinel':
            features_name += [f'{v}_{met}' for v in sentinel_variables for met in methods]
        elif var == "foret":
            features_name += [f'{foretint2str[v]}_{met}' for v in foret_variables for met in methods]
        elif var == 'dynamicWorld':
            features_name += [f'{v}_{met}' for v in dynamic_world_variables for met in methods]
        elif var == 'highway':
            features_name += [f'{osmnxint2str[v]}_{met}' for v in osmnx_variables for met in methods]
        elif var == 'Geo':
            features_name += geo_variables
        elif var == 'vigicrues':
            features_name += [f'{v}_{met}' for v in vigicrues_variables for met in methods]
        elif var == 'nappes':
            features_name += [f'{v}_{met}' for v in nappes_variables for met in methods]
        elif var == 'Historical':
            features_name += [f'{v}_{met}' for v in historical_variables for met in methods]
        elif var == 'AutoRegressionReg':
            features_name += [f'AutoRegressionReg_{v}' for v in auto_regression_variable_reg]
        elif var == 'AutoRegressionBin':
            features_name +=  [f'AutoRegressionBin_{v}' for v in auto_regression_variable_bin]
        elif var == 'elevation':
            features_name += [f'{v}_{met}' for v in elevation_variables for met in methods]
        elif var == 'population':
            features_name += [f'{v}_{met}' for v in population_variabes for met in methods]
        elif True in [tv in var for tv in varying_time_variables]:
            k = var.split('_')[-1]
            met = var.split('_')[1]
            if 'Calendar' in var:
                features_name += [f'{v}_{met}_{k}' for v in calendar_variables]
            elif 'air' in var:
                features_name += [f'{v}_{met}_{k}' for v in air_variables]
            else:
                features_name += [f'{var}']
        elif var == 'temporal_prediction' or var == 'spatial_prediction':
            features_name += [var]
        else:
            features_name += [f'{var}_{met}' for met in methods]

    return features_name, len(features_name)

def get_features_name_lists_2D(shape, features):

    features_name = []
    for var in features:
        if var == 'Calendar':
            features_name.extend(calendar_variables)
        elif var == 'air':
            features_name.extend(air_variables)
        elif var == 'landcover':
            features_name.extend(landcover_variables)
        elif var == 'sentinel':
            features_name.extend(sentinel_variables)
        elif var == "foret":
            features_name.extend([foretint2str[fv] for fv in foret_variables])
        elif var == 'dynamicWorld':
            features_name.extend(dynamic_world_variables)
        elif var == 'highway':
            features_name.extend([osmnxint2str[fv] for fv in osmnx_variables])
        elif var == 'Geo':
            features_name.extend(geo_variables)
        elif var == 'vigicrues':
            features_name.extend(vigicrues_variables)
        elif var == 'nappes':
            features_name.extend(nappes_variables)
        elif var == 'Historical':
            features_name.extend(historical_variables)
        elif var == 'elevation':
            features_name.extend(elevation_variables)
        elif var == 'population':
            features_name.extend(population_variabes)
        elif var == 'AutoRegressionReg':
            features_name.extend([f'AutoRegressionReg_{v}' for v in auto_regression_variable_reg])
        elif var == 'AutoRegressionBin':
            features_name.extend([f'AutoRegressionBin_{v}' for v in auto_regression_variable_bin])
        elif True in [tv in var for tv in varying_time_variables]:
            k = var.split('_')[-1]
            if 'Calendar' in var:
                features_name.extend([f'{v}_{k}' for v in calendar_variables])
            elif 'air' in var:
                features_name.extend([f'{v}_{k}' for v in air_variables])
            else:
                features_name.extend([var])
        elif var == 'temporal_prediction' or var == 'spatial_prediction':
            features_name.extend([var])
        else:
            features_name.extend([var])

    return features_name, len(features_name)

def min_max_scaler(array : np.array, array_train: np.array, concat : bool) -> np.array:
    if concat:
        Xt = np.concatenate((array, array_train))
    else:
        Xt = array_train
    scaler = MinMaxScaler()
    scaler.fit(Xt.reshape(-1,1))
    res = scaler.transform(array.reshape(-1,1))
    return res.reshape(array.shape)

def standard_scaler(array: np.array, array_train: np.array, concat : bool) -> np.array:
    if concat:
        Xt = np.concatenate((array, array_train))
    else:
        Xt = array_train
    scaler = StandardScaler()
    scaler.fit(Xt.reshape(-1,1))
    res = scaler.transform(array.reshape(-1,1))
    return res.reshape(array.shape)

def robust_scaler(array : np.array, array_train: np.array, concat : bool) -> np.array:
    if concat:
        Xt = np.concatenate((array, array_train))
    else:
        Xt = array_train
    scaler = RobustScaler()
    scaler.fit(Xt.reshape(-1,1))
    res = scaler.transform(array.reshape(-1,1))
    return res.reshape(array.shape)

def interpolate_gridd(var, grid, newx, newy):
    x = grid['longitude'].values
    y = grid["latitude"].values
    points = np.zeros((y.shape[0], 2))
    points[:,0] = x
    points[:,1] = y
    return griddata(points, grid[var].values, (newx, newy), method='linear')

def resize(input_image, height, width, dim):
    """
    Resize the input_image into heigh, with, dim
    """
    img = img_as_float(input_image)
    img = transform.resize(img, (dim, height, width))
    return np.asarray(img)

def resize_no_dim(input_image, height, width):
    """
    Resize the input_image into heigh, with, dim
    """
    img = img_as_float(input_image)
    img = transform.resize(img, (height, width), mode='constant', order=0,
                 preserve_range=True, anti_aliasing=True)
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

def pendant_couvrefeux(date):
    # Fonction testant si une date tombe dans une période de confinement
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

def target_encoding(features_name, Xset, x_train, y_train, variableType, size):
    logger.info(f'Target Encoding')
    enc = TargetEncoder(cols=np.arange(features_name.index(variableType), features_name.index(variableType) + size)).fit(x_train, y_train[:, -1])
    Xset = enc.transform(Xset).values
    return Xset

def catboost_encoding(features_name, Xset, x_train, y_train, variableType, size):
    logger.info(f'Catboost Encoding')
    enc = CatBoostEncoder(cols=np.arange(features_name.index(variableType), features_name.index(variableType) + size)).fit(x_train, y_train[:, -1])
    Xset = enc.transform(Xset).values
    return Xset

def log_features(features, features_name):
    if len(features.shape) > 1:
        for fet_index, nb in features:
            logger.info(f'{fet_index, features_name[fet_index]} {nb}')
    else:
        for fet_index in features:
                logger.info(f'{fet_index, features_name[fet_index]}')

def create_feature_map(features : np.array, features_name : dict, dir_output : Path):
    with open(dir_output / 'feature_map.text', 'w') as file:
        for fet_index in features:
            file.write(f'{features_name[fet_index]} q\n')
        file.close()
    return

def calculate_feature_range(fet, scale, methods):
    coef = len(methods) if scale > 0 else 1
    if fet == 'Calendar':
        variables = calendar_variables
        maxi = len(calendar_variables)
        methods = ['raw']
        start = calendar_variables[0]
    elif fet == 'air':
        variables = air_variables
        maxi = len(air_variables)
        methods = ['raw']
        start = air_variables[0]
    elif fet == 'sentinel':
        variables = sentinel_variables
        maxi = coef * len(sentinel_variables)
        methods = methods
        start = f'{sentinel_variables[0]}_mean'
    elif fet == 'Geo':
        variables = geo_variables
        maxi = len(geo_variables)
        methods = ['raw']
        start = f'{geo_variables[0]}'
    elif fet == 'foret':
        variables = foret_variables
        maxi = coef * len(foret_variables)
        methods = methods
        start = f'{foretint2str[foret_variables[0]]}_mean'
    elif fet == 'highway':
        variables = osmnx_variables
        maxi = coef * len(osmnx_variables)
        methods = methods
        start = f'{osmnxint2str[osmnx_variables[0]]}_mean'
    elif fet == 'landcover':
        variables = landcover_variables
        maxi = coef * len(landcover_variables)
        methods = methods
        start = f'{landcover_variables[0]}_mean'
    elif fet == 'dynamicWorld':
        variables = dynamic_world_variables
        maxi = coef * len(dynamic_world_variables)
        methods = methods
        start = f'{dynamic_world_variables[0]}_mean'
    elif fet == 'vigicrues':
        variables = vigicrues_variables
        maxi = coef * len(vigicrues_variables)
        methods = methods
        start = f'{vigicrues_variables[0]}_mean'
    elif fet == 'elevation':
        variables = elevation_variables
        maxi = coef * len(elevation_variables)
        methods = methods
        start = f'{elevation_variables[0]}_mean'
    elif fet == 'population':
        variables = population_variabes
        maxi = coef * len(population_variabes)
        methods = methods
        start = f'{population_variabes[0]}_mean'
    elif fet == 'nappes':
        variables = nappes_variables
        maxi = coef * len(nappes_variables)
        methods = methods
        start = f'{nappes_variables[0]}_mean'
    elif fet == 'Historical':
        variables = historical_variables
        maxi = coef * len(historical_variables)
        methods = methods
        start = f'{historical_variables[0]}_mean'
    elif fet == 'AutoRegressionReg':
        variables = auto_regression_variable_reg
        maxi = len(auto_regression_variable_reg)
        methods = ['raw']
        start = f'AutoRegressionReg_{auto_regression_variable_reg[0]}'
    elif fet == 'AutoRegressionBin':
        variables = auto_regression_variable_bin
        maxi = len(auto_regression_variable_bin)
        methods = ['raw']
        start = f'AutoRegressionBin_{auto_regression_variable_bin[0]}'
    elif fet.find('pca') != -1:
        variables = [fet]
        maxi = 1
        methods = ['raw']
        start = f'{fet}'
    elif fet == 'temporal_prediction' or fet == 'spatial_prediction':
        variables = [fet]
        maxi = 1
        methods = ['raw']
        start = f'{fet}'
    elif fet in cems_variables:
        variables = [fet]
        maxi = coef
        methods = methods
        start = f'{fet}_mean'
    else:
        variables = [fet]
        maxi = 1
        methods = ['raw']
        start = f'{fet}'

    return start, maxi, methods, variables

def select_train_features(train_features, scale, features_name):
    # Select train features
    train_fet_num = []
    features_name_train, _ = get_features_name_list(scale, train_features)
    for fet in features_name_train:
        train_fet_num.append(features_name.index(fet))
    return train_fet_num

def features_selection(doFet, df, dir_output, features_name, NbFeatures, target):

    if NbFeatures == 'all':
        features_selected = features_name
        return features_selected

    if doFet:
        df_weight = df[df['weight'] > 0]
        features_importance = get_features(df_weight[features_name + [target]], features_name, target=target, num_feats=NbFeatures)
        features_importance = np.asarray(features_importance)

        save_object(features_importance, 'features_importance.pkl', dir_output)
    else:
        features_importance = read_object('features_importance.pkl', dir_output)

    logger.info(features_importance)
    features_selected = features_importance[:,0]
    return features_selected

def shapiro_wilk(ypred, ytrue, dir_output, outputname):
    diff = ytrue - ypred
    swtest = scipy.stats.shapiro(diff)
    try:
        logger.info(f'Test statistic : {swtest.statistic}, pvalue {swtest.pvalue}')
        plt.hist(diff)
        plt.title(f'Test statistic : {swtest.statistic}, pvalue {swtest.pvalue}')
        plt.savefig(dir_output / f'{outputname}.png')
        plt.close('all')
    except Exception as e:
        logger.info(e)

def select_n_points(X, Y, dir, nbpoints):
    cls = [0,1,2,3,4]
    oldWeights = Y[:, 5]
    X[:, 5] = 0
    Y[:, 5] = 0
    udept = np.uniuqe(X[:,3])
    for dept in udept:
        mask = np.argwhere(X[:, 3] == dept)
        dir_predictor = root_graph / dir / 'influenceClustering'
        predictor = read_object(int2name[dept]+'Predictor.pkl', dir_predictor)
        classs = predictor.predict(X[mask, -1])
        for cs in cls:
            maskc = np.argwhere(classs == cs)
            choices = np.random.choice(maskc, nbpoints)
            X[choices, 5] = oldWeights[choices]
            Y[choices, 5] = oldWeights[choices]

    return X, Y

def check_class(influence, bin):
    values = influence[~np.isnan(influence)]
    binValues_ = bin[~np.isnan(influence)]
    predictor = Predictor(5, name='test')
    predictor.fit(np.unique(values))
    classs = order_class(predictor, predictor.predict(values))
    predictor.log()
    cls = np.unique(classs)
    for cl in cls:
        mask = classs == cl
        logger.info(f'class {cl}, {np.nanmean(binValues_[mask]), np.nanmean(values[mask])}')

def change_dict_key(d, old_key, new_key, default_value=None):
    d[new_key] = d.pop(old_key, default_value)

def plot_kmeans_class_for_inference(df, date_limit, features_selected, dir_break_point, dir_log, sinister):
    y_dates = np.unique([date_limit - dt.timedelta(days=df.date.values.max() - k) for k in df.date.values])
    unodes = df.id.unique() 

    check_and_create_path(dir_log / 'kmeans_feature')
    for fet in features_selected:
        logger.info(f'############## {fet} #################')

        fig, axs = plt.subplots(unodes.shape[0], figsize=(15,10))
        
        for i, node in enumerate(unodes):
            values = df[df['id'] == node][fet].values
            predictor = read_object(f'{fet}.pkl', dir_break_point / fet)
            predclass = order_class(predictor, predictor.predict(values))
            ax1 = axs[i]
            ax1.set_title(f'{node}')
            ax1.plot(y_dates, predclass, c='b', label=f'{fet}')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Class')
            ax1.set_ylim([0, 4])

            ax2 = ax1.twinx()

            ax2.plot(y_dates, df[df['id'] == node]['AutoRegressionBin_B-1'].shift(1), label=f'{sinister}', c='r')
            ax2.set_ylabel(f'Number of {sinister}', color='r')
            ax2.set_ylim([0, 4])
            
        plt.legend()
        plt.tight_layout()
        plt.savefig(dir_log / 'kmeans_feature' / f'{fet}_{date_limit}.png')
        plt.close('all')

def apply_kmeans_class_on_target(df: pd.DataFrame, dir_break_point: Path, target : str, tresh : int, features_selected, new_val : int):
    dico_correlation = read_object('break_point_dict.pkl', dir_break_point)
    values_risk = df[target].values
    for fet in features_selected:
        logger.info(f'############## {fet} #################')
        values = df[fet].values
        predictor = read_object(f'{fet}.pkl', dir_break_point / fet)
        if predictor is None:
            continue
        predclass = order_class(predictor, predictor.predict(values))
        cls = np.unique(predclass)
        low = False
        previous_class = -1
        for c in cls:
            mask = predclass == c
            if int(c) not in dico_correlation[fet].keys():
                continue
            if dico_correlation[fet][int(c)] < tresh:
                low = True
                previous_class = c
                logger.info(f'{target} : {c, dico_correlation[fet][int(c)]}, {np.unique(values_risk[mask])} -> 0')
                values_risk[mask] = new_val
            elif low:
                logger.info(f'{target} : {c, dico_correlation[fet][int(c)]}, {np.unique(values_risk[mask])} due to {previous_class} -> 0')
                values_risk[mask] = new_val

    df[target] = values_risk
    return df

def group_probability(group):
    # Calculer la probabilité que au moins un des événements se produise
    combined_prob = 1 - (1 - group['prediction']).prod()
    return pd.Series({'prediction': combined_prob})

def est_bissextile(annee):
    return annee % 4 == 0 and (annee % 100 != 0 or annee % 400 == 0)

def ajuster_jour_annee(date, dayoyyear):
    if not est_bissextile(date.year) and date > pd.Timestamp(date.year, 2, 28):
        # Ajuster le jour de l'année pour les années non bissextiles après le 28 février
        return dayoyyear + 1
    else:
        return dayoyyear
    
def generate_season_dict(years):
    res = {}

    dates = {
    'summer' : ('06-01', '08-31'),
    'winter' : ('12-01', '02-28'),
    'autumn' : ('09-01', '11-30'),
    'spring' : ('03-01', '05-31')
    }
    
    for season in ['winter', 'spring', 'summer', 'autumn']:
        res[season] = []
        for year in years:
            if season == 'winter':
                y2 = str(int(year) + 1)
            else:
                y2 = year
            datesBetween = find_dates_between(year+'-'+dates[season][0], y2+'-'+dates[season][1])
            res[season] += [allDates.index(d) for d in datesBetween if d in allDates]

    return res

def show_pcs(pca, size, components, dir_output):
    labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(size)
    )
    fig.update_traces(diagonal_visible=False)
    #fig.show()
    pio.write_image(fig, dir_output / "pca.png")

def find_n_component(thresh, pca):
    nb_component = 0
    sumi = 0.0
    for i in range(pca.explained_variance_ratio_.shape[0]):
        sumi += pca.explained_variance_ratio_[i]
        if sumi >= thresh:
            nb_component = i + 1
            break
    return nb_component

def target_by_day(df: pd.DataFrame, days: int, method: str) -> pd.DataFrame:
    """
    Adjust target values based on specified method (mean or max) over a number of future days.
    
    Parameters:
    df : DataFrame containing the data
    days : number of days to look ahead for calculating the target
    method : method to use for aggregation ('mean' or 'max')
    
    Returns:
    DataFrame with adjusted target values.
    """
    df_res = df.copy()

    unique_nodes = df['id'].unique()
    for node in unique_nodes:
        node_df = df[df['id'] == node]
        unique_dates = node_df['date'].unique()
        for date in unique_dates:
            mask = (df['date'] == date) & (df['id'] == node)
            future_mask = (df['date'] >= date) & (df['date'] < date + days) & (df['id'] == node)
            
            if method == 'mean':
                df_res.loc[mask, 'risk'] = node_df[future_mask]['risk'].mean()
            elif method == 'max':
                df_res.loc[mask, 'risk'] = node_df[future_mask]['risk'].max()
            else:
                raise ValueError(f'Unknown method {method}')
                
            df_res.loc[mask, 'class_risk'] = node_df[future_mask]['class_risk'].max()
            df_res.loc[mask, 'nbsinister'] = node_df[future_mask]['nbsinister'].sum()
            
    return df_res


def add_temporal_spatial_prediction(X : np.array, Y_localized : np.array, Y_daily : np.array, graph_temporal, features_name : dict, target_name : str):
    logger.info(features_name)
    res = np.full((X.shape[0], X.shape[1] + 2), fill_value=0.0)
    res[:, :X.shape[1]] = X

    if target_name == 'nbsinister' or target_name == 'binary':
        band = -2
    else:
        band = -1

    # Add spatial
    res[:, features_name.index('spatial_prediction')] = Y_localized[:, band]
    if target_name == 'binary':
        res[:, features_name.index('spatial_prediction')] = (res[:, features_name.index('spatial_prediction')] > 0).astype(int)

    # Add temporal
    unodes = np.unique(res[:, 0])
    udates = np.unique(res[:, 4])

    for node in unodes:
        lon = np.unique(res[res[:, 0] == node][:, 1])[0]
        lat = np.unique(res[res[:, 0] == node][:, 2])[0]
        temporal_node_pred = graph_temporal._predict_node([[lon, lat]])
        for date in udates:
            mask = np.argwhere((res[:, 0] == node) & (res[:, 4] == date))[:, 0]
            mask_temporal = np.argwhere((Y_daily[:, 0] == temporal_node_pred) & (Y_daily[:, 4] == date))[:, 0]
            if mask.shape[0] == 0 or mask_temporal.shape[0] == 0:
                continue
            
            res[mask, features_name.index('temporal_prediction')] = Y_daily[mask_temporal, band][0]
            if target_name == 'binary':
                res[:, features_name.index('temporal_prediction')] = (res[:, features_name.index('temporal_prediction')] > 0).astype(int)
    return res

def log_metrics_recursively(metrics_dict, prefix=""):
    for key, value in metrics_dict.items():
        # Créer la clé composée pour les métriques
        full_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            # Appel récursif pour les sous-dictionnaires
            log_metrics_recursively(value, full_key)
        elif isinstance(value, (int, float)):
            # Enregistrement de la métrique
            mlflow.log_metric(full_key, value)
        else:
            mlflow.log_metric(full_key, value)
            # Gérer d'autres types si nécessaire (par exemple, des listes)
            #raise ValueError(f"Unsupported metric type: {type(value)} for key: {full_key}")

def get_existing_run(run_name):
    # Récupère tous les runs avec le nom spécifié
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=['0'],  # Spécifiez ici l'ID de l'expérience si nécessaire
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY  # On ne cherche que les runs actifs
    )
    
    # Si un run est trouvé, le retourner
    if runs:
        return runs[0]  # retourne le premier run trouvé
    return None

def frequency_ratio(values: np.array, mask : np.array):
    FF_t = np.sum(values)
    Area_t = len(values)
    FF_i = np.sum(values[mask])
    
    # Calculer Area_i (le nombre total de pixels pour la classe c)
    Area_i = mask.shape[0]
    
    # Calculer FireOcc et Area pour la classe c
    FireOcc = FF_i / FF_t
    Area = Area_i / Area_t
    
    # Calculer le ratio de fréquence (FR) pour la classe c
    FR = FireOcc / Area

    return round(FR, 3)