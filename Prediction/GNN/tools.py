from cv2 import dilate
from torch import cosine_embedding_loss, fill
from GNN.dico_departements import *
from GNN.weigh_predictor import *
from GNN.features_selection import *
from GNN.config import *
from GNN.array_fet import *

if is_pc:
    import datetime as dt
    import geopandas as gpd
    import math
    from matplotlib.dates import datestr2num
    import matplotlib.pyplot as plt
    import numpy as np
    import osmnx as ox
    import pickle
    import plotly.express as px
    import plotly.io as pio
    import random
    from hdbscan import HDBSCAN, approximate_predict
    import rasterio
    import rasterio.features
    import rasterio.warp
    from regex import D
    import scipy.interpolate
    import scipy.stats
    import sys
    import torch
    import warnings
    from arborescence import *
    from array_fet import *
    from category_encoders import TargetEncoder, CatBoostEncoder
    from collections import Counter
    from copy import copy
    from geocube.api.core import make_geocube
    from geocube.rasterize import rasterize_points_griddata
    from lightgbm import LGBMClassifier, LGBMRegressor
    from lifelines.utils import concordance_index
    from ngboost import NGBClassifier, NGBRegressor
    from osgeo import gdal, ogr
    from pathlib import Path
    from scipy import ndimage as ndi
    from scipy.interpolate import griddata
    from scipy.signal import fftconvolve as scipy_fft_conv
    from scipy.stats import kendalltau, pearsonr, spearmanr
    from skimage import img_as_float
    from skimage import measure, segmentation, morphology
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    from skimage import transform
    from sklearn.cluster import SpectralClustering
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, balanced_accuracy_score, \
        mean_absolute_error
    from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
    from sklearn.preprocessing import normalize
    from sklearn.svm import SVR
    from tqdm import tqdm
    from torch import optim
    from xgboost import XGBClassifier, XGBRegressor
    import convertdate
else:
    import datetime as dt
    import geopandas as gpd
    import math
    from matplotlib.dates import datestr2num
    import matplotlib.pyplot as plt
    import numpy as np
    #import osmnx as ox
    import pickle
    #import plotly.express as px
    #import plotly.io as pio
    import random
    #import rasterio
    #import rasterio.features
    #import rasterio.warp
    #from regex import D
    import scipy.interpolate
    import scipy.stats
    import sys
    import torch
    import warnings
    from arborescence import *
    from array_fet import *
    from category_encoders import TargetEncoder, CatBoostEncoder
    from collections import Counter
    from copy import copy
    #from geocube.api.core import make_geocube
    #from geocube.rasterize import rasterize_points_griddata
    from GNN.config import *
    from lightgbm import LGBMClassifier, LGBMRegressor
    from lifelines.utils import concordance_index
    from ngboost import NGBClassifier, NGBRegressor
    #from osgeo import gdal, ogr
    from pathlib import Path
    from scipy import ndimage as ndi
    from scipy.interpolate import griddata
    from scipy.signal import fftconvolve as scipy_fft_conv
    from scipy.stats import kendalltau, pearsonr, spearmanr
    from skimage import img_as_float
    from skimage import measure, segmentation, morphology
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    from skimage import transform
    from sklearn.cluster import SpectralClustering
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, balanced_accuracy_score, \
        mean_absolute_error
    from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
    from sklearn.preprocessing import normalize
    from sklearn.svm import SVR
    from tqdm import tqdm
    from torch import optim
    from weigh_predictor import *
    from xgboost import XGBClassifier, XGBRegressor
    from dico_departements import *
    from features_selection import *
    import convertdate

random.seed(42)

"""# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)"""

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
    logger.info(f'Nan columns : {np.unique(df.columns[df.isna().any()].tolist())}')
    return df.dropna().reset_index(drop=True)

def remove_none_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where the target column has -1 or NaN values.
    """
    return df[(df['risk'] != -1) & (~df['risk'].isna())].reset_index(drop=True)

def remove_bad_period(df: pd.DataFrame, period2ignore: dict, departements: list, ks : int) -> pd.DataFrame:
    global allDates, name2int

    bad_dates = np.array([], dtype=int)
    zeros_dates = np.array([], dtype=int)
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

                ds_idx = allDates.index(ds) - ks
                de_idx = allDates.index(de) - ks
                bad_dates = np.concatenate((bad_dates, np.arange(start=ds_idx, stop=de_idx)))
                zeros_dates = np.concatenate((zeros_dates , np.arange(start=de_idx, stop=de_idx + ks)))
                zeros_dates = np.concatenate((zeros_dates , np.arange(start=ds_idx, stop=ds_idx + ks)))

    for dept in departements:
        df = df[~((df['departement'] == name2int[dept]) & (df['date'].isin(bad_dates)))]
        df.loc[df[(df['departement'] == name2int[dept]) & (df['date'].isin(zeros_dates))].index, weights_columns] = 0

    return df.reset_index(drop=True)

def remove_non_fire_season(df: pd.DataFrame, SAISON_FEUX: dict, departements: list, ks : int) -> pd.DataFrame:
    global allDates, name2int, years

    valid_indices = np.array([], dtype=int)
    zeros_dates = np.array([], dtype=int)
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

            start_idx = allDates.index(date_debut) - ks
            end_idx = allDates.index(date_fin) - ks

            valid_indices = np.concatenate((valid_indices, df[(df['departement'] == name2int[dept]) & (df['date'] >= start_idx) & (df['date'] < end_idx)].index))

            zeros_dates = np.concatenate((zeros_dates, df[(df['departement'] == name2int[dept]) & (df['date'] >= end_idx) & (df['date'] < end_idx + ks)].index))
            zeros_dates = np.concatenate((zeros_dates, df[(df['departement'] == name2int[dept]) & (df['date'] >= start_idx) & (df['date'] < start_idx + ks)].index))

    df.loc[zeros_dates, weights_columns] = 0
    print(zeros_dates)
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

def construct_graph_set(graph, date, X, Y, ks, start_features  : int):
    """
    Construct indexing graph with nodes sort by their id and date and corresponding edges.
    We consider spatial edges and temporal edges
    date : date
    X : train or val nodes
    Y : train or val target
    ks : size of the time series
    """

    date_index_ids_col = ids_columns.index('date')
    weight_index_ids_col = ids_columns.index('weight')

    mask = np.argwhere((X[:,date_index_ids_col] == date) & (X[:, weight_index_ids_col] > 0))
    #mask = np.argwhere((X[:,ids_columns.index('date')] == date))

    x = X[mask[:,0]]
    node_with_weight = np.unique(x[:, 0])

    connection = graph.edges[1][np.argwhere(np.isin(graph.edges[0], x[:, 0]))]

    if ks != 0:
        maskts = np.argwhere(((np.isin(X[:,0], x[:,0]) | np.isin(X[:, 0], connection)) & (X[:,date_index_ids_col] <= date) & (X[:,date_index_ids_col] >= date - ks)))
        if maskts.shape[0] == 0:
            return None, None, None
        xts = X[maskts[:,0]]
        x = np.concatenate((x, xts))

    if Y is not None:
        y = Y[mask[:,0]]
        if ks != 0:
            yts = Y[maskts[:,0]]
            yts[:, ids_columns.index('weight')] = 0
            y = np.concatenate((y, yts))

    # Graph indexing
    ind = np.lexsort((x[:,0], x[:,4]))
    x = x[ind]
    if Y is not None:
        y = y[ind]

    # Get graph specific spatial and temporal edges 
    maskgraph = np.argwhere((np.isin(graph.edges[0], np.unique(node_with_weight))) & (np.isin(graph.edges[1], np.unique(x[:,0]))))
    maskTemp = np.argwhere((np.isin(graph.temporalEdges[0], date)) & (np.isin(graph.temporalEdges[1], np.unique(x[:,4]))))

    spatialEdges = np.asarray([graph.edges[0][maskgraph[:,0]], graph.edges[1][maskgraph[:,0]]])
    temporalEdges = np.asarray([graph.temporalEdges[0][maskTemp[:,0]], graph.temporalEdges[1][maskTemp[:,0]]])

    edges = []
    target = []
    seen_edges = []
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
                    if (i, np.argwhere((x[:,4] == node[4]) & (x[:,0] == sp))[0][0]) not in seen_edges:
                        src.append(i)
                        target.append(np.argwhere((x[:,4] == node[4]) & (x[:,0] == sp))[0][0])
                        time_delta.append(0)
                        seen_edges.append((i, np.argwhere((x[:,4] == node[4]) & (x[:,0] == sp))[0][0]))

            # temporal edges
            if temporalEdges.shape[1] != 0:
                temporalNodes = x[np.argwhere((x[:,0] == node[0]))]
                temporalNodes = temporalNodes.reshape(-1, temporalNodes.shape[-1])
                temporal = temporalEdges[1][(np.isin(temporalEdges[1], temporalNodes[:,4])) & (temporalEdges[0] == node[4])]
                for tm in temporal:
                    if (i, np.argwhere((x[:,4] == tm) & (x[:,0] == node[0]))[0][0]) not in seen_edges:
                        src.append(i)
                        target.append(np.argwhere((x[:,4] == tm) & (x[:,0] == node[0]))[0][0])
                        time_delta.append(abs(node[4] - tm))
                        seen_edges.append((i, np.argwhere((x[:,4] == tm) & (x[:,0] == node[0]))[0][0]))

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
                        if (i, arg[0][0]) not in seen_edges:
                            src.append(i)
                            target.append(arg[0][0])
                            time_delta.append(abs(node[4] - tm))
                            seen_edges.append((i, arg[0][0]))

        edges = np.row_stack((src, target, time_delta)).astype(int)

    return x[:, start_features:], y, edges

def concat_temporal_graph_into_time_series(array : np, ks : int, date : int) -> np.array:
    uniqueNodes = np.unique(array[:,0])
    res = []

    date_limit_min = date - ks
    range_date = np.arange(date_limit_min, date)

    for uNode in uniqueNodes:
        arrayNode = array[array[:,0] == uNode]
        dept = np.unique(arrayNode[:, 3])[0]
        longitude = np.unique(arrayNode[:, 1])[0]
        laitude = np.unique(arrayNode[:, 2])[0]
        ind = np.lexsort([arrayNode[:,4]])

        #ind = np.flip(ind)
        arrayNode = arrayNode[ind]
        if ks + 1 > arrayNode.shape[0]:
            #print(date, allDates[int(date)], uNode)
            """new_data = np.zeros((ks + 1 - arrayNode.shape[0], arrayNode.shape[1]))
            new_data[:, 3] = dept
            new_data[:, 0] = uNode
            udates = np.unique(arrayNode[:, 4])
            for ud in udates:
                if ud < date_limit_min:
                    break
                if ud not in range_date and ud >= date_limit_min and ud <= date:
                    new_data = np.empty((1, arrayNode.shape[1]))
                    new_data[:, 3] = dept
                    new_data[:, 0] = uNode
                    new_data[:, 4] = ud
                    new_data[:, 1] = longitude
                    new_data[:, 1] = laitude
                    new_data[:, 5] = 0
                    for band in range(6, arrayNode.shape[1]):
                        x = arrayNode[:ks + 1, 4]
                        y = arrayNode[:ks + 1, band]
                        f = scipy.interpolate.interp1d(x, y, kind='linear')
                        new_values = f(ud)
                        new_data[:, band] = new_values

                    array_node_before_date = arrayNode[arrayNode[:, 4] < ud]
                    array_node_after_date = arrayNode[arrayNode[:, 4] > ud]
                    print(array_node_before_date.shape, new_data.shape, array_node_after_date.shape)
                    arrayNode = np.concatenate((array_node_before_date, new_data, array_node_after_date), axis=1)"""
            continue

            #ind = np.lexsort([arrayNode[:,4]])
            #arrayNode = arrayNode[ind]

        res.append(arrayNode[:ks + 1])

    if len(res) == 0:
        return None
    res = np.asarray(res)
    res = np.moveaxis(res, 1,2)
    return res

def construct_graph_with_time_series(graph, date : int,
                                     X : np.array, Y : np.array,
                                     ks :int, start_features : int) -> np.array:
    """
    Construct indexing graph with nodes sort by their id and date and corresponding edges.
    We consider spatial edges and time series X
    id : id of the subgraph in X
    X : train or val nodes
    Y : train or val target
    ks : size of the time series
    """
    date_index_ids_col = ids_columns.index('date')
    weight_index_ids_col = ids_columns.index('weight')

    mask = np.argwhere((X[:,date_index_ids_col] == date) & (X[:, weight_index_ids_col] > 0))
    #mask = np.argwhere((X[:,ids_columns.index('date')] == date))

    x = X[mask[:,0]]
    node_with_weight = np.unique(x[:, 0])

    connection = graph.edges[1][np.argwhere(np.isin(graph.edges[0], x[:, 0]))]

    maskts = np.argwhere(((np.isin(X[:,0], x[:,0]) | np.isin(X[:, 0], connection)) & (X[:,date_index_ids_col] <= date) & (X[:,date_index_ids_col] >= date - ks)))
    
    if maskts.shape[0] == 0:
        return None, None, None

    xts = X[maskts[:,0]]
    x = np.concatenate((x, xts))

    if Y is not None:
        y = Y[mask[:,0]]
        yts = Y[maskts[:,0]]
        yts[:,weight_index_ids_col] = 0
        y = np.concatenate((y, yts))

    # Graph indexing
    x = concat_temporal_graph_into_time_series(x, ks, date)
    if x is None:
        return None, None, None
    if Y is not None:
        y = concat_temporal_graph_into_time_series(y, ks, date)

    # Get graph specific spatial
    maskgraph = np.argwhere((np.isin(graph.edges[0], node_with_weight)) & (np.isin(graph.edges[1], np.unique(x[:,0]))))
    spatialEdges = np.asarray([graph.edges[0][maskgraph[:,0]], graph.edges[1][maskgraph[:,0]]])

    edges = []
    target = []
    src = []

    for i, node in enumerate(x):
        spatialNodes = x[np.argwhere((x[:,date_index_ids_col,-1] == node[date_index_ids_col][-1]))][:,:, 0, 0]
        if spatialEdges.shape[1] != 0:
            spatial = spatialEdges[1][(np.isin(spatialEdges[1], spatialNodes[:,0])) & (spatialEdges[0] == node[0][0])]
            for sp in spatial:
                src.append(i)
                target.append(np.argwhere((x[:,date_index_ids_col,-1] == node[date_index_ids_col][-1]) & (x[:,0,0] == sp))[0][0])
        src.append(i)
        target.append(i)
        
    edges = np.row_stack((src, target)).astype(int)
    return x[:, start_features:], y, edges

def construct_time_series(date : int,
                                     X : np.array, Y : np.array,
                                     ks :int, start_features : int) -> np.array:
    """
    Construct time series
    We consider spatial edges and time series X
    id : id of the subgraph in X
    X : train or val nodes
    Y : train or val target
    ks : size of the time series
    """
    date_index_ids_col = ids_columns.index('date')
    weight_index_ids_col = ids_columns.index('weight')

    maskgraph = np.argwhere((X[:,date_index_ids_col] == date) & (X[:, weight_index_ids_col] > 0))
    x = X[maskgraph[:,0]]
    if ks != 0:
        maskts = np.argwhere((np.isin(X[:,0], x[:,0]) & (X[:,date_index_ids_col] < date) & (X[:,date_index_ids_col] >= date - ks)))
        
        if maskts.shape[0] == 0:
            return None, None
    
        xts = X[maskts[:,0]]
        x = np.concatenate((x, xts))

    if Y is not None:
        y = Y[maskgraph[:,0]]
        if ks != 0:
            yts = Y[maskts[:,0]]
            yts[:,weight_index_ids_col] = 0
            y = np.concatenate((y, yts))

    # Graph indexing
    x = concat_temporal_graph_into_time_series(x, ks, date)
    if x is None:
        return None, None
    if Y is not None:
        y = concat_temporal_graph_into_time_series(y, ks, date)

    return x[:, start_features:], y

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
                graph,
                model : Model,
                dir : Path) -> dict:
    
    datesoftest = np.unique(ytrue[:, ids_columns.index('date')]).astype(int)
    years = np.unique([allDates[di].split('-')[0] for di in datesoftest])
    seasons = generate_season_dict(years)

    modes = ['temporal',
             'spatial',
             'spatio-temporal'
             ]
    top = 2

    res = {}
    for mode in modes:
        if mode == 'temporal':
            ytrue_mode = np.copy(ytrue)
            ypred_mode = np.copy(ypred)

        elif mode == 'spatial':
            dir_target = root_target
            raster = read_object(f'{testd_departement[0]}rasterScale{graph.scale}_{graph.base}.pkl', dir / 'raster')
            assert raster is not None
            #raster = raster[0]
            target_values = read_object(f'{testd_departement[0]}binScale0.pkl', dir_target / graph.sinister / graph.dataset_name / graph.sinister_encoding / 'bin' / graph.resolution)
            assert target_values is not None
            target_values = target_values[:, :, int(ytrue[0, 4]):int(ytrue[-1, 4])]
            target_values = np.nansum(target_values, axis=2)

            ytrue_mode = np.full((target_values.shape[0], target_values.shape[1], ytrue.shape[1]), fill_value=np.nan)
            ypred_mode = np.full((target_values.shape[0], target_values.shape[1], ypred.shape[1]), fill_value=np.nan)
            unodes = np.unique(raster)
            unodes = unodes[~np.isnan(unodes)]
            for node in unodes:
                mask_node = raster == node
                ytrue_mode[mask_node, -1] = target_values[mask_node]
                ytrue_mode[mask_node, -2] = target_values[mask_node]
                ytrue_mode[mask_node, -3] = 1
                ytrue_mode[mask_node, 0] = node
                ytrue_mode[mask_node, 1:-3] = 1
                ytrue_mode[mask_node, 3] = name2int[testd_departement[0]]

                ypred_mode[mask_node, 0] = np.nansum(ypred[ytrue[:, 0] == node, 0])
                ypred_mode[mask_node, -1] = 1
            
            ytrue_mode = ytrue_mode.reshape(-1, ytrue.shape[1])
            ypred_mode = ypred_mode.reshape(-1, ypred.shape[1])
            ypred_mode = ypred_mode[~np.isnan(ytrue_mode[:, -1])]
            ytrue_mode = ytrue_mode[~np.isnan(ytrue_mode[:, -1])]
        else:
            dir_target = root_target
            raster = read_object(f'{testd_departement[0]}rasterScale{graph.scale}_{graph.base}.pkl', dir / 'raster')
            assert raster is not None
            #raster = raster[0]
            target_values = read_object(f'{testd_departement[0]}binScale0.pkl', dir_target / graph.sinister / graph.dataset_name / graph.sinister_encoding / 'bin' / graph.resolution)
            assert target_values is not None

            udates = np.unique(ytrue[:, 4]).astype(int)
            target_values = target_values[:, :, udates]

            ytrue_mode = np.full((udates.shape[0], target_values.shape[0], target_values.shape[1], ytrue.shape[1]), fill_value=np.nan)
            ypred_mode = np.full((udates.shape[0], target_values.shape[0], target_values.shape[1], ypred.shape[1]), fill_value=np.nan)
            unodes = np.unique(raster)
            unodes = unodes[~np.isnan(unodes)]
            for node in unodes:
                mask_node = raster == node
                ytrue_mode[:, mask_node, -1] = np.moveaxis(target_values[mask_node], 0, 1)
                ytrue_mode[:, mask_node, -2] = np.moveaxis(target_values[mask_node], 0, 1)
                ytrue_mode[:, mask_node, -3] = 1
                ytrue_mode[:, mask_node, 0] = node
                ytrue_mode[:, mask_node, 1:-3] = 1
                ytrue_mode[:, mask_node, 3] = name2int[testd_departement[0]]

                for d in udates:
                    mask_date = udates == d
                    ypred_mode[mask_date, mask_node, 0] = ypred[(ytrue[:, 0] == node) & (ytrue[:, 4] == d), 0]
                    ypred_mode[mask_date, mask_node, -1] = 1
                    ytrue_mode[mask_date, mask_node, 4] = d

            ytrue_mode = ytrue_mode.reshape(-1, ytrue.shape[1])
            ypred_mode = ypred_mode.reshape(-1, ypred.shape[1])
            ypred_mode = ypred_mode[~np.isnan(ytrue_mode[:, -1])]
            ytrue_mode = ytrue_mode[~np.isnan(ytrue_mode[:, -1])]

        uids = np.unique(ytrue_mode[:, 0])
        ysum = np.empty((uids.shape[0], 2))
        ysum[:, 0] = uids
        for id in uids:
            ysum[np.argwhere(ysum[:, 0] == id)[:, 0], 1] = np.sum(ytrue_mode[np.argwhere(ytrue_mode[:, 0] == id)[:, 0], -2])
        
        ind = np.lexsort([ysum[:,1]])
        ymax = np.flip(ytrue_mode[ind, 0])[:top]
        mask_top = np.argwhere(np.isin(ytrue_mode[:, 0], ymax))[:, 0]

        for name, met, met_type in methods:
            oname = f'{mode}_{name}'
            if met_type == 'proba':
                band = -1
                if target == 'nbsinister' or target == 'binary':
                    band = -2
                    #if target == 'binary':
                    #    ytrue[:,band] = (ytrue[:,band] > 0).astype(int)
            
                mett = met(ypred_mode[:,0], ytrue_mode[:,band], ytrue_mode[:, ids_columns.index('weight')])
                
                if torch.is_tensor(mett):
                    mett = np.mean(mett.detach().cpu().numpy())

                res[oname] = mett

                for season, datesIndex in seasons.items():
                    mask = np.argwhere(np.isin(ytrue_mode[:, ids_columns.index('date')], datesIndex))[:,0]
                    if mask.shape[0] == 0:
                        continue
                    mett = met(ypred_mode[mask, 0], ytrue_mode[mask,band], ytrue_mode[mask, ids_columns.index('weight')])
                    if torch.is_tensor(mett):
                        mett = np.mean(mett.detach().cpu().numpy())

                    res[oname+'_'+season] = mett

                mett = met(ypred_mode[mask_top, 0], ytrue_mode[mask_top,band], ytrue_mode[mask_top, ids_columns.index('weight')])
                if torch.is_tensor(mett):
                    mett = mett.detach().cpu().numpy()
                res[oname+'_top_'+str(top)+'_cluster'] = mett

            elif met_type == 'cal':
                mett = met(ytrue_mode, ypred_mode[:,1], target, ytrue[:, ids_columns.index('weight')])

                if torch.is_tensor(mett):
                    mett = mett.detach().cpu().numpy()
                res[oname] = mett

                for season, datesIndex in seasons.items():
                    mask = np.argwhere(np.isin(ytrue_mode[:, ids_columns.index('date')], datesIndex))[:,0]
                    if mask.shape[0] == 0:
                        continue
                    mett = met(ytrue_mode[:, :], ypred_mode[:,1], target, ytrue_mode[:, ids_columns.index('weight')], mask)
                    if torch.is_tensor(mett):
                        mett = mett.detach().cpu().numpy()

                    res[oname+'_'+season] = mett

                mett = met(ytrue_mode, ypred_mode[:,1], target, None, None)
                if torch.is_tensor(mett):
                    mett = mett.detach().cpu().numpy()
                
                res[oname+'_unweighted'] = mett

                for season, datesIndex in seasons.items():
                    mask = np.argwhere(np.isin(ytrue_mode[:, ids_columns.index('date')], datesIndex))[:,0]
                    if mask.shape[0] == 0:
                        continue
                    mett = met(ytrue_mode[:], ypred_mode[:, 1], target, None, mask)

                    if torch.is_tensor(mett):
                        mett = mett.detach().cpu().numpy()

                    res[oname+'_unweighted_'+season] = mett

                mett = met(ytrue_mode[:], ypred_mode[:, 1], target, None, mask_top)
                if torch.is_tensor(mett):
                    mett = mett.detach().cpu().numpy()
                res[oname+'_top_'+str(top)+'_cluster_unweighted'] = mett

                mett = met(ytrue_mode[:], ypred_mode[:, 1], target, ytrue_mode[:, ids_columns.index('weight')], mask_top)
                if torch.is_tensor(mett):
                    mett = mett.detach().cpu().numpy()
                res[oname+'_top_'+str(top)+'_cluster'] = mett

            elif met_type == 'bin':

                mett = met(ytrue_mode, ypred_mode[:,0], target, ytrue_mode[:, ids_columns.index('weight')])

                if torch.is_tensor(mett):
                    mett = mett.detach().cpu().numpy()
                res[oname] = mett

                for season, datesIndex in seasons.items():
                    mask = np.argwhere(np.isin(ytrue_mode[:, ids_columns.index('date')], datesIndex))[:,0]
                    if mask.shape[0] == 0:
                        continue
                    mett = met(ytrue_mode[:, :], ypred_mode[:,0], target, ytrue_mode[:, ids_columns.index('weight')], mask)
                    if torch.is_tensor(mett):
                        mett = mett.detach().cpu().numpy()

                    res[oname+'_'+season] = mett

                mett = met(ytrue_mode, ypred_mode[:,0], target, None, None)
                if torch.is_tensor(mett):
                    mett = mett.detach().cpu().numpy()
                
                res[oname+'_unweighted'] = mett

                for season, datesIndex in seasons.items():
                    mask = np.argwhere(np.isin(ytrue_mode[:, ids_columns.index('date')], datesIndex))[:,0]
                    if mask.shape[0] == 0:
                        continue
                    mett = met(ytrue_mode[:], ypred_mode[:, 0], target, None, mask)

                    if torch.is_tensor(mett):
                        mett = mett.detach().cpu().numpy()

                    res[oname+'_unweighted_'+season] = mett

                mett = met(ytrue_mode[:], ypred_mode[:, 0], target, None, mask_top)
                if torch.is_tensor(mett):
                    mett = mett.detach().cpu().numpy()
                res[name+'_top_'+str(top)+'_cluster_unweighted'] = mett

                mett = met(ytrue[:], ypred_mode[:, 0], target, ytrue_mode[:, ids_columns.index('weight')], mask_top)
                if torch.is_tensor(mett):
                    mett = mett.detach().cpu().numpy()
                res[oname+'_top_'+str(top)+'_cluster'] = mett

            elif met_type == 'class':
                mett = met(ypred_mode, ytrue_mode, dir, weights=ytrue_mode[:, ids_columns.index('weight')], top=None)
                res[oname] = mett

                for season, datesIndex in seasons.items():
                    mask = np.argwhere(np.isin(ytrue_mode[:, ids_columns.index('date')], datesIndex))[:,0]
                    if mask.shape[0] == 0:
                        continue
                    mett = met(ypred_mode[mask], ytrue_mode[mask], dir, weights=ytrue_mode[mask, ids_columns.index('weight')], top=None)
                    res[oname+'_'+season] = mett

                mett = met(ypred_mode, ytrue_mode, dir, weights=None, top=10)
                res[oname+'top10'] = mett

                mett = met(ypred_mode[mask_top], ytrue_mode[mask_top], dir, weights=ytrue_mode[mask_top, ids_columns.index('weight')], top=None)
                if torch.is_tensor(mett):
                    mett = mett.detach().cpu().numpy()
                res[oname+'_top_'+str(top)+'_cluster'] = mett

                mett = met(ypred_mode[mask_top], ytrue_mode[mask_top], dir, weights=None, top=None)
                if torch.is_tensor(mett):
                    mett = mett.detach().cpu().numpy()
                res[oname+'_top_'+str(top)+'_cluster_unweighted'] = mett

            elif met_type == 'correlation':
                mett = met(ytrue_mode[:, band], ypred_mode[:,0])

                if torch.is_tensor(mett):
                    mett = mett.detach().cpu().numpy()
                res[oname] = mett

                for season, datesIndex in seasons.items():
                    mask = np.argwhere(np.isin(ytrue_mode[:, ids_columns.index('date')], datesIndex))[:,0]
                    if mask.shape[0] == 0:
                        continue
                    mett = met(ytrue_mode[:, band], ypred_mode[:,0], mask)
                    if torch.is_tensor(mett):
                        mett = mett.detach().cpu().numpy()

                    res[oname+'_'+season] = mett

                mett = met(ytrue_mode[:, band], ypred_mode[:, 0], mask_top)
                if torch.is_tensor(mett):
                    mett = mett.detach().cpu().numpy()
                res[name+'_top_'+str(top)+'_cluster'] = mett

            logger.info(f'{oname, res[oname]}')

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

    linear = np.linspace(start=0, stop=1, num=5)

    res = {}

    quantiles = [(0, linear[0]),
                 (1, linear[1]),
                 (2, linear[2]),
                 (3, linear[3]),
                 (4, linear[4]),
                 ]
    
    error = []
    for (cl, expected) in quantiles:
        nf = (ytrue[ypred == cl] > 0).astype(int)
        if nf.shape[0] == 0:
            continue
        else:
            number_of_fire = np.mean(nf)
        error.append(abs((expected - number_of_fire)))
        res[cl] = abs((expected - number_of_fire))
    if len(error) == 0:
        return math.inf
    
    res['mean'] = np.nanmean(error)
    
    return res

def frequency_class_error(Y : np.array, ypred : np.array, target : str, weights = None, mask : np.array = None):
    if mask is None:
        mask = np.arange(ypred.shape[0])

    ytrue = Y[mask, :]
    ypred = ypred[mask]

    res = {}

    quantiles = [(0, ytrue[ytrue[:, -3] == 0].shape[0] / ytrue.shape[0]),
                 (1, ytrue[ytrue[:, -3] == 1].shape[0] / ytrue.shape[0]),
                 (2, ytrue[ytrue[:, -3] == 2].shape[0] / ytrue.shape[0]),
                 (3, ytrue[ytrue[:, -3] == 3].shape[0] / ytrue.shape[0]),
                 (4, ytrue[ytrue[:, -3] == 4].shape[0] / ytrue.shape[0]),
                 ]
    
    error = []
    for (cl, expected) in quantiles:
        fre = ypred[ypred == cl].shape[0] / ytrue.shape[0]
        error.append(abs((expected - fre)))
        res[cl] = abs((expected - fre))
    if len(error) == 0:
        return math.inf
    
    res['mean'] = np.nanmean(error)
    
    return res

# Les fonctions pour calculer les coefficients de corrélation

def kendall_coefficient(y_true, y_pred, mask=None):
    """
    Calcule le coefficient de Kendall en utilisant un masque optionnel.
    
    Paramètres :
    - y_true: Valeurs réelles (np.ndarray).
    - y_pred: Valeurs prédites (np.ndarray).
    - mask: Masque optionnel (np.ndarray de booléens). Si None, pas de masque appliqué.
    
    Retourne :
    - Le coefficient de Kendall (float).
    """
    if mask is None:
       return kendalltau(y_true, y_pred)[0]
    
    mask = mask.reshape(-1)
    return kendalltau(y_true[mask], y_pred[mask])[0]

def pearson_coefficient(y_true, y_pred, mask=None):
    """
    Calcule le coefficient de Pearson en utilisant un masque optionnel.
    
    Paramètres :
    - y_true: Valeurs réelles (np.ndarray).
    - y_pred: Valeurs prédites (np.ndarray).
    - mask: Masque optionnel (np.ndarray de booléens). Si None, pas de masque appliqué.
    
    Retourne :
    - Le coefficient de Pearson (float).
    """
    if mask is None:
       return pearsonr(y_true, y_pred)[0]

    mask = mask.reshape(-1)
    return pearsonr(y_true[mask], y_pred[mask])[0]

def spearman_coefficient(y_true, y_pred, mask=None):
    """
    Calcule le coefficient de Spearman en utilisant un masque optionnel.
    
    Paramètres :
    - y_true: Valeurs réelles (np.ndarray).
    - y_pred: Valeurs prédites (np.ndarray).
    - mask: Masque optionnel (np.ndarray de booléens). Si None, pas de masque appliqué.
    
    Retourne :
    - Le coefficient de Spearman (float).
    """
    if mask is None:
        return spearmanr(y_true, y_pred)[0]
    
    mask = mask.reshape(-1)
    return spearmanr(y_true[mask], y_pred[mask])[0]


def my_f1_score(Y: np.array, ypred: np.array, target: str, weights: np.array = None, mask: np.array = None):
    """
    Calcule le F1 score sur une gamme de seuils et retourne le meilleur score avec les métriques associées.

    Paramètres:
    - Y: Labels réels. Pour la classification binaire, Y doit avoir la forme (n_samples, 1) avec des labels binaires.
         Pour les cibles de régression, Y doit avoir la forme (n_samples, 2), où la dernière colonne contient les valeurs continues.
    - ypred: Valeurs prédites, soit des probabilités, soit des sorties continues.
    - target: Type de cible. Doit être l'un de 'binary', 'nbsinister', 'risk' ou d'autres cibles personnalisées.
    - weights: Poids des échantillons. Si None, tous les échantillons ont un poids égal.
    - mask: Masque booléen pour sélectionner un sous-ensemble d'échantillons. Si None, tous les échantillons sont utilisés.

    Retourne:
    - res: Dictionnaire contenant le meilleur F1 score, la précision, le rappel, le seuil, et la valeur du seuil.
    """
    if mask is None:
        mask = np.arange(ypred.shape[0])
    
    ytrue = Y[:, -2] > 0  # On suppose que les labels binaires sont dans l'avant-dernière colonne
    ytrueReg = Y[:, -1]   # On suppose que les cibles de régression sont dans la dernière colonne

    bounds = np.linspace(0.01, 0.90, 10)  # Seuils de 0.01 à 0.99 par pas de 0.01

    res = {}

    if target in ['binary']:
        maxi = 1.0
    else:
        maxi = np.nanmax(ytrueReg)

    bestScore = 0.0
    bestBound = None
    prec = 0.0
    rec = 0.0

    ypredNumpy = ypred[mask]
    ytrueNumpy = ytrue[mask]
    ytrueRegNumpy = ytrueReg[mask]

    if weights is not None:
        weightsNumpy = weights[mask]
    else:
        weightsNumpy = np.ones(ytrueNumpy.shape[0])

    for bound in bounds:
        if target in ['binary', 'nbsinister', 'risk']:
            yBinPred = (ypredNumpy > bound * maxi).astype(int)
        else:
            yBinPred = (ytrueRegNumpy > bound * maxi).astype(int)

        f1 = f1_score(ytrueNumpy, yBinPred, sample_weight=weightsNumpy)
        if f1 > bestScore:
            bestScore = f1
            bestBound = bound
            prec = precision_score(ytrueNumpy, yBinPred, sample_weight=weightsNumpy)
            rec = recall_score(ytrueNumpy, yBinPred, sample_weight=weightsNumpy)

    if bestBound is None:
        bestBound = bounds[0]
        yBinPred = (ypredNumpy > bestBound * maxi).astype(int)
        prec = precision_score(ytrueNumpy, yBinPred, sample_weight=weightsNumpy)
        rec = recall_score(ytrueNumpy, yBinPred, sample_weight=weightsNumpy)
        bestScore = f1_score(ytrueNumpy, yBinPred, sample_weight=weightsNumpy)

    res['f1'] = bestScore
    res['precision'] = prec
    res['recall'] = rec
    res['threshold'] = bestBound
    res['value'] = bestBound * maxi

    return res

def class_risk(ypred, ytrue, dir : Path, weights = None, top=None) -> dict:
    if torch.is_tensor(ypred):
        ypred = ypred.detach().cpu().numpy().astype(float)
    if torch.is_tensor(ytrue):
        ytrue = ytrue.detach().cpu().numpy().astype(float)
    
    res = {}
        
    uniqueClass = [0,1,2,3,4]

    nbsinisteriny = []
    nbsinisterinpred = []

    for cls in uniqueClass:
        mask2 = np.argwhere(ytrue[:, -3] == cls)[:, 0]
        if mask2.shape[0] == 0:
            nbsinisteriny.append(0)
        else:
            nbsinisteriny.append(frequency_ratio(ytrue[:, -2], mask2))
        
        mask2 = np.argwhere(ypred == cls)[:, 0]
        if mask2.shape[0] == 0:
            nbsinisterinpred.append(0)
        else:
            nbsinisterinpred.append(frequency_ratio(ytrue[:, -2], mask2))

        res[cls] = abs(nbsinisteriny[-1] - nbsinisterinpred[-1])

    res['mean'] = np.mean(np.abs(np.asarray(nbsinisteriny) - np.asarray(nbsinisterinpred)))

    return res

def class_accuracy(ypred, ytrue, dir : Path, weights = None, top=None) -> dict:
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

    yweights = weightsNumpy.reshape(-1)
    res = accuracy_score(ytrue[:, -3], ypred[:, -1], sample_weight=yweights)
            
    return res

def binary_accuracy(Y: np.array, ypred: np.array, target: str, weights: np.array = None, mask: np.array = None) -> dict:
    if mask is None:
        mask = np.arange(ypred.shape[0])
    
    ytrue = Y[:, -2] > 0  # On suppose que les labels binaires sont dans l'avant-dernière colonne
    ytrueReg = Y[:, -1]   # On suppose que les cibles de régression sont dans la dernière colonne

    # Augmentation du nombre de seuils testés
    if target != 'nbsinister':
        bounds = np.linspace(0.01, 0.90, 10)  # Seuils de 0.01 à 0.99 par pas de 0.01
    else:
        bounds = np.linspace(0.90, 1.00, 11)  # Seuils de 0.90 à 1.00 par pas de 0.01

    res = {}

    if target in ['binary', 'nbsinister', 'risk']:
        maxi = 1.0
    else:
        maxi = np.nanmax(ytrueReg)

    bestScore = 0.0

    ypredNumpy = ypred[mask]
    ytrueNumpy = ytrue[mask]
    ytrueRegNumpy = ytrueReg[mask]

    if weights is not None:
        weightsNumpy = weights[mask]
    else:
        weightsNumpy = np.ones(ytrueNumpy.shape[0])

    for bound in bounds:
        if target in ['binary', 'nbsinister', 'risk']:
            yBinPred = (ypredNumpy > bound * maxi).astype(int)
        else:
            yBinPred = (ytrueRegNumpy > bound * maxi).astype(int)

        f1 = accuracy_score(ytrueNumpy, yBinPred, sample_weight=weightsNumpy)
        if f1 > bestScore:
            bestScore = f1
            bestBound = bound

    if 'bestBound' not in locals():
        bestBound = bounds[0]
        yBinPred = (ypredNumpy > bestBound * maxi).astype(int)
        bestScore = accuracy_score(ytrueNumpy, yBinPred, sample_weight=weightsNumpy)

    res['accuracy'] = bestScore
            
    return res

def balanced_class_accuracy(ypred, ytrue, dir : Path, weights = None, top=None) -> dict:
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

    yweights = weightsNumpy.reshape(-1)

    res = balanced_accuracy_score(ytrue[:, -3], ypred[:, -1], sample_weight=yweights)
            
    return res

def class_accuracy(ypred, ytrue, dir : Path, weights = None, top=None) -> dict:
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

    yweights = weightsNumpy.reshape(-1)

    res = accuracy_score(ytrue[:, -3], ypred[:, -1], sample_weight=yweights)
            
    return res

def my_concordance_index(event_times, nb_event, predicted_scores, event_observed, tol_tied=0.0, max_survival_times=None):
    """
    Calculates the Concordance Index (C-index) from the actual survival times, predicted scores,
    a variable indicating if the event was observed (1) or censored (0),
    and an additional comparison with the number of events.
    Includes a tolerance on the equality of predicted scores via tol_tied.
    Elements with survival time greater than their corresponding value in max_survival_times are considered equal and have equal risk.

    :param event_times: Array of actual survival times
    :param predicted_scores: Array of predicted scores by the model
    :param event_observed: Binary array indicating if the event was observed (1) or censored (0)
    :param nb_event: Array of the number of observed events
    :param tol_tied: Floating tolerance for comparing equality of scores (default 0.0)
    :param max_survival_times: Array with the same shape as event_times; times greater than their corresponding values are considered equal
    :return: The three C-indexes: for survival times, for number of events, and combined
    """
    
    n = len(event_times)
    assert len(predicted_scores) == n
    assert len(event_observed) == n
    assert len(nb_event) == n

    # Adjust survival times to consider times greater than their corresponding max_survival_times as equal
    if max_survival_times is not None:
        assert len(max_survival_times) == n
        adjusted_event_times = [
            min(event_times[i], max_survival_times[i]) for i in range(n)
        ]
    else:
        adjusted_event_times = event_times.copy()
    
    num_comparable_pairs_time = 0  # Comparable pairs for survival times
    num_concordant_pairs_time = 0  # Concordant pairs for survival times
    num_tied_pairs_time = 0        # Pairs with identical predicted scores for survival times (with tolerance)
    
    num_comparable_pairs_event = 0  # Comparable pairs for the number of events
    num_concordant_pairs_event = 0  # Concordant pairs for the number of events
    num_tied_pairs_event = 0        # Pairs with identical predicted scores for the number of events (with tolerance)

    # Compare all pairs (i, j)
    for i in range(n):
        for j in range(i + 1, n):
            # Comparison of adjusted survival times
            if adjusted_event_times[i] != adjusted_event_times[j]:
                if event_observed[i] == 1 or event_observed[j] == 1:  # Comparable if at least one event is observed
                    num_comparable_pairs_time += 1
                    
                    if adjusted_event_times[i] < adjusted_event_times[j]:
                        if predicted_scores[i] > predicted_scores[j]:
                            num_concordant_pairs_time += 1
                        elif abs(predicted_scores[i] - predicted_scores[j]) <= tol_tied:  # Tolerance on equality
                            num_tied_pairs_time += 1
                    elif adjusted_event_times[j] < adjusted_event_times[i]:
                        if predicted_scores[j] > predicted_scores[i]:
                            num_concordant_pairs_time += 1
                        elif abs(predicted_scores[j] - predicted_scores[i]) <= tol_tied:
                            num_tied_pairs_time += 1

            # Comparison of the number of events
            if nb_event[i] != nb_event[j]:
                if event_observed[i] == 1 or event_observed[j] == 1: # Comparable if at least one event is observed
                    num_comparable_pairs_event += 1
                    
                    if nb_event[i] > nb_event[j]:
                        if predicted_scores[i] > predicted_scores[j]:
                            num_concordant_pairs_event += 1
                        elif abs(predicted_scores[i] - predicted_scores[j]) <= tol_tied:
                            num_tied_pairs_event += 1
                    elif nb_event[j] > nb_event[i]:
                        if predicted_scores[j] > predicted_scores[i]:
                            num_concordant_pairs_event += 1
                        elif abs(predicted_scores[j] - predicted_scores[i]) <= tol_tied:
                            num_tied_pairs_event += 1

    # Calculation of C-index for survival times
    if num_comparable_pairs_time == 0:
        c_index_time = 0
    else:
        c_index_time = (num_concordant_pairs_time + 0.5 * num_tied_pairs_time) / num_comparable_pairs_time
    
    # Calculation of C-index for the number of events
    if num_comparable_pairs_event == 0:
        c_index_event = 0
    else:
        c_index_event = (num_concordant_pairs_event + 0.5 * num_tied_pairs_event) / num_comparable_pairs_event

    # Total concordant, tied, and comparable pairs
    total_concordant_pairs = num_concordant_pairs_time + num_concordant_pairs_event
    total_tied_pairs = num_tied_pairs_time + num_tied_pairs_event
    total_comparable_pairs = num_comparable_pairs_time + num_comparable_pairs_event
    
    # Calculation of the combined C-index
    if total_comparable_pairs == 0:
        c_index_combined = 0
    else:
        c_index_combined = (total_concordant_pairs + 0.5 * total_tied_pairs) / total_comparable_pairs
    
    return c_index_time, c_index_event, c_index_combined

def c_index(ypred, ytrue, dir: Path, weights=None, top=None) -> dict:
    # Convertir ypred et ytrue en numpy si ce sont des tenseurs PyTorch
    if torch.is_tensor(ypred):
        ypred = ypred.detach().cpu().numpy().astype(float)
    if torch.is_tensor(ytrue):
        ytrue = ytrue.detach().cpu().numpy().astype(float)

    # Gestion des poids
    if weights is not None:
        if torch.is_tensor(weights):
            weightsNumpy = weights.detach().cpu().numpy()
        else:
            weightsNumpy = weights
    else:
        weightsNumpy = np.ones(ytrue.shape[0])

    yweights = weightsNumpy.reshape(-1)

    # Create max_survival_time
    max_survival_times = np.empty(ytrue.shape[0])
    for i, node in enumerate(ytrue):
        date = allDates[int(node[4])]
        month = date.split('-')[1]
        max_survival_times[i] = 30
        """if month in ['10','11','12','01']:
            max_survival_times[i] = 3
        elif month in ['02','03', '04','05']:
            max_survival_times[i] = 5
        elif month in ['06', '07', '08', '09']:
            max_survival_times[i] = 7"""

    # Calcul du C-index with a minimum of 0
    y_events = np.ones(ytrue.shape[0])
    c_index_time_zero, c_index_event_zero, c_index_combined_zero = my_concordance_index(ytrue[:, ids_columns.index('days_until_next_event')],
                                                                         ytrue[:, (ids_columns + targets_columns).index('nbsinister')],
                                                                         ypred[:, 0],
                                                                         y_events,
                                                                         max_survival_times=max_survival_times,
                                                                         tol_tied=1e-8)
    
    # Calcul du C-index with a minimum of 1
    y_events = np.zeros(ytrue.shape[0])
    y_events[ytrue[:, (ids_columns + targets_columns).index('nbsinister')] >= 1] = 1
    c_index_time_one, c_index_event_one, c_index_combined_one = my_concordance_index(ytrue[:, ids_columns.index('days_until_next_event')],
                                                                         ytrue[:, (ids_columns + targets_columns).index('nbsinister')],
                                                                         ypred[:, 0],
                                                                         y_events,
                                                                         max_survival_times=max_survival_times,
                                                                         tol_tied=1e-8)
    
    # Calcul du C-index with a minimum of 2
    y_events = np.zeros(ytrue.shape[0])
    y_events[ytrue[:, (ids_columns + targets_columns).index('nbsinister')] >= 2] = 1
    c_index_time_two, c_index_event_two, c_index_combined_two = my_concordance_index(ytrue[:, ids_columns.index('days_until_next_event')],
                                                                         ytrue[:, (ids_columns + targets_columns).index('nbsinister')],
                                                                         ypred[:, 0],
                                                                         y_events,
                                                                         max_survival_times=max_survival_times,
                                                                         tol_tied=1e-8)
    
    # Calcul du C-index with a minimum of 3
    y_events = np.zeros(ytrue.shape[0])
    y_events[ytrue[:, (ids_columns + targets_columns).index('nbsinister')] >= 3] = 1
    c_index_time_three, c_index_event_three, c_index_combined_three = my_concordance_index(ytrue[:, ids_columns.index('days_until_next_event')],
                                                                         ytrue[:, (ids_columns + targets_columns).index('nbsinister')],
                                                                         ypred[:, 0],
                                                                         y_events,
                                                                         max_survival_times=max_survival_times,
                                                                         tol_tied=1e-8)
    
    return {'c_index_time_0': c_index_time_zero,'c_index_event_0': c_index_event_zero, 'c_index_combined_0' : c_index_combined_zero,
            'c_index_time_1': c_index_time_one, 'c_index_event_1': c_index_event_one, 'c_index_combined_1' : c_index_combined_one,
            'c_index_time_2': c_index_time_two, 'c_index_event_2': c_index_event_two, 'c_index_combined_2' : c_index_combined_two,
            'c_index_time_3': c_index_time_three,'c_index_event_3': c_index_event_three, 'c_index_combined_3' : c_index_combined_three
            }

def c_index_class(ypred, ytrue, dir: Path, weights=None, top=None) -> dict:
    # Convertir ypred et ytrue en numpy si ce sont des tenseurs PyTorch
    if torch.is_tensor(ypred):
        ypred = ypred.detach().cpu().numpy().astype(float)
    if torch.is_tensor(ytrue):
        ytrue = ytrue.detach().cpu().numpy().astype(float)

    # Gestion des poids
    if weights is not None:
        if torch.is_tensor(weights):
            weightsNumpy = weights.detach().cpu().numpy()
        else:
            weightsNumpy = weights
    else:
        weightsNumpy = np.ones(ytrue.shape[0])

    yweights = weightsNumpy.reshape(-1)

    # Create max_survival_time
    max_survival_times = np.empty(ytrue.shape[0])
    for i, node in enumerate(ytrue):
        date = allDates[int(node[4])]
        month = date.split('-')[1]
        max_survival_times[i] = 15
        """if month in ['10','11','12','01']:
            max_survival_times[i] = 3
        elif month in ['02','03', '04','05']:
            max_survival_times[i] = 5
        elif month in ['06', '07', '08', '09']:
            max_survival_times[i] = 7"""

     # Calcul du C-index with a minimum of 0
    y_events = np.ones(ytrue.shape[0])
    c_index_time_zero, c_index_event_zero, c_index_combined_zero = my_concordance_index(ytrue[:, ids_columns.index('days_until_next_event')],
                                                                         ytrue[:, (ids_columns + targets_columns).index('nbsinister')],
                                                                         ypred[:, 1],
                                                                         y_events,
                                                                         max_survival_times=max_survival_times,
                                                                         tol_tied=1e-8)
    
    # Calcul du C-index with a minimum of 1
    y_events = np.zeros(ytrue.shape[0])
    y_events[ytrue[:, (ids_columns + targets_columns).index('nbsinister')] >= 1] = 1
    c_index_time_one, c_index_event_one, c_index_combined_one = my_concordance_index(ytrue[:, ids_columns.index('days_until_next_event')],
                                                                         ytrue[:, (ids_columns + targets_columns).index('nbsinister')],
                                                                         ypred[:, 1],
                                                                         y_events,
                                                                         max_survival_times=max_survival_times,
                                                                         tol_tied=1e-8)
    
    # Calcul du C-index with a minimum of 2
    y_events = np.zeros(ytrue.shape[0])
    y_events[ytrue[:, (ids_columns + targets_columns).index('nbsinister')] >= 2] = 2
    c_index_time_two, c_index_event_two, c_index_combined_two = my_concordance_index(ytrue[:, ids_columns.index('days_until_next_event')],
                                                                         ytrue[:, (ids_columns + targets_columns).index('nbsinister')],
                                                                         ypred[:, 1],
                                                                         y_events,
                                                                         max_survival_times=max_survival_times,
                                                                         tol_tied=1e-8)
    
    # Calcul du C-index with a minimum of 3
    y_events = np.zeros(ytrue.shape[0])
    y_events[ytrue[:, (ids_columns + targets_columns).index('nbsinister')] >= 3] = 3
    c_index_time_three, c_index_event_three, c_index_combined_three = my_concordance_index(ytrue[:, ids_columns.index('days_until_next_event')],
                                                                         ytrue[:, (ids_columns + targets_columns).index('nbsinister')],
                                                                         ypred[:, 1],
                                                                         y_events,
                                                                         max_survival_times=max_survival_times,
                                                                         tol_tied=1e-8)
    
    return {'c_index_time_0': c_index_time_zero,'c_index_event_0': c_index_event_zero, 'c_index_combined_0' : c_index_combined_zero,
            'c_index_time_1': c_index_time_one, 'c_index_event_1': c_index_event_one, 'c_index_combined_1' : c_index_combined_one,
            'c_index_time_2': c_index_time_two, 'c_index_event_2': c_index_event_two, 'c_index_combined_2' : c_index_combined_two,
            'c_index_time_3': c_index_time_three,'c_index_event_3': c_index_event_three, 'c_index_combined_3' : c_index_combined_three
            }


def calibrated_error(ypred, ytrue, dir, weights = None, top = None) -> dict:

    pass

def mean_absolute_error_class(ypred, ytrue,
                              dir : Path, weights = None, top = None) -> dict:
    
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

    yweights = weightsNumpy.reshape(-1)

    if top is not None:
        minBound = np.nanmax(ytrue[:,-1]) * (1 - top/100)
        mask2 = (ytrue[:,-1] > minBound)
        ytrueclass = ytrue[:, -3][mask2]
        ypredclass = ypred[:, -1][mask2]
        yweights = None
    else:
        ytrueclass = ytrue[:, -3]
        ypredclass = ypred[:, -1]

    if ytrueclass.shape[0] == 0 or ypredclass.shape[0] == 0:
        res = None
    else:
        res = mean_absolute_error(ytrueclass, ypredclass, sample_weight=yweights)

    return res

def mode_filter(image, kernel_size=3):
    # Assurez-vous que le kernel_size est impair pour avoir un centre
    assert kernel_size % 2 == 1, "Le kernel_size doit être impair"
    
    # Obtenir le padding pour le centre du kernel
    pad_width = kernel_size // 2
    
    # Padding de l'image pour gérer les bords
    padded_image = np.pad(image, pad_width=pad_width, mode='edge')
    
    # Image de sortie
    filtered_image = np.zeros_like(image)
    
    # Parcourir chaque pixel de l'image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extraire la fenêtre locale
            local_window = padded_image[i:i + kernel_size, j:j + kernel_size]
            local_window = local_window[local_window != -1]
            # Trouver la valeur la plus fréquente dans la fenêtre
            counter_res = Counter(local_window.flatten()).most_common(1)
            #if len(counter_res) == 0:
            #    continue
            most_common_value = Counter(local_window.flatten()).most_common(1)[0][0]
            if most_common_value == -1:
                continue
            
            # Assigner cette valeur au pixel central
            filtered_image[i, j] = most_common_value
    
    return filtered_image

def merge_adjacent_clusters(image, size_threshold, oridata, exclude_label=None, background=-1):
    labeled_image = np.copy(image)
    
    umarker = np.unique(image)
    umarker = umarker[(umarker != 0) & ~(np.isnan(umarker))]
    risk_image = np.full(oridata.shape, fill_value=np.nan)
    for m in umarker:
        mask = (image == m)  # Create a mask for each unique marker
        risk_image[mask] = np.sum(oridata[mask])

    # Obtenir les propriétés des régions labellisées
    regions = measure.regionprops(labeled_image)

    # Masque pour les clusters à fusionner
    #mask = #np.zeros_like(labeled_image, dtype=labeled_image.dtype)
    mask = np.copy(labeled_image)

    nb_attempt = 5
    std_factor = 0.25
    thresh = np.nanstd(np.unique(risk_image)) * std_factor

    changed_labels = []

    # Fusionner les clusters de petite taille
    for i, region in enumerate(regions):
        if region.label in changed_labels:
            continue

        if region.label == exclude_label or region.label == background:
            # Si le cluster est à exclure, on le conserve tel quel
            mask[labeled_image == region.label] = region.label
            continue
        
        if region.area < size_threshold:
            ones = region.area
            label = region.label
            nb_test = 0
            while ones < size_threshold and ones > 0 and nb_test < nb_attempt:
                
                # Obtenir les labels des voisins
                mask_label = mask == label
                neighbors = segmentation.find_boundaries(mask_label, connectivity=1, mode='outer', background=background)
                neighbor_labels = np.unique(mask[neighbors])
                neighbor_labels = neighbor_labels[(neighbor_labels != exclude_label) & (neighbor_labels != background) & (neighbor_labels != label)]  # Exclure le fond et le label à exclure
                dilate = True

                if len(neighbor_labels) > 0: 

                    risk_label = risk_image[mask == label][0]
                    largest_neighbor_label = None

                    # On vérifie la similarité avec le risque de chaque voisin
                    best_simi = math.inf
                    for neighbor_label in neighbor_labels:
                        ori_labels = np.unique(labeled_image[mask == neighbor_label])
                        if ori_labels.shape[0] > 1:
                            for ori_neighbor_label in ori_labels:
                                risk_neighbor_label = risk_image[labeled_image == ori_neighbor_label][0]
                                diff = abs(risk_label - risk_neighbor_label)
                                if diff < thresh and diff < best_simi:
                                    largest_neighbor_label = neighbor_label
                                    best_simi = diff
                    
                    # Si on a trouvé un voisin similaire
                    if largest_neighbor_label is not None:
                        dilate = False
                        mask[mask_label] = largest_neighbor_label
                        changed_labels.append(label)
                        label = largest_neighbor_label

                # Si on a pas de voisin ou que les voisins sont différents
                if dilate:
                    mask_label = morphology.dilation(mask_label, morphology.square(3))
                    mask[(mask_label)] = label
                    nb_test += 1
                            
                mask_label = mask == label
                ones = np.argwhere(mask_label == 1).shape[0]

                if not dilate:
                    break

            # Si la taille est petite et qu'on a jamais trouvé de voisin -> région isolée.
            if ones < size_threshold:
                mask[mask == region.label] = 0

            regions = measure.regionprops(mask)
            
        else:
            # Si le cluster est assez grand, on le conserve tel quel
            mask[labeled_image == region.label] = region.label

    # Image finale après fusion
    merged_image = np.copy(mask)
    
    return merged_image


def find_clusters(image, threshold, clusters_to_ignore=None, background=0):
    """
    Traverse the clusters in an image and return the clusters whose size is greater than a given threshold.
    
    :param image: np.array, 2D image with values representing the clusters
    :param threshold: int, minimum size of the cluster to be considered
    :param background: int, value representing the background (default: 0)
    :param clusters_to_ignore: list, list of clusters to ignore (default: None)
    :return: list, list of cluster IDs whose size is greater than the threshold
    """
    # Initialize the list of valid clusters to return
    valid_clusters = []
    
    # If no clusters to ignore are provided, initialize with an empty list
    if clusters_to_ignore is None:
        clusters_to_ignore = []
    
    # Create a mask where the background is ignored
    mask = image != background
    
    # Label the clusters in the image
    cluster_ids = np.unique(image[mask])
    cluster_ids = cluster_ids[~np.isnan(cluster_ids)]
    
    # Traverse each cluster and check its size
    for cluster_id in cluster_ids:
        # Skip the cluster if it's in the ignore list
        if cluster_id == clusters_to_ignore:
            continue
        
        # Calculate the size of the cluster
        cluster_size = np.sum(image == cluster_id)
        
        # If the cluster size exceeds the threshold, add it to the list
        if cluster_size > threshold:
            valid_clusters.append(cluster_id)
    
    return valid_clusters

def split_large_clusters(image, size_threshold, min_cluster_size, background):
    labeled_image = np.copy(image)
    
    # Obtenir les propriétés des régions labellisées
    regions = measure.regionprops(labeled_image)
    
    # Initialiser une image pour les nouveaux labels après division
    new_labeled_image = np.copy(labeled_image)
    changes_made = False

    for region in regions:

        if region.label == background:
            continue

        if region.area > size_threshold:
            # Si la région est plus grande que le seuil, la diviser
            
            # Extraire le sous-image du cluster
            minr, minc, maxr, maxc = region.bbox
            region_mask = (labeled_image[minr:maxr, minc:maxc] == region.label)
            
            # Obtenir les coordonnées des pixels du cluster
            coords = np.column_stack(np.nonzero(region_mask))
            
            # Appliquer K-means pour diviser en 2 clusters
            if len(coords) > 1:  # Assurez-vous qu'il y a suffisamment de points pour appliquer K-means
                kmeans = KMeans(n_clusters=2, random_state=0).fit(coords)
                labels = kmeans.labels_
                
                # Créer deux nouveaux labels
                new_label_1 = new_labeled_image.max() + 1
                new_label_2 = new_labeled_image.max() + 2
                
                # Assigner les nouveaux labels aux pixels correspondants
                new_labeled_image[minr:maxr, minc:maxc][region_mask] = np.where(labels == 0, new_label_1, new_label_2)
                
                changes_made = True
    
    # Si des changements ont été effectués, vérifier s'il y a des clusters à fusionner
    if changes_made:
        new_labeled_image = split_large_clusters(new_labeled_image, size_threshold, min_cluster_size, background)
    
    return new_labeled_image

def most_frequent_neighbor(image, mask, i, j, non_cluster):
    """
    Retourne la valeur la plus fréquente des voisins d'un pixel, en tenant compte d'un masque.
    """
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < image.shape[0] and 0 <= nj < image.shape[1] and not mask[ni, nj]:
                if non_cluster is not None:
                    if len(image[ni, nj][image[ni, nj] != non_cluster]) > 0:
                        neighbors.append(image[ni, nj][image[ni, nj] != non_cluster][0])
                else: 
                    neighbors.append(image[ni, nj])
    if neighbors:
        return Counter(neighbors).most_common(1)[0][0]
    else:
        return image[i, j]  # En cas d'absence de voisins valides

def merge_small_clusters(image, min_size, non_cluster=None):
    """
    Fusionne les petits clusters en remplaçant leurs pixels par la valeur la plus fréquente de leurs voisins.
    """
    output_image = np.copy(image)
    unique_clusters, counts = np.unique(image, return_counts=True)
    counts = counts[~np.isnan(unique_clusters)]
    unique_clusters = unique_clusters[~np.isnan(unique_clusters)]
    for cluster_id, count in zip(unique_clusters, counts):
        if cluster_id == -1:
            continue
        if count < min_size:
            # Trouver tous les pixels appartenant au cluster
            mask = (image == cluster_id)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if mask[i, j]:
                        output_image[i, j] = most_frequent_neighbor(output_image, mask, i, j, non_cluster)
    
    return output_image

def relabel_clusters(cluster_labels, started):
    """
    Réorganise les labels des clusters pour qu'ils soient séquentiels et croissants.
    """
    unique_labels = np.sort(np.unique(cluster_labels[~np.isnan(cluster_labels)]))

    relabeled_image = np.copy(cluster_labels)
    for ncl, cl in enumerate(unique_labels):
        relabeled_image[cluster_labels == cl] = ncl + started
    return relabeled_image

def get_features_name_list(scale, features, methods):
    features_name = []
    if scale == 0:
        methods = ['mean']
    for var in features:
        if var == 'Calendar':
            features_name += calendar_variables
        elif var == 'air':
            features_name += air_variables
        elif var in landcover_variables:
            features_name += [f'{var}_{met}' for met in methods]
        elif var == 'sentinel':
            features_name += [f'{v}_{met}' for v in sentinel_variables for met in methods]
        elif var == "foret":
            features_name += [f'{foretint2str[v]}_{met}' for v in foret_variables for met in methods]
        elif var == 'dynamicWorld':
            features_name += [f'{v}_{met}' for v in dynamic_world_variables for met in methods]
        elif var == 'cosia':
            features_name += [f'{v}_{met}' for v in cosia_variables for met in methods]
        elif var == 'highway':
            features_name += [f'{osmnxint2str[v]}_{met}' for v in osmnx_variables for met in methods]
        elif var == 'Geo':
            features_name += geo_variables
        elif var == 'vigicrues':
            features_name += [f'{v}_{met}' for v in vigicrues_variables for met in methods]
        elif var == 'nappes':
            features_name += [f'{v}_{met}' for v in nappes_variables for met in methods]
        elif var == 'Historical':
            features_name += [f'{v}' for v in historical_variables]
        elif var == 'AutoRegressionReg':
            features_name += [f'AutoRegressionReg-{v}' for v in auto_regression_variable_reg]
        elif var == 'AutoRegressionBin':
            features_name +=  [f'AutoRegressionBin-{v}' for v in auto_regression_variable_bin]
        elif var == 'elevation':
            features_name += [f'{v}_{met}' for v in elevation_variables for met in methods]
        elif var == 'population':
            features_name += [f'{v}_{met}' for v in population_variabes for met in methods]
        elif var == 'region_class':
            features_name += [var]
        elif var in varying_time_variables_name:
            features_name += [var]
        elif var == 'temporal_prediction' or var == 'spatial_prediction':
            features_name += [var]
        elif var.find('frequencyratio') != -1:
            features_name += [var]
        elif var == 'cluster_encoder':
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
        elif var in landcover_variables:
            features_name.extend([var])
        elif var == 'sentinel':
            features_name.extend(sentinel_variables)
        elif var == "foret":
            features_name.extend([foretint2str[fv] for fv in foret_variables])
        elif var == 'dynamicWorld':
            features_name.extend(dynamic_world_variables)
        elif var == 'cosia':
            features_name.extend(cosia_variables)
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
            features_name.extend([f'AutoRegressionReg-{v}' for v in auto_regression_variable_reg])
        elif var == 'AutoRegressionBin':
            features_name.extend([f'AutoRegressionBin-{v}' for v in auto_regression_variable_bin])
        elif var == 'cluster_encoder':
            features_name.extend([var])
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
    if len(Xt.shape) == 1:
        scaler.fit(Xt.reshape(-1,1))
    else:
        scaler.fit(Xt)
    if len(array.shape) == 1:
        res = scaler.transform(array.reshape(-1,1))
    else:
        res = scaler.transform(array)
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
    coef = len(methods) if (scale == 'departement') or (scale > 0) else 1
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
    elif fet in landcover_variables:
        variables = [fet]
        maxi = coef
        methods = methods
        start = f'{fet}_mean'
    elif fet == 'dynamicWorld':
        variables = dynamic_world_variables
        maxi = coef * len(dynamic_world_variables)
        methods = methods
        start = f'{dynamic_world_variables[0]}_mean'
    elif fet == 'cosia':
        variables = cosia_variables
        maxi = coef * len(cosia_variables)
        methods = methods
        start = f'{cosia_variables[0]}_mean'
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
    elif fet == 'cluster_encoder':
        variables = [fet]
        maxi = coef
        methods = methods
        start = f'cluster_encoder'
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
        maxi = len(historical_variables)
        methods = methods
        start = f'{historical_variables[0]}_mean'
    elif fet == 'AutoRegressionReg':
        variables = auto_regression_variable_reg
        maxi = len(auto_regression_variable_reg)
        methods = ['raw']
        start = f'AutoRegressionReg-{auto_regression_variable_reg[0]}'
    elif fet == 'AutoRegressionBin':
        variables = auto_regression_variable_bin
        maxi = len(auto_regression_variable_bin)
        methods = ['raw']
        start = f'AutoRegressionBin-{auto_regression_variable_bin[0]}'
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

        save_object(features_importance, f'features_importance_{NbFeatures}.pkl', dir_output)
    else:
        logger.info(dir_output)
        features_importance = read_object(f'features_importance_{NbFeatures}.pkl', dir_output)

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
    oldWeights = Y[:, ids_columns.index('weight')]
    X[:, ids_columns.index('weight')] = 0
    Y[:, ids_columns.index('weight')] = 0
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

def check_class(influence, bin, predictor):
    influenceValues = influence[~np.isnan(influence)]
    binValus = influence[~np.isnan(influence)]
    classs = order_class(predictor, predictor.predict(influenceValues))
    predictor.log()
    cls = np.unique(classs)
    for cl in cls:
        mask = classs == cl
        logger.info(f'class {cl}, {np.nanmean(binValus[mask]), np.nanmean(influenceValues[mask])}')

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

            ax2.plot(y_dates, df[df['id'] == node]['AutoRegressionBin-B-1'].shift(1), label=f'{sinister}', c='r')
            ax2.set_ylabel(f'Number of {sinister}', color='r')
            ax2.set_ylim([0, 4])
            
        plt.legend()
        plt.tight_layout()
        plt.savefig(dir_log / 'kmeans_feature' / f'{fet}_{date_limit}.png')
        plt.close('all')

def apply_kmeans_class_on_target(df: pd.DataFrame, dir_break_point: Path, target : str, tresh : float, features_selected, new_val : int):
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

def add_fr_variables(df: pd.DataFrame, dir_break_point: Path, features_selected):
    dico_correlation = read_object('break_point_dict.pkl', dir_break_point)
    new_fet_fr = []
    for fet in features_selected:
        if fet.find('AutoRegression') != -1:
            continue
        logger.info(f'############## {fet} #################')
        values = np.copy(df[fet].values)
        values = values[~np.isnan(values)]
        predictor = read_object(f'{fet}.pkl', dir_break_point / fet)
        if predictor is None:
            continue
        if f'{fet}_frequencyratio' not in df.columns:
            df[f'{fet}_frequencyratio'] = np.nan
            predclass = order_class(predictor, predictor.predict(values))
            cls = np.unique(predclass)
            for c in cls:
                mask = predclass == c
                if int(c) not in dico_correlation[fet].keys():
                    continue
                values[mask] = dico_correlation[fet][int(c)]

            df.loc[df[~df[fet].isna()].index, f'{fet}_frequencyratio'] = values
        new_fet_fr.append(f'{fet}_frequencyratio')

    return df, new_fet_fr

def group_probability(group):
    # Calculer la probabilité que au moins un des événements se produise
    minus_prediction = np.ones(group['prediction'].values.shape[0]) - group['prediction'].values
    combined_prob = 1 - np.prod(minus_prediction)
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

    """dates = {
    'summer' : ('06-01', '08-31'),
    'winter' : ('12-01', '02-28'),
    'autumn' : ('09-01', '11-30'),
    'spring' : ('03-01', '05-31')
    }
    names = ['winter', 'spring', 'summer', 'autumn']
    """

    dates = {'medium' : ('02-01', '05-31'),
             'high' : ('06-01', '09-30'),
             'low' : ('10-01', '01-31'),
            }

    names = ['medium', 'high', 'low']
    
    for season in names:
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
    udates = np.unique(res[:, ids_columns.index('date')])

    for node in unodes:
        lon = np.unique(res[res[:, 0] == node][:, 1])[0]
        lat = np.unique(res[res[:, 0] == node][:, 2])[0]
        temporal_node_pred = graph_temporal._predict_node_with_position([[lon, lat]])
        for date in udates:
            mask = np.argwhere((res[:, 0] == node) & (res[:, ids_columns.index('date')] == date))[:, 0]
            mask_temporal = np.argwhere((Y_daily[:, 0] == temporal_node_pred) & (Y_daily[:, ids_columns.index('date')] == date))[:, 0]
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
            try:
                mlflow.log_metric(full_key, value)
            except:
                mlflow.log_metric(full_key, 0)
            # Gérer d'autres types si nécessaire (par exemple, des listes)
            #raise ValueError(f"Unsupported metric type: {type(value)} for key: {full_key}")

def get_existing_run(run_name):
    # Récupère tous les runs avec le nom spécifié
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=['0'],  # Spécifiez ici l'ID de l'expérience si nécessaire
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        run_view_type=mlflow.entities.ViewType.ALL
    )
    
    # Si un run est trouvé, le retourner
    if runs:
        return runs[0]  # retourne le premier run trouvé
    return None

def frequency_ratio(values: np.array, mask : np.array):
    FF_t = np.sum(values)
    Area_t = len(values)
    FF_i = np.sum(values[mask])

    if FF_t == 0 or Area_t == 0:
        return 0
    
    # Calculer Area_i (le nombre total de pixels pour la classe c)
    Area_i = mask.shape[0]
    
    # Calculer FireOcc et Area pour la classe c
    FireOcc = FF_i / FF_t
    Area = Area_i / Area_t
    if Area == 0:
        return 0
    
    # Calculer le ratio de fréquence (FR) pour la classe c
    FR = FireOcc / Area

    return round(FR, 3)

def remove_0_risk_pixel(dir_target, dir_bin, rasterImage, dept, on, val):
    binImage = read_object(dept+'binScale0.pkl', dir_bin)
    riskImage = read_object(dept+'Influence.pkl', dir_target)
    
    if on == 'bin':
        values = np.nansum(binImage, axis=2)
    else:
        values = np.nansum(riskImage, axis=2)

    zeros_risk = np.argwhere(values == 0)

    rasterImage[zeros_risk[:, 0], zeros_risk[:, 1]] = val

    return rasterImage

def get_features_selected_for_time_series(features, features_name, time_varying_features, nbfeatures):
    features_selected = []
    features_selected_str = []
    for fet in features:
        if len(features_selected_str) == nbfeatures:
            break
        if fet in features_name:
            features_selected.append(features_name.index(fet))
            if fet not in features_selected_str:
                features_selected_str.append(fet)
        elif fet in time_varying_features:
            vec = fet.split('_')
            new_fet = ''
            limit = len(vec) - 3
            for i, v in enumerate(vec):
                new_fet += v
                if i < limit:
                    new_fet += '_'
                else:
                    break
            if new_fet in air_variables or new_fet in calendar_variables or new_fet in historical_variables\
            or new_fet.find('frequencyratio') != -1 or new_fet.find('AutoRegressionReg') != -1 or new_fet.find('AutoRegressionBin') != -1:
                if new_fet not in features_selected_str:
                    features_selected_str.append(f'{new_fet}')
            else:
                if f'{new_fet}_mean' not in features_selected_str:
                    features_selected_str.append(f'{new_fet}_mean')
                
    return features_selected_str

def get_features_selected_for_time_series_for_2D(features, features_name, time_varying_features, nbfeatures):
    
    features_selected = []
    features_selected_str = []
    for fet in features:
        if len(features_selected_str) == nbfeatures:
            break
        if fet in features_name:
            features_selected.append(features_name.index(fet))
            if fet not in features_selected_str:
                features_selected_str.append(fet)
        elif fet in time_varying_features:
            vec = fet.split('_')
            new_fet = ''
            limit = len(vec) - 3
            for i, v in enumerate(vec):
                new_fet += v
                if i < limit:
                    new_fet += '_'
                else:
                    break
            if new_fet not in features_selected_str:
                features_selected_str.append(new_fet)
        elif fet.find('frequencyratio') != -1:
            vec = fet.split('_')
            new_fet = ''
            if vec[0] == 'calendar':
                limit = len(vec) - 2
            else:
                limit = len(vec) - 3
            for i, v in enumerate(vec):
                new_fet += v
                if i < limit:
                    new_fet += '_'
                else:
                    break
            if new_fet not in features_selected_str:
                features_selected_str.append(new_fet)
        else:
            vec = fet.split('_')
            new_fet = ''
            limit = len(vec) - 2
            for i, v in enumerate(vec):
                new_fet += v
                if i < limit:
                    new_fet += '_'
                else:
                    break
            if new_fet not in features_selected_str:
                features_selected_str.append(new_fet)

    return features_selected_str

def calculate_proportion_weights(df):
    res = np.empty((df.shape[0], 1))

    uclass = np.unique(df['class_risk'].values)

    mask_zero_class = np.argwhere(df['class_risk'].values == 0)

    for c in uclass:
        mask_class = np.argwhere(df['class_risk'].values == c)[:, 0]
        w =  mask_zero_class.shape[0] / mask_class.shape[0]
        res[mask_class, 0] = w

    zero_weight = np.argwhere(df['weight'].values == 0)[:, 0]
    res[zero_weight, 0] = 0

    return res

def calculate_class_weights(df):
    return df['weight']

def calculate_nbsinister(df):
    return df['nbsinister'] + 1

def calculate_normalize_weights(df, band):
    res = (df[band] - np.nanmean(df[band]) / np.nanstd(df[band]))
    min_value = np.min(res)
    return (res + abs(min_value)) + np.ones(df.shape[0])

def calculate_weight_proportion_on_zero_sinister(df):
    mask_sinister = np.argwhere(df['nbsinister'].values > 0).shape[0]
    mask_non_sinister = np.argwhere(df['nbsinister'].values == 0).shape[0]
    res = np.ones(df.shape[0])
    res = np.where(df['nbsinister'] == 0, 1, mask_non_sinister / mask_sinister)
    return res

def random_weights(df):
    rand_array = np.random.rand(df.shape[0]) * 10
    return rand_array

def calculate_weighs(weight_col, df):
    # Fonction auxiliaire pour ajuster les poids en fonction du nombre de sinistres
    def adjust_weights_for_zero_sinister(weights, df):
        # Mettre le poids à 1 pour les échantillons où le nombre de sinistres (colonne -2) est égal à 0
        zero_sinister_mask = df['nbsinister'] == 0
        weights[zero_sinister_mask] = 1
        return weights

    if weight_col == 'weight_proportion_on_zero_class':
        return calculate_proportion_weights(df)
    elif weight_col == 'weight_class':
        return calculate_class_weights(df)
    elif weight_col == 'weight_nbsinister':
        weights = calculate_nbsinister(df)
        return adjust_weights_for_zero_sinister(weights, df)
    elif weight_col == 'weight_one':
        return np.ones((df.shape[0], 1))
    elif weight_col == 'weight_normalize':
        return calculate_normalize_weights(df, 'risk')
    elif weight_col == 'weight_proportion_on_zero_sinister':
        weights = calculate_weight_proportion_on_zero_sinister(df)
        return adjust_weights_for_zero_sinister(weights, df)
    elif weight_col == 'weight_random':
        weights = random_weights(df)
        return adjust_weights_for_zero_sinister(weights, df)

    elif weight_col == 'weight_proportion_on_zero_class_nbsinister':
        weights = calculate_proportion_weights(df)
        return adjust_weights_for_zero_sinister(weights, df)
    elif weight_col == 'weight_class_nbsinister':
        weights = calculate_class_weights(df)
        return adjust_weights_for_zero_sinister(weights, df)
    elif weight_col == 'weight_one_nbsinister':
        return np.ones((df.shape[0], 1))
    elif weight_col == 'weight_nbsinister_nbsinister':
        weights = calculate_nbsinister(df)
        return adjust_weights_for_zero_sinister(weights, df)
    elif weight_col == 'weight_normalize_nbsinister':
        weights = calculate_normalize_weights(df, 'nbsinister')
        return adjust_weights_for_zero_sinister(weights, df)
    elif weight_col == 'weight_random_nbsinister':
        weights = random_weights(df)
        return adjust_weights_for_zero_sinister(weights, df)
    elif weight_col == 'weight_proportion_on_zero_sinister_nbsinister':
        weights = calculate_weight_proportion_on_zero_sinister(df)
        return adjust_weights_for_zero_sinister(weights, df)
    else:
        logger.info(f'Unknown value of weight {weight_col}')
        exit(1)

def calculate_days_until_next_event(id_array, date_array, event_array):
    """
    Calcule le nombre de jours jusqu'au prochain événement pour chaque région (id).
    Les jours où un événement se produit auront une valeur de 0, 
    et les jours sans événement auront le nombre de jours jusqu'au prochain événement.

    :param id_array: Tableau contenant les identifiants de région (id)
    :param date_array: Tableau contenant les dates (format datetime ou ordonné)
    :param event_array: Tableau contenant le nombre d'événements observés (nbsinister)
    :return: Un tableau des jours jusqu'au prochain événement
    """
    
    # Créer un DataFrame à partir des trois tableaux
    df = pd.DataFrame({
        'id': id_array,
        'date': date_array,
        'nbsinister': event_array
    })
    
    # Trier par id et date pour garantir que les événements sont dans l'ordre chronologique
    df = df.sort_values(by=['id', 'date'])
    
    # Initialiser une nouvelle colonne pour stocker le nombre de jours avant le prochain événement
    df['days_until_next_event'] = -1  # Initialiser avec -1
    
    # Parcourir chaque région (id)
    for reg_id in df['id'].unique():
        # Filtrer les données pour chaque région
        region_df = df[df['id'] == reg_id]
        
        # Trouver les indices où un événement se produit (nbsinister > 0)
        event_indices = region_df[region_df['nbsinister'] > 0].index
        
        # Parcourir chaque ligne du DataFrame de cette région
        for i in region_df.index:
            if df.loc[i, 'nbsinister'] > 0:
                df.loc[i, 'days_until_next_event'] = 0  # Si un événement se produit, mettre à 0
            else:
                # Trouver le prochain événement
                future_events = event_indices[event_indices > i]
                if len(future_events) > 0:
                    next_event_index = future_events[0]
                    next_event_date = df.loc[next_event_index, 'date']
                    current_date = df.loc[i, 'date']
                    df.loc[i, 'days_until_next_event'] = next_event_date - current_date  # Calculer les jours
    
    # Retourner la colonne des jours jusqu'au prochain événement
    return df['days_until_next_event'].values

def compare_lists(list1, list2):
    """
    Compares two lists and returns the elements that are not only in the second list.
    That is, elements that are either in both lists or only in the first list.

    :param list1: The first list.
    :param list2: The second list.
    :return: A list of elements that are in both lists or only in the first list.
    """
    set1 = set(list1)
    set2 = set(list2)
    # Elements that are only in the second list
    only_in_second = set2 - set1
    # All elements excluding those only in the second list
    result = (set1 | set2) - only_in_second
    return list(result)

def my_convolve(raster, mask, dimS, mode, dim=(90, 150, 3), semi=False, semi2=False):
    dimX, dimY, dimZ = dimS
    if dimX == 0 or dimY == 0 or dimZ == 0:
        return res
    if dim[-1] == 1:
        dimZ = 1
    else:
        dimZ = np.linspace(dim[-1]/2, 0, num=(dim[-1] // 2) + 1)[0] - np.linspace(dim[-1]//2, 0, num=(dim[-1] // 2) + 1)[1]
    if mode == "laplace":
        kernel = myFunctionDistanceDugrandCercle3D(dim, resolution_lon=dimX, resolution_lat=dimY, resolution_altitude=dimZ) + dimZ
        kernel = dimZ / kernel
    elif mode == 'inverse_laplace':
        kernel = myFunctionDistanceDugrandCercle3D(dim, resolution_lon=dimX, resolution_lat=dimY, resolution_altitude=dimZ) + dimZ
        kernel = dimZ / kernel
        kernel = (np.max(kernel) - kernel) + np.min(kernel)
    else:
        kernel = np.full(dim, 1/(dim[0]*dim[1]*dim[2]), dtype=float)

    if semi:
        if dim[2] != 1:
            kernel[:,:,:(dim[2]//2)] = 0.0
    if semi2:
        if dim[2] != 1:
            kernel[:,:,(dim[2]//2):] = 0.0
    res = convolve_fft(raster, kernel, normalize_kernel=False, mask=mask)
    return res