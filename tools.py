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
from sklearn.preprocessing import normalize
from scipy.signal import fftconvolve as scipy_fft_conv
import random
import geopandas as gpd
from torch import optim
from tqdm import tqdm
import torch
import math
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from ngboost import NGBClassifier, NGBRegressor
from sklearn.svm import SVR
import sys
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from scipy.interpolate import griddata
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata
from skimage import img_as_float
from category_encoders import TargetEncoder, CatBoostEncoder
import convertdate
from skimage import transform
from config import *
from weigh_predictor import *
random.seed(42)

def create_larger_scale_image(input, proba, bin):
    probaImageScale = np.full(proba.shape, np.nan)
    binImageScale = np.full(proba.shape, np.nan)
    
    clusterID = np.unique(input)
    for di in range(bin.shape[-1]):
        for id in clusterID:
            mask = np.argwhere(input == id)
            ones = np.ones(proba[mask[:,0], mask[:,1], di].shape)
            probaImageScale[mask[:,0], mask[:,1], di] = 1 - np.prod(ones - proba[mask[:,0], mask[:,1], di])

            binImageScale[mask[:,0], mask[:,1], di] = np.sum(bin[mask[:,0], mask[:,1], di])

    return None, binImageScale

def create_larger_scale_bin(input, bin):
    binImageScale = np.full(bin.shape, np.nan)
    
    clusterID = np.unique(input)
    for di in range(bin.shape[-1]):
        for id in clusterID:
            mask = np.argwhere(input == id)
            binImageScale[mask[:,0], mask[:,1], di] = np.sum(bin[mask[:,0], mask[:,1], di])

    return binImageScale

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

allDates = find_dates_between('2017-06-12', '2023-09-11')

def save_object(obj, filename: str, path : Path):
    check_and_create_path(path)
    with open(path / filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def save_object_torch(obj, filename : str, path : Path):
    check_and_create_path(path)
    torch.save(obj, path/filename)

def read_object(filename: str, path : Path):
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

def stat(c1, c2, clustered, osmnx):
    mask1 = clustered == c1
    mask2 = clustered == c2

    clusterMask1 = clustered.copy()
    clusterMask1[~mask1] = 0

    clusterMask2 = clustered.copy()
    clusterMask2[~mask2] = 0

    morpho1 = morphology.dilation(clusterMask1, morphology.disk(10))
    morpho2 = morphology.dilation(clusterMask2, morphology.disk(10))
    mask13 = (morpho1 == c1) & (morpho2 == c2)

    box = np.argwhere(mask13 == 1)

    if box.shape[0] == 0:
        return np.nan, np.nan, np.nan, np.nan
 
    indexXmin = np.argwhere(mask13 == 1)[:,0].min()
    indexXmax = np.argwhere(mask13 == 1)[:,0].max()
    indexYmin = np.argwhere(mask13 == 1)[:,1].min()
    indexYmax = np.argwhere(mask13 == 1)[:,1].max()

    #print(indexXmin, indexXmax , indexYmin, indexYmax)

    cropOsmnx = osmnx[indexXmin: indexXmax, indexYmin:indexYmax].astype(int)
    cropMask13 = mask13[indexXmin: indexXmax, indexYmin:indexYmax]

    road = cropOsmnx > 0
    road = road.astype(int)

    nonnanmask = np.argwhere(~np.isnan(road))
    density = influence_index(road, nonnanmask).astype(float)
    
    mini = np.nanmin(density[cropMask13])
    max = np.nanmax(density[cropMask13])
    mean = np.nanmean(density[cropMask13])
    std = np.nanstd(density[cropMask13])

    return mini, max, mean, std

def rasterization(ori, lats, longs, column, dir_output, outputname='ori', defVal = np.nan):
    ori.to_file(dir_output.as_posix() + outputname+'.geojson', driver="GeoJSON")
    
    # paths du geojson d'entree et du raster tif de sortie
    input_geojson = dir_output.as_posix() + outputname+'.geojson'
    output_raster = dir_output.as_posix() + outputname+'.tif'

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

def remove_nan_nodes(nodes : np.array, target : np.array) -> np.array:
    """
    Remove non nodes for array
    """
    mask = np.unique(np.argwhere(np.isnan(nodes))[:,0])
    if target is None:
        return np.copy(np.delete(nodes, mask, axis=0)), None
    return np.copy(np.delete(nodes, mask, axis=0)), np.copy(np.delete(target, mask, axis=0))

def add_k_temporal_node(k_days: int, nodes: np.array) -> np.array:

    print(f'Add {k_days} temporal nodes')

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
        #print(nei, date, graphId)
        return newSubNode

    newSubNode[maskNode, -1] = graphId
    newNei = np.unique(edges[1][np.argwhere((edges[0] == nei) &
                                             (np.isin(edges[1], np.unique(newSubNode[:,0])))
                                             )].reshape(-1))
    
    newNei = np.delete(newNei, np.argwhere(np.isin(newNei, nei)))

    #if oldGraphId[0] != -1:
    #    print(oldGraphId, graphId, maskNode, newSubNode[maskNode, 0], newSubNode[maskNode, 5],
    #          newSubNode[newSubNode[:, 5] == oldGraphId[0]].shape, newSubNode[newSubNode[:, 5] == graphId].shape, nei, newNei)

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

        print('Generate sub graph')
        newSubNode = np.full((nodes.shape[0], nodes.shape[1]), -1, dtype=nodes.dtype)
        newSubNode[:,:5] = nodes

        for i, node in enumerate(nodes):
            nei = graph.edges[1][np.argwhere(graph.edges[0] == node[0])].reshape(-1)
            if nei.shape[0] != 0:
                if maxNumber != 0:
                    newSubNode = add_new_random_nodes_from_nei(nei, newSubNode, node, minNumber, maxNumber, graph.nodes, nodes)

                #if newSubNode[i, -1] == -1:
                #    newSubNode[i, -1] = graphId
                #    graphId += 1

                #for ne in nei:
                #    newSubNode = assign_graphID(ne, newSubNode, node[4], newSubNode[i, -1], graph.edges)

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

    maskgraph = np.argwhere((X[:,4] == date) & (Y[:, -3] > 0))
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
            yts[:,-3] = 0
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
                nodes = x[(np.argwhere((x[:,4] != node[4]) & (x[:,0] != node[0])))]
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

    if Y is not None:
        return x, y, edges
    return x, edges

def concat_temporal_graph_into_time_series(array : np.array, ks : int, isY=False) -> np.array:
    uniqueNodes = np.unique(array[:,0])
    res = []

    for uNode in uniqueNodes:
        arrayNode = array[array[:,0] == uNode]
        ind = np.lexsort([arrayNode[:,4]])

        #ind = np.flip(ind)
        arrayNode = arrayNode[ind]

        if ks + 1 > arrayNode.shape[0]:
            return None
        res.append(arrayNode[:ks+1])

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

    maskgraph = np.argwhere((X[:,4] == date) & (Y[:, -3] > 0))
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
            yts[:,-3] = 0
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
    #print(x.shape)
    edges = []
    target = []
    src = []

    if spatialEdges.shape[1] != 0:
        for i, node in enumerate(x):
            # Spatial edges
            if spatialEdges.shape[1] != 0:
                spatialNodes = x[np.argwhere((x[:,4,-1] == node[4][-1]))][:,:, 0, 0]
                #spatialNodes = spatialNodes.reshape(-1, spatialNodes.shape[-1])
                spatial = spatialEdges[1][(np.isin(spatialEdges[1], spatialNodes[:,0])) & (spatialEdges[0] == node[0][0])]
                for sp in spatial:
                    src.append(i)
                    target.append(np.argwhere((x[:,4,-1] == node[4][-1]) & (x[:,0,0] == sp))[0][0])
                src.append(i)
                target.append(i)

        edges = np.row_stack((src, target)).astype(int)

    if Y is not None:
        return x, y, edges
    return x, edges

def load_x_from_pickle(x : np.array, shape : tuple,
                       path : Path,
                       Xtrain : np.array, Ytrain : np.array,
                       scaling : str,
                       pos_feature : dict,
                       pos_feature_2D : dict) -> np.array:

    doCalendar = 'Calendar' in features
    doGeo = 'Geo' in features
    doAir = 'air' in features

    if doCalendar:
        size_calendar = len(calendar_variables)
    if doGeo:
        size_geo = len(geo_variables)
    if doAir:
        size_air = len(air_variables)

    newx = np.empty((x.shape[0], shape[0], shape[1], shape[2], x.shape[-1]))
    for i, node in enumerate(x):
        id = int(node[0][0])
        dates = node[4,:].astype(int)
        for j, date in enumerate(dates):
            array2D = read_object(str(id)+'_'+str(date)+'.pkl', path)
            newx[i,:,:,:, j] = array2D[:,:,:]
            if doCalendar:
                for iv in range(size_calendar):
                    newx[i,:,:,pos_feature_2D['Calendar'] + iv,j] = node[pos_feature['Calendar'] + iv, j]
            if doGeo:
                for iv in range(size_geo):
                    newx[i,:,:,pos_feature_2D['Geo'] + iv,j] = node[pos_feature['Geo'] + iv, j]
            if doAir:
                for iv in range(size_air):
                    newx[i,:,:,pos_feature_2D['air'] + iv,j] = node[pos_feature['air'] + iv, j]

    if scaling == 'MinMax':
        scaler = min_max_scaler
    elif scaling == 'z-score':
        scaler = standart_scaler
    elif scaling == "robust":
        scaler = robust_scaler
    else:
        ValueError('Unknow')
        exit(1)

    if 'sentinel' in features:
        size_sent = len(sentinel_variables)

    for feature in features:
        if feature == 'Calendar' or feature == 'Geo' or features == 'air':
            continue
        if feature == 'sentinel':
            for i in range(size_sent):
                index2D = pos_feature_2D[feature]
                index1D = pos_feature[feature] + (4 * i)
                newx[:,:,:,index2D,:] = scaler(newx[:,:,:,index2D + i,:], Xtrain[:, index1D + 2], concat=False)
        else:
            index2D = pos_feature_2D[feature]
            index1D = pos_feature[feature] + 2
            newx[:,:,:,index2D,:] = scaler(newx[:,:,:,index2D,:], Xtrain[:, index1D], concat=False)

    return newx

def realVspredict(ypred, y, dir_output, name):
    check_and_create_path(dir_output)
    ypred = ypred
    ytrue = y[:,-1]
    _, ax = plt.subplots(1, figsize=(20,10))
    ax.plot(ypred, color='red', label='predict')
    ax.plot(ytrue, color='blue', label='real', alpha=0.5)
    x = np.argwhere( y[:,-2] > 0)
    #ax.scatter(x, ytrue[x], color='black', label='fire', alpha=0.5)
    plt.legend()
    plt.savefig(dir_output/name)

def realVspredict2d(ypred : np.array,
                    ytrue : np.array,
                    dir_output : Path,
                    geo : gpd.GeoDataFrame):
    
    ytrue = ytrue[ytrue[:,4] >= allDates.index('2023-01-01')]
    if ytrue.shape[0] == 0:
        return
    
    check_and_create_path(dir_output)
    uniqueDates = np.sort(np.unique(ytrue[:,4])).astype(int)
    mean = []
    for date in uniqueDates:
        #print(allDates[date])
        mask = np.argwhere(ytrue[:,4] == date)
        ypredDate = ypred[mask]
        ybinDate = ytrue[mask,-2]
        ytrueDate = ytrue[mask,-1]
        dfff = geo.copy(deep=True)
        dfff['id'] = ytrue[mask,0]
        dfff['longitude'] = ytrue[mask,1]
        dfff['latitude'] = ytrue[mask,2]
        dfff['departement'] = ytrue[mask,3]
        dfff['date'] = ytrue[mask,4]
        dfff['gt'] = ytrueDate
        dfff['nbFire'] = ybinDate
        dfff['prediction'] = ypredDate
        dfff = gpd.GeoDataFrame(dfff, geometry=gpd.points_from_xy(dfff.longitude, dfff.latitude))
        name = allDates[date]+'.geojson'
        mean.append(dfff)
        dfff.to_file(dir_output / name, driver='GeoJSON')
    
    mean = pd.concat(mean).reset_index(drop=True)
    name = 'mean.geojson'
    mean.to_file(dir_output / name, driver='GeoJSON')

def train(trainLoader, valLoader, PATIENCE_CNT : int,
          CHECKPOINT: int,
          lr : float,
          epochs : int,
          criterion,
          model : torch.nn.Module,
          dir_output : Path) -> None:
        """
        Train neural network model
        PATIENCE_CNT : early stopping count
        CHECKPOINT : print last train/val loss
        lr : learning rate
        epochs : number of epochs for training
        criterion : loss function
        """
        assert trainLoader is not None and valLoader is not None

        check_and_create_path(dir_output)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        BEST_VAL_LOSS = math.inf
        BEST_MODEL_PARAMS = None
        patience_cnt = 0
        for epoch in tqdm(range(epochs)):
            model.train()
            for i, data in enumerate(trainLoader, 0):
 
                inputs, labels, edges = data
                
                target = labels[:,-1]
                weights = labels[:,-3]
                
                output = model(inputs, edges)
                target = torch.masked_select(target, weights.gt(0))
                output = torch.masked_select(output, weights.gt(0))
                weights = torch.masked_select(weights, weights.gt(0))

                loss = criterion(output, target, weights)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % CHECKPOINT == 0:
                print(f'epochs {epoch}, Train loss {loss.item()}')
                save_object_torch(model.state_dict(), str(epoch)+'.pt', dir_output)

            with torch.no_grad():
                model.eval()
                for i, data in enumerate(valLoader, 0):
                    inputs, labels, edges = data

                    output = model(inputs, edges)
                    
                    target = labels[:,-1]
                    weights = labels[:,-3]

                    target = torch.masked_select(target, weights.gt(0))
                    output = torch.masked_select(output, weights.gt(0))
                    weights = torch.masked_select(weights, weights.gt(0))

                    loss = criterion(output, target, weights)

                    if loss.item() < BEST_VAL_LOSS:
                        BEST_VAL_LOSS = loss.item()
                        BEST_MODEL_PARAMS = model.state_dict()
                        patience_cnt = 0
                    else:
                        patience_cnt += 1
                        if patience_cnt >= PATIENCE_CNT:
                            print(f'Loss has not increased for {patience_cnt} epochs. Last best val loss {BEST_VAL_LOSS}, current val loss {loss.item()}')
                            save_object_torch(model.state_dict(), 'last.pt', dir_output)
                            save_object_torch(BEST_MODEL_PARAMS, 'best.pt', dir_output)
                            return
                        
            if epoch % CHECKPOINT == 0:
                print(f'epochs {epoch}, Best val loss {BEST_VAL_LOSS}')


def config_xgboost(device, binary):
    params = {
        'verbosity':0,
        'early_stopping_rounds':15,
        'learning_rate' :0.01,
        'min_child_weight' : 1.0,
        'max_depth' : 6,
        'max_delta_step' : 1.0,
        'subsample' : 0.5,
        'colsample_bytree' : 0.7,
        'colsample_bylevel': 0.6,
        'reg_lambda' : 1.7,
        'reg_alpha' : 0.7,
        #'gamma' : 2.5, 
        'n_estimators' : 10000,
        'random_state': 42,
        'tree_method':'hist',
        }
        
    
    if device == 'cuda':
        params['device']='cuda'

    if not binary:
        return XGBRegressor(**params,
                            #objective = 'reg:quantileerror'
                            objective = 'reg:squarederror'
                            )
    else:
        return XGBClassifier(**params,
                            #objective = 'reg:quantileerror'
                            objective = 'binary:logistic'
                            )

def config_lightGBM(device, binary):
    params = {'verbosity':-1,
        #'num_leaves':64,
        'learning_rate':0.01,
        'early_stopping_rounds': 15,
        'bagging_fraction':0.7,
        'colsample_bytree':0.6,
        'max_depth' : 4,
        'num_leaves' : 2**4,
        'reg_lambda' : 1,
        'reg_alpha' : 0.27,
        #'gamma' : 2.5,
        'num_iterations' :10000,
        'random_state':42
        }

    if device == 'cuda':
        params['device'] = "gpu"

    if not binary:
        params['objective'] = 'root_mean_squared_error',
        return LGBMRegressor(**params)
    else:
        params['objective'] = 'binary',
        return LGBMClassifier(**params)


def config_ngboost(binary):
    params  = {
        'natural_gradient':True,
        'n_estimators':1000,
        'learning_rate':0.01,
        'minibatch_frac':0.7,
        'col_sample':0.6,
        'verbose':False,
        'verbose_eval':100,
        'tol':1e-4,
    }

    if not binary:
        Base = XGBRegressor(max_depth= 6, n_estimators= 300, verbosity= 1, objective='reg:squarederror',
                            booster= 'gbtree', tree_method= 'hist', n_jobs=-1, learning_rate= 0.05, gamma= 0.15,
                            reg_alpha= 0.20,
                            reg_lambda= 0.50, random_state=42, device='cuda')
        
        """Base = LGBMRegressor(device='gpu', random_state=42, reg_lambda=0.5, reg_alpha=0.2, )"""
        return NGBRegressor(**params)
    else:
        return NGBClassifier(**params)

def train_sklearn_api_model(trainSet, valSet,
          dir_output : Path,
          device : str,
          binary : bool,
          pos_feature : dict,
          encoding : str,
          scaling: str):
    
    models = []
    
    ################# xgboost ##################

    models.append(('xgboost', config_xgboost(device, binary)))

    ################ lightgbm ##################

    models.append(('lightgbm', config_lightGBM(device, binary)))

    ################ ngboost ###################

    models.append(('ngboost', config_ngboost(binary)))

    ############### SVM ########################
    
    # TO DO

    ############## fit ########################

    Xtrain = trainSet[0]
    Ytrain = trainSet[1]

    Xval = valSet[0]
    Yval = valSet[1]

    #################################################################################################
    #                                                                                               #
    #                                     PREPROCESS                                                #
    #                   Like the preprocess function -> TO DO : merge into one                      #
    #                                                                                               #
    #                                                                                               #
    #################################################################################################

        # Define the encoder
    """if encoding == 'Target':
        encoder = target_encoding
        xt = Xtrain
        yt = Ytrain
    elif encoding == 'Catboost':
        encoder = catboost_encoding
        if Yval is not None:
            xt = np.concatenate((Xtrain, Xval))
            yt = np.concatenate((Ytrain, Yval))
            # Sort by date
            ind = np.lexsort([yt[:,4]])
            xt = xt[ind]
            yt = yt[ind]
        else:
            xt = Xtrain
            yt = Ytrain

    # Encode Calendar feature
    if 'Calendar' in features:
        Xtrain = encoder(pos_feature, Xtrain, xt, yt, 'Calendar', len(calendar_variables))
        Xval = encoder(pos_feature, Xval, xt, yt, 'Calendar', len(calendar_variables))

    # Encode Geo feature
    if 'Geo' in features:
        Xtrain = encoder(pos_feature, Xtrain, xt, yt, 'Geo', len(calendar_variables))
        Xval = encoder(pos_feature, Xval, xt, yt,'Geo', len(calendar_variables))

    if 'foret' in features:
        # to do
        pass

    if 'landcover' in features:
        # to do
        pass"""

    # Define the scale method
    if scaling == 'MinMax':
        scaler = min_max_scaler
    elif scaling == 'z-score':
        scaler = standart_scaler
    elif scaling == "robust":
        scaler = robust_scaler
    else:
        ValueError('Unknow')
        exit(1)
    
    # Scale Features
    for featureband in range(6, Xtrain.shape[1]):
        Xval[:,featureband] = scaler(Xval[:,featureband], Xtrain[:, featureband], concat=False)
        Xtrain[:,featureband] = scaler(Xtrain[:,featureband], Xtrain[:, featureband], concat=False)

    print(f'Check {scaling} standardisation Train : {np.nanmax(Xtrain[:,6:])}, {np.nanmin(Xtrain[:,6:])}')
    print(f'Check {scaling} standardisation Val : {np.nanmax(Xval[:,6:])}, {np.nanmin(Xval[:,6:])}')

    print([np.argwhere(Xval[:,6:] == np.nanmax(Xval[:,6:]))])

    Xval = Xval[:, [  6,   7,   8,  10,  11,  12,  13,  14,  15,  16,  17,  22,  23,
        24,  25,  27,  28,  30,  31,  32,  34,  35,  36,  37,  38,  39,
        40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,
        53,  54,  55,  56,  57,  58,  59,  60,  62,  63,  64,  66,  67,
        68,  69,  70,  71,  72,  74,  75,  76,  77,  78,  79,  80,  82,
        85,  86,  87,  88,  94,  95,  96, 100, 102, 104, 105, 108, 113,
       114, 116, 117, 119, 121, 132, 133, 136, 137, 138, 139, 140, 146,
       147, 157, 158, 159, 160, 162, 163, 164, 165]]
    
    Xtrain = Xtrain[:, [  6,   7,   8,  10,  11,  12,  13,  14,  15,  16,  17,  22,  23,
        24,  25,  27,  28,  30,  31,  32,  34,  35,  36,  37,  38,  39,
        40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,
        53,  54,  55,  56,  57,  58,  59,  60,  62,  63,  64,  66,  67,
        68,  69,  70,  71,  72,  74,  75,  76,  77,  78,  79,  80,  82,
        85,  86,  87,  88,  94,  95,  96, 100, 102, 104, 105, 108, 113,
       114, 116, 117, 119, 121, 132, 133, 136, 137, 138, 139, 140, 146,
       147, 157, 158, 159, 160, 162, 163, 164, 165]]
    
    Xtrain = Xtrain[Ytrain[:,-3] > 0]
    Ytrain = Ytrain[Ytrain[:,-3] > 0]

    if not binary:
        Yw = Ytrain[:,-3]
    else:
        Yw = np.ones(Ytrain.shape[0])

    Xval = Xval[Yval[:,-3] > 0]
    Yval = Yval[Yval[:,-3] > 0]

    if not binary:
        Ytrain = Ytrain[:,-1]
        Yval = Yval[:,-1]
    else:
        Ytrain = Ytrain[:,-2] > 0
        Yval = Yval[:,-2] > 0

    for name, model in models:

        if name == 'xgboost':
            fitparams={
            'eval_set':[(Xtrain, Ytrain), (Xval, Yval)],
            'sample_weight' : Yw,
            'verbose' : False
            }

        if name == 'lightgbm':
            fitparams={
            'eval_set':[(Xtrain, Ytrain), (Xval, Yval)],
            'sample_weight' : Yw,
            }

        if name == 'ngboost':
            fitparams={
            'early_stopping_rounds':15,
            'sample_weight' : Yw,
            'X_val':Xval,
            'Y_val':Yval,
            }

        if binary:
            name = name+'_bin'

        print(f'Fitting model {name}')
        model.fit(Xtrain, Ytrain, **fitparams)
        check_and_create_path(dir_output / name)
        outputname = name + '.pkl'
        save_object(model, outputname, Path(dir_output / name))
        
def weighted_rmse_loss(input, target, weights = None):
    if weights is None:
        return torch.sqrt(((input - target) ** 2).mean())
    return torch.sqrt((weights * (input - target) ** 2).sum() / weights.sum())

def add_metrics(methods : list, i : int,
                ypred : torch.tensor,
                ytrue : torch.tensor,
                testDepartement : list,
                isBin : bool,
                scale : int,
                model : str) -> dict:
    res = {}
    for name, met, target in methods:

        if target == 'proba':
            mett = met(ypred, ytrue[:,-1], ytrue[:,-3])
            if torch.is_tensor(mett):
                mett = mett.detach().cpu().numpy()

            res[name] = mett
        elif target == 'bin':
            mett = met(ytrue[:,-2] > 0, ypred, ytrue[:,-3])
            if torch.is_tensor(mett):
                mett = mett.detach().cpu().numpy()
            res[name] = mett

            mett = met(ytrue[:,-2] > 0, ypred, None)
            if torch.is_tensor(mett):
                mett = mett.detach().cpu().numpy()
            res[name+'no_weighted'] = mett
        
        elif target == 'class':
            mett = met(ypred, ytrue, testDepartement, isBin, scale, model, weights=ytrue[:,-3])
            res[name] = mett

        print(name)

    return res

def standard_deviation(inputs, target, weights = None):
    return torch.std(inputs)

def log_factorial(x):
  return torch.lgamma(x + 1)

def factorial(x):
  return torch.exp(log_factorial(x))

def poisson_loss(inputs, target, weights = None):
    if weights is None:
        return torch.exp(inputs) - target*torch.log(inputs) + torch.log(factorial(target))
    else:
        return ((torch.exp(inputs) - target*torch.log(inputs) + torch.log(factorial(target))) * weights).sum() / weights.sum()
    
def quantile_prediction_error(ytrue : torch.tensor, ypred : torch.tensor, weights = None):
    maxi = torch.max(ypred).detach().cpu().numpy()
    ytrue = ytrue.detach().cpu().numpy().astype(int)
    ypred = ypred.detach().cpu().numpy()
    if maxi < 1.0:
        maxi = 1.0
    quantiles = [(0.0,0.1),
                 (0.1, 0.2),
                 (0.2, 0.3),
                 (0.3, 0.4),
                 (0.4, 0.5),
                 (0.5, 0.6),
                 (0.6, 0.7),
                 (0.7, 0.8),
                 (0.8, 0.9),
                 (0.9, 1.0)]
    
    error = []
    for (minB, maxB) in quantiles:
        pred_quantile = ypred[(ypred >= minB * maxi) & (ypred < maxB * maxi)]
        nf = ytrue[(ypred >= minB * maxi) & (ypred < maxB * maxi)]
        if pred_quantile.shape[0] != 0:
            pred_quantile = np.mean(pred_quantile)
            number_of_fire = np.mean(nf)
            #print((minB, maxB), pred_quantile, number_of_fire)
            error.append(abs(((minB + maxB) / 2) - number_of_fire))
    if len(error) == 0:
        return math.inf

    return np.mean(error)

def my_f1_score(ytrue : torch.tensor, ypred : torch.tensor, weights : torch.tensor = None):
    bounds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    maxi = torch.max(ypred).detach().cpu().numpy()
    ytrueNumpy = ytrue.detach().cpu().numpy()
    ypredNumpy = ypred.detach().cpu().numpy()
    if weights is not None:
        weightsNumpy = weights.detach().cpu().numpy()
    else:
        weightsNumpy = np.ones(ytrueNumpy.shape[0])

    weightsNumpy = (weightsNumpy * ytrueNumpy) + 1 ## make all non fire point weights at 1 and keep the seasonality of each fire points

    bestScore = 0.0
    prec = 0.0
    rec = 0.0
    bestBound = 0.0
    for bound in bounds:
        yBinPred = (ypredNumpy > bound * maxi).astype(int)
        f1 = f1_score(ytrueNumpy, yBinPred, sample_weight=weightsNumpy)
        if f1 > bestScore:
            bestScore = f1
            bestBound =  bound
            prec = precision_score(ytrueNumpy, yBinPred, sample_weight=weightsNumpy)
            rec = recall_score(ytrueNumpy, yBinPred, sample_weight=weightsNumpy)

    return (bestScore, prec, rec, bestBound)

def class_risk(ypred, ytrue, departements : list, isBin : bool, scale : int, modelName: str, weights = None) -> dict:
    if torch.is_tensor(ypred):
        ypred = ypred.detach().cpu().numpy().astype(float)
    if torch.is_tensor(ytrue):
        ytrue = ytrue.detach().cpu().numpy().astype(float)
    
    dir_predictor = root / 'Model' / 'HexagonalScale' / 'GNN' / 'influenceClustering'
    ydep = ytrue[:,3]
    
    res = {}
    for nameDep in departements:
        mask = np.argwhere(ydep == name2int[nameDep])
        if mask.shape[0] == 0:
            continue
        if not isBin:
            predictor = read_object(nameDep+'Predictor.pkl', dir_predictor)
        else:
            predictor = Predictor(5)
            predictor.fit(ypred[mask])
            save_object(predictor, nameDep+'Predictor'+modelName+str(scale)+'.pkl', dir_predictor)
        
        yclass = predictor.predict(ypred[mask])
        uniqueClass = np.unique(yclass)
        res[nameDep] = {}
        for c in uniqueClass:
            classIndex = np.argwhere(yclass == c)
            classPred = ypred[mask][classIndex]
            classTrue = ytrue[:,-1][mask][classIndex]
            error = abs(classPred - classTrue)
            classBin = ytrue[:,-2][mask][classIndex] > 0
            meanF = round(np.mean(classBin), 3)
            meanP = round(np.mean(classPred), 3)
            meanT = round(np.mean(classTrue), 3)
            error = round(np.mean(error))
            res[nameDep][c] = (predictor.get_centroid(c)[0], meanP, meanT, error, meanF)
    return res

def class_accuracy(ypred, ytrue, departements : list, isBin : bool, scale : int, modelName: str, weights = None) -> dict:
    if torch.is_tensor(ypred):
        ypred = ypred.detach().cpu().numpy().astype(float)
    if torch.is_tensor(ytrue):
        ytrue = ytrue.detach().cpu().numpy().astype(float)

    if weights is not None:
        weightsNumpy = weights.detach().cpu().numpy()
    else:
        weightsNumpy = np.ones(ytrue.shape[0])
    
    dir_predictor = root / 'Model' / 'HexagonalScale' / 'GNN' / 'influenceClustering'
    ydep = ytrue[:,3]
    res = {}
    for nameDep in departements:
        mask = np.argwhere(ydep == name2int[nameDep])
        if mask.shape[0] == 0:
            continue
        if not isBin:
            predictor = read_object(nameDep+'Predictor.pkl', dir_predictor)
        else:
            predictor = Predictor(5)
            predictor.fit(ypred[mask])
            save_object(predictor, nameDep+'Predictor'+modelName+str(scale)+'.pkl', dir_predictor)
        
        ypredclass = predictor.predict(ypred[mask])
        ytrueclass = predictor.predict(ytrue[mask, -1])
        yweights = weightsNumpy[mask].reshape(-1)

        res[nameDep] = accuracy_score(ytrueclass, ypredclass, sample_weight=yweights)
            
    return res

def create_pos_feature(graph, shape, features):

    pos_feature = {}

    newShape = shape
    if graph.scale > 0:
        for var in features:
            print(newShape, var)
            pos_feature[var] = newShape
            if var == 'Calendar':
                newShape += len(calendar_variables)
            elif var == 'air':
                newShape += len(air_variables)
            #elif var == 'landcover':
            #    newShape += len(landcover_variables)
            elif var == 'sentinel':
                newShape += 4 * len(sentinel_variables)
            #elif var == 'foret':
            #    newShape += len(foret_variables)
            elif var == 'Geo':
                newShape += len(geo_variables)
            else:
                newShape += 4
    else:
        for var in features:
            print(newShape, var)
            pos_feature[var] = newShape
            if var == 'Calendar':
                newShape += len(calendar_variables)
            #elif var == 'landcover':
            #    newShape += len(landcover_variables)
            #elif var == 'foret':
            #    newShape += len(foret_variables)
            elif var == 'sentinel':
                newShape += len(sentinel_variables)
            elif var == 'Geo':
                newShape += len(geo_variables)
            else:
                newShape += 1
    
    return pos_feature, newShape

def create_pos_features_2D(shape, features):

    pos_feature = {}

    newShape = shape
    for var in features:
        print(newShape, var)
        pos_feature[var] = newShape
        if var == 'Calendar':
            newShape += len(calendar_variables)
        elif var == 'air':
            newShape += len(air_variables)
        elif var == 'sentinel':
            newShape += len(sentinel_variables)
        elif var == 'landcover':
            newShape += 1
        elif var == 'foret':
                newShape += 1
        elif var == 'Geo':
            newShape += len(geo_variables)
        else:
            newShape += 1

    return pos_feature, newShape

def min_max_scaler(array : np.array, arrayTrain: np.array, concat : bool) -> np.array:
    if concat:
        Xt = np.concatenate((array, arrayTrain))
    else:
        Xt = arrayTrain
    scaler = MinMaxScaler()
    scaler.fit(Xt.reshape(-1,1))
    res = scaler.transform(array.reshape(-1,1))
    return res.reshape(array.shape)

def standart_scaler(array: np.array, arrayTrain: np.array, concat : bool) -> np.array:
    if concat:
        Xt = np.concatenate((array, arrayTrain))
    else:
        Xt = arrayTrain
    scaler = StandardScaler()
    scaler.fit(Xt.reshape(-1,1))
    res = scaler.transform(array.reshape(-1,1))
    return res.reshape(array.shape)

def robust_scaler(array : np.array, arrayTrain: np.array, concat : bool) -> np.array:
    if concat:
        Xt = np.concatenate((array, arrayTrain))
    else:
        Xt = arrayTrain
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

def target_encoding(pos_feature, Xset, Xtrain, Ytrain, variableType, size):
    print(f'Target Encoding')
    enc = TargetEncoder(cols=np.arange(pos_feature[variableType], pos_feature[variableType] + size)).fit(Xtrain, Ytrain[:, -1])
    Xset = enc.transform(Xset).values
    return Xset

def catboost_encoding(pos_feature, Xset, Xtrain, Ytrain, variableType, size):
    print(f'Catboost Encoding')
    enc = CatBoostEncoder(cols=np.arange(pos_feature[variableType], pos_feature[variableType] + size)).fit(Xtrain, Ytrain[:, -1])
    Xset = enc.transform(Xset).values
    return Xset