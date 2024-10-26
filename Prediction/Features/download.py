import requests
from shapely import unary_union, set_precision
import pandas as pd
import osmnx as ox
from sympy import subsets
import wget
import subprocess
from itertools import chain
import re
import geopandas as gpd
import os
import py7zr
import glob
import zipfile
import geojson
from tools import *

def myround(x):
    return (round(x[0], 3), round(x[1], 3))

def get_latitude(x):
    return x[1]

def get_longitude(x):
    return x[0]

def unzip_7z(file_path, destination_folder) -> None:
    """
    Décompresse un fichier .7z dans le dossier de destination spécifié.

    :param file_path: Le chemin vers le fichier .7z à décompresser.
    :param destination_folder: Le dossier où les fichiers décompressés seront enregistrés.
    """
    # Créer le dossier de destination s'il n'existe pas
    os.makedirs(destination_folder, exist_ok=True)

    # Ouvrir le fichier 7z et extraire son contenu
    with py7zr.SevenZipFile(file_path, mode='r') as archive:
        archive.extractall(path=destination_folder)

def unzip_file(file_path, destination_folder) -> None:
    """
    Décompresse un fichier .zip dans le dossier de destination spécifié.

    :param file_path: Le chemin vers le fichier .zip à décompresser.
    :param destination_folder: Le dossier où les fichiers décompressés seront enregistrés.
    """
    # Créer le dossier de destination s'il n'existe pas
    os.makedirs(destination_folder, exist_ok=True)

    # Ouvrir le fichier .zip et extraire son contenu
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

# Fonction pour télécharger les données de population
def download_population(path, geo, dir_output) -> None:
    check_and_create_path(dir_output)
    print('DONWLOADING POPULATION')
    if not (path / 'kontur_population_FR_20231101.gpkg').is_file():
        print('ERROR : Add population donwload')
        exit(1)
    file = gpd.read_file(path / 'kontur_population_FR_20231101.gpkg')
    file.to_crs('EPSG:4326', inplace=True)
    file['isdep'] = file['geometry'].apply(lambda x : geo.contains(x))
    geo_pop = file[file['isdep']]
    geo_pop['latitude'] = geo_pop['geometry'].apply(lambda x : float(x.centroid.y))
    geo_pop['longitude'] = geo_pop['geometry'].apply(lambda x : float(x.centroid.x))
    geo_pop.to_file(dir_output / 'population.geojson')
    geo_pop.to_csv(dir_output / 'population.csv', index=False)

def download_tourbiere(path, geo, dir_output):
    check_and_create_path(dir_output)
    tourbiere_france = gpd.read_file(path / 'ZonesTourbeuses.shp').to_crs(crs=4326)
    tourbiere = gpd.overlay(tourbiere_france, geo)
    tourbiere['classe'] = 0
    tourbiere.loc[tourbiere['37formvegd'] == 'Bas-marais et marais de transition'] = 1
    tourbiere.loc[tourbiere['37formvegd'] == 'Tourbières hautes'] = 2
    tourbiere.to_file(dir_output / 'ZonesTourbeuses.geojson', driver='GeoJSON')

def create_elevation_geojson(path, bounds, dir_output):
    path = path.as_posix()
    files = glob.glob(path+'/COURBE/**/*.shp', recursive=True)
    gdf = []
    for f in files:
        try:
            gdft = gpd.read_file(f).to_crs(crs=4326)
        except:
            continue
        if "ALTITUDE" not in gdft.columns:
            continue
        gdft['points'] = gdft.apply(lambda x: [y for y in x['geometry'].coords], axis=1)
        gdft['altitude'] = gdft.apply(lambda x: [x['ALTITUDE'] for y in x['geometry'].coords], axis=1)
        points = pd.Series(list(chain.from_iterable(gdft['points'])))
        altitude = pd.Series(list(chain.from_iterable(gdft['altitude'])))
        gdft = pd.DataFrame({'altitude': altitude, 'geometry': points})
        gdft['points'] = gdft['geometry'].apply(myround)
        gb = gdft.groupby('points')
        mean = gb['altitude'].mean().reset_index()
        mean['latitude'] = mean['points'].apply(get_latitude)
        mean['longitude'] = mean['points'].apply(get_longitude)
        mean = gpd.GeoDataFrame(mean, geometry=gpd.points_from_xy(mean.longitude, mean.latitude))
        mean = mean[(mean['latitude'] >= bounds['miny'].values[0]) & (mean['latitude'] <= bounds['maxy'].values[0]) & \
                    (mean['longitude'] >= bounds['minx'].values[0]) & (mean['longitude'] <= bounds['maxx'].values[0])]
        if mean.empty:
            continue
        gdf.append(mean)

    gdf = pd.concat(gdf)
    gdf.drop('points', axis=1, inplace=True)
    gdf.to_csv(dir_output / 'elevation.csv', index=False)
    gpf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(gdf.longitude, gdf.latitude))
    gpf.to_file(dir_output / 'elevation.geojson')

def create_cosia_geojson(path, bounds, dir_output):
    path = path.as_posix()
    files = glob.glob(path+'/*.gpkg', recursive=True)
    gdf = []
    leni = len(files)
    for i, f in enumerate(files):
        try:
            gfile = gpd.read_file(f).to_crs(crs=4326)
        except Exception as e:
            print(e)
            continue
        print(f'{i}/{leni}')
        gfile['geometry'] = set_precision(gfile.geometry, grid_size=0.001, mode='pointwise')
        gfile.drop_duplicates(subset=['geometry'], inplace=True, keep='first')
        gfile = gfile.copy(deep=True)
        gfile = gfile[~gfile.geometry.is_empty]
        gdf.append(gfile)

    gdf = pd.concat(gdf)
    leni = len(gdf)
    gdf = gdf[gdf['geometry'] != None]
    #gdf['is_valid'] = gdf['geometry'].apply(lambda x : x.is_valid)
    #gdf = gdf[gdf['is_valid']]
    print(f'{leni} - > {len(gdf)}')

    gdf.to_file(dir_output / 'cosia.geojson', driver='GeoJSON')

# Fonction pour télécharger les données d'élévation
def download_elevation(code_dept: int, geo, dir_output: str) -> None:
    """
    Télécharge les données d'élévation depuis l'URL spécifiée et les enregistre dans le chemin de destination.

    :param url: L'URL à partir de laquelle télécharger les données d'élévation.
    :param destination_path: Le chemin de destination où enregistrer les données téléchargées.
    """
    check_and_create_path(dir_output)
    print('DONWLOADING ELEVATION')
    if not (dir_output / f'{dir_output}/COURBE_1-0__SHP_LAMB93_D0{code_dept}_2014-04-01.7z').is_file():
        url = f'https://data.geopf.fr/telechargement/download/COURBES/COURBE_1-0__SHP_LAMB93_D0{code_dept}_2021-01-01/COURBE_1-0__SHP_LAMB93_D0{code_dept}_2021-01-01.7z'
        subprocess.run(['wget', url, '-O', f'{dir_output}/COURBE_1-0__SHP_LAMB93_D0{code_dept}_2021-04-01.7z'], check=True)
        unzip_7z(dir_output / f'COURBE_1-0__SHP_LAMB93_D0{code_dept}_2021-04-01.7z', dir_output)
    
    create_elevation_geojson(dir_output / f'COURBE_1-0__SHP_LAMB93_D0{code_dept}_2021-01-01', geo.bounds, dir_output)

# Fonction pour télécharger les données de cosia
def download_cosia(code_dept: int, geo, dir_output: str) -> None:
    """
    Télécharge les données d'élévation depuis l'URL spécifiée et les enregistre dans le chemin de destination.

    :param url: L'URL à partir de laquelle télécharger les données d'élévation.
    :param destination_path: Le chemin de destination où enregistrer les données téléchargées.
    """

    check_and_create_path(dir_output)
    for y in np.arange(2019, 2024):
        if (dir_output / f'{dir_output}/CoSIA_D0{code_dept}_{y}.zip').is_file():
            unzip_file(dir_output / f'CoSIA_D0{code_dept}_{y}.zip', dir_output)
        if (dir_output / f'{dir_output}/CoSIA_D0{code_dept}_{y}').is_dir():
            create_cosia_geojson(dir_output / f'CoSIA_D0{code_dept}_{y}', geo.bounds, dir_output)
            break
    try:
        os.remove(dir_output / f'CoSIA_D0{code_dept}_{y}.zip')
    except:
        pass
    
# Fonction pour télécharger les données de forêt
def download_foret(code_dept: int, dept, dir_output) -> None:
    """
    Télécharge les données de forêt depuis l'URL spécifiée et les enregistre dans le chemin de destination
    
    :param url: L'URL à partir de laquelle télécharger les données de forêt.
    :param destination_path: Le chemin de destination où enregistrer les données téléchargées.
    """
    check_and_create_path(dir_output)
    check_and_create_path(dir_output / 'BDFORET')
    print('DONWLOADING FOREST LANDCOVER')
    url = dico_foret_url[dept]
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    match = re.search(date_pattern, url)

    # Extract the date if found
    if match:
        date = match.group()
        print("Date found:", date)
    else:
        print("No date found in the URL")
    
    if not (dir_output / f'BDFORET_2-0__SHP_LAMB93_D0{code_dept}_{date}.7z').is_file():
        subprocess.run(['wget', url, '-O', f'{dir_output}/BDFORET_2-0__SHP_LAMB93_D0{code_dept}_{date}.7z'], check=True)
    unzip_7z(dir_output / f'BDFORET_2-0__SHP_LAMB93_D0{code_dept}_{date}.7z', dir_output)
    path = (dir_output / f'BDFORET_2-0__SHP_LAMB93_D0{code_dept}_{date}' / 'BDFORET').as_posix()
    files = glob.glob(path+'/**/*.shp', recursive=True)
    print(files, path)
    gdf = gpd.read_file(files[0]).to_crs(crs=4326)
    gdf['code'] = gdf['ESSENCE'].apply(lambda x : valeurs_foret_attribut[x])
    gdf.to_file(dir_output / 'BDFORET' / 'foret.geojson')

def graph2geo(graph, nodes, edges, name):
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph)
    gdf_nodes['name'] = name
    gdf_edges['name'] = name
    nodes.append(gdf_nodes)
    edges.append(gdf_edges)

# Fonction pour télécharger les données OSNMX
def download_osnmx(geo, dir_output) -> None:
    check_and_create_path(dir_output)
    print('DONWLOADING OSMNX')
    graph = ox.graph_from_polygon(unary_union(geo['geometry'].values), network_type='all')
    nodesArray = []
    edgesArray = []
    graph2geo(graph, nodesArray, edgesArray, '')
    edges = pd.concat(edgesArray)
    subedges = edges[edges['highway'].isin(['motorway', 'primary', 'secondary', 'tertiary', 'path'])]
    subedges['coef'] = 1
    subedges['label'] = 0
    for i, hw in enumerate(['motorway', 'primary', 'secondary', 'tertiary', 'path']):
        subedges.loc[subedges['highway'] == hw, 'label'] = i +1

    subedges[['geometry', 'label']].to_file(dir_output / 'osmnx.geojson', driver='GeoJSON')

def download_region(departement, dir_output):
    print('DONWLING REGION')
    check_and_create_path(dir_output)
    url = f'https://france-geojson.gregoiredavid.fr/repo/departements/{name2intstr[departement]}-{name2strlow[departement]}/{departement}.geojson'
    subprocess.run(['wget', url, '-O', f'{dir_output}/geo.json'], check=True)
    region = json.load(open(f'{dir_output}/geo.json'))
    polys = [region["geometry"]]
    geom = [shape(i) for i in polys]
    region = gpd.GeoDataFrame({'geometry':geom})
    region.to_file(dir_output / 'geo.geojson')

def download_hexagones(path, geo, dir_output):
    print('CREATE HEXAGONES')
    check_and_create_path(dir_output)
    hexa_france = gpd.read_file(path / 'hexagones_france.gpkg')
    hexa_france['isdep'] = hexa_france['geometry'].apply(lambda x : geo.contains(x))
    hexa = hexa_france[hexa_france['isdep']]
    hexa.to_file(dir_output / 'hexagones.geojson', driver='GeoJSON')

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

def download_air(path, geo, dir_output):
    check_and_create_path(dir_output)
    stations = pd.read_csv(path / 'Export_stations_france.csv', delimiter=';')
    stations = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations.Longitude, stations.Latitude))
    print(stations.columns)
    center = geo.centroid
    dist_max = 75
    stations['dist'] = stations.apply(lambda x : haversine((float(x['Longitude']), float(x['Latitude'])), (float(center.x), float(center.y))), axis=1)
    stations = stations[stations['dist'] <= dist_max]
    #stations.rename({'Longitude' : 'longitude', 'Latitude' : 'latitude'}, inplace=True, axis=1)
    stations.to_csv(dir_output / 'ExportStations.csv', index=False)

def download_argile(path, code, dir_output):
    print('DOWNLING ARGILE')
    check_and_create_path(dir_output)
    argile_france = gpd.read_file(path / 'AleaRG_Fxx_L93' / 'ExpoArgile_Fxx_L93.shp')
    argile_dept = argile_france[argile_france['DPT'] == str(code)].reset_index(drop=True)
    argile_dept = argile_dept.to_crs('EPSG:4326')
    argile_dept.to_file(dir_output / 'argile.geojson', driver='GeoJSON')