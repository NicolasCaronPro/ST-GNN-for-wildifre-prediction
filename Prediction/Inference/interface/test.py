from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely import unary_union
import os

def integrate_new_geometries(code_geom_max : int, geometries, hexa):
        uids = hexa.id.unique()
        for id in uids:
            geo_id = unary_union(hexa[hexa['id'] == id].geometry.values)
            code_geom_id = code_geom_max + id
            print(f'Incorporate {id} geometry into {code_geom_id}')
            if f'{code_geom_id}' in geometries.code_geom.unique():
                print(f'Dropping geometry {code_geom_id}')
                geometries.drop(geometries[geometries['code_geom'] == f'{code_geom_id}'].index, inplace=True)
            row = {'code_geom' : f'{code_geom_id}', 'type' : 'INCENDIE', 'nom' : f'{id}'}
            if not geo_id.is_valid:
                print(f"Invalid geometry for {id}")
            else:
                geo_id = gpd.GeoDataFrame([row], geometry=[geo_id], crs=geometries.crs)
                geometries = pd.concat((geometries, geo_id))
        geometries.reset_index(drop=True, inplace=True)
        return geometries
 
code_geom_max = 115 + 1

"""root_dir = Path(__file__).absolute().parent.parent.parent.parent.resolve()
dir_data = root_dir / 'data'
dir_geo = dir_data / "geolocalisation"

geometries = gpd.read_file(dir_geo / 'geometries' / 'geometries.gpkg')
print(geometries.code_geom.unique())

dir_incendie_historique = root_dir / 'scripts' / 'global' / 'sinister_prediction' / 'hexagones'

historical_geojson = list(dir_incendie_historique.glob('hexagones_*.geojson'))
hexa = gpd.read_file(historical_geojson[0])"""

geometries = gpd.read_file('geometries.gpkg')
print(geometries.code_geom.unique())
hexa = gpd.read_file('/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/Inference/interface/departement-01-ain/hexagones_2024-08-05.geojson')
hexa['id'] = hexa['id'].astype(int)

geometries = integrate_new_geometries(code_geom_max, geometries, hexa)
print('############################ VALIDATION ###########################')
print(geometries)
#geometries.to_file(dir_geo / 'geometries' / 'geometries.geojson', driver='GeoJSON')
try:
    os.remove('geometries.gpkg')
    geometries.to_file('geometries.gpkg', driver='GPKG')
except Exeption as e:
    print(f'{e}')
print('############################ TEST ###########################')
#geometries = gpd.read_file(dir_geo / 'geometries' / 'geometries.geojson')
geometries = gpd.read_file( 'geometries.gpkg')
print(geometries)