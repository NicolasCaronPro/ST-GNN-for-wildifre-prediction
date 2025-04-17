from tools import *
from category_encoders import TargetEncoder, CatBoostEncoder
import argparse

foret_variables = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
sentinel_variables = ['NDVI', 'NDMI', 'NDBI', 'NDSI', 'NDWI']
cosia_variables = [
    'Other',
    'Building',
    'Bare soil',
    'Water surface',
    'Conifer',
    'Deciduous',
    'Shrubland',
    'Lawn',
    'Crop'
]
osmnx_variables = ['0', '1', '2', '3', '4', '5']
dynamic_world_variables = ['water', 'tree', 'grass', 'crops', 'shrub', 'flooded', 'built', 'bare', 'snow']
landcover_variables = [
                      'foret_encoder',
                      'argile_encoder',
                      'id_encoder',
                      'cosia_encoder'
                        ]

foretint2str = {
    '0': 'PasDeforet',
    '1': 'Châtaignier',
    '2': 'Chênes décidus',
    '3': 'Chênes sempervirents',
    '4': 'Conifères',
    '5': 'Douglas',
    '6': 'Feuillus',
    '7': 'Hêtre',
    '8': 'Mélèze',
    '9': 'Mixtes',
    '10': 'NC',
    '11': 'NR',
    '12': 'Pin à crochets, pin cembro',
    '13': 'Pin autre',
    '14': 'Pin d\'Alep',
    '15': 'Pin laricio, pin noir',
    '16': 'Pin maritime',
    '17': 'Pin sylvestre',
    '18': 'Pins mélangés',
    '19': 'Peuplier',
    '20': 'Robinier',
    '21': 'Sapin, épicéa'
}

osmnxint2str = {
'0' : 'PasDeRoute',
'1':'motorway',
 '2': 'primary',
 '3': 'secondary',
 '4': 'tertiary',
 '5': 'path'}

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

name2str = {
    'departement-01-ain': 'Ain',
    'departement-25-doubs' : 'Doubs',
    'departement-69-rhone' : 'Rhone',
    'departement-78-yvelines' : 'Yvelines'
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sinister', type=str, help='Sinister')
    parser.add_argument('-r', '--resolution', type=str, help='Resolution')
    parser.add_argument('-d', '--database', type=str, help='Database')
    parser.add_argument('-se', '--SinisterEncoding', type=str, help='SinisterEncoding')

    args = parser.parse_args()

    # Input config
    sinister = args.sinister
    resolution = args.resolution
    database = args.database
    sinister_encoding = args.SinisterEncoding

    departements = ['departement-01-ain', 'departement-25-doubs', 'departement-69-rhone', 'departement-78-yvelines']
    variables = ['sentinel', 'osmnx', 'population', 'elevation', 'foret_landcover',  'osmnx_landcover' , 'foret', 'dynamic_world', 'sinister', 'cosia', 'cosia_landcover', 'argile']

    root_data = Path(f'/home/caron/Bureau/csv')
    root_data_disk = Path('/media/caron/X9 Pro/travaille/Thèse/csv')
    root_raster = Path(f'/media/caron/X9 Pro/travaille/Thèse/csv')
    dir_mask = Path(f'/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/Target/{sinister}/{database}/{sinister_encoding}/raster/{resolution}')
    dir_encoder = Path(f'/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/inference/{sinister}/{resolution}/train/Encoder/')
    dir_target = Path(f'/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/Target/{sinister}/{database}/{sinister_encoding}/bin/{resolution}')
    regions = gpd.read_file(f'/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/regions/{sinister}/{database}/regions.geojson')
    regions['scale0'] = regions.index
    regions.index = regions['hex_id']
    dico = regions['scale0'].to_dict()
    regions.reset_index(drop=True, inplace=True)

    allDates = find_dates_between('2017-06-12', '2023-09-11')
    date_to_choose = allDates[-1]
    index = allDates.index(date_to_choose)
    for departement in departements:
        print(departement)
        dir_data = root_data_disk / departement / 'data'
        dir_data_disk = root_data_disk / departement / 'data'
        dir_raster = root_raster / departement / 'raster' / resolution
        h3 = regions[regions['departement'] == departement][['geometry', 'hex_id', 'scale0']]
        mask = read_object(departement+'rasterScale0.pkl', dir_mask)[0]
        for var in variables:
            print(var)
            if var == 'sinister':
                data = read_object(f'{departement}binScale0.pkl', dir_target)
            else:
                data = read_object(var+'.pkl', dir_raster)
            if var in ['dynamic_world_landcover', 'foret_landcover', 'osmnx_landcover', 'cosia_landcover', 'argile']:
                if var != 'argile':
                    vv = var.split('_')[1]
                encoder = read_object('encoder_'+vv+'.pkl', dir_encoder)
            encoder = None
            if var ==  'sentinel':
                data = data[:,:,:, index]
            elif var  == 'dynamic_world':
                data = data[:,:,:, index]
            elif var == 'sinister':
                data = np.nansum(data, axis=2)

            h3 = raster2geojson(data, mask, h3, var, encoder)

        outname = 'hexagones_'+sinister+'.geojson'
        h3.rename({'osmnx_landcover': 'highway_encoder', 'foret_landcover': 'foret_encoder', 'cosia_landcover': 'cosia_encoder', 'argile' : 'argile_encoder'}, axis=1, inplace=True)
        check_and_create_path(dir_data / 'spatial' / outname)
        check_and_create_path(dir_data_disk / 'spatial' / outname)
        print(h3.columns)
        h3.to_file(dir_data / 'spatial' / outname, driver='GeoJSON')
        h3.to_file(dir_data_disk / 'spatial' / outname, driver='GeoJSON')