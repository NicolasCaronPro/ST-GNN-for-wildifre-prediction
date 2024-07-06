from tools import *
from category_encoders import TargetEncoder, CatBoostEncoder
import argparse

foret_variables = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']
sentinel_variables = ['NDVI', 'NDMI', 'NDBI', 'NDSI', 'NDWI']
osmnx_variables = ['0', '1', '2', '3', '4', '5']
dynamic_world_variables = ['water', 'tree', 'grass', 'crops', 'shrub', 'flooded', 'built', 'bare', 'snow']
landcover_variables = [#'landcover',
                        'highway',
                       'foret',
                        ]

def raster2geojson(data, mask, h3, var, encoder):
    if var == 'sentinel':
        h3[sentinel_variables] = 0
    else:
        h3[var] = 0
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
                    h3.loc[h3[h3['scale0'] == node].index, v2] = np.mean(dataNode) 
            elif var == "osmnx":
                for band, v2 in enumerate(osmnx_variables):
                    dataNode = data[band, (mask == node) & ~(np.isnan(data[band]))]
                    h3.loc[h3[h3['scale0'] == node].index, v2] = np.mean(dataNode)
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

    args = parser.parse_args()

    # Input config
    sinister = args.sinister

    departements = ['departement-01-ain', 'departement-25-doubs', 'departement-69-rhone', 'departement-78-yvelines']
    variables = ['sentinel', 'osmnx', 'population', 'elevation', 'foret_landcover', 'foret', 'dynamic_world']

    root_data = Path('/home/caron/Bureau/csv')
    root_raster = Path('/home/caron/Bureau/csv')
    dir_mask = Path('/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/Target/'+sinister+'/raster')
    dir_encoder = Path('/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/final/'+sinister+'/train/Encoder/')

    regions = gpd.read_file('/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/regions/regions.geojson')
    regions['scale0'] = regions.index
    regions.index = regions['hex_id']
    dico = regions['scale0'].to_dict()
    regions.reset_index(drop=True, inplace=True)
    
    allDates = find_dates_between('2017-06-12', '2023-09-11')
    date_to_choose = allDates[-1]
    index = allDates.index(date_to_choose)
    for departement in departements:
        print(departement)
        dir_data = root_data / departement / 'data'
        dir_raster = root_raster / departement / 'raster'
        h3 = regions[regions['departement'] == name2str[departement]][['geometry', 'hex_id', 'scale0']]
        mask = read_object(departement+'rasterScale0.pkl', dir_mask)[0]
        for var in variables:
            if var in ['population', 'elevation']:
                print(var)
                h32 = gpd.read_file(dir_data / 'spatial' / 'hexagones.geojson')
                h3[var] = h32[var].values
                print(h3[var].unique())
            else:
                data = read_object(var+'.pkl', dir_raster)
                if var in ['dynamic_world_landcover', 'foret']:
                    encoder = read_object('encoder_'+var+'.pkl', dir_encoder)
                else:
                    encoder = None
                if var ==  'sentinel':
                    data = data[:,:,:, index]
                elif var  == 'dynamic_world':
                    data = data[:,:,:, index]

                h3 = raster2geojson(data, mask, h3, var, encoder)

        outname = 'hexagones_'+sinister+'.geojson'
        h3.to_file(dir_data / 'spatial' / outname, driver='GeoJSON')