from tools import *
from category_encoders import TargetEncoder, CatBoostEncoder

sentinel_variables = ['NDVI', 'NDMI', 'NDBI', 'NDSI', 'NDWI']

def raster2geojson(dir_raster, mask, h3, var, encoder):
    data = read_object(var, dir_raster)
    if data is None:
        return h3
    else:
        unode = np.unique(mask)
        for node in unode:
            dataNode = data[(mask == node) & ~(np.isnan(data))]
            if encoder is not None:
                dataNode = encoder.transform(dataNode)
            
            h3.loc[h3['scale0'] == node, var] = dataNode 

    return h3

if __name__ == "__main__":

    departements = ['departement-01-ain']
    variables = ['sentinel', 'osmnx', 'landcover', 'population', 'elevation', 'foret']

    dir_data = Path()
    dir_raster = Path()
    dir_mask = Path()

    regions = gpd.read_file('/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/regions.geojson')
    regions['scale0'] = regions.index
    regions.index = regions['hex_id']
    dico = regions['scale0'].to_dict()
    regions.reset_index(drop=True, inplace=True)

    for departement in departements:
        h3 = regions[regions['departement'] == departement]
        mask = read_object(departement+'rasterScale0.pkl', dir_raster)
        for var in variables:
            if var in ['landcover', 'foret']:
                dir_encoder = Path('')
                encoder = ...
            else:
                encoder = None
            h3 = raster2geojson(dir_raster, mask, h3, var, None)

        h3.to_file(dir_data / 'spatial/hexagones.geojson', driver='GeoJSON')