from probabilistic import *
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import datetime as dt
from dico_departements import *
from geopy.distance import geodesic

def process_year(france, regions, df, dir_output):
    
    years = df.year.unique()
    regions['label'] = 0
    for year in years:
        dff = df[df['year'] == year]
        
        region_year = regions.set_index('hex_id').join(dff.set_index('h3')['label'], rsuffix='_fire').reset_index()
        region_year.loc[region_year[region_year['label_fire'].isna()].index, 'label_fire'] = 0

        sat0, _, _ = rasterization(region_year, n_pixel_y, n_pixel_x, 'label_fire', dir_output, f'year_{year}')

        sat0[np.isnan(france)] = np.nan

        print(np.unique(sat0))
        print(dir_output, f'fire_{year}.pkl')
        save_object(sat0, f'fire_{year}.pkl' , dir_output)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Train',
        description='Create graph and database according to config.py and tained model',
    )
    parser.add_argument('-r', '--read', type=str, help='Read or compute mask')
    parser.add_argument('-s', '--sinister', type=str, help='Sininster')
    parser.add_argument('-p', '--past', type=str, help='Past convolution')
    parser.add_argument('-re', '--resolution', type=str, help='Resolution')
    parser.add_argument('-am', '--addMean', type=str, help='Add mean kernel')
    parser.add_argument('-d', '--dataset', type=str, help='Dataset to Use')
    parser.add_argument('-se', '--sinisterEncoding', type=str, help='Value to use for sinister encoding')
    parser.add_argument('-od', '--output_dataset', type=str, help='Value to use for sinister encoding')

    args = parser.parse_args()

    sinister = args.sinister
    read = args.read == 'True'
    doPast = args.past == 'True'
    addMean = args.addMean == 'True'
    resolution = args.resolution
    dataset_name = args.dataset
    sinister_encoding = args.sinisterEncoding
    output_dataset = args.output_dataset

    if output_dataset is None:
        output_dataset = dataset_name

    ###################################### Data loading ###################################
    #root = Path('/home/caron/Bureau/csv')
    root = Path('/media/caron/X9 Pro/travaille/Thèse/csv')
    dir_output = Path('/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/Target/'+sinister+'/'+output_dataset + '/' + sinister_encoding)

    departements = [f'departement-{dept}' for dept in departements]
    spa = 3
    regions = []

    for i, dept in enumerate(departements):
        if not (root / dept / 'data' / 'spatial/hexagones.geojson').is_file():
            departements.pop(departements.index(dept))
            continue
        h3 = gpd.read_file(root / dept / 'data' / 'spatial/hexagones.geojson')
        h3['latitude'] = h3['geometry'].apply(lambda x : float(x.centroid.y))
        h3['longitude'] = h3['geometry'].apply(lambda x : float(x.centroid.x))
        h3['departement'] = dept
        regions.append(h3)

    regions = pd.concat(regions)
    regions['label'] = 1

    regions['code'] = regions['departement'].apply(lambda x : name2int[x])
    
    ################################### Create output directory ###########################
    check_and_create_path(dir_output / 'mask' / 'geo' / resolution)
    check_and_create_path(dir_output / 'mask/tif' / resolution)

    check_and_create_path(dir_output / 'bin' / resolution) # Binary image
    check_and_create_path(dir_output /  'raster' / resolution) # cluster Image

    ################################# Define variable ################################

    resolutions = {'2x2' : {'x' : 0.02875215641173088,'y' :  0.020721094073767096},
                '1x1' : {'x' : 0.01437607820586544,'y' : 0.010360547036883548},
                '0.5x0.5' : {'x' : 0.00718803910293272,'y' : 0.005180273518441774},
                '0.03x0.03' : {'x' : 0.0002694945852326214,'y' :  0.0002694945852352859}}

    n_pixel_x = resolutions[resolution]['x']
    n_pixel_y = resolutions[resolution]['y']

    sdate = '2017-06-12'
    edate = '2024-06-29'

    fp = pd.read_csv(root / 'france' / sinister / f'{sinister}.csv', dtype={'Département': str})

    fp['label'] = 1

    ################################## Process #################################

    fp['year'] = fp['date'].apply(lambda x : x.split('-')[0])
    
    fp = fp.groupby(['h3', 'year'])['label'].sum().reset_index()

    if (dir_output / 'france.pkl').is_file():
        france = read_object('france.pkl', dir_output)
    else:
        france, _, _ = rasterization(regions, n_pixel_y, n_pixel_x, 'label', dir_output, f'france')
        save_object(france, f'france.pkl', dir_output)

    departement, _, _ = rasterization(regions, n_pixel_y, n_pixel_x, 'code', dir_output, f'departement')        
    save_object(departement, f'departement.pkl', dir_output)
    print(np.unique(departement))

    process_year(france, regions, fp, dir_output)