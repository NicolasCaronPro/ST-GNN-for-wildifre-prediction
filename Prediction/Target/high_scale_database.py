from probabilistic import *
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse

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

############################################################## 

##########          My class  

##############################################################

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

    return probaImageScale, binImageScale

def add_hour(x, h):
        if h < 16 or h == 24:
            return x
        else:
            x = datetime.datetime.strptime(x, '%Y-%m-%d') - datetime.timedelta(days=1)
            return x.strftime('%Y-%m-%d')

def process_department(departements, sinister, n_pixel_y, n_pixel_x, read):

    sinisterPoints = []
    input = []

    for dept in departements:
        code = int2strMaj[int(dept.split('-')[1])]

        if not read:
            sat0, _, _ = rasterization(regions[regions['departement'] == dept], n_pixel_y, n_pixel_x, 'scale0', dir_output, dept+'_scale0')
            uniques_ids = np.unique(sat0[~np.isnan(sat0)])
            additionnal_pixel_x = []
            additionnal_pixel_y = []
            for ui in uniques_ids:
                mask = np.argwhere(sat0[0] == ui)
                if mask.shape[0] > 1:
                    additionnal_pixel_x += list(mask[1:, 0])
                    additionnal_pixel_y += list(mask[1:, 1])
            additionnal_pixel_x = np.asarray(additionnal_pixel_x)
            additionnal_pixel_y = np.asarray(additionnal_pixel_y)
            sat0[0, additionnal_pixel_x, additionnal_pixel_y] = np.nan
            save_object(sat0, dept+'rasterScale0.pkl', dir_output / 'raster' / resolution)
        else:
            sat0 = read_object(dept+'rasterScale0.pkl', dir_output / 'raster' / resolution)
        
        if sinister == 'firepoint':
            name = 'NATURELSfire.csv'
        else:
            name = 'inondation.csv'

        try:
            sp = pd.read_csv(root / dept / sinister / name)
        except Exception as e:
            print(e)
            print('Return a full zero image')
            inputDep = np.zeros((*sat0[0].shape, len(creneaux)), dtype=float)
            inputDep[np.isnan(sat0[0])] = np.nan
            input.append(inputDep)
            save_object(inputDep, dept+'binScale0.pkl', dir_output / 'bin' / resolution)
            continue

        sp = sp[(sp['date'] >= sdate) & (sp['date'] < edate)]
        #sp['hour'] = sp['date_debut'].apply(lambda x : int(x.split(' ')[1].split(':')[0]))
        #sp['date'] = sp.apply(lambda x : add_hour(x['date'], x['hour']), axis=1)
        sp['departement'] = dept

        #regions.loc[regions[regions['departement'] == dept].index, 'departement'] = dept
        sp['scale0'] = sp['h3'].replace(dico)

        sat0 = sat0[0]
        if not read:
            inputDep = create_spatio_temporal_sinister_image(sp, regions[regions['departement'] == dept],
                                                             creneaux, sat0, sinister, n_pixel_y, n_pixel_x, dir_output, dept)
            inputDep[additionnal_pixel_x, additionnal_pixel_y, :] = 0
            save_object(inputDep, dept+'binScale0.pkl', dir_output / 'bin' / resolution)

        else: 
            inputDep = read_object(dept+'binScale0.pkl', dir_output / 'bin' / resolution)

        input.append(inputDep)
        sinisterPoints.append(sp)

    return sinisterPoints, input

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

    args = parser.parse_args()

    sinister = args.sinister
    read = args.read == 'True'
    doPast = args.past == 'True'
    addMean = args.addMean == 'True'
    resolution = args.resolution

    ###################################### Data loading ###################################
    root = Path('/home/caron/Bureau/csv')
    dir_output = Path('/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/Target/'+sinister)

    if sinister == "firepoint":
        departements = ['departement-01-ain',
                        'departement-25-doubs',
                        'departement-69-rhone', 
                        'departement-78-yvelines'
                        ]
        
    elif sinister == "inondation":
        departements = ['departement-01-ain',
                        'departement-25-doubs',
                        'departement-69-rhone', 
                        'departement-78-yvelines'
                        ]

    #regions = gpd.read_file('/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/regions/regions.geojson')
    regions = []
    for dept in departements:
        h3 = gpd.read_file(root / dept / 'data' / 'spatial/hexagones.geojson')
        h3['latitude'] = h3['geometry'].apply(lambda x : float(x.centroid.y))
        h3['longitude'] = h3['geometry'].apply(lambda x : float(x.centroid.x))
        h3['departement'] = dept
        regions.append(h3)
    regions = pd.concat(regions)
    regions['scale0'] = regions.index
    regions.index = regions['hex_id']
    dico = regions['scale0'].to_dict()
    regions.reset_index(drop=True, inplace=True)
    regions.to_file('/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/regions/regions.geojson', driver='GeoJSON')

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
    #edate = datetime.datetime.now().date().strftime('%Y-%m-%d')
    edate = '2024-06-29'
    creneaux = find_dates_between(sdate, edate)

    isotonic = IsotonicRegression(y_min=0, y_max=1.0, out_of_bounds='clip')
    logistic = LogisticRegression(random_state=42, solver='liblinear', penalty='l2', C=0.5,
                                intercept_scaling=0.5, fit_intercept=True, class_weight={0: 1, 1 : 1})
    
    spa = 3

    if sinister == "firepoint":
        dims = [[(spa,spa,5), (spa,spa,9), (spa,spa,3)],
                [(spa,spa,3), (spa,spa,5), (spa,spa,3)],
                [(spa,spa,3), (spa,spa,5),(spa,spa,1)],
                [(spa,spa,3), (spa,spa,7), (spa,spa,3)]]
        
    elif sinister == "inondation":
        dims = [(spa,spa,5),
                (spa,spa,5),
                (spa,spa,5),
                (spa,spa,5)]
    else:
        exit(2)

    ################################## Process #################################

    fp, input = process_department(departements=departements, sinister=sinister,
                                    n_pixel_y=n_pixel_y, n_pixel_x=n_pixel_x, read=read)

    fp = pd.concat(fp).reset_index(drop=True)
    fp.to_csv('/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/sinister/'+sinister+'.csv', index=False)

    model = Probabilistic(n_pixel_x, n_pixel_y, 1, logistic, dir_output, resolution)
    model._process_input_raster(dims, input, len(departements), True, departements, doPast, creneaux, departements)