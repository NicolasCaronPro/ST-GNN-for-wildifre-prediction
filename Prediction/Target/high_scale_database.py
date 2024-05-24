from probabilistic import *
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse

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
        code = int(dept.split('-')[1])
        
        if sinister == 'firepoint':
            name = 'NATURELSfire.csv'
        else:
            name = 'inondation.csv'

        sp = pd.read_csv(root / dept / sinister / name)
        sp = sp[(sp['date'] >= sdate) & (sp['date'] < edate)]
        #sp['hour'] = sp['date_debut'].apply(lambda x : int(x.split(' ')[1].split(':')[0]))
        #sp['date'] = sp.apply(lambda x : add_hour(x['date'], x['hour']), axis=1)
        sp['departement'] = code

        regions.loc[regions[regions['departement'] == (dept.split('-')[-1][0].upper()) + dept.split('-')[-1][1:]].index, 'departement'] = code
        sp['scale0'] = sp['h3'].replace(dico)

        if not read:
            sat0, _, _ = rasterization(regions[regions['departement'] == code], n_pixel_y, n_pixel_x, 'scale0', dir_output, dept+'_scale0')
            save_object(sat0, dept+'rasterScale0.pkl', dir_output / 'raster')
        else:
            sat0 = read_object(dept+'rasterScale0.pkl', dir_output / 'raster')
        sat0 = sat0[0]
        if not read:
            inputDep = create_spatio_temporal_sinister_image(sp, regions[regions['departement'] == code],
                                                             creneaux, sat0, sinister, n_pixel_y, n_pixel_x, dir_output, dept)
            save_object(inputDep, dept+'binScale0.pkl', dir_output / 'bin')

        else: 
            inputDep = read_object(dept+'binScale0.pkl', dir_output / 'bin')

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

    args = parser.parse_args()

    sinister = args.sinister
    read = args.read == 'True'
    doPast = args.past == 'True'

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
        departements = [#'departement-01-ain',
                        'departement-25-doubs',
                        #'departement-69-rhone', 
                        #'departement-78-yvelines'
                        ]

    regions = gpd.read_file('/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/regions/regions.geojson')
    regions['scale0'] = regions.index
    regions.index = regions['hex_id']
    dico = regions['scale0'].to_dict()
    regions.reset_index(drop=True, inplace=True)

    ################################### Create output directory ###########################
    check_and_create_path(dir_output / 'mask/geo')
    check_and_create_path(dir_output / 'mask/tif')

    check_and_create_path(dir_output / 'bin') # Binary image
    check_and_create_path(dir_output / 'raster') # cluster Image

    ################################# Define variable ################################

    #n_pixel_x = 0.016133099692723363
    #n_pixel_y = 0.016133099692723363

    n_pixel_x = 0.02875215641173088
    n_pixel_y = 0.020721094073767096

    sdate = '2017-06-12'
    edate = '2023-09-11'
    creneaux = find_dates_between(sdate, edate)

    isotonic = IsotonicRegression(y_min=0, y_max=1.0, out_of_bounds='clip')
    logistic = LogisticRegression(random_state=42, solver='liblinear', penalty='l2', C=0.5,
                                intercept_scaling=0.5, fit_intercept=True, class_weight={0: 1, 1 : 1})
    spa = 3

    if sinister == "firepoint":
        dims = [(spa,spa,11),
                (spa,spa,7),
                (spa,spa,5),
                (spa,spa,9)]
    elif sinister == "inondation":
        dims = [(spa,spa,5)]
    else:
        exit(2)

    ################################## Process #################################

    fp, input = process_department(departements=departements, sinister=sinister,
                                    n_pixel_y=n_pixel_y, n_pixel_x=n_pixel_x, read=read)

    fp = pd.concat(fp).reset_index(drop=True)
    fp.to_csv('/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/'+sinister+'.csv', index=False)

    model = Probabilistic(dims, n_pixel_x, n_pixel_y, 1, logistic, dir_output)

    model._process_input_raster(input, len(departements), True, departements, doPast)