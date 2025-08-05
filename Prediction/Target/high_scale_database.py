from probabilistic import *
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import datetime as dt
from dico_departements import *
from geopy.distance import geodesic

############################################################## 

##########          My class  

##############################################################

# Fonction pour regrouper les points proches d'une même date
def group_close_points(points, distance_threshold=20):
    """
    Regroupe les points géographiques qui sont à une distance inférieure à distance_threshold km.
    points est une liste de tuples (latitude, longitude).
    Renvoie une liste de groupes (clusters).
    """
    groups = []

    for i, point in enumerate(points):
        # Créer un nouveau groupe
        group = [point]
        
        # Comparer ce point avec tous les autres points
        for j, other_point in enumerate(points):
            p1 = Point(point[1], point[0])
            p2 = Point(other_point[1], other_point[0])
            if haversine(p1, p2) < distance_threshold:
                if other_point not in group:
                    group.append(other_point)
        
        groups.append(group)
    
    return groups

# Fonction pour vérifier si un point est proche de tous les points d'un cluster
def are_points_close(cluster, new_points, distance_threshold=20):
    """
    Vérifie si tous les points dans 'new_points' sont proches (moins de distance_threshold km)
    de tous les points dans 'cluster'.
    """
    for point in cluster:
        for new_point in new_points:
            p1 = Point(point[1], point[0])
            p2 = Point(new_point[1], new_point[0])
            if haversine(p1, p2) >= distance_threshold:
                return False
    return True

# Fonction pour vérifier si un point est proche de tous les points d'un cluster
def are_point_close(point, new_points, distance_threshold=20):
    """
    Vérifie si tous les points dans 'new_points' sont proches (moins de distance_threshold km)
    de tous les points dans 'cluster'.
    """
    for new_point in new_points:
        p1 = Point(point[1], point[0])
        p2 = Point(new_point[1], new_point[0])
        if haversine(p1, p2) <= distance_threshold:
            return True, new_point
    return False, None

# Fonction pour calculer les séquences continues pour les groupes de points
def calculate_sequences(df):
    """
    Calcule la taille des séquences continues sans interruption de date pour les points proches.
    Renvoie une liste des tailles de séquences.
    """
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    sequences = []
    td = 1
    
    # Grouper les données par date
    for date, group in df.groupby('date'):
        points = list(zip(group['latitude'], group['longitude']))
        
        # Regrouper les points proches
        grouped_points = group_close_points(points)
        
        # Pour chaque groupe, créer ou étendre les séquences
        for cluster in grouped_points:
            
            if not sequences:
                # Créer la première séquence si elle n'existe pas
                sequences.append({'dates': [date], 'points': cluster})
            else:
                find_seq = False
                for sei, seq in enumerate(sequences):
                    i = 1
                    while (date - seq['dates'][-i]).days <= td and (date != seq['dates'][-i]):
                        is_close, p = are_point_close(seq['points'][-i], cluster, 20)
                        if date in seq['dates'] and p in seq['points']:
                            continue
                        if is_close:
                            seq['dates'].append(date)
                            seq['points'].append(p)
                            find_seq = True
                            break
                        i -= 1
                        if i < 0:
                            break
                    sequences[sei] = seq
                    
                if not find_seq:
                    # Sinon, commencer une nouvelle séquence
                    sequences.append({'dates': [date], 'points': cluster})
    
    # Retourner la taille de chaque séquence
    sequence_lengths = [len(np.unique(seq['dates'])) for seq in sequences]
    return sequence_lengths

def find_kernel_size(dept, months, dataset_name):

    sequences = {}
    if dataset_name == 'firemen':
        if sinister == 'firepoint':
            name = 'NATURELSfire.csv'
        else:
            name = 'inondation.csv'
        try:
            fp = pd.read_csv(root / dept / sinister / name)
        except:
            for i, month in enumerate(months):
                sequences[i] = {}
                sequences[i]['seq'] = [0]
                sequences[i]['ndays'] = [0]
            return sequences, 0

    elif dataset_name == 'bdiff' or dataset_name == 'vigicrues' or dataset_name == 'georisques':
        fp = pd.read_csv(root / 'france' / sinister / f'{sinister}.csv')
        code_dept_str = name2int[dept]
        if code_dept_str < 10:
            code_dept_str = f'0{code_dept_str}'
        else:
            code_dept_str = f'{code_dept_str}'
        fp = fp[fp['Département'] == code_dept_str]

    def dt_2_str(x):
        try:
            return x.strftime('%Y-%m-%d')
        except:
            return x
    
    fp['date'] = fp['date'].apply(lambda x: dt_2_str(x))
    fp = fp[fp['date'] >= sdate]

    if 'month' not in fp.columns:
        fp['month'] = fp['date'].apply(lambda x : int(x.split('-')[1]))

    if 'coef' not in fp.columns:
        fp['coef'] = 1

    for i, month in enumerate(months):

        sequences[i] = {}
        fpp = fp[fp['month'].isin(month)].reset_index(drop=True)
        if len(fpp) == 0:
            sequences[i]['seq'] = [0]
            sequences[i]['ndays'] = [0]
            continue

        temp = fpp.groupby(['date', 'longitude', 'latitude'])['coef'].sum().reset_index()
        temp = temp.sort_values('date')
        
        sequences[i]['seq'] =  calculate_sequences(temp)

    return sequences, len(fp)

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
    # Lists to store the processed data and incident points for each department
    sinisterPoints = []
    input = []

    # Process each department in the list
    for dept in departements:
        
        # If data should be freshly computed (not read from disk)
        if not read:

            # Rasterize the 'scale0' field for the current department. 
            sat0, _, _ = rasterization(
                regions[regions['departement'] == dept], 
                n_pixel_y, n_pixel_x, 
                'scale0', 
                dir_output,
                dept + '_scale0'
            )

            # Find unique IDs in the raster (i.e., unique spatial zones)
            uniques_ids = np.unique(sat0[~np.isnan(sat0)])

            # Remove all but one pixel per unique ID to avoid duplication
            additionnal_pixel_x = []
            additionnal_pixel_y = []
            for ui in uniques_ids:
                mask = np.argwhere(sat0[0] == ui)
                if mask.shape[0] > 1:
                    additionnal_pixel_x += list(mask[1:, 0])
                    additionnal_pixel_y += list(mask[1:, 1])

            # Convert lists to NumPy arrays
            additionnal_pixel_x = np.asarray(additionnal_pixel_x)
            additionnal_pixel_y = np.asarray(additionnal_pixel_y)

            # Set duplicated pixels to NaN
            sat0[0, additionnal_pixel_x, additionnal_pixel_y] = np.nan

            # Save the cleaned raster to disk
            save_object(sat0, dept + 'rasterScale0.pkl', dir_output / 'raster' / resolution)

        else:
            # Load previously saved raster from disk
            sat0 = read_object(dept + 'rasterScale0.pkl', dir_output / 'raster' / resolution)

        # Try loading incident data for the department
        try:
            if dataset_name == 'firemen':
                # Select the correct CSV file depending on the sinister type
                if sinister == 'firepoint':
                    name = 'NATURELSfire.csv'
                elif sinister == 'inondantion':
                    name = 'inondation.csv'
                elif sinister == 'FXAgricoles':
                    name = 'FXAgricoles.csv'
                fp = pd.read_csv(root / dept / sinister / name)
        except:
            # If loading fails, create a zero image for the whole period
            print('Return a full zero image')
            inputDep = np.zeros((*sat0[0].shape, len(creneaux)), dtype=float)
            inputDep[np.isnan(sat0[0])] = np.nan
            input.append(inputDep)
            save_object(inputDep, dept + 'binScale0.pkl', dir_output / 'bin' / resolution)
            continue
        
        # Load data from national datasets and filter by department
        if dataset_name in ['bdiff', 'vigicrues', 'georisques', 'bdiff_small']:
            fp = pd.read_csv(root / 'france' / sinister / f'{sinister}.csv', dtype={'Département': str})
            code_dept_str = name2int[dept]
            code_dept_str = f'{code_dept_str:02}'  # Format with leading zero if needed
            fp = fp[fp['Département'] == code_dept_str]

        else:
            # Convert date columns to datetime safely
            def safe_to_datetime(column):
                return pd.to_datetime(column, errors='coerce')

            fp['date_debut'] = safe_to_datetime(fp['date_debut'])
            fp['date_fin'] = safe_to_datetime(fp['date_fin'])

            # Compute intervention time in hours
            fp['hours_difference'] = (fp['date_fin'] - fp['date_debut']).dt.total_seconds() / 3600

            # Fill missing durations with the mean
            mean_hours = fp['hours_difference'].mean(skipna=True)
            fp['hours_difference'] = fp['hours_difference'].fillna(mean_hours)

        # Filter records to keep only those after the start date
        fp = fp[fp['date'] > sdate]

        if len(fp) == 0:
            # No data: return a zero-filled image
            print('Return a full zero image')
            inputDep = np.zeros((*sat0[0].shape, len(creneaux)), dtype=float)
            inputDep[np.isnan(sat0[0])] = np.nan
            input.append(inputDep)
            save_object(inputDep, dept + 'binScale0.pkl', dir_output / 'bin' / resolution)
            continue

        # Filter incidents that are within the processing period
        sp = fp[(fp['date'] >= sdate) & (fp['date'] < edate)]
        sp['departement'] = dept

        # Replace H3 hexagons with internal scale0 IDs
        sp['scale0'] = sp['h3'].replace(dico)

        sat0 = sat0[0]  # Take the raster array from wrapper list

        # If not reading precomputed files, compute the raster
        if not read:
            inputDep = create_spatio_temporal_sinister_image(
                sp,
                regions[regions['departement'] == dept],
                creneaux,
                sat0,
                sinister,
                sinister_encoding,
                n_pixel_y,
                n_pixel_x,
                dir_output,
                dept
            )
            
            # Set extra pixels to zero (those removed earlier)
            inputDep[additionnal_pixel_x, additionnal_pixel_y, :] = 0.0

            # Save to disk
            save_object(inputDep, dept + 'binScale0.pkl', dir_output / 'bin' / resolution)

        else:
            # Load previously processed raster
            inputDep = read_object(dept + 'binScale0.pkl', dir_output / 'bin' / resolution)

        # Append results for this department
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
    root = Path('/media/caron/X9 Pro1/travaille/Thèse/csv')
    dir_output = Path('/media/caron/X9 Pro1/travaille/Thèse/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/Target/'+sinister+'/'+output_dataset + '/' + sinister_encoding)

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
    
    regions = pd.concat(regions).reset_index(drop=True)
    
    regions['scale0'] = regions.index
    
    regions.index = regions['hex_id']
    
    dico = regions['scale0'].to_dict()
    
    regions.reset_index(drop=True, inplace=True)
    
    check_and_create_path(Path(f'/media/caron/X9 Pro1/travaille/Thèse/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/regions/{sinister}/{output_dataset}'))
    
    regions.to_file(f'/media/caron/X9 Pro1/travaille/Thèse/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/regions/{sinister}/{output_dataset}/regions.geojson', driver='GeoJSON')

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
    creneaux = find_dates_between(sdate, edate)

    dims = {}
    months = []
    months = [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 1]]
    for dept in departements:
        if dept == 'departement-2A-corse-du-sud' or dept == 'departement-2B-haute-corsse':
            departements.pop(departements.index(dept))
            continue
        seq_dept, leni = find_kernel_size(dept, months, dataset_name)
        dim_med = round(np.nanmean(seq_dept[0]['seq'])) * 2 + 1
        dim_high = round(np.nanmean(seq_dept[1]['seq'])) * 2 + 1
        dim_low = round(np.nanmean(seq_dept[2]['seq'])) * 2 + 1
        dims[dept] = ((spa, spa, dim_med), (spa, spa, dim_high), (spa, spa, dim_low))
        print(dept, leni, (dim_med, dim_high, dim_low))
    
    save_object(dims, 'dimension.pkl', Path(f'/media/caron/X9 Pro1/travaille/Thèse/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/regions/{sinister}/{output_dataset}/{sinister_encoding}'))

    ################################## Process #################################

    fp, input = process_department(departements=departements, sinister=sinister,
                                    n_pixel_y=n_pixel_y, n_pixel_x=n_pixel_x, read=read)
    
    for dept in departements:
        # Merge all departmental rasters into a single xarray
        concat_xarrays(Path('/media/caron/X9 Pro1/travaille/Thèse/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/Target/'+sinister+'/'+output_dataset + '/'),
                       allDates, dept, Path('/media/caron/X9 Pro1/travaille/Thèse/csv'), resolution)

    fp = pd.concat(fp).reset_index(drop=True)
    check_and_create_path(Path(f'/media/caron/X9 Pro1/travaille/Thèse/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/sinister/{output_dataset}'))
    fp.to_csv(f'/media/caron/X9 Pro1/travaille/Thèse/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/sinister/{output_dataset}/{sinister}.csv', index=False)

    model = Probabilistic(n_pixel_x, n_pixel_y, 1, None, dir_output, resolution)
    model._process_input_raster(dims, input, len(departements), True, departements, doPast, creneaux, departements, False)