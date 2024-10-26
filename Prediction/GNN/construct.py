import pickle
from nbconvert import export
from sklearn.cluster import KMeans
from copy import copy
from weigh_predictor import Predictor
from GNN.graph_structure import *

def look_for_information(graph, dataset_name : str,
                         maxDate : str,
                         sinister : str, dir : Path,
                         ):

    points = []
    departements = graph.departements.unique()
    for departement in departements:
        if departement in graph.drop_department:
            continue
        
        latitude_points_in_departement = graph.oriLatitudes[graph.departements == departement].values
        longitude_points_in_departement = graph.oriLongitude[graph.departements == departement].values
        # Créer une liste de tuples pour les couples (latitude, longitude)
        coordinates = list(zip(latitude_points_in_departement, longitude_points_in_departement))

        # Utiliser itertools.product pour générer le produit cartésien entre les coordonnées et les dates
        cartesian_product = list(itertools.product(coordinates))

        # Créer le DataFrame
        df = pd.DataFrame(cartesian_product, columns=['Coordinates'])

        # Séparer les colonnes latitude et longitude à partir de la colonne 'Coordinates'
        df[['latitude', 'longitude']] = pd.DataFrame(df['Coordinates'].tolist(), index=df.index)

        # Supprimer la colonne 'Coordinates'
        df = df.drop(columns=['Coordinates'])
        df['departement'] = departement

        logger.info(f'{departement} {len(df)}')

        # Réorganiser les colonnes pour avoir: Longitude, Latitude, Date
        df = df[['longitude', 'latitude', 'departement']]
        points.append(df)
        
    points = pd.concat(points).reset_index(drop=True)
    points.drop_duplicates(inplace=True)
    points['departement'] = points['departement'].apply(lambda x : name2int[x])
    check_and_create_path(dir/sinister)
    name = 'points.csv'
    points.to_csv(dir / sinister / name, index=False)

def construct_graph(scale, maxDist, sinister, dataset_name, sinister_encoding, train_departements,
                    geo, nmax, k_days, dir_output, doRaster, doEdgesFeatures, resolution, graph_construct, train_date, val_date):
    
    graphScale = GraphStructure(scale, geo, maxDist, nmax, resolution, graph_construct, sinister, sinister_encoding, dataset_name, train_departements)

    sucseptibility_map_model_config = {'type': 'Unet',
                                       'device': 'cuda',
                                       'task': 'regression',
                                       'loss' : 'rmse',
                                       'name': 'mapper',
                                       'params' : None}
    
    variables_for_susecptibilty_and_clustering = ['population', 'foret', 'osmnx', #'vigicrues', 'nappes',
                                                  'temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
                                            'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                                            'isi', 'angstroem', 'bui', 'fwi', 'dailySeverityRating',
                                            'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
                                            'days_since_rain', 'sum_consecutive_rainfall',
                                            'sum_rain_last_7_days',
                                            'sum_snow_last_7_days', 'snow24h', 'snow24h16']

    graphScale.train_susecptibility_map(model_config=sucseptibility_map_model_config,
                                        departements=train_departements,
                                        variables=variables_for_susecptibilty_and_clustering, target='risk', train_date=train_date, val_date=val_date,
                                        root_data=rootDisk / 'csv',
                                        root_target=root_target / sinister / dataset_name / sinister_encoding,
                                        dir_output=dir_output)

    graphScale._create_sinister_region(base=graph_construct, doRaster=doRaster,
                                 path=dir_output, sinister=sinister, dataset_name=dataset_name,
                                 sinister_encoding=sinister_encoding,
                                 resolution=resolution, train_date=train_date)
    
    graphScale._create_nodes_list()
    graphScale._create_edges_list()
    graphScale._create_temporal_edges_list(allDates, k_days=k_days)
    graphScale.nodes = graphScale._assign_department(graphScale.nodes)
    
    """graphScale._clusterize_node_with_target(departements=train_departements,
                                            variables=variables_for_susecptibilty_and_clustering,
                                            target='nbsinister', train_date=train_date,
                                            path=dir_output,
                                            root_data=rootDisk / 'csv',
                                            root_target=root_target / sinister / dataset_name / sinister_encoding)"""
    
    graphScale._clusterize_node_with_time_series(departements=train_departements,
                                            variables=variables_for_susecptibilty_and_clustering,
                                            target='nbsinister', train_date=train_date,
                                            path=dir_output,
                                            root_data=rootDisk / 'csv',
                                            root_target=root_target / sinister / dataset_name / sinister_encoding)
    
    graphScale.find_closest_cluster(graphScale.departements.unique(), train_date, dir_output, rootDisk / 'csv')

    if doEdgesFeatures:
        graphScale.edges = edges_feature(graphScale, ['slope', 'highway'], dir_output, geo)
    save_object(graphScale, f'graph_{scale}_{graph_construct}.pkl', dir_output)
    return graphScale

def export_to_all_date(df, dataset_name, sinister, departements, maxDate):
    res = []
    for departement in departements:
        if dataset_name == 'firemen':
            if sinister == 'inondation':
                end_date = '2023-08-03'
                start_date = '2017-06-12'
            else:
                if departement == 'departement-69-rhone':
                    end_date = '2022-12-31'
                    if maxDate < end_date:
                        end_date = maxDate
                else:
                    end_date = allDates[-1]
                if departement == 'departement-69-rhone' or departement == 'departement-01-ain':
                    start_date = '2018-01-01' 
                else:
                    start_date = '2017-06-12'

        elif dataset_name == 'firemen2':
            if sinister == 'inondation':
                end_date = '2023-08-03'
                start_date = '2017-06-12'
            else:
                if departement == 'departement-69-rhone':
                    end_date = '2022-12-31'
                    if maxDate < end_date:
                        end_date = maxDate
                else:
                    end_date = allDates[-1]
                if departement == 'departement-69-rhone' or departement == 'departement-01-ain':
                    start_date = '2018-01-01' 
                elif departement == 'departement-78-yvelines' or departement == 'departement-25-doubs':
                    start_date = '2017-06-12'
                else:
                    start_date = '2023-01-01'

        elif dataset_name == 'bdiff':
            end_date =  allDates[-1]
            start_date = '2017-06-12'
        elif dataset_name == 'georisques':
            end_date =  allDates[-1]
            start_date = '2017-06-12'
        else:
            logger.info(f'Unknown dataset_name {dataset_name}')
            exit(1)

        # Récupérer toutes les dates entre start_date et end_date
        dates = find_dates_between(start_date, end_date)

        # Filtrer les lignes du DataFrame pour le département actuel
        df_filtered = df[df["departement"] == name2int[departement]]
        # Dupliquer chaque ligne pour chaque date dans 'dates'
        for date in dates:
            # Ajouter une colonne "date" avec la date en cours
            df_dup = df_filtered.copy()
            df_dup["date"] = date
            res.append(df_dup)

    # Concaténer toutes les parties dupliquées en un seul DataFrame
    result_df = pd.concat(res, ignore_index=True)
    result_df['date'] = result_df['date'].apply(lambda x : allDates.index(x))
    return result_df

def est_un_entier_chaine(variable):
    try:
        int(variable)
        return True
    except ValueError:
        return False

def construct_database(
    graphScale: GraphStructure,
    ps: pd.DataFrame,
    scale: int,
    k_days: int,
    departements: list,
    features: list,
    sinister: str,
    dataset_name: str,
    sinister_encoding: str,
    dir_output: Path,
    dir_train: Path,
    prefix: str,
    values_per_class: int,
    mode: str,
    resolution: str,
    maxDate: str,
    graph_construct: str
):
    
    
    ######################## Get departments and train departments #######################
    # Select the departments and training departments based on the dataset and sinister type
    departements, train_departements = select_departments(dataset_name, sinister)
    trainCode = [name2int[d] for d in train_departements]

    #######################################################################################
    # Convert department names to their corresponding codes for training departments
    # Prepare data for node prediction by extracting longitude and latitude
    ps.drop_duplicates(subset=['longitude', 'latitude'], inplace=True, keep='first')
    X_kmeans = list(zip(ps.longitude, ps.latitude))
    
    # Predict nodes based on position and assign them to the DataFrame
    
    ps[f'scale{scale}'] = graphScale._predict_node_with_position(X_kmeans, ps.departement)
    ps = ps[~ps[f'scale{scale}'].isna()] 

    # Define the output file name
    name = f'{prefix}.csv'
    # Remove duplicate nodes based on the scale
    ps.drop_duplicates(subset='scale' + str(scale), inplace=True, keep='first')

    # Expand the dataset to include all dates
    ps = export_to_all_date(ps, dataset_name, sinister, departements, maxDate)

    # Save the dataset to a CSV file
    ps.to_csv(dir_output / name, index=False)
    logger.info(f'{len(ps)} point in the dataset. Constructing database')

    # Initialize an array for original nodes with default values
    orinode = np.full((len(ps), 5), -1.0, dtype=float)
    orinode[:, 0] = ps[f'scale{scale}'].values  # Assign node IDs
    orinode[:, 4] = ps['date']  # Assign dates
    orinode[:, 3] = ps['departement']

    print(np.unique(orinode[:, 3]))

    #orinode = generate_subgraph(graphScale, 0, 0, orinode)

    # Add temporal nodes based on the specified number of days (k_days)
    subNode = add_k_temporal_node(k_days=k_days, nodes=orinode)

    # Assign latitude and longitude to the sub-nodes
    subNode = graphScale._assign_latitude_longitude(subNode)

    # Assign departments to the sub-nodes
    #subNode = graphScale._assign_department(subNode)

    # Log information about the graph

    graphScale._info_on_graph(subNode, Path('log'))

    ################################## Try loading Y database #############################
    # Define the filename for the ground truth data
    n = f'Y_full_{scale}_{graph_construct}.pkl'
    #if True:
    if not (dir_output / n).is_file():
        # If the file doesn't exist, generate the ground truth data
        Y = get_sub_nodes_ground_truth(
            graphScale,
            subNode,
            departements,
            orinode,
            dir_output,
            dir_train,
            resolution,
            graph_construct
        )
        # Save the generated ground truth data
        save_object(Y, n, dir_output)
    else:
        # If the file exists, load the ground truth data
        Y = read_object(n, dir_output)

    ################################## Try loading X database #############################
    # Define the filename for the features data
    n = f'X_full_{scale}_{graph_construct}.pkl'
    if (dir_output / n).is_file():
        # Get the list of feature names
        features_name, _ = get_features_name_list(scale, features, METHODS_SPATIAL)
        # Load the features data
        X = read_object(n, dir_output)
    else:
        X = None  # If the file doesn't exist, set X to None

    # Make a copy of Y for backup or further processing
    Y_copy = np.copy(Y)

    ##################################### Not Using all nodes at all dates #########################
    # If we're not using all nodes at all dates, proceed with sampling
    # Split Y into training and testing sets based on the maxDate and trainCode
    y_train = Y[(Y[:, 4] < allDates.index(maxDate)) & (np.isin(Y[:, 3], trainCode))]
    y_test = Y[
        ((Y[:, 4] >= allDates.index(maxDate)) & (np.isin(Y[:, 3], trainCode))) |
        (~np.isin(Y[:, 3], trainCode))
    ]

    ################################# Class selection sample (specify maximum number of samples per class per node) #######################
    # Get the unique node IDs
    unode = np.unique(Y[:, 0])
    udetp = np.unique(Y[:, 3])
    if est_un_entier_chaine(values_per_class):
        # Initialize lists to store new Y and X samples
        new_Y = []
        sumi = 0  # Counter for total samples
        for node in udetp:
            # Get the department code for the current node
            dept = int(np.unique(Y[Y[:, 3] == node, 3])[0])
            for year in years:
                if sinister == 'firepoint':
                    # Define the fire season start and end dates for firepoints
                    fd = f'{year}-{SAISON_FEUX[dept]["mois_debut"]}-{SAISON_FEUX[dept]["jour_debut"]}'
                    ed = f'{year}-{SAISON_FEUX[dept]["mois_fin"]}-{SAISON_FEUX[dept]["jour_fin"]}'
                else:
                    # Use the full year for other sinister types
                    fd = f'{year}-01-01'
                    ed = f'{year}-12-31'
                # Ensure the dates exist in the date list
                if fd not in allDates:
                    fd = allDates[0]
                if ed not in allDates:
                    ed = allDates[-1]

                # Get the indices of the start and end dates
                ifd = allDates.index(fd)
                ied = allDates.index(ed)
                if len(PERIODES_A_IGNORER[dept]['interventions']) > 0:
                    # Exclude certain periods from consideration
                    date_to_ignore = []
                    for per in PERIODES_A_IGNORER[dept]['interventions']:
                        start = allDates.index(per[0].strftime('%Y-%m-%d'))
                        stop = allDates.index(per[0].strftime('%Y-%m-%d'))
                        date_to_ignore.append(np.arange(start, stop))
                    date_to_ignore =  np.asanyarray(date_to_ignore)
                    # Create a mask for the node within the date range and not in ignored dates
                    masknode = np.argwhere(
                        (y_train[:, 3] == node) &
                        (y_train[:, 4] >= ifd) &
                        (y_train[:, 4] <= ied) &
                        ~np.isin(y_train[:, 4], date_to_ignore)
                    )
                else:
                    # Create a mask for the node within the date range
                    masknode = np.argwhere(
                        (y_train[:, 3] == node) &
                        (y_train[:, 4] >= ifd) &
                        (y_train[:, 4] <= ied)
                    )
                if masknode.shape[0] == 0:
                    continue  # Skip if no samples are found
                # Get the unique classes for the current node
                classs = np.unique(y_train[masknode, -3])
                new_Y_index = []
                for cls in classs:
                    # Find indices where the class matches and there is a sinister event
                    mask_sinister = np.argwhere(
                        (y_train[masknode, -3] == cls) & (y_train[masknode, -2] > 0)
                    )[:, 0]
                    sumi += mask_sinister.shape[0]
                    Y_class_index = mask_sinister
                    if Y_class_index.shape[0] < int(values_per_class):
                        # If fewer samples than desired, include all and sample additional non-sinister events
                        number_non_sinister = abs(Y_class_index.shape[0] - int(values_per_class))
                        new_Y_index += list(Y_class_index)
                        Y_class_index = np.argwhere(
                            (y_train[masknode, -3] == cls) & (y_train[masknode, -2] == 0)
                        )[:, 0]
                        choices = np.random.choice(
                            Y_class_index,
                            min(Y_class_index.shape[0], int(number_non_sinister)),
                            replace=False
                        )
                        new_Y_index += list(choices)
                    else:
                        # Randomly select the desired number of samples
                        choices = np.random.choice(
                            Y_class_index,
                            min(Y_class_index.shape[0], int(values_per_class)),
                            replace=False
                        )
                        new_Y_index += list(choices)
                # Add the selected samples to the new lists
                new_Y += list(y_train[masknode][new_Y_index])

    ####################### Using a binary selection (same number of non-fire as fire) per node ########################
    elif values_per_class.find('binary') != -1:
        # Attempt to extract the factor from the values_per_class string
        try:
            factor = int(values_per_class.split('-')[-1])
        except:
            logger.info('Factor undefined, select with factor 1')
            factor = 1  # Default factor is 1
        new_Y = []
        sumi = 0  # Counter for total samples
        for node in udetp:
            # Get the department code for the current node
            dept = int(np.unique(Y[Y[:, 3] == node, 3])[0])
            for year in years:
                if sinister == 'firepoint':
                    # Define the fire season start and end dates for firepoints
                    fd = f'{year}-{SAISON_FEUX[dept]["mois_debut"]}-{SAISON_FEUX[dept]["jour_debut"]}'
                    ed = f'{year}-{SAISON_FEUX[dept]["mois_fin"]}-{SAISON_FEUX[dept]["jour_fin"]}'
                else:
                    # Use the full year for other sinister types
                    fd = f'{year}-01-01'
                    ed = f'{year}-12-31'
                # Ensure the dates exist in the date list
                if fd not in allDates:
                    fd = allDates[0]
                if ed not in allDates:
                    ed = allDates[-1]

                # Get the indices of the start and end dates
                ifd = allDates.index(fd)
                ied = allDates.index(ed)
                if len(PERIODES_A_IGNORER[dept]['interventions']) > 0:
                    # Exclude certain periods from consideration
                    date_to_ignore = []
                    for per in PERIODES_A_IGNORER[dept]['interventions']:
                        start = allDates.index(per[0].strftime('%Y-%m-%d'))
                        stop = allDates.index(per[0].strftime('%Y-%m-%d'))
                        date_to_ignore.append(np.arange(start, stop))
                    date_to_ignore = np.asanyarray(date_to_ignore)
                    # Create a mask for the node within the date range and not in ignored dates
                    masknode = np.argwhere(
                        (y_train[:, 3] == node) &
                        (y_train[:, 4] >= ifd) &
                        (y_train[:, 4] <= ied) &
                        (~np.isnan(y_train[:, -1])) &
                        ~np.isin(y_train[:, 4], date_to_ignore)
                    )
                else:
                    # Create a mask for the node within the date range
                    masknode = np.argwhere(
                        (y_train[:, 3] == node) &
                        (y_train[:, 4] >= ifd) &
                        (y_train[:, 4] <= ied) &
                        (~np.isnan(y_train[:, -1]))
                    )
                if masknode.shape[0] == 0:
                    continue  # Skip if no samples are found
                # Find indices for non-sinister and sinister events
                fire_dates = np.unique(y_train[masknode][y_train[masknode, -2] > 0][:, 4])
                td = 3
                for udates in fire_dates:
                    urange = np.arange(udates-td, udates+td)
                    fire_dates = np.concatenate((fire_dates, urange))
                fire_dates = np.unique(fire_dates)
                #mask_non_sinister = np.argwhere((y_train[masknode, -2] == 0) & ~(np.isin(y_train[masknode, 4], fire_dates)))[:, 0]
                mask_non_sinister = np.argwhere((y_train[masknode, -2] == 0))[:, 0]
                mask_sinister = np.argwhere(y_train[masknode, -2] > 0)[:, 0]
                number_of_samples = mask_sinister.shape[0]
                new_Y_index = []
                if number_of_samples != 0:
                    # Sample non-sinister events proportionally to the number of sinister events
                    choices = np.random.choice(
                        mask_non_sinister,
                        min(mask_non_sinister.shape[0], number_of_samples * factor),
                        replace=False
                    )
                else:
                    # If no sinister events, sample a fixed number of non-sinister events
                    choices = np.random.choice(
                        mask_non_sinister,
                        min(mask_non_sinister.shape[0], factor),
                        replace=False
                    )
                sumi += mask_sinister.shape[0]
                new_Y_index += list(choices)
                new_Y_index += list(mask_sinister)
                print(choices.shape, mask_sinister.shape, mask_non_sinister.shape, y_train[masknode, -2].shape)
                # Add the selected samples to the new lists
                new_Y += list(y_train[masknode][new_Y_index])
                
    elif values_per_class == 'full':
        new_Y = []
        new_Y_index = []
        for node in udetp:
            # Get the department code for the current node
            dept = int(np.unique(Y[Y[:, 3] == node, 3])[0])
            for year in years:
                if sinister == 'firepoint':
                    # Define the fire season start and end dates for firepoints
                    fd = f'{year}-{SAISON_FEUX[dept]["mois_debut"]}-{SAISON_FEUX[dept]["jour_debut"]}'
                    ed = f'{year}-{SAISON_FEUX[dept]["mois_fin"]}-{SAISON_FEUX[dept]["jour_fin"]}'
                else:
                    # Use the full year for other sinister types
                    fd = f'{year}-01-01'
                    ed = f'{year}-12-31'
                # Ensure the dates exist in the date list
                if fd not in allDates:
                    fd = allDates[0]
                if ed not in allDates:
                    ed = allDates[-1]

                # Get the indices of the start and end dates
                ifd = allDates.index(fd)
                ied = allDates.index(ed)
                if len(PERIODES_A_IGNORER[dept]['interventions']) > 0:
                    # Exclude certain periods from consideration
                    date_to_ignore = []
                    for per in PERIODES_A_IGNORER[dept]['interventions']:
                        start = allDates.index(per[0].strftime('%Y-%m-%d'))
                        stop = allDates.index(per[0].strftime('%Y-%m-%d'))
                        date_to_ignore.append(np.arange(start, stop))
                    date_to_ignore = np.asanyarray(date_to_ignore)
                    # Create a mask for the node within the date range and not in ignored dates
                    masknode = np.argwhere(
                        (y_train[:, 3] == node) &
                        (y_train[:, 4] >= ifd) &
                        (y_train[:, 4] <= ied) &
                        ~np.isin(y_train[:, 4], date_to_ignore)
                    )
                else:
                    # Create a mask for the node within the date range
                    masknode = np.argwhere(
                        (y_train[:, 3] == node) &
                        (y_train[:, 4] >= ifd) &
                        (y_train[:, 4] <= ied)
                    )
                new_Y += list(y_train[masknode])

    # Convert the lists to numpy arrays
    new_Y = np.asarray(new_Y).reshape(-1, y_train.shape[-1])
    # Create a mask to include neighboring days within k_days
    mask_days = np.zeros(y_train.shape[0], dtype=bool)
    for k in range(1, k_days + 2):
        mask_days = mask_days | np.isin(y_train[:, 4], new_Y[:, 4] - k)

    # Get the neighboring samples
    new_Y_neighboor = y_train[mask_days]

    set1 = set(tuple(sub_arr) for sub_arr in new_Y_neighboor)
    set2 = set(tuple(sub_arr) for sub_arr in new_Y)
    
    # Find intersection
    intersection = set1.intersection(set2)

    # Remove intersection elements from both arrays
    new_Y_neighboor = np.asarray([sub_arr for sub_arr in new_Y_neighboor if tuple(sub_arr) not in intersection])
    new_Y_neighboor[:, 5] = 0  # Reset weight column in the data
 
    # Concatenate the new and neighboring Y samples with the test set
    #Y = np.concatenate((new_Y, y_test), casting='no')
    Y = np.concatenate((new_Y, new_Y_neighboor, y_test), casting='safe')
    
    # Remove duplicate rows from Y
    Y = np.unique(Y, axis=0)
    if X is None:
        # If X is not loaded, generate the features
        X, features_name = get_sub_nodes_feature(
            graphScale,
            Y[:, :6],
            departements,
            features,
            sinister,
            dataset_name,
            sinister_encoding,
            dir_output,
            dir_train,
            resolution,
            graph_construct
        )
    else:
        # Align X with Y based on node IDs and dates
        X_ = np.empty((Y.shape[0], X.shape[1]))
        for i, instance in enumerate(Y):
            instance_index = np.argwhere(
                (X[:, 0] == instance[0]) & (X[:, 4] == instance[4])
            )[:, 0]
            if instance_index.shape[0] >= 2:
                instance_index = instance_index[0]
            X_[i] = X[instance_index]

        X = X_

        # Boucle sur chaque noeud unique dans la première colonne de X
        for node in np.unique(X[:, 0]):
            # Créer un masque pour sélectionner les lignes correspondant à ce noeud
            mask_node = (X[:, 0] == node)
            
            # Boucle sur chaque colonne (à partir de la 6ème) pour l'interpolation
            for band in range(6, X.shape[1]):
                # Extraire les valeurs non-NaN pour l'interpolation
                x = X[mask_node & ~np.isnan(X[:, band]), 4]  # Dates non-NaN
                y = X[mask_node & ~np.isnan(X[:, band]), band]  # Valeurs non-NaN correspondantes
                if x.shape[0] == 0:
                    continue
                # Vérifier s'il reste des NaN à interpoler dans cette bande
                nan_mask = mask_node & np.isnan(X[:, band])
                if np.any(nan_mask):
                    nan_date = X[nan_mask, 4]  # Dates où il y a des NaN
                    
                    # Créer une fonction d'interpolation
                    f = scipy.interpolate.interp1d(x, y, kind='nearest', fill_value='extrapolate')
                    
                    # Calculer les nouvelles valeurs interpolées
                    new_values = f(nan_date)
                    
                    # Remplacer les NaN par les nouvelles valeurs interpolées
                    X[nan_mask, band] = new_values

    # Sort Y and X based on node IDs and dates
    ind = np.lexsort([Y[:, 4], Y[:, 0]])
    Y = Y[ind]
    if X is not None:
        X = X[ind]

    logger.info(f'{X.shape, Y.shape}')
    # Extract the training samples from Y
    y_train = Y[(Y[:, 4] < allDates.index(maxDate)) & (np.isin(Y[:, 3], trainCode))]
    print(y_train.shape, Y.shape, new_Y.shape)
    # Print unique values and sums for verification
    print(np.unique(y_train[:, -2]))
    print(np.sum(y_train[y_train[:, 5] > 0, -2]))

    # Log the number of events and non-events used for training
    logger.info(
        f'Number of event used for training {np.argwhere((y_train[:, -2] > 0) & (y_train[:, 5] > 0)).shape}'
    )
    logger.info(
        f'Number of non event used for training {np.argwhere((y_train[:, -2] == 0) & (y_train[:, 5] > 0)).shape}'
    )
    # Log the number of samples per class
    for c in np.unique(y_train[:, -3]):
        logger.info(
            f'{c} : , {np.argwhere((y_train[:, -3] == c) & (y_train[:, 5] > 0)).shape}'
        )

    # Return the features, ground truth, and feature names
    return X, Y, features_name

def construct_non_point(firepoints, regions, maxDate, sinister, dir):
    nfps = []
    for dept in firepoints.departement.unique():
        if dept == 'departelent-01-ain':
            ad = find_dates_between('2018-01-01', maxDate)
        else:
            ad = allDates
            
        reg = regions[regions['departement'] == dept]
        fp = firepoints[firepoints['departement'] == dept]

        non_fire_date = [date for date in ad if date not in fp.date]
        non_fire_h3 = reg[~np.isin(reg['hex_id'], fp.h3)][['latitude', 'longitude']]
     
        # Temporal
        iterables = [non_fire_date, list(fp.latitude)]
        nfp = pd.DataFrame(index=pd.MultiIndex.from_product(iterables, names=('date', 'latitude'))).reset_index()

        iterables = [non_fire_date, list(fp.longitude)]
        ps2 = pd.DataFrame(index=pd.MultiIndex.from_product(iterables, names=('date', 'longitude'))).reset_index()
        nfp['longitude'] = ps2['longitude']
        del ps2
        
        # Spatial
        iterables = [fp.date, list(non_fire_h3.latitude)]
        ps3 = pd.DataFrame(index=pd.MultiIndex.from_product(iterables, names=('date', 'latitude'))).reset_index()

        iterables = [fp.date, list(non_fire_h3.longitude)]
        ps2 = pd.DataFrame(index=pd.MultiIndex.from_product(iterables, names=('date', 'longitude'))).reset_index()
        ps3['longitude'] = ps2['longitude']
        del ps2
        
        # Final
        nfp = pd.concat((nfp, ps3)).reset_index(drop=True)
        nfp['departement'] = dept
        nfps.append(nfp)

    nfps = pd.concat(nfps).reset_index(drop=True).sample(2 * len(firepoints), random_state=42)
    print(f'Size of non {sinister} points : {len(nfps)}')

    name = 'non'+sinister+'.csv'
    nfps.to_csv(dir / sinister / name, index=False)

    name = sinister+'.csv'
    firepoints.to_csv(dir / sinister / name, index=False)

def init(args, dir_output, script):
    
    global train_features
    global features

    ######################### Input config #############################
    dataset_name = args.dataset
    name_exp = args.name
    maxDate = args.maxDate
    trainDate = args.trainDate
    doEncoder = args.encoder == "True"
    doPoint = args.point == "True"
    doGraph = args.graph == "True"
    doDatabase = args.database == "True"
    do2D = args.database2D == "True"
    doFet = args.featuresSelection == "True"
    optimize_feature = args.optimizeFeature == "True"
    doTest = args.doTest == "True"
    doTrain = args.doTrain == "True"
    sinister = args.sinister
    values_per_class = args.nbpoint
    scale = int(args.scale) if args.scale != 'departement' else args.scale
    resolution = args.resolution
    doPCA = args.pca == 'True'
    doKMEANS = args.KMEANS == 'True'
    ncluster = int(args.ncluster)
    k_days = int(args.k_days) # Size of the time series sequence use by DL models
    days_in_futur = int(args.days_in_futur) # The target time validation
    dir_output = dir_output
    scaling = args.scaling
    graph_construct = args.graphConstruct
    dataset_name = args.dataset
    sinister_encoding = args.sinisterEncoding
    weights_version = args.weights
    top_cluster = args.top_cluster

    ######################## Get features and train features list ######################

    isInference = name_exp == 'inference'

    features, train_features, kmeans_features = get_features_for_sinister_prediction(dataset_name, sinister, isInference)

    ######################## Get departments and train departments #######################

    departements, train_departements = select_departments(dataset_name, sinister)

    ######################## CONFIG ################################

    if dataset_name == 'firemen2':
        two = True
        dataset_name = 'firemen'
    else:
        two = False

    name_exp = f'{sinister_encoding}_{name_exp}'

    dir_target = root_target / sinister / dataset_name / sinister_encoding / 'log' / resolution

    geo = gpd.read_file(f'regions/{sinister}/{dataset_name}/regions.geojson')
    geo = geo[geo['departement'].isin(departements)].reset_index(drop=True)

    minDate = '2017-06-12' # Starting point

    if values_per_class == 'full':
        prefix = f'{values_per_class}'
    else:
        prefix = f'{values_per_class}_{k_days}'

    prefix += f'_{scale}_{graph_construct}'

    autoRegression = 'AutoRegressionReg' in train_features
    if autoRegression:
        name_exp += '_AutoRegressionReg'

    ########################## Do Graph ######################################

    if doGraph:
        logger.info('#################################')
        logger.info('#      Construct   Graph        #')
        logger.info('#################################')

        graphScale = construct_graph(
                                    train_departements=train_departements,
                                    scale=scale,
                                     maxDist=maxDist[scale],
                                    sinister=sinister,
                                    dataset_name=dataset_name,
                                    sinister_encoding=sinister_encoding,
                                    geo=geo, nmax=nmax, k_days=k_days,
                                    dir_output=dir_output,
                                    doRaster=True,
                                    doEdgesFeatures=False,
                                    resolution=resolution,
                                    graph_construct=graph_construct,
                                    train_date=trainDate,
                                    val_date=maxDate,
                                    )

        #graphScale._clusterize_node(train_departements, ['population', 'foret'], maxDate, dir_output, rootDisk / 'csv')

        """graphScale._plot_risk(mode='time_series_risk', dir_output=dir_output)
        graphScale._plot_risk(mode='2022-07-18_2022-07-18', dir_output=dir_output, path=dir_output)
        graphScale._plot_risk(mode='2022-07-12_2022-07-12', dir_output=dir_output, path=dir_output)
        graphScale._plot_risk(mode='2023-03-01_2023-09-01', dir_output=dir_output, path=dir_output)
        graphScale._plot_risk(mode='2022-03-01_2022-09-01', dir_output=dir_output, path=dir_output)"""

        """graphScale._plot_risk_departement(mode='time_series_class', dir_output=dir_output)
        graphScale._plot_risk_departement(mode='time_series_risk', dir_output=dir_output)
        graphScale._plot_risk_departement(mode='2022-07-18_2022-07-18', dir_output=dir_output, path=dir_output)
        graphScale._plot_risk_departement(mode='2022-07-12_2022-07-12', dir_output=dir_output, path=dir_output)
        graphScale._plot_risk_departement(mode='2022-03-01_2022-09-01', dir_output=dir_output, path=dir_output)"""

    else:
        graphScale = read_object(f'graph_{scale}_{graph_construct}.pkl', dir_output)

    departements = [dept for dept in departements if dept not in graphScale.drop_department] 
    train_departements = [dept for dept in train_departements if dept not in graphScale.drop_department] 

    graphScale._plot(graphScale.nodes, dir_output=dir_output)

    ########################### Create points ################################
    fp = pd.read_csv(f'sinister/{dataset_name}/{sinister}.csv', dtype=str)
    if doPoint:
        
        logger.info('#####################################')
        logger.info('#      Get hexagones point          #')
        logger.info('#      for departement              #')
        logger.info('#####################################')

        check_and_create_path(dir_output)

        #construct_non_point(fp, geo, maxDate, sinister, Path(dataset_name))
        look_for_information(graphScale, dataset_name,
                         maxDate,
                         sinister,
                         Path(dataset_name))

    ######################### Encoding ######################################

    if doEncoder:
        logger.info('#####################################')
        logger.info('#      Calcualte Encoder            #')
        logger.info('#####################################')
        encode(dir_target, trainDate, train_departements, dir_output / 'Encoder', resolution, graphScale)

    graphScale._plot_risk(mode='time_series_class', dir_output=dir_output)

    ########################## Do Database ####################################

    if doDatabase:
        logger.info('#####################################')
        logger.info('#      Construct   Database         #')
        logger.info('#####################################')
        if not dummy:
            name = 'points.csv'
            ps = pd.read_csv(Path(dataset_name) / sinister / name)
            depts = [name2int[dept] for dept in departements]
            logger.info(ps.departement.unique())
            ps = ps[ps['departement'].isin(depts)].reset_index(drop=True)
            logger.info(ps.departement.unique())
        else:
            name = sinister+'.csv'
            fp = pd.read_csv(Path(dataset_name) / sinister / name)
            name = 'non'+sinister+'csv'
            nfp = pd.read_csv(Path(name_exp) / sinister / name)
            ps = pd.concat((fp, nfp)).reset_index(drop=True)
            ps = ps.copy(deep=True)
            ps = ps[(ps['date'].isin(allDates)) & (ps['date'] >= '2017-06-12')]
            ps['date'] = [allDates.index(date) for date in ps.date if date in allDates]

        X, Y, features_name = construct_database(graphScale,
                                            ps, scale, k_days,
                                            departements,
                                            features,
                                            sinister,
                                            dataset_name if not two else f'{dataset_name}2',
                                            sinister_encoding,
                                            dir_output,
                                            dir_output,
                                            prefix,
                                            values_per_class,
                                            'train',
                                            resolution,
                                            trainDate,
                                            graph_construct)
        
        save_object(X, 'X_'+prefix+'.pkl', dir_output)
        save_object(Y, 'Y_'+prefix+'.pkl', dir_output)
    else:
        X = read_object('X_'+prefix+'.pkl', dir_output)
        Y = read_object('Y_'+prefix+'.pkl', dir_output)
        features_name, newshape = get_features_name_list(graphScale.scale, features, METHODS_SPATIAL)
        print(features_name)

    X = X[:, 6:]

    if newFeatures != []:
        X2, features_name_ = get_sub_nodes_feature(graphScale, Y[:, :6], departements, newFeatures, sinister, dir_output, dir_output, resolution, graph_construct)
        for fet in newFeatures:
            start , maxi, _, _ = calculate_feature_range(fet, scale, METHODS_SPATIAL)
            X[:, features_name.index(start): features_name.index(start) + maxi] = X2[:, features_name_.index(start): features_name_.index(start) + maxi]
        save_object(X, 'X_'+prefix+'.pkl', dir_output)
        X = X[:, 6:]

    ############################## Dataframe creation ###################################

    prefix = f'{values_per_class}_{k_days}_{scale}_{graph_construct}_{top_cluster}'

    if (dir_output / f'df_{prefix}.pkl').is_file():
        df = read_object(f'df_{prefix}.pkl', dir_output)
        find_df = not doDatabase

    if True:
        df = pd.DataFrame(columns=ids_columns + targets_columns + features_name, index=np.arange(0, X.shape[0]))
        df[features_name] = X
        df[ids_columns[:-1] + targets_columns] = Y
        find_df = False

    print(len(df[(df['weight'] >  0) & (df['date'] < allDates.index('2022-01-01'))]))

    ############################## Interpolate missing value #########################

    total_nan = df[features_name].isna().sum().sum()

    print(total_nan)

    ####################### Create risk target throught time ##############################

    if not find_df:
        df = graphScale.compute_mean_sequence(df.copy(deep=True), dataset_name, maxDate)

    if top_cluster is not None:
        uvalues = df.cluster_encoder.unique()
        uvalues = np.sort(uvalues)
        top_x = int(top_cluster.split('-')[1])
        thresh = uvalues[-top_x]
        #df.loc[df[df['cluster_encoder'] >= thresh].index, 'weight'] = 0
        print(uvalues, thresh)
        df = df[df['cluster_encoder'] >= thresh]
        uvalues = df.cluster_encoder.unique()
        print(uvalues, thresh)

    print(len(df[(df['weight'] >  0) & (df['date'] < allDates.index('2022-01-01'))]))

    ############################## Global variable #######################################

    global varying_time_variables
    global varying_time_variables_name

    ############################ Frequency ratio ###########################################
    if doKMEANS:
        trainCode = [name2int[d] for d in train_departements]
        train_mask = (df['date'] < allDates.index(trainDate)) & (df['departement'].isin(trainCode))

        features_selected_kmeans, _ = get_features_name_list(scale, features, METHODS_KMEANS)
        train_break_point(df[train_mask].copy(deep=True), features_name, dir_output / 'check_none' / prefix / 'kmeans', ncluster)

        if not find_df:
            features_selected_kmeans, _ = get_features_name_list(scale, kmeans_features, METHODS_KMEANS)
            df = apply_kmeans_class_on_target(df.copy(deep=True), dir_output / 'check_none' / prefix / 'kmeans', 'risk', tresh_kmeans, features_selected_kmeans, new_val=0)
            df = apply_kmeans_class_on_target(df.copy(deep=True), dir_output / 'check_none' / prefix / 'kmeans', 'class_risk', tresh_kmeans, features_selected_kmeans, new_val=0)
            #df = apply_kmeans_class_on_target(df.copy(deep=True), dir_output / 'check_none' / prefix / 'kmeans', 'nbsinister', tresh_kmeans, features_selected_kmeans, new_val=0)
            df = apply_kmeans_class_on_target(df.copy(deep=True), dir_output / 'check_none' / prefix / 'kmeans', 'weight', tresh_kmeans, features_selected_kmeans, new_val=0)
        
        features_selected_kmeans, _ = get_features_name_list(scale, train_features, METHODS_KMEANS)
        
        #df, new_fet_fr = add_fr_variables(df.copy(deep=True), dir_output / 'check_none' / prefix / 'kmeans', features_selected_kmeans)

        #new_varying_time_variables = []

        #features_name += new_fet_fr
        #for col1 in varying_time_variables:
        #    col, met = col1.split('_')
        #    new_varying_time_variables += [f'{col}_frequencyratio_{met}']
            
        #varying_time_variables += new_varying_time_variables

    ################################ Create class ##################################################################

    df = graphScale._create_predictor(df.copy(deep=True), minDate, maxDate, dir_output)
    print(len(df[(df['weight'] >  0) & (df['date'] < allDates.index('2022-01-01'))]))

    ############################## Add varying time features #############################
 
    if k_days > 0:
        df, new_fet_time = add_time_columns(varying_time_variables, k_days, df.copy(deep=True), train_features)
        varying_time_variables_name = new_fet_time
    print(len(df[(df['weight'] >  0) & (df['date'] < allDates.index('2022-01-01'))]))

        ############################## Add weight columns #############################
    for weight_col in weights_columns:
        df[weight_col] = calculate_weighs(weight_col, df.copy(deep=True))
        df[weight_col] = df['weight'] * df[weight_col].values

    print(len(df[(df['weight_one'] >  0) & (df['date'] < allDates.index('2022-01-01'))]))
    print(len(Y[(Y[:, 5] >  0) & (Y[:, 4] < allDates.index('2022-01-01'))]))

    df.drop_duplicates(inplace=True)

    ############################## Add survival column #############################

    df['days_until_next_event'] = calculate_days_until_next_event(df['id'].values, df['date'].values, df['nbsinister'].values)

    ############################## Update features list #############################
    features_name, _ = get_features_name_list(scale, train_features, METHODS_SPATIAL_TRAIN)
    
    if 'new_fet_fr' in locals():
        features_name += varying_time_variables_name + new_fet_fr
    else:
        features_name += varying_time_variables_name

    ############################## Save dataframe ###################################

    save_object(df, 'df_'+prefix+'.pkl', dir_output)

    ################################# 2D Database ###################################
    
    if do2D:
        features_name_2D, newShape2D = get_sub_nodes_feature_2D(graphScale, df, departements, features,
                                                                sinister, dataset_name, dir_output, dir_output,
                                                                resolution, graph_construct, sinister_encoding)
    else:
        features_name_2D, newShape2D = get_features_name_lists_2D(df.shape[1], features)

    return df, graphScale, prefix, fp, list(np.unique(features_name))