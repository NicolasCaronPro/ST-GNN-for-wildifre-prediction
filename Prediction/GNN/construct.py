import pickle
from re import subn
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

def construct_graph(scale, maxDist, sinister, dataset_name, sinister_encoding, train_departements, departements,
                    geo, nmax, k_days, dir_output, doRaster, doEdgesFeatures, resolution, graph_construct, train_date, val_date, graph_method):
    
    graphScale = GraphStructure(scale, geo, maxDist, nmax, resolution, graph_construct, sinister, sinister_encoding, dataset_name, train_departements, graph_method)

    sucseptibility_map_model_config = {'type': 'Unet',
                                       'device': 'cuda',
                                       'task': 'regression',
                                       'loss' : 'rmse',
                                       'name': 'mapper',
                                       'params' : None}
    
    variables_for_susecptibilty_and_clustering = ['population', 'foret', 'osmnx', 'elevation',
                                                  #'vigicrues',
                                                  #'nappes',
                                                  'temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
                                            'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                                            'isi', 'angstroem', 'bui', 'fwi', 'dailySeverityRating',
                                            'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
                                            'days_since_rain', 'sum_consecutive_rainfall',
                                            'sum_rain_last_7_days',
                                            'sum_snow_last_7_days', 'snow24h', 'snow24h16']
    if dataset_name == 'bdiff':
        graphScale.train_susecptibility_map(model_config=sucseptibility_map_model_config,
                                                departements=departements,
                                                variables=variables_for_susecptibilty_and_clustering, target='risk', train_date=train_date, val_date=val_date,
                                                root_data=rootDisk / 'csv',
                                                root_target=root_target / sinister / dataset_name / sinister_encoding,
                                                dir_output=dir_output)
    #else:
    #pass
        
    graphScale._create_sinister_region(base=graph_construct,
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
    save_object(graphScale, f'graph_{scale}_{graph_construct}_{graphScale.graph_method}.pkl', dir_output)
    return graphScale

def export_to_all_date(df, dataset_name, sinister, departements, maxDate):
    res = []
    for departement in departements:
        if dataset_name in ['firemen', 'Ain', 'Doubs', 'Rhone', 'Yvelines']:
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

def process_target(df, graphScale, prefix, find_df, minDate, departements, train_departements, features_name, kmeans_features, features, dir_output, args):

    ######################### Input config #############################
    dataset_name = args.dataset
    maxDate = args.maxDate
    trainDate = args.trainDate
    do2D = args.database2D == "True"
    sinister = args.sinister
    scale = int(args.scale) if args.scale != 'departement' else args.scale
    resolution = args.resolution
    ncluster = int(args.ncluster)
    shift = int(args.shift) 
    graph_construct = args.graphConstruct
    dataset_name = args.dataset
    sinister_encoding = args.sinisterEncoding
    graph_method = args.graph_method

    trainCode = [name2int[d] for d in train_departements]
    train_mask = (df['date'] < allDates.index(trainDate)) & (df['departement'].isin(trainCode))
    target_column_name_list = ['0_0']
    df['nbsinister_0_0'] = df['nbsinister'].values
    limit_day = 9
    shift_list = np.arange(0, 1)
    thresh_kmeans_list = np.arange(0.1, 1, 0.1)
    ############################ Frequency ratio / removing outliers ###########################################

    if (dir_output / f'df_no_weight_{prefix}.pkl').is_file():
        for thresh in thresh_kmeans_list:
            target_column_name_list += [f'{s}_{thresh}' for s in shift_list]
            
        df = read_object(f'df_no_weight_{prefix}.pkl', dir_output)
    else:    
        if dataset_name == 'firemen':
            features_selected_kmeans, _ = get_features_name_list(scale, features, METHODS_KMEANS)
            train_break_point(df[train_mask].copy(deep=True), features_name, dir_output / 'check_none' / prefix / 'kmeans', ncluster, shift_list)
            features_selected_kmeans, _ = get_features_name_list(scale, kmeans_features, METHODS_KMEANS)
            mask = (df['date'] < allDates.index(maxDate)) & (df['departement'].isin(trainCode)) # Train Val mask

            for thresh in thresh_kmeans_list:
                df = apply_kmeans_class_on_target(df.copy(deep=True), dir_output / 'check_none' / prefix / 'kmeans', 'nbsinister', thresh, features_selected_kmeans, new_val=0, shifts=shift_list, mask_df=mask)
                #df = apply_kmeans_class_on_target(df.copy(deep=True), dir_output / 'check_none' / prefix / 'kmeans', 'AutoRegressionBin-B-1', thresh, features_selected_kmeans, new_val=0, shifts=shift_list, mask_df=mask)
                target_column_name_list += [f'{s}_{thresh}' for s in shift_list]

        #mask = ((df['date'] >= allDates.index(maxDate) + k_days) & (df['departement'].isin(trainCode))) | (~df['departement'].isin(trainCode)) # Test mask
        #df = apply_kmeans_class_on_target(df.copy(deep=True), dir_output / 'check_none' / prefix / 'kmeans', 'nbsinister', thresh_kmeans, features_selected_kmeans, new_val=0, shifts=shift_list, mask_df=mask)

        ############################## Add survival column #############################

        for target_spe in target_column_name_list:
            df[f'days_until_next_event_{target_spe}'] = calculate_days_until_next_event(df['id'].values, df['date'].values, df[f'nbsinister_{target_spe}'].values)

        df[f'days_until_next_event'] = df[f'days_until_next_event_0_0'].values

        ####################### Create risk target throught time ##############################
        assert graphScale is not None
        for target_spe in target_column_name_list:
            df = graphScale.compute_mean_sequence(df.copy(deep=True), dataset_name, maxDate, target_spe)
            #df = graphScale.compute_window_class(df, [1], 5, aggregate_funcs=['sum', 'mean', 'max', 'min', 'grad', 'std'], mode='train', column=f'nbsinister_{target_spe}', dir_output=dir_output / 'class_window' / prefix / f'nbsinister_{target_spe}')
            #df = graphScale.compute_window_class(df, [1], 5, aggregate_funcs=['sum', 'mean', 'max', 'min', 'grad', 'std'], mode='train', column=f'risk_{target_spe}', dir_output=dir_output / 'class_window' / prefix / f'risk_{target_spe}')
        
        df['risk'] = df['risk_0_0'].values
        for target_spe in target_column_name_list:
            #df = graphScale._create_predictor(df.copy(deep=True), minDate, maxDate, dir_output, target_spe)
             df[f'class_risk_{target_spe}'] = 0

        ############################## Global variable #######################################

        global varying_time_variables
        global varying_time_variables_name

        ########################################### Add futur risk/nbsinister #############################
        for target_spe in target_column_name_list:
            logger.info(f'Add {target_spe} in {limit_day} days in future')
            df = target_by_day(df, limit_day, futur_met, target_spe)

        save_object(df, f'df_no_weight_{prefix}.pkl', dir_output)

    return df

def construct_database(
    graphScale: GraphStructure,
    ps: pd.DataFrame,
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
):
    
    scale = graphScale.scale
    graph_construct = graphScale.base
    graph_method = graphScale.graph_method
    
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
    
    ps[f'graph_{scale}'], ps[f'scale{scale}'] = graphScale._predict_node_graph_with_position(X_kmeans, ps.departement)
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
    orinode = np.full((len(ps), len(ids_columns) - 1), -1.0, dtype=float)
    orinode[:, graph_id_index] = ps[f'graph_{scale}'].values  # Assign node IDs
    orinode[:, id_index] = ps[f'scale{scale}'].values  # Assign node IDs
    orinode[:, departement_index] = ps['departement']
    orinode[:, date_index] = ps['date']  # Assign dates
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
    n = f'Y_full_{scale}_{graph_construct}_{graph_method}.pkl'
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
        )
        # Save the generated ground truth data
        save_object(Y, n, dir_output)
    else:
        # If the file exists, load the ground truth data
        Y = read_object(n, dir_output)

    ################################## Try loading X database #############################
    # Define the filename for the features data
    n = f'X_full_{scale}_{graph_construct}_{graph_method}.pkl'
    if (dir_output / n).is_file():
        # Get the list of feature names
        features_name, _ = get_features_name_list(scale, features, METHODS_SPATIAL)
        # Load the features data
        X = read_object(n, dir_output)
    else:
        X = None  # If the file doesn't exist, set X to None

    ##################################### Not Using all nodes at all dates #########################
    # If we're not using all nodes at all dates, proceed with sampling
    # Split Y into training and testing sets based on the maxDate and trainCode
    y_train = Y[(Y[:, date_index] < allDates.index(maxDate)) & (np.isin(Y[:, departement_index], trainCode))]
    y_test = Y[
        ((Y[:, date_index] >= allDates.index(maxDate)) & (np.isin(Y[:, departement_index], trainCode))) |
        (~np.isin(Y[:, departement_index], trainCode))
    ]

    ################################# Class selection sample (specify maximum number of samples per class per node) #######################
    # Get the unique node IDs
    unode = np.unique(Y[:, id_index])
    udetp = np.unique(Y[:, departement_index])
    if est_un_entier_chaine(values_per_class):
        # Initialize lists to store new Y and X samples
        new_Y = []
        sumi = 0  # Counter for total samples
        for node in udetp:
            # Get the department code for the current node
            dept = int(np.unique(Y[Y[:, departement_index] == node, departement_index])[0])
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
                        (y_train[:, departement_index] == node) &
                        (y_train[:, date_index] >= ifd) &
                        (y_train[:, date_index] <= ied) &
                        ~np.isin(y_train[:, date_index], date_to_ignore)
                    )
                else:
                    # Create a mask for the node within the date range
                    masknode = np.argwhere(
                        (y_train[:, departement_index] == node) &
                        (y_train[:, date_index] >= ifd) &
                        (y_train[:, date_index] <= ied)
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
            dept = int(np.unique(Y[Y[:, departement_index] == node, departement_index])[0])
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
                        (y_train[:, departement_index] == node) &
                        (y_train[:, date_index] >= ifd) &
                        (y_train[:, date_index] <= ied) &
                        (~np.isnan(y_train[:, -1])) &
                        ~np.isin(y_train[:, date_index], date_to_ignore)
                    )
                else:
                    # Create a mask for the node within the date range
                    masknode = np.argwhere(
                        (y_train[:, departement_index] == node) &
                        (y_train[:, date_index] >= ifd) &
                        (y_train[:, date_index] <= ied) &
                        (~np.isnan(y_train[:, -1]))
                    )
                if masknode.shape[0] == 0:
                    continue  # Skip if no samples are found
                # Find indices for non-sinister and sinister events
                fire_dates = np.unique(y_train[masknode][y_train[masknode, -2] > 0][:, date_index])
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
            dept = int(np.unique(Y[Y[:, departement_index] == node, departement_index])[0])
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
                        (y_train[:, departement_index] == node) &
                        (y_train[:, date_index] >= ifd) &
                        (y_train[:, date_index] <= ied) &
                        ~np.isin(y_train[:, date_index], date_to_ignore)
                    )
                else:
                    # Create a mask for the node within the date range
                    masknode = np.argwhere(
                        (y_train[:, departement_index] == node) &
                        (y_train[:, date_index] >= ifd) &
                        (y_train[:, date_index] <= ied)
                    )
                new_Y += list(y_train[masknode])

    # Convert the lists to numpy arrays
    new_Y = np.asarray(new_Y).reshape(-1, y_train.shape[-1])
    # Create a mask to include neighboring days within k_days
    mask_days = np.zeros(y_train.shape[0], dtype=bool)
    for k in range(1, k_days + 2):
        mask_days = mask_days | np.isin(y_train[:, date_index], new_Y[:, date_index] - k)

    # Get the neighboring samples
    new_Y_neighboor = y_train[mask_days]

    set1 = set(tuple(sub_arr) for sub_arr in new_Y_neighboor)
    set2 = set(tuple(sub_arr) for sub_arr in new_Y)
    
    # Find intersection
    intersection = set1.intersection(set2)

    # Remove intersection elements from both arrays
    new_Y_neighboor = np.asarray([sub_arr for sub_arr in new_Y_neighboor if tuple(sub_arr) not in intersection])
    new_Y_neighboor[:, weight_index] = 0  # Reset weight column in the data
 
    # Concatenate the new and neighboring Y samples with the test set
    #Y = np.concatenate((new_Y, y_test), casting='no')
    Y = np.concatenate((new_Y, new_Y_neighboor, y_test), casting='safe')
    
    # Remove duplicate rows from Y
    Y = np.unique(Y, axis=0)
    if X is None:
        # If X is not loaded, generate the features
        X, features_name = get_sub_nodes_feature(
            graphScale,
            Y[:, :len(ids_columns) - 1],
            departements,
            features,
            sinister,
            dataset_name,
            sinister_encoding,
            dir_output,
            dir_train,
            resolution,
        )
    else:
        # Align X with Y based on node IDs and dates
        X_ = np.empty((Y.shape[0], X.shape[1]))
        for i, instance in enumerate(Y):
            instance_index = np.argwhere(
                (X[:, id_index] == instance[id_index]) & (X[:, date_index] == instance[date_index])
            )[:, 0]
            if instance_index.shape[0] >= 2:
                instance_index = instance_index[0]
            X_[i] = X[instance_index]

        X = X_

        # Boucle sur chaque noeud unique dans la première colonne de X
        for node in np.unique(X[:, id_index]):
            # Créer un masque pour sélectionner les lignes correspondant à ce noeud
            mask_node = (X[:, id_index] == node)
            
            # Boucle sur chaque colonne (à partir de la 6ème) pour l'interpolation
            for band in range(len(ids_columns), X.shape[1]):
                # Extraire les valeurs non-NaN pour l'interpolation
                x = X[mask_node & ~np.isnan(X[:, band]), date_index]  # Dates non-NaN
                y = X[mask_node & ~np.isnan(X[:, band]), band]  # Valeurs non-NaN correspondantes
                if x.shape[0] == 0:
                    continue
                # Vérifier s'il reste des NaN à interpoler dans cette bande
                nan_mask = mask_node & np.isnan(X[:, band])
                if np.any(nan_mask):
                    nan_date = X[nan_mask, date_index]  # Dates où il y a des NaN
                    
                    # Créer une fonction d'interpolation
                    f = scipy.interpolate.interp1d(x, y, kind='nearest', fill_value='extrapolate')
                    
                    # Calculer les nouvelles valeurs interpolées
                    new_values = f(nan_date)
                    
                    # Remplacer les NaN par les nouvelles valeurs interpolées
                    X[nan_mask, band] = new_values

    # Sort Y and X based on node IDs and dates
    ind = np.lexsort([Y[:, date_index], Y[:, id_index]])
    Y = Y[ind]
    if X is not None:
        X = X[ind]

    logger.info(f'{X.shape, Y.shape}')
    # Extract the training samples from Y
    y_train = Y[(Y[:, date_index] < allDates.index(maxDate)) & (np.isin(Y[:, departement_index], trainCode))]
    print(y_train.shape, Y.shape, new_Y.shape)
    # Print unique values and sums for verification
    print(np.unique(y_train[:, -2]))
    print(np.sum(y_train[y_train[:, weight_index] > 0, -2]))

    # Log the number of events and non-events used for training
    logger.info(
        f'Number of event used for training {np.argwhere((y_train[:, -2] > 0) & (y_train[:, weight_index] > 0)).shape}'
    )
    logger.info(
        f'Number of non event used for training {np.argwhere((y_train[:, -2] == 0) & (y_train[:, weight_index] > 0)).shape}'
    )
    # Log the number of samples per class
    for c in np.unique(y_train[:, -3]):
        logger.info(
            f'{c} : , {np.argwhere((y_train[:, -3] == c) & (y_train[:,weight_index] > 0)).shape}'
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
    nbfeatures = args.NbFeatures
    shift = int(args.shift) 
    k_days = int(args.k_days) # Size of the time series sequence use by DL models
    days_in_futur = int(args.days_in_futur) # The target time validation
    dir_output = dir_output
    scaling = args.scaling
    graph_construct = args.graphConstruct
    dataset_name = args.dataset
    sinister_encoding = args.sinisterEncoding
    weights_version = args.weights
    top_cluster = args.top_cluster
    graph_method = args.graph_method
    thresh_kmeans = float(args.thresh_kmeans)

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

    prefix += f'_{scale}_{graph_construct}_{graph_method}'

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
                                    departements=departements,
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
                                    graph_method=graph_method
                                    )
        graphScale._plot(graphScale.nodes, dir_output=dir_output)
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
        graphScale = read_object(f'graph_{scale}_{graph_construct}_{graph_method}.pkl', dir_output)
        graphScale._plot(graphScale.nodes, dir_output=dir_output)

    departements = [dept for dept in departements if dept not in graphScale.drop_department] 
    train_departements = [dept for dept in train_departements if dept not in graphScale.drop_department] 


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

    #graphScale._plot_risk(mode='time_series_class', dir_output=dir_output)

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
                                            ps, k_days,
                                            departements,
                                            features,
                                            sinister,
                                            dataset_name if not two else f'{dataset_name}2',
                                            sinister_encoding,
                                            dir_output,
                                            dir_output,
                                            prefix,
                                            'full',
                                            'train',
                                            resolution,
                                            trainDate)
        
        save_object(X, 'X_'+prefix+'.pkl', dir_output)
        save_object(Y, 'Y_'+prefix+'.pkl', dir_output)
    else:
        X = read_object('X_'+prefix+'.pkl', dir_output)
        Y = read_object('Y_'+prefix+'.pkl', dir_output)
        features_name, newshape = get_features_name_list(graphScale.scale, features, METHODS_SPATIAL)

    X = X[:, len(ids_columns)-1:]

    if newFeatures != []:
        X2, features_name_ = get_sub_nodes_feature(graphScale, Y[:, :6], departements, newFeatures, sinister, dir_output, dir_output, resolution, graph_construct)
        for fet in newFeatures:
            start , maxi, _, _ = calculate_feature_range(fet, scale, METHODS_SPATIAL)
            X[:, features_name.index(start): features_name.index(start) + maxi] = X2[:, features_name_.index(start): features_name_.index(start) + maxi]
        save_object(X, 'X_'+prefix+'.pkl', dir_output)
        X = X[:, len(ids_columns)-1:]

    ############################## Dataframe creation ###################################

    prefix = f'full_{scale}_{graphScale.base}_{graphScale.graph_method}'

    if (dir_output / f'df_{prefix}.pkl').is_file() and not doDatabase:
        features_name = read_object(f'features_name_{prefix}.pkl', dir_output)
        df = read_object(f'df_{prefix}.pkl', dir_output)
        find_df = not doDatabase
    else: 
        df = pd.DataFrame(columns=ids_columns + targets_columns + features_name, index=np.arange(0, X.shape[0]))
        df[features_name] = X 
        df[ids_columns[:-1] + targets_columns] = Y
        find_df = False

    prefix = f'full_{scale}_{graphScale.base}_{graphScale.graph_method}'

    ############################## Generate 2D database #######################

    if do2D:
        features_name_2D, newShape2D = get_sub_nodes_feature_2D(graphScale, df, departements, features,
                                                                    sinister, dataset_name, dir_output, dir_output,
                                                                    resolution, graph_construct, sinister_encoding)
    else:
        features_name_2D, newShape2D = get_features_name_lists_2D(df.shape[1], features)

    if not doDatabase:
        return df, graphScale, prefix, fp, features_name

    ################################ Process Target ###############################################

    if (dir_output / f'df_mid_{prefix}.pkl').is_file():
        df = read_object(f'df_mid_{prefix}.pkl', dir_output)
    else:
        df = process_target(df, graphScale, prefix, find_df, minDate, departements, train_departements, features_name, kmeans_features, features, dir_output, args)
        save_object(df, f'df_mid_{prefix}.pkl', dir_output)

    ################################ Remove bad or correlated features #############################################
    
    if not find_df and (dir_output / 'features_correlation' / f'{scale}_{graphScale.base}_{graphScale.graph_method}_features_name_after_drop_correlated.pkl').is_file():
        features_name, _ = get_features_name_list(scale, train_features, METHODS_SPATIAL_TRAIN)

        old_shape = df.shape
        df = remove_nan_nodes(df, features_name)
        logger.info(f'Removing nan Features DataFrame shape : {old_shape} -> {df.shape}')

        leni = len(features_name)
        df_features = df[features_name].copy(deep=True)

        # Remove low variance Features:
        df_features = variance_threshold(df_features, 0.15)
        features_name = list(df_features.columns)
        logger.info(f'Remove low Variance {leni} -> {len(features_name)}')
        leni = len(features_name)
        
        logger.info('Removing correlated feature')
        tr = SmartCorrelatedSelection(
                variables=None,
                method="pearson",
                threshold=0.8,
                missing_values="raise",
                selection_method="model_performance",
                estimator=XGBRegressor(random_state=42),
                scoring='r2'
        )
            
        df_features = tr.fit_transform(df_features, df['nbsinister'])
        features_name = list(df_features.columns)
        
        check_and_create_path(dir_output / 'features_correlation')

        save_object(features_name, f'{scale}_{graphScale.base}_{graphScale.graph_method}_features_name_after_drop_correlated.pkl', dir_output / 'features_correlation')
        save_object(tr.correlated_feature_dict_, f'{scale}_{graphScale.base}_{graphScale.graph_method}_correlated_group.pkl', dir_output  / 'features_correlation')
    
        logger.info(f'Smart Correlated Selection {leni} -> {len(features_name)}')

        features_name = list(df_features.columns)
    
    features_name = read_object(f'{scale}_{graphScale.base}_{graphScale.graph_method}_features_name_after_drop_correlated.pkl', dir_output / 'features_correlation')
    
    assert features_name is not None

    features_name = list(features_name)

    ############################## Add varying time features #############################

    if True:
        logger.info(f'Adding time columns {7}')
        df, _ = add_time_columns(varying_time_variables, 7, df.copy(deep=True), train_features, features_name)

    ################################ Drop all duplicate ############################

    df.drop_duplicates(subset=['id', 'date'], inplace=True)

    ############################## Round Values #####################################

    df[features_name] = df[features_name].round(3)

    ############################## Save dataframe and features ###################################

    save_object(df, f'df_{prefix}.pkl', dir_output)
    save_object(features_name, f'features_name_{prefix}.pkl', dir_output)

    ############################## Return data, graph, sinister point and features_name ################################
    fp['database'] = dataset_name

    return df, graphScale, prefix, fp, features_name