import pickle
from re import subn

from nbconvert import export
from sklearn.cluster import KMeans
from copy import copy
from weigh_predictor import Predictor
from GNN.graph_structure import *
from feature_engine.selection import SmartCorrelatedSelection

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

def parse_string(s):

    if s is None:
        return {"base" : "None", "attempt" : 0, "reduce" : 0, "tol" : 0}
    
    base = s
    
    if 'degree' in s:
        s = s.split('degree')[1]
    elif 'hexa' in s:
        s = s.split('hexa')[1]
    
    # Initialiser le dictionnaire avec None
    result = {"base": base, "attempt": None, "reduce": None, "tol": None}
    if base == "zonemeteo":
        return result

    # Utiliser findall pour capturer toutes les balises présentes
    matches = re.findall(r"(b(?P<base>[^-]+))|(a(?P<attempt>[^-]+))|(r(?P<reduce>[^-]+))|(t(?P<tol>[^-]+))", s)

    for groups in matches:
        b, a, r, t = groups[1], groups[3], groups[5], groups[7]
        if b:
            result["base"] = b
        if a:
            result["attempt"] = a
        if r:
            result["reduce"] = r
        if t:
            result["tol"] = t

    return result

def construct_graph(scale, maxDist, sinister, dataset_name, sinister_encoding, train_departements, departements,
                    geo, nmax, k_days, dir_output, doRaster, doEdgesFeatures, resolution, graph_construct, train_dates, val_date, graph_method):
    
    train_date = train_dates[-1]
    dico_config = parse_string(graph_construct)
    print(dico_config)
    graphScale = GraphStructure(scale=scale, geo=geo, maxDist=maxDist, numNei=nmax, resolution=resolution, graph_construct=dico_config['base'], sinister=sinister,
                                sinister_encoding=sinister_encoding, dataset_name=dataset_name, train_departements=train_departements, graph_method=graph_method,
                                attempt=dico_config['attempt'], reduce=dico_config['reduce'], tol=dico_config['tol'])

    sucseptibility_map_model_config = {'type': 'Unet',
                                       'device': 'cuda',
                                       'task': 'regression',
                                       'loss' : 'rmse',
                                       'name': 'mapper',
                                       'params' : None}
    
    variables_for_susecptibilty_and_clustering = ['population', 'foret',
                                                  #'bdroute',
                                                  'corine', 'elevation',
                                                'temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
                                        'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                                        'isi', 'angstroem', 'bui', 'fwi', 'dailySeverityRating',
                                        'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
                                        'days_since_rain', 'sum_consecutive_rainfall',
                                        'sum_rain_last_7_days',
                                        'sum_snow_last_7_days', 'snow24h', 'snow24h16']
        
    graphScale._create_sinister_region(
                                 path=dir_output, sinister=sinister, dataset_name=dataset_name,
                                 sinister_encoding=sinister_encoding,
                                 resolution=resolution, train_date=train_date)
    
    graphScale._create_nodes_list()
    graphScale._create_edges_list()
    graphScale._create_temporal_edges_list(allDates, k_days=k_days)
    graphScale.nodes = graphScale._assign_department(graphScale.nodes)

    graphScale._clusterize_node_with_time_series(departements=train_departements,
                                            variables=variables_for_susecptibilty_and_clustering,
                                            target='nbsinister', train_dates=train_dates,
                                            path=dir_output,
                                            root_data=rootDisk / 'csv',
                                            root_target=root_target / sinister / dataset_name / sinister_encoding)
    
    #graphScale.find_closest_cluster(graphScale.departements.unique(), train_date, dir_output, rootDisk / 'csv')

    if doEdgesFeatures:
        graphScale.edges = edges_feature(graphScale, ['slope', 'highway'], dir_output, geo)
    save_object(graphScale, f'graph_{scale}_{graphScale.base}_{graphScale.graph_method}.pkl', dir_output)
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

        elif dataset_name == 'bdiff' or dataset_name == 'bdiff_small':
            end_date = allDates[-1]
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
    do2D = args.database2D
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
    limit_day = 1
    shift_list = np.arange(0, 1)
    thresh_kmeans_list = np.arange(0.1, 1, 0.5)
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
        #for target_spe in target_column_name_list:
        #    logger.info(f'Add {target_spe} in {limit_day} days in future')
        #    df = target_by_day(df, limit_day, futur_met, target_spe)

        save_object(df, f'df_no_weight_{prefix}.pkl', dir_output)

    return df

def construct_database_from_xarray(
    graphScale: GraphStructure,
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
    name_exp,
):
    
    scale = graphScale.scale
    graph_construct = graphScale.base
    graph_method = graphScale.graph_method
    
    """#######################################################################################
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
    graphScale._info_on_graph(subNode, Path('log'))"""

    ################################## Try loading Y database #############################
    # Define the filename for the ground truth data
    n = f'Y_datacube_full_{scale}_{graph_construct}_{graph_method}_{name_exp}.pkl'

    #if not (dir_output / n).is_file():
    if True:
        Y = get_sub_nodes_ground_truth_from_xarray(
            graphScale,
            departements,
            dir_train,
            dir_train,
            dataset_name
        )
        # Save the generated ground truth data
        save_object(Y, n, dir_output)
    else:
        Y = read_object(n, dir_output)

    # If X is not loaded, generate the features
    Y, features_name = get_sub_nodes_features_from_xarray(
        graphScale,
        Y,
        departements,
        features,
        sinister,
        dataset_name,
        sinister_encoding,
        name_exp,
        dir_output,
        dir_train,
        resolution,
    )

    return Y, features_name

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
    name_exp,
):
      
    scale = graphScale.scale
    graph_construct = graphScale.base
    graph_method = graphScale.graph_method
    
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
    n = f'Y_full_{scale}_{graph_construct}_{graph_method}_{name_exp}.pkl'
    if True:
    #if not (dir_output / n).is_file():
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
    n = f'X_full_{scale}_{graph_construct}_{graph_method}_{name_exp}.pkl'
    if (dir_output / n).is_file():
        # Get the list of feature names
        features_name, _ = get_features_name_list(scale, features, METHODS_SPATIAL)
        # Load the features data
        X = read_object(n, dir_output)
    else:
        X = None  # If the file doesn't exist, set X to None

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
            name_exp,
            dir_output,
            dir_train,
            resolution,
        )
    else:

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
    doEncoder = args.encoder
    doPoint = args.point
    doGraph = args.graph
    doDatabase = args.database
    print(f'Do databse -> {args.database}')
    do2D = args.database2D
    sinister = args.sinister
    values_per_class = args.nbpoint
    scale = int(args.scale) if args.scale != 'departement' and args.scale != 'user' else args.scale
    resolution = args.resolution
    ncluster = int(args.ncluster)
    k_days = int(args.k_days) # Size of the time series sequence use by DL models
    dir_output = dir_output
    graph_construct = args.graphConstruct
    dataset_name = args.dataset
    sinister_encoding = args.sinisterEncoding
    graph_method = args.graph_method
    
    if args.likenormal:
        name_exp = 'normal'

    all_train_dates, all_val_dates, all_test_dates = defines_train_dates(args)
    maxDate = all_test_dates[0]
    trainDate = all_train_dates[-1]
    ######################## Get features and train features list ######################

    isInference = name_exp == 'inference'
    
    #features, train_features, kmeans_features = get_features_for_sinister_prediction(dataset_name, sinister, isInference)
    features = args.features
    train_features = args.train_features
    kmeans_features = args.kmeans_features

    ######################## Get departments and train departments #######################

    departements = args.train_departments
    departements += [dept for dept in args.test_departments if dept not in departements]
    departements = sorted(departements)
    train_departements = args.train_departments

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

    prefix += f'_{scale}_{graph_construct}_{graph_method}_{name_exp}'

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
                                    train_dates=all_train_dates,
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
    """if doPoint:
        
        logger.info('#####################################')
        logger.info('#      Get hexagones point          #')
        logger.info('#      for departement              #')
        logger.info('#####################################')
        
        check_and_create_path(dir_output)

        look_for_information(graphScale, dataset_name,
                         maxDate,
                         sinister,
                         Path(dataset_name))
        
        name = 'points.csv'
        ps = pd.read_csv(Path(dataset_name) / sinister / name)
        depts = [name2int[dept] for dept in departements]
        logger.info(ps.departement.unique())
        ps = ps[ps['departement'].isin(depts)].reset_index(drop=True)
        logger.info(ps.departement.unique())"""

    ######################### Encoding ######################################

    if args.encoder:
        logger.info('#####################################')
        logger.info('#      Calcualte Encoder            #')
        logger.info('#####################################')
        #encode(root_target / sinister / dataset_name / sinister_encoding / 'bin' / resolution, all_train_dates, name_exp, train_departements, dir_output / 'Encoder', resolution, graphScale)
        encode_from_xarray('occurence', all_train_dates, name_exp, train_departements, dir_output / 'Encoder', resolution, graphScale)
        encode_from_xarray('burned_area', all_train_dates, name_exp, train_departements, dir_output / 'Encoder', resolution, graphScale)

    ########################## Do Database ####################################
    if doDatabase:
        logger.info('#####################################')
        logger.info('#      Construct   Database         #')
        logger.info('#####################################')

        df, features_name = construct_database_from_xarray(graphScale, k_days,
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
                                            trainDate,
                                            name_exp)
        
        save_object(df, 'df_feat_'+prefix+'.pkl', dir_output)
        #save_object(Y, 'Y_'+prefix+'.pkl', dir_output)
    else:
        df = read_object('df_feat_'+prefix+'.pkl', dir_output)            
        #Y = read_object('Y_'+prefix+'.pkl', dir_output)
        features_name, newshape = get_features_name_list(graphScale.scale, features, METHODS_SPATIAL)

    """for i, col in enumerate(ids_columns):
        print('X', col, np.unique(X[:, i]))
        print('Y', col, np.unique(Y[:, i]))"""
    ################################################ Add a new feature ####################################
    newFeatures = []
    if newFeatures != []:
        X2, features_name_2 = get_sub_nodes_feature(
            graphScale,
            Y[:, :len(ids_columns) - 1],
            departements,
            newFeatures,
            sinister,
            dataset_name,
            sinister_encoding,
            name_exp,
            dir_output,
            dir_output,
            resolution,
            use_log=False
        )
        
        X2 = X2[:, len(ids_columns)-1:]

        features_name_ori, newshape = get_features_name_lis(graphScale.scale, [fet for fet in features if fet not in newFeatures], METHODS_SPATIAL)
            
        new_X = np.empty((X.shape[0], X.shape[1] + X2.shape[1]))

        for fet in features_name:
            if fet in features_name_2:    
                new_X[:, features_name.index(fet) + len(ids_columns)-1] = X2[:, features_name_2.index(fet)]
            else:
                new_X[:, features_name.index(fet) + len(ids_columns)-1] = X[:, features_name_ori.index(fet) + len(ids_columns)-1]
        
        X = new_X

        save_object(X, 'X_'+prefix+'.pkl', dir_output)

    ################################################ Change a Feature Value ####################################
    changeFeature = []
    if changeFeature != []:
        X2, features_name_2 = get_sub_nodes_feature(
            graphScale,
            Y[:, :len(ids_columns) - 1],
            departements,
            changeFeature,
            sinister,
            dataset_name,
            sinister_encoding,
            name_exp,
            dir_output,
            dir_output,
            resolution,
            use_log=False
        )

        X2 = X2[:, len(ids_columns)-1:]

        new_X = np.empty((X.shape[0], X.shape[1]))

        for fet in features_name:
            if fet in features_name_2:    
                new_X[:, features_name.index(fet) + len(ids_columns)-1] = X2[:, features_name_2.index(fet)]
            else:
                new_X[:, features_name.index(fet) + len(ids_columns)-1] = X[:, features_name.index(fet) + len(ids_columns)-1]

        X = new_X
        
        save_object(X, 'X_'+prefix+'.pkl', dir_output)

    logger.info(f'{df.shape}')
    logger.info(f'{df}')
    df.drop_duplicates(subset=['id', 'date'], inplace=True)
    print(df['departement'].unique())
    print(df['id'].unique())
    print(df['date'].unique())
    logger.info(f'{df.shape}')
    #df = datacube.to_dataframe().reset_index()
    df['date'] = df['date'].apply(lambda x : allDates.index(x))
    df['departement'] = df['departement'].apply(lambda x : name2int[x])
    """X = X[:, len(ids_columns)-1:]

    ############################## Dataframe creation ###################################
    prefix = f'full_{scale}_{graphScale.base}_{graphScale.graph_method}_{name_exp}'

    ############### SI CA PLANTE -> SCALE ###################
    #if (dir_output / f'df_{prefix}.pkl').is_file() and not doDatabase and newFeatures == [] and changeFeature == []:
    if False:
        features_name = read_object(f'features_name_{prefix}.pkl', dir_output)
        df = read_object(f'df_{prefix}.pkl', dir_output)
        find_df = not doDatabase
        print('FIRE:', df['nbsinister'].unique())
    else:
        #print(len(features_name), X.shape)
        df = pd.DataFrame(columns=ids_columns + targets_columns + features_name, index=np.arange(0, X.shape[0]))
        df[features_name] = X 
        df[ids_columns + targets_columns] = Y
        find_df = False"""

    if scale == 'departement':
        df['scale'] = 10
    else:
        df['scale'] = scale

    ############################## Generate 2D database #######################
    
    if do2D:
        get_sub_nodes_feature_2D_from_xarray(
                        graphScale,
                        departements,
                        features,
                        dir_output,
                        dir_output,
                        name_exp,
                        True)

    if 'id_encoder_mean' in np.unique(df.columns):
        df['id_encoder'] = df['id_encoder_mean']
        for v in ['id_encoder_mean', 'id_encoder_max', 'id_encoder_min', 'id_encoder_std', 'id_encoder_sum', 'id_encoder_grad']:
            if v in np.unique(df.columns):
                df.drop(v, inplace=True, axis=1)

    df['nbsinister_0_0'] = df['nbsinister'].values
    df['burnedarea_0_0'] = df['burned_area'].values
    df['risk_0_0'] = df['nbsinister'].values
    df['class_risk_0_0'] = 1
    df['month_non_encoder'] = df['date'].apply(lambda x : int(allDates[int(x)].split('-')[1]))

    if args.likenormal:
        name_exp = args.name

    ###################################################################################

    prefix = f'full_{scale}_{graphScale.base}_{graphScale.graph_method}_{name_exp}'
    
    trainCode = [name2int[d] for d in train_departements]

    train_mask = (df['date'].isin([allDates.index(d) for d in all_train_dates])) & (df['departement'].isin(trainCode))
    shift_list = np.arange(0, 1)
    #train_break_point(df[train_mask].copy(deep=True), features_name, dir_output / 'check_none' / prefix / 'kmeans', ncluster, shift_list)

    ################################ Process Target ###############################################

    #if dataset_name == 'bdiff' and not find_df:
    limit_day = [7, 15, 31]
    logger.info(f'Add {limit_day} days in future')
    df = target_by_day(df, limit_day, target_spe='0')

    """if (dir_output / f'df_mid_{prefix}.pkl').is_file():
        df = read_object(f'df_mid_{prefix}.pkl', dir_output)
    else:
        df = process_target(df, graphScale, prefix, find_df, minDate, departements, train_departements, features_name, kmeans_features, features, dir_output, args)
        save_object(df, f'df_mid_{prefix}.pkl', dir_output)"""

    ################################ Remove bad or correlated features #############################################
    if True:
    #if (not (dir_output / 'features_correlation' / f'{scale}_{graphScale.base}_{graphScale.graph_method}_features_name_after_drop_correlated_{name_exp}.pkl').is_file()):

        if name_exp == 'occurence_less_feature':
            features_name, _ = get_features_name_list(scale, train_features, ['mean'])
        else:
            features_name, _ = get_features_name_list(scale, train_features, METHODS_SPATIAL_TRAIN)

        old_shape = df.shape

        if 'Past_risk' in features_name:
            features_name = [fn for fn in features_name if fn != 'Past_risk']
            readdPastRisk = True
        if 'Past_burnedarea' in features_name:
            features_name = [fn for fn in features_name if fn != 'Past_burnedarea']
            readdPastBurned = True

        df = remove_nan_nodes(df, features_name)
        logger.info(f'Removing nan Features DataFrame shape : {old_shape} -> {df.shape}')

        leni = len(features_name)
        df_features = df[features_name].copy(deep=True)

        # Remove low variance Features:
        df_features = variance_threshold(df_features, 0)
        features_name = list(df_features.columns)
        logger.info(f'Remove low Variance {leni} -> {len(features_name)}')
        leni = len(features_name)

        thresholds = 0.95

        logger.info('Removing correlated feature')
        tr = SmartCorrelatedSelection(
                variables=None,
                method="pearson",
                threshold=thresholds,
                missing_values="raise",
                selection_method="variance",
                estimator=None,
        )
        df['binary'] = df['nbsinister'] > 0

        df_features = tr.fit_transform(df_features)
        features_name = list(df_features.columns)
        logger.info(f'Smart Correlated Selection with Pearson {leni} -> {len(features_name)}')
        leni = len(features_name)

        print(features_name)

        tr = SmartCorrelatedSelection(
                variables=None,
                method="spearman",
                threshold=thresholds,
                missing_values="raise",
                selection_method="variance",
                estimator=None,
        )

        df_features = tr.fit_transform(df_features)
        features_name = list(df_features.columns)
        logger.info(f'Smart Correlated Selection with Spearman {leni} -> {len(features_name)}')
        leni = len(features_name)

        print(features_name)

        tr = SmartCorrelatedSelection(
                variables=None,
                method="kendall",
                threshold=thresholds,
                missing_values="raise",
                selection_method="variance",
                estimator=None,
        )
        
        df_features = tr.fit_transform(df_features)
        features_name = list(df_features.columns)
        logger.info(f'Smart Correlated Selection with Kendall {leni} -> {len(features_name)}')
        leni = len(features_name)
        
        print(features_name)

        check_and_create_path(dir_output / 'features_correlation')

        if 'readdPastRisk' in locals():
            features_name.append('Past_risk')
        
        if 'readdPastBurned' in locals():
            features_name.append('Past_burnedarea')

        save_object(features_name, f'{scale}_{graphScale.base}_{graphScale.graph_method}_features_name_after_drop_correlated_{name_exp}.pkl', dir_output / 'features_correlation')
        save_object(tr.correlated_feature_dict_, f'{scale}_{graphScale.base}_{graphScale.graph_method}_correlated_group_{name_exp}.pkl', dir_output  / 'features_correlation')

        logger.info(f'Smart Correlated Selection {leni} -> {len(features_name)}')

        features_name = list(df_features.columns)
    
    features_name = read_object(f'{scale}_{graphScale.base}_{graphScale.graph_method}_features_name_after_drop_correlated_{name_exp}.pkl', dir_output / 'features_correlation')
    
    assert features_name is not None

    features_name = list(features_name)

    ############################## Add varying time features #############################

    #if dataset_name == 'bdiff' and not find_df:
    #if not find_df:
    #    logger.info(f'Adding time columns {10}')
    #    logger.info(f'WARNING: NO TIME COLUMN ARE ADDED')
    #    df, _ = add_time_columns(varying_time_variables, 20, df.copy(deep=True), train_features, features_name)

    ################################ Drop all duplicate ############################

    df.drop_duplicates(subset=['id', 'date'], inplace=True)

    ############################## Round Values #####################################

    for fet in features_name:
        if fet not in list(df.columns):
            continue
        df[fet] = df[fet].round(3)

    ############################## Save dataframe and features ###################################

    df['saison'] = df['date'].apply(get_saison)
    df['saison-encoding'] = df['date'].apply(get_saison_encoding)
    df['mediterranean'] = df['departement'].apply(is_mediterranean_dept)
    df['cluster-encoder'] = df['cluster_encoder']

    save_object(df, f'df_{prefix}.pkl', dir_output)
    save_object(features_name, f'features_name_{prefix}.pkl', dir_output)

    ############################## Return data, graph, sinister point and features_name ################################
    fp['database'] = dataset_name
    
    prefix = f'full_{scale}_{graphScale.base}_{graphScale.graph_method}'
    return df, graphScale, prefix, fp, features_name