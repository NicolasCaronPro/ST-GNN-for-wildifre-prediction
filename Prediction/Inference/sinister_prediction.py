from graph_structure import *
import datetime as dt
import json
from probabilistic import *
import argparse
import socket
from array_fet import *
from generate_database import GenerateDatabase, launch
from itertools import product


# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

####################### Function #################################

def create_ps(geo):
    assert graph is not None

    X_kmeans = list(zip(geo.longitude, geo.latitude))

    if dept in graph.departements.unique():
        geo['id'] = graph._predict_node_with_position(X_kmeans)
    else:
        geo['id'] = graph._predict_node_with_features(dept,  dir_script / 'log', resolution,
                                                        dir_script / 'log', dir_target,
                                                        dir_target_bin, dir_raster, dir_encoder,
                                                        geo)

    logger.info(f'Unique id : {np.unique(geo["id"].values)}')
    logger.info(f'{len(geo)} point in the dataset. Constructing database')

    return geo

def fire_prediction(interface, departement, features, train_features, scaling, dir_data, dates, df_train, apply_break_point):

    assert graph is not None

    features = copy.copy(features)
    train_features = copy.copy(train_features)
    
    logger.info('Assign hexagon to graph nodes')
    geoDT = create_ps(interface.copy(deep=True))
    geoDT.dropna(subset='id', inplace=True)
    logger.info(f'Fix missing features {air_variables, vigicrues_variables, nappes_variables}')
    # Add thoses variables in predictops
    geoDT[air_variables] = df_train[air_variables].mean()
    geoDT[vigicrues_variables] = 0
    geoDT[nappes_variables] = 0
    geoDT[geo_variables] = int(departement.split('-')[1])

    if sinister == 'firepoint':
        geoDT['AutoRegressionBin'] = geoDT['nbfirepoint']

    gbpgeoDT = geoDT.groupby(['id', 'date'])['AutoRegressionBin'].sum().reset_index()
    geoDT.drop('AutoRegressionBin', axis=1, inplace=True)
    geoDT = geoDT.set_index(['id', 'date']).join(gbpgeoDT.set_index(['id', 'date'])['AutoRegressionBin'], on=['id', 'date']).reset_index()

    logger.info('Drop duplicated point due to graph scale + assign weight to correct point')
    geoDT['weights'] = 0
    geoDT['departement'] = int(departement.split('-')[1])
    columns = ['id', 'longitude', 'latitude', 'departement', 'date', 'weights']
    orinode = geoDT.drop_duplicates(['id', 'date'])[columns].values
    orinode = graph._assign_latitude_longitude(orinode)
    orinode[:,3] = name2int[departement]
    orinode[orinode[:, 4] == k_limit, 5] = 1

    model_name = f'{model}_{target}_{task_type}_{loss}'
    
    logger.info(f'Load model {model_name}')
    if model not in traditionnal_models:
        name_model =  model + '/' + 'best.pt'
        dir = 'check_'+scaling + '/' + prefix_train + '/'
        graph._load_model_from_path(dir_data / datatset_name/ dir / name_model, dico_model[model], device)
    else:
        dir = 'check_'+scaling + '/' + prefix_train + '/' + '/baseline'
        model_ = read_object(f'{model_name}.pkl', dir_data / datatset_name/ dir / model_name)
        graph._set_model(model_)

    logger.info('Generate X from dataframe -> could be improove')
    if USE_IMAGE:
        X, features_name = get_sub_nodes_feature_with_images(graph=graph,
                        subNode=orinode,
                        departements=[departement],
                        features=features, sinister=sinister,
                        dir_mask=dir_mask,
                        dir_data=dir_raster,
                        dir_train=dir_data,
                        resolution=resolution,
                        dates=[d.strftime('%Y-%m-%d') for d in dates],
                        graph_construct=graphConstruct)
    else:
        X, features_name = get_sub_nodes_feature_with_geodataframe(graph=graph,
                                subNode=orinode,
                                features=features,
                                departement=departement,
                                geo=geoDT,
                                path=dir_data,
                                dates=[d.strftime('%Y-%m-%d') for d in dates],
                                resolution=resolution)
        
    df = pd.DataFrame(columns=ids_columns + features_name, index=np.arange(0, X.shape[0]))
    df[features_name] = X
    df[ids_columns] = orinode
    df, _ = add_time_columns(varying_time_variables, 7, df.copy(deep=True), train_features, features)
    df.to_csv(f'{dir_log}/{date_limit}.csv', index=False)
        
    fire_feature = ['fire_prediction_raw', 'fire_prediction', 'fire_prediction_dept']

    logger.info(f'Preprocess X (scale {scaling})')

    df = preprocess_inference(df, df_train, scaling, features_name, fire_feature,
                              apply_break_point,
                              scale, kmeans_features, dir_break_point,
                              thresh=thresh_kmeans,
                              shift=days_in_futur)

    if df is None:
        return None
    
    df.to_csv(f'{dir_log}/{date_limit}_after_scale.csv', index=False)
    index_to_predict = df[df['fire_prediction'].isna()].index
    logger.info(f'Found {len(df) - index_to_predict.shape[0]} values where fire has no chance')

    logger.info(f'Load features')
    features_selected = read_object('features.pkl', dir_data / datatset_name/ dir / model_name)

    if df is not None:
        if model in traditionnal_models:
            Y = graph.predict_model_api_sklearn(X=df.loc[index_to_predict], features=features_selected, target_name=target, autoRegression=False)
        else:
            logger.info('Generate X and edges, use for GNN')
            # Predict Today
            X, E = create_inference(graph,
                            df.values,
                            device,
                            False,
                            k_limit,
                            int(args.days))
                    
            if X is None:
                raise Exception('Today x is None')
            Y = graph._predict_tensor(X, E)
    else:
        Y = None

    df['do_prediction']= False
    if index_to_predict.shape[0] != 0:
        df.loc[index_to_predict, 'fire_prediction'] = Y
        df.loc[index_to_predict, 'do_prediction'] = True
    
    logger.info('Features importances')    
    if model in traditionnal_models:
        if date_limit == dt.datetime.now().date():
            output = 'today'
        elif date_limit == dt.datetime.now().date() + dt.timedelta(days=1):
            output = 'tomorrow'
        else:
            output = date_limit.strftime('%Y-%m-%d')
        if output == 'today' or output == 'tomorrow':
            check_and_create_path(dir_interface / 'features_importance_fire')
            samples = df[df['do_prediction'] == True].index.values.astype(int)
            samples_name = list(df.loc[samples, 'id'].values.astype(int))
            samples_name += [f'{id}_today' for id in samples_name]
            graph.model.shapley_additive_explanation(df[features_selected], output, dir_interface, mode='beeswarm', samples=samples, samples_name=samples_name, figsize=(30,15))
        else:
            check_and_create_path(dir_log / 'features_importance_fire')
            check_and_create_path(dir_log / 'features_importance_fire' / 'sample')
            samples = df[df['do_prediction'] == True].index.values.astype(int)
            if len(samples) != 0:
                samples_name = list(df.loc[samples, 'id'].values.astype(int))
                samples_date = int(df.loc[samples, 'date'].values[0])
                samples_name += [f'{id}_{dates[samples_date].strftime("%Y-%m-%d")}' for id in samples_name]
                graph.model.shapley_additive_explanation(df[features_selected], output, dir_log / 'features_importance_fire', mode='beeswarm', samples=samples, samples_name=samples_name, figsize=(30,15))
    else:
        logger.info('No Features importances implemented for DL')

    logger.info('Saving in dataframes')
    geoDT = geoDT.set_index('id').join(df.set_index('id')[fire_feature], on='id').reset_index()
    return geoDT

def prepare_data(date_limit):
    assert graph is not None
    # Date
    date_ks = (date_limit - dt.timedelta(days=k_days_conv))

    if 'incendie_feather_ori' in locals():
        incendie_feather = incendie_feather_ori[(incendie_feather_ori['date'] >= date_ks) & (incendie_feather_ori['date'] <= date_limit)]
        dates = np.sort(incendie_feather.date.unique())
    else:
        dates_str = find_dates_between(date_ks.strftime('%Y-%m-%d'), (d + dt.timedelta(days=1)).strftime('%Y-%m-%d'))
        dates = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dates_str]
    if date_limit not in dates:
        logger.info(f'No data for {date_limit}')
        return None, _
    logger.info(dates)
    if len(dates) == 0:
        return None, _
    ######################### Interface ##################################
    interface = []

    for ks, date in enumerate(dates):            
        geo_data = spatialGeo.copy(deep=True)
        geo_data.index = geo_data.hex_id
        if 'incendie_feather_ori' in locals():
            geo_data = geo_data.set_index('hex_id').join(incendie_feather[incendie_feather['date'] == date][features_in_feather].set_index('hex_id'),
                                                on='hex_id')
            geo_data.reset_index(inplace=True)
            geo_data[spatialGeo['population'].isna()][features_in_feather[1:]] = np.nan
            
        geo_data['date'] = ks
        geo_data['date_str'] = date.strftime('%Y-%m-%d')
        geo_data['longitude'] = geo_data['geometry'].apply(lambda x : float(x.centroid.x))
        geo_data['latitude'] = geo_data['geometry'].apply(lambda x : float(x.centroid.y))

        interface.append(geo_data)

    interface = pd.concat(interface).reset_index(drop=True)

    if sinister == 'firepoint':
        if 'feux' in interface.columns:
            interface.rename({'feux': 'nbfirepoint'}, axis=1, inplace=True)
            interface['nbfirepoint'].fillna(0, inplace=True)
        else:
            interface['nbfirepoint'] = 0

    logger.info('Rename features to match with train fet')
    interface.rename({'temp12' : 'temp', 'dwpt12' : 'dwpt',
                    'rhum12' : 'rhum', 'prcp12' : 'prcp', 'wdir12' : 'wdir',
                    'wspd12' : 'wspd', 'prec24h12' : 'prec24h',
                    'snow24h12': 'snow24h',
                    'daily_severity_rating' : 'dailySeverityRating',
                    'osmnx' : 'highway'}, inplace=True, axis=1)

    ######################### Preprocess #################################

    n_pixel_x = 0.02875215641173088
    n_pixel_y = 0.020721094073767096

    mask = read_object(f'{dept}rasterScale{scale}_{graphConstruct}.pkl', dir_mask)
    if mask is None:
        geo = interface[interface['date'] == 0].reset_index()
        geo[f'id_{scale}'] = graph._predict_node_with_features(dept,  dir_mask, resolution,
                                                    dir_script / 'log', dir_target,
                                                    dir_target_bin, dir_raster, dir_encoder,
                                                    geo)

    if dept in knowed_departement: 
        logger.info(f'Create past influence using convolution and PCA')
        inputDep = create_spatio_temporal_sinister_image(interface,
                                                         geo,
                                                        list(dates),
                                                        mask,
                                                        sinister,
                                                        'occurence',
                                                        n_pixel_y,
                                                        n_pixel_x,
                                                        dir_script / 'log',
                                                        dept)

        Probabilistic(n_pixel_x, n_pixel_y, 1, None, Path(__file__).absolute().parent / dir_script, resolution)._process_input_raster(dims, [inputDep], 1, True,
                                                                                                [dept], True, dates, [dept], True)
        
    
    ######################### Load autoregression ####################
    interface['AutoRegressionReg'] = 0
    return interface, dates

if __name__ == "__main__":

    MACHINES_DEPLOIEMENT = {
     'ns3021266': {'name': 'ovh1', 'depts': ['25']}, 
     'ns3007021': {'name': 'ovh2', 'depts': ['01', '78']}, 
    }

    features = [
                'temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
                'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                'isi', 'angstroem', 'bui', 'fwi', 'dailySeverityRating',
                'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
                'days_since_rain', 'sum_consecutive_rainfall',
                'sum_rain_last_7_days',
                'sum_snow_last_7_days', 'snow24h', 'snow24h16',
                'elevation',
                'population',
                'sentinel',
                'foret_encoder',
                'argile_encoder',
                'id_encoder',
                'cluster_encoder',
                'cosia_encoder',
                'vigicrues',
                'foret',
                'highway',
                'cosia',
                'Calendar',
                'Historical',
                'Geo',
                'air',
                'nappes',
                'AutoRegressionReg',
                'AutoRegressionBin'
                ]

        # Train features 
    train_features = [
                    'temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
                    'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                    'isi', 'angstroem', 'bui',
                    'fwi',
                    'dailySeverityRating',
                    'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
                    'days_since_rain', 'sum_consecutive_rainfall',
                    'sum_rain_last_7_days',
                    'sum_snow_last_7_days', 'snow24h', 'snow24h16',
                    'elevation',
                    'population',
                    #'sentinel',
                    'foret_encoder',
                    'argile_encoder',
                    'id_encoder',
                    'cluster_encoder',
                    'cosia_encoder',
                    'vigicrues',
                    'foret',
                    'highway',
                    'cosia',
                    'Calendar',
                    'Historical',
                    'Geo',
                    'air',
                    'nappes',
                    #'AutoRegressionReg',
                'AutoRegressionBin',
                    ]
        
    kmeans_features = [
                #'rhum',
                'prec24h',
                #'rhum16',
                'prec24h16',
                'sum_consecutive_rainfall',
                'sum_rain_last_7_days',
                'sum_snow_last_7_days',
                'snow24h', 'snow24h16',
                ]

    features_in_feather = ['hex_id', 'date',
                            'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
                            'temp12', 'dwpt12', 'rhum12', 'prcp12', 'wdir12', 'wspd12', 'prec24h12',
                            'dc', 'ffmc', 'dmc', 'isi', 'bui', 'fwi',
                            'daily_severity_rating', 'nesterov', 'munger', 'kbdi', 'angstroem',
                            'days_since_rain', 'sum_consecutive_rainfall', 'sum_rain_last_7_days',
                            'sum_snow_last_7_days', 'snow24h12', 'snow24h16',
                            'feux',
                            ]

    knowed_departement = ['departement-01-ain', 'departement-25-doubs', 'departement-69-rhone', 'departement-78-yvelines']

    traditionnal_models = ['xgboost', 'lightgbm', 'ngboost']
        

    parser = argparse.ArgumentParser(
        prog='Inference',
        description='Inference prediction',
    )
    parser.add_argument('-d', '--days', type=str, help='Number of days')
    parser.add_argument('-l', '--loss', type=str, help='Loss used from training')
    parser.add_argument('-t', '--task_type', type=str, help='Regression or classification')
    parser.add_argument('-ta', '--target', type=str, help='Target')
    parser.add_argument('-dept', '--departement', type=str, help='departement')
    parser.add_argument('-sinis', '--sinister', type=str, help='Sinister')
    parser.add_argument('-m', '--model', type=str, help='Model')
    parser.add_argument('-np', '--nbpoint', type=str, help='Number of point')
    parser.add_argument('-nf', '--nbFeature', type=str, help='Number of feature')
    parser.add_argument('-dh', '--doHistorical', type=str, help='Do Historical')
    parser.add_argument('-r', '--resolution', type=str, help='Pixel resolution')
    parser.add_argument('-graphConstruct', '--graphConstruct', type=str, help='Apply Kmeans on target')
    parser.add_argument('-dataset', '--dataset', type=str, help='Dataset to use')
    parser.add_argument('-graph_method', '--graph_method', type=str, help='Dataset to use')
    parser.add_argument('-thresh_kmeans', '--thresh_kmeans', type=str, help='Dataset to use')
    parser.add_argument('-shift_kmeans', '--shift_kmeans', type=str, help='Dataset to use')
    parser.add_argument('-doKMEANS', '--doKMEANS', type=str, help='Dataset to use')
    
    args = parser.parse_args()

    # Input config
    k_days = int(args.days)
    dept = args.departement
    sinister = args.sinister
    model = args.model
    minPoint = args.nbpoint
    target = args.target
    loss = args.loss
    task_type = args.task_type
    nbFeature = args.nbFeature
    resolution = args.resolution
    doHistorical = args.doHistorical == 'True'
    scaling = 'z-score' # Scale to used
    datatset_name= args.dataset
    graphConstruct = args.graphConstruct
    dataset_name = args.dataset
    graph_method = args.graph_method
    thresh_kmeans = args.thresh_kmeans
    shift = args.shift_kmeans
    doKMEANS = args.doKMEANS == 'True'

    scale_list = [4, 5, 6, 7, 'departement']
    days_in_futur_list = [0]
    combinations = list(product(scale_list, days_in_futur_list))

    interfaces = []
    
    #spec = 'occurence_inference'
    spec = 'occurence_default'

    for scale, days_in_futur in combinations:

        prefix_train = f'full_{k_days}_{nbFeature}_{scale}_{days_in_futur}_{graphConstruct}_{graph_method}'
        prefix_df = f'full_{scale}_{days_in_futur}_{graphConstruct}_{graph_method}'

        if doKMEANS:
            prefix_train += f'_kmeans_{shift}_{thresh_kmeans}'

        prefix_kmeans =  f'full_{scale}_{graphConstruct}_{graph_method}'

        # Arborescence
        if socket.gethostname() in MACHINES_DEPLOIEMENT:
            USE_IMAGE = False
            root_dir = Path(__file__).absolute().parent.parent.parent.parent.resolve()
            dir_data = root_dir / 'data' / 'features' / 'incendies' / dataset_name / sinister / resolution / 'train'
            dir_feather = root_dir / 'data' / 'dataframes' 
            dir_interface = root_dir / 'interface'
            dir_log = root_dir / 'scripts' / 'global' / 'sinister_prediction' / 'hexagones'
            dir_incendie = root_dir / 'data' / 'features' / 'incendies'
            dir_feather = root_dir / 'data' / 'dataframes'
            dir_raster = dir_incendie / 'raster' / resolution
            dir_mask = dir_data / 'raster'
            dir_script = Path('./')
        else:
            dir_data = Path(f'../GNN/{dataset_name}/{sinister}/{resolution}/train/')
            dir_feather = Path('interface')
            dir_log = Path(f'interface/{dept}')
            dir_incendie = Path(f'/home/caron/Bureau/csv/{dept}/data/')
            dir_incendie_disk = Path(f'/media/caron/X9 Pro/travaille/Thèse/csv/{dept}/data')
            dir_interface = Path('interface')
            dir_feather = Path('interface')
            dir_mask = dir_data /'raster'
            dir_encoder = dir_data / 'Encoder'
            dir_target = Path(f'../Target/{dataset_name}/{sinister}/log/{resolution}')
            dir_target_bin = Path(f'../Target/{sinister}/bin/{resolution}')
            dir_raster = dir_interface / 'raster'
            dir_script = Path('./')

        dir_output = dir_log / 'output'
        dir_break_point = dir_data / datatset_name/ 'check_none' / prefix_kmeans / 'kmeans'
        check_and_create_path(dir_output)
        check_and_create_path(dir_log)

        graph = read_object(f'graph_{scale}_{graphConstruct}_{graph_method}.pkl', dir_data)
        assert graph is not None

        df_train = read_object(f'df_train_{prefix_df}.pkl', dir_data / spec)

        assert df_train is not None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if socket.gethostname() in MACHINES_DEPLOIEMENT:
            incendie_feather_ori = pd.read_feather(dir_feather / f'incendie_final.feather')
        else:
            if (dir_feather / f'incendie_final_{name2str[dept]}.feather').is_file():
                incendie_feather_ori = pd.read_feather(dir_feather / f'incendie_final_{name2str[dept]}.feather')
                USE_IMAGE = False
            else:
                compute_meteostat_features = False
                compute_temporal_features = False
                compute_spatial_features= False
                compute_air_features = False
                compute_trafic_features = False
                compute_vigicrues_features = False
                compute_nappes_features = True
                histo = 14
                start = (dt.datetime.now() - dt.timedelta(days=histo)).date().strftime('%Y-%m-%d')
                stop = (dt.datetime.now() + dt.timedelta(days=2)).date().strftime('%Y-%m-%d')
                launch(dir_incendie, dir_incendie_disk, dir_raster,
                    dept, resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features,
                    start, stop)
                USE_IMAGE = True

        # Load feather and spatial
        if (dir_incendie / 'spatial' / f'hexagones_{sinister}.geojson').is_file():
            spatialGeo = gpd.read_file(dir_incendie / 'spatial' / f'hexagones_{sinister}.geojson')
        else:
            if not (dir_incendie / 'spatial' / f'hexagones.geojson').is_file():
                logger.info(f'{dir_incendie}/spatial/hexagones_{sinister}.geojson doesn t exist, please provide hexagones !!')
                exit(2)
            spatialGeo = gpd.read_file(dir_incendie / 'geo' / f'hexagones.geojson')
            if not USE_IMAGE:
                spatialGeo = process_department(dir_data, dir_incendie, dir_mask, dir_encoder,
                                spatialGeo, sinister, dept, resolution,
                                ['sentinel', 'osmnx', 'population', 'elevation', 'foret_landcover',  'osmnx_landcover' ,'foret', 'dynamic_world', 'cosia', 'cosia_landcover', 'argile_landcover'])
        spa = 3

        if sinister == "firepoint":
            if dept == 'departement-01-ain':
                dims = [(spa,spa,5), (spa,spa,9), (spa,spa,3)],
                k_days_conv = max(9, k_days)
            elif dept == 'departement-25-doubs':
                dims = [(spa,spa,3), (spa,spa,5), (spa,spa,3)],
                k_days_conv = max(5, k_days)
            elif dept == 'departement-69-rhone':
                dims =  [(spa,spa,3), (spa,spa,5),(spa,spa,1)],
                k_days_conv = max(5, k_days)
            elif dept == 'departement-78-yvelines':
                dims =  [(spa,spa,3), (spa,spa,7), (spa,spa,3)],
                k_days_conv = max(7, k_days)
            else:
                dims = [(spa,spa,5), (spa,spa,9), (spa,spa,3)],
                k_days_conv = max(9, k_days)
        elif sinister == "inondation":
            """if dept == 'departement-01-ain':
                dims = [(spa,spa,5), (spa,spa,9), (spa,spa,3)],
                k_days_conv = max(9, k_days)
            elif dept == 'departement-25-doubs':
                dims = [(spa,spa,3), (spa,spa,5), (spa,spa,3)],
                k_days_conv = max(5, k_days)
            elif dept == 'departement-69-rhone':
                dims =  [(spa,spa,3), (spa,spa,5),(spa,spa,1)],
                k_days_conv = max(5, k_days)
            elif dept == 'departement-78-yvelines':
                dims =  [(spa,spa,3), (spa,spa,7), (spa,spa,3)],
                k_days_conv = max(7, k_days)"""
            raise ValueError(f'Sinister must be firepoint got {sinister}')
        else:
            exit(2)

        if doHistorical:
            sd = dt.datetime.now().date() - dt.timedelta(days=8)
            ed = dt.datetime.now().date() - dt.timedelta(days=6)
        else:
            sd = dt.datetime.now().date()
            ed = dt.datetime.now().date() + dt.timedelta(days=2)
        d = sd

        while d != ed:
            date_limit = d
        
            interface, dates = prepare_data(date_limit)
            if interface is None:
                d += dt.timedelta(days=1)
                continue

            k_limit = interface.date.max()
            ######################### Predict ################################

            logger.info(f'Launch {sinister} Prediction')
            interface = fire_prediction(interface, dept, features, train_features, scaling, dir_data, dates, df_train, apply_break_point)
            if interface is None:
                logger.info(f'Can t produce fire prediction for {dates[-1]}')
                d += dt.timedelta(days=1)
                continue

            interface['scale'] = scale
            interface['days_in_futur'] = days_in_futur
            interfaces.append(interface)

        interfaces = pd.concat(interfaces)

        ######################### Saving ###############################
        if date_limit == dt.datetime.now().date():
            input = 'today'
            output = f'today'
        elif date_limit == dt.datetime.now().date() + dt.timedelta(days=1):
            input = 'tomorrow'
            output = f'tomorrow'
        else:
            input = 'today'

        if socket.gethostname() not in MACHINES_DEPLOIEMENT:
            input = f'{input}_{name2str[dept]}'
            if 'output' in locals():
                output = f'{output}_{name2str[dept]}'
        else:
            input = f'{input}0'

        fire_feature = ['fire_prediction_raw', 'fire_prediction', 'fire_prediction_dept']
        
        if (dir_interface / f'hexagones_{input}.geojson').is_file():
            output_geojson = gpd.read_file(dir_interface / f'hexagones_{input}.geojson')
        else:
            output_geojson = interfaces[interfaces['date'] == 0].reset_index()

        for ff in fire_feature:
            if ff in output_geojson.columns:
                output_geojson.drop(ff, inplace=True, axis=1)

        if date_limit == dt.datetime.now().date() or date_limit == dt.datetime.now().date() + dt.timedelta(days=1):
            logger.info(f'Saving to hexagones_{output}_final.geojson')
            output_geojson = output_geojson.set_index('hex_id').join(interfaces[interfaces['date'] == k_limit].set_index('hex_id')[fire_feature],
                                                        on='hex_id')
            output_geojson.reset_index(inplace=True)
            output_geojson.to_file(dir_interface / f'hexagones_{output}_final.geojson', driver='GeoJSON')
        
        if date_limit != dt.datetime.now().date() + dt.timedelta(days=1):
            previous_date = (date_limit - dt.timedelta(days=1)).strftime('%Y-%m-%d')
            on = f'hexagones_{previous_date}.geojson'
            logger.info(f'Open {on} to save last recorded fires')
            if (dir_log / on).is_file():
                previous_log_interfaces = gpd.read_file(dir_log / on)
                previous_log_interfaces.drop('nbfirepoint', inplace=True, axis=1)
                previous_log_interfaces = previous_log_interfaces.set_index('hex_id').join(interfaces[interfaces['date'] == k_limit - 1].set_index('hex_id')['nbfirepoint'], on='hex_id').reset_index()
                previous_log_interfaces.to_file(dir_log / on, driver='GeoJSON')                    
                try:
                    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
                    previous_log_interfaces.plot(column='fire_prediction', vmin=1, vmax=6, cmap='jet', legend=True, ax=ax[0][0])
                    previous_log_interfaces.plot(column='fire_prediction_raw', cmap='jet',  ax=ax[0][1])
                    previous_log_interfaces.plot(column='fire_prediction_dept', vmin=1, vmax=6, cmap='jet', legend=True, ax=ax[1][0])
                    previous_log_interfaces.plot(column='id', legend=True, categorical=True, ax=ax[1][1])
                    logger.info(f'{previous_date}, {dept}')
                    if previous_date == '2024-08-18' and dept == 'departement-34-herault':
                        lats = [43.289746816650435, 43.48800209415416]
                        longs = [3.0975390912695886, 3.753774492722863]
                        fire_herault = pd.DataFrame(index=np.arange(2))
                        fire_herault['latitude'] = lats
                        fire_herault['longitude'] = longs
                        fire_herault['nbfirepoint'] = 1
                        fire_herault = gpd.GeoDataFrame(fire_herault, geometry=gpd.points_from_xy(fire_herault.longitude, fire_herault.latitude))
                        fire_herault.plot(column='nbfirepoint', cmap='binary', ax=ax[0][0], vmin=0)
                        fire_herault.plot(column='nbfirepoint', cmap='binary', ax=ax[0][1], vmin=0)
                        fire_herault.plot(column='nbfirepoint', cmap='binary', ax=ax[1][0], vmin=0)
                    else:
                        if previous_log_interfaces['nbfirepoint'].sum() > 0:
                            previous_log_interfaces[previous_log_interfaces['nbfirepoint'] > 0].plot(column='nbfirepoint', cmap='binary', ax=ax[0][0], vmin=0)
                            previous_log_interfaces[previous_log_interfaces['nbfirepoint'] > 0].plot(column='nbfirepoint', cmap='binary', ax=ax[0][1], vmin=0)
                            previous_log_interfaces[previous_log_interfaces['nbfirepoint'] > 0].plot(column='nbfirepoint', cmap='binary', ax=ax[1][0], vmin=0)
                    check_and_create_path(dir_log / 'image')
                    plt.savefig(dir_log / 'image' / f'hexagones_{previous_date}.png')
                except Exception as e:
                    logger.info(f'Can t produce image for {previous_date}, {e}')

            output = date_limit.strftime('%Y-%m-%d')
            log_interfaces = interfaces[interfaces['date'] == k_limit]     
            on = f'hexagones_{output}.geojson'
            logger.info(f'Saving to {on}')
            log_interfaces.to_file(dir_log / on, driver='GeoJSON')

        d += dt.timedelta(days=1)