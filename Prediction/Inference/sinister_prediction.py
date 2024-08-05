from graph_structure import *
import datetime as dt
import json
from probabilistic import *
import argparse
from array_fet import *

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

####################### Function #################################

def create_ps(geo):
    X_kmeans = list(zip(geo.longitude, geo.latitude))
    geo['id'] = graph._predict_node(X_kmeans)

    logger.info(f'Unique id : {np.unique(geo["id"].values)}')
    logger.info(f'{len(geo)} point in the dataset. Constructing database')

    return geo

def fire_prediction(interface, departement, features, train_features, scaling, dir_data, spec, dates, df_train, apply_break_point):

    features = copy.copy(features)
    train_features = copy.copy(train_features)
    
    logger.info('Assign hexagon to graph nodes')
    geoDT = create_ps(interface)

    logger.info(f'Fix missing features {air_variables, vigicrues_variables, nappes_variables}')
    # Add thoses variables in predictops
    geoDT[air_variables] = 0
    geoDT[vigicrues_variables] = 0
    geoDT[nappes_variables] = 0
    geoDT[geo_variables] = int(departement.split('-')[1])

    logger.info(f'Rename {sinister} to AutoRegressionBin')
    if sinister == 'firepoint':
        geoDT.rename({'nbfirepoint' : 'AutoRegressionBin'}, axis=1, inplace=True)

    logger.info('Drop duplicated point due to graph scale + assign weight to correct point')
    geoDT['weights'] = 0
    geoDT['departement'] = int(departement.split('-')[1])
    columns = ['id', 'longitude', 'latitude', 'departement', 'date', 'weights']
    orinode = geoDT.drop_duplicates(['id', 'date'])[columns].values
    orinode = graph._assign_latitude_longitude(orinode)
    orinode[:,3] = name2int[departement]
    orinode[orinode[:, 4] == k_limit, 5] = 1

    if suffix != '':
        model_name = f'{model}_{target}_{task_type}_{loss}_{suffix}'
    else:
        model_name = f'{model}_{target}_{task_type}_{loss}'

    logger.info(f'Load model {model}')
    if model not in traditionnal_models:
        name_model =  model + '/' + 'best.pt'
        dir = 'check_'+scaling + '/' + prefix_train + '/'
        graph._load_model_from_path(dir_data / spec/ dir / name_model, dico_model[model], device)
    else:
        dir = 'check_'+scaling + '/' + prefix_train + '/' + '/baseline'
        model_ = read_object(f'{model_name}.pkl', dir_data / spec / dir / model_name)
        graph._set_model(model_)

    logger.info(f'Load features')
    features_selected = read_object('features.pkl', dir_data / spec / dir / model_name)

    logger.info('Generate X from dataframe -> could be improove')
    X, _ = get_sub_nodes_feature(graph=graph,
                              subNode=orinode,
                              features=features,
                              departement=departement,
                              geo=geoDT,
                              path=dir_data,
                              dates=[d.strftime('%Y-%m-%d') for d in dates],
                              resolution=resolution)

    if int(args.days) > 0:
        logger.info(f'Add time vayring features from 1 -> {args.days}')
        for k in range(1, int(args.days) + 1):
            new_fet = [f'{v}_{k}' for v in varying_time_variables]
            train_features += [nf for nf in new_fet if nf.split('_')[0] in train_features]
            features += new_fet 
            features_name, newShape = get_features_name_list(graph.scale, features, METHODS_TEMPORAL)
            X = add_varying_time_features(X=X, Y=orinode, features=new_fet, newShape=newShape, features_name=features_name, ks=k, scale=scale, methods=METHODS_TEMPORAL)
    
    df = pd.DataFrame(columns=ids_columns + features_name, index=np.arange(0, X.shape[0]))
    df[features_name] = X
    df[ids_columns] = orinode

    df.to_csv(f'log/{date_ks}.csv', index=False)
    
    features_name, _ = get_features_name_list(graph.scale, train_features, METHODS_SPATIAL_TRAIN)
    
    fire_feature = ['fire_prediction_raw', 'fire_prediction', 'fire_prediction_dept']

    df_shape = df.shape
    df = df[ids_columns + features_name]
    logger.info(f'Select train feature : shape : {df_shape} -> {df.shape}')
    logger.info(f'Preprocess X (scale {scaling})')
    df = preprocess_inference(df, df_train, scaling, features_name, fire_feature, apply_break_point, scale, kmeans_features, dir_break_point)
    index_to_predict = df[df['fire_prediction_raw'].isna()].index
    logger.info(f'Found {len(df) - index_to_predict.shape[0]} values where fire has no chance')

    if df is None:
        return None
    
    df.to_csv(f'log/{date_ks}_after_scale.csv', index=False)

    if df is not None:
        if model in traditionnal_models:
            Y = graph.predict_model_api_sklearn(X=df.loc[index_to_predict], features=features_selected, isBin=False, autoRegression=False)
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
    logger.info('Features importances')
    
    df.loc[index_to_predict, 'fire_prediction_raw'] = Y
    df.loc[index_to_predict, 'fire_prediction'] = order_class(predictor, predictor.predict(Y), 1)
    df = df[df['date'] == k_limit]
    risk = np.sum(df['fire_prediction_raw'].values)
    df['fire_prediction_dept'] = order_class(predictorDept, predictorDept.predict(np.array(risk).reshape(-1,1)), 1)[0]

    if model in traditionnal_models:
        graph.model.plot_features_importance(df[features_selected], df['fire_prediction_raw'], features_selected, date_limit.strftime('%Y-%m-%d'),
                                            dir_interface, mode = 'bar', figsize=(15,5))
    else:
        logger.info('No Features importances implemented for DL')

    logger.info('Saving in dataframes')

    geoDT = geoDT.set_index('id').join(df.set_index('id')[fire_feature], on='id').reset_index()
    return geoDT

######################## Read geo dataframe ######################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Inference',
        description='Inference prediction',
    )
    parser.add_argument('-d', '--days', type=str, help='Number of days')
    parser.add_argument('-s', '--scale', type=str, help='Scale')
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
    parser.add_argument('-spec', '--spec', type=str, help='Specifity')
    parser.add_argument('-exp', '--experiment', type=str, help='Experiment')
    parser.add_argument('-su', '--suffix', type=str, help='Suffix of model name')
    parser.add_argument('-abp', '--applyBreakpoint', type=str, help='Apply Kmeans on target')
    args = parser.parse_args()

    # Input config
    k_days = int(args.days)
    scale = int(args.scale)
    dept = args.departement
    sinister = args.sinister
    model = args.model
    minPoint = args.nbpoint
    target = args.target
    loss = args.loss
    task_type = args.task_type
    nbFeature = int(args.nbFeature)
    resolution = args.resolution
    doHistorical = args.doHistorical == 'True'
    scaling = 'z-score' # Scale to used
    spec = args.spec
    suffix = args.suffix
    exp = args.experiment
    apply_break_point = args.applyBreakpoint == 'True'

    dir_data = Path('../GNN/'+exp+'/' + sinister + '/' + resolution + '/train/')

    #dir_data = dir_incendie / 'inference' / sinister / 'train'
    #dir_feather = Path('/home/ubuntu/Predictops_'+dept.split('-')[1]+'/data/dataframes')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # mean max min std ans sum for scale > 0 else value
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
                'landcover',
                'vigicrues',
                'foret',
                'highway',
                'dynamicWorld',
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
                    'isi', 'angstroem', 'bui', 'fwi', 'dailySeverityRating',
                    'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
                    'days_since_rain', 'sum_consecutive_rainfall',
                    'sum_rain_last_7_days',
                    'sum_snow_last_7_days', 'snow24h', 'snow24h16',
                    'elevation',
                    'population',
                    #'sentinel',
                    'landcover',
                    #'vigicrues',
                    'foret',
                    'highway',
                    'dynamicWorld',
                    'Calendar',
                    'Historical',
                    'Geo',
                    #'air',
                    #'nappes',
                    #'AutoRegressionReg',
                    'AutoRegressionBin'
                    ]
    

    kmeans_features = [
            #'temp',
            #'dwpt',
            'rhum',
            #'prcp',
            #'wdir',
            #'wspd',
            'prec24h',
            #'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
            #'isi', 'angstroem', 'bui', 'fwi', 'dailySeverityRating',
            #'temp16',
            #'dwpt16',
            'rhum16',
            #'prcp16',
            #'wdir16', 'wspd16',
            'prec24h16',
            #'days_since_rain',
            'sum_consecutive_rainfall',
            'sum_rain_last_7_days',
            'sum_snow_last_7_days',
            'snow24h', 'snow24h16',
            #'elevation',
            #'population',
            #'sentinel',
            #'landcover',
            #'vigicrues',
            #'foret',
            #'highway',
            #'dynamicWorld',
            #'Calendar',
            #'Historical',
            #'Geo',
            #'air',
            #'nappes',
            #'AutoRegressionReg',
            #'AutoRegressionBin'
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
        

    traditionnal_models = ['xgboost', 'lightgbm', 'ngboost']

    # Load train
    graph = read_object(f'graph_{scale}.pkl', dir_data)
    predictor = read_object(dept+'Predictor'+str(scale)+'.pkl', dir_data / 'influenceClustering')
    predictorDept = read_object(dept+'PredictorDepartement.pkl', dir_data / 'influenceClustering')

    prefix_train = f'{str(minPoint)}_{str(k_days)}_{str(scale)}_{str(nbFeature)}'
    
    df_train = read_object(f'df_train_{prefix_train}.pkl', dir_data / spec)
    dir_break_point = dir_data / spec / 'check_none' / prefix_train / 'kmeans'
    
    assert df_train is not None

    # Arborescence
    dir_incendie = Path('/home/caron/Bureau/csv/'+dept+'/data/')
    dir_interface = Path('interface')
    dir_output = dir_interface / 'output'
    check_and_create_path(dir_output)

    # Load feather and spatial
    name = 'hexagones_'+sinister+'.geojson'
    spatialGeo = gpd.read_file(dir_incendie / 'spatial' / name)
    incendie_feather_ori = pd.read_feather(dir_interface / 'incendie_final.feather')
    spa = 3

    #dir_output = Path('log')
    #check_and_create_path(dir_output)

    # Load feather and spatial
    #name = 'hexagones_'+sinister+'.geojson'
    #spatialGeo = gpd.read_file(dir_incendie / 'spatial' / name)
    #incendie_feather_ori = pd.read_feather(dir_feather / 'incendie_final.feather')

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
    elif sinister == "inondation":
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
        exit(2)

    if doHistorical:
        sd = dt.datetime.strptime('2024-07-01', '%Y-%m-%d').date()
        ed = dt.datetime.now().date() + dt.timedelta(days=2)
    else:
        sd = dt.datetime.now().date() - dt.timedelta(days=2)
        ed = dt.datetime.now().date() + dt.timedelta(days=2)
    d = sd
    while d != ed:
        # Date
        date_limit = d
        date_ks = (date_limit - dt.timedelta(days=k_days_conv))
        incendie_feather = incendie_feather_ori[(incendie_feather_ori['date'] >= date_ks) & (incendie_feather_ori['date'] <= date_limit)]
        dates = np.sort(incendie_feather.date.unique())
        if date_limit not in dates:
            logger.info(f'No data for {date_limit}')
            d += dt.timedelta(days=1)
            continue
        logger.info(dates)
        if len(dates) == 0:
            continue
        ######################### Interface ##################################
        interface = []

        for ks, date in enumerate(dates):

            geo_data = spatialGeo.copy(deep=True)
            geo_data.index = geo_data.hex_id
            geo_data = geo_data.set_index('hex_id').join(incendie_feather[incendie_feather['date'] == date][features_in_feather].set_index('hex_id'),
                                                    on='hex_id')
            
            geo_data.reset_index(inplace=True)
            geo_data[spatialGeo['population'].isna()][features_in_feather[:1]] = np.nan
            geo_data['date'] = ks
            geo_data['date_str'] = date.strftime('%Y-%m-%d')
            geo_data['longitude'] = geo_data['geometry'].apply(lambda x : float(x.centroid.x))
            geo_data['latitude'] = geo_data['geometry'].apply(lambda x : float(x.centroid.y))

            interface.append(geo_data)

        interface = pd.concat(interface).reset_index(drop=True)
        k_limit = interface.date.max()

        if sinister == 'firepoint':
            interface.rename({'feux': 'nbfirepoint'}, axis=1, inplace=True)
            interface['nbfirepoint'].fillna(0, inplace=True)

        logger.info('Rename features to match with train fet')
        interface.rename({'temp12' : 'temp', 'dwpt12' : 'dwpt',
                        'rhum12' : 'rhum', 'prcp12' : 'prcp', 'wdir12' : 'wdir',
                        'wspd12' : 'wspd', 'prec24h12' : 'prec24h',
                        'snow24h12': 'snow24h',
                        'daily_severity_rating' : 'dailySeverityRating',
                        'osmnx' : 'highway'}, inplace=True, axis=1)

        ######################### Preprocess #################################
        logger.info(f'Create past influence using convolution and PCA')
        n_pixel_x = 0.02875215641173088
        n_pixel_y = 0.020721094073767096

        mask = read_object(f'{dept}rasterScale{scale}.pkl', dir_data / 'raster')
        check_and_create_path(Path('log'))
        inputDep = create_spatio_temporal_sinister_image(interface,
                                                        mask,
                                                        sinister,
                                                        n_pixel_y,
                                                        n_pixel_x,
                                                        Path('log'),
                                                        dept)

        Probabilistic(n_pixel_x, n_pixel_y, 1, None, Path('./'), resolution)._process_input_raster(dims, [inputDep], 1, True,
                                                                                                 [dept], True, dates, [dept], True)
        
        ######################### Load autoregression ####################
        interface['AutoRegressionReg'] = 0
        if 'AutoRegressionReg' in train_features:
            previous_date = (date_limit - dt.timdelta(days=1)).strftime('%Y-%m-%d')
            on = f'hexagones_{previous_date}.geojson'
            logger.info('AutoRegressionReg mode -> loading {previous_date}')
            if (dir_interface / '..' / 'scripts' / 'global' / 'sinister_prediction' / 'interface' / on).is_file():
                logger.info(f'{on} found')
                previous_log_interface = gpd.read_file(dir_interface / '..' / 'scripts' / 'global' / 'sinister_prediction' / 'interface' / on)
                if 'fire_prediction_raw' not in previous_log_interface.columns:
                    logger.info(f'fire_prediction_raw not found in previous_log_interface.columns. \
                                Please launch with doHistorical at True to construct previous prediction, current {doHistorical}')
                    logger.info('AutoRegressionReg is set to 0')
                else:
                    interface.loc[interface[interface['date'] == k_limit].index, 'AutoRegressionReg'] = previous_log_interface['fire_prediction_raw']
            else:
                logger.info(f'{on} not found. Please launch with doHistorical at True to construct previous prediction, current {doHistorical}')
                logger.info('AutoRegressionReg is set to 0')

        ######################### Predict ################################

        logger.info(f'Launch {sinister} Prediction')
        interface = fire_prediction(interface, dept, features, train_features, scaling, dir_data, spec, dates, df_train, apply_break_point)
        if interface is None:
            logger.info(f'Can t produce fire prediction for {dates[-1]}')
            d += dt.timedelta(days=1)
            continue

        ######################### Saving ###############################
        if date_limit == dt.datetime.now().date():
            input = 'today'
            output = 'today'
        elif date_limit == dt.datetime.now().date() + dt.timedelta(days=1):
            input = 'tomorrow'
            output = 'tomorrow'
        else:
            input = 'today'
            
        output_geojson = gpd.read_file(dir_interface / f'hexagones_{input}0.geojson')
        fire_feature = ['fire_prediction_raw', 'fire_prediction', 'fire_prediction_dept']

        for ff in fire_feature:
            if ff in output_geojson.columns:
                output_geojson.drop(ff, inplace=True, axis=1)

        if date_limit == dt.datetime.now().date() or date_limit == dt.datetime.now().date() + dt.timedelta(days=1):
            logger.info(f'Saving to hexagones_{output}_final.geojson')
            output_geojson = output_geojson.set_index('hex_id').join(interface[interface['date'] == k_limit].set_index('hex_id')[fire_feature],
                                                        on='hex_id')
            output_geojson.reset_index(inplace=True)
            output_geojson.to_file(dir_interface / f'hexagones_{output}_final.geojson', driver='GeoJSON')
        
        previous_date = (date_limit - dt.timedelta(days=1)).strftime('%Y-%m-%d')
        on = f'hexagones_{previous_date}.geojson'
        logger.info(f'Open {on} to save last recorded fires')
        #if (dir_interface / '..' / 'scripts' / 'global' / 'sinister_prediction' / 'interface' / on).is_file():
        if (dir_interface / dept / "interface" / on).is_file():
            previous_log_interface = gpd.read_file(dir_interface / dept / "interface" / on)
            #previous_log_interface = gpd.read_file(dir_interface / '..' / 'scripts' / 'global' / 'sinister_prediction' / 'interface' / on)
            previous_log_interface['nbfirepoint'] = interface[interface['date'] == k_limit - 1]['AutoRegressionBin'].values
            #previous_log_interface.to_file(dir_interface / '..' / 'scripts' / 'global' / 'sinister_prediction' / 'interface' / on, driver='GeoJSON')
            previous_log_interface.to_file(dir_interface / dept / "interface" / on, driver='GeoJSON')
        
        output = date_limit.strftime('%Y-%m-%d')
        log_interface = interface[interface['date'] == k_limit]            
        on = f'hexagones_{output}.geojson'
        logger.info(f'Saving to {on}')
        log_interface.to_file(dir_interface / dept / 'interface' / on, driver='GeoJSON')

        d += dt.timedelta(days=1)