from graph_structure import *
import datetime as dt
import json
from probabilistic import *
import argparse
from array_fet import *

####################### Function #################################

def create_ps(geo):
    X_kmeans = list(zip(geo.longitude, geo.latitude))
    geo['id'] = graph._predict_node(X_kmeans)

    logger.info(f'Unique id : {np.unique(geo["id"].values)}')
    logger.info(f'{len(geo)} point in the dataset. Constructing database')

    return geo

def fire_prediction(interface, departement, features, trainFeatures, scaling, dir_data, spec, dates, Xtrain, Ytrain):
    
    logger.info('Assign hexagon to graph nodes')
    geoDT = create_ps(interface)
    
    logger.info(f'Fix missing features {landcover_variables, air_variables, vigicrues_variables, nappes_variables}')
    # Add thoses variables in predictops
    geoDT[landcover_variables] = 0
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

    logger.info(f'Load model {model}')
    if model not in traditionnal_models:
        name_model =  model + '/' + 'best.pt'
        dir = 'check_'+scaling + '/' + prefix_train + '/'
        graph._load_model_from_path(dir_data / spec/ dir / name_model, dico_model[model], device)
    else:
        dir = 'check_'+scaling + '/' + prefix_train + '/' + '/baseline'
        model_ = read_object(model+'.pkl', dir_data / spec / dir / model)
        graph._set_model(model_)

    logger.info(f'Load features')
    features_selected = read_object('features.pkl', dir_data / spec / dir / model).astype(int)

    logger.info('Generate X from dataframe -> could be improove')
    X, _ = get_sub_nodes_feature(graph=graph,
                              subNode=orinode,
                              features=features,
                              departement=departement,
                              geo=geoDT,
                              path=dir_data,
                              dates=[d.strftime('%Y-%m-%d') for d in dates])

    if int(args.days) > 0:
        logger.info(f'Add time vayring features from 1 -> {args.days}')
        for k in range(1, int(args.days) + 1):
            new_fet = [f'{v}_{k}' for v in varying_time_variables]
            trainFeatures += [nf for nf in new_fet if nf.split('_')[0] in trainFeatures]
            features += new_fet 
            features_name, newShape = get_features_name_list(graph.scale, 6, features)
            X = add_varying_time_features(X=X, features=new_fet, newShape=newShape, features_name=features_name, ks=k, scale=scale)
            Xtrain = add_varying_time_features(X=Xtrain, features=new_fet, newShape=newShape, features_name=features_name, ks=k, scale=scale)

    x_shape = X.shape
    train_fet_num = select_train_features(trainFeatures, scale, features_name)
    X = X[:, train_fet_num]
    Xtrain = Xtrain[:, train_fet_num]

    logger.info(f'Select train feature : shape : {x_shape} -> {X.shape}')
    logger.info(f'Preprocess X (scale {scaling})')
    X = preprocess_inference(X, Xtrain, scaling, features_name)

    logger.info('Generate X and edges, use for GNN')
    # Predict Today
    X, E = create_inference(graph,
                     X,
                     device,
                     False,
                     k_limit,
                     int(args.days))
            
    if X is None:
        raise Exception('Today x is None')

    if X is not None:
        if model in traditionnal_models:
            X = X.detach().cpu().numpy()
            Y = graph.predict_model_api_sklearn(X=X, features=features_selected, isBin=False, autoRegression=False, features_name=features_name)
        else:
            Y = graph._predict_tensor(X, E)
    else:
        Y = None

    logger.info('Features importances')
    #model._plot_features_importance(X[:, features_selected], Y, features_name, date_limit.strftime('%Y-%m-%d'),
    #                                dir_interface / '..' / 'scripts' / 'global' / 'sinister_prediction' / 'interface', mode = 'bar')
    if model in traditionnal_models:
        graph.model._plot_features_importance(X[:, features_selected], Y, features_name, date_limit.strftime('%Y-%m-%d'),
                                            dir_interface, mode = 'bar')
    else:
        logger.info('No Features importances implemented for DL')

    logger.info('Saving in dataframes') 
    fire_feature = ['fire_prediction_raw', 'fire_prediction', 'fire_prediction_dept']
    geoDT[fire_feature] = np.nan
    unodes = np.unique(orinode[:,0])
    for node in unodes:
        if Y is not None:
            geoDT.loc[geoDT[(geoDT['date'] == k_limit) &
                                    (geoDT['id'] == node)].index, 'fire_prediction_raw'] = Y[np.argwhere(X[:, 0] == node)[:,0]][0]
            
            geoDT.loc[geoDT[(geoDT['date'] == k_limit) &
                                    (geoDT['id'] == node)].index, 'fire_prediction'] = \
                                        order_class(predictor, predictor.predict(Y[np.argwhere(X[:, 0] == node)[:,0]], 1))[0]
            
    geoDT['fire_prediction_dept'] = np.nan
    if X is not None:
        risk = geoDT[(geoDT['date'] == k_limit)].groupby('id')['fire_prediction_raw'].mean().values
        risk = np.nansum(risk)
        geoDT.loc[geoDT[(geoDT['date'] == k_limit)].index, 'fire_prediction_dept'] = order_class(predictorDept, predictorDept.predict(np.array(risk).reshape(-1,1)))[0]
        
    return geoDT

######################## Read geo dataframe ######################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Inference',
        description='Inference prediction',
    )
    parser.add_argument('-d', '--days', type=str, help='Number of days')
    parser.add_argument('-s', '--scale', type=str, help='Scale')
    parser.add_argument('-dept', '--departement', type=str, help='departement')
    parser.add_argument('-sinis', '--sinister', type=str, help='Sinister')
    parser.add_argument('-m', '--model', type=str, help='Model')
    parser.add_argument('-np', '--nbpoint', type=str, help='Number of point')
    parser.add_argument('-nf', '--nbFeature', type=str, help='Number of feature')
    parser.add_argument('-dh', '--doHistorical', type=str, help='Do Historical')
    parser.add_argument('-r', '--resolution', type=str, help='Pixel resolution')
    parser.add_argument('-spec', '--spec', type=str, help='Specifity')
    args = parser.parse_args()

    # Input config
    k_days = int(args.days)
    scale = int(args.scale)
    dept = args.departement
    sinister = args.sinister
    model = args.model
    minPoint = args.nbpoint
    nbFeature = int(args.nbFeature)
    resolution = args.resolution
    doHistorical = args.doHistorical == 'True'
    scaling = 'z-score' # Scale to used
    spec = args.spec

    dir_data = Path('../GNN/final/' + sinister + '/' + resolution + '/train/')

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
    trainFeatures = [
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
    
    features_in_feather = ['hex_id', 'date',
                           'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
                           'temp12', 'dwpt12', 'rhum12', 'prcp12', 'wdir12', 'wspd12', 'prec24h12',
                           'dc', 'ffmc', 'dmc', 'isi', 'bui', 'fwi',
                           'daily_severity_rating', 'nesterov', 'munger', 'kbdi', 'angstroem',
                           'days_since_rain', 'sum_consecutive_rainfall', 'sum_last_7_days',
                           'sum_snow_last_7_days', 'snow24h12', 'snow24h16',
                           'feux',
                           ]
        
    if minPoint == 'full':
        prefix_train = f'{str(minPoint)}_{str(scale)}'
    else:
        prefix_train = f'{str(minPoint)}_{str(k_days)}_{str(scale)}'

    prefix = str(scale)

    traditionnal_models = ['xgboost_risk', 'lightgbm_risk', 'ngboost_risk', 'xgboost_bin', 'lightgbm_bin', 'ngboost_bin', 
                           'xgboost_unweighted', 'lightgbm_unweighted', 'ngboost_unweighted',
                           'xgboost_bin_unweighted', 'lightgbm_bin_unweighted', 'ngboost_bin_unweighted']

    # Load train
    graph = read_object(f'graph_{scale}.pkl', dir_data)
    Xtrain = read_object(f'X_{prefix_train}.pkl', dir_data)
    Ytrain = read_object(f'Y_{prefix_train}.pkl', dir_data)
    predictor = read_object(dept+'Predictor'+str(scale)+'.pkl', dir_data / 'influenceClustering')
    predictorDept = read_object(dept+'PredictorDepartement.pkl', dir_data / 'influenceClustering')

    prefix_train = f'{str(minPoint)}_{str(k_days)}_{str(scale)}_{str(nbFeature)}'

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

    if sinister == "firepoint":
        if dept == 'departement-01-ain':
            dims = [(spa,spa,5), (spa,spa,9), (spa,spa,3)]
            k_days = max(9, k_days)
        elif dept == 'departement-25-doubs':
            dims = [(spa,spa,3), (spa,spa,5), (spa,spa,3)],
            k_days = max(5, k_days)
        elif dept == 'departement-69-rhone':
            dims =  [(spa,spa,3), (spa,spa,5),(spa,spa,1)],
            k_days = max(5, k_days)
        elif dept == 'departement-78-yvelines':
            dims =  [(spa,spa,3), (spa,spa,7), (spa,spa,3)]
            k_days = max(7, k_days)
    elif sinister == "inondation":
        if dept == 'departement-01-ain':
            dims = [(spa,spa,5), (spa,spa,9), (spa,spa,3)]
            k_days = max(9, k_days)
        elif dept == 'departement-25-doubs':
            dims = [(spa,spa,3), (spa,spa,5), (spa,spa,3)],
            k_days = max(5, k_days)
        elif dept == 'departement-69-rhone':
            dims =  [(spa,spa,3), (spa,spa,5),(spa,spa,1)],
            k_days = max(5, k_days)
        elif dept == 'departement-78-yvelines':
            dims =  [(spa,spa,3), (spa,spa,7), (spa,spa,3)]
            k_days = max(7, k_days)
    else:
        exit(2)

    if doHistorical:
        sd = dt.datetime.strptime('2024-01-01', '%Y-%m-%d').date()
        ed = dt.datetime.now().date() + dt.timedelta(days=2)
    else:
        sd = dt.datetime.now().date() - dt.timedelta(days=1)
        ed = dt.datetime.now().date() + dt.timedelta(days=2)
    d = sd
    while d != ed:
        # Date
        date_limit = d
        date_ks = (date_limit - dt.timedelta(days=k_days))
        incendie_feather = incendie_feather_ori[(incendie_feather_ori['date'] >= date_ks) & (incendie_feather_ori['date'] <= date_limit)]
        dates = np.sort(incendie_feather.date.unique())
        if date_limit not in dates:
            logger.info(f'No data for {date_limit}')
            exit(0)
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
            geo_data['date'] = ks
            geo_data['date_str'] = date.strftime('%Y-%m-%d')
            geo_data['longitude'] = geo_data['geometry'].apply(lambda x : float(x.centroid.x))
            geo_data['latitude'] = geo_data['geometry'].apply(lambda x : float(x.centroid.y))
            interface.append(geo_data)

        interface = pd.concat(interface).reset_index(drop=True)

        k_limit = interface.date.max()

        interface.fillna(0, inplace=True)

        if sinister == 'firepoint':
            interface.rename({'feux': 'nbfirepoint'}, axis=1, inplace=True)

        logger.info('Rename features to match with train fet')
        interface.rename({'temp12' : 'temp', 'dwpt12' : 'dwpt',
                        'rhum12' : 'rhum', 'prcp12' : 'prcp', 'wdir12' : 'wdir',
                        'wspd12' : 'wspd', 'prec24h12' : 'prec24h',
                        'snow24h12': 'snow24h',
                        'daily_severity_rating' : 'dailySeverityRating',
                        'osmnx' : 'highway',
                        'sum_last_7_days' : 'sum_rain_last_7_days'}, inplace=True, axis=1)

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
        if 'AutoRegressionReg' in trainFeatures:
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
        interface = fire_prediction(interface, dept, features, trainFeatures, scaling, dir_data, spec, dates, Xtrain, Ytrain)

        ######################### Saving ###############################
        if date_limit == dt.datetime.now():
            input = 'today'
            output = 'today'
        elif date_limit == dt.datetime.now() + dt.timedelta(days=1):
            input = 'tomorrow'
            output = 'tomorrow'
        else:
            input = 'today'
            output = date_limit.strftime('%Y-%m-%d')

        output_geojson = gpd.read_file(dir_interface / f'hexagones_{input}.geojson')
        fire_feature = ['fire_prediction_raw', 'fire_prediction', 'fire_prediction_dept']

        for ff in fire_feature:
            if ff in output_geojson.columns:
                output_geojson.drop(ff, inplace=True, axis=1)

        if date_limit == dt.datetime.now() or date_limit == dt.datetime.now() + dt.timedelta(days=1):
            logger.info(f'Saving to {output}_final.geojson')
            output_geojson = output_geojson.set_index('hex_id').join(interface[interface['date'] == k_limit].set_index('hex_id')[fire_feature],
                                                        on='hex_id')
            output_geojson.reset_index(inplace=True)
            output_geojson.to_file(dir_interface / f'hexagones_{output}_final.geojson', driver='GeoJSON')
        
        previous_date = (date_limit - dt.timedelta(days=1)).strftime('%Y-%m-%d')
        on = f'hexagones_{previous_date}.geojson'
        logger.info(f'Open {on} to save last recorded fires')
        if (dir_interface / '..' / 'scripts' / 'global' / 'sinister_prediction' / 'interface' / on).is_file():
            previous_log_interface = gpd.read_file(dir_interface / '..' / 'scripts' / 'global' / 'sinister_prediction' / 'interface' / on)
            previous_log_interface['nbfirepoint'] = interface[interface['date'] == k_limit - 1]['nbfirepoint'].values
            previous_log_interface.to_file(dir_interface / '..' / 'scripts' / 'global' / 'sinister_prediction' / 'interface' / on, driver='GeoJSON')
        
        log_interface = interface[interface['date'] == k_limit]            
        on = f'hexagones_{output}.geojson'
        logger.info(f'Saving to {on}')
        log_interface.to_file(dir_interface, driver='GeoJSON')

        d += dt.timedelta(days=1)