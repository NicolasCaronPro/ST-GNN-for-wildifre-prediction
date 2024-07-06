from tools_inference import *
import datetime as dt
import json
from probabilistic import *
import argparse
import logging
from array_fet import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")

# Handler pour afficher les logs dans le terminal
if (logger.hasHandlers()):
    logger.handlers.clear()
streamHandler = logging.StreamHandler(stream=sys.stdout)
streamHandler.setFormatter(logFormatter)
logger.addHandler(streamHandler)

####################### Function #################################

def create_ps(geo):
    X_kmeans = list(zip(geo.longitude, geo.latitude))
    geo['id'] = graph._predict_node(X_kmeans)

    logger.info(f'Unique id : {np.unique(geo["id"].values)}')
    logger.info(f'{len(geo)} point in the dataset. Constructing database')

    return geo

def fire_prediction(interface, departement, features, trainFeatures, scaling, dir_data, dates, Xtrain, Ytrain):
    
    geoDT = create_ps(interface)

    # Based tree models
    traditionnal_models = ['xgboost', 'lightgbm', 'ngboost']

    geoDT[geo_variables] = int(departement.split('-')[1])

    geoDT[auto_regression_variable_bin] = 0
    geoDT[auto_regression_variable_reg] = 0
    geoDT[landcover_variables] = 0
    geoDT[air_variables] = 0
    geoDT[vigicrues_variables] = 0
    geoDT[nappes_variables] = 0

    geoDT['weights'] = 0
    geoDT['departement'] = int(departement.split('-')[1])
    columns = ['id', 'longitude', 'latitude', 'departement', 'date', 'weights']
    orinode = geoDT.drop_duplicates(['id', 'date'])[columns].values
    orinode = graph._assign_latitude_longitude(orinode)
    #orinode = graph._assign_department(orinode)

    orinode[:,3] = name2int[departement]
    orinode[np.isin(orinode[:, 4], [k_today, k_tomorrow]), 5] = 1

    if model not in traditionnal_models:
        name_model =  model + '/' + 'best.pt'
        dir = 'check_'+scaling + '/' + prefix_train + '/'
        graph._load_model_from_path(dir_data / dir / name_model, dico_model[model], device)
    else:
        dir = 'check_'+scaling + '/' + prefix_train + '/' + '/baseline'
        model_ = read_object(model+'.pkl', dir_data / dir / model)
        graph._set_model(model_)

    features_selected = read_object('features.pkl', dir_data / dir / model).astype(int)
    X, _ = get_sub_nodes_feature(graph=graph,
                              subNode=orinode,
                              features=features,
                              dept=departement,
                              mask=mask,
                              geo=geoDT,
                              path=dir_data,
                              dates=dates,
                              logger=logger)
    
    logger.info(np.unique(np.argwhere(np.isnan(X))[:,1]))
    logger.info(X.shape)

    if int(args.days) > 0:
        trainFeatures += varying_time_variables
        features += varying_time_variables
        pos_feature, newShape = create_pos_feature(graph.scale, 6, features)
        X = add_varying_time_features(X=X, features=varying_time_variables, newShape=newShape, pos_feature=pos_feature, ks=k_days)

    # Predict Today
    today_X, E = create_inference(graph,
                     X,
                     Xtrain,
                     device,
                     False,
                     k_today,
                     int(args.days),
                     scaling)
            
    if today_X is None:
        raise Exception('Today x is None')

    if today_X is not None:
        if model in traditionnal_models:
            today_X = today_X.detach().cpu().numpy()
            today_Y = graph.predict_model_api_sklearn(today_X[:, features_selected], False)
        else:
            today_Y = graph._predict_tensor(today_X, E)
    else:
        today_Y = None

    # Predict Tomorrow
    tomorrow_Y = None
    if k_tomorrow in np.unique(X[:,4]):
        tomorrow_X, E = create_inference(graph,
                        X,
                        Xtrain,
                        device,
                        False,
                        k_tomorrow,
                        int(args.days),
                        scaling)
        
        if tomorrow_X is not None:
            if model in traditionnal_models:
                tomorrow_X = tomorrow_X.detach().cpu().numpy()
                tomorrow_Y = graph.predict_model_api_sklearn(tomorrow_X[:, features_selected], False)
            else:
                tomorrow_Y = graph._predict_tensor(tomorrow_X, E)
        else:
            tomorrow_Y = None

    # Saving in dataframes 
    fire_feature = ['fire_prediction_raw', 'fire_prediction', 'fire_prediction_dept']
    geoDT[fire_feature] = np.nan
    unodes = np.unique(orinode[:,0])
    for node in unodes:
        if today_Y is not None:
            geoDT.loc[geoDT[(geoDT['date'] == k_today) &
                                    (geoDT['id'] == node)].index, 'fire_prediction_raw'] = today_Y[np.argwhere(today_X[:, 0] == node)[:,0]][0]
            
            geoDT.loc[geoDT[(geoDT['date'] == k_today) &
                                    (geoDT['id'] == node)].index, 'fire_prediction'] = order_class(predictor, predictor.predict(today_Y[np.argwhere(today_X[:, 0] == node)[:,0]], 1))[0]
            
        if k_tomorrow in np.unique(X[:,4]) and tomorrow_Y is not None:

            geoDT.loc[interface[(geoDT['date'] == k_tomorrow) &
                                    (geoDT['id'] == node)].index, 'fire_prediction_raw'] = tomorrow_Y[np.argwhere(tomorrow_X[:, 0] == node)[:,0]][0]
            
            geoDT.loc[geoDT[(geoDT['date'] == k_tomorrow) &
                                (geoDT['id'] == node)].index, 'fire_prediction'] = order_class(predictor, predictor.predict(tomorrow_Y[np.argwhere(tomorrow_X[:, 0] == node)[:,0]], 1))[0]
    
    geoDT['fire_prediction_dept'] = 0

    if today_X is not None:
        today_risk = geoDT[(geoDT['date'] == k_today)].groupby('id')['fire_prediction_raw'].mean().values
        today_risk = np.nansum(today_risk)
        geoDT.loc[geoDT[(geoDT['date'] == k_today)].index, 'fire_prediction_dept'] = order_class(predictorDept, predictorDept.predict(np.array(today_risk)))[0]
        
    if k_tomorrow in np.unique(X[:,4]) and tomorrow_Y is not None:
        tomorrow_risk = geoDT[(geoDT['date'] == k_tomorrow)].groupby('id')['fire_prediction_raw'].mean().values
        tomorrow_risk = np.nansum(tomorrow_risk)
        geoDT.loc[geoDT[(geoDT['date'] == k_tomorrow)].index, 'fire_prediction_dept'] = order_class(predictorDept, predictorDept.predict(np.array(tomorrow_risk)))[0]

    return geoDT

def add_varying_time_features(X : np.array, features : list, newShape : int, pos_feature : dict, ks : int):
    res = np.empty((X.shape[0], newShape))
    res[:, :X.shape[1]] = X
    for feat in features:
        logger.info(feat)
        vec = feat.split('_')
        name = vec[0]
        methods = vec[1:]
        index_feat = pos_feature[feat]
        index_feat_X = pos_feature[name]
        for met in methods:
            if met == "mean":
                for node in X:
                    index_node = np.argwhere((X[:,0] == node[0]) & (X[:,4] == node[4]))
                    index_nodes_ks = np.argwhere((X[:,0] == node[0]) & (X[:,4] < node[4]) & (X[:,4] >= node[4] - ks))
                    if name == 'Calendar':
                        for i, _ in enumerate(calendar_variables):
                            res[index_node, index_feat + i] = np.mean(X[index_nodes_ks, index_feat_X + i])
                    elif name == 'Historical':
                        for i, _ in enumerate(historical_variables):
                            res[index_node, index_feat + i] = np.mean(X[index_nodes_ks, index_feat_X + i])
                    elif name == "air":
                        for i, _ in enumerate(air_variables):
                            res[index_node, index_feat + i] = np.mean(X[index_nodes_ks, index_feat_X + i])
                    else:
                        res[index_node, index_feat] = np.mean(X[index_nodes_ks, index_feat_X])
            elif met == "sum":
                    index_node = np.argwhere((X[:,0] == node[0]) & (X[:,4] == node[4]))
                    index_nodes_ks = np.argwhere((X[:,0] == node[0]) & (X[:,4] < node[4]) & (X[:,4] >= node[4] - ks))
                    res[index_node, index_feat] = np.sum(X[index_nodes_ks, index_feat_X])
            else:
                logger.info(f'{met} unknow method')
                exit(1)
    return res

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
    args = parser.parse_args()

    # Input config
    k_days = int(args.days)
    scale = args.scale
    dept = args.departement
    sinister = args.sinister
    model = args.model
    minPoint = args.nbpoint
    nbFeature = int(args.nbFeature)
    doHistorical = args.doHistorical == 'True'
    scaling='z-score' # Scale to used

    dir_data = Path('../GNN/inference/' + sinister + '/train')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # mean max min and std for scale > 0 else value
    features = [
                'temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
                'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                'isi', 'angstroem', 'bui', 'fwi', 'dailySeverityRating',
                'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
                'days_since_rain', 'sum_consecutive_rainfall', 'sum_last_7_days',
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
                'days_since_rain', 'sum_consecutive_rainfall', 'sum_last_7_days',
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
                #'AutoRegressionBin'
                ]
    
    features_in_feather = ['hex_id', 'date',
                           'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
                           'temp12', 'dwpt12', 'rhum12', 'prcp12', 'wdir12', 'wspd12', 'prec24h12',
                           'dc', 'ffmc', 'dmc', 'isi', 'bui', 'fwi',
                           'daily_severity_rating', 'nesterov', 'munger', 'kbdi', 'angstroem',
                           'days_since_rain', 'sum_consecutive_rainfall', 'sum_last_7_days',
                           'feux',
                           ]
    
    if minPoint == 'full':
        prefix_train = str(minPoint)+'_'+str(scale)
    else:
        prefix_train = str(minPoint)+'_'+str(k_days)+'_'+str(scale)

    prefix = str(scale)

    traditionnal_models = ['xgboost', 'lightgbm', 'ngboost', 'xgboost_bin', 'lightgbm_bin', 'ngboost_bin', 
                           'xgboost_unweighted', 'lightgbm_unweighted', 'ngboost_unweighted',
                           'xgboost_bin_unweighted', 'lightgbm_bin_unweighted', 'ngboost_bin_unweighted']

    # Load train
    graph = read_object('graph_'+scale+'.pkl', dir_data)
    Xtrain = read_object('X_'+prefix_train+'.pkl', dir_data)
    Ytrain = read_object('Y_'+prefix_train+'.pkl', dir_data)
    predictor = read_object(dept+'Predictor'+str(scale)+'.pkl', dir_data / 'influenceClustering')
    predictorDept = read_object(dept+'PredictorDepartement.pkl', dir_data / 'influenceClustering')

    prefix_train = str(minPoint)+'_'+str(k_days)+'_'+str(scale)+'_'+str(nbFeature)

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
            dims = [(spa, spa, 11)]
        elif dept == 'departement-25-doubs':
            dims = [(spa,spa, 7)]
        elif dept == 'departement-69-rhone':
            dims = [(spa, spa, 5)]
        elif dept == 'departement-78-yvelines':
            dims = [(spa, spa, 9)]
    elif sinister == "inondation":
        dims = [(spa,spa,5)]

    k_days = max(dims[0][2], k_days)

    if doHistorical:
        sd = dt.datetime.strptime('2024-01-01', '%Y-%m-%d').date()
        ed = dt.datetime.now().date() + dt.timedelta(days=1)
    else:
        sd = dt.datetime.now().date()
        ed = dt.datetime.now().date() + dt.timedelta(days=1)
    d = sd
    while d != ed:
        # Date
        date_today = d
        date_ks = (date_today - dt.timedelta(days=k_days))
        incendie_feather = incendie_feather_ori[(incendie_feather_ori['date'] >= date_ks) & (incendie_feather_ori['date'] <= date_today)]
        dates = np.sort(incendie_feather.date.unique())
        if date_today - dt.timedelta(days=1) not in dates:
            logger.info(f'No data for {date_today- dt.timedelta(days=1)}')
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

        interface.fillna(0, inplace=True)

        if date_today in incendie_feather.date.unique():
            k_today = interface.date.max() - 1
            k_tomorrow = interface.date.max()
        else:
            k_today = interface.date.max()
            k_tomorrow = interface.date.max() + 1

        if sinister == 'firepoint':
            interface.rename({'feux': 'nbfirepoint'}, axis=1, inplace=True)

        interface.rename({'temp12' : 'temp', 'dwpt12' : 'dwpt',
                        'rhum12' : 'rhum', 'prcp12' : 'prcp', 'wdir12' : 'wdir',
                        'wspd12' : 'wspd', 'prec24h12' : 'prec24h',
                        'daily_severity_rating' : 'dailySeverityRating',
                        'osmnx' : 'highway'}, inplace=True, axis=1)

        ######################### Preprocess #################################
        n_pixel_x = 0.02875215641173088
        n_pixel_y = 0.020721094073767096

        mask = read_object(dept+'rasterScale'+scale+'.pkl', dir_data / 'raster' / '2x2')
        check_and_create_path(Path('log'))
        inputDep = create_spatio_temporal_sinister_image(interface,
                                                        mask,
                                                        sinister,
                                                        n_pixel_y,
                                                        n_pixel_x,
                                                        Path('log'),
                                                        dept)

        Probabilistic(dims, n_pixel_x, n_pixel_y, 1, None, Path('./'))._process_input_raster([inputDep], 1, True, [dept], True)

        ######################### Predict ################################

        interface = fire_prediction(spatialGeo, interface, dept, features, trainFeatures, features_selected, scaling, dir_data, dates)

        ######################### Saving ###############################

        today = gpd.read_file(dir_interface / 'hexagones_today.geojson')
        fire_feature = ['fire_prediction_raw', 'fire_prediction', 'fire_prediction_dept']

        for ff in fire_feature:
            if ff in today.columns:
                today.drop(ff, inplace=True, axis=1)

        today = today.set_index('hex_id').join(interface[interface['date'] == k_today].set_index('hex_id')[fire_feature],
                                                    on='hex_id')
        today.reset_index(inplace=True)
        today.to_file(dir_interface / 'hexagones_today_final.geojson', driver='GeoJSON')
        
        # Save daily fire in previous day prediction

        on = 'hexagones_today_'+(date_today - dt.timedelta(days=1)).strftime('%Y-%m-%d')+'.geojson'
        if (dir_interface / '..' / 'scripts' / 'global' / 'sinister_prediction' / 'interface' / on).is_file():
            previous_log_interface = gpd.read_file(dir_interface / '..' / 'scripts' / 'global' / 'sinister_prediction' / 'interface' / on)
            previous_log_interface['nbfirepoint'] = interface[interface['date'] == k_today]['nbfirepoint'].values
            previous_log_interface.to_file(dir_interface / '..' / 'scripts' / 'global' / 'sinister_prediction' / 'interface' / on, driver='GeoJSON')
        
        # Save current prediction
        log_interface = interface[interface['date'] == k_today]
        if (k_today + 1) in interface.date.unique():
            log_interface['nbfirepoint'] = interface[interface['date'] == k_today + 1]['nbfirepoint'].values
        
        on = 'hexagones_today_'+date_today.strftime('%Y-%m-%d')+'.geojson'
        log_interface.to_file(dir_interface / '..' / 'scripts' / 'global' / 'sinister_prediction' / 'interface' / on, driver='GeoJSON')

        if k_tomorrow in interface.date.unique():

            tomorrow = gpd.read_file(dir_interface / 'hexagones_tomorrow.geojson')
            for ff in fire_feature:
                if ff in tomorrow.columns:
                    tomorrow.drop(ff, inplace=True, axis=1)

            tomorrow = tomorrow.set_index('hex_id').join(interface[interface['date'] == k_tomorrow].set_index('hex_id')[fire_feature],
                                                    on='hex_id')
            tomorrow.reset_index(inplace=True)
            tomorrow.to_file(dir_interface / 'hexagones_tomorrow_final.geojson',  driver='GeoJSON')

        d += dt.timedelta(days=1)