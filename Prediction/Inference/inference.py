from tools_inference import *
import datetime as dt
import json
from probabilistic import *
import argparse
from array_fet import *

####################### Function #################################

def get_sub_nodes_feature(graph, subNode: np.array,
                        features : list,
                        path : Path,
                        geo : gpd.GeoDataFrame) -> np.array:
    
    print('Load nodes features')

    pos_feature, newShape = create_pos_feature(graph, subNode.shape[1], features)

    def save_values(array, indexVvar, indexNode, mask):
        if False not in np.unique(np.isnan(array[mask])):
            return
        if graph.scale > -1:
            X[indexNode, indexVvar] = round(np.nanmean(array[mask]), 3)
            X[indexNode, indexVvar+1] = round(np.nanmin(array[mask]), 3)
            X[indexNode, indexVvar+2] = round(np.nanmax(array[mask]), 3)
            X[indexNode, indexVvar+3] = round(np.nanstd(array[mask]), 3)
        else:
            X[indexNode, indexVvar] = np.nanmean(array[mask])

    def save_value(array, indexVvar, indexNode, mask):
        if False not in np.unique(np.isnan(array[mask])):
            return
        X[indexNode, indexVvar] = np.nanmean(array[mask])

    def save_values_count(array, lc, indexVvar, indexNode, mask):
        X[indexNode, indexVvar] = np.argwhere(array[mask] == float(lc)).shape[0]

    def save_value_with_encoding(array, indexVvar, indexNode, mask, encoder):
        values = array[mask].reshape(-1,1)
        encode_values = encoder.transform(values).values
        if graph.scale > -1:
            X[indexNode, indexVvar] = round(np.nanmean(encode_values), 3)
            X[indexNode, indexVvar+1] = round(np.nanmin(encode_values), 3)
            X[indexNode, indexVvar+2] = round(np.nanmax(encode_values), 3)
            X[indexNode, indexVvar+3] = round(np.nanstd(encode_values), 3)
        else:
            X[indexNode, indexVvar] = np.nanmean(encode_values)

    X = np.full((subNode.shape[0], newShape), np.nan, dtype=float)
    X[:,:subNode.shape[1]] = subNode

    dir_encoder = path / 'Encoder'

    if 'landcover' in features:
        encoder_landcover = read_object('encoder_landcover.pkl', dir_encoder)

    if 'foret' in features:
        encoder_foret = read_object('encoder_foret.pkl', dir_encoder)

    if 'Calendar' in features:
        size_calendar = len(calendar_variables)
        encoder_calendar = read_object('encoder_calendar.pkl', dir_encoder)

    if 'Geo' in features:
        encoder_geo = read_object('encoder_geo.pkl', dir_encoder)

    print('Calendar')
    if 'Calendar' in features:
        for node in subNode:
            index = np.argwhere(subNode[:,4] == node[4])
            X[index, pos_feature['Calendar']] = geo[geo['date'] == node[4]]['month'].values[0] # month
            X[index, pos_feature['Calendar'] + 1] = geo[geo['date'] == node[4]]['dayofweek'].values[0] # dayofweek
            X[index, pos_feature['Calendar'] + 2] = geo[geo['date'] == node[4]]['dayofyear'].values[0] # dayofyear
            X[index, pos_feature['Calendar'] + 3] = geo[geo['date'] == node[4]]['dayofweek'].values[0]  >= 5 # isweekend
            X[index, pos_feature['Calendar'] + 4] = geo[geo['date'] == node[4]]['couvrefeux'].values[0]  # couvrefeux
            X[index, pos_feature['Calendar'] + 5] = geo[geo['date'] == node[4]]['confinement'].values[0]  # confinement
            X[index, pos_feature['Calendar'] + 6] = geo[geo['date'] == node[4]]['ramadan'].values[0]  # ramadan
            X[index, pos_feature['Calendar'] + 7] = geo[geo['date'] == node[4]]['bankHolidays'].values[0]  # bankHolidays
            X[index, pos_feature['Calendar'] + 8] = geo[geo['date'] == node[4]]['bankHolidaysEve'].values[0]  # bankHolidaysEve
            X[index, pos_feature['Calendar'] + 9] = geo[geo['date'] == node[4]]['holidays'].values[0]  # holidays
            X[index, pos_feature['Calendar'] + 10] = geo[geo['date'] == node[4]]['holidaysBorder'].values[0]  # holidaysBorder

        X[:, pos_feature['Calendar'] : pos_feature['Calendar'] + size_calendar] = \
                encoder_calendar.transform((X[:, pos_feature['Calendar'] : \
                        pos_feature['Calendar'] + size_calendar]).reshape(-1, size_calendar)).values.reshape(-1, size_calendar)

    ### Geo spatial
    print('Geo')
    if 'Geo' in features:
        X[:, pos_feature['Geo']] = encoder_geo.transform(geo['departement'].values).values[0] # departement
  
    print('Meteorological')
    ### Meteo
    for i, var in enumerate(cems_variables):
        for node in subNode:
            maskNode = geo[geo['id'] == node[0]].index
            if maskNode.shape[0] == 0:
                continue
            index = np.argwhere((subNode[:,0] == node[0])
                                & (subNode[:,4] == node[4])
                                )

            save_values(geo[var].values, pos_feature[var], index, maskNode)

    print('Air Quality')
    for i, var in enumerate(air_variables):
        for node in subNode:
            maskNode = geo[geo['id'] == node[0]].index
            if maskNode.shape[0] == 0:
                continue
            index = np.argwhere((subNode[:,0] == node[0])
                                & (subNode[:,4] == node[4])
                                )
            save_value(geo[var].values, pos_feature['air'] + i, index, maskNode)

    print('Population elevation Highway Sentinel')
    for node in subNode:
        maskNode = geo[geo['id'] == node[0]].index
        index = np.argwhere(subNode[:,0] == node[0])
        if 'population' in features:
            save_values(geo['population'].values, pos_feature['population'], index, maskNode)
        if 'elevation' in features:
            save_values(geo['elevation'].values, pos_feature['elevation'], index, maskNode)
        if 'highway' in features:
            save_values(geo['highway'].values, pos_feature['highway'], index, maskNode)
        if 'foret' in features:
            save_value_with_encoding(geo['foret'].values, pos_feature['foret'], index, maskNode, encoder_foret)

    coef = 4 if graph.scale > -1 else 1
    for node in subNode:
        maskNode = geo[geo['id'] == node[0]].index
        index = np.argwhere((subNode[:,0] == node[0])
                            & (subNode[:,4] == node[4])
                            )

        if 'sentinel' in features:
            for band, var in enumerate(sentinel_variables):
                save_values(geo[var].values, pos_feature['sentinel'] + (sentinel_variables.index(var) * coef) , index, maskNode)

        if 'landcover' in features: 
            save_value_with_encoding(geo['landcover'].values, pos_feature['landcover'], index, maskNode, encoder_landcover)

    print('Historical')
    if 'Historical' in features:
        name = dept+'pastInfluence.pkl'
        arrayInfluence = pickle.load(open(dir_data / 'log' / name, 'rb'))
        for node in subNode:
            index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
            maskNode = mask == node[0]
            #if node[4] - 1 < 0:
            #    continue

            save_values(arrayInfluence[:,:, int(node[4] - 1)], pos_feature['Historical'], index, maskNode)

    return X

def create_ps(geo):
    X_kmeans = list(zip(geo.longitude, geo.latitude))
    geo['id'] = graph._predict_node(X_kmeans)

    print(f'Unique id : {np.unique(geo["id"].values)}')
    print(f'{len(geo)} point in the dataset. Constructing database')

    return geo

def fire_prediction(spatialGeo, interface, departement, features, features_selected, scaling, dir_data):

    geoDT = create_ps(interface)

    # Based tree models
    traditionnal_models = ['xgboost', 'lightgbm', 'ngboost']

    spatialGeo = spatialGeo.rename({'osmnx' : 'highway'}, axis=1)
    
    spatial = ['NDVI', 'NDMI', 'NDBI', 'NDSI', 'NDWI',
               'landcover', 'highway', 'elevation', 'population']
    
    temporal = ['temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
                'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                'isi', 'angstroem', 'bui', 'fwi', 'daily_severity_rating',
                'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16'
                ]
    
    for s in spatial:
        geoDT[s] = spatialGeo[s]

    geoDT[geo_variables] = int(departement.split('-')[1])

    geoDT[landcover_variables] = 0
    for cv in cems_variables:
        if cv not in geoDT.columns:
            geoDT[cv] = 0
            
    geoDT[foret_variables] = 0
    geoDT[calendar_variables] = 0
    geoDT[historical_variables] = 0
    geoDT[air_variables] = 0

    geoDT['weights'] = 0
    geoDT['departement'] = int(departement.split('-')[1])
    columns = ['id', 'longitude', 'latitude', 'departement', 'date', 'weights']
    orinode = geoDT.drop_duplicates(['id', 'date'])[columns].values
    orinode = graph._assign_latitude_longitude(orinode)
    orinode = graph._assign_department(orinode)
    orinode[np.isin(orinode[:, 4], [ks, ks+1]), 5] = 1
    
    prefix = str(minPoint)+'_'+str(k_days)+'_'+str(scale)
    if model not in traditionnal_models:
        name_model =  model + '/' + 'best.pt'
        dir = 'check_'+scaling + '/' + prefix + '/'
        graph._load_model_from_path(dir_data / dir / name_model, dico_model[model], device)
    else:
        dir = 'check_'+scaling + '/' + prefix + '/' + '/baseline'
        model_ = read_object(model+'.pkl', dir_data / dir / model)
        graph._set_model(model_)

    X = get_sub_nodes_feature(graph=graph,
                              subNode=orinode,
                              features=features,
                              geo=geoDT,
                              path=dir_data)
    
    print(np.unique(np.argwhere(np.isnan(X))[:,1]))

    # Today
    today_X, E = create_inference(graph,
                     X,
                     Xtrain,
                     device,
                     False,
                     k_days,
                     k_days,
                     scaling)
    
    if model in traditionnal_models:
        today_X = today_X.detach().cpu().numpy()
        today_X_fet = today_X[:, features_selected, :]
        today_X_fet = today_X_fet.reshape(today_X_fet.shape[0], -1)
        today_Y = graph.predict_model_api_sklearn(today_X_fet, False)
    else:
        today_Y = graph._predict_tensor(today_X_fet, E)

    # Tomorrow
    tomorrow_X, E = create_inference(graph,
                     X,
                     Xtrain,
                     device,
                     False,
                     k_days + 1,
                     k_days,
                     scaling)
    
    if model in traditionnal_models:
        tomorrow_X = tomorrow_X.detach().cpu().numpy()
        tomorrow_X_fet = tomorrow_X[:, features_selected, :]
        tomorrow_X_fet = tomorrow_X_fet.reshape(tomorrow_X_fet.shape[0], -1)
        tomorrow_Y = graph.predict_model_api_sklearn(tomorrow_X_fet, False)
    else:
        tomorrow_Y = graph._predict_tensor(tomorrow_X_fet, E)

    unodes = np.unique(orinode[:,0])

    for node in unodes:

        interface.loc[interface[(interface['date'] == k_days) &
                                (interface['id'] == node)].index, 'fire_prediction'] = today_Y[np.argwhere(today_X[:, 0, 0] == node)[:,0]][0]
        
        interface.loc[interface[(interface['date'] == k_days + 1) &
                                (interface['id'] == node)].index, 'fire_prediction'] = tomorrow_Y[np.argwhere(tomorrow_X[:, 0, 0] == node)[:,0]][0]
        
    return interface

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
    args = parser.parse_args()

    # Input config
    k_days = int(args.days)
    scale = args.scale
    dept = args.departement
    sinister = args.sinister
    model = args.model
    minPoint = args.nbpoint
    scaling='z-score' # Scale to used
    dir_data = Path('../GNN/final/' + sinister + '/train')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Features 
    features = [
            'temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
            'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
            'isi', 'angstroem', 'bui', 'fwi', 'daily_severity_rating',
            'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
            'elevation', 'highway', 'population',
            'sentinel',
            'landcover',
            'foret',
            'Calendar',
            'Historical',
            'Geo',
            'air',
            ]

    # Load train
    graph = read_object('graph_'+scale+'.pkl', dir_data)
    Xtrain = read_object('X_'+str(minPoint)+'_'+str(k_days)+'_'+str(graph.scale)+'.pkl', dir_data)
    Ytrain = read_object('Y_'+str(minPoint)+'_'+str(k_days)+'_'+str(graph.scale)+'.pkl', dir_data)
    features_importance = read_object('features_importance.pkl', dir_data)
    features_selected = np.unique(features_importance[:,0]).astype(int)

    # Arborescence
    dir_incendie = Path('/home/caron/Bureau/csv/'+dept+'/data/')
    dir_interface = Path('interface')
    dir_output = dir_interface / 'output'
    check_and_create_path(dir_output)

    # Load save
    spatialGeo = gpd.read_file(dir_incendie / 'spatial' / 'hexagones.geojson')

    ######################### Interface ##################################

    interface = []

    originalCols = None

    for ks in range(k_days + 1):

        todays = gpd.read_file(dir_interface / 'hexagones_today.geojson')

        if originalCols is None:
            originalCols = list(todays.columns)

        todays['longitude'] = todays['geometry'].apply(lambda x : float(x.centroid.x))
        todays['latitude'] = todays['geometry'].apply(lambda x : float(x.centroid.y))
        todays['date'] = ks
        interface.append(todays)

    tomorrow = gpd.read_file(dir_interface / 'hexagones_tomorrow.geojson')
    tomorrow['longitude'] = todays['longitude']
    tomorrow['latitude'] = todays['latitude']
    spatialGeo['longitude'] = todays['longitude']
    spatialGeo['latitude'] = todays['latitude']
    tomorrow['date'] = k_days + 1

    interface.append(tomorrow)
    interface = pd.concat(interface).reset_index(drop=True)

    # Patch
    interface['nb'+sinister] = 1

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
    
    spa = 3
    if sinister == "firepoint":
        dims = [(spa,spa,11),
                (spa,spa,7),
                (spa,spa,5),
                (spa,spa,9)]
    elif sinister == "inondation":
        dims = [(spa,spa,5)]

    Probabilistic(dims, n_pixel_x, n_pixel_y, 1, None, dir_data)._process_input_raster([inputDep], 1, True, [dept], True)

    ######################### Construct X ################################

    interface = fire_prediction(spatialGeo, interface, dept, features, features_selected, scaling, dir_data)

    if 'fire_prediction' not in originalCols:
        originalCols.append('fire_prediction')

    today = interface[interface['date'] == k_days][originalCols]
    tomorrow = interface[interface['date'] == k_days + 1][originalCols]

    #today.to_file(dir_interface / 'hexagones_today.geojson', driver='GeoJSON')
    #tomorrow.to_file(dir_interface / 'hexagones_tomorrow.geojson',  driver='GeoJSON')