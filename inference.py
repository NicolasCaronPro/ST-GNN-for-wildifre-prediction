from tools_inference import *
import datetime as dt
import json
####################### Function #################################

def get_sub_nodes_feature(graph, subNode, departements, features, geo):
    assert graph.nodes is not None

    print('Load nodes features')

    cems_variables = ['temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
                    'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                    'isi', 'angstroem', 'bui', 'fwi', 'daily_severity_rating',
                    'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16']
    
    sentinel_variables = ['NDVI', 'NDMI', 'NDBI', 'NDSI', 'NDWI']
    landcover_variables = ['0.0','1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0',
                           '8.0'
                           ]
    osmnx_variables = ['highway']
    elevation_variables = ['elevation']
    population_variabes = ['population']
    calendar_variables = ['month', 'dayofweek', 'dayofyear', 'isweekend', 'couvrefeux', 'confinemenent',
                        'ramadan', 'bankHolidays', 'bankHolidaysEve', 'holidays', 'holidaysBorder']
    
    pos_feature = {}

    newShape = subNode.shape[1]
    if graph.scale > 0:
        for var in features:
            print(newShape, var)
            pos_feature[var] = newShape
            if var == 'Calendar':
                newShape += len(calendar_variables)
            elif var == 'landcover':
                newShape += len(landcover_variables)
            elif var == 'sentinel':
                newShape += 4 * len(sentinel_variables)
            else:
                newShape += 4
    else:
        for var in features:
            print(newShape, var)
            pos_feature[var] = newShape
            if var == 'Calendar':
                newShape += len(calendar_variables)
            elif var == 'landcover':
                newShape += len(landcover_variables)
            elif var == 'sentinel':
                newShape += len(sentinel_variables)
            else:
                newShape += 1

    def save_values(array, indexVvar, indexNode, mask):
        if False not in np.unique(np.isnan(array[mask])):
            return
        if graph.scale > 0:
            X[indexNode, indexVvar] = round(np.nanmean(array[mask]), 3)
            X[indexNode, indexVvar+1] = round(np.nanmin(array[mask]), 3)
            X[indexNode, indexVvar+2] = round(np.nanmax(array[mask]), 3)
            X[indexNode, indexVvar+3] = round(np.nanstd(array[mask]), 3)
        else:
            X[indexNode, indexVvar] = np.nanmean(array[mask])

    def save_values_count(array, lc, indexVvar, indexNode, mask):
        X[indexNode, indexVvar] = np.argwhere(array[mask] == float(lc)).shape[0]

    X = np.full((subNode.shape[0], newShape), np.nan, dtype=float)

    X[:,:subNode.shape[1]] = subNode

    for departement in departements:
        print(departement)
        dir_data = root / departement / 'raster'

        nodeDepartementMask = np.argwhere(subNode[:,3] == str2int[departement.split('-')[-1]])
        nodeDepartement = subNode[nodeDepartementMask].reshape(-1, subNode.shape[1])

        print('Meteorological')
        ### Meteo
        for i, var in enumerate(cems_variables):
            if var not in features:
                continue
            print(var)
            for  node in nodeDepartement:
                maskNode = geo['scale'] == node[0]
                if True not in maskNode:
                    continue
                index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                save_values(geo[geo['date'] == node[4]][var], pos_feature[var], index, maskNode)

        print('Population elevation Highway Sentinel')

        coef = 4 if graph.scale > 0 else 1
        for node in nodeDepartement:
            maskNode = geo['scale'] == node[0]
            index = np.argwhere(subNode[:,0] == node[0])
            if 'population' in features:
                save_values(geo[geo['date'] == node[4]]['population'], pos_feature['population'], index, maskNode)
            if 'elevation' in features:
                save_values(geo[geo['date'] == node[4]]['elevation'], pos_feature['elevation'], index, maskNode)
            if 'highway' in features:
                save_values(geo[geo['date'] == node[4]]['highway'], pos_feature['highway'], index, maskNode)

            index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
            if 'sentinel' in features:
                for band, var in enumerate(sentinel_variables):
                    save_values(geo[geo['date'] == node[4]][var], pos_feature['sentinel'] + (sentinel_variables.index(var) * coef) , index, maskNode)

            if 'landcover' in features: 
                for lc, var in enumerate(landcover_variables):
                    save_values_count(geo[geo['date'] == node[4]][var], lc, pos_feature['landcover'] + landcover_variables.index(var), index, maskNode)

        print('Calendar')

        if 'Calendar' not in features:
            return X
        for node in nodeDepartement:
            date = allDates[int(node[4])]
            ddate = dt.datetime.strptime(date, '%Y-%m-%d')
            index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
            X[index, pos_feature['Calendar']] = int(date.split('-')[1]) # month
            X[index, pos_feature['Calendar'] + 1] = ddate.timetuple().tm_yday # dayofweek
            X[index, pos_feature['Calendar'] + 2] = ddate.weekday() # dayofyear
            X[index, pos_feature['Calendar'] + 3] = ddate.weekday() >= 5 # isweekend
            X[index, pos_feature['Calendar'] + 4] = pendant_couvrefeux(ddate) # couvrefeux
            X[index, pos_feature['Calendar'] + 5] = (1 if dt.datetime(2020, 3, 17, 12) <= ddate <= dt.datetime(2020, 5, 11) else 0) or 1 if dt.datetime(2020, 10, 30) <= ddate <= dt.datetime(2020, 12, 15) else 0# confinement
            X[index, pos_feature['Calendar'] + 6] = 1 if convertdate.islamic.from_gregorian(ddate.year, ddate.month, ddate.day)[1] == 9 else 0 # ramadan
            X[index, pos_feature['Calendar'] + 7] = 1 if ddate in jours_feries else 0 # bankHolidays
            X[index, pos_feature['Calendar'] + 8] = 1 if ddate in veille_jours_feries else 0 # bankHolidaysEve
            X[index, pos_feature['Calendar'] + 9] = 1 if vacances_scolaire.is_holiday_for_zone(ddate.date(), get_academic_zone(ACADEMIES[str(name2int[departement])], ddate)) else 0 # holidays
            X[index, pos_feature['Calendar'] + 10] = (1 if vacances_scolaire.is_holiday_for_zone(ddate.date() + dt.timedelta(days=1), get_academic_zone(ACADEMIES[str(name2int[departement])], ddate)) else 0 ) \
                or (1 if vacances_scolaire.is_holiday_for_zone(ddate.date() - dt.timedelta(days=1), get_academic_zone(ACADEMIES[str(name2int[departement])], ddate)) else 0) # holidaysBorder
        
        print('Historical')
        if 'Historical' in features:
            dir_proba = Path('/home/caron/Bureau/Model/HexagonalScale/log')
            name = departement+'pastInfluence.pkl'
            arrayInfluence = pickle.load(open(dir_proba / name, 'rb'))
            for node in nodeDepartement:
                index = np.argwhere((subNode[:,0] == node[0]) & (subNode[:,4] == node[4]))
                maskNode = geo['scale'] == node[0]
                if node[4] - 1 < 0:
                    continue
                
                save_values(geo[geo['date'] == node[4]]['Historical'], pos_feature['Historical'], index, maskNode)
                
            del arrayInfluence
    return X

######################## Read geo dataframe ######################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEPARTEMENT = 'departement-01-ain'
graph = read_object('graph_5.pkl', Path('train'))
use_temporal_as_edges = False
ks = 7
Xtrain = read_object('X_'+str(minPoint)+'_'+str(k_days)+'_'+str(graph.scale)+'.pkl', Path('train')) # Scaling isue
Ytrain = read_object('Y_'+str(minPoint)+'_'+str(k_days)+'_'+str(graph.scale)+'.pkl', Path('train')) # Scaling isue
in_dim = Xtrain[:,6:].shape[1]

dico_graph_model = {#'GAT': GAT([in_dim, 64, 64], [4, 4, 2], 0.06, False, device),
                    #'GUNET' : GRAPH_UNET(in_dim, in_dim * 2, in_dim // 2, 0.0, [0.9,0.8,0.7], 'gelu').to(device),
                    'ST-GATTCN': STGATCN(n_sequences=k_days, num_of_layers=3, in_channels=in_dim,
                                         end_channels=32, skip_channels=64, residual_channels=64,
                                         dilation_channels=64, dropout=0.03, heads=6, act_func='gelu', device=device),
                    #'ST-GATCONV': STGATCONV(k_days, 3,
                    #                        in_dim,
                    #                        64,
                    #                        32,
                    #                        0.03, 6, 'gelu', device=device),
                    #'TGAT' : TGAT(n_sequences=k_days,
                    #              in_channels_list=[in_dim],
                    #              heads=[4], bias = True, dropout=0.0, device=device)
                    }

gnn = 'ST-GATTCN'

graph._load_model_from_path(Path('check/' + str(minPoint)+'_'+str(k_days)+'_'+str(graph.scale) + '/' + gnn +  '/' + 'best.pt'),
                                         dico_graph_model[gnn], device)

dir_incendie = Path('/home/caron/Bureau/csv/'+DEPARTEMENT+'/data/')
dir_meteostat = Path('test_inference')
dir_interface = Path('interface')
dir_output = dir_interface / 'output'

check_and_create_path(dir_meteostat)
check_and_create_path(dir_output)

geoSpatial = gpd.read_file(dir_incendie / 'spatial' / 'hexagones.geojson')
pos_feature = create_pos_feature(graph, 5, features)

######################### Construct X ################################

def create_ps(geo, isToday):
    
    X_kmeans = list(zip(geo.longitude, geo.latitude))
    geo['scale'] = graph._predict_node(X_kmeans)
    print(np.unique(geo['scale'].values))

    print(f'{len(geo)} point in the dataset. Constructing database')

    orinode = np.full((len(geo), 6), -1.0, dtype=float)
    orinode[:,0] = geo['scale'].values
    orinode[:,4] = k_days if isToday else k_days + 1

    orinode = generate_subgraph(graph, 0, 0, orinode)
    return orinode, geo

def fire_prediction(today, tomorrow, spatialGeo):
    originalCols = today.columns
    today['longitude'] = today['geometry'].apply(lambda x : float(x.centroid.x))
    today['latitude'] = today['geometry'].apply(lambda x : float(x.centroid.y))
    tomorrow['longitude'] = today['longitude']
    tomorrow['latitude'] = today['latitude']
    spatialGeo['longitude'] = today['longitude']
    spatialGeo['latitude'] = today['latitude']

    todayPS, today = create_ps(today, True)
    tomorrowPS, tomorrow = create_ps(tomorrow, True)

    orinode = np.concatenate((todayPS, tomorrowPS))
    orinode[:,5] = 1
    
    today['date'] = k_days
    tomorrow['date'] = k_days + 1
    geoDT = pd.concat((today, tomorrow))
    spatialGeo = spatialGeo.rename(columns={'osmnx' : 'highway'})
    spatial = ['NDVI', 'NDMI', 'NDBI', 'NDSI', 'NDWI',
               '0.0','1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0',
               #'8.0',
               'highway', 'elevation', 'population']
    
    temporal = ['temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
                'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                'isi', 'angstroem', 'bui', 'fwi', 'daily_severity_rating',
                'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16']
    
    for s in spatial:
        today[s] = spatialGeo[s]
        tomorrow[s] = spatialGeo[s]

    today['8.0'] = 0                #
    today['Historical'] = 0         #  <------------------ TO DO
    tomorrow['Historical'] = 0      #

    ########### Load historical ################
    print(f'Check if historical exist (normally no it doesn t just for the first iteration)')
    try:
        subNode = read_object('X.pkl', dir_output)
        subNode[:,5] = 0
        CONSTRUCT_HISTORICAL_METEO = False
    except Exception as e:
        print(f'{e}')
        CONSTRUCT_HISTORICAL_METEO = True
        region_path = dir_incendie / 'geo/geo.geojson'
        region = json.load(open(region_path))
        polys = [region["geometry"]]
        geom = [shape(i) for i in polys]
        region = gpd.GeoDataFrame({'geometry':geom})
        cems = construct_historical_meteo(dt.datetime.now().strftime('%Y-%m-%d'), region, dir_meteostat)
        print(cems)
        orinode = add_k_temporal_node(k_days=k_days, nodes=orinode)
        orinode[:,5] = 1
        geoDT = [today, tomorrow]
        for i in range(k_days):
            geoDT2 = today.copy(deep=True)
            #geoDT2['date'] = (dt.datetime.now() - dt.timedelta(days=i)).strftime('%Y-%m-%d')
            geoDT2['date'] = i
            for t in temporal:
                cems_grid = create_grid_cems(cems, (dt.datetime.now() - dt.timedelta(days=i)).strftime('%Y-%m-%d'), 0, t)

                geoDT2[t] = interpolate_gridd(t, cems_grid, today.longitude.values, today.latitude.values)

            geoDT.append(geoDT2)

        geoDT = pd.concat(geoDT).reset_index(drop=True)
    print(today.columns)

    orinode[(orinode[:, 4] != k_days) & (orinode[:, 4] != k_days + 1)][:,5] = 0
    orinode = graph._assign_latitude_longitude(orinode)
    orinode = graph._assign_department(orinode)

    graph._info_on_graph(orinode, Path('log'))

    X = get_sub_nodes_feature(graph, orinode, departements, features, geoDT)

    if not CONSTRUCT_HISTORICAL_METEO:
        X = np.concatenate((X, subNode))

    X, E = create_inference(graph, X, Xtrain, Ytrain, device, use_temporal_as_edges, ks, pos_feature)
    Y = graph._predict_tensor(X, E)
    Y = torch.masked_select(Y, X[:,5].gt(0))

    today['fire_prediction'] = Y[X[:,4] == k_days]
    tomorrow['fire_prediction'] = Y[X[:,4] == k_days + 1]
    X[:,4] = X[:,4] - 1
    X = X[(X[:,4] > 0) & (X[:,4] < k_days)]
    save_object(X, 'X.pkl', dir_output)

    return today[originalCols], tomorrow[originalCols]

tomorrow = gpd.read_file(dir_interface / 'hexagones_today.geojson')
today = gpd.read_file(dir_interface / 'hexagones_tomorrow.geojson')

today, tomorrow = fire_prediction(today, tomorrow, geoSpatial)

today.to_file(dir_interface / 'hexagones_tomorrow.geojson')
tomorrow.to_file(dir_interface / 'hexagones_today.geojson')