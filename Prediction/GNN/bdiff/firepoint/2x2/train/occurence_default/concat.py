from math import pi
import pickle
import pandas as pd


file1 = pickle.load(open('df_train_full_departement_0_None_node.pkl', 'rb'))
file3 = pickle.load(open('df_val_full_departement_0_None_node.pkl', 'rb'))
file2 = pickle.load(open('df_test_full_departement_0_None_node.pkl', 'rb'))

files = pd.concat((file1, file2, file3)).reset_index()

cems_variables = ['temp',
                  'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
                'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                'isi', 'angstroem', 'bui', 'fwi', 'dailySeverityRating',
                'temp16',
                'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
                'days_since_rain', 'sum_consecutive_rainfall', 'sum_rain_last_7_days',
                'sum_snow_last_7_days', 'snow24h', 'snow24h16',
                'precipitationIndexN3', 'precipitationIndexN5', 'precipitationIndexN7'
                ]

air_variables = ['O3', 'NO2', 'PM10', 'PM25']

# Encoder
sentinel_variables = ['NDVI', 'NDMI', 'NDBI', 'NDSI', 'NDWI']
landcover_variables = [
                      'foret_encoder',
                      'argile_encoder',
                      'cosia_encoder',
                      #'corine_encoder',
                      'bdroute_encoder'
                        ]

cluster_encoder = ['cluster_encoder', 'id_encoder']

calendar_variables = ['month', 'dayofyear', 'dayofweek', 'isweekend', 'couvrefeux', 'confinement',
                    'ramadan', 'bankHolidays', 'bankHolidaysEve', 'holidays', 'holidaysBorder',
                    'calendar_mean', 'calendar_min', 'calendar_max', 'calendar_sum']

geo_variables = ['departement_encoder']
region_variables = ['region_class']

# Percentage
foret_variables = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
cosia_variables = [
    'Other',
    'Building',
    'Bare soil',
    'Water surface',
    'Conifer',
    'Deciduous',
    'Shrubland',
    'Lawn',
    'Crop'
]

corine_variable = [
    'Corine_Other',
    'Corine_urban',
    'Corine_transport',
    'Corine_agricultural',
    'Corine_grass',
    'Corine_forest',
    'Corine_vegetation',
    'Corine_moisture',
    'Corine_water',
    'Corine_littoral',
    'Corine_rock'
]

bdroute_variables = ['NoRoad', 'Road']

osmnx_variables = ['0', '1', '2', '3', '4', '5']
dynamic_world_variables = ['water', 'tree', 'grass', 'crops', 'shrub', 'flooded', 'built', 'bare', 'snow']

# Influence 
historical_variables = ['pastinfluence']
auto_regression_variable_reg = ['J-1',
                                 #'J-2', 'J-3', 'J-4', 'J-5', 'J-6', 'J-7'
                                ]
auto_regression_variable_bin = ['B-1',
                                #'B-2', 'B-3', 'B-4', 'B-5', 'B-6', 'B-7'
                                ]

# Other
elevation_variables = ['elevation']
population_variabes = ['population']
vigicrues_variables = ['12',
                       #'16'
                      ]

nappes_variables = ['niveau_nappe_eau', 'profondeur_nappe']

# Time varying
varying_time_variables = ['temp_mean', 'dwpt_mean', 'rhum_mean', 'wdir_mean', 'wspd_mean',
                            'dc_mean', 'ffmc_mean', 'dmc_mean', 'nesterov_mean', 'munger_mean', 'kbdi_mean',
                            'isi_mean', 'angstroem_mean', 'bui_mean', 'fwi_mean', 'dailySeverityRating_mean',
                            'temp16_mean', 'dwpt16_mean', 'rhum16_mean', 'wdir16_mean', 'wspd16_mean',
                            #'air_mean',
                            #'Calendar_mean',

                            'temp_min', 'dwpt_min', 'rhum_min', 'wdir_min', 'wspd_min',
                            'dc_min', 'ffmc_min', 'dmc_min', 'nesterov_min', 'munger_min', 'kbdi_min',
                            'isi_min', 'angstroem_min', 'bui_min', 'fwi_min', 'dailySeverityRating_min',
                            'temp16_min', 'dwpt16_min', 'rhum16_min', 'wdir16_min', 'wspd16_min',
                            #'air_min',
                            #'Calendar_min',

                            'temp_max', 'dwpt_max', 'rhum_max', 'wdir_max', 'wspd_max',
                            'dc_max', 'ffmc_max', 'dmc_max', 'nesterov_max', 'munger_max', 'kbdi_max',
                            'isi_max', 'angstroem_max', 'bui_max', 'fwi_max', 'dailySeverityRating_max',
                            'temp16_max', 'dwpt16_max', 'rhum16_max', 'wdir16_max', 'wspd16_max',
                            #'air_max',
                            #'Calendar_max',

                            #'Historical_sum',
                            #'Historical_grad',
                            #'AutoRegressionBin_sum',
                            #'AutoRegressionReg_sum',
                            #'AutoRegressionReg_grad'
                            ]

varying_time_variables_name = []

foretint2str = {
    '0': 'NoForest',
    '1': 'Châtaignier',
    '2': 'Chênes décidus',
    '3': 'Chênes sempervirents',
    '4': 'Conifères',
    '5': 'Douglas',
    '6': 'Feuillus',
    '7': 'Hêtre',
    '8': 'Mélèze',
    '9': 'Mixte',
    '10': 'NC',
    '11': 'NR',
    '12': 'Pin à crochets, pin cembro',
    '13': 'Pin autre',
    '14': 'Pin d\'Alep',
    '15': 'Pin laricio, pin noir',
    '16': 'Pin maritime',
    '17': 'Pin sylvestre',
    '18': 'Pins mélangés',
    '19': 'Peuplier',
    '20': 'Robinier',
    '21': 'Sapin, épicéa'
}

osmnxint2str = {
'0' : 'PasDeRoute',
'1':'motorway',
 '2': 'primary',
 '3': 'secondary',
 '4': 'tertiary', 
 '5': 'path'}

def get_features_name_list(scale, features, methods):
    features_name = []
    if scale == 0:
        methods = ['mean']
    for var in features:
        if var == 'Calendar':
            features_name += calendar_variables
        elif var == 'air':
            features_name += air_variables
        elif var in landcover_variables:
            features_name += [f'{var}_{met}' for met in methods]
        elif var == 'sentinel':
            features_name += [f'{v}_{met}' for v in sentinel_variables for met in methods]
        elif var == "foret":
            features_name += [f'{foretint2str[v]}_{met}' for v in foret_variables for met in methods]
        elif var == 'dynamicWorld':
            features_name += [f'{v}_{met}' for v in dynamic_world_variables for met in methods]
        elif var == 'cosia':
            features_name += [f'{v}_{met}' for v in cosia_variables for met in methods]
        elif var == 'corine':
            features_name += [f'{v}_{met}' for v in corine_variable for met in methods]
        elif var == 'bdroute':
            features_name += [f'{v}_{met}' for v in bdroute_variables for met in methods]
        elif var == 'highway':
            features_name += [f'{osmnxint2str[v]}_{met}' for v in osmnx_variables for met in methods]
        elif var == 'Geo':
            features_name += geo_variables
        elif var == 'vigicrues':
            features_name += [f'{v}_{met}' for v in vigicrues_variables for met in methods]
        elif var == 'nappes':
            features_name += [f'{v}_{met}' for v in nappes_variables for met in methods]
        elif var == 'Historical':
            features_name += [f'{v}' for v in historical_variables]
        elif var == 'AutoRegressionReg':
            features_name += [f'AutoRegressionReg-{v}' for v in auto_regression_variable_reg]
        elif var == 'AutoRegressionBin':
            features_name +=  [f'AutoRegressionBin-{v}' for v in auto_regression_variable_bin]
        elif var == 'elevation':
            features_name += [f'{v}_{met}' for v in elevation_variables for met in methods]
        elif var == 'population':
            features_name += [f'{v}_{met}' for v in population_variabes for met in methods]
        elif var == 'region_class':
            features_name += [var]
        elif var == 'Past_risk' or var == 'Past_bunredarea':
            features_name += [var]
        elif var in varying_time_variables_name:
            features_name += [var]
        elif var == 'temporal_prediction' or var == 'spatial_prediction':
            features_name += [var]
        elif var.find('frequencyratio') != -1:
            features_name += [var]
        elif var in cluster_encoder:
            features_name += [var]
        elif var == 'Past_risk' or var == 'Past_burnedarea':
            features_name += [var]
        else:
            features_name += [f'{var}_{met}' for met in methods]
            
    return features_name, len(features_name)

features_name = ["temp", "dwpt", "rhum", "prcp", "wdir", "wspd", "prec24h",
        "dc", "ffmc", "dmc", "nesterov", "munger", "kbdi",
        "isi", "angstroem", "bui", "fwi", "dailySeverityRating",
        "temp16", "dwpt16", "rhum16", "prcp16", "wdir16", "wspd16", "prec24h16",
        "days_since_rain", "sum_consecutive_rainfall",
        "sum_rain_last_7_days",
        "sum_snow_last_7_days", "snow24h", "snow24h16",
        "precipitationIndexN3", "precipitationIndexN5", "precipitationIndexN7",
        "elevation",
        "population",
        "sentinel",
        "foret_encoder",
        "corine_encoder",
        "cluster_encoder",
        "bdroute_encoder",
        "id_encoder",
        "Geo",
        "foret",
        "bdroute",
        "corine",
        "Calendar",
        "Past_risk",
        "Past_burnedarea"]

METHODS = ['mean', 'min', 'max',
            #'std',
           #'sum',
           #'grad'
           ]

#features, _ = get_features_name_list('departement', features_name, METHODS)

features = pickle.load(open('features_name.pkl', 'rb'))
#features = ['temp_max', 'NDVI_max']
target = 'nbsinister-kmeans-5-Class-Dept'

def find_dates_between(start, end):
    import datetime as dt
    start_date = dt.datetime.strptime(start, '%Y-%m-%d').date()
    end_date = dt.datetime.strptime(end, '%Y-%m-%d').date()

    delta = dt.timedelta(days=1)
    date = start_date
    res = []
    while date < end_date:
            res.append(date.strftime("%Y-%m-%d"))
            date += delta
    return res

allDates = find_dates_between('2017-06-12', '2024-12-31')
files['year'] = files['date'].apply(lambda x : allDates[x].split('-')[0])

features.append(target)
files = files[list(features) + ['graph_id', 'id', 'longitude', 'latitude', 'departement', 'date', 'weight', 'scale', 'days_until_next_event', 'year']]

def get_saison(x):
    date = allDates[(int(x))]
    month = int(date.split('-')[1])
    group_month = [
                [2, 3, 4, 5],    # Medium season
                [6, 7, 8, 9],    # High season
                [10, 11, 12, 1]  # Low season
            ]
    
    if month in [2, 3, 4, 5]:
        return 'medium'
    if month in [6, 7, 8, 9]:
        return 'high'
    return 'low'
    
files['saison'] = files['date'].apply(lambda x : get_saison(x)) 

files['target'] = files[target]
files.drop(target, inplace=True, axis=1)

files.to_csv('full_dataframe.csv', index=False)
