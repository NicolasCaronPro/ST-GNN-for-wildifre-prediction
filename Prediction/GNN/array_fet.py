cems_variables = ['temp',
                  'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
                'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                'isi', 'angstroem', 'bui', 'fwi', 'dailySeverityRating',
                'temp16',
                'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
                'days_since_rain', 'sum_consecutive_rainfall', 'sum_rain_last_7_days',
                'sum_snow_last_7_days', 'snow24h', 'snow24h16'
                ]

air_variables = ['O3', 'NO2', 'PM10', 'PM25']

# Encoder
sentinel_variables = ['NDVI', 'NDMI', 'NDBI', 'NDSI', 'NDWI']
landcover_variables = [
                      'foret_encoder',
                      'argile_encoder',
                      'id_encoder',
                      'cosia_encoder'
                        ]

cluster_encoder = ['cluster_encoder']

calendar_variables = ['month', 'dayofyear', 'dayofweek', 'isweekend', 'couvrefeux', 'confinemenent',
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
                            'air_mean',
                            'Calendar_mean',

                            'temp_min', 'dwpt_min', 'rhum_min', 'wdir_min', 'wspd_min',
                            'dc_min', 'ffmc_min', 'dmc_min', 'nesterov_min', 'munger_min', 'kbdi_min',
                            'isi_min', 'angstroem_min', 'bui_min', 'fwi_min', 'dailySeverityRating_min',
                            'temp16_min', 'dwpt16_min', 'rhum16_min', 'wdir16_min', 'wspd16_min',
                            'air_min',
                            'Calendar_min',

                            'temp_max', 'dwpt_max', 'rhum_max', 'wdir_max', 'wspd_max',
                            'dc_max', 'ffmc_max', 'dmc_max', 'nesterov_max', 'munger_max', 'kbdi_max',
                            'isi_max', 'angstroem_max', 'bui_max', 'fwi_max', 'dailySeverityRating_max',
                            'temp16_max', 'dwpt16_max', 'rhum16_max', 'wdir16_max', 'wspd16_max',
                            'air_max',
                            'Calendar_max',

                            'Historical_sum',
                            #'Historical_grad',
                            'AutoRegressionBin_sum',
                            #'AutoRegressionReg_sum',
                            #'AutoRegressionReg_grad'
                            ]

varying_time_variables_name = []