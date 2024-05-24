cems_variables = ['temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
                'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                'isi', 'angstroem', 'bui', 'fwi', 'dailySeverityRating',
                'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
                'days_since_rain', 'sum_consecutive_rainfall', 'sum_last_7_days',]

air_variables = ['O3', 'NO2', 'PM10', 'PM25']

# Encoder
sentinel_variables = ['NDVI', 'NDMI', 'NDBI', 'NDSI', 'NDWI']
landcover_variables = [#'landcover',
                        #'osmnx',
                       'foret',
                       ]
calendar_variables = ['month', 'dayofweek', 'dayofyear', 'isweekend', 'couvrefeux', 'confinemenent',
                    'ramadan', 'bankHolidays', 'bankHolidaysEve', 'holidays', 'holidaysBorder']
geo_variables = ['departement']

# Percentage
foret_variables = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']
osmnx_variables = ['0', '1', '2', '3', '4']
dynamic_world_variables = ['water',
                            'tree', 'grass', 'crops', 'shrub', 'flooded', 'built', 'bare', 'snow'
                         ]

# Influence
foret_influence_variables = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']
osmnx_influence_variables = ['0', '1', '2', '3', '4']
dynamic_world_influence_variables = ['water',
                            'tree', 'grass', 'crops', 'shrub', 'flooded', 'built', 'bare', 'snow'
                         ]
historical_variables = ['influence']

elevation_variables = ['elevation']
population_variabes = ['population']
vigicrues_variables = ['12',
                       #'16'
                    ]

# Time varying
varying_time_variables = ['temp_mean', 'dwpt_mean', 'rhum_mean', 'wdir_mean', 'wspd_mean', 'prec24h_mean',
                            'dc_mean', 'ffmc_mean', 'dmc_mean', 'nesterov_mean', 'munger_mean', 'kbdi_mean',
                            'isi_mean', 'angstroem_mean', 'bui_mean', 'fwi_mean', 'dailySeverityRating_mean',
                            'temp16_mean', 'dwpt16_mean', 'rhum16_mean', 'wdir16_mean', 'wspd16_mean', 'prec24h16_mean',
                            'Calendar_mean',
                            'Historical_mean',
                            'air_mean']