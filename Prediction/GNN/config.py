from pathlib import Path
import vacances_scolaires_france
import jours_feries_france
import datetime as dt
from forecasting_models.pytorch.tools_2 import *
from forecasting_models.pytorch.loss import *
from forecasting_models.sklearn.sklearn_api_models_config import *
import logging
import socket

def get_machine_info():
    try:
        # Obtenir le nom d'hôte de la machine
        hostname = socket.gethostname()
        print(f"Nom de l'hôte : {hostname}")

        # Obtenir l'adresse IP locale
        local_ip = socket.gethostbyname(hostname)
        print(f"Adresse IP locale : {local_ip}")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

    return hostname

is_pc = get_machine_info() == 'caron-Precision-7780'

MLFLOW = False
if is_pc:
    MLFLOW = True
if MLFLOW:
    import mlflow
    from mlflow import MlflowClient
    from mlflow.models import infer_signature

# Setting see for reproductible results
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

eps = {'departement-01-ain' : {'risk' : 0.4,
       'population' : 0.01,
       'elevation' : 5,
       'foret_landcover' : 0.5,
       'dynamicworld' : 1},
       'departement-25-doubs' : {'risk' : 0.4,
       'population' : 0.01,
       'elevation' : 5,
       'foret_landcover' : 0.5,
       'dynamicworld' : 1},
       'departement-69-rhone' : {'risk' : 0.5,
       'population' : 0.01,
       'elevation' : 5,
       'foret_landcover' : 0.5,
       'dynamicworld' : 1},
       'departement-78-yvelines' : {'risk' : 0.5,
       'population' : 0.01,
       'elevation' : 5,
       'foret_landcover' : 0.5,
       'dynamicworld' : 1},
       }

metric = {'risk' : 'euclidean',
       'population' : 'cosine',
       'elevation' : 'l1',
       'foret_landcover' : 'cosine',
       'dynamicworld' : ...}

ACADEMIES = {
    '1': 'Lyon',
    '2': 'Amiens',
    '3': 'Clermont-Ferrand',
    '4': 'Aix-Marseille',
    '5': 'Aix-Marseille',
    '6': 'Nice',
    '7': 'Grenoble',
    '8': 'Reims',
    '9': 'Toulouse',
    '10': 'Reims',
    '11': 'Montpellier',
    '12': 'Toulouse',
    '13': 'Aix-Marseille',
    '14': 'Caen',
    '15': 'Clermont-Ferrand',
    '16': 'Poitiers',
    '17': 'Poitiers',
    '18': 'Orléans-Tours',
    '19': 'Limoges',
    '21': 'Dijon',
    '22': 'Rennes',
    '23': 'Limoges',
    '24': 'Bordeaux',
    '25': 'Besançon',
    '26': 'Grenoble',
    '27': 'Normandie',
    '28': 'Orléans-Tours',
    '29': 'Rennes',
    '30': 'Montpellier',
    '31': 'Toulouse',
    '32': 'Toulouse',
    '33': 'Bordeaux',
    '34': 'Montpellier',
    '35': 'Rennes',
    '36': 'Orléans-Tours',
    '37': 'Orléans-Tours',
    '38': 'Grenoble',
    '39': 'Besançon',
    '40': 'Bordeaux',
    '41': 'Orléans-Tours',
    '42': 'Lyon',
    '43': 'Clermont-Ferrand',
    '44': 'Nantes',
    '45': 'Orléans-Tours',
    '46': 'Toulouse',
    '47': 'Bordeaux',
    '48': 'Montpellier',
    '49': 'Nantes',
    '50': 'Normandie',
    '51': 'Reims',
    '52': 'Reims',
    '53': 'Nantes',
    '54': 'Nancy-Metz',
    '55': 'Nancy-Metz',
    '56': 'Rennes',
    '57': 'Nancy-Metz',
    '58': 'Dijon',
    '59': 'Lille',
    '60': 'Amiens',
    '61': 'Normandie',
    '62': 'Lille',
    '63': 'Clermont-Ferrand',
    '64': 'Bordeaux',
    '65': 'Toulouse',
    '66': 'Montpellier',
    '67': 'Strasbourg',
    '68': 'Strasbourg',
    '69': 'Lyon',
    '70': 'Besançon',
    '71': 'Dijon',
    '72': 'Nantes',
    '73': 'Grenoble',
    '74': 'Grenoble',
    '75': 'Paris',
    '76': 'Normandie',
    '77': 'Créteil',
    '78': 'Versailles',
    '79': 'Poitiers',
    '80': 'Amiens',
    '81': 'Toulouse',
    '82': 'Toulouse',
    '83': 'Nice',
    '84': 'Aix-Marseille',
    '85': 'Nantes',
    '86': 'Poitiers',
    '87': 'Limoges',
    '88': 'Nancy-Metz',
    '89': 'Dijon',
    '90': 'Besançon',
    '91': 'Versailles',
    '92': 'Versailles',
    '93': 'Créteil',
    '94': 'Créteil',
    '95': 'Versailles',
    '971': 'Guadeloupe',
    '972': 'Martinique',
    '973': 'Guyane',
    '974': 'La Réunion',
    '976': 'Mayotte',
}

foret = {
    "Châtaignier": 1,
    "Chênes décidus": 2,
    "Chênes sempervirents": 3,
    "Conifères": 4,
    "Douglas": 5,
    "Feuillus": 6,
    "Hêtre": 7,
    "Mélèze": 8,
    "Mixtes": 9,
    "NC": 10,
    "NR": 11,
    "Pin à crochets, pin cembro": 12,
    "Pin autre": 13,
    "Pin d'Alep": 14,
    "Pin laricio, pin noir": 15,
    "Pin maritime": 16,
    "Pin sylvestre": 17,
    "Pins mélangés": 18,
    "Peuplier": 19,
    "Robinier": 20,
    "Sapin, épicéa": 21
}

foretint2str = {
    '0': 'PasDeforet',
    '1': 'Châtaignier',
    '2': 'Chênes décidus',
    '3': 'Chênes sempervirents',
    '4': 'Conifères',
    '5': 'Douglas',
    '6': 'Feuillus',
    '7': 'Hêtre',
    '8': 'Mélèze',
    '9': 'Mixtes',
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

newFeatures = []

import datetime as dt

def get_academic_zone(name, date):
    dict_zones = {
        'Aix-Marseille': ('B', 'B'),
        'Amiens': ('B', 'B'),
        'Besançon': ('B', 'A'),
        'Bordeaux': ('C', 'A'),
        'Caen': ('A', 'B'),
        'Clermont-Ferrand': ('A', 'A'),
        'Créteil': ('C', 'C'),
        'Dijon': ('B', 'A'),
        'Grenoble': ('A', 'A'),
        'Lille': ('B', 'B'),
        'Limoges': ('B', 'A'),
        'Lyon': ('A', 'A'),
        'Montpellier': ('A', 'C'),
        'Nancy-Metz': ('A', 'B'),
        'Nantes': ('A', 'B'),
        'Nice': ('B', 'B'),
        'Orléans-Tours': ('B', 'B'),
        'Paris': ('C', 'C'),
        'Poitiers': ('B', 'A'),
        'Reims': ('B', 'B'),
        'Rennes': ('A', 'B'),
        'Rouen': ('B', 'B'),
        'Strasbourg': ('B', 'B'),
        'Toulouse': ('A', 'C'),
        'Versailles': ('C', 'C'),
        'Guadeloupe': ('C', 'C'),
        'Martinique': ('C', 'C'),
        'Guyane': ('C', 'C'),
        'La Réunion': ('C', 'C'),
        'Mayotte': ('C', 'C'),
        'Normandie': ('A', 'B'),  # Choix arbitraire de zone pour l'académie Normandie après 2020
    }

    if name == 'Normandie':
        if date < dt.datetime(2020, 1, 1):
            if date < dt.datetime(2016, 1, 1):
                # Avant 2016, on prend en compte l'ancienne académie de Caen ou Rouen
                return 'A'  # Zone de Caen
            return 'B'  # Zone de Rouen après 2016
        else:
            return dict_zones[name][1]  # Zone après la fusion en 2020
    
    # Cas général pour les autres académies
    if date < dt.datetime(2016, 1, 1):
        return dict_zones[name][0]
    return dict_zones[name][1]

ids_columns = ['id', 'longitude', 'latitude', 'departement', 'date', 'weight', 'days_until_next_event']
targets_columns = ['class_risk', 'nbsinister', 'risk']
weights_columns = ['weight_proportion_on_zero_class',
                   'weight_class',
                   'weight_one',
                   'weight_normalize',
                   'weight_nbsinister',
                   'weight_proportion_on_zero_sinister',
                   'weight_random',
                   
                   'weight_proportion_on_zero_class_nbsinister',
                   'weight_class_nbsinister',
                   'weight_proportion_on_zero_sinister_nbsinister',
                   'weight_one_nbsinister',
                   'weight_normalize_nbsinister',
                   'weight_nbsinister_nbsinister',
                   'weight_random_nbsinister']

############################ Logger ######################################

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")

# Handler pour afficher les logs dans le terminal
streamHandler = logging.StreamHandler(stream=sys.stdout)
streamHandler.setFormatter(logFormatter)
logger.addHandler(streamHandler)

futur_met = 'mean'
dummy = False # Is a dummy test (we don't use the complete time sequence)
nmax = 6 # Number maximal for each node (usually 6 will be 1st order nodes)
tresh_kmeans = 0.15

FAST = False

if MLFLOW:
    try:
       tracking_uri = "http://127.0.0.1:8080"
       mlflow.set_tracking_uri(uri=tracking_uri)
       client = MlflowClient(tracking_uri=tracking_uri)
    except Exception as e:
        logger.info(f'{e}')
        exit(1)


maxDist = {0 : 5,
           1 : 10,
           2 : 15,
           3 : 30,
        4 : 20,
        5 : 25,
        6 : 30,
        7 : 35,
        8 : 35,
        9 : 35,
        10 : 35,
        15 : 45,
        20 : 45,
        25 : 50,
        30 : 50,
        35 : 60,
        40 : 60,
        60 : 50,
        70 : 55,
        80 : 60,
        100 : 75,
        143 : 100,
        'departement': 150,
        }

resolutions = {'2x2' : {'x' : 0.02875215641173088,'y' :  0.020721094073767096},
                '1x1' : {'x' : 0.01437607820586544,'y' : 0.010360547036883548},
                '0.5x0.5' : {'x' : 0.00718803910293272,'y' : 0.005180273518441774},
                '0.03x0.03' : {'x' : 0.0002694945852326214,'y' :  0.0002694945852352859}}

shape2D = {10: (24, 24),
           30 : (30, 30),
           3 : (15,15),
            8 : (30,30),
            'departement' : (64,64)}

jours_feries = sum([list(jours_feries_france.JoursFeries.for_year(k).values()) for k in range(2017,2023)],[]) # French Jours fériés, used in features_*.py 
veille_jours_feries = sum([[l-dt.timedelta(days=1) for l \
            in jours_feries_france.JoursFeries.for_year(k).values()] for k in range(2017,2023)],[]) # French Veille Jours fériés, used in features_*.py 
vacances_scolaire = vacances_scolaires_france.SchoolHolidayDates() # French Holidays used in features_*.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # The device on which we train each models
device = torch.device('cpu')
Rewrite = True
#scaling='MinMax' # Scale to used
#scaling='none' # Scale to used
encoding='Catboost' # How we encode the categorical variable
# Methods for reducing features

epochs = 10000
lr = 0.0000005
PATIENCE_CNT = 50
CHECKPOINT = 100
batch_size = 64

METHODS_TEMPORAL = ['mean', 'min', 'max',
        # 'std',
        #   'sum',
           #'grad'
           ]

METHODS_SPATIAL = ['mean', 'min', 'max',
            'std',
           'sum',
           #'grad'
           ]

METHODS_TEMPORAL_TRAIN = ['mean', 'min', 'max',
            #'std',
           #'sum',
           #'grad'
           ]

METHODS_SPATIAL_TRAIN = ['mean', 'min', 'max',
            #'std',
           #'sum',
           #'grad'
           ]

METHODS_KMEANS = ['mean', 'min', 'max',
                  #'std'
                  ]
METHODS_KMEANS_TRAIN = ['mean', 'min', 'max',
                        #'std'
                        ]

num_lstm_layers = 3
dropout = 0.03

sklearn_model_list = ['xgboost', 'lightgbm', 'svm', 'rf', 'dt', 'ngboost']
models_2D = ['Zhang', 'Unet', 'ConvLSTM']
models_hybrid = ['ConvGraphNet', 'ST-ConvGraphNet', 'HybridConvGraphNet']
temporal_model_list = ['LSTM', 'DST-GCN', 'ST-GCN', 'ST-GAT', 'ATGCN', 'ST-GATLSTM']
daily_model_list = ['GCN', 'GAT', 'KAN']

zhang_layer_conversion = {
    8 : 15,
    10 : 15,
    9 : 15,
    }