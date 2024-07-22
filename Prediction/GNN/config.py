from pathlib import Path
import vacances_scolaires_france
import jours_feries_france
import datetime as dt
from forecasting_models.models import *
from forecasting_models.models_2D import *
import logging

# Setting see for reproductible results
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

ACADEMIES = {
    '1': 'Lyon',
    '69': 'Lyon',
    '25': 'Besançon',
    '78': 'Versailles',
}

foret = {
'PasDeforet' : 0,
'Châtaignier': 1,
 'Chênes décidus': 2,
 'Conifères': 3,
 'Douglas': 4,
 'Feuillus': 5,
 'Hêtre': 6,
 'Mixte': 7,
 'Mélèze': 8,
 'NC': 9,
 'NR': 10,
 'Peuplier': 11,
 'Pin autre': 12,
 'Pin laricio, pin noir': 13,
 'Pin maritime': 14,
 'Pin sylvestre': 15,
 'Pins mélangés': 16,
 'Robinier': 17,
 'Sapin, épicéa': 18}

foretint2str = {
'0' : 'PasDeforet',
'1':'Châtaignier',
 '2': 'Chênes décidus',
 '3': 'Conifères',
 '4': 'Douglas',
 '5': 'Feuillus',
 '6': 'Hêtre',
 '7': 'Mixte',
 '8': 'Mélèze',
 '9': 'NC',
 '10': 'NR',
 '11': 'Peuplier',
 '12': 'Pin autre',
 '13': 'Pin laricio, pin noir',
 '14': 'Pin maritime',
 '15': 'Pin sylvestre',
 '16': 'Pins mélangés',
 '17': 'Robinier',
 '18': 'Sapin, épicéa'}

osmnxint2str = {
'0' : 'PasDeRoute',
'1':'motorway',
 '2': 'primary',
 '3': 'secondary',
 '4': 'tertiary',
 '5': 'path'}

# Variables : 
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
                'sentinel',
                'landcover',
                #'vigicrues',
                'foret',
                'highway',
                'dynamicWorld',
                'Calendar',
                'Historical',
                'Geo',
                'air',
                #'nappes',
                #'AutoRegressionReg',
                'AutoRegressionBin'
                ]

newFeatures = []

kmeansFeatures = [
            #'temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd',
            #'prec24h',
            #'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
            #'isi', 'angstroem', 'bui', 'fwi', 'dailySeverityRating',
            #'temp16', 'dwpt16', 'rhum16',
            # 'prcp16', # 'wdir16', 'wspd16',
            # 'prec24h16',
            #'days_since_rain', 'sum_consecutive_rainfall', 'sum_last_7_days',
            #'elevation',
            #'population',
            #'sentinel',
            #'landcover',
            #'vigicrues',
            #'foret',
            #'highway',
            #'dynamicWorld',
            'Calendar',
            #'Historical',
            #'Geo',
            #'air',
            #'nappes',
            #'AutoRegressionReg',
            #'AutoRegressionBin'
            ]

PERIODES_A_IGNORER = {
    1: {'interventions': [(dt.datetime(2023, 7, 3), dt.datetime(2023, 7, 26)),
                             (dt.datetime(2024, 3, 10), dt.datetime(2024,5,29))],
           'appels': [(dt.datetime(2023, 3, 5), dt.datetime(2023, 7, 19)),
                      (dt.datetime(2024, 4, 1), dt.datetime(2024,5,24))]},
    25: {'interventions': [],
           'appels': []},
    78: {'interventions': [],
           'appels': []},
    69: {'interventions': [(dt.datetime(2023, 1, 1), dt.datetime(2024, 6, 26))],
           'appels': []}       
}

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
            'Rouen ': ('B', 'B'),
            'Strasbourg': ('B', 'B'),
            'Toulouse': ('A', 'C'),
            'Versailles': ('C', 'C')
        }
        if date < dt.datetime(2016, 1, 1):
            return dict_zones[name][0]
        return dict_zones[name][1]

# All department for which we have data
departements = ['departement-01-ain',
                'departement-25-doubs',
                'departement-69-rhone',
                'departement-78-yvelines'
                ]

# We trained only on those 3 departmenet
trainDepartements = [
                'departement-01-ain',
                'departement-25-doubs',
                'departement-69-rhone',
                'departement-78-yvelines',
                ]

############################ Logger ######################################

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")

# Handler pour afficher les logs dans le terminal
streamHandler = logging.StreamHandler(stream=sys.stdout)
streamHandler.setFormatter(logFormatter)
logger.addHandler(streamHandler)

k_days = 0 # Size of the time series sequence
days_in_futur = 7 # The target time validation
futur_met = 'mean'
dummy = False # Is a dummy test (we don't use the complete time sequence)
nmax = 6 # Number maximal for each node (usually 6 will be 1st order nodes)

FAST = False

maxDist = {0 : 5,
           1 : 10,
           2 : 15,
           3 : 15,
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
        }

resolutions = {'2x2' : {'x' : 0.02875215641173088,'y' :  0.020721094073767096},
                '1x1' : {'x' : 0.01437607820586544,'y' : 0.010360547036883548},
                '0.5x0.5' : {'x' : 0.00718803910293272,'y' : 0.005180273518441774},
                '0.03x0.03' : {'x' : 0.0002694945852326214,'y' :  0.0002694945852352859}}

shape2D = (25,25)

jours_feries = sum([list(jours_feries_france.JoursFeries.for_year(k).values()) for k in range(2017,2023)],[]) # French Jours fériés, used in features_*.py 
veille_jours_feries = sum([[l-dt.timedelta(days=1) for l \
            in jours_feries_france.JoursFeries.for_year(k).values()] for k in range(2017,2023)],[]) # French Veille Jours fériés, used in features_*.py 
vacances_scolaire = vacances_scolaires_france.SchoolHolidayDates() # French Holidays used in features_*.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # The device on which we train each models
#device = torch.device('cpu')
Rewrite = True
scaling='z-score' # Scale to used
#scaling='MinMax' # Scale to used
#scaling='none' # Scale to used
encoding='Catboost' # How we encode the categorical variable
METHODS = ['mean', 'min', 'max', 'std',
           'sum',
           #'grad'
           ] # Methods for reducing features
num_lstm_layers = 1

def make_models(in_dim, in_dim_2D, dropout, act_func):
    # Neural networks
    dico_model = {'GAT_risk': GAT(in_dim=[in_dim, 64, 64, 64],
                            heads=[4, 4, 2],
                            dropout=dropout,
                            bias=True,
                            device=device,
                            act_func=act_func,
                            n_sequences=k_days + 1,
                            binary=False),

                        'DST-GCN_risk': DSTGCN(n_sequences=k_days+1,
                                        in_channels=in_dim,
                                        end_channels=64,
                                        dilation_channels=[64],
                                        dilations=[1],
                                        dropout=dropout,
                                        act_func=act_func,
                                        device=device,
                                        binary=False),
                                                                    
                        'ST-GAT_risk': STGAT(n_sequences=k_days + 1,
                                    in_channels=in_dim,
                                    hidden_channels=[64],
                                    end_channels=64,
                                    dropout=dropout, heads=6,
                                    act_func=act_func, device=device,
                                    binary=False),

                        'ST-GCN_risk' : STGCN(n_sequences=k_days + 1,
                                        in_channels=in_dim,
                                        hidden_channels=[64],
                                        end_channels=64,
                                        dropout=dropout,
                                        act_func=act_func,
                                        device=device,
                                        binary=False),

                        'SDT-GCN_risk' : SDSTGCN(n_sequences=k_days + 1,
                                    in_channels=in_dim,
                                    hidden_channels_temporal=[64],
                                    dilations=[1],
                                    hidden_channels_spatial=[64],
                                    end_channels=64,
                                    dropout=dropout,
                                    act_func=act_func,
                                    device=device,
                                    binary=False),

                        'ATGN_risk' : TemporalGNN(in_channels=in_dim,
                                            hidden_channels=64,
                                            out_channels=64,
                                            n_sequences=k_days + 1,
                                            device=device,
                                            act_func=act_func,
                                            dropout=dropout,
                                            binary=False),

                        'ST-GATLTSM_risk' : ST_GATLSTM(in_channels=in_dim,
                                                hidden_channels=64,
                                                residual_channels=64,
                                                end_channels=32,
                                                n_sequences=k_days + 1,
                                                num_layers=num_lstm_layers,
                                                device=device, act_func=act_func, heads=6, dropout=dropout,
                                                concat=False,
                                                binary=False),

                        'LTSM_risk' : LSTM(in_channels=in_dim, residual_channels=64,
                                      hidden_channels=64,
                                      end_channels=32, n_sequences=k_days + 1,
                                      device=device, act_func=act_func, binary=False,
                                      dropout=dropout, num_layers=num_lstm_layers),
                        }
    return dico_model