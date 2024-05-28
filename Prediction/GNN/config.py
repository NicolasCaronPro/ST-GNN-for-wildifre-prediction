from pathlib import Path
import vacances_scolaires_france
import jours_feries_france
import datetime as dt
from models.models import *
from models.models_2D import *
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

# Variables : 
# mean max min and std for scale > 0 else value
features = [
            'temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
            'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
            'isi', 'angstroem', 'bui', 'fwi', 'dailySeverityRating',
            'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
            'days_since_rain', 'sum_consecutive_rainfall', 'sum_last_7_days',
            'elevation', 'population',
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
            'elevation', 'population',
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
departements = [#'departement-01-ain',
                'departement-25-doubs',
                #'departement-69-rhone',
                #'departement-78-yvelines'
                ]

# We trained only on those 3 departmenet
trainDepartements = [
                #'departement-01-ain',
                'departement-25-doubs',
                #'departement-69-rhone',
                #'departement-78-yvelines',
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
dummy = False # Is a dummy test (we don't use the complete time sequence)
nmax = 6 # Number maximal for each node (usually 6 will be 1st order nodes)

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
        10 : 35
        }

shape2D = (25,25)

jours_feries = sum([list(jours_feries_france.JoursFeries.for_year(k).values()) for k in range(2017,2023)],[]) # French Jours fériés, used in features_*.py 
veille_jours_feries = sum([[l-dt.timedelta(days=1) for l \
            in jours_feries_france.JoursFeries.for_year(k).values()] for k in range(2017,2023)],[]) # French Veille Jours fériés, used in features_*.py 
vacances_scolaire = vacances_scolaires_france.SchoolHolidayDates() # French Holidays used in features_*.py
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # The device on which we train each models
device = torch.device('cpu')
in_dim = 100 # Number of features we use to train GNN and based tree models
in_dim_2D = 52 # Number of features we use to train CNN models
dropout = 0.03
act_func = 'relu'
Rewrite = True
scaling='z-score' # Scale to used
encoding='Catboost' # How we encode the categorical variable

"""# Neural networks
dico_model = {'GAT': GAT(in_dim=[in_dim, 64, 64, 64],
                        heads=[4, 4, 2],
                        dropout=dropout,
                        bias=True,
                        device=device,
                        act_func=act_func,
                        n_sequences=k_days),

                    'ST-GATTCN': STGATCN(n_sequences=k_days,
                                        num_of_layers=3,
                                        in_channels=in_dim,
                                         end_channels=64,
                                         skip_channels=32,
                                         residual_channels=32,
                                         dilation_channels=32,
                                         dropout=dropout,
                                         heads=6, act_func=act_func,
                                         device=device),
                                        
                    'ST-GATCONV': STGATCONV(k_days,
                                            num_of_layers=3,
                                            in_channels=in_dim,
                                            hidden_channels=32,
                                            residual_channels=64,
                                            end_channels=32,
                                            dropout=dropout,
                                            heads=6,
                                            act_func=act_func,
                                            device=device),

                    'ST-GCNCONV' : STGCNCONV(k_days,
                                            num_of_layers=3,
                                            in_channels=in_dim,
                                            hidden_channels=64,
                                            residual_channels=64,
                                            end_channels=32,
                                            dropout=dropout,
                                            act_func=act_func,
                                            device=device),

                    'ATGN' : TemporalGNN(in_channels=in_dim,
                                        hidden_channels=64,
                                        out_channels=64,
                                        n_sequences=k_days,
                                        device=device,
                                        act_func=act_func,
                                        dropout=dropout),

                   'ST-GATLTSM' : ST_GATLSTM(in_channels=in_dim,
                                            hidden_channels=[64, 64, 64],
                                             out_channels=64,
                                             end_channels=32,
                                             n_sequences=k_days,
                                             device=device, act_func=act_func, heads=6, dropout=dropout),

                    'Zhang' : Zhang(in_channels=in_dim_2D,
                                hidden_channels=64,
                                    end_channels=128,
                                    dropout=dropout,
                                    binary=False,
                                    device=device,
                                    n_sequences=k_days),

                    'ConvLSTM' : CONVLSTM(in_channels=in_dim_2D,
                                        hidden_dim=[32, 32, 32],
                                          end_channels=64,
                                          n_sequences=k_days,
                                          dropout=dropout,
                                          device=device, act_func=act_func)
                    }"""