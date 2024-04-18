from pathlib import Path
import vacances_scolaires_france
import jours_feries_france
import datetime as dt
from models.models import *
from models.models_2D import *

# Setting see for reproductible results
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Arboressence
rootDisk = Path('/media/caron/X9 Pro/travaille/Thèse/csv')
root = Path('/home/caron/Bureau')

# Dictionnaire de gestion des départements
int2str = {
    1 : 'ain',
    25 : 'doubs',
    69 : 'rhone',
    78 : 'yvelines'
}

int2strMaj = {
    1 : 'Ain',
    25 : 'Doubs',
    69 : 'Rhone',
    78 : 'Yvelines'
}

int2name = {
    1 : 'departement-01-ain',
    25 : 'departement-25-doubs',
    69 : 'departement-69-rhone',
    78 : 'departement-78-yvelines'
}

str2int = {
    'ain': 1,
    'doubs' : 25,
    'rhone' : 69,
    'yvelines' : 78
}

str2intMaj = {
    'Ain': 1,
    'Doubs' : 25,
    'Rhone' : 69,
    'Yvelines' : 78
}

str2name = {
    'Ain': 'departement-01-ain',
    'Doubs' : 'departement-25-doubs',
    'Rhone' : 'departement-69-rhone',
    'Yvelines' : 'departement-78-yvelines'
}

name2str = {
    'departement-01-ain': 'Ain',
    'departement-25-doubs' : 'Doubs',
    'departement-69-rhone' : 'Rhone',
    'departement-78-yvelines' : 'Yvelines'
}

name2int = {
    'departement-01-ain': 1,
    'departement-25-doubs' : 25,
    'departement-69-rhone' : 69,
    'departement-78-yvelines' : 78
}

ACADEMIES = {
    '1': 'Lyon',
    '69': 'Lyon',
    '25': 'Besançon',
    '78': 'Versailles',
}

foret = {'Châtaignier': 1,
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

# Variables : 
# mean max min and std for scale > 0 else value
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

cems_variables = ['temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
                'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                'isi', 'angstroem', 'bui', 'fwi', 'daily_severity_rating',
                'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16']

sentinel_variables = ['NDVI', 'NDMI', 'NDBI', 'NDSI', 'NDWI']
landcover_variables = ['landcover']
foret_variables = ['foret']
osmnx_variables = ['highway']
elevation_variables = ['elevation']
population_variabes = ['population']
calendar_variables = ['month', 'dayofweek', 'dayofyear', 'isweekend', 'couvrefeux', 'confinemenent',
                    'ramadan', 'bankHolidays', 'bankHolidaysEve', 'holidays', 'holidaysBorder']
geo_variables = ['departement']
historical_variables = ['influence']
air_variables = ['O3', 'NO2', 'PM10', 'PM25']

# Variable à encoder
TARGET_ENCODE = ['jourSemaine', 'jourAnnee', 
                 'confinement1', 'confinement2', 'couvrefeux', 
                 'ramadan', 
                 'bankHolidays', 'bankHolidaysEve', 'holidays', 'holidaysBorder']

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
trainDepartements = ['departement-01-ain',
                'departement-25-doubs',
                'departement-78-yvelines'
                ]

k_days = 7 # Size of the time series sequence
minPoint = 100 # Define which train set use
scaling='z-score' # Scale to used
encoding='Catboost' # How we encode the categorical variable
dummy = False # Is a dummy test (we don't use the complete time sequence)
binary = True # Do we train "traditionnal models" (xgboost, lightgbm, etc...) as binary
nmax = 6 # Number maximal for each node (usually 6 will be 1st order nodes)
maxDist = 30 # Maximum distance to create a edge (depend of the scale, for scale 5 20 is good)
shape2D = (25,25)
jours_feries = sum([list(jours_feries_france.JoursFeries.for_year(k).values()) for k in range(2017,2023)],[]) # French Jours fériés, used in features_*.py 
veille_jours_feries = sum([[l-dt.timedelta(days=1) for l \
            in jours_feries_france.JoursFeries.for_year(k).values()] for k in range(2017,2023)],[]) # French Veille Jours fériés, used in features_*.py 
vacances_scolaire = vacances_scolaires_france.SchoolHolidayDates() # French Holidays used in features_*.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # The device on which we train each models
in_dim = 100 # Number of features we use to train GNN and based tree models
in_dim_2D = 52 # Number of features we use to train CNN models
dropout = 0.3

# The neural network we train
dico_model = {      'GAT': GAT(in_dim=[in_dim, 64, k_days+1],
                        heads=[4, 4, 2],
                        dropout=dropout,
                        bias=True,
                        device=device,
                        act_func='relu',
                        n_sequences=k_days+1),

                    'ST-GATTCN': STGATCN(n_sequences=k_days+1,
                                        num_of_layers=4,
                                        in_channels=in_dim,
                                         end_channels=64,
                                         skip_channels=64,
                                         residual_channels=64,
                                         dilation_channels=64,
                                         dropout=dropout,
                                         heads=6, act_func='gelu',
                                         device=device),

                    'ST-GATCONV': STGATCONV(k_days+1,
                                            num_of_layers=4,
                                            in_channels=in_dim,
                                            hidden_channels=64,
                                            residual_channels=64,
                                            end_channels=64,
                                            dropout=dropout,
                                            heads=6,
                                            act_func='gelu',
                                            device=device),

                    'ST-GCNCONV' : STGCNCONV(k_days+1,
                                            num_of_layers=4,
                                            in_channels=in_dim,
                                            hidden_channels=64,
                                            residual_channels=64,
                                            end_channels=64,
                                            dropout=dropout,
                                            act_func='gelu',
                                            device=device),

                    'ATGN' : TemporalGNN(in_channels=in_dim,
                                        hidden_channels=64,
                                        out_channels=32,
                                        n_sequences=k_days + 1,
                                        device=device,
                                        act_func='relu'),

                   'ST-GATLTSM' : ST_GATLSTM(in_channels=in_dim,
                                            hidden_channels=64,
                                             out_channels=64,
                                             end_channels=64,
                                             n_sequences=k_days + 1,
                                             device=device, act_func='relu', heads=6, dropout=0.3),


                    'Zhang' : Zhang(in_channels=in_dim_2D,
                                hidden_channels=64,
                                    end_channels=128,
                                    dropout=dropout,
                                    binary=False,
                                    device=device,
                                    n_sequences=k_days+1),

                    'ConvLSTM' : CONVLSTM(in_channels=in_dim_2D,
                                        hidden_dim=[32],
                                          end_channels=64,
                                          n_sequences=k_days+1,
                                          device=device, act_func='gelu')
                    }