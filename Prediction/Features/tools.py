import firedanger
import copy
import meteostat
import datetime as dt
from pathlib import Path
import random

from sympy import EX, true

random.seed(0)
import pandas as pd
import geopandas as gpd
import os
from sklearn.cluster import KMeans
import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import shape, Point
import math
from osgeo import gdal, ogr
from scipy.signal import fftconvolve as scipy_fft_conv
from astropy.convolution import convolve_fft
import rasterio
import rasterio.features
import rasterio.warp
from skimage import img_as_float
from skimage import transform
from sklearn.linear_model import LinearRegression
import pickle
import json
import warnings
from skimage import morphology
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, normalize
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.geometry import box
import xarray as xr
from skimage import morphology
from skimage.segmentation import watershed
from scipy.interpolate import interp1d
from scipy import ndimage as ndi
from skimage.filters import rank
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from scipy.interpolate import griddata
import sys
from itertools import chain
from dico_departements import *
import time
import requests
from rasterio.warp import reproject, Resampling, calculate_default_transform

##################################################################################################
#                                       Meteostat
##################################################################################################

def compute_fire_indices(point, date_debut, date_fin, saison_feux):
    meteostat.Point.radius = 200000
    meteostat.Point.alt_range = 1000
    meteostat.Point.max_count = 5
    location = meteostat.Point(point[0], point[1])
    # logger.info(f"Calcul des indices incendie pour le point de coordonnées {point}")
    df = meteostat.Hourly(location, date_debut-dt.timedelta(hours=24), date_fin)
    df = df.normalize()
    df = df.fetch()
    assert len(df)>0
    df.drop(['tsun', 'coco', 'wpgt'], axis=1, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df['snow'] = df['snow'].fillna(0)
    df['prcp'] = df['prcp'].fillna(0)
    df['rhum'] = df['rhum'].apply(lambda x : min(x, 100))
    df.reset_index(inplace=True)
    df.rename({'time': 'creneau'}, axis=1, inplace=True)
    # La vitesse du vent doit être en m/s
    df['wspd'] =  df['wspd'] * 1000 / 3600
    df.sort_values(by='creneau', inplace=True)
    # Calculer la somme des précipitations des 24 heures précédentes
    df['prec24h'] = df['prcp'].rolling(window=24, min_periods=1).sum()
    df['snow24h'] = df['snow'].rolling(window=24, min_periods=1).sum()
    # Pour s'assurer que prcp24 ne contient des valeurs calculées qu'à midi (12:00)
    # mettre à NaN les lignes qui ne correspondent pas à midi, 
    # puis utiliser ffill pour propager la dernière valeur calculée
    df['hour'] = df['creneau'].dt.hour
    df['prec24h12'] = np.where(df['hour'] == 12, df['prec24h'], np.nan)
    df['snow24h12'] = np.where(df['hour'] == 12, df['snow24h'], np.nan)
    df['prec24h12'].ffill(inplace=True)
    df['snow24h12'].ffill(inplace=True)
    # Données à 16h, utiles pour Nicolas
    for col in ['temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd']:
        df[f'{col}16'] = df[col].copy()
        df[f'{col}16'] = np.where(df['hour'] == 16, df[f'{col}16'], np.nan)
        df[f'{col}16'].ffill(inplace=True)
    df['prec24h16'] = np.where(df['hour'] == 16, df['prec24h'], np.nan)
    df['snow24h16'] = np.where(df['hour'] == 16, df['snow24h'], np.nan)
    df['prec24h16'].ffill(inplace=True)
    df['snow24h16'].ffill(inplace=True)

    # Données à 12h, utiles pour Nicolas
    for col in ['temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd']:
        df[f'{col}12'] = df[col].copy()
        df[f'{col}12'] = np.where(df['hour'] == 12, df[f'{col}12'], np.nan)
        df[f'{col}12'].ffill(inplace=True)

    # Données à 15h, utiles pour Nesterov
    df['temp15h'] = df['temp'].copy()
    df['temp15h'] = np.where(df['hour'] == 15, df['temp15h'], np.nan)
    df['temp15h'].ffill(inplace=True)
    df['rhum15h'] = df['rhum'].copy()
    df['rhum15h'] = np.where(df['hour'] == 15, df['rhum15h'], np.nan)
    df['rhum15h'].ffill(inplace=True)
    # Données à 12h, utiles pour Angstroem
    df['temp12h'] = df['temp'].copy()
    df['temp12h'] = np.where(df['hour'] == 12, df['temp12h'], np.nan)
    df['temp12h'].ffill(inplace=True)
    df['rhum12h'] = df['rhum'].copy()
    df['rhum12h'] = np.where(df['hour'] == 12, df['rhum12h'], np.nan)
    df['rhum12h'].ffill(inplace=True)
    df.drop('hour', axis=1, inplace=True)

    # Température maximale de la veille, pour le KBDI
    df.set_index(df['creneau'], inplace=True)
    df.drop('creneau', axis=1, inplace=True)
    daily_max_temp = df.resample('D').max()
    daily_max_temp['temp24max'] = daily_max_temp['temp'].shift(1)
    df = df.merge(daily_max_temp['temp24max'].asfreq('h', method='ffill'), left_index=True, right_index=True, how='left')    
    
    # Précipitations de la veille, pour le KBDI
    daily_prec = df.resample('D').sum()
    daily_prec['prec24veille'] = daily_max_temp['prcp'].shift(1)
    df = df.merge(daily_prec['prec24veille'].asfreq('h', method='ffill'), left_index=True, right_index=True, how='left')    
    # Somme des précipitations de la semaine écoulée, pour le KBDI
    df['sum_rain_last_7_days'] = df['prcp'].rolling('7D').sum()
    df['sum_snow_last_7_days'] = df['snow'].rolling('7D').sum()
    
    df.reset_index(inplace=True)    
    # Somme des précipitations consécutives, toujours pour le KBDI
    df['no_rain'] = df['prcp'] < 1.8 # Identifier les jours sans précipitations
    df['consecutive_rain_group'] = (df['no_rain']).cumsum() # Calculer les groupes de jours consécutifs avec précipitations
    df['sum_consecutive_rainfall'] = df.groupby('consecutive_rain_group')['prcp'].transform('sum') # Calculer la somme des précipitations pour chaque groupe de jours consécutifs
    df.loc[df['no_rain'], 'sum_consecutive_rainfall'] = 0 # Réinitialiser la somme à 0 pour les jours sans pluie
    df.drop(['no_rain', 'consecutive_rain_group'], axis=1, inplace=True)
    # On peut maintenant calculer les indices
    df = df.loc[df.creneau>=date_debut]
    df.reset_index(inplace=True)

    t = time.time()
    df = df.loc[df.creneau.dt.hour == 12]
    temps = df['temp'].to_numpy()
    temps12 = df['temp12h'].to_numpy()
    temps15 = df['temp15h'].to_numpy()
    temps24max = df['temp24max'].to_numpy()
    wspds = df['wspd'].to_numpy()
    rhums = df['rhum'].to_numpy()
    rhums12 = df['rhum12h'].to_numpy()
    rhums15 = df['rhum15h'].to_numpy()
    months = df['creneau'].dt.month
    months += 1
    months = months.to_numpy()
    prec24h12s = df['prec24h12'].to_numpy()
    prec24hs = df['prec24h'].to_numpy()
    prec24veilles = df['prec24veille'].to_numpy()
    snow24 = df['snow24h'].to_numpy()
    sum_rain_last_7_days = df['sum_rain_last_7_days'].to_numpy()
    sum_snow_last_7_days = df['sum_snow_last_7_days'].to_numpy()
    sum_consecutive_rainfall = df['sum_consecutive_rainfall'].to_numpy()    
    months = df['creneau'].dt.month.to_numpy() + 1
    latitudes = np.full_like(temps, point[0])

    # Day since rain
    treshPrec24 = 1.8
    dsr = np.empty_like(prec24h12s)
    dsr[0] = int(prec24hs[0] > treshPrec24)
    for i in range(1, len(prec24hs)):
        dsr[i] = dsr[i - 1] + 1 if prec24h12s[i] < treshPrec24 else 0
    df['days_since_rain'] = dsr

    """if not saison_feux: # -> pas content car pas envie de réduire mon dataset mais on verra plus tard
        df['dc'] = 0
        df['ffmc'] = 0
        df['dmc'] = 0
        df['isi'] = 0
        df['bui'] = 0
        df['fwi'] = 0
        df['daily_severity_rating'] = 0
        df['nesterov'] = 0
        df['munger'] = 0
        df['kbdi'] = 0
        df['angstroem'] = 0
        return df"""

    # Calcul du DC en passant par numpy
    '''
    Le Drought Code (DC) fait partie du Canadian Forest Fire Weather Index (FWI) System, qui est un système 
    complet utilisé pour estimer le risque d'incendie de forêt. Le DC est spécifiquement conçu pour mesurer 
    les effets séchants à long terme des périodes de temps sec sur les combustibles forestiers profonds. En 
    d'autres termes, il est utilisé pour évaluer la quantité d'humidité séchée dans les matériaux organiques 
    profonds et épais sur le sol de la forêt, qui peuvent s'enflammer et soutenir un feu de forêt même sans 
    l'apport d'humidité pendant une longue période.
    '''
    dc = np.empty_like(temps)
    dc[0] = 0
    consecutive = 0
    for i in range(1, len(temps)):
        if temps[i] > 12 and snow24[i] < 1:
            consecutive += 1
        elif consecutive < 3:
            consecutive = 0

        if consecutive < 3:
            dc[i] = 0
        elif consecutive == 3:
            dc[i] = 15
            consecutive += 1
        else:
            dc[i] = firedanger.indices.dc(temps[i], prec24h12s[i], months[i], latitudes[i], dc[i-1])
            continue

    df['dc'] = dc
    # Calcul du FFMC en passant par numpy
    '''
    Le Fine Fuel Moisture Code (FFMC) est un autre composant du Canadian Forest Fire Weather Index (FWI) System, 
    conçu pour estimer la teneur en humidité des combustibles fins et légers à la surface du sol, qui sont 
    susceptibles de s'enflammer rapidement. Il s'agit essentiellement d'une mesure de la facilité avec laquelle 
    ces combustibles peuvent s'enflammer et soutenir la propagation initiale d'un feu de forêt. Ces combustibles 
    fins comprennent des éléments tels que les feuilles mortes, les brindilles, l'herbe et les petites branches 
    qui ont un diamètre de moins de 6 mm.
    '''
    ffmc = np.empty_like(temps)
    ffmc[0] = 0
    consecutive = 0
    for i in range(1, len(temps)):
                
        if temps[i] > 12 and snow24[i] < 1:
            consecutive += 1
        elif consecutive < 3:
            consecutive = 0

        if consecutive < 3:
            ffmc[i] = 0
        elif consecutive == 3:
            ffmc[i] = 6
            consecutive += 1
        else:
            ffmc[i] = firedanger.indices.ffmc(temps[i], prec24h12s[i], wspds[i], rhums[i], ffmc[i-1])
            continue

    df['ffmc'] = ffmc
    # Calcul du DMC
    '''
    Le Duff Moisture Code (DMC) est un autre indicateur important du Canadian Forest Fire Weather Index (FWI) 
    System. Contrairement au Fine Fuel Moisture Code (FFMC), qui évalue la teneur en humidité des combustibles 
    fins et légers à la surface, le DMC se concentre sur la teneur en humidité des couches de litière et de 
    matières organiques en décomposition situées juste sous la surface. Ces matériaux sont plus épais et moins 
    volatils que les combustibles fins, et ils nécessitent donc plus de temps pour sécher et s'enflammer, mais 
    une fois allumés, ils peuvent soutenir un feu pendant une période prolongée.
    '''
    dmc = np.empty_like(temps)
    dmc[0] = 0
    consecutive = 0
    for i in range(1, len(temps)):

        if temps[i] > 12 and snow24[i] < 1:
            consecutive += 1
        elif consecutive < 3:
            consecutive = 0

        if consecutive < 3:
            dmc[i] = 0
        elif consecutive == 3:
            dmc[i] = 85
            consecutive += 1
        else:
            dmc[i] = firedanger.indices.dmc(temps[i], prec24h12s[i], rhums[i], months[i], latitudes[i], dmc[i-1])
            continue

    df['dmc'] = dmc
    # Calcul des derniers indices du FWI canadian
    '''
    L'Initial Spread Index ISI est spécifiquement conçu pour prédire la vitesse de propagation initiale d'un 
    feu nouvellement allumé, en se basant sur les conditions météorologiques.
    '''
    df['isi'] = df.apply(lambda x: firedanger.indices.isi(x.wspd, x.ffmc), axis=1)
    '''
    Le BUI, ou Buildup Index, est conçu pour quantifier la quantité de combustible disponible pour alimenter un 
    feu de forêt, en se concentrant principalement sur les combustibles moyens et lourds. Le BUI est utilisé pour
    estimer la quantité totale de combustible accumulé et sa capacité à brûler, offrant ainsi une mesure de la 
    lourdeur potentielle et de l'intensité d'un incendie.
    '''
    df['bui'] = firedanger.indices.bui(df['dmc'], df['dc'])
    '''
    Le FWI, ou Fire Weather Index, est l'indice principal du système canadien d'évaluation du danger d'incendie 
    de forêt (Canadian Forest Fire Weather Index System). Il représente une mesure globale du danger d'incendie,
    intégrant plusieurs sous-indices pour fournir une estimation de l'intensité potentielle d'un incendie de forêt. 
    Le FWI est conçu pour refléter les effets combinés des conditions météorologiques actuelles sur le comportement 
    du feu, notamment en termes de propagation et d'intensité.
    '''
    df['fwi'] = firedanger.indices.fwi(df['isi'], df['bui'])
    '''
    Le Daily Severity Rating (DSR) est une composante du système canadien d'évaluation du danger d'incendie de 
    forêt (Canadian Forest Fire Weather Index System), qui fournit une évaluation quantitative de l'intensité 
    d'un incendie de forêt potentiel pour une journée donnée. Le DSR est conçu pour traduire l'indice Fire Weather 
    Index (FWI) en une échelle qui reflète plus directement la difficulté potentielle de contrôler un incendie, 
    ainsi que l'intensité et l'énergie qu'un incendie pourrait dégager.
    '''
    df['daily_severity_rating'] = firedanger.indices.daily_severity_rating(df['fwi'])
    # Calcul de l'indice Nesterov
    '''
    L'indice de Nesterov est un indicateur utilisé pour évaluer le risque d'incendie de forêt. Il est particulièrement 
    utile dans les régions où la végétation est principalement constituée d'herbes et de broussailles. Cet indice est 
    calculé à partir de données météorologiques, en particulier la température de l'air et la quantité de précipitations, 
    et il est conçu pour refléter la sécheresse de la surface et la disponibilité des combustibles fins (comme l'herbe sèche) 
    pour le feu.
    '''
    nesterov = np.empty_like(temps)
    nesterov[0] = 0
    start = False
    for i in range(1, len(temps)):
        if prec24hs[i] > 1:
            start = True
        if start: 
            nesterov[i] = firedanger.indices.nesterov(temps15[i], rhums15[i], prec24hs[i], nesterov[i-1])
        else:
            nesterov[i] = 0
            
    df['nesterov'] = nesterov

    # Calcul de l'indice de sécheresse Munger
    munger = np.empty_like(temps)
    munger[0] = 0
    start = False
    for i in range(1, len(temps)):
        if prec24hs[i] > 0.05:
            start = True
        if start: 
            munger[i] = firedanger.indices.munger(prec24hs[i], munger[i-1])
        else:
            munger[i] = 0

    df['munger'] = munger

    # Calcul du kbdi
    """
    Le Keetch-Byram Drought Index (KBDI) est un indice utilisé principalement pour évaluer le risque de feu de forêt 
    en se basant sur la quantité d'humidité présente dans le sol. L'indice varie typiquement de 0 (sol saturé) à 800 
    (conditions extrêmement sèches).
    """
    dg = meteostat.Hourly(location, 
                          dt.datetime(date_debut.year, 1, 1), 
                          min(dt.datetime(date_debut.year+1, 1, 1),dt.datetime.now()))
    dg = dg.normalize()
    dg = dg.fetch()
    pAnnualAvg = dg['prcp'].mean() # Annual rainfall average [mm].
    kbdi = np.empty_like(temps)
    kbdi[0] = 0
    start = False
    for i in range(1, len(temps)):
        if sum_rain_last_7_days[i] > 152:
            start = True
        if start:
            kbdi[i] = max(0, min(800, firedanger.indices.kbdi(temps24max[i], 
                                                            prec24veilles[i],
                                                            kbdi[i-1], 
                                                            sum_consecutive_rainfall[i],
                                                            sum_rain_last_7_days[i],
                                                            30, # weekly rain threshold to initialize index [mm]
                                                            pAnnualAvg)))
        else:
            kbdi[i] = 0

    df['kbdi'] = kbdi

    # Calcul de l'indice Angstroem
    '''
    Indice météorologique qui a été principalement utilisé pour estimer la probabilité et l'intensité des 
    incendies de forêt. Cet indice se concentre sur deux variables météorologiques principales : la précipitation 
    et l'humidité relative. Il a été conçu pour fournir une estimation rapide du potentiel d'incendie basé sur l'état 
    de sécheresse de la végétation et les conditions atmosphériques.
    '''
    angstroem = np.empty_like(temps)
    angstroem[0] = 0
    for i in range(1, len(temps)):
        angstroem[i] = firedanger.indices.angstroem(temps12[i], rhums12[i-1])
    df['angstroem'] = angstroem

    df['dc'] = df['dc'].apply(lambda x : max(x, 0))
    df['ffmc'] = df['ffmc'].apply(lambda x : max(x, 0))
    df['dmc'] = df['dmc'].apply(lambda x : max(x, 0))
    df['isi'] = df['isi'].apply(lambda x : max(x, 0))
    df['bui'] = df['bui'].apply(lambda x : max(x, 0))
    df['fwi'] = df['fwi'].apply(lambda x : max(x, 0))
    df['daily_severity_rating'] = df['daily_severity_rating'].apply(lambda x : max(x, 0))
    df['nesterov'] = df['nesterov'].apply(lambda x : max(x, 0))
    df['munger'] = df['munger'].apply(lambda x : max(x, 0))
    df['kbdi'] = df['kbdi'].apply(lambda x : max(x, 0))
    df['kbdi'] = df['kbdi'].apply(lambda x : min(x, 800))
    df['angstroem'] = df['angstroem'].apply(lambda x : max(x, 0))

    return df

def compute_flood_indices(point, date_debut, date_fin):
    meteostat.Point.radius = 200000
    meteostat.Point.alt_range = 1000
    meteostat.Point.max_count = 5
    location = meteostat.Point(point[0], point[1])
    # logger.info(f"Calcul des indices incendie pour le point de coordonnées {point}")
    df = meteostat.Hourly(location, date_debut-dt.timedelta(hours=24), date_fin)
    df = df.normalize()
    df = df.fetch()
    assert len(df)>0
    df.drop(['tsun', 'coco', 'wpgt'], axis=1, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df['snow'] = df['snow'].fillna(0)
    df['prcp'] = df['prcp'].fillna(0)
    df.reset_index(inplace=True)
    df.rename({'time': 'creneau'}, axis=1, inplace=True)
    # La vitesse du vent doit être en m/s
    df['wspd'] =  df['wspd'] * 1000 / 3600
    df.sort_values(by='creneau', inplace=True)
    # Calculer la somme des précipitations des 24 heures précédentes
    df['prec24h'] = df['prcp'].rolling(window=24, min_periods=1).sum()
    df['snow24h'] = df['snow'].rolling(window=24, min_periods=1).sum()
    # Pour s'assurer que prcp24 ne contient des valeurs calculées qu'à midi (12:00)
    # mettre à NaN les lignes qui ne correspondent pas à midi, 
    # puis utiliser ffill pour propager la dernière valeur calculée
    df['hour'] = df['creneau'].dt.hour

    df.drop('hour', axis=1, inplace=True)
    # Température maximale de la veille, pour le KBDI
    df.set_index(df['creneau'], inplace=True)
    df.drop('creneau', axis=1, inplace=True)
    daily_max_temp = df.resample('D').max()
    daily_max_temp['temp24max'] = daily_max_temp['temp'].shift(1)
    df = df.merge(daily_max_temp['temp24max'].asfreq('H', method='ffill'), left_index=True, right_index=True, how='left')    
    # Précipitations de la veille, pour le KBDI
    daily_prec = df.resample('D').sum()
    daily_prec['prec24veille'] = daily_max_temp['prcp'].shift(1)
    df = df.merge(daily_prec['prec24veille'].asfreq('H', method='ffill'), left_index=True, right_index=True, how='left')    
    
    df['sum_rain_last_7_days'] = df['prcp'].rolling('7D').sum()
    df['sum_snow_last_7_days'] = df['snow'].rolling('7D').sum()
    df.reset_index(inplace=True)

    # Somme des précipitations consécutives, toujours pour le KBDI
    df['no_rain'] = df['prcp'] < 1.8 # Identifier les jours sans précipitations
    df['consecutive_rain_group'] = (df['no_rain']).cumsum() # Calculer les groupes de jours consécutifs avec précipitations
    df['sum_consecutive_rainfall'] = df.groupby('consecutive_rain_group')['prcp'].transform('sum') # Calculer la somme des précipitations pour chaque groupe de jours consécutifs
    df.loc[df['no_rain'], 'sum_consecutive_rainfall'] = 0 # Réinitialiser la somme à 0 pour les jours sans pluie
    df.drop(['no_rain', 'consecutive_rain_group'], axis=1, inplace=True)
    # On peut maintenant calculer les indices
    df = df.loc[df.creneau>=date_debut]
    df.reset_index(inplace=True)
    df.loc[0, 'dc'] = 15
    t = time.time()
    months = df['creneau'].dt.month
    months += 1
    months = months.to_numpy()
    prcps = df['prcp'].to_numpy()
    months = df['creneau'].dt.month.to_numpy() + 1

    # hour since rain
    treshPrec24 = 1.8
    dsr = np.empty_like(prcps)
    dsr[0] = int(prcps[0] > treshPrec24)
    for i in range(1, len(prcps)):
        dsr[i] = dsr[i - 1] + 1 if prcps[i] < treshPrec24 else 0
    df['hour_since_rain'] = dsr

        # Calculate daily values for the new columns
    daily_df = df.resample('D', on='creneau').agg({
        'prcp': 'sum',
        'snow': 'sum'
    })

    # Day since rain
    daily_df['day_since_rain'] = 0
    day_since_rain = 0
    for i in range(1, len(daily_df)):
        if daily_df.iloc[i]['prcp'] < treshPrec24:
            day_since_rain += 1
        else:
            day_since_rain = 0
        daily_df.loc[daily_df.index[i], 'day_since_rain'] = day_since_rain

    # Day since snow
    daily_df['day_since_snow'] = 0
    day_since_snow = 0
    for i in range(1, len(daily_df)):
        if daily_df.iloc[i]['snow'] == 0:
            day_since_snow += 1
        else:
            day_since_snow = 0
        daily_df.loc[daily_df.index[i], 'day_since_snow'] = day_since_snow

    # Last 7-day snow cover
    daily_df['last_7_day_snow_cover'] = daily_df['snow'].rolling(7, min_periods=1).sum()

    # Merge daily columns back to the original hourly DataFrame
    df = df.merge(daily_df[['day_since_rain', 'day_since_snow', 'last_7_day_snow_cover']], left_on=df['creneau'].dt.floor('D'), right_index=True, how='left')

    return df

def get_fire_indices(point, date_debut, date_fin, departement):
    if departement not in SAISON_FEUX.keys():
        SAISON_FEUX[departement] = {}
        SAISON_FEUX[departement]['mois_debut'] = 3
        SAISON_FEUX[departement]['jour_debut'] = 1
        SAISON_FEUX[departement]['mois_fin'] = 11
        SAISON_FEUX[departement]['jour_fin'] = 1

    for annee in range(date_debut.year, date_fin.year+1):
        debut = max(date_debut, dt.datetime(annee, 1, 1))
        fin = min(date_fin, dt.datetime(annee+1, 1, 1))
        debut_saison = dt.datetime(annee, 
                                   SAISON_FEUX[departement]['mois_debut'], 
                                   SAISON_FEUX[departement]['jour_debut'])

        fin_saison = dt.datetime(annee, 
                                   SAISON_FEUX[departement]['mois_fin'], 
                                   SAISON_FEUX[departement]['jour_fin'])
        
        dg = compute_fire_indices(point, debut, fin, False)
        if 'df' not in locals():
                df = dg
        else:
            df = pd.concat((df, dg)).reset_index(drop=True)

        """dg2 = compute_fire_indices(point, debut_saison, fin_saison, True)
        if fin_saison < fin:
            dg3 = compute_fire_indices(point, fin_saison, fin, False)
            if 'df' not in locals():
                df = pd.concat((dg, dg2, dg3)).reset_index(drop=True)
            else:
                df = pd.concat((df, dg, dg2, dg3)).reset_index(drop=True)
        else:
            if 'df' not in locals():
                df = pd.concat((dg, dg2)).reset_index(drop=True)
            else:
                df = pd.concat((df, dg, dg2)).reset_index(drop=True)"""
    df = df[(df['creneau'] >= date_debut) & (df['creneau'] <= date_fin)]
    return df

def construct_historical_meteo(start, end, region, dir_meteostat, departement):
    START = dt.datetime.strptime(start, '%Y-%m-%d') #- dt.timedelta(days=10)
    END = dt.datetime.strptime(end, '%Y-%m-%d')

    END += dt.timedelta(hours=1)
    if not (dir_meteostat / 'liste_de_points.pkl').is_file():
            N = 11
            range_x = np.linspace(
                *region.iloc[0].geometry.buffer(0.15).envelope.boundary.xy[0][:2], N)
            range_y = np.linspace(
                *region.iloc[0].geometry.buffer(0.15).envelope.boundary.xy[1][1:3], N)
            points = []
            for point_y in range_y:
                for point_x in range_x:
                    if region.iloc[0].geometry.buffer(0.15).contains(Point((point_x, point_y))):
                        points.append((point_y, point_x))
            print(f"Nombre de points de surveillance pour Meteostat : {len(points)}")
            print(f"On sauvegarde ces points")
            with open(dir_meteostat / 'liste_de_points.pkl', 'wb') as f:
                pickle.dump(points, f)
    else:
        print("On relit les points de Meteostat")
        with open(dir_meteostat / 'liste_de_points.pkl', 'rb') as f:
            points = pickle.load(f)

    print("On récupère les variables du risque d'incendie par hexagone")
    data_plein, data_creux, liste = {}, {}, []
    for index, point in enumerate(sorted(points)):
        print(f"Intégration du point de coordonnées {point}")
        data_plein[point] = get_fire_indices(point, START, END, departement)
        data_plein[point]['latitude'] = point[0]
        data_plein[point]['longitude'] = point[1]
        liste.append(data_plein[point])
    
    def get_date(x):
        return x.strftime('%Y-%m-%d')

    liste = pd.concat(liste)

    liste['creneau'] = liste['creneau'].apply(get_date)
    liste.sort_values('creneau', inplace=True)
    liste.reset_index(drop=False)
    return liste

def old_onstruct_historical_meteo(start, end, region, dir_meteostat):
    MAX_NAN = 5000
    if not (dir_meteostat / 'liste_de_points.pkl').is_file():
        print("Pas de listes de points trouvés pour Meteostat, on les calcule")

        #N = config.getint('PARAMETERS', 'cote_meteostat')
        N = 11
        range_x = np.linspace(
            *region.iloc[0].geometry.buffer(0.15).envelope.boundary.xy[0][:2], N)
        range_y = np.linspace(
            *region.iloc[0].geometry.buffer(0.15).envelope.boundary.xy[1][1:3], N)
        points = []
        for point_y in range_y:
            for point_x in range_x:
                if region.iloc[0].geometry.buffer(0.15).contains(Point((point_x, point_y))):
                    points.append((point_y, point_x))

        print(f"Nombre de points de surveillance pour Meteostat : {len(points)}")
        print(f"On sauvegarde ces points")
        with open(dir_meteostat / 'liste_de_points.pkl', 'wb') as f:
            pickle.dump(points, f)
    else:
        print("On relit les points de Meteostat")
        with open(dir_meteostat / 'liste_de_points.pkl', 'rb') as f:
            points = pickle.load(f)
    print(points)
    print(f"Nombre de points de surveillance pour Meteostat : {len(points)}")
    print("On récupère les archives Meteostat")

    START = dt.datetime.strptime(start, '%Y-%m-%d') #- dt.timedelta(days=10)
    END = dt.datetime.strptime(end, '%Y-%m-%d')

    END += dt.timedelta(hours=1)

    data, data12h, liste, datakbdi = {}, {}, [], {}
    last_point = None
    #meteostat.Stations.max_age = 0
    meteostat.Point.radius = 2000000
    #meteostat.Point.method = "weighted"
    meteostat.Point.alt_range = 10000
    meteostat.Point.max_count = 6
    res = []
    for index, point in enumerate(sorted(points)):
        location = meteostat.Point(point[0], point[1])
        print(f"Intégration du point de coordonnées {point}")
        try:
            data[point] = meteostat.Hourly(location, START, END + dt.timedelta(days=1))
            data[point] = data[point].normalize()
            data[point] = data[point].interpolate()
            data[point] = data[point].fetch()
            assert len(data[point]) > 0
            data[point]['snow'].fillna(0, inplace=True)
            data[point].drop(['tsun', 'coco', 'wpgt'], axis=1, inplace=True)
            data[point].reset_index(inplace=True)
            
            for k in sorted(data[point].columns):
                if data[point][k].isna().sum() > MAX_NAN:
                    print(f"{k} du point {point} possède trop de NaN ({data[point][k].isna().sum()}")
                    if last_point is None:
                        assert False
                    data[point][k] = last_point[k].values

                elif data[point][k].isna().sum() > 0:
                    x = data[point][data[point][k].notna()].index
                    y = data[point][data[point][k].notna()][k]

                    f2 = interp1d(x, y, kind='linear', fill_value='extrapolate')
                    
                    newx = data[point][data[point][k].isna()].index
                    newy = f2(newx)

                    data[point].loc[newx, k] = newy

       
            data[point].rename({'time': 'creneau'}, axis=1, inplace=True)
            data[point].set_index('creneau', inplace=True)
            last_point = data[point]
        except Exception as e:
            print(e)
            if last_point is None:
                print(f"Pas de données pour {point}, et pas de voisin")
                del data[point]
                continue
            else:
                print(f"Pas de données pour {point}, on prend ceux des voisins")
                data[point] = copy.deepcopy(last_point)

        data12h[point] = copy.deepcopy(data[point])
        datakbdi[point] = copy.deepcopy(data[point])

        for k in range(0, 24):
            data12h[point][f"prcp+{k}"] = data12h[point].prcp.shift(k)
            datakbdi[point][f"prcp+{k}"] = datakbdi[point].prcp.shift(k)
            data12h[point][f"snow+{k}"] = data12h[point].prcp.shift(k)
            datakbdi[point][f"snow+{k}"] = datakbdi[point].prcp.shift(k)

        data12h[point]['prec24h'] = data12h[point][[f"prcp+{k}" for k in range(0, 24)]].sum(axis=1)
        datakbdi[point]['prec24h'] = datakbdi[point][[f"prcp+{k}" for k in range(0, 24)]].sum(axis=1)

        data12h[point]['snow24h'] = data12h[point][[f"snow+{k}" for k in range(0, 24)]].sum(axis=1)
        datakbdi[point]['snow24h'] = datakbdi[point][[f"snow+{k}" for k in range(0, 24)]].sum(axis=1)

        data12h[point] = data12h[point][['temp',
                                'dwpt',
                                'rhum',
                                'prcp',
                                'snow',
                                'wdir',
                                'wspd',
                                'snow24h',
                                'prec24h']]
        datakbdi[point] = datakbdi[point][['temp',
                                'dwpt',
                                'rhum',
                                'prcp',
                                'snow',
                                'wdir',
                                'wspd',
                                'prec24h',
                                'snow24h',]]
        
        data12h[point]['wspd'] =  data12h[point]['wspd'] * 1000 / 3600
        datakbdi[point]['wspd'] =  datakbdi[point]['wspd'] * 1000 / 3600

        data16h = data12h[point].loc[data12h[point].index.hour == 16].reset_index()
        data15h = data12h[point].loc[data12h[point].index.hour == 15].reset_index()
        data12h[point] = data12h[point].loc[data12h[point].index.hour == 12]

        treshPrec24 = 1.8
        treshSnow24 = 1.8

        #### data12h
        data12h[point].reset_index(inplace=True)
        data12h[point].loc[0, 'dc'] = 15
        data12h[point].loc[0, 'ffmc'] = 85
        data12h[point].loc[0, 'dmc'] = 6
        data12h[point].loc[0, 'nesterov'] = 0
        data12h[point].loc[0, 'munger'] = 0
        data12h[point].loc[0, 'kbdi'] = 0
        data12h[point].loc[0, 'days_since_rain'] = int(data12h[point].loc[0, 'prec24h'] > treshPrec24)
        data12h[point].loc[0, 'days_since_snow'] = int(data12h[point].loc[0, 'snow24h'] > treshSnow24)
        data12h[point].loc[0, 'sum_consecutive_rainfall'] = data12h[point].loc[0, 'prec24h']
        data12h[point].loc[0, 'sum_consecutive_snowfall'] = data12h[point].loc[0, 'snow24h']
        data12h[point]['rhum'] = data12h[point]['rhum'].apply(lambda x: min(x, 100))

        datakbdi[point].reset_index(inplace=True)
        datakbdi[point]['day'] = datakbdi[point]['creneau'].apply(lambda x : x.date())

        tmax = datakbdi[point].groupby('day')['temp'].max()
        prec = datakbdi[point].groupby('day')['prcp'].sum()
        snow = datakbdi[point].groupby('day')['snow'].sum()
        leni = len(data12h[point])

        consecutive = 0
        print(leni)
        for i in range(1, len(data12h[point])):
            #print(i, '/', leni, data12h[point].loc[i, 'creneau'], consecutive)
            if consecutive == 3:
                data12h[point].loc[i, 'dc'] = firedanger.indices.dc(data12h[point].loc[i, 'temp'],
                                                    data12h[point].loc[i, 'prec24h'],
                                                    data12h[point].loc[i, 'creneau'].month,
                                                    point[0],
                                                    data12h[point].loc[i - 1, 'dc'])
                if data12h[point].loc[i, 'dc'] > 1000:
                    print(data12h[point].loc[i - 1, 'dc'], data12h[point].loc[i, 'dc'], data12h[point].loc[i, 'prec24h'], data12h[point].loc[i-1, 'days_since_rain'])

                data12h[point].loc[i, 'ffmc'] = firedanger.indices.ffmc(data12h[point].loc[i, 'temp'],
                                                    data12h[point].loc[i, 'prec24h'],
                                                    data12h[point].loc[i, 'wspd'],
                                                    data12h[point].loc[i, 'rhum'],
                                                    data12h[point].loc[i - 1, 'ffmc'])
                
                data12h[point].loc[i, 'dmc'] = firedanger.indices.dmc(data12h[point].loc[i, 'temp'],
                                                    data12h[point].loc[i, 'prec24h'],
                                                    data12h[point].loc[i, 'rhum'],
                                                    data12h[point].loc[i, 'creneau'].month,
                                                    point[0],
                                                    data12h[point].loc[i - 1, 'dmc'])
            else:
                data12h[point].loc[i, 'dc'] = data12h[point].loc[i - 1, 'dc']
                data12h[point].loc[i, 'ffmc'] = data12h[point].loc[i - 1, 'ffmc']
                data12h[point].loc[i, 'dmc'] = data12h[point].loc[i - 1, 'dmc']

                if data12h[point].loc[i, 'temp'] >= 12:
                    consecutive += 1
                else:
                    consecutive = 0

            data12h[point].loc[i, 'nesterov'] = firedanger.indices.nesterov(data15h.loc[i, 'temp'], data15h.loc[i, 'rhum'], data15h.loc[i, 'prec24h'],
                                                                   data12h[point].loc[i - 1, 'nesterov'])
            
            data12h[point].loc[i, 'munger'] = firedanger.indices.munger(data12h[point].loc[i, 'prec24h'], data12h[point].loc[i - 1, 'munger'])
            data12h[point].loc[i, 'days_since_rain'] = data12h[point].loc[i - 1, 'days_since_rain'] + 1 if (data12h[point].loc[i, 'prec24h'] < treshPrec24)  else 0
            data12h[point].loc[i, 'days_since_snow'] = data12h[point].loc[i - 1, 'days_since_snow'] + 1 if ( data12h[point].loc[i, 'snow24h'] < treshSnow24)  else 0
            
            creneau = data12h[point].loc[i, 'creneau'].date()
            
            data12h[point].loc[i, 'sum_consecutive_rainfall'] = data12h[point].loc[i, 'sum_consecutive_rainfall'].rolling(data12h[point].loc[i, 'days_since_rain']).sum()
            data12h[point].loc[i, 'sum_consecutive_snowfall'] = data12h[point].loc[i, 'sum_consecutive_snowfall'].rolling(data12h[point].loc[i, 'days_since_snow']).sum()

            data12h[point].loc[i, 'sum_rain_last_7_days'] = data12h[point].loc[i, 'sum_rain_last_7_days'].rolling('7D').sum()
            data12h[point].loc[i, 'sum_snow_last_7_days'] = data12h[point].loc[i, 'sum_snow_last_7_days'].rolling('7D').sum()

            preci = prec[prec.index == creneau].values[0]
            tmaxi = tmax[tmax.index == creneau].values[0]
            data12h[point].loc[i, 'kbdi'] = firedanger.indices.kbdi(tmaxi, preci, data12h[point].loc[i - 1, 'kbdi'],
                                                                    data12h[point].loc[i, 'sum_consecutive_rainfall'].rolling(data12h[point].loc[i, 'days_since_rain']).sum(),
                                                                    data12h[point].loc[i, 'sum_rain_last_7_days'], 30, 3.467326)
        
        data12h[point]['isi'] = data12h[point].apply(lambda x: firedanger.indices.isi(x.wspd, x.ffmc), axis=1)
        data12h[point]['angstroem'] = data12h[point].apply(lambda x: firedanger.indices.angstroem(x.temp, x.rhum), axis=1)
        data12h[point]['bui'] = firedanger.indices.bui(data12h[point]['dmc'], data12h[point]['dc'])
        data12h[point]['fwi'] = firedanger.indices.bui(data12h[point]['isi'], data12h[point]['bui'])
        data12h[point]['daily_severity_rating'] = data12h[point].apply(lambda x: firedanger.indices.daily_severity_rating(x.fwi), axis=1)

        var16h = ['temp','dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h', 'snow', 'snow24h']
        for v in var16h:
            data12h[point][f"{v}16"] = data16h[v].values

        for k in sorted(data12h[point].columns):
            if data12h[point][k].isna().sum() > MAX_NAN:
                print(f"{k} du point {point} possède trop de NaN ({data12h[point][k].isna().sum()}")
                if last_point is None:
                    assert False
                data12h[point][k] = last_point[k].values

            elif data12h[point][k].isna().sum() > 0:
                x = data12h[point][data12h[point][k].notna()].index
                y = data12h[point][data12h[point][k].notna()][k]

                f2 = interp1d(x, y, kind='linear', fill_value='extrapolate')
                
                newx = data12h[point][data12h[point][k].isna()].index
                newy = f2(newx)

                data12h[point].loc[newx, k] = newy

        data12h[point]['latitude'] = point[0]
        data12h[point]['longitude'] = point[1]
        res.append(data12h[point].reset_index())

    def get_date(x):
        return x.strftime('%Y-%m-%d')

    res = pd.concat(res)

    res['creneau'] =  res['creneau'].apply(get_date)
    
    return res

def check_and_create_path(path: Path):
    """
    Creer un dossier s'il n'existe pas
    """
    path_way = path.parent if path.is_file() else path

    path_way.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()

def create_grid_cems(df, time, delta, variables):
    interdate = dt.datetime.strptime(time, '%Y-%m-%d') - dt.timedelta(days=delta)
    interdate = interdate.strftime('%Y-%m-%d')
    dff = df[(df["creneau"] >= interdate) & (df["creneau"] <= time)]
    dff[variables] = dff[variables].astype(float)
    if len(dff) == 0:
        return None
    return dff.reset_index(drop=True)

def interpolate_gridd(var, grid, newx, newy, met, fill_value=0):
    x = grid['longitude'].values
    y = grid["latitude"].values
    points = np.zeros((y.shape[0], 2))
    points[:,0] = x
    points[:,1] = y
    return griddata(points, grid[var].values, (newx, newy), method=met, fill_value=0)

def create_dict_from_arry(array):
    res = {}
    for var in array:
        res[var] = 0
    return res

def myRasterization(geo, tif, maskNan, sh, column):

    res = np.full(sh, np.nan, dtype=float)
    
    if maskNan is not None:
        res[maskNan[:,0], maskNan[:,1]] = np.nan

    for index, row in geo.iterrows():
        #inds = indss[row['hex_id']]
        """ mask = np.zeros((sh[0], sh[1]))
        cv2.fillConvexPoly(mask, inds, 1)
        mask = mask > 0"""
        mask = tif == row['cluster']
        res[mask] = row[column]

    return res

def rasterise_meteo_data(h3, maskh3, cems, sh, dates, dir_output):
    cems_variables = [
                    'temp', 'dwpt',
                    'rhum', 'prcp', 'wdir', 'wspd', 'snow', 'prec24h', 'snow24h',
                    'dc',
                    'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                    'isi', 'angstroem', 'bui',
                    'fwi', 'daily_severity_rating',
                    'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16', 'snow24h16',
                    'days_since_rain', 'sum_consecutive_rainfall',
                    'sum_rain_last_7_days',
                    'sum_snow_last_7_days',
                    ]
    
    lenDates = len(dates)
    print(lenDates)
    for var in cems_variables:
        print(var)
        spatioTemporalRaster = np.full((sh[0], sh[1], lenDates), np.nan)

        for i, date in enumerate(dates):
            if i % 200 == 0:
                print(date)

            #ddate = dt.datetime.strptime(date, "%Y-%m-%d")
            cems_grid = create_grid_cems(cems, date, 0, var)
            if cems_grid is None:
                print('Cems is None')

            cems_grid.fillna(0, inplace=True)

            h3[var] = interpolate_gridd(var, cems_grid, h3.longitude.values, h3.latitude.values, 'cubic')
            h3[var].fillna(value=np.nanmean(h3[var]), inplace=True)
            
            h3[var] = [max(0, u) for u in  h3[var].values]

            rasterVar = myRasterization(h3, maskh3, None, maskh3.shape, var)

            if rasterVar.shape != sh:
                rasterVar = resize(rasterVar, sh[0], sh[1], 1)

            spatioTemporalRaster[:,:, i] = rasterVar
        if var == 'daily_severity_rating':
            var = 'dailySeverityRating'
        outputName = var+'raw.pkl'
        f = open(dir_output / outputName,"wb")
        pickle.dump(spatioTemporalRaster, f)

def rasterise_vigicrues(h3, maskh3, cems, sh, dates, dir_output, name):
    
    lenDates = len(dates)
    spatioTemporalRaster = np.full((sh[0], sh[1], lenDates), np.nan)

    for i, date in enumerate(dates):
        if i % 200 == 0:
            print(date)
        #ddate = dt.datetime.strptime(date, "%Y-%m-%d")
        cems_grid = create_grid_cems(cems, date, 0, 'hauteur')

        if cems_grid is None:
            continue

        #if len(cems_grid) < 4:
        #    continue

        cems_grid = cems_grid[cems_grid['hauteur'] >= 0]
        h3['hauteur'] = interpolate_gridd('hauteur', cems_grid, h3.longitude.values, h3.latitude.values, 'nearest', fill_value=0)

        rasterVar = myRasterization(h3, maskh3, None, maskh3.shape, 'hauteur')

        if rasterVar.shape != sh:
            rasterVar = resize(rasterVar, sh[0], sh[1], 1)

        spatioTemporalRaster[:,:, i] = rasterVar

        outputName = name+'.pkl'
        f = open(dir_output / outputName,"wb")
        pickle.dump(spatioTemporalRaster, f)

def rasterise_nappes(h3, maskh3, cems, sh, dates, dir_output, var):
    
    lenDates = len(dates)
    spatioTemporalRaster = np.full((sh[0], sh[1], lenDates), np.nan)

    for i, date in enumerate(dates):
        if i % 200 == 0:
            print(date)
        cems_grid = create_grid_cems(cems, date, 0, var)
        if cems_grid is None:
            continue
        cems_grid = cems_grid[cems_grid[var] >= 0]
        if len(cems_grid) == 0:
            continue
        h3[var] = interpolate_gridd(var, cems_grid, h3.longitude.values, h3.latitude.values, 'nearest', fill_value=0)

        rasterVar = myRasterization(h3, maskh3, None, maskh3.shape, var)

        if rasterVar.shape != sh:
            rasterVar = resize(rasterVar, sh[0], sh[1], 1)

        spatioTemporalRaster[:,:, i] = rasterVar

        outputName = var+'.pkl'
        f = open(dir_output / outputName,"wb")
        pickle.dump(spatioTemporalRaster, f)

def rasterise_air_qualite(h3, maskh3, air, sh, dates, dir_output):
    air_variables = ['NO2', 'O3', 'PM10', 'PM25',
                     'NO216', 'O316', 'PM1016', 'PM2516']
    
    lenDates = len(dates)
    print(lenDates)
    for var in air_variables:
        print(var)
        spatioTemporalRaster = np.full((sh[0], sh[1], lenDates), np.nan)

        for i, date in enumerate(dates):
            if i % 200 == 0:
                print(date)
            #ddate = dt.datetime.strptime(date, "%Y-%m-%d")
            air_grid = create_grid_cems(air, date, 0, var)
            if air_grid is None:
                continue
            air_grid.dropna(subset=var, inplace=True)
            if np.unique(air_grid[var].shape[0] != 0):
                #try:
                #    h3[var] = interpolate_gridd(var, air_grid, h3.longitude.values, h3.latitude.values, 'linear')
                #except:
                h3[var] = interpolate_gridd(var, air_grid, h3.longitude.values, h3.latitude.values, 'nearest', fill_value=0)

                rasterVar = myRasterization(h3, maskh3, None, maskh3.shape, var)

                if rasterVar.shape != sh:
                    rasterVar = resize(rasterVar, sh[0], sh[1], 1)

                spatioTemporalRaster[:,:, i] = rasterVar

        outputName = var+'raw.pkl'
        f = open(dir_output / outputName,"wb")
        pickle.dump(spatioTemporalRaster, f)

def find_dates_between(start, end):
    start_date = dt.datetime.strptime(start, '%Y-%m-%d').date()
    end_date = dt.datetime.strptime(end, '%Y-%m-%d').date()

    delta = dt.timedelta(days=1)
    date = start_date
    res = []
    while date < end_date:
            res.append(date.strftime("%Y-%m-%d"))
            date += delta
    return res

def get_hourly(x):
    return x.hour

def get_date(x):
    return x.date().strftime('%Y-%m-%d')


warnings.filterwarnings("ignore")


##################################################################################################
#                                       Spatial
##################################################################################################

def read_tif(name):
    """
    Open a satellite images and return bands, latitude and longitude of each pixel.
    """
    with rasterio.open(name) as src:
        dt = src.read()
        height = dt.shape[1]
        width = dt.shape[2]
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        lons = np.array(xs)
        lats = np.array(ys)
        src.close()
    return dt, lons, lats

def find_pixel(lats, lons, lat, lon):

    lonIndex = (np.abs(lons - lon)).argmin()
    latIndex = (np.abs(lats - lat)).argmin()

    lonValues = lons.reshape(-1,1)[lonIndex]
    latValues = lats.reshape(-1,1)[latIndex]
    #print(lonValues, latValues)
    return np.where((lons == lonValues) & (lats == latValues))

def resize(input_image, height, width, dim):
    """
    Resize the input_image into heigh, with, dim
    """
    img = img_as_float(input_image)
    img = transform.resize(img, (dim, height, width), mode='constant', order=0,
                 preserve_range=True, anti_aliasing=True)
    return np.asarray(img)

def resize_no_dim(input_image, height, width, mode='constant', order=0,
                 preserve_range=True, anti_aliasing=True):
    """
    Resize the input_image into heigh, with, dim
    """
    img = img_as_float(input_image)
    img = transform.resize(img, (height, width), mode=mode, order=order,
                 preserve_range=preserve_range, anti_aliasing=anti_aliasing)
    return np.asarray(img)

def create_geocube(df, variables, reslons, reslats):
    """
    Create a image representing variables with the corresponding resolution from df
    """
    geo_grid = make_geocube(
        vector_data=df,
        measurements=variables,
        resolution=(reslons, reslats),
        rasterize_function=rasterize_points_griddata,
        fill = 0
    )
    return geo_grid

def raster_population(tifFile, tifFile_high, dir_output, reslon, reslat, dir_data):
    population = pd.read_csv(dir_data / 'population' / 'population.csv')
    population = gpd.GeoDataFrame(population, geometry=gpd.points_from_xy(population.longitude, population.latitude))

    population = create_geocube(population, ['population'], -reslon, reslat)
    population = population.to_array().values[0]
    population = resize_no_dim(population, tifFile.shape[0], tifFile.shape[1])

    mask = np.argwhere(np.isnan(tifFile))
    population[mask[:,0], mask[:,1]] = np.nan

    outputName = 'population.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(population,f)

def raster_elevation(tifFile, dir_output, reslon, reslat, dir_data, dept):
    #elevation = gpd.read_file(dir_data / 'elevation' / 'elevation.geojson')
    elevation = pd.read_csv(dir_data / 'elevation' / 'elevation.csv')
    elevation['latitude'] = elevation['latitude'].apply(lambda x : round(x, 3))
    elevation['longitude'] = elevation['longitude'].apply(lambda x : round(x, 3))

    elevation = elevation.groupby(['longitude', 'latitude'], as_index=False)['altitude'].mean()

    try:
        elevation = gpd.GeoDataFrame(elevation, geometry=gpd.points_from_xy(elevation.longitude, elevation.latitude))
        elevation = rasterisation(elevation, reslat, reslon, 'altitude', defval=0, name=dept)
        elevation = resize_no_dim(elevation, tifFile.shape[0], tifFile.shape[1])
    except Exception as e:
        print(e)
        elevation = np.zeros(tifFile.shape)
    minusMask = np.argwhere(tifFile == -1)
    minusMask = np.argwhere(np.isnan(tifFile))
    elevation[minusMask[:,0], minusMask[:,1]] = np.nan
    outputName = 'elevation.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(elevation,f)

valeurs_foret_attribut = {
    "NoForest": 0,
    "Châtaignier": 1,
    "Chênes décidus": 2,
    "Chênes sempervirents": 3,
    "Conifères": 4,
    "Douglas": 5,
    "Feuillus": 6,
    "Hêtre": 7,
    "Mélèze": 8,
    "Mixte": 9,
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

valeurs_cosia_couverture = {
    'Cosia_Building': 1,
    'Cosia_Bare soil': 2,
    'Cosia_Water surface': 3,
    'Cosia_Conifer': 4,
    'Cosia_Deciduous': 5,
    'Cosia_Shrubland': 6,
    'Cosia_Lawn': 7,
    'Cosia_Crop': 8,
}

valeurs_tourbiere_couverture = {
    'Marais': 1,
    'Haute': 2,
}

def arrondir_avec_seuil(array, seuil):
    """
    Arrondit les éléments d'un tableau NumPy en fonction d'un seuil décimal.
    Si la partie décimale est supérieure ou égale au seuil, l'élément est arrondi au supérieur.
    Sinon, il est arrondi à l'inférieur.

    :param array: Le tableau NumPy à arrondir.
    :param seuil: Le seuil décimal pour l'arrondi (entre 0 et 1).
    :return: Le tableau NumPy arrondi en fonction du seuil.
    """
    # Séparer la partie entière et la partie décimale
    partie_entière = np.floor(array)
    partie_décimale = array - partie_entière

    # Condition pour arrondir au supérieur
    arrondir_au_sup = partie_décimale >= seuil

    # Ajouter 1 à la partie entière là où on doit arrondir au supérieur
    partie_entière[arrondir_au_sup] += 1

    # Retourner la partie entière comme résultat arrondi
    return partie_entière

def raster_argile(tifFile, tifFile_high, dir_output, reslon, reslat, dir_data, dept):
    argile = gpd.read_file(dir_data / 'argile' / 'argile.geojson')
    #argile = create_geocube(argile, ['NIVEAU'], -reslon, reslat)
    try:
        argile = rasterisation(argile, reslat, reslon, 'NIVEAU', defval=0, name=dept)
    except:
        argile = np.zeros(tifFile.shape)
    argile = resize_no_dim(argile, tifFile.shape[0], tifFile.shape[1])
    argile = arrondir_avec_seuil(argile, 0.2)
    argile[np.isnan(tifFile)] = np.nan

    outputName = 'argile.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(argile,f)

def raster_tourbiere(tifFile, tifFile_high, dir_output, reslon, reslat, dir_data, region):
    ZonesTourbeuses = gpd.read_file(dir_data / 'tourbiere' / 'ZonesTourbeuses.geojson')

    ZonesTourbeuses = rasterisation(ZonesTourbeuses, reslat, reslon, 'classe', defval=0, name=dept)
    ZonesTourbeuses = resize_no_dim(ZonesTourbeuses, tifFile_high.shape[0], tifFile_high.shape[1])
    bands = valeurs_tourbiere_couverture.values()
    bands = np.asarray(list(bands))
    res = np.full((np.max(bands) + 1, tifFile.shape[0], tifFile.shape[1]), fill_value=0.0)
    res2 = np.full((tifFile.shape[0], tifFile.shape[1]), fill_value=np.nan)
    res3 = np.full((np.max(bands) + 1, tifFile.shape[0], tifFile.shape[1]), fill_value=0.0)

    unodes = np.unique(tifFile)
    ZonesTourbeuses_2 = np.empty((np.max(bands) + 1,*ZonesTourbeuses.shape))
    
    for band in bands:
        ZonesTourbeuses_2[band] = influence_index(ZonesTourbeuses == band, np.isnan(ZonesTourbeuses))

    for node in unodes:
        if node not in tifFile_high:
            continue

        mask1 = tifFile == node
        mask2 = tifFile_high == node

        for band in bands:
            res[band, mask1] = (np.argwhere(ZonesTourbeuses[mask2] == band).shape[0] / ZonesTourbeuses[mask2].shape[0]) * 100
            res3[band, mask1] = np.nanmean(ZonesTourbeuses_2[band, mask2])

        if res[:, mask1].shape[1] == 1:
            res2[mask1] = np.nanargmax(res[:, mask1])
        else:
            res2[mask1] = np.nanargmax(res[:, mask1][:,0])

    res[:, np.isnan(tifFile)] = np.nan
    res2[np.isnan(tifFile)] = np.nan
    res3[:, np.isnan(tifFile)] = np.nan
    
    outputName = 'ZonesTourbeuses.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res,f)

    outputName = 'ZonesTourbeuses_landcover.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res2,f)

    outputName = 'ZonesTourbeuses_influence.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res3,f)


def raster_cosia(tifFile, tifFile_high, dir_output, reslon, reslat, dir_data, region):
    cosia = gpd.read_file(dir_data / 'cosia' / 'cosia.geojson')
    cosia = gpd.overlay(region, cosia)
    # Modify the classes
    cosia = cosia[(cosia['numero'] != 7)]
    cosia.loc[cosia[cosia['numero'].isin([18, 2, 4])].index, 'numero'] = 1
    cosia.loc[cosia[cosia['numero'].isin([3, 5])].index, 'numero'] = 2
    cosia.loc[cosia[cosia['numero'].isin([6, 7])].index, 'numero'] = 3
    cosia.loc[cosia[cosia['numero'] == 8].index, 'numero'] = 4
    cosia.loc[cosia[cosia['numero'] == 10].index, 'numero'] = 5
    cosia.loc[cosia[cosia['numero'] == 9].index, 'numero'] = 6
    cosia.loc[cosia[cosia['numero'] == 15].index, 'numero'] = 7
    cosia.loc[cosia[cosia['numero'].isin([14, 17, 16])].index, 'numero'] = 8

    cosia = rasterisation(cosia, reslat, reslon, 'numero', defval=0, name=dept)
    cosia = resize_no_dim(cosia, tifFile_high.shape[0], tifFile_high.shape[1])
    bands = valeurs_cosia_couverture.values()
    bands = np.asarray(list(bands))
    res = np.full((np.max(bands) + 1, tifFile.shape[0], tifFile.shape[1]), fill_value=0.0)
    res2 = np.full((tifFile.shape[0], tifFile.shape[1]), fill_value=np.nan)
    res3 = np.full((np.max(bands) + 1, tifFile.shape[0], tifFile.shape[1]), fill_value=0.0)

    unodes = np.unique(tifFile)
    cosia_2 = np.empty((np.max(bands) + 1,*cosia.shape))
    
    for band in bands:
        cosia_2[band] = influence_index(cosia == band, np.isnan(cosia))

    for node in unodes:
        if node not in tifFile_high:
            continue

        mask1 = tifFile == node
        mask2 = tifFile_high == node

        for band in bands:
            res[band, mask1] = (np.argwhere(cosia[mask2] == band).shape[0] / cosia[mask2].shape[0]) * 100
            res3[band, mask1] = np.nanmean(cosia_2[band, mask2])

        if res[:, mask1].shape[1] == 1:
            res2[mask1] = np.nanargmax(res[:, mask1])
        else:
            res2[mask1] = np.nanargmax(res[:, mask1][:,0])

    res[:, np.isnan(tifFile)] = np.nan
    res2[np.isnan(tifFile)] = np.nan
    res3[:, np.isnan(tifFile)] = np.nan
    
    outputName = 'cosia.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res,f)

    outputName = 'cosia_landcover.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res2,f)

    outputName = 'cosia_influence.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res3,f)

def raster_foret(tifFile, tifFile_high, dir_output, reslon, reslat, dir_data, dept):
    foret = gpd.read_file(dir_data / 'BDFORET' / 'foret.geojson')
    foret = rasterisation(foret, reslat, reslon, 'code', defval=0, name=dept)
    foret = resize_no_dim(foret, tifFile_high.shape[0], tifFile_high.shape[1])
    bands = valeurs_foret_attribut.values()
    bands = np.asarray(list(bands))
    res = np.full((np.max(bands) + 1, tifFile.shape[0], tifFile.shape[1]), fill_value=0.0)
    res2 = np.full((tifFile.shape[0], tifFile.shape[1]), fill_value=np.nan)
    res3 = np.full((np.max(bands) + 1, tifFile.shape[0], tifFile.shape[1]), fill_value=0.0)

    unodes = np.unique(tifFile)
    foret_2 = np.empty((np.max(bands) + 1,*foret.shape))

    for band in bands:
        foret_2[band] = influence_index(foret == band, np.isnan(foret))

    for node in unodes:
        if node not in tifFile_high:
            continue

        mask1 = tifFile == node
        mask2 = tifFile_high == node

        for band in bands:
            res[band, mask1] = (np.argwhere(foret[mask2] == band).shape[0] / foret[mask2].shape[0]) * 100
            res3[band, mask1] = np.nanmean(foret_2[band, mask2])

        if res[:, mask1].shape[1] == 1:
            res2[mask1] = np.nanargmax(res[:, mask1])
        else:
            res2[mask1] = np.nanargmax(res[:, mask1][:,0])

    res[:, np.isnan(tifFile)] = np.nan
    res2[np.isnan(tifFile)] = np.nan
    res3[:, np.isnan(tifFile)] = np.nan

    outputName = 'foret.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res,f)

    outputName = 'foret_landcover.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res2,f)

    outputName = 'foret_influence.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res3,f)

def raster_osmnx(tifFile, tifFile_high, dir_output, reslon, reslat, dir_data, dept):
    #osmnx, _, _ = read_tif(dir_data / 'osmnx' / 'osmnx.tif')
    osmnx = gpd.read_file(dir_data / 'osmnx' / 'osmnx.geojson')
    osmnx = rasterisation(osmnx, reslat, reslon, 'label', defval=0, name=dept)

    mask = np.isnan(tifFile_high)
    osmnx = resize_no_dim(osmnx, tifFile_high.shape[0], tifFile_high.shape[1])
    osmnx[mask] = np.nan
    bands = np.asarray([0,1,2,3,4,5])
    bands = bands[~np.isnan(bands)].astype(int)
    res = np.zeros(((np.nanmax(bands) + 1), tifFile.shape[0], tifFile.shape[1]), dtype=float)
    res2 = np.full((tifFile.shape[0], tifFile.shape[1]), fill_value=np.nan)
    res3 = np.full(((np.nanmax(bands) + 1), tifFile.shape[0], tifFile.shape[1]), fill_value=0.0, dtype=float)
    res4 = np.zeros((tifFile.shape[0], tifFile.shape[1]), dtype=float)
    
    osmnx_2 = np.empty(((np.nanmax(bands) + 1), *osmnx.shape))

    for band in bands:
        osmnx_2[band] = influence_index(osmnx == band, mask)

    unodes = np.unique(tifFile)
    for node in unodes:
        mask1 = tifFile == node
        mask2 = tifFile_high == node

        if True not in np.unique(mask2):
            continue
        
        for band in bands:
            res[band, mask1] = (np.argwhere(osmnx[mask2] == band).shape[0] / osmnx[mask2].shape[0]) * 100
            res3[band, mask1] = np.nanmean(osmnx_2[band, mask2])
            if band > 0:
                res4[mask1] = res4[mask1] + res[band, mask1]

        if res[:, mask1].shape[1] == 1:
            res2[mask1] = np.nanargmax(res[:, mask1])
        else:
            res2[mask1] = np.nanargmax(res[:, mask1][:,0])

    res[:, np.isnan(tifFile)] = np.nan
    res2[np.isnan(tifFile)] = np.nan
    res3[:, np.isnan(tifFile)] = np.nan
    res4[np.isnan(tifFile)] = np.nan

    outputName = 'osmnx.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res,f)

    outputName = 'osmnx_landcover.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res2,f)

    outputName = 'osmnx_influence.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res3,f)

    outputName = 'osmnx_landcover_2.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res4,f)

def raster_sat(base, dir_reg, dir_output, dates):
    size = '30m'
    dir_sat = dir_reg / 'GEE' / size
    res = np.full((5, base.shape[0], base.shape[1],  len(dates)), np.nan)
    minusMask = np.argwhere(np.isnan(base))
    
    for tifFile in dir_sat.glob('sentinel/*.tif'):
        
        tifFile = tifFile.as_posix()
        dateFile = tifFile.split('/')[-1]
        date = dateFile.split('.')[0]

        if date not in dates:
            continue
        
        i = dates.index(date)
        print(dateFile, i)

        sentinel, _, _ = read_tif(tifFile)
        
        for band in range(sentinel.shape[0]):
            values = resize_no_dim(sentinel[band], base.shape[0], base.shape[1])
            values[np.isnan(values)] = np.nanmean(values)
            if i + 1 > res.shape[3]:
                lmin = i - 14 
                lmax = res.shape[3]
            elif i == 0:
                lmin = 0
                lmax = i + 1
            else:
                lmin = i - 14
                lmax = i + 1

            k = lmax - lmin
            res[band,:,:, lmin:lmax] = np.repeat(values[:,:, np.newaxis], k, axis=2)

            res[:, minusMask[:,0], minusMask[:,1], :] = np.nan

    
    res[:, minusMask[:,0], minusMask[:,1], :] = np.nan

    outputName = 'sentinel.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res,f)
    res = np.full((base.shape[0], base.shape[1], len(dates)), np.nan)

from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from shapely.geometry import Point
import rasterio
from rasterio.mask import mask

def raster_sat_from_france(base, geo, dir_output, dir_france, dates):
    size = '30m'
    res = np.full((5, base.shape[0], base.shape[1], len(dates)), np.nan)
    minusMask = np.argwhere(np.isnan(base))
    
    polygons = unary_union(geo.geometry) 

    for tifFile in dir_france.glob('*.tif'):
        tifFile = tifFile.as_posix()
        dateFile = tifFile.split('/')[-1]
        date = dateFile.split('.')[0]

        if date not in dates:
            continue

        i = dates.index(date)
        print(dateFile, i)

        with rasterio.open(tifFile) as src:
            # Masquage par polygone
            out_image, out_transform = mask(src, [polygons], crop=True, nodata=np.nan)
            out_image = out_image.astype(np.float32)
            out_image[out_image == src.nodata] = np.nan

            # Resize chaque bande à la forme de `base`
            for b in range(out_image.shape[0]):
                # Crée un tableau vide pour le résultat interpolé
                target = np.full(base.shape, np.nan, dtype=np.float32)

                #plt.imshow(out_image[b])
                #plt.show()

                """reproject(
                    source=out_image[b],
                    destination=target,
                    src_transform=out_transform,
                    src_crs=src.crs,
                    dst_transform=from_origin(0, 0, 1, 1),  # Remplacer si nécessaire
                    dst_crs=src.crs,
                    resampling=Resampling.nearest
                )"""

                target = resize_no_dim(out_image[b], base.shape[0], base.shape[1])
                #target[np.isnan(base)] = np.nan
                #plt.imshow(target)
                #plt.show()

                if i + 8 > res.shape[-1]:
                    length = res.shape[-1] - i - 1
                    target = np.repeat(target[:, :, np.newaxis], length, axis=-1)
                    res[b, :, :, i:-1] = target
                else:
                    target = np.repeat(target[:, :, np.newaxis], 8, axis=-1)
                    res[b, :, :, i:i+8] = target

    # Masque les pixels NaN d'origine
    res[:, minusMask[:, 0], minusMask[:, 1], :] = np.nan

    print(dir_output)
    outputName = 'sentinel.pkl'
    with open(dir_output / outputName, "wb") as f:
        pickle.dump(res, f)

    # Optionnel : une image moyenne ou indice NDVI par date
    return res

def raster_land(tifFile, tifFile_high, dir_reg, dir_output, dates):

    size = '30m'
    dir_sat = dir_reg / 'GEE' / size / 'landcover_season'
    bands = [0,1,2,3,4,5,6,7,8]
    res = np.full((np.max(bands) + 1, tifFile.shape[0], tifFile.shape[1], len(dates)), fill_value=0.0)
    res2 = np.full((tifFile.shape[0], tifFile.shape[1]), fill_value=np.nan)
    res3 = np.full((np.max(bands) + 1, tifFile.shape[0], tifFile.shape[1], len(dates)), fill_value=np.nan)
    unodes = np.unique(tifFile)
    minusMask = np.argwhere(np.isnan(tifFile))

    season = {
    'summer' : ('06-01', '08-31'),
    'winter' : ('12-01', '02-28'),
    'autumn' : ('09-01', '11-30'),
    'spring' : ('03-01', '05-31')
    }

    years = ['2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']

    for year in years:
        print(year)
        for seas, item in season.items():
            print(seas)
            if year == '2024' and (seas == 'winter' or seas == 'autumn'):
                continue
            name = seas+'_'+year+'.tif'
            dynamicWorld, _, _ = read_tif(dir_sat / name)
            dynamicWorld = dynamicWorld[0]
            dynamicWorld = resize_no_dim(dynamicWorld, tifFile_high.shape[0], tifFile_high.shape[1])
            dynamicWorld_2 = np.empty((len(bands),*dynamicWorld.shape))

            for band in bands:
                dynamicWorld_2[band] = influence_index(dynamicWorld == band, np.isnan(dynamicWorld)) 
            
            date_min = year+'-'+item[0]
            date_max = year+'-'+item[1]
            if date_min in dates:
                index_min = dates.index(date_min)
            else:
                index_min = 0
            if date_max in dates:
                index_max = dates.index(date_max)
            else:
                index_max = -1

            for node in unodes:
                if node not in tifFile_high:
                    continue
                mask1 = tifFile == node
                mask2 = tifFile_high == node
                if True not in np.unique(mask2):
                    continue
                for band in bands:
                    res[band, mask1, index_min:index_max] = (np.argwhere(dynamicWorld[mask2] == band).shape[0] / dynamicWorld[mask2].shape[0]) * 100
                    res3[band, mask1] = np.nanmean(dynamicWorld_2[band, mask2])
                try:
                    if res[:, mask1, index_min:index_max].shape[1] == 1:
                        res2[mask1] = np.nanargmax(res[:, mask1, index_min:index_max])
                    else:
                        res2[mask1] = np.nanargmax(res[:, mask1, index_min:index_max][:,0])
                except:
                    continue

    res[:, np.isnan(tifFile)] = np.nan
    res2[np.isnan(tifFile)] = np.nan
    res3[:, np.isnan(tifFile)] = np.nan

    outputName = 'dynamic_world.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res,f)

    outputName = 'dynamic_world_landcover.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res2,f)

    outputName = 'dynamic_world_influence.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res3,f)

def rasterisation(h3, lats, longs, column='cluster', defval = 0, name='default', dir_output='/media/caron/X9 Pro1/corbeille', return_lat_lon=False):
    #h3['cluster'] = h3.index

    h3.to_file(dir_output + '/' + name+'.geojson', driver='GeoJSON')

    input_geojson = dir_output + '/' + name+'.geojson'
    output_raster = dir_output + '/' + name+'.tif'

    # Si on veut rasteriser en fonction de la valeur d'un attribut du vecteur, mettre son nom ici 
    attribute_name = column

    # Taille des pixels
    if isinstance(lats, float):
        pixel_size_y = lats
        pixel_size_x = longs
    else:
        pixel_size_x = abs(longs[0][0] - longs[0][1])
        pixel_size_y = abs(lats[0][0] - lats[1][0])
    print(f'px {pixel_size_x}, py {pixel_size_y}')
    #pixel_size_x = res[dim][0]
    #pixel_size_y = res[dim][1]

    source_ds = ogr.Open(input_geojson)
    source_layer = source_ds.GetLayer()

    # On obtient l'étendue du raster
    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    # On calcule le nombre de pixels
    width = int((x_max - x_min) / pixel_size_x)
    height = int((y_max - y_min) / pixel_size_y)

    # Oncrée un nouveau raster dataset et on passe de "coordonnées image" (pixels) à des coordonnées goréférencées
    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(output_raster, width, height, 1, gdal.GDT_Float32)
    output_ds.GetRasterBand(1).Fill(defval)
    output_ds.SetGeoTransform([x_min, pixel_size_x, 0, y_max, 0, -pixel_size_y])
    output_ds.SetProjection(source_layer.GetSpatialRef().ExportToWkt())

    if attribute_name != '' :
        # On  rasterise en fonction de l'attribut donné
        gdal.RasterizeLayer(output_ds, [1], source_layer, options=["ATTRIBUTE=" + attribute_name])
    else :
        # On  rasterise. Le raster prend la valeur 1 là où il y a un vecteur
        gdal.RasterizeLayer(output_ds, [1], source_layer)

    output_ds = None
    source_ds = None

    if not return_lat_lon:
        res, _, _ = read_tif(dir_output + '/' + name+'.tif')
        os.remove(dir_output + '/' + name+'.tif')
        return res[0]
    else:
        return read_tif(dir_output + '/' + name+'.tif')

def reclass_corine_by_index(array):
    """
    Reclasse un raster CORINE à base d'indices (1 à 44) vers 10 classes logiques.
    Le mapping est basé sur la position des indices dans l'ordre CORINE officiel.

    Entrée :
        array : np.ndarray contenant les indices de 1 à 44 (ex : [1, 2, ..., 44])
    
    Sortie :
        np.ndarray avec valeurs de 1 à 10 représentant les classes logiques
    """

    # Table des 10 classes (indexé de 0 à 43 → 44 valeurs)
    # Cette table correspond à l'ordre des indices dans ton raster
    mapping_10_classes = np.array([
        1, 1, 1, 1, 1, 1,        # 1–6 urbain
        2, 2, 1, 1,              # 7–10 (transport, ZI, urbain diffus)
        3, 3, 3, 3, 3, 3, 3,     # 11–17 agriculture
        4,                      # 18 prairie
        3, 3, 3, 3,              # 19–22 agriculture
        5, 5, 5,                # 23–25 forêts
        6, 6, 6, 6,             # 26–29 végétation naturelle
        10, 10, 10, 10, 10,     # 30–34 roche, minéral, etc.
        7, 7,                  # 35–36 zones humides
        9, 9, 9,               # 37–39 littoral
        8, 8,                  # 40–41 plans d’eau
        10, 10, 10             # 42–44 surfaces peu végétalisées / neige
    ])

    # Préparer un tableau LUT de 256 pour supporter tout uint8
    lut = np.zeros(256, dtype=np.uint8)
    lut[1:45] = mapping_10_classes  # les indices valides sont 1–44

    # Remplace les valeurs invalides (-128 ou 0) par 0
    array_clean = np.where((array >= 1) & (array <= 44), array, 0)
    reclassified = lut[array_clean]

    return reclassified

def load_shp_from_dir(subpath):
    gdfs = []
    # Parcours de tous les fichiers .shp
    shp_files = chain(
    subpath.glob("*.shp"),
    subpath.glob("*.SHP")
    )

    for shp_file in shp_files:
        try:
            gdf = gpd.read_file(shp_file)
            gdfs.append(gdf)
        except Exception as e:
            print(f"Erreur lors de la lecture de {shp_file.name}: {e}")

    # Concaténation finale
    if gdfs:
        gdf_concat = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    else:
        print("Aucun shapefile trouvé ou tous les fichiers ont échoué à la lecture.")

    return gdf_concat

def raster_corine(geo, dir_output, file_subpaths, tifFile, tifFile_high, years, dates):
    """
    Pour chaque TIFF dans dir_france :
      - masque selon la géométrie 'geo'
      - reprojette à la résolution donnée
      - convertit chaque pixel en couleur RGB (i, i, i)
      - exporte le résultat comme GIF couleur
    """
    dir_output = Path(dir_output)
    dir_output.mkdir(parents=True, exist_ok=True)

    unodes = np.unique(tifFile[~np.isnan(tifFile)])

    bands = np.arange(0, 10)

    res = np.empty(tifFile.shape)
    res2 = np.copy(res)
    res3 = np.copy(res2)

    height, width = tifFile_high[0].shape

    for i, file_path in enumerate(file_subpaths):

        year = years[i]
        
        if i == len(years):
            index_max = -1
        else:
            
            date_min = f'{year}-01-01'
            date_max = f'{years[i + 1]}-01-01'

            if date_min < dates[0]:
                date_min = dates[0]
            if date_max > dates[-1]:
                date_max = dates[-1]

            index_min = dates.index(date_min)
            index_max = dates.index(date_max)

        for tif_file in Path(file_path).glob("*.tif"):
            print(f"📂 Traitement de {tif_file.name}")

            with rasterio.open(tif_file) as src:
                src_crs = src.crs
                target_crs = "EPSG:4326"  # CRS géographique (degrés)

                # Reprojeter la géométrie dans le bon CRS
                if geo.crs != src_crs:
                    geo_proj = geo.to_crs(src_crs)
                else:
                    geo_proj = geo
                
                #src_bounds = src_bounds.to_crs(src_crs)

                polygons = unary_union(geo_proj.geometry)
                # Masquage spatial
                out_image, out_transform = mask(src, [polygons], crop=True, nodata=0)
                
                # Calcul de la résolution cible
                dst_transform, width, height = calculate_default_transform(
                    src.crs, target_crs,
                    out_image.shape[1], out_image.shape[0],
                    *polygons.bounds,
                    dst_width=width, dst_height=height,
                    #resolution=(reslon, reslat)  # ICI en degrés
                )
                
                # Destination reprojetée (1 bande attendue)
                dst_array = np.empty((1, height, width), dtype=np.float32)

                reproject(
                    source=out_image,  # on suppose 1 bande représentant les classes
                    destination=dst_array[0],
                    src_transform=out_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )

                print(np.unique(dst_array), np.unique(out_image))
                
                # Reclassification 1–44 → 1–10
                classified = reclass_corine_by_index(dst_array[0].astype(np.uint8))

                for node in unodes:
                    if node not in tifFile_high:
                        continue
                    mask1 = tifFile == node
                    mask2 = tifFile_high == node
                    if True not in np.unique(mask2):
                        continue
                    for band in bands:
                        res[band, mask1, index_min:index_max] = (np.argwhere(classified[mask2] == band).shape[0] / classified[mask2].shape[0]) * 100
                        res3[band, mask1] = np.nanmean(classified[band, mask2])
                    try:
                        if res[:, mask1, index_min:index_max].shape[1] == 1:
                            res2[mask1] = np.nanargmax(res[:, mask1, index_min:index_max])
                        else:
                            res2[mask1] = np.nanargmax(res[:, mask1, index_min:index_max][:,0])
                    except:
                        continue

    res[:, np.isnan(tifFile)] = np.nan
    res2[np.isnan(tifFile)] = np.nan
    res3[:, np.isnan(tifFile)] = np.nan

    outputName = 'corine.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res,f)

    outputName = 'corine_landcover.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res2,f)

    outputName = 'corine_influence.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res3,f)

def raster_route(dir_output, tifFile, tifFile_high, reslon, reslat, file_paths, france, dates, years):
    
    res = np.empty(tifFile.shape)
    res2 = np.copy(res)
    res3 = np.copy(res2)

    unodes = np.unique(tifFile[~np.isnan(tifFile)])

    bands = [0]

    for i, file_path in file_paths:
        
        if i == len(years):
            index_max = -1
        else:
            
            date_min = f'{year}-01-01'
            date_max = f'{years[i + 1]}-01-01'

            if date_min < dates[0]:
                date_min = dates[0]
            if date_max > dates[-1]:
                date_max = dates[-1]

            index_min = dates.index(date_min)
            index_max = dates.index(date_max)

        osmnx = load_shp_from_dir(file_path)
        #osmnx = osmnx[osmnx['NB_VOIES'].isin(['1 voie ou 2 voies étroites', '2 voies larges', '3 voies', '4 voies', 'Plus de 4 voies'])].copy()
        osmnx = osmnx[osmnx['NB_VOIES'].isin(['2 voies larges', '3 voies', '4 voies', 'Plus de 4 voies'])].copy()
        #osmnx = osmnx[osmnx['NB_VOIES'].isin(['3 voies', '4 voies', 'Plus de 4 voies'])].copy()
        #osmnx = osmnx[osmnx['NB_VOIES'].isin(['4 voies', 'Plus de 4 voies'])].copy()

        osmnx['label'] = 1  # utile uniquement pour rasterisation ultérieure
        print(f"Lignes routières filtrées : {osmnx.shape}")
        
        # Harmoniser CRS
        if osmnx.crs != france.crs:
            france = france.to_crs(osmnx.crs)

        # -- SPATIAL JOIN : marquer les polygones qui intersectent au moins une route --
        france = france.copy()
        france['label'] = 0  # initialisation

        # Utiliser sjoin pour récupérer les intersections
        intersections = gpd.sjoin(france, osmnx, how='left', predicate='intersects')

        # Récupérer les index de polygones qui intersectent des lignes
        matched_idx = intersections[~intersections.index_right.isna()].index.unique()

        # Mettre leur label à 1
        france.loc[matched_idx, 'label'] = 1
        
        france = france.to_crs("EPSG:4326")

        print(f"Polygones labellisés : {france['label'].sum()} sur {len(france)}")

        # Rasterisation sur les polygones maintenant labellisés
        image = rasterisation(france, reslat, reslon, 'label', dir_output=dir_output, name='route')

        for node in unodes:
            if node not in tifFile_high:
                continue
            mask1 = tifFile == node
            mask2 = tifFile_high == node
            if True not in np.unique(mask2):
                continue
            for band in bands:
                res[band, mask1, index_min:index_max] = (np.argwhere(image[mask2] == band).shape[0] / image[mask2].shape[0]) * 100
                res3[band, mask1] = np.nanmean(image[band, mask2])
            try:
                if res[:, mask1, index_min:index_max].shape[1] == 1:
                    res2[mask1] = np.nanargmax(res[:, mask1, index_min:index_max])
                else:
                    res2[mask1] = np.nanargmax(res[:, mask1, index_min:index_max][:,0])
            except:
                continue

        res[:, np.isnan(tifFile)] = np.nan
        res2[np.isnan(tifFile)] = np.nan
        res3[:, np.isnan(tifFile)] = np.nan

        outputName = 'route.pkl'
        f = open(dir_output / outputName,"wb")
        pickle.dump(res,f)

        outputName = 'route_landcover.pkl'
        f = open(dir_output / outputName,"wb")
        pickle.dump(res2,f)

        outputName = 'route_influence.pkl'
        f = open(dir_output / outputName,"wb")
        pickle.dump(res3,f)
    
def myFunctionDistanceDugrandCercle(outputShape, earth_radius=6371.0, resolution_lon=0.0002694945852352859, resolution_lat=0.0002694945852326214):
    half_rows = outputShape[0] // 2
    half_cols = outputShape[1] // 2

    # Créer une grille de coordonnées géographiques avec les résolutions souhaitées
    latitudes = np.linspace(-half_rows * resolution_lat, half_rows * resolution_lat, outputShape[0])
    longitudes = np.linspace(-half_cols * resolution_lon, half_cols * resolution_lon, outputShape[1])
    latitudes, longitudes = np.meshgrid(latitudes, longitudes, indexing='ij')

    # Coordonnées du point central
    center_lat = latitudes[outputShape[0] // 2, outputShape[1] // 2]
    center_lon = longitudes[outputShape[0] // 2, outputShape[1] // 2]

    # Convertir les coordonnées géographiques en radians
    latitudes_rad = np.radians(latitudes)
    longitudes_rad = np.radians(longitudes)

    # Calculer la distance du grand cercle entre chaque point et le point central
    delta_lon = longitudes_rad - np.radians(center_lon)
    delta_lat = latitudes_rad - np.radians(center_lat)
    a = np.sin(delta_lat/2)**2 + np.cos(latitudes_rad) * np.cos(np.radians(center_lat)) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distances = earth_radius * c
    
    return distances

def influence_index(categorical_array, mask):
    res = np.full(categorical_array.shape, np.nan)

    kernel = myFunctionDistanceDugrandCercle((30,30))
    #kernel = normalize(kernel, norm='l2')
    #res[mask[:,0], mask[:,1]] = (scipy_fft_conv(categorical_array, kernel, mode='same')[mask[:,0], mask[:,1]])

    res = convolve_fft(categorical_array, kernel, normalize_kernel=True, mask=mask)

    return res

def add_osmnx(clusterSum, dataset, dir_reg, kmeans, tifFile):
    print('Add osmnx')
    dir_data = dir_reg / 'osmnx'

    tif, _, _ = read_tif(dir_data / 'osmnx.tif')
    
    sentinel, lons, lats = read_tif(tifFile)
    nanmask = np.argwhere(np.isnan(sentinel[0]))
    sentinel = sentinel[0]

    coord = np.empty((sentinel.shape[0], sentinel.shape[1], 2))
    tif = resize(tif, sentinel.shape[0], sentinel.shape[1], 1).astype(int)[0]
    coord[:,:,0] = lats
    coord[:,:,1] = lons

    clustered = kmeans.predict(np.reshape(coord, (-1, coord.shape[-1])))
    clustered = clustered.reshape(sentinel.shape[0], sentinel.shape[1]).astype(float)
    clustered[nanmask[:,0], nanmask[:,1]] = np.nan

    mask = np.argwhere(~np.isnan(tif))
    density = influence_index(tif.astype(int), mask)

    dataset['highway_min'] = 0
    dataset['highway_max'] = 0
    dataset['highway_std'] = 0
    dataset['highway_mean'] = 0
    dataset['highway_sum'] = 0

    for cluster in clusterSum['cluster'].unique():
        clusterMask = clustered == cluster
        indexDataset = dataset[dataset['cluster'] == cluster].index.values

        dataset.loc[indexDataset, 'highway_sum'] = np.nansum(density[clusterMask])

        dataset.loc[indexDataset, 'highway_max'] = np.nanmax(density[clusterMask])

        dataset.loc[indexDataset, 'highway_min'] = np.nanmin(density[clusterMask])

        dataset.loc[indexDataset, 'highway_mean'] = np.nanmean(density[clusterMask])

        dataset.loc[indexDataset, 'highway_std'] =  np.nanstd(density[clusterMask])
    
    return dataset

def read_object(filename: str, path : Path):
    if not (path / filename).is_file():
        print(f'{path / filename} not found')
        return None
    return pickle.load(open(path / filename, 'rb'))


############################### Raster loaders #################################

def _expand_static(data: np.ndarray, n_dates: int) -> np.ndarray:
    """Expand a static raster to have a date dimension."""
    if data.ndim == 2:
        return np.repeat(data[..., np.newaxis], n_dates, axis=2)
    if data.ndim == 3:
        return np.repeat(data[..., np.newaxis], n_dates, axis=3)
    return data

def load_rasterise_meteo(dir_raster: Path, dates: list, lat, lon) -> xr.Dataset:
    """Load meteo rasters produced by ``rasterise_meteo_data`` into an xarray."""
    cems_variables = [
                'temp', 'dwpt',
                'rhum', 'prcp', 'wdir', 'wspd', 'snow', 'prec24h', 'snow24h',
                'dc',
                'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                'isi', 'angstroem', 'bui',
                'fwi', 'daily_severity_rating',
                'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16', 'snow24h16',
                'days_since_rain', 'sum_consecutive_rainfall',
                'sum_rain_last_7_days',
                'sum_snow_last_7_days',
                ]
    data_vars = {}
    
    for var in cems_variables:
        if var == 'daily_severity_rating':
            var = 'dailySeverityRating'
        file = f'{var}raw.pkl'
        with open(dir_raster / file, "rb") as f:
            values = pickle.load(f)
        data_vars[var] = (("latitude", "longitude", "date"), values)

    coords = {"latitude": lat, "longitude": lon, "date": dates}
    return xr.Dataset(data_vars, coords=coords)

def load_raster_cosia(dir_raster: Path, dates: list, lat, lon) -> xr.Dataset:
    """Load COSIA rasters and broadcast them on the date dimension."""
    cosia = pickle.load(open(dir_raster / "cosia.pkl", "rb"))
    cosia_landcover = pickle.load(open(dir_raster / "cosia_landcover.pkl", "rb"))
    cosia_influence = pickle.load(open(dir_raster / "cosia_influence.pkl", "rb"))

    cosia = _expand_static(cosia, len(dates))
    cosia_landcover = _expand_static(cosia_landcover, len(dates))
    #cosia_influence = _expand_static(cosia_influence, len(dates))

    bands = valeurs_cosia_couverture.keys()

    data_vars = {
        "cosia_landcover": (("latitude", "longitude", "date"), cosia_landcover),
        #"cosia_influence": (("band", "latitude", "longitude", "date"), cosia_influence),
    }

    for band in bands:
        data_vars[band] = (("latitude", "longitude", "date"), cosia[valeurs_cosia_couverture[band]])

    coords = {"latitude": lat, "longitude": lon, "date": dates}
    return xr.Dataset(data_vars, coords=coords)

def load_raster_corine(dir_raster: Path, dates: list, lat, lon) -> xr.Dataset:
    """Load CORINE rasters into an xarray structure."""
    corine = pickle.load(open(dir_raster / "corine.pkl", "rb"))
    corine_land = pickle.load(open(dir_raster / "corine_landcover.pkl", "rb"))
    corine_inf = pickle.load(open(dir_raster / "corine_influence.pkl", "rb"))

    corine = _expand_static(corine, len(dates))
    corine_land = _expand_static(corine_land, len(dates))
    corine_inf = _expand_static(corine_inf, len(dates))
    
    bands = [
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

    data_vars = {
        "corine_landcover": (("latitude", "longitude", "date"), corine_land),
        #"cosia_influence": (("band", "latitude", "longitude", "date"), cosia_influence),
    }

    for i, band in enumerate(bands):
        data_vars[band] = (("latitude", "longitude", "date"), corine[i])

    coords = { "latitude": lat, "longitude": lon, "date": dates}
    return xr.Dataset(data_vars, coords=coords)

def load_raster_elevation(dir_raster: Path, dates: list, lat, lon) -> xr.Dataset:
    """Load elevation raster and expand it across dates."""
    elevation = pickle.load(open(dir_raster / "elevation.pkl", "rb"))
    elevation = _expand_static(elevation, len(dates))

    data_vars = {
        "elevation": (("latitude", "longitude", "date"), elevation),
    }

    coords = {"latitude": lat, "longitude": lon, "date": dates}
    return xr.Dataset(data_vars, coords=coords)


def load_raster_population(dir_raster: Path, dates: list, lat, lon) -> xr.Dataset:
    """Load population raster and expand it across dates."""
    population = pickle.load(open(dir_raster / "population.pkl", "rb"))
    population = _expand_static(population, len(dates))

    data_vars = {"population": (("latitude", "longitude", "date"), population)}

    coords = {"latitude": lat, "longitude": lon, "date": dates}
    return xr.Dataset(data_vars, coords=coords)

def load_raster_vigicrues(dir_raster: Path, dates: list, lat, lon) -> xr.Dataset:
    """Load vigicrues rasters (12h and 16h) into an xarray."""
    data_vars = {}
    for file in sorted(dir_raster.glob("vigicrues*.pkl")):
        values = pickle.load(open(file, "rb"))
        data_vars[file.stem] = ("latitude", "longitude", "date"), values

    coords = {"latitude": lat, "longitude": lon, "date": dates}
    return xr.Dataset(data_vars, coords=coords)

def load_raster_air_quality(dir_raster: Path, dates: list, lat, lon) -> xr.Dataset:
    """Load air quality rasters produced by ``rasterise_air_qualite``."""
    data_vars = {}
    for file in sorted(dir_raster.glob("*raw.pkl")):
        if any(file.stem.startswith(prefix) for prefix in [
            "NO2", "O3", "PM10", "PM25", "NO216", "O316", "PM1016", "PM2516"]):
            values = pickle.load(open(file, "rb"))
            data_vars[file.stem.replace("raw", "")] = (
                "latitude",
                "longitude",
                "date",
            ), values

    coords = {"latitude": lat, "longitude": lon, "date": dates}
    return xr.Dataset(data_vars, coords=coords)

def load_raster_sat(dir_raster: Path, dates: list, lat, lon) -> xr.Dataset:
    """Load Sentinel satellite rasters."""
    sentinel = pickle.load(open(dir_raster / "sentinel.pkl", "rb"))

    coords = {"latitude": lat, "longitude": lon, "date": dates}
    data_vars = {}
    for i, var in enumerate(['NDVI', 'NDMI', 'NDBI', 'NDSI', 'NDWI']):
        data_vars[var] = (("latitude", "longitude", "date"), sentinel[i]
    )
    return xr.Dataset(data_vars, coords=coords)

def load_raster_argile(dir_raster: Path, dates: list, lat, lon) -> xr.Dataset:
    """Load argile raster and expand across dates."""
    argile = pickle.load(open(dir_raster / "argile.pkl", "rb"))
    argile = _expand_static(argile, len(dates))

    data_vars = {"argile": (("latitude", "longitude", "date"), argile)}

    coords = {"latitude": lat, "longitude": lon, "date": dates}
    return xr.Dataset(data_vars, coords=coords)

def load_raster_foret(dir_raster: Path, dates: list, lat, lon) -> xr.Dataset:
    """Load forest rasters and broadcast them on the date dimension."""
    foret = pickle.load(open(dir_raster / "foret.pkl", "rb"))
    foret_land = pickle.load(open(dir_raster / "foret_landcover.pkl", "rb"))
    foret_inf = pickle.load(open(dir_raster / "foret_influence.pkl", "rb"))

    foret = _expand_static(foret, len(dates))
    foret_land = _expand_static(foret_land, len(dates))
    foret_inf = _expand_static(foret_inf, len(dates))

    bands = valeurs_foret_attribut.keys()

    data_vars = {
        "forest_landcover": (("latitude", "longitude", "date"), foret_land),
        #"cosia_influence": (("band", "latitude", "longitude", "date"), cosia_influence),
    }

    for band in bands:
        data_vars[band] = (("latitude", "longitude", "date"), foret[valeurs_foret_attribut[band]])

    coords = {"latitude": lat, "longitude": lon, "date": dates}
    return xr.Dataset(data_vars, coords=coords)

def load_raster_bdroute(dir_raster: Path, dates: list, lat, lon) -> xr.Dataset:
    """Load forest rasters and broadcast them on the date dimension."""
    route = pickle.load(open(dir_raster / "route.pkl", "rb"))
    route_land = pickle.load(open(dir_raster / "route_landcover.pkl", "rb"))
    route_inf = pickle.load(open(dir_raster / "route_influence.pkl", "rb"))

    route = _expand_static(route, len(dates))
    route_land = _expand_static(route_land, len(dates))
    route_inf = _expand_static(route_inf, len(dates))

    print(route.shape)

    bands = ['NoRoad', 'Road']

    data_vars = {
        "route_landcover": (("latitude", "longitude", "date"), route_land),
        #"cosia_influence": (("band", "latitude", "longitude", "date"), cosia_influence),
    }

    for i, band in enumerate(bands):
        data_vars[band] = (("latitude", "longitude", "date"), route[i])

    coords = {"latitude": lat, "longitude": lon, "date": dates}
    return xr.Dataset(data_vars, coords=coords)

def load_raster_nappes(dir_raster: Path, dates: list, lat, lon) -> xr.Dataset:
    """Load groundwater level rasters into an xarray."""
    data_vars = {}
    for var in ["niveau_nappe_eau", "profondeur_nappe"]:
        file = dir_raster / f"{var}.pkl"
        if not file.is_file():
            continue
        values = pickle.load(open(file, "rb"))
        data_vars[var] = (("latitude", "longitude", "date"), values)

    coords = {"latitude": lat, "longitude": lon, "date": dates}
    return xr.Dataset(data_vars, coords=coords)

def concat_xarrays(dir_raster: Path, dates: list) -> xr.Dataset:
    """Concatenate all available rasters into a single xarray dataset.

    Parameters
    ----------
    dir_raster : Path
        Directory containing the pickled rasters.
    dates : list
        Dates associated with the rasters.

    Returns
    -------
    xr.Dataset
        A merged dataset containing every raster that could be loaded.
    """

    latitude = read_object('latitude.pkl', dir_raster)
    longitude = read_object('longitude.pkl', dir_raster)

    latitude = latitude[:, 0]
    longitude = longitude[0]

    loaders = [
        (load_rasterise_meteo),
        #(load_raster_cosia),
        (load_raster_corine),
        (load_raster_elevation),
        (load_raster_population),
        #(load_raster_vigicrues, list(dir_raster.glob("vigicrues*.pkl"))),
        #(load_raster_air_quality, list(dir_raster.glob("*raw.pkl"))),
        (load_raster_sat),
        (load_raster_argile),
        (load_raster_foret),
        (load_raster_bdroute),
        #(load_raster_nappes, [dir_raster / "niveau_nappe_eau.pkl", dir_raster / "profondeur_nappe.pkl"]),
    ]

    datasets = []
    for loader in loaders:
        datasets.append(loader(dir_raster, dates, latitude, longitude))
        
    if not datasets:
        raise ValueError("No raster data found in the provided directory")

    f = open(dir_raster / f'datacube.pkl',"wb")
    pickle.dump(xr.merge(datasets),f)