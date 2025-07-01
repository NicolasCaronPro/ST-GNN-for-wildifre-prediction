import firedanger
import meteostat
import pandas as pd
import numpy as np

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