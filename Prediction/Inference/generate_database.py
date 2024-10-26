from tools_database import *
import os
import logging
import feedparser
import re
from bs4 import BeautifulSoup
import argparse

############################### Logger #####################################

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")

# Handler pour afficher les logs dans le terminal
if (logger.hasHandlers()):
    logger.handlers.clear()
streamHandler = logging.StreamHandler(stream=sys.stdout)
streamHandler.setFormatter(logFormatter)
logger.addHandler(streamHandler)
################################ Database ####################################

class GenerateDatabase():
    def __init__(self, departement,
                 compute_meteostat_features, meteostatParams,
                 compute_temporal_features, outputParams,
                 compute_spatial_features, spatialParams,
                 compute_air_features, airParams,
                 compute_trafic_features, bouchonParams,
                 compute_vigicrues_features, vigicrueParams,
                 compute_nappes_features, nappesParams,
                 region, h3,
                 dir_raster
                 ):
        
        self.departement = departement

        self.compute_meteostat_features = compute_meteostat_features
        self.meteostatParams = meteostatParams

        self.compute_temporal_features = compute_temporal_features
        self.outputParams = outputParams

        self.compute_spatial_features = compute_spatial_features
        self.spatialParams = spatialParams

        self.compute_air_features = compute_air_features
        self.airParams = airParams

        self.compute_trafic_features = compute_trafic_features
        self.bouchonParams = bouchonParams

        self.compute_vigicrues_features = compute_vigicrues_features
        self.vigicrueParams = vigicrueParams

        self.compute_nappes_features = compute_nappes_features
        self.nappesParams = nappesParams

        self.dir_raster = dir_raster

        self.region = region

        self.h3 = h3

        #self.elevation, self.lons, self.lats = read_tif(self.spatialParams['dir'] / 'elevation' / self.spatialParams['elevation_file'])

    def compute_meteo_stat(self):
        logger.info('Compute compute_meteo_stat')
        self.meteostat = construct_historical_meteo(self.meteostatParams['start'], self.meteostatParams['end'],
                                                    self.region, self.meteostatParams['dir'], self.departement)
        
        self.meteostat['year'] = self.meteostat['creneau'].apply(lambda x : x.split('-')[0])
        self.meteostat.to_csv(self.meteostatParams['dir'] / 'meteostat.csv', index=False)

    def compute_temporal(self):
        logger.info('Compute compute_temporal')
        rasterise_meteo_data(self.clusterSum, self.h3tif, self.meteostat, self.h3tif.shape, self.dates, self.dir_raster)

    def add_spatial(self):
        logger.info('Add spatial')

        #raster_sat(self.h3tif, self.spatialParams['dir_sat'], self.dir_raster, self.dates)
        #raster_land(self.h3tif, self.h3tif_high, self.spatialParams['dir_sat'], self.dir_raster, self.dates)
        if not (self.spatialParams['dir'] / 'population' / 'population.csv').is_file():
            pass
        raster_population(self.h3tif, self.h3tif_high, self.dir_raster, self.resLon, self.resLat, self.spatialParams['dir'])
        if not (self.spatialParams['dir'] / 'osmnx' / 'osmnx.geojson').is_file():
            pass
        raster_osmnx(self.h3tif, self.h3tif_high, self.dir_raster, self.resLon, self.resLat, self.spatialParams['dir'], self.departement)
        if not (self.spatialParams['dir'] / 'elevation' / 'elevation.csv').is_file():
            pass
        raster_elevation(self.h3tif, self.dir_raster, self.resLon, self.resLat, self.spatialParams['dir'], self.departement)
        if not (self.spatialParams['dir'] / 'BDFORET' / 'population.csv').is_file():
            pass
        raster_foret(self.h3tif, self.h3tif_high, self.dir_raster, self.resLon_high, self.resLat_high, self.spatialParams['dir'], self.departement)

    def add_air_qualite(self):
        air_stations = pd.read_csv(self.airParams['dir'] / 'air' / 'ExportStations.csv', sep=';')
        air_stations["Date d'entrée en service (à 00h00)"] = pd.to_datetime(air_stations["Date d'entrée en service (à 00h00)"], dayfirst=True)
        gdf = gpd.GeoDataFrame(air_stations, geometry=gpd.points_from_xy(air_stations.Longitude, air_stations.Latitude))
        gdf = gdf.set_crs(epsg=4326)
        polluants = {
        #'CO' : '04',
        #'C6H6': 'V4', 
        'NO2': '03',
        'O3': '08',
        #'SO2': '01',
        'PM10': '24',
        'PM25': '39',
        }
        start = dt.datetime.strptime(self.airParams['start'], '%Y-%m-%d')
        #if not (self.airParams['dir'] / 'air' / 'Air_archive').is_dir():
        if True:
            token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiR2VvZGFpclJlc3RBUEkiXSwic2NvcGUiOlsicmVhZCJdLCJleHAiOjE3MjQ3MTk3NDgsImF1dGhvcml0aWVzIjpbIlNFUlZJQ0UiXSwianRpIjoiYTM2MzRmMWEtNzMxZi00NjM0LTlhYWEtYjZlOWRhMDQwYTliIiwiY2xpZW50X2lkIjoiR2VvZGFpckZyb250ZW5kQ2xpZW50In0.w4KFBqOt8gFurOnJ358taZgkY9UUu6lNKhbReTc36yY'
            headers = {
                            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/111.0',
                            'Accept': 'application/json, text/plain, */*',
                            'Accept-Language': 'fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3',
                            'Authorization': f'Bearer {token}',
                            'Connection': 'keep-alive',
                            'Referer': 'https://www.geodair.fr/donnees/export-advanced',
                            'Sec-Fetch-Mode': 'cors',
                            'Sec-Fetch-Site': 'same-origin',
                        }
            for polluant in polluants:
                logger.info(f" - polluant : {polluant}")
                for annee in range(start.year, dt.datetime.now().year + 1, 2):
                    logger.info(f"   année : {annee}")
                    #if not (self.airParams['dir'] / 'air' / 'Air_archive' / f"{polluant}_{annee}_{annee+2}.csv").is_file():
                    if True:
                        headers1 = headers.copy()
                        headers1['Content-Type']= 'multipart/form-data; boundary=---------------------------14490471821690778178105310701'
                        headers1['Origin'] ='https://www.geodair.fr'
                        data = f'''-----------------------------14490471821690778178105310701\r\nContent-Disposition: form-data; name="polluant"\r\n\r\n{polluants[polluant]}\r\n-----------------------------14490471821690778178105310701\r\nContent-Disposition: form-data; name="region"\r\n\r\n\r\n-----------------------------14490471821690778178105310701\r\nContent-Disposition: form-data; name="departement"\r\n\r\n{self.departement.split('-')[1]}\r\n-----------------------------14490471821690778178105310701\r\nContent-Disposition: form-data; name="typologie"\r\n\r\n\r\n-----------------------------14490471821690778178105310701\r\nContent-Disposition: form-data; name="influence"\r\n\r\n\r\n-----------------------------14490471821690778178105310701\r\nContent-Disposition: form-data; name="date"\r\n\r\n{annee}-01-01 00:00\r\n-----------------------------14490471821690778178105310701\r\nContent-Disposition: form-data; name="dateFin"\r\n\r\n{annee+2}-01-01 00:00\r\n-----------------------------14490471821690778178105310701--\r\n'''
                        response1 = requests.post('https://www.geodair.fr/priv/api/station/filtered-list', headers=headers1, data=data)
                        reponse = json.loads(response1.text)
                        logger.info(reponse)
                        if len(reponse)>0:
                            headers2 = headers.copy()
                            headers2['Content-Type'] = 'multipart/form-data; boundary=---------------------------200640252532634071971631722713'
                            headers2['Origin'] = 'https://www.geodair.fr'
                            data = f'''-----------------------------200640252532634071971631722713\r\nContent-Disposition: form-data; name="polluant"\r\n\r\n{polluants[polluant]}\r\n-----------------------------200640252532634071971631722713\r\nContent-Disposition: form-data; name="codeStat"\r\n\r\na1\r\n-----------------------------200640252532634071971631722713\r\nContent-Disposition: form-data; name="typeStat"\r\n\r\nhoraire\r\n-----------------------------200640252532634071971631722713\r\nContent-Disposition: form-data; name="dateDebut"\r\n\r\n{annee}-01-01 00:00\r\n-----------------------------200640252532634071971631722713\r\nContent-Disposition: form-data; name="dateFin"\r\n\r\n{annee+2}-01-01 00:00\r\n-----------------------------200640252532634071971631722713\r\nContent-Disposition: form-data; name="stations"\r\n\r\n{','.join([reponse[k]["code"] for k in range(len(reponse))])}\r\n-----------------------------200640252532634071971631722713--\r\n'''
                            response2 = requests.post('https://www.geodair.fr/priv/api/statistique/export/grand-publique/periode', headers=headers2, data=data)
                            time.sleep(1)
                            while True:
                                cpt = 0
                                response3 = requests.get(
                                    f'https://www.geodair.fr/priv/api/download?id={response2.text}',
                                    headers=headers,
                                )
                                if "error" not in response3.text:
                                    with open('/tmp/txt.csv', 'w') as f:
                                        f.write(response3.text)
                                    dg = pd.read_csv('/tmp/txt.csv', sep=';')
                                    (self.airParams['dir'] / 'air' / 'Air_archive').mkdir(exist_ok=True, parents=True)
                                    dg.to_csv(self.airParams['dir'] / 'air' / 'Air_archive' / f"{polluant}_{annee}_{annee+2}.csv", sep=';', index=False)
                                    break
                                else:
                                    logger.info(response3)
                                    cpt+= 1
                                    time.sleep(1)
                                if cpt == 5:
                                    break
                                
        if True:
            polluants_csv = []
            for polluant in polluants:
                for annee in range(start.year, dt.datetime.now().year + 1, 2):
                    po = pd.read_csv(self.airParams['dir'] / 'air' / 'Air_archive' / f"{polluant}_{annee}_{annee+2}.csv", sep=';')
                    po.rename({'valeur' : polluant}, inplace=True, axis='columns')
                    polluants_csv.append(po)

            polluants_csv = pd.concat(polluants_csv).reset_index(drop=True)
            polluants_csv = polluants_csv.sort_values(by='Date de début')
            
            polluants_csv['creneau'] = polluants_csv['Date de début'].apply(lambda x : x.split(' ')[0])
            polluants_csv['horaire'] = polluants_csv['Date de début'].apply(lambda x : x.split(' ')[1])
            polluants_csv.rename({'Longitude' : 'longitude', 'Latitude' : 'latitude'}, inplace=True, axis='columns')
            polluants_csv = polluants_csv[(polluants_csv['horaire'] == '12:00:00')]
            polluants16 = polluants_csv[(polluants_csv['horaire'] == '16:00:00')]
            for polluant in polluants.keys():
                polluants_csv[f'{polluant}16'] = polluants16[polluant]
            
            polluants_csv.to_csv(self.airParams['dir'] / 'air' / 'Air_archive' / 'polluants1216.csv', sep=',', index=False)

        else:
            polluants_csv = pd.read_csv(self.airParams['dir'] / 'air' / 'Air_archive' / 'polluants1216.csv')

        logger.info(polluants_csv.columns)
        rasterise_air_qualite(self.clusterSum, self.h3tif, polluants_csv, self.h3tif.shape, self.dates, self.dir_raster)

    def compute_hauteur_riviere(self):
        code = self.departement.split('-')[1]
        dir_logs = Path('./')
        dir_hydroreel = self.vigicrueParams['dir'] / 'hydroreel'
        dir_france_hydroreel = Path('/home/caron/Bureau/csv/france/data/hydroreel')
        check_and_create_path(dir_france_hydroreel)

        start = dt.datetime.strptime(self.vigicrueParams['start'], '%Y-%m-%d')
        stop = dt.datetime.strptime(self.vigicrueParams['stop'], '%Y-%m-%d')

        ################################ Create file ####################################################
        def hydroreel_date_parser(chaine):
            try:
                return dt.datetime.strptime(chaine, "%d/%m/%Y %H:%M")
            except:
                return dt.datetime.strptime(chaine, "%Y-%m-%d %H:%M:%S+00:00")

        # On récupère la liste des stations hydroréel au besoin
        stations_a_valider = False
        if not (dir_hydroreel / 'stations_hydroreel.csv').is_file():
            logger.info("On récupère sur le net les stations hydroréel")
            url="https://www.hydro.eaufrance.fr/rechercher/ajax/entites-hydrometriques?hydro_entities_search%5Blabel%5D%5Bvalue%5D=&hydro_entities_search%5Bcode%5D=&hydro_entities_search%5BhydroRegion%5D=&hydro_entities_search%5BsiteTypes%5D%5B%5D=53b64673-5725-4967-9880-b775b65bdc9e&hydro_entities_search%5BsiteTypes%5D%5B%5D=913a1b84-0e48-44e1-a7c7-ab8a867b34ee&hydro_entities_search%5BsiteTypes%5D%5B%5D=672aecb1-d629-426e-85a1-60882db8b30e&hydro_entities_search%5BsiteTypes%5D%5B%5D=7a358b42-a4c6-4ee8-b019-722c1871cc3a&hydro_entities_search%5BtopoWatershedAreaMin%5D=&hydro_entities_search%5BtopoWatershedAreaMax%5D=&hydro_entities_search%5BsiteStatus%5D=&hydro_entities_search%5BstartedDateMin%5D=&hydro_entities_search%5BstartedDateMax%5D=&hydro_entities_search%5BkmPointMin%5D=&hydro_entities_search%5BkmPointMax%5D=&hydro_entities_search%5BstationTypes%5D%5B%5D=a642c792-d0fc-40ec-91c2-f0790391dba6&hydro_entities_search%5BstationTypes%5D%5B%5D=fbe2b93c-e34d-43bf-8f7b-4acbdaee8af0&hydro_entities_search%5BstationTypes%5D%5B%5D=5df2125a-ad73-4147-bb73-32da7982a978&hydro_entities_search%5BrddSandreCode%5D=&hydro_entities_search%5Bactive%5D=1"
            r = requests.get(url)
            dico = json.loads(r.text)
            txt = "id;nom;X;Y\n"
            txt += '\n'.join([';'.join([dico['sites'][k]['stations'][0]['CdStationHydro'], dico['sites'][k]['stations'][0]['LbStationHydro'].replace(';', ',').replace('[', '').replace(']','').replace(' ','_'), dico['sites'][k]['stations'][0]['CoordStationHydro']['CoordXStationHydro'], dico['sites'][k]['stations'][0]['CoordStationHydro']['CoordYStationHydro']]) for k in range(len(dico['sites']))])
            with open(dir_hydroreel / 'stations.csv', 'w') as f:
                f.write(txt)
            df = pd.read_csv(dir_hydroreel / 'stations.csv', sep=';')
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y))
            # Il est généralement préférable d'utiliser un système de coordonnées projeté 
            # comme EPSG:2154 (Lambert-93) pour effectuer des opérations géospatiales
            gdf.crs = "EPSG:4326"
            gdf = gdf.to_crs(epsg=2154)
            gdf_depts = code
            if gdf_depts.crs.to_string() != 'EPSG:2154':
                gdf_depts = gdf_depts.to_crs(epsg=2154)
            # Création d'une zone tampon de 5 km autour de chaque géométrie, pour récupérer les stations
            # des cours d'eau potentiellement à la "frontière" entre deux départements
            gdf_depts['geometry'] = gdf_depts['geometry'].buffer(5000)
            gdf_depts.set_geometry('geometry', inplace=True)
            #gdf_depts = gdf_depts.set_geometry('buffered_geometry').drop(columns=['buffered_geometry'])
            df = gpd.sjoin(gdf_depts, gdf, how="inner", op="intersects")
            #df = gpd.sjoin(gdf, gdf_depts)
            stations_hydroreel = df[['id', 'nom_right', 'X', 'Y', 'code_insee']]
            stations_hydroreel.columns = ['station', 'nom', 'longitude', 'latitude', 'departement']
            stations_hydroreel = stations_hydroreel[stations_hydroreel.departement==code]
            stations_hydroreel.to_csv(dir_hydroreel / 'stations_hydroreel.csv', index=False, sep=';')
            stations_a_valider = True
        else:
            stations_hydroreel = pd.read_csv(dir_hydroreel / 'stations_hydroreel.csv', sep=';', dtype={'departement': str})

        # On récupère l'archive historique au besoin
        liste = list(stations_hydroreel.station)
        for index, station in enumerate(sorted(liste)):
            if not (dir_hydroreel / f"{station}.feather").is_file():
                logger.info(f"On collecte l'archive de {station} ({index+1}/{len(liste)})")
                df = pd.DataFrame()
                for annee in range(start.year, stop.year + 1):
                    # Laisser à raw pour avoir un relevé par heure
                    url1 = f"https://www.hydro.eaufrance.fr/stationhydro/ajax/{station}/series?hydro_series%5BstartAt%5D=01%2F01%2F{annee}&hydro_series%5BendAt%5D=01%2F07%2F{annee}&hydro_series%5BvariableType%5D=simple_and_interpolated_and_hourly_variable&hydro_series%5BsimpleAndInterpolatedAndHourlyVariable%5D=H&hydro_series%5BstatusData%5D=raw" 
                    url2 = f"https://www.hydro.eaufrance.fr/stationhydro/ajax/{station}/series?hydro_series%5BstartAt%5D=01%2F07%2F{annee}&hydro_series%5BendAt%5D=01%2F01%2F{annee+1}&hydro_series%5BvariableType%5D=simple_and_interpolated_and_hourly_variable&hydro_series%5BsimpleAndInterpolatedAndHourlyVariable%5D=H&hydro_series%5BstatusData%5D=raw"
                    for url in [url1, url2]:
                        r = requests.get(url)
                        if r.status_code == 200:
                            logger.info(f" -> Des données ont été collectées pour l'année {annee}")
                            dg = pd.DataFrame.from_dict(json.loads(r.text)['series'])
                            dg = dg.loc[dg.metric=='H']
                            logger.debug(f"    Taille de la série : {len(dg)}")
                            if len(dg) == 0:
                                logger.info("    Pas de données -> on ignore la station")
                                break
                            logger.debug(f"    Unité de la série: {dg.iloc[0].unit}")
                            assert dg.iloc[0].unit=='mm'
                            dg['H (cm)'] = dg['data'].apply(lambda dico: 0.1*dico['v'])
                            dg['Date'] = dg['data'].apply(lambda dico: dico['t'])
                            dg['Date'] = pd.to_datetime(dg['Date'])
                            dg = dg[['Date', 'H (cm)']]
                            taille = df.shape
                            df = pd.concat([df, dg])
                            logger.info(f"    Le dataframe de {station} passe de {taille} à {df.shape}")
                            df.reset_index(inplace=True, drop=True)
                            # on reçoit les données en mm
                            if df['H (cm)'].isna().sum() <= 5000:
                                df.to_feather(dir_hydroreel / f"{station}.feather")
                            else:
                                if stations_a_valider:
                                    stations_hydroreel = stations_hydroreel.loc[stations_hydroreel.station != station]
                                    stations_hydroreel.to_csv(dir_hydroreel / 'stations_hydroreel.csv', index=False, sep=';')
                            #time.sleep(1)
                        else:
                            logger.info(f" -> Erreur : code {r.status_code}")
                            if stations_a_valider:
                                stations_hydroreel = stations_hydroreel.loc[stations_hydroreel.station != station]
                                stations_hydroreel.to_csv(dir_hydroreel / 'stations_hydroreel.csv', index=False, sep=';')
                            break
                
        # On récupère les nouvelles mises à jour, seulement pour une station sur 3
        url_initial = "https://hubeau.eaufrance.fr/api/v1/hydrometrie/observations_tr?"
        url_outputs = "&pretty&grandeur_hydro=H&fields=code_station,date_obs,resultat_obs,continuite_obs_hydro"
        moyennes = {}
        total = []
        for index, station in enumerate(sorted(stations_hydroreel.station)):
            # On met à jour les stations, uniquement dans le cas Tous, et soit une fois
            # sur trois, soit en mode slow
            logger.info(f" * On tente de mettre à jour {station}")
            if dir_hydroreel / (station+'.feather') in dir_hydroreel.iterdir():
                df = pd.read_feather(dir_hydroreel / (station+'.feather'))
            else:
                if not (dir_hydroreel / (station+'.csv')).is_file():
                    continue
                df = pd.read_csv(dir_hydroreel / (station+'.csv'), 
                        parse_dates=['Date'], 
                        date_parser=hydroreel_date_parser)
            Date_debut = max(df['Date'])
            Date_debut += dt.timedelta(hours=1)
            try:
                Date_fin = min(pd.to_datetime(dt.datetime.now().replace(minute=0, second=0, microsecond=0), utc=True),
                                Date_debut + dt.timedelta(days=3))
            except:
                Date_fin = min(dt.datetime.now().replace(minute=0, second=0, microsecond=0),
                                Date_debut + dt.timedelta(days=3))
            cpt = 0
            while Date_fin != Date_debut:
                cpt += 1
                if cpt == 10:
                    # On a récolté un mois d'archives nouvelles. On arrête là, pour éviter d'être blacklisté
                    # On reprendra à la prochaine exécution du script
                    break
                date_debut = Date_debut.strftime('%Y-%m-%dT%H:%M:%S')
                date_fin = Date_fin.strftime('%Y-%m-%dT%H:%M:%S')[:-5] + "00:00"
                logger.info(f"  - début des relevés à télécharger : {date_debut}")
                logger.info(f"  - fin des relevés : {date_fin}")        
                url_parameters = "code_entite=" + station + "&date_debut_obs=" + date_debut + "&date_fin_obs=" + date_fin
                url = url_initial + url_parameters + url_outputs
                logger.info("  - on envoie notre requête au site hydroréel")
                logger.info(f"  - URL : {url}")
                try:
                    response = requests.get(url)
                    text = response.json()
                    if 'data' in text and len(text['data'])>0:
                        logger.info("  - des données ont été collectées")
                        dg = pd.DataFrame(text['data'])
                        dg['date_obs'] = pd.to_datetime(dg['date_obs'], format='%Y-%m-%dT%H:%M:%SZ')
                        dg = dg[['date_obs', 'resultat_obs']]
                        dg.columns = ['Date', 'H (cm)']
                        dg['H (cm)'] = dg['H (cm)']*0.1
                        df = pd.concat([df, dg])
                        df.drop_duplicates()
                    else:
                        logger.info("  - rien n'a été reçu")
                except Exception as e:
                    logger.info(f"Erreur {e} pour cette station")
                    with open(dir_logs / 'probleme_de_connexion.csv', 'a') as f:
                        f.write(f'{dt.datetime.now()},{url}\n')
                Date_debut = Date_fin
                try: 
                    Date_fin = min(pd.to_datetime(dt.datetime.now().replace(minute=0, second=0, microsecond=0), utc=True),
                                    Date_debut + dt.timedelta(days=3))            
                except:
                    Date_fin = min(dt.datetime.now().replace(minute=0, second=0, microsecond=0),
                                    Date_debut + dt.timedelta(days=3))            
                df.reset_index(inplace=True)
                df = df[['Date', 'H (cm)']]
                df.to_feather(dir_hydroreel / (station+'.feather'))
            else:
                logger.info(f" * On mettra à jour {station} lors d'une prochaine itération")
                if not (dir_hydroreel / (station+'.feather')).is_file():
                    continue
                else:
                    df = pd.read_feather(dir_hydroreel / (station+'.feather'))
            if not df.empty:
                df['Date'] = pd.to_datetime(df['Date'], utc=True)
                df = df[df['Date']>=start]
                df = df[df['Date']<=stop]
                df1 = df.groupby(pd.Grouper(key='Date', freq='H')).size()
                df2 = df.groupby(pd.Grouper(key='Date', freq='H')).mean()
                df = pd.concat([df2, df1], axis=1)
                nom = stations_hydroreel.loc[stations_hydroreel.station == station].nom.iloc[0]
                df.columns = [station+'_height', station+'_measures']
                moyennes[station] = df[station+'_height'].mean()
                if "df_hydro" not in locals():
                    df_hydro = df
                else:
                    df_hydro = pd.concat([df_hydro, df], axis=1)
            df['longitude'] = stations_hydroreel[stations_hydroreel['station'] == station]['longitude'].values[0]
            df['latitude'] = stations_hydroreel[stations_hydroreel['station'] == station]['latitude'].values[0]
            total.append(df)

        total = pd.concat(total).reset_index(drop=True)
        total['creneau'] = total['Date'].apply(lambda x : hydroreel_date_parser(x))
        total['date'] = total['Date'].apply(lambda x : hydroreel_date_parser(x))
        total['hour'] = total['Date'].apply(lambda x : x.split(' ')[1])
        total.rename({'H (cm)' : 'hauteur'}, inplace=True, axis=1)
        total = total[total['hour'] == '12:00:00']
        total16 = total[total['hour'] == '16:00:00']

        rasterise_vigicrues(self.clusterSum, self.h3tif, total, self.h3tif.shape, self.dates, self.dir_raster, 'vigicrues12')
        rasterise_vigicrues(self.clusterSum, self.h3tif, total16, self.h3tif.shape, self.dates, self.dir_raster, 'vigicrues16')

    def compute_nappes_phréatique(self):
        code = self.departement.split('-')[1]
        dir_logs = Path('./')
        dir_nappes = self.nappesParams['dir'] / 'nappes'
        check_and_create_path(dir_nappes)
        if not (dir_nappes / 'liste_stations.csv').is_file():
            date_mesure = '2017-06-12'
            url=f"https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/stations?code_departement={code}&format=geojson&size=20&date_recherche={date_mesure}"
            r = requests.get(url)
            if r.status_code == 200 or r.status_code == 206:
                response = json.loads(r.text)
                response_features = response['features']

                dg = []

                for i, elt in enumerate(response_features):
                    dico_elt = {}
                    dico_elt['longitude'] = [elt['properties']['x']]
                    dico_elt['latitude'] = [elt['properties']['y']]
                    dico_elt['code_bss'] = [elt['properties']['code_bss']]
                    dg.append(pd.DataFrame.from_dict(dico_elt))
                dg = pd.concat(dg)
                dg.to_csv(dir_nappes / 'liste_stations.csv', index=False)
        else:
            dg = pd.read_csv(dir_nappes / 'liste_stations.csv')

        dg['code_bss_suff'] = dg['code_bss'].apply(lambda x : x.split('/')[1])
        dg['code_bss_pre'] = dg['code_bss'].apply(lambda x : x.split('/')[0])
        date_debut = self.nappesParams['start']
        date_fin = self.nappesParams['end']
        df = []
        for index, row in dg.iterrows():
            url=f"https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/chroniques?code_bss={row['code_bss_pre']}%2F{row['code_bss_suff']}&date_debut_mesure={date_debut}&date_fin_mesure={date_fin}"
            r = requests.get(url)
            if r.status_code == 200 or r.status_code == 206:
                response = json.loads(r.text)
                datas = response['data']
                df.append(pd.DataFrame.from_dict(datas))
            else:
                logger.info(f'Error {r.status_code}')
        df = pd.concat(df).reset_index(drop=True)
        df = df.set_index('code_bss').join(dg.set_index('code_bss')).reset_index()
        df.rename({'date_mesure': 'creneau'}, inplace=True, axis=1)
        df.to_csv(dir_nappes / 'mesures.csv', index=False)
        rasterise_nappes(self.clusterSum, self.h3tif, df, self.h3tif.shape, self.dates, self.dir_raster, 'niveau_nappe_eau')
        rasterise_nappes(self.clusterSum, self.h3tif, df, self.h3tif.shape, self.dates, self.dir_raster, 'profondeur_nappe')

    def process(self, start, stop, resolution):
        logger.info(self.departement)
        
        if self.compute_meteostat_features:
            self.compute_meteo_stat()
        else:
            self.meteostat = pd.read_csv(self.meteostatParams['dir'] / 'meteostat.csv')

        self.h3['latitude'] = self.h3['geometry'].apply(lambda x : float(x.centroid.y))
        self.h3['longitude'] = self.h3['geometry'].apply(lambda x : float(x.centroid.x))
        self.h3['altitude'] = 0

        self.clusterSum = self.h3.copy(deep=True)
        self.clusterSum['cluster'] = self.clusterSum.index.values.astype(int)
        self.cluster = None
        sdate = start
        edate = stop
        self.dates = find_dates_between(sdate, edate)

        resolutions = {'2x2' : {'x' : 0.02875215641173088,'y' :  0.020721094073767096},
                '1x1' : {'x' : 0.01437607820586544,'y' : 0.010360547036883548},
                '0.5x0.5' : {'x' : 0.00718803910293272,'y' : 0.005180273518441774},
                '0.03x0.03' : {'x' : 0.0002694945852326214,'y' :  0.0002694945852352859}}

        #n_pixel_x = 0.016133099692723363
        #n_pixel_y = 0.016133099692723363

        n_pixel_x = resolutions[resolution]['x']
        n_pixel_y = resolutions[resolution]['y']

        self.resLon = n_pixel_x
        self.resLat = n_pixel_y
        self.h3tif = rasterisation(self.clusterSum, n_pixel_y, n_pixel_x, column='cluster', defval=np.nan, name=self.departement+'_low')
        logger.info(f'Low scale {self.h3tif.shape}')

        n_pixel_x = resolutions['0.03x0.03']['x']
        n_pixel_y = resolutions['0.03x0.03']['y']

        self.resLon_high = n_pixel_x
        self.resLat_high = n_pixel_y
        self.h3tif_high = rasterisation(self.clusterSum, n_pixel_y, n_pixel_x, column='cluster', defval=np.nan, name=self.departement+'_high')
        logger.info(f'High scale {self.h3tif_high.shape}')

        if self.compute_temporal_features:
            self.compute_temporal()

        if self.compute_spatial_features:
            self.add_spatial()

        if self.compute_air_features:
            self.add_air_qualite()
        
        if self.compute_nappes_features:
            self.compute_nappes_phréatique()
        
        if self.compute_vigicrues_features:
            self.compute_hauteur_riviere()

def launch(dir_data, dir_data_disk, dir_raster, departement, resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, 
           compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop):

    dir_meteostat = dir_data / 'meteostat'
    check_and_create_path(dir_raster)
    
    meteostatParams = {'start' : '2016-01-01',
                    'end' : stop,
                    'dir' : dir_meteostat}

    outputParams  = {'start' : start,
                    'end' : stop}
    
    spatialParams = {'dir_sat':  dir_data_disk,
                    'dir' : dir_data,
                    'elevation_file' : 'elevation.tif',
                    'sentBand': ['NDVI', 'NDMI', 'NDBI', 'NDSI', 'NDWI'],
                    'landCoverBand' : ['0.0','1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0'],
                    'addsent': True,
                    'addland': True}
    
    airParams = {'dir' : dir_data,
                 'start' : start}

    bouchonParams = {'dir' : dir_data}

    vigicrueParams = {'dir' : dir_data,
                      'start' : start,
                        'end' : stop}
    
    nappesParams = {'dir' : dir_data,
                      'start' : start,
                        'end' : stop}

    region_path = dir_data / 'geo/geo.geojson'
    region = json.load(open(region_path))
    polys = [region["geometry"]]
    geom = [shape(i) for i in polys]
    region = gpd.GeoDataFrame({'geometry':geom})

    h3 = gpd.read_file(dir_data / 'spatial/hexagones.geojson')

    database = GenerateDatabase(departement,
                    compute_meteostat_features, meteostatParams,
                    compute_temporal_features, outputParams,
                    compute_spatial_features, spatialParams,
                    compute_air_features, airParams,
                    compute_trafic_features, bouchonParams,
                    compute_vigicrues_features, vigicrueParams,
                    compute_nappes_features, nappesParams,
                    region, h3,
                    dir_raster)

    database.process(start, stop, resolution)

if __name__ == '__main__':
    RASTER = True
    depts = gpd.read_file('/home/caron/Bureau/csv/france/data/departements/departements-20180101.shp')
    depts = depts.to_crs("EPSG:4326")
    parser = argparse.ArgumentParser(
        prog='Train',
        description='Create graph and database according to config.py and tained model',
    )
    parser.add_argument('-m', '--meteostat', type=str, help='Compute meteostat')
    parser.add_argument('-t', '--temporal', type=str, help='Temporal')
    parser.add_argument('-s', '--spatial', type=str, help='Spatial')
    parser.add_argument('-a', '--air', type=str, help='Air')
    parser.add_argument('-v', '--vigicrues', type=str, help='Vigicrues')
    parser.add_argument('-n', '--nappes', type=str, help='Nappes')
    parser.add_argument('-r', '--resolution', type=str, help='Resolution')

    args = parser.parse_args()

    # Input config
    compute_meteostat_features = args.meteostat == "True"
    compute_temporal_features = args.temporal == "True"
    compute_spatial_features = args.spatial == "True"
    compute_air_features = args.air == 'True'
    compute_trafic_features = False
    compute_nappes_features = args.nappes == 'True'
    compute_vigicrues_features = args.vigicrues == "True"
    resolution = args.resolution

    start = '2017-06-12'
    stop = '2024-06-28'

    ################## Ain ######################
    #launch('departement-01-ain', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## DOUBS ######################
    #launch('departement-25-doubs', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## HERAULT ####################
    #launch('departement-34-herault', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## YVELINES ######################
    #launch('departement-78-yvelines', resolution,  compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Rhone ######################
    #launch('departement-69-rhone', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)