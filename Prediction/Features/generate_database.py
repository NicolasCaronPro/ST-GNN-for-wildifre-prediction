from download import *
import os
import logging
import feedparser
import re
from bs4 import BeautifulSoup
import io
import argparse
import pytz

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
        check_and_create_path(self.meteostatParams['dir'])
        self.meteostat = construct_historical_meteo(self.meteostatParams['start'], self.meteostatParams['end'],
                                                    self.region, self.meteostatParams['dir'], self.departement)
        
        self.meteostat['year'] = self.meteostat['creneau'].apply(lambda x : x.split('-')[0])
        self.meteostat.to_csv(self.meteostatParams['dir'] / 'meteostat.csv', index=False)

    def compute_temporal(self):
        logger.info('Compute compute_temporal')
        rasterise_meteo_data(self.clusterSum, self.h3tif, self.meteostat, self.h3tif.shape, self.dates, self.dir_raster)

    def add_spatial(self):
        logger.info('Add spatial')
        code_dept = name2int[self.departement]
        if code_dept < 10:
            code_dept = f'0{code_dept}'
        else:
            code_dept = f'{code_dept}'

        #raster_sat(self.h3tif, self.spatialParams['dir_sat'], self.dir_raster, self.dates)
        #raster_land(self.h3tif, self.h3tif_high, self.spatialParams['dir_sat'], self.dir_raster, self.dates)

        #if not (self.spatialParams['dir'] / 'cosia' / 'cosia.geojson').is_file():
        #if True:
        #    download_cosia(code_dept, self.region, self.spatialParams['dir'] / 'cosia')
        #raster_cosia(self.h3tif, self.h3tif_high, self.dir_raster, self.resLon_high, self.resLat_high, self.spatialParams['dir'], self.region)

        #if not (self.spatialParams['dir'] / 'tourbiere' / 'ZonesTourbeuses.geojson').is_file():
        #if True:
        #    download_tourbiere(Path('/home/caron/Bureau/csv/france/data/tourbiere/ZonesTourbeuses'), self.region, self.spatialParams['dir'] / 'tourbiere')
        #raster_tourbiere(self.h3tif, self.h3tif_high, self.dir_raster, self.resLon_high, self.resLat_high, self.spatialParams['dir'], self.region)

        """if not (self.spatialParams['dir'] / 'population' / 'population.csv').is_file():
            download_population(Path('/home/caron/Bureau/csv/france/data/population'), self.region, self.spatialParams['dir'] / 'population')
        raster_population(self.h3tif, self.h3tif_high, self.dir_raster, self.resLon, self.resLat, self.spatialParams['dir'])"""

        if not (self.spatialParams['dir'] / 'osmnx' / 'osmnx.geojson').is_file():
            download_osnmx(self.region, self.spatialParams['dir'] / 'osmnx')
        raster_osmnx(self.h3tif, self.h3tif_high, self.dir_raster, self.resLon, self.resLat, self.spatialParams['dir'], self.departement)

        """if not (self.spatialParams['dir'] / 'elevation' / 'elevation.csv').is_file():
            download_elevation(code_dept, self.region, self.spatialParams['dir'] / 'elevation')
        raster_elevation(self.h3tif, self.dir_raster, self.resLon, self.resLat, self.spatialParams['dir'], self.departement)"""
        
        #if True:
        #if not (self.spatialParams['dir'] / 'BDFORET' / 'foret.geojson').is_file():
        #    download_foret(code_dept, self.departement, self.spatialParams['dir'])
        #raster_foret(self.h3tif, self.h3tif_high, self.dir_raster, self.resLon_high, self.resLat_high, self.spatialParams['dir'], self.departement)

        """if not (self.spatialParams['dir'] / 'argile' / 'argile.geojson').is_file():
            download_argile(Path('/home/caron/Bureau/csv/france/data/argile'), code_dept, self.spatialParams['dir'] / 'argile')"""
        
        #raster_argile(self.h3tif, self.h3tif_high, self.dir_raster, self.resLon_high, self.resLat_high, self.spatialParams['dir'], self.departement)

    def add_air_qualite(self):
        if not (self.airParams['dir'] / 'air' / 'ExportStations.csv').is_file():
            download_air(Path('/home/caron/Bureau/csv/france/data/air'), self.region, self.airParams['dir'] / 'air')

        air_stations = pd.read_csv(self.airParams['dir'] / 'air' / 'ExportStations.csv')
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
        if not (self.airParams['dir'] / 'air' / 'Air_archive').is_dir():
        #if True:
            token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiR2VvZGFpclJlc3RBUEkiXSwic2NvcGUiOlsicmVhZCJdLCJleHAiOjE3MjU1NjYzNDYsImF1dGhvcml0aWVzIjpbIlNFUlZJQ0UiXSwianRpIjoiYjhhOTY0NTktZDRjNi00YTc1LTlhMzctYjJlOTMxOTliM2I2IiwiY2xpZW50X2lkIjoiR2VvZGFpckZyb250ZW5kQ2xpZW50In0.-Vc8Dx17OJDG0ymqMUYCVQMwT8acMZrqEszooE8fKwI'           
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
                    if not (self.airParams['dir'] / 'air' / 'Air_archive' / f"{polluant}_{annee}_{annee+2}.csv").is_file():
                    #if True:
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
                    if (self.airParams['dir'] / 'air' / 'Air_archive' / f"{polluant}_{annee}_{annee+2}.csv").is_file():
                        po = pd.read_csv(self.airParams['dir'] / 'air' / 'Air_archive' / f"{polluant}_{annee}_{annee+2}.csv", sep=';')
                        po.rename({'valeur' : polluant}, inplace=True, axis='columns')
                        polluants_csv.append(po)

            if len(polluants_csv) == 0:
                polluants_csv = pd.DataFrame(index=np.arange(len(self.dates)))
                polluants_csv['creneau'] = self.dates
                polluants_csv['horaire'] = '12:00:00'
                polluants_csv['Longitude'] = np.nan
                polluants_csv['Latitude'] = np.nan
                polluants_csv[['NO2', 'O3', 'PM10', 'PM25']] = np.nan
            else:
                polluants_csv = pd.concat(polluants_csv).reset_index(drop=True)
                polluants_csv = polluants_csv.sort_values(by='Date de début')
            
                polluants_csv['creneau'] = polluants_csv['Date de début'].apply(lambda x : x.split(' ')[0])
                polluants_csv['horaire'] = polluants_csv['Date de début'].apply(lambda x : x.split(' ')[1])

            for polluant in ['NO2', 'O3', 'PM10', 'PM25']:
                if polluant not in polluants_csv.columns:
                    polluants_csv[polluant] = np.nan

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

        check_and_create_path(dir_hydroreel)

        START = dt.datetime.strptime(self.vigicrueParams['start'], '%Y-%m-%d')
        STOP = dt.datetime.strptime(self.vigicrueParams['end'], '%Y-%m-%d')

        ################################ Create file ####################################################
        #if not (dir_hydroreel / 'stations_hydroreel.csv').is_file():
        if True: 
            if name2int[self.departement] < 10:
                url = f"https://hubeau.eaufrance.fr/api/v1/hydrometrie/referentiel/stations.csv?code_departement=0{name2int[self.departement]}&en_service=true&size=1000"
            else:
                url = f"https://hubeau.eaufrance.fr/api/v1/hydrometrie/referentiel/stations.csv?code_departement={name2int[self.departement]}&en_service=true&size=1000"
            r = requests.get(url)
            assert r.status_code in [200, 206]
            csv_data = io.StringIO(r.text)
            stations_hydroreel = pd.read_csv(csv_data, sep=';')
            stations_hydroreel.rename({'code_station': 'station', 'libelle_station': 'nom', 'longitude_station': 'longitude', 'latitude_station': 'latitude'}, axis=1, inplace=True)
            stations_hydroreel['departement'] = name2int[self.departement]
            stations_hydroreel = stations_hydroreel[['station', 'nom', 'longitude', 'latitude', 'departement']]
            stations_hydroreel.to_csv(dir_hydroreel / 'stations_hydroreel.csv', index=False, sep=';')
            logger.info("fichier des stations sauvegardé")
            stations_a_valider = True
        else:
            stations_hydroreel = pd.read_csv(dir_hydroreel / 'stations_hydroreel.csv', sep=';', dtype={'departement': str})

        #########################################################################

        rname = 'stations_'+self.departement.split('-')[1]+'_full.csv'
        fic_name = dir_hydroreel / rname

        def hydroreel_date_parser(chaine):
            try:
                return dt.datetime.strptime(chaine, "%d/%m/%Y %H:%M").strftime("%Y-%m-%d %H:%M:%S")
            except:
                try:
                #'2018-01-01 00:00:00+00:00'
                    return dt.datetime.strptime(chaine, "%Y-%m-%d %H:%M:%S+00:00").strftime("%Y-%m-%d %H:%M:%S")
                except:
                    return dt.datetime.strptime(chaine, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")

        logger.info('Collecting Hydroreel stations list')
        stations_a_valider = True
        count_max_ignore = 3
        stations_a_ignorer = []
        for station in sorted(list(stations_hydroreel.station)):
            if not (dir_hydroreel / f"{station}.csv").is_file():
            #if True:
                logger.info(f"On collecte l'archive de {station}")
                df = pd.DataFrame()
                cm = 0
                for annee in range(START.year, STOP.year + 1):
                        url1 = f"https://www.hydro.eaufrance.fr/stationhydro/ajax/{station}/series?hydro_series%5BstartAt%5D=01%2F01%2F{annee}&hydro_series%5BendAt%5D=01%2F07%2F{annee}&hydro_series%5BvariableType%5D=simple_and_interpolated_and_hourly_variable&hydro_series%5BsimpleAndInterpolatedAndHourlyVariable%5D=H&hydro_series%5BstatusData%5D=raw" 
                        url2 = f"https://www.hydro.eaufrance.fr/stationhydro/ajax/{station}/series?hydro_series%5BstartAt%5D=01%2F07%2F{annee}&hydro_series%5BendAt%5D=01%2F01%2F{annee+1}&hydro_series%5BvariableType%5D=simple_and_interpolated_and_hourly_variable&hydro_series%5BsimpleAndInterpolatedAndHourlyVariable%5D=H&hydro_series%5BstatusData%5D=raw"
                        for url in [url1, url2]:
                            r = requests.get(url)
                            if r.status_code == 200:
                                try:
                                    logger.info(f" -> Des données ont été collectées pour l'année {annee}")
                                    dg = pd.DataFrame.from_dict(json.loads(r.text)['series'])
                                    dg = dg.loc[dg.metric=='H']
                                    logger.debug(f"    Taille de la série : {len(dg)}")
                                    if len(dg) == 0:
                                        cm += 1
                                        if cm >= count_max_ignore and stations_a_valider:
                                            logger.info("    Pas de données -> on ignore la station")
                                            stations_a_ignorer.append(station)
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
                                            stations_a_ignorer.append(station)
                                except Exception as e:
                                    logger.info(f" -> Erreur : {e}")
                                    stations_a_ignorer.append(station)
                                    break
                            
                            else:
                                logger.info(f" -> Erreur : code {r.status_code}")
                                stations_a_ignorer.append(station)
                                break

                if station not in stations_a_ignorer:
                    df['latitude'] = stations_hydroreel[stations_hydroreel['station'] == station]['latitude'].values[0]
                    df['longitude'] = stations_hydroreel[stations_hydroreel['station'] == station]['longitude'].values[0]
                    df.to_csv(dir_hydroreel / f"{station}.csv", index=False)

        logger.info(f"On supprime les stations Hydroreel qui n'ont pas répondues : {', '.join(stations_a_ignorer)}")
        logger.info(f"Taille du dataframe avant suppression : {len(stations_hydroreel)}")
        stations_hydroreel = stations_hydroreel.loc[~stations_hydroreel['station'].isin(stations_a_ignorer)]
        logger.info(f"Taille du dataframe après suppression : {len(stations_hydroreel)}")
        stations_hydroreel.to_csv(dir_hydroreel / 'stations_hydroreel.csv', sep=';', index=False)

        url_initial = "https://hubeau.eaufrance.fr/api/v1/hydrometrie/observations_tr?"
        url_outputs = "&pretty&grandeur_hydro=H&fields=code_station,date_obs,resultat_obs,continuite_obs_hydro"
        #url_outputsQ = "&pretty&grandeur_hydro=Q&fields=code_station,date_obs,resultat_obs,continuite_obs_hydro"

        total = []
        for index, station in enumerate(sorted(stations_hydroreel.station)):
            logger.info(f" loading {station}")
            try:
                df = pd.read_csv(dir_hydroreel / (station+'.csv'))
            except Exception as e:
                print(e)
                continue
            df['station_hydroreel'] = station
            if 'Date' not in df.columns:
                continue
            df['Date'] = df['Date'].apply(hydroreel_date_parser)    
            hmin = 0
            hmax = 100
            NUM_MAX = 5000

            # Compter les valeurs en dessous de hmin
            count_below_hmin = (df['H (cm)'] < hmin).sum()
            # Compter les valeurs au-dessus de hmax
            count_above_hmax = (df['H (cm)'] > hmax).sum()

            # Si le nombre de points inférieurs à hmin est inférieur à NUM_MAX
            if count_below_hmin < NUM_MAX:
                # Interpolation des valeurs inférieures à hmin
                df.loc[df['H (cm)'] < hmin, 'H (cm)'] = np.nan
                df['H (cm)'] = df['H (cm)'].interpolate()
            else:
                # Ignorer la station si le nombre de points est trop élevé
                continue

            # Si le nombre de points supérieurs à hmax est inférieur à NUM_MAX
            if count_above_hmax < NUM_MAX:
                # Interpolation des valeurs supérieures à hmax
                df.loc[df['H (cm)'] > hmax, 'H (cm)'] = np.nan
                df['H (cm)'] = df['H (cm)'].interpolate()
            else:
                # Ignorer la station si le nombre de points est trop élevé
                continue

            total.append(df)

        if len(total) == 0:
            logger.info('No data available')
            return
        
        total = pd.concat(total).reset_index(drop=True)
        total['creneau'] = total['Date'].apply(lambda x : hydroreel_date_parser(x).split(' ')[0])
        total['date'] = total['Date'].apply(lambda x : hydroreel_date_parser(x))
        total['hour'] = total['Date'].apply(lambda x : x.split(' ')[1])
        total.rename({'H (cm)' : 'hauteur'}, inplace=True, axis=1)
        total = total[total['hour'] == '12:00:00']
        total16 = total[total['hour'] == '16:00:00']
        print(total)
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
                if len(dg) > 0:
                    dg = pd.concat(dg)
                    dg.to_csv(dir_nappes / 'liste_stations.csv', index=False)
                else:
                    dg = pd.DataFrame(columns=['longitude','latitude','code_bss'])
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
        if len(df) == 0:
            logger.info('No data')
            df = pd.DataFrame(index=np.arange(0, len(self.dates)))
            df['profondeur_nappe'] = np.nan
            df['niveau_nappe_eau'] = np.nan
            df['code_bss'] = np.nan
            df['date_mesure'] = self.dates
        else:
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
            if (self.meteostatParams['dir'] / 'meteostat.csv').is_file():
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

def launch(departement, resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, 
           compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop):
    
    #dir_data = Path('/home/caron/Bureau/csv') / departement / 'data'
    dir_data_disk = Path('/media/caron/X9 Pro/travaille/Thèse') / 'csv' / departement / 'data'
    dir_data = dir_data_disk 
    #dir_raster = Path('/home/caron/Bureau/csv') / departement / 'raster'
    dir_raster =  Path('/media/caron/X9 Pro/travaille/Thèse') / 'csv' / departement / 'raster' / resolution
    
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
    
    if not (dir_data / 'geo/geo.geojson').is_file():
        download_region(departement, dir_data / 'geo')

    region_path = dir_data / 'geo/geo.geojson'
    region = gpd.read_file(region_path)

    if not (dir_data / 'spatial/hexagones.geojson').is_file():
        download_hexagones(Path('/home/caron/Bureau/csv/france/data/geo'), region, dir_data / 'spatial')

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
    stop = '2024-06-29'
    
    ################## Ain ######################
    """launch('departement-01-ain', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Aisne ######################
    launch('departement-02-aisne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Allier ######################
    launch('departement-03-allier', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Alpes-de-Haute-Provence ######################
    launch('departement-04-alpes-de-haute-provence', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Hautes-Alpes ######################
    launch('departement-05-hautes-alpes', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Alpes-Maritimes ######################
    launch('departement-06-alpes-maritimes', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Ardeche ######################
    launch('departement-07-ardeche', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Ardennes ######################
    launch('departement-08-ardennes', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Ariege ######################
    launch('departement-09-ariege', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Aube ######################
    launch('departement-10-aube', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Aude ######################
    launch('departement-11-aude', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Aveyron ######################
    launch('departement-12-aveyron', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Bouches-du-Rhone ######################
    launch('departement-13-bouches-du-rhone', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start,   stop)
    
    ################## Calvados ######################
    launch('departement-14-calvados', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Cantal ######################
    launch('departement-15-cantal', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Charente ######################
    launch('departement-16-charente', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Charente-Maritime ######################
    launch('departement-17-charente-maritime', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    """
    ################## Cher ######################
    launch('departement-18-cher', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Correze ######################
    launch('departement-19-correze', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Cote-d-Or ######################
    launch('departement-21-cote-d-or', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Cotes-d-Armor ######################
    launch('departement-22-cotes-d-armor', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Creuse ######################
    launch('departement-23-creuse', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Dordogne ######################
    launch('departement-24-dordogne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Doubs ######################
    launch('departement-25-doubs', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Drome ######################
    launch('departement-26-drome', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Eure ######################
    launch('departement-27-eure', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Eure-et-Loir ######################
    launch('departement-28-eure-et-loir', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Finistere ######################
    launch('departement-29-finistere', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Corse-du-Sud ######################
    #launch('departement-2A-corse-du-sud', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Haute-Corse ######################
    #launch('departement-2B-haute-corse', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Gard ######################
    launch('departement-30-gard', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Haute-Garonne ######################
    launch('departement-31-haute-garonne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Gers ######################
    launch('departement-32-gers', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Gironde ######################
    launch('departement-33-gironde', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Herault ######################
    launch('departement-34-herault', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Ille-et-Vilaine ######################
    launch('departement-35-ille-et-vilaine', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Indre ######################
    launch('departement-36-indre', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Indre-et-Loire ######################
    launch('departement-37-indre-et-loire', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Isere ######################
    launch('departement-38-isere', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Jura ######################
    launch('departement-39-jura', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Landes ######################
    launch('departement-40-landes', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Loir-et-Cher ######################
    launch('departement-41-loir-et-cher', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Loire ######################
    launch('departement-42-loire', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Haute-Loire ######################
    launch('departement-43-haute-loire', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Loire-Atlantique ######################
    launch('departement-44-loire-atlantique', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Loiret ######################
    launch('departement-45-loiret', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Lot ######################
    launch('departement-46-lot', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Lot-et-Garonne ######################
    launch('departement-47-lot-et-garonne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Lozere ######################
    launch('departement-48-lozere', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Maine-et-Loire ######################
    launch('departement-49-maine-et-loire', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Manche ######################
    launch('departement-50-manche', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Marne ######################
    launch('departement-51-marne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Haute-Marne ######################
    launch('departement-52-haute-marne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Mayenne ######################
    launch('departement-53-mayenne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Meurthe-et-Moselle ######################
    launch('departement-54-meurthe-et-moselle', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Meuse ######################
    launch('departement-55-meuse', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Morbihan ######################
    launch('departement-56-morbihan', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Moselle ######################
    launch('departement-57-moselle', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Nievre ######################
    launch('departement-58-nievre', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Nord ######################
    launch('departement-59-nord', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Oise ######################
    launch('departement-60-oise', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Orne ######################
    launch('departement-61-orne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Pas-de-Calais ######################
    launch('departement-62-pas-de-calais', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Puy-de-Dome ######################
    launch('departement-63-puy-de-dome', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Pyrenees-Atlantiques ######################
    launch('departement-64-pyrenees-atlantiques', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Hautes-Pyrenees ######################
    launch('departement-65-hautes-pyrenees', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Pyrenees-Orientales ######################
    launch('departement-66-pyrenees-orientales', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Bas-Rhin ######################
    launch('departement-67-bas-rhin', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Haut-Rhin ######################
    launch('departement-68-haut-rhin', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Rhone ######################
    launch('departement-69-rhone', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Haute-Saone ######################
    launch('departement-70-haute-saone', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Saone-et-Loire ######################
    launch('departement-71-saone-et-loire', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Sarthe ######################
    launch('departement-72-sarthe', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Savoie ######################
    launch('departement-73-savoie', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Haute-Savoie ######################
    launch('departement-74-haute-savoie', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Paris ######################
    launch('departement-75-paris', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Seine-Maritime ######################
    launch('departement-76-seine-maritime', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Seine-et-Marne ######################
    launch('departement-77-seine-et-marne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Yvelines ######################
    launch('departement-78-yvelines', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Deux-Sevres ######################
    launch('departement-79-deux-sevres', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Somme ######################
    launch('departement-80-somme', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Tarn ######################
    launch('departement-81-tarn', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Tarn-et-Garonne ######################
    launch('departement-82-tarn-et-garonne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Var ######################
    launch('departement-83-var', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Vaucluse ######################
    launch('departement-84-vaucluse', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Vendee ######################
    launch('departement-85-vendee', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Vienne ######################
    launch('departement-86-vienne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Haute-Vienne ######################
    launch('departement-87-haute-vienne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Vosges ######################
    launch('departement-88-vosges', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Yonne ######################
    launch('departement-89-yonne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Territoire-de-Belfort ######################
    launch('departement-90-territoire-de-belfort', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Essonne ######################
    launch('departement-91-essonne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Hauts-de-Seine ######################
    launch('departement-92-hauts-de-seine', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Seine-Saint-Denis ######################
    launch('departement-93-seine-saint-denis', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Val-de-Marne ######################
    launch('departement-94-val-de-marne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Val-d-Oise ######################
    launch('departement-95-val-d-oise', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    """
    ################## Guadeloupe ######################
    launch('departement-971-guadeloupe', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Martinique ######################
    launch('departement-972-martinique', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Guyane ######################
    launch('departement-973-guyane', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## La-Reunion ######################
    launch('departement-974-la-reunion', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Mayotte ######################
    launch('departement-976-mayotte', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)"""