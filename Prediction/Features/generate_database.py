from tools import *
from config import *
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
                 computeMeteoStat, meteostatParams,
                 computeTemporal, outputParams,
                 addSpatial, spatialParams,
                 addAir, airParams,
                 addBouchon, bouchonParams,
                 addVigicrue, vigicrueParams,
                 region, h3,
                 dir_raster
                 ):
        
        self.departement = departement

        self.computeMeteoStat = computeMeteoStat
        self.meteostatParams = meteostatParams

        self.computeTemporal = computeTemporal
        self.outputParams = outputParams

        self.addSpatial = addSpatial
        self.spatialParams = spatialParams

        self.addAir = addAir
        self.airParams = airParams

        self.addBouchon = addBouchon
        self.bouchonParams = bouchonParams

        self.addVigicrue = addVigicrue
        self.vigicrueParams = vigicrueParams

        self.dir_raster = dir_raster

        self.region = region

        self.h3 = h3

        self.elevation, self.lons, self.lats = read_tif(self.spatialParams['dir'] / 'elevation' / self.spatialParams['elevation_file'])

    def compute_meteo_stat(self):
        logger.info('Compute compute_meteo_stat')
        self.meteostat = construct_historical_meteo(self.meteostatParams['start'], self.meteostatParams['end'],
                                                    self.region, self.meteostatParams['dir'])
        
        self.meteostat['year'] = self.meteostat['creneau'].apply(lambda x : x.split('-')[0])
        self.meteostat.to_csv(self.meteostatParams['dir'] / 'meteostat.csv', index=False)

    def compute_temporal(self):
        logger.info('Compute compute_temporal')
        rasterise_meteo_data(self.clusterSum, self.h3tif, self.meteostat, self.h3tif.shape, self.dates, self.dir_raster)

    def add_spatial(self):
        logger.info('Add spatial')
        #raster_sat(self.h3tif, self.spatialParams['dir_sat'], self.dir_raster, self.dates)
        #raster_land(self.h3tif, self.h3tif_high, self.spatialParams['dir_sat'], self.dir_raster, self.dates)
        #raster_population(self.h3tif, self.h3tif_high, self.dir_raster, self.resLon, self.resLat, self.spatialParams['dir'])
        #raster_elevation(self.h3tif, self.dir_raster, self.elevation)
        #raster_osmnx(self.h3tif, self.h3tif_high, self.dir_raster, self.resLon, self.resLat, self.spatialParams['dir'])
        raster_foret(self.h3tif, self.h3tif_high, self.dir_raster, self.resLon_high, self.resLat_high, self.spatialParams['dir'], self.departement)

    def add_air_qualite(self):
        assert RASTER == True
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

        if not (self.airParams['dir'] / 'air' / 'Air_archive').is_dir():
            token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiR2VvZGFpclJlc3RBUEkiXSwic2NvcGUiOlsicmVhZCJdLCJleHAiOjE3MTI4NzA1NjAsImF1dGhvcml0aWVzIjpbIlNFUlZJQ0UiXSwianRpIjoiZTI5YmJjOGItODY2Ny00ODcxLTk5YzUtYjE1NGM4NWJiNTIzIiwiY2xpZW50X2lkIjoiR2VvZGFpckZyb250ZW5kQ2xpZW50In0.J6axigDBoJY4kmMLbQfUwl-MCudxF5cm43R1WTXtt7Q'
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
                for annee in range(2017, dt.datetime.now().year + 1, 2):
                    logger.info(f"   année : {annee}")
                    if not (self.airParams['dir'] / 'air' / 'Air_archive' / f"{polluant}_{annee}_{annee+2}.csv").is_file():
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
        if not (self.airParams['dir'] / 'air' / 'archive' / 'polluants1216.csv').is_file():

            polluants_csv = []
            for polluant in polluants:
                for annee in range(2017, dt.datetime.now().year + 1, 2):
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

        START = dt.datetime.strptime(self.vigicrueParams['start'], '%Y-%m-%d') 
        STOP = dt.datetime.strptime(self.vigicrueParams['end'], '%Y-%m-%d') 
        dir_logs = Path('./')
        dir_hydroreel = self.vigicrueParams['dir'] / 'hydroreel'
        rname = 'stations_'+self.departement.split('-')[1]+'_full.csv'
        fic_name = dir_hydroreel / rname

        def hydroreel_date_parser(chaine):
            try:
                return dt.datetime.strptime(chaine, "%d/%m/%Y %H:%M").strftime("%Y-%m-%d %H:%M:%S")
            except:
                #'2018-01-01 00:00:00+00:00'
                return dt.datetime.strptime(chaine, "%Y-%m-%d %H:%M:%S+00:00").strftime("%Y-%m-%d %H:%M:%S")

        logger.info('Collecting Hydroreel stations list')

        stations_hydroreel = pd.read_csv(fic_name, sep=',', dtype={'departement': str})
        stations_a_ignorer = []
        for station in sorted(list(stations_hydroreel.station)):
            #if not (dir_hydroreel / f"{station}.csv").is_file():
            if True:
                logger.info(f"On collecte l'archive de {station}")
                df = pd.DataFrame()
                for annee in range(START.year, dt.datetime.now().year + 1):
                    url=f"https://www.hydro.eaufrance.fr/stationhydro/ajax/{station}/series?hydro_series%5BstartAt%5D=01%2F01%2F{annee}&hydro_series%5BendAt%5D=01%2F01%2F{annee+1}&hydro_series%5BvariableType%5D=simple_and_interpolated_and_hourly_variable&hydro_series%5BsimpleAndInterpolatedAndHourlyVariable%5D=H&hydro_series%5BstatusData%5D=raw"
                    r = requests.get(url)
                    if r.status_code == 200:
                        logger.info(f" -> Des données ont été collectées pour l'année {annee}")
                        dg = pd.DataFrame.from_dict(json.loads(r.text)['series'])
                        dg = dg.loc[dg.metric=='H']
                        logger.debug(f"    Taille de la série : {len(dg)}")
                        if len(dg) == 0:
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
                    else:
                        logger.info(f" -> Erreur : code {r.status_code} pour la station {station}")
                        #stations_a_ignorer.append(station)
                        break
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
            logger.info(f" * on tente de mettre à jour {station}") 
            df = pd.read_csv(dir_hydroreel / (station+'.csv'))
            Date_debut = max(df['Date'])
            Date_debut += dt.timedelta(hours=1)
            Date_fin = min(dt.datetime.now().replace(minute=0, second=0, microsecond=0),
                        Date_debut + dt.timedelta(days=3))
            with open(dir_hydroreel / (station + '.csv'), 'a') as f:
                cpt = 0
                while Date_fin != Date_debut:
                    cpt += 1
                    if cpt == 10:
                        break
                    date_debut = Date_debut.strftime('%Y-%m-%dT%H:%M:%S')
                    date_fin = Date_fin.strftime('%Y-%m-%dT%H:%M:%S')[:-5] + "00:00"
                    logger.info(f"  - start of records: {date_debut}")
                    logger.info(f"  - end of records: {date_fin}")
                    url_parameters = "code_entite=" + station + "&date_debut_obs=" + date_debut + "&date_fin_obs=" + date_fin
                    url = url_initial + url_parameters + url_outputs
                    logger.info("  - sending request to the hydroreel website")
                    logger.info(f"  - URL requested: {url}")
                    try:
                        response = requests.get(url)
                        text = response.json()
                        if 'data' in text and len(text['data'])>0:
                            logger.info("  - some data have been received")
                            for data in text['data'][::-1]:
                                f.write(
                                f"{dt.datetime.strptime(data['date_obs'], '%Y-%m-%dT%H:%M:%SZ').strftime('%d/%m/%Y %H:%M')},{data['resultat_obs'] / 10}" + os.linesep)
                        else:
                            logger.info("  - nothing received")
                    except Exception as e:
                        logger.error(f"Erreur {e} pour cette station")
                        with open(dir_logs / 'probleme_de_connexion.csv', 'a') as f:
                            f.write(f'{dt.datetime.now()},{url}\n')
                    Date_debut = Date_fin
                    Date_fin = min(dt.datetime.now().replace(minute=0, second=0, microsecond=0),
                                Date_debut + dt.timedelta(days=3))

            total.append(pd.read_csv(dir_hydroreel / (station+'.csv')))

        total = pd.concat(total).reset_index(drop=True)
        total['Date'] = total['Date'].apply(lambda x : hydroreel_date_parser(x))
        total['hour'] = total['Date'].apply(lambda x : x.split(' ')[1])
        total['Date'] = total['Date'].apply(lambda x : x.split(' ')[0])
        total.rename({'Date' : 'creneau', 'H (cm)' : 'hauteur'}, inplace=True, axis=1)
        total = total[total['hour'] == '12:00:00']
        total16 = total[total['hour'] == '16:00:00']

        rasterise_vigicrues(self.clusterSum, self.h3tif, total, self.h3tif.shape, self.dates, self.dir_raster, 'vigicrues12')
        rasterise_vigicrues(self.clusterSum, self.h3tif, total16, self.h3tif.shape, self.dates, self.dir_raster, 'vigicrues16')

        logger.info('Constructing Hydroreel dataframe')
        df_hydro = pd.DataFrame()
        moyennes = {}
        for Hydro_station in dir_hydroreel.iterdir():
            if Hydro_station.suffix == '.csv' and Hydro_station.stem != "stations_hydroreel":
                hydro_station = Hydro_station.stem
                if hydro_station in list(stations_hydroreel.station):
                    logger.info(f"  - Integrating {hydro_station} file")
                    df = pd.read_csv(Hydro_station, parse_dates=['Date'], date_parser=hydroreel_date_parser)
                    df1 = df.groupby(pd.Grouper(key='Date', freq='H')).size()
                    df2 = df.groupby(pd.Grouper(key='Date', freq='H')).mean()
                    df = pd.concat([df2, df1], axis=1)
                    nom = stations_hydroreel.loc[stations_hydroreel.station == hydro_station].nom.iloc[0]
                    df.columns = [nom+'_height', nom+'_measures']
                    df_hydro = pd.concat([df_hydro, df], axis=1)
                    moyennes[hydro_station] = df[nom+'_height'].mean()
                else:
                    logger.info(f"Je ne trouve pas {hydro_station}")

        del df
        del df1
        del df2

        stations_hydroreel['Moyenne'] = stations_hydroreel['station'].map(moyennes)
        stations_hydroreel.to_csv(dir_hydroreel / 'stations_hydroreel.csv', sep=';', index=False)

        logger.info('Filling NAN in Hydroreel dataframe')
        df_hydro.fillna(method='ffill', inplace=True)
        df_hydro = df_hydro[START:STOP]

        logger.info(f"On sauvegarde le dataframe hydroréel, de taille {df_hydro.shape}")

        df_hydro['creneau'] = df_hydro.index
    
    def compute_bouchon(self):
        
        for i, date in enumerate(self.dates):
            logger.info(date)
            bison_urls = ["https://tipi.bison-fute.gouv.fr/bison-fute-ouvert/publicationsDIR/Evenementiel-DIR/cnir/RecapBouchonsFranceEntiere.html",
                        "https://tipi.bison-fute.gouv.fr/bison-fute-ouvert/publicationsDIR/Evenementiel-DIR/cnir/RecapTraficFranceEntiere.html"]

            dir_trafic = self.bouchonParams['dir'] / 'trafic'
            (dir_trafic).mkdir(exist_ok=True, parents=True)
            (dir_trafic / 'raw').mkdir(exist_ok=True, parents=True)

            keys = ['nature', 'horodate', 'axe', 'sens_cardinal', 'point_repere', 'longueur_pr', 'commune']
            qnames = ['nature', 'ligne_horodate_fin_exception_ve', 'axe', 'sens_cardinal', 'pr', 'longueur', 'commune']

            def process_data(next_element, keys, qnames, importance, department_name, df):
                data = {}

                for key, qname in zip(keys, qnames):
                    try:
                        element = next_element.find('span', {'qname': qname})
                        element = next_element.find('span', class_=qname) if not element else element # if qname is not found, try with class
                        data[key] = element.get_text(strip=True) if element else None
                    except AttributeError:
                        data[key] = None

                data['date'] = date
                data['department'] = department_name
                data['importance'] = len(importance)

                new_row_df = pd.DataFrame(data, index=[0])
                updated_df = pd.concat([df, new_row_df], ignore_index=True)

                return updated_df

            for index, bison_url in enumerate(bison_urls):
                html = requests.get(bison_url)

                bsobj = BeautifulSoup(html.content, 'html.parser')

                df = pd.DataFrame()

                departments = bsobj.find_all('span', class_='rupture')
                initial_rows = df.shape[0]

                for department in departments:
                    department_name = department.find('a').get_text(strip=True)
                    if department_name.split(' ')[1] != self.departement.split('-')[1]:
                        continue
                    next_element = department.find_next_sibling()
                    while next_element and next_element.name != 'span' and 'rupture' not in next_element.get('class', []):
                        if next_element.name == 'div' and 'interligne' in next_element['class'][0]:
                            elements = next_element.find_all('span', {'qname': 'element'})
                            for element in elements:
                                importance = next_element.find('span', {'qname': 'importance_vr_reduit'}).get_text(strip=True)
                                df = process_data(element, keys, qnames, importance, department_name, df)

                        next_element = next_element.find_next_sibling()

                df.drop_duplicates(inplace=True) # I don't want to add elements I already have in my df

                df.to_csv(dir_trafic / f'raw/bison_fute_{"trafic" if "Trafic" in bison_url else "bouchons"}_raw_{date}.csv', index=False)

                df.fillna(value=np.nan, inplace=True)
                df['sens_cardinal'] = df['sens_cardinal'].str.replace(r'^\(|sens(?=\s)', '', regex=True).str.strip()
                df['longueur_pr'] = df['longueur_pr'].str.replace(' environ', '')
                df['longueur_pr'].fillna('', inplace=True)
                df['longueur_pr'] = df['longueur_pr'].apply(lambda x: x.replace("(sur ", ''))
                df['longueur_pr'] = df['longueur_pr'].apply(lambda x: x.replace(' km)', ''))
                df['longueur_pr'] = df['longueur_pr'].str.replace('de longueur indéterminée', 'nan')
                df['longueur_pr'] = df['longueur_pr'].str.replace(',', '.')
                df['longueur_pr'] = df['longueur_pr'].apply(lambda x: float(x) if x not in ['nan', ''] else np.nan)
                df = df.rename(columns={'longueur_pr': 'longueur_pr (km)'})

                df['department_number'] = df['department'].str.extract(r'Département (\d+)')
                df['department_name'] = df['department'].str.extract(r'\(([^()]+)\)')
                df.drop(columns=['department'], inplace=True)

                new_order = ['date', 'department_number', 'department_name'] + [col for col in df.columns if col not in ['date', 'department_number', 'department_name']]
                df = df[new_order]

                df.to_csv(dir_trafic / f'bison_fute_{"trafic" if "Trafic" in bison_url else "bouchons"}_{date}.csv', index=False)
                if index == 0:
                    dg = df.loc[(df.department_number==department)&(df.importance>=2)]
                    dg['news'] = dg.apply(lambda x: f"{x['nature']} de {x['longueur_pr (km)']} km sur l'axe {x['axe']}, {x['commune']}", axis=1)
                else:
                    liste = ['Accident', 'Incendie', 'Manifestation sociale', 'Effondrement', 'Éboulement',
                            'Glissement de terrain', 'Animal errant', 'Produit sur chaussée',
                            'Véhicule en panne', 'Obstacle', 'Véhicule abandonné', 'Véhicule arrêté']
                    dg = df.loc[(df.nature.isin(liste))&(df.importance>=2)&(df.department_number==department)]
                    dg['news'] = dg.apply(lambda x: f"{x['nature']} {x['commune']}, axe {x['axe']}", axis=1)
                news_trafic = list(dg.news.values)                

    def compute_nappes_phréatique(self):
        pass

    def process(self, start, stop):
        logger.info(self.departement)
        
        if self.computeMeteoStat:
            self.compute_meteo_stat()
        else:
            self.meteostat = pd.read_csv(self.meteostatParams['dir'] / 'meteostat.csv')

        self.h3['latitude'] = self.h3['geometry'].apply(lambda x : float(x.centroid.y))
        self.h3['longitude'] = self.h3['geometry'].apply(lambda x : float(x.centroid.x))
        self.h3['altitude'] = 0

        if RASTER:
            self.clusterSum = self.h3.copy(deep=True)
            self.clusterSum['cluster'] = self.clusterSum.index.values.astype(int)
            self.cluster = None
            sdate = start
            edate = stop
            self.dates = find_dates_between(sdate, edate)

            #n_pixel_x = 0.016133099692723363
            #n_pixel_y = 0.016133099692723363

            n_pixel_x = 0.02875215641173088
            n_pixel_y = 0.020721094073767096

            self.resLon = n_pixel_x
            self.resLat = n_pixel_y
            self.h3tif = rasterisation(self.clusterSum, n_pixel_y, n_pixel_x, column='cluster', defval=np.nan, name=self.departement+'_low')
            logger.info(f'Low scale {self.h3tif.shape}')

            n_pixel_x = 0.0002694945852326214
            n_pixel_y = 0.0002694945852352859

            self.resLon_high = n_pixel_x
            self.resLat_high = n_pixel_y
            self.h3tif_high = rasterisation(self.clusterSum, n_pixel_y, n_pixel_x, column='cluster', defval=np.nan, name=self.departement+'_high')
            logger.info(f'High scale {self.h3tif_high.shape}')

        if self.computeTemporal:
            self.compute_temporal()

        if self.addSpatial:
            self.add_spatial()

        if self.addAir:
            self.add_air_qualite()

        if self.addBouchon:
            self.compute_bouchon()
        
        if self.addVigicrue:
            self.compute_hauteur_riviere()

def launch(departement, computeMeteoStat, computeTemporal, addSpatial, 
           addAir, addBouchon, addVigicrue):

    dir_data = root / departement / 'data'
    dir_meteostat = dir_data / 'meteostat'
    dir_raster = root / departement / 'raster'
    check_and_create_path(dir_raster)

    start = '2017-06-12'
    stop = '2024-05-16'

    meteostatParams = {'start' : '2016-01-01',
                    'end' : stop,
                    'dir' : dir_meteostat}

    outputParams  = {'start' : start,
                    'end' : stop}
    

    spatialParams = {'dir_sat':  rootDisk / departement / 'data',
                    'dir' : dir_data,
                    'elevation_file' : 'elevation.tif',
                    'sentBand': ['NDVI', 'NDMI', 'NDBI', 'NDSI', 'NDWI'],
                    'landCoverBand' : ['0.0','1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0'],
                    'addsent': True,
                    'addland': True}
    
    airParams = {'dir' : dir_data}

    bouchonParams = {'dir' : dir_data}

    vigicrueParams = {'dir' : dir_data,
                      'start' : start,
                        'end' : stop}

    region_path = dir_data / 'geo/geo.geojson'
    region = json.load(open(region_path))
    polys = [region["geometry"]]
    geom = [shape(i) for i in polys]
    region = gpd.GeoDataFrame({'geometry':geom})

    h3 = gpd.read_file(dir_data / 'spatial/hexagones.geojson')

    database = GenerateDatabase(departement,
                    computeMeteoStat, meteostatParams,
                    computeTemporal, outputParams,
                    addSpatial, spatialParams,
                    addAir, airParams,
                    addBouchon, bouchonParams,
                    addVigicrue, vigicrueParams,
                    region, h3,
                    dir_raster)

    database.process(start, stop)

if __name__ == '__main__':
    RASTER = True

    parser = argparse.ArgumentParser(
        prog='Train',
        description='Create graph and database according to config.py and tained model',
    )
    parser.add_argument('-m', '--meteostat', type=str, help='Compute meteostat')
    parser.add_argument('-t', '--temporal', type=str, help='temporal')
    parser.add_argument('-s', '--spatial', type=str, help='spatial')
    parser.add_argument('-a', '--air', type=str, help='air')
    parser.add_argument('-v', '--vigicrues', type=str, help='Vigicrues')

    args = parser.parse_args()

    # Input config
    computeMeteoStat = args.meteostat == "True"
    computeTemporal = args.temporal == "True"
    addSpatial = args.spatial == "True"
    addAir = args.air == 'True'
    addBouchon = False
    addVigicrue = args.vigicrues == "True"

    ################## Ain ######################
    launch('departement-01-ain', computeMeteoStat, computeTemporal, addSpatial, addAir, addBouchon, addVigicrue)

    ################## DOUBS ######################
    launch('departement-25-doubs', computeMeteoStat, computeTemporal, addSpatial, addAir, addBouchon, addVigicrue)

    ################## YVELINES ######################
    launch('departement-78-yvelines', computeMeteoStat, computeTemporal, addSpatial, addAir, addBouchon, addVigicrue)

    ################## Rhone ######################
    launch('departement-69-rhone', computeMeteoStat, computeTemporal, addSpatial, addAir, addBouchon, addVigicrue)