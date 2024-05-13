from tools import *
from config import *

class GenerateDatabase():
    def __init__(self, departement,
                 computeMeteoStat, meteostatParams,
                 computeTemporal, outputParams,
                 addSpatial, spatialParams,
                 addAir, airParams,
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

        self.dir_raster = dir_raster

        self.region = region

        self.h3 = h3

        self.elevation, self.lons, self.lats = read_tif(self.spatialParams['dir'] / 'elevation' / self.spatialParams['elevation_file'])

    def compute_meteo_stat(self):
        print('Compute compute_meteo_stat')
        self.meteostat = construct_historical_meteo(self.meteostatParams['start'], self.meteostatParams['end'],
                                                    self.region, self.meteostatParams['dir'])
        
        self.meteostat['year'] = self.meteostat['creneau'].apply(lambda x : x.split('-')[0])
        self.meteostat['cluster'] = self.meteostat.apply(lambda x :get_cluster(x['latitude'], x['longitude'], x['altitude'], self.cluster, None), axis=1)
        self.meteostat.to_csv(self.meteostatParams['dir'] / 'meteostat.csv', index=False)

    def compute_temporal(self):
        print('Compute compute_temporal')
        rasterise_meteo_data(self.clusterSum, self.h3tif, self.meteostat, self.h3tif.shape, self.dates, self.dir_raster)

    def add_spatial(self):
        print('Add spatial')

        raster_sat(self.h3tif, self.spatialParams['dir_sat'], self.dir_raster, self.dates)
        raster_population(self.h3tif, self.dir_raster, self.resLon, self.resLat, self.spatialParams['dir_sat'])
        raster_elevation(self.h3tif, self.dir_raster, self.elevation, self.spatialParams['dir_sat'])
        raster_osmnx(self.h3tif, self.dir_raster, self.resLon, self.resLat, self.spatialParams['dir_sat'])
        raster_foret(self.h3tif, self.dir_raster, self.resLon, self.resLat, self.spatialParams['dir_sat'])

    def add_air_quailte(self):
        assert RASTER == True
        air_stations = pd.read_csv(self.air_params['dir'] / 'air' / 'ExportStations.csv', sep=';')
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

        if not (self.air_params['dir'] / 'air' / 'Air_archive').is_dir():
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
                print(f" - polluant : {polluant}")
                for annee in range(2017, dt.datetime.now().year + 1, 2):
                    print(f"   année : {annee}")
                    if not (self.air_params['dir'] / 'air' / 'Air_archive' / f"{polluant}_{annee}_{annee+2}.csv").is_file():
                        headers1 = headers.copy()
                        headers1['Content-Type']= 'multipart/form-data; boundary=---------------------------14490471821690778178105310701'
                        headers1['Origin'] ='https://www.geodair.fr'
                        data = f'''-----------------------------14490471821690778178105310701\r\nContent-Disposition: form-data; name="polluant"\r\n\r\n{polluants[polluant]}\r\n-----------------------------14490471821690778178105310701\r\nContent-Disposition: form-data; name="region"\r\n\r\n\r\n-----------------------------14490471821690778178105310701\r\nContent-Disposition: form-data; name="departement"\r\n\r\n{self.departement.split('-')[1]}\r\n-----------------------------14490471821690778178105310701\r\nContent-Disposition: form-data; name="typologie"\r\n\r\n\r\n-----------------------------14490471821690778178105310701\r\nContent-Disposition: form-data; name="influence"\r\n\r\n\r\n-----------------------------14490471821690778178105310701\r\nContent-Disposition: form-data; name="date"\r\n\r\n{annee}-01-01 00:00\r\n-----------------------------14490471821690778178105310701\r\nContent-Disposition: form-data; name="dateFin"\r\n\r\n{annee+2}-01-01 00:00\r\n-----------------------------14490471821690778178105310701--\r\n'''
                        response1 = requests.post('https://www.geodair.fr/priv/api/station/filtered-list', headers=headers1, data=data)
                        reponse = json.loads(response1.text)
                        print(reponse)
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
                                    (self.air_params['dir'] / 'air' / 'Air_archive').mkdir(exist_ok=True, parents=True)
                                    dg.to_csv(self.air_params['dir'] / 'air' / 'Air_archive' / f"{polluant}_{annee}_{annee+2}.csv", sep=';', index=False)
                                    break
                                else:
                                    print(response3)
                                    cpt+= 1
                                    time.sleep(1)
                                if cpt == 5:
                                    break
        if not (self.air_params['dir'] / 'air' / 'archive' / 'polluants1216.csv').is_file():

            polluants_csv = []
            for polluant in polluants:
                for annee in range(2017, dt.datetime.now().year + 1, 2):
                    po = pd.read_csv(self.air_params['dir'] / 'air' / 'Air_archive' / f"{polluant}_{annee}_{annee+2}.csv", sep=';')
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
            
            polluants_csv.to_csv(self.air_params['dir'] / 'air' / 'Air_archive' / 'polluants1216.csv', sep=',', index=False)

        else:
            polluants_csv = pd.read_csv(self.air_params['dir'] / 'air' / 'Air_archive' / 'polluants1216.csv')

        print(polluants_csv.columns)
        rasterise_air_qualite(self.clusterSum, self.h3tif, polluants_csv, self.h3tif.shape, self.dates, self.dir_raster)

    def process(self):
        print(self.departement)
        
        if self.computeMeteoStat:
            self.compute_meteo_stat()
        else:
            self.meteostat = pd.read_csv(self.meteostatParams['dir'] / 'meteostat.csv')
            self.meteostat.drop('cluster', inplace=True, axis=1)

        self.h3['latitude'] = self.h3['geometry'].apply(lambda x : float(x.centroid.y))
        self.h3['longitude'] = self.h3['geometry'].apply(lambda x : float(x.centroid.x))
        self.h3['altitude'] = 0

        if RASTER:
            self.clusterSum = self.h3.copy(deep=True)
            self.clusterSum['cluster'] = self.clusterSum.index.values.astype(int)
            self.cluster = None
            sdate = '2017-06-12'
            edate = '2023-09-12'
            self.dates = find_dates_between(sdate, edate)
            #n_pixel_x = 0.0002694945852326214
            #n_pixel_y = 0.0002694945852352859

            #n_pixel_x = 0.016133099692723363
            #n_pixel_y = 0.016133099692723363

            n_pixel_x = 0.02875215641173088
            n_pixel_y = 0.020721094073767096

            self.resLon = n_pixel_x
            self.resLat = n_pixel_y
            self.h3tif = rasterisation(self.clusterSum, n_pixel_y, n_pixel_x)
            print(self.h3tif.shape)

        if self.computeTemporal:
            self.compute_temporal()

        if self.addSpatial:
            self.add_spatial()

        if self.addAir:
            self.add_air_quailte()


def launch(departement):

    dir_data = root / departement / 'data'
    dir_meteostat = dir_data / 'meteostat'
    dir_raster = root / departement / 'raster'
    check_and_create_path(dir_raster)

    computeMeteoStat = False
    meteostatParams = {'start' : '2016-01-01',
                    'end' : '2023-12-04',
                    'dir' : dir_meteostat}

    computeTemporal = False
    outputParams  = {'start' : '2017-06-12',
                    'end' : '2023-09-11'}

    addSpatial = True
    spatialParams = {'dir_sat':  rootDisk / departement / 'data',
                    'dir' : dir_data,
                    'elevation_file' : 'elevation.tif',
                    'sentBand': ['NDVI', 'NDMI', 'NDBI', 'NDSI', 'NDWI'],
                    'landCoverBand' : ['0.0','1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0'],
                    'addsent': True,
                    'addland': True}
    
    addAir = False
    airParams = {'dir' : dir_data}

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
                    region, h3,
                    dir_raster)

    database.process()

if __name__ == '__main__':
    RASTER = True
    ################## Ain ######################
    launch('departement-01-ain')

    ################## DOUBS ######################
    launch('departement-25-doubs')

    ################## YVELINES ######################
    launch('departement-78-yvelines')

    ################## Rhone ######################
    launch('departement-69-rhone')