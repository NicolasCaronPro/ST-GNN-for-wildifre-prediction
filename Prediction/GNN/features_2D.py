from raster_utils import *
from config import *

def find_spatial_date(x, firepoint):
    return firepoint[firepoint['h3'] != x].sample(1)['date'].values[0]

def find_temporal_date(dates):
    return sample(dates, k = 1)[0]

def get_sub_nodes_feature_2D(graph, shape: int,
                            Xset : np.array,
                            scaling : str,
                            pos_feature_1D : dict,
                            departements : list,
                            features : list,
                            sinister : str,
                            path : Path,
                            prefix : str) -> tuple:
    
    # Assert that graph structure is build (not really important for that part BUT it is preferable to follow the api)
    assert graph.nodes is not None

    # Create output diretory
    dir_output = path / '2D'
    check_and_create_path(path)
    check_and_create_path(path / '2D')
    
    # Define pixel size
    n_pixel_y = 0.009
    n_pixel_x = 0.012

    # Define Features for 2D database
    pos_feature, newShape = create_pos_features_2D(shape, features)

    # Load mask
    dir_mask = path / 'raster' / '1x1'

    doMeteo = 'temp' in features
    doSent = 'sentinel' in features
    doLand = 'landcover' in features
    doForet = 'foret' in features
    doEle = 'elevation' in features
    doPop = 'population' in features
    doHighway = 'highway' in features
    doCalendar = 'Calendar' in features
    doGeo = 'Geo' in features
    doHistorical = 'Historical' in features
    doAir = 'air' in features
    doDW = 'dynamicWorld' in features
    doNappes = 'nappes' in features
    doVigi = 'vigicrues' in features
    doHighway = 'highway' in features

    dir_encoder = root_graph / path / 'Encoder'

    if doLand:
        encoder_landcover = read_object('encoder_landcover.pkl', dir_encoder)
        encoder_osmnx = read_object('encoder_osmnx.pkl', dir_encoder)
        encoder_foret = read_object('encoder_foret.pkl', dir_encoder)

    if doCalendar:
        size_calendar = len(calendar_variables)

    if doGeo:
        size_geo = len(geo_variables)
        
    # For each departement
    for departement in departements:
        logger.info(departement)
        # Define the directory for the data
        dir_data = rootDisk / departement / 'data'
        
        # Loading the mask where each pixel value are nodes from graph
        name = departement+'rasterScale'+str(graph.scale)+'.pkl'
        mask = pickle.load(open(dir_mask / name, 'rb'))

        # Get nodes from that department
        nodeDepartementMask = np.argwhere(Xset[:,3] == str2int[departement.split('-')[-1]])
        nodeDepartement = Xset[nodeDepartementMask].reshape(-1, Xset.shape[1])
        if nodeDepartementMask.shape[0] == 0:
            continue

        # Load meteostat
        if doMeteo:
            #meteostat = pd.read_csv(rootDisk / departement / 'data' / 'meteostat' / 'meteostat.csv')
            #for cem in cems_variables:
            #    meteostat[cem] = scaler(meteostat[cem].values, Xtrain[:, pos_feature_1D[cem] + 2], concat=False)
            pass

        # Load elevation raster
        if doEle:
            elevation, _, _ = read_tif(dir_data / 'elevation' / 'elevation.tif') # Read the tif file
            elevation = elevation[0]
            elevation = resize_no_dim(elevation, mask.shape[0], mask.shape[1]).astype(int) # Resize to the current shape
            save_object(elevation, 'elevation.pkl', dir_output)

        if doForet:
            check_and_create_path(dir_output / 'foret')
            foret = gpd.read_file(dir_data / 'BDFORET' / 'foret.geojson')
            foret, _, _ = rasterization(foret, n_pixel_y, n_pixel_x, 'code', defVal=0, dir_output=dir_output / 'foret')
            foret = foret[-1]
            foret = encoder_foret.transform(foret.reshape(-1,1)).values.reshape(foret.shape)
            outputName = 'foret.pkl'
            save_object(foret, outputName, dir_output)

        geo = gpd.read_file(dir_data / 'spatial' / 'hexagones.geojson')

         # Load population raster
        if doPop:
            check_and_create_path(dir_output / 'population') # Create temporal directory (nothign to do with meteo features)
            population, _, _ = rasterization(geo, n_pixel_y, n_pixel_x, 'population', dir_output / 'population') # Create a raster - > (1, H, W)
            population = population[0] # Get first band - > (H, W)
            save_object(population, 'population.pkl', dir_output)

        # Load highway raster
        if doHighway:
            check_and_create_path(dir_output / 'osmnx') # Create temporal directory (nothign to do with meteo features)
            osmnx, _, _ = rasterization(geo, n_pixel_y, n_pixel_x, 'osmnx', dir_output / 'osmnx') # Create a raster - > (1, H, W)
            osmnx = osmnx[0] # Get first band - > (H, W)
            save_object(osmnx, 'osmnx.pkl', dir_output)

        # Load landcover
        if doLand:
            landcover, _, _ = read_tif(rootDisk / departement / 'data' / 'GEE' / 'landcover' / 'summer.tif')
            landcover = landcover[-1]
            landcover[~np.isnan(landcover)] = np.round(landcover[~np.isnan(landcover)])
            landcover[np.isnan(landcover)] = -1
            landcover = encoder_landcover.transform(landcover.reshape(-1,1)).values.reshape(landcover.shape)
            landcover = (resize_no_dim(landcover, mask.shape[0], mask.shape[-1]))
            save_object(landcover, 'landcover.pkl', dir_output)

        if doSent:
            logger.info('Sentinel')

            summer, _, _ = read_tif(rootDisk / departement / 'data' / 'GEE' / 'sentinel' / 'summer.tif')
            
            for band in range(5):
                summer[band, np.isnan(summer[band])] = np.nanmean(summer[band])

            summer = resize(summer, mask.shape[0], mask.shape[-1], summer.shape[0])
            save_object(summer, 'summer.pkl', dir_output)

            winter, _, _ = read_tif(rootDisk / departement / 'data' / 'GEE' / 'sentinel' / 'winter.tif')

            for band in range(5):
                winter[band, np.isnan(winter[band])] = np.nanmean(winter[band])

            winter = resize(winter, mask.shape[0], mask.shape[-1], winter.shape[0])
            save_object(winter, 'winter.pkl', dir_output)

            autumn, _, _ = read_tif(rootDisk / departement / 'data' / 'GEE' / 'sentinel' / 'autumn.tif')

            for band in range(5):
                autumn[band, np.isnan(autumn[band])] = np.nanmean(autumn[band])

            autumn = resize(autumn, mask.shape[0], mask.shape[-1], autumn.shape[0])
            save_object(autumn, 'autumn.pkl', dir_output)

            spring, _, _ = read_tif(rootDisk / departement / 'data' / 'GEE' / 'sentinel' / 'spring.tif')

            for band in range(5):
                spring[band, np.isnan(spring[band])] = np.nanmean(spring[band])

            spring = resize(spring, mask.shape[0], mask.shape[-1], spring.shape[0])
            save_object(spring, 'spring.pkl', dir_output)

            logger.info(summer.shape, winter.shape, spring.shape, autumn.shape)

        Xk = list(zip(geo.longitude, geo.latitude))
        geo['scale'] = graph._predict_node(Xk)

        # This will take timmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmme
        for i, node in enumerate(nodeDepartement):
            if i % 500 == 0:
                logger.info(f'Node {i} / {nodeDepartement.shape} : {(node[:5])}')

            indD = (node[4]).astype(int)
            index_1D = np.argwhere((Xset[:,0] == node[0]) & (Xset[:,4] == node[4]))
    
            spatioTemporalRaster = np.full((shape2D[0], shape2D[1], newShape), np.nan, dtype=float)
            spatioTemporalRaster[:,:,0] = node[0]
            spatioTemporalRaster[:,:,1] = node[1]
            spatioTemporalRaster[:,:,2] = node[2]
            spatioTemporalRaster[:,:,3] = node[3]
            spatioTemporalRaster[:,:,4] = node[4]
            spatioTemporalRaster[:,:,5] = node[5]

            # Add historical
            if doHistorical:
                spatioTemporalRaster[:,:,pos_feature['Historical']] = Xset[index_1D, pos_feature_1D['Historical'] + 2]

            # Add calendar
            if doCalendar:
                for iv in range(size_calendar):
                    spatioTemporalRaster[:,:, pos_feature['Calendar'] + iv] = Xset[index_1D, pos_feature_1D['Calendar'] + iv]

            # Add geo
            if doGeo:
                for iv in range(size_geo):
                    spatioTemporalRaster[:,:, pos_feature['Geo'] + iv] = Xset[index_1D, pos_feature_1D['Geo'] + iv]

            # Add meteorological
            if doMeteo:
                #meteostatDate = meteostat[meteostat['creneau'] == allDates[indD]]
                #raster_meteo raster_meteo(spatioTemporalRaster, geo[geo['scale'] == node[0]], meteostatDate, \
                #pos_feature['temp], n_pixel_x, n_pixel_y, dir_output)
                for cems in cems_variables:
                    spatioTemporalRaster[:,:, pos_feature[cems]] = Xset[index_1D, pos_feature_1D[cems]]

            if doAir:
                for iv, pol in enumerate(air_variables):
                    spatioTemporalRaster[:,:, pos_feature['air'] + iv] = Xset[index_1D, pos_feature_1D['air'] + iv ]

            ########################################### This is time consuming ##########################################################
            # Add sentinel
            if doSent:
                #raster_sentinel(spatioTemporalRaster, allDates[indD], node[0], mask, rootDisk / departement / 'data', pos_feature['sentinel'])
                month = allDates[node[4].astype(int)].split('-')[1]
                if month >= '3' and month <= '5':
                    sent = spring
                elif month >= '6' and month <= '8':
                    sent = summer
                elif month >= '9' and month <= '11':
                    sent = autumn
                else:
                    sent = winter

                raster_sentinel_2(spatioTemporalRaster, node[0], mask, sent, pos_feature['sentinel'])

            # Add landcover
            if doLand:
                raster_landcover_2(spatioTemporalRaster, node[0], mask, landcover, pos_feature['landcover'], encoder_landcover)
                raster_foret(spatioTemporalRaster, mask, node[0], foret, pos_feature['foret'], encoder_foret)
                raster_osmnx(spatioTemporalRaster, mask, node[0], osmnx, pos_feature['highway'], encoder_osmnx)
            ############################################################################################################################

            # Add elevation
            if doEle:
                raster_elevation(spatioTemporalRaster, mask, node[0], elevation, pos_feature['elevation'])

            # Add population
            if doPop:
                raster_population(spatioTemporalRaster, mask, node[0], population, n_pixel_x, n_pixel_y, pos_feature['population'])

            if doForet:
                pass

            if doDW:
                pass

            if doHighway:
                pass

            if doNappes:
                pass

            if doVigi:
                pass
        
            # Save ouptut file
            save_object(spatioTemporalRaster, str(int(node[0]))+'_'+str(indD)+'.pkl', dir_output / prefix / 'data')

    # Return original subnode
    return pos_feature, newShape

if __name__ == '__main__':

    dir_output = 'temp/'
    check_and_create_path(dir_output)

    graph = read_object('graph_5.pkl', Path('train/'))
    regions = gpd.read_file('regions.geojson')

    n_pixel_y = 0.009
    n_pixel_x = 0.012

    fp = []
    nfp = []

    fp = pd.read_csv(dir_output / 'firepoints.csv')
    nfp = pd.read_csv(dir_output / 'nonfirepoints.csv')
    Xk = list(zip(fp.longitude, fp.latitude)) 
    fp['scale'] = graph._predict_node(Xk)

    Xk = list(zip(nfp.longitude, nfp.latitude)) 
    nfp['scale'] = graph._predict_node(Xk)

    for dept in departements:
        code = int(dept.split('-')[1])

        regions.loc[regions[regions['departement'] == (dept.split('-')[-1][0].upper()) + dept.split('-')[-1][1:]].index, 'departement'] = code
        sat0, lons, lats = rasterization(regions[regions['departement'] == code], n_pixel_y, n_pixel_x, 'scale', dir_output, dept+'_scale')
        sat0 = sat0[0]
        logger.info(sat0.shape)
        #sat0 = read_object(dept+'rasterScale5.pkl', GNN_PATH / 'test/2023/raster/')
        #sat0 = resize_no_dim(sat0, 104, 122).round(0)
        logger.info(np.unique(sat0))
        meteostat = pd.read_csv(rootDisk / dept / 'data' / 'meteostat' / 'meteostat.csv')

        """firepoint = pd.read_csv(root / dept / 'firepoint' / 'NATURELSfire.csv')
        firepoint['departement'] = code

        nonfirepointSpatial = regions[regions['departement'] == code].copy(deep=True)
        nonfirepointSpatial.drop(nonfirepointSpatial[nonfirepointSpatial['hex_id'].isin(firepoint['h3'])].index, inplace=True)
        nonfirepointSpatial['date'] = nonfirepointSpatial.apply(lambda x : find_spatial_date(x['hex_id'], firepoint), axis=1)
        nonfirepointSpatial['h3'] = nonfirepointSpatial['hex_id']

        allDates = find_dates_between(datesDico[dept]['start'], datesDico[dept]['end'])
        nonFireDates = [date  for date in allDates if date not in firepoint['date']]
        nonfirepointTemporal = firepoint.copy(deep=True)
        nonfirepointTemporal['date'] = nonfirepointTemporal.apply(lambda x : find_temporal_date(nonFireDates), axis=1)

        nonfirepoint = pd.concat((nonfirepointSpatial, nonfirepointTemporal)).reset_index(drop=True)

        Xk = list(zip(firepoint.longitude, firepoint.latitude)) 
        firepoint['scale'] = graph._predict_node(Xk)

        Xk = list(zip(nonfirepoint.longitude, nonfirepoint.latitude)) 
        nonfirepoint['scale'] = graph._predict_node(Xk)

        logger.info(len(nonfirepoint), len(firepoint))

        firepoint.drop_duplicates(subset=('date', 'scale'), inplace=True)
        nonfirepoint.drop_duplicates(subset=('date', 'scale'), inplace=True)

        firepoint.dropna(subset='scale', inplace=True)
        nonfirepoint.dropna(subset='scale', inplace=True)

        
        logger.info(len(nonfirepoint), len(firepoint))

        trainfp = firepoint[firepoint['date'] <= '2022-12-31']
        trainnfp = nonfirepoint[nonfirepoint['date'] <= '2022-12-31']
        
        test_date = find_dates_between('2023-01-01', '2023-09-11')

        iterables = [test_date, list(regions[regions['departement'] == code].hex_id)]
        test = pd.DataFrame(index=pd.MultiIndex.from_product(iterables=iterables,  names=['date', 'h3'])).reset_index()
        test['longitude'] = test['h3'].apply(lambda x : float(regions[regions['hex_id'] == x].latitude))
        test['latitude'] = test['h3'].apply(lambda x : float(regions[regions['hex_id'] == x].longitude))

        testfp = firepoint[firepoint['date'] > '2022-12-31']
        for _, row in testfp.iterrows():
            index = test[(test['h3'] == row['h3']) & (test['date'] == row['date'])].index
            test.drop(index, inplace=True)"""
        
        fpp = fp[fp['departement'] == int(dept.split('-')[1])]
        nfpp = nfp[nfp['departement'] == int(dept.split('-')[1])]

        trainfp = fpp[fpp['date'] < '2023-01-01']
        testfp = fpp[fpp['date'] >= '2023-01-01']
        
        trainnfp = nfpp[nfpp['date'] < '2023-01-01']
        test = nfpp[nfpp['date'] > '2023-01-01']

        logger.info(f'{len(trainfp)}')

        get_sub_nodes_feature_2D(trainfp.values, 'X')
        get_sub_nodes_feature_2D(trainnfp.values, 'X')

        get_sub_nodes_feature_2D(testfp.values, 'test')
        get_sub_nodes_feature_2D(test.values, 'test')

        #fp.append(firepoint)
        #nfp.append(nonfirepoint)

    """fp = pd.concat(fp)
    nfp = pd.concat(nfp)

    fp.to_csv(dir_output / 'firepoints.csv', index=False)
    nfp.to_csv(dir_output / 'nonfirepoints.csv', index=False)"""