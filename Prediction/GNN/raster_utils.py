from tools import *
from config import cems_variables

def raster_meteo(spatioTemporalRaster, h3, grid, index, n_pixel_x, n_pixel_y, dir_output):
    h3 = gpd.GeoDataFrame(h3, )
    for i, var in enumerate(cems_variables):

        h3[var] = interpolate_gridd(var, grid, h3.longitude, h3.latitude)
        values = rasterization(h3, n_pixel_y, n_pixel_x, var, dir_output, var)
        values = resize_no_dim(values, spatioTemporalRaster.shape[0], spatioTemporalRaster.shape[1])
        spatioTemporalRaster[:,:,index] = values
        masknan = np.argwhere(np.isnan(spatioTemporalRaster[:,:,index]))

        spatioTemporalRaster[masknan[:,0], masknan[:,1], index] = np.nanmean(spatioTemporalRaster[:,:, index])
        index += 1

def raster_sentinel(spatioTemporalRaster, date, id, mask, dir_reg, index):
    size = '30m'
    dir_sat = dir_reg / 'GEE' / size
    for tifFile in dir_sat.glob('sentinel/*.tif'):
        
        tifFile = tifFile.as_posix()
        dateFile = tifFile.split('/')[-1]
        dateFile = dateFile.split('.')[0]

        infDate = (dt.datetime.strptime(dateFile, '%Y-%m-%d') - dt.timedelta(days=8)).strftime('%Y-%m-%d')
        endDate = (dt.datetime.strptime(dateFile, '%Y-%m-%d') + dt.timedelta(days=8)).strftime('%Y-%m-%d')
        if date < infDate or date >= endDate:
            continue

        sentinel, _, _ = read_tif(tifFile)
        sentinel = resize(sentinel, mask.shape[0], mask.shape[1], sentinel.shape[0])
        ind = np.argwhere(mask == id)
        xmin = np.min(ind[:,0])
        xmax = np.max(ind[:,0])
        ymin = np.min(ind[:,1])
        ymax = np.max(ind[:,1])
        for band in range(sentinel.shape[0]):
            values = sentinel[band][xmin:xmax, ymin:ymax]
            values = resize_no_dim(values, spatioTemporalRaster.shape[0], spatioTemporalRaster.shape[1])
            spatioTemporalRaster[:,:, index] = values
            masknan = np.argwhere(np.isnan(spatioTemporalRaster[:,:,index]))
            spatioTemporalRaster[masknan[:,0], masknan[:,1], index] = np.nanmean(spatioTemporalRaster[:,:, index])
            index += 1
        return
    
def raster_sentinel_2(spatioTemporalRaster, mask, id, sent, index):
    ind = np.argwhere(mask == id)
    xmin = np.min(ind[:,0])
    xmax = np.max(ind[:,0])
    ymin = np.min(ind[:,1])
    ymax = np.max(ind[:,1])
    #values = values[:,:,np.newaxis]
    #values = np.repeat(values, repeats=spatioTemporalRaster.shape[-1], axis=2)
    for band in range(sent.shape[0]):
        values = resize_no_dim(sent[band, xmin:xmax, ymin:ymax], spatioTemporalRaster.shape[0], spatioTemporalRaster.shape[1])
        spatioTemporalRaster[:,:,index + band] = values
        masknan = np.argwhere(np.isnan(spatioTemporalRaster[:,:,index + band]))
        spatioTemporalRaster[masknan[:,0], masknan[:,1], index + band] = np.nanmean(spatioTemporalRaster[:,:, index + band])

def raster_landcover(spatioTemporalRaster, date, id, mask, dir_reg, index, encoder):
    size = '30m'
    dir_sat = dir_reg / 'GEE' / size

    for tifFile in dir_sat.glob('dynamic_world/*.tif'):

        tifFile = tifFile.as_posix()
        dateFile = tifFile.split('/')[-1]
        dateFile = dateFile.split('.')[0]

        infDate = (dt.datetime.strptime(dateFile, '%Y-%m-%d') - dt.timedelta(days=8)).strftime('%Y-%m-%d')
        endDate = (dt.datetime.strptime(dateFile, '%Y-%m-%d') + dt.timedelta(days=8)).strftime('%Y-%m-%d')

        if date < infDate or date >= endDate:
            continue
        
        dw, _, _ = read_tif(tifFile)
        dw = dw[-1]
        dw = resize_no_dim(dw, mask.shape[0], mask.shape[1])
        ind = np.argwhere(mask == id)
        xmin = np.min(ind[:,0])
        xmax = np.max(ind[:,0])
        ymin = np.min(ind[:,1])
        ymax = np.max(ind[:,1])
        values = resize_no_dim(dw[xmin:xmax, ymin:ymax], spatioTemporalRaster.shape[0], spatioTemporalRaster.shape[1])
        values = encoder.transform(values.reshape(-1,1)).values.reshape(values.shape)
        spatioTemporalRaster[:,:, index] = values
        masknan = np.argwhere(np.isnan(spatioTemporalRaster[:,:,index]))
        spatioTemporalRaster[masknan[:,0], masknan[:,1], index] = np.nanmean(spatioTemporalRaster[:,:, index])
        return
    
def raster_landcover_2(spatioTemporalRaster, mask, id, land, index, encoder):
    ind = np.argwhere(mask == id)
    xmin = np.min(ind[:,0])
    xmax = np.max(ind[:,0])
    ymin = np.min(ind[:,1])
    ymax = np.max(ind[:,1])
    values = resize_no_dim(land[xmin:xmax, ymin:ymax], spatioTemporalRaster.shape[0], spatioTemporalRaster.shape[1])
    #values = values[:,:,np.newaxis]
    #values = np.repeat(values, repeats=spatioTemporalRaster.shape[-1], axis=2)
    #values = encoder.transform(values.reshape(-1,1)).values.reshape(values.shape)
    spatioTemporalRaster[:,:, index] = values
    masknan = np.argwhere(np.isnan(spatioTemporalRaster[:,:,index]))
    spatioTemporalRaster[masknan[:,0], masknan[:,1], index] = np.nanmean(spatioTemporalRaster[:,:, index])
    
def raster_foret(spatioTemporalRaster, mask, id, foret, index, encoder):
    ind = np.argwhere(mask == id)
    xmin = np.min(ind[:,0])
    xmax = np.max(ind[:,0])
    ymin = np.min(ind[:,1])
    ymax = np.max(ind[:,1])
    values = resize_no_dim(foret[xmin:xmax, ymin:ymax], spatioTemporalRaster.shape[0], spatioTemporalRaster.shape[1])
    #values = values[:,:,np.newaxis]
    #values = np.repeat(values, repeats=spatioTemporalRaster.shape[-1], axis=2)
    #values = encoder.transform(values.reshape(-1,1)).values.reshape(values.shape)
    spatioTemporalRaster[:,:, index] = values
    masknan = np.argwhere(np.isnan(spatioTemporalRaster[:,:,index]))
    spatioTemporalRaster[masknan[:,0], masknan[:,1], index] = np.nanmean(spatioTemporalRaster[:,:, index])

def raster_elevation(spatioTemporalRaster, mask, id, elevation, index):
    ind = np.argwhere(mask == id)
    xmin = np.min(ind[:,0])
    xmax = np.max(ind[:,0])
    ymin = np.min(ind[:,1])
    ymax = np.max(ind[:,1])
    values = resize_no_dim(elevation[xmin:xmax, ymin:ymax], spatioTemporalRaster.shape[0], spatioTemporalRaster.shape[1])
    #values = values[:,:,np.newaxis]
    #values = np.repeat(values, repeats=spatioTemporalRaster.shape[-1], axis=2)
    spatioTemporalRaster[:,:, index] = values
    masknan = np.argwhere(np.isnan(spatioTemporalRaster[:,:,index]))
    spatioTemporalRaster[masknan[:,0], masknan[:,1], index] = np.nanmean(spatioTemporalRaster[:,:, index])

def raster_population(spatioTemporalRaster, mask, id, population, reslon, reslat, index):
    ind = np.argwhere(mask == id)
    xmin = np.min(ind[:,0])
    xmax = np.max(ind[:,0])
    ymin = np.min(ind[:,1])
    ymax = np.max(ind[:,1])

    values = resize_no_dim(population[xmin:xmax, ymin:ymax], spatioTemporalRaster.shape[0], spatioTemporalRaster.shape[1])
    #values = values[:,:,np.newaxis]
    #values = np.repeat(values, repeats=spatioTemporalRaster.shape[-1], axis=2)
    spatioTemporalRaster[:,:, index] = values
    masknan = np.argwhere(np.isnan(spatioTemporalRaster[:,:,index]))
    spatioTemporalRaster[masknan[:,0], masknan[:,1], index] = np.nanmean(spatioTemporalRaster[:,:, index])

def raster_osmnx(spatioTemporalRaster, mask, id, res, index):
    ind = np.argwhere(mask == id)
    xmin = np.min(ind[:,0])
    xmax = np.max(ind[:,0])
    ymin = np.min(ind[:,1])
    ymax = np.max(ind[:,1])
    values = resize_no_dim(res[xmin:xmax, ymin:ymax], spatioTemporalRaster.shape[0], spatioTemporalRaster.shape[1])
    #values = values[:,:,np.newaxis]
    #values = np.repeat(values, repeats=spatioTemporalRaster.shape[-1], axis=2)
    spatioTemporalRaster[:,:, index] = values
    masknan = np.argwhere(np.isnan(spatioTemporalRaster[:,:,index]))
    spatioTemporalRaster[masknan[:,0], masknan[:,1], index] = np.nanmean(spatioTemporalRaster[:,:, index])

def raster_target(spatioTemporalRaster, mask, id, res, index):
    res[np.isnan(res)] = -1.0
    res = resize_no_dim(res, mask.shape[0], mask.shape[1])
    if index == -2:
        res = res.astype(int)
        save_object(res, 'testBin.pkl', Path('./'))
    ind = np.argwhere(mask == id)
    xmin = np.min(ind[:,0])
    xmax = np.max(ind[:,0])
    ymin = np.min(ind[:,1])
    ymax = np.max(ind[:,1])
    values = resize_no_dim(res[xmin:xmax, ymin:ymax], spatioTemporalRaster.shape[0], spatioTemporalRaster.shape[1])
    spatioTemporalRaster[:,:, index] = values
    masknan = np.argwhere(np.isnan(spatioTemporalRaster[:,:,index]))
    spatioTemporalRaster[masknan[:,0], masknan[:,1], index] = np.nanmean(spatioTemporalRaster[:,:, index])