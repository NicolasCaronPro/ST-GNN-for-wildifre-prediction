from tools import *
from config import calendar_variables

def encode(path_to_target, maxDate, trainDepartments, dir_output):

    print(f'Create encoder for categorical features using {trainDepartments}, at max {maxDate}')
    trainMax = allDates.index(maxDate)
    foret = []
    gt = []
    landcover = []
    calendar_array = [[] for j in calendar_variables]
    geo_array = []
    for dep in trainDepartements:
        
        dir_data = root / 'csv' / dep /  'raster'

        tar = read_object(dep+'Influence.pkl', path_to_target)
        if tar is None:
            continue
        tar = tar[:,:,:trainMax]
        gt += list(tar[~np.isnan(tar)].reshape(-1))

        fore = read_object('foret.pkl', dir_data)

        fore = resize_no_dim(fore, tar.shape[0], tar.shape[1])
        fore = fore[:,:, np.newaxis]
        fore = np.repeat(fore, repeats=tar.shape[2], axis=2)

        foret += list(fore[~np.isnan(tar)])

        land = read_object('landcover.pkl', dir_data)[:,:,:trainMax]
        land[~np.isnan(tar)] = np.round(land[~np.isnan(tar)])
        landcover += list(land[~np.isnan(tar)])

        calendar = np.empty((tar.shape[0], tar.shape[1], tar.shape[2], len(calendar_variables)))
        for i, date in enumerate(allDates):
            if date == '2022-01-01':
                break
            ddate = dt.datetime.strptime(date, '%Y-%m-%d')
            calendar[:,:,i, 0] = int(date.split('-')[1]) # month
            calendar[:,:,i, 1] = ddate.timetuple().tm_yday # dayofweek
            calendar[:,:,i, 2] = ddate.weekday() # dayofyear
            calendar[:,:,i, 3] = ddate.weekday() >= 5 # isweekend
            calendar[:,:,i, 4] = pendant_couvrefeux(ddate) # couvrefeux
            calendar[:,:,i, 5] = (1 if dt.datetime(2020, 3, 17, 12) <= ddate <= dt.datetime(2020, 5, 11) else 0) or 1 if dt.datetime(2020, 10, 30) <= ddate <= dt.datetime(2020, 12, 15) else 0# confinement
            calendar[:,:,i, 6] = 1 if convertdate.islamic.from_gregorian(ddate.year, ddate.month, ddate.day)[1] == 9 else 0 # ramadan
            calendar[:,:,i, 7] = 1 if ddate in jours_feries else 0 # bankHolidays
            calendar[:,:,i, 8] = 1 if ddate in veille_jours_feries else 0 # bankHolidaysEve
            calendar[:,:,i, 9] = 1 if vacances_scolaire.is_holiday_for_zone(ddate.date(), get_academic_zone(ACADEMIES[str(name2int[dep])], ddate)) else 0 # holidays
            calendar[:,:,i, 10] = (1 if vacances_scolaire.is_holiday_for_zone(ddate.date() + dt.timedelta(days=1), get_academic_zone(ACADEMIES[str(name2int[dep])], ddate)) else 0 ) \
                or (1 if vacances_scolaire.is_holiday_for_zone(ddate.date() - dt.timedelta(days=1), get_academic_zone(ACADEMIES[str(name2int[dep])], ddate)) else 0) # holidaysBorder

        for j, _ in enumerate(calendar_variables):
            calendar_array[j] += list(calendar[~np.isnan(tar), j])

        geo = np.empty((tar.shape[0], tar.shape[1], tar.shape[2], len(geo_variables)))
        geo[:,:,:] = name2int[dep]
        geo_array += list(geo[~np.isnan(tar)])
        
    gt = np.asarray(gt)
    foret = np.asarray(foret)
    landcover = np.asarray(landcover)
    calendar_array = np.asarray(calendar_array)
    geo_array = np.asarray(geo_array)

    print(gt.shape, foret.shape, landcover.shape, calendar_array.shape, geo_array.shape)

    gt = gt.reshape(-1,1)
    foret = foret.reshape(-1,1)
    landcover = landcover.reshape(-1,1)
    calendar_array = np.moveaxis(calendar_array, 0, 1)
    geo_array = geo_array.reshape(-1,1)

    print(gt.shape, foret.shape, landcover.shape, calendar_array.shape, geo_array.shape)

    # Calendar
    encoder = CatBoostEncoder(cols=np.arange(0, len(calendar_variables)))
    encoder.fit(calendar_array, gt)
    save_object(encoder, 'encoder_calendar.pkl', dir_output)

    # Landcover
    encoder = CatBoostEncoder(cols=np.arange(0, 1))
    encoder.fit(landcover, gt)
    save_object(encoder, 'encoder_landcover.pkl', dir_output)

    # Foret
    encoder = CatBoostEncoder(cols=np.arange(0, 1))
    encoder.fit(foret, gt)
    save_object(encoder, 'encoder_foret.pkl', dir_output)

    # Geo
    encoder = CatBoostEncoder(cols=np.arange(0, 1))
    encoder.fit(geo_array, gt)
    save_object(encoder, 'encoder_geo.pkl', dir_output)


if __name__ == '__main__':
    traindepartements = ['departement-01-ain', 'departement-25-doubs', 'departement-78-yvelines']
    departements = ['departement-01-ain', 'departement-25-doubs', 'departement-69-rhone', 'departement-78-yvelines']

    path_to_target = root / 'Model/HexagonalScale/log'

    trainMax = allDates.index('2022-01-01')
    print(trainMax)

