from segmentation import *

if __name__ == '__main__':


    ####################### Load geodataframe ########################

    ######################## Define parameters #######################
    ######### fix parameters (do not change those)
    resolution = '2x2'
    #train_departements = ['departement-13-bouches-du-rhone', 'departement-34-herault']
    train_departements = ['departement-01-ain']
    train_date = '2022-01-01'
    geo = gpd.read_file(root_target / 'regions.geojson')
    geo = geo[geo['departement'].isin(train_departements)].reset_index(drop=True)

    ######## Variables parameters
    scale = 5 # possibles values 4 5 6 7
    nb_attempt = 3 # possibles 0-inf 
    n_reduce_class = 4 # possibles 1-inf
    base = 'risk-size-watershed'
    
    """################# Watershed segmentation #############
    Segmentation_obj = Segmentation(scale=scale, nb_attempt=nb_attempt, n_reduce_class=n_reduce_class, geo=geo, resolution=resolution, base=base)
    dir_output = Path('./risk-size-watershed')

    Segmentation_obj._create_sinister_region(path=dir_output,
                                 resolution=resolution, train_date=train_date)"""
     
    """################# HDBSCAN segmentation ##############
    base = 'risk-size-clustering'
    Segmentation_obj = Segmentation(scale=scale, nb_attempt=nb_attempt, n_reduce_class=n_reduce_class, geo=geo, resolution=resolution, base=base)
    dir_output = Path('./risk-size-clustering')

    Segmentation_obj._create_sinister_region(path=dir_output,
                                 resolution=resolution, train_date=train_date)
    
     ################# Grid segmentation #################
    base = 'risk-regular'
    Segmentation_obj = Segmentation(scale=scale, nb_attempt=nb_attempt, n_reduce_class=n_reduce_class, geo=geo, resolution=resolution, base=base)
    dir_output = Path('./risk-regular')

    Segmentation_obj._create_sinister_region(path=dir_output,
                                 resolution=resolution, train_date=train_date)"""
    
    """################# Foret Watershed segmentation #############
    base = 'risk-BrayCurtis-watershed'
    Segmentation_obj = Segmentation(scale=scale, nb_attempt=nb_attempt, n_reduce_class=n_reduce_class, geo=geo, resolution=resolution, base=base, features_name=['foret'])
    dir_output = Path('./risk-BrayCurtis-watershed')

    Segmentation_obj._create_sinister_region(path=dir_output,
                                 resolution=resolution, train_date=train_date)"""
    
    ################# Foret Watershed segmentation #############
    base = 'risk-timeSeriesSimilarity-watershed'
    Segmentation_obj = Segmentation(scale=scale, nb_attempt=nb_attempt, n_reduce_class=n_reduce_class, geo=geo, resolution=resolution, base=base)
    dir_output = Path('./risk-timeSeriesSimilarity-watershed')

    Segmentation_obj._create_sinister_region(path=dir_output,
                                 resolution=resolution, train_date=train_date)