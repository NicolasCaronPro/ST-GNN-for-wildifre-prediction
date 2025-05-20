from segmentation import *
from pathlib import Path
from itertools import product

def launch(method):
    ###################################################################################
    resolution = '2x2'
    train_departements = ['departement-01-ain', 'departement-25-doubs', 'departement-78-yvelines', 'departement-69-rhone']
    train_date = '2022-01-01'
    geo = gpd.read_file(root_target / 'regions.geojson')
    geo = geo[geo['departement'].isin(train_departements)].reset_index(drop=True)

    # Définir les plages de paramètres pour le grid search
    scales = [1, 2, 3, 4, 5, 6, 7]
    nb_attempts = [None, 1, 2, 3, 4, 5]  # À adapter selon ton besoin
    n_reduce_classes = [None, 2, 3, 4, 5, 6]  # À adapter aussi
    bases = [
        'risk-size-watershed',
        #'risk-size-clustering',
        'risk-regular',
        #'risk-BrayCurtis-watershed',
        #'risk-timeSeriesSimilarity-watershed'
    ]
    compare_experiment(train_departements, scales, nb_attempts, n_reduce_classes, bases, method, 'firemen')

    ###################################################################################
    resolution = '2x2'
    #train_departements = ['departement-13-bouches-du-rhone', 'departement-34-herault']
    train_departements = ['departement-13-bouches-du-rhone', 'departement-34-herault']
    train_date = '2022-01-01'
    geo = gpd.read_file(root_target / 'regions.geojson')
    geo = geo[geo['departement'].isin(train_departements)].reset_index(drop=True)

    # Définir les plages de paramètres pour le grid search
    scales = [1, 2, 3, 4, 5, 6, 7]
    nb_attempts = [None, 1, 2, 3, 4, 5]  # À adapter selon ton besoin
    n_reduce_classes = [None, 2, 3, 4, 5, 6]  # À adapter aussi
    bases = [
        'risk-size-watershed',
        #'risk-size-clustering',
        'risk-regular',
        #'risk-BrayCurtis-watershed',
        #'risk-timeSeriesSimilarity-watershed'
    ]
    compare_experiment(train_departements, scales, nb_attempts, n_reduce_classes, bases, method, 'bdiff_small')

    ###################################################################################

    resolution = '2x2'
    #train_departements = ['departement-13-bouches-du-rhone', 'departement-34-herault']
    train_departements = ['departement-01-ain', 'departement-13-bouches-du-rhone', 'departement-25-doubs', 'departement-78-yvelines', 'departement-69-rhone', 'departement-34-herault']

    train_date = '2022-01-01'
    geo = gpd.read_file(root_target / 'regions.geojson')
    geo = geo[geo['departement'].isin(train_departements)].reset_index(drop=True)

    # Définir les plages de paramètres pour le grid search
    scales = [1, 2, 3, 4, 5, 6, 7]
    nb_attempts = [None, 1, 2, 3, 4, 5]  # À adapter selon ton besoin
    n_reduce_classes = [None, 2, 3, 4, 5, 6]  # À adapter aussi
    bases = [
        'risk-size-watershed',
        #'risk-size-clustering',
        'risk-regular',
        #'risk-BrayCurtis-watershed',
        #'risk-timeSeriesSimilarity-watershed'
    ]
    for dept in train_departements:
        compare_experiment([dept], scales, nb_attempts, n_reduce_classes, bases, method, dept)

if __name__ == '__main__':
    launch('fr')
    launch('pearson')
    launch('spearman')