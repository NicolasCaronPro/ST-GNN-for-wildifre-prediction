from segmentation import *
from pathlib import Path
from itertools import product

if __name__ == '__main__':
    resolution = '2x2'
    #train_departements = ['departement-13-bouches-du-rhone', 'departement-34-herault']
    train_departements = [#'departement-01-ain', 'departement-13-bouches-du-rhone', 'departement-25-doubs', 'departement-78-yvelines', 'departement-69-rhone',
                          'departement-34-herault']
    train_date = '2022-01-01'
    geo = gpd.read_file(root_target / 'regions.geojson')
    geo = geo[geo['departement'].isin(train_departements)].reset_index(drop=True)

    # Définir les plages de paramètres pour le grid search
    scales = [1, 2, 3, 4, 5, 6, 7]
    nb_attempts = [1, 2, 3, 4, 5]  # À adapter selon ton besoin
    n_reduce_classes = [2, 3, 4, 5, 6]  # À adapter aussi
    bases = [
        'risk-size-watershed',
        #'risk-size-clustering',
        #'risk-regular',
        #'risk-BrayCurtis-watershed',
        #'risk-timeSeriesSimilarity-watershed'
    ]

    # Itérer sur toutes les combinaisons de paramètres
    for scale, nb_attempt, n_reduce_class, base in product(scales, nb_attempts, n_reduce_classes, bases):
        print(f"\nRunning segmentation for scale={scale}, nb_attempt={nb_attempt}, "
            f"n_reduce_class={n_reduce_class}, base={base}")

        # Définir les arguments communs
        kwargs = {
            'scale': scale,
            'nb_attempt': nb_attempt,
            'n_reduce_class': n_reduce_class,
            'geo': geo,
            'resolution': resolution,
            'base': base
        }

        # Ajouter des features spécifiques pour certaines bases
        if base == 'risk-BrayCurtis-watershed':
            kwargs['features_name'] = ['foret', 'sentinel', 'population', 'cosia']

        # Créer l'objet Segmentation
        Segmentation_obj = Segmentation(**kwargs)

        # Définir le répertoire de sortie
        dir_output = Path(f'/media/caron/X9 Pro/travaille/Thèse/segmentation/{base}/s{scale}_a{nb_attempt}_r{n_reduce_class}')
        dir_output.mkdir(parents=True, exist_ok=True)

        # Lancer la segmentation
        Segmentation_obj._create_sinister_region(path=dir_output,
                                                resolution=resolution,
                                                train_date=train_date)

    # Définir les plages de paramètres pour le grid search
    scales = [1, 2, 3, 4, 5, 6, 7]
    bases = [
        #'risk-size-watershed',
        #'risk-size-clustering',
        'risk-regular',
        #'risk-BrayCurtis-watershed',
        #'risk-timeSeriesSimilarity-watershed'
    ]
    nb_attempt = None
    n_reduce_class = None
    # Itérer sur toutes les combinaisons de paramètres
    for scale, base in product(scales, bases):
        print(f"\nRunning segmentation for scale={scale}, nb_attempt={nb_attempt}, "
            f"n_reduce_class={n_reduce_class}, base={base}")

        # Définir les arguments communs
        kwargs = {
            'scale': scale,
            'nb_attempt': nb_attempt,
            'n_reduce_class': n_reduce_class,
            'geo': geo,
            'resolution': resolution,
            'base': base
        }

        # Ajouter des features spécifiques pour certaines bases
        if base == 'risk-BrayCurtis-watershed':
            kwargs['features_name'] = ['foret']

        # Créer l'objet Segmentation
        Segmentation_obj = Segmentation(**kwargs)

        # Définir le répertoire de sortie
        dir_output = Path(f'/media/caron/X9 Pro/travaille/Thèse/segmentation/{base}/s{scale}_aNone_rNone')
        dir_output.mkdir(parents=True, exist_ok=True)

        # Lancer la segmentation
        Segmentation_obj._create_sinister_region(path=dir_output,
                                                resolution=resolution,
                                                train_date=train_date)
