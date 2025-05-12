from arborescence import *
from tools import *
from array_fet import *
from dico_departements import *
from matplotlib.colors import ListedColormap
import pandas as pd

class Myloader():
    def __init__(self, args) -> None:
        self.name = args.name
        self.sinister = args.sinister
        self.doDatabase = args.database == 'True'
        self.resolution = args.resolution
        self.aggregation = args.aggregation
        self.dataset = args.dataset
        self.sinister_encoding = args.sinisterEncoding

        self.dir_output = Path(self.dataset) / self.name / self.sinister_encoding / self.aggregation
        check_and_create_path(self.dir_output)
    
    def load_target(self):
        if self.sinister != 'firepoint':
            raise ValueError(f'{self.sinister} not implmented as sinister')
        
        dir_target = root_target / self.sinister / self.dataset / self.sinister_encoding
        
        self.france = read_object(f'france.pkl', dir_target)
        self.images = []
        years = ['2018', '2019', '2020', '2021', '2022', '2023']
        for year in years:
            im = read_object(f'fire_{year}.pkl', dir_target)
            assert im is not None
            im = im[0]

            self.images.append(im)

    def apply_aggregation(self):
        aggregation_params = self.aggregation.split('_')
        
        if aggregation_params[0] == 'None':
            self.images_aggregated = self.images
        elif aggregation_params[0] == 'MaxPool':
            kernel_size = int(aggregation_params[1])
            self.images_aggregated =  [nanmax_pool_same_resolution(im, kernel_size=kernel_size) for im in self.images]
        else:
            raise ValueError(f'{self.aggregation} not implmented')
        
        plot_matches_on_images(self.images_aggregated, ['2018', '2019', '2020', '2021', '2022', '2023'], cmap='plasma', figsize=(25,15), dir_output=self.dir_output, name=f'{self.aggregation}_firepoint')

    def apply_watershed_clustering(self, image, mask, **args):
        valid_mask = ~np.isnan(image)
        res_watershed = my_watershed(image, valid_mask)

        merge_region = merge_adjacent_clusters(res_watershed, **args)

        valid_cluster = find_clusters(merge_region, args['min_cluster_size'], 0, -1)

        valid_cluster = [val + 1 for val in valid_cluster]
        merge_region[valid_mask] += 1
        merge_region = split_large_clusters(merge_region, args['max_cluster_size'], args['min_cluster_size'], valid_cluster)
        valid_cluster = [val - 1 for val in valid_cluster]
        merge_region[valid_mask] -= 1
        
        valid_mask = ~mask
        merge_region = merge_region.astype(np.float32)
        merge_region[mask] = np.nan
        return merge_region
    
    def compute_features(self, departement_mask, features_list=default_features):
        dir_output_features = self.dir_output / '../' / 'features'
        check_and_create_path(dir_output_features)
        H, W = departement_mask.shape
        res_total = []
        unique_departements = np.unique(departement_mask[~np.isnan(departement_mask)])

        features_name = list(final_dict.keys())
        
        allDates = find_dates_between('2017-06-12', '2024-06-29')

        root_data_disk = Path('/media/caron/X9 Pro/travaille/Thèse/csv')

        if not self.doDatabase and (dir_output_features / 'features.pkl').is_file():
            res_total = read_object('features.pkl', dir_output_features)
            #plot_feature_distributions(res_total, list(final_dict.keys()), dir_output_features, ['2018', '2019', '2020', '2021', '2022', '2023'])
            #return res_total, list(final_dict.keys())
        else:
            for i, year in enumerate(['2018', '2019', '2020', '2021', '2022', '2023']):
                res = []
                
                for idx, feature in enumerate(features_list):
                    
                    print(f'{year}, {feature}')

                    if feature == 'Tourisme':
                        tourisme_data = pd.read_csv(root_data_disk / 'Tourisme.csv')
                    
                    france_feature = None

                    if feature == "LastState":
                        continue

                    for departement in unique_departements:
                        dir_raster = root_data_disk / int2name[int(departement)] / 'raster' / self.resolution
                        if feature == 'Tourisme':
                            n_bands = 1
                        elif feature == "LastState":
                            # Crée une image vide (0) de taille identique aux autres features
                            data_features = np.zeros((1, H, W), dtype=np.float32)
                        else:
                            if feature in cems_variables:
                                data_features = read_object(f'{feature}raw.pkl', dir_raster)
                            else:
                                data_features = read_object(f'{feature}.pkl', dir_raster)
                            assert data_features is not None
                            # S'assurer que les données ont la forme (bands, h, w)
                            if data_features.ndim == 2:
                                data_features = data_features[np.newaxis, ...]  # (1, h, w)
                            elif feature in cems_variables:
                                data_features = data_features[:, :, allDates.index(f'{year}-05-01') : allDates.index(f'{year}-10-01')]
                                data_features = np.nanmean(data_features, axis=-1)[np.newaxis, :, :]
                            elif feature == 'sentinel':
                                data_features = data_features[:, :, :, allDates.index(f'{year}-05-01') : allDates.index(f'{year}-10-01')]
                                data_features = np.nanmean(data_features, axis=-1)

                        n_bands = data_features.shape[0]

                        if france_feature is None:
                            france_feature = np.full((n_bands, H, W), fill_value=np.nan)

                        # Créer un masque binaire pour ce département
                        mask = (departement_mask == departement)
                        mask_2d  = extract_bbox_region(mask == 1)

                        if feature != 'Tourisme':
                            # Redimensionner chaque bande vers la taille du mask national
                            for b in range(n_bands):
                                band = data_features[b]
                                resized_band = interpolate_image_to_match_target(band, mask_2d)
                                france_feature[b][mask] = resized_band[mask_2d == 1]
                        else:
                            france_feature[0][mask] = tourisme_data[tourisme_data['Code'] == departement]['Valeur'].values[0]

                    res.append(france_feature)
                    
                # Si "LastState" a été demandé, on injecte self.images_aggregated[i - 1] dans sa place
                if i > 0:
                    res.append(self.transition_rules[i - 1][np.newaxis, :, :])
                else:
                    res.append(np.zeros((1, H, W)))

                # Continue ici avec le reste du traitement sur `res` pour cette année
                res = np.concatenate(res, axis=0).astype(np.float32)  
                #res = interpolate_image_3d(res, np.isnan(self.images_aggregated[0]))
                res_total.append(res)

            save_object(res_total, 'features.pkl', dir_output_features)
        
        print(len(list(final_dict.keys())))
        res = res_total[0]

        test_res = [res_total[0][i] for i in range(res.shape[0] - 1)]
        plot_matches_on_images(test_res, list(final_dict.keys()), cmap='plasma', figsize=(50,50), dir_output=dir_output_features, name=f'features_2018', colorbar=False)
        
        test_res = [res_total[1][i] for i in range(res.shape[0])]
        plot_matches_on_images(test_res, list(final_dict.keys()), cmap='plasma', figsize=(50,50), dir_output=dir_output_features, name=f'features_2019', colorbar=False)
        
        test_res = [res_total[2][i] for i in range(res.shape[0])]
        plot_matches_on_images(test_res, list(final_dict.keys()), cmap='plasma', figsize=(50,50), dir_output=dir_output_features, name=f'features_2020', colorbar=False)
        
        test_res = [res_total[3][i] for i in range(res.shape[0])]
        plot_matches_on_images(test_res, list(final_dict.keys()), cmap='plasma', figsize=(50,50), dir_output=dir_output_features, name=f'features_2021', colorbar=False)

        test_res = [res_total[4][i] for i in range(res.shape[0] - 1)]
        plot_matches_on_images(test_res, list(final_dict.keys()), cmap='plasma', figsize=(50,50), dir_output=dir_output_features, name=f'features_2022', colorbar=False)

        test_res = [res_total[5][i] for i in range(res.shape[0])]
        plot_matches_on_images(test_res, list(final_dict.keys()), cmap='plasma', figsize=(50,50), dir_output=dir_output_features, name=f'features_2023', colorbar=False)
        
        plot_feature_distributions(res_total, list(final_dict.keys()), dir_output_features, ['2018', '2019', '2020', '2021', '2022', '2023'])
        
        return res_total, list(final_dict.keys()) # Shape: (total_bands, H, W)
    
    def compute_target_rules(self):
        """
        Crée une liste d'images représentant les règles de transition entre paires successives
        d'images dans self.images_aggregated (modifiée pour inclure une première image de base).
        Les règles par pixel sont :
            - 0 : si i == j
            - 1 : si i >= 1 et j == 0
            - 2 : si i == 0 et j >= 1
            - 3 : si i >= 1 et j >= 1
            - np.nan : si i ou j est nan
        """
        # Ajouter l'image de base tout au début de self.images_aggregated
        base_img = np.zeros_like(self.images_aggregated[0], dtype=float)
        base_img[np.isnan(self.images_aggregated[0])] = np.nan
        self.images_aggregated = [base_img] + self.images_aggregated

        self.transition_rules = []

        years = ['2018', '2019', '2020', '2021', '2022', '2023']
        years = ['null'] + years  # pour gérer null_2018

        cmap = ListedColormap([
            "blue",   # 0 -> 0
            "green",  # 1 -> 0
            "yellow", # 0 -> 1
            "red"     # 1 -> 1
        ])

        titles = []
        for i in range(len(self.images_aggregated) - 1):
            img1 = self.images_aggregated[i]
            img2 = self.images_aggregated[i + 1]

            # Initialiser l'image de sortie avec des NaN
            rule_image = np.full_like(img1, np.nan, dtype=float)

            # Masques de validité (ni i ni j ne doivent être nan)
            valid_mask = ~np.isnan(img1) & ~np.isnan(img2)

            # Appliquer les règles sur les pixels valides
            i_vals = img1[valid_mask]
            j_vals = img2[valid_mask]

            rules = np.zeros_like(i_vals)

            rules[(i_vals >= 1) & (j_vals == 0)] = 1
            rules[(i_vals == 0) & (j_vals >= 1)] = 2
            rules[(i_vals >= 1) & (j_vals >= 1)] = 3
            # Les autres cas restent à 0 (quand i == j)
            
            # Placer les valeurs calculées dans l'image résultat
            rule_image[valid_mask] = rules

            self.valid_transitions_dict = {0 : [0, 2],
                                           1 : [0, 2],
                                           2 : [3, 1],
                                           3 : [3, 1]}

            self.transition_rules.append(rule_image)
            titles.append(f'{years[i]}_{years[i + 1]}')

        plot_matches_on_images(
            self.transition_rules,
            titles,
            cmap=cmap,
            figsize=(25, 15),
            dir_output=self.dir_output,
            name=f'{self.aggregation}_target_rules'
        )
        return self.transition_rules