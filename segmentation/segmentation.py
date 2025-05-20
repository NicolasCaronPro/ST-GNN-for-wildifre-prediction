from tools import *

class Segmentation():
    def __init__(self, scale : int,
                  n_reduce_class : int,
                  nb_attempt: int,
                 geo : gpd.GeoDataFrame,
                 resolution : str,
                 base : str,
                 features_name=None,
                ):

        for col in ['latitude', 'longitude', 'geometry', 'departement']:
            if col not in geo.columns:
                logger.info(f'{col} not in geo columns. Please send a correct geo dataframe')
                exit(2)

        self.scale = scale # current scale define by numUniqueNode // 6
        self.n_reduce_class = n_reduce_class
        self.nb_attempt = nb_attempt
        self.base = base

        #################################### Fix parameters ########################
        self.oriLatitudes = geo.latitude # all original latitude of interest
        self.oriLongitude = geo.longitude # all original longitude of interest
        self.oriLen = self.oriLatitudes.shape[0] # len of data
        self.oriGeometry = geo['geometry'].values # original geometry
        self.oriIds = geo['scale0'].values if 'scale0' in list(geo.columns) else geo.index.values
        self.orihexid = geo['hex_id'].values
        self.departements = geo.departement # original dept of each geometry
        self.sinister_regions = [] # fire region id
        self.resolution = resolution
        self.sinister = 'firepoint'
        self.graph_construct = 'node'
        self.graph_method = 'node'
        self.sinister_encoding = 'occurence'
        self.dataset_name = 'icml'
        self.features_name = features_name

    def _create_sinister_region(self, path: Path, resolution, train_date) -> None:
        udept = np.unique(self.departements)

        if '-' in self.base:
            vec_base = self.base.split('-')
            self.base = self.base
        else:
            raise ValueError(f'{self.base} is a unknow format try risk-size-watershed, risk-size-clustering or risk-regular')

        self.train_kmeans = True
        node_already_predicted = 0
        self.numCluster = 0
        self.resolution = resolution
        
        logger.info(f'Create node via {vec_base}')
        
        self.ids = np.full(self.oriLatitudes.shape[0], fill_value=np.nan)
        self.graph_ids = np.full(self.oriLatitudes.shape[0], fill_value=np.nan)

        for dept in udept:
            mask = self.departements == dept
            logger.info(f'######################### {dept} ####################')
            if self.scale == 'departement':
                assert self.base == 'None'
                dir_raster = root_target
                raster = read_object(f'{dept}rasterScale0.pkl', dir_raster)
                assert raster is not None
                raster = raster[0]
                pred = np.full(raster.shape, fill_value=np.nan)
                pred[~np.isnan(raster)] = node_already_predicted
                # Process post-watershed results
                self._post_process_result(pred, raster, mask, node_already_predicted, 'graph')
                self._save_feature_image(path, dept, 'pred_final', pred, raster)
            else:
                if 'clustering' in vec_base:
                    self.create_geometry_with_clustering(dept, vec_base, path, mask, node_already_predicted, train_date)
                elif 'watershed' in vec_base:
                    self.create_geometry_with_watershed(dept, vec_base, path,  mask, node_already_predicted, train_date)
                elif 'regular' in vec_base:
                    self.create_geometry_with_regular(dept, path, mask, node_already_predicted)

            current_cluster = np.nanmax(self.graph_ids[mask][~np.isnan(self.graph_ids[mask])]) + 1
            logger.info(f'{dept} Unique cluster : {np.unique(self.graph_ids[mask])}, {current_cluster}. {node_already_predicted}')
            node_already_predicted = current_cluster
            
        self.oriIds = self.oriIds[~np.isnan(self.graph_ids)]
        self.oriLatitudes = self.oriLatitudes[~np.isnan(self.graph_ids)].reset_index(drop=True)
        self.oriGeometry = self.oriGeometry[~np.isnan(self.graph_ids)]
        self.oriLongitude = self.oriLongitude[~np.isnan(self.graph_ids)].reset_index(drop=True)
        self.departements = self.departements[~np.isnan(self.graph_ids)].reset_index(drop=True)
        self.graph_ids = self.graph_ids[~np.isnan(self.graph_ids)].astype(int)
        self.graph_ids = relabel_clusters(self.graph_ids, 0)
        self.numCluster = np.unique(self.graph_ids).shape[0]
        self.oriLen = self.oriLatitudes.shape[0]
        self.numCluster = np.shape(np.unique(self.graph_ids))[0]

        self._raster(path=path, sinister=self.sinister, base=self.base, resolution=resolution, train_date=train_date, dataset_name=self.dataset_name, sinister_encoding=self.sinister_encoding)

    def create_geometry_with_clustering(self, dept, vec_base, path,  mask, node_already_predicted, train_date):

        dir_raster = root_target
        dir_target = root_target
        raster = read_object(f'{dept}rasterScale0.pkl', dir_raster)

        self.max_target_value = None
        
        if raster is None:
            exit(1)
        raster = raster[0]
        pred = np.full(raster.shape, fill_value=-1)

        vb = vec_base[0]
        mode = vec_base[1]

        #raster = remove_0_risk_pixel(dir_target, dir_target_bin, raster, dept, 'risk', 0)
        valid_mask = (raster != -1) & (~np.isnan(raster))

        data = self._process_base_data(vb, dept, dir_target, valid_mask, train_date)
        if data is None:
            logger.info(f'Can t find {vb}')
            exit(1)

        if self.max_target_value is None:
            self.max_target_value = np.nanmax(data)
        else:
            self.max_target_value = max(self.max_target_value, np.nanmax(data))

        oridata = np.copy(data) 

        reducor = KMeans(n_clusters=self.n_reduce_class, random_state=42, n_init=10)
        reducor.fit(data[valid_mask].reshape(-1,1))
        data[valid_mask] = reducor.predict(data[valid_mask].reshape(-1,1))
        data[valid_mask] = order_class(reducor, data[valid_mask])
        data[~valid_mask] = 0

        data[valid_mask] = morphology.erosion(data, morphology.square(1))[valid_mask]
        self._save_feature_image(path, dept, f'{vb}_pred', data, raster)

        width, height = raster.shape[1], raster.shape[0]
        positions = np.array([[x, y] for y in range(height) for x in range(width)])
        X_image = np.zeros((height, width), dtype=float)
        Y_image = np.zeros((height, width), dtype=float)

        # Remplir les images avec les indices x et y
        for pos in positions:
            x, y = pos
            X_image[y, x] = x  # Indice selon x
            Y_image[y, x] = y  # Indice selon y

        X_image[data == 0] = np.nan
        Y_image[data== 0] = np.nan
        values = np.stack((X_image, Y_image), axis=2)

        values = np.moveaxis(values, 2, 0)
        self._save_feature_image(path, dept, 'latitude', values[0], raster)
        self._save_feature_image(path, dept, 'longitude', values[1], raster)
        pred_mask = ~np.isnan(values[0])
        values = values[:, pred_mask]
        values = np.moveaxis(values, 0, 1)
        min_cluster_size = 1 + 3 * self.scale * (self.scale + 1)
        max_cluster_size = int(min_cluster_size * 2.5)

        model = HDBSCAN(min_cluster_size=2, max_cluster_size=max_cluster_size)
        pred[pred_mask] = model.fit_predict(values)
        pred[~pred_mask] = 0
        pred[pred == -1] = 0
        self._save_feature_image(path, dept, 'pred_clustering_pred', pred, raster)

        # Merge an split
        # Merge and split clusters
        umarker = np.unique(pred)
        umarker = umarker[(umarker != 0) & ~(np.isnan(umarker))]
        risk_image = np.full(data.shape, fill_value=np.nan)
        for m in umarker:
            mask_temp = (pred == m)
            risk_image[mask_temp] = np.sum(oridata[mask_temp])
            
        self._save_feature_image(path, dept, 'pred_risk', risk_image, raster)

        pred[~valid_mask] = -1

        pred = self.create_cluster(pred, dept, path, self.scale, mode, raster, valid_mask, 'pred')

        raster_ = np.copy(raster)
        raster_closing = morphology.closing(~np.isnan(raster_), morphology.disk(3))
        pred[np.isnan(raster_)] = np.nan

        raster_node_with_nan = np.where(np.isnan(data), np.nan, data)

        # Calculer une carte de distance pour chaque NaN vers le point le plus proche non-NaN
        # Cette méthode remplit les NaN par les valeurs les plus proches

        nan_mask = np.isnan(raster_node_with_nan)  # Masque des NaN
        filled_raster = raster_node_with_nan.copy()  # Copie du tableau original

        nearest_indices = ndimage.distance_transform_edt(
            nan_mask,
            return_distances=False,
            return_indices=True
        )

        # Utiliser les indices pour remplir les NaN avec les valeurs les plus proches
        filled_raster = raster_node_with_nan.copy()
        filled_raster[nan_mask] = raster_node_with_nan[tuple(nearest_indices[:, nan_mask])]

        # Remettre à jour raster_node
        data = filled_raster
        data[raster_closing == 0] = np.nan

        # Process post-watershed results
        pred = self._post_process_result(pred, raster, mask, node_already_predicted, 'graph') 
        self._save_feature_image(path, dept, 'pred_final', pred, raster)

    def my_watershed(self, dept, data, valid_mask, raster, path, vb, image_type):
        reducor = KMeans(n_clusters=self.n_reduce_class, random_state=42, n_init=10)
        reducor.fit(data[valid_mask].reshape(-1,1))
        data[valid_mask] = reducor.predict(data[valid_mask].reshape(-1,1))
        data[valid_mask] = order_class(reducor, data[valid_mask])
        data[~valid_mask] = 0

        data[valid_mask] = morphology.erosion(data, morphology.square(1))[valid_mask]
        self._save_feature_image(path, dept, f'{vb}_{image_type}', data, raster)

        # High Fire region
        # Détection des contours avec l'opérateur Sobel
        edges = filters.sobel(data)
        self._save_feature_image(path, dept, f'edges_{image_type}', edges, raster)

        # Créer une carte de distance
        distance = np.full(data.shape, fill_value=0.0)
        distance = ndi.distance_transform_edt(edges)
        self._save_feature_image(path, dept, f'distance_{image_type}', distance, raster)

        # Marquer les objets (régions connectées) dans l'image
        local_maxi = np.full(data.shape, fill_value=0)
        markers = np.full(data.shape, fill_value=0)
        local_maxi = morphology.local_maxima(distance)
        markers, _ = ndi.label(local_maxi)

        # Appliquer la segmentation Watershed
        pred = watershed(-data, markers, mask=data, connectivity=1)
        self._save_feature_image(path, dept, f'pred_watershed_{image_type}', pred, raster)
        return pred
        
    def create_geometry_with_watershed(self, dept, vec_base, path, mask, node_already_predicted, train_date):
        
        dir_raster = root_target
        dir_target = root_target
        raster = read_object(f'{dept}rasterScale0.pkl', dir_raster)
        assert raster is not None
        raster = raster[0]
        pred = np.full(raster.shape, fill_value=np.nan)
        valid_mask = (raster != -1) & (~np.isnan(raster))

        self.max_target_value = None

        vb = vec_base[0]

        data = self._process_base_data(vb, dept, dir_target, valid_mask, train_date)
        if data is None:
            logger.info(f'Can t find {vb}')
            exit(1)

        if self.max_target_value is None:
            self.max_target_value = np.nanmax(data)
        else:
            self.max_target_value = max(self.max_target_value, np.nanmax(data))
        oridata = np.copy(data)
        
        self._save_feature_image(path, dept, 'sum', data, raster, 0, self.max_target_value)

        pred = self.my_watershed(dept, data, valid_mask, raster, path, vb, 'pred')

        # Merge and split clusters
        umarker = np.unique(pred)
        umarker = umarker[(umarker != 0) & ~(np.isnan(umarker))]
        risk_image = np.full(oridata.shape, fill_value=np.nan)
        for m in umarker:
            mask_temp = (pred == m)
            risk_image[mask_temp] = np.sum(oridata[mask_temp])
        self._save_feature_image(path, dept, 'pred_risk', risk_image, raster)
        print(vec_base[1])
        pred = self.create_cluster(pred, dept, path, self.scale, vec_base[1], raster, valid_mask, 'pred')

        pred[~valid_mask] = -1

        # Process post-watershed results
        pred = self._post_process_result(pred, raster, mask, node_already_predicted, 'graph') 
        self._save_feature_image(path, dept, 'pred_final', pred, raster)
        
    def create_cluster(self, pred, dept, path, scale, mode, raster, valid_mask, type):
        
        #min_cluster_size = 1 + 3 * scale * (scale + 1)
        #max_cluster_size = (int)(min_cluster_size * 2.5)
        size = count_pixels_in_france_deg_square(deg_size=float(f'0.{scale}'))[-1]
        max_cluster_size = int(size + (0.05 * size))
        min_cluster_size = int(size - (0.05 * size))
        
        if mode == 'size':
            pred = merge_adjacent_clusters(pred, nb_attempt=self.nb_attempt, min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size, mode=mode, exclude_label=0, background=-1)

        elif mode == 'timeSeriesSimilarity':
            features = read_object(f'{dept}Influence.pkl', root_target)
            pred = merge_adjacent_clusters(pred, nb_attempt=self.nb_attempt, min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size, mode=mode, exclude_label=0, background=-1, features=features)

        elif mode == 'BrayCurtis':
            root_features = Path(f'/media/caron/X9 Pro/travaille/Thèse/csv/{dept}/raster/2x2')
            features = load_features(self.features_name, '2022-06-01', '2022-07-01', root_features)
            pred = merge_adjacent_clusters(pred, nb_attempt=self.nb_attempt, min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size, mode=mode, exclude_label=0, background=-1, features=features)
        else:
            raise ValueError

        valid_cluster = find_clusters(pred, min_cluster_size, 0, -1)
        self._save_feature_image(path, dept, 'pred_merge', pred, raster)
        pred_save = np.copy(pred).astype(float)
        pred_save[np.isnan(raster)] = np.nan
        pred_save = binary_closing_id(pred_save, disk(1))
        save_object(pred_save, f'pred_merge_{dept}.pkl', path / f'{self.scale}_{self.base}_{self.graph_method}' / dept)

        logger.info(np.unique(pred))
        logger.info(f'{dept} : We found {len(valid_cluster)} to build geometry.')
        mask_valid = np.isin(pred, valid_cluster)

        valid_cluster = [val + 1 for val in valid_cluster]
        pred[valid_mask] += 1
        pred = split_large_clusters(pred, max_cluster_size, min_cluster_size, size, valid_cluster)
        valid_cluster = [val - 1 for val in valid_cluster]
        pred[valid_mask] -= 1
        
        pred = pred.astype(float)
        self._save_feature_image(path, dept, f'{type}_split', pred, raster)

        return pred

    def create_geometry_with_regular(self, dept, path, mask, node_already_predicted):
        
        dir_raster = root_target
        raster = read_object(f'{dept}rasterScale0.pkl', dir_raster)
        assert raster is not None
        raster = raster[0]
        pred = np.full(raster.shape, fill_value=1)
        valid_mask = (raster != -1) & (~np.isnan(raster))
        pred[~valid_mask] = -1
        
        self.max_target_value = None

        #min_cluster_size = 1 + 3 * self.scale * (self.scale + 1)
        #max_cluster_size = (int)(min_cluster_size * 2.5)

        size = count_pixels_in_france_deg_square(deg_size=float(f'0.{self.scale}'))[-1]
        max_cluster_size = int(size + (0.10 * size))
        min_cluster_size = int(size - (0.10 * size))
        
        pred = split_large_clusters(pred, max_cluster_size, min_cluster_size, size, [-1])
        #pred = cluster_image_pixels(pred, max_cluster_size, min_cluster_size, [-1])
        pred -= 1
        pred = pred.astype(float)
        pred[~valid_mask] = np.nan
        self._save_feature_image(path, dept, 'pred', pred, raster)
        pred_save = np.copy(pred).astype(float)
        pred_save[np.isnan(raster)] = np.nan
        save_object(pred_save, f'pred_merge_{dept}.pkl', path / f'{self.scale}_{self.base}_{self.graph_method}' / dept)
        # Process post-watershed results
        self._post_process_result(pred, raster, mask, node_already_predicted, 'graph')
        self._save_feature_image(path, dept, 'pred_final', pred, raster)

    def _process_base_data(self, vb, dept, dir_target, valid_mask, train_date):
        
        if vb == 'risk':
            data = read_object(f'{dept}Influence.pkl', dir_target)
            if data is None:
                raise ValueError(f'Can t load {dir_target}/{dept}Influence.pkl')
            else:
                data = data[:, :, :allDates.index(train_date)]
                data = np.nansum(data, axis=2)
        else:
            raise ValueError(f'{vb} unknwon type. Try with risk')

        data[~valid_mask] = np.nan

        return data
    
    def _raster2(self, path, base):

        dir_bin = root_target
        dir_target = root_target
        dir_raster = root_target

        for dept in np.unique(self.departements):

            outputName = f'{dept}rasterScale0.pkl'
            raster = read_object(outputName, dir_raster)
            assert raster is not None
            raster = raster[0]

            mask = np.full(raster.shape, fill_value = np.nan)
            for uid in np.unique(raster[~np.isnan(raster)]):
                mask[raster == uid] = self.graph_ids[self.oriIds == uid]

            logger.info(f'{dept, mask.shape, raster.shape}')

            mask[np.isnan(raster)] = np.nan

            self.ids = np.copy(self.graph_ids)
            raster_node = np.copy(mask)

            save_object(mask, f'{dept}rasterScale{self.scale}_{base}_{self.graph_method}_node.pkl', path / 'raster')
            save_object(mask, f'{dept}rasterScale{self.scale}_{base}_{self.graph_method}.pkl', path / 'raster')
           
            unique_ids = np.unique(mask)
            unique_ids = unique_ids[~np.isnan(unique_ids)]
            plt.figure(figsize=(15, 5))
            plt.imshow(mask, label='ID')

            # Annotate each unique ID on the image
            for unique_id in unique_ids:
                # Find the positions of the current ID in the mask
                positions = np.column_stack(np.where(mask == unique_id))
                
                # Calculate the center of these positions
                center_y, center_x = np.mean(positions, axis=0).astype(int)
                
                # Place the text at the center
                plt.text(center_x, center_y, f'{unique_id}', color='white', fontsize=12, ha='center', va='center')

            plt.title(dept)
            plt.savefig(path / 'raster' / f'{dept}_{self.scale}_{self.base}_{self.graph_method}.png')
            plt.close('all')
            
            outputName = f'{dept}binScale0.pkl'
            bin = read_object(outputName, dir_bin)
            outputName = f'{dept}Influence.pkl'
            influence = read_object(outputName, dir_target)

            binImageScale, influenceImageScale = create_larger_scale_bin(mask, bin, influence, raster)
            save_object(binImageScale, f'{dept}binScale{self.scale}_{base}_{self.graph_method}.pkl', path / 'bin')
            save_object(influenceImageScale, f'{dept}InfluenceScale{self.scale}_{base}_{self.graph_method}.pkl', path / 'influence')

            binImageScale, influenceImageScale, = create_larger_scale_bin(raster_node, bin, influence, raster)
            save_object(binImageScale, f'{dept}binScale{self.scale}_{base}_{self.graph_method}_node.pkl', path / 'bin')
            save_object(influenceImageScale, f'{dept}InfluenceScale{self.scale}_{base}_{self.graph_method}_node.pkl', path / 'influence')

            self.numCluster = np.shape(np.unique(self.ids))[0]
            
    def _raster(self, path : Path,
                sinister : str, 
                dataset_name: str,
                sinister_encoding,
                resolution : str,
                base: str,
                train_date) -> None:
        """
        Create new raster mask
        """
        check_and_create_path(path)
        check_and_create_path(path / 'raster')
        check_and_create_path(path / 'bin')
        check_and_create_path(path / 'proba')

        self._raster2(path, base)

    def _save_feature_image(self, path, dept, vb, image, raster, mini=None, maxi=None):
        data = np.copy(image)
        check_and_create_path(path / 'features_geometry' / f'{self.scale}_{self.base}_{self.graph_method}' / dept)
        data = data.astype(float)
        data[np.isnan(raster)] = np.nan
        plt.figure(figsize=(15, 15))
        if mini is None:
            mini = np.nanmin(image)
        if maxi is None:
            maxi = np.nanmax(image)
        img = plt.imshow(data, vmin=mini, vmax=maxi)
        plt.colorbar(img)
        plt.title(vb)
        plt.savefig(path / 'features_geometry' / f'{self.scale}_{self.base}_{self.graph_method}' / dept / f'{vb}.png')
        plt.close('all')

    def _post_process_result(self, pred, raster, mask, node_already_predicted, graph_or_node='node'):

        array = np.full(self.ids.shape[0], fill_value=np.nan)
        pred = relabel_clusters(pred, node_already_predicted)
        X = np.unique(raster)
        X = X[~np.isnan(X)]
        for node in X:
            mask_node = self.oriIds == node
            mask2 = raster == node
            array[mask_node] = pred[mask2]

       # Incorporate nan hexagone
        if True in np.isnan(array[mask]):

            valid_mask = ~np.isnan(array[mask])
            invalid_mask = np.isnan(array[mask])

            if True in valid_mask and True in invalid_mask:
                #valid_coords = np.column_stack(np.where(valid_mask))
                valid_coords = np.asarray(list(zip(self.oriLongitude[mask][valid_mask], self.oriLatitudes[mask][valid_mask])))
                valid_values = array[mask][valid_mask]

                # Coordonnées des pixels à remplacer
                invalid_coords = np.asarray(list(zip(self.oriLongitude[mask][invalid_mask], self.oriLatitudes[mask][invalid_mask])))
                #invalid_coords = np.column_stack(np.where(invalid_mask))

                # Utilisation de KNeighbors pour prédire les valeurs manquantes
                knn = KNeighborsRegressor(n_neighbors=1, weights='distance')
                knn.fit(valid_coords, valid_values)

                predicted_values = knn.predict(invalid_coords)

                predicted_values = np.round(predicted_values)

                array[np.where(mask)[0][invalid_mask]] = predicted_values.astype(int)

        if graph_or_node == 'node': # graph = node
            self.ids[mask] = array[mask]
        elif graph_or_node == 'graph': # nodes per graph
            self.graph_ids[mask] = array[mask]
        else:
            logger.info(f'Unknow graph_or_node value {graph_or_node}')
            exit(1)
            
        #self.ids[mask] = relabel_clusters(self.ids[mask], node_already_predicted)
        return pred