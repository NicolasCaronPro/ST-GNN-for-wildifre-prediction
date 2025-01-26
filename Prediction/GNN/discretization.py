from GNN.statistical_model import *

class KMeansRisk:
    """
    Classe utilisant KMeans pour identifier les classes de risque.
    """
    
    def __init__(self, n_clusters):
        """
        :param n_clusters: Nombre de clusters pour KMeans.
        """
        self.n_clusters = n_clusters
        self.name = f'KMeansRisk_{n_clusters}'
        self.label_map = None

    def fit(self, X, y):
        """
        Ajuste le modèle KMeans aux données.

        :param X: Données à ajuster.
        """
        if np.unique(X).shape[0] < self.n_clusters:
            self.zeros = True
            return np.zeros(X.shape)
        
        self.zeros = False
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.model.fit(np.unique(X).reshape(-1,1))
        #self.model.fit(X)
        centroids = self.model.cluster_centers_

        if centroids.shape[1] > 1:
            magnitudes = np.linalg.norm(centroids, axis=1)
            sorted_indices = np.argsort(magnitudes)
        else:
            centroids = centroids.flatten()
            sorted_indices = np.argsort(centroids)

        self.label_map = {label: i for i, label in enumerate(sorted_indices)}

    def predict(self, X):
        """
        Prédit les classes en utilisant le modèle KMeans.

        :param X: Données à classer.
        :return: Classes prédites.
        """
        if self.zeros:
            return np.zeros_like(X)
        kmeans_labels = self.model.predict(X)
        return np.vectorize(self.label_map.get)(kmeans_labels)
    
class KMeansRiskZerosHandle:
    """
    Classe utilisant KMeans pour identifier les classes de risque.
    """
    
    def __init__(self, n_clusters):
        """
        :param n_clusters: Nombre de clusters pour KMeans.
        """
        self.n_clusters = n_clusters - 1
        self.name = f'KMeansRisk_{n_clusters}'
        self.label_map = None

    def fit(self, X, y):
        """
        Ajuste le modèle KMeans aux données.

        :param X: Données à ajuster.
        """
        X_val = X[X > 0].reshape(-1,1)
        if np.unique(X_val).shape[0] == 0:
            self.model = None
            return
        self.model = KMeans(n_clusters=min(self.n_clusters, np.unique(X_val).shape[0]), random_state=42, n_init=10)
        self.model.fit(X_val)
        centroids = self.model.cluster_centers_

        if centroids.shape[1] > 1:
            magnitudes = np.linalg.norm(centroids, axis=1)
            sorted_indices = np.argsort(magnitudes)
        else:
            centroids = centroids.flatten()
            sorted_indices = np.argsort(centroids)

        self.label_map = {label: i + 1 for i, label in enumerate(sorted_indices)} 

    def predict(self, X):
        """
        Prédit les classes en utilisant le modèle KMeans.

        :param X: Données à classer.
        :return: Classes prédites.
        """
        res = np.zeros(X.shape)
        if self.model is None:
            return res.reshape(-1,1)
        X_val = X[X > 0].reshape(-1,1)
        
        if X_val.shape[0] == 0:
            return res.reshape(-1)
        kmeans_labels = self.model.predict(X_val)
        res[X > 0] = np.vectorize(self.label_map.get)(kmeans_labels)
        return res.reshape(-1)

class ThresholdRisk:
    """
    Classe utilisant des seuils définis pour créer les classes de risque.
    """
    def __init__(self, thresholds):
        """
        :param thresholds: Liste de seuils pour définir les classes.
        """
        self.n_clusters = len(thresholds)
        self.thresholds = np.array(thresholds)
        self.name = f'ThresholdRisk_{thresholds}'

    def fit(self, X, y):
        """
        Aucun ajustement requis pour des seuils prédéfinis.
        """
        pass

    def predict(self, X):
        """
        Prédit les classes en fonction des seuils.

        :param X: Données à classer.
        :return: Classes prédites.
        """
        return np.digitize(X.flatten(), self.thresholds, right=True)

class QuantileRisk:
    """
    Classe utilisant des quartiles pour créer les classes de risque,
    avec une classe supplémentaire pour la valeur minimale.
    """
    def __init__(self):
        """
        Initialise sans seuils prédéfinis. Les seuils seront calculés lors de l'ajustement.
        """
        self.n_clusters = None
        self.thresholds = None
        self.min_value = None
        self.name = "QuantileRisk"

    def fit(self, X, y):
        """
        Calcule les seuils basés sur les quartiles des données.

        :param X: Données à partir desquelles calculer les seuils.
        """
        self.min_value = np.min(X)  # Déterminer la valeur minimale
        self.thresholds = np.unique(np.quantile(X[X > self.min_value].flatten(), [0.50, 0.65, 0.85]))

        logger.info(f'Threshold of quantile risk : {self.thresholds}')

        self.n_clusters = 5

    def predict(self, X):
        """
        Prédit les classes en fonction des quartiles calculés et une classe spécifique pour la valeur minimale.

        :param X: Données à classer.
        :return: Classes prédites.
        """
        if self.thresholds is None or self.min_value is None:
            raise ValueError("Les seuils (quartiles) et la valeur minimale n'ont pas été calculés. Appelez `fit` d'abord.")
        
        result = np.empty(X.shape).reshape(-1)

        # Initialiser les classes avec les indices basés sur les quartiles
        result[X.flatten() > self.min_value] = np.digitize(X.flatten()[X.flatten() > self.min_value], self.thresholds, right=True)
        
        result += 1
        # Assigner une classe distincte pour la valeur minimale
        result[X.flatten() == self.min_value] = 0

        return result

class KSRisk:
    def __init__(self, thresholds, score_col='ks_stat', thresh_col='optimal_score', dir_output=Path('./')):
        self.thresholds_ks = thresholds
        self.score_col = score_col
        self.thresh_col = thresh_col
        self.ks_results = None
        self.dir_output = dir_output
        self.name = f'KSRisk_{thresholds}'
        self.n_clusters = len(thresholds)

    def fit(self, y_pred, y_true):
        test_window = pd.DataFrame(index=np.arange(0, y_true.shape[0]))
        test_window['y_pred'] = y_pred
        test_window['y_true'] = y_true 
        self.ks_results = calculate_ks(test_window, 'y_pred', 'y_true', self.thresholds_ks, self.dir_output)
        return self

    def predict(self, y_pred):
        assert self.ks_results is not None
        new_pred = np.full(y_pred.shape[0], fill_value=0.0)
        max_value = self.ks_results[self.ks_results['threshold'] == self.thresholds_ks[0]][self.score_col].values
        for threshold in self.thresholds_ks:
            if self.ks_results[self.ks_results['threshold'] == threshold][self.score_col].values[0] >= max_value:
                new_pred[y_pred >= self.ks_results[self.ks_results['threshold'] == threshold][self.thresh_col].values[0]] = threshold
        return new_pred

class PreprocessorConv:
    def __init__(self, graph, kernel='Specialized', conv_type='laplace+mean', id_col=None, persistence=False,):
        """
        Initialize the PreprocessorConv.

        :param graph: An object containing `sequences_month` data structure.
        :param conv_type: Type of convolution to apply ('laplace', 'mean', 'laplace+mean', 'sum', or 'max').
        :param id_col: List of column names or indices representing IDs. Defaults to None.
        """
        assert conv_type in ['laplace', 'laplace+median', 'mean', 'laplace+mean', 'sum', 'max', 'median'], \
            "conv_type must be 'laplace', 'mean', 'laplace+mean', 'sum', or 'max'."
        self.graph = graph
        self.conv_type = conv_type
        self.id_col = id_col if id_col is not None else ['id']
        self.kernel = kernel
        self.persistence = persistence

    def _get_season_name(self, month):
        """
        Determine the season name based on the month.

        :param month: Integer representing the month (1 to 12).
        :return: Season name as a string ('medium', 'high', or 'low').
        """
        group_month = [
            [2, 3, 4, 5],    # Medium season
            [6, 7, 8, 9],    # High season
            [10, 11, 12, 1]  # Low season
        ]
        names = ['medium', 'high', 'low']
        for i, group in enumerate(group_month):
            if month in group:
                return names[i]
        raise ValueError(f"Month {month} does not belong to any season group.")

    def _laplace_convolution(self, X, kernel_size):
        """
        Apply Laplace convolution using a custom kernel from Astropy.

        :param X: Input array for convolution.
        :param kernel_size: Size of the convolution kernel.
        :return: Convoluted array.
        """
        # Create Laplace kernel
        kernel_daily = np.abs(np.arange(-(kernel_size // 2), kernel_size // 2 + 1))
        kernel_daily += 1
        kernel_daily = 1 / kernel_daily
        kernel_daily = kernel_daily.reshape(-1,1)

        if self.persistence:
            if kernel_size > 1:
                kernel_daily[:kernel_size // 2] = 0
            else:
                return np.zeros(X.shape).reshape(-1)

        if np.unique(kernel_daily)[0] == 0 and np.unique(kernel_daily).shape[0] == 1:
            kernel_daily += 1
            
        # Perform convolution using Astropy
        #print(np.unique(X))
        #print(np.unique(kernel_daily))
        res = convolve_fft(X.reshape(-1), kernel_daily.reshape(-1), normalize_kernel=False, fill_value=0.)

        return res
    
    def _mean_convolution(self, X, kernel_size):
        """
        Apply mean convolution using a custom kernel from Astropy.

        :param X: Input array for convolution.
        :param kernel_size: Size of the convolution kernel.
        :return: Convoluted array.
        """
        # Create mean kernel
        if kernel_size == 1:
            return np.zeros_like(X).reshape(-1)
        
        kernel_season = np.ones(kernel_size, dtype=float)
        kernel_season /= kernel_size  # Normalize the kernel

        if self.persistence:
            kernel_season[:kernel_size // 2] = 0
 
        #kernel_season[kernel_size // 2] = 0
        res = convolve_fft(X.reshape(-1), kernel_season.reshape(-1), normalize_kernel=False, fill_value=0.)

        return res
    
    def median_convolution(self, X, kernel_size):
        """
        Apply median convolution using a custom kernel, excluding the center pixel.

        Parameters:
            X (array-like): Input array for convolution.
            kernel_size (int): Size of the convolution kernel (must be odd).

        Returns:
            numpy.ndarray: Convoluted array with median filtering applied.
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer.")

        if kernel_size == 1:
            return np.zeros(X.shape).reshape(-1)

        def custom_median_filter(window):
            # Exclude the center element from the kernel (middle pixel)
            center = len(window) // 2
            if self.persistence:
                window[center:] = np.nan  # Exclude earlier values
            window = np.delete(window, center)  # Remove center element
            return np.nanmedian(window)
        
        # Apply the custom median filter
        return generic_filter(X, custom_median_filter, size=kernel_size).reshape(-1)


    def _sum_convolution(self, X, kernel_size):
        """
        Apply sum operation on the input with a sliding window of size `kernel_size`.

        :param X: Input array for summation (2D: samples x features).
        :param kernel_size: Size of the sliding window.
        :return: Array where each element is the sum of elements in the sliding window.
        """
        from scipy.ndimage import uniform_filter1d

        if self.persistence:
            mask = np.zeros_like(X, dtype=float)
            mask[kernel_size // 2:] = 0
            mask[:kernel_size // 2] = 1  # Persistence applies only to later values
            X = X * mask

        return uniform_filter1d(X, size=kernel_size, mode='constant', origin=0, axis=0) * kernel_size

    def _max_convolution(self, X, kernel_size):
        """
        Apply max operation on the input with a sliding window of size `kernel_size`.

        :param X: Input array for max operation (2D: samples x features).
        :param kernel_size: Size of the sliding window.
        :return: Array where each element is the max of elements in the sliding window.
        """
        from scipy.ndimage import maximum_filter1d

        if self.persistence:
            mask = np.zeros_like(X, dtype=float)
            mask[kernel_size // 2:] = 0
            mask[:kernel_size // 2] = 1
            X = X * mask

        res = maximum_filter1d(X, size=kernel_size, mode='constant', origin=0, axis=0)
        return res


    def apply(self, X, ids):
        """
        Apply the convolution based on the given type to the input data.

        :param X: Input array of shape (n_samples, n_features).
        :param ids: Array of shape (n_samples, len(id_col)) if `id_col` is a list, otherwise (n_samples, 1).
        :return: Processed array with convolutions applied.
        """
        X_processed = np.zeros_like(X, dtype=np.float32).reshape(-1)
        unique_ids = np.unique(ids, axis=0)

        for unique_id in unique_ids:

            if len(self.id_col) > 1 and not isinstance(self.id_col, str):
                mask = (ids[:, 0] == unique_id[0]) & (ids[:, 1] == unique_id[1])
            else:
                mask = ids[:, 0] == unique_id[0]

            if not np.any(mask):
                continue

            if self.kernel == 'Specialized':
                # Handle month_non_encoder
                if 'month_non_encoder' in self.id_col:
                    month_idx = self.id_col.index('month_non_encoder')
                    month = unique_id[month_idx]
                    season_name = self._get_season_name(month)
                    kernel_size = int(self.graph.sequences_month[season_name][unique_id[1]]['mean_size'])
                else:
                    raise ValueError(
                        "Error: 'month_non_encoder' is not specified in id_col. Please include it in id_col to proceed."
                    )
            else:
                kernel_size = int(self.kernel) + 2 # Add 2 as self.kernel == num_day before and after
            
            # Apply Laplace convolution if specified
            if self.conv_type == 'laplace':
                laplace_result = self._laplace_convolution(X[mask], kernel_size)
                X_processed[mask] = laplace_result.reshape(-1)

            # Apply mean convolution if specified
            elif self.conv_type == 'mean':
                mean_result = self._mean_convolution(X[mask], kernel_size)
                #X_processed[mask] = (X[mask].reshape(-1) + mean_result).reshape(-1)
                X_processed[mask] = mean_result.reshape(-1)

            # Apply laplace+mean convolutions if specified
            elif self.conv_type == 'laplace+mean':
                laplace_result = self._laplace_convolution(X[mask], kernel_size)
                """if unique_id[1] == 4:
                    fig, ax = plt.subplots(1, figsize=(15,6))
                    ax.plot(laplace_result)
                    ax.plot(X[mask])
                    plt.show()"""
                mean_result = self._mean_convolution(X[mask], kernel_size)
                X_processed[mask] = (laplace_result + mean_result).reshape(-1)

            # Apply sum operation if specified
            elif self.conv_type == 'sum':
                sum_result = self._sum_convolution(X[mask], kernel_size)
                X_processed[mask] = sum_result.reshape(-1)

            # Apply max operation if specified
            elif self.conv_type == 'max':
                max_result = self._max_convolution(X[mask], kernel_size)
                X_processed[mask] = max_result.reshape(-1)

            elif self.conv_type == 'median':
                med_result = self.median_convolution(X[mask], kernel_size)
                #X_processed[mask] = (X[mask].reshape(-1) + med_result).reshape(-1)
                X_processed[mask] = med_result.reshape(-1)

            elif self.conv_type == 'laplace+median':
                laplace_result = self._laplace_convolution(X[mask], kernel_size)
                med_result = self.median_convolution(X[mask], kernel_size)
                X_processed[mask] = (laplace_result + med_result).reshape(-1)

            else:
                raise ValueError(
                    f"Error: conv_type must be in  ['laplace', 'mean', 'laplace+mean', 'sum', 'max'], got {self.conv_type}"
                )
            
        return X_processed
    
class ScalerClassRisk:
    def __init__(self, col_id, dir_output, target, scaler=None, class_risk=None, preprocessor=None):
        """
        Initialize the ScalerClassRisk.

        :param col_id: Column ID used to group data.
        :param dir_output: Directory to save output files (e.g., histograms).
        :param target: Target name used for labeling outputs.
        :param scaler: A scaler object (e.g., StandardScaler) for normalization. If None, no scaling is applied.
        :param class_risk: An object with `fit` and `predict` methods for risk classification.
        :param preprocessor: An object with an `apply` method to preprocess X before fitting.
        """
        assert class_risk is not None
        self.n_clusters = 5
        self.col_id = col_id

        if scaler is None:
            scaler_name = None
        elif isinstance(scaler, MinMaxScaler):
            scaler_name = 'MinMax'
        elif isinstance(scaler, StandardScaler):
            scaler_name = 'Standard'
        elif isinstance(scaler, RobustScaler):
            scaler_name = 'Robust'
        else:
            raise ValueError(f'{scaler} wrong value')
        
        class_risk_name = class_risk.name
        preprocessor_name = f'{preprocessor.conv_type}_{preprocessor.kernel}' if preprocessor is not None else None
        self.name = f'ScalerClassRisk_{preprocessor_name}_{scaler_name}_{class_risk_name}_{target}'
        self.dir_output = dir_output / self.name
        self.target = target
        self.scaler = scaler
        self.class_risk = class_risk
        self.preprocessor = preprocessor
        self.preprocessor_id_col = preprocessor.id_col if preprocessor is not None else None
        check_and_create_path(self.dir_output)
        self.models_by_id = {}
        self.is_fit = False

    def fit(self, X, sinisters, ids, ids_preprocessor=None):
        """
        Fit models for each unique ID and calculate statistics.

        :param X: Array of values to scale and classify.
        :param sinisters: Array of sinisters values corresponding to each value in X.
        :param ids: Array of IDs corresponding to each value in X.
        :param ids_preprocessor: IDs to use for preprocessing (if preprocessor is not None).
        """
        self.is_fit = True
        logger.info(f'########################################## {self.name} ##########################################')

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if sinisters is not None and len(sinisters.shape) == 1:
            sinisters = sinisters.reshape(-1)

        X_ = np.copy(X)

        # Apply preprocessor if provided
        if self.preprocessor is not None:
            if ids_preprocessor is None:
                raise ValueError("ids_preprocessor must be provided when preprocessor is not None.")
            X = self.preprocessor.apply(X, ids_preprocessor)

        lambda_function = lambda x: max(x, 1)
        self.models_by_id = {}

        for unique_id in np.unique(ids):
            mask = ids == unique_id
            X_id = X[mask]

            if sinisters is not None:
                sinisters_id = sinisters[mask]
            else:
                sinisters_id = None

            # Scale data if scaler is provided
            if self.scaler is not None:
                scaler = deepcopy(self.scaler)
                scaler.fit(X_id.reshape(-1, 1))
                X_scaled = scaler.transform(X_id.reshape(-1, 1))
            else:
                scaler = None
                X_scaled = X_id

            X_scaled = X_scaled.astype(np.float32)

            # Fit the class_risk model
            class_risk = deepcopy(self.class_risk)

            if len(X_scaled.shape) == 1:
                X_scaled = np.reshape(X_scaled, (-1,1))

            class_risk.fit(X_scaled, sinisters_id)

            # Predict classes for current ID
            classes = class_risk.predict(X_scaled)

            if sinisters is not None:
            
                sinisters = sinisters.reshape(-1)
                # Apply the lambda function using np.vectorize for the condition `sinisters > 0`
                classes[sinisters[mask] > 0] = np.vectorize(lambda_function)(classes[sinisters[mask] > 0])

                # Calculate statistics for each class
                sinisters_mean = []
                sinisters_min = []
                sinisters_max = []

                for cls in np.unique(classes):
                    class_mask = classes == cls
                    sinisters_class = sinisters_id[class_mask]
                    sinisters_mean.append(np.mean(sinisters_class))
                    sinisters_min.append(np.min(sinisters_class))
                    sinisters_max.append(np.max(sinisters_class))

                sinisters_mean = np.array(sinisters_mean)
                sinisters_min = np.array(sinisters_min)
                sinisters_max = np.array(sinisters_max)
            else:
                sinisters_mean = None
                sinisters_min = None
                sinisters_max = None

            self.models_by_id[unique_id] = {
                'scaler': scaler,
                'class_risk': class_risk,
                'sinistres_mean': sinisters_mean,
                'sinistres_min': sinisters_min,
                'sinistres_max': sinisters_max
            }

        if sinisters is None:
            return

        pred = self.predict(X_, sinisters, ids, ids_preprocessor)

        # Plot histogram of the classes in X[mask]
        plt.figure(figsize=(8, 6))
        plt.hist(pred, bins=np.unique(pred).shape[0], color='blue', alpha=0.7, edgecolor='black')
        plt.title(f'Histogram')
        plt.xlabel('Values')
        plt.ylabel('Frequency')

        # Save the plot
        output_path = os.path.join(self.dir_output, f'histogram_train.png')
        plt.savefig(output_path)
        plt.close()
        for cl in np.unique(pred):
            logger.info(f'{cl} -> {pred[pred == cl].shape[0]}')

    def predict(self, X, sinisters, ids, ids_preprocessor=None):
        """
        Predict class labels using the appropriate model for each ID.

        :param X: Array of values to predict.
        :param ids: Array of IDs corresponding to each value in X.
        :param ids_preprocessor: IDs to use for preprocessing (if preprocessor is not None).
        :return: Array of predicted class labels.
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # Apply preprocessor if provided
        if self.preprocessor is not None:
            if ids_preprocessor is None:
                raise ValueError("ids_preprocessor must be provided when preprocessor is not None.")
            X = self.preprocessor.apply(X, ids_preprocessor)

        predictions = np.zeros(X.shape[0], dtype=int)

        for unique_id, model in self.models_by_id.items():
            if unique_id not in np.unique(ids):
                continue

            mask = ids == unique_id
            scaler = model['scaler']
            class_risk = model['class_risk']

            # Scale data if scaler exists
            if scaler is not None:
                X_scaled = scaler.transform(X[mask].reshape(-1,1))
            else:
                X_scaled = X[mask]

            if len(X_scaled.shape) == 1:
                X_scaled = np.reshape(X_scaled, (-1,1))

            predictions[mask] = class_risk.predict(X_scaled.astype(np.float32))
        
        predictions[predictions >= self.n_clusters] = self.n_clusters - 1

        # Define the lambda function
        lambda_function = lambda x: max(x, 1)

        if sinisters is not None:
            sinisters = sinisters.reshape(-1)
            # Apply the lambda function using np.vectorize for the condition `sinisters > 0`
            predictions[sinisters > 0] = np.vectorize(lambda_function)(predictions[sinisters > 0])

        """if ids_preprocessor is not None:
            fig, ax = plt.subplots(2, figsize=(15,5))
            ax[0].plot(predictions[ids_preprocessor[:, 1] == 4])
            ax[1].plot(X[ids_preprocessor[:, 1] == 4])
            plt.show()"""

        return predictions

    def fit_predict(self, X, ids, sinisters, ids_preprocessor=None):
        self.fit(X, ids, sinisters, ids_preprocessor)
        return self.predict(X, ids, ids_preprocessor)

    def predict_stat(self, X, ids, stat_key):
        """
        Generic method to predict statistics (mean, min, max) for sinistres based on the class of each sample.

        :param X: Array of input values.
        :param ids: Array of IDs corresponding to each value in X.
        :param stat_key: Key to fetch the required statistic ('sinistres_mean', 'sinistres_min', 'sinistres_max').
        :return: Array of predicted statistics for each sample based on its class.
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        predictions = np.zeros(X.shape[0])

        for unique_id, model in self.models_by_id.items():
            if unique_id not in np.unique(ids):
                continue

            mask = ids == unique_id
            
            stats = model[stat_key]
            predictions[mask] = np.array([stats[int(cls)] for cls in X[mask]])

        return predictions

    def predict_nbsinister(self, X, ids):
        """
        Predict the mean sinistres for the class of each instance in X.

        :param X: Array of input values.
        :param ids: Array of IDs corresponding to each value in X.
        :return: Array of predicted mean sinistres for each class.
        """
        return self.predict_stat(X, ids, stat_key='sinistres_mean')

    def predict_nbsinister_min(self, X, ids):
        """
        Predict the minimum sinistres for the class of each instance in X.

        :param X: Array of input values.
        :param ids: Array of IDs corresponding to each value in X.
        :return: Array of predicted minimum sinistres for each class.
        """
        return self.predict_stat(X, ids, stat_key='sinistres_min')

    def predict_nbsinister_max(self, X, ids):
        """
        Predict the maximum sinistres for the class of each instance in X.

        :param X: Array of input values.
        :param ids: Array of IDs corresponding to each value in X.
        :return: Array of predicted maximum sinistres for each class.
        """
        return self.predict_stat(X, ids, stat_key='sinistres_max')
    
    def predict_risk(self, X, sinisters, ids, ids_preprocessor=None):
        if self.is_fit:
            return self.predict(X, sinisters, ids, ids_preprocessor)
        else:
            return self.fit_predict(X, sinisters, ids, ids_preprocessor)

# Fonction pour calculer la somme dans une fenêtre de rolling, incluant les fenêtres inversées
def calculate_rolling_sum(dataset, column, shifts, group_col, func):
    """
    Calcule la somme rolling sur une fenêtre donnée pour chaque groupe.
    Combine les fenêtres normales et inversées.

    :param dataset: Le DataFrame Pandas.
    :param column: Colonne sur laquelle appliquer le rolling.
    :param shifts: Taille de la fenêtre rolling.
    :param group_col: Colonne pour le groupby.
    :param func: Fonction à appliquer sur les fenêtres.
    :return: Colonne calculée avec la somme rolling bidirectionnelle.
    """
    if shifts == 0:
        return dataset[column].values
    
    dataset.reset_index(drop=True)
    
    # Rolling forward
    forward_rolling = dataset.groupby(group_col)[column].rolling(window=shifts).apply(func).values
    forward_rolling[np.isnan(forward_rolling)] = 0
    
    # Rolling backward (inversé)
    backward_rolling = (
        dataset.iloc[::-1]
        .groupby(group_col)[column]
        .rolling(window=shifts, min_periods=1)
        .apply(func, raw=True)
        .iloc[::-1]  # Remettre dans l'ordre original
    ).values
    backward_rolling[np.isnan(backward_rolling)] = 0

    # Somme des deux fenêtres
    return forward_rolling + backward_rolling - dataset[column].values
    #return forward_rolling

def calculate_rolling_sum_per_col_id(dataset, column, shifts, group_col, func):
    """
    Calcule la somme rolling sur une fenêtre donnée pour chaque col_id sans utiliser des boucles sur les indices internes.
    Combine les fenêtres normales et inversées, tout en utilisant rolling.

    :param dataset: Le DataFrame Pandas.
    :param column: Colonne sur laquelle appliquer le rolling.
    :param shifts: Taille de la fenêtre rolling.
    :param group_col: Colonne identifiant les groupes (col_id).
    :param func: Fonction à appliquer sur les fenêtres.
    :return: Numpy array contenant les sommes rolling bidirectionnelles pour chaque col_id.
    """
    if shifts == 0:
        return dataset[column].values

    # Initialiser un tableau pour stocker les résultats
    result = np.zeros(len(dataset))

    # Obtenir les valeurs uniques de col_id
    unique_col_ids = dataset[group_col].unique()

    # Parcourir chaque groupe col_id
    for col_id in unique_col_ids:
        # Filtrer le groupe correspondant
        group_data = dataset[dataset[group_col] == col_id]
        group_data.sort_values('date', inplace=True)

        # Calculer rolling forward
        forward_rolling = group_data[column].rolling(window=shifts, min_periods=1).apply(func, raw=True).values

        # Calculer rolling backward (fenêtres inversées)
        backward_rolling = (
            group_data[column][::-1]
            .rolling(window=shifts, min_periods=1)
            .apply(func, raw=True)[::-1]
            .values
        )

        # Combine forward et backward
        group_result = forward_rolling + backward_rolling - group_data[column].values

        # Affecter le résultat au tableau final
        result[group_data.index] = group_result

    return result

def class_window_sum(dataset, group_col, column, shifts):
    # Initialize a column to store the rolling aggregation
    column_name = f'nbsinister_sum_{shifts}'
    dataset[column_name] = 0.0

    # Case when window_size is 1
    if shifts == 0:
        dataset[column_name] = dataset[column].values
        return dataset
    else:
        # For each unique graph_id
        for graph_id in dataset[group_col].unique():
            # Filter data for the current graph_id
            df_graph = dataset[dataset[group_col] == graph_id]

            # Iterate through each row in df_graph
            for idx, row in df_graph.iterrows():
                # Define the window bounds
                date_min = row['date'] - shifts
                date_max = row['date'] + shifts
                
                # Filter rows within the date window
                window_df = df_graph[(df_graph['date'] >= date_min) & (df_graph['date'] <= date_max)]
                
                # Apply the aggregation function
                dataset.at[idx, column_name] = window_df[column].sum()
    return dataset

def class_window_max(dataset, group_col, column, shifts):
    # Initialize a column to store the rolling aggregation
    column_name = f'nbsinister_max_{shifts}'
    dataset[column_name] = 0.0

    # Case when window_size is 1
    if shifts == 0:
        dataset[column_name] = dataset[column].values
        return dataset
    else:
        # For each unique graph_id
        for graph_id in dataset[group_col].unique():
            # Filter data for the current graph_id
            df_graph = dataset[dataset[group_col] == graph_id]

            # Iterate through each row in df_graph
            for idx, row in df_graph.iterrows():
                # Define the window bounds
                date_min = row['date'] - shifts
                date_max = row['date'] + shifts
                
                # Filter rows within the date window
                window_df = df_graph[(df_graph['date'] >= date_min) & (df_graph['date'] <= date_max)]
                
                # Apply the aggregation function
                dataset.at[idx, column_name] = window_df[column].max()
    
    return dataset

def post_process_model(train_dataset, val_dataset, test_dataset, dir_post_process, graph):

    graph_method = graph.graph_method

    new_cols = []

    if graph_method == 'node':
        train_dataset_ = train_dataset.copy(deep=True)
        val_dataset_ = val_dataset.copy(deep=True)
        test_dataset_ = test_dataset.copy(deep=True)
    else:
        def keep_one_per_pair(dataset):
            # Supprime les doublons en gardant uniquement la première occurrence par paire (graph_id, date)
            return dataset.drop_duplicates(subset=['graph_id', 'date'], keep='first')

        train_dataset_ = keep_one_per_pair(train_dataset)
        val_dataset_ = keep_one_per_pair(val_dataset)
        test_dataset_ = keep_one_per_pair(test_dataset)

    res = {}

    ####################################################################################

    """obj1 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target='nbsinister', scaler=MinMaxScaler(), class_risk=ThresholdRisk([0.05, 0.15, 0.30, 0.60, 1.0]))

    obj1.fit(train_dataset_['nbsinister'].values, train_dataset_['nbsinister'].values, train_dataset_['departement'].values)

    train_dataset_['nbsinister-MinMax-thresholds-5-Class-Dept'] = obj1.predict(train_dataset_['nbsinister'].values, train_dataset_['nbsinister'].values, train_dataset_['departement'].values)
    val_dataset_['nbsinister-MinMax-thresholds-5-Class-Dept'] = obj1.predict(val_dataset_['nbsinister'].values, val_dataset_['nbsinister'].values, val_dataset_['departement'].values)
    test_dataset_['nbsinister-MinMax-thresholds-5-Class-Dept'] = obj1.predict(test_dataset_['nbsinister'].values, test_dataset_['nbsinister'].values, test_dataset_['departement'].values)
    
    res[obj1.name] = obj1

    new_cols.append('nbsinister-MinMax-thresholds-5-Class-Dept')"""

    ####################################################################################

    """obj11 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target='nbsinister', scaler=RobustScaler(), class_risk=QuantileRisk())

    obj11.fit(train_dataset_['nbsinister'].values, train_dataset_['nbsinister'].values, train_dataset_['departement'].values)

    train_dataset_['nbsinister-Robust-Quantile-5-Class-Dept'] = obj11.predict(train_dataset_['nbsinister'].values, train_dataset_['nbsinister'].values, train_dataset_['departement'].values)
    val_dataset_['nbsinister-Robust-Quantile-5-Class-Dept'] = obj11.predict(val_dataset_['nbsinister'].values, val_dataset_['nbsinister'].values, val_dataset_['departement'].values)
    test_dataset_['nbsinister-Robust-Quantile-5-Class-Dept'] = obj11.predict(test_dataset_['nbsinister'].values, test_dataset_['nbsinister'].values, test_dataset_['departement'].values)
    
    res[obj11.name] = obj11

    new_cols.append('nbsinister-Robust-Quantile-5-Class-Dept')"""

    ####################################################################################
    
    obj2 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target='nbsinister', scaler=None, class_risk=KMeansRiskZerosHandle(5))

    obj2.fit(train_dataset_['nbsinister'].values, train_dataset_['nbsinister'].values, train_dataset_['departement'].values)

    train_dataset_['nbsinister-kmeans-5-Class-Dept'] = obj2.predict(train_dataset_['nbsinister'].values,  train_dataset_['nbsinister'].values, train_dataset_['departement'].values)
    val_dataset_['nbsinister-kmeans-5-Class-Dept'] = obj2.predict(val_dataset_['nbsinister'].values,  val_dataset_['nbsinister'].values, val_dataset_['departement'].values)
    test_dataset_['nbsinister-kmeans-5-Class-Dept'] = obj2.predict(test_dataset_['nbsinister'].values,  test_dataset_['nbsinister'].values, test_dataset_['departement'].values)
    
    res[obj2.name] = obj2

    new_cols.append('nbsinister-kmeans-5-Class-Dept')

    ###################################################################################

    """obj4 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target='risk', scaler=None, class_risk=KMeansRisk(5))

    obj4.fit(train_dataset_['risk'].values, train_dataset_['nbsinister'].values, train_dataset_['departement'].values)

    train_dataset_['risk-kmeans-5-Class-Dept'] = obj4.predict(train_dataset_['risk'].values, train_dataset_['nbsinister'].values, train_dataset_['departement'].values)
    val_dataset_['risk-kmeans-5-Class-Dept'] = obj4.predict(val_dataset_['risk'].values, val_dataset_['nbsinister'].values, val_dataset_['departement'].values)
    test_dataset_['risk-kmeans-5-Class-Dept'] = obj4.predict(test_dataset_['risk'].values, test_dataset_['nbsinister'].values, test_dataset_['departement'].values)
    
    res[obj4.name] = obj4

    new_cols.append('risk-kmeans-5-Class-Dept')"""

    ##################################################################################

    """obj3 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target='risk-nbsinister', scaler=RobustScaler(), class_risk=KMeansRisk(5))

    obj3.fit(train_dataset_[['risk', 'nbsinister']].values, train_dataset_['nbsinister'].values, train_dataset_['departement'].values)

    train_dataset_['risk-nbsinister-Robust-kmeans-5-Class-Dept'] = obj3.predict(train_dataset_[['risk', 'nbsinister']].values, train_dataset_['nbsinister'].values, train_dataset_['departement'].values)
    val_dataset_['risk-nbsinister-Robust-kmeans-5-Class-Dept'] = obj3.predict(val_dataset_[['risk', 'nbsinister']].values, val_dataset_['nbsinister'].values, val_dataset_['departement'].values)
    test_dataset_['risk-nbsinister-Robust-kmeans-5-Class-Dept'] = obj3.predict(test_dataset_[['risk', 'nbsinister']].values, test_dataset_['nbsinister'].values, test_dataset_['departement'].values)

    res[obj3.name] = obj3

    new_cols.append('risk-nbsinister-Robust-kmeans-5-Class-Dept')"""

    #######################################################################################
    
    """shifts = 3

    obj14 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target=f'nbsinister-max-{shifts}+{shifts}', scaler=None, class_risk=KMeansRiskZerosHandle(5))

    train_dataset_.sort_values(by=['graph_id', 'date'], inplace=True)
    val_dataset_.sort_values(by=['graph_id', 'date'], inplace=True)
    test_dataset_.sort_values(by=['graph_id', 'date'], inplace=True)

    # Application sur les jeux de données
    train_dataset_ = class_window_max(
        dataset=train_dataset_.copy(deep=True),
        column='nbsinister',
        shifts=shifts,
        group_col='graph_id',
    )

    val_dataset_ = class_window_max(
            dataset=val_dataset_.copy(deep=True),
            column='nbsinister',
            shifts=shifts,
            group_col='graph_id',
        )
    
    test_dataset_ = class_window_max(
                dataset=test_dataset_.copy(deep=True),
                column='nbsinister',
                shifts=shifts,
                group_col='graph_id',
            )

    obj14.fit(train_dataset_[f'nbsinister_max_{shifts}'].values, train_dataset_['nbsinister'].values, train_dataset_['departement'].values)

    train_dataset_[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj14.predict(train_dataset_[f'nbsinister_max_{shifts}'].values, train_dataset_['nbsinister'].values, train_dataset_['departement'].values)
    val_dataset_[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj14.predict(val_dataset_[f'nbsinister_max_{shifts}'].values, val_dataset_['nbsinister'].values, val_dataset_['departement'].values)
    test_dataset_[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj14.predict(test_dataset_[f'nbsinister_max_{shifts}'].values, test_dataset_['nbsinister'].values, test_dataset_['departement'].values)

    res[obj14.name] = obj14

    new_cols.append(f'nbsinister-max-{shifts}-kmeans-5-Class-Dept')"""

    #######################################################################################
    """shifts = 2
    
    obj15 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target=f'nbsinister-max-{shifts}+{shifts}', scaler=None, class_risk=KMeansRiskZerosHandle(5))

    train_dataset_.sort_values(by=['graph_id', 'date'], inplace=True)
    val_dataset_.sort_values(by=['graph_id', 'date'], inplace=True)
    test_dataset_.sort_values(by=['graph_id', 'date'], inplace=True)

    # Application sur les jeux de données
    train_dataset_ = class_window_max(
        dataset=train_dataset_.copy(deep=True),
        column='nbsinister',
        shifts=shifts,
        group_col='graph_id',
    )

    val_dataset_ = class_window_max(
            dataset=val_dataset_.copy(deep=True),
            column='nbsinister',
            shifts=shifts,
            group_col='graph_id',
        )
    
    test_dataset_ = class_window_max(
                dataset=test_dataset_.copy(deep=True),
                column='nbsinister',
                shifts=shifts,
                group_col='graph_id',
            )

    obj15.fit(train_dataset_[f'nbsinister_max_{shifts}'].values, train_dataset_['nbsinister'].values, train_dataset_['departement'].values)

    train_dataset_[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj15.predict(train_dataset_[f'nbsinister_max_{shifts}'].values, train_dataset_['nbsinister'].values, train_dataset_['departement'].values)
    val_dataset_[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj15.predict(val_dataset_[f'nbsinister_max_{shifts}'].values, val_dataset_['nbsinister'].values, val_dataset_['departement'].values)
    test_dataset_[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj15.predict(test_dataset_[f'nbsinister_max_{shifts}'].values, test_dataset_['nbsinister'].values, test_dataset_['departement'].values)

    res[obj15.name] = obj15

    new_cols.append(f'nbsinister-max-{shifts}-kmeans-5-Class-Dept')"""

    #######################################################################################
    """shifts = 1
    
    obj16 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target=f'nbsinister-max-{shifts}+{shifts}', scaler=None, class_risk=KMeansRiskZerosHandle(5))

    train_dataset_.sort_values(by=['graph_id', 'date'], inplace=True)
    val_dataset_.sort_values(by=['graph_id', 'date'], inplace=True)
    test_dataset_.sort_values(by=['graph_id', 'date'], inplace=True)

    # Application sur les jeux de données
    train_dataset_ = class_window_max(
        dataset=train_dataset_.copy(deep=True),
        column='nbsinister',
        shifts=shifts,
        group_col='graph_id',
    )

    val_dataset_ = class_window_max(
            dataset=val_dataset_.copy(deep=True),
            column='nbsinister',
            shifts=shifts,
            group_col='graph_id',
        )
    
    test_dataset_ = class_window_max(
                dataset=test_dataset_.copy(deep=True),
                column='nbsinister',
                shifts=shifts,
                group_col='graph_id',
            )

    obj16.fit(train_dataset_[f'nbsinister_max_{shifts}'].values, train_dataset_['nbsinister'].values, train_dataset_['departement'].values)

    train_dataset_[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj16.predict(train_dataset_[f'nbsinister_max_{shifts}'].values,  train_dataset_['nbsinister'].values, train_dataset_['departement'].values)
    val_dataset_[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj16.predict(val_dataset_[f'nbsinister_max_{shifts}'].values,  val_dataset_['nbsinister'].values,val_dataset_['departement'].values)
    test_dataset_[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj16.predict(test_dataset_[f'nbsinister_max_{shifts}'].values, test_dataset_['nbsinister'].values, test_dataset_['departement'].values)

    res[obj16.name] = obj16

    new_cols.append(f'nbsinister-max-{shifts}-kmeans-5-Class-Dept')"""

    #######################################################################################
    """shifts = 0
    
    obj17 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target=f'nbsinister-max-{shifts}+{shifts}', scaler=None, class_risk=KMeansRiskZerosHandle(5))

    train_dataset_.sort_values(by=['graph_id', 'date'], inplace=True)
    val_dataset_.sort_values(by=['graph_id', 'date'], inplace=True)
    test_dataset_.sort_values(by=['graph_id', 'date'], inplace=True)

    # Application sur les jeux de données
    train_dataset_ = class_window_max(
        dataset=train_dataset_.copy(deep=True),
        column='nbsinister',
        shifts=shifts,
        group_col='graph_id',
    )

    val_dataset_ = class_window_max(
            dataset=val_dataset_.copy(deep=True),
            column='nbsinister',
            shifts=shifts,
            group_col='graph_id',
        )
    
    test_dataset_ = class_window_max(
                dataset=test_dataset_.copy(deep=True),
                column='nbsinister',
                shifts=shifts,
                group_col='graph_id',
            )

    obj17.fit(train_dataset_[f'nbsinister_max_{shifts}'].values, train_dataset_['nbsinister'].values, train_dataset_['departement'].values)

    train_dataset_[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj17.predict(train_dataset_[f'nbsinister_max_{shifts}'].values, train_dataset_['nbsinister'].values, train_dataset_['departement'].values)
    val_dataset_[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj17.predict(val_dataset_[f'nbsinister_max_{shifts}'].values, val_dataset_['nbsinister'].values, val_dataset_['departement'].values)
    test_dataset_[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj17.predict(test_dataset_[f'nbsinister_max_{shifts}'].values, test_dataset_['nbsinister'].values, test_dataset_['departement'].values)

    res[obj17.name] = obj17

    new_cols.append(f'nbsinister-max-{shifts}-kmeans-5-Class-Dept')"""

    ###############################################################################

    if graph.sequences_month is None:
        graph.compute_sequence_month(pd.concat([train_dataset, test_dataset]), graph.dataset_name)

    conv_types = ['laplace', 'mean', 'laplace+mean', 'median', 'laplace+median', 'max', 'sum']

    kernels = ['Specialized', 1, 3, 5]
    """
    n_clusters = 3

    for conv_type in conv_types:
        for kernel in kernels:
            logger.info(f"Testing with convolution type: {conv_type}")

            # Sélection du préprocesseur
            preprocessor = PreprocessorConv(graph=graph, conv_type=conv_type, kernel=kernel, id_col=['month_non_encoder', 'graph_id'])

            # Définition de l'objet ScalerClassRisk
            class_risk = KMeansRisk(n_clusters=n_clusters) if conv_type in ['laplace', 'mean', 'laplace+mean', 'laplace+median'] else KMeansRiskZerosHandle(n_clusters)
            obj = ScalerClassRisk(
                col_id='departement',
                dir_output=dir_post_process,
                target='nbsinister',
                scaler=None,
                class_risk=class_risk,
                preprocessor=preprocessor
            )

            # Application du fit et prédictions
            obj.fit(
                train_dataset_['nbsinister'].values,
                train_dataset_['nbsinister'].values,
                train_dataset_['departement'].values,
                train_dataset_[['month_non_encoder', 'graph_id']].values
            )

            train_col = f"nbsinister-kmeans-{n_clusters}-Class-Dept-{conv_type}-{kernel}"
            val_col = f"nbsinister-kmeans-{n_clusters}-Class-Dept-{conv_type}-{kernel}"
            test_col = f"nbsinister-kmeans-{n_clusters}-Class-Dept-{conv_type}-{kernel}"

            train_dataset_[train_col] = obj.predict(
                train_dataset_['nbsinister'].values,
                train_dataset_['nbsinister'].values,  # Ajout de dataset['nbsinister'] comme 2ème argument
                train_dataset_['departement'].values,
                train_dataset_[['month_non_encoder', 'graph_id']].values
            )

            val_dataset_[val_col] = obj.predict(
                val_dataset_['nbsinister'].values,
                val_dataset_['nbsinister'].values,  # Ajout de dataset['nbsinister'] comme 2ème argument
                val_dataset_['departement'].values,
                val_dataset_[['month_non_encoder', 'graph_id']].values
            )
            test_dataset_[test_col] = obj.predict(
                test_dataset_['nbsinister'].values,
                test_dataset_['nbsinister'].values,  # Ajout de dataset['nbsinister'] comme 2ème argument
                test_dataset_['departement'].values,
                test_dataset_[['month_non_encoder', 'graph_id']].values
            )

            # Stockage des résultats
            res[obj.name] = deepcopy(obj)
            new_cols.append(train_col)


        logger.info(f"Completed processing for convolution type: {conv_type} with kernel {kernel}")"""

    ###############################################################################

    n_clusters = 5

    for conv_type in conv_types:
        for kernel in kernels:
            logger.info(f"Testing with convolution type: {conv_type}")

            # Sélection du préprocesseur
            preprocessor = PreprocessorConv(graph=graph, conv_type=conv_type, kernel=kernel, id_col=['month_non_encoder', 'graph_id'])

            # Définition de l'objet ScalerClassRisk
            class_risk = KMeansRisk(n_clusters=n_clusters) if conv_type in ['laplace', 'mean', 'laplace+mean', 'laplace+median'] else KMeansRiskZerosHandle(n_clusters)
            obj = ScalerClassRisk(
                col_id='departement',
                dir_output=dir_post_process,
                target='nbsinister',
                scaler=None,
                class_risk=class_risk,
                preprocessor=preprocessor
            )

            # Application du fit et prédictions
            obj.fit(
                train_dataset_['nbsinister'].values,
                train_dataset_['nbsinister'].values,
                train_dataset_['departement'].values,
                train_dataset_[['month_non_encoder', 'graph_id']].values
            )

            train_col = f"nbsinister-kmeans-{n_clusters}-Class-Dept-{conv_type}-{kernel}"
            val_col = f"nbsinister-kmeans-{n_clusters}-Class-Dept-{conv_type}-{kernel}"
            test_col = f"nbsinister-kmeans-{n_clusters}-Class-Dept-{conv_type}-{kernel}"

            train_dataset_[train_col] = obj.predict(
                train_dataset_['nbsinister'].values,
                train_dataset_['nbsinister'].values,  # Ajout de dataset['nbsinister'] comme 2ème argument
                train_dataset_['departement'].values,
                train_dataset_[['month_non_encoder', 'graph_id']].values
            )

            val_dataset_[val_col] = obj.predict(
                val_dataset_['nbsinister'].values,
                val_dataset_['nbsinister'].values,  # Ajout de dataset['nbsinister'] comme 2ème argument
                val_dataset_['departement'].values,
                val_dataset_[['month_non_encoder', 'graph_id']].values
            )
            test_dataset_[test_col] = obj.predict(
                test_dataset_['nbsinister'].values,
                test_dataset_['nbsinister'].values,  # Ajout de dataset['nbsinister'] comme 2ème argument
                test_dataset_['departement'].values,
                test_dataset_[['month_non_encoder', 'graph_id']].values
            )

            # Stockage des résultats
            res[obj.name] = deepcopy(obj)
            new_cols.append(train_col)

            train_col = f"nbsinister-kmeans-{n_clusters}-Class-Dept-{conv_type}-{kernel}-Past"
    
            # Sélection du préprocesseur
            preprocessor = PreprocessorConv(graph=graph, conv_type=conv_type, kernel=kernel, id_col=['month_non_encoder', 'graph_id'], persistence=True)

            # Définition de l'objet ScalerClassRisk
            class_risk = KMeansRisk(n_clusters=n_clusters) if conv_type in ['laplace', 'mean', 'laplace+mean', 'laplace+median'] else KMeansRiskZerosHandle(n_clusters)
            obj = ScalerClassRisk(
                col_id='departement',
                dir_output=dir_post_process,
                target='nbsinister',
                scaler=None,
                class_risk=class_risk,
                preprocessor=preprocessor
            )

            # Application du fit et prédictions
            obj.fit(
                train_dataset_['nbsinister'].values,
                train_dataset_['nbsinister'].values,
                train_dataset_['departement'].values,
                train_dataset_[['month_non_encoder', 'graph_id']].values
            )

            train_dataset_[train_col] = obj.predict(
                train_dataset_['nbsinister'].values,
                train_dataset_['nbsinister'].values,
                train_dataset_['departement'].values,
                train_dataset_[['month_non_encoder', 'graph_id']].values
            )

            val_dataset_[train_col] = obj.predict(
                val_dataset_['nbsinister'].values,
                val_dataset_['nbsinister'].values,
                val_dataset_['departement'].values,
                val_dataset_[['month_non_encoder', 'graph_id']].values
            )
            test_dataset_[train_col] = obj.predict(
                test_dataset_['nbsinister'].values,
                test_dataset_['nbsinister'].values,
                test_dataset_['departement'].values,
                test_dataset_[['month_non_encoder', 'graph_id']].values
            )

            res[obj.name] = deepcopy(obj)
            new_cols.append(train_col)

    logger.info(f"Completed processing for convolution type: {conv_type} with kernel {kernel}")

    """for conv_type in conv_types:
        for kernel in kernels:
            logger.info(f"Testing with convolution type: {conv_type}")

            # Sélection du préprocesseur
            preprocessor = PreprocessorConv(graph=graph, conv_type=conv_type, kernel=kernel, id_col=['month_non_encoder', 'graph_id'])

            # Définition de l'objet ScalerClassRisk
            class_risk = ThresholdRisk(thresholds=[0.05, 0.15, 0.30, 0.60, 1.0])
            obj = ScalerClassRisk(
                col_id='departement',
                dir_output=dir_post_process,
                target='nbsinister',
                scaler=MinMaxScaler(),
                class_risk=class_risk,
                preprocessor=preprocessor
            )

            # Application du fit et prédictions
            obj.fit(
                train_dataset_['nbsinister'].values,
                train_dataset_['nbsinister'].values,
                train_dataset_['departement'].values,
                train_dataset_[['month_non_encoder', 'graph_id']].values
            )

            train_col = f"nbsinister-MinMax-{n_clusters}-Class-Dept-{conv_type}-{kernel}"
            val_col = f"nbsinister-MinMax-{n_clusters}-Class-Dept-{conv_type}-{kernel}"
            test_col = f"nbsinister-MinMax-{n_clusters}-Class-Dept-{conv_type}-{kernel}"

            train_dataset_[train_col] = obj.predict(
                train_dataset_['nbsinister'].values,
                train_dataset_['nbsinister'].values,  # Ajout de dataset['nbsinister'] comme 2ème argument
                train_dataset_['departement'].values,
                train_dataset_[['month_non_encoder', 'graph_id']].values
            )

            val_dataset_[val_col] = obj.predict(
                val_dataset_['nbsinister'].values,
                val_dataset_['nbsinister'].values,  # Ajout de dataset['nbsinister'] comme 2ème argument
                val_dataset_['departement'].values,
                val_dataset_[['month_non_encoder', 'graph_id']].values
            )
            test_dataset_[test_col] = obj.predict(
                test_dataset_['nbsinister'].values,
                test_dataset_['nbsinister'].values,  # Ajout de dataset['nbsinister'] comme 2ème argument
                test_dataset_['departement'].values,
                test_dataset_[['month_non_encoder', 'graph_id']].values
            )

            # Stockage des résultats
            res[obj.name] = deepcopy(obj)
            new_cols.append(train_col)"""
    
    ##################### Union des risk ###########################
    col_raw = 'nbsinister-kmeans-5-Class-Dept'
    col_derived = 'nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized'

    train_dataset_['union'] = np.maximum(train_dataset_[col_derived].values, train_dataset_[col_raw].values)
    train_dataset_.loc[train_dataset_[(train_dataset_[col_derived] > 0) & (train_dataset_[col_raw] == 0)].index, 'union'] = 0
    train_dataset_['potential_risk'] = np.maximum(train_dataset_[col_derived].values, train_dataset_[col_raw].values)

    test_dataset_['union'] = np.maximum(test_dataset_[col_derived].values, test_dataset_[col_raw].values)
    test_dataset_.loc[test_dataset_[(test_dataset_[col_derived] > 0) & (test_dataset_[col_raw] == 0)].index, 'union'] = 0
    test_dataset_['potential_risk'] = np.maximum(test_dataset_[col_derived].values, test_dataset_[col_raw].values)

    val_dataset_['union'] = np.maximum(val_dataset_[col_derived].values, val_dataset_[col_raw].values)
    val_dataset_.loc[val_dataset_[(val_dataset_[col_derived] > 0) & (val_dataset_[col_raw] == 0)].index, 'union'] = 0
    val_dataset_['potential_risk'] = np.maximum(val_dataset_[col_derived].values, val_dataset_[col_raw].values)

    new_cols.append('union')
    new_cols.append('potential_risk')

    ################################################

    logger.info(f'Post process Model -> {res}')

    if graph_method == 'node':
        train_dataset = train_dataset_
        val_dataset = val_dataset_
        test_dataset = test_dataset_
    else:
        def join_on_index_with_new_cols(original_dataset, updated_dataset, new_cols):
            """
            Effectue un join sur les index (graph_id, date) pour ajouter de nouvelles colonnes.
            :param original_dataset: DataFrame original
            :param updated_dataset: DataFrame avec les index et colonnes à joindre
            :param new_cols: Liste des colonnes à ajouter
            :return: DataFrame mis à jour avec les nouvelles colonnes
            """
            # Joindre les deux DataFrames sur leurs index
            original_dataset.reset_index(drop=True, inplace=True)
            updated_dataset.reset_index(drop=True, inplace=True)

            joined_dataset = original_dataset.set_index(['graph_id', 'date']).join(
                updated_dataset.set_index(['graph_id', 'date'])[new_cols],
                on=['graph_id', 'date'],
                how='left'
            ).reset_index()
            return joined_dataset

        # Mise à jour des datasets
        train_dataset = join_on_index_with_new_cols(train_dataset, train_dataset_, new_cols)
        val_dataset = join_on_index_with_new_cols(val_dataset, val_dataset_, new_cols)
        test_dataset = join_on_index_with_new_cols(test_dataset, test_dataset_, new_cols)

    return res, train_dataset, val_dataset, test_dataset, new_cols

class KSGraphDiscretizer(BaseEstimator):
    def __init__(self, thresholds_ks, score_col='ks_stat', thresh_col='optimal_score', dir_output=Path('./')):
        self.thresholds_ks = thresholds_ks
        self.score_col = score_col
        self.thresh_col = thresh_col
        self.group_results = None
        self.dir_output = dir_output

    def fit(self, test_window, y_pred, col):
        self.group_results = calculate_ks_per_season_and_graph_id(test_window, 'prediction', col, self.thresholds_ks, self.dir_output)
        return self

    def predict(self, test_window, y_pred):
        assert self.group_results is not None
        new_pred = np.full(y_pred.shape[0], fill_value=0.0)
        for (saison, graph_id), df_ks in self.group_results.items():
            subset_indices = (test_window['saison'] == saison) & (test_window['graph_id'] == graph_id)
            y_pred_subset = y_pred[subset_indices]
            new_pred_subset = np.zeros_like(y_pred_subset)
            for threshold in self.thresholds_ks:
                if threshold in df_ks['threshold'].values:
                    optimal_score = df_ks[df_ks['threshold'] == threshold][self.thresh_col].values[0]
                    new_pred_subset[y_pred_subset >= optimal_score] = threshold
            new_pred[subset_indices] = new_pred_subset
        return new_pred
    
class APRDiscretizer(BaseEstimator):
    def __init__(self, thresholds, score_col='f1_score', thresh_col='optimal_score', dir_output=Path('./')):
        self.thresholds = thresholds
        self.score_col = score_col
        self.thresh_col = thresh_col
        self.apr_results = None
        self.dir_output = dir_output

    def fit(self, test_window, y_pred, col):
        """
        Calcule l'APR (Area Precision-Recall) et optimise les seuils sur le F1-Score.
        """
        results = []
        for threshold in self.thresholds:
            # Binarisation des prédictions selon le seuil
            y_true = test_window[col]
            y_pred_binary = (y_pred >= threshold).astype(int)
            
            # Calcul des métriques Precision-Recall et F1-Score
            precision, recall, _ = precision_recall_curve(y_true, y_pred_binary)
            f1_scores = 2 * (precision * recall) / (precision + recall)
            f1_scores = np.nan_to_num(f1_scores)  # Remplacer NaN par 0

            # Trouver le meilleur F1-Score pour ce seuil
            best_f1_index = np.argmax(f1_scores)
            best_f1 = f1_scores[best_f1_index]
            best_threshold = threshold

            results.append({
                'threshold': best_threshold,
                self.score_col: best_f1,
                self.thresh_col: best_threshold
            })

        # Sauvegarder les résultats comme DataFrame
        self.apr_results = pd.DataFrame(results)
        # Optionnel : sauvegarder les résultats sur disque
        output_file = self.dir_output / 'apr_results.csv'
        self.apr_results.to_csv(output_file, index=False)
        return self

    def predict(self, y_pred):
        """
        Applique les seuils optimisés pour les prédictions.
        """
        assert self.apr_results is not None, "La méthode fit doit être appelée avant predict."
        new_pred = np.full(y_pred.shape[0], fill_value=0.0)

        # Application des seuils optimaux
        for threshold in self.thresholds:
            optimal_score = self.apr_results[self.apr_results['threshold'] == threshold][self.thresh_col].values[0]
            new_pred[y_pred >= optimal_score] = threshold

        return new_pred

class ClassifierDiscretizer(BaseEstimator):
    def __init__(self, model_params, dir_output=Path('./')):
        self.model_params = model_params
        self.model = None
        self.dir_output = dir_output

    def fit(self, test_window, y_pred, col):
        df = test_window.copy(deep=True)
        df['risk'] = y_pred
        train_df, _ = train_test_split(df, test_size=0.2, random_state=42)
        self.model = get_model(
            model_type='dt',
            name='discretization_model_dt',
            device='cpu',
            task_type='classification',
            params=self.model_params,
            loss='logloss'
        )
        self.model.fit(train_df[['risk']], train_df[col])
        return self

    def predict(self, y_pred):
        assert self.model is not None
        df = pd.DataFrame({'risk': y_pred})
        return self.model.predict(df[['risk']])
    
class MDLPDiscretizer(BaseEstimator):
    def __init__(self, dir_output=Path('./')):
        self.dir_output = dir_output
        self.discretizer = MDLP()  # Initialiser l'algorithme MDLP

    def fit(self, test_window, y_pred, col):
        df = test_window.copy(deep=True)
        df['risk'] = y_pred
        train_df, _ = train_test_split(df, test_size=0.2, random_state=42)
        
        # Utiliser MDLP pour la discretisation
        discretized_column = self.discretizer.fit_transform(train_df[['risk']].values, train_df[col].values)
        train_df[col] = discretized_column
        
        self.discretized_data_ = train_df  # Stocker les données discrétisées pour future utilisation
        return self

    def transform(self, X):
        """
        Transformer les données avec les intervalles de discretisation obtenus par MDLP.
        """
        discretized_column = self.discretizer.transform(X)
        return discretized_column

    def fit_transform(self, test_window, y_pred, col):
        """
        Utilise MDLP pour discrétiser les données puis retourne les données discrétisées.
        """
        self.fit(test_window, y_pred, col)
        return self.transform(test_window[['risk']].values)

class RoundDiscretizer(BaseEstimator):
    def __init__(self, dir_output=Path('./')):
        """
        Initializer for the RoundDiscretizer class.
        
        :param dir_output: The directory for storing output files.
        """
        self.dir_output = dir_output

    def fit(self, test_window, y_pred, col):
        """
        Fit method for the rounder model (no fitting necessary).

        :param test_window: The input data to fit.
        :param y_pred: Predicted risk values.
        :param col: The target column name in the dataframe.
        :return: self
        """
        return self

    def predict(self, y_pred):
        """
        Rounds the predicted risk values to the nearest integer.
        
        :param y_pred: Array-like, predicted risk values.
        :return: Array-like, rounded risk values.
        """
        return self.roundDiscretize(y_pred)

    def roundDiscretize(self, y_pred):
        """
        Rounds the predicted risk values to the nearest integer.
        
        :param y_pred: Array-like, predicted risk values.
        :return: Array-like, rounded risk values.
        """
        rounded_predictions = y_pred.round()
        return rounded_predictions

def discretization(method, test_window, y_pred, pred_max, pred_min, col, target_name, dir_output):
    thresholds_ks = np.sort(np.unique(test_window[col]))
    thresholds_ks = thresholds_ks[(~np.isnan(thresholds_ks)) & (thresholds_ks > 0)]

    if method == 'apr':
        apr_discretizer = APRDiscretizer(thresholds=thresholds_ks, dir_output=dir_output)
        apr_discretizer.fit(test_window, y_pred, col)
        if pred_max is not None:
            return apr_discretizer.predict(y_pred), apr_discretizer.predict(pred_max), apr_discretizer.predict(pred_min)
        return apr_discretizer.predict(y_pred), None, None
    
    elif method == 'round':
        round_discretizer = RoundDiscretizer(thresholds=thresholds_ks, dir_output=dir_output)
        round_discretizer.fit(test_window, y_pred, col)
        if pred_max is not None:
            return round_discretizer.predict(y_pred), round_discretizer.predict(pred_max), round_discretizer.predict(pred_min)
        return round_discretizer.predict(y_pred), None, None
    
    elif method == 'ks':
        ks_discretizer = KSDiscretizer(thresholds_ks=thresholds_ks, dir_output=dir_output)
        ks_discretizer.fit(test_window, y_pred, col)
        if pred_max is not None:
            return ks_discretizer.predict(y_pred), ks_discretizer.predict(pred_max), ks_discretizer.predict(pred_min)
        return ks_discretizer.predict(y_pred), None, None
    
    elif method == 'graph_ks':
        ks_graph_discretizer = KSGraphDiscretizer(thresholds_ks=thresholds_ks, dir_output=dir_output)
        ks_graph_discretizer.fit(test_window, y_pred, col)
        
        if pred_max is not None:
            return ks_graph_discretizer.predict(test_window, y_pred), ks_graph_discretizer.predict(test_window, pred_max), ks_graph_discretizer.predict(test_window, pred_min)

        return ks_graph_discretizer.predict(test_window, y_pred), None, None
    
    elif method == 'dt':
        model_params = {
            'max_depth': 6,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
        }

        classifier_discretizer = ClassifierDiscretizer(model_params=model_params, dir_output=dir_output)
        if target_name == 'risk':
            classifier_discretizer.fit(test_window, test_window['risk'].values, col)
        else:
            classifier_discretizer.fit(test_window, y_pred, col)

        if pred_max is not None:
            return classifier_discretizer.predict(y_pred), classifier_discretizer.predict(pred_max), classifier_discretizer.predict(pred_min)
        return classifier_discretizer.predict(y_pred), None, None
    
    elif method == 'classification':
        return y_pred, pred_max, pred_min

    else:
        raise ValueError(f'{method} unknow')