from cProfile import label
from curses import raw, window
from matplotlib.pyplot import sca
from sqlalchemy import column
from GNN.train import *

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
        self.model = KMeans(n_clusters=min(self.n_clusters, np.unique(X).shape[0]), random_state=42, n_init=10)
        self.model.fit(X)
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
        self.thresholds = np.quantile(X[X > self.min_value].flatten(), [0.25, 0.5, 0.75])
        self.n_clusters = len(self.thresholds) + 2  # 5 classes : 4 quartiles + classe pour la valeur minimale
        self.name = f'QuantileRisk_{self.thresholds}_MinValue'

    def predict(self, X):
        """
        Prédit les classes en fonction des quartiles calculés et une classe spécifique pour la valeur minimale.

        :param X: Données à classer.
        :return: Classes prédites.
        """
        if self.thresholds is None or self.min_value is None:
            raise ValueError("Les seuils (quartiles) et la valeur minimale n'ont pas été calculés. Appelez `fit` d'abord.")
        
        # Initialiser les classes avec les indices basés sur les quartiles
        result = np.digitize(X.flatten(), self.thresholds, right=True)
        
        # Assigner une classe distincte pour la valeur minimale
        result[X.flatten() == self.min_value] = self.n_clusters - 1  # Classe pour la valeur minimale
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
        print(self.ks_results)
        return self

    def predict(self, y_pred):
        assert self.ks_results is not None
        new_pred = np.full(y_pred.shape[0], fill_value=0.0)
        max_value = self.ks_results[self.ks_results['threshold'] == self.thresholds_ks[0]][self.score_col].values
        for threshold in self.thresholds_ks:
            if self.ks_results[self.ks_results['threshold'] == threshold][self.score_col].values[0] >= max_value:
                new_pred[y_pred >= self.ks_results[self.ks_results['threshold'] == threshold][self.thresh_col].values[0]] = threshold
        return new_pred

class ScalerClassRisk:
    def __init__(self, col_id, dir_output, target, scaler=None, class_risk=None):
        """
        Initialize the ScalerClassRisk.

        :param n_clusters: Number of clusters for classification (if using KMeans).
        :param col_id: Column ID used to group data.
        :param dir_output: Directory to save output files (e.g., histograms).
        :param target: Target name used for labeling outputs.
        :param scaler: A scaler object (e.g., StandardScaler) for normalization. If None, no scaling is applied.
        :param class_risk: An object with `fit` and `predict` methods for risk classification.
        """
        assert class_risk is not None
        self.n_clusters = class_risk.n_clusters
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
        self.name = f'ScalerClassRisk_{scaler_name}_{class_risk_name}_{target}'
        self.dir_output = dir_output / self.name
        self.target = target
        self.scaler = scaler
        self.class_risk = class_risk
        check_and_create_path(self.dir_output)
        self.models_by_id = {}

    def fit(self, X, sinisters, ids):
        """
        Fit models for each unique ID and calculate statistics.

        :param X: Array of values to scale and classify.
        :param ids: Array of IDs corresponding to each value in X.
        :param sinisters: Array of sinisters values corresponding to each value in X.
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(sinisters.shape) == 1:
            sinisters = sinisters.reshape(-1, 1)

        self.models_by_id = {}

        for unique_id in np.unique(ids):
            mask = ids == unique_id
            X_id = X[mask]
            sinisters_id = sinisters[mask]

            # Scale data if scaler is provided
            if self.scaler is not None:
                scaler = deepcopy(self.scaler)
                scaler.fit(X_id)
                X_scaled = scaler.transform(X_id)
            else:
                scaler = None
                X_scaled = X_id

            # Fit the class_risk model
            class_risk = deepcopy(self.class_risk)
            class_risk.fit(X_scaled, sinisters[mask])

            # Predict classes for current ID
            classes = class_risk.predict(X_scaled)

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

            self.models_by_id[unique_id] = {
                'scaler': scaler,
                'class_risk': class_risk,
                'sinistres_mean': sinisters_mean,
                'sinistres_min': sinisters_min,
                'sinistres_max': sinisters_max
            }

        pred = self.predict(X, ids)
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
        logger.info(f'########################################## {self.name} ##########################################')
        for cl in np.unique(pred):
            logger.info(f'{cl} -> {pred[pred == cl].shape[0]}')

    def predict(self, X, ids):
        """
        Predict class labels using the appropriate model for each ID.

        :param X: Array of values to predict.
        :param ids: Array of IDs corresponding to each value in X.
        :return: Array of predicted class labels.
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        predictions = np.zeros(X.shape[0], dtype=int)

        for unique_id, model in self.models_by_id.items():
            if unique_id not in np.unique(ids):
                continue

            mask = ids == unique_id
            scaler = model['scaler']
            class_risk = model['class_risk']

            # Scale data if scaler exists
            if scaler is not None:
                X_scaled = scaler.transform(X[mask])
            else:
                X_scaled = X[mask]

            predictions[mask] = class_risk.predict(X_scaled.astype(float))

        return predictions
    
    def fit_predict(self, X, ids, sinisters):
        self.fit(X, ids, sinisters)
        return self.predict(X, ids)

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
    
    def predict_risk(self, X, ids):
        return self.predict(X, ids)

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

def post_process_model(train_dataset, val_dataset, test_dataset, dir_post_process):

    res = {}

    ####################################################################################

    obj1 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target='nbsinister', scaler=MinMaxScaler(), class_risk=ThresholdRisk([0.05, 0.15, 0.30, 0.60, 1.0]))

    obj1.fit(train_dataset['nbsinister'].values, train_dataset['nbsinister'].values, train_dataset['departement'].values)

    train_dataset['nbsinister-MinMax-thresholds-5-Class-Dept'] = obj1.predict(train_dataset['nbsinister'].values, train_dataset['departement'].values)
    val_dataset['nbsinister-MinMax-thresholds-5-Class-Dept'] = obj1.predict(val_dataset['nbsinister'].values, val_dataset['departement'].values)
    test_dataset['nbsinister-MinMax-thresholds-5-Class-Dept'] = obj1.predict(test_dataset['nbsinister'].values, test_dataset['departement'].values)
    
    res[obj1.name] = obj1

    ####################################################################################

    obj11 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target='nbsinister', scaler=RobustScaler(), class_risk=QuantileRisk())

    obj11.fit(train_dataset['nbsinister'].values, train_dataset['nbsinister'].values, train_dataset['departement'].values)

    train_dataset['nbsinister-Robust-Quantile-5-Class-Dept'] = obj1.predict(train_dataset['nbsinister'].values, train_dataset['departement'].values)
    val_dataset['nbsinister-Robust-Quantile-5-Class-Dept'] = obj1.predict(val_dataset['nbsinister'].values, val_dataset['departement'].values)
    test_dataset['nbsinister-Robust-Quantile-5-Class-Dept'] = obj1.predict(test_dataset['nbsinister'].values, test_dataset['departement'].values)

    """fig, ax = plt.subplots(1, figsize=(15,5))
    test_dataset[test_dataset['graph_id'] == 0]['nbsinister-Robust-Quantile-5-Class-Dept'].plot(ax=ax)
    test_dataset[test_dataset['graph_id'] == 0]['nbsinister'].plot(ax=ax)
    plt.show()"""
    
    res[obj11.name] = obj11

    ####################################################################################
    
    obj2 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target='nbsinister', scaler=None, class_risk=KMeansRiskZerosHandle(5))

    obj2.fit(train_dataset['nbsinister'].values, train_dataset['nbsinister'].values, train_dataset['departement'].values)

    train_dataset['nbsinister-kmeans-5-Class-Dept'] = obj2.predict(train_dataset['nbsinister'].values, train_dataset['departement'].values)
    val_dataset['nbsinister-kmeans-5-Class-Dept'] = obj2.predict(val_dataset['nbsinister'].values, val_dataset['departement'].values)
    test_dataset['nbsinister-kmeans-5-Class-Dept'] = obj2.predict(test_dataset['nbsinister'].values, test_dataset['departement'].values)
    
    res[obj2.name] = obj2

    ###################################################################################
    
    obj4 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target='risk', scaler=None, class_risk=KMeansRisk(5))

    obj4.fit(train_dataset['risk'].values, train_dataset['nbsinister'].values, train_dataset['departement'].values)

    train_dataset['risk-kmeans-5-Class-Dept'] = obj4.predict(train_dataset['risk'].values, train_dataset['departement'].values)
    val_dataset['risk-kmeans-5-Class-Dept'] = obj4.predict(val_dataset['risk'].values, val_dataset['departement'].values)
    test_dataset['risk-kmeans-5-Class-Dept'] = obj4.predict(test_dataset['risk'].values, test_dataset['departement'].values)
    
    res[obj4.name] = obj4

    ##################################################################################

    obj3 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target='risk-nbsinister', scaler=None, class_risk=KMeansRisk(5))

    obj3.fit(train_dataset[['risk', 'nbsinister']].values, train_dataset['nbsinister'].values, train_dataset['departement'].values)

    train_dataset['risk-nbsinister-kmeans-5-Class-Dept'] = obj3.predict(train_dataset[['risk', 'nbsinister']].values, train_dataset['departement'].values)
    val_dataset['risk-nbsinister-kmeans-5-Class-Dept'] = obj3.predict(val_dataset[['risk', 'nbsinister']].values, val_dataset['departement'].values)
    test_dataset['risk-nbsinister-kmeans-5-Class-Dept'] = obj3.predict(test_dataset[['risk', 'nbsinister']].values, test_dataset['departement'].values)

    res[obj3.name] = obj3

    #######################################################################################
    shifts = 7
    
    obj6 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target=f'nbsinister-sum-{shifts}+{shifts}', scaler=None, class_risk=KMeansRiskZerosHandle(5))

    train_dataset.sort_values(by=['graph_id', 'date'], inplace=True)
    val_dataset.sort_values(by=['graph_id', 'date'], inplace=True)
    test_dataset.sort_values(by=['graph_id', 'date'], inplace=True)

    # Application sur les jeux de données
    train_dataset = class_window_sum(
        dataset=train_dataset,
        column='nbsinister',
        shifts=shifts,
        group_col='graph_id',
    )

    val_dataset = class_window_sum(
            dataset=val_dataset,
            column='nbsinister',
            shifts=shifts,
            group_col='graph_id',
        )
    
    test_dataset = class_window_sum(
                dataset=test_dataset,
                column='nbsinister',
                shifts=shifts,
                group_col='graph_id',
            )

    obj6.fit(train_dataset[f'nbsinister_sum_{shifts}'].values, train_dataset['nbsinister'].values, train_dataset['departement'].values)

    train_dataset[f'nbsinister-sum-{shifts}-kmeans-5-Class-Dept'] = obj6.predict(train_dataset[f'nbsinister_sum_{shifts}'].values, train_dataset['departement'].values)
    val_dataset[f'nbsinister-sum-{shifts}-kmeans-5-Class-Dept'] = obj6.predict(val_dataset[f'nbsinister_sum_{shifts}'].values, val_dataset['departement'].values)
    test_dataset[f'nbsinister-sum-{shifts}-kmeans-5-Class-Dept'] = obj6.predict(test_dataset[f'nbsinister_sum_{shifts}'].values, test_dataset['departement'].values)

    res[obj6.name] = obj6

    #######################################################################################
    shifts = 5
    
    obj7 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target=f'nbsinister-sum-{shifts}+{shifts}', scaler=None, class_risk=KMeansRiskZerosHandle(5))

    train_dataset.sort_values(by=['graph_id', 'date'], inplace=True)
    val_dataset.sort_values(by=['graph_id', 'date'], inplace=True)
    test_dataset.sort_values(by=['graph_id', 'date'], inplace=True)

    # Application sur les jeux de données
    train_dataset = class_window_sum(
        dataset=train_dataset,
        column='nbsinister',
        shifts=shifts,
        group_col='graph_id',
    )

    val_dataset = class_window_sum(
            dataset=val_dataset,
            column='nbsinister',
            shifts=shifts,
            group_col='graph_id',
        )
    
    test_dataset = class_window_sum(
                dataset=test_dataset,
                column='nbsinister',
                shifts=shifts,
                group_col='graph_id',
            )

    obj7.fit(train_dataset[f'nbsinister_sum_{shifts}'].values, train_dataset['nbsinister'].values, train_dataset['departement'].values)

    train_dataset[f'nbsinister-sum-{shifts}-kmeans-5-Class-Dept'] = obj7.predict(train_dataset[f'nbsinister_sum_{shifts}'].values, train_dataset['departement'].values)
    val_dataset[f'nbsinister-sum-{shifts}-kmeans-5-Class-Dept'] = obj7.predict(val_dataset[f'nbsinister_sum_{shifts}'].values, val_dataset['departement'].values)
    test_dataset[f'nbsinister-sum-{shifts}-kmeans-5-Class-Dept'] = obj7.predict(test_dataset[f'nbsinister_sum_{shifts}'].values, test_dataset['departement'].values)

    res[obj7.name] = obj7

    #######################################################################################
    shifts = 3
    
    obj8 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target=f'nbsinister-sum-{shifts}+{shifts}', scaler=None, class_risk=KMeansRiskZerosHandle(5))

    train_dataset.sort_values(by=['graph_id', 'date'], inplace=True)
    val_dataset.sort_values(by=['graph_id', 'date'], inplace=True)
    test_dataset.sort_values(by=['graph_id', 'date'], inplace=True)

    # Application sur les jeux de données
    train_dataset = class_window_sum(
        dataset=train_dataset,
        column='nbsinister',
        shifts=shifts,
        group_col='graph_id',
    )

    val_dataset = class_window_sum(
            dataset=val_dataset,
            column='nbsinister',
            shifts=shifts,
            group_col='graph_id',
        )
    
    test_dataset = class_window_sum(
                dataset=test_dataset,
                column='nbsinister',
                shifts=shifts,
                group_col='graph_id',
            )

    obj8.fit(train_dataset[f'nbsinister_sum_{shifts}'].values, train_dataset['nbsinister'].values, train_dataset['departement'].values)

    train_dataset[f'nbsinister-sum-{shifts}-kmeans-5-Class-Dept'] = obj8.predict(train_dataset[f'nbsinister_sum_{shifts}'].values, train_dataset['departement'].values)
    val_dataset[f'nbsinister-sum-{shifts}-kmeans-5-Class-Dept'] = obj8.predict(val_dataset[f'nbsinister_sum_{shifts}'].values, val_dataset['departement'].values)
    test_dataset[f'nbsinister-sum-{shifts}-kmeans-5-Class-Dept'] = obj8.predict(test_dataset[f'nbsinister_sum_{shifts}'].values, test_dataset['departement'].values)

    res[obj8.name] = obj8

    #######################################################################################
    shifts = 1
    
    obj9 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target=f'nbsinister-sum-{shifts}+{shifts}', scaler=None, class_risk=KMeansRiskZerosHandle(5))

    train_dataset.sort_values(by=['graph_id', 'date'], inplace=True)
    val_dataset.sort_values(by=['graph_id', 'date'], inplace=True)
    test_dataset.sort_values(by=['graph_id', 'date'], inplace=True)

    # Application sur les jeux de données
    train_dataset = class_window_sum(
        dataset=train_dataset,
        column='nbsinister',
        shifts=shifts,
        group_col='graph_id',
    )

    val_dataset = class_window_sum(
            dataset=val_dataset,
            column='nbsinister',
            shifts=shifts,
            group_col='graph_id',
        )
    
    test_dataset = class_window_sum(
                dataset=test_dataset,
                column='nbsinister',
                shifts=shifts,
                group_col='graph_id',
            )

    obj9.fit(train_dataset[f'nbsinister_sum_{shifts}'].values, train_dataset['nbsinister'].values, train_dataset['departement'].values)

    train_dataset[f'nbsinister-sum-{shifts}-kmeans-5-Class-Dept'] = obj9.predict(train_dataset[f'nbsinister_sum_{shifts}'].values, train_dataset['departement'].values)
    val_dataset[f'nbsinister-sum-{shifts}-kmeans-5-Class-Dept'] = obj9.predict(val_dataset[f'nbsinister_sum_{shifts}'].values, val_dataset['departement'].values)
    test_dataset[f'nbsinister-sum-{shifts}-kmeans-5-Class-Dept'] = obj9.predict(test_dataset[f'nbsinister_sum_{shifts}'].values, test_dataset['departement'].values)

    res[obj9.name] = obj9

    #######################################################################################
    shifts = 0
    
    obj10 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target=f'nbsinister-sum-{shifts}+{shifts}', scaler=None, class_risk=KMeansRiskZerosHandle(5))

    train_dataset.sort_values(by=['graph_id', 'date'], inplace=True)
    val_dataset.sort_values(by=['graph_id', 'date'], inplace=True)
    test_dataset.sort_values(by=['graph_id', 'date'], inplace=True)

    # Application sur les jeux de données
    train_dataset = class_window_sum(
        dataset=train_dataset,
        column='nbsinister',
        shifts=shifts,
        group_col='graph_id',
    )

    val_dataset = class_window_sum(
            dataset=val_dataset,
            column='nbsinister',
            shifts=shifts,
            group_col='graph_id',
        )
    
    test_dataset = class_window_sum(
                dataset=test_dataset,
                column='nbsinister',
                shifts=shifts,
                group_col='graph_id',
            )

    obj10.fit(train_dataset[f'nbsinister_sum_{shifts}'].values, train_dataset['nbsinister'].values, train_dataset['departement'].values)

    train_dataset[f'nbsinister-sum-{shifts}-kmeans-5-Class-Dept'] = obj10.predict(train_dataset[f'nbsinister_sum_{shifts}'].values, train_dataset['departement'].values)
    val_dataset[f'nbsinister-sum-{shifts}-kmeans-5-Class-Dept'] = obj10.predict(val_dataset[f'nbsinister_sum_{shifts}'].values, val_dataset['departement'].values)
    test_dataset[f'nbsinister-sum-{shifts}-kmeans-5-Class-Dept'] = obj10.predict(test_dataset[f'nbsinister_sum_{shifts}'].values, test_dataset['departement'].values)

    res[obj10.name] = obj10

    #######################################################################################
    shifts = 7
    
    obj12 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target=f'nbsinister-max-{shifts}+{shifts}', scaler=None, class_risk=KMeansRiskZerosHandle(5))

    train_dataset.sort_values(by=['graph_id', 'date'], inplace=True)
    val_dataset.sort_values(by=['graph_id', 'date'], inplace=True)
    test_dataset.sort_values(by=['graph_id', 'date'], inplace=True)

    # Application sur les jeux de données
    train_dataset = class_window_max(
        dataset=train_dataset,
        column='nbsinister',
        shifts=shifts,
        group_col='graph_id',
    )

    val_dataset = class_window_max(
            dataset=val_dataset,
            column='nbsinister',
            shifts=shifts,
            group_col='graph_id',
        )
    
    test_dataset = class_window_max(
                dataset=test_dataset,
                column='nbsinister',
                shifts=shifts,
                group_col='graph_id',
            )

    obj12.fit(train_dataset[f'nbsinister_max_{shifts}'].values, train_dataset['nbsinister'].values, train_dataset['departement'].values)

    train_dataset[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj12.predict(train_dataset[f'nbsinister_max_{shifts}'].values, train_dataset['departement'].values)
    val_dataset[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj12.predict(val_dataset[f'nbsinister_max_{shifts}'].values, val_dataset['departement'].values)
    test_dataset[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj12.predict(test_dataset[f'nbsinister_max_{shifts}'].values, test_dataset['departement'].values)

    res[obj12.name] = obj12

    #######################################################################################
    shifts = 5
    
    obj13 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target=f'nbsinister-max-{shifts}+{shifts}', scaler=None, class_risk=KMeansRiskZerosHandle(5))

    train_dataset.sort_values(by=['graph_id', 'date'], inplace=True)
    val_dataset.sort_values(by=['graph_id', 'date'], inplace=True)
    test_dataset.sort_values(by=['graph_id', 'date'], inplace=True)

    # Application sur les jeux de données
    train_dataset = class_window_max(
        dataset=train_dataset,
        column='nbsinister',
        shifts=shifts,
        group_col='graph_id',
    )

    val_dataset = class_window_max(
            dataset=val_dataset,
            column='nbsinister',
            shifts=shifts,
            group_col='graph_id',
        )
    
    test_dataset = class_window_max(
                dataset=test_dataset,
                column='nbsinister',
                shifts=shifts,
                group_col='graph_id',
            )

    obj13.fit(train_dataset[f'nbsinister_max_{shifts}'].values, train_dataset['nbsinister'].values, train_dataset['departement'].values)

    train_dataset[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj13.predict(train_dataset[f'nbsinister_max_{shifts}'].values, train_dataset['departement'].values)
    val_dataset[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj13.predict(val_dataset[f'nbsinister_max_{shifts}'].values, val_dataset['departement'].values)
    test_dataset[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj13.predict(test_dataset[f'nbsinister_max_{shifts}'].values, test_dataset['departement'].values)

    res[obj13.name] = obj13

    #######################################################################################
    shifts = 3
    
    obj14 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target=f'nbsinister-max-{shifts}+{shifts}', scaler=None, class_risk=KMeansRiskZerosHandle(5))

    train_dataset.sort_values(by=['graph_id', 'date'], inplace=True)
    val_dataset.sort_values(by=['graph_id', 'date'], inplace=True)
    test_dataset.sort_values(by=['graph_id', 'date'], inplace=True)

    # Application sur les jeux de données
    train_dataset = class_window_max(
        dataset=train_dataset,
        column='nbsinister',
        shifts=shifts,
        group_col='graph_id',
    )

    val_dataset = class_window_max(
            dataset=val_dataset,
            column='nbsinister',
            shifts=shifts,
            group_col='graph_id',
        )
    
    test_dataset = class_window_max(
                dataset=test_dataset,
                column='nbsinister',
                shifts=shifts,
                group_col='graph_id',
            )

    obj14.fit(train_dataset[f'nbsinister_max_{shifts}'].values, train_dataset['nbsinister'].values, train_dataset['departement'].values)

    train_dataset[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj14.predict(train_dataset[f'nbsinister_max_{shifts}'].values, train_dataset['departement'].values)
    val_dataset[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj14.predict(val_dataset[f'nbsinister_max_{shifts}'].values, val_dataset['departement'].values)
    test_dataset[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj14.predict(test_dataset[f'nbsinister_max_{shifts}'].values, test_dataset['departement'].values)

    res[obj14.name] = obj14

    #######################################################################################
    shifts = 1
    
    obj15 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target=f'nbsinister-max-{shifts}+{shifts}', scaler=None, class_risk=KMeansRiskZerosHandle(5))

    train_dataset.sort_values(by=['graph_id', 'date'], inplace=True)
    val_dataset.sort_values(by=['graph_id', 'date'], inplace=True)
    test_dataset.sort_values(by=['graph_id', 'date'], inplace=True)

    # Application sur les jeux de données
    train_dataset = class_window_max(
        dataset=train_dataset,
        column='nbsinister',
        shifts=shifts,
        group_col='graph_id',
    )

    val_dataset = class_window_max(
            dataset=val_dataset,
            column='nbsinister',
            shifts=shifts,
            group_col='graph_id',
        )
    
    test_dataset = class_window_max(
                dataset=test_dataset,
                column='nbsinister',
                shifts=shifts,
                group_col='graph_id',
            )

    obj15.fit(train_dataset[f'nbsinister_max_{shifts}'].values, train_dataset['nbsinister'].values, train_dataset['departement'].values)

    train_dataset[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj15.predict(train_dataset[f'nbsinister_max_{shifts}'].values, train_dataset['departement'].values)
    val_dataset[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj15.predict(val_dataset[f'nbsinister_max_{shifts}'].values, val_dataset['departement'].values)
    test_dataset[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj15.predict(test_dataset[f'nbsinister_max_{shifts}'].values, test_dataset['departement'].values)

    res[obj15.name] = obj15

    #######################################################################################
    shifts = 0
    
    obj16 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target=f'nbsinister-max-{shifts}+{shifts}', scaler=None, class_risk=KMeansRiskZerosHandle(5))

    train_dataset.sort_values(by=['graph_id', 'date'], inplace=True)
    val_dataset.sort_values(by=['graph_id', 'date'], inplace=True)
    test_dataset.sort_values(by=['graph_id', 'date'], inplace=True)

    # Application sur les jeux de données
    train_dataset = class_window_max(
        dataset=train_dataset,
        column='nbsinister',
        shifts=shifts,
        group_col='graph_id',
    )

    val_dataset = class_window_max(
            dataset=val_dataset,
            column='nbsinister',
            shifts=shifts,
            group_col='graph_id',
        )
    
    test_dataset = class_window_max(
                dataset=test_dataset,
                column='nbsinister',
                shifts=shifts,
                group_col='graph_id',
            )

    obj16.fit(train_dataset[f'nbsinister_max_{shifts}'].values, train_dataset['nbsinister'].values, train_dataset['departement'].values)

    train_dataset[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj16.predict(train_dataset[f'nbsinister_max_{shifts}'].values, train_dataset['departement'].values)
    val_dataset[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj16.predict(val_dataset[f'nbsinister_max_{shifts}'].values, val_dataset['departement'].values)
    test_dataset[f'nbsinister-max-{shifts}-kmeans-5-Class-Dept'] = obj16.predict(test_dataset[f'nbsinister_max_{shifts}'].values, test_dataset['departement'].values)

    res[obj16.name] = obj16

    ###############################################################################

    logger.info(f'Post process Model -> {res}')

    return res

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