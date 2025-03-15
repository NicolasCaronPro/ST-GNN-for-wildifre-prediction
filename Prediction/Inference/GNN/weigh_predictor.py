from sklearn.cluster import KMeans, DBSCAN
import numpy as np

class Predictor():
    def __init__(self, n_clusters, name='', type='kmeans', eps=0.5, binary=False):
        if type == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif type == 'dbscan':
            self.model = DBSCAN(eps=eps)
        elif type == 'fix':
            self.model = FixedThresholdPredictor(n_clusters, name, 1 if binary else None)
        elif type == 'jenks':
            self.model = JenksThresholdPredictor(n_clusters=n_clusters, name=name)

        self.binary = binary
        self.bounds = None
        self.n_clusters = n_clusters
        self.type = type
        self.name = name

    def fit(self, X : np.array):
        self.train_X = X
        if len(X.shape) == 1:
            self.model.fit(X.reshape(-1,1))
            Xpre = self.model.predict(X.reshape(-1,1))
        else:
            self.model.fit(X)
            Xpre = self.model.predict(X)

        if self.type == 'kmeans':
            self.cluster_centers_ = self.model.cluster_centers_
        else:
            self.cluster_centers_= []
            cls = np.unique(Xpre)
            for c in cls:
                mask = np.argwhere(Xpre == c)
                self.cluster_centers_.append(np.mean(X[mask]))
            self.cluster_centers_ = np.asarray(self.cluster_centers_)
                
        self.histogram = np.bincount(Xpre)
        self.highNumberClass = np.argwhere(self.histogram == np.max(self.histogram))

    def predict(self, X : np.array, min_class : int = 0):
        X = X.astype(float)
        if len(X.shape) == 1:
            pred = self.model.predict(X.reshape(-1,1)) + min_class
        else:
            pred = self.model.predict(X) + min_class
        mask = self.train_X > np.nanmax(self.train_X)
        if True in np.unique(mask):
            pred[mask] += 1
        return pred
    
    def weight(self, c : int):
        #valueOfClass = self.histogram[c]
        #valueOfMinClass = self.histogram[self.highNumberClass][:,0][0]
        #return valueOfMinClass / valueOfClass
        return c + 1
    
    def weight_array(self, array : np.array):
        lis = [self.weight(c) for c in array]
        return np.asarray(lis)
    
    def get_centroid(self, c : int):
        if self.binary:
            return self.bounds[c]
        return self.model.cluster_centers_[c]

    def log(self, logger = None):
        if logger is not None:
            logger.info(f'############# Predictor {self.name} ###############')
            logger.info('Histogram')
            logger.info(self.histogram)
            logger.info('Cluster Centers')
            logger.info(self.model.cluster_centers_)
            logger.info(f'####################################')
        else:
            print(f'############# Predictor {self.name} ###############')
            print('Histogram')
            print(self.histogram)
            print('Cluster Centers')
            print(self.model.cluster_centers_)
            print(f'####################################')

class FixedThresholdPredictor:
    def __init__(self, n_clusters=5, name='', max_value=None):
        """
        Initialise un modèle avec 5 seuils fixes pour les clusters.
        
        Args:
            n_clusters (int): Le nombre de clusters, fixé à 5 pour ce modèle.
            name (str): Nom du modèle.
            binary (bool): Toujours True pour ce type de modèle.
        """
        self.bounds = None
        self.n_clusters = n_clusters
        self.name = name
        self.cluster_centers_ = None
        self.histogram = None
        self.highNumberClass = None
        self.max_value = max_value

    def fit(self, X: np.array):
        """
        Calcule les seuils et génère les clusters en fonction des données X.
        
        Args:
            X (np.array): Tableau 1D de données pour l'entraînement.
        """
        # Définir des bornes fixes pour 5 clusters
        if self.max_value is None:
            self.max_value = np.max(X)

        self.bounds = np.array([0.0, 0.1, 0.40, 0.64, 0.95, 1.0])

        # Clusteriser les valeurs selon les bornes
        Xpre = np.zeros(X.shape[0], dtype=int)

        # Appliquer les bornes pour chaque classe
        for c in range(self.n_clusters):
            lower_bound = self.bounds[c]
            upper_bound = self.bounds[c + 1]
            mask = np.argwhere((X >= self.max_value * lower_bound) & (X < self.max_value * upper_bound))[:, 0]
            Xpre[mask] = c

        # Pour les valeurs exactement égales au max, assigner au dernier cluster
        max_mask = np.argwhere(X == self.max_value)[:, 0]
        Xpre[max_mask] = self.n_clusters - 1

        self.cluster_centers_ = np.zeros(self.n_clusters)
        for c in range(self.n_clusters):
            self.cluster_centers_[c] = (self.bounds[c] * self.max_value + self.bounds[c + 1] * self.max_value) / 2

        # Histogramme des points dans chaque cluster
        self.histogram = np.bincount(Xpre, minlength=self.n_clusters)
        self.highNumberClass = np.argwhere(self.histogram == np.max(self.histogram))

    def predict(self, X: np.array):
        """
        Prédire les classes pour les nouvelles données X.
        
        Args:
            X (np.array): Données à classer.
        
        Returns:
            np.array: Tableau des classes prédites.
        """
        pred = np.zeros(X.shape[0], dtype=int)

        # Appliquer les bornes pour chaque classe
        for c in range(self.n_clusters):
            lower_bound = self.bounds[c]
            upper_bound = self.bounds[c + 1]
            mask = np.argwhere((X >= self.max_value * lower_bound) & (X < self.max_value * upper_bound))[:, 0]
            pred[mask] = c

        # Pour les valeurs exactement égales au max, assigner au dernier cluster
        max_mask = np.argwhere(X == self.max_value)[:, 0]
        pred[max_mask] = self.n_clusters - 1

        return pred
    

import numpy as np

class JenksThresholdPredictor:
    def __init__(self, n_clusters=5, name='Jenks Natural Breaks'):
        """
        Initialise un modèle basé sur la méthode Jenks Natural Breaks.
        
        Args:
            n_clusters (int): Nombre de clusters/classes souhaités.
            name (str): Nom du modèle.
        """
        self.n_clusters = n_clusters
        self.name = name
        self.bounds = None
        self.cluster_centers_ = None
        self.histogram = None

    def _calculate_jenks_breaks(self, X, n_clusters):
        """
        Implémente l'algorithme de Jenks Natural Breaks.
        
        Args:
            X (np.array): Données triées (1D).
            n_clusters (int): Nombre de classes souhaitées.
        
        Returns:
            np.array: Tableau des bornes calculées (breaks).
        """
        data = np.sort(X)
        n = len(data)

        # Matrices pour la minimisation de la variance intra-classe
        lower_class_limits = np.zeros((n + 1, n_clusters + 1), dtype=np.float64)
        variance_combinations = np.zeros((n + 1, n_clusters + 1), dtype=np.float64)

        # Initialisation
        for i in range(1, n + 1):
            lower_class_limits[i][1] = 1
            variance_combinations[i][1] = np.sum((data[:i] - np.mean(data[:i]))**2)

        for k in range(2, n_clusters + 1):
            for i in range(2, n + 1):
                best_variance = float("inf")
                for j in range(1, i):
                    variance = variance_combinations[j][k - 1] + np.sum((data[j:i] - np.mean(data[j:i]))**2)
                    if variance < best_variance:
                        best_variance = variance
                        lower_class_limits[i][k] = j
                variance_combinations[i][k] = best_variance

        # Récupération des bornes optimales
        k = n_clusters
        breaks = np.zeros(n_clusters + 1)
        breaks[-1] = data[-1]
        for i in range(n_clusters - 1, 0, -1):
            breaks[i] = data[int(lower_class_limits[int(breaks[i + 1])][k]) - 1]
            k -= 1
        breaks[0] = data[0]

        return breaks

    def fit(self, X: np.array):
        """
        Calcule les seuils Jenks et prépare le modèle pour la prédiction.
        
        Args:
            X (np.array): Tableau 1D de données pour l'entraînement.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Les données doivent être un tableau numpy (np.array).")

        # Calculer les bornes via la méthode de Jenks
        self.bounds = self._calculate_jenks_breaks(X, self.n_clusters)

        # Calcul des centres des clusters
        self.cluster_centers_ = []
        for i in range(self.n_clusters):
            lower_bound = self.bounds[i]
            upper_bound = self.bounds[i + 1]
            mask = (X >= lower_bound) & (X < upper_bound)
            if np.any(mask):
                self.cluster_centers_.append(np.mean(X[mask]))
            else:
                self.cluster_centers_.append((lower_bound + upper_bound) / 2)

        # Histogramme des points dans chaque cluster
        labels = self.predict(X)
        self.histogram = np.bincount(labels, minlength=self.n_clusters)

    def predict(self, X: np.array):
        """
        Prédire les classes pour les nouvelles données X.
        
        Args:
            X (np.array): Données à classer.
        
        Returns:
            np.array: Tableau des classes prédites.
        """
        if self.bounds is None:
            raise ValueError("Le modèle n'a pas encore été entraîné. Appelez 'fit' d'abord.")
        
        pred = np.zeros(X.shape[0], dtype=int)
        for i in range(self.n_clusters):
            lower_bound = self.bounds[i]
            upper_bound = self.bounds[i + 1]
            mask = (X >= lower_bound) & (X < upper_bound)
            pred[mask] = i
        
        # Assignation des valeurs égales au max dans le dernier cluster
        pred[X == self.bounds[-1]] = self.n_clusters - 1
        return pred