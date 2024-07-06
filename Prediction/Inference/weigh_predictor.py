from sklearn.cluster import KMeans
import numpy as np

class Predictor():
    def __init__(self, n_clusters, name):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.name = name

    def fit(self, X : np.array):
        self.train_X = X
        self.kmeans.fit(X.reshape(-1,1))
        self.cluster_centers = self.kmeans.cluster_centers_
        Xpre = self.kmeans.predict(X.reshape(-1,1))
        self.histogram = np.bincount(Xpre)
        self.highNumberClass = np.argwhere(self.histogram == np.max(self.histogram))

    def predict(self, X : np.array, min_class : int = 0):
        X = X.astype(float)
        pred = self.kmeans.predict(X.reshape(-1,1)) + min_class
        mask = self.train_X > np.nanmax(self.train_X)
        if True in np.unique(mask):
            pred[mask] += 1
        return pred
    
    def weight(self, c : int):
        valueOfClass = self.histogram[c]
        valueOfMinClass = self.histogram[self.highNumberClass][:,0][0]
        return valueOfMinClass / valueOfClass
    
    def weight_array(self, array : np.array):
        lis = [self.weight(c) for c in array]
        return np.asarray(lis)
    
    def get_centroid(self, c : int):
        return self.kmeans.cluster_centers_[c]

    def log(self, logger = None):
        if logger is not None:
            logger.info(f'############# Predictor {self.name} ###############')
            logger.info('Histogram')
            logger.info(self.histogram)
            logger.info('Cluster Centers')
            logger.info(self.kmeans.cluster_centers_)
            logger.info(f'####################################')
        else:
            print(f'############# Predictor {self.name} ###############')
            print('Histogram')
            print(self.histogram)
            print('Cluster Centers')
            print(self.kmeans.cluster_centers_)
            print(f'####################################')
