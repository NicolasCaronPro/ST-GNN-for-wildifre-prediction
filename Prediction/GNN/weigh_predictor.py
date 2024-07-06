from sklearn.cluster import KMeans, DBSCAN
import numpy as np

class Predictor():
    def __init__(self, n_clusters, name, type='kmeans', eps=0.5):
        if type == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif type == 'dbscan':
            self.model = DBSCAN(eps=0.5)
        
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
            self.cluster_centers = self.model.cluster_centers_
        else:
            self.cluster_centers= []
            cls = np.unique(Xpre)
            for c in cls:
                mask = np.argwhere(Xpre == c)
                self.cluster_centers.append(np.mean(X[mask]))
            self.cluster_centers = np.asarray(self.cluster_centers)
            
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
        valueOfClass = self.histogram[c]
        valueOfMinClass = self.histogram[self.highNumberClass][:,0][0]
        return valueOfMinClass / valueOfClass
    
    def weight_array(self, array : np.array):
        lis = [self.weight(c) for c in array]
        return np.asarray(lis)
    
    def get_centroid(self, c : int):
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
