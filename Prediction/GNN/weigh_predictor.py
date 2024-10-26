from sklearn.cluster import KMeans, DBSCAN
import numpy as np

class Predictor():
    def __init__(self, n_clusters, name='', type='kmeans', eps=0.5, binary=False):
        if type == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif type == 'dbscan':
            self.model = DBSCAN(eps=0.5)

        self.binary = binary
        self.bounds = None
        self.n_clusters = n_clusters
        self.type = type
        self.name = name

    def fit(self, X : np.array):
        self.train_X = X
        if self.binary:
            self.cluster_centers = np.asarray([0.0,0.2,0.4,0.6,0.8])
            Xpre = np.full(X.shape[0], fill_value=0, dtype=int)
            for c, bound in enumerate(self.cluster_centers):
                if c == 0:
                    mask = np.argwhere((X < 0.1))[:, 0]
                elif c == 1:
                    mask = np.argwhere(((X >= 0.1) & (X < 0.25)))[:, 0]
                elif c == 2:
                    mask = np.argwhere(((X >= 0.25) & (X < 0.64)))[:, 0]
                elif c == 3:
                    mask = np.argwhere(((X >= 0.65) & (X < 0.95)))[:, 0]
                elif c == 4:
                    mask = np.argwhere(((X >= 0.95)))[:, 0]

                if mask.shape[0] == 0:
                    continue
                else:
                    Xpre[mask] = c
        else:
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
        if self.binary:
            pred = np.full(X.shape[0], fill_value=0, dtype=int)
            assert self.cluster_centers is not None
            for c, bound in enumerate(self.cluster_centers):
                if c == 0:
                    mask = np.argwhere((X < 0.1))[:, 0]
                elif c == 1:
                    mask = np.argwhere(((X >= 0.1) & (X < 0.25)))[:, 0]
                elif c == 2:
                    mask = np.argwhere(((X >= 0.25) & (X < 0.65)))[:, 0]
                elif c == 3:
                    mask = np.argwhere(((X >= 0.65) & (X < 0.95)))[:, 0]
                elif c == 4:
                    mask = np.argwhere(((X >= 0.95)))[:, 0]

                if mask.shape[0] == 0:
                    continue
                else:
                    pred[mask] = c
        else:
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
