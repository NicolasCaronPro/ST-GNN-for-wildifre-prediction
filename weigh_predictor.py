from sklearn.cluster import KMeans
import numpy as np

class Predictor():
    def __init__(self, n_clusters):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    def fit(self, X : np.array):
        self.kmeans.fit(X.reshape(-1,1))
        self.cluster_centers = self.kmeans.cluster_centers_
        Xpre = self.kmeans.predict(X.reshape(-1,1))
        self.histogram = np.bincount(Xpre)
        self.highNumberClass = np.argwhere(self.histogram == np.max(self.histogram))
        print(f'High number class found at index {self.highNumberClass} \
              with value of {self.cluster_centers[self.highNumberClass]} \
              and number of {self.histogram[self.highNumberClass]}')

    def predict(self, X : np.array):
        pred = self.kmeans.predict(X.reshape(-1,1))
        return pred
    
    def weight(self, c : int):
        valueOfClass = self.histogram[c]
        valueOfMinClass = self.histogram[self.highNumberClass][:,0]
        return valueOfMinClass / valueOfClass
    
    def weight_array(self, array : np.array):
        return np.asarray([self.weight(c) for c in array])
    
    def get_centroid(self, c : int):
        return self.kmeans.cluster_centers_[c]

    def log(self):
        print(self.histogram)