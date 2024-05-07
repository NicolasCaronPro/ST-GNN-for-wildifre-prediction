from pathlib import Path
import numpy as np
from sklearn.metrics import log_loss
from sklearn.isotonic import IsotonicRegression
from tools_inference import *

class Probabilistic():
    def __init__(self, dims,
                 precision_x: float,
                 precision_y: float,
                 precision_z :float,
                 learner,
                 dir_output: Path):
        
        self.dim = dims # (longitude, latitude, temporal)
        self.precision = (precision_x, precision_y, precision_z)
        self.learner = learner
        self.dir_output = dir_output
        self.config = {}

    def _metrics(self, O : np.array,
                 Y : np.array,
                 W : np.array):
        loglossProba = np.zeros((O.shape[0], 2))
        loglossProba[:,0] = 1 - O
        loglossProba[:,1] = O

        self.config['logloss_'+self.sinisterType] = log_loss(Y, loglossProba, sample_weight=W)

        ones = np.argwhere(Y == 1)
        loglossProba = np.zeros(shape=(len(ones), 2))
        loglossProba[:,0] = 1 - O[ones[:,0]]
        loglossProba[:,1] = O[ones[:,0]]

        self.config['fireLogLoss_'+self.sinisterType] = log_loss(Y[ones[:,0]], \
                                                        loglossProba, sample_weight=W[ones[:,0]], labels=[0,1])

        bins, ECE = expected_calibration_error(Y, O)
        self.config['ECE_bins'+self.sinisterType] = bins
        self.config['ECE_val'+self.sinisterType] = ECE
        
    def _clustering(self, input : np.array,
                    nclass : int):
        print('Clusterized')

        self.config['n_class'] = nclass
        kmeansClass = KMeans(n_clusters=nclass, random_state=0).fit(input.reshape(-1,1))

        print(kmeansClass.cluster_centers_)
        kcc = np.sort(kmeansClass.cluster_centers_.reshape(-1))
        value4class = {}

        for i in range(nclass):
            arg = np.where(kmeansClass.cluster_centers_[:,0] == kcc[i])[0]
            value4class[arg[0]] = i

        def get_class(x):
            cluster = kmeansClass.predict([[x]])[0]
            return value4class[cluster]
        
        res = [get_class(x) for x in input]
        return np.array(res)
    
    def _weights(self, input : np.array):
        print('Caclulate samples weights using', (np.unique(input)))

        weights = {}

        for c in np.unique(input):
  
            wr = (len(np.where(input == 0)[0]) / len(np.where(input == c)[0]))
            
            print(f'Class {c} : Weights {wr}')
            weights[c] = wr

        return weights
    
    def _process_input_raster(self, input : list,
                                number_of_input : int,
                                save : bool,
                                names : list,
                                doPast : bool):
        if save:
            check_and_create_path(self.dir_output / 'log')

        self.X = []
        self.Y = []
        self.W = []
        for band in range(number_of_input):

            dim = self.dim[band]

            density = np.array(influence_index3D(input[band], np.isnan(input[band]),
                                                            dimS=self.precision, mode='laplace',
                                                            dim=(dim[0], dim[1], dim[2]), semi=doPast))
            influence = np.copy(density)

            influence[np.isnan(input[band])] = np.nan

            influence = np.round(influence, 3)
            print(np.nanmin(influence), np.nanmean(influence), np.nanmax(influence), np.nanstd(influence))
            if save:
                if not doPast:
                    outputName = names[band]+'Influence.pkl'
                else:
                    outputName = names[band]+'pastInfluence.pkl'
                save_object(influence, outputName ,self.dir_output / 'log')

    def fit(self):
        X, Y = quantile_prediction_error(self.Y, self.X)
        #X, Y = self.X, self.Y
        #X = MinMaxScaler().fit_transform(X.reshape(-1,1)).reshape(-1)
        self.learner.fit(X.reshape(-1,1), Y)

        if isinstance(self.learner, IsotonicRegression):
            output = self.learner.predict(X.reshape(-1,1))
            print(output.shape)
        else:
            output = self.learner.predict_proba(self.X.reshape(-1,1))
            print(output.shape)
            output = output[:,1]

        quantile_prediction_error(self.Y, output)

    def predict_proba(self, X : np.array):
        return self.learner.predict_proba(X)[:,1]