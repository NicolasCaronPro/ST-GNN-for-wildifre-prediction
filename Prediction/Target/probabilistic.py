from pathlib import Path
import numpy as np
from sklearn.metrics import log_loss
from sklearn.isotonic import IsotonicRegression
from tools_functions import *

class Probabilistic():
    def __init__(self,
                 precision_x: float,
                 precision_y: float,
                 precision_z :float,
                 learner,
                 dir_output: Path,
                 resolution : str):

        self.precision = (precision_x, precision_y, precision_z)
        self.learner = learner
        self.dir_output = dir_output
        self.resolution = resolution
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
    
    def _process_input_raster(self, dims : list, 
                                input : list,
                                number_of_input : int,
                                save : bool,
                                names : list,
                                doPast : bool,
                                allDates : list,
                                departement : list,
                                read_pca : bool):
        if save:
            check_and_create_path(self.dir_output / 'log')

        self.X = []
        self.Y = []
        self.W = []

        if isinstance(allDates[0], str):
            months = [d.split('-')[1] for d in allDates]
        else:
            months = [d.strftime('%Y-%m-%d').split('-')[1] for d in allDates]

        high_season = []
        low_season = []
        medium_season = []

        for i, month in enumerate(months):
            if month in ['06', '07', '08', '09']:
                high_season.append(i)
            elif month in ['02', '03', '04', '05']:
                medium_season.append(i)
            else:
                low_season.append(i)

        for band in range(number_of_input):

            dim = dims[band]
            med_dim = dim[0]
            high_dim = dim[1]
            low_dim = dim[-1]

            daily = np.full(input[band].shape, fill_value=np.nan)
            seasonaly = np.full(input[band].shape, fill_value=np.nan)

            daily[:, :, high_season] = np.array(influence_index3D(input[band], np.isnan(input[band]),
                                                            dimS=self.precision, mode='laplace',
                                                            dim=(high_dim[0], high_dim[1], high_dim[2]), semi=doPast))[:, :, high_season]
            
            seasonaly[:, :, high_season] = np.array(influence_index3D(input[band], np.isnan(input[band]),
                                                          dimS=self.precision, mode='mean',
                                                           dim=(high_dim[0], high_dim[1], high_dim[2]), semi=doPast))[:, :, high_season]


            daily[:, :, medium_season] = np.array(influence_index3D(input[band], np.isnan(input[band]),
                                                            dimS=self.precision, mode='laplace',
                                                            dim=(med_dim[0], med_dim[1], med_dim[2]), semi=doPast))[:, :, medium_season]
            
            seasonaly[:, :, medium_season] = np.array(influence_index3D(input[band], np.isnan(input[band]),
                                                          dimS=self.precision, mode='mean',
                                                           dim=(med_dim[0], med_dim[1], med_dim[2]), semi=doPast))[:, :, medium_season]
            

            daily[:, :, low_season] = np.array(influence_index3D(input[band], np.isnan(input[band]),
                                                            dimS=self.precision, mode='laplace',
                                                            dim=(low_dim[0], low_dim[1], low_dim[2]), semi=doPast))[:, :, low_season]
            
            seasonaly[:, :, low_season] = np.array(influence_index3D(input[band], np.isnan(input[band]),
                                                          dimS=self.precision, mode='mean',
                                                           dim=(low_dim[0], low_dim[1], low_dim[2]), semi=doPast))[:, :, low_season]
            
            years = ['2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']
            if departement[band] == 'departement-01-ain' or departement[band] == 'departement-69-rhone':
                years = years[1:]
            if departement == 'departement-69-rhone':
                years = years[:-2]
            
            historical = np.zeros((daily.shape))
            for index, date in enumerate(allDates):
                if isinstance(date, str):
                    vec = date.split('-')
                else:
                     vec = date.strftime('%Y-%m-%d').split('-')
                month = vec[1]
                day = vec[2]
                for y in years:
                    newd = y+'-'+month+'-'+day
                    if newd not in allDates:
                        continue
                    newindex = allDates.index(newd)
                    if doPast and newindex > index:
                        continue
                    historical[:, :, index] += input[band][:, :, newindex]
                historical[:, :, index] /= len(years)
            
            influence = np.full(daily.shape, np.nan)

            historical[np.isnan(input[band])] = np.nan
            seasonaly[np.isnan(input[band])] = np.nan
            daily[np.isnan(input[band])] = np.nan

            #influence = daily + historical + seasonaly
            #influence = (daily + historical + seasonaly) / 3
            influence = np.copy(daily)

            mask = ~np.isnan(input[band])
            df = pd.DataFrame(index=np.arange(0, historical[mask].shape[0]))
            df['1'] = historical[mask]
            df['2'] = seasonaly[mask]
            df['3'] = daily[mask]
            if not read_pca:
                pca =  PCA(n_components='mle', svd_solver='full')
                components = pca.fit_transform(df.values)
                check_and_create_path(self.dir_output / 'pca')
                show_pcs(pca, 2, components, self.dir_output / 'pca')
                print('99 % :',find_n_component(0.99, pca))
                print('95 % :', find_n_component(0.95, pca))
                print('90 % :' , find_n_component(0.90, pca))
                print('85 % :' , find_n_component(0.85, pca))
                print('75 % :' , find_n_component(0.75, pca))
                print('70 % :' , find_n_component(0.70, pca))
                print('60 % :' , find_n_component(0.60, pca))
                print('50 % :' , find_n_component(0.50, pca))

                pca =  PCA(n_components=1, svd_solver='full')
            else:
                if not doPast:
                    pca = read_object(f'{names[band]}Influence.pkl', self.dir_output / 'log' / self.resolution)
                else:
                    pca = read_object(f'{names[band]}pastInfluence.pkl', self.dir_output / 'log' / self.resolution)

            res = pca.fit_transform(df.values)

            influence[mask] = res[:, 0]
            influence[np.isnan(input[band])] = np.nan
            if np.nanmin(influence) < 0:
                influence += abs(np.nanmin(influence))
            
            print(np.nanmin(influence), np.nanmean(influence), np.nanmax(influence), np.nanstd(influence))
            if save:
                if not doPast:
                    outputName = names[band]+'Influence.pkl'
                    histoname = names[band]+'Historical.pkl'
                    dailyname = names[band]+'Daily.pkl'
                    seasoname = names[band]+'Season.pkl'
                    pcaname = names[band]+'PCA.pkl'
                else:
                    outputName = names[band]+'pastInfluence.pkl'
                    histoname = names[band]+'pastHistorical.pkl'
                    dailyname = names[band]+'pastDaily.pkl'
                    seasoname = names[band]+'pastSeason.pkl'
                    pcaname = names[band]+'pastPCA.pkl'

                save_object(pca, pcaname, self.dir_output / 'log' / self.resolution)
                save_object(influence, outputName ,self.dir_output / 'log' / self.resolution)
                save_object(historical, histoname ,self.dir_output / 'log' / self.resolution)
                save_object(daily, dailyname ,self.dir_output / 'log' / self.resolution)
                save_object(seasonaly, seasoname ,self.dir_output / 'log' / self.resolution)

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

    def cluster(self, n_clusters, X, dir_output):
        predictor = KMeans(n_clusters=n_clusters, random_state=42)
        predictor.fit(X)
        save_object(predictor, 'classifier', dir_output / 'classifier')

    def predict_proba(self, X : np.array):
        return self.learner.predict_proba(X)[:,1]