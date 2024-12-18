from GNN.train import *

class MinMaxClass(BaseEstimator):
    def __init__(self, thresholds):
        self.num_class = len(thresholds) + 1
        self.thresholds = thresholds
        self.scaler = MinMaxScaler()
        self.class_means = None  # Mean values for each class
        self.class_mins = None   # Minimum values for each class
        self.class_maxs = None   # Maximum values for each class

    def fit(self, X):
        # Apply the scaler to normalize the data
        self.scaler.fit(X.reshape(-1,1))
        x_scaled = self.scaler.transform(X.reshape(-1,1)).flatten()

        # Initialize lists to store class statistics
        self.class_means = [0]
        self.class_mins = [0]
        self.class_maxs = [0]

        prev_thresh = 0  # Start with the lowest threshold (zero)

        for thresh in self.thresholds:
            # Mask for values in the range [prev_thresh, thresh]
            mask = (x_scaled > prev_thresh) & (x_scaled <= thresh)
            
            # Compute statistics
            if mask.any():
                values = X[mask]
                self.class_means.append(values.mean())
                self.class_mins.append(values.min())
                self.class_maxs.append(values.max())
            else:
                self.class_means.append(0)
                self.class_mins.append(float('inf'))
                self.class_maxs.append(float('-inf'))
            
            prev_thresh = thresh
        
        # Add statistics for the upper class (above the last threshold)
        mask = x_scaled > prev_thresh
        if mask.any():
            values = X[mask]
            self.class_means.append(values.mean())
            self.class_mins.append(values.min())
            self.class_maxs.append(values.max())
        else:
            self.class_means.append(0)
            self.class_mins.append(float('inf'))
            self.class_maxs.append(float('-inf'))

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def predict(self, X):
        # Predict classes based on normalized thresholds

        x_scaled = self.scaler.transform(X.reshape(-1,1)).flatten()
            
        return x_scaled

    def predict_nbsinister(self, Xc):
        if self.class_means is None:
            raise ValueError("The model must be fitted before calling predict_nbsinister.")

        predictions = np.zeros(Xc.shape[0])

        print(self.class_means)

        for i, thresh in enumerate(self.thresholds):
            mask = Xc == i
            predictions[mask] = self.class_means[i]
        
        return predictions

    def predict_nbsinister_min(self, Xc):
        """
        Predict the minimum value for the class of each instance in X.
        """
        if self.class_mins is None:
            raise ValueError("The model must be fitted before calling predict_nbsinister_min.")

        predictions = np.zeros(Xc.shape[0])

        for i, thresh in enumerate(self.thresholds):
            mask = Xc == i
            predictions[mask] = self.class_mins[i]
        
        return predictions

    def predict_nbsinister_max(self, Xc):
        """
        Predict the maximum value for the class of each instance in X.
        """
        if self.class_maxs is None:
            raise ValueError("The model must be fitted before calling predict_nbsinister_max.")
        
        predictions = np.zeros(Xc.shape[0])

        for i, thresh in enumerate(self.thresholds):
            mask = Xc == i
            predictions[mask] = self.class_maxs[i]
        
        return predictions

    def predict_risk(self, X):
        pred = self.predict(X)
        res = np.zeros(X.shape[0])
        prev_thresh = 0
        for i, thresh in enumerate(self.thresholds):
            res[(pred > prev_thresh) & (pred <= thresh)] = i + 1
            prev_thresh = thresh

        return res

class KSDiscretizer(BaseEstimator):
    def __init__(self, thresholds_ks, score_col='ks_stat', thresh_col='optimal_score', dir_output=Path('./')):
        self.thresholds_ks = thresholds_ks
        self.score_col = score_col
        self.thresh_col = thresh_col
        self.ks_results = None
        self.dir_output = dir_output

    def fit(self, test_window, y_pred, col):
        self.ks_results = calculate_ks(test_window, 'prediction', col, self.thresholds_ks, self.dir_output)
        return self

    def predict(self, y_pred):
        assert self.ks_results is not None
        new_pred = np.full(y_pred.shape[0], fill_value=0.0)
        max_value = self.ks_results[self.ks_results['threshold'] == self.thresholds_ks[0]][self.score_col].values
        for threshold in self.thresholds_ks:
            if self.ks_results[self.ks_results['threshold'] == threshold][self.score_col].values[0] >= max_value:
                new_pred[y_pred >= self.ks_results[self.ks_results['threshold'] == threshold][self.thresh_col].values[0]] = threshold
        return new_pred

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