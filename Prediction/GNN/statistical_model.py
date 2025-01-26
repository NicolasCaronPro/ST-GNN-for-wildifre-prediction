from sympy import centroid
from GNN.train import *

class Statistical_Model:
    def __init__(self, variables, thresholds, num_clusters, target, task_type):
        self.thresholds = thresholds
        self.variables = variables
        self.num_clusters = num_clusters
        self.task_type = task_type
        self.target = target
        self.name = f'{variables.replace("_", "-")}-{thresholds}-{num_clusters}_full_one_{target}_{task_type}_None'

    def fit(self, df_train, col_id):
        """
        Entraîne les KMeans individuellement pour chaque ID.
        :param df_train: DataFrame contenant les données d'entraînement
        :param ids: Tableau contenant les IDs pour chaque échantillon
        """

        ids = df_train[col_id].values
        self.models = {}

        if self.thresholds is None:
            return

        if self.thresholds == 'auto':
            for unique_id in np.unique(ids):
                # Sélectionne les échantillons correspondant à l'ID courant
                mask = ids == unique_id
                X = np.unique(df_train[self.variables].values[mask]).reshape(-1, 1)

                # Entraîne un modèle KMeans pour cet ID
                model = KMeans(n_clusters=self.num_clusters, n_init=10, random_state=42)
                model.fit(X)

                # Trie les labels par ordre croissant des centroids
                centroids = model.cluster_centers_.flatten()
                sorted_indices = np.argsort(centroids)
                logger.info(f'Centroids : {self.variables} -> {centroids}')
                label_map = {label: i for i, label in enumerate(sorted_indices)}

                # Stocke le modèle et les labels triés
                self.models[unique_id] = {
                    'model': model,
                    'label_map': label_map
                }

    def predict(self, df, col_id):
        """
        Prédit les classes pour les données en fonction des modèles entraînés.
        :param df_train: DataFrame contenant les données d'entrée
        :param ids: Tableau contenant les IDs pour chaque échantillon
        :return: Tableau des prédictions
        """
        predictions = np.zeros(df.shape[0], dtype=int)
        ids = df[col_id].values
        ids_graph = df['graph_id'].values

        if self.thresholds == 'shift':
            res = np.zeros_like(df[self.variables].values).reshape(-1)
            for unique_id in np.unique(ids_graph):
                mask = ids_graph == unique_id
                shifted_values = np.roll(df[self.variables].values[mask], shift=1)
                shifted_values[0] = 0
                res[mask] = shifted_values

            return res

        elif self.thresholds == 'auto':
            for unique_id in np.unique(ids):
                # Sélectionne les échantillons correspondant à l'ID courant
                mask = ids == unique_id
                X = df[self.variables].values[mask].reshape(-1, 1)

                # Applique le modèle correspondant
                model_info = self.models[unique_id]
                model = model_info['model']
                label_map = self.models[unique_id]['label_map']

                # Prédit les labels et les mappe aux labels triés
                raw_labels = model.predict(X)
                predictions[mask] = np.vectorize(label_map.get)(raw_labels)
        elif self.thresholds != None:
            X = df[self.variables].values
            predictions = np.digitize(X, self.thresholds).astype(int)
        else:
            predictions = df[self.variables].values

        return predictions