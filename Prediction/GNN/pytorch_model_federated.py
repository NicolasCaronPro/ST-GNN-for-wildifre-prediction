from GNN.pytorch_model_tools import *

import numpy as np
from copy import deepcopy
import torch
import pandas as pd

########################################## Federated Learning #########################################

class FederatedLearningModel(RegressorMixin, ClassifierMixin):
    def __init__(self, federated_model, features, federated_cluster='departement', loss='mse',
                 name='FederatedModel', dir_log=Path('../'), under_sampling='full', over_sampling='full',
                 target_name='nbsinister', post_process=None, task_type='classification',
                 aggregation_method='max', nbfeatures='all', n_run=1, horizon=0):
        """
        Initialize the Federated Learning Model.

        Parameters:
        - federated_model: The base model to be used across all clusters.
        - federated_cluster: Column name used to identify clusters (default: 'departement').
        - aggregation_method: Method to aggregate local models ('mean', 'median', 'weighted', etc.).
        """
        super().__init__()
        self.features_name = features
        self.federated_cluster = federated_cluster
        self.name = name
        self.loss = loss
        self.dir_log = dir_log
        self.under_sampling = under_sampling
        self.over_sampling = over_sampling
        self.target_name = target_name
        self.post_process = post_process
        self.task_type = task_type
        self.aggregation_method = aggregation_method  # Méthode d'agrégation
        self.global_model = deepcopy(federated_model)  # Modèle global
        self.nbfeatures = nbfeatures
        self.n_run = n_run

        self.horizon = horizon

    def fit(self, df_train, df_val, df_test, graph, args):
        """
        Train local models for each federated cluster, aggregate them into a global model, 
        and stop training once the global score does not improve for patience_count_global epochs.
        """

        #importance_df = calculate_and_plot_feature_importance(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        #importance_df = calculate_and_plot_feature_importance_shapley(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        #features95, featuresAll = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)
        
        #if self.nbfeatures != 'all':
        #    self.features_name = featuresAll[:int(self.nbfeatures)]
        #else:
            #features_name = featuresAll

        self.global_model.features_name = self.features_name
        self.global_model.nbfeatures = 'all'
        self.global_model.graph = graph

        initiate_model, model_params = self.global_model.make_model(graph, custom_model_params=None)
        
        # Vérifier que la méthode d'agrégation est implémentée
        if self.aggregation_method not in ['mean', 'median', 'weighted', 'max', 'fltg']:
            raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' is not implemented.")
        
        print(args)
        # Récupération des paramètres d'entraînement
        global_epochs = args.get('global_epochs')
        local_epochs = args.get('local_epochs')
        patience_count_global = args.get('patience_count_global')
        patience_count_local = args.get('patience_count_local')

        if self.federated_cluster in np.unique(df_train.columns):
            clusters = df_train[self.federated_cluster].unique()
        else:
            raise NotImplementedError(f"Aggregation method '{self.federated_cluster}' is not implemented.")
        # If using FLTG aggregation, prepare structures to record per-client weights over epochs
        if self.aggregation_method == 'fltg':
            try:
                self.fltg_clients = list(clusters)
                self.fltg_scores_history = []
            except Exception:
                self.fltg_clients = None
                self.fltg_scores_history = []

        print(f"\n--- Training Federated Model for {global_epochs} global epochs ---")
        
        self.metrics = {}

        tp = 'client-based'

        class_freq = self.global_model.get_class_freq(df_train)

        for run in range(self.n_run):

            self.global_model.model = deepcopy(initiate_model)
            self.global_model.model_params = deepcopy(model_params)
            local_models = {}
            self.score_per_epochs = {}
            self.score_per_epochs['epoch'] = []
            self.score_per_epochs['score'] = []

            seed = int(random.random())

            best_global_score = float('-inf')
            patience_counter = 0  # Compteur pour l'arrêt anticipé
            for epoch in range(global_epochs):
                print(f"\n--- Global Epoch {epoch + 1}/{global_epochs} ---")

                local_weights = []
                sample_counts = []
                
                for cluster in clusters:

                    if self.federated_cluster == 'departement':
                        if not is_below_threshold():   # threshold par défaut = 0.35
                            print(f"\nSkip: {cluster}")
                            continue

                    print(f"\nTraining local model for cluster: {cluster}")

                    # Création des datasets pour le cluster fédéré
                    df_train_cluster, df_val_cluster, df_test_cluster = self.create_cluster_set(
                        df_train, df_val, df_test, cluster
                    )

                    if np.all(df_train[self.global_model.target_name].values == 0) or np.all(df_val[self.global_model.target_name].values == 0) or np.all(df_test[self.global_model.target_name].values == 0):
                        print(f'Skipping {cluster} due to no positif samples')
                        continue
                    
                    if df_val_cluster.shape[0] == 0 or df_train_cluster.shape[0] == 0:
                        print(f'Skipping {cluster} due to empty dataset')
                        continue

                    if cluster in local_models.keys():
                        local_model = local_models[cluster]
                        local_model.model.load_state_dict(self.global_model.model.state_dict())
                    else:
                        # Initialisation du modèle local
                        local_model = deepcopy(self.global_model)
                        local_model.class_freq = class_freq
                        local_model.seed = seed
                        local_model.name = f'{self.federated_cluster}_{cluster}_{self.global_model.name}'
                        local_model.dir_log = self.dir_log / local_model.name
                        if epoch == 0:
                            check_and_create_path(local_model.dir_log)
                        local_model.features_name = self.features_name
                        local_model.nbfeatures = 'all'

                    if epoch == 0 or cluster not in local_models.keys():
                        local_model.create_train_val_test_loader(graph, df_train_cluster, df_val_cluster, df_test_cluster, local_epochs, patience_count_local, CHECKPOINT, False)
                        
                    # Entraînement du modèle local
                    local_model.train(graph, patience_count_local, CHECKPOINT, local_epochs, verbose=False, custom_model_params=None, new_model=False)

                    # Verification des poids
                    w_global = self.global_model.model.state_dict()
                    w_local = local_model.model.state_dict()
                    diff = sum((w_local[k] - w_global[k].to(w_local[k].device)).abs().sum().item() for k in w_global.keys())
                    if diff == 0:
                        print(f"⚠️ WARNING: Local model for cluster {cluster} did NOT update (diff=0). Check LR or gradients.")
                    else:
                        print(f"✅ Local model for cluster {cluster} updated (L1 diff={diff:.4f}).")
                    
                    local_models[cluster] = local_model

                    # Stocker les poids des modèles locaux
                    local_weights.append(deepcopy(local_model.model.state_dict()))
                    sample_counts.append(len(df_train_cluster))

                # Agréger les modèles locaux dans le modèle global
                last_scores = self.aggregate_models(local_weights, sample_counts, epoch)

                # If FLTG returned per-client raw scores, record them (aligning with clusters order)
                if self.aggregation_method == 'fltg' and last_scores is not None:
                    try:
                        arr = last_scores
                        if isinstance(arr, torch.Tensor):
                            arr = arr.detach().cpu().numpy()
                        self.fltg_scores_history.append(arr)
                    except Exception:
                        logger.exception('Failed to append FLTG last_scores to history')

                # Évaluer le modèle global
                global_score = self.global_model.score(df_val, df_val[self.target_name])
                print(f"\nGlobal Model Score after epoch {epoch + 1}: {global_score:.4f}")
                self.score_per_epochs['epoch'].append(epoch)
                self.score_per_epochs['score'].append(global_score)

                # Vérifier si le score s'est amélioré
                if global_score > best_global_score:
                    best_global_score = global_score
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Arrêt anticipé si le score global ne s'améliore plus
                if patience_counter >= patience_count_global:
                    print("\nEarly stopping: Global model score did not improve.")
                    break
            
            loader = self.create_test_loader(graph, df_test)
            test_output, y = self._predict_test_loader(loader)
            test_output = test_output.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            ###################### TEST SET ########################
            loader = self.create_test_loader(graph, df_test)
            test_output, y = self._predict_test_loader(loader)

            test_output = test_output.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            import pandas as pd
            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]
            
            metrics_run = evaluate_metrics(dff[self.target_name], test_output, zones=dff['departement'])
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'test')

            ###################### VAL SET ########################
            loader = self.create_test_loader(graph, df_val)
            test_output, y = self._predict_test_loader(loader)
            test_output = test_output.detach().cpu().numpy()
            
            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]
            
            metrics_run = evaluate_metrics(dff[self.target_name], test_output, zones=dff['departement'])
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'val')
            plot_score_per_epochs(self.score_per_epochs, self.dir_log, f'score_per_epoch_run_{run}')

        self.metrics['best_tp'] = tp

        self.is_fitted_ = True
        # If FLTG was used, save/plot client influence based on recorded weights
        if self.aggregation_method == 'fltg' and getattr(self, 'fltg_scores_history', None) is not None and len(self.fltg_scores_history) > 0:
            try:
                import matplotlib.pyplot as plt
                import pandas as pd

                arr = np.vstack(self.fltg_scores_history)  # shape (n_epochs, K)
                mean_scores = arr.mean(axis=0)

                # Create dataframe with client labels if available
                if getattr(self, 'fltg_clients', None) is not None:
                    clients = list(self.fltg_clients)
                else:
                    clients = [f'client_{i}' for i in range(mean_scores.shape[0])]

                dfw = pd.DataFrame({'client': clients, 'mean_score': mean_scores})
                check_and_create_path(self.dir_log)
                dfw.to_csv(self.dir_log / 'fltg_client_mean_scores.csv', index=False)

                plt.figure(figsize=(12, 6))
                plt.bar(dfw['client'], dfw['mean_score'])
                plt.xticks(rotation=90)
                plt.ylabel('Mean FLTG raw score')
                plt.tight_layout()
                plt.savefig(self.dir_log / 'fltg_client_influence_scores.png')
                plt.close('all')
            except Exception:
                logger.exception('Failed to save/plot FLTG client influence')

        print("\n--- Federated Learning Training Complete ---")

    def create_cluster_set(self, df_train, df_val, df_test, cluster):
        """
        Create training, validation, and test datasets for a given cluster.
        """
        if self.federated_cluster in np.unique(df_train.columns):
            X_cluster = df_train[df_train[self.federated_cluster] == cluster].reset_index(drop=True)
            X_val_cluster = df_val[df_val[self.federated_cluster] == cluster].reset_index(drop=True)
            X_test_cluster = df_test[df_test[self.federated_cluster] == cluster].reset_index(drop=True)

            return X_cluster, X_val_cluster, X_test_cluster
        
        raise NotImplementedError(f"Federated clustering method '{self.federated_cluster}' is not implemented.")

    def aggregate_models(self, local_weights, sample_counts, epoch):
        """
        Aggregate local models into the global model using the chosen method.
        """
        print(f"\n--- Aggregating models using {self.aggregation_method} ---")
        # Special-case FLTG: delegate to the FLTG implementation which contains
        # the robust aggregation logic. We instantiate a temporary FLTG
        # aggregator, run its aggregation and copy the resulting weights back
        # to this federated model's global_model.
        if self.aggregation_method == 'fltg' and epoch > 0:
            try:
                temp_fltg = FLTG(federated_model=self.global_model,
                                 features=self.features_name,
                                 federated_cluster=self.federated_cluster,
                                 loss=self.loss,
                                 name=self.name,
                                 dir_log=self.dir_log,
                                 under_sampling=self.under_sampling,
                                 over_sampling=self.over_sampling,
                                 target_name=self.target_name,
                                 post_process=self.post_process,
                                 task_type=self.task_type,
                                 aggregation_method='fltg',
                                 nbfeatures=self.nbfeatures,
                                 n_run=self.n_run,
                                 horizon=getattr(self, 'horizon', 0))

                # Work on a deepcopy of the global model to avoid accidental side-effects
                temp_fltg.global_model = deepcopy(self.global_model)

                # Call FLTG aggregation which will update temp_fltg.global_model
                print(f"DEBUG: Calling temp_fltg.aggregate_models. Global norm: {sum(p.float().norm().item() for p in self.global_model.model.parameters())}")
                if len(local_weights) > 0:
                     print(f"DEBUG: Local weights[0] norm: {sum(v.float().norm().item() for v in local_weights[0].values())}")
                
                temp_fltg.aggregate_models(local_weights, sample_counts)

                # Copy updated state back to our global_model
                updated_state = temp_fltg.global_model.model.state_dict()
                self.global_model.update_weight(updated_state)

                # If FLTG computed raw scores, return them so caller (fit) can record per-client history
                last_scores = getattr(temp_fltg, '_last_scores', None)
                print("\n--- Global Model Weights Updated (FLTG) ---")
                return last_scores
            except Exception:
                # If FLTG delegation fails, fall back to weighted mean below
                logger.exception('FLTG aggregation failed, falling back to default aggregation')

        else:
            # Default aggregations (mean / median / max / weighted)
            param_keys = local_weights[0].keys()
            new_state_dict = {}

            for key in param_keys:
                stacked_params = torch.stack([weights[key] for weights in local_weights])

                if self.aggregation_method == 'mean':
                    new_state_dict[key] = torch.mean(stacked_params, dim=0)
                elif self.aggregation_method == 'median':
                    new_state_dict[key] = torch.median(stacked_params, dim=0)[0]
                elif self.aggregation_method == 'max':
                    new_state_dict[key] = torch.max(stacked_params, dim=0)[0]
                elif self.aggregation_method == 'weighted' or (self.aggregation_method == 'fltg' and epoch == 0):
                    if sample_counts is not None and len(sample_counts) == len(local_weights):
                        weights = torch.tensor(sample_counts, dtype=torch.float32)
                        weights = weights / weights.sum()
                    else:
                        weights = torch.tensor([1 / len(local_weights)] * len(local_weights), dtype=torch.float32)
                    view_shape = [len(local_weights)] + [1] * (stacked_params.dim() - 1)
                    new_state_dict[key] = torch.sum(stacked_params * weights.view(*view_shape), dim=0)

            # Mettre à jour les poids du modèle global
            self.global_model.update_weight(new_state_dict)
            print("\n--- Global Model Weights Updated ---")

    def predict(self, X, graph=None, return_y=False):
        """
        Predict using the aggregated global model.
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Please train the model before predicting.")

        print(f'Predicting using Global Model')
        return self.global_model.predict(X, graph, return_y)

    def predict_proba(self, X):
        """
        Predict probabilities using the aggregated global model.
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Please train the model before predicting.")

        if hasattr(self.global_model, "predict_proba"):
            return self.global_model.predict_proba(X)
        else:
            raise AttributeError("The global model does not support predict_proba.")
        
    def make_model(self, graph, custom_model_params):
        return self.global_model.make_model(graph, custom_model_params)
    
    def create_test_loader(self, graph, df):
        return self.global_model.create_test_loader(graph, df)
    
    def _predict_test_loader(self, X):
        return self.global_model._predict_test_loader(X)

############################################ ALA Federated Model ##############################################################
class FederatedALA(FederatedLearningModel):
    """Federated learning strategy relying on the :class:`Training` class for local training."""

    def __init__(self, federated_model, eta, features, federated_cluster='departement', loss='mse',
                 name='FederatedModel', dir_log=Path('../'), under_sampling='full', over_sampling='full',
                 target_name='nbsinister', post_process=None, task_type='classification',
                 aggregation_method='max', nbfeatures='all', n_run=1, params_to_update=['linear2'], horizon=0):


        super().__init__(federated_model=federated_model, features=features, federated_cluster=federated_cluster,
                         loss=loss, name=name, dir_log=dir_log, under_sampling=under_sampling,
                         over_sampling=over_sampling, target_name=target_name, post_process=post_process,
                         task_type=task_type, aggregation_method=aggregation_method, nbfeatures=nbfeatures,
                         n_run=n_run, horizon=horizon)
        self.eta = eta
        self.weight = 0.5
        self.params_to_update = params_to_update
        self.horizon = horizon

    def pick_params_by_name(self, model):
        names, params = [], []
        for n, p in model.named_parameters():
            for pick_parm in self.params_to_update:
                if pick_parm in n:
                    names.append(n); params.append(p)
        return names, params
    
    def pick_params_to_replace(self, model):
        names, params = [], []
        for n, p in model.named_parameters():
            add = True
            for pick_parm in self.params_to_update:
                if pick_parm in n:
                    add = False
                    break
            if add:
                names.append(n); params.append(p)
        return names, params

    def fedALA_params(self, local_model):
        with torch.no_grad():
            for p_t, p_g in zip(local_model.params_erase,
                                local_model.params_gp_erase,
                                ):
                p_t.copy_(p_g)

            for p_t, p_prev, p_g, w in zip(local_model.params_p,
                                        local_model.params_tp,
                                        local_model.params_gp,
                                        local_model.weights):
                # met tout sur le bon device/dtype
                p_prev = p_prev.to(local_model.device, dtype=p_t.dtype)
                p_g    = p_g.detach().to(local_model.device, dtype=p_t.dtype)
                w      = w.to(local_model.device, dtype=p_t.dtype)
                print(w)
                w = w.clamp_(0, 1)
                p_t.copy_(p_prev + (p_g - p_prev) * w)

    def fit(self, df_train, df_val, df_test, graph, args):
        """
        Train local models for each federated cluster, aggregate them into a global model, 
        and stop training once the global score does not improve for patience_count_global epochs.
        """

        importance_df = calculate_and_plot_feature_importance(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        #importance_df = calculate_and_plot_feature_importance_shapley(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        features95, featuresAll = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)
        
        #if self.nbfeatures != 'all':
        #    self.features_name = featuresAll[:int(self.nbfeatures)]
        #else:
        #    self.features_name = featuresAll

        self.global_model.features_name = self.features_name
        self.global_model.nbfeatures = 'all'
        self.global_model.graph = graph

        initiate_model, model_params = self.global_model.make_model(graph, custom_model_params=None)

        # Vérifier que la méthode d'agrégation est implémentée
        if self.aggregation_method not in ['mean', 'median', 'weighted', 'max', 'fltg']:
            raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' is not implemented.")
        
        # Récupération des paramètres d'entraînement
        global_epochs = args.get('global_epochs')
        local_epochs = args.get('local_epochs')
        patience_count_global = args.get('patience_count_global')
        patience_count_local = args.get('patience_count_local')

        if self.federated_cluster in np.unique(df_train.columns):
            clusters = df_train[self.federated_cluster].unique()
        else:
            raise NotImplementedError(f"Aggregation method '{self.federated_cluster}' is not implemented.")

        # If using FLTG aggregation, prepare structures to record per-client weights over epochs
        if self.aggregation_method == 'fltg':
            try:
                self.fltg_clients = list(clusters)
                self.fltg_scores_history = []
            except Exception:
                self.fltg_clients = None
                self.fltg_scores_history = []

        print(f"\n--- Training ALA Federated Model for {global_epochs} global epochs ---")

        tp = 'client-based'
        self.metrics = {}

        class_freq = self.global_model.get_class_freq(df_train)
        
        for run in range(self.n_run):
            
            self.score_per_epochs = {}
            self.model_params = deepcopy(model_params)
            self.global_model.model = deepcopy(initiate_model)
            self.global_model.model_params = deepcopy(model_params)
            self.global_model.eta = self.eta
            local_models = {}
            self.score_per_epochs['epoch'] = []
            self.score_per_epochs['score'] = []
            
            best_global_score = float('-inf')
            patience_counter = 0  # Compteur pour l'arrêt anticipé
            seed = int(random.random())

            for epoch in range(global_epochs):
                print(f"\n--- Global Epoch {epoch + 1}/{global_epochs} ---")

                local_weights = []
                sample_counts = []
                
                for cluster in clusters:

                    if self.federated_cluster == 'departement':
                        if not is_below_threshold():   # threshold par défaut = 0.35
                            print(f"\nSkip: {cluster}")
                            continue

                    print(f"\nTraining local model for cluster: {cluster}")
                    # Création des datasets pour le cluster fédéré
                    df_train_cluster, df_val_cluster, df_test_cluster = self.create_cluster_set(
                        df_train, df_val, df_test, cluster
                    )

                    if np.all(df_train[self.global_model.target_name].values == 0) or np.all(df_val[self.global_model.target_name].values == 0) or np.all(df_test[self.global_model.target_name].values == 0):
                        print(f'Skipping {cluster} due to no positif samples')
                        continue

                    if df_val_cluster.shape[0] == 0 or df_train_cluster.shape[0] == 0:
                        print(f'Skipping {cluster} due to empty dataset')
                        continue

                    # Initialisation du modèle local
                    if cluster not in local_models.keys():
                        local_model = deepcopy(self.global_model)
                        local_model.class_freq = class_freq
                        local_model.ALATraining = False
                        local_model.seed = seed
                        local_model.features_name = self.features_name
                        local_model.nbfeatures = 'all'
                        self.metrics[cluster] = local_model.metrics
                        local_model.name = f'{self.federated_cluster}_{cluster}_{self.global_model.name}'
                        local_model.dir_log = self.dir_log / local_model.name
                        local_model.create_train_val_test_loader(graph, df_train_cluster, df_val_cluster, df_test_cluster, local_epochs, patience_count_local, CHECKPOINT, False, use_log=args['use_log'])
                        check_and_create_path(local_model.dir_log)
                    else:
                        local_model = local_models[cluster]
                        local_model.ALATraining = True
                        local_model.ala_weight_only = False

                        local_model.global_model = self.global_model.model
                        _, params = self.pick_params_by_name(local_model.model)
                        _, params_g = self.pick_params_by_name(self.global_model.model)

                        _, params_erase = self.pick_params_to_replace(local_model.model)
                        _, params_gp_erase = self.pick_params_to_replace(self.global_model.model)

                        assert len(params) > 0
                        
                        local_model.params_p  = params
                        local_model.params_gp = params_g
                        local_model.params_erase = params_erase
                        local_model.params_gp_erase = params_gp_erase

                        # snapshot des poids locaux précédents (gelés)
                        local_model.params_tp = [p.detach().clone() for p in local_model.params_p]
                        
                        for param in local_model.params_tp:
                            param.requires_grad = False
                        
                        if not hasattr(local_model, "weights") and cluster in local_models.keys():
                            local_model_log_params = deepcopy(local_model.model.state_dict())
                            w_t = torch.as_tensor(self.weight, device=local_model.device, dtype=local_model.params_p[0].dtype)
                            local_model.weights = [
                                w_t.expand_as(p).clone() for p in local_model.params_p
                            ]
                            local_model.ala_weight_only = True
                            wil = local_model.weights
                            print("weights", local_model.weights)
                            local_model.train(graph, patience_count_local, CHECKPOINT, local_epochs, verbose=False, custom_model_params={'return_hidden' : True}, new_model=False)
                            local_model.model.load_state_dict(local_model_log_params)
                            print("weights", local_model.weights)
                            
                        # Fed ala params
                        self.fedALA_params(local_model)

                    # Entraînement du modèle local
                    local_model.train(graph, patience_count_local, CHECKPOINT, local_epochs if not hasattr(local_model, "weights") else 1, verbose=False, custom_model_params={'return_hidden' : True}, new_model=False)
                    
                    local_models[cluster] = local_model
                    
                    # Stocker les poids des modèles locaux
                    local_weights.append(deepcopy(local_model.model.state_dict()))
                    sample_counts.append(len(df_train_cluster))

                # Agréger les modèles locaux dans le modèle global
                last_scores = self.aggregate_models(local_weights, sample_counts, epoch)

                # If FLTG returned per-client raw scores, record them (aligning with clusters order)
                if self.aggregation_method == 'fltg' and last_scores is not None:
                    try:
                        arr = last_scores
                        if isinstance(arr, torch.Tensor):
                            arr = arr.detach().cpu().numpy()
                        self.fltg_scores_history.append(arr)
                    except Exception:
                        logger.exception('Failed to append FLTG last_scores to history')

                # Évaluer le modèle global
                global_score = self.global_model.score(df_val, df_val[self.target_name])
                self.score_per_epochs['score'].append(global_score)
                self.score_per_epochs['epoch'].append(epoch)
                print(f"\nGlobal Model Score after epoch {epoch + 1}: {global_score:.4f}")

                # Vérifier si le score s'est amélioré
                if global_score > best_global_score:
                    best_global_score = global_score
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Arrêt anticipé si le score global ne s'améliore plus
                if patience_counter >= patience_count_global:
                    print("\nEarly stopping: Global model score did not improve.")
                    break

            ###################### TEST SET ########################
            loader = self.create_test_loader(graph, df_test)
            test_output, y = self._predict_test_loader(loader)

            test_output = test_output.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]
            
            metrics_run = evaluate_metrics(dff[self.target_name], test_output, zones=dff['departement'])
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'test')

            ###################### VAL SET ########################
            loader = self.create_test_loader(graph, df_val)
            test_output, y = self._predict_test_loader(loader)
            test_output = test_output.detach().cpu().numpy()
            
            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]
            
            metrics_run = evaluate_metrics(dff[self.target_name], test_output, zones=dff['departement'])
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'val')
            plot_score_per_epochs(self.score_per_epochs, self.dir_log, f'score_per_epoch_run_{run}')

        self.metrics['best_tp'] = tp

        self.is_fitted_ = True

        # If FLTG was used, save/plot client influence based on recorded weights
        if self.aggregation_method == 'fltg' and getattr(self, 'fltg_scores_history', None) is not None and len(self.fltg_scores_history) > 0:
            try:
                import matplotlib.pyplot as plt

                arr = np.vstack(self.fltg_scores_history)  # shape (n_epochs, K)
                mean_scores = arr.mean(axis=0)

                # Create dataframe with client labels if available
                if getattr(self, 'fltg_clients', None) is not None:
                    clients = list(self.fltg_clients)
                else:
                    clients = [f'client_{i}' for i in range(mean_scores.shape[0])]

                dfw = pd.DataFrame({'client': clients, 'mean_score': mean_scores})
                check_and_create_path(self.dir_log)
                dfw.to_csv(self.dir_log / 'fltg_client_mean_scores.csv', index=False)

                plt.figure(figsize=(12, 6))
                plt.bar(dfw['client'], dfw['mean_score'])
                plt.xticks(rotation=90)
                plt.ylabel('Mean FLTG raw score')
                plt.tight_layout()
                plt.savefig(self.dir_log / 'fltg_client_influence_scores.png')
                plt.close('all')
            except Exception:
                logger.exception('Failed to save/plot FLTG client influence')

        print("\n--- ALA Federated Learning Training Complete ---")

############################################ MOON Federated Model ##############################################################

class MOONFederatedLearning(FederatedLearningModel):
    def __init__(self, federated_model, features, federated_cluster='departement', loss='mse', 
                 name='MoonFederatedModel', dir_log=Path('../'), under_sampling='full', over_sampling='full',
                 target_name='nbsinister', post_process=None, task_type='classification', 
                 aggregation_method='max', nbfeatures='all', n_run=1, temperature=1, smooth=0):
        
        super().__init__(federated_model=federated_model, features=features, federated_cluster=federated_cluster, loss=loss,
                         name=name, dir_log=dir_log, under_sampling=under_sampling, over_sampling=over_sampling,
                         target_name=target_name, post_process=post_process, task_type=task_type,
                         aggregation_method=aggregation_method, nbfeatures=nbfeatures, n_run=n_run)
        
        self.moon_temperature_value = temperature
        self.smooth_value = smooth
    
    def fit(self, df_train, df_val, df_test, graph, args):
        """
        Train local models for each federated cluster, aggregate them into a global model, 
        and stop training once the global score does not improve for patience_count_global epochs.
        """

        #importance_df = calculate_and_plot_feature_importance(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        #importance_df = calculate_and_plot_feature_importance_shapley(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        #features95, featuresAll = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)
        
        #if self.nbfeatures != 'all':
        #    self.features_name = featuresAll[:int(self.nbfeatures)]
        #else:
        #    self.features_name = featuresAll

        self.global_model.features_name = self.features_name
        self.global_model.nbfeatures = 'all'
        self.global_model.graph = graph

        initiate_model, model_params = self.global_model.make_model(graph, custom_model_params={'return_hidden' : True})

        # Vérifier que la méthode d'agrégation est implémentée
        if self.aggregation_method not in ['mean', 'median', 'weighted', 'max', 'fltg']:
            raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' is not implemented.")

        # Récupération des paramètres d'entraînement
        global_epochs = args.get('global_epochs')
        local_epochs = args.get('local_epochs')
        patience_count_global = args.get('patience_count_global')
        patience_count_local = args.get('patience_count_local')

        if self.federated_cluster in np.unique(df_train.columns):
            clusters = df_train[self.federated_cluster].unique()
        else:
            raise NotImplementedError(f"Aggregation method '{self.federated_cluster}' is not implemented.")
        
        # If using FLTG aggregation, prepare structures to record per-client weights over epochs
        if self.aggregation_method == 'fltg':
            try:
                self.fltg_clients = list(clusters)
                self.fltg_scores_history = []
            except Exception:
                self.fltg_clients = None
                self.fltg_scores_history = []
        
        print(f"\n--- Training Federated Model for {global_epochs} global epochs ---")

        tp = 'client-based'
        
        self.metrics = {}
        class_freq = self.global_model.get_class_freq(df_train)

        for run in range(self.n_run):

            self.score_per_epochs = {}
            self.global_model.model = deepcopy(initiate_model)
            self.global_model.model_params = deepcopy(model_params)
            local_models = {}
            self.score_per_epochs['epoch'] = []
            self.score_per_epochs['score'] = []
            
            best_global_score = float('-inf')
            patience_counter = 0  # Compteur pour l'arrêt anticipé
            seed = int(random.random())

            for epoch in range(global_epochs):
                print(f"\n--- Global Epoch {epoch + 1}/{global_epochs} ---")

                local_weights = []
                sample_counts = []
                
                for cluster in clusters:

                    if self.federated_cluster == 'departement':
                        if not is_below_threshold():   # threshold par défaut = 0.35
                            print(f"\nSkip: {cluster}")
                            continue

                    print(f"\nTraining local model for cluster: {cluster}")
                    # Création des datasets pour le cluster fédéré
                    df_train_cluster, df_val_cluster, df_test_cluster = self.create_cluster_set(
                        df_train, df_val, df_test, cluster
                    )

                    if np.all(df_train[self.global_model.target_name].values == 0) or np.all(df_val[self.global_model.target_name].values == 0) or np.all(df_test[self.global_model.target_name].values == 0):
                        print(f'Skipping {cluster} due to no positif samples')
                        continue

                    if df_val_cluster.shape[0] == 0 or df_train_cluster.shape[0] == 0:
                        print(f'Skipping {cluster} due to empty dataset')
                        continue

                    # Initialisation du modèle local
                    if cluster in local_models.keys():
                        local_model = local_models[cluster]
                        local_model.model.load_state_dict(self.global_model.model.state_dict())
                    else:
                        local_model = deepcopy(self.global_model)
                        local_model.class_freq = class_freq
                        local_model.seed = seed
                        local_model.name = f'{self.federated_cluster}_{cluster}_{self.global_model.name}'
                        local_model.dir_log = self.dir_log / local_model.name
                        print(local_model.dir_log)
                        local_model.global_model = self.global_model
                        local_model.moon_temperature_value = self.moon_temperature_value
                        local_model.smooth_value = self.smooth_value
                        local_model.constrastive = True

                    if epoch == 0 or cluster not in local_models.keys():
                        local_model.prev_model = self.global_model.model
                        local_model.constrastive = False
                        check_and_create_path(local_model.dir_log)
                    else:
                        local_model.prev_model = deepcopy(local_models[cluster].model)
                    
                    local_model.global_model = deepcopy(self.global_model.model)

                    local_model.features_name = self.features_name
                    local_model.nbfeatures = 'all'

                    if epoch == 0 or cluster not in local_models.keys():
                        local_model.create_train_val_test_loader(graph, df_train_cluster, df_val_cluster, df_test_cluster, local_epochs, patience_count_local, CHECKPOINT, False, custom_model_params={'return_hidden' : True})
                        self.metrics[cluster] = local_model.metrics

                    local_model.train(graph, patience_count_local, CHECKPOINT, local_epochs, verbose=False, custom_model_params=None, new_model=False)
                    
                    local_models[cluster] = local_model

                    # Stocker les poids des modèles locaux
                    local_weights.append(deepcopy(local_model.model.state_dict()))
                    sample_counts.append(len(df_train_cluster))

                # Agréger les modèles locaux dans le modèle global
                last_scores = self.aggregate_models(local_weights, sample_counts, epoch)

                # If FLTG returned per-client raw scores, record them (aligning with clusters order)
                if self.aggregation_method == 'fltg' and last_scores is not None:
                    try:
                        arr = last_scores
                        if isinstance(arr, torch.Tensor):
                            arr = arr.detach().cpu().numpy()
                        self.fltg_scores_history.append(arr)
                    except Exception:
                        logger.exception('Failed to append FLTG last_scores to history')

                # Évaluer le modèle global
                global_score = self.global_model.score(df_val, df_val[self.target_name])
                print(f"\nGlobal Model Score after epoch {epoch + 1}: {global_score:.4f}")
                self.score_per_epochs['epoch'].append(epoch)
                self.score_per_epochs['score'].append(global_score)

                # Vérifier si le score s'est amélioré
                if global_score > best_global_score:
                    best_global_score = global_score
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Arrêt anticipé si le score global ne s'améliore plus
                if patience_counter >= patience_count_global:
                    print("\nEarly stopping: Global model score did not improve.")
                    break
            
            loader = self.create_test_loader(graph, df_test)
            test_output, y = self._predict_test_loader(loader)
            test_output = test_output.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]
            
            metrics_run = evaluate_metrics(dff[self.target_name], test_output, zones=dff['departement'])
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'test')
            
            loader = self.create_test_loader(graph, df_val)
            test_output, y = self._predict_test_loader(loader)
            test_output = test_output.detach().cpu().numpy()
            
            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]
            
            metrics_run = evaluate_metrics(dff[self.target_name], test_output, zones=dff['departement'])
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'val')
            plot_score_per_epochs(self.score_per_epochs, self.dir_log, f'score_per_epoch_run_{run}')

        self.metrics['best_tp'] = tp
        self.is_fitted_ = True
        print("\n--- Federated Learning Training Complete ---")

############################################ FED PROX #######################################################

class FederatedProx(FederatedLearningModel):
    def __init__(self, federated_model, features, federated_cluster='departement', loss='mse', 
                 name='ProxFederatedModel', dir_log=Path('../'), under_sampling='full', over_sampling='full',
                 target_name='nbsinister', post_process=None, task_type='classification', 
                 aggregation_method='max', nbfeatures='all', n_run=1, prox_value=1, fed_prox_names=[]):
        
        super().__init__(federated_model=federated_model, features=features, federated_cluster=federated_cluster, loss=loss,
                         name=name, dir_log=dir_log, under_sampling=under_sampling, over_sampling=over_sampling,
                         target_name=target_name, post_process=post_process, task_type=task_type,
                         aggregation_method=aggregation_method, nbfeatures=nbfeatures, n_run=n_run)
        
        self.prox_value = prox_value
        self.fed_prox_names = fed_prox_names
    
    def fit(self, df_train, df_val, df_test, graph, args):
        """
        Train local models for each federated cluster, aggregate them into a global model, 
        and stop training once the global score does not improve for patience_count_global epochs.
        """


        self.global_model.features_name = self.features_name
        self.global_model.nbfeatures = 'all'
        self.global_model.graph = graph

        initiate_model, model_params = self.global_model.make_model(graph, custom_model_params={'return_hidden' : True})

        # Vérifier que la méthode d'agrégation est implémentée
        if self.aggregation_method not in ['mean', 'median', 'weighted', 'max', 'fltg']:
            raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' is not implemented.")

        # Récupération des paramètres d'entraînement
        global_epochs = args.get('global_epochs')
        local_epochs = args.get('local_epochs')
        patience_count_global = args.get('patience_count_global')
        patience_count_local = args.get('patience_count_local')

        if self.federated_cluster in np.unique(df_train.columns):
            clusters = df_train[self.federated_cluster].unique()
        else:
            raise NotImplementedError(f"Aggregation method '{self.federated_cluster}' is not implemented.")
        
        print(f"\n--- Training Federated Model for {global_epochs} global epochs ---")

        tp = 'client-based'
        
        self.metrics = {}
        class_freq = self.global_model.get_class_freq(df_train)

        for run in range(self.n_run):

            self.score_per_epochs = {}
            self.global_model.model = deepcopy(initiate_model)
            self.global_model.model_params = deepcopy(model_params)
            local_models = {}
            self.score_per_epochs['epoch'] = []
            self.score_per_epochs['score'] = []
            
            best_global_score = float('-inf')
            patience_counter = 0  # Compteur pour l'arrêt anticipé
            seed = int(random.random())

            for epoch in range(global_epochs):
                print(f"\n--- Global Epoch {epoch + 1}/{global_epochs} ---")

                local_weights = []
                sample_counts = []
                
                for cluster in clusters:

                    if self.federated_cluster == 'departement':
                        if not is_below_threshold():   # threshold par défaut = 0.35
                            print(f"\nSkip: {cluster}")
                            continue

                    print(f"\nTraining local model for cluster: {cluster}")
                    # Création des datasets pour le cluster fédéré
                    df_train_cluster, df_val_cluster, df_test_cluster = self.create_cluster_set(
                        df_train, df_val, df_test, cluster
                    )

                    if np.all(df_train[self.global_model.target_name].values == 0) or np.all(df_val[self.global_model.target_name].values == 0) or np.all(df_test[self.global_model.target_name].values == 0):
                        print(f'Skipping {cluster} due to no positif samples')
                        continue

                    if df_val_cluster.shape[0] == 0 or df_train_cluster.shape[0] == 0:
                        print(f'Skipping {cluster} due to empty dataset')
                        continue

                    # Initialisation du modèle local
                    if cluster in local_models.keys():
                        local_model = local_models[cluster]
                        local_model.model.load_state_dict(self.global_model.model.state_dict())
                    else:
                        local_model = deepcopy(self.global_model)
                        local_model.class_freq = class_freq
                        local_model.seed = seed
                        local_model.fed_prox_names = self.fed_prox_names
                        local_model.name = f'{self.federated_cluster}_{cluster}_{self.global_model.name}'
                        local_model.dir_log = self.dir_log / local_model.name
                        local_model.global_model = self.global_model
                        local_model.prox_term = True
                        local_model.prox_value = self.prox_value

                    if epoch == 0 or cluster not in local_models.keys():
                        local_model.prev_model = self.global_model.model
                        local_model.prox_term = False
                        check_and_create_path(local_model.dir_log)
                    else:
                        local_model.prev_model = deepcopy(local_models[cluster].model)
                    
                    local_model.global_model = deepcopy(self.global_model.model)

                    local_model.features_name = self.features_name
                    local_model.nbfeatures = 'all'

                    if epoch == 0 or cluster not in local_models.keys():
                        local_model.create_train_val_test_loader(graph, df_train_cluster, df_val_cluster, df_test_cluster, local_epochs, patience_count_local, CHECKPOINT, False, custom_model_params={'return_hidden' : True})
                        self.metrics[cluster] = local_model.metrics

                    local_model.train(graph, patience_count_local, CHECKPOINT, local_epochs, verbose=False, custom_model_params=None, new_model=False)
                    
                    local_models[cluster] = local_model

                    # Stocker les poids des modèles locaux
                    local_weights.append(deepcopy(local_model.model.state_dict()))
                    sample_counts.append(len(df_train_cluster))

                # Agréger les modèles locaux dans le modèle global
                last_scores = self.aggregate_models(local_weights, sample_counts, epoch)

                # Évaluer le modèle global
                global_score = self.global_model.score(df_val, df_val[self.target_name])
                print(f"\nGlobal Model Score after epoch {epoch + 1}: {global_score:.4f}")
                self.score_per_epochs['epoch'].append(epoch)
                self.score_per_epochs['score'].append(global_score)

                # Vérifier si le score s'est amélioré
                if global_score > best_global_score:
                    best_global_score = global_score
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Arrêt anticipé si le score global ne s'améliore plus
                if patience_counter >= patience_count_global:
                    print("\nEarly stopping: Global model score did not improve.")
                    break
            
            loader = self.create_test_loader(graph, df_test)
            test_output, y = self._predict_test_loader(loader)
            test_output = test_output.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]
            
            metrics_run = evaluate_metrics(dff[self.target_name], test_output, zones=dff['departement'])
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'test')
            
            loader = self.create_test_loader(graph, df_val)
            test_output, y = self._predict_test_loader(loader)
            test_output = test_output.detach().cpu().numpy()
            
            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]
            
            metrics_run = evaluate_metrics(dff, self.target_name, test_output)
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'val')
            plot_score_per_epochs(self.score_per_epochs, self.dir_log, f'score_per_epoch_run_{run}')

        self.metrics['best_tp'] = tp
        self.is_fitted_ = True
        print("\n--- Federated Learning Training Complete ---")
        
############################################ Proto Federated Learning ############################################

class FLTG(FederatedLearningModel):
    """
    FLTG = Angle/consensus-based robust aggregation (no server learning).
    - Works well for non-IID extreme clients (acts like "soft Byzantine-robust").
    - Keeps your design: local/global models are instances of another class.
    """

    def __init__(
        self,
        federated_model,
        features,
        federated_cluster='departement',
        loss='mse',
        name='FLTG',
        dir_log=Path('../'),
        under_sampling='full',
        over_sampling='full',
        target_name='nbsinister',
        post_process=None,
        task_type='classification',
        aggregation_method='fltg',   # force method name
        nbfeatures='all',
        n_run=1,
        horizon=0,
        # --- FLTG specific knobs ---
        fltg_clip_tau=100.0,           # clipping threshold on delta norm
        fltg_temperature=1.0,         # softmax temperature for weights
        fltg_min_clients=2,           # safety: if too few clients, fallback to weighted mean
        fltg_key_filter=None,         # None or callable(key)->bool or list/tuple of substrings
        fltg_relu_cos=True,           # only keep positive cosine contributions
        fltg_eps=1e-12
    ):
        super().__init__(
            federated_model=federated_model,
            features=features,
            federated_cluster=federated_cluster,
            loss=loss,
            name=name,
            dir_log=dir_log,
            under_sampling=under_sampling,
            over_sampling=over_sampling,
            target_name=target_name,
            post_process=post_process,
            task_type=task_type,
            aggregation_method=aggregation_method,
            nbfeatures=nbfeatures,
            n_run=n_run,
            horizon=horizon
        )

        self.fltg_clip_tau = fltg_clip_tau
        self.fltg_temperature = fltg_temperature
        self.fltg_min_clients = fltg_min_clients
        self.fltg_key_filter = fltg_key_filter
        self.fltg_relu_cos = fltg_relu_cos
        self.fltg_eps = fltg_eps
        # History for debugging/analysis: record per-client scores/weights per aggregation call
        self._last_scores = None
        self._last_weights = None
        self.fltg_scores_history = []

    # -------------------------
    # Helpers
    # -------------------------
    def _key_is_selected(self, key: str) -> bool:
        """
        Select which parameters participate in FLTG logic.
        - None => all keys
        - callable => user-defined filter
        - list/tuple of substrings => keep keys that contain any substring
        """
        if self.fltg_key_filter is None:
            # Default: exclude batchnorm stats which can dominate norms
            if 'num_batches_tracked' in key: return False
            if 'running_mean' in key: return False
            if 'running_var' in key: return False
            return True
        if callable(self.fltg_key_filter):
            return bool(self.fltg_key_filter(key))
        if isinstance(self.fltg_key_filter, (list, tuple)):
            return any(s in key for s in self.fltg_key_filter)
        return True

    @torch.no_grad()
    def _flatten_delta(self, delta_state_dict: dict) -> torch.Tensor:
        """
        Flatten selected deltas into a 1D vector for cosine computations.
        """
        vecs = []
        for k, v in delta_state_dict.items():
            if not self._key_is_selected(k):
                continue
            if not torch.is_tensor(v):
                continue
            vecs.append(v.reshape(-1).float().cpu())
        if len(vecs) == 0:
            return torch.zeros(1, dtype=torch.float32)
        return torch.cat(vecs, dim=0)

    @torch.no_grad()
    def _compute_deltas(self, local_weights: list, global_weights: dict) -> list:
        """
        deltas[k][param] = local[param] - global[param]
        """
        deltas = []
        for lw in local_weights:
            d = {}
            for key in lw.keys():
                # Always compute delta for all keys (we may only use some for scoring)
                d[key] = (lw[key] - global_weights[key])
                #print(d[key])
            deltas.append(d)
            
            # DEBUG: Check delta norms
            delta_norm = sum(v.float().norm().item() for v in d.values())
            print(f"DEBUG: _compute_deltas client {len(deltas)-1} delta norm: {delta_norm}")
            if delta_norm == 0:
                 print(f"DEBUG: ZERO DELTA DETECTED for client {len(deltas)-1}")
                 # Print first few keys comparison
                 for k in list(d.keys())[:3]:
                     print(f"  Key {k}: local={lw[k].float().norm().item()}, global={global_weights[k].float().norm().item()}, diff={(lw[k]-global_weights[k]).float().norm().item()}")
        
        return deltas

    @torch.no_grad()
    def _clip_delta_state(self, delta_state: dict) -> dict:
        """
        Clip delta_state by global L2 norm over selected keys (for stability).
        """
        flat = self._flatten_delta(delta_state)
        norm = torch.norm(flat) + self.fltg_eps
        scale = min(1.0, float(self.fltg_clip_tau / norm))
        
        print(f"DEBUG: _clip_delta_state norm={norm.item()}, tau={self.fltg_clip_tau}, scale={scale}")
        
        if scale >= 1.0:
            return delta_state

        clipped = {}
        for k, v in delta_state.items():
            clipped[k] = v * scale
        return clipped

    @torch.no_grad()
    def _weighted_mean_fallback(self, local_weights, sample_counts):
        """
        Fallback aggregation (FedAvg weighted by sample_counts).
        """
        param_keys = local_weights[0].keys()
        new_state_dict = {}
        if sample_counts is not None and len(sample_counts) == len(local_weights):
            w = torch.tensor(sample_counts, dtype=torch.float32)
            w = w / (w.sum() + self.fltg_eps)
        else:
            w = torch.ones(len(local_weights), dtype=torch.float32) / max(1, len(local_weights))

        for key in param_keys:
            stacked = torch.stack([lw[key] for lw in local_weights], dim=0)
            view_shape = [len(local_weights)] + [1] * (stacked.dim() - 1)
            new_state_dict[key] = torch.sum(stacked * w.view(*view_shape), dim=0)

        self.global_model.update_weight(new_state_dict)

    # -------------------------
    # Core: FLTG aggregation
    # -------------------------
    def aggregate_models(self, local_weights, sample_counts=None):
        """
        FLTG aggregation:
        1) Compute deltas to global
        2) Clip deltas
        3) Build consensus delta_bar
        4) Compute cosine scores per client: cos(delta_k, delta_bar) on selected layers
        5) Convert scores -> weights (softmax), optionally ReLU-cosine
        6) Update global: w_new = w_global + sum_k a_k * delta_k
        """

        if len(local_weights) == 0:
            print("[FLTG] No local models to aggregate. Skipping.")
            return

        # If too few clients, FLTG scoring is unstable => fallback
        if len(local_weights) < self.fltg_min_clients:
            print(f"[FLTG] Too few clients ({len(local_weights)}). Fallback to weighted mean.")
            return self._weighted_mean_fallback(local_weights, sample_counts)

        print(f"\n--- Aggregating models using FLTG (K={len(local_weights)}) ---")

        global_w = deepcopy(self.global_model.model.state_dict())
        deltas = self._compute_deltas(local_weights, global_w)
        
        print(f"DEBUG: FLTG.aggregate_models global_w norm: {sum(v.float().norm().item() for v in global_w.values())}")
        if len(local_weights) > 0:
             print(f"DEBUG: FLTG.aggregate_models local_weights[0] norm: {sum(v.float().norm().item() for v in local_weights[0].values())}")

        # 1) Clip deltas
        deltas = [self._clip_delta_state(d) for d in deltas]

        # 2) Consensus delta_bar (simple mean of deltas)
        #    You can replace by trimmed mean/median if you want even more robustness.
        delta_bar = {}
        for key in global_w.keys():
            stacked = torch.stack([d[key] for d in deltas], dim=0)
            if stacked.is_floating_point():
                delta_bar[key] = torch.mean(stacked, dim=0)
            else:
                delta_bar[key] = torch.mean(stacked.float(), dim=0).type(stacked.dtype)

        # 3) Cosine scores between each delta_k and consensus delta_bar (selected keys only)
        delta_bar_vec = self._flatten_delta(delta_bar)
        bar_norm = torch.norm(delta_bar_vec) + self.fltg_eps

        scores = []
        
        # DEBUG: Inspect per-layer cosine
        if len(deltas) >= 2:
            print("\nDEBUG: Per-layer cosine similarity (Client 0 vs Client 1):")
            d0 = deltas[0]
            d1 = deltas[1]
            for k in d0.keys():
                if not self._key_is_selected(k): continue
                t0 = d0[k].float().flatten()
                t1 = d1[k].float().flatten()
                if t0.numel() == 0: continue
                
                n0 = torch.norm(t0)
                n1 = torch.norm(t1)
                if n0 > 0 and n1 > 0:
                    cos_k = torch.dot(t0, t1) / (n0 * n1)
                    print(f"  Key {k}: shape={d0[k].shape}, norm0={n0:.4f}, norm1={n1:.4f}, cos={cos_k:.6f}")
                else:
                    print(f"  Key {k}: shape={d0[k].shape}, norm0={n0:.4f}, norm1={n1:.4f}, cos=NaN")

        for d in deltas:
            v = self._flatten_delta(d)
            v_norm = torch.norm(v) + self.fltg_eps
            cos = torch.dot(v, delta_bar_vec) / (v_norm * bar_norm)
            scores.append(cos)
        scores = torch.stack(scores)  # [K]

        print(scores)

        # Optional ReLU on cosine: ignore opposite-direction updates
        if self.fltg_relu_cos:
            scores = torch.clamp(scores, min=0.0)

        # If all scores are ~0 (e.g., consensus vector ~0), fallback
        if float(scores.sum()) <= 1e-8:
            print("[FLTG] Scores collapsed (sum~0). Fallback to weighted mean.")
            return self._weighted_mean_fallback(local_weights, sample_counts)

        # 4) Convert scores -> weights
        T = max(self.fltg_temperature, 1e-6)
        a = torch.softmax(scores / T, dim=0)  # [K]

        # Save last raw scores and normalized weights for external inspection
        try:
            # detach & move to cpu numpy for safe storage
            self._last_scores = scores.detach().cpu().numpy()
            self._last_weights = a.detach().cpu().numpy()
            # Also append to history (list of arrays)
            self.fltg_scores_history.append(self._last_weights.copy())
        except Exception:
            # best-effort, do not break aggregation on failure to record
            logger.exception('Failed to save FLTG scores/weights')

        # 5) Aggregate deltas with weights and update global
        new_state = {}
        for key in global_w.keys():
            stacked = torch.stack([d[key] for d in deltas], dim=0)  # [K, ...]
            view_shape = [len(deltas)] + [1] * (stacked.dim() - 1)
            delta_agg = torch.sum(stacked * a.view(*view_shape).to(stacked.device), dim=0)
            new_state[key] = global_w[key] + delta_agg

        self.global_model.update_weight(new_state)
        print("[FLTG] Global Model Weights Updated (FLTG).")
        print(f"[FLTG] Weights summary: min={float(a.min()):.4f} max={float(a.max()):.4f} entropy~={float((-a*torch.log(a+self.fltg_eps)).sum()):.4f}")

############################################ Proto Federated Learning ############################################

class ProtoFederatedLearning(FederatedLearningModel):
    def __init__(self, federated_model, features, federated_cluster='departement', loss='mse',
                 name='ProtoFederatedModel', dir_log=Path('../'), under_sampling='full', over_sampling='full',
                 target_name='nbsinister', post_process=None, task_type='classification', nbfeatures='all', n_run=1, prototype_weight=1.0,
                 horizon=0):

        super().__init__(federated_model=federated_model, features=features, federated_cluster=federated_cluster, loss=loss,
                         name=name, dir_log=dir_log, under_sampling=under_sampling, over_sampling=over_sampling,
                         target_name=target_name, post_process=post_process, task_type=task_type,
                         aggregation_method='median', nbfeatures=nbfeatures, n_run=n_run, horizon=horizon)

        self.prototype_weight = prototype_weight

        self.horizon = horizon
        self.global_prototypes = {}

    def compute_local_prototypes(self, model):
        prototypes = {}
        counts = {}
        loader = model.train_loader
        model.model.eval()
        with torch.no_grad():
            for data in loader:
                inputs, labels, edges = data
                _, hidden = model.model(inputs, edges)
                try:
                    target, _ = model.compute_weights_and_target(labels, -1, ids_columns, model.model.is_graph_or_node, None)
                except:
                    target, _ = model.compute_weights_and_target(labels, -1, ids_columns, False, None)

                target = target.long()
                for cls in torch.unique(target):
                    cls_idx = int(cls.item())
                    mask = target == cls
                    if mask.sum() == 0:
                        continue
                    
                    vec = hidden[mask].mean(dim=0)
                    if cls_idx in prototypes:
                        prototypes[cls_idx] += vec * mask.sum()
                        counts[cls_idx] += mask.sum()
                    else:
                        prototypes[cls_idx] = vec * mask.sum()
                        counts[cls_idx] = mask.sum()

        for k in prototypes:
            prototypes[k] /= counts[k]
        return prototypes

    def aggregate_prototypes(self, prototypes_list):
        aggregated = {}
        counts = {}
        for proto in prototypes_list:
            for cls, vec in proto.items():
                if cls in aggregated:
                    aggregated[cls] += vec
                    counts[cls] += 1
                else:
                    aggregated[cls] = vec.clone()
                    counts[cls] = 1
        for cls in aggregated:
            aggregated[cls] /= counts[cls]
        self.global_prototypes = aggregated

    def fit(self, df_train, df_val, df_test, graph, args):
        #importance_df = calculate_and_plot_feature_importance(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        #features95, featuresAll = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)

        if self.nbfeatures != 'all':
            self.features_name = featuresAll[:int(self.nbfeatures)]
        else:
            self.features_name = self.features_name

        self.global_model.features_name = self.features_name
        self.global_model.nbfeatures = 'all'
        self.global_model.graph = graph

        initiate_model, model_params = self.global_model.make_model(graph, custom_model_params={'return_hidden' : True})
        self.global_model.model = deepcopy(initiate_model)
        self.global_model.model_params = deepcopy(model_params)
        self.model_params = deepcopy(model_params)
        del initiate_model
        del model_params

        if self.aggregation_method not in ['mean', 'median', 'weighted', 'max']:
            raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' is not implemented.")

        global_epochs = args.get('global_epochs')
        local_epochs = args.get('local_epochs')
        patience_count_global = args.get('patience_count_global')
        patience_count_local = args.get('patience_count_local')

        if self.federated_cluster in np.unique(df_train.columns):
            clusters = df_train[self.federated_cluster].unique()
        else:
            raise NotImplementedError(f"Aggregation method '{self.federated_cluster}' is not implemented.")

        best_global_score = float('-inf')
        patience_counter = 0
        self.local_models = {}

        print(f"\n--- Training Federated Model for {global_epochs} global epochs ---")

        for epoch in range(global_epochs):
            print(f"\n--- Global Epoch {epoch + 1}/{global_epochs} ---")

            local_weights = []
            sample_counts = []
            local_prototypes = []

            for cluster in clusters:
                print(f"\nTraining local model for cluster: {cluster}")

                # Création des datasets pour le cluster fédéré
                df_train_cluster, df_val_cluster, df_test_cluster = self.create_cluster_set(
                    df_train, df_val, df_test, cluster
                )

                if np.all(df_train[self.global_model.target_name].values == 0) or np.all(df_val[self.global_model.target_name].values == 0) or np.all(df_test[self.global_model.target_name].values == 0):
                    print(f'Skipping {cluster} due to no positif samples')
                    continue

                if df_val_cluster.shape[0] == 0 or df_train_cluster.shape[0] == 0:
                    print(f'Skipping {cluster} due to empty dataset')
                    continue
                
                if epoch == 0:
                    local_model = deepcopy(self.global_model)
                else:
                    local_model = self.local_models[cluster]

                local_model.name = f'{self.federated_cluster}_{cluster}_{self.global_model.name}'
                local_model.dir_log = self.global_model.dir_log / local_model.name
                if epoch == 0:
                    check_and_create_path(local_model.dir_log)

                local_model.features_name = self.features_name
                local_model.nbfeatures = 'all'
                local_model.use_prototypes = len(self.global_prototypes) > 0
                local_model.prototype_weight = self.prototype_weight
                local_model.prototypes = self.global_prototypes
                
                if epochs == 0:
                    local_model.create_train_val_test_loader(graph, df_train_cluster, df_val_cluster, df_test_cluster, local_epochs, patience_count_local, CHECKPOINT, False, custom_model_params={'return_hidden' : True})
                    self.metrics[cluster] = local_model.metrics

                local_model.train(graph, patience_count_local, CHECKPOINT, local_epochs, verbose=False, custom_model_params=None, new_model=False)
                
                self.local_models[cluster] = local_model
                local_weights.append(deepcopy(local_model.model.state_dict()))
                sample_counts.append(len(df_train_cluster))
                local_prototypes.append(self.compute_local_prototypes(local_model))

            self.aggregate_prototypes(local_prototypes)

            global_score = self.score(df_val, df_val[self.target_name])
            print(f"\nGlobal Model Score after epoch {epoch + 1}: {global_score:.4f}")

            if global_score > best_global_score:
                best_global_score = global_score
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience_count_global:
                print("\nEarly stopping: Global model score did not improve.")
                break

        self.is_fitted_ = True
        print("\n--- Federated Learning Training Complete ---")

    def create_cluster_set(self, df_train, df_val, df_test, cluster):
        """
        Create training, validation, and test datasets for a given cluster.
        """
        if self.federated_cluster in np.unique(df_train.columns):
            X_cluster = df_train[df_train[self.federated_cluster] == cluster].reset_index(drop=True)
            X_val_cluster = df_val[df_val[self.federated_cluster] == cluster].reset_index(drop=True)
            X_test_cluster = df_test[df_test[self.federated_cluster] == cluster].reset_index(drop=True)

            return X_cluster, X_val_cluster, X_test_cluster
        
        raise NotImplementedError(f"Federated clustering method '{self.federated_cluster}' is not implemented.")

    def score_with_prediction(self, y_pred, y, sample_weight=None):
        
        return iou_score(y, y_pred)
    
    def score(self, X, y, sample_weight=None):
        """
        Evaluate the model's performance for each ID.

        Parameters:
        - X_val: Validation data.
        - y_val: True labels.
        - id_val: List of IDs corresponding to validation data.

        Returns:
        - Mean score across all IDs.
        """
        predictions, y = self.predict(X, return_y=True)
        y = y[:, -1]
        return self.score_with_prediction(predictions, y, sample_weight)
    
    def predict(self, df, graph=None, return_y=False):
        """
        Prédit les classes via le modèle local et les prototypes globaux.
        """
        if graph is None:
            graph = self.global_model.graph

        if self.target_name not in df.columns:
            df = df.copy()
            df[self.target_name] = 0

        if self.federated_cluster not in np.unique(df.columns):
            logger.info(f'{self.federated_cluster} must be in dataframe for inference')
            exit(1)

        final_pred = []
        ys = []
        clusters_values = df[self.federated_cluster].values
        for cluster in np.unique(clusters_values):

            loader = self.create_test_loader(graph, df[df[self.federated_cluster] == cluster])

            pred, y = self._predict_test_loader(loader, prediction_type='Class', cluster=cluster)
            
            final_pred.append(pred)
            ys.append(y)

        final_pred = torch.concatenate(final_pred, dim=0)
        ys = torch.concatenate(ys, dim=0)

        return (final_pred, ys) if return_y else pred

    def predict_proba(self, df, graph=None, return_y=False):
        """
        Renvoie les probabilités de classe via softmax inversé sur les distances aux prototypes.
        """
        if graph is None:
            graph = self.global_model.graph
            
        if self.target_name not in df.columns:
            df = df.copy()
            df[self.target_name] = 0

        final_pred = []
        clusters_values = df[self.federated_cluster].values
        ys = []
        for cluster in np.unique(clusters_values):
            loader = self.create_test_loader(graph, df[df[self.federated_cluster] == cluster])

            pred, y = self._predict_test_loader(loader, prediction_type='Probability', cluster=cluster)

            final_pred.append(pred)
            ys.append(y)
        
        final_pred = torch.concatenate(final_pred, dim=0)
        ys = torch.concatenate(ys, dim=0)

        return (final_pred, ys) if return_y else pred

    def _predict_test_loader(self, loader, prediction_type='Class', cluster = None):
        """
        Effectue la prédiction sur un DataLoader en utilisant le modèle local
        + les prototypes globaux (style FedProto).
        """
        if cluster is None:
            logger.info(f'Need to provide the correspond cluster of the batch ({self.federated_cluster})')
        assert cluster is not None
        if cluster in self.local_models.keys():
            model = self.local_models[cluster].model
            model.eval()
        else:
            model = None
        preds = []
        targets = []
        
        with torch.no_grad():
            for data in loader:

                inputs, y, _ = data
                y = y[:, :, -1]

                batch_preds = []
                
                # Obtenir l'embedding via le modèle local
                if model is not None:
                    _, embeddings = model(inputs)
                else:
                    embeddings = torch.zeros(inputs.shape[0])

                # Comparer aux prototypes globaux
                for emb in embeddings:
                    distances = {
                        cls: torch.norm(emb - proto.to(embeddings.device))
                        for cls, proto in self.global_prototypes.items()
                    }
                    if prediction_type == 'Probability':
                        # Convertir les distances en "scores inversés"
                        inv = torch.tensor(
                            [-d.item() for d in distances.values()]
                        )
                        probs = torch.softmax(inv, dim=0)
                        batch_preds.append(probs)
                    else:
                        pred_class = min(distances, key=distances.get)
                        batch_preds.append(pred_class)

                preds.extend(batch_preds)
                targets.extend(y.cpu().tolist())

        if prediction_type == 'Probability':
            preds = torch.stack(preds)  # [N, num_classes]
        else:
            preds = torch.tensor(preds)

        return preds, torch.tensor(targets)