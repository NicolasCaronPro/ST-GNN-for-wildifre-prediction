from GNN.pytorch_model_tools import *

import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader

class ModelKnowledgeDistillation(Training):
    def __init__(self, temperature, alpha, distillation_training_mode, teacher_name, student_name, model_name, batch_size, lr, delta_lr, patience_cnt_lr, out_channels, dir_log, features_name, ks, loss, name, device,
                under_sampling, over_sampling, nbfeatures, weight_type, target_name, task_type, teacher_loss, beta=None, gamma=None, horizon=0, loss_param_search=False):

        super().__init__(f'{model_name}', nbfeatures, batch_size, lr, delta_lr, patience_cnt_lr, target_name, task_type, features_name, ks, \
        out_channels, dir_log, loss=loss, name=name, device=device, under_sampling=under_sampling, over_sampling=over_sampling, horizon=horizon, loss_param_search=loss_param_search)
        
        self.teacher_loss = teacher_loss
        self.distillation_training_mode = distillation_training_mode
        self.teacher_name = teacher_name
        self.student_name = student_name
        self.weight_type = weight_type
        self.temperature = temperature
        self.alpha = alpha
        self.temperature_value = float(temperature) if temperature != 'search' else None
        self.alpha_value = float(alpha) if alpha != 'search' else None
        self.beta_value = float(beta) if beta is not None and beta != 'search' else None
        self.gamma_value = float(gamma) if gamma is not None and gamma != 'search' else None
        self.student_train = True
        self.load_teacher = True
        self.distillation_log = []

        if 'group' in self.distillation_training_mode:
            self.model_list = []

        if self.distillation_training_mode not in ['normal', 'allTeacher', 'RelationMLP', 'RelationAtt', 'RelationATT', 'AdaptativeMLP', 'Confidence', 'MATTKD']:
            raise ValueError(f'{self.distillation_training_mode} unknow')

        self.horizon = int(horizon)

    def create_train_val_test_loader(self, graph, df_train, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT, features_importance=True, custom_model_params=None, use_log=True):
        self.graph = graph

        #if self.teacher_name in sklearn_model_list or 'xgboost' in self.target_name:
        #    print(self.dir_log / '..' / 'baseline' / self.teacher_name)
        #    self.teacher = read_object(f'{self.teacher_name}.pkl', self.dir_log / '..' / 'baseline' / self.teacher_name)
        #else:
        #    self.teacher = read_object(f'{self.teacher_name}', self.dir_log / self.teacher_name)
        
        check_and_create_path(self.dir_log)

        if self.load_teacher:
            filter_name, model_type, hard_or_soft, weights_average, top_model = self.teacher_name.split('-')
            self.hard_or_soft = hard_or_soft
            self.weights_average = weights_average
            self.top_model = top_model
            full_teacher_name = f'{filter_name}-{model_type}_search_{self.over_sampling}_{self.ks}_{self.horizon}_{self.nbfeatures}_{self.weight_type}_{self.target_name}_{self.task_type}_{self.teacher_loss}'
            print(full_teacher_name)
            if model_type in ['catboost', 'xgboost']:
                self.teacher = read_object(f'{full_teacher_name}.pkl', self.dir_log / '..' / 'baseline' / full_teacher_name)
            else:
                self.teacher = read_object(f'{full_teacher_name}.pkl', self.dir_log / '..' / full_teacher_name)
            assert self.teacher is not None
            try:
                self.teacher.clean()
            except:
                pass
            self.load_teacher = False

                # Initialize RelationMLP for RelationMLP distillation mode
        if self.distillation_training_mode == 'RelationMLP':
            assert self.beta_value is not None, "beta parameter must be set for RelationMLP distillation mode"
            
            # Get number of teachers and classes
            models_to_mean, _, _ = self.teacher.get_weights(self.top_model, return_self_model_idx=True)
            num_teachers = len(models_to_mean) - 1
            num_classes = self.out_channels  # Number of output classes
            
            # Create RelationMLP to learn ensemble embedding from teacher logits
            self.relation_mlp = RelationMLP(
                num_teachers=num_teachers,
                num_classes=64,
                embedding_dim=64,  # Output dimension matches student logits
                mlp_hidden_dim=128
            ).to(self.device)
            
            logger.info(f'Initialized RelationMLP with {num_teachers} teachers, {num_classes} classes')

        # Initialize RelationAttention for RelationATT distillation mode
        if self.distillation_training_mode == 'RelationATT':
            assert self.beta_value is not None, "beta parameter must be set for RelationATT distillation mode"
            
            # Get number of teachers and classes
            models_to_mean, _, _ = self.teacher.get_weights(self.top_model, return_self_model_idx=True)
            num_teachers = len(models_to_mean) - 1
            num_classes = self.out_channels  # Number of output classes
            
            # Create RelationAttention to learn ensemble embedding from teacher logits
            self.relation_att = RelationAttention(
                num_teachers=num_teachers,
                num_classes=64,
                embedding_dim=64,  # Output dimension matches student logits
                hidden_dim=128,
                num_heads=4
            ).to(self.device)
            
            logger.info(f'Initialized RelationAttention with {num_teachers} teachers, {num_classes} classes')

        if self.distillation_training_mode == 'AdaptativeMLP':
            assert self.gamma_value is not None
            # Initialize Adapter
            models_to_mean, _, _ = self.teacher.get_weights(self.top_model, return_self_model_idx=True)
            #num_teachers = len(models_to_mean) - 1
            num_teachers = len(models_to_mean)
            self.adapter = Adapter(d=64, num_teachers=num_teachers).to(self.device)

            # Initialize FitNets
            self.fitnets = torch.nn.ModuleList([
                FitNet(c_student=128, c_teacher=64).to(self.device)
                for _ in range(num_teachers)
            ])

        if self.distillation_training_mode == 'Confidence':
            # Initialize FitNets for Confidence mode
            # FitNet projects student features (128) to teacher features (assumed 64 or inferred)
            # BUT for Confidence mode, we need to project to teacher logits dimension (out_channels) 
            # to calculate CE with labels.
            # Wait, the user code says:
            # L_inter : feature distillation with weight w_inter
            # w_inter comes from CE(teacher_classifier(fitnet(student_feat)), labels)
            # MSE is between teacher_feat and fitnet(student_feat)
            
            models_to_mean, _, _ = self.teacher.get_weights(self.top_model, return_self_model_idx=True)
            num_teachers = len(models_to_mean) - 1
            
            # We assume student feature dim is 128 (from StudentMLP)
            # We need to know teacher feature dim.
            # Let's assume 64 as in AdaptativeMLP for now, or try to infer if possible.
            # In AdaptativeMLP it was hardcoded c_teacher=64.
            
            self.fitnets = torch.nn.ModuleList([
                FitNet(c_student=64, c_teacher=64).to(self.device)
                for _ in range(num_teachers)
            ])

        elif self.distillation_training_mode == 'MATTKD':

            models_to_mean, _, _ = self.teacher.get_weights(self.top_model, return_self_model_idx=True)
            num_teachers = len(models_to_mean) - 1

            # Initialize FitNets
            self.fitnets = torch.nn.ModuleList([
                FitNet(c_student=128, c_teacher=64).to(self.device)
                for _ in range(num_teachers)
            ])

            self.relation_att = RelationAttention(
                num_teachers=num_teachers,
                num_classes=self.out_channels,
                embedding_dim=64,  # Output dimension matches student logits
                hidden_dim=128,
                num_heads=4
            ).to(self.device)
            
        if self.distillation_training_mode in ['normal', 'allTeacher', 'RelationMLP', 'RelationAtt', 'RelationATT', 'AdaptativeMLP', 'Confidence', 'MATTKD']:
            if self.under_sampling != 'full':
                old_shape = df_train.shape
                y = df_train[self.target_name]
                if 'binary' in self.under_sampling:
                    vec = self.under_sampling.split('-')
                    try:
                        nb = int(vec[-1]) * len(df_train[df_train[self.target_name] > 0])
                    except:
                        logger.info(f'{self.under_sampling} with undefined factor, set to 1 -> {len(df_train[df_train[self.target_name] > 0])}')
                        nb = len(df_train[df_train[self.target_name] > 0])

                    df_combined = self.split_dataset(df_train, nb)

                    # Mettre à jour df_train pour l'entraînement
                    df_train = df_combined

                    logger.info(f'Train mask df_train shape: {old_shape} -> {df_train.shape}')

                elif self.under_sampling == 'search' or 'percentage' in self.under_sampling:
                        if self.under_sampling == 'search':
                            best_tp, find_log = self.search_samples_proportion(graph, df_train, df_val, df_test, is_unknowed_risk=False,
                                                                               epochs=epochs, PATIENCE_CNT=PATIENCE_CNT, CHECKPOINT=CHECKPOINT,
                                                                               custom_model_params=custom_model_params, use_log=use_log)
                            self.find_log = find_log
                        else:
                            vec = self.under_sampling.split('-')
                            try:
                                best_tp = float(vec[-1])
                            except ValueError:
                                logger.info(f'{self.under_sampling} with undefined factor, set to 0.3 -> {0.3 * len(y[y == 0])}')
                                best_tp = 0.3

                        nb = int(best_tp * len(y[y == 0]))

                        df_combined = self.split_dataset(df_train, nb)
                        df_train = df_combined
                        logger.info(f'Train mask df_train shape: {old_shape} -> {df_train.shape}')

        self.df_train = df_train
        self.df_test = df_test
        self.df_val = df_val

        ##################################### Create loader #########################################
        if False:
            train_dataset = read_object('train_dataset.pkl', self.dir_log)
            val_dataset = read_object('val_dataset.pkl', self.dir_log)
            test_dataset = read_object('test_dataset.pkl', self.dir_log)
        else:
            train_dataset, val_dataset, test_dataset = create_dataset(graph,
                                                                self.df_train,
                                                                df_val,
                                                                df_test,
                                                                self.features_name,
                                                                self.target_name,
                                                                None,
                                                                self.device, self.ks, self.horizon)

            #save_object_torch(train_dataset, 'train_dataset.pkl', self.dir_log)
            #save_object_torch(val_dataset, 'val_dataset.pkl', self.dir_log)
            #save_object_torch(test_dataset, 'test_dataset.pkl', self.dir_log)

            train_loader = DataLoader(train_dataset, batch_size, True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(val_dataset, val_dataset.__len__(), False, worker_init_fn=seed_worker, generator=g)
            test_loader = DataLoader(test_dataset, test_dataset.__len__(), False, worker_init_fn=seed_worker, generator=g)

            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader

    def create_train_val_test_loader_teacher(self, graph, df_train, df_val, df_test, teacher, features_importance=True):
        self.graph = graph

        #if self.teacher_name in sklearn_model_list or 'xgboost' in self.target_name:
        #    print(self.dir_log / '..' / 'baseline' / self.teacher_name)
        #    self.teacher = read_object(f'{self.teacher_name}.pkl', self.dir_log / '..' / 'baseline' / self.teacher_name)
        #else:
        #    self.teacher = read_object(f'{self.teacher_name}', self.dir_log / self.teacher_name)
        
        check_and_create_path(self.dir_log)
        print(teacher.dir_log)

        percentage = read_object('test_percentage_scores.pkl', teacher.dir_log)
        test_percentage, under_prediction_score_scores, over_prediction_score_scores = percentage[0], percentage[1], percentage[2]

        score_differences = np.array(under_prediction_score_scores) - np.array(over_prediction_score_scores)
        index_max = np.argmin(np.abs(score_differences))
        best_tp = test_percentage[index_max]
 
        nb = int(best_tp * len(df_train[df_train[teacher.target_name] == 0]))

        df_combined = self.split_dataset(df_train, nb)
        
        self.df_train = df_combined
        self.df_test = df_test
        self.df_val = df_val

        ##################################### Create loader #########################################
        if False:
            train_dataset = read_object('train_dataset.pkl', self.dir_log)
            val_dataset = read_object('val_dataset.pkl', self.dir_log)
            test_dataset = read_object('test_dataset.pkl', self.dir_log)
        else:
            train_dataset, val_dataset, test_dataset = create_dataset(graph,
                                                                self.df_train,
                                                                df_val,
                                                                df_test,
                                                                self.features_name,
                                                                self.target_name,
                                                                None,
                                                                self.device, self.ks, self.horizon)

            #save_object_torch(train_dataset, 'train_dataset.pkl', self.dir_log)
            #save_object_torch(val_dataset, 'val_dataset.pkl', self.dir_log)
            #save_object_torch(test_dataset, 'test_dataset.pkl', self.dir_log)

            train_loader = DataLoader(train_dataset, batch_size, True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(val_dataset, val_dataset.__len__(), False, worker_init_fn=seed_worker, generator=g)
            test_loader = DataLoader(test_dataset, test_dataset.__len__(), False, worker_init_fn=seed_worker, generator=g)

            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader

    def create_test_loader(self, graph, df):
        loader = create_test_loader(graph, df,
                       self.features_name,
                       self.device,
                       None,
                       self.target_name,
                       self.ks,
                       horizon=self.horizon)

        return loader
    
    def search_temperature_alpha(self, graph, PATIENCE_CNT, CHECKPOINT, epoch, verbose=True, custom_model_params=None, new_model=True):
        """temperature_grid = [4.0, 6.0] if self.temperature == 'search' else [self.temperature_value]
        alpha_grid = [.0, .1, .2, .3, .4, .5,] if self.alpha == 'search' else [self.alpha_value]
        
        if False and (self.dir_log / 'log_parameters.pkl').is_file():
            parameters = read_object('log_parameters.pkl', self.dir_log)
            self.temperature_value = parameters['temperature']
            self.alpha_value = parameters['alpha']
        else:
            score_0 = 0
            patience_i = 0
            patience_c = 5
            top_temp = 0
            top_al = 0
            for temp in temperature_grid:
                
                for al in alpha_grid:
                    
                    logger.info(f'########### temperature {temp}, alpha {al} #############')

                    self.temperature_value = temp
                    self.alpha_value = al
                    self.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose, custom_model_params, new_model, search=False)
                    score = self.score(self.df_val, self.df_val[self.target_name])
                    if score > score_0:
                        logger.info(f'temperature {temp}, alpha {al} -> {score}')
                        score_0 = score
                        top_temp = temp
                        top_al = al
                    else:
                        patience_i += 1
                        if patience_i == patience_c:
                            break
                            
            parameters = {'temperature' : top_temp, 'alpha' : top_al}
            self.alpha_value = top_al
            self.temperature_value = top_temp
            save_object(parameters, 'log_parameters.pkl', self.dir_log)"""
        
        self.temperature_value = torch.nn.Parameter(torch.tensor(3.0))
        self.alpha_value = torch.nn.Parameter(torch.tensor(0.2))
            
    def train(self, graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose=True, custom_model_params=None, new_model=True, search=True):

        if (self.temperature == 'search' or self.alpha == 'search') and search:
            self.search_temperature_alpha(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose, custom_model_params, new_model)
            
        if self.distillation_training_mode == 'iterate':
            sub_teachers = np.asarray([estimator for estimator in self.teacher.best_estimator_])
            weights2use = self.teacher.weights_for_model
            weights2use = np.asarray(weights2use)
            key = np.argsort(weights2use)
            key = np.flip(key)
            sub_teachers = sub_teachers[np.asarray(key)]
            score_0 = 0
            
            for i, sub_teacher in enumerate(sub_teachers):
                
                if i > 0:
                    model_params_log = self.model.state_dict()
                    new_model = False
                else:
                    new_model = True

                self.create_train_val_test_loader_teacher(graph, self.df_train, self.df_val, self.df_test, sub_teacher, features_importance=False)

                super().train(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose, custom_model_params, new_model)

                test_output, y = self._predict_test_loader(self.test_loader)
                test_output = test_output.detach().cpu().numpy()

                y = y.detach().cpu().numpy()

                plt.figure(figsize=(15,5))
                plt.plot(y[y[:, departement_index] == 1, -1])
                plt.plot(test_output[y[:, departement_index] == 1])
                plt.savefig(self.dir_log / 'test.png')

                score = self.score(self.df_test, self.df_test[self.target_name])
                logger.info(f'Teacher : {sub_teacher.name} -> {score}')
                teacher_score = sub_teacher.score(self.df_test, self.df_test[self.target_name])
                logger.info(f'Teacher score -> {teacher_score}')
                if score > score_0:
                    score_0 = score
                else:
                    break
                    self.update_weight(model_params_log)

        elif self.distillation_training_mode in ['normal', 'allTeacher', 'RelationMLP', 'RelationATT', 'AdaptativeMLP', 'Confidence', 'MATTKD', 'Confidence']:
            super().train(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose, custom_model_params, new_model)

        elif 'group' in self.distillation_training_mode:
            nbgroup = int(self.distillation_training_mode.split('-')[1])
            if nbgroup == 'search':
                score_0 = 0
                for sub_teacher in self.teacher.best_estimator_:
                    
                    model_params_log = self.model.model.state_dict()
                    super().train(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose, custom_model_params, new_model)

                    score = self.score(self.df_test, self.df_test[self.target_name])
                    if score > score_0:
                        score_0 = score
                    else:
                        self.update_weight(model_params_log)
                        self.model_list.append(deepcopy(self.model))
                        score_0 = 0
            else:
                nbgroup = int(nbgroup)
                score_0 = 0
                current_group = 0
                for sub_teacher in self.teacher.best_estimator_:
                    
                    model_params_log = self.model.model.state_dict()
                    super().train(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose, custom_model_params, new_model)
                    
                    score = self.score(self.df_test, self.df_test[self.target_name])
                    if score > score_0:
                        score_0 = score
                    else:
                        self.update_weight(model_params_log)
                        self.model_list.append(deepcopy(self.model))
                        score_0 = 0
                        current_group += 1
                        if current_group > nbgroup:
                            break
            
    def get_temperature_alpha(self):
        return self.temperature_value, self.alpha_value

    def _save_temperature_alpha_plot(self):

        # Extraction directe (car distillation_log est un dict {epoch: {"kappa":..., "xi":...}})
        
        print(self.distillation_log)
        kappas = [distillation_log["temperature"] for distillation_log in self.distillation_log]
        xis = [distillation_log["xi"] for distillation_log in self.distillation_log]
        epochs = [distillation_log["epoch"] for distillation_log in self.distillation_log]

        # Sauvegarde pickle
        dist_to_save = {"epoch": epochs, "temperature": kappas, "xi": xis}
        save_object(dist_to_save, 'egpd_kappa_xi.pkl', self.dir_log)

        # temperature vs epoch
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, kappas, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('temperature')
        plt.title('temperature over epochs')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(self.dir_log / 'temperature_over_epochs.png')
        plt.close()

        # alpha vs epoch
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, xis, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('alpha')
        plt.title('alpha over epochs')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(self.dir_log / 'alpha_over_epochs.png')
        plt.close()

############################################ VOTING MODEL ##############################################################

class ModelVotingPytorchAndSklearn(RegressorMixin, ClassifierMixin):
    def __init__(self, models, features, loss='mse', name='ModelVoting', dir_log=Path('../'), under_sampling='full', target_name='nbsinister', post_process=None, task_type='classification', horizon=0):
        """
        Initialize the ModelVoting class.

        Parameters:
        - models: A list of base models to use (must follow the sklearn API).
        - name: The name of the model.
        - loss: Loss function to use ('logloss', 'hinge_loss', 'mse', 'rmse', etc.).
        """
        super().__init__()
        self.best_estimator_ = models  # Now a list of models
        self.feature_names = features
        self.name = name
        self.loss = loss
        self.is_fitted_ = [False] * len(models)  # Keep track of fitted models
        self.features_per_model = []
        self.dir_log = dir_log
        self.post_process = post_process
        self.under_sampling = under_sampling
        self.target_name = target_name
        self.task_type = task_type
        self.horizon = horizon

    def fit(self, X, y, X_val, y_val, X_test, y_test, args, use_log=True):
        """
        Train each model on the corresponding data.

        Parameters:
        - X_list: List of training data for each model.
        - y_list: List of labels for the training data for each model.
        - optimization: Optimization method to use ('grid' or 'bayes').
        - grid_params_list: List of parameters to optimize for each model.
        - fit_params_list: List of additional parameters for the fit function for each model.
        - cv_folds: Number of cross-validation folds.
        """

        ######## Pytorch params
        PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params_list = args['PATIENCE_CNT'], args['CHECKPOINT'], args['epochs'], args['custom_model_params_list']
        graph = args['graph']
        training_mode = args['training_mode']
        optimization = args['optimization']
        grid_params_list = args['grid_params_list']
        fit_params_list = args['fit_params_list']
        cv_folds = args['cv_folds']
        
        df_val = X_val.copy(deep=True)
        df_val[y_val.columns] = y_val

        self.cv_results_ = []
        self.is_fitted_ = [True] * len(self.best_estimator_)
        self.weights_for_model = []
        for i, model in enumerate(self.best_estimator_):
            model.dir_log = self.dir_log / '..' / model.name
            print(f'Fitting model -> {model.name}')

            if (model.dir_log / f'best.pt').is_file():
                model.graph = graph
                model._load_model_from_path(model.dir_log / 'best.pt', model.model)
                print(f'Loading model -> {model.name}')
                continue

            #if issubclass(model, Model_Torch):
            model.fit(graph, X, y, X_val, y_val, X_test, y_test, PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params=None, use_log=use_log)
            #else:
            #    model.fit(graph, X, y[targets[i]], X_val, y_val[targets[i]], training_mode=training_mode, optimization=optimization, grid_params=grid_params_list[i], fit_params=fit_params_list[i], cv_folds=cv_folds)
            target_name_model = model.target_name
            model.target_name = self.target_name
            print(f'Change target name {target_name_model} to {model.target_name}')
            test_loader = model.create_test_loader(graph, df_val)
            test_output, y_test_val = model._predict_test_loader(test_loader)
            test_output = test_output.detach().cpu().numpy()

            test_output = test_output[:, 0]
            y_test_val = y_test_val[:, :, 0]
            
            y_test_val = y_test_val.detach().cpu().numpy()[:, -1]
                
            model.target_name = target_name_model

            score_model = self.score_with_prediction(y_test_val, test_output)
            self.weights_for_model.append(score_model)

        self.weights_for_model = np.asarray(self.weights_for_model)
        # Affichage des poids et des modèles
        print("\n--- Final Model Weights ---")
        for model, weight in zip(self.best_estimator_, self.weights_for_model):
            print(f"Model: {model.name}, Weight: {weight:.4f}")

        self.plot_weights_by_target()

    def predict_nbsinister(self, X, ids=None, preprocessor_ids=None, hard_or_soft='soft', weights_average=True):
        
        if self.target_name == 'nbsinister':
            return self.predict(X)
        else:
            assert self.post_process is not None
            predict = self.predict(X)
            return self.post_process.predict_nbsinister(predict, ids)
    
    def predict_risk(self, X, ids=None, preprocessor_ids=None, hard_or_soft='soft', weights_average=True):

        if self.task_type == 'classification':
            return self.predict(X)
        
        elif self.task_type == 'binary':
                assert self.post_process is not None
                predict = self.predict_proba(X, return_y=False)[:, 1]

                if isinstance(ids, pd.Series):
                    ids = ids.values
                if isinstance(preprocessor_ids, pd.Series):
                    preprocessor_ids = preprocessor_ids.values
    
                return self.post_process.predict_risk(predict, None, ids, preprocessor_ids)
        else:
            assert self.post_process is not None
            predict = self.predict(X)

            if isinstance(ids, pd.Series):
                ids = ids.values
            if isinstance(preprocessor_ids, pd.Series):
                preprocessor_ids = preprocessor_ids.values

            return self.post_process.predict_risk(predict, None, ids, preprocessor_ids)

    def predict_with_weight(self, X, hard_or_soft='soft', weights_average='weight', weights2use=[], top_model='all', prediction_type="Class", aggregation=True):
        
        models_list = np.asarray([estimator.name for estimator in self.best_estimator_])
        weights2use = np.asarray(weights2use)
        
        if hard_or_soft == 'hard':
            if top_model != 'all':
                top_model = int(top_model)
                key = np.argsort(weights2use)
                models_list = models_list[key]
                models_list = models_list[-top_model:]
                #weights2use = weights2use[np.asarray(key)]
                #weights2use = weights2use[-top_model:]
            else:
                key = np.arange(0, len(self.best_estimator_))
                
            models_to_mean = []
            predictions = []
            for i, estimator in enumerate(self.best_estimator_):
                if estimator.target_name == self.target_name:
                    pred, y = estimator.predict(X, return_y=True, prediction_type=prediction_type)
                if estimator.name not in models_list:
                    continue
                else:
                    if estimator.target_name != self.target_name:
                        pred = estimator.predict(X, return_y=False, prediction_type=prediction_type)

                    predictions.append(pred.detach().cpu().numpy())

                models_to_mean.append(key[i])

            try:
                weights2use = weights2use[models_to_mean]
            except:
                pass
            # Aggregate predictions
            if aggregation:
                aggregated_pred = self.aggregate_predictions(predictions, models_to_mean, weights2use)
            else:
                aggregated_pred = predictions
            return aggregated_pred, y.detach().cpu().numpy()
        elif hard_or_soft == 'None':
            top_model = int(top_model)
            key = np.argsort(weights2use)
            idx = key[-top_model]
            estimator = self.best_estimator_[idx]
            if estimator.target_name == self.target_name:
                pred, y = estimator.predict(X, return_y=True, prediction_type=prediction_type)
                return pred.detach().cpu().numpy(), y.detach().cpu().numpy(), y
            else:
                pred = estimator.predict(X, return_y=False)
                y = None
                for estimator in self.best_estimator_:
                    if estimator.target_name == self.target_name:
                        _, y = estimator.predict(X, return_y=True, prediction_type=prediction_type)
                        return pred.detach().cpu().numpy(), y.detach().cpu().numpy(),
        else:
            aggregated_pred, y = self.predict_proba_with_weights(X, weights_average=weights_average, top_model=top_model,
                                                                 weights2use=weights2use, prediction_type=prediction_type,
                                                                 aggregation=aggregation)
            if prediction_type == 'Class':
                predictions = np.argmax(aggregated_pred, axis=-1)
            else:
                predictions = aggregated_pred
            
            return predictions, y
        
    def get_weights(self, top_model, return_self_model_idx=False, name=None):
        weights = np.asarray(self.weights_for_model)
        n_models = len(weights)

        # Liste des modèles
        estimators = list(self.best_estimator_)
        names = np.asarray([est.name for est in estimators])

        # Indices triés par poids (croissant)
        order = np.argsort(weights)

        if top_model != 'all':
            k = int(top_model)
            k = max(1, min(k, n_models))  # clamp entre 1 et n_models
            # indices des top-k modèles (les plus gros poids)
            selected_idx = order[-k:]
        else:
            selected_idx = np.arange(n_models)

        # On repère l'indice (dans l'ensemble global) du modèle "self"
        self_idx = math.inf
        for i, idx in enumerate(selected_idx):
            if estimators[idx].name == name:
                self_idx = i
                break

        if return_self_model_idx:
            # on renvoie: indices sélectionnés, leurs poids, et l'indice du modèle "self"
            return selected_idx.tolist(), weights[selected_idx], self_idx
        else:
            return selected_idx.tolist(), weights[selected_idx], self_idx

    def predict_proba_with_weights(self, X, hard_or_soft='soft', weights_average='weight', top_model='all', weights2use=[], id_col=(None, None), prediction_type='Proba', aggregation=True):
        """
        Predict probabilities for input data using each model and aggregate the results.

        Parameters:
        - X_list: List of data to predict probabilities for.
        
        Returns:
        - Aggregated predicted probabilities.
        """
        models_list = np.asarray([estimator.name for estimator in self.best_estimator_])
        weights2use = np.asarray(weights2use)

        if top_model != 'all':
                top_model = int(top_model)
                key = np.argsort(weights2use)
                models_list = models_list[np.asarray(key)]
                models_list = models_list[-top_model:]
                #weights2use = weights2use[np.asarray(key)]
                #weights2use = weights2use[-top_model:]
        else:
            key = np.arange(0, len(self.best_estimator_))
        
        probas = []
        models_to_mean = []

        if hard_or_soft == 'None':
            top_model = int(top_model)
            idx = np.argsort(weights2use)[-top_model]
            estimator = self.best_estimator_[idx]
            if estimator.target_name == self.target_name:
                pred, y = estimator.predict_proba(X, return_y=True, prediction_type=prediction_type)
                return pred.detach().cpu().numpy(), y.detach().cpu().numpy()
            else:
                pred = estimator.predict(X, return_y=False)
                y = None
                for estimator in self.best_estimator_:
                    if estimator.target_name == self.target_name:
                        _, y = estimator.predict_proba(X, return_y=True, prediction_type=prediction_type)
                        return pred.detach().cpu().numpy(), y.detach().cpu().numpy()

        for i, estimator in enumerate(self.best_estimator_):
            X_ = X
            if estimator.target_name == self.target_name:
                proba, y = estimator.predict_proba(X, return_y=True, prediction_type=prediction_type)
            if estimator.name not in models_list:
                continue
            else:
                if estimator.target_name != self.target_name:
                    proba = estimator.predict_proba(X_, return_y=False, prediction_type=prediction_type)
            if proba.shape[-1] != 5:
                continue
            #print(estimator.name, np.asarray(probas).shape)
            models_to_mean.append(key[i])
            probas.append(proba)
        try:
            weights2use = weights2use[models_to_mean]
        except:
            pass
        # Aggregate probabilities
        if aggregation:
            aggregated_proba = self.aggregate_probabilities(probas, models_to_mean, weights2use)
            return aggregated_proba, y
        else:
            return probas, y
        
    def predict_with_tasks(
        self,
        X,
        hard_or_soft="soft",
        weights_average="weight",
        model_per_task=None,
        generalized_departement=None,
        id_col=(None, None),
        prediction_type='Class',
        aggregation=True
    ):
        """Predict with a specific ``top_model`` per task.

        Parameters
        ----------
        X : pandas.DataFrame
            Input dataframe.
        hard_or_soft : str
            Mode passed to :func:`predict_with_weight`.
        weights_average : str
            Weighting mode for aggregation.
        model_per_task : dict
            Mapping of task names to number of models to use.
        generalized_departement : list
            Departments for which to apply ``generalized_prediction`` task.
        id_col : tuple
            Id column information for weighted predictions.
        """

        if model_per_task is None:
            model_per_task = {}
        
        if model_per_task == 'default':
            model_per_task={'normal_predictions' : 4,
                                'generalized_prediction' : 12,
                                'class_value_2_predictions' : 12,
                                'class_value_3_predictions' : 20,
                                'class_value_4_predictions' : 20
                                }
            
            generalized_departement = [
                1.,  2.,  3.,  4.,  5.,  8.,  9., 10., 12., 14.,
                15., 16., 17., 18., 19., 21., 22., 23., 24., 25.,
                26., 27., 28., 29., 31., 32., 35., 36., 37., 38.,
                39., 41., 42., 43., 44., 45., 46., 47., 48., 49.,
                50., 51., 52., 53., 54., 55., 56., 57., 58., 59.,
                60., 61., 62., 63., 64., 65., 67., 68., 69., 70.,
                71., 72., 73., 74., 75., 76., 77., 78., 79., 80.,
                81., 82., 85., 86., 87., 88., 89., 90., 91., 92.,
                93., 94., 95.
            ]
        

        # Normal prediction for all samples
        top_model = model_per_task.get("normal_predictions", "all")
        if prediction_type == 'Class' or prediction_type == 'RawFormulaVal':
            predictions, y = self.predict_with_weight(
                X,
                hard_or_soft=hard_or_soft,
                weights_average=weights_average,
                weights2use=self.weights_for_model,
                top_model=top_model,
                prediction_type=prediction_type,
                aggregation=aggregation
            )
        else:
            predictions, y = self.predict_proba_with_weights(
                X,
                hard_or_soft=hard_or_soft,
                weights_average=weights_average,
                weights2use=self.weights_for_model,
                top_model=top_model,
                prediction_type=prediction_type,
                aggregation=aggregation
            )

        predictions = np.asarray(predictions)

        # Generalized prediction
        if (
            model_per_task.get("generalized_prediction") is not None
            and generalized_departement is not None
            and "departement" in X.columns
        ):
            mask = X["departement"].isin(generalized_departement).values
            if mask.any():
                if prediction_type == 'Class' or prediction_type == 'RawFormulaVal':
                    preds_gen, _  = self.predict_with_weight(
                        X[mask],
                        hard_or_soft=hard_or_soft,
                        weights_average=weights_average,
                        weights2use=self.weights_for_model,
                        top_model=model_per_task["generalized_prediction"],
                        prediction_type=prediction_type,
                        aggregation=aggregation   
                    )
                else:
                    preds_gen, _  = self.predict_proba_with_weights(
                        X[mask],
                        hard_or_soft=hard_or_soft,
                        weights_average=weights_average,
                        weights2use=self.weights_for_model,
                        top_model=model_per_task["generalized_prediction"],
                        prediction_type=prediction_type,
                        aggregation=aggregation
                    )
                mask = np.isin(y[:, departement_index], generalized_departement)
                predictions[mask] = preds_gen

        for val in [2, 3, 4]:
            task_name = f"class_value_{val}_predictions"
            if task_name in model_per_task:
                if prediction_type == 'Class' or prediction_type == 'RawFormulaVal':
                    preds_cls, _ = self.predict_with_weight(
                        X,
                        hard_or_soft=hard_or_soft,
                        weights_average=weights_average,
                        weights2use=self.weights_for_model,
                        top_model=model_per_task[task_name],
                        prediction_type=prediction_type,
                        aggregation=aggregation
                    )
                    mask = (preds_cls >= val) | (predictions >= val)
                    if mask.any():
                        predictions[mask] = preds_cls[mask]
                else:
                    preds_cls, _ = self.predict_proba_with_weights(
                        X,
                        hard_or_soft=hard_or_soft,
                        weights_average=weights_average,
                        weights2use=self.weights_for_model,
                        top_model=model_per_task[task_name],
                        prediction_type=prediction_type,
                        aggregation=aggregation
                    )
                    # prendre les lignes où la classe la plus probable est >= val
                    mask = (np.argmax(preds_cls, axis=1) >= val) | (np.argmax(predictions, axis=1) >= val)
                    if mask.any():
                        predictions[mask] = preds_cls[mask]
                        
        return predictions[:, None], y
    
    def plot_weights_by_target(self, save_path=None):
        """
        Plot the weights of each model against its target name.
        """
        if save_path is None:
            save_path = self.dir_log / 'weights_by_target.png'
            
        target_names = []
        weights = []
        
        for model, weight in zip(self.best_estimator_, self.weights_for_model):
            target_names.append(model.target_name)
            weights.append(weight)
            
        plt.figure(figsize=(12, 6))
        plt.scatter(target_names, weights, alpha=0.7)
        plt.xlabel('Target Name')
        plt.ylabel('Weight')
        plt.title('Model Weights by Target Name')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def predict(self, X, hard_or_soft='soft', weights_average='weight', top_model='all', id_col=(None, None), prediction_type="Class", aggregation=True):
        """
        Predict labels for input data using each model and aggregate the results.

        Parameters:
        - X_list: List of data to predict labels for.
        
        Returns:
        - Aggregated predicted labels.
        """

        if weights_average not in ['None', 'weight']:
            assert id_col[0] is not None and id_col[1] is not None
            vals = id_col[1]
            unique_ids = np.unique(vals)
            prediction = np.empty(X.shape[0], dtype=int)
            y = np.empty(X.shape[0], dtype=int)
            for id in unique_ids:
                print(f'Prediction for {id_col[0]} {id}')
                mask = (id_col[1] == id)
                prediction[mask], y[mask] = self.predict_with_weight(X[mask], hard_or_soft=hard_or_soft, weights_average='weight', \
                    weights2use=self.weights_id_model[id_col[0]][id], top_model=top_model, prediction_type=prediction_type, aggregation=aggregation)
            return prediction, y
        else:
            return self.predict_with_weight(X, hard_or_soft=hard_or_soft, weights_average='weight', weights2use=self.weights_for_model, top_model=top_model,  prediction_type=prediction_type, aggregation=aggregation)
    
    def remove_graph(self):
        for teacher in self.best_estimator_:
            del teacher.graph
            
    def predict_proba(self, X, weights_average='weight', top_model='all', id_col=(None, None)):
        """
        Predict probabilities for input data using each model and aggregate the results.

        Parameters:
        - X_list: List of data to predict probabilities for.
        
        Returns:
        - Aggregated predicted probabilities.
        """

        if weights_average not in ['None', 'weight']:
            assert id_col[0] is not None and id_col[1] is not None
            vals = id_col[1]
            unique_ids = np.unique(vals)
            prediction = np.empty(X.shape[0], dtype=int)
            y = np.empty(X.shape[0], dtype=int)
            for id in unique_ids:
                print(f'Prediction for {id_col[0]} {id}')
                mask = (id_col[1] == id)
                prediction[mask], y[mask] = self.predict_proba_with_weights(X[mask], hard_or_soft='soft', weights_average='weight', weights2use=self.weights_id_model[id_col[0]][id], top_model=top_model)
            return prediction, y

        else:
            return self.predict_proba_with_weights(X, hard_or_soft='soft', weights_average='weight', weights2use=self.weights_for_model, top_model=top_model)
    
    def clean(self):
        for teacher in self.best_estimator_:
            del teacher.df_train
            del teacher.df_test
            del teacher.df_val
            del teacher.train_loader
            del teacher.val_loader
            del teacher.test_loader

    def aggregate_predictions(self, predictions_list, models_to_mean, weight2use=[], id_col=(None, None), prediction_type="RawFormulaVal"):
        """
        Aggregate predictions from multiple models with weights.

        Parameters:
        - predictions_list: List of predictions from each model.

        Returns:
        - Aggregated predictions.
        """
        
        if prediction_type == 'RawFormulaVal' or prediction_type == 'Probability':
            return self.aggregate_probabilities(predictions_list, models_to_mean, weight2use=weight2use, id_col=id_col)

        predictions_array = np.array(predictions_list)
        if len(weight2use) == 0 or weight2use is None:
            weight2use = np.ones_like(self.weights_for_model)[models_to_mean]

        # Normalize weights
        weight2use = weight2use / np.sum(weight2use)

        if self.task_type == 'classification' or self.task_type == 'ordinal-classification':
            # Weighted vote for classification
            unique_classes = np.arange(0, 5)
            weighted_votes = np.zeros((len(unique_classes), predictions_array.shape[1]))

            for i, cls in enumerate(unique_classes):
                mask = (predictions_array == cls)
                weighted_votes[i] = np.sum(mask * weight2use.reshape(mask.shape[0], 1), axis=0)

            aggregated_pred = unique_classes[np.argmax(weighted_votes, axis=0)]
        else:
            # Weighted average for regression
            weighted_sum = np.sum(predictions_array * weight2use[:, None], axis=0)
            aggregated_pred = weighted_sum
            #aggregated_pred = np.max(predictions_array * weight2use[:, None], axis=0)
        
        return aggregated_pred

    """def aggregate_predictions_id(self, predictions_array, models_to_mean, id_col=(None, None)):
        assert id_col[0] is not None and id_col[1] is not None
        id = id_col[0]
        vals = id_col[1]
        uvals = np.unique(vals)

        weight2use = np.zeros((len(models_to_mean), predictions_array.shape[1]))
        for val in uvals:
            mask = (vals == val)
            weight2use[:, mask] = self.weights_id_model[id][val][models_to_mean]
            if np.all(weight2use[:, mask] == 0):
                weight2use[:, mask] = self.weights_for_model[models_to_mean]

        unique_classes = np.arange(0, 5)
        weighted_votes = np.zeros((len(unique_classes), predictions_array.shape[1]))

        for i, cls in enumerate(unique_classes):
            mask = (predictions_array == cls)
            weighted_votes[i] = np.sum(mask * weight2use, axis=0)

        aggregated_pred = unique_classes[np.argmax(weighted_votes, axis=0)]
        return aggregated_pred"""   

    def aggregate_probabilities(self, probas_list, models_to_mean, weight2use=[], id_col=(None, None)):
        """
        Aggregate probabilities from multiple models with weights.

        Parameters:
        - probas_list: List of probability predictions from each model.

        Returns:
        - Aggregated probabilities.
        """
        probas_array = np.array(probas_list)
        if weight2use is None or len(weight2use) == 0:
            weight2use = np.ones_like(self.weights_for_model)[models_to_mean]
        
        # Weighted average for probabilities
        weighted_sum = np.sum(probas_array * weight2use[:, None, None, None], axis=0)
        aggregated_proba = weighted_sum / np.sum(weight2use)
        #aggregated_proba = np.max(probas_array * weight2use[:, None, None], axis=0)
        return aggregated_proba
    
    def aggregate_probabilities_tensor(self, probas_list, models_to_mean, weight2use=None, id_col=(None, None)):
        """
        Agrège des probabilités issues de plusieurs modèles (torch tensors).

        Paramètres
        ----------
        probas_list : torch.Tensor ou list[torch.Tensor]
            Liste ou tensor de taille [M, B, ..., C] contenant les probabilités des M modèles.
        models_to_mean : list[int] ou torch.Tensor
            Indices des modèles à inclure dans l’agrégation.
        weight2use : list[float] ou torch.Tensor ou None
            Poids associés à chaque modèle (même longueur que models_to_mean).
            Si None ou vide, poids uniformes.
        id_col : tuple (optionnel)
            Non utilisé ici (inclus pour compatibilité).

        Retour
        ------
        aggregated_proba : torch.Tensor
            Tensor agrégé de taille [B, ..., C] (moyenne pondérée sur les modèles).
        """
        # Convertir en tensor unique si liste
        if isinstance(probas_list, (list, tuple)):
            probas_tensor = torch.stack(probas_list, dim=0)  # [M, B, ..., C]
        else:
            probas_tensor = probas_list  # déjà un tensor

        device = probas_tensor.device
        dtype = probas_tensor.dtype

        M = probas_tensor.shape[0]

        # Gérer les poids
        if weight2use is None or len(weight2use) == 0:
            weights = torch.as_tensor(self.weights_for_model, dtype=dtype)
            if models_to_mean is not None:
                weights = weights[models_to_mean]
            else:
                weights = torch.ones(M, device=device, dtype=dtype)
        else:
            weights = torch.as_tensor(weight2use, device=device, dtype=dtype)
        
        weights = weights / weights.sum().clamp_min(1e-12)

        # Agrégation pondérée (diffusée sur les dimensions suivantes)
        # poids shape [M, 1, 1, ...] compatible avec probas_tensor [M, B, ..., C]
        view_shape = [M] + [1] * (probas_tensor.dim() - 1)
        aggregated_proba = (probas_tensor * weights.view(view_shape)).sum(dim=0)

        return aggregated_proba

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
        predictions = self.predict(X)
        return self.score_with_prediction(predictions, y, sample_weight)
    
    def score_with_prediction(self, y_pred, y, sample_weight=None):
        
        return iou_score(y, y_pred)
    
class ModelPerID(RegressorMixin, ClassifierMixin):
    def __init__(self, model, dir_log, cluster="departement", horizon=0):
        self.base_model = model
        self.cluster_col = cluster
        self.models = {}
        self.is_fitted_ = False
        self.name = f'unique-{cluster}-{model.name}'

        self.horizon = horizon
        self.dir_log = dir_log

    def fit(self, df_train, df_val, df_test, graph, PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params, **args):
        """Train one model per cluster value."""
        self.models = {}
        values = df_train[self.cluster_col].unique()
        for val in values:
            train_subset = df_train[df_train[self.cluster_col] == val].reset_index(drop=True)
            val_subset = df_val[df_val[self.cluster_col] == val].reset_index(drop=True)
            test_subset = df_test[df_test[self.cluster_col] == val].reset_index(drop=True)
            model = deepcopy(self.base_model)
            model.dir_log = self.base_model.dir_log / f'unique-{val}-{self.base_model.name}'
            if hasattr(model, "name"):
                model.name = f"{val}_{model.name}"
            if hasattr(model, "fit"):
                args['PATIENCE_CNT'] = PATIENCE_CNT
                args['CHECKPOINT'] = CHECKPOINT
                args['epochs'] = epochs
                args['custom_model_params'] = custom_model_params
                model.fit(train_subset, val_subset, test_subset, graph, **args)
            else:
                raise ValueError("Base model must implement fit method")
            self.models[val] = model
        self.is_fitted_ = True

    def predict_proba(self, X, **kwargs):
        proba = None
        result = []
        for val, model in self.models.items():
            mask = X[self.cluster_col] == val
            if mask.any():
                preds = model.predict_proba(X[mask], **kwargs)
                if proba is None:
                    proba = np.zeros((len(X), preds.shape[1]))
                proba[mask] = preds
        return proba

    def predict(self, X, **kwargs):
        preds = np.zeros(len(X))
        for val, model in self.models.items():
            mask = X[self.cluster_col] == val
            if mask.any():
                preds[mask] = model.predict(X[mask], **kwargs)
        return preds

    def _predict_test_loader(self, loader):
        preds_list = []
        ys_list = []
        for val, model in self.models.items():
            pred, y = model._predict_test_loader(loader)
            preds_list.append(pred)
            ys_list.append(y)
        return torch.cat(preds_list, 0), torch.cat(ys_list, 0)