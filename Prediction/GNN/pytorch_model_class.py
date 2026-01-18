from GNN.pytorch_model_tools import *

class ModelCNN(SplitTraining):
    def __init__(self, model_name, nbfeatures, batch_size, lr, target_name, task_type, out_channels, dir_log, features_name, features, features_1D,
                 ks, loss, name, device, under_sampling, over_sampling, path, image_per_node, n_run, training_mode='normal', federated_cluster='', cut_layer_name='', input_server_model=0,
                post_process=None,
                 **kwargs):

        super().__init__(federated_cluster=federated_cluster, cut_layer_name=cut_layer_name, input_server_model=input_server_model, model_name=model_name, nbfeatures=nbfeatures, batch_size=batch_size, lr=lr,
                         target_name=target_name, task_type=task_type, features_name=features_name, ks=ks,
                         out_channels=out_channels, dir_log=dir_log, loss=loss, name=name, device=device, under_sampling=under_sampling,
                         over_sampling=over_sampling, n_run=n_run, post_process=post_process, **kwargs)

        self.training_mode = training_mode
        self.path = path
        self.features = features
        self.features_1D = features_1D
        self.image_per_node = image_per_node
        self.nbfeatures = nbfeatures
        
    def create_train_val_test_loader(self, graph, df_train, df_val, df_test, epochs,
                                     PATIENCE_CNT, CHECKPOINT, features_importance=True,
                                     varying_time_variables=[], train_features=[], custom_model_params=None, name_exp=None):

        if features_importance:
            importance_df = calculate_and_plot_feature_importance(df_train[self.features_1D], df_train[self.target_name], self.features_1D, self.dir_log / '../importance', self.target_name)
            #importance_df = calculate_and_plot_feature_importance_shapley(df_train[self.features], df_train[self.target_name], self.features, self.dir_log / '../importance', self.target_name)
            features95, self.features_1D = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)
        
        if self.nbfeatures != 'all':
            self.nbfeatures = int(self.nbfeatures)
            varying_time_variables_2 = get_time_columns(varying_time_variables, self.ks, df_train.copy(deep=True), train_features)
            
            self.features_1D = [fet for fet in self.features_1D if fet in df_train.columns]
            
            self.features_1D = self.features_1D[:self.nbfeatures]

            print(self.features_1D)

            features_name_2D, newShape2D = get_features_name_lists_2D(6, train_features)
            self.features_name = get_features_selected_for_time_series_for_2D(self.features_1D, features_name_2D, varying_time_variables_2, self.nbfeatures)
            
            self.features_name = list(np.unique(self.features_name))
            print(self.features_name)
        
        if self.under_sampling != 'full':
            y = df_train[self.target_name]
            old_shape = df_train.shape
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
                    if self.training_mode == 'splittraining':
                        best_combinaison = self.search_samples_proportion_per_cluster(graph, df_train, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT)
                        df_parts = []
                        for cluster, nb in best_combinaison:
                            df_cluster = df_train[df_train[self.federated_cluster] == cluster]
                            sampled = self.split_dataset(df_cluster, nb, reset=True)[cluster]
                            df_parts.append(sampled)

                        df_train = pd.concat(df_parts).reset_index(drop=True)
                    else:
                        if self.under_sampling == 'search':
                            best_tp, find_log = self.search_samples_proportion(graph, df_train, df_val, df_test, False, epochs, PATIENCE_CNT, CHECKPOINT)
                            self.find_log = find_log
                        else:
                            vec = self.under_sampling.split('-')
                            try:
                                best_tp = float(vec[-1])
                            except ValueError:
                                logger.info(f'{self.under_sampling} with undefined factor, set to 0.3 -> {0.3 * len(y[y == 0])}')
                                best_tp = 0.3

                        nb = int(best_tp * len(y[y == 0]))

                        df_combined = self.split_dataset(df_train, nb, reset=False)
                        df_train['weight'] = 0

                        # Mettre à jour df_train pour l'entraînement
                        df_train.loc[df_combined.index, 'weight'] = 1
                        logger.info(f'Train mask df_train shape: {old_shape} -> {df_train.shape}')
                
        if False:
            self.train_loader = read_object('train_loader.pkl', self.dir_log)
            self.val_loader = read_object('val_loader.pkl', self.dir_log)
            self.test_loader = read_object('test_loader.pkl', self.dir_log)
        else:
            train_dataset, val_dataset, test_dataset = create_dataset_2D(graph=graph,
                                            df_train=df_train,
                                            df_val=df_val,
                                            df_test=df_test,
                                            features_name_2D=self.features_name,
                                            features=self.features,
                                            features_1D=self.features_1D,
                                            target_name=self.target_name,
                                            use_temporal_as_edges=None,
                                            image_per_node=self.image_per_node,
                                            device=self.device, ks=self.ks,
                                            path=self.path,
                                            name_exp=name_exp)

            train_loader = DataLoader(train_dataset, 16, True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(val_dataset, 16, False, worker_init_fn=seed_worker, generator=g)
            test_loader = DataLoader(test_dataset, 16, False, worker_init_fn=seed_worker, generator=g)

            #save_object_torch(train_loader, 'train_loader.pkl', self.dir_log)
            #save_object_torch(val_loader, 'val_loader.pkl', self.dir_log)
            #save_object_torch(test_loader, 'test_loader.pkl', self.dir_log)

            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader

    def create_test_loader(self, graph, df):

        x_test, y_test = df[ids_columns + self.features_1D].values, df[ids_columns + [self.target_name]].values

        dateTest = np.sort(np.unique(y_test[np.argwhere(y_test[:, weight_index] > 0), date_index]))
        
        XsTe, YsTe, EsTe = create_dataset_2D_2(graph, x_test, y_test, self.ks, dateTest,
                    self.features_name, self.features, self.features_1D, False)
        
        sub_dir = f'image_per_node_{len(self.features_name)}' if self.image_per_node else f'image_per_departement_{len(self.features_name)}'
        test_dataset = ReadGraphDataset_2D_from_xarray(XsTe, YsTe, EsTe, len(XsTe), device, rootDisk / 'csv', self.features_name, self.features_1D, self.ks,
                                                graph.scale,
                                                graph.graph_method,
                                                graph.base,
                                                self.path / 'datacube')
        
        loader = DataLoader(test_dataset, test_dataset.__len__(), False)

        return loader

class ModelGNN(SplitTraining):
    def __init__(self, graph_method, mesh, mesh_file, model_name, nbfeatures, batch_size, lr, target_name, task_type,
                 out_channels, dir_log, features_name, ks, loss, name, device, under_sampling, over_sampling,
                 n_run, training_mode='normal', federated_cluster='', cut_layer_name='', input_server_model=0,
                 horizon=0, post_process=None):

        super().__init__(federated_cluster=federated_cluster, cut_layer_name=cut_layer_name, input_server_model=input_server_model, model_name=model_name, nbfeatures=nbfeatures, batch_size=batch_size, lr=lr, target_name=target_name, task_type=task_type, features_name=features_name, ks=ks,
                         out_channels=out_channels, dir_log=dir_log, loss=loss, name=name, device=device, under_sampling=under_sampling,
                         over_sampling=over_sampling, n_run=n_run, horizon=horizon, post_process=post_process)
        self.training_mode = training_mode
        self.mesh = mesh
        self.mesh_file = mesh_file
        self.graph_method = graph_method
        self.mesh2graph = None
        self.gridh2mesh = None
        self.graph_mesh = None
        self.horizon = horizon

    def create_train_val_test_loader(self, graph, df_train, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT, features_importance=True, custom_model_params=None, use_log=True):

        self.graph = graph
        if self.mesh and self.graph_mesh is None:
            
            df = pd.concat((df_train, df_val, df_test))

            latitudes = torch.Tensor(df['latitude'].values.reshape(-1,1))
            longitudes = torch.Tensor(df['longitude'].values.reshape(-1,1))
            departement = torch.Tensor(df['departement'].values.reshape(-1,))

            g_lat_lon_grid = torch.concat((latitudes, longitudes), dim=1).to('cpu')
            g_lat_lon_grid = torch.unique(g_lat_lon_grid, dim=0)
             
            graph_builder = GraphBuilder(self.mesh_file, g_lat_lon_grid, doPrint=False)

            self.graph_mesh = graph_builder.create_mesh_graph(None)
            self.gridh2mesh, self.graph_mesh = graph_builder.create_g2m_graph(None, self.graph_mesh)
            self.mesh2graph = graph_builder.create_m2g_graph(None)

            def _check_non_empty_edges(g, etype, name):
                e = g.num_edges(etype)
                if e == 0:
                    print(f"[WARN] {name}: 0 edges pour etype {etype}.")
                return e

            _check_non_empty_edges(self.gridh2mesh, ("grid","g2m","mesh"), "gridh2mesh")
            _check_non_empty_edges(self.mesh2graph, ("mesh","m2g","grid"), "mesh2graph")

            if '_ID' not in self.graph_mesh.edata:
                self.graph_mesh.edata['_ID'] = torch.arange(self.graph_mesh.num_edges(), dtype=torch.int32)
            if '_ID' not in self.graph_mesh.ndata:
                self.graph_mesh.ndata['_ID'] = torch.arange(self.graph_mesh.num_nodes(), dtype=torch.int32)

        if features_importance:
            importance_df = calculate_and_plot_feature_importance(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
            #importance_df = calculate_and_plot_feature_importance_shapley(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
            features95, featuresAll = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)
            
            #exit(1)
            if self.nbfeatures != 'all':
                self.features_name = featuresAll[:int(self.nbfeatures)]
            else:
                self.features_name = featuresAll
        
        if self.model_name in ['GRUGNN']:
            static_idx, temporal_idx = get_static_temporal_idx(self.features_name)
            logger.info(f'Num temporal features {len(temporal_idx)} num spatial features {len(static_idx)}')
            custom_model_params = {'static_idx': static_idx, 'temporal_idx' : temporal_idx}

        if self.val_loader is None:
            if self.mesh == 'mesh':
                val_dataset, test_dataset = create_test_val_dataset(graph,
                                                    df_val,
                                                    df_test,
                                                    self.features_name,
                                                    self.target_name,
                                                    None,
                                                    self.device, self.ks,
                                                    self.horizon,
                                                    graph_mesh=self.graph_mesh,
                                                    gridh2mesh=self.gridh2mesh,
                                                    mesh2graph=self.mesh2graph
                                                    )
            else:
                val_dataset, test_dataset = create_test_val_dataset(graph,
                                                    df_val,
                                                    df_test,
                                                    self.features_name,
                                                    self.target_name,
                                                    False,
                                                    self.device, self.ks,
                                                    self.horizon,
                                                    graph_mesh=self.graph_mesh,
                                                    gridh2mesh=self.gridh2mesh,
                                                    mesh2graph=self.mesh2graph)
                
            if not self.mesh or self.mesh == False:            
                self.val_loader = DataLoader(val_dataset, val_dataset.__len__(), False, collate_fn=graph_collate_fn)
                self.test_loader = DataLoader(test_dataset, test_dataset.__len__(), False, collate_fn=graph_collate_fn)
            
            elif self.mesh == 'mesh':
                self.val_loader = DataLoader(val_dataset, val_dataset.__len__(), False, collate_fn=graph_collate_fn_mesh)
                self.test_loader = DataLoader(test_dataset, test_dataset.__len__(), False, collate_fn=graph_collate_fn_mesh)

            elif self.mesh == 'mygraph':
                self.val_loader = DataLoader(val_dataset, val_dataset.__len__(), False, collate_fn=graph_collate_fn_multiple_graph)
                self.test_loader = DataLoader(test_dataset, test_dataset.__len__(), False, collate_fn=graph_collate_fn_multiple_graph)

        if self.under_sampling != 'full':
            y = df_train[self.target_name]
            old_shape = df_train.shape
            if 'binary' in self.under_sampling:
                vec = self.under_sampling.split('-')
                try:
                    nb = int(vec[-1]) * len(df_train[df_train[self.target_name] > 0])
                except:
                    logger.info(f'{self.under_sampling} with undefined factor, set to 1 -> {len(df_train[df_train[self.target_name] > 0])}')
                    nb = len(df_train[df_train[self.target_name] > 0])

                df_combined = self.split_dataset(df_train, nb, reset=False)
                df_train['weight'] = 0
                # Mettre à jour df_train pour l'entraînement
                df_train.loc[df_combined.index, 'weight'] = 1

                logger.info(f'Train mask df_train shape: {old_shape} -> {df_train[df_train["weight"] > 0].shape}')
                
            elif self.under_sampling == 'search' or 'percentage' in self.under_sampling:
                    if self.training_mode == 'splittraining':
                        best_combinaison = self.search_samples_proportion_per_cluster(graph, df_train, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT)
                        df_parts = []
                        for cluster, nb in best_combinaison:
                            df_cluster = df_train[df_train[self.federated_cluster] == cluster]
                            sampled = self.split_dataset(df_cluster, nb, reset=True)[cluster]
                            df_parts.append(sampled)

                        df_train = pd.concat(df_parts).reset_index(drop=True)
                    else:
                        if self.under_sampling == 'search':
                            best_tp, find_log = self.search_samples_proportion(graph, df_train, df_val, df_test, False,
                                                                               epochs, PATIENCE_CNT, CHECKPOINT, False,
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

                        df_combined = self.split_dataset(df_train, nb, reset=False)
                        df_train['weight'] = 0

                        # Mettre à jour df_train pour l'entraînement
                        df_train.loc[df_combined.index, 'weight'] = 1
                        logger.info(f'Train mask df_train shape: {old_shape} -> {df_train[df_train["weight"] > 0].shape}')

        self.df_train = df_train

        if False:
            self.train_loader = read_object('train_loader.pkl', self.dir_log)
            self.val_loader = read_object('val_loader.pkl', self.dir_log)
            self.test_loader = read_object('test_loader.pkl', self.dir_log)
        else:
            if self.mesh == 'mesh':
                train_dataset = create_train_dataset(graph,
                                                    df_train,
                                                    self.features_name,
                                                    self.target_name,
                                                    None,
                                                    self.device, self.ks,
                                                    self.horizon,
                                                    graph_mesh=self.graph_mesh,
                                                    gridh2mesh=self.gridh2mesh,
                                                    mesh2graph=self.mesh2graph)
            else:
                train_dataset = create_train_dataset(graph,
                                                    df_train,
                                                    self.features_name,
                                                    self.target_name,
                                                    False,
                                                    self.device, self.ks,
                                                    self.horizon,
                                                    graph_mesh=self.graph_mesh,
                                                    gridh2mesh=self.gridh2mesh,
                                                    mesh2graph=self.mesh2graph)

            if not self.mesh or self.mesh == False:            
                self.train_loader = DataLoader(train_dataset, self.batch_size, True, collate_fn=graph_collate_fn)
            
            elif self.mesh == 'mesh':
                self.train_loader = DataLoader(train_dataset, self.batch_size, True, collate_fn=graph_collate_fn_mesh)

            elif self.mesh == 'mygraph':
                self.train_loader = DataLoader(train_dataset, self.batch_size, True, collate_fn=graph_collate_fn_multiple_graph)

        #save_object_torch(self.train_loader, 'train_loader.pkl', self.dir_log)
        #save_object_torch(self.val_loader, 'val_loader.pkl', self.dir_log)
        #save_object_torch(self.test_loader, 'test_loader.pkl', self.dir_log)
        
    def create_test_loader(self, graph, df):
        loader = create_test_loader(graph, df,
                       self.features_name,
                       self.device,
                       False,
                       self.target_name,
                       self.ks,
                        self.horizon,
                       self.graph_mesh,
                        self.gridh2mesh,
                        self.mesh2graph)

        return loader

    def launch_batch(self, data, criterion, batch_type, do_update):

        if not self.mesh or self.mesh == False:
            inputs, labels, graphs, graphs_id = data
            model_args = (graphs,)
        else:
            inputs, labels, DGLgraphs, graphs_id = data

        if inputs.shape[0] == 1:
            return 0

        band = -1
        total_loss = None

        hidden_past: List[torch.Tensor] = []  # contiendra des tenseurs (B, D)
        output_past : List(torch.Tensor) = []

        for H in range(self.horizon + 1):
            horizon_index = -1 - (self.horizon - H)

            try:
                target, weights = self.compute_weights_and_target(
                    labels,
                    band,
                    ids_columns,
                    getattr(self.model, 'is_graph_or_node', False),
                    graphs_id,
                    horizon_index
                )
            except Exception:
                target, weights = self.compute_weights_and_target(
                    labels,
                    band,
                    ids_columns,
                    False,
                    graphs_id,
                    horizon_index
                )

            if self.loss not in ['kldivloss']:
                target = target.long()

            inputs_horizon = self.compute_inputs(inputs, horizon_index, "current" if H == 0 else "futur")

            if H == 0:
                z_prev = None
            else:
                if self.ks > 0:
                    # on prend les ks derniers états cachés déjà vus
                    history = hidden_past[-(self.ks + 1):]
                    # empilement (B, D, L) avec L = len(history)
                    z_prev = torch.stack(history, dim=2)  # (B, D, L)

                    # padding à gauche si L < ks
                    L = z_prev.size(2)
                    if L < (self.ks + 1):
                        B, D = z_prev.size(0), z_prev.size(1)
                        pad = torch.zeros(
                            (B, D, self.ks + 1 - L),
                            device=z_prev.device,
                            dtype=z_prev.dtype
                        )
                        z_prev = torch.cat([pad, z_prev], dim=2)  # (B, D, ks)
                else:
                    z_prev = hidden_past[-1]
                    
            if H > 0:
                if self.id_past_risk is not None:
                    inputs_horizon[:, self.id_past_risk, -H:] = 0
                if self.id_past_ba is not None:
                    inputs_horizon[:, self.id_past_ba, -H:] = 0
                if self.prev_idx is not None:
                    inputs_horizon[:, self.prev_idx, -H:] = torch.stack(output_past, dim=2)
            else: 
                z_prev = None

            output, logits, hidden = self.model(inputs_horizon, DGLgraphs[0], DGLgraphs[1], DGLgraphs[2], z_prev=z_prev)
            if batch_type == 'train' and do_update:
                if has_method(criterion, 'update_after_batch'):
                    criterion.update_after_batch(logits, target)

            hidden_past.append(hidden)
            output_past.append(output)

            loss = self.calculate_loss(criterion, logits, target, weights, labels)

            if self.student_train:
                criterion_teacher = self.get_loss('kldivloss')
                #df_test = pd.DataFrame(inputs_horizon[:, :, -1], columns=self.features_name)
                #df_test.columns = df_test.columns.astype(str)
                pred_teacher = self.teacher.predict_proba(
                    df_test,
                    weights_average=self.weights_average,
                    top_model=self.top_model,
                    id_col=(None, None)
                )
                target_teacher = torch.Tensor(pred_teacher, device=inputs.device).to(torch.float32)
                target_teacher = target_teacher / self.temperature_value
                
                loss2 = self.calculate_loss(criterion_teacher, output, target_teacher, weights, labels, tolong=False)

                loss = self.alpha_value * loss2 + (1 - self.alpha_value) * loss

            if self.constrastive:
                _, _, zprev = self.prev_model(inputs, *model_args)
                _, _, zglob = self.global_model(inputs, *model_args)
                loss_constrastive = self.calculate_contrastive_moon_loss(hidden, zprev, zglob, self.temperature_value)
                loss = loss + self.smooth_value * loss_constrastive

            if self.use_prototypes and self.prototypes is not None:
                proto_loss = self.calculate_prototype_alignment_loss(hidden, target, self.prototypes)
                loss = loss + self.prototype_weight * proto_loss

            if self.model_name in ['BayesianMLP', 'BayesianCNN', 'BayesianRNN']:
                loss += self.model.kl_loss()

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        return total_loss

    def _predict_test_loader(self, X: DataLoader, prediction_type='Class', output_pdf='test', calibrate=False) -> torch.tensor:
        """
        Generates predictions using the model on the provided DataLoader, with optional autoregression.

        Note:
        - 'node_id' and 'date_id' are not included in features_name but can be found in labels[:, 0] and labels[:, 4].
        Parameters:
        - X_: DataLoader providing the test data.
        - features: Numpy array of feature indices to use.
        - device: Torch device to use for computations.
        - target_name: Name of the target variable.
        - autoRegression: Boolean indicating whether to use autoregression.
        - features_name: List mapping feature names to indices.
        - dataset: Pandas DataFrame containing 'node_id', 'date_id', and 'nbsinister' columns.

        Returns:
        - pred: Tensor containing the model's predictions.
        - y: Tensor containing the true labels.
        """
        assert self.model is not None
        self.model.eval()
        criterion = self.get_loss(self.loss, {})
        if len(self.criterion_params) > 0:
            if has_method(criterion, 'update_params'):
                criterion.update_params(self.criterion_params[self.best_epoch])
                criterion.eval()

        with torch.no_grad():
            pred = []
            y = []

            for i, data in enumerate(X, 0):

                if not self.mesh:
                    inputs, orilabels_, graphs, graphs_id = data
                    model_args = (graphs,)
                else:
                    inputs, orilabels_, DGLgraphs, graphs_id = data
                    
                orilabels_ = orilabels_.to(device)

                pred_horizon = []
                labels_horizon = []

                prev_output = None

                hidden_past: List[torch.Tensor] = []  # contiendra des tenseurs (B, D)
                output_past: List[torch.Tensor] = []
                for H in range(self.horizon + 1):

                    horizon_index = -1 - (self.horizon - H)
                    orilabels = orilabels_[:, :, horizon_index]

                    inputs_horizon = self.compute_inputs(inputs, horizon_index, "current" if H == 0 else "futur")
                    if H == 0:
                        z_prev = None
                    else:
                        if self.ks > 0:
                            # on prend les ks derniers états cachés déjà vus
                            history = hidden_past[-(self.ks + 1):]
                            # empilement (B, D, L) avec L = len(history)
                            z_prev = torch.stack(history, dim=2)  # (B, D, L)

                            # padding à gauche si L < ks
                            L = z_prev.size(2)
                            if L < (self.ks + 1):
                                B, D = z_prev.size(0), z_prev.size(1)
                                pad = torch.zeros(
                                    (B, D, self.ks + 1 - L),
                                    device=z_prev.device,
                                    dtype=z_prev.dtype
                                )
                                z_prev = torch.cat([pad, z_prev], dim=2)  # (B, D, ks)
                        else:
                            z_prev = hidden_past[-1]
                    if H > 0:
                        if self.id_past_risk is not None:
                            inputs_horizon[:, self.id_past_risk, -H:] = 0
                        if self.id_past_ba is not None:
                            inputs_horizon[:, self.id_past_ba, -H:] = 0
                        if self.prev_idx is not None and prev_output is not None:
                            inputs_horizon[:, self.prev_idx, -H:] = torch.stack(output_past, dim=2)

                    output, logits, hidden = self.model(inputs_horizon, DGLgraphs[0], DGLgraphs[1], DGLgraphs[2], z_prev=z_prev)

                    hidden_past.append(hidden)
                    output_past.append(output)
                    
                    if 'criterion' in locals() and hasattr(criterion, 'calibrate') and calibrate:
                            if 'clusters_ids' in required_params(criterion.transform):
                                clusters_ids = orilabels[:, criterion.id].long()
                                calibration = criterion.calibrate(inputs=logits, y_true=orilabels[:, -1], score_fn=iou_score, clusters_ids=clusters_ids, dir_output=self.dir_log)
                            else:
                                calibration = criterion.calibrate(inputs=logits, y_true=orilabels[:, -1], score_fn=iou_score, dir_output=self.dir_log)
                            
                            self.calibration = calibration
                        
                    elif 'criterion' in locals() and hasattr(criterion, 'calibrate'):
                        assert hasattr(self, 'calibration')
                            
                    if 'criterion' in locals() and hasattr(criterion, 'transform'):
                        params = {'inputs' : logits}
                        if 'clusters_ids' in required_params(criterion.transform):
                            clusters_ids = orilabels[:, criterion.id].long()
                            params['clusters_ids'] = clusters_ids
                            
                        if 'output_pdf' in required_params(criterion.transform):
                            assert output_pdf is not None and self.dir_log is not None
                            params['output_pdf'] = output_pdf
                        
                        if 'dir_output' in required_params(criterion.transform):    
                            params['dir_output'] = self.dir_log
                            
                        if 'areas' in required_params(criterion.transform):
                            params['areas'] = orilabels[:, area_index]
                            
                        if 'p_thresh' in required_params(criterion.transform):
                            params['p_thresh'] = self.calibration
                            
                        params['prediction_type'] = prediction_type

                        output = criterion.transform(**params)

                    if prediction_type == 'Class':

                        if self.task_type in ['classification', 'binary']:
                            output = torch.argmax(output, dim=1)

                        elif self.task_type == 'regression' and output.ndim > 1 and output.shape[1] > 1:
                            output = torch.argmax(output, dim=1)

                    elif prediction_type == 'RawFormulaVal':
                        output = logits

                    pred_horizon.append(output[:, None])
                    labels_horizon.append(orilabels[:, :, None])

                pred_horizon = torch.cat(pred_horizon, dim=1)
                labels_horizon = torch.cat(labels_horizon, dim=2)
                pred.append(pred_horizon)
                y.append(labels_horizon)

            y = torch.cat(y, 0)
            pred = torch.cat(pred, 0)
            
            if prediction_type == 'Class' and pred.dtype != torch.long:
                pred = torch.round(pred, decimals=1)

            return pred, y

class Model_Torch(SplitTraining):
    def __init__(self, model_name, nbfeatures, batch_size, lr, target_name, task_type, out_channels,
                 dir_log, features_name, ks, loss, name, device, under_sampling, over_sampling, n_run,
                 training_mode='normal', federated_cluster='', cut_layer_name='', input_server_model=0,
                 horizon=0, post_process=None):

        #federated_cluster, model_name, nbfeatures, batch_size, lr, target_name, task_type, out_channels,
        #         dir_log, features_name, ks, loss, name, device, under_sampling, over_sampling, n_run
        
        super().__init__(federated_cluster=federated_cluster, cut_layer_name=cut_layer_name, input_server_model=input_server_model,
                         model_name=model_name, nbfeatures=nbfeatures, batch_size=batch_size, lr=lr,
                         target_name=target_name, task_type=task_type, features_name=features_name, ks=ks,
                         out_channels=out_channels, dir_log=dir_log, loss=loss, name=name, device=device, under_sampling=under_sampling,
                         over_sampling=over_sampling, n_run=n_run, horizon=horizon, post_process=post_process)

        self.training_mode = training_mode

    def create_train_val_test_loader(self, graph, df_train, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT, features_importance=True, custom_model_params=None, use_log=True):
        self.graph = graph

        if 'learnable-area' in self.loss:
            area_parameters = np.sort(df_train['graph_id'].unique())
            self.area_parameters = torch.nn.Parameter(torch.Tensor(torch.ones_like(torch.tensor(area_parameters))))
        elif 'area' in self.loss:
            area_parameters = np.sort(df_train['graph_id'].unique())
            self.area_parameters = torch.ones_like(torch.tensor(area_parameters))
        
        if False:
            ##################################### Select features #########################################
            importance_df = calculate_and_plot_feature_importance(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
            #importance_df = calculate_and_plot_feature_importance_shapley(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
            features95, featuresAll = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)

            if self.nbfeatures != 'all':
                self.features_name = featuresAll[:int(self.nbfeatures)]
            else:
                self.features_name = featuresAll

        if self.val_loader is None:
            val_dataset, test_dataset = create_test_val_dataset(graph,
                                                                df_val,
                                                                df_test,
                                                                self.features_name,
                                                                self.target_name,
                                                                None,
                                                                self.device, self.ks,
                                                                self.horizon,
                                                                graph_mesh=None,
                                                                gridh2mesh=None,
                                                                mesh2graph=None)
            
            val_loader = DataLoader(val_dataset, val_dataset.__len__(), False, worker_init_fn=seed_worker, generator=g)
            test_loader = DataLoader(test_dataset, test_dataset.__len__(), False, worker_init_fn=seed_worker, generator=g)
            self.val_loader = val_loader
            self.test_loader = test_loader
        ##################################### Define percentage of 0 samples #########################################
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
                    if self.training_mode == 'splittraining':
                        best_combinaison = self.search_samples_proportion_per_cluster(graph, df_train, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT)
                        df_parts = []
                        for cluster, nb in best_combinaison:
                            df_cluster = df_train[df_train[self.federated_cluster] == cluster]
                            sampled = self.split_dataset(df_cluster, nb, reset=True)[cluster]
                            df_parts.append(sampled)

                        df_train = pd.concat(df_parts).reset_index(drop=True)
                    else:
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
                        print(best_tp, nb, y[y == 0].shape)
                        df_combined = self.split_dataset(df_train, nb, reset=False)
                        df_train['weight'] = 0
                        weight = egpd_trunc_discrete_weights(df_combined[self.target_name].values, df_combined['graph_id'].values)

                        # Mettre à jour df_train pour l'entraînement
                        df_train.loc[df_combined.index, 'weight'] = 1
                        #df_train.loc[df_combined.index, 'weight'] = weight
                        logger.info(f'Train mask df_train shape: {old_shape} -> {df_train.shape}')

        if 'smote' in self.over_sampling:
            
            if isinstance(self, ModelCNN) or isinstance(self, ModelGNN):
                raise ValueError(f'Smote is not adaptable to images or GNN')
            
            if self.ks > 0:
                raise ValueError(f'Smote is not adaptbale to time series')

            # Exemple : si ta cible est dans une colonne 'target'
            matching_ids_columns = [col for col in ids_columns if col != 'date']  # on exclut 'date' du matching

            # 2. Construire X (features + ids_columns), y (target)
            X_full = df_train[self.features_name + matching_ids_columns].copy()
            y = df_train[self.target_name].copy()

            smote_coef = int(self.over_sampling.split('-')[1])
            y_negative = y[y == 0].shape[0]
            
            y_one = min(y[y == 1].shape[0] * smote_coef, y_negative)
            y_two = min(y[y == 2].shape[0] * smote_coef, y_negative)
            y_three = min(y[y == 3].shape[0] * smote_coef, y_negative)
            y_four = min(y[y == 4].shape[0] * smote_coef, y_negative)

            if self.task_type in ['classification', 'ordinal-classification']:
                sampling_strategy = {
                    0: y_negative,
                    1: y_one,
                    2: y_two,
                    3: y_three,
                    4: y_four
                }
                smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
            
            elif self.task_type == 'binary':
                smote = SMOTE(random_state=42, sampling_strategy='auto')

            X_resampled, y_resampled = smote.fit_resample(X_full, y)

            df_train = X_resampled
            df_train[self.target_name] = y_resampled

            for uy in np.unique(df_train[self.target_name]):
                    print(f'Number of {uy} class : {df_train[df_train[self.target_name] == uy].shape}') 

        self.df_train = df_train
        self.df_test = df_test
        self.df_val = df_val

        print(df_train.shape, df_test.shape, df_val.shape)
        print(df_train[df_train['weight'] > 0].shape, df_val[df_val['weight'] > 0].shape)

        ##################################### Create loader #########################################        
        if False:
            train_dataset = read_object('train_dataset.pkl', self.dir_log)
            val_dataset = read_object('val_dataset.pkl', self.dir_log)
            test_dataset = read_object('test_dataset.pkl', self.dir_log)
        else:
            train_dataset = create_train_dataset(graph,
                                                df_train,
                                                self.features_name,
                                                self.target_name,
                                                None,
                                                self.device, self.ks, self.horizon,
                                                graph_mesh=None,
                                                gridh2mesh=None,
                                                mesh2graph=None)
    
            #save_object_torch(train_dataset, 'train_dataset.pkl', self.dir_log)
            #save_object_torch(val_dataset, 'val_dataset.pkl', self.dir_log)
            #save_object_torch(test_dataset, 'test_dataset.pkl', self.dir_log)

            train_loader = DataLoader(train_dataset, batch_size, True, worker_init_fn=seed_worker, generator=g)

            self.train_loader = train_loader

    def create_test_loader(self, graph, df):
        loader = create_test_loader(graph, df,
                       self.features_name,
                       self.device,
                       None,
                       self.target_name,
                       self.ks,
                       self.horizon,
                        graph_mesh=None,
                        gridh2mesh=None,
                        mesh2graph=None)

        return loader