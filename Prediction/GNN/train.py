from visualize import *

#################################### TREES ########################################


def train_sklearn_api_model(trainDataset, valDataset, testDataset,
          dir_output : Path,
          device : str,
          features : np.array,
          autoRegression : bool):
            
        Xtrain = trainDataset[0]
        Ytrain = trainDataset[1]

        Xval = valDataset[0]
        Yval = valDataset[1]

        Xtest = testDataset[0]
        Ytest = testDataset[1]

        Xtrain = Xtrain[Xtrain[:, 5] > 0]
        Ytrain = Ytrain[Ytrain[:, 5] > 0]

        Xval = Xval[Xval[:, 5] > 0]
        Yval = Yval[Yval[:, 5] > 0]

        Xtest = Xtest[Xtest[:, 5] > 0]
        Ytest = Ytest[Ytest[:, 5] > 0]

        logger.info(f'Xtrain shape : {Xtrain.shape}, Xval shape : {Xval.shape}, Xtest shape {testDataset[0].shape}, Ytrain shape {Ytrain.shape}, Yval shape : {Yval.shape},  Xtest shape {Xtest.shape}')

        ########################## Easy model ######################
        ################# xgboost ##################
        # xgboost 1
        Yw = Ytrain[:,-4]
        Y_train = Ytrain[:,-1]
        Y_val = Yval[:,-1]
        Y_test = Ytest[:, -1]
        W_test = Ytest[:, -4]
        fitparams={
                #'eval_set':[(Xtrain[:, features], Y_train), (Xval[:, features], Y_val)],
                #'sample_weight' : Yw,
                #'verbose' : False
                }
        
        model, grid = config_xgboost(device, False, 'reg:squarederror')
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, features,
            'xgboost', Model(model, 'rmse', 'xgboost'), fitparams, False, 'skip', grid, dir_output)

        model, grid = config_xgboost(device, False, 'reg:squaredlogerror')
        #fit(Xtrain, Y_train, Xtest, Y_test, W_test, features,
        #    'xgboost_rmlse', Model(model, 'rmsle', 'xgboost_rmlse'), fitparams, False, 'skip', grid, dir_output)
        
        # xgboost 2
        Yw = np.ones(Ytrain.shape[0])
        W_test = np.ones(Ytest.shape[0])
        fitparams={
                'eval_set':[(Xtrain[:, features], Y_train), (Xval[:, features], Y_val)],
                'sample_weight' : Yw,
                'verbose' : False
                }
        model, grid = config_xgboost(device, False, 'reg:squarederror')
        #fit(Xtrain, Y_train, Xtest, Y_test, W_test, features,
        #    'xgboost_features', Model(model, 'rmse', 'xgboost_features'), fitparams, True, 'skip', grid, dir_output)

        # xgboost 3
        Yw = Ytrain[:,-4]
        Yw[Ytrain[:,-2] == 0] = 1
        W_test = Ytest[:,-4]
        W_test[Ytest[:,-2] == 0] = 1
        Y_train = Ytrain[:,-2] > 0
        Y_val = Yval[:,-2] > 0
        Y_test = Ytest[:, -2] > 0
        fitparams={
                'eval_set':[(Xtrain[:, features], Y_train), (Xval[:, features], Y_val)],
                'sample_weight' : Yw,
                'verbose' : False
                }

        model, grid = config_xgboost(device, True, 'reg:logistic')
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, features,
            'xgboost_binary', Model(model, 'log_loss', 'xgboost_binary'), fitparams, False, 'skip', grid, dir_output)

        # xgboost 4
        Yw = np.ones(Y_train.shape[0])
        W_test = np.ones(Ytest.shape[0])
        fitparams={
                'eval_set':[(Xtrain[:, features], Y_train), (Xval[:, features], Y_val)],
                'sample_weight' : Yw,
                'verbose' : False
                }
        model, grid = config_xgboost(device, True, 'reg:logistic')
        #fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, 'xgboost_binary_unweighted',
        #    Model(model, 'rmse', 'xgboost_binary_unweighted'), fitparams, False, 'skip', grid, dir_output)

        ################ lightgbm ##################
        
        # lightgbm 1
        Yw = Ytrain[:,-4]
        Y_train = Ytrain[:,-1]
        Y_val = Yval[:,-1]
        W_test = Ytest[:,-4]
        W_test[Ytest[:,-2] == 0] = 1
        Y_test = Ytest[:, -1]
        fitparams={
                'eval_set':[(Xtrain[:, features], Y_train), (Xval[:, features], Y_val)],
                'sample_weight' : Yw,
                }
        model, grid = config_lightGBM(device, False, 'root_mean_squared_error')
        #fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, 'lightgbm', Model(model, 'rmse', 'lightgbm'), fitparams, False, 'skip', grid, dir_output)

        # lightgbm 2
        Yw = np.ones(Ytrain.shape[0])
        W_test = np.ones(Ytest.shape[0])
        fitparams={
                'eval_set':[(Xtrain[:, features], Y_train), (Xval[:, features], Y_val)],
                'sample_weight' : Yw,
                }
        model, grid = config_lightGBM(device, False, 'root_mean_squared_error')
        #fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, 'lightgbm_unweighted', Model(model, 'rmse', 'lightgbm_unweighted'),
        #   fitparams, False, 'skip', grid, dir_output)

        # lightgbm 3
        Yw = Ytrain[:,-4]
        Yw[Ytrain[:,-2] == 0] = 1
        W_test = Ytest[:, -4]
        W_test[Ytest[:,-2] == 0] = 1
        Y_train = Ytrain[:,-2] > 0
        Y_val = Yval[:,-2] > 0
        Y_test = Ytest[:, -2] > 0 
        fitparams={
                'eval_set':[(Xtrain[:, features], Y_train), (Xval[:, features], Y_val)],
                'sample_weight' : Yw,
                }
        model, grid = config_lightGBM(device, True, 'binary')
        #fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, 'lightgbm_binary', Model(model, 'log_loss', 'lightgbm_binary'),
        #    fitparams, False, 'skip', grid, dir_output)

        # lightgbm 4
        Yw = np.ones(Y_train.shape[0])
        fitparams={
                'eval_set':[(Xtrain[:, features], Y_train), (Xval[:, features], Y_val)],
                'sample_weight' : Yw,
                }
        model, grid = config_lightGBM(device, True, 'binary')
        #fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, 'lightgbm_binary_unweighted', Model(model, 'log_loss', 'lightgbm_binary_unweighted'),
        #    fitparams, False, 'skip', grid, dir_output)

        ################ ngboost ###################
        
        # ngboost 1
        Yw = Ytrain[:,-4]
        Y_train = Ytrain[:,-1]
        Y_val = Yval[:,-1]
        W_test = Ytest[:,-4]
        W_test[Ytest[:,-2] == 0] = 1
        Y_test = Ytest[:, -1]
        fitparams={
                'early_stopping_rounds':15,
                'sample_weight' : Yw,
                'X_val':Xval[:, features],
                'Y_val':Y_val,
                }
        model, grid = config_ngboost( False)
        #fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, 'ngboost', Model(model, 'rmse', 'ngboost'),
        #    fitparams, False, 'skip', grid, dir_output)

        # ngboost 2
        Yw = np.ones(Y_train.shape[0])
        W_test = np.ones(Ytest.shape[0])
        fitparams={
                'early_stopping_rounds':15,
                'sample_weight' : Yw,
                'X_val':Xval[:, features],
                'Y_val':Y_val,
                }
        model, grid = config_ngboost(False)
        #fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, 'ngboost_unweighted', Model(model, 'rmse', 'ngboost_unweighted'),
        #    fitparams, False, 'skip', grid, dir_output)

        ############################ Grid Search model ##################################

        # xgboost 1
        Yw = Ytrain[:,-4]
        Y_train = Ytrain[:,-1]
        Y_val = Yval[:,-1]
        W_test = Ytest[:,-4]
        W_test[Ytest[:,-2] == 0] = 1
        Y_test = Ytest[:, -1]
        fitparams={
                'eval_set':[(Xtrain[:, features], Y_train), (Xval[:, features], Y_val)],
                'sample_weight' : Yw,
                'verbose' : False
                }
        
        model, grid = config_xgboost(device, False, 'reg:squarederror')
        #fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, 'xgboost_grid_search', Model(model, 'rmse', 'xgboost_grid_search'),
        #    fitparams, False, 'grid', grid, dir_output)

        ############################# Features Model #######################################

        # xgboost 1
        Yw = Ytrain[:,-4]
        Y_train = Ytrain[:,-1]
        Y_val = Yval[:,-1]
        W_test = Ytest[:,-4]
        W_test[Ytest[:,-2] == 0] = 1
        Y_test = Ytest[:, -1]
        fitparams={
                'eval_set':[(Xtrain[:, features], Y_train), (Xval[:, features], Y_val)],
                'sample_weight' : Yw,
                'verbose' : False
                }
        
        model, grid = config_xgboost(device, False, 'reg:squarederror')
        #fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, 'xgboost_features', Model(model, 'rmse', 'xgboost'),
        #    fitparams, True, 'skip', grid, dir_output)
    
def fit(Xtrain, y, Xtest, Ytest, Wtest, features,
        name, model, fitparams, evaluate_individual_features,
        parameter_optimization_method, grid, dir_output):
    
    save_object(features, 'features.pkl', dir_output / name)

    logger.info(f'Fitting model {name}')
    model.fit(X=Xtrain[:, features], y=y,
              X_test=Xtest[:, features], y_test=Ytest, w_test=Wtest,
                evaluate_individual_features=evaluate_individual_features,
                optimization=parameter_optimization_method,
                param_grid=grid, fit_params=fitparams)

    check_and_create_path(dir_output / name)
    save_object(model, name + '.pkl', dir_output / name)

    """pred = model.predict(Xtest)
    
    realVspredict(pred, Ytest.reshape(-1,1), -1,
                      dir_output / name, 'raw')"""

    logger.info(f'Model score {model.score(Xtest, Ytest, Wtest)}')

def train_break_point(Xset : np.array, Yset : np.array, features : list, dir_output : Path, pos_feature : dict, scale : int, n_clusters : int):
    check_and_create_path(dir_output)
    logger.info('###################" Calculate break point ####################')
    res_cluster = {}
    features_selected_2 = []

    # For each feature
    for fet in features:
        train_fet_num = []
        check_and_create_path(dir_output / fet)
        # Select the numerical values
        maxi, methods, variales = calculate_feature_range(fet, scale)
        logger.info(f'{fet}, {variales}, {methods}')
        train_fet_num += list(np.arange(pos_feature[fet], pos_feature[fet] + maxi))
        len_met = len(methods)
        
        i_var = -1
        # For each method
        for i, xs in enumerate(train_fet_num):
            #if xs not in features_selected:
            #    continue

            if len_met == len_met:
                i_met = i % len_met
            else:
                i_met = 0

            if i_met % len_met == 0:
                i_var += 1

            on = fet+'_'+variales[i_var]+'_'+methods[i_met] # Get the right name
            res_cluster[xs] = {}
            res_cluster[xs]['name'] = on
            res_cluster[xs]['fet'] = fet
            X_fet = Xset[:, xs] # Select the feature
            if np.unique(X_fet).shape[0] < n_clusters:
                continue
            X_cluster = np.copy(X_fet)

            # Create the cluster model
            model = Predictor(n_clusters, on)
            model.fit(X_cluster)

            save_object(model, on+'.pkl', dir_output / fet)

            pred = order_class(model, model.predict(X_cluster)) # Predict and order class to 0 1 2 3 4

            cls = np.unique(pred)
            if cls.shape[0] != n_clusters:
                continue

            plt.figure(figsize=(5,5)) # Create figure for ploting

            # For each class calculate the target values
            for c in cls:
                mask = np.argwhere(pred == c)[:,0]
                res_cluster[xs][c] = np.mean(Yset[mask, -2])
                plt.scatter(X_fet[mask], Yset[mask,-2], label=c)
            
            logger.info(on)

            # Calculate Pearson coefficient
            df = pd.DataFrame(res_cluster, index=np.arange(n_clusters)).reset_index()
            df.rename({'index': 'class'}, axis=1, inplace=True)
            df = df[['class', xs]]
            correlation = df['class'].corr(df[xs])
            res_cluster[xs]['correlation'] = correlation
            # Plot
            plt.ylabel('Sinister')
            plt.xlabel(fet+'_'+methods[i_met])
            plt.title(fet+'_'+methods[i_met])
            on += '.png'
            plt.legend()
            plt.savefig(dir_output / fet / on)
            plt.close('all')

            # If no correlation then pass
            if abs(correlation) > 0.7:
                # If negative linearity, inverse class
                if correlation < 0:
                    new_class = np.flip(np.arange(n_clusters))
                    temp = {}
                    for i, nc in enumerate(new_class):
                        temp[nc] = res_cluster[xs][i]
                    temp['correlation'] = res_cluster[xs]['correlation']
                    temp['name'] = res_cluster[xs]['name']
                    temp['fet'] = res_cluster[xs]['fet']
                    res_cluster[xs] = temp
                
                # Add in selected features
                features_selected_2.append(xs)
        
    logger.info(len(features_selected_2))
    save_object(res_cluster, 'break_point_dict.pkl', dir_output)
    save_object(features_selected_2, 'features.pkl', dir_output)

################################ DEEP LEARNING #########################################

def launch_loader(model, loader, type,
                  features, binary,
                  criterion, optimizer,
                  autoRegression,
                  pos_feature):

    if type == 'train':
        model.train()
    else:
        model.eval()
        prev_input = torch.Tensor([], device=device)

    for i, data in enumerate(loader, 0):

        inputs, labels, edges = data
        
        if not binary:
            target = labels[:,-1]
        else:
            target = (labels[:,-2] > 0).long()

        weights = labels[:,-4]

        if autoRegression and type != 'train':
            inode = inputs[:, 0]
            idate = inputs[:, 4]
            if inode in prev_input[:, 0] and idate - 1 in prev_input[:, 4]:
                mask = torch.argwhere((prev_input[:, 0] == inode) & (prev_input[:, 4] == idate - 1))
                inputs[:, pos_feature['AutoRegressionReg']] = prev_input[mask, pos_feature['AutoRegressionReg']]
            else:
                inputs[:, pos_feature['AutoRegressionReg']] = 0

        inputs = inputs[:, features]
        output = model(inputs, edges)

        if autoRegression and type != 'train':
            inputs[:, pos_feature['AutoRegressionReg']] = output
            prev_input = torch.cat((prev_input, inputs))

        target = torch.masked_select(target, weights.gt(0))
        weights = torch.masked_select(weights, weights.gt(0))

        if not binary:
            target = target.view(output.shape)
            weights = weights.view(output.shape)
            loss = criterion(output, target, weights)
        else:
            loss = criterion(output, target)

        if type == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss

def func_epoch(model, trainLoader, valLoader, features,
               optimizer, criterion, binary,
                autoRegression,
                pos_feature):
    
    launch_loader(model, trainLoader, 'train', features, binary, criterion, optimizer,  autoRegression,
                  pos_feature)

    with torch.no_grad():
        loss = launch_loader(model, valLoader, 'val', features, binary, criterion, optimizer,  autoRegression,
                  pos_feature)

    return loss.items()

def compute_optimisation_features(modelname, lr, scale, pos_feature, trainLoader, valLoader, testLoader, criterion, binary, epochs, PATIENCE_CNT,
                                autoRegression):
    newFet = [features[0]]
    BEST_VAL_LOSS_FET = math.inf
    patience_cnt_fet = 0
    for fi, fet in enumerate(1, features):
        testFet = copy(newFet)
        testFet.append(fet)
        dico_model = make_models(len(testFet), 52, 0.03, 'relu')
        model = dico_model[modelname]
        optimizer = optim.Adam(model.parameters(), lr=lr)
        BEST_VAL_LOSS = math.inf
        BEST_MODEL_PARAMS = None
        patience_cnt = 0
        logger.info(f'Train {model} with')
        log_features(testFet, scale, pos_feature, ["min", "mean", "max", "std"])
        for epoch in tqdm(range(epochs)):
            loss = func_epoch(model, trainLoader, valLoader, testFet, optimizer, criterion, binary,  autoRegression,
                  pos_feature)
            if loss.item() < BEST_VAL_LOSS:
                BEST_VAL_LOSS = loss.item()
                BEST_MODEL_PARAMS = model.state_dict()
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE_CNT:
                    logger.info(f'Loss has not increased for {patience_cnt} epochs. Last best val loss {BEST_VAL_LOSS}, current val loss {loss.item()}')
            
            with torch.no_grad():
                loss = launch_loader(model, testLoader, 'test', features, binary, criterion, optimizer,  autoRegression,
                  pos_feature)

            if loss.item() < BEST_VAL_LOSS_FET:
                logger.info(f'{loss.item()} < {BEST_VAL_LOSS_FET}, we add the feature')
                BEST_VAL_LOSS_FET = loss.item()
                newFet.append(fet)
                patience_cnt_fet = 0
            else:
                patience_cnt_fet += 1
                if patience_cnt_fet >= 30:
                    logger.info('Loss has not increased for 30 epochs')
                break
    return newFet

def train(trainLoader, valLoader, testLoader,
          scale: int,
          optmize_feature : bool,
          PATIENCE_CNT : int,
          CHECKPOINT: int,
          lr : float,
          epochs : int,
          criterion,
          pos_feature : dict,
          modelname : str,
          features : np.array,
          dir_output : Path,
          binary : bool,
          autoRegression : bool) -> None:
        """
        Train neural network model
        PATIENCE_CNT : early stopping count
        CHECKPOINT : logger.info last train/val loss
        lr : learning rate
        epochs : number of epochs for training
        criterion : loss function
        """
        assert trainLoader is not None and valLoader is not None

        check_and_create_path(dir_output)

        if optmize_feature:
            features = compute_optimisation_features(modelname, lr, scale, pos_feature, trainLoader, valLoader, testLoader, criterion, binary, epochs, PATIENCE_CNT, autoRegression)
            
        dico_model = make_models(len(features), 52, 0.03, 'relu')
        model = dico_model[modelname]
        optimizer = optim.Adam(model.parameters(), lr=lr)
        BEST_VAL_LOSS = math.inf
        BEST_MODEL_PARAMS = None
        patience_cnt = 0

        logger.info('Train model with')
        log_features(features, scale, pos_feature, ["min", "mean", "max", "std"])
        save_object(features, 'features.pkl', dir_output)
        for epoch in tqdm(range(epochs)):
            loss = func_epoch(model, trainLoader, valLoader, features, optimizer, criterion, binary)
            if epoch % CHECKPOINT == 0:
                logger.info(f'epochs {epoch}, Train loss {loss.item()}')
                save_object_torch(model.state_dict(), str(epoch)+'.pt', dir_output)
            if loss.item() < BEST_VAL_LOSS:
                BEST_VAL_LOSS = loss.item()
                BEST_MODEL_PARAMS = model.state_dict()
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE_CNT:
                    logger.info(f'Loss has not increased for {patience_cnt} epochs. Last best val loss {BEST_VAL_LOSS}, current val loss {loss.item()}')
                    save_object_torch(model.state_dict(), 'last.pt', dir_output)
                    save_object_torch(BEST_MODEL_PARAMS, 'best.pt', dir_output)
                    return
                
            if epoch % CHECKPOINT == 0:
                logger.info(f'epochs {epoch}, Best val loss {BEST_VAL_LOSS}')

#############################  PCA ###################################

def train_pca(X, percent, dir_output, features_selected):
    pca =  PCA(n_components='mle', svd_solver='full')
    components = pca.fit_transform(X[:, features_selected])
    check_and_create_path(dir_output / 'pca')
    show_pcs(pca, 2, components, dir_output / 'pca')
    print('99 % :',find_n_component(0.99, pca))
    print('95 % :', find_n_component(0.95, pca))
    print('90 % :' , find_n_component(0.90, pca))
    pca_number = find_n_component(percent, pca)
    save_object(pca, 'pca.pkl', dir_output)
    pca =  PCA(n_components=pca_number, svd_solver='full')
    pca.fit(X[:, features_selected])
    return pca, pca_number

def apply_pca(X, pca, pca_number, features_selected):
    res = pca.transform(X[:, features_selected])
    new_X = np.empty((X.shape[0], 6 + pca_number))
    new_X[:, :6] = X[:, :6]
    new_X[:, 6:] = res
    return new_X