from visualize import *
from sklearn.linear_model import LinearRegression, LogisticRegression


############################################## Some tools ###############################################

def switch_target(target, y_train, y_val, y_test, use_weight):
    if target == 'risk':
        Y_train = y_train[:,-1]
        W_train = y_train[:,-4]
        Y_val = y_val[:,-1]
        Y_test = y_test[:,-1]
        W_test = y_test[:,-4]
    elif target == 'nbsinister':
        Y_train = y_train[:,-2]
        W_train = y_train[:,-4]
        Y_val = y_val[:,-2]
        Y_test = y_test[:,-2]
        W_test = y_test[:,-4]
    elif target == 'binary':
        Y_train = y_train[:,-2] > 0
        W_train = y_train[:,-4]
        W_train[Y_train == 0] = 1
        Y_val = y_val[:,-2] > 0
        Y_test = y_test[:,-2] > 0
        W_test = y_test[:,-4]
        W_test[Y_test == 0] = 1

    if not use_weight:
        W_train = np.ones(W_train.shape[0])
        W_test = np.ones(W_test.shape[0])

    return Y_train, W_train, Y_val, Y_test, W_test

def get_model_and_fit_params(x_train, Y_train, x_val, Y_val, W_train,
                      features, name, model_type, task_type, 
                      params, loss):
    
    if model_type == 'xgboost':
        fit_params = {
                'eval_set': [(x_train[:, features], Y_train), (x_val[:, features], Y_val)],
                'sample_weight': W_train,
                'verbose': False
         }
        
    elif model_type == 'ngboost':
        fit_params = {
            'X_val': x_val[:, features],
            'Y_val': Y_val,
            'sample_weight': W_train,
            'early_stopping_rounds': 100,
            'monitor': None
        }
    elif model_type == 'linear':
        fit_params = {}
    else:
        raise ValueError(f"Unsupported model model_type: {model_type}")
    
    model = get_model(model_type=model_type, name=name, device=device, task_type=task_type, params=params, loss=loss)
    return model, fit_params

############################################### SKLEARN ##################################################

def fit(params):
    """
    Entraîne et évalue un modèle en utilisant les paramètres fournis.

    Parameters:
    ----------
    params : dict
        Un dictionnaire contenant les paramètres nécessaires pour l'entraînement et l'évaluation du modèle.

        Clés attendues dans le dictionnaire :
        - x_train : np.array
        - y : np.array
        - x_test : np.array
        - y_test : np.array
        - w_test : np.array
        - features : np.array
        - features_name : list
        - name : str
        - model : object
        - fit_params : dict
        - parameter_optimization_method : str
        - grid_params : dict
        - dir_output : Path
    """
    required_keys = [
        'x_train', 'y', 'x_test', 'y_test', 'w_test', 'features', 'features_name',
        'name', 'model', 'fit_params', 'parameter_optimization_method',
        'grid_params', 'dir_output'
    ]

    for key in required_keys:
        assert key in params, f"Missing required parameter: {key}"

    x_train = params['x_train']
    y = params['y']
    x_test = params['x_test']
    y_test = params['y_test']
    w_test = params['w_test']
    features = params['features']
    features_name = params['features_name']
    name = params['name']
    model = params['model']
    fit_params = params['fit_params']
    parameter_optimization_method = params['parameter_optimization_method']
    grid_params = params['grid_params']
    dir_output = params['dir_output']

    save_object(features, 'features.pkl', dir_output / name)
    logger.info(f'Fitting model {name}')
    model.fit(X=x_train[:, features], y=y,
              optimization=parameter_optimization_method,
              grid_params=grid_params, fit_params=fit_params)

    check_and_create_path(dir_output / name)
    save_object(model, name + '.pkl', dir_output / name)
    logger.info(f'Model score {model.score(x_test[:, features], y_test, w_test)}')

    create_feature_map(features=features, features_name=features_name, dir_output=dir_output / name)

    """if isinstance(model, Model):
        if name.find('features') != -1:
            logger.info('Features importance')
            model.plot_features_importance(x_train[:, features], y, [features_name[i] for i in features], 'train', dir_output / name, mode='bar')
            model.plot_features_importance(x_test[:, features], y_test, [features_name[i] for i in features], 'test', dir_output / name, mode='bar')

        if name.find('search') != -1:
            model.plot_param_influence('max_depth', dir_output / name)

    if isinstance(model, ModelTree):
        model.plot_tree(features_name=features_name, outname=name, dir_output= dir_output / name, figsize=(50,25))"""

def train_sklearn_api_model(params):
    """
    Entraîne un modèle en utilisant les paramètres fournis.

    Parameters:
    ----------
    params : dict
        Un dictionnaire contenant les paramètres nécessaires pour l'entraînement du modèle.
        Les clés requises dans ce dictionnaire sont :

        - x_train : np.array
            Les données d'entraînement, contenant les features (caractéristiques) à utiliser pour entraîner le modèle.

        - y_train : np.array
            Les étiquettes (labels) correspondantes aux données d'entraînement.

        - x_val : np.array
            Les données de validation, utilisées pour valider le modèle pendant l'entraînement.

        - y_val : np.array
            Les étiquettes correspondantes aux données de validation.

        - x_test : np.array
            Les données de test, utilisées pour tester les performances finales du modèle.

        - y_test : np.array
            Les étiquettes correspondantes aux données de test.

        - features : list
            La liste des indices des features à utiliser pour l'entraînement.

        - target : str
            Le nom de la cible (target) que le modèle essaie de prédire. Peut être `risk`, `nbsinister` ou `binary`.
            Example: `target = 'risk'`

        - features_name : list
            Une list contenant les noms des features.

        - optimize_feature : bool
            Indicateur pour savoir si une optimisation des features doit être réalisée avant l'entraînement.
            Example: `optimize_feature = True`

        - dir_output : Path
            Le chemin du répertoire où les résultats et les modèles seront sauvegardés.
            Example: `dir_output = Path('./output')`

        - device : str
            Le dispositif à utiliser pour l'entraînement (`cpu` ou `cuda`).
            Example: `device = 'cpu'`

        - do_grid_search : bool
            Indicateur pour savoir si une recherche de grille (GridSearch) doit être effectuée pour l'optimisation des hyperparamètres.
            Example: `do_grid_search = True`

        - do_bayes_search : bool
            Indicateur pour savoir si une recherche bayésienne (BayesSearch) doit être effectuée pour l'optimisation des hyperparamètres.
            Example: `do_bayes_search = False`

        - task_type : str
            Le model_type de tâche (`regression` ou `classification`).
            Example: `task_type = 'regression'`

        - loss : str
            La fonction de perte à utiliser pour l'entraînement (par exemple, `rmse`, `log_loss`).
            Example: `loss = 'rmse'`

        - model_type : str
            Le model_type de modèle à entraîner (par exemple, `xgboost`, `lightgbm`, `ngboost`).
            Example: `model_type = 'xgboost'`

        - name : str
            Le nom du modèle, utilisé pour l'identifier et sauvegarder les résultats.
            Example: `name = 'xgboost_risk_regression_rmse'`

        - model_params : dict
            Les paramètres du modèle, spécifiques au model_type de modèle utilisé.
            Example: `model_params = {'objective': 'reg:squarederror', 'learning_rate': 0.01, 'n_estimators': 10000}`

        - grid_params : dict
            La grille des hyperparamètres à explorer lors de l'optimisation.
            Example: `grid_params = {'max_depth': [2, 5, 10], 'learning_rate': [0.01, 0.1]}`
    """

    required_params = [
        'x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test', 'features', 'target',
        'features_name', 'optimize_feature', 'dir_output', 'device',
        'do_grid_search', 'do_bayes_search', 'task_type', 'loss', 'model_type', 'name', 'model_params', 'grid_params'
    ]

    # Ensure all required parameters are present
    for param in required_params:
        assert param in params, f"Missing required parameter: {param}"

    x_train = params['x_train']
    y_train = params['y_train']
    x_val = params['x_val']
    y_val = params['y_val']
    x_test = params['x_test']
    y_test = params['y_test']
    features = params['features']
    target = params['target']
    features_name = params['features_name']
    optimize_feature = params['optimize_feature']
    dir_output = params['dir_output']
    device = params['device']
    do_grid_search = params['do_grid_search']
    do_bayes_search = params['do_bayes_search']
    task_type = params['task_type']
    loss = params['loss']
    model_type = params['model_type']
    name = params['name']
    model_params = params['model_params']
    grid_params = params['grid_params']

    Y_train, W_train, Y_val, Y_test, W_test = switch_target(target=target, y_train=y_train, y_val=y_val, y_test=y_test, use_weight=True)

    model, fit_params = get_model_and_fit_params(x_train, Y_train, x_val, Y_val, W_train,
                                        features, name, model_type, task_type, 
                                        model_params, loss)

    fit_params_dict = {
        'x_train': x_train,
        'y': Y_train,
        'x_test': x_test,
        'y_test': Y_test,
        'w_test': W_test,
        'features': features,
        'features_name': features_name,
        'name': f'{name}',
        'model': model,
        'fit_params': fit_params,
        'parameter_optimization_method': 'skip',
        'grid_params': grid_params,
        'dir_output': dir_output
    }

    fit(fit_params_dict)
    
    if optimize_feature:
        model = get_model(model_type=model_type, name=f'{name}_features', device=device, task_type=task_type, params=model_params, loss=loss)
        relevant_features = explore_features(model=model,
                                                features=features,
                                                X=x_train, y=Y_train, w=W_train, X_val=x_val, y_val=Y_val, w_val=None,
                                                X_test=x_test, y_test=Y_test, w_test=W_test)

        log_features(relevant_features, features_name)

        save_object(relevant_features, f'relevant_features_{target}.pkl', dir_output)
    else:
        relevant_features = read_object(f'relevant_features_{target}.pkl', dir_output)
        if relevant_features is None:
            logger.info('No relevant features found, train with all features')
            relevant_features = features
    
    model, fit_params = get_model_and_fit_params(x_train, Y_train, x_val, Y_val, W_train,
                                        relevant_features, f'{name}_features', model_type, task_type, 
                                        model_params, loss)
    
    fit_params_dict = {
        'x_train': x_train,
        'y': Y_train,
        'x_test': x_test,
        'y_test': Y_test,
        'w_test': W_test,
        'features': relevant_features,
        'features_name': features_name,
        'name': f'{name}_features',
        'model': model,
        'fit_params': fit_params,
        'parameter_optimization_method': 'skip',
        'grid_params': grid_params,
        'dir_output': dir_output
    }

    fit(fit_params_dict)

    if do_grid_search:
        model = get_model(model_type=model_type, name=f'{name}_grid_search', device=device, task_type=task_type, params=model_params, loss=loss)
        fit_params_dict['name'] = f'{name}_grid_search'
        fit_params_dict['model'] = model
        fit_params_dict['parameter_optimization_method'] = 'grid'

        fit(fit_params_dict)    

    if do_bayes_search:
        model = get_model(model_type=model_type, name=f'{name}_bayes_search', device=device, task_type=task_type, params=model_params, loss=loss)
        fit_params_dict['name'] = f'{name}_bayes_search'
        fit_params_dict['model'] = model
        fit_params_dict['parameter_optimization_method'] = 'bayes'

        fit(fit_params_dict)

def train_xgboost(params):
    """
    Train xgboost model
    """
    x_train = params['x_train']
    y_train = params['y_train']
    x_val = params['x_val']
    y_val = params['y_val']
    x_test = params['x_test']
    y_test = params['y_test']
    features = params['features']
    features_name = params['features_name']
    optimize_feature = params['optimize_feature']
    dir_output = params['dir_output']
    device = params['device']
    do_grid_search = params['do_grid_search']
    do_bayes_search = params['do_bayes_search']

    targets = ['risk', 'nbsinister', 'binary']

    for target in targets:
        if target == 'risk':
            loss = 'rmse'
            task_type = 'regression'
            objective = 'reg:squarederror'
        elif target == 'nbsinister':
            loss = 'rmse'
            task_type = 'regression'
            objective = 'reg:squarederror'
        else:
            loss = 'log_loss'
            task_type = 'classification'
            objective = 'reg:logistic'

        name = f'xgboost_{target}_{task_type}_{loss}'
        model_params = {
            'objective': objective,
            'verbosity': 0,
            'early_stopping_rounds': None,
            'learning_rate': 0.01,
            'min_child_weight': 1.0,
            'max_depth': 6,
            'max_delta_step': 1.0,
            'subsample': 0.3,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'reg_lambda': 1.0,
            'reg_alpha': 0.9,
            'n_estimators': 10000,
            'random_state': 42,
            'tree_method': 'hist',
        }
        grid_params = {'max_depth': [2, 5, 6, 8, 10, 15]}

        train_sklearn_api_model({
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test,
            'features': features,
            'target': target,
            'features_name': features_name,
            'optimize_feature': optimize_feature,
            'dir_output': dir_output,
            'device': device,
            'do_grid_search': do_grid_search,
            'do_bayes_search': do_bayes_search,
            'task_type': task_type,
            'loss': loss,
            'model_type': 'xgboost',
            'name': name,
            'model_params': model_params,
            'grid_params': grid_params,
        })

def train_ngboost(params):
    model_type = 'ngboost'
    target = 'risk'
    loss = 'rmse'
    task_type = 'regression'
    name = f'{model_type}_{target}_{task_type}_{loss}'
    model_params = {
        'Dist': 'normal',
        'Score': 'MLE',
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'random_state': 42,
    }
    grid_params = {
        'learning_rate': [0.01, 0.1, 0.2],
    }

    train_sklearn_api_model({
        'x_train': params['x_train'],
        'y_train': params['y_train'],
        'x_val': params['x_val'],
        'y_val': params['y_val'],
        'x_test': params['x_test'],
        'y_test': params['y_test'],
        'features': params['features'],
        'target': target,
        'features_name': params['features_name'],
        'optimize_feature': params['optimize_feature'],
        'dir_output': params['dir_output'],
        'device': params['device'],
        'do_grid_search': params['do_grid_search'],
        'do_bayes_search': params['do_bayes_search'],
        'task_type': task_type,
        'loss': loss,
        'model_type': model_type,
        'name': name,
        'model_params': model_params,
        'grid_params': grid_params,
    })


def train_lightgbm(params):
    model_type = 'lightgbm'
    target = 'risk'
    loss = 'rmse'
    task_type = 'regression'
    name = f'{model_type}_{target}_{task_type}_{loss}'
    model_params = {
        'objective': 'regression',
        'learning_rate': 0.01,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'max_depth': -1,
        'verbosity': -1,
        'metric': 'rmse',
        'random_state': 42,
    }
    grid_params = {
        'num_leaves': [31, 127],
        'learning_rate': [0.01, 0.1],
    }

    train_sklearn_api_model({
        'x_train': params['x_train'],
        'y_train': params['y_train'],
        'x_val': params['x_val'],
        'y_val': params['y_val'],
        'x_test': params['x_test'],
        'y_test': params['y_test'],
        'features': params['features'],
        'target': target,
        'features_name': params['features_name'],
        'optimize_feature': params['optimize_feature'],
        'dir_output': params['dir_output'],
        'device': params['device'],
        'do_grid_search': params['do_grid_search'],
        'do_bayes_search': params['do_bayes_search'],
        'task_type': task_type,
        'loss': loss,
        'model_type': model_type,
        'name': name,
        'model_params': model_params,
        'grid_params': grid_params,
    })


def train_svm(params):
    model_type = 'svm'
    target = 'risk'
    loss = 'rmse'
    task_type = 'regression'
    name = f'{model_type}_{target}_{task_type}_{loss}'
    model_params = {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'random_state': 42,
    }
    grid_params = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
    }

    train_sklearn_api_model({
        'x_train': params['x_train'],
        'y_train': params['y_train'],
        'x_val': params['x_val'],
        'y_val': params['y_val'],
        'x_test': params['x_test'],
        'y_test': params['y_test'],
        'features': params['features'],
        'target': target,
        'features_name': params['features_name'],
        'optimize_feature': params['optimize_feature'],
        'dir_output': params['dir_output'],
        'device': params['device'],
        'do_grid_search': params['do_grid_search'],
        'do_bayes_search': params['do_bayes_search'],
        'task_type': task_type,
        'loss': loss,
        'model_type': model_type,
        'name': name,
        'model_params': model_params,
        'grid_params': grid_params,
    })


def train_rf(params):
    model_type = 'rf'
    target = 'risk'
    loss = 'rmse'
    task_type = 'regression'
    name = f'{model_type}_{target}_{task_type}_{loss}'
    model_params = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
    }
    grid_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, None],
    }

    train_sklearn_api_model({
        'x_train': params['x_train'],
        'y_train': params['y_train'],
        'x_val': params['x_val'],
        'y_val': params['y_val'],
        'x_test': params['x_test'],
        'y_test': params['y_test'],
        'features': params['features'],
        'target': target,
        'features_name': params['features_name'],
        'optimize_feature': params['optimize_feature'],
        'dir_output': params['dir_output'],
        'device': params['device'],
        'do_grid_search': params['do_grid_search'],
        'do_bayes_search': params['do_bayes_search'],
        'task_type': task_type,
        'loss': loss,
        'model_type': model_type,
        'name': name,
        'model_params': model_params,
        'grid_params': grid_params,
    })

def train_decision_tree(params):
    model_type = 'decision_tree'
    target = 'risk'
    loss = 'rmse'
    task_type = 'regression'
    name = f'{model_type}_{target}_{task_type}_{loss}'
    model_params = {
        'criterion': 'mse',
        'splitter': 'best',
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
    }
    grid_params = {
        'max_depth': [2, 5, 10, None],
        'criterion': ['mse', 'mae'],
    }

    train_sklearn_api_model({
        'x_train': params['x_train'],
        'y_train': params['y_train'],
        'x_val': params['x_val'],
        'y_val': params['y_val'],
        'x_test': params['x_test'],
        'y_test': params['y_test'],
        'features': params['features'],
        'target': target,
        'features_name': params['features_name'],
        'optimize_feature': params['optimize_feature'],
        'dir_output': params['dir_output'],
        'device': params['device'],
        'do_grid_search': params['do_grid_search'],
        'do_bayes_search': params['do_bayes_search'],
        'task_type': task_type,
        'loss': loss,
        'model_type': model_type,
        'name': name,
        'model_params': model_params,
        'grid_params': grid_params,
    })

def wrapped_train_sklearn_api_model(train_dataset, val_dataset, test_dataset,
                            dir_output: Path,
                            device: str,
                            features: np.array,
                            features_name: dict,
                            autoRegression: bool,
                            optimize_feature: bool,
                            do_grid_search: bool,
                            do_bayes_search: bool):
    x_train = train_dataset[0]
    y_train = train_dataset[1]

    x_val = val_dataset[0]
    y_val = val_dataset[1]

    x_test = test_dataset[0]
    y_test = test_dataset[1]

    x_train = x_train[x_train[:, 5] > 0]
    y_train = y_train[y_train[:, 5] > 0]

    x_val = x_val[x_val[:, 5] > 0]
    y_val = y_val[y_val[:, 5] > 0]

    x_test = x_test[x_test[:, 5] > 0]
    y_test = y_test[y_test[:, 5] > 0]

    logger.info(f'x_train shape: {x_train.shape}, x_val shape: {x_val.shape}, x_test shape: {test_dataset[0].shape}, y_train shape: {y_train.shape}, y_val shape: {y_val.shape}, x_test shape: {x_test.shape}')

    params = {
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test': x_test,
        'y_test': y_test,
        'features': features,
        'features_name': features_name,
        'optimize_feature': optimize_feature,
        'dir_output': dir_output,
        'device': device,
        'do_grid_search': do_grid_search,
        'do_bayes_search': do_bayes_search,
    }

    train_xgboost(params)
    #train_lightgbm(params)
    #train_ngboost(params)
    #train_rf(params)
    #train_svm(params)
    #train_decision_tree(params)

############################################################# Fusion ######################################################################

def wrapped_train_sklearn_api_fusion_model(train_dataset, val_dataset,
                                           test_dataset,
                                           train_dataset_list : list,
                                           val_dataset_list : list,
                                           test_dataset_list : list,
                                           model_list : list,
                                           dir_model_list : list,
                            dir_output: Path,
                            device: str,
                            features_name: dict,
                            autoRegression: bool,
                            optimize_feature: bool,
                            do_grid_search: bool,
                            do_bayes_search: bool,
                            deep : bool):
    
    params_list = []
    
    for i, train_dataset in enumerate(train_dataset_list):
        x_train = train_dataset[0]
        y_train = train_dataset[1]

        x_val = val_dataset_list[i][0]
        y_val = val_dataset_list[i][1]

        x_test = test_dataset_list[i][0]
        y_test = test_dataset_list[i][1]

        train_dataset_list[i][0] = x_train[x_train[:, 5] > 0]
        train_dataset_list[i][1] = y_train[y_train[:, 5] > 0]

        val_dataset_list[i][0] = x_val[x_val[:, 5] > 0]
        val_dataset_list[i][1] = y_val[y_val[:, 5] > 0]

        test_dataset_list[i][0] = x_test[x_test[:, 5] > 0]
        test_dataset_list[i][1] = y_test[y_test[:, 5] > 0]

        model_name = model_list[i]
        dir_model = dir_model_list[i]

        model_type, target, task_type, loss  = model_name.split('_')
        
        params = {
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test': x_test,
        'y_test': y_test,
        'features': features,
        'features_name': features_name,
        'optimize_feature': optimize_feature,
        'dir_output': dir_model,
        'device': device,
        'do_grid_search': do_grid_search,
        'do_bayes_search': do_bayes_search,
        'model_type': model_type,
        'name' : model_name,
        'task_type' : task_type,
        'loss' : loss
        }

        params_list.append(params)

    x_train = train_dataset[0]
    y_train = train_dataset[1]

    x_val = val_dataset[0]
    y_val = val_dataset[1]

    x_test = test_dataset[0]
    y_test = test_dataset[1]

    x_train = x_train[x_train[:, 5] > 0]
    y_train = y_train[y_train[:, 5] > 0]

    x_val = x_val[x_val[:, 5] > 0]
    y_val = y_val[y_val[:, 5] > 0]

    x_test= x_test[x_test[:, 5] > 0]
    y_test = y_test[y_test[:, 5] > 0]

    params_fusion = {
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test': x_test,
        'y_test': y_test,
        'features': features,
        'features_name': features_name,
        'optimize_feature': optimize_feature,
        'dir_output': dir_output,
        'device': device,
        'do_grid_search': do_grid_search,
        'do_bayes_search': do_bayes_search,
        }
    
    targets = ['risk', 'nbsinister', 'binary']

    for target in targets:
        if target == 'risk':
            loss = 'rmse'
            task_type = 'regression'
        elif target == 'nbsinister':
            loss = 'rmse'
            task_type = 'regression'
        else:
            loss = 'log_loss'
            task_type = 'classification'

        ####################### Linear ############################
        params_fusion['model_type'] = 'linear'
        params_fusion['name'] = 'linear_fusion'
        params_fusion['parameter_optimization_method'] = 'skip'
        params_fusion['fit_params'] = 'skip'
        params_fusion['grid_params'] = {}
        params_fusion['deep'] = deep

        train_sklearn_api_model_fusion(params_list, params_fusion)
    
def train_sklearn_api_model_fusion(params_list, params_fusion):
    """
    Entraîne plusieurs modèles en utilisant les ensembles de paramètres fournis.

    Parameters:
    ----------
    params : dict
        Contient les paramètres du modèle de fusion. Même clefs que params_list
    params_list : list of dict
        Une liste de dictionnaires contenant les paramètres nécessaires pour l'entraînement de chaque modèle.
        Chaque dictionnaire dans la liste doit contenir les clés suivantes :

        - x_train : np.array
            Les données d'entraînement, contenant les features (caractéristiques) à utiliser pour entraîner le modèle.
            Example: `x_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])`

        - y_train : np.array
            Les étiquettes (labels) correspondantes aux données d'entraînement.
            Example: `y_train = np.array([0, 1, 1])`

        - x_val : np.array
            Les données de validation, utilisées pour valider le modèle pendant l'entraînement.
            Example: `x_val = np.array([[1, 2, 3], [4, 5, 6]])`

        - y_val : np.array
            Les étiquettes correspondantes aux données de validation.
            Example: `y_val = np.array([0, 1])`

        - x_test : np.array
            Les données de test, utilisées pour tester les performances finales du modèle.
            Example: `x_test = np.array([[1, 2, 3], [4, 5, 6]])`

        - y_test : np.array
            Les étiquettes correspondantes aux données de test.
            Example: `y_test = np.array([0, 1])`

        - features : list
            La liste des indices des features à utiliser pour l'entraînement.
            Example: `features = [0, 1, 2]`

        - target : str
            Le nom de la cible (target) que le modèle essaie de prédire. Peut être `risk`, `nbsinister` ou `binary`.
            Example: `target = 'risk'`

        - features_name : dict
            Un dictionnaire contenant les noms des features, mappés par leurs indices.
            Example: `features_name = {0: 'feature1', 1: 'feature2', 2: 'feature3'}`

        - optimize_feature : bool
            Indicateur pour savoir si une optimisation des features doit être réalisée avant l'entraînement.
            Example: `optimize_feature = True`

        - dir_output : Path
            Le chemin du répertoire où les résultats et les modèles seront sauvegardés.
            Example: `dir_output = Path('./output')`

        - device : str
            Le dispositif à utiliser pour l'entraînement (`cpu` ou `cuda`).
            Example: `device = 'cpu'`

        - do_grid_search : bool
            Indicateur pour savoir si une recherche de grille (GridSearch) doit être effectuée pour l'optimisation des hyperparamètres.
            Example: `do_grid_search = True`

        - do_bayes_search : bool
            Indicateur pour savoir si une recherche bayésienne (BayesSearch) doit être effectuée pour l'optimisation des hyperparamètres.
            Example: `do_bayes_search = False`

        - task_type : str
            Le model_type de tâche (`regression` ou `classification`).
            Example: `task_type = 'regression'`

        - loss : str
            La fonction de perte à utiliser pour l'entraînement (par exemple, `rmse`, `log_loss`).
            Example: `loss = 'rmse'`

        - model_type : str
            Le model_type de modèle à entraîner (par exemple, `xgboost`, `lightgbm`, `ngboost`).
            Example: `model_type = 'xgboost'`

        - name : str
            Le nom du modèle, utilisé pour l'identifier et sauvegarder les résultats.
            Example: `name = 'xgboost_risk_regression_rmse'`

        - model_params : dict
            Les paramètres du modèle, spécifiques au model_type de modèle utilisé.
            Example: `model_params = {'objective': 'reg:squarederror', 'learning_rate': 0.01, 'n_estimators': 10000}`

        - grid_params : dict
            La grille des hyperparamètres à explorer lors de l'optimisation.
            Example: `grid_params = {'max_depth': [2, 5, 10], 'learning_rate': [0.01, 0.1]}`
    """
    required_params = [
        'x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test', 'features', 'target',
        'features_name', 'optimize_feature', 'dir_output', 'device',
        'do_grid_search', 'do_bayes_search', 'task_type', 'loss', 'model_type', 'name', 'model_params', 'grid_params'
    ]

    for param in required_params:
            assert param in params_fusion, f"Missing required parameter: {param}"

    x_train = params_fusion['x_train']
    y_train = params_fusion['y_train']
    x_val = params_fusion['x_val']
    y_val = params_fusion['y_val']
    x_test = params_fusion['x_test']
    y_test = params_fusion['y_test']
    features = params_fusion['features']
    target = params_fusion['target']
    features_name = params_fusion['features_name']
    optimize_feature = params_fusion['optimize_feature']
    dir_output = params_fusion['dir_output']
    device = params_fusion['device']
    do_grid_search = params_fusion['do_grid_search']
    do_bayes_search = params_fusion['do_bayes_search']
    task_type = params_fusion['task_type']
    loss = params_fusion['loss']
    model_type = params_fusion['model_type']
    name = params_fusion['name']
    model_params = params_fusion['model_params']
    grid_params = params_fusion['grid_params']
    deep = params_fusion['deep']

    Y_train, W_train, Y_val, Y_test, W_test = switch_target(target=target, y_train=y_train, y_val=y_val, y_test=y_test, use_weight=True)

    model, fit_params = get_model_and_fit_params(x_train, Y_train, x_val, Y_val, W_train,
                                        features, name, model_type, task_type, 
                                        model_params, loss)
    
    # Valeurs par défaut pour le dictionnaire
    params = {
        'x_train_list': [],
        'y_train_list': [],
        'x_test_list': [],
        'y_test_list': [],
        'w_test_list': [],
        'features_name_list': [],
        'fit_params_list': [],
        'features_list': [],
        'grid_params_list': [],
        'y_train': Y_train,
        'y_test': Y_test,
        'w_test': W_test,
        'name': name,
        'model': model,
        'deep': deep,
        'fit_params': fit_params,
        'grid_params': grid_params,
        'dir_output': dir_output
    }

    model_list = []

    for params in params_list:
        # Ensure all required parameters are present
        for param in required_params:
            assert param in params, f"Missing required parameter: {param}"

        x_train_p = params['x_train']
        y_train_p = params['y_train']
        x_val_p = params['x_val']
        y_val_p = params['y_val']
        x_test_p = params['x_test']
        y_test_p = params['y_test']
        features_p = params['features']
        target_p = params['target']
        features_name_p = params['features_name']
        optimize_feature_p = params['optimize_feature']
        dir_output_p = params['dir_output']
        device_p = params['device']
        do_grid_search_p = params['do_grid_search']
        do_bayes_search_p = params['do_bayes_search']
        task_type_p = params['task_type']
        loss_p = params['loss']
        model_type_p = params['model_type']
        name_p = params['name']
        model_params_p = params['model_params']
        grid_params_p = params['grid_params']


        Y_train_p, W_train_p, Y_val_p, Y_test_p, W_test_p = switch_target(target=target_p, y_train=y_train_p, y_val=y_val_p, y_test=y_test_p, use_weight=True)
        if deep:
            model_p, fit_params_p = get_model_and_fit_params(x_train_p, Y_train_p, x_val_p, Y_val_p, W_train_p,
                                          features_p, name_p, model_type_p, task_type_p,
                                            model_params_p, loss_p)
        
        else:
            model_p = read_object(f'{name_p}.pkl', dir_output_p)
            if model_p is None:
                raise ValueError(f'Error : train model first or use deep at True, now {deep}')
        
        params['x_train_list'].append(x_train_p)
        params['y_train_list'].append(Y_train_p)
        params['x_test_list'].append(x_test_p)
        params['y_test_list'].append(Y_test_p)
        params['w_test_list'].append(W_test_p)
        params['features_name_list'].append(features_name_p)
        params['fit_params_list'].append(fit_params_p)
        params['features_list'].append(features_p)
        params['grid_params_list'].append(grid_params_p)
        params['do_grid_search_list'].append(do_grid_search_p)
        params['do_bayes_search_list'].append(do_bayes_search_p)
        params['dir_output_list'].append(dir_output_p)
        params['optimize_feature_list'].append(optimize_feature_p)
        params['device_list'].append(device_p)
        model_list.append(model_p)

    model_fusion = ModelFusion(model_list=model_list, model=model, loss=loss, name=name)
    params['model'] = model_fusion

    fit_fusion(params)

def fit_fusion(params):
    """
    Entraîne et évalue un modèle de fusion en utilisant les paramètres fournis.

    Parameters:
    ----------
    params : dict
        Un dictionnaire contenant les paramètres nécessaires pour l'entraînement et l'évaluation du modèle de fusion.

        Clés attendues dans le dictionnaire :
        - x_train_list : list of np.array
        - y_train_list : list of np.array
        - x_test_list : list of np.array
        - y_test_list : np.array
        - w_test_list : np.array
        - features_name_list : list of list
        - fit_params_list : list of dict
        - features_list : list of list
        - grid_params_list : list of dict
        - y_train : np.array
        - y_test : np.array
        - w_test : np.array
        - name : model name
        - model : model
        - deep : bool
        - fit_params : dict
        - grid_params : dict 
        - dir_output : Path
    """
    required_keys = [
        'x_train_list', 'y_train_list', 'x_test_list', 'y_test_list', 'w_test_list', 'features_name_list', 'fit_params_list',
        'features_list', 'grid_params_list', 'y_train', 'w_test', 'y_test',
        'name', 'model', 'deep', 'fit_params', 'grid_params', 'dir_output'
    ]

    for key in required_keys:
        assert key in params, f"Missing required parameter: {key}"

    x_train_list = params['x_train_list']
    y_train_list = params['y_train_list']
    x_test_list = params['x_test_list']
    y_test_list = params['y_test_list']
    w_test_list = params['w_test_list']
    features_name_list = ['features_name_list']
    fit_params_list = params['fit_params_list']
    features_list = params['features_list']
    grid_params_list = params['grid_params_list']

    y_train = params['y_train']
    y_test = params['y_test']
    w_test = params['w_test']
    name = params['name']
    model = params['model']
    parameter_optimization_method = params['parameter_optimization_method']
    grid_params = params['grid_params']
    fit_params = params['fit_params']
    deep = params['deep']
    dir_output = params['dir_output']

    save_object(features_list, 'features.pkl', dir_output / name)
    logger.info(f'Fitting model {name}')

    model.fit(X=x_train_list, y_list=y_train_list, y=y_train, w_test=w_test,
              optimization=parameter_optimization_method,
              grid_params_list=grid_params_list, fit_params_list=fit_params_list,
              grid_params=grid_params, fit_params=fit_params, deep=deep)

    check_and_create_path(dir_output / name)
    save_object(model, name + '.pkl', dir_output / name)
    logger.info(f'Model score {model.score(x_test_list, y_test, w_test)}')

    if name.find('features') != -1:
        logger.info('Features importance')
        model.plot_features_importance(x_train_list, y_train_list, y_test, features_name_list, f'test', dir_output / name, mode='bar', deep=True)
        model.plot_features_importance(x_test_list, y_test_list, y_test, features_name_list, f'train', dir_output / name, mode='bar', deep=True)

    if name.find('search') != -1:
        model.plot_param_influence('max_depth', dir_output / name)

############################################################# BREAK POINT #################################################################

def train_break_point(Xset : np.array, Yset : np.array, features : list, dir_output : Path, features_name : dict, scale : int, n_clusters : int):
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
        train_fet_num += list(np.arange(features_name.index(fet), features_name.index(fet) + maxi))
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

            # Create the cluster model
            model = Predictor(n_clusters, on)
            model.fit(np.unique(X_fet))

            save_object(model, on+'.pkl', dir_output / fet)

            pred = order_class(model, model.predict(X_fet)) # Predict and order class to 0 1 2 3 4

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

def launch_loader(model, loader, model_type,
                  features, target_name,
                  criterion, optimizer,
                  autoRegression,
                  features_name):

    if model_type == 'train':
        model.train()
    else:
        model.eval()
        prev_input = torch.empty((0, 5))

    for i, data in enumerate(loader, 0):

        inputs, labels, edges = data
        
        if target_name == 'risk' or target_name == 'nbsinister':
            target = labels[:,-1]
        else:
            target = (labels[:,-2] > 0).long()

        weights = labels[:,-4]

        if autoRegression and model_type != 'train':
            inode = inputs[:, 0]
            idate = inputs[:, 4]
            if inode in prev_input[:, 0] and idate - 1 in prev_input[:, 4]:
                mask = torch.argwhere((prev_input[:, 0] == inode) & (prev_input[:, 4] == idate - 1))
                inputs[:, features_name.index('AutoRegressionReg')] = prev_input[mask, features_name.index('AutoRegressionReg')]
            else:
                inputs[:, features_name.index('AutoRegressionReg')] = 0

        inputs = inputs[:, features]
        output = model(inputs, edges)

        if autoRegression and model_type != 'train':
            inputs[:, features_name.index('AutoRegressionReg')] = output
            prev_input = torch.cat((prev_input, inputs))

        target = torch.masked_select(target, weights.gt(0))
        weights = torch.masked_select(weights, weights.gt(0))

        if target_name == 'risk' or target_name == 'nbsinister':
            target = target.view(output.shape)
            weights = weights.view(output.shape)
            loss = criterion(output, target, weights)
        else:
            loss = criterion(output, target)

        if model_type == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss

def func_epoch(model, trainLoader, valLoader, features,
               optimizer, criterion, binary,
                autoRegression,
                features_name):
    
    launch_loader(model, trainLoader, 'train', features, binary, criterion, optimizer,  autoRegression,
                  features_name)

    with torch.no_grad():
        loss = launch_loader(model, valLoader, 'val', features, binary, criterion, optimizer,  autoRegression,
                  features_name)

    return loss

def compute_optimisation_features(modelname, lr, scale, features_name,
                                  trainLoader, valLoader, testLoader,
                                  criterion, target_name, epochs, PATIENCE_CNT,
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
        log_features(features_name)()
        for epoch in tqdm(range(epochs)):
            loss = func_epoch(model, trainLoader, valLoader, testFet, optimizer, criterion, target_name,  autoRegression,
                  features_name)
            if loss.item() < BEST_VAL_LOSS:
                BEST_VAL_LOSS = loss.item()
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE_CNT:
                    logger.info(f'Loss has not increased for {patience_cnt} epochs. Last best val loss {BEST_VAL_LOSS}, current val loss {loss.item()}')

            with torch.no_grad():
                loss = launch_loader(model, testLoader, 'test', features, target_name, criterion, optimizer,  autoRegression,
                  features_name)

            if loss.item() < BEST_VAL_LOSS_FET:
                logger.info(f'{fet} : {loss.item()} -> {BEST_VAL_LOSS_FET}')
                BEST_VAL_LOSS_FET = loss.item()
                newFet.append(fet)
                patience_cnt_fet = 0
            #else:
            #    patience_cnt_fet += 1
            #    if patience_cnt_fet >= 30:
            #        logger.info('Loss has not increased for 30 epochs')
            #    break
    return newFet

def train(params):
    """
    Train neural network model
    """
    trainLoader = params['trainLoader']
    valLoader = params['valLoader']
    testLoader = params['testLoader']
    scale = params['scale']
    optimize_feature = params['optimize_feature']
    PATIENCE_CNT = params['PATIENCE_CNT']
    CHECKPOINT = params['CHECKPOINT']
    lr = params['lr']
    epochs = params['epochs']
    criterion = params['criterion']
    features_name = params['features_name']
    modelname = params['modelname']
    features = params['features']
    dir_output = params['dir_output']
    target_name = params['target_name']
    autoRegression = params['autoRegression']

    assert trainLoader is not None and valLoader is not None

    check_and_create_path(dir_output)

    if optimize_feature:
        features = compute_optimisation_features(modelname, lr, scale, features_name, trainLoader, valLoader, testLoader, criterion, target_name, epochs, PATIENCE_CNT, autoRegression)

    dico_model = make_models(len(features), 52, 0.03, 'relu')
    model = dico_model[modelname]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    BEST_VAL_LOSS = math.inf
    BEST_MODEL_PARAMS = None
    patience_cnt = 0

    logger.info('Train model with')
    log_features(features_name)()
    save_object(features, 'features.pkl', dir_output)
    for epoch in tqdm(range(epochs)):
        loss = func_epoch(model, trainLoader, valLoader, features, optimizer, criterion, target_name, autoRegression, features_name)
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
            logger.info(f'epochs {epoch}, Val loss {loss.item()}')
            logger.info(f'epochs {epoch}, Best val loss {BEST_VAL_LOSS}')
            save_object_torch(model.state_dict(), str(epoch)+'.pt', dir_output)
            
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


############################### LOGISTIC #############################

def train_logistic(Y : np.array, dir_output : Path):
    udept = np.unique(Y[:, 3])

    for idept in udept:
        mask = np.argwhere(Y[:, 3] == idept)

        x_train = Y[mask, -1]
        y_train = Y[mask, -2]
        w_train = Y[mask, -3]

        logistic = logistic()
        model = Model(logistic, 'log_loss', 'logistic_'+str(idept))
        model.fit(x_train, y_train, w_train)
        save_object(model, 'logistic_regression_'+int2name[idept], dir_output)

########################### LINEAR ####################################

def train_linear(train_dataset, test_dataset, features, scale, features_name, dir_output, target_name):

    if target_name == 'risk':
        band = -1
    else:
        band = -2

    x_train = train_dataset[0]
    y_train = train_dataset[1][:,band]
    x_test = test_dataset[0]
    y_test = test_dataset[1][:, band]

    if target_name == 'binary':
        linear = LogisticRegression()
        model = Model(linear, 'log_loss', f'linear_{target_name}')
    else:
        linear = LinearRegression()
        model = Model(linear, 'rmse', f'linear_{target_name}')
    
    fit(x_train, y_train, x_test, y_test, None, features, scale, features_name, f'linear_{target_name}', model, {}, 'skip', None, dir_output)

def train_mean(train_dataset, test_dataset, features, scale, features_name, dir_output, target_name):

    if target_name == 'risk':
        band = -1
    else:
        band = -2
    
    x_train = train_dataset[0]
    y_train = train_dataset[1][:, band]

    x_test = test_dataset[0]
    y_test = test_dataset[1][:, band]

    if target_name != 'binary':
        model = MeanFeaturesModel()
        model = Model(model, 'rmse', f'mean_{target_name}')
    else:
        model = MeanFeaturesModel()
        model = Model(model, 'log_loss', f'mean_{target_name}')
    
    fit(x_train, y_train, x_test, y_test, None, features, scale, features_name, f'mean_{target_name}', model, {}, 'skip', None, dir_output)