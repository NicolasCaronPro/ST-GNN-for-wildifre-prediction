from visualize import *
from sklearn.linear_model import LinearRegression, LogisticRegression


############################################## Some tools ###############################################

def create_weight_binary(df_train, df_val, df_test, use_weight):
    if not use_weight:
        df_train['weight'] = 1
        df_val['weight'] = 1
        df_test['weight'] = 1

    df_train['weight_binary'] = np.where(df_train['nbsinister'] == 0, 1,  df_train['weight'])
    df_val['weight_binary'] = np.where(df_val['nbsinister'] == 0, 1,  df_val['weight'])
    df_test['weight_binary'] = np.where(df_test['nbsinister'] == 0, 1,  df_test['weight'])

    df_train['binary'] = df_train['nbsinister'] > 0
    df_val['binary'] = df_val['nbsinister'] > 0
    df_test['binary'] = df_test['nbsinister'] > 0

    return df_train, df_val, df_test

def get_model_and_fit_params(df_train, df_val, df_test, target, weight_col,
                      features, name, model_type, task_type, 
                      params, loss):
    
    if model_type == 'xgboost':
        fit_params = {
                'eval_set': [(df_train[features], df_train[target]), (df_val[features], df_val[target])],
                'sample_weight': df_train[weight_col],
                'verbose': False
         }
        
    elif model_type == 'ngboost':
        fit_params = {
            'X_val': df_val[features],
            'Y_val': df_val[target],
            'sample_weight': df_train[weight_col],
            'early_stopping_rounds': 100,
            'monitor': None
        }
    elif model_type == 'linear':
        fit_params = {}
    else:
        raise ValueError(f"Unsupported model model_type: {model_type}")
    
    model = get_model(model_type=model_type, name=name, device=device, task_type=task_type, params=params, loss=loss)
    return model, fit_params

def explore_features(model,
                    features,
                    df_train,
                    df_val,
                    df_test,
                    weight_col,
                    target):
    
    features_importance = []
    selected_features_ = []
    base_score = -math.inf
    count_max = 70
    c = 0
    for i, fet in enumerate(features):
        
        selected_features_.append(fet)

        X_train_single = df_train[selected_features_]

        fitparams={
                'eval_set':[(X_train_single, df_train[target]), (df_val[selected_features_], df_val[target])],
                'sample_weight' : df_train[weight_col],
                'verbose' : False
                }

        model.fit(X=X_train_single, y=df_train[target], fit_params=fitparams)

        # Calculer le score avec cette seule caractéristique
        single_feature_score = model.score(df_test[selected_features_], df_test[target], sample_weight=df_test[weight_col])

        # Si le score ne s'améliore pas, on retire la variable de la liste
        if single_feature_score <= base_score:
            selected_features_.pop(-1)
            c += 1
        else:
            logger.info(f'With {fet} number {i}: {base_score} -> {single_feature_score}')
            if MLFLOW:
                mlflow.log_metric(model.get_scorer(), single_feature_score, step=i)
            base_score = single_feature_score
            c = 0

        if c > count_max:
            logger.info(f'Score didn t improove for {count_max} features, we break')
            break
        features_importance.append(single_feature_score)

    return selected_features_

############################################### SKLEARN ##################################################

def fit(params):
    """
    """
    required_keys = [
        'df_train', 'df_test', 'target', 'weight_col', 'features',
        'name', 'model', 'fit_params', 'parameter_optimization_method',
        'grid_params', 'dir_output'
    ]

    for key in required_keys:
        assert key in params, f"Missing required parameter: {key}"

    df_train = params['df_train']
    df_test = params['df_test']
    target = params['target']
    weight_col = params['weight_col']
    features = params['features']
    name = params['name']
    model = params['model']
    fit_params = params['fit_params']
    parameter_optimization_method = params['parameter_optimization_method']
    grid_params = params['grid_params']
    dir_output = params['dir_output']


    save_object(features, 'features.pkl', dir_output / name)
    logger.info(f'Fitting model {name}')
    model.fit(X=df_train[features], y=df_train[target],
              optimization=parameter_optimization_method,
              grid_params=grid_params, fit_params=fit_params)

    check_and_create_path(dir_output / name)
    save_object(model, name + '.pkl', dir_output / name)
    logger.info(f'Model score {model.score(df_test[features], df_test[target], df_test[weight_col])}')

    """ if isinstance(model, Model):
        if name.find('features') != -1:
            logger.info('Features importance')
            model.plot_features_importance(df_train[features], df_train[target], features, 'train', dir_output / name, mode='bar')
            model.plot_features_importance(df_test[features], df_test[target], features, 'test', dir_output / name, mode='bar')

        if name.find('search') != -1:
            model.plot_param_influence('max_depth', dir_output / name)

    if isinstance(model, ModelTree):
        model.plot_tree(features_name=features, outname=name, dir_output= dir_output / name, figsize=(50,25))"""
    
    if MLFLOW:
        existing_run = get_existing_run(name)
        if existing_run:
            mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
        else:
            mlflow.start_run(run_name=name, nested=True)

        signature = infer_signature(df_train[features], model.predict(df_train[features]))

        mlflow.set_tag(f"Training", f"{name}")
        mlflow.log_param(f'relevant_feature_name', features)
        mlflow.log_params(model.get_params(deep=True))
        model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=f'{name}',
        signature=signature,
        input_example=df_train[features],
        registered_model_name=name,
        )

        save_object(model_info, f'mlflow_signature_{name}.pkl', dir_output)
        mlflow.end_run()

def train_sklearn_api_model(params):

    required_params = [
        'df_train', 'df_val', 'df_test', 'features', 'target',
        'optimize_feature', 'dir_output', 'device',
        'do_grid_search', 'do_bayes_search', 'task_type', 'loss', 'model_type', 'name', 'model_params', 'grid_params'
    ]

    # Ensure all required parameters are present
    for param in required_params:
        assert param in params, f"Missing required parameter: {param}"

    df_train = params['df_train']
    df_val = params['df_val']
    df_test = params['df_test']
    features = params['features']
    target = params['target']
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

    df_train, df_val, df_test = create_weight_binary(df_train, df_val, df_test, use_weight=True)

    if target == 'binary':
        weight_col = 'weight_binary'
    else:
        weight_col = 'weight'

    model, fit_params = get_model_and_fit_params(df_train, df_val, df_test, target, weight_col,
                                        features, name, model_type, task_type, 
                                        model_params, loss)

    fit_params_dict = {
        'df_train': df_train,
        'df_test': df_test,
        'weight_col' : weight_col,
        'target' : target,
        'features': features,
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
                                                df_train=df_train, weight_col=weight_col, df_val=df_val,
                                                df_test=df_test,
                                                target=target)

        logger.info(relevant_features)

        save_object(relevant_features, f'relevant_features_{target}.pkl', dir_output)
    else:
        relevant_features = read_object(f'relevant_features_{target}.pkl', dir_output)
        if relevant_features is None:
            logger.info('No relevant features found, train with all features')
            relevant_features = features
    
    model, fit_params = get_model_and_fit_params(df_train, df_val, df_test, target, weight_col,
                                        relevant_features, f'{name}_features', model_type, task_type, 
                                        model_params, loss)
    
    fit_params_dict = {
        'df_train': df_train,
        'df_test': df_test,
        'weight_col' : weight_col,
        'target' : target,
        'features': relevant_features,
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
    df_train = params['df_train']
    df_val = params['df_val']
    df_test = params['df_test']
    features = params['features']
    optimize_feature = params['optimize_feature']
    dir_output = params['dir_output']
    device = params['device']
    do_grid_search = params['do_grid_search']
    do_bayes_search = params['do_bayes_search']

    targets = ['risk',
               #'nbsinister',
               #'binary'
               ]

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
            'subsample': 0.8,
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
            'df_train': df_train,
            'df_test': df_test,
            'df_val': df_val,
            'features': features,
            'target': target,
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
                            features: list,
                            autoRegression: bool,
                            optimize_feature: bool,
                            do_grid_search: bool,
                            do_bayes_search: bool):
    
    train_dataset = train_dataset[train_dataset['weight'] > 0]
    val_dataset = val_dataset[val_dataset['weight'] > 0]
    test_dataset = test_dataset[test_dataset['weight'] > 0]

    logger.info(f'x_train shape: {train_dataset.shape}, x_val shape: {val_dataset.shape}, x_test shape: {test_dataset.shape}')

    params = {
        'df_train': train_dataset,
        'df_val': val_dataset,
        'df_test': test_dataset,
        'features': features,
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
                                           target_list : list,
                                           model_list : list,
                                           dir_model_list : list,
                                           features: list,
                            dir_output: Path,
                            device: str,
                            autoRegression: bool,
                            optimize_feature: bool,
                            do_grid_search: bool,
                            do_bayes_search: bool,
                            deep : bool):
    
    params_list = []
    
    for i, train_dataset in enumerate(train_dataset_list):

        train_dataset_list[i] = train_dataset_list[i][train_dataset_list[i]['weight'] > 0]
        val_dataset_list[i] = val_dataset_list[i][val_dataset_list[i]['weight'] > 0]
        test_dataset_list[i] = test_dataset_list[i][test_dataset_list[i]['weight'] > 0]

        model_name = model_list[i]
        dir_model = dir_model_list[i]
        vec = model_name.split('_')
        model_type, target, task_type, loss = vec[0], vec[1], vec[2], vec[3]
        
        params = {
        'df_train': train_dataset_list[i],
        'df_val': val_dataset_list[i],
        'df_test': test_dataset_list[i],
        'target' : target_list[i],
        'features': features,
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

    train_dataset = train_dataset[train_dataset['weight'] > 0]
    val_dataset = val_dataset[val_dataset['weight'] > 0]
    test_dataset = test_dataset[test_dataset['weight'] > 0]

    params_fusion = {
        'df_train': train_dataset,
        'df_val': val_dataset,
        'df_test': test_dataset,
        'features': features,
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
    """
    required_params = [
        'df_train', 'df_val', 'df_test', 'features', 'target',
        'features_name', 'optimize_feature', 'dir_output', 'device',
        'do_grid_search', 'do_bayes_search', 'task_type', 'loss', 'model_type', 'name', 'model_params', 'grid_params'
    ]

    for param in required_params:
            assert param in params_fusion, f"Missing required parameter: {param}"

    df_train = params_fusion['df_train']
    df_val = params_fusion['df_val']
    df_test = params_fusion['df_test']
    features = params_fusion['features']
    target = params_fusion['target']
    weight_col = params_fusion['weight_col']
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

    df_train, df_val, df_test = create_weight_binary(df_train, df_val, df_test, use_weight=True)

    if target == 'binary':
        weight_col = 'weight_binary'
    else:
        weight_col = 'weight'

    model, fit_params = get_model_and_fit_params(df_train, df_val, df_test, target, weight_col,
                                        features, name, model_type, task_type, 
                                        model_params, loss)
    
    # Valeurs par défaut pour le dictionnaire
    params = {
        'df_train_list': [],
        'df_val_list': [],
        'df_test_list': [],
        'targt_list' : [],
        'weight_col_list' : [],
        'features_name_list': [],
        'fit_params_list': [],
        'features_list': [],
        'grid_params_list': [],
        'df_train': df_train,
        'df_test': df_test,
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

        df_train_p = params['df_train']
        df_val_p = params['df_val']
        df_test_p = params['df_test']
        features_p = params['features']
        target_p = params['target']
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

        df_train, df_val, df_test = create_weight_binary(df_train, df_val, df_test, use_weight=True)

        if target == 'binary':
            weight_col = 'weight_binary'
        else:
            weight_col = 'weight'

        if deep:
            model_p, fit_params_p = get_model_and_fit_params(df_train_p, df_val_p, df_test_p, target_p, weight_col,
                                        features_p, name_p, model_type_p, task_type_p, 
                                        model_params_p, loss_p)
        
        else:
            model_p = read_object(f'{name_p}.pkl', dir_output_p)
            if model_p is None:
                raise ValueError(f'Error : train model first or use deep at True, now {deep}')
        
        params['df_train_list'].append(df_train_p)
        params['df_val_list'].append(df_val_p)
        params['df_test_list'].append(df_test_p)
        params['weigh_col_list'].append(weight_col)
        params['df_val_list'].append(df_val_p)
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
    required_keys = [
        'df_train_list', 'df_val_list', 'df_test_list', 'weight_col_list', 'target_list', 'features_name_list', 'fit_params_list',
        'features_list', 'grid_params_list', 'df_train', 'df_test', 'target', 'weight_col',
        'name', 'model', 'deep', 'fit_params', 'grid_params', 'dir_output'
    ]

    for key in required_keys:
        assert key in params, f"Missing required parameter: {key}"

    df_train_list = params['df_train_list']
    df_val_list = params['df_val_list']
    df_test_list = params['df_test_list']
    weight_col_list = params['weight_col_list']
    target_list = ['target_list']
    features_name_list = ['features_name_list']
    fit_params_list = params['fit_params_list']
    features_list = params['features_list']
    grid_params_list = params['grid_params_list']

    df_train = params['df_train']
    df_test = params['df_test']
    target = params['target']
    weight_col = params['weight_col']
    name = params['name']
    model = params['model']
    parameter_optimization_method = params['parameter_optimization_method']
    grid_params = params['grid_params']
    fit_params = params['fit_params']
    deep = params['deep']
    dir_output = params['dir_output']

    save_object(features_list, 'features.pkl', dir_output / name)
    logger.info(f'Fitting model {name}')
    y_train_list = [df_train_list[tar] for tar in target_list]
    y_test_list = [df_test_list[tar] for tar in target_list]
    w_train = [df_train_list[wcol] for wcol in weight_col_list]
    w_test = [df_test_list[wcol] for wcol in weight_col_list]
    model.fit(X=df_train_list, y_train_list=y_train_list, w=w_train, y=df_train[target],
              optimization=parameter_optimization_method,
              grid_params_list=grid_params_list, fit_params_list=fit_params_list,
              grid_params=grid_params, fit_params=fit_params, deep=deep)

    check_and_create_path(dir_output / name)
    save_object(model, name + '.pkl', dir_output / name)
    logger.info(f'Model score {model.score(df_test_list=df_test_list, y=df_test[target], w=df_test[weight_col])}')

    if name.find('features') != -1:
        logger.info('Features importance')
        model.plot_features_importance(df_train_list, y_train_list, df_train[target], features_name_list, f'train', dir_output / name, mode='bar', deep=True)
        model.plot_features_importance(df_test_list, y_test_list, df_test[target], features_name_list, f'test', dir_output / name, mode='bar', deep=True)

    if name.find('search') != -1:
        model.plot_param_influence('max_depth', dir_output / name)

############################################################# BREAK POINT #################################################################

def train_break_point(df : pd.DataFrame, features : list, dir_output : Path, n_clusters : int):
    check_and_create_path(dir_output)
    logger.info('###################" Calculate break point ####################')

    #logger.info(f'Check for break_point_dict in {dir_output}')
    #if (dir_output / 'break_point_dict.pkl').is_file():
    #    logger.info('Found file, return')
    #    return

    res_cluster = {}
    features_selected_2 = []

    FF_t = np.sum(df['nbsinister'])
    Area_t = len(df)

    # For each feature
    for fet in features:
        check_and_create_path(dir_output / fet)
        logger.info(fet)
        res_cluster[fet] = {}
        X_fet = df[fet].values # Select the feature
        if np.unique(X_fet).shape[0] < n_clusters:
            continue

        # Create the cluster model
        model = Predictor(n_clusters, fet)
        model.fit(np.unique(X_fet))

        save_object(model, fet+'.pkl', dir_output / fet)

        pred = order_class(model, model.predict(X_fet)) # Predict and order class to 0 1 2 3 4

        cls = np.unique(pred)
        if cls.shape[0] != n_clusters:
            continue

        plt.figure(figsize=(5,5)) # Create figure for ploting

        # For each class calculate the target values
        for c in cls:
            mask = np.argwhere(pred == c)[:,0]
            #res_cluster[fet][c] = round(np.sum(df['nbsinister'].values[mask]) / np.sum(df['nbsinister']) * 100, 3)

            # Calculer FF_i (le nombre de sinistres pour la classe c)
            FF_i = np.sum(df['nbsinister'].values[mask])
            
            # Calculer Area_i (le nombre total de pixels pour la classe c)
            Area_i = mask.shape[0]
            
            # Calculer FireOcc et Area pour la classe c
            FireOcc = FF_i / FF_t
            Area = Area_i / Area_t
            
            # Calculer le ratio de fréquence (FR) pour la classe c
            FR = FireOcc / Area

            res_cluster[fet][c] = round(FR, 3)

            plt.scatter(X_fet[mask], df['nbsinister'].values[mask], label=f'{c} : {res_cluster[fet][c]}')

        #logger.info(f'{fet} : {res_cluster[fet]}')        
        # Calculate Pearson coefficient
        #dff = pd.DataFrame(res_cluster, index=np.arange(n_clusters)).reset_index()
        #dff.rename({'index': 'class'}, axis=1, inplace=True)
        #dff['nbsinister'] = df['nbsinister']
        #correlation = dff['class'].corr(dff['nbsinister'])
        correlation = 0.8
        res_cluster[fet]['correlation'] = correlation

        # Plot
        plt.xlabel(fet)
        plt.ylabel('nbsinister')
        plt.title(fet)
        plt.legend()
        plt.savefig(dir_output / fet / f'{fet}.png')
        plt.close('all')

        # If no correlation then pass
        if abs(correlation) > 0.7:
            # If negative linearity, inverse class
            if correlation < 0:
                new_class = np.flip(np.arange(n_clusters))
                temp = {}
                for i, nc in enumerate(new_class):
                    temp[nc] = res_cluster[fet][i]
                temp['correlation'] = res_cluster[fet]['correlation']
                res_cluster[fet] = temp
            
            # Add in selected features
            features_selected_2.append(fet)
        
    logger.info(len(features_selected_2))
    save_object(res_cluster, 'break_point_dict.pkl', dir_output)

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
            target = labels[:, -1, :]
        else:
            target = (labels[:, -2, :] > 0).long()

        weights = labels[:, -4, :]
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
        output = torch.masked_select(output, weights.gt(0))
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

def func_epoch(model, train_loader, val_loader, features,
               optimizer, criterion, binary,
                autoRegression,
                features_name):
    
    launch_loader(model, train_loader, 'train', features, binary, criterion, optimizer,  autoRegression,
                  features_name)

    with torch.no_grad():
        loss = launch_loader(model, val_loader, 'val', features, binary, criterion, optimizer,  autoRegression,
                  features_name)

    return loss

def compute_optimisation_features(modelname, lr, scale, features_name,
                                  train_loader, val_loader, test_loader,
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
        log_features(features_name)
        for epoch in tqdm(range(epochs)):
            loss = func_epoch(model, train_loader, val_loader, testFet, optimizer, criterion, target_name,  autoRegression,
                  features_name)
            if loss.item() < BEST_VAL_LOSS:
                BEST_VAL_LOSS = loss.item()
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE_CNT:
                    logger.info(f'Loss has not increased for {patience_cnt} epochs. Last best val loss {BEST_VAL_LOSS}, current val loss {loss.item()}')

            with torch.no_grad():
                loss = launch_loader(model, test_loader, 'test', features, target_name, criterion, optimizer,  autoRegression,
                  features_name)

            if loss.item() < BEST_VAL_LOSS_FET:
                logger.info(f'{fet} : {loss.item()} -> {BEST_VAL_LOSS_FET}')
                BEST_VAL_LOSS_FET = loss.item()
                newFet.append(fet)

    return newFet

def train(params):
    """
    Train neural network model
    """
    train_loader = params['train_loader']
    val_loader = params['val_loader']
    test_loader = params['test_loader']
    scale = params['scale']
    optimize_feature = params['optimize_feature']
    PATIENCE_CNT = params['PATIENCE_CNT']
    CHECKPOINT = params['CHECKPOINT']
    lr = params['lr']
    epochs = params['epochs']
    loss_name = params['loss_name']
    features_name = params['features_name']
    modelname = params['modelname']
    features = params['features']
    dir_output = params['dir_output']
    target_name = params['target_name']
    autoRegression = params['autoRegression']
    k_days = params['k_days']

    if MLFLOW:
        mlflow.start_run(run_name=modelname)
        mlflow.log_params(params)

    assert train_loader is not None and val_loader is not None

    features_selected = [i for i in range(len(features_name)) if f'{features_name[i]}_mean' in features or features_name[i] in features]

    check_and_create_path(dir_output)

    criterions_dict = {'rmse': weighted_rmse_loss,
                       'log_loss': weighted_cross_entropy}

    criterion = criterions_dict[loss_name]

    if optimize_feature:
        features = compute_optimisation_features(modelname, lr, scale, features_name, train_loader, val_loader, test_loader,
                                                 criterion, target_name, epochs, PATIENCE_CNT, autoRegression)

    dico_model = make_models(len(features), len(features_selected), scale, 0.03, 'relu', k_days, target_name == 'binary')
    model = dico_model[modelname]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    BEST_VAL_LOSS = math.inf
    BEST_MODEL_PARAMS = None
    patience_cnt = 0

    logger.info('Train model with')
    save_object(features, 'features.pkl', dir_output)
    for epoch in tqdm(range(epochs)):
        loss = func_epoch(model, train_loader, val_loader, features_selected, optimizer, criterion, target_name, autoRegression, features_name)
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
        if MLFLOW:
            mlflow.log_metric('loss', loss.item(), step=epoch)
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