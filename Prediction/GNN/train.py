from copy import deepcopy
from turtle import mode
from matplotlib.pyplot import grid
from GNN.visualize import *
from sklearn.linear_model import LinearRegression, LogisticRegression
from pygam import s, te, f, intercept, l

############################################## Some tools ###############################################

def create_weight_binary(df_train, df_val, df_test, use_weight):
    """if not use_weight:
        df_train['weight'] = 1
        df_val['weight'] = 1
        df_test['weight'] = 1

    df_train['weight_binary'] = np.where(df_train['nbsinister'] == 0, 1,  df_train['nbsinister'])
    df_val['weight_binary'] = np.where(df_val['nbsinister'] == 0, 1,  df_val['nbsinister'])
    df_test['weight_binary'] = np.where(df_test['nbsinister'] == 0, 1,  df_test['nbsinister'])"""

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
            'early_stopping_rounds': 15,
        }

    elif model_type == 'rf':
        fit_params = {
            'sample_weight': df_train[weight_col]
        }

    elif model_type == 'dt':
        fit_params = {
            'sample_weight': df_train[weight_col]
        }

    elif model_type == 'lightgbm':
        fit_params = {
            'eval_set': [(df_val[features], df_val[target])],
            'eval_sample_weight': [df_val[weight_col]],
            'early_stopping_rounds': 15,
            'verbose': False
        }

    elif model_type == 'svm':
        fit_params = {
            'sample_weight': df_train[weight_col]
        }
    elif model_type == 'poisson':
        fit_params = {
            'sample_weight': df_train[weight_col]
        }

    elif model_type == 'gam':
        fit_params = {
        'weights': df_train[weight_col]
        }

    elif model_type ==  'linear':
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
                    target,
                    return_mode='best',
                    num_iteration=1):

    best_score_iter = -math.inf
    selected_features_iter = []
    all_selected_features = []
    all_fit_params = []
    all_models = []
    
    for num_iter in range(num_iteration):
        logger.info(f'###############################################################')
        logger.info(f'#                                                             #')
        logger.info(f'#                 Iteration {num_iter}                            #')
        logger.info(f'#                                                             #')
        logger.info(f'###############################################################')
        features_importance = []
        selected_features_ = []
        base_score = -math.inf
        count_max = 50
        bre = False
        c = 0
        if num_iter != 0:
            random.shuffle(features)
        for i, fet in enumerate(features):
            
            selected_features_.append(fet)

            X_train_single = df_train[selected_features_]

            if model.name.find('xgboost') != -1:

                fitparams={
                        'eval_set':[(X_train_single, df_train[target]), (df_val[selected_features_], df_val[target])],
                        'sample_weight' : df_train[weight_col],
                        'verbose' : False
                        }
            elif model.name.find('gam') != -1:
                fitparams = {
                    #'weights' : df_train[weight_col]
                }

            
            all_fit_params.append(fitparams)
            model.fit(X=X_train_single, y=df_train[target], fit_params=fitparams)

            # Calculer le score avec cette seule caractéristique
            single_feature_score = model.score(df_test[selected_features_], df_test[target], sample_weight=df_test[weight_col])
            
            # Si le score ne s'améliore pas, on retire la variable de la liste
            if single_feature_score <= base_score:
                selected_features_.pop(-1)
                #logger.info(f'With {fet} number {i}: {base_score} -> {single_feature_score}')
                c += 1
            else:
                logger.info(f'With {fet} number {i}: {base_score} -> {single_feature_score}')
                if MLFLOW:
                    mlflow.log_metric(model.get_scorer(), single_feature_score, step=i)
                base_score = single_feature_score
                c = 0
                last_model = deepcopy(model)

            if c > count_max:
                logger.info(f'Score didn t improove for {count_max} features, we break')
                all_selected_features.append(selected_features_)
                all_models.append(last_model)
                bre = True
                break
            features_importance.append(single_feature_score)

        if base_score > best_score_iter:
            logger.info(f'Iteration {num_iter} goes from {best_score_iter} -> {base_score}')
            best_score_iter = base_score
            selected_features_iter = selected_features_

        if not bre:
            all_selected_features.append(selected_features_)
            all_models.append(last_model)

    if return_mode == 'best':
        return selected_features_iter
    elif return_mode == 'all':
        return all_selected_features, all_models, all_fit_params
    elif return_mode == 'both':
        return selected_features_iter, all_selected_features, all_models, all_fit_params

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
    df_val = params['df_val']
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
    save_object(grid_params, 'grid_params.pkl', dir_output / name)
    save_object(fit_params, 'fit_params.pkl', dir_output / name)
    logger.info(f'Fitting model {name}')

    """if 'AutoRegressionReg-J-1' in features:
        if 'early_stopping_rounds' in fit_params.keys():
            early_stopping_rounds = fit_params['early_stopping_rounds']
            fit_params['early_stopping_rounds'] = 0
            model.recursive_fit(df_train[features], df_train[target], (df_val[features],
                            df_val[['id', 'date', target]]), features, 1, early_stopping_rounds)
        else:
            #try:
                early_stopping_rounds = model.get_params('early_stopping_rounds')
                model.set_params(early_stopping_rounds=0)
                model.recursive_fit(df_train[features], df_train[target], (df_val[features],
                            df_val[['id', 'date', target]]), features, 1, early_stopping_rounds)
                except Exception as e:
                logger.info(e)
                model.fit(X=df_train[features], y=df_train[target],
                    optimization=parameter_optimization_method,
                    grid_params=grid_params, fit_params=fit_params)"""
        
    #else:
    model.fit(X=df_train[features], y=df_train[target],
                optimization=parameter_optimization_method,
                grid_params=grid_params, fit_params=fit_params)

    check_and_create_path(dir_output / name)
    save_object(model, name + '.pkl', dir_output / name)
    logger.info(f'Model score {model.score(df_test[features], df_test[target], df_test[weight_col])}')

    if isinstance(model, Model):
        if name.find('features') != -1:
            logger.info('Features importance')
            model.shapley_additive_explanation(df_train[features], 'train', dir_output / name, 'beeswarm', figsize=(30,15))
            model.shapley_additive_explanation(df_test[features], 'test',  dir_output / name, 'beeswarm', figsize=(30,15))

        if name.find('search') != -1:
            model.plot_param_influence('max_depth', dir_output / name)

    if isinstance(model, ModelTree):
        model.plot_tree(features_name=features, outname=name, dir_output= dir_output / name, figsize=(50,25))
    
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

    if target == 'binary' or target == 'nbsinister':
        weight_col = 'weight_nbsinister'
    else:
        weight_col = 'weight'

    model, fit_params = get_model_and_fit_params(df_train, df_val, df_test, target, weight_col,
                                        features, name, model_type, task_type, 
                                        model_params, loss)

    fit_params_dict = {
        'df_train': df_train,
        'df_test': df_test,
        'df_val': df_val,
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

    #if model_type != 'gam':
    #fit(fit_params_dict)
    #else:
    #    logger.info('GAM can t being train with all features. SKIP')
    
    if optimize_feature:
        model = get_model(model_type=model_type, name=f'{name}_features', device=device, task_type=task_type, params=model_params, loss=loss)
        relevant_features = explore_features(model=model,
                                                features=features,
                                                df_train=df_train, weight_col=weight_col, df_val=df_val,
                                                df_test=df_test,
                                                target=target,
                                                num_iteration=1)

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
        'df_val' : df_val,
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
        if model_type == 'gam':
            # Liste des formes fonctionnelles possibles pour chaque variable
            functional_forms = [s, l, te, f]

            # Générer toutes les combinaisons possibles pour chaque variable
            all_combinations = list(itertools.product(functional_forms, repeat=len(features)))

            # Initialiser la grille de paramètres pour les termes
            grid_params = {'terms': []}

            # Boucle pour tester toutes les combinaisons de formes fonctionnelles
            for combination in all_combinations:
                terms = [func(i) for i, func in enumerate(combination)]  # Appliquer chaque fonction aux variables
                grid_params['terms'].append(terms)

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
    model = params['name']
    
    name, target, task_type, loss = model.split('_')

    if loss == 'rmse':
        task_type = 'regression'
        objective = 'reg:squarederror'
    elif loss == 'poisson':
        task_type = 'regression'
        objective = 'count:poisson'
    elif loss == 'logloss':
        task_type = 'classification'
        objective = 'reg:logistic'

    model_params = {
        'objective': objective,
        'verbosity': 0,
        'early_stopping_rounds': 15,
        'learning_rate': 0.0001,
        'min_child_weight': 1.0,
        'max_depth': 6,
        'max_delta_step': 1.0,
        'subsample': 0.55,
        'colsample_bytree': 0.73,
        'colsample_bylevel': 0.62,
        'reg_lambda': 9.638,
        'reg_alpha': 0.53,
        'n_estimators': 10000,
        'random_state': 42,
        'tree_method': 'hist',
        'device':"cuda"
    }

    grid_params = {'max_depth': [6],
                    'min_child_weight': [1.0,1.5,3.5,5,10],
                    'reg_lambda' : [1.0,2.0,3.0,5.0],
                    'reg_alpha':[0.9,0.6,0.7],
                    'learning_rate':[0.01,0.001,0.005,0.00005]
                    }

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
        'name': model,
        'model_params': model_params,
        'grid_params': grid_params,
    })

def train_ngboost(params):
    """
    Train NGBoost model
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
    model = params['name']
    
    _, target, task_type, loss = model.split('_')

    if target == 'crps':
        objective = 'normal'
    else:
        loss = 'log_loss'
        objective = 'bernoulli'

    name = f'ngboost_{target}_{task_type}_{loss}'
    model_params = {
        'learning_rate': 0.01,
        'n_estimators': 10000,
        'minibatch_frac': 0.8,
        'random_state': 42,
        'Base': None,
    }
    grid_params = {'n_estimators': [100, 500, 1000, 5000, 10000]}

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
        'model_type': 'ngboost',
        'name': name,
        'model_params': model_params,
        'grid_params': grid_params,
    })

def train_lightgbm(params):
    """
    Train LightGBM model
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
    model = params['name']
    
    _, target, task_type, loss = model.split('_')

    if loss == 'rmse':
        objective = 'regression'
    else:
        objective = 'binary'

    max_depth = 6
    name = f'lightgbm_{target}_{task_type}_{loss}'
    model_params = {
        'objective': objective,
        'verbosity': -1,
        'learning_rate': 0.01,
        'min_child_weight': 1.0,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 1.0,
        'reg_alpha': 0.9,
        'n_estimators': 10000,
        'random_state': 42,
        'num_leaves': 2**max_depth,
        'device': device,
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
        'model_type': 'lightgbm',
        'name': name,
        'model_params': model_params,
        'grid_params': grid_params,
    })

def train_svm(params):
    """
    Train SVM model
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
    model = params['name']

    model_type, target, task_type, loss = model.split('_')
    
    model_params = {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
    }
    grid_params = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}

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
        'model_type': model_type,
        'name': name,
        'model_params': model_params,
        'grid_params': grid_params,
    })

def train_random_forest(params):
    """
    Train Random Forest model
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

    name = params['name']
    
    model_type, target, task_type, loss = name.split('_')

    name = f'{model_type}_{target}_{task_type}_{loss}'
    model_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'bootstrap': True,
        'random_state': 42,
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
        'model_type': model_type,
        'name': name,
        'model_params': model_params,
        'grid_params': grid_params,
    })

def train_decision_tree(params):
    """
    Train Decision Tree model
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
    model = params['name']

    model_type, target, task_type, loss = model.split('_')
    
    model_params = {
        'max_depth': 6,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
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
        'model_type': model_type,
        'name': model,
        'model_params': model_params,
        'grid_params': grid_params,
    })

def train_poisson(params):
    """
    Train Decision Tree model
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
    model = params['name']

    model_type, target, task_type, loss = model.split('_')
    
    model_params = {
            'alpha': 1.0,            # Regularization strength (L2 penalty)
            'max_iter': 100,          # Maximum number of iterations
            'tol': 1e-4,              # Tolerance for stopping criteria
            'fit_intercept': True,    # Whether to fit the intercept
            'verbose': 0,             # Verbosity level
            'warm_start': False,      # Reuse solution of the previous call to fit
        }
    grid_params = {'alpha': [1.0,1.5,2.0]}

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
        'model_type': model_type,
        'name': model,
        'model_params': model_params,
        'grid_params': grid_params,
    })

def train_gam(params):
    """
    Train Decision Tree model
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
    model = params['name']

    model_type, target, task_type, loss = model.split('_')
    
    if target == 'nbsinister':
        model_params = {
            'distribution' : 'poisson',
            'link' : 'log'
            }
    elif target == 'risk':
        model_params = {'distribution' : 'normal',
                    'link': 'identity'
                }
    elif target == 'binary':
        model_params = {
                'distribution' : 'binomial',
                'link': 'log'
                }
    else:
        ValueError(f'Unknow {target}, select in [risk, nbsinister, binary]')

    grid_params = {'terms' : []}

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
        'model_type': model_type,
        'name': model,
        'model_params': model_params,
        'grid_params': grid_params,
    })

def wrapped_train_sklearn_api_model(train_dataset, val_dataset, test_dataset,
                                    model,
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
        'name': model[0],
    }

    name, _,_ ,_ = model[0].split('_')
    if name == 'xgboost':
        train_xgboost(params)
    elif name == 'lightgbm':
        train_lightgbm(params)
    elif name == 'rf':
        train_random_forest(params)
    elif name == 'svm':
        train_svm(params)
    elif name == 'dt':
        train_decision_tree(params)
    elif name == 'ngboost':
        train_ngboost(params)
    elif name == 'poisson':
        train_poisson(params)
    elif name == 'gam':
        train_gam(params)

############################################################# Voting ######################################################################

def fit_voting(params):
    """
    Function to fit a voting or stacking model with given parameters.
    """
    required_keys = [
        'df_train', 'df_test', 'df_val', 'target', 'weight_col', 'features',
        'name', 'model', 'fit_params', 'parameter_optimization_method',
        'grid_params', 'dir_output'
    ]

    # Ensure all required parameters are provided
    for key in required_keys:
        assert key in params, f"Missing required parameter: {key}"

    # Unpack parameters
    df_train_list = params['df_train']
    df_test_list = params['df_test']
    df_val_list = params['df_val']
    target_list = params['target']
    weight_col_list = params['weight_col']
    features_list = params['features']
    name = params['name']
    model = params['model']
    fit_params = params['fit_params']
    parameter_optimization_method = params['parameter_optimization_method']
    grid_params = params['grid_params']
    dir_output = params['dir_output']

    # Save features used
    save_object(features_list, 'features.pkl', dir_output / name)
    logger.info(f'Fitting model {name}')

    # Ensure all lists are of the same length
    n_models = len(df_train_list)
    print(n_models, len(features_list))
    assert len(features_list) == n_models, "features_list length mismatch"
    assert len(target_list) == n_models, "target_list length mismatch"
    assert len(weight_col_list) == n_models, "weight_col_list length mismatch"
    assert len(df_test_list) == n_models, "df_test_list length mismatch"

    # Prepare training data lists
    X_list = [df_train_list[i][features_list[i]] for i in range(n_models)]
    y_list = [df_train_list[i][target_list[i]] for i in range(n_models)]

    # Fit the model
    model.fit(
        X_list=X_list,
        y_list=y_list,
        optimization=parameter_optimization_method,
        grid_params_list=grid_params,
        fit_params_list=fit_params,
        cv_folds=10
    )

    # Save the trained model
    check_and_create_path(dir_output / name)
    save_object(model, name + '.pkl', dir_output / name)

    # Prepare test data lists
    X_test_list = [df_test_list[i][features_list[i]] for i in range(n_models)]
    y_test = df_test_list[0][target_list[0]]
    weight = df_test_list[0][weight_col_list[0]]

    # Evaluate the model
    model_score = model.score(X_test_list, y_test, sample_weight=weight)
    logger.info(f'Model score: {model_score}')

    # Log model and parameters to MLflow if enabled
    if MLFLOW:
        existing_run = get_existing_run(name)
        if existing_run:
            mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
        else:
            mlflow.start_run(run_name=name, nested=True)

        # Prepare signature for logging
        y_pred = model.predict(X_list)
        #signature = infer_signature(X_list, y_pred)

        mlflow.set_tag("Training", name)
        mlflow.log_param('relevant_feature_name', features_list)
        mlflow.log_params(model.get_params(deep=True))
        """model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=name,
            #signature=signature,
            input_example=X_list,
            registered_model_name=name,
        )"""

        #save_object(model_info, f'mlflow_signature_{name}.pkl', dir_output)
        mlflow.end_run()

def wrapped_train_sklearn_api_voting_model(train_dataset, val_dataset, test_dataset,
                                    model_name,
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

    if do_grid_search:
        parameter_optimization_method = 'grid'
    elif do_bayes_search:
        parameter_optimization_method = 'bayes'
    else:
        parameter_optimization_method = 'skip'

    model_type, target, task_type, loss = model_name[0].split('_')
    
    if target == 'binary' or target == 'nbsinister':
        weight_col = 'weight_nbsinister'
    else:
        weight_col = 'weight'

    if model_type == 'xgboost':

        if loss == 'rmse':
            task_type = 'regression'
            objective = 'reg:squarederror'
        else:
            task_type = 'classification'
            objective = 'reg:logistic'

        model_params = {
        'objective': objective,
        'verbosity': 0,
        'early_stopping_rounds': 15,
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
        'device':"cuda"
        }
        
        grid_params = {'max_depth': [6],
                'min_child_weight': [1.0,1.5,3.5,5,10],
                'reg_lambda' : [1.0,2.0,3.0,5.0],
                'reg_alpha':[0.9,0.6,0.7],
                'learning_rate':[0.01,0.001,0.005,0.00005]
                }
        
    elif name == 'lightgbm':
        pass
    elif name == 'rf':
        pass
    elif name == 'svm':
        pass
    elif name == 'dt':
        pass
    elif name == 'ngboost':
        pass

    num_iteration = 10
    
    train_dataset, val_dataset, test_dataset = create_weight_binary(train_dataset, val_dataset, test_dataset, False)

    model, fit_params = get_model_and_fit_params(train_dataset, val_dataset, test_dataset, target, weight_col,
                                    features, name, model_type, task_type,
                                    model_params, loss)
    
    all_selected_features, all_models, all_fit_params = explore_features(model=model,
                                                features=features,
                                                df_train=train_dataset, weight_col=weight_col, df_val=val_dataset,
                                                df_test=test_dataset,
                                                target=target,
                                                return_mode='all',
                                                num_iteration=num_iteration)
    
    all_models = [class_model.best_estimator_ for class_model in all_models]
    
    estimator = ModelVoting(all_models, loss=loss, name=f'{model_name}')

    fit_params_dict = {
        'df_train': [train_dataset for i in range(num_iteration)],
        'df_test': [test_dataset for i in range(num_iteration)],
        'df_val': [val_dataset for i in range(num_iteration)],
        'weight_col' : [weight_col for i in range(num_iteration)],
        'target' : [target for i in range(num_iteration)],
        'features': all_selected_features,
        'name': f'{model_name[0]}_voting_{num_iteration}',
        'model': estimator,
        'fit_params': all_fit_params,
        'parameter_optimization_method': parameter_optimization_method,
        'grid_params': [grid_params for i in range(num_iteration)],
        'dir_output': dir_output
    }

    fit_voting(fit_params_dict)

def wrapped_train_voting_model_list(train_dataset, val_dataset, test_dataset,
                                    models_list,
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

    if do_grid_search:
        parameter_optimization_method = 'grid'
    elif do_bayes_search:
        parameter_optimization_method = 'bayes'
    else:
        parameter_optimization_method = 'skip'

    all_models = []
    all_features = []
    all_grid_params = []

    model_type, target, task_type, loss = models_list[0].split('_')

    if target == 'binary' or target == 'nbsinister':
        weight_col = 'weight_nbsinister'
    else:
        weight_col = 'weight'

    for model_name in models_list:

        model = None
        relevant_features = None
        fit_params = None
        grid_params = None

        if model_name in sklearn_model_list:
            if (dir_output / model_name / f'{model_name}.pkl').is_file():
                model = read_object(f'{model_name}.pkl', dir_output / model_name)
                relevant_features = read_object('features.pkl', dir_output / model_name)
                fit_params = read_object('fit_params.pkl', dir_output / model_name)
                grid_params = read_object('grid_params.pkl', dir_output / model_name)

            if not (dir_output / model_name / f'{model_name}.pkl').is_file() or model is None or relevant_features is None or fit_params is None or grid_params is None:
                wrapped_train_sklearn_api_model(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, model=model_name,
                                                            dir_output=dir_output, device=device, features=features, autoRegression=autoRegression,
                                                            optimize_feature=optimize_feature, do_grid_search=do_grid_search, do_bayes_search=do_bayes_search)
                
                model = read_object(f'{model_name}.pkl', dir_output / model_name)
                relevant_features = read_object('features.pkl', dir_output / model_name)
                fit_params = read_object('features.pkl', dir_output / model_name)
                grid_params = read_object('grid_params.pkl', dir_output / model_name)

        elif model_name in models_2D:
            
            if (dir_output / model_name / f'{model_name}.pkl').is_file():
                model = read_object(f'{model_name}.pkl', dir_output / model_name)
                relevant_features = read_object('features.pkl', dir_output / model_name)
                
            if not (dir_output / model_name / f'{model_name}.pkl').is_file() or model is None or relevant_features is None:
                raise FileNotFoundError('Le modèle entraîné est introuvable ou les caractéristiques pertinentes sont manquantes.')

            elif model_name in models_hybrid:
                if (dir_output / model_name / f'{model_name}.pkl').is_file():
                    model = read_object(f'{model_name}.pkl', dir_output / model_name)
                    relevant_features = read_object('features.pkl', dir_output / model_name)

                if not (dir_output / model_name / f'{model_name}.pkl').is_file() or model is None or relevant_features is None:
                    raise FileNotFoundError('Le modèle hybride entraîné est introuvable ou les caractéristiques pertinentes sont manquantes.')

            elif model_name in temporal_model_list:
                if (dir_output / model_name / f'{model_name}.pkl').is_file():
                    model = read_object(f'{model_name}.pkl', dir_output / model_name)
                    relevant_features = read_object('features.pkl', dir_output / model_name)

                if not (dir_output / model_name / f'{model_name}.pkl').is_file() or model is None or relevant_features is None:
                    raise FileNotFoundError('Le modèle temporel entraîné est introuvable ou les caractéristiques pertinentes sont manquantes.')

            elif model_name in daily_model_list:
                if (dir_output / model_name / f'{model_name}.pkl').is_file():
                    model = read_object(f'{model_name}.pkl', dir_output / model_name)
                    relevant_features = read_object('features.pkl', dir_output / model_name)
                    
                if not (dir_output / model_name / f'{model_name}.pkl').is_file() or model is None or relevant_features is None:
                    raise FileNotFoundError('Le modèle journalier entraîné est introuvable ou les caractéristiques pertinentes sont manquantes.')

        else:
            ValueError(f'Unknow {model_name}')
            
        all_models.append(model)
        all_features.append(relevant_features)
        all_grid_params.append(grid_params)

    estimator = ModelVoting(all_models, loss=loss, name=f'{models_list}')

    len_model = len(all_models)
    fit_params_dict = {
        'df_train': [train_dataset for i in range(len_model)],
        'df_test': [test_dataset for i in range(len_model)],
        'df_val': [val_dataset for i in range(len_model)],
        'weight_col' : [weight_col for i in range(len_model)],
        'target' : [target for target in range(len_model)],
        'features': all_features,
        'name': f'{name}',
        'model': estimator,
        'fit_params': fit_params,
        'parameter_optimization_method': parameter_optimization_method,
        'grid_params': all_grid_params,
        'dir_output': dir_output
    }

    fit_voting(fit_params_dict)

############################################################# Stacked ######################################################################

def wrapped_train_sklearn_api_stacked_model_list(train_dataset, val_dataset, test_dataset,
                                    models_list,
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

    if do_grid_search:
        parameter_optimization_method = 'grid'
    elif do_bayes_search:
        parameter_optimization_method = 'bayes'
    else:
        parameter_optimization_method = 'skip'

    all_models = []

    for model_name in models_list[:-1]:
        model_type, target, task_type, loss = model_name.split('_')

        if target == 'binary' or target == 'nbsinister':
            weight_col = 'weight_nbsinister'
        else:
            weight_col = 'weight'

        if model_type == 'xgboost':
            if loss == 'rmse':
                objective = 'regression'
            else:
                objective = 'binary'
            model_params = {
            'objective': objective,
            'verbosity': 0,
            'early_stopping_rounds': 15,
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
            'device':"cuda"
            }
        elif model_type == 'lightgbm':
            pass
        elif model_type == 'rf':
            pass
        elif model_type == 'svm':
            pass
        elif model_type == 'dt':
            pass
        elif model_type == 'ngboost':
            pass
        
        model, _ = get_model_and_fit_params(train_dataset, val_dataset, test_dataset, target, weight_col,
                                    features, model_type, model_type, task_type, 
                                    model_params, loss)

        all_models.append(model)

    model_type, target, task_type, loss = models_list[-1].split('_')

    if model_type == 'xgboost':
        if loss == 'rmse':
            objective = 'regression'
        else:
            objective = 'binary'
        model_params = {
        'objective': objective,
        'verbosity': 0,
        'early_stopping_rounds': 15,
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
        'device':"cuda"
        }
    elif model_type == 'lightgbm':
        pass
    elif model_type == 'rf':
        pass
    elif model_type == 'svm':
        pass
    elif model_type == 'dt':
        pass
    elif model_type == 'ngboost':
        pass
        
    model, _ = get_model_and_fit_params(train_dataset, val_dataset, test_dataset, target, weight_col,
                                    features, name, model_type, task_type, 
                                    model_params, loss)

    estimator = ModelStacking(all_models, final_estimator=model, loss=loss, name=f'{models_list}')
    fit_params = {'sample_weight': train_dataset[weight_col]}
    fit_params_dict = {
        'df_train': train_dataset,
        'df_test': test_dataset,
        'df_val': val_dataset,
        'weight_col' : weight_col,
        'target' : target,
        'features': all_features,
        'name': f'{name}',
        'model': estimator,
        'fit_params': fit_params,
        'parameter_optimization_method': 'skip',
        'grid_params': {},
        'dir_output': dir_output
    }

    fit(fit_params_dict)

############################################################# BREAK POINT #################################################################

def train_break_point(df : pd.DataFrame, features : list, dir_output : Path, n_clusters : int):
    check_and_create_path(dir_output)
    logger.info('###################" Calculate break point ####################')

    logger.info(f'Check for break_point_dict in {dir_output}')
    if (dir_output / 'break_point_dict.pkl').is_file():
        res_cluster = read_object('break_point_dict.pkl', dir_output)
    else:
        res_cluster = {}
    features_selected_2 = []

    FF_t = np.sum(df['nbsinister'])
    Area_t = len(df)

    # For each feature
    for fet in features:
        if fet in res_cluster.keys():
            continue
        check_and_create_path(dir_output / fet)
        logger.info(fet)
        res_cluster[fet] = {}
        X_fet = df[fet].values # Select the feature
        val_sinister = df['nbsinister'].values[~np.isnan(X_fet)]
        X_fet = X_fet[~np.isnan(X_fet)]
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
            FF_i = np.sum(val_sinister[mask])
            
            # Calculer Area_i (le nombre total de pixels pour la classe c)
            Area_i = mask.shape[0]
            
            # Calculer FireOcc et Area pour la classe c
            FireOcc = FF_i / FF_t
            Area = Area_i / Area_t
            
            # Calculer le ratio de fréquence (FR) pour la classe c
            FR = FireOcc / Area

            res_cluster[fet][c] = round(FR, 3)

            plt.scatter(X_fet[mask], val_sinister[mask], label=f'{c} : {res_cluster[fet][c]}')

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

def compute_weights_and_target(target_name, labels, band, ids_columns, is_grap_or_node, graphs):
    weight_idx = ids_columns.index('weight')
    target_is_binary = target_name == 'binary'

    if len(labels.shape) == 3:
        weights = labels[:, weight_idx, -1]
        target = (labels[:, band, -1] > 0).long() if target_is_binary else labels[:, band, -1]

    elif len(labels.shape) == 5:
        weights = labels[:, :, :, weight_idx, -1]
        target = (labels[:, :, :, band, -1] > 0).long() if target_is_binary else labels[:, :, :, band, -1]

    elif len(labels.shape) == 4:
        weights = labels[:, :, :, weight_idx,]
        target = (labels[:, :, :, band] > 0).long() if target_is_binary else labels[:, :, :, band]

    else:
        weights = labels[:, weight_idx]
        target = (labels[:, band] > 0).long() if target_is_binary else labels[:, band]
    
    if is_grap_or_node:
        unique_elements = torch.unique(graphs, return_inverse=False, return_counts=False, sorted=True)
        first_indices = torch.tensor([torch.nonzero(graphs == u, as_tuple=True)[0][0] for u in unique_elements])
        weights = weights[first_indices]
        target = target[first_indices]

    return target, weights

def compute_labels(labels, is_grap_or_node, graphs):
    if len(labels.shape) == 3:
        labels = labels[:, :, -1]
    elif len(labels.shape) == 5:
        labels = labels[:, :, :, :, -1]
    elif len(labels.shape) == 4:
        labels = labels
    else:
        labels = labels
    
    if is_grap_or_node:
        unique_elements = torch.unique(graphs, return_inverse=False, return_counts=False, sorted=True)
        first_indices = torch.tensor([torch.nonzero(graphs == u, as_tuple=True)[0][0] for u in unique_elements])
        labels = labels[first_indices]

    return labels

def launch_train_loader(model, loader,
                  features, target_name,
                  criterion, optimizer,
                  model_name):
    
    model.train()
    for i, data in enumerate(loader, 0):
        
        try:
            if model_name in models_hybrid:
                inputs_1D, inputs_2D, labels, edges, graphs = data
            else:
                inputs, labels, edges, graphs = data
        except:
            if model_name in models_hybrid:
                inputs_1D, inputs_2D, labels, edges = data
            else:
                inputs, labels, edges = data
            graphs = None

        if target_name == 'binary' or target_name == 'nbsinister':
            band = -2
        else:
            band = -1

        try:
            target, weights = compute_weights_and_target(target_name, labels, band, ids_columns, model.is_graph_or_node, graphs)
        except Exception as e:
            target, weights = compute_weights_and_target(target_name, labels, band, ids_columns, False, graphs)

        # Prepare inputs for the model
        if model_name in models_hybrid:
            inputs_1D = inputs_1D[:, features[0]]
            inputs_2D = inputs_2D[:, features[1]]
            output = model(inputs_1D, inputs_2D, edges, graphs)
        elif model_name in models_2D:
            inputs_model = inputs[:, features]
            output = model(inputs_model, edges, graphs)
        elif model_name in temporal_model_list:
            inputs_model = inputs[:, features]
            output = model(inputs_model, edges, graphs)
        else:
            inputs_model = inputs[:, features]
            output = model(inputs_model, edges, graphs)

        if target_name == 'risk' or target_name == 'nbsinister':
            #print(inputs.shape, labels.shape, target.shape, output.shape)

            target = target.view(output.shape)
            weights = weights.view(output.shape)
            
            target = torch.masked_select(target, weights.gt(0))
            output = torch.masked_select(output, weights.gt(0))
            weights = torch.masked_select(weights, weights.gt(0))
            loss = criterion(output, target, weights)
        else:
            target = torch.masked_select(target, weights.gt(0))
            output = output[weights.gt(0)]
            weights = torch.masked_select(weights, weights.gt(0))
            loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss

def launch_val_test_loader(model, loader,
                           features, target_name,
                           criterion, optimizer,
                           model_name):
    """
    Evaluates the model using the provided data loader, with optional autoregression.

    Note:
    - 'node_id' and 'date_id' are not included in features_name but can be found in labels[:, 0] and labels[:, 4].

    Parameters:
    - model: The PyTorch model to evaluate.
    - loader: DataLoader providing the validation or test data.
    - features: List or tuple of feature indices to use.
    - target_name: Name of the target variable.
    - criterion: Loss function.
    - optimizer: Optimizer (not used during evaluation but included for consistency).
    - autoRegression: Boolean indicating whether to use autoregression.
    - features_name: List of feature names.
    - hybrid: Boolean indicating if the model uses hybrid inputs.

    Returns:
    - total_loss: The cumulative loss over the dataset.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():

        for i, data in enumerate(loader, 0):
            
            try:
                if model_name in models_hybrid:
                    inputs_1D, inputs_2D, labels, edges, graphs = data
                else:
                    inputs, labels, edges, graphs = data
            except:
                if model_name in models_hybrid:
                    inputs_1D, inputs_2D, labels, edges = data
                else:
                    inputs, labels, edges = data
                graphs = None

            # Determine the index of the target variable in labels
            if target_name == 'binary' or target_name == 'nbsinister':
                band = -2
            else:
                band = -1

            try:
                target, weights = compute_weights_and_target(target_name, labels, band, ids_columns, model.is_graph_or_node, graphs)
            except Exception as e:
                target, weights = compute_weights_and_target(target_name, labels, band, ids_columns, False, graphs)

            # Prepare inputs for the model
            if model_name in models_hybrid:
                inputs_1D = inputs_1D[:, features[0]]
                inputs_2D = inputs_2D[:, features[1]]
                output = model(inputs_1D, inputs_2D, edges, graphs)
            elif model_name in models_2D:
                inputs_model = inputs[:, features]
                output = model(inputs_model, edges, graphs)
            elif model_name in temporal_model_list:
                inputs_model = inputs[:, features]
                output = model(inputs_model, edges, graphs)
            else:
                inputs_model = inputs[:, features]
                output = model(inputs_model, edges, graphs)

            # Compute loss
            if target_name == 'risk' or target_name == 'nbsinister':
                target = target.view(output.shape)
                weights = weights.view(output.shape)

                # Mask out invalid weights
                valid_mask = weights.gt(0)
                target = torch.masked_select(target, valid_mask)
                output = torch.masked_select(output, valid_mask)
                weights = torch.masked_select(weights, valid_mask)
                loss = criterion(output, target, weights)
            else:
                valid_mask = weights.gt(0)
                target = torch.masked_select(target, valid_mask)
                output = output[valid_mask]
                weights = torch.masked_select(weights, valid_mask)
                loss = criterion(output, target)

            total_loss += loss.item()

    return total_loss

def func_epoch(model, train_loader, val_loader, features,
               optimizer, criterion, target_name,
                model_name):
    
    train_loss = launch_train_loader(model, train_loader,
                  features, target_name,
                  criterion, optimizer,
                  model_name)

    if val_loader is not None:
        val_loss = launch_val_test_loader(model, val_loader,
                  features, target_name,
                  criterion, optimizer,
                  model_name)
    
    else:
        val_loss = train_loss

    return val_loss, train_loss

def plot_train_val_loss(epochs, train_loss_list, val_loss_list, dir_output):
    # Création de la figure et des axes
    plt.figure(figsize=(10, 6))

    # Tracé de la courbe de val_loss
    plt.plot(epochs, val_loss_list, label='Validation Loss', color='blue')

    # Ajout de la légende
    plt.legend()

    # Ajout des labels des axes
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Ajout d'un titre
    plt.title('Validation Loss over Epochs')
    plt.savefig(dir_output / 'Validation.png')
    plt.close('all')

    # Tracé de la courbe de train_loss
    plt.plot(epochs, train_loss_list, label='Training Loss', color='red')

    # Ajout de la légende
    plt.legend()

    # Ajout des labels des axes
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Ajout d'un titre
    plt.title('Training Loss over Epochs')
    plt.savefig(dir_output / 'Training.png')
    plt.close('all')

def train(params):
    """
    Train neural network model
    """
    train_loader = params['train_loader']
    val_loader = params['val_loader']
    test_loader = params['test_loader']
    graph = params['graph']
    optimize_feature = params['optimize_feature']
    PATIENCE_CNT = params['PATIENCE_CNT']
    CHECKPOINT = params['CHECKPOINT']
    lr = params['lr']
    epochs = params['epochs']
    loss_name = params['loss_name']
    features_selected_str = params['features_selected_str']
    features_selected = params['features_selected']
    modelname = params['modelname']
    dir_output = params['dir_output']
    target_name = params['target_name']
    autoRegression = params['autoRegression']
    k_days = params['k_days']
    if 'custom_model_params' in params.keys():
        custom_model_params = params['custom_model_params']
    else:
        custom_model_params = None

    if MLFLOW:
        existing_run = get_existing_run(f'{modelname}_')
        if existing_run:
            mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
        else:
            mlflow.start_run(run_name=f'{modelname}_', nested=True)

    assert train_loader is not None and val_loader is not None

    check_and_create_path(dir_output)

    criterion = get_loss_function(loss_name)

    if modelname in models_hybrid:
        model, _ = make_model(modelname, len(features_selected[0]), len(features_selected[1]),
                              graph, dropout, 'relu',
                              k_days,
                              target_name == 'binary',
                              device, num_lstm_layers,
                              custom_model_params)
    else:
        model, _ = make_model(modelname, len(features_selected), len(features_selected),
                              graph, dropout, 'relu',
                              k_days,
                              target_name == 'binary',
                              device, num_lstm_layers,
                              custom_model_params)
    
    #if (dir_output / '100.pt').is_file():
    #    model.load_state_dict(torch.load((dir_output / '100.pt'), map_location=device, weights_only=True), strict=False)
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    BEST_VAL_LOSS = math.inf
    BEST_MODEL_PARAMS = None
    patience_cnt = 0

    val_loss_list = []
    train_loss_list = []
    epochs_list = []

    logger.info('Train model with')
    save_object(features_selected, 'features.pkl', dir_output)
    save_object(features_selected_str, 'features_str.pkl', dir_output)
    for epoch in tqdm(range(epochs)):
        val_loss, train_loss = func_epoch(model, train_loader, val_loader, features_selected,
                                            optimizer, criterion, target_name,
                                                modelname)
        train_loss = train_loss.item()
        val_loss = round(val_loss, 3)
        train_loss = round(train_loss, 3)
        val_loss_list.append(val_loss)
        train_loss_list.append(train_loss)
        epochs_list.append(epoch)
        if val_loss < BEST_VAL_LOSS:
            BEST_VAL_LOSS = val_loss
            BEST_MODEL_PARAMS = model.state_dict()
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE_CNT:
                logger.info(f'Loss has not increased for {patience_cnt} epochs. Last best val loss {BEST_VAL_LOSS}, current val loss {val_loss}')
                save_object_torch(model.state_dict(), 'last.pt', dir_output)
                save_object_torch(BEST_MODEL_PARAMS, 'best.pt', dir_output)
                plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, dir_output)
                if MLFLOW:
                    mlflow.end_run()
                return
        if MLFLOW:
            mlflow.log_metric('loss', val_loss, step=epoch)
        if epoch % CHECKPOINT == 0:
            logger.info(f'epochs {epoch}, Val loss {val_loss}')
            logger.info(f'epochs {epoch}, Best val loss {BEST_VAL_LOSS}')
            save_object_torch(model.state_dict(), str(epoch)+'.pt', dir_output)

    logger.info(f'Last val loss {val_loss}')
    save_object_torch(model.state_dict(), 'last.pt', dir_output)
    save_object_torch(BEST_MODEL_PARAMS, 'best.pt', dir_output)
    plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, dir_output)
    
# Fonction pour sélectionner la fonction de perte via son nom
def get_loss_function(loss_name):
    # Dictionnaire pour associer le nom de la fonction de perte à sa classe correspondante
    loss_dict = {
        "poisson": PoissonLoss,
        "rmsle": RMSLELoss,
        "rmse": RMSELoss,
        "mse": MSELoss,
        "huber": HuberLoss,
        "logcosh": LogCoshLoss,
        "tukeybiweight": TukeyBiweightLoss,
        "exponential": ExponentialLoss,
        "weightedcrossentropy": WeightedCrossEntropyLoss
    }
    loss_name = loss_name.lower()
    if loss_name in loss_dict:
        return loss_dict[loss_name]()
    else:
        raise ValueError(f"Loss function '{loss_name}' not found in loss_dict.")

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