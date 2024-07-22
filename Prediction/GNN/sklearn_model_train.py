from visualize import *
from sklearn.linear_model import LinearRegression

def train_xgboost(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, features,
                  scale, pos_feature, optimize_feature, dir_output, device,
                  doGridSearch, doBayesSearch):
    
    ######################################## RISK ############################################ 
    Yw = Ytrain[:,-4]
    Y_train = Ytrain[:,-1]
    Y_val = Yval[:,-1]
    Y_test = Ytest[:, -1]
    W_test = Ytest[:, -4]
    fitparams = {
        'eval_set': [(Xtrain[:, features], Y_train), (Xval[:, features], Y_val)],
        'sample_weight': Yw,
        'verbose': False
    }
    
    model, grid = config_xgboost(device, False, 'reg:squarederror')
    fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, scale, pos_feature,
        'xgboost_risk', Model(model, 'rmse', 'xgboost_risk'), fitparams, 'skip', grid, dir_output)

    if optimize_feature:
        relavant_features = find_relevant_features(model=Model(model, 'rmse', 'noName'),
                                                   features=features,
                                                   X=Xtrain, y=Y_train, w=Yw, X_val=Xval, y_val=Y_val, w_val=None,
                                                   X_test=Xtest, y_test=Y_test, w_test=W_test)

        log_features(relavant_features, scale=scale, pos_feature=pos_feature, methods=METHODS)

        save_object(relavant_features, 'relevent_features.pkl', dir_output)

        fitparams = {
            'eval_set': [(Xtrain[:, relavant_features], Y_train), (Xval[:, relavant_features], Y_val)],
            'sample_weight': Yw,
            'verbose': False
        }

        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'xgboost_risk_features', Model(model, 'rmse', 'xgboost_risk_features'), fitparams, 'skip', grid, dir_output)
    else:
        relavant_features = features

    if doGridSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'xgboost_risk_grid_search', Model(model, 'rmse', 'xgboost_risk_grid_search'), fitparams, 'grid', grid, dir_output)    

    if doBayesSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'xgboost_risk_bayes_search', Model(model, 'rmse', 'xgboost_risk_bayes_search'), fitparams, 'bayes', grid, dir_output)    

    ############################################# NB SINISTER ##################################################

    Yw = Ytrain[:,-4]
    Y_train = Ytrain[:,-2]
    Y_val = Yval[:,-2]
    Y_test = Ytest[:, -2]
    W_test = Ytest[:, -4]
    fitparams = {
        'eval_set': [(Xtrain[:, features], Y_train), (Xval[:, features], Y_val)],
        'sample_weight': Yw,
        'verbose': False
    }
    model, grid = config_xgboost(device, False, 'reg:squarederror')
    fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, scale, pos_feature,
        'xgboost_nbsinister', Model(model, 'rmse', 'xgboost_nbsinister'), fitparams, 'skip', grid, dir_output)
    

    if optimize_feature:
        relavant_features = find_relevant_features(model=Model(model, 'rmse', 'noName'),
                                                   features=features,
                                                   X=Xtrain, y=Y_train, w=Yw, X_val=Xval, y_val=Y_val, w_val=None,
                                                   X_test=Xtest, y_test=Y_test, w_test=W_test)

        log_features(relavant_features, scale=scale, pos_feature=pos_feature, methods=METHODS)

        save_object(relavant_features, 'relevent_features.pkl', dir_output)

        fitparams = {
            'eval_set': [(Xtrain[:, relavant_features], Y_train), (Xval[:, relavant_features], Y_val)],
            'sample_weight': Yw,
            'verbose': False
        }

        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'xgboost_nbsinister_features', Model(model, 'rmse', 'xgboost_nbsinister_features'), fitparams, 'skip', grid, dir_output)
    else:
        relavant_features = features

    if doGridSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'xgboost_nbsinister_grid_search', Model(model, 'rmse', 'xgboost_nbsinister_grid_search'), fitparams, 'grid', grid, dir_output)    

    if doBayesSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'xgboost_nbsinister_bayes_search', Model(model, 'rmse', 'xgboost_nbsinister_bayes_search'), fitparams, 'bayes', grid, dir_output)   
    
    ################################################### BINARY ########################################################
    # xgboost Binary
    Yw = Ytrain[:,-4]
    Yw[Ytrain[:,-2] == 0] = 1
    W_test = Ytest[:,-4]
    W_test[Ytest[:,-2] == 0] = 1
    Y_train = Ytrain[:,-2] > 0
    Y_val = Yval[:,-2] > 0
    Y_test = Ytest[:, -2] > 0
    fitparams = {
        'eval_set': [(Xtrain[:, features], Y_train), (Xval[:, features], Y_val)],
        'sample_weight': Yw,
        'verbose': False
    }

    model, grid = config_xgboost(device, True, 'reg:logistic')
    fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, scale, pos_feature,
        'xgboost_binary', Model(model, 'log_loss', 'xgboost_binary'), fitparams, 'skip', grid, dir_output)
    
    if optimize_feature:
        model, grid = config_xgboost(device, True, 'reg:logistic')
        relavant_features_binary = find_relevant_features(model=Model(model, 'log_loss', 'noName'),
                                                          features=features,
                                                          X=Xtrain, y=Y_train, w=Yw, X_val=Xval, y_val=Y_val, w_val=None,
                                                          X_test=Xtest, y_test=Y_test, w_test=W_test)

        log_features(relavant_features_binary, scale=scale, pos_feature=pos_feature, methods=METHODS)

        save_object(relavant_features_binary, 'relevent_features_binary.pkl', dir_output)

        fitparams = {
            'eval_set': [(Xtrain[:, relavant_features_binary], Y_train), (Xval[:, relavant_features_binary], Y_val)],
            'sample_weight': Yw,
            'verbose': False
        }
    else:
        relavant_features_binary = features

    if doGridSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'xgboost_binary_grid_search', Model(model, 'rmse', 'xgboost_binary_grid_search'), fitparams, 'grid', grid, dir_output)    

    if doBayesSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'xgboost_binary_bayes_search', Model(model, 'rmse', 'xgboost_binary_bayes_search'), fitparams, 'bayes', grid, dir_output)    

def train_lightgbm(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, features,
                   scale, pos_feature, optimize_feature, dir_output, device,
                   doGridSearch, doBayesSearch):
    
    ######################################## RISK ############################################ 
    Yw = Ytrain[:,-4]
    Y_train = Ytrain[:,-1]
    Y_val = Yval[:,-1]
    Y_test = Ytest[:, -1]
    W_test = Ytest[:, -4]
    fitparams = {
        'eval_set': [(Xval[:, features], Y_val)],
        'sample_weight': Yw,
    }
    
    model, grid = config_lightGBM(device, False, 'root_mean_squared_error')
    fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, scale, pos_feature,
        'lightgbm_risk', Model(model, 'rmse', 'lightgbm_risk'), fitparams, 'skip', grid, dir_output)

    if optimize_feature:
        relavant_features = find_relevant_features(model=Model(model, 'rmse', 'noName'),
                                                   features=features,
                                                   X=Xtrain, y=Y_train, w=Yw, X_val=Xval, y_val=Y_val, w_val=None,
                                                   X_test=Xtest, y_test=Y_test, w_test=W_test)

        log_features(relavant_features, scale=scale, pos_feature=pos_feature, methods=METHODS)

        save_object(relavant_features, 'relevent_features.pkl', dir_output)

        fitparams = {
            'eval_set': [(Xval[:, relavant_features], Y_val)],
            'sample_weight': Yw,
        }

        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'lightgbm_risk_features', Model(model, 'rmse', 'lightgbm_risk_features'), fitparams, 'skip', grid, dir_output)
    else:
        relavant_features = features

    if doGridSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'lightgbm_risk_grid_search', Model(model, 'rmse', 'lightgbm_risk_grid_search'), fitparams, 'grid', grid, dir_output)    

    if doBayesSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'lightgbm_risk_bayes_search', Model(model, 'rmse', 'lightgbm_risk_bayes_search'), fitparams, 'bayes', grid, dir_output)    

    ############################################# NB SINISTER ##################################################

    Yw = Ytrain[:,-4]
    Y_train = Ytrain[:,-2]
    Y_val = Yval[:,-2]
    Y_test = Ytest[:, -2]
    W_test = Ytest[:, -4]
    fitparams = {
        'eval_set': [(Xval[:, features], Y_val)],
        'sample_weight': Yw,
    }
    model, grid = config_lightGBM(device, False, 'root_mean_squared_error')
    fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, scale, pos_feature,
        'lightgbm_nbsinister', Model(model, 'rmse', 'lightgbm_nbsinister'), fitparams, 'skip', grid, dir_output)
    

    if optimize_feature:
        relavant_features = find_relevant_features(model=Model(model, 'rmse', 'noName'),
                                                   features=features,
                                                   X=Xtrain, y=Y_train, w=Yw, X_val=Xval, y_val=Y_val, w_val=None,
                                                   X_test=Xtest, y_test=Y_test, w_test=W_test)

        log_features(relavant_features, scale=scale, pos_feature=pos_feature, methods=METHODS)

        save_object(relavant_features, 'relevent_features.pkl', dir_output)

        fitparams = {
            'eval_set': [(Xval[:, relavant_features], Y_val)],
            'sample_weight': Yw,
        }

        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'lightgbm_nbsinister_features', Model(model, 'rmse', 'lightgbm_nbsinister_features'), fitparams, 'skip', grid, dir_output)
    else:
        relavant_features = features

    if doGridSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'lightgbm_nbsinister_grid_search', Model(model, 'rmse', 'lightgbm_nbsinister_grid_search'), fitparams, 'grid', grid, dir_output)    

    if doBayesSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'lightgbm_nbsinister_bayes_search', Model(model, 'rmse', 'lightgbm_nbsinister_bayes_search'), fitparams, 'bayes', grid, dir_output)   
    
    ################################################### BINARY ########################################################
    # lightgbm Binary
    Yw = Ytrain[:,-4]
    Yw[Ytrain[:,-2] == 0] = 1
    W_test = Ytest[:,-4]
    W_test[Ytest[:,-2] == 0] = 1
    Y_train = Ytrain[:,-2] > 0
    Y_val = Yval[:,-2] > 0
    Y_test = Ytest[:, -2] > 0
    fitparams = {
        'eval_set': [(Xval[:, features], Y_val)],
        'sample_weight': Yw,
    }

    model, grid = config_lightGBM(device, True, 'binary')
    fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, scale, pos_feature,
        'lightgbm_binary', Model(model, 'log_loss', 'lightgbm_binary'), fitparams, 'skip', grid, dir_output)
    
    if optimize_feature:
        model, grid = config_lightGBM(device, True, 'binary')
        relavant_features_binary = find_relevant_features(model=Model(model, 'log_loss', 'noName'),
                                                          features=features,
                                                          X=Xtrain, y=Y_train, w=Yw, X_val=Xval, y_val=Y_val, w_val=None,
                                                          X_test=Xtest, y_test=Y_test, w_test=W_test)

        log_features(relavant_features_binary, scale=scale, pos_feature=pos_feature, methods=METHODS)

        save_object(relavant_features_binary, 'relevent_features_binary.pkl', dir_output)

        fitparams = {
            'eval_set': [(Xval[:, relavant_features_binary], Y_val)],
            'sample_weight': Yw,
        }
    else:
        relavant_features_binary = features

    if doGridSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'lightgbm_binary_grid_search', Model(model, 'log_loss', 'lightgbm_binary_grid_search'), fitparams, 'grid', grid, dir_output)    

    if doBayesSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'lightgbm_binary_bayes_search', Model(model, 'log_loss', 'lightgbm_binary_bayes_search'), fitparams, 'bayes', grid, dir_output)    

def train_ngboost(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, features,
                  scale, pos_feature, optimize_feature, dir_output,
                  doGridSearch, doBayesSearch):
    
    ######################################## RISK ############################################ 
    Yw = Ytrain[:,-4]
    Y_train = Ytrain[:,-1]
    Y_val = Yval[:,-1]
    Y_test = Ytest[:, -1]
    W_test = Ytest[:, -4]
    fitparams = {
        'early_stopping_rounds': 15,
        'sample_weight': Yw,
        'X_val': Xval[:, features],
        'Y_val': Y_val,
    }
    
    model, grid = config_ngboost(False)
    fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, scale, pos_feature,
        'ngboost_risk', Model(model, 'rmse', 'ngboost_risk'), fitparams, 'skip', grid, dir_output)

    if optimize_feature:
        relavant_features = find_relevant_features(model=Model(model, 'rmse', 'noName'),
                                                   features=features,
                                                   X=Xtrain, y=Y_train, w=Yw, X_val=Xval, y_val=Y_val, w_val=None,
                                                   X_test=Xtest, y_test=Y_test, w_test=W_test)

        log_features(relavant_features, scale=scale, pos_feature=pos_feature, methods=METHODS)

        save_object(relavant_features, 'relevent_features.pkl', dir_output)

        fitparams = {
            'early_stopping_rounds': 15,
            'sample_weight': Yw,
            'X_val': Xval[:, relavant_features],
            'Y_val': Y_val,
        }

        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'ngboost_risk_features', Model(model, 'rmse', 'ngboost_risk_features'), fitparams, 'skip', grid, dir_output)
    else:
        relavant_features = features

    if doGridSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'ngboost_risk_grid_search', Model(model, 'rmse', 'ngboost_risk_grid_search'), fitparams, 'grid', grid, dir_output)    

    if doBayesSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'ngboost_risk_bayes_search', Model(model, 'rmse', 'ngboost_risk_bayes_search'), fitparams, 'bayes', grid, dir_output)    

    ############################################# NB SINISTER ##################################################

    Yw = Ytrain[:,-4]
    Y_train = Ytrain[:,-2]
    Y_val = Yval[:,-2]
    Y_test = Ytest[:, -2]
    W_test = Ytest[:, -4]
    fitparams = {
        'early_stopping_rounds': 15,
        'sample_weight': Yw,
        'X_val': Xval[:, features],
        'Y_val': Y_val,
    }
    model, grid = config_ngboost(False)
    fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, scale, pos_feature,
        'ngboost_nbsinister', Model(model, 'rmse', 'ngboost_nbsinister'), fitparams, 'skip', grid, dir_output)
    

    if optimize_feature:
        relavant_features = find_relevant_features(model=Model(model, 'rmse', 'noName'),
                                                   features=features,
                                                   X=Xtrain, y=Y_train, w=Yw, X_val=Xval, y_val=Y_val, w_val=None,
                                                   X_test=Xtest, y_test=Y_test, w_test=W_test)

        log_features(relavant_features, scale=scale, pos_feature=pos_feature, methods=METHODS)

        save_object(relavant_features, 'relevent_features.pkl', dir_output)

        fitparams = {
            'early_stopping_rounds': 15,
            'sample_weight': Yw,
            'X_val': Xval[:, relavant_features],
            'Y_val': Y_val,
        }

        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'ngboost_nbsinister_features', Model(model, 'rmse', 'ngboost_nbsinister_features'), fitparams, 'skip', grid, dir_output)
    else:
        relavant_features = features

    if doGridSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'ngboost_nbsinister_grid_search', Model(model, 'rmse', 'ngboost_nbsinister_grid_search'), fitparams, 'grid', grid, dir_output)    

    if doBayesSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'ngboost_nbsinister_bayes_search', Model(model, 'rmse', 'ngboost_nbsinister_bayes_search'), fitparams, 'bayes', grid, dir_output)   
    
    ################################################### BINARY ########################################################
    # ngboost Binary
    Yw = Ytrain[:,-4]
    Yw[Ytrain[:,-2] == 0] = 1
    W_test = Ytest[:,-4]
    W_test[Ytest[:,-2] == 0] = 1
    Y_train = Ytrain[:,-2] > 0
    Y_val = Yval[:,-2] > 0
    Y_test = Ytest[:, -2] > 0
    fitparams = {
        'early_stopping_rounds': 15,
        'sample_weight': Yw,
        'X_val': Xval[:, features],
        'Y_val': Y_val,
    }

    model, grid = config_ngboost(True)
    fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, scale, pos_feature,
        'ngboost_binary', Model(model, 'log_loss', 'ngboost_binary'), fitparams, 'skip', grid, dir_output)
    
    if optimize_feature:
        model, grid = config_ngboost(True)
        relavant_features_binary = find_relevant_features(model=Model(model, 'log_loss', 'noName'),
                                                          features=features,
                                                          X=Xtrain, y=Y_train, w=Yw, X_val=Xval, y_val=Y_val, w_val=None,
                                                          X_test=Xtest, y_test=Y_test, w_test=W_test)

        log_features(relavant_features_binary, scale=scale, pos_feature=pos_feature, methods=METHODS)

        save_object(relavant_features_binary, 'relevent_features_binary.pkl', dir_output)

        fitparams = {
            'early_stopping_rounds': 15,
            'sample_weight': Yw,
            'X_val': Xval[:, relavant_features_binary],
            'Y_val': Y_val,
        }
    else:
        relavant_features_binary = features

    if doGridSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'ngboost_binary_grid_search', Model(model, 'log_loss', 'ngboost_binary_grid_search'), fitparams, 'grid', grid, dir_output)    

    if doBayesSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'ngboost_binary_bayes_search', Model(model, 'log_loss', 'ngboost_binary_bayes_search'), fitparams, 'bayes', grid, dir_output)
        
def train_svm(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, features,
              scale, pos_feature, optimize_feature, dir_output, device,
              doGridSearch, doBayesSearch):
    
    ######################################## RISK ############################################ 
    Yw = Ytrain[:, -4]
    Y_train = Ytrain[:, -1]
    Y_val = Yval[:, -1]
    Y_test = Ytest[:, -1]
    W_test = Ytest[:, -4]
    fitparams = {
        'eval_set': [(Xval[:, features], Y_val)],
        'sample_weight': Yw,
    }
    
    model, grid = config_svm(device, 'regression')
    fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, scale, pos_feature,
        'svm_risk', Model(model, 'rmse', 'svm_risk'), fitparams, 'skip', grid, dir_output)

    if optimize_feature:
        relavant_features = find_relevant_features(model=Model(model, 'rmse', 'noName'),
                                                   features=features,
                                                   X=Xtrain, y=Y_train, w=Yw, X_val=Xval, y_val=Y_val, w_val=None,
                                                   X_test=Xtest, y_test=Y_test, w_test=W_test)

        log_features(relavant_features, scale=scale, pos_feature=pos_feature, methods=METHODS)

        save_object(relavant_features, 'relevent_features.pkl', dir_output)

        fitparams = {
            'eval_set': [(Xval[:, relavant_features], Y_val)],
            'sample_weight': Yw,
        }

        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'svm_risk_features', Model(model, 'rmse', 'svm_risk_features'), fitparams, 'skip', grid, dir_output)
    else:
        relavant_features = features

    if doGridSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'svm_risk_grid_search', Model(model, 'rmse', 'svm_risk_grid_search'), fitparams, 'grid', grid, dir_output)    

    if doBayesSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'svm_risk_bayes_search', Model(model, 'rmse', 'svm_risk_bayes_search'), fitparams, 'bayes', grid, dir_output)    

    ############################################# NB SINISTER ##################################################

    Yw = Ytrain[:, -4]
    Y_train = Ytrain[:, -2]
    Y_val = Yval[:, -2]
    Y_test = Ytest[:, -2]
    W_test = Ytest[:, -4]
    fitparams = {
        'eval_set': [(Xval[:, features], Y_val)],
        'sample_weight': Yw,
    }
    model, grid = config_svm(device, 'regression')
    fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, scale, pos_feature,
        'svm_nbsinister', Model(model, 'rmse', 'svm_nbsinister'), fitparams, 'skip', grid, dir_output)
    

    if optimize_feature:
        relavant_features = find_relevant_features(model=Model(model, 'rmse', 'noName'),
                                                   features=features,
                                                   X=Xtrain, y=Y_train, w=Yw, X_val=Xval, y_val=Y_val, w_val=None,
                                                   X_test=Xtest, y_test=Y_test, w_test=W_test)

        log_features(relavant_features, scale=scale, pos_feature=pos_feature, methods=METHODS)

        save_object(relavant_features, 'relevent_features.pkl', dir_output)

        fitparams = {
            'eval_set': [(Xval[:, relavant_features], Y_val)],
            'sample_weight': Yw,
        }

        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'svm_nbsinister_features', Model(model, 'rmse', 'svm_nbsinister_features'), fitparams, 'skip', grid, dir_output)
    else:
        relavant_features = features

    if doGridSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'svm_nbsinister_grid_search', Model(model, 'rmse', 'svm_nbsinister_grid_search'), fitparams, 'grid', grid, dir_output)    

    if doBayesSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'svm_nbsinister_bayes_search', Model(model, 'rmse', 'svm_nbsinister_bayes_search'), fitparams, 'bayes', grid, dir_output)   
    
    ################################################### BINARY ########################################################
    # svm Binary
    Yw = Ytrain[:, -4]
    Yw[Ytrain[:, -2] == 0] = 1
    W_test = Ytest[:, -4]
    W_test[Ytest[:, -2] == 0] = 1
    Y_train = Ytrain[:, -2] > 0
    Y_val = Yval[:, -2] > 0
    Y_test = Ytest[:, -2] > 0
    fitparams = {
        'eval_set': [(Xval[:, features], Y_val)],
        'sample_weight': Yw,
    }

    model, grid = config_svm(device, 'classification')
    fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, scale, pos_feature,
        'svm_binary', Model(model, 'log_loss', 'svm_binary'), fitparams, 'skip', grid, dir_output)
    
    if optimize_feature:
        model, grid = config_svm(device, 'classification')
        relavant_features_binary = find_relevant_features(model=Model(model, 'log_loss', 'noName'),
                                                          features=features,
                                                          X=Xtrain, y=Y_train, w=Yw, X_val=Xval, y_val=Y_val, w_val=None,
                                                          X_test=Xtest, y_test=Y_test, w_test=W_test)

        log_features(relavant_features_binary, scale=scale, pos_feature=pos_feature, methods=METHODS)

        save_object(relavant_features_binary, 'relevent_features_binary.pkl', dir_output)

        fitparams = {
            'eval_set': [(Xval[:, relavant_features_binary], Y_val)],
            'sample_weight': Yw,
        }
    else:
        relavant_features_binary = features

    if doGridSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features_binary, scale, pos_feature,
            'svm_binary_grid_search', Model(model, 'log_loss', 'svm_binary_grid_search'), fitparams, 'grid', grid, dir_output)    

    if doBayesSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features_binary, scale, pos_feature,
            'svm_binary_bayes_search', Model(model, 'log_loss', 'svm_binary_bayes_search'), fitparams, 'bayes', grid, dir_output)
        
def train_random_forest(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, features,
                        scale, pos_feature, optimize_feature, dir_output, device,
                        doGridSearch, doBayesSearch):
    
    ######################################## RISK ############################################ 
    Yw = Ytrain[:, -4]
    Y_train = Ytrain[:, -1]
    Y_val = Yval[:, -1]
    Y_test = Ytest[:, -1]
    W_test = Ytest[:, -4]
    fitparams = {
        'eval_set': [(Xval[:, features], Y_val)],
        'sample_weight': Yw,
    }
    
    model, grid = config_random_forest(device, 'regression')
    fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, scale, pos_feature,
        'random_forest_risk', Model(model, 'rmse', 'random_forest_risk'), fitparams, 'skip', grid, dir_output)

    if optimize_feature:
        relavant_features = find_relevant_features(model=Model(model, 'rmse', 'noName'),
                                                   features=features,
                                                   X=Xtrain, y=Y_train, w=Yw, X_val=Xval, y_val=Y_val, w_val=None,
                                                   X_test=Xtest, y_test=Y_test, w_test=W_test)

        log_features(relavant_features, scale=scale, pos_feature=pos_feature, methods=METHODS)

        save_object(relavant_features, 'relevent_features.pkl', dir_output)

        fitparams = {
            'eval_set': [(Xval[:, relavant_features], Y_val)],
            'sample_weight': Yw,
        }

        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'random_forest_risk_features', Model(model, 'rmse', 'random_forest_risk_features'), fitparams, 'skip', grid, dir_output)
    else:
        relavant_features = features

    if doGridSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'random_forest_risk_grid_search', Model(model, 'rmse', 'random_forest_risk_grid_search'), fitparams, 'grid', grid, dir_output)    

    if doBayesSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'random_forest_risk_bayes_search', Model(model, 'rmse', 'random_forest_risk_bayes_search'), fitparams, 'bayes', grid, dir_output)    

    ############################################# NB SINISTER ##################################################

    Yw = Ytrain[:, -4]
    Y_train = Ytrain[:, -2]
    Y_val = Yval[:, -2]
    Y_test = Ytest[:, -2]
    W_test = Ytest[:, -4]
    fitparams = {
        'eval_set': [(Xval[:, features], Y_val)],
        'sample_weight': Yw,
    }
    model, grid = config_random_forest(device, 'regression')
    fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, scale, pos_feature,
        'random_forest_nbsinister', Model(model, 'rmse', 'random_forest_nbsinister'), fitparams, 'skip', grid, dir_output)
    

    if optimize_feature:
        relavant_features = find_relevant_features(model=Model(model, 'rmse', 'noName'),
                                                   features=features,
                                                   X=Xtrain, y=Y_train, w=Yw, X_val=Xval, y_val=Y_val, w_val=None,
                                                   X_test=Xtest, y_test=Y_test, w_test=W_test)

        log_features(relavant_features, scale=scale, pos_feature=pos_feature, methods=METHODS)

        save_object(relavant_features, 'relevent_features.pkl', dir_output)

        fitparams = {
            'eval_set': [(Xval[:, relavant_features], Y_val)],
            'sample_weight': Yw,
        }

        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'random_forest_nbsinister_features', Model(model, 'rmse', 'random_forest_nbsinister_features'), fitparams, 'skip', grid, dir_output)
    else:
        relavant_features = features

    if doGridSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'random_forest_nbsinister_grid_search', Model(model, 'rmse', 'random_forest_nbsinister_grid_search'), fitparams, 'grid', grid, dir_output)    

    if doBayesSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'random_forest_nbsinister_bayes_search', Model(model, 'rmse', 'random_forest_nbsinister_bayes_search'), fitparams, 'bayes', grid, dir_output)   
    
    ################################################### BINARY ########################################################
    # random_forest Binary
    Yw = Ytrain[:, -4]
    Yw[Ytrain[:, -2] == 0] = 1
    W_test = Ytest[:, -4]
    W_test[Ytest[:, -2] == 0] = 1
    Y_train = Ytrain[:, -2] > 0
    Y_val = Yval[:, -2] > 0
    Y_test = Ytest[:, -2] > 0
    fitparams = {
        'eval_set': [(Xval[:, features], Y_val)],
        'sample_weight': Yw,
    }

    model, grid = config_random_forest(device, 'classification')
    fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, scale, pos_feature,
        'random_forest_binary', Model(model, 'log_loss', 'random_forest_binary'), fitparams, 'skip', grid, dir_output)
    
    if optimize_feature:
        model, grid = config_random_forest(device, 'classification')
        relavant_features_binary = find_relevant_features(model=Model(model, 'log_loss', 'noName'),
                                                          features=features,
                                                          X=Xtrain, y=Y_train, w=Yw, X_val=Xval, y_val=Y_val, w_val=None,
                                                          X_test=Xtest, y_test=Y_test, w_test=W_test)

        log_features(relavant_features_binary, scale=scale, pos_feature=pos_feature, methods=METHODS)

        save_object(relavant_features_binary, 'relevent_features_binary.pkl', dir_output)

        fitparams = {
            'eval_set': [(Xval[:, relavant_features_binary], Y_val)],
            'sample_weight': Yw,
        }
    else:
        relavant_features_binary = features

    if doGridSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features_binary, scale, pos_feature,
            'random_forest_binary_grid_search', Model(model, 'log_loss', 'random_forest_binary_grid_search'), fitparams, 'grid', grid, dir_output)    

    if doBayesSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features_binary, scale, pos_feature,
            'random_forest_binary_bayes_search', Model(model, 'log_loss', 'random_forest_binary_bayes_search'), fitparams, 'bayes', grid, dir_output)    

def train_decision_tree(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, features,
                        scale, pos_feature, optimize_feature, dir_output, device,
                        doGridSearch, doBayesSearch):
    
    ######################################## RISK ############################################ 
    Yw = Ytrain[:, -4]
    Y_train = Ytrain[:, -1]
    Y_val = Yval[:, -1]
    Y_test = Ytest[:, -1]
    W_test = Ytest[:, -4]
    fitparams = {
        'eval_set': [(Xval[:, features], Y_val)],
        'sample_weight': Yw,
    }
    
    model, grid = config_decision_tree(device, 'regression')
    fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, scale, pos_feature,
        'decision_tree_risk', Model(model, 'rmse', 'decision_tree_risk'), fitparams, 'skip', grid, dir_output)

    if optimize_feature:
        relavant_features = find_relevant_features(model=Model(model, 'rmse', 'noName'),
                                                   features=features,
                                                   X=Xtrain, y=Y_train, w=Yw, X_val=Xval, y_val=Y_val, w_val=None,
                                                   X_test=Xtest, y_test=Y_test, w_test=W_test)

        log_features(relavant_features, scale=scale, pos_feature=pos_feature, methods=METHODS)

        save_object(relavant_features, 'relevent_features.pkl', dir_output)

        fitparams = {
            'eval_set': [(Xval[:, relavant_features], Y_val)],
            'sample_weight': Yw,
        }

        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'decision_tree_risk_features', Model(model, 'rmse', 'decision_tree_risk_features'), fitparams, 'skip', grid, dir_output)
    else:
        relavant_features = features

    if doGridSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'decision_tree_risk_grid_search', Model(model, 'rmse', 'decision_tree_risk_grid_search'), fitparams, 'grid', grid, dir_output)    

    if doBayesSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'decision_tree_risk_bayes_search', Model(model, 'rmse', 'decision_tree_risk_bayes_search'), fitparams, 'bayes', grid, dir_output)    

    ############################################# NB SINISTER ##################################################

    Yw = Ytrain[:, -4]
    Y_train = Ytrain[:, -2]
    Y_val = Yval[:, -2]
    Y_test = Ytest[:, -2]
    W_test = Ytest[:, -4]
    fitparams = {
        'eval_set': [(Xval[:, features], Y_val)],
        'sample_weight': Yw,
    }
    model, grid = config_decision_tree(device, 'regression')
    fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, scale, pos_feature,
        'decision_tree_nbsinister', Model(model, 'rmse', 'decision_tree_nbsinister'), fitparams, 'skip', grid, dir_output)
    

    if optimize_feature:
        relavant_features = find_relevant_features(model=Model(model, 'rmse', 'noName'),
                                                   features=features,
                                                   X=Xtrain, y=Y_train, w=Yw, X_val=Xval, y_val=Y_val, w_val=None,
                                                   X_test=Xtest, y_test=Y_test, w_test=W_test)

        log_features(relavant_features, scale=scale, pos_feature=pos_feature, methods=METHODS)

        save_object(relavant_features, 'relevent_features.pkl', dir_output)

        fitparams = {
            'eval_set': [(Xval[:, relavant_features], Y_val)],
            'sample_weight': Yw,
        }

        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'decision_tree_nbsinister_features', Model(model, 'rmse', 'decision_tree_nbsinister_features'), fitparams, 'skip', grid, dir_output)
    else:
        relavant_features = features

    if doGridSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'decision_tree_nbsinister_grid_search', Model(model, 'rmse', 'decision_tree_nbsinister_grid_search'), fitparams, 'grid', grid, dir_output)    

    if doBayesSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features, scale, pos_feature,
            'decision_tree_nbsinister_bayes_search', Model(model, 'rmse', 'decision_tree_nbsinister_bayes_search'), fitparams, 'bayes', grid, dir_output)   
    
    ################################################### BINARY ########################################################
    # decision_tree Binary
    Yw = Ytrain[:, -4]
    Yw[Ytrain[:, -2] == 0] = 1
    W_test = Ytest[:, -4]
    W_test[Ytest[:, -2] == 0] = 1
    Y_train = Ytrain[:, -2] > 0
    Y_val = Yval[:, -2] > 0
    Y_test = Ytest[:, -2] > 0
    fitparams = {
        'eval_set': [(Xval[:, features], Y_val)],
        'sample_weight': Yw,
    }

    model, grid = config_decision_tree(device, 'classification')
    fit(Xtrain, Y_train, Xtest, Y_test, W_test, features, scale, pos_feature,
        'decision_tree_binary', Model(model, 'log_loss', 'decision_tree_binary'), fitparams, 'skip', grid, dir_output)
    
    if optimize_feature:
        model, grid = config_decision_tree(device, 'classification')
        relavant_features_binary = find_relevant_features(model=Model(model, 'log_loss', 'noName'),
                                                          features=features,
                                                          X=Xtrain, y=Y_train, w=Yw, X_val=Xval, y_val=Y_val, w_val=None,
                                                          X_test=Xtest, y_test=Y_test, w_test=W_test)

        log_features(relavant_features_binary, scale=scale, pos_feature=pos_feature, methods=METHODS)

        save_object(relavant_features_binary, 'relevent_features_binary.pkl', dir_output)

        fitparams = {
            'eval_set': [(Xval[:, relavant_features_binary], Y_val)],
            'sample_weight': Yw,
        }
    else:
        relavant_features_binary = features

    if doGridSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features_binary, scale, pos_feature,
            'decision_tree_binary_grid_search', Model(model, 'log_loss', 'decision_tree_binary_grid_search'), fitparams, 'grid', grid, dir_output)    

    if doBayesSearch:
        fit(Xtrain, Y_train, Xtest, Y_test, W_test, relavant_features_binary, scale, pos_feature,
            'decision_tree_binary_bayes_search', Model(model, 'log_loss', 'decision_tree_binary_bayes_search'), fitparams, 'bayes', grid, dir_output)    
        

def fit(Xtrain, y, Xtest, Ytest, Wtest, features,
        scale, pos_feature,
        name, model, fitparams,
        parameter_optimization_method,
        grid, dir_output):
    
    save_object(features, 'features.pkl', dir_output / name)
    logger.info(f'Fitting model {name}')
    model.fit(X=Xtrain[:, features], y=y,
                optimization=parameter_optimization_method,
                param_grid=grid, fit_params=fitparams)

    check_and_create_path(dir_output / name)
    save_object(model, name + '.pkl', dir_output / name)

    create_feature_map(features=features, pos_feature=pos_feature, dir_output=dir_output/name, methods=METHODS, scale=scale)

    logger.info(f'Model score {model.score(Xtest[:, features], Ytest, Wtest)}')