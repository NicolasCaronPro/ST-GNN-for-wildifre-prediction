import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory
parent_dir = os.path.dirname(current_dir)

# Insert the parent directory into sys.path
sys.path.insert(0, parent_dir)
from GNN.construct import *
import geopandas as gpd
import argparse

########################### Input Arg ######################################

parser = argparse.ArgumentParser(
    prog='Train',
    description='Create graph and database according to config.py and tained model',
)
parser.add_argument('-n', '--name', type=str, help='Name of the experiment')
parser.add_argument('-e', '--encoder', type=str, help='Create encoder model')
parser.add_argument('-s', '--sinister', type=str, help='Sinister type')
parser.add_argument('-p', '--point', type=str, help='Construct points')
parser.add_argument('-np', '--nbpoint', type=str, help='Number of points')
parser.add_argument('-d', '--database', type=str, help='Do database')
parser.add_argument('-g', '--graph', type=str, help='Construct graph')
parser.add_argument('-mxd', '--maxDate', type=str, help='Limit train and validation date')
parser.add_argument('-mxdv', '--trainDate', type=str, help='Limit training date')
parser.add_argument('-f', '--featuresSelection', type=str, help='Do features selection')
parser.add_argument('-dd', '--database2D', type=str, help='Do 2D database')
parser.add_argument('-sc', '--scale', type=str, help='Scale')
parser.add_argument('-dataset', '--dataset', type=str, help='Dataset to use')
parser.add_argument('-nf', '--NbFeatures', type=str, help='Number de Features')
parser.add_argument('-of', '--optimizeFeature', type=str, help='Launch test')
parser.add_argument('-gs', '--GridSearch', type=str, help='GridSearch')
parser.add_argument('-bs', '--BayesSearch', type=str, help='BayesSearch')
parser.add_argument('-test', '--doTest', type=str, help='Launch test')
parser.add_argument('-train', '--doTrain', type=str, help='Launch train')
parser.add_argument('-r', '--resolution', type=str, help='Resolution of image')
parser.add_argument('-pca', '--pca', type=str, help='Apply PCA')
parser.add_argument('-kmeans', '--KMEANS', type=str, help='Apply kmeans preprocessing')
parser.add_argument('-ncluster', '--ncluster', type=str, help='Number of cluster for kmeans')
parser.add_argument('-shift', '--shift', type=str, help='Shift of kmeans', default='0')
parser.add_argument('-thresh_kmeans', '--thresh_kmeans', type=str, help='Thresh of fr to remove sinister', default='0')
parser.add_argument('-k_days', '--k_days', type=str, help='k_days')
parser.add_argument('-days_in_futur', '--days_in_futur', type=str, help='days_in_futur')
parser.add_argument('-scaling', '--scaling', type=str, help='scaling methods')
parser.add_argument('-graphConstruct', '--graphConstruct', type=str, help='')
parser.add_argument('-sinisterEncoding', '--sinisterEncoding', type=str, help='')
parser.add_argument('-weights', '--weights', type=str, help='Type of weights')
parser.add_argument('-top_cluster', '--top_cluster', type=str, help='Top x cluster (on 5)')
parser.add_argument('-graph_method', '--graph_method', type=str, help='Top x cluster (on 5)', default='node')
parser.add_argument('-training_mode', '--training_mode', type=str, help='training_mode', default='normal')

args = parser.parse_args()

QUICK = True

# Input config
dataset_name = args.dataset
name_exp = args.name
maxDate = args.maxDate
trainDate = args.trainDate
doEncoder = args.encoder == "True"
doPoint = args.point == "True"
doGraph = args.graph == "True"
doDatabase = args.database == "True"
do2D = args.database2D == "True"
doFet = args.featuresSelection == "True"
optimize_feature = args.optimizeFeature == "True"
doTest = args.doTest == "True"
doTrain = args.doTrain == "True"
nbfeatures = args.NbFeatures
sinister = args.sinister
values_per_class = args.nbpoint
scale = int(args.scale) if args.scale != 'departement' else args.scale
resolution = args.resolution
doPCA = args.pca == 'True'
doKMEANS = args.KMEANS == 'True'
ncluster = int(args.ncluster)
do_grid_search = args.GridSearch ==  'True'
do_bayes_search = args.BayesSearch == 'True'
k_days = int(args.k_days) # Size of the time series sequence use by DL models
days_in_futur = int(args.days_in_futur) # The target time validation
scaling = args.scaling
graph_construct = args.graphConstruct
sinister_encoding = args.sinisterEncoding
weights_version = args.weights
top_cluster = args.top_cluster
graph_method = args.graph_method
shift = args.shift
thresh_kmeans = args.thresh_kmeans
training_mode = args.training_mode
######################## Get features and train features list ######################

isInference = name_exp == 'inference'

features, train_features, kmeasn_features = get_features_for_sinister_prediction(dataset_name, sinister, isInference)

######################## Get departments and train departments #######################

departements, train_departements = select_departments(dataset_name, sinister)

############################# GLOBAL VARIABLES #############################

name_exp = f'{sinister_encoding}_{name_exp}'

dir_target = root_target / sinister / dataset_name / sinister_encoding / 'log' / resolution

name_dir = dataset_name + '/' + sinister + '/' + resolution + '/' + 'train' +  '/'
dir_output = Path(name_dir)
check_and_create_path(dir_output)
if dataset_name == 'firemen2':
    two = True
    dataset_name = 'firemen'
else:
    two = False

minDate = '2017-06-12' # Starting point

if dataset_name== '':
    dataset_name= 'firemen'

autoRegression = 'AutoRegressionReg' in train_features
if autoRegression:
    name_exp += '_AutoRegressionReg'

if not QUICK:
    ####################### INIT ################################

    df, graphScale, prefix, fp, features_selected = init(args, dir_output, 'train_gnn')
    save_object(df.columns, 'features_name.pkl', dir_output)

    if MLFLOW:
        exp_name = f"{dataset_name}_train"
        experiments = client.search_smote_experiments()

        if exp_name not in list(map(lambda x: x.name, experiments)):
            tags = {
                "mlflow.note.content": "This experiment is an example of how to use mlflow. The project allows to predict housing prices in california.",
            }

            client.create_experiment(name=exp_name, tags=tags)

        mlflow.set_experiment(exp_name)
        existing_run = get_existing_run('Preprocessing')
        if existing_run:
            mlflow.start_run(run_id=existing_run.info.run_id)
        else:
            mlflow.start_run(run_name='Preprocessing')

    ############################# Train, Val, test ###########################
    dir_output = dir_output / name_exp

    prefix_config = deepcopy(prefix)

    train_dataset, val_dataset, test_dataset, train_dataset_unscale, val_dataset_unscale, test_dataset_unscale, prefix, features_selected = get_train_val_test_set(graphScale, df,
                                                                                        features_selected, train_departements,
                                                                                        prefix,
                                                                                        dir_output,
                                                                                        args)

    varying_time_variables_2 = get_time_columns(varying_time_variables, k_days, train_dataset.copy(), train_features)
    features_name, newshape = get_features_name_list(graphScale.scale, train_features, METHODS_SPATIAL_TRAIN)
    features_selected_str = get_features_selected_for_time_series(features_selected, features_name, varying_time_variables_2)

    features_selected_str = list(features_selected_str)
    features_selected = np.arange(0, len(features_selected_str))
    logger.info((features_selected_str, len(features_selected_str)))

    if MLFLOW:
        train_dataset_ml_flow = mlflow.data.from_pandas(train_dataset)
        val_dataset_ml_flow = mlflow.data.from_pandas(val_dataset)
        test_dataset_ml_flow = mlflow.data.from_pandas(test_dataset)

        mlflow.log_param('train_features', train_features)
        mlflow.log_param('features', features)
        mlflow.log_param('features_selected_str', features_selected_str)
        mlflow.log_input(train_dataset_ml_flow, context='trainig')
        mlflow.log_input(val_dataset_ml_flow, context='validation')
        mlflow.log_input(test_dataset_ml_flow, context='testing')

        mlflow.end_run()

    ############################# Training ##################################

    test_dataset_unscale['weight_nbsinister'] = 1
    test_dataset['weight'] = 1

    name = 'check_'+scaling + '/' + prefix + '/' + 'baseline'

    ###################### Defined ClassRisk model ######################

    dir_post_process = dir_output / 'post_process'

    post_process_model_dico, train_dataset, val_dataset, test_dataset, new_cols = post_process_model(train_dataset, val_dataset, test_dataset, dir_post_process, graphScale)
    features_selected_str.append('Past_risk')
    features_selected = np.arange(0, len(features_selected_str))

    train_dataset = add_past_risk(train_dataset, 'nbsinister-kmeans-5-Class-Dept-cubic-Specialized-Past')
    test_dataset = add_past_risk(test_dataset, 'nbsinister-kmeans-5-Class-Dept-cubic-Specialized-Past')
    val_dataset = add_past_risk(val_dataset, 'nbsinister-kmeans-5-Class-Dept-cubic-Specialized-Past')

    save_object(train_dataset, 'df_train_'+prefix+'.pkl', dir_output)
    save_object(val_dataset, 'df_val_'+prefix+'.pkl', dir_output)
    save_object(test_dataset, 'df_test_'+prefix+'.pkl', dir_output)

else:
    post_process_model_dico = None

    prefix = f'full_{scale}_{days_in_futur}_{graph_construct}_{graph_method}'

    name = 'check_'+scaling + '/' + prefix + '/' + 'baseline'
    
    graphScale = read_object(f'graph_{scale}_{graph_construct}_{graph_method}.pkl', dir_output)

    dir_output = dir_output / name_exp
    
    train_dataset = read_object(f'df_train_{prefix}.pkl', dir_output)
    val_dataset = read_object(f'df_val_{prefix}.pkl', dir_output)
    test_dataset = read_object(f'df_test_{prefix}.pkl', dir_output)

    train_dataset_unscale = read_object(f'df_unscaled_train_{prefix}.pkl', dir_output)
    val_dataset_unscale = read_object(f'df_unscaled_val_{prefix}.pkl', dir_output)
    test_dataset_unscale = read_object(f'df_unscaled_test_{prefix}.pkl', dir_output)

    features_selected_str = read_object('features_importance.pkl', dir_output / 'features_importance' / f'{values_per_class}_{k_days}_{scale}_{days_in_futur}_{graphScale.base}_{graphScale.graph_method}')
    features_selected_str = np.asarray(features_selected_str)
    features_selected_str = list(features_selected_str[:,0])

    varying_time_variables_2 = get_time_columns(varying_time_variables, k_days, train_dataset.copy(), train_features)
    features_name, newshape = get_features_name_list(graphScale.scale, train_features, METHODS_SPATIAL_TRAIN)
    features_selected_str = get_features_selected_for_time_series(features_selected_str, features_name, varying_time_variables_2)

    features_selected_str = list(features_selected_str)
    features_selected = np.arange(0, len(features_selected_str))
    logger.info((features_selected_str, len(features_selected_str)))

    features_selected_str.append('Past_risk')

prefix = f'full_{k_days}_{nbfeatures}_{scale}_{days_in_futur}_{graph_construct}_{graph_method}'
prefix_config = deepcopy(prefix)
name = 'check_'+scaling + '/' + prefix + '/' + 'baseline'

###################### Define models to train ######################

if name_exp.find('voting') != -1:
    #voting_models = define_voting_dl_models(training_mode, dataset_name, scale, graph_construct, post_process_model_dico)
    voting_models = []
    models = [#('DilatedCNN', 'full_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5),
            #('LSTM', 'full_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5),
            #('NetMLP', 'search_smote_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5),
              ]
    
    federated_models = [('NetMLP', False, 'departement', 'full_full_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5),]

    gnn_models = []
    staking_models = []
    gnn_models = [
                #('STGCN', False, 'full_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5),
                #('STGAT', False, 'full_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5),
                #('STGATLSTM', False, 'full_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5),
                #('DSTGCN', False, 'full_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5),
                #('DSTGAT', False, 'full_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5),
                #('NetGCN', False, 'full_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5),
                ]
else:
    models = [
            #('LSTM', 'search_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5),
            #('LSTM', 'search_smote_one_nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized_classification_weightedcrossentropy', 5),

            #('DilatedCNN', 'search_smote_one_nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized_classification_weightedcrossentropy', 5),
            #('NetMLP', 'full_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5),
            ]

    gnn_models = [
            #('net', False, 'full_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5),
            #('ST-GCN', False, 'full_proportion-on-zero-class_nbsinister-kmeans-5-Class-Dept-both_classification_weightedcrossentropy', 5),
            #('ST-GCN', False, 'full_proportion-on-zero-class_nbsinister-max-0-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5),
    ]

    voting_models = []

test_dataset_unscale['weight_nbsinister'] = 1
test_dataset['weight'] = 1

train_loader = None
val_loader = None
last_bool = None

params = {
    "graphScale": graphScale,
    "train_dataset": train_dataset,
    "val_dataset": val_dataset,
    "test_dataset": test_dataset,
    "features_selected": features_selected,
    "features_selected_str": features_selected_str,
    "k_days": k_days,
    "device": device,
    "optimize_feature": optimize_feature,
    "PATIENCE_CNT": PATIENCE_CNT,
    "CHECKPOINT": CHECKPOINT,
    "epochs": epochs,
    "lr": lr,
    "scaling": scaling,
    "encoding": encoding,
    "prefix": prefix,
    "Rewrite": Rewrite,
    "dir_output": dir_output,
    "train_dataset_unscale": train_dataset_unscale,
    "graph": graphScale,
    "name_dir": name_dir,
    'k_days' : k_days,
    'nbfeatures':nbfeatures,
    'graph_method' : graph_method
}

if doTrain:
    for gnn_model in gnn_models:
        params['model'] = gnn_model[0]
        params['use_temporal_as_edges'] = gnn_model[1]
        params['infos'] = gnn_model[2]
        params['out_channels'] = gnn_model[3]
        params['torch_structure'] = 'Model_gnn'

        wrapped_train_deep_learning_1D(params)

    if graph_method != 'graph':
        for model in models:
            params['model'] = model[0]
            params['infos'] = model[1]
            params['out_channels'] = model[2]
            params['use_temporal_as_edges'] = None
            params['torch_structure'] = 'Model_Torch'
            
            wrapped_train_deep_learning_1D(params)

    for models in voting_models:
        params['model'] = models[0]
        params['out_channels'] = models[2]
        params['use_temporal_as_edges'] = None
        params['torch_structure'] = 'Model_Torch'
        
        wrapped_train_sklearn_api_and_pytorch_voting_model(train_dataset=train_dataset.copy(deep=True),
                            val_dataset=val_dataset.copy(deep=True),
                            test_dataset=test_dataset.copy(deep=True),
                            graph_method=graph_method,
                            dir_output=dir_output / name,
                            autoRegression=autoRegression,
                            training_mode=training_mode,
                            do_grid_search=do_grid_search,
                            do_bayes_search=do_bayes_search,
                            model=models,
                            scale=scale,
                            input_params=params)

    for models in federated_models:
        params['model'] = models[0]
        params['use_temporal_as_edges'] = models[1]
        params['federated_cluster'] = models[2]
        params['infos'] = models[3]
        params['aggregation_method'] = 'mean'
        params['out_channels'] = models[-1]
        params['torch_structure'] = 'Model_Torch'

        wrapped_train_deep_learning_1D_federated(params)

if doTest:

    host = 'pc'

    logger.info('############################# TEST ###############################')

    dn = dataset_name
    if two:
        dn += '2'

    name_dir = dn + '/' + sinister + '/' + resolution + '/train' + '/'
    dir_train = Path(name_dir)
    
    name_dir = dn + '/' + sinister + '/' + resolution + '/test' + '/' + name_exp
    dir_output = Path(name_dir)

    if graph_method == 'node':

        models = [
                #('LSTM_search_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy'),
                # ('LSTM_search_smote_one_nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized_classification_weightedcrossentropy'),
                #('filter-soft-weight-all_full_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy'),
                #('DilatedCNN_full_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy'),
                #('LSTM_full_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy'),
                ('NetMLP_full_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy'),
                ('federated-NetMLP-departement_full_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy'),
                #('NetGCN_full_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy'),
                #('DSTGCN_full_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy'),
                #('DSTGAT_full_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy'),
                #('STGCN_full_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy'),
                #('STGAT_full_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy'),
                #('STGATLSTM_full_smote_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy'),
        ]
    elif graph_method == 'graph':

        models = [
        ]

    prefix_kmeans = f'{values_per_class}_{k_days}_{scale}_{graph_construct}_{top_cluster}'

    if days_in_futur > 0:
        prefix_kmeans += f'_{days_in_futur}_{futur_met}'

    aggregated_prediction = []
    aggregated_prediction_dept = []

    ####################################################### Test on all Dataset ####################################################
    metrics, metrics_dept, res, res_dept = test_dl_model(args=vars(args), graphScale=graphScale, test_dataset_dept=test_dataset, train_dataset=train_dataset_unscale,
                            test_dataset_unscale_dept=test_dataset_unscale,
                            test_name='all',
                            features_name=features_selected_str,
                            prefix_train=prefix,
                            prefix_config=prefix_config,
                            models=models,
                            dir_output=dir_output / 'all' / prefix,
                            device=device,
                            k_days=k_days,
                            encoding=encoding,
                            scaling=scaling,
                            test_departement=['all'],
                            dir_train=dir_train,
                            features=features,
                            dir_break_point=dir_train / 'check_none' / prefix_kmeans / 'kmeans',
                            name_exp=name_exp,
                            doKMEANS=doKMEANS,
                            suffix='temp')

    df_metrics = pd.DataFrame.from_dict(metrics, orient='index').reset_index()

    ####################################################### Test by departmenent ###################################################

    for dept in departements:
        if MLFLOW:
            dn = dataset_name
            if two:
                dn += '2'
            exp_name = f"{name_exp}_{name_exp}_{dn}_{dept}_{sinister}_{sinister_encoding}_test"
            experiments = client.search_smote_experiments()

            if exp_name not in list(map(lambda x: x.name, experiments)):
                tags = {
                    "",
                }

                client.create_experiment(name=exp_name, tags=tags)
        test_dataset_dept = test_dataset[(test_dataset['departement'] == name2int[dept])].reset_index(drop=True)
        if test_dataset_unscale is not None:
            test_dataset_unscale_dept = test_dataset_unscale[(test_dataset_unscale['departement'] == name2int[dept])].reset_index(drop=True)
            #test_dataset_unscale_dept = train_dataset_unscale[(train_dataset_unscale['departement'] == name2int[dept]) & (train_dataset['weight_one'] > 0) ].reset_index(drop=True)
        else:
            test_dataset_unscale_dept = None

        if test_dataset_dept.shape[0] < 5:
            continue

        logger.info(f'{dept} test : {test_dataset_dept.shape}')

        if host == 'pc':
            metrics, metrics_dept, res, res_dept = test_dl_model(args=vars(args), graphScale=graphScale, test_dataset_dept=test_dataset_dept, train_dataset=train_dataset_unscale,
                            test_dataset_unscale_dept=test_dataset_unscale,
                            test_name=dept,
                            features_name=features_selected_str,
                            prefix_train=prefix,
                            prefix_config=prefix_config,
                            models=models,
                            dir_output=dir_output / dept / prefix,
                            device=device,
                            k_days=k_days,
                            encoding=encoding,
                            scaling=scaling,
                            test_departement=[dept],
                            dir_train=dir_train,
                            features=features,
                            dir_break_point=dir_train / 'check_none' / prefix_kmeans / 'kmeans',
                            name_exp=name_exp,
                            doKMEANS=doKMEANS,
                            suffix='temp')
            
            aggregated_prediction.append(res)
        else:
            metrics = read_object('metrics'+'_'+prefix+'_'+scaling+'_'+encoding+'_'+dept+'_temp.pkl', dir_output / dept / prefix)
            #metrics_gnn = read_object('metrics'+'_'+prefix+'_'+scaling+'_'+encoding+'_'+dept+'_temp_gnn.pkl', dir_output / dept / prefix)
            metrics.update(metrics_gnn)
            assert metrics is not None
            for name, _, target_name, _ in models:
                res = read_object(name+'_'+prefix+'_'+scaling+'_'+encoding+'_'+dept+'_pred.pkl', dir_output / dept / prefix / name)
                aggregated_prediction.append(res)
                run = f'{dept}_{name}_{prefix}'
                if MLFLOW:
                    existing_run = get_existing_run(f'{run}')
                    if existing_run:
                        mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
                    else:
                        mlflow.start_run(run_name=f'{run}', nested=True)
                    if name in metrics.keys():
                        log_metrics_recursively(metrics[run], prefix='')
                    else:
                        logger.info(f'{name} not found')
                    mlflow.end_run()

            for name, _, target_name, _ in gnn_models:
                res = read_object(name+'_'+prefix+'_'+scaling+'_'+encoding+'_'+dept+'_pred.pkl', dir_output / dept / prefix / name)
                aggregated_prediction.append(res)
                run = f'{dept}_{name}_{prefix}'
                if MLFLOW:
                    existing_run = get_existing_run(f'{run}')
                    if existing_run:
                        mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
                    else:
                        mlflow.start_run(run_name=f'{run}', nested=True)
                    if name in metrics.keys():
                        log_metrics_recursively(metrics[run], prefix='')
                    else:
                        logger.info(f'{name} not found')
                    mlflow.end_run()

        if 'df_metrics' not in locals():
            df_metrics = pd.DataFrame.from_dict(metrics, orient='index').reset_index()
        else:
            df_metrics = pd.concat((df_metrics, pd.DataFrame.from_dict(metrics, orient='index').reset_index()))
        #aggregated_prediction_dept.append(res_dept)

    df_metrics.rename({'index': 'Run'}, inplace=True, axis=1)
    df_metrics.reset_index(drop=True, inplace=True)
    
    check_and_create_path(dir_output / prefix)
    df_metrics.to_csv(dir_output / prefix / f'df_metrics_{graph_method}.csv')
 
    #wrapped_compare(df_metrics,  dir_output / prefix, 'Model')
    #wrapped_compare(df_metrics,  dir_output / prefix, 'Loss_function')
    #wrapped_compare(df_metrics,  dir_output / prefix, 'Target')
    #wrapped_compare(df_metrics,  dir_output / prefix, 'Days_in_futur')