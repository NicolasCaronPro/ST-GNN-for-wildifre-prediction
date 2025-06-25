import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory
parent_dir = os.path.dirname(current_dir)

# Insert the parent directory into sys.path
sys.path.insert(0, parent_dir)
from GNN.construct import *
from GNN.config_parser import ConfigParser
import geopandas as gpd
import argparse

########################### Input Arg ######################################

parser = argparse.ArgumentParser(
    prog="Train",
    description="Create graph and database according to config file",
)
parser.add_argument(
    "-c", "--config", type=str, default="config/config.json", help="Path to config file"
)

args = parser.parse_args()

QUICK = True

config = ConfigParser(args.config)

# Input config
dataset_name = config.dataset
name_exp = config.name
maxDate = config.maxDate
trainDate = config.trainDate
doEncoder = bool(config.get("encoder", False))
doPoint = bool(config.get("point", False))
doGraph = bool(config.get("graph", False))
doDatabase = bool(config.get("database", False))
do2D = bool(config.get("database2D", False))
doFet = bool(config.featuresSelection)
optimize_feature = bool(config.optimizeFeature)
doTest = bool(config.doTest)
doTrain = bool(config.doTrain)
nbfeatures = config.NbFeatures
sinister = config.sinister
values_per_class = config.nbpoint
scale = int(config.scale) if config.scale != "departement" else config.scale
resolution = config.resolution
doKMEANS = bool(config.KMEANS)
do_grid_search = bool(config.GridSearch)
do_bayes_search = bool(config.BayesSearch)
days_in_futur = int(config.days_in_futur)
scaling = config.scaling
graph_construct = config.graphConstruct
sinister_encoding = config.sinisterEncoding
top_cluster = config.top_cluster
graph_method = config.graph_method
training_mode = config.training_mode
######################## Get features and train features list ######################


isInference = name_exp == 'inference'

features = config.features
train_features = config.features
kmeasn_features = config.kmeans_features

######################## Get departments and train departments #######################

departements = config.train_departments + config.test_departments
train_departements = config.train_departments

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

    df, graphScale, prefix, fp, features_selected = init(config, dir_output, 'train_gnn')
    save_object(df.columns, 'features_name.pkl', dir_output)

    if MLFLOW:
        exp_name = f"{dataset_name}_train"
        experiments = client.search_experiments()

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
        config,
        config)
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

    name = 'check_'+scaling + '/' + prefix + '/' + 'baseline'

    ###################### Defined ClassRisk model ######################

    dir_post_process = dir_output / 'post_process'

    post_process_model_dico, train_dataset, val_dataset, test_dataset, new_cols = post_process_model(train_dataset, val_dataset, test_dataset, dir_post_process, graphScale)

    features_selected_str.append('Past_risk')
    features_selected_str.append('Past_burnedarea')
    features_selected = np.arange(0, len(features_selected_str))

    train_dataset = add_past_risk(train_dataset, 'nbsinisterDaily-kmeans-5-Class-Dept-cubic-Specialized-Past', 'risk')
    test_dataset = add_past_risk(test_dataset, 'nbsinisterDaily-kmeans-5-Class-Dept-cubic-Specialized-Past', 'risk')
    val_dataset = add_past_risk(val_dataset, 'nbsinisterDaily-kmeans-5-Class-Dept-cubic-Specialized-Past', 'risk')

    train_dataset = add_past_risk(train_dataset, 'burnedareaDaily-kmeans-5-Class-Dept-cubic-Specialized-Past', 'burnedarea')
    test_dataset = add_past_risk(test_dataset, 'burnedareaDaily-kmeans-5-Class-Dept-cubic-Specialized-Past', 'burnedarea')
    val_dataset = add_past_risk(val_dataset, 'burnedareaDaily-kmeans-5-Class-Dept-cubic-Specialized-Past', 'burnedarea')

    save_object(train_dataset, 'df_train_'+prefix+'.pkl', dir_output)
    save_object(val_dataset, 'df_val_'+prefix+'.pkl', dir_output)
    save_object(test_dataset, 'df_test_'+prefix+'.pkl', dir_output)

else:
    post_process_model_dico = None

    prefix = f'full_{scale}_{days_in_futur}_{graph_construct}_{graph_method}'
    
    graphScale = read_object(f'graph_{scale}_{graph_construct}_{graph_method}.pkl', dir_output)

    dir_output = dir_output / name_exp

    train_dataset = read_object(f'df_train_{prefix}.pkl', dir_output)
    val_dataset = read_object(f'df_val_{prefix}.pkl', dir_output)
    test_dataset = read_object(f'df_test_{prefix}.pkl', dir_output)

    train_dataset_unscale = read_object(f'df_unscaled_train_{prefix}.pkl', dir_output)
    val_dataset_unscale = read_object(f'df_unscaled_val_{prefix}.pkl', dir_output)
    test_dataset_unscale = read_object(f'df_unscaled_test_{prefix}.pkl', dir_output)

    features_selected_str = read_object('features_importance.pkl', dir_output / 'features_importance' / f'{values_per_class}_0_{scale}_{days_in_futur}_{graphScale.base}_{graphScale.graph_method}')
    features_selected_str = np.asarray(features_selected_str)
    features_selected_str = list(features_selected_str[:,0])

    #varying_time_variables_2 = get_time_columns(varying_time_variables, k_days, train_dataset.copy(), train_features)
    features_name, newshape = get_features_name_list(graphScale.scale, train_features, METHODS_SPATIAL_TRAIN)
    #features_selected_str = get_features_selected_for_time_series(features_selected_str, features_name, varying_time_variables_2)

    features_selected = np.arange(0, len(features_selected_str))
    logger.info((features_selected_str, len(features_selected_str)))

    if 'nbsinisterDaily-kmeans-5-Class-Dept-cubic-Specialized-Past' not in np.unique(train_dataset.columns) or 'burnedareaDaily-kmeans-5-Class-Dept-cubic-Specialized-Past' not in np.unique(train_dataset.columns):
        dir_post_process = dir_output / 'post_process'
        post_process_model_dico, train_dataset, val_dataset, test_dataset, new_cols = post_process_model(train_dataset, val_dataset, test_dataset, dir_post_process, graphScale)

    train_dataset = add_past_risk(train_dataset, 'nbsinisterDaily-kmeans-5-Class-Dept-cubic-Specialized-Past', 'risk')
    test_dataset = add_past_risk(test_dataset, 'nbsinisterDaily-kmeans-5-Class-Dept-cubic-Specialized-Past', 'risk')
    val_dataset = add_past_risk(val_dataset, 'nbsinisterDaily-kmeans-5-Class-Dept-cubic-Specialized-Past', 'risk')

    train_dataset = add_past_risk(train_dataset, 'burnedareaDaily-kmeans-5-Class-Dept-cubic-Specialized-Past', 'burnedarea')
    test_dataset = add_past_risk(test_dataset, 'burnedareaDaily-kmeans-5-Class-Dept-cubic-Specialized-Past', 'burnedarea')
    val_dataset = add_past_risk(val_dataset, 'burnedareaDaily-kmeans-5-Class-Dept-cubic-Specialized-Past', 'burnedarea')

    save_object(train_dataset, 'df_train_'+prefix+'.pkl', dir_output)
    save_object(val_dataset, 'df_val_'+prefix+'.pkl', dir_output)
    save_object(test_dataset, 'df_test_'+prefix+'.pkl', dir_output)

    features_selected_str.append('Past_risk')
    features_selected_str.append('Past_burnedarea')
    features_selected = np.arange(0, len(features_selected_str))

######################### Tourisme ###########################
"""tourisme_data = pd.read_csv(rootDisk / 'csv' / 'tourisme.csv')

train_dataset = train_dataset.set_index('departement').join(tourisme_data.set_index('Code')['Tourisme'], on='Code').reset_index()
val_dataset = val_dataset.set_index('departement').join(tourisme_data.set_index('Code')['Tourisme'], on='Code').reset_index()
test_dataset = test_dataset.set_index('departement').join(tourisme_data.set_index('Code')['Tourisme'], on='Code').reset_index()

features_selected_str.append('Tourisme')"""

##############################################################
    
prefix_config = deepcopy(prefix)
name = 'check_'+scaling + '/' + prefix + '/' + 'baseline'

train_dataset['saison'] = train_dataset['date'].apply(get_saison)
val_dataset['saison'] = val_dataset['date'].apply(get_saison)
test_dataset['saison'] = test_dataset['date'].apply(get_saison)

train_dataset['isBassin'] = train_dataset['departement'].apply(is_mediterranean_dept)
val_dataset['isBassin'] = val_dataset['departement'].apply(is_mediterranean_dept)
test_dataset['isBassin'] = test_dataset['departement'].apply(is_mediterranean_dept)

#features_selected_str.append('isBassin')

train_dataset['cluster-encoder'] = train_dataset['cluster_encoder']
val_dataset['cluster-encoder'] = val_dataset['cluster_encoder']
test_dataset['cluster-encoder'] = test_dataset['cluster_encoder']

if scale == 'departement':
    train_dataset['scale'] = 10
    val_dataset['scale'] = 10
    test_dataset['scale'] = 10
else:
    train_dataset['scale'] = scale
    val_dataset['scale'] = scale
    test_dataset['scale'] = scale
    
save_object(train_dataset, 'df_train_'+prefix+'.pkl', dir_output)
save_object(val_dataset, 'df_val_'+prefix+'.pkl', dir_output)
save_object(test_dataset, 'df_test_'+prefix+'.pkl', dir_output)

print(allDates[int(test_dataset.date.min())])
print(allDates[int(test_dataset[test_dataset['weight'] > 0].date.min())])

prefix = f'full_all_{scale}_{days_in_futur}_{graph_construct}_{graph_method}'

###################### Define models to train ######################

if name_exp.find('voting') != -1:
    if dataset_name != 'bdiff':
        voting_models = define_voting_dl_models(training_mode, dataset_name, scale, graph_construct, post_process_model_dico)
    else:
        voting_models = []
    
    voting_models = []

    models = [
            ]
    
    staking_models = []
    federated_models = [
        #('NetMLP', False, 'cluster-encoder', 'search_full_10_all_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5),
        #('NetMLP', False, 'saison', 'search_full_10_all_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5),
        #('NetMLP', False, 'departement', 'search_full_10_all_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5),
        ]
    
    gnn_models = [
                #('SepGRUGNN', False, None,'search_full_5_all_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5, 1),
                #('SepGRUGNN', False, None,'search_full_10_all_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5, 1),
                #('SepGRUGNN', False, None,'search_full_15_all_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5, 1),
                #('SepLSTMGNN', False, None,'search_full_20_all_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5),
                
                #('SepGRUGNN', False, None,'search_full_5_all_one_burnedarea-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5, 1),
                #('SepGRUGNN', False, None,'search_full_10_all_one_burnedarea-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5, 1),
                #('SepGRUGNN', False, None,'search_full_15_all_one_burnedarea-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5, 1),
                #('SepLSTMGNN', False, None,'search_full_20_all_one_burnedarea-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5),
                ]
    
    if days_in_futur == 0:
        gnn_models = [
            ('SepGRUGNN', False, None, 'search_full_10_all_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5, 1),
            ('SepGRUGNN', False, None, 'search_full_15_all_one_burnedarea-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5, 5),
                ]
        
    elif days_in_futur == 7:
        gnn_models = [
            ('SepGRUGNN', False, None, 'search_full_10_all_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5, 1),
            ('SepGRUGNN', False, None, 'search_full_10_all_one_burnedarea-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5, 5),
        ]
    elif days_in_futur == 15:
        gnn_models = [
            ('SepGRUGNN', False, None, 'search_full_5_all_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5, 1),
            ('SepGRUGNN', False, None, 'search_full_5_all_one_burnedarea-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5, 5),
        ]
    elif days_in_futur == 31:
        gnn_models = [
            ('SepGRUGNN', False, None, 'search_full_10_all_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5, 1),
            ('SepGRUGNN', False, None, 'search_full_5_all_one_burnedarea-kmeans-5-Class-Dept_classification_weightedcrossentropy', 5, 5),
        ]
    
else:
    models = [
            ]
    
    gnn_models = [
    ]

    voting_models = []

train_loader = None
val_loader = None
last_bool = None

global_params = {
    "graphScale": graphScale,
    "train_dataset": train_dataset,
    "val_dataset": val_dataset,
    "test_dataset": test_dataset,
    "features_selected": features_selected,
    "features_selected_str": features_selected_str,
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
    'graph_method' : graph_method
}

import multiprocessing

def train_gnn(gnn_model, params):
    params.update({
        'model': gnn_model[0],
        'use_temporal_as_edges': gnn_model[1],
        'mesh_file': gnn_model[2],
        'infos': gnn_model[3],
        'out_channels': gnn_model[4],
        'torch_structure': 'Model_gnn'
    })
    wrapped_train_deep_learning_1D(params)

def train_classic(model, params):
    params.update({
        'model': model[0],
        'infos': model[1],
        'out_channels': model[2],
        'use_temporal_as_edges': None,
        'torch_structure': 'Model_Torch'
    })
    wrapped_train_deep_learning_1D(params)

def train_voting(models, params):
    params.update({
        'model': models[0],
        'out_channels': models[2],
        'use_temporal_as_edges': None,
        'torch_structure': 'Model_Torch'
    })

    wrapped_train_sklearn_api_and_pytorch_voting_model(
        train_dataset=train_dataset.copy(deep=True),
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
        input_params=params
    )

def train_federated(models, params):
    params.update({
        'model': models[0],
        'use_temporal_as_edges': models[1],
        'federated_cluster': models[2],
        'infos': models[3],
        'aggregation_method': 'median',
        'out_channels': models[-1],
        'torch_structure': 'Model_Torch'
    })

    wrapped_train_deep_learning_1D_federated(params)

import traceback

# ---------------- MAIN LOGIQUE ---------------- #
"""if doTrain:
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        tasks = []

        if len(gnn_models) > 0:
            tasks.append(("GNN", pool.starmap_async(train_gnn, [(m, global_params) for m in gnn_models])))

        if graph_method != 'graph' and len(models) > 0:
            tasks.append(("Classic", pool.starmap_async(train_classic, [(m, global_params) for m in models])))

        if len(voting_models) > 0:
            tasks.append(("Voting", pool.starmap_async(train_voting, [(m, global_params) for m in voting_models])))

        if len(federated_models) > 0:
            tasks.append(("Federated", pool.starmap_async(train_federated, [(m, global_params) for m in federated_models])))

        # Attendre les résultats et gérer les erreurs
        for name, task in tasks:
            try:
                task.get()  # récupère les résultats et lève l'exception si une tâche a planté
                print(f"✅ {name} models trained successfully.")
            except Exception as e:
                print(f"❌ Error while training {name} models:")
                traceback.print_exc()"""
if doTrain:
    for gnn_model in gnn_models:
        global_params['model'] = gnn_model[0]
        global_params['use_temporal_as_edges'] = gnn_model[1]
        global_params['mesh_file'] = gnn_model[2]
        global_params['infos'] = gnn_model[3]
        global_params['out_channels'] = gnn_model[4]
        global_params['n_run'] = gnn_model[-1]
        global_params['torch_structure'] = 'Model_gnn'

        wrapped_train_deep_learning_1D(global_params)

    if graph_method != 'graph':
        for model in models:
            global_params['model'] = model[0]
            global_params['infos'] = model[1]
            global_params['out_channels'] = model[2]
            global_params['use_temporal_as_edges'] = None
            global_params['torch_structure'] = 'Model_Torch'
            global_params['n_run'] = models[-1]
            
            wrapped_train_deep_learning_1D(global_params)

    for models in voting_models:
        global_params['model'] = models[0]
        global_params['out_channels'] = models[2]
        global_params['use_temporal_as_edges'] = None
        global_params['torch_structure'] = 'Model_Torch'
        global_params['n_run'] = models[-1]
        
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
                            input_global_params=global_params)

    for models in federated_models:
        global_params['model'] = models[0]
        global_params['use_temporal_as_edges'] = models[1]
        global_params['federated_cluster'] = models[2]
        global_params['infos'] = models[3]
        global_params['aggregation_method'] = 'median'
        global_params['out_channels'] = models[-1]
        global_params['torch_structure'] = 'Model_Torch'

        wrapped_train_deep_learning_1D_federated(global_params)

if doTest:

    #test_dataset = test_dataset[test_dataset['weight'] > 0]
    #test_dataset_unscale = test_dataset_unscale[test_dataset_unscale['weight'] > 0]
    #if days_in_futur > 0:
    #    test_dataset = set_weight_every_n_days(test_dataset, days_in_futur)
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
            ('SepGRUGNN_search_full_5_all_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy'),
            ('SepGRUGNN_search_full_10_all_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy'),
            ('SepGRUGNN_search_full_15_all_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy'),
            ('SepGRUGNN_search_full_20_all_one_nbsinister-kmeans-5-Class-Dept_classification_weightedcrossentropy'),

            #('SepGRUGNN_search_full_5_all_one_burnedarea-kmeans-5-Class-Dept_classification_weightedcrossentropy'),
            #('SepGRUGNN_search_full_10_all_one_burnedarea-kmeans-5-Class-Dept_classification_weightedcrossentropy'),
            #('SepGRUGNN_search_full_15_all_one_burnedarea-kmeans-5-Class-Dept_classification_weightedcrossentropy'),
            #('SepGRUGNN_search_full_20_all_one_burnedarea-kmeans-5-Class-Dept_classification_weightedcrossentropy'),

        ]
        
    elif graph_method == 'graph':

        models = [
        ]
        
    prefix_kmeans = f'{values_per_class}_0_{scale}_{graph_construct}_{top_cluster}'

    if days_in_futur > 0:
        prefix_kmeans += f'_{days_in_futur}_{futur_met}'
        
    aggregated_prediction = []
    aggregated_prediction_dept = []

    ####################################################### Test on all Dataset ####################################################
    metrics, metrics_dept, res, res_dept = test_dl_model(args=vars(config), graphScale=graphScale, test_dataset_dept=test_dataset, train_dataset=train_dataset_unscale,
                            test_dataset_unscale_dept=test_dataset_unscale,
                            test_name='all',
                            features_name=features_selected_str,
                            prefix_train=prefix,
                            prefix_config=prefix_config,
                            models=models,
                            dir_output=dir_output / 'all' / prefix,
                            device=device,
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
    for re in res:
        aggregated_prediction.append(re)

    ####################################################### Test by departmenent ###################################################

    for dept in departements:
        if MLFLOW:
            dn = dataset_name
            if two:
                dn += '2'
            exp_name = f"{name_exp}_{name_exp}_{dn}_{dept}_{sinister}_{sinister_encoding}_test"
            experiments = client.search_experiments()

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
            metrics, metrics_dept, res, res_dept = test_dl_model(args=vars(config), graphScale=graphScale, test_dataset_dept=test_dataset_dept, train_dataset=train_dataset_unscale,
                            test_dataset_unscale_dept=test_dataset_unscale,
                            test_name=dept,
                            features_name=features_selected_str,
                            prefix_train=prefix,
                            prefix_config=prefix_config,
                            models=models,
                            dir_output=dir_output / dept / prefix,
                            device=device,
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
    df_metrics.to_csv(dir_output / prefix / f'df_metrics_STGNN_{graph_method}.csv')
 
    #wrapped_compare(df_metrics,  dir_output / prefix, 'Model')
    #wrapped_compare(df_metrics,  dir_output / prefix, 'Loss_function')
    #wrapped_compare(df_metrics,  dir_output / prefix, 'Target')
    #wrapped_compare(df_metrics,  dir_output / prefix, 'Days_in_futur')
    exit(1)
    if dataset_name != 'bdiff':
        exit(1)

    aggregated_prediction = pd.concat(aggregated_prediction).reset_index(drop=True)
    #aggregated_prediction_dept = pd.concat(aggregated_prediction_dept).reset_index(drop=True)

    aggregated_prediction.to_csv(dir_output / 'aggregated_prediction.csv', index=False)
    fp = pd.read_csv(f'sinister/{dataset_name}/{sinister}.csv', dtype=str)

    for name in models:
        model_name, under_sampling, over_sampling, nbfeatures, weight_type, target_name, task_type, loss = name.split('_')
        band = 'prediction'
        if target_name == 'binary':
            vmax_band = 1
        else:
            vmax_band = np.nanmax(aggregated_prediction[target_name].values)

        aggregated_prediction_model = aggregated_prediction[aggregated_prediction['model'] == name]
        #aggregated_prediction_dept_model = aggregated_prediction_dept[aggregated_prediction_dept['model'] == name]

        #dates = aggregated_prediction[aggregated_prediction['nbsinister'] == aggregated_prediction['nbsinister'].max()]['date'].values
        dates = [
            allDates.index('2023-04-26'),
            allDates.index('2023-05-25'),
            allDates.index('2023-03-06'),
            allDates.index('2023-04-17'),
            allDates.index('2023-06-15'),
            allDates.index('2023-09-03'),
            allDates.index('2023-09-17'),
            allDates.index('2023-09-13'),
            allDates.index('2023-08-15'),
            allDates.index('2023-05-08')
        ]

        region_france = gpd.read_file(root / 'csv/france/data/geo/hexagones_france.gpkg')
        region_france['latitude'] = region_france['geometry'].apply(lambda x : float(x.centroid.y))
        region_france['longitude'] = region_france['geometry'].apply(lambda x : float(x.centroid.x))

        ####################### Prediction #####################
        #check_and_create_path(dir_output / name)
        susectibility_map_france_daily_geojson(aggregated_prediction_model, region_france, graphScale, np.unique(dates).astype(int), f'prediction_{target_name}', f'{scale}_france',
                                    dir_output / name, vmax_band, dept_reg=False, sinister=sinister, sinister_point=fp)
        
        train_dataset_dept = train_dataset.groupby(['departement', 'date'])[target_name].sum().reset_index()
        vmax_band = np.nanmax(train_dataset_dept[target_name].values)
        
        #susectibility_map_france_daily_geojson(aggregated_prediction_dept, region_france.copy(deep=True), graphScale, np.unique(dates).astype(int), 'prediction', f'departemnnt_france',
        #                            dir_output / name, vmax_band, dept_reg=True, sinister=sinister, sinister_point=fp)"""
    
    """######################## Ground Truth ####################
    name = 'GT'
    check_and_create_path(dir_output / name)
    susectibility_map_france_daily_geojson(aggregated_prediction, region_france.copy(deep=True), graphScale, np.unique(dates).astype(int), target_name,
                                       f'{scale}_gt', dir_output / name, vmax_band=vmax_band,
                                       dept_reg=False, sinister=sinister, sinister_point=fp)

    susectibility_map_france_daily_geojson(aggregated_prediction_dept,
                                       region_france.copy(deep=True), graphScale, np.unique(dates).astype(int),
                                       target_name, 'departement_gt',
                                       dir_output / name, vmax_band=vmax_band,
                                       dept_reg=True, sinister=sinister, sinister_point=fp)"""