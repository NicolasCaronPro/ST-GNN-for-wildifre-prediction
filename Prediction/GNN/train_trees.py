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
parser.add_argument('-quick', '--quick', type=str, help='isquick', default='False')

args = parser.parse_args()

QUICK = args.quick == 'True'

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

assert values_per_class == 'full'
assert graph_method == 'node'

######################## Get features and train features list ######################

isInference = name_exp == 'inference'

features, train_features, kmeans_features = get_features_for_sinister_prediction(dataset_name, sinister, isInference)

######################## Get departments and train departments #######################

departements, train_departements = select_departments(dataset_name, sinister)

############################# GLOBAL VARIABLES #############################

name_dir = dataset_name + '/' + sinister + '/' + resolution + '/' + 'train' +  '/'
dir_output = Path(name_dir)
check_and_create_path(dir_output)

if dataset_name == 'firemen2':
    two = True
    dataset_name = 'firemen'
else:
    two = False

name_exp = f'{sinister_encoding}_{name_exp}'

dir_target = root_target / sinister / dataset_name / sinister_encoding / 'log' / resolution

minDate = '2017-06-12' # Starting point

if dataset_name == '':
    dataset_name = 'firemen'

autoRegression = 'AutoRegressionReg' in train_features
if autoRegression:
    name_exp += '_AutoRegressionReg'

if not QUICK:
    #geo = gpd.read_file(f'regions/{sinister}/{dataset_name}/regions.geojson')
    #geo = geo[geo['departement'].isin(departements)].reset_index(drop=True)
    
    ####################### INIT ################################

    df, graphScale, prefix, fp, features_selected = init(config, dir_output, 'train_trees')

    if MLFLOW:
        exp_name = f"{dataset_name}_train"
        experiments = client.search()

        if exp_name not in list(map(lambda x: x.name, experiments)):
            tags = {
                "",
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
    if MLFLOW:
        train_dataset_ml_flow = mlflow.data.from_pandas(train_dataset)
        val_dataset_ml_flow = mlflow.data.from_pandas(val_dataset)
        test_dataset_ml_flow = mlflow.data.from_pandas(test_dataset)

        mlflow.log_param('train_features', train_features)
        mlflow.log_param('features', features)
        mlflow.log_param('features_selected', features_selected)
        mlflow.log_input(train_dataset_ml_flow, context='trainig')
        mlflow.log_input(val_dataset_ml_flow, context='validation')
        mlflow.log_input(test_dataset_ml_flow, context='testing')
        mlflow.end_run()

    ######################## Training ##############################

    name = 'check_'+scaling + '/' + prefix + '/' + 'baseline'

    ###################### Defined ClassRisk model ######################

    features_selected.append('Past_risk')
    features_selected.append('Past_burnedarea')

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

else:
    post_process_model_dico = None

    prefix = f'full_{scale}_{days_in_futur}_{graph_construct}_{graph_method}'
    
    graphScale = read_object(f'graph_{scale}_{graph_construct}_{graph_method}.pkl', dir_output)

    dir_output = dir_output / name_exp
    
    train_dataset = read_object(f'df_train_{prefix}.pkl', dir_output)
    val_dataset = read_object(f'df_val_{prefix}.pkl', dir_output)
    test_dataset = read_object(f'df_test_{prefix}.pkl', dir_output)

    #train_dataset_unscale = read_object(f'df_unscaled_train_{prefix}.pkl', dir_output)
    #val_dataset_unscale = read_object(f'df_unscaled_val_{prefix}.pkl', dir_output)
    #test_dataset_unscale = read_object(f'df_unscaled_test_{prefix}.pkl', dir_output)

    #train_dataset_unscale = read_object(f'df_unscaled_train_{prefix}.pkl', dir_output)
    #val_dataset_unscale = read_object(f'df_unscaled_val_{prefix}.pkl', dir_output)
    #test_dataset_unscale = read_object(f'df_unscaled_test_{prefix}.pkl', dir_output)

    features_selected = read_object('features_importance.pkl', dir_output / 'features_importance' / f'{values_per_class}_0_{scale}_{days_in_futur}_{graphScale.base}_{graphScale.graph_method}')
    features_importance = np.asarray(features_selected)
    features_selected = list(features_importance[:,0])

    if 'nbsinisterDaily-kmeans-5-Class-Dept-cubic-Specialized-Past' not in np.unique(train_dataset.columns) or 'burnedareaDaily-kmeans-5-Class-Dept-cubic-Specialized-Past' not in np.unique(train_dataset.columns):
        dir_post_process = dir_output / 'post_process'
        post_process_model_dico, train_dataset, val_dataset, test_dataset, new_cols = post_process_model(train_dataset, val_dataset, test_dataset, dir_post_process, graphScale)

    train_dataset = add_past_risk(train_dataset, 'nbsinisterDaily-kmeans-5-Class-Dept-cubic-Specialized-Past', 'risk')
    test_dataset = add_past_risk(test_dataset, 'nbsinisterDaily-kmeans-5-Class-Dept-cubic-Specialized-Past', 'risk')
    val_dataset = add_past_risk(val_dataset, 'nbsinisterDaily-kmeans-5-Class-Dept-cubic-Specialized-Past', 'risk')

    train_dataset = add_past_risk(train_dataset, 'burnedareaDaily-kmeans-5-Class-Dept-cubic-Specialized-Past', 'burnedarea')
    test_dataset = add_past_risk(test_dataset, 'burnedareaDaily-kmeans-5-Class-Dept-cubic-Specialized-Past', 'burnedarea')
    val_dataset = add_past_risk(val_dataset, 'burnedareaDaily-kmeans-5-Class-Dept-cubic-Specialized-Past', 'burnedarea')

    train_dataset['nbsinister-binary'] = (train_dataset['nbsinister'] > 0).astype(int)
    test_dataset['nbsinister-binary'] = (test_dataset['nbsinister'] > 0).astype(int)
    val_dataset['nbsinister-binary'] = (val_dataset['nbsinister'] > 0).astype(int)

    save_object(train_dataset, 'df_train_'+prefix+'.pkl', dir_output)
    save_object(val_dataset, 'df_val_'+prefix+'.pkl', dir_output)
    save_object(test_dataset, 'df_test_'+prefix+'.pkl', dir_output)

######################### Tourisme ###########################
"""tourisme_data = pd.read_csv(rootDisk / 'csv' / 'tourisme.csv')

train_dataset = train_dataset.set_index('departement').join(tourisme_data.set_index('Code')['Tourisme'], on='Code').reset_index()
val_dataset = val_dataset.set_index('departement').join(tourisme_data.set_index('Code')['Tourisme'], on='Code').reset_index()
test_dataset = test_dataset.set_index('departement').join(tourisme_data.set_index('Code')['Tourisme'], on='Code').reset_index()

features_selected.append('Tourisme')"""

##############################################################

prefix_config = deepcopy(prefix)
name = 'check_'+scaling + '/' + prefix + '/' + 'baseline'

train_dataset['saison'] = train_dataset['date'].apply(get_saison)
val_dataset['saison'] = val_dataset['date'].apply(get_saison)
test_dataset['saison'] = test_dataset['date'].apply(get_saison)

train_dataset['isBassin'] = train_dataset['departement'].apply(is_mediterranean_dept)
val_dataset['isBassin'] = val_dataset['departement'].apply(is_mediterranean_dept)
test_dataset['isBassin'] = test_dataset['departement'].apply(is_mediterranean_dept)

train_dataset['cluster-encoder'] = train_dataset['cluster_encoder']
val_dataset['cluster-encoder'] = val_dataset['cluster_encoder']
test_dataset['cluster-encoder'] = test_dataset['cluster_encoder']

print(allDates[int(test_dataset.date.min())])

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

prefix = f'full_all_{scale}_{days_in_futur}_{graph_construct}_{graph_method}'
name = 'check_'+scaling + '/' + prefix + '/' + 'baseline'

###################### Define models to train ######################

if name_exp.find('default') != -1: 
    voting_models = []
    models = [
            ('lg_search_smote-4_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2', None, None, None, 1),
            ('xgboost_search_smote-4_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax', None, None, None, 1),
            ('catboost_search_smote-4_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax', None, None, None, 1),
        
            ('lg_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2', None, None, None, 1),
            ('xgboost_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax', None, None, None, 1),
            ('catboost_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax', None, None, None, 1),

            ('lg_search_smote-6_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2', None, None, None, 1),
            ('xgboost_search_smote-6_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax', None, None, None, 1),
            ('catboost_search_smote-6_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax', None, None, None, 1),
        
            ('xgboost_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax', None, None,None, 1),
            #('lg_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2', None, None, None, 1),
            #('xgboost_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax', None, None, None, 1),
            #('catboost_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax', None, None, None, 1),
        
            #('xgboost_search_full_0_all_one_burnedarea-kmeans-5-Class-Dept_classification_softmax', None, None, None, 5),
            #('catboost_search_full_0_all_one_burnedarea-kmeans-5-Class-Dept_classification_softmax', None, None, None, 5),
            #('lg_search_full_0_all_one_burnedarea-kmeans-5-Class-Dept_classification_l2', None, None, None, 5),

            #('lg_search_full_0_all_one_nbsinister-binary_classification_l2', None, None, None),
            #('xgboost_search_full_0_all_one_nbsinister-binary_classification_softmax', None, None, None),
            #('catboost_search_full_0_all_one_nbsinister-binary_classification_logloss', None, None, None, 5),

            #('ngboost_search_smote-2_0_all_one_burnedarea-kmeans-5-Class-Dept_classification_CRPScore', None, None, None),

            #('xgboost_search_smote-2_all_one_nbsinister-kmeans-5-Class-Dept-cubic-5_classification_softmax-dual', None, None, None),
            #('xgboost_search_smote-2_all_one_nbsinister-kmeans-5-Class-Dept-cubic-5_classification_softmax', None, None, None),
            #('catboost_search_smote-2_all_one_nbsinister-kmeans-5-Class-Dept-circular-5_classification_softmax', None, None, None),
    ]
    staking_models = []

elif name_exp.find('stacking') != -1:
    staking_models = define_staking_trees_model(training_mode, dataset_name, scale, graph_construct, post_process_model_dico)
    models = []
    voting_models = []

elif name_exp.find('normal') != -1 or name_exp.find('2022') != -1 or 'year' in name_exp:
    #voting_models = define_voting_trees_model(training_mode, dataset_name, scale, graph_construct, post_process_model_dico)    
    voting_models = []
    models = [
            #('lg_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2', None, None, None, 1),
            ('xgboost_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax', None, None,None, 1),
            ('catboost_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax', None, None, None, 1),
        ]
    staking_models = []

elif name_exp.find('voting') != -1:
    voting_models = define_voting_trees_model(training_mode, dataset_name, scale, graph_construct, post_process_model_dico)
    #voting_models = []
    staking_models = []
    models = [
            #('lg_search_smote-4_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2', None, None, None, 1),
            #('xgboost_search_smote-4_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax', None, None, None, 1),
            #('catboost_search_smote-4_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax', None, None, None, 1),
        ##
            #('lg_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2', None, None, None, 1),
            #('xgboost_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax', None, None, None, 1),
            #('catboost_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax', None, None, None, 1),
##
            #('lg_search_smote-6_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2', None, None, None, 1),
            #('xgboost_search_smote-6_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax', None, None, None, 1),
            #('catboost_search_smote-6_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax', None, None, None, 1),
        ##
            #('lg_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2', None, None, None, 1),
            ('xgboost_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_mcewk', None, None,None, 1),
            #('catboost_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_mcewk', None, None, None, 1),
        ]
else:
    #models = define_trees_model(training_mode, dataset_name, scale, graph_construct, post_process_model_dico)
    models = []
    voting_models = []
    staking_models = []

dual_models = [
               ]

gam_models = [
        #('gam_binary_classification_logloss', 'binary'),
        #('gam_risk_regression_rmse', 'risk'),
        #('gam_nbsinister_regression_rmse', 'nbsinister'),
]

one_by_id_model = [
       #('xgboost_full_all_one_nbsinister-max-0-kmeans-5-Class-Dept_classification_softmax', 'unique', 'id', post_process_model_dico['ScalerClassRisk_None_None_KMeansRisk_5_nbsinister']),
       #('xgboost_full_all_one_nbsinister-kmeans-5-Class-Dept-both_classification_softmax', 'unique', 'id', post_process_model_dico['ScalerClassRisk_both_None_KMeansRisk_5_nbsinister'])
]

federate_model_by_id_model = []

voting_models_list = [
                    #[('xgboost_risk_regression_rmse', 'risk'),
                    # ('svm_risk_regression_rmse', 'risk'),
                    # ('dt_risk_regresion_rmse', 'risk')]
                    ]

if doTrain:

    for model in models:
        wrapped_train_sklearn_api_model(train_dataset=train_dataset.copy(deep=True),
                                val_dataset=val_dataset.copy(deep=True),
                                test_dataset=test_dataset.copy(deep=True),
                                graph_method=graph_method,
                                dir_output=dir_output / name,
                                device='gpu',
                                autoRegression=autoRegression,
                                features=features_selected,
                                training_mode=training_mode,
                                do_grid_search=do_grid_search,
                                do_bayes_search=do_bayes_search,
                                model=model,
                                scale=scale)
        
    for model in dual_models:
        wrapped_train_sklearn_api_dual_model(train_dataset=train_dataset.copy(deep=True),
                                val_dataset=val_dataset.copy(deep=True),
                                test_dataset=test_dataset.copy(deep=True),
                                graph_method=graph_method,
                                dir_output=dir_output / name,
                                device='gpu',
                                autoRegression=autoRegression,
                                features=features_selected,
                                training_mode=training_mode,
                                do_grid_search=do_grid_search,
                                do_bayes_search=do_bayes_search,
                                model=model,
                                scale=scale)
        
    for model in voting_models:
        wrapped_train_sklearn_api_voting_model(train_dataset=train_dataset.copy(deep=True),
                                val_dataset=val_dataset.copy(deep=True),
                                test_dataset=test_dataset.copy(deep=True),
                                graph_method=graph_method,
                                dir_output=dir_output / name,
                                device='cpu',
                                autoRegression=autoRegression,
                                features=features_selected,
                                training_mode=training_mode,
                                do_grid_search=do_grid_search,
                                do_bayes_search=do_bayes_search,
                                model=model,
                                scale=scale)
        
    for model in staking_models:
        wrapped_train_sklearn_api_stacked_model_list(train_dataset=train_dataset.copy(deep=True),
                                val_dataset=val_dataset.copy(deep=True),
                                test_dataset=test_dataset.copy(deep=True),
                                graph_method=graph_method,
                                dir_output=dir_output / name,
                                device='cpu',
                                autoRegression=autoRegression,
                                features=features_selected,
                                training_mode=training_mode,
                                do_grid_search=do_grid_search,
                                do_bayes_search=do_bayes_search,
                                model=model,
                                scale=scale)
        
    for model in gam_models:
         wrapped_train_sklearn_api_model(train_dataset=train_dataset.copy(deep=True),
                                val_dataset=val_dataset.copy(deep=True),
                                test_dataset=test_dataset.copy(deep=True),
                                graph_method=graph_method,
                                dir_output=dir_output / name,
                                device='cpu',
                                autoRegression=autoRegression,
                                features=features_selected,
                                training_mode=training_mode,
                                do_grid_search=do_grid_search,
                                do_bayes_search=do_bayes_search,
                                model=model,
                                scale=scale)
        
    for model in one_by_id_model:
        wrapped_train_sklearn_api_model(train_dataset=train_dataset.copy(deep=True),
                                val_dataset=val_dataset.copy(deep=True),
                                test_dataset=test_dataset.copy(deep=True),
                                graph_method=graph_method,
                                dir_output=dir_output / name,
                                device='cpu',
                                autoRegression=autoRegression,
                                features=features_selected,
                                training_mode=training_mode,
                                do_grid_search=do_grid_search,
                                do_bayes_search=do_bayes_search,
                                model=model,
                                scale=scale)
        
    for model in federate_model_by_id_model:
        wrapped_train_sklearn_api_model(train_dataset=train_dataset.copy(deep=True),
                                val_dataset=val_dataset.copy(deep=True),
                                test_dataset=test_dataset.copy(deep=True),
                                graph_method=graph_method,
                                dir_output=dir_output / name,
                                device='cpu',
                                autoRegression=autoRegression,
                                features=features_selected,
                                training_mode=training_mode,
                                do_grid_search=do_grid_search,
                                do_bayes_search=do_bayes_search,
                                model=model,
                                scale=scale)

if doTest:

    test_dataset = test_dataset[test_dataset['weight'] > 0]
    #test_dataset_unscale = test_dataset_unscale[test_dataset_unscale['weight'] > 0]
    test_dataset_unscale = None

    print(allDates[int(test_dataset.date.min())])

    host = 'pc'

    logger.info('############################# TEST ###############################')

    dn = dataset_name
    if two:
        dn += '2'

    name_dir = dn + '/' + sinister + '/' + resolution + '/train' + '/'
    dir_train = Path(name_dir)

    name_dir = dn + '/' + sinister + '/' + resolution + '/test' + '/' + name_exp
    dir_output = Path(name_dir)

    #if training_mode == 'normal':
    """if graph_construct == 'risk-size-watershed':
            if scale == 4:
                models = [
                ('xgboost_percentage-0.20-0.0_one_union_classification_softmax'),
                ('xgboost_percentage-0.25-0.0_one_union_classification_softmax'),
                #('xgboost_percentage-0.25-1.0_one_union_classification_softmax'),
                #('xgboost_percentage-0.25_one_union_classification_softmax'),
                #('xgboost_percentage-0.40_one_nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized_classification_softmax'),
                ('xgboost_percentage-0.40-1.0_one_nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized_classification_softmax'),
                ]

            elif scale == 5:
                models = [
                    ('xgboost_percentage-0.20-0.0_one_union_classification_softmax'),
                    ('xgboost_percentage-0.60-0.0_one_union_classification_softmax'),
                    #('xgboost_percentage-0.35_one_union_classification_softmax'),
                    #('xgboost_percentage-0.45_one_nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized_classification_softmax'),
                    ('xgboost_npercentage-0.45-1.0_one_nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized_classification_softmax'),
                ]

            elif scale == 6:
                models = [
                    ('xgboost_percentage-0.20-0.0_one_union_classification_weighted'),
                    #('xgboost_percentage-0.70_one_union_classification_weighted'),
                    #('xgboost_percentage-0.30_one_union_classification_softmax'),
                    #('xgboost_percentage-0.30_one_union_classification_softmax'),
                    ('xgboost_percentage-0.6-1.0_one_nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized_classification_softmax'),
                ]

            elif scale == 7:
                models = [
                    ('xgboost_percentage-0.15-0.0_one_union_classification_softmax'),
                    ('xgboost_percentage-0.40-1.0_one_union_classification_softmax'),
                    #('xgboost_percentage-0.40_one_union_classification_softmax'),
                    #('xgboost_full_all_one_nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized_classification_softmax'),
                    ('xgboost_percentage-0.90-1.0_one_nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized_classification_softmax'),
                ]
        elif scale == 'departement':
                models = [
                    ('xgboost_full_all_one_union_classification_softmax'),                    
                    ('xgboost_full_all_one_union_classification_softmax'),                    
                    ('xgboost_full_all_one_nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized_classification_softmax'),
                    ('xgboost_full_all_one_nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized_classification_softmax'),
                ]
        else:
            if scale == 4:
                models = [
                    ('xgboost_percentage-0.15-0.0_one_union_classification_softmax'),
                ]

            elif scale == 5: 
                models = [
                    ('xgboost_percentage-0.20-0.0_one_union_classification_softmax'),
                ]

            elif scale == 6:
                models = [
                    ('xgboost_percentage-0.20-0.0_one_union_classification_softmax'),
                ]

            elif scale == 7:
                models = [
                    ('xgboost_percentage-0.30-0.0_one_union_classification_softmax'),
                ]
    else:"""
    models = [
        #('xgboost_search_smote-2_all_one_union_classification_weighted'),
        #('filter_full_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),

        #('filter-catboost-soft-departement-all_full_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('filter-catboost-soft-graphid-all_full_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),

        #('catboost_search_smote-2_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('catboost_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('xgboost_search_smote-2_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),

        #('xgboost_search_smote-4_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('catboost_search_smote-4_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('lg_search_smote-4_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
#
        #('xgboost_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('catboost_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('lg_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
#
        #('xgboost_search_smote-6_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('catboost_search_smote-6_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('lg_search_smote-6_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
#
        #('xgboost_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('catboost_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('lg_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
#
        #('xgboost_search_smote-4_0_all_one_burnedarea-kmeans-5-Class-Dept_classification_softmax'),
        #('catboost_search_smote-4_0_all_one_burnedarea-kmeans-5-Class-Dept_classification_softmax'),
        #('lg_search_smote-4_0_all_one_burnedarea-kmeans-5-Class-Dept_classification_l2'),
#
        #('xgboost_search_smote-2_0_all_one_burnedarea-kmeans-5-Class-Dept_classification_softmax'),
        #('catboost_search_smote-2_0_all_one_burnedarea-kmeans-5-Class-Dept_classification_softmax'),
        #('lg_search_smote-2_0_all_one_burnedarea-kmeans-5-Class-Dept_classification_l2'),
        #
        #('xgboost_search_smote-6_0_all_one_burnedarea-kmeans-5-Class-Dept_classification_softmax'),
        #('catboost_search_smote-6_0_all_one_burnedarea-kmeans-5-Class-Dept_classification_softmax'),
        #('lg_search_smote-6_0_all_one_burnedarea-kmeans-5-Class-Dept_classification_l2'),

        #('catboost_search_full_0_all_one_nbsinister-binary_classification_logloss'),

        #('xgboost_search_full_0_all_one_burnedarea-kmeans-5-Class-Dept_classification_softmax'),
        #('catboost_search_full_0_all_one_burnedarea-kmeans-5-Class-Dept_classification_softmax'),
        #('lg_search_full_0_all_one_burnedarea-kmeans-5-Class-Dept_classification_l2'),

        ('xgboost_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('catboost_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),

        ('xgboost_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_mcewk'),
        ('catboost_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_mcewk'),

        ('lg_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),

        ('xgboost_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('catboost_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('lg_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),

        ('xgboost_search_smote-4_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('catboost_search_smote-4_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('lg_search_smote-4_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),

        ('xgboost_search_smote-6_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('catboost_search_smote-6_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('lg_search_smote-6_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
        
        #('ngboost_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_CRPScore'),

        #('xgboost_search_smote-2_0_all_one_burnedarea-kmeans-5-Class-Dept_classification_softmax'),
        #('catboost_search_smote-2_0_all_one_burnedarea-kmeans-5-Class-Dept_classification_softmax'),
        #('lg_search_smote-2_0_all_one_burnedarea-kmeans-5-Class-Dept_classification_l2'),
        #('ngboost_search_smote-2_0_all_one_burnedarea-kmeans-5-Class-Dept_classification_CRPScore'),
        
        #('lg_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_elasticnet'),
        #('ngboost_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_normal'),
        
        #('filter-catboost-soft-weight-all_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('filter-catboost-hard-None-all_full_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('filter-catboost-hard-weight-all_full_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        
        #('filter-catboost-soft-weight-all_full_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('filter-catboost-soft-weight-5_full_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('filter-catboost-soft-weight-1_full_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
#
        ('filter-xgboost-soft-weight-all_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-catboost-soft-weight-all_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-lg-soft-weight-all_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),

        ('filter-lg-soft-weight-1_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
        ('filter-lg-soft-weight-2_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
        ('filter-lg-soft-weight-3_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
        ('filter-lg-soft-weight-4_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
        ('filter-lg-soft-weight-5_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
        ('filter-lg-soft-weight-6_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
        ('filter-lg-soft-weight-7_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
        ('filter-lg-soft-weight-8_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
        ('filter-lg-soft-weight-9_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
        ('filter-lg-soft-weight-10_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
        ('filter-lg-soft-weight-11_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
        ('filter-lg-soft-weight-12_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
        ('filter-lg-soft-weight-13_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
        ('filter-lg-soft-weight-14_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
        ('filter-lg-soft-weight-15_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
        ('filter-lg-soft-weight-16_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
        ('filter-lg-soft-weight-17_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
        ('filter-lg-soft-weight-18_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
        ('filter-lg-soft-weight-19_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),
        ('filter-lg-soft-weight-20_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_l2'),

        ('filter-catboost-soft-weight-1_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-catboost-soft-weight-2_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-catboost-soft-weight-3_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-catboost-soft-weight-4_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-catboost-soft-weight-5_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-catboost-soft-weight-6_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-catboost-soft-weight-7_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-catboost-soft-weight-8_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-catboost-soft-weight-9_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-catboost-soft-weight-10_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-catboost-soft-weight-11_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-catboost-soft-weight-12_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-catboost-soft-weight-13_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-catboost-soft-weight-14_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-catboost-soft-weight-15_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-catboost-soft-weight-16_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-catboost-soft-weight-17_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-catboost-soft-weight-18_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-catboost-soft-weight-19_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-catboost-soft-weight-20_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),

        ('filter-xgboost-soft-weight-1_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-xgboost-soft-weight-2_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-xgboost-soft-weight-3_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-xgboost-soft-weight-4_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-xgboost-soft-weight-5_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-xgboost-soft-weight-6_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-xgboost-soft-weight-7_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-xgboost-soft-weight-8_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-xgboost-soft-weight-9_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-xgboost-soft-weight-10_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-xgboost-soft-weight-11_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-xgboost-soft-weight-12_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-xgboost-soft-weight-13_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-xgboost-soft-weight-14_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-xgboost-soft-weight-15_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-xgboost-soft-weight-16_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-xgboost-soft-weight-17_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-xgboost-soft-weight-18_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-xgboost-soft-weight-19_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        ('filter-xgboost-soft-weight-20_search_full_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),


        #('filter-xgboost-soft-weight-all_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
#
        #('filter-xgboost-soft-None-all_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('filter-xgboost-hard-weight-all_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('filter-xgboost-hard-None-all_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
#
        #('filter-xgboost-soft-weight-1_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('filter-xgboost-soft-weight-5_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('filter-xgboost-soft-weight-10_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('filter-xgboost-soft-weight-15_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('filter-xgboost-soft-weight-20_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
#
        #('filter-xgboost-soft-departement-all_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('filter-xgboost-soft-graphid-all_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
#
        #('filter-xgboost-soft-departement-1_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('filter-xgboost-soft-graphid-1_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
#
        #('filter-xgboost-soft-departement-5_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),
        #('filter-xgboost-soft-graphid-5_search_smote-2_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),

        #('xgboost_search_smote-2_all_one_nbsinister-kmeans-5-Class-Dept-cubic-5_classification_softmax'),
        #('xgboost_search_smote-2_all_one_nbsinister-kmeans-5-Class-Dept-cubic-5_classification_softmax-dual'),
        #('catboost_search_smote-2_all_one_nbsinister-kmeans-5-Class-Dept-circular-5_classification_softmax'),

        #('xgboost_search_smote-2_all_one_nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized_classification_softmax'),
        #('staking_full_all_one_nbsinister-kmeans-5-Class-Dept_classification_softmax'),

        #('xgboost_search_smote-2_all_one_nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized_classification_weighted'),
        #('xgboost_search_smote-2_all_one_nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized_classification_softmax'),
    ]
        
    prefix_kmeans = f'full_{k_days}_{scale}_{graph_construct}_{top_cluster}'

    if days_in_futur > 0:
        prefix_kmeans += f'_{days_in_futur}_{futur_met}'

    ####################################################### Test on all Dataset ####################################################
    metrics, metrics_dept, res, res_dept = test_sklearn_api_model(config, graphScale, test_dataset,
                                None,
                                    'all',
                                    prefix,
                                    prefix_config,
                                    models,
                                    dir_output / 'all' / prefix,
                                    device,
                                    encoding,
                                    scaling,
                                    test_dataset.departement.unique(),
                                    dir_train,
                                    name_exp,
                                    dir_train / 'check_none' / prefix_kmeans / 'kmeans',
                                    doKMEANS
                                    )

    df_metrics = pd.DataFrame.from_dict(metrics, orient='index').reset_index()

    ####################################################### Test by departmenent ###################################################

    aggregated_prediction = []
    aggregated_prediction_dept = []

    """for dept in departements:
        if MLFLOW:
            dn = dataset_name
            if two:
                dn += '2'
            exp_name = f"{name_exp}_{dn}_{dept}_{sinister}_{sinister_encoding}_test"
            experiments = client.search_experiments()
            
            if exp_name not in list(map(lambda x: x.name, experiments)):

                client.create_experiment(name=exp_name)

            mlflow.set_experiment(exp_name)

        test_dataset_dept = test_dataset[(test_dataset['departement'] == name2int[dept])].reset_index(drop=True)
        if test_dataset_unscale is not None:
            test_dataset_unscale_dept = test_dataset_unscale[(test_dataset_unscale['departement'] == name2int[dept])].reset_index(drop=True)
        else:
            test_dataset_unscale_dept = None
        
        if test_dataset_dept.shape[0] < 5:
            continue

        logger.info(f'{dept} test : {test_dataset_dept.shape}, {np.unique(test_dataset_dept["id"].values)}')

        if host == 'pc':
            metrics, metrics_dept, res, res_dept = test_sklearn_api_model(config, graphScale, test_dataset_dept,
                                test_dataset_unscale_dept,
                                    dept,
                                    prefix,
                                    prefix_config,
                                    models,
                                    dir_output / dept / prefix,
                                    device,
                                    encoding,
                                    scaling,
                                    [dept],
                                    dir_train,
                                    name_exp,
                                    dir_train / 'check_none' / prefix_kmeans / 'kmeans',
                                    doKMEANS
                                    )
        
            res = pd.concat(res).reset_index(drop=True)
        else:
            metrics = read_object('metrics'+'_'+prefix+'_'+scaling+'_'+encoding+'_'+dept+'_tree.pkl', dir_output / dept / prefix)
            assert metrics is not None
            run = f'{dept}_{name}_{prefix}'
            for name in models:
                res = read_object(name+'_'+prefix+'_'+scaling+'_'+encoding+'_'+dept+'_pred.pkl', dir_output / dept / name / prefix)
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

        aggregated_prediction.append(res)

        #aggregated_prediction_dept.append(res_dept)"""

    df_metrics.rename({'index': 'Run'}, inplace=True, axis=1)
    df_metrics.reset_index(drop=True, inplace=True)
    
    check_and_create_path(dir_output / prefix)
    df_metrics.to_csv(dir_output / prefix / 'df_metrics_trees.csv')

    #wrapped_compare(df_metrics,  dir_output / prefix, 'Model')
    #wrapped_compare(df_metrics,  dir_output / prefix, 'Loss_function')
    #wrapped_compare(df_metrics,  dir_output / prefix, 'Target')
    #wrapped_compare(df_metrics,  dir_output / prefix, 'Days_in_futur')

    """############## Prediction ######################
    for name, target_name, _ in models:
        if name not in res.model.unique():
            continue
        if target_name == 'risk':
            predictor = read_object(f'{dept}Predictor{scale}_{graphScale.base}.pkl', dir_train / 'influenceClustering')
        else:
            predictor = read_object(f'{dept}Predictor{name}{scale}_{graphScale.base}.pkl', dir_train / 'influenceClustering')

        if target_name == 'binary':
            vmax_band = 1
        else:
            #vmax_band = np.nanmax(train_dataset[train_dataset['departement'] == name2int[dept]][target_name].values)
            vmax_band = np.nanmax(train_dataset[target_name].values)

        res_test = res[res['model'] == name]
        res_test_dept = res_dept[res_dept['model'] == name]
        regions_test = geo[geo['departement'] == dept]
        #dates = res_test[(res_test['risk'] == res_test['risk'].max())]['date'].values
        dates = allDates.index('2023-07-14')

        susectibility_map_france_daily_geojson(res_test, regions_test, graphScale, np.unique(dates).astype(int), 'prediction',
                                    scale, dir_output / dept / prefix / name, vmax_band=vmax_band,
                                    dept_reg=False, sinister=sinister, sinister_point=fp)

        #susectibility_map_france_daily_image(df=res_test, vmax=vmax_band, graph=graphScale, departement=dept, dates=dates,
        #                resolution='2x2', region_dept=regions_test, column='prediction',
        #                dir_output=dir_output / dept / prefix / name, predictor=predictor, sinister_point=fp.copy(deep=True))

        train_dataset_dept = train_dataset[train_dataset['departement'] == name2int[dept]].groupby(['departement', 'date'])[target_name].sum().reset_index()
        vmax_band = np.nanmax(train_dataset_dept[target_name].values)

        susectibility_map_france_daily_geojson(res_test_dept,
                                    regions_test, graphScale, np.unique(dates).astype(int),
                                    target_name, 'departement',
                                    dir_output / dept / prefix / name, vmax_band=vmax_band,
                                    dept_reg=True, sinister=sinister, sinister_point=fp)

        ################# Ground Truth #####################
        name = 'GT'
        susectibility_map_france_daily_geojson(res_test, regions_test, graphScale, np.unique(dates).astype(int), target_name,
                                    f'{scale}_gt', dir_output / dept / prefix / name, vmax_band=vmax_band,
                                    dept_reg=False, sinister=sinister, sinister_point=fp)
        
        susectibility_map_france_daily_geojson(res_test_dept,
                                    regions_test, graphScale, np.unique(dates).astype(int),
                                    target_name, 'departement_gt',
                                    dir_output / dept / prefix / name, vmax_band=vmax_band,
                                    dept_reg=True, sinister=sinister, sinister_point=fp)"""

    if dataset_name != 'bdiff':
        exit(1)

    aggregated_prediction = pd.concat(aggregated_prediction).reset_index(drop=True)
    #aggregated_prediction_dept = pd.concat(aggregated_prediction_dept).reset_index(drop=True)

    aggregated_prediction.to_csv(dir_output / 'aggregated_prediction.csv', index=False)
    fp = pd.read_csv(f'sinister/{dataset_name}/{sinister}.csv', dtype=str)

    """for name in models:
        model_name, under_sampling, over_sampling, kdays, nbfeatures, weight_type, target_name, task_type, loss = name.split('_')
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
        vmax_band = np.nanmax(train_dataset_dept[target_name].values)"""
        
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