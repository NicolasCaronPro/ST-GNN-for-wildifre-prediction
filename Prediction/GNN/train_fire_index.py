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

args = parser.parse_args()

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

######################## Get features and train features list ######################

isInference = name_exp == 'inference'

features, train_features, kmeasn_features = get_features_for_sinister_prediction(dataset_name, sinister, isInference)

######################## Get departments and train departments #######################

departements, train_departements = select_departments(dataset_name, sinister)

############################# GLOBAL VARIABLES #############################

name_exp = f'{sinister_encoding}_{name_exp}'

dir_target = root_target / sinister / dataset_name / sinister_encoding / 'log' / resolution

geo = gpd.read_file(f'regions/{sinister}/{dataset_name}/regions.geojson')

geo = geo[geo['departement'].isin(departements)].reset_index(drop=True)

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

####################### INIT ################################

df, graphScale, prefix, fp, features_selected = init(args, dir_output, 'train_gnn')
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
                                                                                    ['mean'], args)

if nbfeatures == 'all':
    nbfeatures = len(features_selected)
else:
    nbfeatures = int(nbfeatures)

varying_time_variables_2 = get_time_columns(varying_time_variables, k_days, train_dataset.copy(), train_features)
features_name, newshape = get_features_name_list(graphScale.scale, train_features, METHODS_SPATIAL_TRAIN)
features_selected_str = get_features_selected_for_time_series(features_selected, features_name, varying_time_variables_2, nbfeatures)

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

def plot_unknowed_sample(df, dir_output, outname):
    index_to_drop = df[(df['nbsinister-kmeans-5-Class-Dept'] == 0) & (df['nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized'] > 0)].index
    risky_day = df.loc[index_to_drop]
    risky_day.sort_values('month_non_encoder')

    plt.figure(figsize=(10, 6))
    sns.histplot(risky_day['month_non_encoder'], bins=12, kde=False, color='blue', edgecolor='black')
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Frquency of unknowed risk', fontsize=14)
    plt.xticks(rotation=45)  # Rotation des labels des mois si nécessaire
    
    # Affichage du graphique
    plt.tight_layout()
    plt.savefig(dir_output / f'{outname}_risk_day_month.png')

    plt.figure(figsize=(10, 6))
    sns.histplot(risky_day['graph_id'], bins=12, kde=False, color='blue', edgecolor='black')
    plt.xlabel('Graph_id', fontsize=14)
    plt.ylabel('Frquency of unknowed risk', fontsize=14)
    plt.xticks(rotation=45)  # Rotation des labels des mois si nécessaire
    
    # Affichage du graphique
    plt.tight_layout()
    plt.savefig(dir_output / f'{outname}_risk_graph_id.png')
    plt.close('all')
    
    dff = df.copy(deep=True)
    
    y_true_temp = np.ones((dff.shape[0], 7))
    y_true_temp[:, graph_id_index] = dff['graph_id']
    y_true_temp[:, id_index] = dff['graph_id']
    y_true_temp[:, departement_index] = dff['departement']
    y_true_temp[:, date_index] = dff['date']
    y_true_temp[:, -1] = dff['nbsinister-kmeans-5-Class-Dept'].values
    y_true_temp[:, -2] = dff['nbsinister']
    y_true_temp[:, -3] = dff['nbsinister']

    iou_vis(dff['nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized'].values, y_true_temp, -1, dir_output, f'{outname}_unknowed_samples')

plot_unknowed_sample(train_dataset, dir_output, f'train_dataset_{scale}_{graph_construct}_{graph_method}')
plot_unknowed_sample(test_dataset, dir_output, f'test_dataset_{scale}_{graph_construct}_{graph_method}')
plot_unknowed_sample(val_dataset, dir_output, f'val_dataset_{scale}_{graph_construct}_{graph_method}')

models = []

#for col in new_cols:
#    models.append(Statistical_Model(col, None, int(col.split('-')[2]), col, 'classification'))
    #test_dataset_unscale[col] = test_dataset[col]
#    

models.append(Statistical_Model('nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized-Past', 'shift', 5, 'nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized', 'classification'))
models.append(Statistical_Model('nbsinister-kmeans-5-Class-Dept', 'shift', 5, 'nbsinister-kmeans-5-Class-Dept', 'classification'))
models.append(Statistical_Model('union', None, 5, 'union', 'classification'))

cols = ['nbsinister-kmeans-5-Class-Dept', 'nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized',  'nbsinister-kmeans-5-Class-Dept-laplace+mean-Specialized-Past']
test_dataset_unscale = test_dataset_unscale.set_index(['graph_id', 'date']).join(test_dataset.set_index(['graph_id', 'date'])[cols], on=['graph_id', 'date']).reset_index()

for model in models:
    model.fit(train_dataset_unscale, 'departement')
    save_object(model, f'{model.name}.pkl', dir_output / Path('check_'+scaling + '/' + prefix + '/' + 'baseline') / model.name)

if doTest:

    host = 'pc'

    logger.info('############################# TEST ############## #################')

    name_dir = dataset_name + '/' + sinister + '/' + resolution + '/train' + '/'
    dir_train = Path(name_dir)

    name_dir = dataset_name + '/' + sinister + '/' + resolution + '/test' + '/' + name_exp
    dir_output = Path(name_dir)
    
    str_models = []
    for model in models:
        str_models.append((model.name, model.target))

    prefix_kmeans = f'full_{k_days}_{scale}_{graph_construct}_{graph_method}'

    if days_in_futur > 0:
        prefix_kmeans += f'_{days_in_futur}_{futur_met}'

    aggregated_prediction = []
    aggregated_prediction_dept = []

    for dept in departements:
        if MLFLOW:
            dn = dataset_name
            if two:
                dn += '2'
            exp_name = f"{name_exp}_{dn}_{dept}_{sinister}_{sinister_encoding}_test"
            experiments = client.search_experiments()

            if exp_name not in list(map(lambda x: x.name, experiments)):
                tags = {
                    "",
                }

                client.create_experiment(name=exp_name, tags=tags)

        test_dataset_dept = test_dataset[(test_dataset['departement'] == name2int[dept])].reset_index(drop=True)
        if test_dataset_unscale is not None:
            test_dataset_unscale_dept = test_dataset_unscale[(test_dataset_unscale['departement'] == name2int[dept])].reset_index(drop=True)
        else:
            test_dataset_unscale_dept = None

        if test_dataset_dept.shape[0] < 5:
            continue
        
        if host == 'pc':
            logger.info(f'{dept} test : {test_dataset_dept.shape}, {np.unique(test_dataset_dept["graph_id"].values)}')

            metrics, metrics_dept, res, res_dept = test_fire_index_model(vars(args), graphScale, test_dataset_dept.copy(deep=True),
                                test_dataset_unscale_dept.copy(deep=True),
                                    dept,
                                    prefix,
                                    str_models,
                                    dir_output / dept / prefix,
                                    prefix_config,
                                    encoding,
                                    name_exp,
                                    scaling,
                                    [dept],
                                    dir_train,
                                    )
        else:
            metrics = read_object('metrics'+'_'+prefix+'_'+scaling+'_'+encoding+'_'+dept+'.pkl', dir_output / dept / prefix)
            assert metrics is not None
            run = f'{dept}_{name}_{prefix}'
            for name, target_name in models:
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
            
    df_metrics.rename({'index': 'Run'}, inplace=True, axis=1)
    df_metrics.reset_index(drop=True, inplace=True)

    check_and_create_path(dir_output / prefix)
    df_metrics.to_csv(dir_output / prefix / 'df_metrics.csv')
    
    """res = pd.concat(res).reset_index(drop=True)
    res_dept = pd.concat(res_dept).reset_index(drop=True)

    ############## Prediction ######################
    for name, target_name in models:
        if name not in res.model.unique():
            continue
        if target_name == 'risk':
            predictor = read_object(f'{dept}Predictor{scale}_{graphScale.base}.pkl', dir_train / 'influenceClustering')
        else:
            predictor = read_object(f'{dept}Predictor{name}{scale}_{graphScale.base}.pkl', dir_train / 'influenceClustering')

        vmax_band = np.nanmax(train_dataset_unscale[train_dataset_unscale['departement'] == name2int[dept]][name].values)

        res_test = res[res['model'] == name]
        res_test_dept = res_dept[res_dept['model'] == name]
        regions_test = geo[geo['departement'] == dept]
        dates = res_test_dept[(res_test_dept['nbsinister'] == res_test_dept['nbsinister'].max())]['date'].values
        #dates = res_test_dept[(res_test_dept['nbsinister'] == res_test_dept['nbsinister'].max()) | (res_test_dept['risk'] >= res_test_dept['risk'].max() * 0.90)]['date'].values
        
        susectibility_map_france_daily_geojson(res_test, regions_test, graphScale, np.unique(dates).astype(int), 'prediction',
                                    scale, dir_output / dept / prefix / name, vmax_band=vmax_band,
                                    dept_reg=False, sinister=sinister, sinister_point=fp)
        
        susectibility_map_france_daily_image(df=res_test, vmax=vmax_band, graph=graphScale, departement=dept, dates=dates,
                        resolution='0.03x0.03', region_dept=regions_test, column='prediction',
                        dir_output=dir_output / dept / prefix / name, predictor=predictor, sinister_point=fp.copy(deep=True))

    ################# Ground Truth #####################
    name = 'GT'
    susectibility_map_france_daily_geojson(res_test, regions_test, graphScale, np.unique(dates).astype(int), 'risk',
                                f'{scale}_gt', dir_output / dept / prefix / name, vmax_band=vmax_band,
                                dept_reg=False, sinister=sinister, sinister_point=fp)
    
    susectibility_map_france_daily_geojson(res_test_dept,
                                regions_test, graphScale, np.unique(dates).astype(int),
                                target_name, 'departement_gt',
                                dir_output / dept / prefix / name, vmax_band=vmax_band,
                                dept_reg=True, sinister=sinister, sinister_point=fp)
    
    aggregated_prediction.append(res)
    aggregated_prediction_dept.append(res_dept)"""