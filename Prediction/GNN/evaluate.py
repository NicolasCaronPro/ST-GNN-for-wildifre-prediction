from tools import *
import matplotlib.pyplot as plt
import argparse
import matplotlib.colors as mcolors
from glob import glob

XKCD_COLORS = {
    'cloudy blue': '#acc2d9',
    'dark pastel green': '#56ae57',
    'dust': '#b2996e',
    'electric lime': '#a8ff04',
    'fresh green': '#69d84f',
    'light eggplant': '#894585',
    'nasty green': '#70b23f',
    'really light blue': '#d4ffff',
    'tea': '#65ab7c',
    'warm purple': '#952e8f',
    'yellowish tan': '#fcfc81',
    'cement': '#a5a391',
    'dark grass green': '#388004',
    'dusty teal': '#4c9085',
    'grey teal': '#5e9b8a',
    'macaroni and cheese': '#efb435',
    'pinkish tan': '#d99b82',
    'spruce': '#0a5f38',
    'strong blue': '#0c06f7',}

def evaluate_ca(metrics, met):
    df = pd.DataFrame(index=np.arange(len(metrics)), columns=['model', 'departement', met])
    i = 0
    for key, value in metrics.items():
        if met not in value.keys():
                df.loc[i, 'model'] = key
                df.loc[i, 'departement'] = -1
                df.loc[i, met] = -1
                i += 1
        else:
            for keys in value[met].keys():
                df.loc[i, 'model'] = key
                df.loc[i, 'departement'] = keys
                df.loc[i, met] = round(value[met][keys], 3)
                i += 1
    return df

def evaluate_f1(metrics, met):
    df = pd.DataFrame(index=np.arange(len(metrics)), columns=['model', 'f1', 'precision', 'recall'])
    i = 0
    for key, value in metrics.items():
        if met not in value.keys():
            df.loc[i, 'model'] = key
            df.loc[i, 'f1'] = -1
            df.loc[i, 'precision'] = -1
            df.loc[i, 'recall'] = -1
            df.loc[i, 'tresh'] = -1
        else:
            df.loc[i, 'model'] = key
            df.loc[i, 'f1'] = value[met][0]
            df.loc[i, 'precision'] = value[met][1]
            df.loc[i, 'recall'] = value[met][2]
            df.loc[i, 'tresh'] = value[met][3]
            #df.loc[i, 'value'] = value[met][4]
        i += 1
    return df.sort_values('f1')

def evaluate_met(metrics, met):
    df = pd.DataFrame(index=np.arange(len(metrics)), columns=['model', met])
    i = 0
    for key, value in metrics.items():
        if met not in value.keys():
            df.loc[i, 'model'] = key
            df.loc[i, met] = -1
            i += 1
        else:
            df.loc[i, 'model'] = key
            df.loc[i, met] = value[met]
            i += 1
    return df.sort_values(met)

def evaluate_cr(metrics, met):
    df = pd.DataFrame(index=np.arange(len(metrics)), columns=['model', met])
    i = 0
    for key, value in metrics.items():
        df.loc[i, 'model'] = key
        print(value[met])
        df.loc[i, met] = value[met]
        i += 1
    return df.sort_values(met)

def plot_cr(df, met, dir_output, out_name, color):
    pass

def plot(df, met, dir_output, out_name, color):
    index = df['index'].values
    label = df['label'].values
    model = df.model.values
    mets = df[met].values
    
    scatters = []
    labels = []
    scatters_unique = []

    fig = plt.figure(figsize=(30, 15))
    for i, m in enumerate(mets):
        sc = plt.scatter(m, model[i], c=color[index[i]], alpha=0.8)
        scatters.append(sc)
        if label[i] not in labels:
            labels.append(label[i])
            scatters_unique.append(sc)

    plt.xlabel(met)
    plt.ylabel('Model')
    plt.grid(True)
    plt.legend(scatters_unique, labels, loc='upper left', ncol=3)
    out_name += '.png'
    plt.savefig(dir_output / out_name)
    plt.close('all')

def plot_by_index(df, met, dir_output, out_name, color, index):
    label = df['label'].values
    model = df.model.values
    mets = df[met].values
    scales = df['scale'].values

    dfgp = df.groupby(['model', index])[met].mean().reset_index()
    dfgp.index = dfgp[index]

    fig, ax = plt.subplots(1, figsize=(15,10))

    umodels = dfgp.model.unique()
    for i, model in enumerate(umodels):
        dfgp[dfgp['model'] == model][met].plot(label=model, color=color[model], ax=ax)

    plt.xlabel('scale')
    plt.ylabel(met)
    plt.grid(True)
    plt.legend(loc='upper left', ncol=3)
    out_name += f'_{index}.png'
    plt.savefig(dir_output / out_name)
    plt.close('all')

def plot_features_importances(df, name, dir_output, color):
    dfb = df.groupby(['scale', 'features_name'])['Permutation_importance'].mean().reset_index()

    fig, ax = plt.subplots(1, figsize=(15,10))
    dfb.index = dfb.scale
    features_name = dfb.features_name
    print(color)
    for i, fet in enumerate(features_name):
        dfb[dfb['features_name'] == fet].plot(label=fet, color=color[fet], ax=ax)
    plt.title(name)
    plt.legend(loc='upper left', ncol=3)
    plt.savefig(dir_output / f'{name}.png')

def plot_variation(df, base_label, met, dir_output, out_name, color):

    base_df = df[df['label'] == base_label]

    # Les données du DataFrame
    base_mets = base_df[met].values
    base_mets = np.repeat(base_mets, len(df) / len(base_mets))

    index = df['index'].values
    label = df['label'].values
    model = df.model.values
    mets = base_mets - df[met].values

    scatters = []
    labels = []
    scatters_unique = []

    fig = plt.figure(figsize=(15, 5))
    for i, m in enumerate(mets):
        sc = plt.scatter(m, model[i], c=color[index[i]], alpha=0.8)
        scatters.append(sc)
        if label[i] not in labels:
            labels.append(label[i])
            scatters_unique.append(sc)

    plt.xlabel(met)
    plt.ylabel('Model')
    #plt.xlim(-0.1,1.1)
    plt.grid(True)
    plt.legend(scatters_unique, labels, loc='upper right', ncol=3)

    out_name += '.png'
    plt.savefig(dir_output / out_name)

def load_and_evaluate(experiments, test_name, dir_output, dir_input, sinister):

    f1s, maes, bcas, rmses, maetop10s, crs, cals = [], [], [], [], [], [], []
    features_importance_test = []
    features_importance_train = []
    features_name_full = []
    (_, _, base_label, _) = experiments[0]

    for elt in experiments:
        (expe, index, label, prefix) = elt
        dir_train = dir_input / Path(sinister + '/' + resolution + '/train/' + expe + '/check_z-score/' + label + '/baseline' )
        dir_test = dir_input / Path(sinister + '/' + resolution + '/test/' + expe + '/' + test_name + '/' + label + '/')
        vec = label.split('_')
        values_per_class, k_days, scale, nf = vec[0], vec[1], vec[2], vec[3]
        label = expe + ' ' + str(index) + ' : ' + label 
        name = 'metrics_'+prefix+'.pkl'
        metrics = read_object(name, dir_test)

        # Features importance
        sub_features_importance_train = []
        sub_features_importance_test = []
        sub_features_importance_index = []
        sub_features_name = []
        features_name = read_object('features_name_train.pkl', dir_input / sinister / resolution / 'train' / expe)
        features_name_full.extend(features_name)

        filename = 'test_permutation_importances.pkl'
        filename2 = 'features.pkl'
        for root, dirs, files in os.walk(dir_test):
            for dir in dirs:
                importances = read_object(filename, dir_test / dir)

                if importances is None:
                    continue
                importances = importances.importances_mean
                indices = importances.argsort()[:10]
                sub_features_importance_test.extend(importances[indices])

                fet = read_object(filename2, dir_train / dir)
                if fet is None:
                    continue
                sub_features_name.extend(fet)
        
        """if len(sub_features_importance_test) > 0:
            sub_features_importance_test_df = pd.DataFrame(index=np.arange(0, len(sub_features_importance_test)))
            sub_features_importance_test_df['Permutation_importance'] = sub_features_importance_test
            sub_features_importance_test_df['features_name'] = sub_features_name
            sub_features_importance_test_df['expe'] = expe
            sub_features_importance_test_df['index'] = index
            sub_features_importance_test_df['label'] = label
            sub_features_importance_test_df['values_per_class'] = values_per_class
            sub_features_importance_test_df['k_days'] = k_days
            sub_features_importance_test_df['scale'] = scale
            sub_features_importance_test_df['nf'] = nf
            features_importance_test.append(sub_features_importance_test_df)
        else:
            logger.info('No file found for test permutation importance')
        sub_features_name = [] 
        filename = 'train_permutation_importances.pkl'
        for root, dirs, files in os.walk(dir_train):
            for dir in dirs:
                importances = read_object(filename, dir_train / dir)
                if importances is None:
                    continue
                importances = importances.importances_mean
                indices = importances.argsort()[:10]
                sub_features_importance_test.extend(importances[indices])

                fet = read_object(filename2, dir_train / dir)
                if fet is None:
                    continue

                sub_features_name.extend(fet)

        if len(sub_features_importance_train) > 0:
            sub_features_importance_train_df = pd.DataFrame(index=np.arange(0, len(sub_features_importance_train)))
            sub_features_importance_train_df['Permutation_importance'] = sub_features_importance_train
            sub_features_importance_train_df['features_name'] = sub_features_name
            sub_features_importance_train_df['expe'] = expe
            sub_features_importance_train_df['index'] = index
            sub_features_importance_train_df['label'] = label
            sub_features_importance_train_df['values_per_class'] = values_per_class
            sub_features_importance_train_df['k_days'] = k_days
            sub_features_importance_train_df['scale'] = scale
            sub_features_importance_train_df['nf'] = nf
            features_importance_train.append(sub_features_importance_train_df)
        else:
            logger.info('No file found for train permutation importance')"""

        # F1 score
        f1 = evaluate_f1(metrics, 'f1')
        f1['f1'] = f1['f1'].astype(float)
        f1['precision'] = evaluate_f1(metrics, 'f1')['precision'].astype(float)
        f1['recall'] = evaluate_f1(metrics, 'f1')['recall'].astype(float)
        f1['expe'] = expe
        f1['index'] = index
        f1['label'] = label
        f1['values_per_class'] = values_per_class
        f1['k_days'] = k_days
        f1['scale'] = scale
        f1['nf'] = nf

        # F1 score
        f1['f1_unweighted'] = evaluate_f1(metrics, 'f1_unweighted')['f1']

        f1['f1_winter'] = evaluate_f1(metrics, 'f1_winter')['f1']
        f1['f1_unweighted_winter'] = evaluate_f1(metrics, 'f1_unweighted_winter')['f1']

        f1['f1_spring'] = evaluate_f1(metrics, 'f1_spring')['f1']
        f1['f1_unweighted_spring'] = evaluate_f1(metrics, 'f1_unweighted_spring')['f1']

        f1['f1_summer'] = evaluate_f1(metrics, 'f1_summer')['f1']
        f1['f1_unweighted_summer'] = evaluate_f1(metrics, 'f1_unweighted_summer')['f1']

        f1['f1_autumn'] = evaluate_f1(metrics, 'f1_winter')['f1']
        f1['f1_unweighted_autumn'] = evaluate_f1(metrics, 'f1_unweighted_autumn')['f1']

        f1['f1_top_5_cluster'] = evaluate_f1(metrics, 'f1_top_5_cluster')['f1']
        f1['f1_top_5_cluster_unweighted'] = evaluate_f1(metrics, 'f1_top_5_cluster_unweighted')['f1']

        # Precision
        f1['precision_unweighted'] = evaluate_f1(metrics, 'f1_unweighted')['precision']

        f1['precision_winter'] = evaluate_f1(metrics, 'f1_winter')['precision']
        f1['precision_unweighted_winter'] = evaluate_f1(metrics, 'f1_unweighted_winter')['precision']

        f1['precision_spring'] = evaluate_f1(metrics, 'f1_spring')['precision']
        f1['precision_unweighted_spring'] = evaluate_f1(metrics, 'f1_unweighted_spring')['precision']

        f1['precision_summer'] = evaluate_f1(metrics, 'f1_summer')['precision']
        f1['precision_unweighted_summer'] = evaluate_f1(metrics, 'f1_unweighted_summer')['precision']

        f1['precision_autumn'] = evaluate_f1(metrics, 'f1_winter')['precision']
        f1['precision_unweighted_autumn'] = evaluate_f1(metrics, 'f1_unweighted_autumn')['precision']

        f1['precision_top_5_cluster'] = evaluate_f1(metrics, 'f1_top_5_cluster')['precision']
        f1['precision_top_5_cluster_unweighted'] = evaluate_f1(metrics, 'f1_top_5_cluster')['precision']

        # Recall
        f1['recall_unweighted'] = evaluate_f1(metrics, 'f1_unweighted')['f1']

        f1['recall_winter'] = evaluate_f1(metrics, 'f1_winter')['recall']
        f1['recall_unweighted_winter'] = evaluate_f1(metrics, 'f1_unweighted_winter')['recall']

        f1['recall_spring'] = evaluate_f1(metrics, 'f1_spring')['recall']
        f1['recall_unweighted_spring'] = evaluate_f1(metrics, 'f1_unweighted_spring')['recall']

        f1['recall_summer'] = evaluate_f1(metrics, 'f1_summer')['recall']
        f1['recall_unweighted_summer'] = evaluate_f1(metrics, 'f1_unweighted_summer')['recall']

        f1['recall_autumn'] = evaluate_f1(metrics, 'f1_winter')['recall']
        f1['recall_unweighted_autumn'] = evaluate_f1(metrics, 'f1_unweighted_autumn')['recall']

        f1['recall_top_5_cluster'] = evaluate_f1(metrics, 'f1_top_5_cluster')['recall']
        f1['recall_top_5_cluster_unweighted'] = evaluate_f1(metrics, 'f1_top_5_cluster')['recall']

        f1s.append(f1)

        # MAE score
        mae = evaluate_ca(metrics, 'maec')
        mae = mae.groupby('model')['maec'].mean().reset_index()
        mae['maec'] = mae['maec'].astype(float)
        mae['expe'] = expe
        mae['index'] = index
        mae['label'] = label
        mae['maec_winter'] = evaluate_ca(metrics, 'maec_winter')['maec_winter']
        mae['maec_spring'] = evaluate_ca(metrics, 'maec_spring')['maec_spring']
        mae['maec_summer'] = evaluate_ca(metrics, 'maec_summer')['maec_summer']
        mae['maec_autumn'] = evaluate_ca(metrics, 'maec_winter')['maec_winter']
        mae['maec_top_5_cluster'] = evaluate_ca(metrics, 'maec_top_5_cluster')['maec_top_5_cluster']
        mae['maec_top_5_cluster_unweighted'] = evaluate_ca(metrics, 'maec_top_5_cluster_unweighted')['maec_top_5_cluster_unweighted']

        mae['mae'] = evaluate_met(metrics, 'mae')['mae']
        mae['mae_winter'] = evaluate_met(metrics, 'mae_winter')['mae_winter']
        mae['mae_spring'] = evaluate_met(metrics, 'mae_spring')['mae_spring']
        mae['mae_summer'] = evaluate_met(metrics, 'mae_summer')['mae_summer']
        mae['mae_autumn'] = evaluate_met(metrics, 'mae_winter')['mae_winter']
        #mae['mae_top_5_cluster'] = evaluate_met(metrics, 'mae_top_5_cluster')['mae_top_5_cluster']
        #mae['mae_top_5_cluster_unweighted'] = evaluate_met(metrics, 'mae_top_5_cluster_unweighted')['mae_top_5_cluster_unweighted']

        mae['values_per_class'] = values_per_class
        mae['k_days'] = k_days
        mae['scale'] = scale
        mae['nf'] = nf

        maes.append(mae)

        maetop10 = evaluate_ca(metrics, 'maectop10')
        maetop10 = maetop10.groupby('model')['maectop10'].mean().reset_index()
        maetop10['maectop10'] = maetop10['maectop10'].astype(float)
        maetop10['expe'] = expe
        maetop10['index'] = index
        maetop10['label'] = label
        maetop10['values_per_class'] = values_per_class
        maetop10['k_days'] = k_days
        maetop10['scale'] = scale
        maetop10['nf'] = nf
        maetop10s.append(maetop10)

        # Bcas score
        bca = evaluate_ca(metrics, 'bca')
        bca = bca.groupby('model')['bca'].mean().reset_index()
        bca['bca'] = bca['bca'].astype(float)
        bca['expe'] = expe
        bca['label'] = label
        bca['index'] = index
        bca['bca_winter'] = evaluate_ca(metrics, 'bca_winter')['bca_winter']
        bca['bca_spring'] = evaluate_ca(metrics, 'bca_spring')['bca_spring']
        bca['bca_summer'] = evaluate_ca(metrics, 'bca_summer')['bca_summer']
        bca['bca_autumn'] = evaluate_ca(metrics, 'bca_autumn')['bca_autumn']
        bca['bca_top_5_cluster'] = evaluate_ca(metrics, 'bca_top_5_cluster')['bca_top_5_cluster']
        bca['bca_top_5_cluster_unweighted'] = evaluate_ca(metrics, 'bca_top_5_cluster_unweighted')['bca_top_5_cluster_unweighted']
        bca['values_per_class'] = values_per_class
        bca['k_days'] = k_days
        bca['scale'] = scale
        bca['nf'] = nf
        bcas.append(bca)

        # RMSE score
        rmse = evaluate_met(metrics, 'rmse')
        rmse = rmse.groupby('model')['rmse'].mean().reset_index()
        rmse['rmse'] = rmse['rmse'].astype(float)
        rmse['expe'] = expe
        rmse['label'] = label
        rmse['index'] = index
        rmse['rmse_winter'] = evaluate_met(metrics, 'rmse_winter')['rmse_winter']
        rmse['rmse_spring'] = evaluate_met(metrics, 'rmse_spring')['rmse_spring']
        rmse['rmse_summer'] = evaluate_met(metrics, 'rmse_summer')['rmse_summer']
        rmse['rmse_autumn'] = evaluate_met(metrics, 'rmse_autumn')['rmse_autumn']
        rmse['rmse_top_5_cluster'] = evaluate_met(metrics, 'rmse_top_5_cluster')['rmse_top_5_cluster']
        #rmse['rmse_top_5_cluster_unweighted'] = evaluate_met(metrics, 'rmse_top_5_cluster_unweighted')['rmse_top_5_cluster_unweighted']
        rmse['values_per_class'] = values_per_class
        rmse['k_days'] = k_days
        rmse['scale'] = scale
        rmse['nf'] = nf
        rmses.append(rmse)

        # CAL
        # RMSE score
        cal = evaluate_met(metrics, 'cal')
        cal = cal.groupby('model')['cal'].mean().reset_index()
        cal['cal'] = cal['cal'].astype(float)
        cal['expe'] = expe
        cal['label'] = label
        cal['index'] = index
        cal['cal_winter'] = evaluate_met(metrics, 'cal_winter')['cal_winter']
        cal['cal_spring'] = evaluate_met(metrics, 'cal_spring')['cal_spring']
        cal['cal_summer'] = evaluate_met(metrics, 'cal_summer')['cal_summer']
        cal['cal_autumn'] = evaluate_met(metrics, 'cal_autumn')['cal_autumn']
        cal['cal_top_5_cluster'] = evaluate_met(metrics, 'cal_top_5_cluster')['cal_top_5_cluster']
        #rmse['rmse_top_5_cluster_unweighted'] = evaluate_met(metrics, 'rmse_top_5_cluster_unweighted')['rmse_top_5_cluster_unweighted']
        cal['values_per_class'] = values_per_class
        cal['k_days'] = k_days
        cal['scale'] = scale
        cal['nf'] = nf
        cals.append(cal)


    f1s = pd.concat(f1s).reset_index(drop=True)
    maes = pd.concat(maes).reset_index(drop=True)
    bcas = pd.concat(bcas).reset_index(drop=True)
    rmses = pd.concat(rmses).reset_index(drop=True)
    maetop10s = pd.concat(maetop10s).reset_index(drop=True)
    cals = pd.concat(cals).reset_index(drop=True)

    f1s.to_csv(dir_output / 'f1s.csv')
    maes.to_csv(dir_output / 'maes.csv')
    bcas.to_csv(dir_output / 'bcas.csv')
    rmses.to_csv(dir_output / 'rmses.csv')
    maetop10.to_csv(dir_output / 'maetop10.csv')
    cals.to_csv(dir_output / 'cals.csv')

    index_colors = np.unique(f1s['index'].values)

    color = {}
    for i in index_colors:
        c = random.choice(list(mcolors.XKCD_COLORS.keys()))
        print(c)
        if c.find('yellow') > -1:
            continue
        color[i] = c

    plot(f1s, 'f1', dir_output, 'f1', color)
    plot(f1s, 'f1_unweighted', dir_output, 'f1_unweighted', color)
    plot(f1s, 'f1_top_5_cluster', dir_output, 'f1_top_5_cluster', color)
    plot(f1s, 'f1_top_5_cluster_unweighted', dir_output, 'f1_top_5_cluster_unweighted', color)

    plot(f1s, 'precision', dir_output, 'precision', color)
    plot(f1s, 'precision_unweighted', dir_output, 'precision_unweighted', color)
    plot(f1s, 'precision_top_5_cluster', dir_output, 'precision_top_5_cluster', color)
    plot(f1s, 'precision_top_5_cluster_unweighted', dir_output, 'precision_top_5_cluster_unweighted', color)

    plot(f1s, 'recall', dir_output, 'recall', color)
    plot(f1s, 'recall_unweighted', dir_output, 'recall_unweighted', color)
    plot(f1s, 'recall_top_5_cluster', dir_output, 'recall_top_5_cluster', color)
    plot(f1s, 'recall_top_5_cluster_unweighted', dir_output, 'recall_top_5_cluster_unweighted', color)

    plot(cals, 'cal', dir_output, 'cal', color)
    plot(maes, 'maec', dir_output, 'maec', color)
    plot(maes, 'mae', dir_output, 'mae', color)
    plot(bcas, 'bca', dir_output, 'bca', color)
    plot(rmses, 'rmse', dir_output, 'rmse', color)
    plot(maetop10s, 'maectop10', dir_output, 'maectop10', color)

    plot(maes, 'maec_top_5_cluster', dir_output, 'maec_top_5_cluster', color)
    #plot(maes, 'mae_top_5_cluster', dir_output, 'mae_top_5_cluster', color)
    plot(bcas, 'bca_top_5_cluster', dir_output, 'bca_top_5_cluster', color)
    plot(rmses, 'rmse_top_5_cluster', dir_output, 'rmse_top_5_cluster', color)
    plot(maetop10s, 'maectop10', dir_output, 'maectop10', color)

    plot(f1s, 'f1_winter', dir_output, 'f1_winter', color)
    plot(f1s, 'f1_unweighted_winter', dir_output, 'f1_unweighted_winter', color)

    plot(f1s, 'precision_winter', dir_output, 'precision_winter', color)
    plot(f1s, 'precision_unweighted_winter', dir_output, 'precision_unweighted_winter', color)

    plot(f1s, 'recall_winter', dir_output, 'recall_winter', color)
    plot(f1s, 'recall_unweighted_winter', dir_output, 'recall_unweighted_winter', color)

    plot(cals, 'cal_winter', dir_output, 'cal_winter', color)
    plot(maes, 'maec_winter', dir_output, 'maec_winter', color)
    plot(maes, 'mae_winter', dir_output, 'mae_winter', color)
    plot(bcas, 'bca_winter', dir_output, 'bca_winter', color)
    plot(rmses, 'rmse_winter', dir_output, 'rmse_winter', color)

    plot(f1s, 'f1_summer', dir_output, 'f1_summer', color)
    plot(f1s, 'f1_unweighted_summer', dir_output, 'f1_unweighted_summer', color)

    plot(f1s, 'precision_summer', dir_output, 'precision_summer', color)
    plot(f1s, 'precision_unweighted_summer', dir_output, 'precision_unweighted_summer', color)

    plot(f1s, 'recall_summer', dir_output, 'recall_summer', color)
    plot(f1s, 'recall_unweighted_summer', dir_output, 'recall_unweighted_summer', color)

    plot(cals, 'cal_summer', dir_output, 'cal_summer', color)
    plot(maes, 'maec_summer', dir_output, 'maec_summer', color)
    plot(maes, 'mae_summer', dir_output, 'mae_summer', color)
    plot(bcas, 'bca_summer', dir_output, 'bca_summer', color)
    plot(rmses, 'rmse_summer', dir_output, 'rmse_summer', color)

    plot(cals, 'cal_spring', dir_output, 'cal_spring', color)
    plot(f1s, 'f1_spring', dir_output, 'f1_spring', color)
    plot(f1s, 'f1_unweighted_spring', dir_output, 'f1_unweighted_spring', color)

    plot(f1s, 'precision_spring', dir_output, 'precision_spring', color)
    plot(f1s, 'precision_unweighted_spring', dir_output, 'precision_unweighted_spring', color)

    plot(f1s, 'recall_spring', dir_output, 'recall_spring', color)
    plot(f1s, 'recall_unweighted_spring', dir_output, 'recall_unweighted_spring', color)

    plot(maes, 'maec_spring', dir_output, 'maec_spring', color)
    plot(maes, 'mae_spring', dir_output, 'mae_spring', color)
    plot(bcas, 'bca_spring', dir_output, 'bca_spring', color)
    plot(rmses, 'rmse_spring', dir_output, 'rmse_spring', color)

    plot(cals, 'cal_autumn', dir_output, 'cal_autumn', color)
    plot(f1s, 'f1_autumn', dir_output, 'f1_autumn', color)
    plot(f1s, 'f1_unweighted_autumn', dir_output, 'f1_unweighted_autumn', color)

    plot(f1s, 'precision_autumn', dir_output, 'precision_autumn', color)
    plot(f1s, 'precision_unweighted_autumn', dir_output, 'precision_unweighted_autumn', color)

    plot(f1s, 'recall_autumn', dir_output, 'recall_autumn', color)
    plot(f1s, 'recall_unweighted_autumn', dir_output, 'recall_unweighted_autumn', color)

    plot(maes, 'maec_autumn', dir_output, 'maec_autumn', color)
    plot(maes, 'mae_autumn', dir_output, 'mae_autumn', color)
    plot(bcas, 'bca_autumn', dir_output, 'bca_autumn', color)
    plot(rmses, 'rmse_autumn', dir_output, 'rmse_autumn', color)

    ################ Plot by scale ####################

    index = 'scale'

    index_colors = np.unique(f1s['model'].values)

    color = {}
    for i in index_colors:
        c = random.choice(list(mcolors.XKCD_COLORS.keys()))
        print(c)
        if c.find('yellow') > -1:
            continue
        color[i] = c

    plot_by_index(f1s, 'f1', dir_output, 'f1', color, index)
    plot_by_index(f1s, 'f1_unweighted', dir_output, 'f1_unweighted', color, index)
    plot_by_index(f1s, 'f1_top_5_cluster', dir_output, 'f1_top_5_cluster', color, index)
    plot_by_index(f1s, 'f1_top_5_cluster_unweighted', dir_output, 'f1_top_5_cluster_unweighted', color, index)

    plot_by_index(f1s, 'precision', dir_output, 'precision', color, index)
    plot_by_index(f1s, 'precision_unweighted', dir_output, 'precision_unweighted', color, index)
    plot_by_index(f1s, 'precision_top_5_cluster', dir_output, 'precision_top_5_cluster', color, index)
    plot_by_index(f1s, 'precision_top_5_cluster_unweighted', dir_output, 'precision_top_5_cluster_unweighted', color, index)

    plot_by_index(f1s, 'recall', dir_output, 'recall', color, index)
    plot_by_index(f1s, 'recall_unweighted', dir_output, 'recall_unweighted', color, index)
    plot_by_index(f1s, 'recall_top_5_cluster', dir_output, 'recall_top_5_cluster', color, index)
    plot_by_index(f1s, 'recall_top_5_cluster_unweighted', dir_output, 'recall_top_5_cluster_unweighted', color, index)

    plot_by_index(cals, 'cal', dir_output, 'cal', color, index)
    plot_by_index(maes, 'maec', dir_output, 'maec', color, index)
    plot_by_index(maes, 'mae', dir_output, 'mae', color, index)
    plot_by_index(bcas, 'bca', dir_output, 'bca', color, index)
    plot_by_index(rmses, 'rmse', dir_output, 'rmse', color, index)
    plot_by_index(maetop10s, 'maectop10', dir_output, 'maectop10', color, index)

    plot_by_index(maes, 'maec_top_5_cluster', dir_output, 'maec_top_5_cluster', color, index)
    #plot_by_index(maes, 'mae_top_5_cluster', dir_output, 'mae_top_5_cluster', color, index)
    plot_by_index(bcas, 'bca_top_5_cluster', dir_output, 'bca_top_5_cluster', color, index)
    plot_by_index(rmses, 'rmse_top_5_cluster', dir_output, 'rmse_top_5_cluster', color, index)
    plot_by_index(maetop10s, 'maectop10', dir_output, 'maectop10', color, index)

    plot_by_index(f1s, 'f1_winter', dir_output, 'f1_winter', color, index)
    plot_by_index(f1s, 'f1_unweighted_winter', dir_output, 'f1_unweighted_winter', color, index)

    plot_by_index(f1s, 'precision_winter', dir_output, 'precision_winter', color, index)
    plot_by_index(f1s, 'precision_unweighted_winter', dir_output, 'precision_unweighted_winter', color, index)

    plot_by_index(f1s, 'recall_winter', dir_output, 'recall_winter', color, index)
    plot_by_index(f1s, 'recall_unweighted_winter', dir_output, 'recall_unweighted_winter', color, index)

    plot_by_index(cals, 'cal_winter', dir_output, 'cal_winter', color, index)
    plot_by_index(maes, 'maec_winter', dir_output, 'maec_winter', color, index)
    plot_by_index(maes, 'mae_winter', dir_output, 'mae_winter', color, index)
    plot_by_index(bcas, 'bca_winter', dir_output, 'bca_winter', color, index)
    plot_by_index(rmses, 'rmse_winter', dir_output, 'rmse_winter', color, index)

    plot_by_index(f1s, 'f1_summer', dir_output, 'f1_summer', color, index)
    plot_by_index(f1s, 'f1_unweighted_summer', dir_output, 'f1_unweighted_summer', color, index)

    plot_by_index(f1s, 'precision_summer', dir_output, 'precision_summer', color, index)
    plot_by_index(f1s, 'precision_unweighted_summer', dir_output, 'precision_unweighted_summer', color, index)

    plot_by_index(f1s, 'recall_summer', dir_output, 'recall_summer', color, index)
    plot_by_index(f1s, 'recall_unweighted_summer', dir_output, 'recall_unweighted_summer', color, index)

    plot_by_index(cals, 'cal_summer', dir_output, 'cal_summer', color, index)
    plot_by_index(maes, 'maec_summer', dir_output, 'maec_summer', color, index)
    plot_by_index(maes, 'mae_summer', dir_output, 'mae_summer', color, index)
    plot_by_index(bcas, 'bca_summer', dir_output, 'bca_summer', color, index)
    plot_by_index(rmses, 'rmse_summer', dir_output, 'rmse_summer', color, index)

    plot_by_index(cals, 'cal_spring', dir_output, 'cal_spring', color, index)
    plot_by_index(f1s, 'f1_spring', dir_output, 'f1_spring', color, index)
    plot_by_index(f1s, 'f1_unweighted_spring', dir_output, 'f1_unweighted_spring', color, index)

    plot_by_index(f1s, 'precision_spring', dir_output, 'precision_spring', color, index)
    plot_by_index(f1s, 'precision_unweighted_spring', dir_output, 'precision_unweighted_spring', color, index)

    plot_by_index(f1s, 'recall_spring', dir_output, 'recall_spring', color, index)
    plot_by_index(f1s, 'recall_unweighted_spring', dir_output, 'recall_unweighted_spring', color, index)

    plot_by_index(maes, 'maec_spring', dir_output, 'maec_spring', color, index)
    plot_by_index(maes, 'mae_spring', dir_output, 'mae_spring', color, index)
    plot_by_index(bcas, 'bca_spring', dir_output, 'bca_spring', color, index)
    plot_by_index(rmses, 'rmse_spring', dir_output, 'rmse_spring', color, index)

    plot_by_index(cals, 'cal_autumn', dir_output, 'cal_autumn', color, index)
    plot_by_index(f1s, 'f1_autumn', dir_output, 'f1_autumn', color, index)
    plot_by_index(f1s, 'f1_unweighted_autumn', dir_output, 'f1_unweighted_autumn', color, index)

    plot_by_index(f1s, 'precision_autumn', dir_output, 'precision_autumn', color, index)
    plot_by_index(f1s, 'precision_unweighted_autumn', dir_output, 'precision_unweighted_autumn', color, index)

    plot_by_index(f1s, 'recall_autumn', dir_output, 'recall_autumn', color, index)
    plot_by_index(f1s, 'recall_unweighted_autumn', dir_output, 'recall_unweighted_autumn', color, index)

    plot_by_index(maes, 'maec_autumn', dir_output, 'maec_autumn', color, index)
    plot_by_index(maes, 'mae_autumn', dir_output, 'mae_autumn', color, index)
    plot_by_index(bcas, 'bca_autumn', dir_output, 'bca_autumn', color, index)
    plot_by_index(rmses, 'rmse_autumn', dir_output, 'rmse_autumn', color, index)

    ################################## FEATURES IMPORTANCE #################################

    index_colors = np.unique(features_name_full)
    color = {}
    for i in index_colors:
        c = random.choice(list(mcolors.XKCD_COLORS.keys()))
        if c.find('yellow') > -1:
            continue
        color[i] = c

    index_colors = np.unique(index)

    if len(features_importance_test) > 0:
        features_importance_test = pd.concat(features_importance_test).reset_index()
        plot_features_importances(features_importance_test, 'features_importance_test', dir_output, color=color)

    if len(features_importance_train) > 0:
        features_importance_train = pd.concat(features_importance_train).reset_index()
        plot_features_importances(features_importance_train, 'features_importance_train', dir_output, color=color)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Inference',
        description='Inference prediction',
    )
    parser.add_argument('-n', '--name', type=str, help='Test name')
    parser.add_argument('-spec', '--spec', type=str, help='Test name')
    parser.add_argument('-o', '--output', type=str, help='Output directory')
    parser.add_argument('-i', '--input', type=str, help='Input directory')
    parser.add_argument('-s', '--sinister', type=str, help='Sinister')
    parser.add_argument('-r', '--resolution', type=str, help='resolution')

    args = parser.parse_args()

    test_name = args.name
    sinister = args.sinister
    resolution = args.resolution
    datatset_name= args.spec
    dir_output = Path(args.output + '/' + datatset_name+ '/' + test_name)
    dir_input = Path(args.input)

    # features
    experiments_features = [('exp1', 0, 'All', 'full_0_10_100_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp1', 1, 'noForet', 'full_0_10_100_noForet_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp1', 2, 'noHistorical', 'full_0_10_100_noHistorical_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp1', 3, 'noMeteo', 'full_0_10_100_noMeteo_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp1', 4, 'noIndex', 'full_0_10_100_noIndex_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp1', 5, 'noSentinel', 'full_0_10_100_noSentinel_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp1', 6, 'noLandcover', 'full_0_10_100_noLandcover_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp1', 7, 'noCalendar', 'full_0_10_100_noCalendar_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp1', 8, 'noGeo', 'full_0_10_100_noGeo_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp1', 9, 'noOSMNX', 'full_0_10_100_noOSMNX_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp1', 10, 'noAutoReg', 'full_0_10_100_noAutoReg_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp1', 11, 'noAutoBin', 'full_0_10_100_noAutoBin_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp1', 12, 'noPop', 'full_0_10_100_noPop_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp1', 13, 'noNappes', 'full_0_10_100_noNappes_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp1', 14, 'noAir', 'full_0_10_100_noAir_10_z-score_Catboost_'+test_name+'_tree'),
                   #('exp1', 15, 'noEle', 'full_0_10_100_noEle_10_z-score_Catboost_'+test_name+'_tree'),
                   ]

    # Ks
    experiments_ks = [('exp_ks', 0, '0', 'full_0_10_100_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp_ks', 1, '1', 'full_1_10_100_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp_ks', 2, '2', 'full_2_10_100_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp_ks', 3, '3', 'full_3_10_100_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp_ks', 4, '4', 'full_4_10_100_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp_ks', 5, '5', 'full_5_10_100_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp_ks', 6, '6', 'full_6_10_100_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp_ks', 7, '7', 'full_7_10_100_10_z-score_Catboost_'+test_name+'_tree'),
                   ]

    # Scale
    experiments_scale = [('ecai', 10, '10', 'full_0_10_100_10_z-score_Catboost_'+test_name+'_tree'),
                        ('ecai', 2, '2', '20_0_2_100_2_z-score_Catboost_'+test_name+'_tree'),
                        ('ecai', 1, '1', '20_0_1_100_1_z-score_Catboost_'+test_name+'_tree'),
                        ('try_0_scale', 0, '0', '20_0_0_20_0_z-score_Catboost_'+test_name+'_tree'),
                        ]

    # Inference
    experiments_inference = [
                             #(spec, 1, 'full_0_7_10_100', 'full_0_7_10_100_z-score_Catboost_'+test_name+'_dl'),
                             #(spec, 2, 'full_0_7_30_100', 'full_0_7_30_100_z-score_Catboost_'+test_name+'_dl'),
                             #(spec, 3, 'full_0_30_100', 'full_0_30_100_z-score_Catboost_'+test_name+'_tree'),
                             #(spec, 4, 'full_0_10_100', 'full_0_10_100_z-score_Catboost_'+test_name+'_tree'),
                             #(spec, 4, 'full_0_35_100', 'full_0_35_100_z-score_Catboost_'+test_name+'_tree'),
                             #(spec, 6, 'full_0_40_100', 'full_0_40_100_z-score_Catboost_'+test_name+'_tree'),
                             #(spec, 7, 'full_7_60_700', 'full_7_60_700_z-score_Catboost_'+test_name+'_tree'),
                             (spec, 8, '100_7_30_300_kmeans', '100_7_30_300_kmeans_z-score_Catboost_'+test_name+'_tree'),
                             (spec, 9, '200_7_30_300_kmeans', '200_7_30_300_kmeans_z-score_Catboost_'+test_name+'_tree'),
                             (spec, 13, '50_7_30_300_kmeans', '50_7_30_300_kmeans_z-score_Catboost_'+test_name+'_tree'),
                             (spec, 10, '500_7_30_300_kmeans', '500_7_30_300_kmeans_z-score_Catboost_'+test_name+'_tree'),
                             (spec, 12, 'binary_7_30_300', 'binary_7_30_300_z-score_Catboost_'+test_name+'_tree'),
                             #(spec, 9, 'full_7_30_300', 'full_7_30_300_z-score_Catboost_'+test_name+'_tree'),
                             #(spec, 9, 'full_7_60_300', 'full_7_60_300_z-score_Catboost_'+test_name+'_tree'),
                             #(spec, 9, 'full_7_60_300', 'full_7_60_300_none_Catboost_'+test_name+'_bp'),
                             #(spec, 8, 'full_7_30_300', 'full_7_30_300_none_Catboost_'+test_name+'_bp'),
                             (spec, 11, 'full_7_30_300_kmeans', 'full_7_30_300_kmeans_z-score_Catboost_'+test_name+'_tree'),
                             #(#spec, 12, 'full_7_60_300_kmeans', 'full_7_60_300_kmeans_z-score_Catboost_'+test_name+'_tree'),
                             #(spec, 4, 'full_0_10_100_fusion', 'full_0_10_100_fusion_10_z-score_Catboost_'+test_name+'_fusion'),
                            ]
    
    # ECAI
    experiments_ecai = [ ('Ain', 0, 'full_0_10_100', 'full_0_10_100_10_z-score_Catboost_'+test_name+'_tree'),
                         ('final', 1, 'full_0_10_100', 'full_0_10_100_10_z-score_Catboost_'+test_name+'_tree'),
                        ]

    # exp
    experiments_dl = [('exp1', 0, 'Tree based', 'full_0_10_100_10_z-score_Catboost_'+test_name+'_tree'),
                             ('exp1', 1, 'GNN', 'full_7_10_100_10_z-score_Catboost_'+test_name+'_dl')
                            ]

    check_and_create_path(dir_output)
    load_and_evaluate(experiments=experiments_inference, test_name=test_name, dir_output=dir_output, sinister=sinister, dir_input=dir_input)