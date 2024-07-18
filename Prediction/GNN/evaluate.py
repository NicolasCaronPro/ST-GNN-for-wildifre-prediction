from tools import *
import matplotlib.pyplot as plt
import argparse
import matplotlib.colors as mcolors

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

    fig = plt.figure(figsize=(15, 10))
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

def load_and_evaluate(experiments, test_name, dir_output, sinister):

    f1s, maes, bcas, rmses, maetop10s, crs = [], [], [], [], [], []
    (_, _, base_label, _) = experiments[0]

    for elt in experiments:
        (expe, index, label, prefix) = elt
        dir_test = Path(expe + '/' + sinister + '/' + resolution + '/test/' + test_name + '/' + label + '/')
        
        name = 'metrics_'+prefix+'.pkl'
        metrics = read_object(name, dir_test)

        # F1 score
        f1 = evaluate_f1(metrics, 'f1')
        f1['f1'] = f1['f1'].astype(float)
        f1['precision'] = evaluate_f1(metrics, 'f1')['precision'].astype(float)
        f1['recall'] = evaluate_f1(metrics, 'f1')['recall'].astype(float)
        f1['expe'] = expe
        f1['index'] = index
        f1['label'] = label

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
        mae = evaluate_ca(metrics, 'meac')
        mae = mae.groupby('model')['meac'].mean().reset_index()
        mae['meac'] = mae['meac'].astype(float)
        mae['expe'] = expe
        mae['index'] = index
        mae['label'] = label
        mae['meac_winter'] = evaluate_ca(metrics, 'meac_winter')['meac_winter']
        mae['meac_spring'] = evaluate_ca(metrics, 'meac_spring')['meac_spring']
        mae['meac_summer'] = evaluate_ca(metrics, 'meac_summer')['meac_summer']
        mae['meac_autumn'] = evaluate_ca(metrics, 'meac_winter')['meac_winter']
        mae['meac_top_5_cluster'] = evaluate_ca(metrics, 'meac_top_5_cluster')['meac_top_5_cluster']
        mae['meac_top_5_cluster_unweighted'] = evaluate_ca(metrics, 'meac_top_5_cluster_unweighted')['meac_top_5_cluster_unweighted']

        maes.append(mae)

        maetop10 = evaluate_ca(metrics, 'meactop10')
        maetop10 = maetop10.groupby('model')['meactop10'].mean().reset_index()
        maetop10['meactop10'] = maetop10['meactop10'].astype(float)
        maetop10['expe'] = expe
        maetop10['index'] = index
        maetop10['label'] = label
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
        
        rmses.append(rmse)

    f1s = pd.concat(f1s).reset_index(drop=True)
    maes = pd.concat(maes).reset_index(drop=True)
    bcas = pd.concat(bcas).reset_index(drop=True)
    rmses = pd.concat(rmses).reset_index(drop=True)
    maetop10s = pd.concat(maetop10s).reset_index(drop=True)
    #crs = pd.concat(crs).reset_index(drop=True)

    f1s.to_csv(dir_output / 'f1s.csv')
    maes.to_csv(dir_output / 'maes.csv')
    bcas.to_csv(dir_output / 'bcas.csv')
    rmses.to_csv(dir_output / 'rmses.csv')
    maetop10.to_csv(dir_output / 'maetop10.csv')
    #crs.to_csv(dir_output / 'crs.csv')

    index_colors = np.unique(f1s['index'].values)

    color = {}
    for i in index_colors:
        c = random.choice(list(mcolors.XKCD_COLORS.keys()))
        print(c)
        if c.find('yellow') > -1:
            continue
        color[i] = c

    index_colors = np.unique(index)

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

    plot(maes, 'meac', dir_output, 'meac', color)
    plot(bcas, 'bca', dir_output, 'bca', color)
    plot(rmses, 'rmse', dir_output, 'rmse', color)
    plot(maetop10s, 'meactop10', dir_output, 'maectop10', color)

    plot(maes, 'meac_top_5_cluster', dir_output, 'meac_top_5_cluster', color)
    plot(bcas, 'bca_top_5_cluster', dir_output, 'bca_top_5_cluster', color)
    plot(rmses, 'rmse_top_5_cluster', dir_output, 'rmse_top_5_cluster', color)
    plot(maetop10s, 'meactop10', dir_output, 'maectop10', color)

    plot(f1s, 'f1_winter', dir_output, 'f1_winter', color)
    plot(f1s, 'f1_unweighted_winter', dir_output, 'f1_unweighted_winter', color)

    plot(f1s, 'precision_winter', dir_output, 'precision_winter', color)
    plot(f1s, 'precision_unweighted_winter', dir_output, 'precision_unweighted_winter', color)

    plot(f1s, 'recall_winter', dir_output, 'recall_winter', color)
    plot(f1s, 'recall_unweighted_winter', dir_output, 'recall_unweighted_winter', color)

    plot(maes, 'meac_winter', dir_output, 'meac_winter', color)
    plot(bcas, 'bca_winter', dir_output, 'bca_winter', color)
    plot(rmses, 'rmse_winter', dir_output, 'rmse_winter', color)

    plot(f1s, 'f1_summer', dir_output, 'f1_summer', color)
    plot(f1s, 'f1_unweighted_summer', dir_output, 'f1_unweighted_summer', color)

    plot(f1s, 'precision_summer', dir_output, 'precision_summer', color)
    plot(f1s, 'precision_unweighted_summer', dir_output, 'precision_unweighted_summer', color)

    plot(f1s, 'recall_summer', dir_output, 'recall_summer', color)
    plot(f1s, 'recall_unweighted_summer', dir_output, 'recall_unweighted_summer', color)

    plot(maes, 'meac_summer', dir_output, 'meac_summer', color)
    plot(bcas, 'bca_summer', dir_output, 'bca_summer', color)
    plot(rmses, 'rmse_summer', dir_output, 'rmse_summer', color)

    plot(f1s, 'f1_spring', dir_output, 'f1_spring', color)
    plot(f1s, 'f1_unweighted_spring', dir_output, 'f1_unweighted_spring', color)

    plot(f1s, 'precision_spring', dir_output, 'precision_spring', color)
    plot(f1s, 'precision_unweighted_spring', dir_output, 'precision_unweighted_spring', color)

    plot(f1s, 'recall_spring', dir_output, 'recall_spring', color)
    plot(f1s, 'recall_unweighted_spring', dir_output, 'recall_unweighted_spring', color)

    plot(maes, 'meac_spring', dir_output, 'meac_spring', color)
    plot(bcas, 'bca_spring', dir_output, 'bca_spring', color)
    plot(rmses, 'rmse_spring', dir_output, 'rmse_spring', color)

    plot(f1s, 'f1_autumn', dir_output, 'f1_autumn', color)
    plot(f1s, 'f1_unweighted_autumn', dir_output, 'f1_unweighted_autumn', color)

    plot(f1s, 'precision_autumn', dir_output, 'precision_autumn', color)
    plot(f1s, 'precision_unweighted_autumn', dir_output, 'precision_unweighted_autumn', color)

    plot(f1s, 'recall_autumn', dir_output, 'recall_autumn', color)
    plot(f1s, 'recall_unweighted_autumn', dir_output, 'recall_unweighted_autumn', color)

    plot(maes, 'meac_autumn', dir_output, 'meac_autumn', color)
    plot(bcas, 'bca_autumn', dir_output, 'bca_autumn', color)
    plot(rmses, 'rmse_autumn', dir_output, 'rmse_autumn', color)

    """plot_variation(f1s, base_label, 'f1', dir_output, 'f1_variation', color)
    plot_variation(f12s, base_label, 'f1no_weighted', dir_output, 'f1no_weighted_variation', color)
    plot_variation(maes, base_label, 'meac', dir_output, 'meac_variation', color)
    plot_variation(bcas, base_label, 'bca', dir_output, 'bca_variation', color)
    plot_variation(rmses, base_label, 'rmse', dir_output, 'rmse_variation', color)"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Inference',
        description='Inference prediction',
    )
    parser.add_argument('-n', '--name', type=str, help='Test name')
    parser.add_argument('-o', '--output', type=str, help='Output directory')
    parser.add_argument('-s', '--sinister', type=str, help='Sinister')
    parser.add_argument('-r', '--resolution', type=str, help='resolution')

    args = parser.parse_args()

    test_name = args.name
    sinister = args.sinister
    resolution = args.resolution
    dir_output = Path(args.output + '/' + test_name)

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
                             ('final', 1, 'full_0_30_100', 'full_0_30_100_30_z-score_Catboost_'+test_name+'_tree'),
                             ('final', 2, 'full_0_10_100', 'full_0_10_100_10_z-score_Catboost_'+test_name+'_tree'),
                             ('final', 3, 'full_0_10_100_fusion', 'full_0_10_100_fusion_10_z-score_Catboost_'+test_name+'_fusion'),
                             ('final', 0, 'full_0_10_100_7_mean', 'full_0_10_100_7_mean_10_z-score_Catboost_'+test_name+'_tree'),
                             #('final', 2, 'full_0_25_100_TrainOnYvelines', 'full_0_25_100_TrainOnYvelines_25_z-score_Catboost_'+test_name+'_tree'),
                             #('final', 1, 'full_0_10_100_3_mean', 'full_0_10_100_3_mean_10_z-score_Catboost_'+test_name+'_tree'),
                             #('final', 1, 'full_0_10_100_7_mean', 'full_0_10_100_7_mean_10_MinMax_Catboost_'+test_name+'_tree'),
                             #('final', 3, 'full_0_10_100_3_mean', 'full_0_10_100_3_mean_10_z-score_Catboost_'+test_name+'_tree'),
                             #('final', 4, 'full_0_10_100_3_mean', 'full_0_10_100_3_mean_10_MinMax_Catboost_'+test_name+'_tree'),
                             #('final', 2, 'full_0_15_100_7_mean', 'full_0_15_100_7_mean_15_MinMax_Catboost_'+test_name+'_tree'),
                             #('final', 0, 'full_0_10_100', 'full_0_10_100_10_none_Catboost_'+test_name+'_tree'),
                             #('final', 2, 'full_0_15_100', 'full_0_15_100_15_z-score_Catboost_'+test_name+'_tree'),
                             #('final', 3, '1000_0_15_100', '1000_0_15_100_15_z-score_Catboost_'+test_name+'_tree'),
                             #('final', 4, '100_0_15_100', '100_0_15_100_15_z-score_Catboost_'+test_name+'_tree'),
                             #('final', 5, '1000_7_15_100', '1000_7_15_100_15_z-score_Catboost_'+test_name+'_tree'),
                             #('final', 6, 'full_0_20_100', 'full_0_20_100_20_z-score_Catboost_'+test_name+'_tree'),
                             #('final', 7, 'full_0_35_100', 'full_0_35_100_35_z-score_Catboost_'+test_name+'_tree'),
                             #('final', 3, 'full_0_10_3_Simple', 'full_0_10_3_Simple_10_z-score_Catboost_'+test_name+'_tree'),
                             #('final', 1, 'full_0_10_100_noHistorical', 'full_0_10_100_noHistorical_10_z-score_Catboost_'+test_name+'_tree'),
                             #('final', 2, '200_0_10_100', '200_0_10_100_10_z-score_Catboost_'+test_name+'_tree'),
                             #('final', 3, '100_0_5_100', '100_0_5_100_5_z-score_Catboost_'+test_name+'_tree'),
                             #('final', 1, 'full_0_10_100_pca', 'full_0_10_100_pca_10_z-score_Catboost_'+test_name+'_tree'),
                             #('final', 2, 'full_0_10_100_pca_kmeans_10', 'full_0_10_100_pca_kmeans_10_10_z-score_Catboost_'+test_name+'_tree'),
                             #('final', 3, 'full_0_10_100_kmeans_10', 'full_0_10_100_kmeans_10_10_z-score_Catboost_'+test_name+'_tree'),
                             #('final', 4, '500_0_10_100_TempFetForKMEANS', '500_0_10_100_TempFetForKMEANS_10_z-score_Catboost_'+test_name+'_tree'),
                             #('final', 4, 'full_7_10_100_10', 'full_7_10_100_10_z-score_Catboost_'+test_name+'_dl'),
                            ]

    # ECAI
    experiments_ecai = [('ecai', 0, 'ecai', 'full_0_10_100_10_z-score_Catboost_'+test_name+'_tree'),
                        ('inference', 1, 'inference', 'full_0_10_100_10_z-score_Catboost_'+test_name+'_tree'),
                        ('final', 2, 'final', 'full_7_10_73_10_z-score_Catboost_'+test_name+'_tree'),
                        ]

    # exp
    experiments_dl = [('exp1', 0, 'Tree based', 'full_0_10_100_10_z-score_Catboost_'+test_name+'_tree'),
                             ('exp1', 1, 'GNN', 'full_7_10_100_10_z-score_Catboost_'+test_name+'_dl')
                            ]

    check_and_create_path(dir_output)
    load_and_evaluate(experiments=experiments_inference, test_name=test_name, dir_output=dir_output, sinister=sinister)