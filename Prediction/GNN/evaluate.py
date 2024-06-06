from tools import *
import matplotlib.pyplot as plt
import argparse
import matplotlib.colors as mcolors

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
    df = pd.DataFrame(index=np.arange(len(metrics)), columns=['model', 'f1', 'prec', 'rec'])
    i = 0
    for key, value in metrics.items():
        df.loc[i, 'model'] = key
        df.loc[i, 'f1'] = value[met][0]
        df.loc[i, 'prec'] = value[met][1]
        df.loc[i, 'rec'] = value[met][2]
        df.loc[i, 'tresh'] = value[met][3]
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

    fig = plt.figure(figsize=(15, 5))
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

    f1s, f12s, maes, bcas, rmses, maetop10s, crs = [], [], [], [], [], [], []

    (_, _, base_label, _) = experiments[0]

    for elt in experiments:
        (expe, index, label, prefix) = elt
        dir_test = Path(expe + '/' + sinister +  '/test/' + test_name + '/')
        name = 'metrics_'+prefix+'.pkl'
        metrics = read_object(name, dir_test)

        # F1 score
        f1 = evaluate_f1(metrics, 'f1')
        f1['f1'] = f1['f1'].astype(float)
        f1['expe'] = expe
        f1['index'] = index
        f1['label'] = label
        f1s.append(f1)

        # F1 score
        f12 = evaluate_f1(metrics, 'f1no_weighted')
        f12['f1no_weighted'] = f12['f1'].astype(float)
        f12['expe'] = expe
        f12['index'] = index
        f12['label'] = label
        f12s.append(f12)

        # MAE score
        mae = evaluate_ca(metrics, 'meac')
        mae = mae.groupby('model')['meac'].mean().reset_index()
        mae['meac'] = mae['meac'].astype(float)
        mae['expe'] = expe
        mae['index'] = index
        mae['label'] = label
        maes.append(mae)

        # MAE score
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
        bcas.append(bca)

        # RMSE score
        rmse = evaluate_met(metrics, 'rmse')
        rmse = rmse.groupby('model')['rmse'].mean().reset_index()
        rmse['rmse'] = rmse['rmse'].astype(float)
        rmse['expe'] = expe
        rmse['label'] = label
        rmse['index'] = index
        rmses.append(rmse)

        """# CR score
        cr = evaluate_cr(metrics, 'class')
        cr = cr.groupby('model')['class'].mean().reset_index()
        cr['class'] = cr['cr'].astype(float)
        cr['expe'] = expe
        cr['label'] = label
        cr['index'] = index
        crs.append(cr)"""

    f1s = pd.concat(f1s).reset_index(drop=True)
    maes = pd.concat(maes).reset_index(drop=True)
    bcas = pd.concat(bcas).reset_index(drop=True)
    f12s = pd.concat(f12s).reset_index(drop=True)
    rmses = pd.concat(rmses).reset_index(drop=True)
    maetop10s = pd.concat(maetop10s).reset_index(drop=True)
    #crs = pd.concat(crs).reset_index(drop=True)

    f1s.to_csv(dir_output / 'f1s.csv')
    maes.to_csv(dir_output / 'maes.csv')
    bcas.to_csv(dir_output / 'bcas.csv')
    f12s.to_csv(dir_output / 'f12s.csv')
    rmses.to_csv(dir_output / 'rmses.csv')
    maetop10.to_csv(dir_output / 'maetop10.csv')
    #crs.to_csv(dir_output / 'crs.csv')

    index_colors = f1s['index'].values

    color = {}
    for i in index_colors:
        color[i] = random.choice(list(mcolors.XKCD_COLORS.keys()))

    index_colors = np.unique(index)

    plot(f1s, 'f1', dir_output, 'f1', color)
    plot(f12s, 'f1no_weighted', dir_output, 'f1no_weighted', color)
    plot(maes, 'meac', dir_output, 'meac', color)
    plot(bcas, 'bca', dir_output, 'bca', color)
    plot(rmses, 'rmse', dir_output, 'rmse', color)
    plot(maetop10s, 'meactop10', dir_output, 'maectop10', color)
    #plot_cr(cr, 'cr', dir_output, 'cr', color)

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

    args = parser.parse_args()

    test_name = args.name
    sinister = args.sinister
    dir_output = Path(args.output)

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
    experiments_inference = [('inference', 0, '0', 'full_0_10_10_z-score_Catboost_'+test_name+'_tree'),
                            ]

    # ECAI
    experiments_ecai = [('ecai', 0, '0', 'full_0_10_100_10_z-score_Catboost_'+test_name+'_tree'),
                            ]

    # exp
    experiments_dl = [('exp1', 0, 'Tree based', 'full_0_10_100_10_z-score_Catboost_'+test_name+'_tree'),
                             ('exp1', 1, 'GNN', 'full_7_10_100_10_z-score_Catboost_'+test_name+'_dl')
                            ]

    check_and_create_path(dir_output)
    load_and_evaluate(experiments=experiments_scale, test_name=test_name, dir_output=dir_output, sinister=sinister)