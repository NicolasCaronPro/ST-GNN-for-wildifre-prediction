from tools import *
import matplotlib.pyplot as plt
import argparse

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

def plot(df, met, dir_output, out_name):
    
    # Les données du DataFrame
    index = df['index'].values
    label = df['label'].values
    model = df.model.values
    mets = df[met].values

    # Création du graphique
    fig = plt.figure(figsize=(15, 5))
    scatter = plt.scatter(mets, model, c=index, label=label, cmap='jet', alpha=0.8)
    plt.xlabel('MAE')
    plt.ylabel('Model')
    plt.xlim(-0.1,1.1)
    plt.grid(True)
    fig.colorbar(scatter, label='Index')
    plt.legend()

    out_name += '.png'
    plt.savefig(dir_output / out_name)

def plot_variation(df, base_label, met, dir_output, out_name):

    base_df = df[df['index'] == base_label]

    # Les données du DataFrame
    base_mets = base_df[met].values

    index = df['index'].values
    label = df['label'].values
    model = df.model.values
    mets = base_mets - df[met].values

    # Création du graphique
    fig = plt.figure(figsize=(15, 5))
    scatter = plt.scatter(mets, model, c=index, label=label, cmap='jet', alpha=0.8)
    plt.xlabel('MAE')
    plt.ylabel('Model')
    plt.xlim(-0.1,1.1)
    plt.grid(True)
    fig.colorbar(scatter, label='Index')
    plt.legend()

    out_name += '.png'
    plt.savefig(dir_output / out_name)

def load_and_evaluate(experiments, test_name, dir_output, sinister):

    f1s, f12s, maes, bcas, rmses, = [], [], [], [], []

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

    f1s = pd.concat(f1s).reset_index(drop=True)
    maes = pd.concat(maes).reset_index(drop=True)
    bcas = pd.concat(bcas).reset_index(drop=True)
    f12s = pd.concat(f12s).reset_index(drop=True)
    rmses = pd.concat(rmses).reset_index(drop=True)

    f1s.to_csv(dir_output / 'f1s.csv')
    maes.to_csv(dir_output / 'maes.csv')
    bcas.to_csv(dir_output / 'bcas.csv')
    f12s.to_csv(dir_output / 'f12s.csv')
    rmses.to_csv(dir_output / 'rmses.csv')

    plot(f1s, 'f1', dir_output, 'f1')
    plot(f12s, 'f1no_weighted', dir_output, 'f1no_weighted')
    plot(maes, 'meac', dir_output, 'meac')
    plot(bcas, 'bca', dir_output, 'bca')
    plot(rmses, 'rmse', dir_output, 'rmse')
    
    plot_variation(f1s, base_label, 'f1_variation', dir_output, 'f1_variation')
    plot_variation(f12s, base_label, 'f1no_weighted_variation', dir_output, 'f1no_weighted_variation')
    plot_variation(maes, base_label, 'meac_variation', dir_output, 'meac_variation')
    plot_variation(bcas, base_label, 'bca_variation', dir_output, 'bca_variation')
    plot_variation(rmses, base_label, 'rmse_variation', dir_output, 'rmse_variation')

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
    experiments_features = [('exp_features', 0, 'noTopo', 'full_0_10_noTopo_10_z-score_Catboost_'+test_name),
                   ('exp_features', 1, 'noMeteo', 'full_0_10_noMeteo_10_z-score_Catboost_'+test_name),
                   ('exp_features', 2, 'noHistorical', 'full_0_10_noHistorical_10_z-score_Catboost_'+test_name),
                   #('exp_features', 3, '100_7_10_noIndex_10_z-score_Catboost_2023')
                   ]
    
    # Ks
    experiments_ks = [('exp_ks', 0, '0', 'full_0_10_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp_ks', 1, '1', 'full_1_10_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp_ks', 2, '2', 'full_2_10_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp_ks', 3, '3', 'full_3_10_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp_ks', 4, '4', 'full_4_10_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp_ks', 5, '5', 'full_5_10_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp_ks', 6, '6', 'full_6_10_10_z-score_Catboost_'+test_name+'_tree'),
                   ('exp_ks', 7, '7', 'full_7_10_10_z-score_Catboost_'+test_name+'_tree'),
                   ]
                   
    # Inference
    experiments_inference = [('inference', 0, '0', 'full_0_10_10_z-score_Catboost_'+test_name+'_tree'),
                            ]
                       
    # exp
    experiments_inference = [('exp1', 0, '0', 'full_0_10_10_z-score_Catboost_'+test_name+'_tree'),
                             ('exp1', 1, '0', 'full_7_10_10_z-score_Catboost_'+test_name+'_dl')
                            ]
    
    check_and_create_path(dir_output)
    load_and_evaluate(experiments=experiments_inference, test_name=test_name, dir_output=dir_output, sinister=sinister)