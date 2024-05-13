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
    model = df.model.values
    mets = np.round(df[met].values, 2)

    # Création du graphique
    fig = plt.figure(figsize=(10, 5))
    scatter = plt.scatter(mets, model, c=index, cmap='jet', alpha=0.8)
    plt.xlabel('MAE')
    plt.ylabel('Model')
    plt.xlim(-0.1,1.1)
    plt.grid(True)
    fig.colorbar(scatter, label='Index')

    out_name += '.png'
    plt.savefig(dir_output / out_name)

def load_and_evaluate(experiments, test_name, dir_output, sinister):

    f1s, maes, bcas, rmses, = [], [], [], []

    for elt in experiments:
        (expe, index, prefix) = elt
        dir_test = Path(expe + '/' + sinister +  '/test/' + test_name + '/')
        name = 'metrics_'+prefix+'.pkl'
        metrics = read_object(name, dir_test)

        # F1 score
        f1 = evaluate_f1(metrics, 'f1')
        f1['f1'] = f1['f1'].astype(float)
        f1['expe'] = expe
        f1['index'] = index
        f1s.append(f1)

        # MAE score
        mae = evaluate_ca(metrics, 'meac')
        mae['meac'] = mae['meac'].astype(float)
        mae['expe'] = expe
        mae['index'] = index
        maes.append(mae)

        # Bcas score
        bca = evaluate_ca(metrics, 'bca')
        bca['bca'] = bca['bca'].astype(float)
        bca['expe'] = expe
        bca['index'] = index
        bcas.append(bca)

        # RMSE score
        rmse = evaluate_met(metrics, 'rmse')
        rmse['rmse'] = rmse['rmse'].astype(float)
        rmse['expe'] = expe
        rmse['index'] = index
        rmses.append(rmse)

    f1s = pd.concat(f1s).reset_index(drop=True)
    maes = pd.concat(maes).reset_index(drop=True)
    bcas = pd.concat(bcas).reset_index(drop=True)
    rmses = pd.concat(rmses).reset_index(drop=True)

    f1s.to_csv(dir_output / 'f1s.csv')
    maes.to_csv(dir_output / 'maes.csv')
    bcas.to_csv(dir_output / 'bcas.csv')
    rmses.to_csv(dir_output / 'rmses.csv')

    plot(f1s, 'f1', dir_output, 'f1')
    plot(maes, 'meac', dir_output, 'meac')
    plot(bcas, 'bca', dir_output, 'bca')
    plot(rmses, 'rmse', dir_output, 'rmse')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Inference',
        description='Inference prediction',
    )
    parser.add_argument('-n', '--name', type=str, help='Test name')
    parser.add_argument('-o', '--output', type=str, help='Output directory')
    parser.add_argument('-s', '--sinister', type=str, help='Sinister')

    args = parser.parse_args()

    experiments = [('exp_features', 0, 'full_0_10_noTopo_10_z-score_Catboost_2023'),
                   ('exp_features', 1, 'full_0_10_noMeteo_10_z-score_Catboost_2023'),
                   ('exp_features', 2, 'full_0_10_noHistorical_10_z-score_Catboost_2023'),
                   #('exp_features', 3, '100_7_10_noIndex_10_z-score_Catboost_2023')
                   ]
    
    test_name = args.name
    sinister = args.sinister
    dir_output = Path(args.output)
    
    check_and_create_path(dir_output)
    load_and_evaluate(experiments=experiments, test_name=test_name, dir_output=dir_output, sinister=sinister)