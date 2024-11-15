from matplotlib import axis, markers
from sqlalchemy import column
from torch import ge
from GNN.encoding import *
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

def realVspredict(ypred, y, band, dir_output, on, pred_min=None, pred_max=None):
    check_and_create_path(dir_output)
    
    ytrue = y[:, band]

    if band == -3:
        band = -1
    elif band == -1 or band == -2:
        band = 0

    dept = np.unique(y[:, departement_index])
    classes = np.unique(y[:, -3])
    colors = plt.cm.get_cmap('jet', 5)
    
    for d in dept:
        mask = np.argwhere(y[:, departement_index] == d)[:, 0]
        maxi = max(np.nanmax(ypred[mask]), np.nanmax(ytrue[mask]))
        mini = min(np.nanmin(ypred[mask]), np.nanmin(ytrue[mask]))
        ids = np.unique(y[mask, graph_id_index])
        if ids.shape[0] == 1:
            fig, ax = plt.subplots(ids.shape[0], figsize=(15, 5))
            ax.plot(ypred[mask], color='red', label='predict')
            ax.plot(ytrue[mask], color='blue', label='real', alpha=0.5)

            # Ajout des courbes et bandes si pred_min et pred_max ne sont pas None
            if pred_min is not None and pred_max is not None:
                ax.plot(pred_min[mask, band], color='green', linestyle='--', label='pred_min')
                ax.plot(pred_max[mask, band], color='purple', linestyle='--', label='pred_max')
                ax.fill_between(np.arange(len(mask)), pred_min[mask, band], pred_max[mask, band], color='lightcoral', alpha=0.5, label='prediction range')

            ax.set_title(f'{ids[0]}')
            for class_value in classes:
                class_mask = np.argwhere((y[mask, -3] == class_value) & (y[mask,-2] > 0))[:, 0]
                ax.scatter(class_mask, ypred[mask][class_mask], color=colors(class_value / 4), label=f'class {class_value}', alpha=1, marker='x', s=50)
            ax.set_ylim(ymin=mini, ymax=maxi)
        else:
            fig, ax = plt.subplots(ids.shape[0], figsize=(50, 50))
            for i, id in enumerate(ids):
                mask2 = np.argwhere(y[:, graph_id_index] == id)[:, 0]
                unode = np.unique(y[mask2, id_index])
                mask2 = np.argwhere(y[:, id_index] == unode[0])[:, 0]
                ax[i].plot(ypred[mask2], color='red', label='predict')
                ax[i].plot(ytrue[mask2], color='blue', label='real', alpha=0.5)
                
                # Ajout des courbes et bandes si pred_min et pred_max ne sont pas None
                if pred_min is not None and pred_max is not None:
                    ax[i].plot(pred_min[mask2, band], color='green', linestyle='--', label='pred_min')
                    ax[i].plot(pred_max[mask2, band], color='purple', linestyle='--', label='pred_max')
                    ax[i].fill_between(np.arange(len(mask2)), pred_min[mask2, band], pred_max[mask2, band], color='lightcoral', alpha=0.5, label='prediction range')

                ax[i].set_title(f'{id}')
                for class_value in classes:
                    class_mask = np.argwhere((y[mask2, -3] == class_value) & (y[mask2,-2] > 0))[:, 0]
                    #ax[i].scatter(class_mask, ypred[mask2][class_mask], color=colors(class_value / 4), label=f'class {class_value}', alpha=1, linewidths=5, marker='x', s=200)
                    ax[i].scatter(class_mask, ypred[mask2][class_mask], color='black', label=f'Fire', alpha=1, linewidths=5, marker='x', s=200)
                ax[i].set_ylim(ymin=mini, ymax=maxi)
        plt.legend()
        outn = str(d) + '_' + on + '.png'
        if MLFLOW:
            mlflow.log_figure(fig, outn)
        plt.savefig(dir_output / outn)
        plt.close('all')

    uids = np.unique(y[:, graph_id_index])
    if uids.shape[0] < 5:
        return
    ysum = np.empty((uids.shape[0], 2))
    ysum[:, graph_id_index] = uids
    for id in uids:
        ysum[np.argwhere(ysum[:, graph_id_index] == id)[:, 0], 1] = np.sum(y[np.argwhere(y[:, graph_id_index] == id)[:, 0], -2])

    tops = 1  # or any other number
    ind = np.lexsort([ysum[:, 1]])
    ymax = np.flip(ysum[ind, graph_id_index])[:tops]
    _, ax = plt.subplots(1 if tops == 1 else np.unique(ymax).shape[0], figsize=(50, 25))

    # Ensure ax is iterable, even when tops == 1
    if tops == 1:
        ax = [ax]

    for i, idtop in enumerate(ymax):
        mask = np.argwhere(y[:, graph_id_index] == idtop)[:, 0]
        dept = np.unique(y[mask, departement_index])[0]
        ids = np.unique(y[mask, graph_id_index])
        
        ax[i].plot(ypred[mask], color='red', label='predict')
        ax[i].plot(ytrue[mask], color='blue', label='real', alpha=0.5)

        # Add curves and bands if pred_min and pred_max are not None
        if pred_min is not None and pred_max is not None:
            ax[i].plot(pred_min[mask, band], color='green', linestyle='--', label='pred_min')
            ax[i].plot(pred_max[mask, band], color='purple', linestyle='--', label='pred_max')
            ax[i].fill_between(np.arange(len(mask)), pred_min[mask, band], pred_max[mask, band], color='lightcoral', alpha=0.5, label='prediction range')

        for class_value in classes:
            class_mask = np.argwhere((y[mask, -3] == class_value) & (y[mask, -2] > 0))[:, 0]
            ax[i].scatter(class_mask, ypred[mask][class_mask], color=colors(class_value / 4), label=f'class {class_value}', alpha=1, linewidths=5, marker='x', s=200)

        ax[i].set_ylim(ymin=0, ymax=np.nanmax(ytrue[mask]))
        ax[i].set_title(f'{dept}_{idtop}')

    # Show the plot
    plt.tight_layout()
    outn = f'top_{tops}{on}.png'
    
    if MLFLOW:
        mlflow.log_figure(fig, dir_output / outn)

    plt.savefig(dir_output / outn)
    plt.close('all')

def plot_and_save_roc_curve(Y: np.array, ypred: np.array, dir_output: Path, target_name : str, isDept):
    """
    Trace et sauvegarde la courbe ROC dans un répertoire spécifié.

    Paramètres:
    - Y : np.array
        Les vraies étiquettes binaires.
    - ypred : np.array
        Les probabilités prédites pour la classe positive.
    - dir_output : str
        Chemin du répertoire où sauvegarder l'image de la courbe ROC.
    """
    if target_name != 'binary':
        ypred = MinMaxScaler().fit_transform(ypred.reshape(-1,1))
    # Calcul des taux de faux positifs et vrais positifs
    fpr, tpr, _ = roc_curve(Y, ypred)
    roc_auc = auc(fpr, tpr)

    # Tracé de la courbe ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    # Création du répertoire de sortie s'il n'existe pas
    check_and_create_path(dir_output)

    # Chemin de l'image
    if isDept:
        output_path = os.path.join(dir_output, 'roc_curve_departement.png')
    else:
        output_path = os.path.join(dir_output, 'roc_curve.png')

    # Sauvegarde de la figure
    plt.savefig(output_path)
    plt.close()
    print(f"Courbe ROC sauvegardée à l'emplacement : {output_path}")

def plot_and_save_pr_curve(Y: np.array, ypred: np.array, dir_output: Path, target_name: str, isDept):
    """
    Trace et sauvegarde la courbe Precision-Recall (PR) dans un répertoire spécifié.

    Paramètres:
    - Y : np.array
        Les vraies étiquettes binaires.
    - ypred : np.array
        Les probabilités prédites pour la classe positive.
    - dir_output : Path
        Chemin du répertoire où sauvegarder l'image de la courbe PR.
    - target_name : str
        Type de cible. Si non binaire, applique une transformation MinMax à `ypred`.
    """
    # Si la cible n'est pas binaire, applique MinMaxScaler à ypred pour normaliser entre 0 et 1
    if target_name != 'binary':
        ypred = MinMaxScaler().fit_transform(ypred.reshape(-1, 1)).flatten()

    # Calcul des taux de précision et rappel
    precision, recall, _ = precision_recall_curve(Y, ypred)
    pr_auc = auc(recall, precision)

    # Tracé de la courbe Precision-Recall
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    # Création du répertoire de sortie s'il n'existe pas
    check_and_create_path(dir_output)

    # Chemin de l'image
    if isDept:
        output_path = os.path.join(dir_output, 'pr_curve_departement.png')
    else:
        output_path = os.path.join(dir_output, 'pr_curve.png')

    # Sauvegarde de la figure
    plt.savefig(output_path)
    plt.close()
    print(f"Courbe Precision-Recall sauvegardée à l'emplacement : {output_path}")

def sinister_distribution_in_class(ypredclassfull, ytrue, dir_output, outputname):
    check_and_create_path(dir_output)
    udept = np.unique(ytrue[:, departement_index])

    for dept in udept:
        y = ytrue[ytrue[:, departement_index] == dept]
        ypredclass = ypredclassfull[ytrue[:, departement_index] == dept] 
        nbclass = [0,1,2,3,4]

        meansinisterinpred = []
        meansinisteriny = []

        nbsinisterinpred = []
        nbsinisteriny = []

        frequency_ratio_pred = []
        frequency_ratio_y = []

        for cls in nbclass:
            mask = np.argwhere(y[:, -3] == cls)[:, 0]

            if mask.shape[0] == 0:
                nbsinisteriny.append(-1)
                meansinisteriny.append(-1)
                frequency_ratio_y.append(0)
                frequency_ratio_pred.append(0)
                continue

            if np.nansum(y[:, -2]) == 0:
                nbsinisteriny.append(0)
            else:
                nbsinisteriny.append(round(100 * np.nansum(y[mask, -2]) / np.nansum(y[:, -2])))
            meansinisteriny.append(np.nanmean((y[mask, -2] > 0).astype(int)))

            frequency_ratio_y.append(frequency_ratio(y[:, -2], mask))

            mask = np.argwhere(ypredclass == cls)[:, 0]
            if np.nansum(y[:, -2]) == 0:
                nbsinisterinpred.append(0)
            else:
                nbsinisterinpred.append(round(100 * np.nansum(y[mask, -2]) / np.nansum(y[:, -2])))
            meansinisterinpred.append(np.nanmean((y[mask, -2] > 0).astype(int)))

            frequency_ratio_pred.append(frequency_ratio(y[:, -2], mask))

        fig, ax = plt.subplots(5, figsize=(50,30))
        ax[0].plot(nbclass, meansinisteriny, label='GT')
        ax[0].plot(nbclass, meansinisterinpred, label='Prediction')
        ax[0].set_xlabel('Class')
        ax[0].set_ylabel('Mean of sinister')
        ax[0].set_title('Mean per class')
        ax[0].legend()

        ax[1].plot(nbclass, nbsinisteriny, label='GT')
        ax[1].plot(nbclass, nbsinisterinpred, label='Prediction')
        ax[1].set_xlabel('Class')
        ax[1].set_ylabel('Nb of sinister')
        ax[1].set_title('Number per class')
        ax[1].legend()

        ax[2].plot(nbclass, frequency_ratio_y, label='GT')
        ax[2].plot(nbclass, frequency_ratio_pred, label='Prediction')
        ax[2].set_xlabel('Class')
        ax[2].set_ylabel('Frequency ratio')
        ax[2].set_title('Frequency ratio')
        ax[2].legend()

        for c in np.unique(y[:, -3]).astype(int):
            mask = np.argwhere(y[:, -3] == c)[:,0]
            ax[3].scatter(y[mask, -1], y[mask, -2], label=f'{c} : {frequency_ratio_y[c]}')

        ax[3].set_title('Frequency ratio of GT')
        ax[3].set_xlabel('Risk')
        ax[3].set_ylabel('nbsinister')
        ax[3].legend()

        for c in np.unique(y[:, -3]).astype(int):
            mask = np.argwhere(ypredclass == c)[:,0]
            ax[4].scatter(ypredclass[mask], y[mask, -2], label=f'{c} : {frequency_ratio_pred[c]}')

        ax[4].set_title('Frequency ratio of Prediction')
        ax[4].set_xlabel('Risk')
        ax[4].set_ylabel('nbsinister')
        ax[4].legend()

        plt.tight_layout()
        plt.savefig(dir_output / f'{int2name[dept]}_{outputname}.png')
        if MLFLOW:
            mlflow.log_figure(fig, dir_output/f'{int2name[dept]}_{outputname}.png')
        plt.close('all')

def plot_custom_confusion_matrix(y_true, y_pred, labels, dir_output, figsize=(10, 8), title='Confusion Matrix', filename='confusion_matrix', normalize=None):
    """
    Generates a custom confusion matrix with specific colors and annotations, 
    and saves the image to the specified directory.
    
    :param y_true: List of true classes
    :param y_pred: List of predicted classes
    :param labels: List of class labels
    :param dir_output: Output directory to save the image
    :param figsize: Size of the matplotlib figure (width, height)
    :param title: Title of the chart
    :param filename: Name of the image file to save
    :param mlflow_enabled: Boolean to determine if mlflow should log the figure
    """
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    cm = np.append(cm, cm.sum(axis=0).reshape(1, -1), axis=0)
    cm = np.append(cm, cm.sum(axis=1).reshape(-1, 1), axis=1)

    # Configure the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create a heatmap with seaborn
    sns.heatmap(cm, annot=True, fmt='.2f', cmap="Blues", cbar=False,
                xticklabels=labels + ['TOTAL'], yticklabels=labels + ['TOTAL'],
                linewidths=1, linecolor='black', square=True, ax=ax)

    # Personnalisation des couleurs
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            if i == len(cm) - 1 or j == len(cm[0]) - 1:  # Ligne ou colonne des totaux
                plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='magenta', lw=3))
            elif i == j:  # Diagonale principale
                plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='blue', lw=3))

    # Add labels
    ax.set_xlabel('PREDICTION', fontsize=14)
    ax.set_ylabel('REAL', fontsize=14)
    ax.set_title(title, fontsize=16)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    # Ensure the output directory exists
    Path(dir_output).mkdir(parents=True, exist_ok=True)

    # Save the image to the specified directory
    output_path = Path(dir_output) / f'{filename}_{normalize}.png'
    plt.savefig(output_path)
    
    # If mlflow is enabled, log the figure
    if MLFLOW:
        mlflow.log_figure(fig, str(output_path))
        
    plt.close(fig)

def realVspredict2d(ypred : np.array,
                    ytrue : np.array,
                    isBin : np.array,
                    modelName : str,
                    scale : int,
                    dir_output : Path,
                    dir : Path,
                    Geo : gpd.GeoDataFrame,
                    testd_departement : list,
                    graph):

    dir_predictor = root_graph / dir / 'influenceClustering'

    yclass = np.empty(ypred.shape)
    for nameDep in testd_departement:
        mask = np.argwhere(ytrue[:,3] == name2int[nameDep])
        if mask.shape[0] == 0:
            continue
        if not isBin:
            predictor = read_object(nameDep+'Predictor'+str(scale)+'.pkl', dir_predictor)
        else:
            predictor = read_object(nameDep+'Predictor'+modelName+str(scale)+'.pkl', dir_predictor)
        
        classpred = predictor.predict(ypred[mask].astype(float))

        yclass[mask] = order_class(predictor, classpred).reshape(-1, 1)

    geo = Geo[Geo['departement'].isin([name2str[dep] for dep in testd_departement])].reset_index(drop=True)

    x = list(zip(geo.longitude, geo.latitude))

    geo['id'], _ = graph._predict_node_graph_with_position(x)

    def add_value_from_array(date, x, array, index):
        try:
            if index is not None:
                return array[(ytrue[:,date_index] == date) & (ytrue[:,graph_id_index] == x)][:,index][0]
            else:
                return array[(ytrue[:,date_index] == date) & (ytrue[:,graph_id_index] == x)][0]
        except Exception as e:
            #logger.info(e)
            return 0
    
    check_and_create_path(dir_output)
    uniqueDates = np.sort(np.unique(ytrue[:,date_index])).astype(int)
    mean = []
    for date in uniqueDates:
        try:
            dfff = geo.copy(deep=True)
            dfff['date'] = date
            dfff['gt'] = dfff['id'].apply(lambda x : add_value_from_array(date, x, ytrue, -1))
            dfff['Fire'] = dfff['id'].apply(lambda x : add_value_from_array(date, x, ytrue, -2))
            dfff['prediction'] = dfff['id'].apply(lambda x : add_value_from_array(date, x, ypred, None))
            dfff['class'] = dfff['id'].apply(lambda x : add_value_from_array(date, x, yclass, None))
            mean.append(dfff)

            fig, ax = plt.subplots(1,2, figsize=(20,10))
            plt.title(allDates[date] + ' prediction')
            fp = dfff[dfff['Fire'] > 0].copy(deep=True)
            fp.geometry = fp['geometry'].apply(lambda x : x.centroid)

            dfff.plot(ax=ax[0], column='prediction', vmin=np.nanmin(ytrue[:,-1]), vmax=np.nanmax(ytrue[:,-1]), cmap='jet')
            if len(fp) != 0:
                fp.plot(column='Fire', color='black', ax=ax[0], alpha=0.7, legend=True)
            ax[0].set_title('Predicted value')

            dfff.plot(ax=ax[1], column='class', vmin=0, vmax=4, cmap='jet', categorical=True)
            if len(fp) != 0:
                fp.plot(column='Fire', color='black', ax=ax[1], alpha=0.7, legend=True)
            ax[1].set_title('Predicted class')
            fig.suptitle(f'Prediction for {allDates[date + 1]}')

            plt.tight_layout()
            name = allDates[date]+'.png'
            plt.savefig(dir_output / name)
            plt.close('all')
        except Exception as e:
            logger.info(e)
    if len(mean) == 0:
        return
    mean = pd.concat(mean).reset_index(drop=True)
    plt.title('Mean prediction')
    _, ax = plt.subplots(2,2, figsize=(20,10))

    plt.title('Mean prediction')

    mean.plot(ax=ax[0][0], column='Fire', cmap='jet', categorical=True)
    ax[0][0].set_title('Number of fire the next day')

    mean.plot(ax=ax[0][1], column='gt', cmap='jet')
    ax[0][1].set_title('Ground truth value')

    mean.plot(ax=ax[1][0], column='prediction', cmap='jet')
    ax[1][0].set_title('Predicted value')

    mean.plot(ax=ax[1][1], column='prediction', cmap='jet')
    ax[1][1].set_title('Predicted class')
    
    plt.tight_layout()
    name = 'Mean.png'
    plt.savefig(dir_output / name)
    plt.close('all')

def calibrated_curve(ypred, y, dir_output, on):
    # Créer la figure
    plt.figure(figsize=(15, 10))
    plt.title('Calibration Curve')

    # Trier ypred et y selon la deuxième dernière colonne de y
    ind = np.lexsort((y[:, -2],))  # On ajoute une virgule pour garantir un tuple à un seul élément
    Y_axis = y[ind][:, -2]
    X_axis = ypred[ind]

    # Afficher les dimensions des données après tri pour débogage
    print("Taille après tri de Y_axis :", Y_axis.shape)
    
    # Tracer la courbe de prédiction
    plt.scatter(X_axis, Y_axis, label='Prediction curve')

    # Ajouter des labels aux axes
    plt.xlabel('Prediction')
    plt.ylabel('Number of sinister')

    # Ajouter une légende
    plt.legend()

    # Sauvegarder le graphique
    print(dir_output / f'{on}.png')
    plt.savefig(dir_output / f'{on}.png')
    plt.close('all')

def susectibility_map_france_daily_geojson(df_res, regions_france, graph, dates, band,
                                           outname, dir_output, vmax_band, dept_reg, sinister,
                                           sinister_point):

    sinister_point['departement'] = np.nan
    sinister_point['id'] = np.nan
    sinister_point['graph_id'] = 0
    values = graph._assign_department(sinister_point[['graph_id', 'id', 'longitude', 'latitude', 'departement']].values)
    sinister_point['departement'] = values[:, departement_index]

    regions_france['departement'] = np.nan
    regions_france['id'] = np.nan
    regions_france['graph_id'] = np.nan
    values = graph._assign_department(regions_france[['graph_id', 'id', 'longitude', 'latitude', 'departement']].values)
    regions_france['departement'] = values[:, departement_index]

    for date_index in dates:
        date = allDates[date_index]
        sinister_point_date = sinister_point[sinister_point['date'] == date]
        logger.info(f'susectibility_map_france_daily for {date}')
        df_res_daily = df_res[df_res['date'] == date_index]
        regions_france['id'] = -1
        regions_france_in_graph = regions_france[regions_france['departement'].isin(df_res.departement.unique())]
        sinister_point_date = sinister_point_date[sinister_point_date['departement'].isin(df_res.departement.unique())]
        
        if not dept_reg:
            array = regions_france_in_graph[['longitude', 'latitude']].values
            values, _ = graph._predict_node_graph_with_position(array)
            regions_france.loc[regions_france_in_graph.index, 'id'] = values
        else:
            regions_france['id'] = regions_france['departement'].values

        for col in ['risk', 'nbsinister', 'prediction', 'class']:
            if col in regions_france.columns:
                regions_france.drop(col, inplace=True, axis=1)

        regions_france = regions_france.set_index('id').join(df_res_daily.set_index('id')[['risk', 'nbsinister', 'prediction', 'class']], on='id').reset_index()
        regions_france['latitude'] = regions_france['geometry'].apply(lambda x : float(x.centroid.y))
        regions_france['longitude'] = regions_france['geometry'].apply(lambda x : float(x.centroid.x))
        # Vérification des valeurs manquantes dans 'time_series_risk'
        missing_mask = regions_france[band].isna()
        # Si des valeurs manquantes existent, on les interpole
        """if missing_mask.any():
            # Récupérer les points non nuls (valides) pour l'interpolation
            valid_points = ~missing_mask
            points = regions_france.loc[valid_points, ['longitude', 'latitude']].values
            values = regions_france.loc[valid_points, band].values
            
            # Points où les valeurs sont manquantes
            missing_points = regions_france.loc[missing_mask, ['longitude', 'latitude']].values

            # Interpolation 2D avec griddata
            interpolated_values = griddata(points, values, missing_points, method='linear', fill_value=0)

            # Remplir les valeurs manquantes dans 'time_series_class'
            regions_france.loc[missing_mask, band] = interpolated_values.astype(int)"""

        #for dept_code in regions_france.departement.unique():
        #    if dept_code == np.nan:
        #        continue
        #    dir_data = rootDisk / 'csv' / int2name[dept_code] / 'data' / 'spatial'
        #    geo_spa = gpd.read_file(f'{dir_data}/hexagones.geojson')
            #non_sinister_h3 = geo_spa[geo_spa['sinister'] == 0]['hex_id']
            #regions_france[regions_france['hex_id'].isin(non_sinister_h3)][band] = 0

        database_in_fp = sinister_point['database'].unique()
        colors = {'bdiff': 'red',
                  'firemen': 'yellow'}
        
        regions_france.loc[regions_france[~regions_france['departement'].isin(df_res.departement.unique())].index, band] = 0
        fig, ax = plt.subplots(1, figsize=(30,30))
        ax.set_title(date)
        regions_france.plot(column=band, vmin=0, vmax=vmax_band, cmap='jet', ax=ax)
        sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=vmax_band))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation="vertical", label=band)
        if len(sinister_point_date) > 0:
            sinister_point_date = gpd.GeoDataFrame(sinister_point_date, geometry=gpd.points_from_xy(sinister_point_date.longitude, sinister_point_date.latitude))
            for database_name in database_in_fp:
                sinister_point_date[sinister_point_date['database'] == database_name].plot(ax=ax, marker='X', color=colors[database_name], edgecolor='black', markersize=100)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(dir_output / f'susectibility_map_daily_{outname}_{band}_{date}.png')
        plt.close('all')

def find_pixel(lats, lons, lat, lon):

    lonIndex = (np.abs(lons - lon)).argmin()
    latIndex = (np.abs(lats - lat)).argmin()

    lonValues = lons.reshape(-1,1)[lonIndex]
    latValues = lats.reshape(-1,1)[latIndex]

    return np.where((lons == lonValues) & (lats == latValues))

def susectibility_map_france_daily_image(df, vmax, graph, departement, dates, resolution, region_dept, column,
                                         dir_output, predictor, sinister_point):
    for date in dates.astype(int):
        df_date = df[(df['date'] == date) & (df['departement'] == name2int[departement])]
        #print(sinister_point.departement.unique())
        #print(df.departement.unique())
        sinister_point_date = sinister_point[(sinister_point['departement'].isin(df.departement.unique())) & (sinister_point['date'] == allDates[date])]
        n_pixel_x = resolutions[resolution]['x']
        n_pixel_y = resolutions[resolution]['y']
        array = region_dept[['longitude', 'latitude']].values
        values, _ = graph._predict_node_graph_with_position(array)
        region_dept['id'] = values

        if column in region_dept.columns:
            region_dept.drop(column, axis=1, inplace=True)
        regions_dept = region_dept.set_index('id').join(df_date.set_index('id')[column], on='id').reset_index()
        
        high_scale_mask, longs, lats = rasterization(regions_dept, n_pixel_y, n_pixel_x, 'id',
                                                  dir_output, outputname=f'{departement}_{allDates[date]}_mask', defVal = np.nan)
            
        high_scale_pred, _, _ = rasterization(regions_dept, n_pixel_y, n_pixel_x, column,
                                              dir_output, outputname=f'{departement}_{allDates[date]}_{column}', defVal = np.nan)
        
        os.remove(dir_output / f'{departement}_{allDates[date]}_{column}.geojson')
        os.remove(dir_output / f'{departement}_{allDates[date]}_mask.geojson')
        os.remove(dir_output / f'{departement}_{allDates[date]}_{column}.tif')
        os.remove(dir_output / f'{departement}_{allDates[date]}_mask.tif')

        high_scale_pred = high_scale_pred[0]
        high_scale_mask = high_scale_mask[0]

        # Assign no fire pixel
        dir_data = rootDisk / 'csv' / departement / 'data' / 'GEE' / 'landcover'
        dir_data_2 = rootDisk / 'csv' / departement / 'data' / 'GEE' / 'sentinel'
        sentinel, _, _ = read_tif(dir_data_2  / 'summer.tif')
        landcover, _, _ = read_tif(dir_data  / 'summer.tif')
        sentinel = sentinel[0]
        landcover = landcover[0]
        if landcover.shape != high_scale_pred.shape:
            landcover = resize_no_dim(landcover, high_scale_pred.shape[0], high_scale_pred.shape[1])
            longs = resize_no_dim(longs, high_scale_pred.shape[0], high_scale_pred.shape[1])
            lats = resize_no_dim(lats, high_scale_pred.shape[0], high_scale_pred.shape[1])
            sentinel = resize_no_dim(sentinel, high_scale_pred.shape[0], high_scale_pred.shape[1])

        # 0 is water
        # 6 is building
        # 8 is snow
        non_fire_pixel = np.isin(landcover, [0,5,6,7,8])
        high_scale_pred[non_fire_pixel] = 0
        high_scale_pred[np.isnan(sentinel)] = np.nan
        longs[np.isnan(high_scale_pred)] = -math.inf
        lats[np.isnan(high_scale_pred)] = -math.inf

        label_positions = np.sort(predictor.cluster_centers.reshape(-1))

        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

        plt.figure(figsize=(15,15))

        plt.title(f'30m fire susecptibility {departement} {allDates[date]}')
        fp = list(zip(sinister_point_date.latitude, sinister_point_date.longitude))
        if len(fp) > 0:
            # Conversion des latitudes/longitudes des points en indices de la grille de l'image
            i = 0
            for lat, lon in fp:
                # Trouver l'indice le plus proche dans l'image en fonction des latitudes/longitudes
                pos = find_pixel(lats, longs, lat, lon)
                if i == 0:
                    plt.scatter(pos[1], pos[0], color='red', s=100, label='Fire points', edgecolor='black', marker='X')
                    i = 1
                else:
                    plt.scatter(pos[1], pos[0], color='red', s=100, edgecolor='black', marker='X')
                    
            # Ajouter la légende pour les points de feux
            plt.legend(loc='upper right')

        img = plt.imshow(high_scale_pred, cmap='jet', vmin=0, vmax=vmax)
        # Add color bar
        cbar = plt.colorbar(img)
        # Set the positions and labels for the color bar
        cbar.set_ticks(label_positions)
        #cbar.set_ticklabels(labels)
        plt.savefig(dir_output / f'{departement}_{allDates[date]}_{column}.png')
        plt.close('all')

        plt.figure(figsize=(15,15))
        img = plt.imshow(high_scale_mask)
        plt.title(f'30m raster {departement}')
        plt.savefig(dir_output / f'{departement}_raster.png')
        plt.close('all')