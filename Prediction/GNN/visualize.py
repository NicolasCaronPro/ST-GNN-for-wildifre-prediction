from construct import *

def realVspredict(ypred, y, band, dir_output, on):
    check_and_create_path(dir_output)
    ytrue = y[:,band]
    dept = np.unique(y[:, 3])

    for d in dept:
        mask = np.argwhere(y[:,3] == d)
        maxi = max(np.nanmax(ypred[mask]), np.nanmax(ytrue[mask]))
        mini = min(np.nanmin(ypred[mask]), np.nanmin(ytrue[mask]))
        ids = np.unique(y[mask, 0])
        if ids.shape[0] == 1:
            _, ax = plt.subplots(ids.shape[0], figsize=(15,5))
            ax.plot(ypred[mask], color='red', label='predict')
            ax.plot(ytrue[mask], color='blue', label='real', alpha=0.5)
            x = np.argwhere(y[mask,-2] > 0)[:,0]
            ax.scatter(x, ypred[mask][x], color='black', label='fire', alpha=0.5)
            ax.set_ylim(ymin=mini, ymax=maxi)
        else:
            _, ax = plt.subplots(ids.shape[0], figsize=(50,50))
            for i, id in enumerate(ids):
                mask2 = np.argwhere(y[:, 0] == id)
                ax[i].plot(ypred[mask2], color='red', label='predict')
                ax[i].plot(ytrue[mask2], color='blue', label='real', alpha=0.5)
                x = np.argwhere(y[mask2,-2] > 0)[:,0]
                ax[i].scatter(x, ypred[mask2][x], color='black', label='fire', alpha=0.5)
                ax[i].set_ylim(ymin=mini, ymax=maxi)
        plt.legend()
        outn = str(d) + '_' + on + '.png'
        plt.savefig(dir_output / outn)

    uids = np.unique(y[:, 0])
    ysum = np.empty((uids.shape[0], 2))
    ysum[:, 0] = uids
    for id in uids:
        ysum[np.argwhere(ysum[:, 0] == id)[:, 0], 1] = np.sum(y[np.argwhere(y[:, 0] == id)[:, 0], -2])

    ind = np.lexsort([ysum[:,1]])
    ymax = np.flip(ysum[ind, 0])[:5]
    _, ax = plt.subplots(np.unique(ymax).shape[0], figsize=(50,25))
    for i, idtop in enumerate(ymax):
        mask = np.argwhere(y[:,0] == idtop)
        dept = np.unique(y[mask, 3])[0]
        ids = np.unique(y[mask, 0])
        ax[i].plot(ypred[mask], color='red', label='predict')
        ax[i].plot(ytrue[mask], color='blue', label='real', alpha=0.5)
        ax[i].set_ylim(ymin=0, ymax=np.nanmax(ytrue[mask]))
        x = np.argwhere(y[mask,-2] > 0)[:,0]
        ax[i].scatter(x, ypred[mask][x], color='black', label='fire', alpha=0.5)
        ax[i].set_title(str(dept)+'_'+str(idtop))

    plt.tight_layout()
    outn =  'top_5' + on + '.png'
    plt.savefig(dir_output / outn)

def sinister_distribution_in_class(ypredclass, y, dir_output):
    check_and_create_path(dir_output)
    nbclass = np.unique(y[:, -3])

    meansinisterinpred = []
    meansinisteriny = []

    nbsinisterinpred = []
    nbsinisteriny = []

    for cls in nbclass:
        mask = np.argwhere(y[:, -3] == cls)[:, 0]
        if mask.shape[0] == 0:
            nbsinisteriny.append(-1)
            meansinisteriny.append(-1)
            
        nbsinisteriny.append(round(100 * np.nansum(y[mask, -2]) / np.nansum(y[:, -2])))
        meansinisteriny.append(np.nanmean(y[mask, -2]))

        mask = np.argwhere(ypredclass == cls)[:, 0]

        nbsinisterinpred.append(round(100 * np.nansum(y[mask, -2]) / np.nansum(y[:, -2])))
        meansinisterinpred.append(np.nanmean(y[mask, -2]))


    fig, ax = plt.subplots(2, figsize=(15,10))
    ax[0].plot(nbclass, meansinisteriny, label='GT')
    ax[0].plot(nbclass, meansinisterinpred, label='Prediction')
    ax[0].set_xlabel('Class')
    ax[0].set_ylabel('Mean of sinister')
    ax[0].set_title('Mean per class')

    ax[1].plot(nbclass, nbsinisteriny, label='GT')
    ax[1].plot(nbclass, nbsinisterinpred, label='Prediction')
    ax[1].set_xlabel('Class')
    ax[1].set_ylabel('Nb of sinister')
    ax[1].set_title('Number per class')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_output / 'mean_fire.png')
    plt.close('all')

def realVspredict2d(ypred : np.array,
                    ytrue : np.array,
                    isBin : np.array,
                    modelName : str,
                    scale : int,
                    dir_output : Path,
                    dir : Path,
                    Geo : gpd.GeoDataFrame,
                    testDepartement : list,
                    graph : GraphStructure):

    dir_predictor = root_graph / dir / 'influenceClustering'

    yclass = np.empty(ypred.shape)
    for nameDep in testDepartement:
        mask = np.argwhere(ytrue[:,3] == name2int[nameDep])
        if mask.shape[0] == 0:
            continue
        if not isBin:
            predictor = read_object(nameDep+'Predictor'+str(scale)+'.pkl', dir_predictor)
        else:
            predictor = read_object(nameDep+'Predictor'+modelName+str(scale)+'.pkl', dir_predictor)
        
        classpred = predictor.predict(ypred[mask].astype(float))

        yclass[mask] = order_class(predictor, classpred).reshape(-1, 1)

    geo = Geo[Geo['departement'].isin([name2str[dep] for dep in testDepartement])].reset_index(drop=True)

    x = list(zip(geo.longitude, geo.latitude))

    geo['id'] = graph._predict_node(x)

    def add_value_from_array(date, x, array, index):
        try:
            if index is not None:
                return array[(ytrue[:,4] == date) & (ytrue[:,0] == x)][:,index][0]
            else:
                return array[(ytrue[:,4] == date) & (ytrue[:,0] == x)][0]
        except Exception as e:
            #logger.info(e)
            return 0
    
    check_and_create_path(dir_output)
    uniqueDates = np.sort(np.unique(ytrue[:,4])).astype(int)
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

def susectibility_map(dept : str, sinister : str,
                      train_dir : Path, dir_output : Path, k_days : int,
                      isBin : bool, scale : int, name : str, res : np.array, n : str,
                      minDate : str, maxDate : str, resolution : str):
    
    # Wildfire susceptibility map
    dir_predictor = root_graph / train_dir / 'influenceClustering'
    dir_mask = root_graph / train_dir / 'raster'
    name_0 = dept + 'rasterScale0.pkl'
    sinister_file = sinister+'.csv' 
    sinister_df = pd.read_csv(root_graph / 'sinister' / sinister_file)
    sinister_df = sinister_df[sinister_df['departement'] == dept]
    if not (dir_mask / name_0).is_file():
        geo = gpd.read_file(root_graph / 'regions' / 'regions.geojson')
        graphScale = construct_graph(0,
                                maxDist[scale],
                                sinister,
                                geo, 6, k_days,
                                train_dir,
                                True,
                                False, 
                                resolution)
    else:
        graphScale = read_object('graph_0.pkl', train_dir)
        
    X_kmeans = list(zip(sinister_df.longitude, sinister_df.latitude))
    sinister_df['scale'+str(scale)] = graphScale._predict_node(X_kmeans)
    sinister_df['label'] = 1
    sinister_df = sinister_df.groupby(by=['scale0', 'date'])['label'].sum().reset_index()

    sinisterNode = np.full((len(sinister_df), 6), -1.0, dtype=float)
    sinisterNode[:,0] = sinister_df['scale0'].values
    sinisterNode[:,4] = [allDates.index(d) for d in sinister_df['date']]
    sinisterNode[:,-1] = sinister_df['label']

    res = res[(res[:, 4] >= allDates.index(minDate)) & (res[:, 4] < allDates.index(maxDate))]
    sinisterNode = sinisterNode[(sinisterNode[:, 4] >= allDates.index(minDate)) & (sinisterNode[:, 4] < allDates.index(maxDate))]
    logger.info(f'{dept}, nb sinister :{sinisterNode.shape}')

    if isBin:
        predictor = read_object(dept+'Predictor'+name+str(scale)+'.pkl', dir_predictor)
    else:
        predictor = read_object(dept+'Predictor'+str(scale)+'.pkl', dir_predictor)
        
    class_risk = np.copy(res)

    class_risk[:, -3] = predictor.predict(res[:, -3])
    class_risk[:, -4] = predictor.predict(res[:, -4])

    resSum = array2image(class_risk, dir_mask, scale, dept, 'mean', -3, dir_output / n, dept+'_prediction.pkl')
    ySum = array2image(class_risk, dir_mask, scale, dept, 'mean', -4, dir_output / n, dept+'_gt.pkl')
    fireSum = array2image(sinisterNode, dir_mask, 0, dept, 'sum', -1, dir_output / n, dept+'_fire.pkl')

    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    im = ax[0].imshow(resSum, vmin=1, vmax=6,  cmap='jet')
    ax[0].set_title('Sum of Prediction')
    fig.colorbar(im, ax=ax[0], orientation='vertical')
    im = ax[1].imshow(ySum, vmin=1, vmax=6,  cmap='jet')
    ax[1].set_title('Sum of Target')
    fig.colorbar(im, ax=ax[1], orientation='vertical')
    x = np.argwhere((fireSum > 0) & (~np.isnan(fireSum)))
    ax[0].scatter(x[:,1], x[:,0], color='black', label='fire', alpha=0.5)
    ax[1].scatter(x[:,1], x[:,0], color='black', label='fire', alpha=0.5)
    susec = dept+'_susecptibility.png'
    plt.tight_layout()
    plt.savefig(dir_output / n / susec)
    plt.close('all')