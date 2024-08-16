from torch_geometric.data import Dataset
from torch.utils.data import DataLoader
import torch
from train import *

#########################################################################################################
#                                                                                                       #
#                                         Graph collate                                                 #
#                                         Dataset reader                                                #
#                                                                                                       #
#########################################################################################################

def graph_collate_fn(batch):
    #node_indice_list = []
    edge_index_list = []
    node_features_list = []
    node_labels_list = []
    num_nodes_seen = 0

    for features_labels_edge_index_tuple in batch:
        # Just collect these into separate lists
        node_features_list.append(features_labels_edge_index_tuple[0])
        node_labels_list.append(features_labels_edge_index_tuple[1])
        #node_indice_list.append(features_labels_edge_index_tuple[2])
        edge_index = features_labels_edge_index_tuple[2]  # all of the components are in the [0, N] range
        if edge_index.shape[0] > 2:
            edge_index[0] += num_nodes_seen
            edge_index[1] += num_nodes_seen
            edge_index_list.append(edge_index)
        else:
            edge_index_list.append(edge_index + num_nodes_seen)  # very important! translate the range of this component
        num_nodes_seen += len(features_labels_edge_index_tuple[1])  # update the number of nodes we've seen so far

    # Merge the PPI graphs into a single graph with multiple connected components
    node_features = torch.cat(node_features_list, 0)
    node_labels = torch.cat(node_labels_list, 0)
    edge_index = torch.cat(edge_index_list, 1)
    #node_indices = torch.cat(node_indice_list, 0)
    return node_features, node_labels, edge_index

def graph_collate_fn_no_label(batch):
    #node_indice_list = []
    edge_index_list = []
    node_features_list = []
    num_nodes_seen = 0

    for features_labels_edge_index_tuple in batch:
        # Just collect these into separate lists
        node_features_list.append(features_labels_edge_index_tuple[0])
        #node_indice_list.append(features_labels_edge_index_tuple[1])
        edge_index = features_labels_edge_index_tuple[1]  # all of the components are in the [0, N] range
        edge_index_list.append(edge_index + num_nodes_seen)  # very important! translate the range of this component
        num_nodes_seen += len(features_labels_edge_index_tuple[0])  # update the number of nodes we've seen so far

    # Merge the PPI graphs into a single graph with multiple connected components
    node_features = torch.cat(node_features_list, 0)
    edge_index = torch.cat(edge_index_list, 1)
    #node_indices = torch.cat(node_indice_list, 0)

    return node_features, edge_index

def graph_collate_fn_adj_mat(batch):
    node_features_list = []
    #node_indice_list = []
    node_labels_list = []
    edge_index_list = []
    num_nodes_seen = 0

    for features_labels_edge_index_tuple in batch:
        # Just collect these into separate lists
        node_features_list.append(features_labels_edge_index_tuple[0])
        node_labels_list.append(features_labels_edge_index_tuple[1])
        #node_indice_list.append(features_labels_edge_index_tuple[2])
        edge_index = features_labels_edge_index_tuple[2]  # all of the components are in the [0, N] range
        edge_index_list.append(edge_index + num_nodes_seen)  # very important! translate the range of this component
        num_nodes_seen += len(features_labels_edge_index_tuple[1])  # update the number of nodes we've seen so far

    node_features = torch.cat(node_features_list, 0)
    node_labels = torch.cat(node_labels_list, 0)
    #node_indices = torch.cat(node_indice_list, 0)
    edge_index = torch.cat(edge_index_list, 1)

    adjacency_matrix = torch.zeros(num_nodes_seen, num_nodes_seen)
    for edge in edge_index:
        node1, node2 = edge[0].item(), edge[1].item()
        adjacency_matrix[node1, node2] = 1
        adjacency_matrix[node2, node1] = 1  # Since it's an undirected graph

    return node_features, node_labels, adjacency_matrix

class InplaceGraphDataset(Dataset):
    def __init__(self, X : list, Y : list, edges : list, leni : int, device : torch.device) -> None:
        self.X = X
        self.Y = Y
        self.device = device
        self.edges = edges
        self.leni = leni

    def __getitem__(self, index) -> tuple:
        x = self.X[index]
        y = self.Y[index]

        edges = self.edges[index]

        return torch.tensor(x, dtype=torch.float32, device=self.device), \
            torch.tensor(y, dtype=torch.float32, device=self.device), \
            torch.tensor(edges, dtype=torch.long, device=self.device)
        
    def __len__(self) -> int:
        return self.leni
    
    def len(self):
        pass

    def get(self):
        pass
    
class ReadGraphDataset_2D(Dataset):
    def __init__(self, X : list,
                 Y : list,
                 edges : list,
                 leni : int,
                 device : torch.device,
                 path : Path) -> None:
        
        self.X = X
        self.Y = Y
        self.device = device
        self.edges = edges
        self.leni = leni
        self.path = path

    def __getitem__(self, index) -> tuple:

        x = read_object(self.X[index], self.path)
        #y = read_object(self.Y[index], self.path)
        y = self.Y[index]
        edges = self.edges[index]

        return torch.tensor(x, dtype=torch.float32, device=self.device), \
            torch.tensor(y, dtype=torch.float32, device=self.device), \
            torch.tensor(edges, dtype=torch.long, device=self.device)
        
    def __len__(self) -> int:
        return self.leni
    
    def len(self):
        pass

    def get(self):
        pass

#################################### NUMPY #############################################################
def get_train_val_test_set(graphScale, df, train_features, train_departements, prefix, dir_output, METHODS, args):
    # Input config
    maxDate = args.maxDate
    trainDate = args.trainDate
    doFet = args.featuresSelection == "True"
    nbfeatures = int(args.NbFeatures)
    values_per_class = args.nbpoint
    scale = int(args.scale)
    spec = args.spec
    doPCA = args.pca == 'True'
    doKMEANS = args.KMEANS == 'True'
    ncluster = int(args.ncluster)
    k_days = int(args.k_days) # Size of the time series sequence use by DL models
    days_in_futur = int(args.days_in_futur) # The target time validation
    scaling = args.scaling

    features_name, newshape = get_features_name_list(graphScale.scale, train_features, METHODS)

    save_object(features, 'features.pkl', dir_output)
    save_object(train_features, 'train_features.pkl', dir_output)
    save_object(features_name, 'features_name_train.pkl', dir_output)

    if days_in_futur > 1:
        prefix += '_'+str(days_in_futur)
        prefix += '_'+futur_met

    df = df[ids_columns + features_name + targets_columns]
    logger.info(f'Ids in dataset {np.unique(df["id"])}')

    # Preprocess
    train_dataset, val_dataset, test_dataset, train_dataset_unscale, test_dataset_unscale = preprocess(df=df, scaling=scaling, maxDate=maxDate,
                                                    trainDate=trainDate, train_departements=train_departements,
                                                    departements = departements,
                                                    ks=k_days, dir_output=dir_output, prefix=prefix, features_name=features_name,
                                                    days_in_futur=days_in_futur,
                                                    futur_met=futur_met, doKMEANS=doKMEANS, ncluster=ncluster, scale=scale)
    
    realVspredict(test_dataset['risk'].values, test_dataset[ids_columns + targets_columns].values, -1, dir_output / prefix, 'raw')

    realVspredict(test_dataset['class_risk'].values, test_dataset[ids_columns +targets_columns].values, -3, dir_output / prefix, 'class')

    sinister_distribution_in_class(test_dataset['class_risk'].values, test_dataset[ids_columns + targets_columns].values, dir_output / prefix, 'mean_fire')

    features_selected = features_selection(doFet, train_dataset, dir_output / prefix, features_name, nbfeatures, 'risk')

    if values_per_class == 'full':
        prefix = f'{values_per_class}'
    else:
        prefix = f'{values_per_class}_{k_days}'

    prefix += f'_{scale}_{args.graphConstruct}'

    if doKMEANS:
        prefix += '_kmeans'

    if days_in_futur > 1:
        prefix += '_'+str(days_in_futur)
        prefix += '_'+futur_met

    if doPCA:
        prefix += '_pca'
        x_train = train_dataset[0]
        pca, components = train_pca(x_train, 0.99, dir_output / prefix, features_selected)
        train_dataset = apply_pca(train_dataset[0], pca, components, features_selected)
        val_dataset = apply_pca(val_dataset[0], pca, components, features_selected)
        test_dataset = apply_pca(test_dataset[0], pca, components, features_selected)
        train_features = ['pca_'+str(i) for i in range(components)]
        features_name, _ = get_features_name_list(0, train_features, METHODS_SPATIAL_TRAIN)
        features_selected = features_name
        
    logger.info(f'Train dates are between : {allDates[int(np.min(train_dataset["date"]))], allDates[int(np.max(train_dataset["date"]))]}')
    logger.info(f'Val dates are bewteen : {allDates[int(np.min(val_dataset["date"]))], allDates[int(np.max(val_dataset["date"]))]}')

    return train_dataset, val_dataset, test_dataset, train_dataset_unscale, test_dataset_unscale, prefix, features_selected

#########################################################################################################
#                                                                                                       #
#                                         Preprocess                                                    #
#                                                                                                       #
#########################################################################################################

def preprocess(df: pd.DataFrame, scaling: str, maxDate: str, trainDate: str, train_departements: list, departements: list, ks: int,
               dir_output: Path, prefix: str, features_name: list, days_in_futur: int, futur_met: str, doKMEANS: bool, ncluster: int, scale : int,
               save = True):


    global features

    # Remove nodes where there is a feature at NaN
    df = remove_nan_nodes(df)
    df = remove_none_target(df)
    df = remove_bad_period(df, PERIODES_A_IGNORER, departements)
    df = remove_non_fire_season(df, SAISON_FEUX, departements)

    logger.info(f'DataFrame shape after removal of NaN and -1 target: {df.shape}')

    assert df.shape[0] > 0

    # Sort by date
    df = df.sort_values('date')

    if days_in_futur > 1:
        logger.info(f'{futur_met} target by {days_in_futur} days in future')
        df = target_by_day(df, days_in_futur, futur_met)
        logger.info(f'Done')

    trainCode = [name2int[departement] for departement in train_departements]

    train_mask = (df['date'] < allDates.index(trainDate)) & (df['departement'].isin(trainCode))
    val_mask = (df['date'] >= allDates.index(trainDate) + ks) & (df['date'] < allDates.index(maxDate)) & (df['departement'].isin(trainCode))
    test_mask = ((df['date'] >= allDates.index(maxDate) + ks) & (df['departement'].isin(trainCode))) | (~df['departement'].isin(trainCode))

    train_dataset_unscale = df[train_mask].reset_index(drop=True).copy(deep=True)

    if doKMEANS:
        features_selected_kmeans, _ = get_features_name_list(scale, kmeans_features, METHODS_KMEANS)
        train_break_point(df[train_mask], features_selected_kmeans, dir_output / 'check_none' / prefix / 'kmeans', ncluster)
        df = apply_kmeans_class_on_target(df.copy(deep=True), dir_output / 'check_none' / prefix / 'kmeans', 'risk', tresh_kmeans, features_selected_kmeans, new_val=0)
        df = apply_kmeans_class_on_target(df.copy(deep=True), dir_output / 'check_none' / prefix / 'kmeans', 'class_risk', tresh_kmeans, features_selected_kmeans, new_val=0)
        #df = apply_kmeans_class_on_target(df.copy(deep=True), dir_output / 'check_none' / prefix / 'kmeans', 'nbsinister', tresh_kmeans, features_selected_kmeans, new_val=0)
        df = apply_kmeans_class_on_target(df.copy(deep=True), dir_output / 'check_none' / prefix / 'kmeans', 'weight', tresh_kmeans, features_selected_kmeans, new_val=0)
        test_dataset_unscale = df[test_mask].reset_index(drop=True).copy(deep=True)
    else:
        test_dataset_unscale = None

    if save:
        save_object(df[train_mask], 'df_train_'+prefix+'.pkl', dir_output)

    # Define the scale method
    if scaling == 'MinMax':
        scaler = min_max_scaler
    elif scaling == 'z-score':
        scaler = standard_scaler
    elif scaling == "robust":
        scaler = robust_scaler
    elif scaling == 'none':
        scaler = None
    else:
        raise ValueError('Unknown scaling method')

    if 'AutoRegressionReg' in features_name:
        autoregressiveBand = features_name.index('AutoRegressionReg')
    else:
        autoregressiveBand = -1

    df_no_scale = df.copy(deep=True)

    if scaler is not None:
        # Scale Features
        for feature in features_name:
            if features_name.index(feature) == autoregressiveBand:
                logger.info(f'Avoid scale for Autoregressive feature')
                continue
            df[feature] = scaler(df[feature].values, df[train_mask][feature].values, concat=False)

    logger.info(f'Max of the train set: {df[train_mask][features_name].max().max(), df[train_mask][features_name].max().idxmax()}, before scale : {df_no_scale[train_mask][features_name].max().max(), df_no_scale[train_mask][features_name].max().idxmax()}')
    logger.info(f'Max of the val set: {df[val_mask][features_name].max().max(), df[val_mask][features_name].max().idxmax()}, before scale : {df_no_scale[val_mask][features_name].max().max(), df_no_scale[val_mask][features_name].max().idxmax()}')
    logger.info(f'Max of the test set: {df[test_mask][features_name].max().max(), df[test_mask][features_name].max().idxmax()}, before scale : {df_no_scale[test_mask][features_name].max().max(), df_no_scale[test_mask][features_name].max().idxmax()}')
    logger.info(f'Min of the train set: {df[train_mask][features_name].min().min(), df[train_mask][features_name].min().idxmin()}, before scale : {df_no_scale[train_mask][features_name].min().min(), df_no_scale[train_mask][features_name].min().idxmin()}')
    logger.info(f'Min of the val set: {df[val_mask][features_name].max().min(), df[val_mask][features_name].max().idxmin()}, before scale : {df_no_scale[val_mask][features_name].min().min(), df_no_scale[val_mask][features_name].min().idxmin()}')
    logger.info(f'Min of the test set: {df[test_mask][features_name].min().min(), df[test_mask][features_name].min().idxmin()}, before scale : {df_no_scale[test_mask][features_name].min().min(), df_no_scale[test_mask][features_name].min().idxmin()}')
    logger.info(f'Size of the train set: {df[train_mask].shape[0]}, size of the val set: {df[val_mask].shape[0]}, size of the test set: {df[test_mask].shape[0]}')
    logger.info(f'Unique train dates: {np.unique(df[train_mask]["date"]).shape[0]}, val dates: {np.unique(df[val_mask]["date"]).shape[0]}, test dates: {np.unique(df[test_mask]["date"]).shape[0]}')

    return df[train_mask].reset_index(drop=True), df[val_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True), train_dataset_unscale, test_dataset_unscale

def preprocess_test(df_X: pd.DataFrame, df_Y: pd.DataFrame, x_train: pd.DataFrame, scaling: str):
    df_XY = df_X.join(df_Y)

    df_XY_clean = remove_nan_nodes(df_XY)
    
    logger.info(f'DataFrame shape before removal of NaNs: {df_X.shape}, after removal: {df_XY_clean.shape}')
    assert df_XY_clean.shape[0] > 0

    # Define the scaler
    if scaling == 'MinMax':
        scaler = min_max_scaler
    elif scaling == 'z-score':
        scaler = standard_scaler
    elif scaling == "robust":
        scaler = robust_scaler
    else:
        raise ValueError('Unknown scaling method. Scaling is obligatory.')

    # Scaling Features
    for feature in df_X.columns[6:]:
        df_XY_clean[feature] = scaler(df_XY_clean[feature], x_train[feature], concat=False)

    logger.info(f'Check {scaling} standardisation test: max {df_XY_clean.iloc[:, 6:].max().max()}, min {df_XY_clean.iloc[:, 6:].min().min()}')

    return df_XY_clean

def preprocess_inference(df_X: pd.DataFrame, x_train: pd.DataFrame, scaling: str, features_name : list, fire_feature : list, apply_break_point : bool, 
                         scale, kmeans_features, dir_break_point):
    df_X_clean = remove_nan_nodes(df_X)

    logger.info(f'DataFrame shape before removal of NaNs: {df_X.shape}, after removal: {df_X_clean.shape}')

    logger.info(f'Nan columns : {df_X.columns[df_X.isna().any()].tolist()}')

    df_X_clean[fire_feature] = np.nan

    if apply_break_point:
        logger.info('Apply Kmeans to remove useless node')
        features_selected_kmeans, _ = get_features_name_list(scale, kmeans_features, METHODS_SPATIAL_TRAIN)
        df_X_clean['fire_prediction_raw'] = apply_kmeans_class_on_target(df_X_clean.copy(deep=True), dir_break_point, 'fire_prediction_raw', 0.15, features_selected_kmeans, new_val=0)['fire_prediction_raw'].values

    if df_X_clean.shape[0] == 0:
        return None

    if scaling == 'MinMax':
        scaler = min_max_scaler
    elif scaling == 'z-score':
        scaler = standard_scaler
    elif scaling == "robust":
        scaler = robust_scaler
    else:
        raise ValueError('Unknown scaling method.')
    
    df_X_copy = df_X_clean.copy(deep=True)

    # Scaling Features
    for feature in features_name:
        df_X_clean[feature] = scaler(df_X_clean[feature].values, x_train[feature].values, concat=False)


    logger.info(f'Check {scaling} standardisation inference: max {df_X_clean[df_X_clean["date"] == df_X_clean.date.max()][features_name].max().max(), df_X_clean[df_X_clean["date"] == df_X_clean.date.max()][features_name].max().idxmax()}')
    logger.info(f'Check {scaling} standardisation inference: min  {df_X_clean[df_X_clean["date"] == df_X_clean.date.max()][features_name].min().min(), df_X_clean[df_X_clean["date"] == df_X_clean.date.max()][features_name].min().idxmin()}')
    
    logger.info(f'Check before {scaling} standardisation inference: max {df_X_copy[df_X_copy["date"] == df_X_copy.date.max()][features_name].max().max(), df_X_copy[df_X_copy["date"] == df_X_copy.date.max()][features_name].max().idxmax()}')
    logger.info(f'Check before {scaling} standardisation inference: min  {df_X_copy[df_X_copy["date"] == df_X_copy.date.max()][features_name].min().min(), df_X_copy[df_X_copy["date"] == df_X_copy.date.max()][features_name].min().idxmin()}')
    
    logger.info(f'Check before xtrain: max {x_train[features_name].max().max(), x_train[features_name].max().idxmax()}')
    logger.info(f'Check before xtrain: min  {x_train[features_name].min().min(), x_train[features_name].min().idxmin()}')

    return df_X_clean


#########################################################################################################
#                                                                                                       #
#                                         Train Val                                                     #
#                                                                                                       #
#########################################################################################################

def create_dataset(graph,
                    train,
                    val,
                    test,
                    features_name,
                    use_temporal_as_edges : bool,
                    device,
                    ks : int):

    x_train, y_train = train[ids_columns + features_name].values, train[ids_columns + targets_columns].values
    
    x_val, y_val = val[ids_columns + features_name].values, val[ids_columns + targets_columns].values

    x_test, y_test = test[ids_columns + features_name].values, test[ids_columns + targets_columns].values

    dateTrain = np.unique(y_train[np.argwhere(y_train[:, 5] > 0),4])
    dateVal = np.unique(y_val[np.argwhere(y_val[:, 5] > 0),4])
    dateTest = np.unique(y_test[:,4])
    
    Xst = []
    Yst = []
    Est = []

    XsV = []
    YsV = []
    EsV = []

    XsTe = []
    YsTe = []
    EsTe = []

    for id in dateTrain:
        if use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, id, x_train, y_train, ks)
        else:
            x, y, e = construct_graph_with_time_series(graph, id, x_train, y_train, ks)

        if x is None:
            continue
    
        Xst.append(x)
        Yst.append(y)
        Est.append(e)

    for id in dateVal:
        if use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, id, x_val, y_val, ks)
        else:
            x, y, e = construct_graph_with_time_series(graph, id, x_val, y_val, ks)

        if x is None:
            continue
        
        XsV.append(x)
        YsV.append(y)
        EsV.append(e)

    for id in dateTest:
        if use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, id, x_test, y_test, ks)
        else:
            x, y, e = construct_graph_with_time_series(graph, id, x_test, y_test, ks)

        if x is None:
            continue
    
        XsTe.append(x)
        YsTe.append(y)
        EsTe.append(e)
    
    assert len(Xst) > 0
    assert len(XsV) > 0
    assert len(XsTe) > 0

    train_dataset = InplaceGraphDataset(Xst, Yst, Est, len(Xst), device)
    val_dataset = InplaceGraphDataset(XsV, YsV, EsV, len(XsV), device)
    testDatset = InplaceGraphDataset(XsTe, YsTe, EsTe, len(XsTe), device)
    return train_dataset, val_dataset, testDatset

def load_x_from_pickle(date : int,
                       path : Path,
                       df_train : pd.DataFrame,
                       scaling : str,
                       features_name_2D : list) -> np.array:

    leni = len(features_name_2D)
    x_2D = read_object(f'X_{date}.pkl', path)
    new_x_2D = np.empty((leni, x_2D.shape[1], x_2D.shape[2]))
    if scaling == 'MinMax':
        scaler = min_max_scaler
    elif scaling == 'z-score':
        scaler = standard_scaler
    elif scaling == "robust":
        scaler = robust_scaler
    elif scaling == 'none':
        scaler = None
    else:
        raise ValueError(f'Unknown scaling method {scaling}')
    
    for i, fet_2D in enumerate(features_name_2D):
        if fet_2D in calendar_variables or fet_2D in geo_variables or fet_2D.find('AutoRegression') != -1:
            var_1D = fet_2D
        else:
            var_1D = f'{fet_2D}_mean'

        new_x_2D[i, :, :] = x_2D[features_name_2D.index(fet_2D), :, :]
        if False not in np.isnan(new_x_2D[i, :, :]):
            continue
        new_x_2D[i, :, :][np.isnan(new_x_2D)[i, :, :]] = np.nanmean(new_x_2D[i, :, :])
        new_x_2D[i, :, :] = scaler(x_2D[features_name_2D.index(fet_2D), :, :].reshape(-1,1), df_train[var_1D].values, concat=False).reshape((x_2D.shape[1], x_2D.shape[2]))

    return new_x_2D

def generate_image_y(y, y_risk_image, y_binary_image):
    res = np.empty((y.shape[0], y.shape[1], *y_risk_image.shape, y.shape[-1]))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            for k in range(y.shape[2]):
                res[i, j, :, :, k] = y[i, j, k]

        res[i, -1, :,  :, -1] = y_risk_image
        res[i, -1, :,  :, -1] = y_binary_image

    return res

def create_dataset_2D_2(graph, X_np, Y_np, ks, dates, df_train, scaling,
                        features_name_2D, path,
                        use_temporal_as_edges, image_per_node,
                        scale, context):
    
    Xst = []
    Yst = []
    Est = []
    for id in dates:
        if use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, id, X_np, Y_np, ks)
        else:
            x, y, e = construct_graph_with_time_series(graph, id, X_np, Y_np, ks, )
        if x is None:
            continue
        depts = np.unique(y[:,3].astype(int))
        if image_per_node:
            X = np.empty((y.shape[0], len(features_name_2D), *shape2D[scale], ks + 1))
            Y = []
        else:
            X = np.empty((depts.shape, len(features_name_2D), *shape2D[scale], ks + 1))
            Y = np.empty((depts.shape, y.shape[1], *shape2D[scale], ks + 1))

        for i, dept in enumerate(depts):
            index = np.argwhere((y[:, 3, 0] == dept))[:,0]
            y_dept = []
            y_date = read_object(f'Y_{int(id)}.pkl', path / '2D_database' / int2name[dept] / str(scale) / 'influence')
            y_bin_date = read_object(f'Y_{int(id)}.pkl', path / '2D_database' / int2name[dept] / str(scale) / 'binary')

            y_date = resize_no_dim(y_date, 64, 64)
            y_bin_date = resize_no_dim(y_bin_date, 64, 64)

            if image_per_node:
                raster_dept = read_object(f'{int2name[dept]}rasterScale{scale}_{graph.base}.pkl', path / 'raster')
                unodes = np.unique(raster_dept)
                unodes = unodes[~np.isnan(unodes)]

            for k in range(ks + 1):
                date = int(id) - ks
                x_date = load_x_from_pickle(date, path / '2D_database' / int2name[dept], df_train, scaling, features_name_2D)
                if image_per_node:
                    for node in unodes:
                        if node not in np.unique(x[:, 0]):
                            continue
                        mask = np.argwhere(raster_dept == node)
                        minx = np.min(mask[:, 0])
                        miny = np.min(mask[:, 1])
                        maxx = np.max(mask[:, 0])
                        maxy = np.max(mask[:, 1])
                        index = np.argwhere((x[:, 0, 0] == node))
                        x_node = x_date[:, minx:maxx, miny:maxy]
                        for band in range(x_node.shape[0]):
                            x_band = x_node[band]
                            x_band = resize_no_dim(x_band, *shape2D[scale])
                            if False not in np.isnan(x_band):
                                x_band[np.isnan(x_band)] = -1
                                continue
                            x_band[np.isnan(x_band)] = np.nanmean(x_band[:, :])
                            X[index[:, 0], band, :, :, k] = x_band
                else:
                    x_date = resize(x_date, 64, 64, x_date.shape[2])
                    X[i, :, :, :, k] = x_date

            if not image_per_node:
                y_date = generate_image_y(y[y[:, 3, 0] == dept], y_date, y_bin_date)
                Y[i, :, :, :, k] = y_date

        if image_per_node:
            Yst.append(y)

        save_object(X, f'X_{int(id)}.pkl', path / '2D_database' / scaling / context)
        Xst.append(f'X_{int(id)}.pkl')
        if not image_per_node:
            Yst.append(Y)
        Est.append(e)

    return Xst, Yst, Est

def create_dataset_2D(graph,
                    df_train,
                    df_val,
                    df_test,
                    scaling,
                    path,
                    features_name_2D,
                    features_name,
                    image_per_node,
                    use_temporal_as_edges : bool,
                    scale,
                    device,
                    ks : int):
    
    x_train, y_train = df_train[ids_columns + features_name].values, df_train[ids_columns + targets_columns].values

    x_val, y_val = df_val[ids_columns + features_name].values, df_val[ids_columns + targets_columns].values

    x_test, y_test = df_test[ids_columns + features_name].values, df_test[ids_columns + targets_columns].values

    dateTrain = np.unique(y_train[np.argwhere(y_train[:, 5] > 0),4])
    dateVal = np.unique(y_val[np.argwhere(y_val[:, 5] > 0),4])
    dateTest = np.unique(y_test[:,4])

    XsTe = []
    YsTe = []
    EsTe = []
    
    logger.info(f'Model configuration : image_per_node {image_per_node}, use_temporal_as_edges {use_temporal_as_edges}')
 
    logger.info('Creating train dataset')
    Xst, Yst, Est = create_dataset_2D_2(graph, x_train, y_train, ks, dateTrain, df_train, scaling,
                        features_name_2D, path, use_temporal_as_edges, image_per_node, scale, context='train') 
    
    logger.info('Creating val dataset')
    # Val
    XsV, YsV, EsV = create_dataset_2D_2(graph, x_val, y_val, ks, dateVal, df_train, scaling,
                        features_name_2D, path, use_temporal_as_edges, image_per_node, scale, context='val') 

    logger.info('Creating Test dataset')
    # Test
    XsTe, YsTe, EsTe = create_dataset_2D_2(graph, x_test, y_test, ks, dateTest, df_train, scaling,
                    features_name_2D, path, use_temporal_as_edges, image_per_node, scale, context='test') 

    
    assert len(Xst) > 0
    assert len(XsV) > 0
    assert len(XsTe) > 0

    train_dataset = ReadGraphDataset_2D(Xst, Yst, Est, len(Xst), device, path / '2D_database' / scaling / 'train')
    val_dataset = ReadGraphDataset_2D(XsV, YsV, EsV, len(XsV), device, path / '2D_database' / scaling / 'val')
    testDatset = ReadGraphDataset_2D(XsTe, YsTe, EsTe, len(XsTe), device, path / '2D_database' / scaling / 'test')
    return train_dataset, val_dataset, testDatset

def train_val_data_loader(graph,
                          train,
                          val,
                          test,
                          features_name,
                        use_temporal_as_edges : bool,
                        batch_size,
                        device,
                        ks : int) -> None:

    train_dataset, val_dataset, test_dataset = create_dataset(graph,
                                                            train,
                                                            val,
                                                            test,
                                                            features_name,
                                                            use_temporal_as_edges,
                                                            device, ks)

    train_loader = DataLoader(train_dataset, batch_size, True, collate_fn=graph_collate_fn)
    val_loader = DataLoader(val_dataset, val_dataset.__len__(), False, collate_fn=graph_collate_fn)
    test_loader = DataLoader(test_dataset, test_dataset.__len__(), False, collate_fn=graph_collate_fn)
    return train_loader, val_loader, test_loader

def train_val_data_loader_2D(graph,
                            df_train,
                            df_val,
                            df_test,
                            use_temporal_as_edges : bool,
                            image_per_node : bool,
                            batch_size : int,
                            device: torch.device,
                            scaling : str,
                            features_name_2D : list,
                            features_name : list,
                            scale,
                            ks : int,
                            path : Path) -> None:
    
    train_dataset, val_dataset, test_dataset = create_dataset_2D(graph,
                                                                df_train,
                                                                df_val,
                                                                df_test,
                                                                scaling,
                                                                path,
                                                                features_name_2D,
                                                                features_name,
                                                                image_per_node,
                                                                use_temporal_as_edges,
                                                                scale,
                                                                device,
                                                                ks)
    
    train_loader = DataLoader(train_dataset, batch_size, True, collate_fn=graph_collate_fn)
    val_loader = DataLoader(val_dataset, val_dataset.__len__(), False, collate_fn=graph_collate_fn)
    test_loader = DataLoader(test_dataset, test_dataset.__len__(), False, collate_fn=graph_collate_fn)
    return train_loader, val_loader, test_loader

#########################################################################################################
#                                                                                                       #
#                                         test                                                          #
#                                                                                                       #
#########################################################################################################

def create_test(graph, df,
                device : torch.device, use_temporal_as_edges: bool,
                ks :int,) -> tuple:
    """
    """
    assert use_temporal_as_edges is not None

    Xset, Yset = df[ids_columns + features_name], df[ids_columns + targets_columns]

    print(np.nanmax(Xset[:,6:]), np.nanmin(Xset[:,6:]))
    graphId = np.unique(Xset[:,4])

    batch = []

    i = 0
    for id in graphId:
        if use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, id, Xset, Yset, ks)
        else:
            x, y, e = construct_graph_with_time_series(graph, id, Xset, Yset, ks)
        if x is None:
            i += 1
            continue

        batch.append([torch.tensor(x, dtype=torch.float32, device=device),
                        torch.tensor(y, dtype=torch.float32, device=device),
                        torch.tensor(e, dtype=torch.long, device=device)])  

    features, labels, edges = graph_collate_fn(batch)
    return features, labels, edges

def create_test_loader_2D(graph,
                df,
                df_train : pd.DataFrame,
                use_temporal_as_edges : bool,
                image_per_node : bool,
                device: torch.device,
                scaling : str,
                features_name_2D : list,
                ks : int,
                scale : int,
                path : Path) -> tuple:
    """
    """
    assert use_temporal_as_edges is not None

    x_test, y_test = df[ids_columns].values, df[ids_columns + targets_columns].values

    XsTe, YsTe, EsTe = create_dataset_2D_2(graph, x_test, y_test, ks, np.unique(x_test[:,4]), df_train, scaling,
                    features_name_2D, path, use_temporal_as_edges, image_per_node, scale) 

    dataset = ReadGraphDataset_2D(XsTe, YsTe, EsTe, len(XsTe), device, path)
    loader = DataLoader(dataset, 64, False, collate_fn=graph_collate_fn)
    return loader

def create_test_loader(graph, df,
                       features_name,
                       device : torch.device,
                       use_temporal_as_edges : bool,
                       ks :int) -> tuple:
    
    assert use_temporal_as_edges is not None

    Xset, Yset = df[ids_columns + features_name].values, df[ids_columns + targets_columns].values 

    graphId = np.unique(Xset[:,4])

    X = []
    Y = []
    E = []

    i = 0
    for id in graphId:
        if use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, id, Xset, Yset, ks)
        else:
            x, y, e = construct_graph_with_time_series(graph, id, Xset, Yset, ks)

        if x is None:
            continue

        X.append(x)
        Y.append(y)
        E.append(e)

        i += 1

    dataset = InplaceGraphDataset(X, Y, E, len(X), device)
    loader = DataLoader(dataset, 64, False, collate_fn=graph_collate_fn)

    return loader

def load_tensor_test(use_temporal_as_edges, graphScale, dir_output,
                     X, Y, device, encoding,
                     k_days, test, scaling, prefix, Rewrite):

    if not Rewrite:
        XTensor = read_object('XTensor_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
        if XTensor is not None:
            XTensor = XTensor.to(device)
            YTensor = read_object('YTensor_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output).to(device)
            ETensor = read_object('ETensor_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output).to(device)
            return XTensor, YTensor, ETensor

    XTensor, YTensor, ETensor = create_test(graph=graphScale, Xset=X, Yset=Y,
                                device=device, use_temporal_as_edges=use_temporal_as_edges, ks=k_days)

    save_object(XTensor, 'XTensor_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
    save_object(YTensor, 'YTensor_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
    save_object(ETensor, 'ETensor_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)

    return XTensor, YTensor, ETensor

def load_loader_test(use_temporal_as_edges, graphScale, dir_output,
                     df, device, k_days,
                     test, scaling, encoding, prefix, Rewrite):
    loader = None
    if Rewrite:
        loader = create_test_loader(graph=graphScale, df=df, device=device,
                                    use_temporal_as_edges=use_temporal_as_edges, ks=k_days)

        save_object(loader, 'loader_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
        loader = read_object('loader_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
    else:
        loader = read_object('loader_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
    if loader is None:
        loader = create_test_loader(graph=graphScale, df=df, device=device,
                                    use_temporal_as_edges=use_temporal_as_edges, ks=k_days)

        save_object(loader, 'loader_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)

    return loader


def load_loader_test_2D(use_temporal_as_edges, image_per_node, scale, graphScale, dir_output,
                     df, df_train, device,
                     k_days, test, features_name_2D, scaling, encoding, prefix, Rewrite):
    loader = None
    if Rewrite:
        loader = create_test_loader_2D(graphScale,
                                    df,
                                    df_train,
                                    use_temporal_as_edges,
                                    image_per_node,
                                    device,
                                    scaling,
                                    features_name_2D,
                                    k_days,
                                    scale,
                                    dir_output)

        save_object(loader, 'loader_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'_2D.pkl', dir_output)
        loader = read_object('loader_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'_2D.pkl', dir_output)
    else:
        loader = read_object('loader_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'_2D.pkl', dir_output)
    if loader is None:
        loader = create_test_loader_2D(graphScale,
                                    df,
                                    df_train,
                                    use_temporal_as_edges,
                                    image_per_node,
                                    device,
                                    scaling,
                                    features_name_2D,
                                    k_days,
                                    scale,
                                    dir_output)

        save_object(loader, 'loader_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'_2D.pkl', dir_output)

    return loader

def evaluate_pipeline(dir_train, pred, y, scale, graph_structure, methods, i, test_departement, target_name, name, testname, dir_output):
    metrics = {}
    dir_predictor = root_graph / dir_train / '../influenceClustering'

    for nameDep in test_departement:
        mask = np.argwhere(y[:, 3] == name2int[nameDep])
        if mask.shape[0] == 0:
            continue
        if target_name == 'risk':
            if scale == 'Departement':
                predictor = read_object(f'{nameDep}Predictor{scale}.pkl', dir_predictor)
            else:
                predictor = read_object(f'{nameDep}Predictor{scale}_{graph_structure}.pkl', dir_predictor)
        else:
            create_predictor(pred, name, nameDep, dir_predictor, scale, target_name == 'binary', graph_construct=graph_structure)
            if scale == 'Departement':
                predictor = read_object(f'{nameDep}Predictor{name}{scale}.pkl', dir_predictor)
            else:
                predictor = read_object(f'{nameDep}Predictor{name}{scale}_{graph_structure}.pkl', dir_predictor)

        pred[mask[:,0], 1] = order_class(predictor, predictor.predict(pred[mask[:, 0], 0]))

    metrics = add_metrics(methods, i, pred, y, test_departement, target_name, scale, name, dir_train)

    if target_name == 'binary':
        y[:,-1] = y[:,-1] / np.nanmax(y[:,-1])

    realVspredict(pred[:, 0], y, -1,
                dir_output / name, f'raw_{scale}')
            
    realVspredict(pred[:, 1], y, -3,
                dir_output / name, f'class_{scale}')
    
    realVspredict(pred[:, 0], y, -2,
                dir_output / name, f'nbfire_{scale}')

    sinister_distribution_in_class(pred[:, 1], y, dir_output / name, f'mean_fire_{scale}')

    if target_name == 'binary':
        y_test = (y[:, -2] > 0).astype(int)
    elif target_name == 'nbsinister':
        y_test = y[:, -2]
    elif target_name == 'risk':
        y_test = y[:, -1]
    else:
        raise ValueError(f'Unknow {target_name}')
    
    res = pd.DataFrame(columns=ids_columns + targets_columns + ['prediction', 'class', 'mae', 'rmse'], index=np.arange(pred.shape[0]))
    res[ids_columns + targets_columns] = y
    res['prediction'] = pred[:, 0]
    res['class'] = pred[:, 1]
    res['mae'] = np.sqrt((y[:,-4] * (pred[:,0] - y[:,-1]) ** 2))
    res['rmse'] = np.sqrt(((pred[:,0] - y[:,-1]) ** 2))

    if testname == '69':
        start = '2018-01-01'
        stop = '2022-09-30'
    else:
        start = '2023-06-01'
        stop = '2023-09-30'
    
    """for departement in testd_departement:
        susectibility_map(departement, sinister,
                    dir_train, dir_output, k_days,
                    target_name, scale, name, res, n,
                    start, stop, resolution)"""

    shapiro_wilk(pred[:,0], y[:,-1], dir_output / name, f'shapiro_wilk_{scale}')

    """
    realVspredict2d(pred,
        y,
        target_name,
        name,
        scale,
        dir_output / n,
        dir_train,
        geo,
        testd_departement,
        graphScale)"""
    return metrics, res

def test_sklearn_api_model(graphScale,
                          test_dataset_dept,
                          test_dataset_unscale_dept,
                           methods,
                           testname,
                           prefix_train,
                           models,
                           dir_output,
                           device,
                           encoding,
                           scaling,
                           test_departement,
                           dir_train,
                           dir_break_point,
                           doKMEANS):
    
    scale = graphScale.scale
    metrics = {}
    metrics_dept = {}

    ################################ Ground Truth ############################################
    logger.info('#########################')
    logger.info(f'       GT              ')
    logger.info('#########################')

    y = test_dataset_dept[ids_columns + targets_columns].values
    pred = np.empty((y.shape[0], 2))
    pred[:, 0] = test_dataset_dept['risk'].values
    pred[:, 1] = test_dataset_dept['class_risk'].values
    metrics['GT'] = add_metrics(methods, 0, pred, y, test_departement, 'risk', scale, 'gt', dir_train)
    i = 0

    #################################### Traditionnal ##################################################

    for name, target_name, autoRegression in models:

        if MLFLOW:
            existing_run = get_existing_run(name)
            if existing_run:
                mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
            else:
                mlflow.start_run(run_name=name, nested=True)

        y = test_dataset_dept[ids_columns + targets_columns].values
        model_dir = dir_train / Path('check_'+scaling + '/' + prefix_train + '/baseline/' + name + '/')
        
        logger.info('#########################')
        logger.info(f'      {name}            ')
        logger.info('#########################')

        model = read_object(name+'.pkl', model_dir)

        if model is None:
            continue

        graphScale._set_model(model)

        features_selected = read_object('features.pkl', model_dir)
        
        pred = np.empty((y.shape[0], 2))

        pred[:, 0] = graphScale.predict_model_api_sklearn(test_dataset_dept, features_selected, target_name == 'binary', autoRegression)
        logger.info(f'pred min {np.nanmin(pred[:, 0])}, pred max : {np.nanmax(pred[:, 0])}')

        if doKMEANS:
            df = pd.DataFrame(columns=ids_columns + targets_columns, index=np.arange(pred.shape[0]))
            df[ids_columns] = test_dataset_dept[ids_columns]
            features_selected_kmeans,_ = get_features_name_list(scale, kmeans_features, METHODS_KMEANS_TRAIN)
            df[features_selected_kmeans] = test_dataset_unscale_dept[features_selected_kmeans]
            df[target_name] = pred[:, 0]
            df = apply_kmeans_class_on_target(df, dir_break_point=dir_break_point, target=target_name, tresh=tresh_kmeans,
                                              features_selected=features_selected_kmeans, new_val=0)

            pred[:, 0] = df[target_name]

        metrics[name], res = evaluate_pipeline(dir_train, pred, y, scale, graphScale.base,
                                               methods, i, test_departement, target_name, name, testname,
                                               dir_output)

        if MLFLOW:
            log_metrics_recursively(metrics[name], prefix='')

        if len(features_selected) < 30:
            if target_name == 'binary':
                band = -2
            else:
                band = -1
            pass
            #model.plot_features_importance(test_dataset_dept[features_selected], y[:, band], 'test', dir_output / name, mode='bar')
            #model.shapley_additive_explanation(test_dataset_dept[features_selected], 'test', dir_output, mode = 'bar', figsize=(50,25), samples=200)

        # Analyse departement prediction
        res_dept = res.groupby(['departement', 'date'])['risk'].sum().reset_index()
        res_dept['id'] = res_dept['departement']
        res_dept['latitude'] = res.groupby(['departement'])['latitude'].mean().reset_index()['latitude']
        res_dept['longitude'] = res.groupby(['departement'])['longitude'].mean().reset_index()['longitude']
        res_dept['nbsinister'] = res.groupby(['departement', 'date'])['nbsinister'].sum().reset_index()['nbsinister']

        dir_predictor = root_graph / dir_train / '../influenceClustering'
        for nameDep in test_departement:
            mask = np.argwhere(y[:, 3] == name2int[nameDep])
            if mask.shape[0] == 0:
                continue
            predictor = read_object(nameDep+'PredictorDepartement.pkl', dir_predictor)

        res_dept['class_risk'] = order_class(predictor, predictor.predict(res_dept['risk'].values, 0))
        res_dept['weight'] =  predictor.weight_array(predictor.predict(res_dept['risk'].values, 0).astype(int)).reshape(-1)

        if target_name == 'binary':
            res_dept['prediction'] = res.groupby(['departement', 'date']).apply(group_probability).reset_index()['prediction']
        else:
            res_dept['prediction'] = res.groupby(['departement', 'date']).sum().reset_index()['prediction']

        metrics_dept[f'{name}'], res = evaluate_pipeline(dir_train, res_dept[['prediction', 'class_risk']].values,
                                                                res_dept[ids_columns + targets_columns].values, 'Departement',
                                                                graphScale.base,
                                                                methods, i, test_departement, target_name, name, testname,
                                                                dir_output)

        if MLFLOW:
            log_metrics_recursively(metrics_dept[name], prefix='departement')

        save_object(res, name+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output / name)

        i += 1

        mlflow.end_run()

    ########################################## Save metrics ################################ 
    outname = 'metrics'+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+testname+'_tree.pkl'
    save_object(metrics, outname, dir_output)

    outname = 'metrics'+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+testname+'_dept_tree.pkl'
    save_object(metrics_dept, outname, dir_output)

    return metrics, metrics_dept

def test_dl_model(graphScale,test_dataset_dept,
                          test_dataset_unscale_dept,
                          train_dataset,
                           methods,
                           testname,
                           features_name,
                           prefix,
                           prefix_train,
                           models,
                           dir_output,
                           device,
                           k_days,
                           encoding,
                           scaling,
                           test_departement,
                           dir_train,
                           features_name_2D,
                           doKMEANS,
                           dir_break_point,
                           spec,
                           ):
    
    scale = graphScale.scale
    metrics = {}
    metrics_dept = {}

    ################################ Ground Truth ############################################
    logger.info('#########################')
    logger.info(f'       GT              ')
    logger.info('#########################')

    y = test_dataset_dept[ids_columns + targets_columns].values
    pred = np.empty((y.shape[0], 2))
    pred[:, 0] = test_dataset_dept['risk'].values
    pred[:, 1] = test_dataset_dept['class_risk'].values
    metrics['GT'] = add_metrics(methods, 0, pred, y, test_departement, 'risk', scale, 'gt', dir_train / spec)
    i = 1

    #################################### GNN ###################################################
    for name, use_temporal_as_edges, target_name, is_2D_model, autoRegression in models:

        model_dir = dir_train / spec / f'check_{scaling}' / prefix_train / name

        logger.info('#########################')
        logger.info(f'       {name}          ')
        logger.info('#########################')

        if not is_2D_model:
            test_loader = load_loader_test(use_temporal_as_edges=use_temporal_as_edges, graphScale=graphScale, dir_output=dir_output,
                                        df=test_dataset_dept,
                                        device=device, k_days=k_days, test=testname, features_name=features_name,
                                        scaling=scaling, encoding=encoding, prefix=prefix, Rewrite=False)

        else:
            test_loader = load_loader_test_2D(use_temporal_as_edges=use_temporal_as_edges, graphScale=graphScale,
                                              dir_output=dir_train, df=test_dataset_dept,
                                            df_train=train_dataset, device=device, k_days=k_days, test=testname,
                                            scaling=scaling, prefix=prefix,
                                            features_name_2D=features_name_2D,
                                            encoding=encoding,
                                            Rewrite=False)

        graphScale._load_model_from_path(model_dir / 'best.pt', dico_model[name], device)

        features_selected = read_object('features.pkl', model_dir)

        dico_model = make_models(len(features_selected), len(features_selected), scale, 0.03, 'relu', k_days, target_name == 'binary')

        predTensor, YTensor = graphScale._predict_test_loader(test_loader, features_selected, device=device, target_name=target_name, 
                                                              autoRegression=autoRegression, features_name=features_name)
        if doKMEANS:
            df = pd.DataFrame(columns=ids_columns + targets_columns, index=np.arange(pred.shape[0]))
            df[ids_columns] = test_dataset_dept[ids_columns]
            features_selected_kmeans,_ = get_features_name_list(scale, kmeans_features, METHODS_KMEANS_TRAIN)
            df[features_selected_kmeans] = test_dataset_unscale_dept[features_selected_kmeans]
            df[target_name] = pred[:, 0]
            df = apply_kmeans_class_on_target(df, dir_break_point=dir_break_point, target=target_name, tresh=tresh_kmeans,
                                              features_selected=features_selected_kmeans, new_val=0)
            
            pred[:, 0] = df[target_name]

        y = YTensor.detach().cpu().numpy()
        pred = np.empty((y.shape[0], 2))
        pred[:, 0] = predTensor.detach().cpu().numpy()

        metrics[name], res = evaluate_pipeline(dir_train, pred, y, scale, graphScale.base,
                                               methods, i, test_departement, target_name, name, testname, dir_output)

        if MLFLOW:
            log_metrics_recursively(metrics[name], prefix='')

        # Analyse departement prediction
        res_dept = res.groupby(['departement', 'date'])['risk'].sum().reset_index()
        res_dept['id'] = res_dept['departement']
        res_dept['latitude'] = res.groupby(['departement'])['latitude'].mean().reset_index()['latitude']
        res_dept['longitude'] = res.groupby(['departement'])['longitude'].mean().reset_index()['longitude']
        res_dept['nbsinister'] = res.groupby(['departement', 'date'])['nbsinister'].sum().reset_index()['nbsinister']

        dir_predictor = root_graph / dir_train / '../influenceClustering'
        for nameDep in test_departement:
            mask = np.argwhere(y[:, 3] == name2int[nameDep])
            if mask.shape[0] == 0:
                continue
            predictor = read_object(nameDep+'PredictorDepartement.pkl', dir_predictor)

        res_dept['class_risk'] = order_class(predictor, predictor.predict(res_dept['risk'].values, 0))
        res_dept['weight'] =  predictor.weight_array(predictor.predict(res_dept['risk'].values, 0).astype(int)).reshape(-1)

        if target_name == 'binary':
            res_dept['prediction'] = res.groupby(['departement', 'date']).apply(group_probability).reset_index()['prediction']
        else:
            res_dept['prediction'] = res.groupby(['departement', 'date']).sum().reset_index()['prediction']

        metrics_dept[f'{name}_departement'], res = evaluate_pipeline(dir_train, res_dept[['prediction', 'class_risk']].values,
                                                                res_dept[ids_columns + targets_columns].values, 'Departement',
                                                                graphScale.base,
                                                                methods, i, test_departement, target_name, name, testname,
                                                                dir_output)

        save_object(res, name+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output / name)

        if MLFLOW:
            log_metrics_recursively(metrics_dept[name], prefix='departement')
        
        i += 1

    ########################################## Save metrics ################################ 
    outname = 'metrics'+'_'+prefix_train+'_'+'_'+scaling+'_'+encoding+'_'+testname+'_dl.pkl'
    save_object(metrics, outname, dir_output / name)

def test_simple_model(graphScale, Xset, Yset,
                           methods,
                           testname,
                           features_name,
                           prefix_train,
                           dummy,
                           dir_output,
                           encoding,
                           scaling,
                           testd_departement,
                           scale,
                           dir_train
                           ):
    metrics = {}
    y = Yset[Yset[:,-4] > 0]
    x = Xset[Yset[:,-4] > 0]
    YTensor = torch.tensor(y, dtype=torch.float32, device=device)
    if not dummy:

        ITensor = Yset[:,-4]
        if testname != 'dummy':

            logger.info('#########################')
            logger.info(f'      perf_Y            ')
            logger.info('#########################')
            name = 'perf_Y'
        
            pred = graphScale._predict_perference_with_Y(y, False)
            predTensor = torch.tensor(pred, dtype=torch.float32, device=device)
            metrics[name] = add_metrics(methods, 0, predTensor, YTensor, testd_departement, False, scale, name, dir_train)

            res = np.empty((pred.shape[0], y.shape[1] + 3))
            res[:, :y.shape[1]] = y
            res[:,y.shape[1]] = pred
            res[:,y.shape[1]+1] = np.sqrt((y[:,-4] * (pred - y[:,-1]) ** 2))
            res[:,y.shape[1]+2] = np.sqrt(((pred - Yset[:,-1]) ** 2))

            #save_object(res, name+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output)
            save_object(res, name+'_pred.pkl', dir_output)

            logger.info('#########################')
            logger.info(f'      perf_Y_bin         ')
            logger.info('#########################')
            name = 'perf_Y_bin'
            pred = graphScale._predict_perference_with_Y(y, True)

            predTensor = torch.tensor(pred, dtype=torch.float32, device=device)
            metrics[name] = add_metrics(methods, 1, predTensor, YTensor, testd_departement, True, scale, name, dir_train)

            res = np.empty((pred.shape[0], y.shape[1] + 3))
            res[:, :y.shape[1]] = Yset
            res[:,y.shape[1]] = pred
            res[:,y.shape[1]+1] = np.sqrt((y[:,-4] * (pred - y[:,-1]) ** 2))
            res[:,y.shape[1]+2] = np.sqrt(((pred - y[:,-1]) ** 2))

            #save_object(res, name+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output)
            save_object(res, name+'_pred.pkl', dir_output)

        logger.info('#########################')
        logger.info(f'      perf_X         ')
        logger.info('#########################')
        name = 'perf_X'
        pred = graphScale._predict_perference_with_X(x, features_name)

        predTensor = torch.tensor(pred, dtype=torch.float32, device=device)
        metrics[name] = add_metrics(methods, 2, predTensor, YTensor, testd_departement, False, scale, name, dir_train)

        res = np.empty((pred.shape[0], y.shape[1] + 3))
        res[:, :y.shape[1]] = y
        res[:,y.shape[1]] = pred
        res[:,y.shape[1]+1] = np.sqrt((y[:,-4] * (pred - y[:,-1]) ** 2))
        res[:,y.shape[1]+2] = np.sqrt(((pred - y[:,-1]) ** 2))

        save_object(res, name+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output)
        outname = 'metrics'+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+testname+'_metrics.pkl'
        save_object(metrics, outname, dir_output)

def test_break_points_model(test_dataset_dept,
                           methods,
                           testname,
                           prefix_train,
                           dir_output,
                           encoding,
                           scaling,
                           test_departement,
                           scale,
                           dir_train,
                           features):    
    metrics = {}
    
    n = 'break_point'
    name = 'break_point'
    dir_break_point = dir_train / 'check_none' / prefix_train / 'kmeans'

    pred = np.zeros((len(test_dataset_dept), 2))
    pred[:, 0] = test_dataset_dept['risk']
    pred[:, 1] = test_dataset_dept['class_risk']
    y = test_dataset_dept[ids_columns + targets_columns].values
            
    metrics[name], res = evaluate_pipeline(dir_train, pred, y, scale, methods, 0, test_departement, 'risk', name, testname, dir_output)

    #save_object(res, name+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output)
    save_object(res, name+'_pred.pkl', dir_output)
    outname = 'metrics'+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+testname+'_bp.pkl'
    save_object(metrics, outname, dir_output)

def test_fusion_prediction(models,
                           args1,
                           args2,
                           XsetDaily,
                           XsetLocalized,
                           Xset,
                           Yset,
                           dir_output,
                           dir_train,
                           graphScale,
                        testd_departement,
                        testname,
                        methods,
                        prefix_train,
                        features_name,):
    
    scale = graphScale.scale

    y = Yset[Yset[:,-4] > 0]
    
    dir_predictor = root_graph / args2['dir_train'] / 'influenceClustering'

    metrics = {}

    # Graphs

    graph1 = read_object(args1['graph'], args1['dir_train'])
    graph2 = read_object(args1['graph'], args1['dir_train'])

    # Models
    n1 = args1['model_name']+'.pkl'
    n2 = args2['model_name']+'.pkl'
    model1 = read_object(n1, args1['dir_model'] /  args1['model_name'])
    model2 = read_object(n2, args2['dir_model'] / args2['model_name'])
    for name, target_name, autoRegression in models:
        # Prediction
        if args1['model_type'] == 'traditionnal':
            features_selected = read_object('features.pkl', args1['dir_model'] / args1['model_name'])
            graph1._set_model(model1)
            pred1 = np.empty((XsetDaily.shape[0], y.shape[1]))
            pred1[:, :6] = XsetDaily[:, :6]
            pred1[:, -1] = graph1.predict_model_api_sklearn(X=XsetDaily,
                                                            features=features_selected, autoRegression=args1['autoRegression'],
                                                            features_name=args1['features_name'], isBin=target_name == 'binary')
        else:
            test_loader = read_object('test_loader.pkl', args1['dir_train'])
            features_selected = read_object('features.pkl', args1['dir_model'] / args1['model_name'])

            graph1._load_model_from_path(args1['dir_train'] / 'best.pt', args1['model_name']+'.pkl', device)

            predTensor, YTensor = graph1._predict_test_loader(test_loader, features_selected, device=device, isBin=target_name == 'binary',
                                                            autoRegression=args1['autoRegression'], features_name=args1['features_name'])

            pred1 = np.copy(y)
            pred1[:, -1] = predTensor.detach().cpu().numpy()

        if args2['model_type'] == 'traditionnal':
            features_selected = read_object('features.pkl', args2['dir_model'] / args2['model_name'])
            graph2._set_model(model2)
            pred2 = graph2.predict_model_api_sklearn(X=XsetLocalized,
                                                            features=features_selected, autoRegression=args2['autoRegression'],
                                                            features_name=args2['features_name'], isBin=target_name == 'binary')
        else:
            test_loader = read_object('test_loader.pkl', args2['dir_model'])
            features_selected = read_object('features.pkl', args2['dir_model'] / args2['model_name'])

            graph2._load_model_from_path(args2['dir_train'] / 'best.pt', args2['model_name'], device)

            predTensor, YTensor = graph2._predict_test_loader(test_loader, features_selected, device=device, isBin=target_name == 'binary',
                                                            autoRegression=args2['autoRegression'], features_name=args2['features_name'])
            y = YTensor.detach().cpu().numpy()
            pred2 = predTensor.detach().cpu().numpy()
        # Fusion
        Xset = add_temporal_spatial_prediction(Xset[:, :-2], pred2.reshape(-1,1), pred1, graph1, features_name=features_name, target_name=target_name)
        i = 0
    
        y = Yset[Yset[:,-4] > 0]
        n = name
        model_dir = dir_train / Path('check_'+scaling + '/' + prefix_train + '/baseline/' + name + '/')
        
        logger.info('#########################')
        logger.info(f'      {name}            ')
        logger.info('#########################')
    
        model = read_object(name+'.pkl', model_dir)

        if model is None:
            continue

        graphScale._set_model(model)

        features_selected = read_object('features.pkl', model_dir)

        pred = np.empty((Xset[:, features_selected].shape[0], 2))

        pred[:, 0] = graphScale.predict_model_api_sklearn(X=Xset,
                                                            features=features_selected, autoRegression=autoRegression,
                                                            features_name=features_name, isBin=target_name == 'binary')

        logger.info(f'pred min {np.nanmin(pred[:, 0])}, pred max : {np.nanmax(pred[:, 0])}')

        for nameDep in departements:
            mask = np.argwhere(y[:, 3] == name2int[nameDep])
            if mask.shape[0] == 0:
                continue
            if not target_name:
                predictor = read_object(f'{nameDep}Predictor{scale}_{graph_structure}.pkl', dir_predictor)
            else:
                create_predictor(pred, name, nameDep, dir_predictor, scale, target_name == 'binary')
                predictor = read_object(f'{nameDep}Predictor{name}{scale}_{graph_structure}.pkl', dir_predictor)

            pred[mask[:,0], 1] = order_class(predictor, predictor.predict(pred[mask[:, 0], 0]))

        metrics[name] = add_metrics(methods, i, pred, y, testd_departement, target_name, scale, name, None)

        if target_name == 'binary':
            y[:,-1] = y[:,-1] / np.nanmax(y[:,-1])

        realVspredict(pred[:, 0], y, -1,
                      dir_output / n, 'raw')
                
        realVspredict(pred[:, 1], y, -3,
                      dir_output / n, 'class')

        sinister_distribution_in_class(pred[:, 1], y, dir_output / n)

        res = np.empty((pred.shape[0], y.shape[1] + 4))
        res[:, :y.shape[1]] = y
        res[:,y.shape[1]] = pred[:, 0]
        res[:,y.shape[1] + 1] = pred[:, 1]
        res[:,y.shape[1]+2] = np.sqrt((y[:,-4] * (pred[:,0] - y[:,-1]) ** 2))
        res[:,y.shape[1]+3] = np.sqrt(((pred[:,0] - y[:,-1]) ** 2))

        save_object(res, name+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output / n)

        if testname == '69':
            start = '2018-01-01'
            stop = '2022-09-30'
        else:
            start = '2023-06-01'
            stop = '2023-09-30'
        
        """for departement in testd_departement:
            susectibility_map(departement, sinister,
                        dir_train, dir_output, k_days,
                        target_name, scale, name, res, n,
                        start, stop, resolution)"""

        shapiro_wilk(pred[:,0], y[:,-1], dir_output / n)

        """
        realVspredict2d(pred,
            y,
            target_name,
            name,
            scale,
            dir_output / n,
            dir_train,
            geo,
            testd_departement,
            graphScale)"""

        i += 1
    #save_object(res, name+'_pred.pkl', dir_output)
    outname = 'metrics'+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+testname+'_fusion.pkl'
    save_object(metrics, outname, dir_output)

#########################################################################################################
#                                                                                                       #
#                                         Inference                                                     #
#                                                                                                       #
#########################################################################################################

def create_inference(graph,
                     X : np.array,
                     device : torch.device,
                     use_temporal_as_edges,
                     graphId : int,
                     ks : int,) -> tuple:

    graphId = [graphId]

    batch = []

    for id in graphId:
        if use_temporal_as_edges:
            x, e = construct_graph_set(graph, id, X, None, ks)
        else:
            x, e = construct_graph_with_time_series(graph, id, x, None, ks)
        if x is None:
            continue
        batch.append([torch.tensor(x, dtype=torch.float32, device=device), torch.tensor(e, dtype=torch.long, device=device)])
    if len(batch) == 0:
        return None, None
    
    features, edges = graph_collate_fn_no_label(batch)

    return features[:,:,0], edges


def create_inference_2D(graph, X : np.array, x_train : np.array, y_train : np.array,
                     device : torch.device, use_temporal_as_edges, ks : int, features_name : dict,
                     scaling : str) -> tuple:

    pass

    return


############################################################################################################
#                                                                                                          #
#                                           WRAPPED TRAIN                                                  # 
#                                                                                                          #
############################################################################################################

import torch
import logging
from pathlib import Path

# Assuming these functions are defined somewhere else in your code
# from your_module import read_object, save_object, train_val_data_loader, train_val_data_loader_2D, weighted_rmse_loss, weighted_cross_entropy, train

logger = logging.getLogger(__name__)

def wrapped_train_deep_learning_1D(params):
    model = params['model']
    use_temporal_as_edges = params['use_temporal_as_edges']
    infos = params['infos']
    autoRegression = params['autoRegression']

    target_name, task_type, loss = infos.split('_')
    loss_name = loss

    logger.info(f'Fitting model {model}')
    logger.info('Try loading loader')

    train_loader = read_object(f'train_loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{str(use_temporal_as_edges)}.pkl', params['dir_output'])
    val_loader = read_object(f'val_loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{str(use_temporal_as_edges)}.pkl', params['dir_output'])

    if train_loader is None or params['Rewrite']:
        logger.info(f'Building loader, loading loader_{str(use_temporal_as_edges)}.pkl')
        train_loader, val_loader, test_loader = train_val_data_loader(
            graph=params['graphScale'],
            train=params['train_dataset'],
            val=params['val_dataset'],
            test=params['test_dataset'],
            features_name=params['features_selected'],
            batch_size=64,
            device=params['device'],
            use_temporal_as_edges=use_temporal_as_edges,
            ks=params['k_days']
        )
        save_object(train_loader, f'train_loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{str(use_temporal_as_edges)}.pkl', params['dir_output'])
        save_object(val_loader, f'val_loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{str(use_temporal_as_edges)}.pkl', params['dir_output'])

    train_params = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "k_days": params['k_days'],
        "optimize_feature": params['optimize_feature'],
        "PATIENCE_CNT": params['PATIENCE_CNT'],
        "CHECKPOINT": params['CHECKPOINT'],
        "epochs": params['epochs'],
        "features": params['features_selected'],
        "lr": params['lr'],
        "features_name": params['features_name'],
        "loss_name": loss_name,
        "modelname": model,
        "dir_output": params['dir_output'] / Path(f'check_{params["scaling"]}/{params["prefix"]}/ {model}_infos'),
        "target_name": target_name,
        'infos' : infos,
        "autoRegression": autoRegression,
        'k_days' : params['k_days']

    }

    train(train_params)

def wrapped_train_deep_learning_2D(params):
    model = params['model']
    use_temporal_as_edges = params['use_temporal_as_edges']
    infos = params['infos']
    autoRegression = params['autoRegression']
    image_per_node = params['image_per_node']

    target_name, task_type, loss = infos.split('_')
    loss_name = loss

    logger.info(f'Fitting model {model}')
    logger.info('Try loading loader')

    train_loader = read_object(f'train_loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{image_per_node}_{use_temporal_as_edges}_2D.pkl', params['dir_output'])
    val_loader = read_object(f'val_loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{image_per_node}_{use_temporal_as_edges}_2D.pkl', params['dir_output'])
    test_loader = read_object(f'test_loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{image_per_node}_{use_temporal_as_edges}_2D.pkl', params['dir_output'])

    if (train_loader is None or val_loader is None or test_loader is None) or params['Rewrite']:
        logger.info(f'Building loader, loading loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{image_per_node}_{use_temporal_as_edges}_2D.pkl')
        train_loader, val_loader, test_loader = train_val_data_loader_2D(
            graph=params['graphScale'],
            df_train=params['train_dataset_unscale'],
            df_val=params['val_dataset'],
            df_test=params['test_dataset'],
            use_temporal_as_edges=use_temporal_as_edges,
            image_per_node=image_per_node,
            scale=params['scale'],
            batch_size=64,
            device=params['device'],
            scaling=params['scaling'],
            features_name=params['features_name'],
            features_name_2D=params['features_name_2D'],
            ks=params['k_days'],
            path=Path(params['name_dir'])
        )

        save_object(train_loader, f'train_loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{image_per_node}_{use_temporal_as_edges}_2D.pkl', params['dir_output'])
        save_object(val_loader, f'val_loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{image_per_node}_{use_temporal_as_edges}_2D.pkl', params['dir_output'])
        save_object(test_loader, f'test_loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{image_per_node}_{use_temporal_as_edges}_2D.pkl', params['dir_output'])

    train_params = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "k_days": params['k_days'],
        "optimize_feature": params['optimize_feature'],
        "PATIENCE_CNT": params['PATIENCE_CNT'],
        "CHECKPOINT": params['CHECKPOINT'],
        "epochs": params['epochs'],
        "features": params['features_selected'],
        "lr": params['lr'],
        "features_name": params['features_name_2D'],
        "loss_name": loss_name,
        "modelname": model,
        'scale' : params['scale'],
        "dir_output": params['dir_output'] / Path(f'check_{params["scaling"]}/{params["prefix"]}/ {model}_infos'),
        "target_name": target_name,
        "autoRegression": autoRegression,
        'k_days' : params['k_days']
    }

    train(train_params)