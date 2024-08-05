from torch_geometric.data import Dataset
from torch.utils.data import DataLoader
import torch
from tools_inference import *

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

        file = self.X[index]
        x = read_object(file, self.path)
        x = x[:,:,:, 6:, :]
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
def get_train_val_test_set(graphScale, df, train_features, train_departements, prefix, dir_output, args):
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

    dir_output =  dir_output / spec

    features_name, newshape = get_features_name_list(graphScale.scale, train_features, METHODS)

    save_object(features, 'features.pkl', dir_output)
    save_object(train_features, 'train_features.pkl', dir_output)
    save_object(features_name, 'features_name_train.pkl', dir_output)

    if days_in_futur > 1:
        prefix += '_'+str(days_in_futur)
        prefix += '_'+futur_met

    df = df[ids_columns + features_name + targets_columns]
    logger.info(np.nanmax(df[features_name]))
    logger.info(np.unique(df['id']))

    # Preprocess
    train_dataset, val_dataset, test_dataset = preprocess(df=df, scaling=scaling, maxDate=maxDate,
                                                    trainDate=trainDate, train_departements=train_departements,
                                                    departements = departements,
                                                    ks=k_days, dir_output=dir_output, prefix=prefix, features_name=features_name,
                                                    days_in_futur=days_in_futur,
                                                    futur_met=futur_met)
    
    realVspredict(test_dataset['risk'].values, test_dataset[ids_columns + targets_columns].values, -1, dir_output / prefix, 'raw')

    realVspredict(test_dataset['class_risk'].values, test_dataset[ids_columns +targets_columns].values, -3, dir_output / prefix, 'class')

    sinister_distribution_in_class(test_dataset['class_risk'].values, test_dataset[ids_columns + targets_columns].values, dir_output / prefix)

    features_selected = features_selection(doFet, train_dataset, dir_output / prefix, features_name, nbfeatures, 'risk')

    prefix = f'{str(values_per_class)}_{str(k_days)}_{str(scale)}_{str(nbfeatures)}'

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
        features_name, _ = get_features_name_list(0, train_features, [])
        features_selected = features_name
        kmeans_features = train_features

    if doKMEANS:
        prefix += f'_kmeans_{str(ncluster)}'
        train_break_point(train_dataset, features_selected, kmeansFeatures, dir_output / 'varOnValue' / prefix, features_name, scale, ncluster)

        train_dataset = apply_kmeans_class_on_target(train_dataset, dir_output / 'varOnValue' / prefix, True)
        val_dataset = apply_kmeans_class_on_target(val_dataset, dir_output / 'varOnValue' / prefix, True)
        test_dataset = apply_kmeans_class_on_target(test_dataset, dir_output / 'varOnValue' / prefix, True)

        realVspredict(test_dataset['risk'], test_dataset[ids_columns + targets_columns], -1, dir_output / prefix, 'raw')

        realVspredict(test_dataset['class_risk'], test_dataset[ids_columns +targets_columns], -3, dir_output / prefix, 'class')

        sinister_distribution_in_class(test_dataset['class_risk'], test_dataset[ids_columns + targets_columns], dir_output / prefix)
        
    logger.info(f'Train dates are between : {allDates[int(np.min(train_dataset["date"]))], allDates[int(np.max(train_dataset["date"]))]}')
    logger.info(f'Val dates are bewteen : {allDates[int(np.min(val_dataset["date"]))], allDates[int(np.max(val_dataset["date"]))]}')

    return train_dataset, val_dataset, test_dataset, features_selected

#########################################################################################################
#                                                                                                       #
#                                         Preprocess                                                    #
#                                                                                                       #
#########################################################################################################

def preprocess(df: pd.DataFrame, scaling: str, maxDate: str, trainDate: str, train_departements: list, departements: list, ks: int,
               dir_output: Path, prefix: str, features_name: list, days_in_futur: int, futur_met: str):
    """
    Preprocess the input features:
        df: combined dataframe of features and target
        scaling: scale method
        maxDate: maximum date for validation set
        trainDate: date to split training and validation
        train_departements: list of departments for training
        departements: list of all departments
        ks: offset for validation and test sets
        dir_output: directory to save the outputs
        prefix: prefix for the saved files
        features_name: list of feature names
        days_in_futur: days in future for the target
        futur_met: future metric for the target
    """

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

    if scaler is not None:
        # Scale Features
        for feature in features_name:
            if features_name.index(feature) == autoregressiveBand:
                logger.info(f'Avoid scale for Autoregressive feature')
                continue
            df[feature] = scaler(df[feature].values, df[train_mask][feature].values, concat=False)

    logger.info(f'Max of the train set: {df[train_mask][features_name].max()}, Max of the val set: {df[val_mask][features_name].max()}, Max of the test set: {df[test_mask][features_name].max()}')
    logger.info(f'Size of the train set: {df[train_mask].shape[0]}, size of the val set: {df[val_mask].shape[0]}, size of the test set: {df[test_mask].shape[0]}')
    logger.info(f'Unique train dates: {np.unique(df[train_mask]["date"]).shape[0]}, val dates: {np.unique(df[val_mask]["date"]).shape[0]}, test dates: {np.unique(df[test_mask]["date"]).shape[0]}')

    return df[train_mask], df[val_mask], df[test_mask]

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
                    use_temporal_as_edges : bool,
                    device,
                    ks : int):
    """
    """

    x_train, y_train = train
    
    x_val, y_val = val

    x_test, y_test = test

    dateTrain = np.unique(x_train[np.argwhere(x_train[:, 5] > 0),4])
    dateVal = np.unique(x_val[np.argwhere(x_val[:, 5] > 0),4])
    dateTest = np.unique(x_test[:,4])
    
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

def train_val_data_loader(graph,
                          train,
                          val,
                          test,
                        use_temporal_as_edges : bool,
                        batch_size,
                        device,
                        ks : int) -> None:

    train_dataset, val_dataset, test_dataset = create_dataset(graph,
                                train,
                                val,
                                test, use_temporal_as_edges,
                                device, ks)

    trainLoader = DataLoader(train_dataset, batch_size, True, collate_fn=graph_collate_fn)
    valLoader = DataLoader(val_dataset, val_dataset.__len__(), False, collate_fn=graph_collate_fn)
    testLoader = DataLoader(test_dataset, test_dataset.__len__(), False, collate_fn=graph_collate_fn)
    return trainLoader, valLoader, testLoader


def train_val_data_loader_2D(graph,
                            train,
                            val,
                            test,
                            use_temporal_as_edges : bool,
                            batch_size : int,
                            device: torch.device,
                            scaling : str,
                            features_name : dict,
                            features_name_2D : dict,
                            ks : int,
                            shape : tuple,
                            path : Path) -> None:
    

    x_train, y_train = train

    x_val, y_val = val

    x_test, y_test = test

    dateTrain = np.unique(x_train[:,4])
    dateVal = np.unique(x_val[:,4])
    dateTest = np.unique(x_test[:,4])

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

        x = load_x_from_pickle(x, shape, path, x_train, y_train, scaling, features_name, features_name_2D)
        save_object(x, str(id)+'.pkl', path / scaling)
    
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

        x = load_x_from_pickle(x, shape, path, x_train, y_train, scaling, features_name, features_name_2D)
        save_object(x, str(id)+'.pkl', path / scaling)
        
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

        x = load_x_from_pickle(x, shape, path, x_train, y_train, scaling, features_name, features_name_2D)
        save_object(x, str(id)+'.pkl', path / scaling)
    
        XsTe.append(x)
        YsTe.append(y)
        EsTe.append(e)
    
    assert len(Xst) > 0
    assert len(XsV) > 0
    assert len(XsTe) > 0

    
    assert len(Xst) > 0
    assert len(XsV) > 0
    
    train_dataset = ReadGraphDataset_2D(Xst, Yst, Est, len(Xst), device, path / scaling)
    val_dataset = ReadGraphDataset_2D(XsV, YsV, EsV, len(XsV), device, path  / scaling)
    test_dataset = ReadGraphDataset_2D(XsTe, YsTe, EsTe, len(XsTe), device, path  / scaling)
    
    trainLoader = DataLoader(train_dataset, batch_size, True, collate_fn=graph_collate_fn)
    valLoader = DataLoader(val_dataset, val_dataset.__len__(), False, collate_fn=graph_collate_fn)
    testLoader = DataLoader(test_dataset, test_dataset.__len__(), False, collate_fn=graph_collate_fn)
    return trainLoader, valLoader, testLoader

#########################################################################################################
#                                                                                                       #
#                                         test                                                          #
#                                                                                                       #
#########################################################################################################

def create_test(graph, Xset : np.array, Yset : np.array,
                device : torch.device, use_temporal_as_edges: bool,
                ks :int,) -> tuple:
    """
    """
    assert use_temporal_as_edges is not None

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
                Xset : np.array,
                Yset : np.array,
                x_train : np.array,
                y_train : np.array,
                use_temporal_as_edges : bool,
                device: torch.device,
                scaling : str,
                features_name : dict,
                features_name_2D : dict,
                ks : int,
                shape : tuple,
                path : Path) -> tuple:
    """
    """
    assert use_temporal_as_edges is not None

    print(np.nanmax(Xset[:,6:]), np.nanmin(Xset[:,6:]))
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
    
        x = load_x_from_pickle(x, shape, path, x_train, y_train, scaling, features_name, features_name_2D)
        save_object(x, str(id)+'.pkl', path / scaling)

        X.append(str(id)+'.pkl')
        Y.append(y)
        E.append(e)

        i += 1

    dataset = ReadGraphDataset_2D(X, Y, E, len(X), device, path / scaling)
    loader = DataLoader(dataset, 64, False, collate_fn=graph_collate_fn)
    return loader

def create_test_loader(graph, Xset : np.array,
                       Yset : np.array,
                       device : torch.device,
                       use_temporal_as_edges : bool,
                       ks :int) -> tuple:
    
    assert use_temporal_as_edges is not None

    print(np.nanmax(Xset[:,6:]), np.nanmin(Xset[:,6:]))
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
                     X, Y, x_train, y_train, device, encoding,
                     k_days, test, features_name, scaling, prefix, Rewrite):

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
                     X, Y, x_train, y_train, device, k_days,
                     test, features_name, scaling, encoding, prefix, Rewrite):
    loader = None
    if Rewrite:
        loader = create_test_loader(graph=graphScale, Xset=X, Yset=Y, device=device,
                                    use_temporal_as_edges=use_temporal_as_edges, ks=k_days)

        save_object(loader, 'loader_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
        loader = read_object('loader_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
    else:
        loader = read_object('loader_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
    if loader is None:
        loader = create_test_loader(graph=graphScale, Xset=X, Yset=Y, device=device,
                                    use_temporal_as_edges=use_temporal_as_edges, ks=k_days)

        save_object(loader, 'loader_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)

    return loader


def load_loader_test_2D(use_temporal_as_edges, graphScale, dir_output,
                     X, Y, x_train, y_train, device,
                     k_days, test, features_name, scaling, encoding, prefix,
                     shape, features_name_2D, Rewrite):
    loader = None
    if Rewrite:
        loader = create_test_loader_2D(graphScale,
                                    X,
                                    Y,
                                    x_train,
                                    y_train,
                                    use_temporal_as_edges,
                                    device,
                                    scaling,
                                    features_name,
                                    features_name_2D,
                                    k_days,
                                    shape,
                                    dir_output)

        save_object(loader, 'loader_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'_2D.pkl', dir_output)
        loader = read_object('loader_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'_2D.pkl', dir_output)
    else:
        loader = read_object('loader_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'_2D.pkl', dir_output)
    if loader is None:
        loader = create_test_loader_2D(graphScale,
                                    X,
                                    Y,
                                    x_train,
                                    y_train,
                                    use_temporal_as_edges,
                                    device,
                                    scaling,
                                    features_name,
                                    features_name_2D,
                                    k_days,
                                    shape,
                                    dir_output)

        save_object(loader, 'loader_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'_2D.pkl', dir_output)

    return loader

def test_sklearn_api_model(graphScale,
                          test_dataset_dept,
                           methods,
                           testname,
                           prefix_train,
                           models,
                           dir_output,
                           device,
                           encoding,
                           scaling,
                           testd_departement,
                           dir_train,
                           features_name):
    
    
    scale = graphScale.scale
    metrics = {}

    ################################ Ground Truth ############################################
    logger.info('#########################')
    logger.info(f'       GT              ')
    logger.info('#########################')

    y = test_dataset_dept[test_dataset_dept['weight'] > 0][[ids_columns + targets_columns]].values
    pred = np.empty((y.shape[0], 2))
    pred[:, 0] = test_dataset_dept[test_dataset_dept['weight'] > 0]['risk'].values
    pred[:, 1] = test_dataset_dept[test_dataset_dept['weight'] > 0]['class_risk'].values
    metrics['GT'] = add_metrics(methods, 0, pred, y, testd_departement, 'risk', scale, 'gt', dir_train)
    i = 1

    #################################### Traditionnal ##################################################

    for name, target_name, autoRegression in models:
        y = test_dataset_dept[test_dataset_dept['weight'] > 0][[ids_columns + targets_columns]].values
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
        logger.info(f'{features_selected}')
        
        pred = np.empty((y.shape[0], 2))

        pred[:, 0] = graphScale.predict_model_api_sklearn(test_dataset_dept, features_selected, target_name == 'binary', autoRegression, features_name)
        logger.info(f'pred min {np.nanmin(pred[:, 0])}, pred max : {np.nanmax(pred[:, 0])}')

        dir_predictor = root_graph / dir_train / '../influenceClustering'
        for nameDep in departements:
            mask = np.argwhere(y[:, 3] == name2int[nameDep])
            if mask.shape[0] == 0:
                continue
            if target_name == 'risk':
                predictor = read_object(nameDep+'Predictor'+str(scale)+'.pkl', dir_predictor)
            else:
                create_predictor(pred, name, nameDep, dir_predictor, scale, target_name == 'binary')
                predictor = read_object(nameDep+'Predictor'+name+str(scale)+'.pkl', dir_predictor)

            pred[mask[:,0], 1] = order_class(predictor, predictor.predict(pred[mask[:, 0], 0]))

        metrics[name] = add_metrics(methods, i, pred, y, testd_departement, target_name, scale, name, dir_train)

        if target_name == 'binary':
            y[:,-1] = y[:,-1] / np.nanmax(y[:,-1])

        realVspredict(pred[:, 0], y, -1,
                      dir_output / n, 'raw')
                
        realVspredict(pred[:, 1], y, -3,
                      dir_output / n, 'class')
        
        realVspredict(pred[:, 0], y, -2,
                      dir_output / n, 'nbfire')

        sinister_distribution_in_class(pred[:, 1], y, dir_output / n)

        if target_name == 'binary':
            y_test = (y[:, -2] > 0).astype(int)
        elif target_name == 'nbsinister':
            y_test = y[:, -2]
        elif target_name == 'risk':
            y_test = y[:, -1]
        else:
            raise ValueError(f'Unknow {target_name}')

        #model.plot_features_importance(Xset[:, features_selected], y_test, [features_name[int(i)] for i in list(features_selected)], 'test', dir_output / n, mode='bar')

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
        try:
            pred[:, 0] = apply_kmeans_class_on_target(test_dataset_dept, pred, dir_train / 'varOnValue' / prefix_train, False).reshape(-1)
            metrics[name+'_kmeans_preprocessing'] = add_metrics(methods, i, pred, y, testd_departement, target_name, scale, name, dir_train)
        except:
            pass
        i += 1

    ########################################## Save metrics ################################ 
    outname = 'metrics'+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+testname+'_tree.pkl'
    save_object(metrics, outname, dir_output)

def test_dl_model(graphScale, Xset, Yset, x_train, y_train,
                           methods,
                           testname,
                           features_name,
                           prefix,
                           prefix_train,
                           dummy,
                           models,
                           dir_output,
                           device,
                           k_days,
                           Rewrite,
                           encoding,
                           scaling,
                           testd_departement,
                           dir_train,
                           dico_model,
                           features_name_2D,
                           shape2D,
                           sinister,
                           resolution):
    
    scale = graphScale.scale
    metrics = {}
    ################################ Ground Truth ############################################
    logger.info('#########################')
    logger.info(f'       GT              ')
    logger.info('#########################')

    y = Yset[Yset[:,-4] > 0]
    Xset = Xset[Yset[:, -4] > 0]
    pred = np.empty((Yset[Yset[:,-4] > 0].shape[0], 2))
    pred[:, 0] = Yset[Yset[:,-4] > 0][:, -1]
    pred[:, 1] = Yset[Yset[:,-4] > 0][:, -3]
    metrics['GT'] = add_metrics(methods, 0, pred, y, testd_departement, False, scale, 'gt', dir_train)
    i = 1

    #################################### GNN ###################################################
    for mddel, use_temporal_as_edges, target_name, is_2D_model, autoRegression in models:
        #n = mddel+'_'+prefix_train+'_'+str(scale)+'_'+scaling + '_' + encoding+'_'+testname
        n = mddel
        model_dir = dir_train / Path('check_'+scaling+'/' + prefix_train + '/' + mddel +  '/')
        logger.info('#########################')
        logger.info(f'       {mddel}          ')
        logger.info('#########################')

        if not is_2D_model:
            test_loader = load_loader_test(use_temporal_as_edges=use_temporal_as_edges, graphScale=graphScale, dir_output=dir_output,
                                        X=Xset, Y=Yset, x_train=x_train, y_train=y_train,
                                        device=device, k_days=k_days, test=testname, features_name=features_name,
                                        scaling=scaling, encoding=encoding, prefix=prefix, Rewrite=False)

        else:
            test_loader = load_loader_test_2D(use_temporal_as_edges=use_temporal_as_edges, graphScale=graphScale,
                                              dir_output=dir_output / '2D' / prefix / 'data', X=Xset, Y=Yset,
                                            x_train=x_train, y_train=y_train, device=device, k_days=k_days, test=testname,
                                            features_name=features_name, scaling=scaling, prefix=prefix,
                                        shape=(shape2D[0], shape2D[1], shape2D[2]), features_name_2D=features_name_2D, encoding=encoding,
                                        Rewrite=False)

        graphScale._load_model_from_path(model_dir / 'best.pt', dico_model[mddel], device)

        features_selected = read_object('features.pkl', model_dir)

        predTensor, YTensor = graphScale._predict_test_loader(test_loader, features_selected, device=device, target_name=target_name, 
                                                              autoRegression=autoRegression, features_name=features_name)
        y = YTensor.detach().cpu().numpy()

        dir_predictor = root_graph / dir_train / 'influenceClustering'
        pred = np.empty((predTensor.shape[0], 2))
        pred[:, 0] = predTensor.detach().cpu().numpy()
        for nameDep in departements:
            mask = np.argwhere(y[:, 3] == name2int[nameDep])
            if mask.shape[0] == 0:
                continue
            if not target_name:
                predictor = read_object(nameDep+'Predictor'+str(scale)+'.pkl', dir_predictor)
            else:
                create_predictor(pred, mddel, nameDep, dir_predictor, scale, target_name == 'binary')
                predictor = read_object(nameDep+'Predictor'+mddel+str(scale)+'.pkl', dir_predictor)

            pred[mask[:,0], 1] = order_class(predictor, predictor.predict(pred[mask[:, 0], 0]))

        metrics[mddel] = add_metrics(methods, i, pred, y, testd_departement, target_name, scale, mddel, dir_train)

        realVspredict(pred[:, 0], y, -1,
                      dir_output / n, 'raw')
        
        realVspredict(pred[:, 1], y, -3,
                      dir_output / n, 'class')

        realVspredict(pred[:, 1], y, -2,
                      dir_output / n, 'nbfire')
        
        sinister_distribution_in_class(pred[:, 1], y, dir_output / n)

        res = np.empty((pred.shape[0], y.shape[1] + 4))
        res[:, :y.shape[1]] = y
        res[:,y.shape[1]] = pred[:,0]
        res[:,y.shape[1] + 1] = pred[:, 1]
        res[:,y.shape[1]+2] = np.sqrt((y[:,-4] * (pred[:,0] - y[:,-1]) ** 2))
        res[:,y.shape[1]+3] = np.sqrt(((pred[:,0] - y[:,-1]) ** 2))

        save_object(res, mddel+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output / n)

        if testname == '69':
            start = '2022-06-01'
            stop = '2022-09-30'
        else:
            start = '2023-06-01'
            stop = '2023-09-30'
        
        """for departement in testd_departement:
            susectibility_map(departement, sinister,
                        dir_train, dir_output, k_days,
                        target_name, scale, mddel, res, n,
                        start, stop, resolution)"""

        shapiro_wilk(pred[:,0], y[:,-1], dir_output / n)

        """n = mddel+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname
        realVspredict2d(pred,
                    y,
                    target_name,
                    mddel,
                    scale,
                    dir_output / n,
                    dir_train,
                    geo,
                    testd_departement,
                    graphScale)"""
        
        i += 1

    ########################################## Save metrics ################################ 
    outname = 'metrics'+'_'+prefix_train+'_'+'_'+scaling+'_'+encoding+'_'+testname+'_dl.pkl'
    save_object(metrics, outname, dir_output / n)

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

def test_break_points_model(Xset, Yset,
                           methods,
                           testname,
                           prefix_train,
                           dir_output,
                           encoding,
                           scaling,
                           testd_departement,
                           scale,
                           dir_train,
                           sinister,
                           resolution):
    
    #n = 'break_point'+'_'+prefix_train+'_'+str(scale)+'_'+scaling + '_' + encoding+'_'+testname
    n = 'break_point'
    name = 'break_point'
    dir_break_point = dir_train / 'varOnValue' / prefix_train
    features_selected = read_object('features.pkl', dir_break_point)
    dico_correlation = read_object('break_point_dict.pkl', dir_break_point)
    metrics = {}

    y = Yset[Yset[:,-4] > 0]
    x = Xset[Yset[:,-4] > 0]
    pred = np.zeros((Xset.shape[0], 2))
    leni = 0
    for fet in features_selected:
        if 'name' not in dico_correlation[fet].keys():
            continue
        on = dico_correlation[fet]['name']+'.pkl'
        predictor = read_object(on, dir_break_point / dico_correlation[fet]['fet'])
        predclass = order_class(predictor, predictor.predict(x[:, fet]))
        leni += 1
        if abs(dico_correlation[fet]['correlation']) > 0.7:
            # If negative linearity, inverse class
            if dico_correlation[fet]['correlation'] < 0:
                new_class = np.flip(np.arange(predictor.n_clusters))
                for i, nc in enumerate(new_class):
                    mask = np.argwhere(predclass == i)
                    predclass[mask] = nc
            cls = np.unique(predclass)
            for c in cls:
                mask = np.argwhere(predclass == c)
                pred[mask, 1] += dico_correlation[fet][int(c)]
            pred[:,0] += predclass
            
    logger.info(f'{leni, len(features_selected)}')
    pred /= leni
    y[:,-1] = y[:,-1] / np.nanmax(y[:,-1])
    metrics['break_point'] = add_metrics(methods, 0, pred, y, testd_departement, True, scale, name, dir_train)
    
    realVspredict(pred[:, 1], y, -1,
                      dir_output / n, 'raw')
    
    realVspredict(pred[:, 0], y, -3,
                      dir_output / n, 'class')
    
    sinister_distribution_in_class(predclass, y, dir_output / n)
        
    res = np.empty((pred.shape[0], y.shape[1] + 3))
    res[:, :y.shape[1]] = y
    res[:,y.shape[1]] = pred[:,1]
    res[:,y.shape[1]+1] = np.sqrt((y[:,-4] * (pred[:,1] - y[:,-1]) ** 2))
    res[:,y.shape[1]+2] = np.sqrt(((pred[:,1] - y[:,-1]) ** 2))

    #save_object(res, name+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output / n)
    save_object(res, name+'_pred.pkl', dir_output / n)

    """if testname == '69':
        start = '2018-01-01'
        stop = '2022-09-30'
    else:
        start = '2023-06-01'
        stop = '2023-09-30'
    
    for departement in testd_departement:
        susectibility_map(departement, sinister,
                    dir_train, dir_output, k_days,
                    False, scale, name, res, n,
                    start, stop, resolution)"""

    shapiro_wilk(pred[:,1], y[:,-1], dir_output / n)

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
            logger.info(features_selected)
            logger.info(XsetDaily.shape)
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
        logger.info(f'{features_selected}')

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
                predictor = read_object(nameDep+'Predictor'+str(scale)+'.pkl', dir_predictor)
            else:
                create_predictor(pred, name, nameDep, dir_predictor, scale, target_name == 'binary')
                predictor = read_object(nameDep+'Predictor'+name+str(scale)+'.pkl', dir_predictor)

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
            train_dir,
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