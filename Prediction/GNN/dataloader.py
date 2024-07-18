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

#########################################################################################################
#                                                                                                       #
#                                         Preprocess                                                    #
#                                                                                                       #
#########################################################################################################

def preprocess(X : np.array, Y : np.array, scaling : str, maxDate : str,
               trainDate : str, trainDepartements : list, departements : list, ks : int, dir_output : Path,
               prefix : str, pos_feature : dict, days_in_futur : int, futur_met :str):
    """
    Preprocess the input features:
        X : input features
        Y  input target
        test_size : fraction of validation
        pos_feature : dict storing feature's index
        scaling : scale method
    """

    global features

    # Remove nodes where the is a feature at nan
    Xset, Yset = remove_nan_nodes(X, Y)
    Xset, Yset = remove_none_target(Xset, Yset)
    Xset, Yset = remove_bad_period(Xset, Yset, PERIODES_A_IGNORER, departements)

    print(f'We found nan at {np.unique(np.argwhere(np.isnan(X))[:,1])}')

    print(f'X shape {X.shape}, X shape after removal nan and -1 target {Xset.shape}, Y shape is {Yset.shape}')

    assert Xset.shape[0] > 0
    assert Xset.shape[0] == Yset.shape[0]

    # Sort by date
    ind = np.lexsort([Yset[:,4]])
    Xset = Xset[ind]
    Yset = Yset[ind]

    # Seperate date into training and validation
    #dateTrain, dateVal = train_test_split(dateUseForLossCalculation, test_size=test_size, random_state=42)
    #dateTrain = dateUseForLossCalculation[: int(dateUseForLossCalculation.shape[0] * (1 - test_size))]
    #dateVal = dateUseForLossCalculation[int(dateUseForLossCalculation.shape[0] * (1 - test_size)):]

    if days_in_futur > 1:
        logger.info(f'{futur_met} target by {days_in_futur} days in futur')
        Yset = target_by_day(Yset, days_in_futur, futur_met)
        logger.info(f'Done')

    trainCode = [name2int[dept] for dept in trainDepartements]

    Xtrain, Ytrain = (Xset[(Xset[:,4] < allDates.index(trainDate)) & (np.isin(Xset[:,3], trainCode))],
                      Yset[(Xset[:,4] < allDates.index(trainDate)) & (np.isin(Xset[:,3], trainCode))])
    
    save_object(Xtrain, 'Xtrain_'+prefix+'.pkl', dir_output)
    save_object(Ytrain, 'Ytrain_'+prefix+'.pkl', dir_output)
    
    #if minPoint != 'full':
    #    Xtrain, Ytrain = select_n_points(Xtrain, Ytrain, dir_output, int(minPoint))

    # Define the scale method
    if scaling == 'MinMax':
        scaler = min_max_scaler
    elif scaling == 'z-score':
        scaler = standart_scaler
    elif scaling == "robust":
        scaler = robust_scaler
    elif scaling == 'none':
        scaling = None
    else:
        raise ValueError('Unknow')
        exit(1)

    if 'AutoRegressionReg' in trainFeatures:
        autoregressiveBand = pos_feature['AutoRegressionReg']
    else:
        autoregressiveBand = - 1

    if scaling is not None:
        # Scale Features
        for featureband in range(6, Xset.shape[1]):
            if featureband == autoregressiveBand:
                logger.info(f'Avoid scale for Autoregressive feature')
                continue
            Xset[:,featureband] = scaler(Xset[:,featureband], Xtrain[:, featureband], concat=False)
    
    Xtrain, Ytrain = (Xset[(Xset[:,4] < allDates.index(trainDate)) & (np.isin(Xset[:,3], trainCode))],
                      Yset[(Xset[:,4] < allDates.index(trainDate)) & (np.isin(Xset[:,3], trainCode))])

    Xval, Yval = (Xset[(Xset[:,4] >= allDates.index(trainDate)) & (Xset[:,4] < allDates.index(maxDate)) & (np.isin(Xset[:,3], trainCode))],
                    Yset[(Xset[:,4] >= allDates.index(trainDate)) & (Xset[:,4] < allDates.index(maxDate)) & (np.isin(Xset[:,3], trainCode))])

    Xtest, Ytest = (Xset[((Xset[:,4] >= allDates.index(maxDate)) & (np.isin(Xset[:,3], trainCode))) |  (~np.isin(Xset[:,3], trainCode))],
                    Yset[((Xset[:,4] >= allDates.index(maxDate)) & (np.isin(Xset[:,3], trainCode))) |  (~np.isin(Xset[:,3], trainCode))])
    
    logger.info(f'Size of the train set {Xtrain.shape[0]}, size of the val set {Xval.shape[0]}, size of the test set {Xtest.shape[0]}')
    logger.info(f'Unique train dates set {np.unique(Xtrain[:,4]).shape[0]}, val dates {np.unique(Xval[:,4]).shape[0]}, test dates {np.unique(Xtest[:,4]).shape[0]}')

    # Log
    logger.info(f'Check {scaling} standardisation Train : {np.nanmax(Xtrain[:,6:]), np.nanargmax(Xtrain[:,6:])}, {np.nanmin(Xtrain[:,6:]), np.nanargmin(Xtrain[:,6:])}')
    logger.info(f'Check {scaling} standardisation Val : {np.nanmax(Xval[:,6:]), np.nanargmax(Xval[:,6:])}, {np.nanmin(Xval[:,6:]), np.nanargmin(Xval[:,6:])}')
    logger.info(f'Check {scaling} standardisation Test : {np.nanmax(Xtest[:,6:]), np.nanargmax(Xtrain[:,6:])}, {np.nanmin(Xtest[:,6:]), np.nanargmin(Xtest[:,6:])}')

    return (Xtrain, Ytrain), (Xval, Yval), (Xtest, Ytest)

def preprocess_test(X, Y, Xtrain, scaling):

    Xset, Yset = remove_nan_nodes(X, Y)
    
    logger.info(f'X shape {X.shape}, X shape after removal nan {Xset.shape}')
    assert Xset.shape[0] > 0

    # Define the scaler
    if scaling == 'MinMax':
        scaler = min_max_scaler
    elif scaling == 'z-score':
        scaler = standart_scaler
    elif scaling == "robust":
        scaler = robust_scaler
    else:
        ValueError('Unknow : scaling is obligatory')
        exit(1)

    # Scaling Features
    for featureband in range(6, Xset.shape[1]):
        Xset[:,featureband] = scaler(Xset[:,featureband], Xtrain[:, featureband], concat=False)

    logger.info(f'Check {scaling} standardisation test : {np.nanmax(Xset[:,6:])}, {np.nanmin(Xset[:,6:])}')

    return Xset, Yset

def preprocess_inference(X, Xtrain, scaling):

    Xset, _ = remove_nan_nodes(X, None)

    print(f'X shape {X.shape}, X shape after removal nan {Xset.shape}')
    assert Xset.shape[0] > 0

    if scaling == 'MinMax':
        scaler = min_max_scaler
    elif scaling == 'z-score':
        scaler = standart_scaler
    elif scaling == "robust":
        scaler = robust_scaler
    else:
        ValueError('Unknow')
        exit(1)
    
    # Scaling Features
    for featureband in range(6, Xset.shape[1]):
        Xset[:,featureband] = scaler(Xset[:,featureband], Xtrain[:, featureband], concat=False)

    print(f'Check {scaling} standardisation inference : {np.nanmax(Xset[:,6:])}, {np.nanmin(Xset[:,6:])}')
    
    return Xset

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

    Xtrain, Ytrain = train
    
    Xval, Yval = val

    Xtest, Ytest = test

    dateTrain = np.unique(Xtrain[np.argwhere(Xtrain[:, 5] > 0),4])
    dateVal = np.unique(Xval[np.argwhere(Xval[:, 5] > 0),4])
    dateTest = np.unique(Xtest[:,4])
    
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
            x, y, e = construct_graph_set(graph, id, Xtrain, Ytrain, ks)
        else:
            x, y, e = construct_graph_with_time_series(graph, id, Xtrain, Ytrain, ks)

        if x is None:
            continue
    
        Xst.append(x)
        Yst.append(y)
        Est.append(e)

    for id in dateVal:
        if use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, id, Xval, Yval, ks)
        else:
            x, y, e = construct_graph_with_time_series(graph, id, Xval, Yval, ks)

        if x is None:
            continue
        
        XsV.append(x)
        YsV.append(y)
        EsV.append(e)

    for id in dateTest:
        if use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, id, Xtest, Ytest, ks)
        else:
            x, y, e = construct_graph_with_time_series(graph, id, Xtest, Ytest, ks)

        if x is None:
            continue
    
        XsTe.append(x)
        YsTe.append(y)
        EsTe.append(e)
    
    assert len(Xst) > 0
    assert len(XsV) > 0
    assert len(XsTe) > 0

    trainDataset = InplaceGraphDataset(Xst, Yst, Est, len(Xst), device)
    valDataset = InplaceGraphDataset(XsV, YsV, EsV, len(XsV), device)
    testDatset = InplaceGraphDataset(XsTe, YsTe, EsTe, len(XsTe), device)
    return trainDataset, valDataset, testDatset

def train_val_data_loader(graph,
                          train,
                          val,
                          test,
                        use_temporal_as_edges : bool,
                        batch_size,
                        device,
                        ks : int) -> None:

    trainDataset, valDataset, testDataset = create_dataset(graph,
                                train,
                                val,
                                test, use_temporal_as_edges,
                                device, ks)

    trainLoader = DataLoader(trainDataset, batch_size, True, collate_fn=graph_collate_fn)
    valLoader = DataLoader(valDataset, valDataset.__len__(), False, collate_fn=graph_collate_fn)
    testLoader = DataLoader(testDataset, testDataset.__len__(), False, collate_fn=graph_collate_fn)
    return trainLoader, valLoader, testLoader


def train_val_data_loader_2D(graph,
                            train,
                            val,
                            test,
                            use_temporal_as_edges : bool,
                            batch_size : int,
                            device: torch.device,
                            scaling : str,
                            pos_feature : dict,
                            pos_feature_2D : dict,
                            ks : int,
                            shape : tuple,
                            path : Path) -> None:
    

    Xtrain, Ytrain = train

    Xval, Yval = val

    Xtest, Ytest = test

    dateTrain = np.unique(Xtrain[:,4])
    dateVal = np.unique(Xval[:,4])
    dateTest = np.unique(Xtest[:,4])

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
            x, y, e = construct_graph_set(graph, id, Xtrain, Ytrain, ks)
        else:
            x, y, e = construct_graph_with_time_series(graph, id, Xtrain, Ytrain, ks)

        if x is None:
            continue

        x = load_x_from_pickle(x, shape, path, Xtrain, Ytrain, scaling, pos_feature, pos_feature_2D)
        save_object(x, str(id)+'.pkl', path / scaling)
    
        Xst.append(x)
        Yst.append(y)
        Est.append(e)

    for id in dateVal:
        if use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, id, Xval, Yval, ks)
        else:
            x, y, e = construct_graph_with_time_series(graph, id, Xval, Yval, ks)

        if x is None:
            continue

        x = load_x_from_pickle(x, shape, path, Xtrain, Ytrain, scaling, pos_feature, pos_feature_2D)
        save_object(x, str(id)+'.pkl', path / scaling)
        
        XsV.append(x)
        YsV.append(y)
        EsV.append(e)

    for id in dateTest:
        if use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, id, Xtest, Ytest, ks)
        else:
            x, y, e = construct_graph_with_time_series(graph, id, Xtest, Ytest, ks)

        if x is None:
            continue

        x = load_x_from_pickle(x, shape, path, Xtrain, Ytrain, scaling, pos_feature, pos_feature_2D)
        save_object(x, str(id)+'.pkl', path / scaling)
    
        XsTe.append(x)
        YsTe.append(y)
        EsTe.append(e)
    
    assert len(Xst) > 0
    assert len(XsV) > 0
    assert len(XsTe) > 0

    
    assert len(Xst) > 0
    assert len(XsV) > 0
    print(len(Xst))
    
    trainDataset = ReadGraphDataset_2D(Xst, Yst, Est, len(Xst), device, path / scaling)
    valDataset = ReadGraphDataset_2D(XsV, YsV, EsV, len(XsV), device, path  / scaling)
    testDataset = ReadGraphDataset_2D(XsTe, YsTe, EsTe, len(XsTe), device, path  / scaling)
    
    trainLoader = DataLoader(trainDataset, batch_size, True, collate_fn=graph_collate_fn)
    valLoader = DataLoader(valDataset, valDataset.__len__(), False, collate_fn=graph_collate_fn)
    testLoader = DataLoader(testDataset, testDataset.__len__(), False, collate_fn=graph_collate_fn)
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
                Xtrain : np.array,
                Ytrain : np.array,
                use_temporal_as_edges : bool,
                device: torch.device,
                scaling : str,
                pos_feature : dict,
                pos_feature_2D : dict,
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
    
        x = load_x_from_pickle(x, shape, path, Xtrain, Ytrain, scaling, pos_feature, pos_feature_2D)
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
                     X, Y, Xtrain, Ytrain, device, encoding,
                     k_days, test, pos_feature, scaling, prefix, Rewrite):

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
                     X, Y, Xtrain, Ytrain, device, k_days,
                     test, pos_feature, scaling, encoding, prefix, Rewrite):
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
                     X, Y, Xtrain, Ytrain, device,
                     k_days, test, pos_feature, scaling, encoding, prefix,
                     shape, pos_feature_2D, Rewrite):
    loader = None
    if Rewrite:
        loader = create_test_loader_2D(graphScale,
                                    X,
                                    Y,
                                    Xtrain,
                                    Ytrain,
                                    use_temporal_as_edges,
                                    device,
                                    scaling,
                                    pos_feature,
                                    pos_feature_2D,
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
                                    Xtrain,
                                    Ytrain,
                                    use_temporal_as_edges,
                                    device,
                                    scaling,
                                    pos_feature,
                                    pos_feature_2D,
                                    k_days,
                                    shape,
                                    dir_output)

        save_object(loader, 'loader_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'_2D.pkl', dir_output)

    return loader

def test_sklearn_api_model(graphScale,
                           Xset, Yset,
                           methods,
                           testname,
                           prefix_train,
                           models,
                           dir_output,
                           device,
                           encoding,
                           scaling,
                           testDepartement,
                           train_dir,
                           pos_feature):
    
    
    if Xset.shape[0] == 0:
        logger.info(f'{testname} is empty, skip')
        return
    
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
    metrics['GT'] = add_metrics(methods, 0, pred, y, testDepartement, False, scale, 'gt', train_dir)
    i = 1

    #################################### Traditionnal ##################################################

    for name, isBin, autoRegression in models:
        y = Yset[Yset[:,-4] > 0]
        n = name
        model_dir = train_dir / Path('check_'+scaling + '/' + prefix_train + '/baseline/' + name + '/')
        
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

        pred[:, 0] = graphScale.predict_model_api_sklearn(Xset, features_selected, isBin, autoRegression, pos_feature)
        logger.info(f'pred min {np.nanmin(pred[:, 0])}, pred max : {np.nanmax(pred[:, 0])}')

        dir_predictor = root_graph / train_dir / 'influenceClustering'
        for nameDep in departements:
            mask = np.argwhere(y[:, 3] == name2int[nameDep])
            if mask.shape[0] == 0:
                continue
            if not isBin:
                predictor = read_object(nameDep+'Predictor'+str(scale)+'.pkl', dir_predictor)
            else:
                create_binary_predictor(pred, name, nameDep, dir_predictor, scale)
                predictor = read_object(nameDep+'Predictor'+name+str(scale)+'.pkl', dir_predictor)

            pred[mask[:,0], 1] = order_class(predictor, predictor.predict(pred[mask[:, 0], 0]))

        metrics[name] = add_metrics(methods, i, pred, y, testDepartement, isBin, scale, name, train_dir)

        if isBin:
            y[:,-1] = y[:,-1] / np.nanmax(y[:,-1])

        realVspredict(pred[:, 0], y, -1,
                      dir_output / n, 'raw')
                
        realVspredict(pred[:, 1], y, -3,
                      dir_output / n, 'class')
        
        realVspredict(pred[:, 0], y, -2,
                      dir_output / n, 'nbfire')

        sinister_distribution_in_class(pred[:, 1], y, dir_output / n)

        res = np.empty((pred.shape[0], y.shape[1] + 4))
        res[:, :y.shape[1]] = y
        res[:,y.shape[1]] = pred[:, 0]
        res[:,y.shape[1] + 1] = pred[:, 1]
        res[:,y.shape[1]+2] = np.sqrt((y[:,-4] * (pred[:,0] - y[:,-1]) ** 2))
        res[:,y.shape[1]+3] = np.sqrt(((pred[:,0] - y[:,-1]) ** 2))

        save_object(res, name+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output / n)

        if testname == '69':
            start = '2018-01-01'
            stop = '2022-09-30'
        else:
            start = '2023-06-01'
            stop = '2023-09-30'
        
        """for dept in testDepartement:
            susectibility_map(dept, sinister,
                        train_dir, dir_output, k_days,
                        isBin, scale, name, res, n,
                        start, stop, resolution)"""

        shapiro_wilk(pred[:,0], y[:,-1], dir_output / n)

        """
        realVspredict2d(pred,
            y,
            isBin,
            name,
            scale,
            dir_output / n,
            train_dir,
            geo,
            testDepartement,
            graphScale)"""
        try:
            pred[:, 0] = apply_kmeans_class_on_target(Xset, pred, train_dir / 'varOnValue' / prefix_train, False).reshape(-1)
            metrics[name+'_kmeans_preprocessing'] = add_metrics(methods, i, pred, y, testDepartement, isBin, scale, name, train_dir)
        except:
            pass
        i += 1

    ########################################## Save metrics ################################ 
    outname = 'metrics'+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_tree.pkl'
    save_object(metrics, outname, dir_output)

def test_dl_model(graphScale, Xset, Yset, Xtrain, Ytrain,
                           methods,
                           testname,
                           pos_feature,
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
                           testDepartement,
                           train_dir,
                           dico_model,
                           pos_feature_2D,
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
    metrics['GT'] = add_metrics(methods, 0, pred, y, testDepartement, False, scale, 'gt', train_dir)
    i = 1

    #################################### GNN ###################################################
    for mddel, use_temporal_as_edges, isBin, is_2D_model, autoRegression in models:
        #n = mddel+'_'+prefix_train+'_'+str(scale)+'_'+scaling + '_' + encoding+'_'+testname
        n = mddel
        model_dir = train_dir / Path('check_'+scaling+'/' + prefix_train + '/' + mddel +  '/')
        logger.info('#########################')
        logger.info(f'       {mddel}          ')
        logger.info('#########################')

        if not is_2D_model:
            test_loader = load_loader_test(use_temporal_as_edges=use_temporal_as_edges, graphScale=graphScale, dir_output=dir_output,
                                        X=Xset, Y=Yset, Xtrain=Xtrain, Ytrain=Ytrain,
                                        device=device, k_days=k_days, test=testname, pos_feature=pos_feature,
                                        scaling=scaling, encoding=encoding, prefix=prefix, Rewrite=False)

        else:
            test_loader = load_loader_test_2D(use_temporal_as_edges=use_temporal_as_edges, graphScale=graphScale,
                                              dir_output=dir_output / '2D' / prefix / 'data', X=Xset, Y=Yset,
                                            Xtrain=Xtrain, Ytrain=Ytrain, device=device, k_days=k_days, test=testname,
                                            pos_feature=pos_feature, scaling=scaling, prefix=prefix,
                                        shape=(shape2D[0], shape2D[1], shape2D[2]), pos_feature_2D=pos_feature_2D, encoding=encoding,
                                        Rewrite=False)

        graphScale._load_model_from_path(model_dir / 'best.pt', dico_model[mddel], device)

        features_selected = read_object('features.pkl', model_dir)

        predTensor, YTensor = graphScale._predict_test_loader(test_loader, features_selected, device=device, isBin=isBin, autoRegression=autoRegression, pos_feature=pos_feature)
        y = YTensor.detach().cpu().numpy()

        dir_predictor = root_graph / train_dir / 'influenceClustering'
        pred = np.empty((predTensor.shape[0], 2))
        pred[:, 0] = predTensor.detach().cpu().numpy()
        for nameDep in departements:
            mask = np.argwhere(y[:, 3] == name2int[nameDep])
            if mask.shape[0] == 0:
                continue
            if not isBin:
                predictor = read_object(nameDep+'Predictor'+str(scale)+'.pkl', dir_predictor)
            else:
                create_binary_predictor(pred, mddel, nameDep, dir_predictor, scale)
                predictor = read_object(nameDep+'Predictor'+mddel+str(scale)+'.pkl', dir_predictor)

            pred[mask[:,0], 1] = order_class(predictor, predictor.predict(pred[mask[:, 0], 0]))

        metrics[mddel] = add_metrics(methods, i, pred, y, testDepartement, isBin, scale, mddel, train_dir)


        realVspredict(pred[:, 0], y, -1,
                      dir_output / n, 'raw')
        
        realVspredict(pred[:, 1], y, -3,
                      dir_output / n, 'class')
        
        sinister_distribution_in_class(pred[:, 1], y, dir_output / n)

        res = np.empty((pred.shape[0], y.shape[1] + 4))
        res[:, :y.shape[1]] = y
        res[:,y.shape[1]] = pred[:,0]
        res[:,y.shape[1] + 1] = pred[:, 1]
        res[:,y.shape[1]+2] = np.sqrt((y[:,-4] * (pred[:,0] - y[:,-1]) ** 2))
        res[:,y.shape[1]+3] = np.sqrt(((pred[:,0] - y[:,-1]) ** 2))

        save_object(res, mddel+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output / n)

        if testname == '69':
            start = '2022-06-01'
            stop = '2022-09-30'
        else:
            start = '2023-06-01'
            stop = '2023-09-30'
        
        """for dept in testDepartement:
            susectibility_map(dept, sinister,
                        train_dir, dir_output, k_days,
                        isBin, scale, mddel, res, n,
                        start, stop, resolution)"""

        shapiro_wilk(pred[:,0], y[:,-1], dir_output / n)

        """n = mddel+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname
        realVspredict2d(pred,
                    y,
                    isBin,
                    mddel,
                    scale,
                    dir_output / n,
                    train_dir,
                    geo,
                    testDepartement,
                    graphScale)"""
        
        i += 1

    ########################################## Save metrics ################################ 
    outname = 'metrics'+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_dl.pkl'
    save_object(metrics, outname, dir_output)

def test_simple_model(graphScale, Xset, Yset,
                           methods,
                           testname,
                           pos_feature,
                           prefix_train,
                           dummy,
                           dir_output,
                           encoding,
                           scaling,
                           testDepartement,
                           scale,
                           train_dir
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
            metrics[name] = add_metrics(methods, 0, predTensor, YTensor, testDepartement, False, scale, name, train_dir)

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
            metrics[name] = add_metrics(methods, 1, predTensor, YTensor, testDepartement, True, scale, name, train_dir)

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
        pred = graphScale._predict_perference_with_X(x, pos_feature)

        predTensor = torch.tensor(pred, dtype=torch.float32, device=device)
        metrics[name] = add_metrics(methods, 2, predTensor, YTensor, testDepartement, False, scale, name, train_dir)

        res = np.empty((pred.shape[0], y.shape[1] + 3))
        res[:, :y.shape[1]] = y
        res[:,y.shape[1]] = pred
        res[:,y.shape[1]+1] = np.sqrt((y[:,-4] * (pred - y[:,-1]) ** 2))
        res[:,y.shape[1]+2] = np.sqrt(((pred - y[:,-1]) ** 2))

        save_object(res, name+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output)
        outname = 'metrics'+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_metrics.pkl'
        save_object(metrics, outname, dir_output)

def test_break_points_model(Xset, Yset,
                           methods,
                           testname,
                           prefix_train,
                           dir_output,
                           encoding,
                           scaling,
                           testDepartement,
                           scale,
                           train_dir,
                           sinister,
                           resolution):
    
    #n = 'break_point'+'_'+prefix_train+'_'+str(scale)+'_'+scaling + '_' + encoding+'_'+testname
    n = 'break_point'
    name = 'break_point'
    dir_break_point = train_dir / 'varOnValue' / prefix_train
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
    metrics['break_point'] = add_metrics(methods, 0, pred, y, testDepartement, True, scale, name, train_dir)
    
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
    
    for dept in testDepartement:
        susectibility_map(dept, sinister,
                    train_dir, dir_output, k_days,
                    False, scale, name, res, n,
                    start, stop, resolution)"""

    shapiro_wilk(pred[:,1], y[:,-1], dir_output / n)

    """
    realVspredict2d(pred,
        y,
        isBin,
        name,
        scale,
        dir_output / n,
        train_dir,
        geo,
        testDepartement,
        graphScale)"""

    #save_object(res, name+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output)
    save_object(res, name+'_pred.pkl', dir_output)
    outname = 'metrics'+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_bp.pkl'
    save_object(metrics, outname, dir_output)

def test_fusion_prediction(models,
                           args1,
                           args2,
                           XsetDaily,
                           XsetLocalized,
                           Xset,
                           Yset,
                           dir_output,
                           graphScale,
                        testDepartement,
                        testname,
                        methods,
                        prefix_train,
                        pos_feature,):
    
    scale = graphScale.scale

    y = Yset[Yset[:,-4] > 0]
    
    dir_predictor = root_graph / args2['train_dir'] / 'influenceClustering'

    metrics = {}

    # Graphs

    graph1 = read_object(args1['graph'], args1['dir_train'])
    graph2 = read_object(args1['graph'], args1['dir_train'])

    # Models
    n1 = args1['model_name']+'.pkl'
    n2 = args2['model_name']+'.pkl'
    model1 = read_object(n1, args1['dir_model'] /  args1['model_name'])
    model2 = read_object(n2, args2['dir_model'] / args2['model_name'])

    # Prediction,
    if args1['model_type'] == 'traditionnal':
        feature_selected = read_object('features.pkl', args1['dir_model'] / args1['model_name'])
        graph1._set_model(model1)
        pred1 = np.empty((XsetDaily.shape[0], y.shape[1]))
        pred1[:, :6] = XsetDaily[:, :6]
        pred1[:, -1] = graph1.predict_model_api_sklearn(XsetDaily, feature_selected,  args1['isBin'], args1['autoRegression'], args1['pos_feature'])
    else:
        test_loader = read_object('test_loader.pkl', args1['train_dir'])
        feature_selected = read_object('features.pkl', args1['dir_model'] / args1['model_name'])

        graph1._load_model_from_path(args1['train_dir'] / 'best.pt', args1['model_name']+'.pkl', device)

        predTensor, YTensor = graph1._predict_test_loader(test_loader, feature_selected, device=device, isBin= args1['isBin'],
                                                          autoRegression=args1['autoRegression'], pos_feature=args1['pos_feature'])

        pred1 = np.copy(y)
        pred1[:, -1] = predTensor.detach().cpu().numpy()

    if args2['model_type'] == 'traditionnal':
        feature_selected = read_object('features.pkl', args2['dir_model'] / args2['model_name'])
        graph2._set_model(model2)
        pred2 = graph2.predict_model_api_sklearn(XsetLocalized, feature_selected,  args1['isBin'], args2['autoRegression'], args2['pos_feature'])
    else:
        test_loader = read_object('test_loader.pkl', args2['dir_model'])
        feature_selected = read_object('features.pkl', args2['dir_model'] / args2['model_name'])

        graph2._load_model_from_path(args2['train_dir'] / 'best.pt', args2['model_name'], device)

        predTensor, YTensor = graph2._predict_test_loader(test_loader, feature_selected, device=device, isBin= args1['isBin'],
                                                          autoRegression=args2['autoRegression'], pos_feature=args2['pos_feature'])
        y = YTensor.detach().cpu().numpy()
        pred2 = predTensor.detach().cpu().numpy()
    # Fusion
    Xset = add_temporal_spatial_prediction(Xset[:, :-2], pred2.reshape(-1,1), pred1, graph1, pos_feature=pos_feature)
    i = 0
    for name, isBin, autoRegression in models:
        y = Yset[Yset[:,-4] > 0]
        n = name
        model_dir = args2['train_dir'] / Path('check_'+scaling + '/' + prefix_train + '/baseline/' + name + '/')
        
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

        pred[:, 0] = graphScale.predict_model_api_sklearn(Xset, features_selected, isBin, autoRegression, pos_feature)
        logger.info(f'pred min {np.nanmin(pred[:, 0])}, pred max : {np.nanmax(pred[:, 0])}')

        for nameDep in departements:
            mask = np.argwhere(y[:, 3] == name2int[nameDep])
            if mask.shape[0] == 0:
                continue
            if not isBin:
                predictor = read_object(nameDep+'Predictor'+str(scale)+'.pkl', dir_predictor)
            else:
                create_binary_predictor(pred, name, nameDep, dir_predictor, scale)
                predictor = read_object(nameDep+'Predictor'+name+str(scale)+'.pkl', dir_predictor)

            pred[mask[:,0], 1] = order_class(predictor, predictor.predict(pred[mask[:, 0], 0]))

        metrics[name] = add_metrics(methods, i, pred, y, testDepartement, isBin, scale, name, None)

        if isBin:
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

        save_object(res, name+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output / n)

        if testname == '69':
            start = '2018-01-01'
            stop = '2022-09-30'
        else:
            start = '2023-06-01'
            stop = '2023-09-30'
        
        """for dept in testDepartement:
            susectibility_map(dept, sinister,
                        train_dir, dir_output, k_days,
                        isBin, scale, name, res, n,
                        start, stop, resolution)"""

        shapiro_wilk(pred[:,0], y[:,-1], dir_output / n)

        """
        realVspredict2d(pred,
            y,
            isBin,
            name,
            scale,
            dir_output / n,
            train_dir,
            geo,
            testDepartement,
            graphScale)"""

        i += 1
    save_object(res, name+'_pred.pkl', dir_output)
    outname = 'metrics'+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_fusion.pkl'
    save_object(metrics, outname, dir_output)

#########################################################################################################
#                                                                                                       #
#                                         Inference                                                     #
#                                                                                                       #
#########################################################################################################

def create_inference(graph,
                     X : np.array,
                     Xtrain : np.array,
                     device : torch.device,
                     use_temporal_as_edges,
                     graphId : int,
                     ks : int,
                     scaling : str) -> tuple:

    Xset = preprocess_inference(X, Xtrain, scaling)

    graphId = [graphId]

    batch = []

    for id in graphId:
        if use_temporal_as_edges:
            x, e = construct_graph_set(graph, id, Xset, None, ks)
        else:
            x, e = construct_graph_with_time_series(graph, id, Xset, None, ks)
        if x is None:
            continue
        batch.append([torch.tensor(x, dtype=torch.float32, device=device), torch.tensor(e, dtype=torch.long, device=device)])
    if len(batch) == 0:
        return None, None
    features, edges = graph_collate_fn_no_label(batch)

    return features, edges


def create_inference_2D(graph, X : np.array, Xtrain : np.array, Ytrain : np.array,
                     device : torch.device, use_temporal_as_edges, ks : int, pos_feature : dict,
                     scaling : str) -> tuple:

    pass

    return