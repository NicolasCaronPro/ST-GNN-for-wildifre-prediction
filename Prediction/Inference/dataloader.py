from torch_geometric.data import Dataset
from torch.utils.data import DataLoader
import torch
from tools import *

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
        #x = self.X[index]
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

def preprocess(X : np.array, Y : np.array, scaling : str, maxDate : str, trainDate : str, trainDepartements : list, ks : int):
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
    print(f'We found nan at {np.unique(np.argwhere(np.isnan(X))[:,1])}')

    print(f'X shape {X.shape}, X shape after removal nan {Xset.shape}, Y shape is {Yset.shape}')

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

    trainCode = [name2int[dept] for dept in trainDepartements]

    Xtrain, Ytrain = (Xset[(Xset[:,4] < allDates.index(trainDate)) & (np.isin(Xset[:,3], trainCode))],
                      Yset[(Xset[:,4] < allDates.index(trainDate)) & (np.isin(Xset[:,3], trainCode))])
    

    # Define the scale method
    if scaling == 'MinMax':
        scaler = min_max_scaler
    elif scaling == 'z-score':
        scaler = standart_scaler
    elif scaling == "robust":
        scaler = robust_scaler
    else:
        ValueError('Unknow')
        exit(1)

    # Scale Features
    for featureband in range(6, Xset.shape[1]):
        Xset[:,featureband] = scaler(Xset[:,featureband], Xtrain[:, featureband], concat=False)
    
    Xtrain, Ytrain = (Xset[(Xset[:,4] < allDates.index(trainDate)) & (np.isin(Xset[:,3], trainCode))],
                      Yset[(Xset[:,4] < allDates.index(trainDate)) & (np.isin(Xset[:,3], trainCode))])

    Xval, Yval = (Xset[(Xset[:,4] >= allDates.index(trainDate)) & (Xset[:,4] < allDates.index(maxDate)) & (np.isin(Xset[:,3], trainCode))],
                    Yset[(Xset[:,4] >= allDates.index(trainDate)) & (Xset[:,4] < allDates.index(maxDate)) & (np.isin(Xset[:,3], trainCode))])
    
    Xtest, Ytest = (Xset[((Xset[:,4] >= allDates.index(maxDate)) & (np.isin(Xset[:,3], trainCode))) | ((Xset[:,4] >= allDates.index('2018-01-01')) & (Xset[:,4] < allDates.index(maxDate)) & (~np.isin(Xset[:,3], trainCode)))],
                    Yset[((Xset[:,4] >= allDates.index(maxDate)) & (np.isin(Xset[:,3], trainCode))) | ((Xset[:,4] >= allDates.index('2018-01-01')) & (Xset[:,4] < allDates.index(maxDate)) & (~np.isin(Xset[:,3], trainCode)))])
    
    logger.info(f'Size of the train set {Xtrain.shape[0]}, size of the val set {Xval.shape[0]}, size of the test set {Xtest.shape[0]}')
    logger.info(f'Unique train dates set {np.unique(Xtrain[:,4]).shape[0]}, val dates {np.unique(Xval[:,4]).shape[0]}, test dates {np.unique(Xtest[:,4]).shape[0]}')

    # Log
    logger.info(f'Check {scaling} standardisation Train : {np.nanmax(Xtrain[:,6:]), np.nanargmax(Xtrain[:,6:])}, {np.nanmin(Xtrain[:,6:]), np.nanargmin(Xtrain[:,6:])}')
    logger.info(f'Check {scaling} standardisation Val : {np.nanmax(Xval[:,6:]), np.nanargmax(Xval[:,6:])}, {np.nanmin(Xval[:,6:]), np.nanargmin(Xval[:,6:])}')
    logger.info(f'Check {scaling} standardisation Test : {np.nanmax(Xtest[:,6:]), np.nanargmax(Xtrain[:,6:])}, {np.nanmin(Xtest[:,6:]), np.nanargmin(Xtest[:,6:])}')

    return (Xtrain, Ytrain), (Xval, Yval), (Xtest, Ytest)

def preprocess_test(X, Y, Xtrain, scaling):

    Xset, Yset = remove_nan_nodes(X, Y)
    
    print(f'X shape {X.shape}, X shape after removal nan {Xset.shape}')
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

def test_sklearn_api_model(graphScale, Xset, Yset, Xtrain, Ytrain,
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
                           train_dir):
    
    if Xset.shape[0] == 0:
        logger.info(f'{testname} is empty, skip')
        return
    
    scale = graphScale.scale
    metrics = {}

    ################################ Ground Truth ############################################
    logger.info('#########################')
    logger.info(f'       GT              ')
    logger.info('#########################')

    if Rewrite:
        XTensor, YTensor, ETensor = load_tensor_test(use_temporal_as_edges=True, graphScale=graphScale, dir_output=dir_output,
                                                            X=Xset, Y=Yset, Xtrain=Xtrain, Ytrain=Ytrain, device=device, k_days=k_days,
                                                            test=testname, pos_feature=pos_feature,
                                                            scaling=scaling, prefix=prefix, encoding=encoding,
                                                            Rewrite=False)
        
    ITensor = YTensor[:,-3]
    Ypred = torch.masked_select(YTensor[:,-1], ITensor.gt(0))
    YTensor = YTensor[ITensor.gt(0)]
    
    metrics['GT'] = add_metrics(methods, 0, Ypred, YTensor, testDepartement, False, scale, 'gt', train_dir)
    i = 1

    #################################### Traditionnal ##################################################

    for name, isBin in models:
        n = name+'_'+prefix_train+'_'+str(scale)+'_'+scaling + '_' + encoding+'_'+testname
        model_dir = train_dir / Path('check_'+scaling + '/' + prefix_train + '/baseline/' + name + '/')
        
        logger.info('#########################')
        logger.info(f'      {name}            ')
        logger.info('#########################')
        XTensor, YTensor, _ = load_tensor_test(use_temporal_as_edges=False, graphScale=graphScale, dir_output=dir_output,
                                                        X=Xset, Y=Yset, Xtrain=Xtrain, Ytrain=Ytrain, device=device, k_days=k_days,
                                                        test=testname, pos_feature=pos_feature,
                                                        scaling=scaling, prefix=prefix, encoding=encoding,
                                                        Rewrite=True)
        
        features_selected = read_object('features.pkl', model_dir)
        log_features(features_selected, graphScale.scale, pos_feature, ['min', 'mean', 'max', 'std'])


        XTensor = XTensor.detach().cpu().numpy()
        XTensor = XTensor[:, features_selected, -1]

        YTensor = YTensor[:,:, -1]
        ITensor = YTensor[:,-3]

        model = read_object(name+'.pkl', model_dir)
        if model is None:
            continue
        graphScale._set_model(model)
        Ypred = graphScale.predict_model_api_sklearn(XTensor, isBin)
        
        Ypred = torch.tensor(Ypred, dtype=torch.float32, device=device)
        YTensor = YTensor[ITensor.gt(0)]
        Ypred = torch.masked_select(Ypred, ITensor.gt(0))

        if not dummy:
            metrics[name] = add_metrics(methods, i, Ypred, YTensor, testDepartement, isBin, scale, name, train_dir)
        else:
            metrics[name +'_dummy'] = add_metrics(methods, i, Ypred, YTensor, testDepartement, isBin, scale, name, train_dir)

        pred = Ypred.detach().cpu().numpy()
        y = YTensor.detach().cpu().numpy()

        if isBin:
            y[:,-1] = y[:,-1] / np.nanmax(y[:,-1])

        realVspredict(pred, y,
                      dir_output / n, name+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'.png')
        
        res = np.empty((pred.shape[0], y.shape[1] + 3))
        res[:, :y.shape[1]] = y
        res[:,y.shape[1]] = pred
        res[:,y.shape[1]+1] = np.sqrt((y[:,-3] * (pred - y[:,-1]) ** 2))
        res[:,y.shape[1]+2] = np.sqrt(((pred - y[:,-1]) ** 2))

        save_object(res, name+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output / n)

        for dept in testDepartement:
            dir_mask = train_dir / 'raster' / '2x2'
            resSum = array2image(res, dir_mask, scale, dept, 'sum', -3, dir_output / n, dept+'_prediction.pkl')
            ySum = array2image(y, dir_mask, scale, dept, 'sum', -1, dir_output / n, dept+'_gt.pkl')
            fireSum = array2image(y, dir_mask, scale, dept, 'sum', -2, dir_output / n, dept+'_fire.pkl')
            
            fig, ax = plt.subplots(1, 3, figsize=(10,5))
            im = ax[0].imshow(resSum, vmin=np.nanmin(resSum), vmax=np.nanmax(resSum),  cmap='jet')
            ax[0].set_title('Sum of Prediction')
            fig.colorbar(im, ax=ax[0], orientation='vertical')
            im = ax[1].imshow(ySum, vmin=np.nanmin(ySum), vmax=np.nanmax(ySum),  cmap='jet')
            ax[1].set_title('Sum of Target')
            fig.colorbar(im, ax=ax[1], orientation='vertical')
            im = ax[2].imshow(fireSum, vmin=np.nanmin(fireSum), vmax=np.nanmax(fireSum),  cmap='jet')
            ax[2].set_title('Sum of Fire')
            fig.colorbar(im, ax=ax[2], orientation='vertical')
            susec = dept+'_susecptibility.png'
            plt.tight_layout()
            plt.savefig(dir_output / n / susec)
            plt.close('all')

        shapiro_wilk(pred, y[:,-1], dir_output / n)

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

    ######################################## Simple model ##################################
    if not dummy:
        XTensor, YTensor, ETensor = load_tensor_test(use_temporal_as_edges=True, graphScale=graphScale, dir_output=dir_output,
                                                            X=Xset, Y=Yset, Xtrain=Xtrain, Ytrain=Ytrain, device=device, k_days=k_days,
                                                            test=testname, pos_feature=pos_feature,
                                                            scaling=scaling, prefix=prefix, encoding=encoding,
                                                            Rewrite=False)
        ITensor = YTensor[:,-3]
        if testname != 'dummy':
            """y2 = YTensor.detach().cpu().numpy()

            logger.info('#########################')
            logger.info(f'      perf_Y            ')
            logger.info('#########################')
            name = 'perf_Y'
            Ypred = graphScale._predict_perference_with_Y(y2, False)
            Ypred = torch.tensor(Ypred, dtype=torch.float32, device=device)
            YTensor = YTensor[ITensor.gt(0)]
            Ypred = torch.masked_select(Ypred, ITensor.gt(0))
            metrics[name] = add_metrics(methods, 0, Ypred, YTensor, testDepartement, False, scale, name)

            pred = Ypred.detach().cpu().numpy()
            y = YTensor.detach().cpu().numpy()

            res = np.empty((pred.shape[0], y.shape[1] + 3))
            res[:, :y.shape[1]] = y
            res[:,y.shape[1]] = pred
            res[:,y.shape[1]+1] = np.sqrt((y[:,-3] * (pred - y[:,-1]) ** 2))
            res[:,y.shape[1]+2] = np.sqrt(((pred - y[:,-1]) ** 2))

            save_object(res, name+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output)

            logger.info('#########################')
            logger.info(f'      perf_Y_bin         ')
            logger.info('#########################')
            name = 'perf_Y_bin'
            Ypred = graphScale._predict_perference_with_Y(y2, True)
            Ypred = torch.tensor(Ypred, dtype=torch.float32, device=device)
            Ypred = torch.masked_select(Ypred, ITensor.gt(0))
            metrics[name] = add_metrics(methods, 0, Ypred, YTensor, testDepartement, True, scale, name, path(nameExp))

            pred = Ypred.detach().cpu().numpy()
            y = YTensor.detach().cpu().numpy()

            res = np.empty((pred.shape[0], y.shape[1] + 3))
            res[:, :y.shape[1]] = y
            res[:,y.shape[1]] = pred
            res[:,y.shape[1]+1] = np.sqrt((y[:,-3] * (pred - y[:,-1]) ** 2))
            res[:,y.shape[1]+2] = np.sqrt(((pred - y[:,-1]) ** 2))

            save_object(res, name+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output)"""

        """logger.info('#########################')
        logger.info(f'      perf_X         ')
        logger.info('#########################')
        name = 'perf_X'
        XTensor, YTensor, ETensor = load_tensor_test(use_temporal_as_edges=True, graphScale=graphScale, dir_output=dir_output,
                                                            X=X, Y=Y, Xtrain=Xtrain, Ytrain=Ytrain, device=device, k_days=k_days,
                                                            test=testname, pos_feature=pos_feature,
                                                            scaling=scaling, prefix=prefix, encoding=encoding)
                                                            
        x = XTensor.detach().cpu().numpy()
        
        Ypred = graphScale._predict_perference_with_X(x, pos_feature)
        Ypred = torch.tensor(Ypred, dtype=torch.float32, device=device)
        YTensor = YTensor[ITensor.gt(0)]
        Ypred = torch.masked_select(Ypred, ITensor.gt(0))
        metrics['perf_X'] = add_metrics(methods, 0, Ypred, YTensor, testDepartement, False, scale, name, path(nameExp))

        pred = Ypred.detach().cpu().numpy()
        y = YTensor.detach().cpu().numpy()

        res = np.empty((pred.shape[0], y.shape[1] + 3))
        res[:, :y.shape[1]] = y
        res[:,y.shape[1]] = pred
        res[:,y.shape[1]+1] = np.sqrt((y[:,-3] * (pred - y[:,-1]) ** 2))
        res[:,y.shape[1]+2] = np.sqrt(((pred - y[:,-1]) ** 2))

        save_object(res, name+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output)"""

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
                           shape2D):
    
    scale = graphScale.scale
    metrics = {}
    ################################ Ground Truth ############################################
    logger.info('#########################')
    logger.info(f'       GT              ')
    logger.info('#########################')

    if Rewrite:
        XTensor, YTensor, ETensor = load_tensor_test(use_temporal_as_edges=True, graphScale=graphScale, dir_output=dir_output,
                                                            X=Xset, Y=Yset, Xtrain=Xtrain, Ytrain=Ytrain, device=device, k_days=k_days,
                                                            test=testname, pos_feature=pos_feature,
                                                            scaling=scaling, prefix=prefix, encoding=encoding, Rewrite=True)

        _ = load_loader_test(use_temporal_as_edges=False, graphScale=graphScale, dir_output=dir_output,
                                        X=Xset, Y=Yset, Xtrain=Xtrain, Ytrain=Ytrain,
                                        device=device, k_days=k_days, test=testname, pos_feature=pos_feature,
                                        scaling=scaling, encoding=encoding, prefix=prefix, Rewrite=Rewrite)
        
        """_= load_loader_test_2D(use_temporal_as_edges=False, graphScale=graphScale,
                                              dir_output=dir_output / '2D' / prefix / 'data', X=Xset, Y=Yset,
                                            Xtrain=Xtrain, Ytrain=Ytrain, device=device, k_days=k_days, test=testname,
                                            pos_feature=pos_feature, scaling=scaling, prefix=prefix,
                                        shape=(shape2D[0], shape2D[1], newShape2D), pos_feature_2D=pos_feature_2D, encoding=encoding,
                                        Rewrite=Rewrite)"""
        
    ITensor = YTensor[:,-3]
    Ypred = torch.masked_select(YTensor[:,-1], ITensor.gt(0))
    YTensor = YTensor[ITensor.gt(0)]
    
    metrics['GT'] = add_metrics(methods, 0, Ypred, YTensor, testDepartement, False, scale, 'gt', train_dir)

    i = 1

    #################################### GNN ###################################################
    for mddel, use_temporal_as_edges, isBin, is_2D_model  in models:
        n = mddel+'_'+prefix_train+'_'+str(scale)+'_'+scaling + '_' + encoding+'_'+testname
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

        Ypred, YTensor = graphScale._predict_test_loader(test_loader, features_selected, device=device)

        metrics[mddel] = add_metrics(methods, i, Ypred, YTensor, testDepartement, isBin, scale, mddel, train_dir)

        pred = Ypred.detach().cpu().numpy()
        y = YTensor.detach().cpu().numpy()

        realVspredict(pred, y,
                      dir_output / n,
                      mddel+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'.png')
        
        res = np.empty((pred.shape[0], y.shape[1] + 3))
        res[:, :y.shape[1]] = y
        res[:,y.shape[1]] = pred
        res[:,y.shape[1]+1] = np.sqrt((y[:,-3] * (pred - y[:,-1]) ** 2))
        res[:,y.shape[1]+2] = np.sqrt(((pred - y[:,-1]) ** 2))

        save_object(res, mddel+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+testname+'_pred.pkl', dir_output / n)

        for dept in testDepartement:
            dir_mask = train_dir / 'raster' / '2x2'
            resSum = array2image(res, dir_mask, scale, dept, 'sum', -3, dir_output / n, dept+'_prediction.pkl')
            ySum = array2image(y, dir_mask, scale, dept, 'sum', -1, dir_output / n, dept+'_gt.pkl')
            fireSum = array2image(y, dir_mask, scale, dept, 'sum', -2, dir_output / n, dept+'_fire.pkl')
            
            fig, ax = plt.subplots(1, 3, figsize=(10,5))
            im = ax[0].imshow(resSum, vmin=np.nanmin(resSum), vmax=np.nanmax(resSum),  cmap='jet')
            ax[0].set_title('Sum of Prediction')
            fig.colorbar(im, ax=ax[0], orientation='vertical')
            im = ax[1].imshow(ySum, vmin=np.nanmin(ySum), vmax=np.nanmax(ySum),  cmap='jet')
            ax[1].set_title('Sum of Target')
            fig.colorbar(im, ax=ax[1], orientation='vertical')
            im = ax[2].imshow(fireSum, vmin=np.nanmin(fireSum), vmax=np.nanmax(fireSum),  cmap='jet')
            ax[2].set_title('Sum of Fire')
            fig.colorbar(im, ax=ax[2], orientation='vertical')
            susec = dept+'_susecptibility.png'
            plt.tight_layout()
            plt.savefig(dir_output / n / susec)
            plt.close('all')

        shapiro_wilk(pred, y[:,-1], dir_output / n)

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

    return features[:,:,0], edges


def create_inference_2D(graph, X : np.array, Xtrain : np.array, Ytrain : np.array,
                     device : torch.device, use_temporal_as_edges, ks : int, pos_feature : dict,
                     scaling : str) -> tuple:

    pass

    return