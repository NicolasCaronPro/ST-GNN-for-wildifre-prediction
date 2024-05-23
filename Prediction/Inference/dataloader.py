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

def preprocess(X : np.array, Y : np.array, scaling : str, maxDate : str, ks : int):
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
    print(np.argwhere(np.isnan(X))[:,1])

    print(f'X shape {X.shape}, X shape after removal nan {Xset.shape}, Y shape is {Yset.shape}')

    assert Xset.shape[0] > 0
    assert Xset.shape[0] == Yset.shape[0]

    # Get node's id and date for which weight is greater than 0
    nodeUseForLossCalculation = Yset[(Yset[:,-3] > 0) & (Yset[:,4] > ks)]
    dateUseForLossCalculation = np.unique(nodeUseForLossCalculation[:,4])

    # Sort by date
    ind = np.lexsort([Yset[:,4]])
    Xset = Xset[ind]
    Yset = Yset[ind]

    # Seperate date into training and validation
    #dateTrain, dateVal = train_test_split(dateUseForLossCalculation, test_size=test_size, random_state=42)
    #dateTrain = dateUseForLossCalculation[: int(dateUseForLossCalculation.shape[0] * (1 - test_size))]
    #dateVal = dateUseForLossCalculation[int(dateUseForLossCalculation.shape[0] * (1 - test_size)):]
    dateTrain =  dateUseForLossCalculation[dateUseForLossCalculation < allDates.index(maxDate)]
    dateVal =  dateUseForLossCalculation[dateUseForLossCalculation >= allDates.index(maxDate)]

    print(f'Size of the train set {dateTrain.shape[0]}, size of the val set {dateVal.shape[0]}')
    
    # Generate Xtrain and Xval
    trainMask = np.isin(Xset[:,4], dateTrain)

    valMask = np.isin(Xset[:,4], dateVal)

    Xtrain = Xset[trainMask, :]

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
    
    # Log
    print(f'Check {scaling} standardisation Train : {np.nanmax(Xset[trainMask,6:])}, {np.nanmin(Xset[trainMask,6:])}')
    print(f'Check {scaling} standardisation Val : {np.nanmax(Xset[valMask,6:])}, {np.nanmin(Xset[valMask,6:])}')

    return Xset, Yset

def preprocess_test(X, Y, Xtrain, scaling):

    Xset, Yset = remove_nan_nodes(X, Y)
    
    print(f'X shape {X.shape}, X shape after removal nan {Xset.shape}')
    assert Xset.shape[0] > 0

    # Sort by date
    ind = np.lexsort([Yset[:,4]])
    Xset = Xset[ind]
    Yset = Yset[ind]

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
        
    print(f'Check {scaling} standardisation test : {np.nanmax(Xset[:,6:])}, {np.nanmin(Xset[:,6:])}')

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
                    Xset,
                    Yset,
                    maxDate,
                    use_temporal_as_edges : bool,
                    device,
                    ks : int):
    """
    
    """
    dateTrain = np.unique(Xset[Xset[:,4] < allDates.index(maxDate)][:,4])
    dateVal = np.unique(Xset[Xset[:,4] > allDates.index(maxDate)][:,4])
    
    nodeUseForLossCalculation = Yset[Yset[:,-3] > 0]

    dateUseForLossCalculation = np.unique(nodeUseForLossCalculation[:,4])
    
    Xst = []
    Yst = []
    Est = []

    XsV = []
    YsV = []
    EsV = []

    for id in dateUseForLossCalculation:
        if use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, id, Xset, Yset, ks)
        else:
            x, y, e = construct_graph_with_time_series(graph, id, Xset, Yset, ks)
        if x is None:
            continue
        
        if id in dateTrain:
            Xst.append(x)
            Yst.append(y)
            Est.append(e)
        elif id in dateVal:
            XsV.append(x)
            YsV.append(y)
            EsV.append(e)
        else:
            ValueError('Unknow date', id)
    
    assert len(Xst) > 0
    assert len(XsV) > 0

    trainDataset = InplaceGraphDataset(Xst, Yst, Est, len(Xst), device)
    valDataset = InplaceGraphDataset(XsV, YsV, EsV, len(XsV), device)
    return trainDataset, valDataset

def train_val_data_loader(graph,
                          Xset,
                          Yset,
                          maxDate,
                        use_temporal_as_edges : bool,
                        batch_size,
                        device,
                        ks : int) -> None:

    trainDataset, valDataset = create_dataset(graph,
                                Xset, Yset, maxDate, use_temporal_as_edges,
                                device, ks)

    trainLoader = DataLoader(trainDataset, batch_size, True, collate_fn=graph_collate_fn)
    valLoader = DataLoader(valDataset, valDataset.__len__(), False, collate_fn=graph_collate_fn)
    return trainLoader, valLoader


def train_val_data_loader_2D(graph,
                             Xset : np.array,
                             Yset : np.array,
                             maxDate : str,
                            use_temporal_as_edges : bool,
                            batch_size : int,
                            device: torch.device,
                            scaling : str,
                            pos_feature : dict,
                            pos_feature_2D : dict,
                            ks : int,
                            shape : tuple,
                            path : Path) -> None:
    
    
    nodeUseForLossCalculation = Yset[Yset[:,-3] > 0]

    dateUseForLossCalculation = np.unique(nodeUseForLossCalculation[:,4])
    
    Xst = []
    Yst = []
    Est = []

    XsV = []
    YsV = []
    EsV = []

    Xtrain = Xset[Xset[:,4] < allDates.index(maxDate)]
    Ytrain = Yset[Yset[:,4] < allDates.index(maxDate)]

    Xval = Xset[Xset[:,4] < allDates.index(maxDate)]

    dateTrain = np.unique(Xtrain[:,4])
    dateVal = np.unique(Xval[:,4])
    
    for id in dateUseForLossCalculation:
        if use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, id, Xset, Yset, ks)
        else:
            x, y, e = construct_graph_with_time_series(graph, id, Xset, Yset, ks)

        if x is None:
            continue

        x = load_x_from_pickle(x, shape, path, Xtrain, Ytrain, scaling, pos_feature, pos_feature_2D)
        save_object(x, str(id)+'.pkl', path / scaling)
        
        if id in dateTrain:
            Xst.append(str(id)+'.pkl')
            Yst.append(y)
            Est.append(e)
        elif id in dateVal:
            XsV.append(str(id)+'.pkl')
            YsV.append(y)
            EsV.append(e)
        else:
            ValueError('Unknow date', id)
    
    assert len(Xst) > 0
    assert len(XsV) > 0
    print(len(Xst))
    
    trainDataset = ReadGraphDataset_2D(Xst, Yst, Est, len(Xst), device, path / scaling)
    valDataset = ReadGraphDataset_2D(XsV, YsV, EsV, len(XsV), device, path  / scaling)
    
    trainLoader = DataLoader(trainDataset, batch_size, True, collate_fn=graph_collate_fn)
    valLoader = DataLoader(valDataset, valDataset.__len__(), False, collate_fn=graph_collate_fn)
    return trainLoader, valLoader

#########################################################################################################
#                                                                                                       #
#                                         test                                                          #
#                                                                                                       #
#########################################################################################################

def create_test(graph, X : np.array, Y : np.array, Xtrain : np.array, 
                device : torch.device, use_temporal_as_edges: bool,
                ks :int, scaling :str) -> tuple:
    """
    """
    assert use_temporal_as_edges is not None

    Xset, Yset = preprocess_test(X, Y , Xtrain, scaling)

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

    XTensor, YTensor, ETensor = create_test(graph=graphScale, X=X, Y=Y, Xtrain=Xtrain,
                                device=device, use_temporal_as_edges=use_temporal_as_edges, ks=k_days, scaling=scaling)

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