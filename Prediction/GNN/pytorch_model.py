from torch_geometric.data import Dataset
from torch.utils.data import DataLoader
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
from feature_engine.selection import SmartCorrelatedSelection

from GNN.discretization import *

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

        if len(self.edges) > 0:
            edges = self.edges[index]
        else:
            edges = []

        return torch.tensor(x, dtype=torch.float32, device=self.device), \
            torch.tensor(y, dtype=torch.float32, device=self.device), \
            torch.tensor(edges, dtype=torch.long, device=self.device)

    def __len__(self) -> int:
        return self.leni
    
    def len(self):
        pass

    def get(self):
        pass

def graph_collate_fn(batch):
    #node_indice_list = []
    edge_index_list = []
    node_features_list = []
    node_labels_list = []
    graph_labels_list = []
    num_nodes_seen = 0
    for graph_id, features_labels_edge_index_tuple in enumerate(batch):
        # Collecter les caractéristiques et les étiquettes des nœuds
        node_features_list.append(features_labels_edge_index_tuple[0])
        node_labels_list.append(features_labels_edge_index_tuple[1])
        
        edge_index = features_labels_edge_index_tuple[2]  # Tous les composants sont dans la plage [0, N]
        
        # Ajuster l'index des arêtes en fonction du nombre de nœuds vus
        if edge_index.shape[0] > 2:
            edge_index[0] += num_nodes_seen
            edge_index[1] += num_nodes_seen
            edge_index_list.append(edge_index)
        else:
            edge_index_list.append(edge_index + num_nodes_seen)
        
        # Ajouter l'ID du graphe pour chaque nœud
        num_nodes = features_labels_edge_index_tuple[1].size(0)
        graph_labels_list.append(torch.full((num_nodes,), graph_id, dtype=torch.long))  # Création d'un tensor d'IDs

        num_nodes_seen += num_nodes  # Mettre à jour le nombre de nœuds vus

    # Merge the PPI graphs into a single graph with multiple connected components
    node_features = torch.cat(node_features_list, 0)
    node_labels = torch.cat(node_labels_list, 0)
    edge_index = torch.cat(edge_index_list, 1)
    graph_labels_list = torch.cat(graph_labels_list, 0).to(device)
    #node_indices = torch.cat(node_indice_list, 0)
    return node_features, node_labels, edge_index, graph_labels_list

def graph_collate_fn_hybrid(batch):
    edge_index_list = []
    node_features_list = []
    node_features_list_2D = []
    node_labels_list = []
    graph_labels_list = []
    num_nodes_seen = 0

    for graph_id, features_labels_edge_index_tuple in enumerate(batch):
        node_features_list.append(features_labels_edge_index_tuple[0])
        node_features_list_2D.append(features_labels_edge_index_tuple[1])
        node_labels_list.append(features_labels_edge_index_tuple[2])
        edge_index = features_labels_edge_index_tuple[3]  # all of the components are in the [0, N] range
        
        # Adjust edge indices
        if edge_index.shape[0] > 2:
            edge_index[0] += num_nodes_seen
            edge_index[1] += num_nodes_seen
            edge_index_list.append(edge_index)
        else:
            edge_index_list.append(edge_index + num_nodes_seen)
        
        # Add graph ID for each node
        num_nodes = features_labels_edge_index_tuple[1].size(0)
        graph_labels_list.append(torch.full((num_nodes,), graph_id, dtype=torch.long))

        num_nodes_seen += num_nodes  # Update the number of nodes seen

    # Merge all components into single tensors
    node_features = torch.cat(node_features_list, 0)
    node_features_2D = torch.cat(node_features_list_2D, 0)
    node_labels = torch.cat(node_labels_list, 0)
    edge_index = torch.cat(edge_index_list, 1)
    graph_labels = torch.cat(graph_labels_list, 0).to(device)

    return node_features, node_features_2D, node_labels, edge_index, graph_labels

def graph_collate_fn_no_label(batch):
    edge_index_list = []
    node_features_list = []
    graph_labels_list = []
    num_nodes_seen = 0

    for graph_id, features_labels_edge_index_tuple in enumerate(batch):
        node_features_list.append(features_labels_edge_index_tuple[0])
        edge_index = features_labels_edge_index_tuple[1]
        
        # Adjust edge indices
        edge_index_list.append(edge_index + num_nodes_seen)
        
        # Add graph ID for each node
        num_nodes = features_labels_edge_index_tuple[0].size(0)
        graph_labels_list.append(torch.full((num_nodes,), graph_id, dtype=torch.long))

        num_nodes_seen += len(features_labels_edge_index_tuple[0])  # Update the number of nodes seen

    # Merge all components into single tensors
    node_features = torch.cat(node_features_list, 0)
    edge_index = torch.cat(edge_index_list, 1)
    graph_labels = torch.cat(graph_labels_list, 0).to(device)

    return node_features, edge_index, graph_labels

def graph_collate_fn_adj_mat(batch):
    node_features_list = []
    node_labels_list = []
    edge_index_list = []
    graph_labels_list = []
    num_nodes_seen = 0

    for graph_id, features_labels_edge_index_tuple in enumerate(batch):
        node_features_list.append(features_labels_edge_index_tuple[0])
        node_labels_list.append(features_labels_edge_index_tuple[1])
        edge_index = features_labels_edge_index_tuple[2]
        
        # Adjust edge indices
        edge_index_list.append(edge_index + num_nodes_seen)
        
        # Add graph ID for each node
        num_nodes = features_labels_edge_index_tuple[1].size(0)
        graph_labels_list.append(torch.full((num_nodes,), graph_id, dtype=torch.long))

        num_nodes_seen += num_nodes  # Update the number of nodes seen

    # Merge all components into single tensors
    node_features = torch.cat(node_features_list, 0)
    node_labels = torch.cat(node_labels_list, 0)
    edge_index = torch.cat(edge_index_list, 1)

    # Create adjacency matrix
    adjacency_matrix = torch.zeros(num_nodes_seen, num_nodes_seen)
    for edge in edge_index.t():  # Transpose to iterate through edge pairs
        node1, node2 = edge[0].item(), edge[1].item()
        adjacency_matrix[node1, node2] = 1
        adjacency_matrix[node2, node1] = 1  # Since it's an undirected graph

    graph_labels = torch.cat(graph_labels_list, 0)

    return node_features, node_labels, adjacency_matrix, graph_labels.to(device)

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

        if len(self.edges) > 0:
            edges = self.edges[index]
        else:
            edges = []

        return torch.tensor(x, dtype=torch.float32, device=self.device), \
            torch.tensor(y, dtype=torch.float32, device=self.device), \
            torch.tensor(edges, dtype=torch.long, device=self.device),  \

    def __len__(self) -> int:
        return self.leni
    
    def len(self):
        pass

    def get(self):
        pass

def construct_dataset(date_ids, x_data, y_data, graph, ids_columns, ks, use_temporal_as_edges):
    Xs, Ys, Es = [], [], []
    
    if graph.graph_method == 'graph':
        # Traiter par identifiant de graph
        graphId = np.unique(x_data[:, graph_id_index])
        for id in graphId:
            x_data_graph = x_data[x_data[:, graph_id_index] == id]
            y_data_graph = y_data[y_data[:, graph_id_index] == id]
            for date_id in date_ids:
                if date_id not in np.unique(x_data_graph[:, date_index]):
                    continue
                if use_temporal_as_edges is None:
                    x, y = construct_time_series(date_id, x_data_graph, y_data_graph, ks, len(ids_columns))
                    if x is not None:
                        for i in range(x.shape[0]):                
                            Xs.append(x[i])
                            Ys.append(y[i])
                    continue
                elif use_temporal_as_edges:
                    x, y, e = construct_graph_set(graph, date_id, x_data_graph, y_data_graph, ks, len(ids_columns))
                else:
                    x, y, e = construct_graph_with_time_series(graph, date_id, x_data_graph, y_data_graph, ks, len(ids_columns))
                
                if x is None:
                    continue
                
                Xs.append(x)
                Ys.append(y)
                Es.append(e)
    
    else:
        # Traiter par date
        for id in date_ids:
            if use_temporal_as_edges is None:
                x, y = construct_time_series(id, x_data, y_data, ks, len(ids_columns))
                if x is not None:
                    for i in range(x.shape[0]):                
                        Xs.append(x[i])
                        Ys.append(y[i])
                continue
            elif use_temporal_as_edges:
                x, y, e = construct_graph_set(graph, id, x_data, y_data, ks, len(ids_columns))
            else:
                x, y, e = construct_graph_with_time_series(graph, id, x_data, y_data, ks, len(ids_columns))
            
            if x is None:
                continue
            
            Xs.append(x)
            Ys.append(y)
            Es.append(e)
    
    return Xs, Ys, Es

def create_dataset(graph,
                    df_train,
                    df_val,
                    df_test,
                    features_name,
                    target_name,
                    use_temporal_as_edges : bool,
                    device,
                    ks : int):
    
    x_train, y_train = df_train[ids_columns + features_name].values, df_train[ids_columns + targets_columns + [target_name]].values
    
    x_val, y_val = df_val[ids_columns + features_name].values, df_val[ids_columns + targets_columns + [target_name]].values

    x_test, y_test = df_test[ids_columns + features_name].values, df_test[ids_columns + targets_columns + [target_name]].values

    dateTrain = np.sort(np.unique(y_train[y_train[:, weight_index] > 0, date_index]))
    dateVal = np.sort(np.unique(y_val[y_val[:, weight_index] > 0, date_index]))
    dateTest = np.sort(np.unique(y_test[y_test[:, weight_index] > 0, date_index]))

    logger.info(f'Constructing train Dataset')
    Xst, Yst, Est = construct_dataset(dateTrain, x_train, y_train, graph, ids_columns, ks, use_temporal_as_edges)

    logger.info(f'Constructing val Dataset')
    XsV, YsV, EsV = construct_dataset(dateVal, x_val, y_val, graph, ids_columns, ks, use_temporal_as_edges)

    logger.info(f'Constructing test Dataset')
    XsTe, YsTe, EsTe = construct_dataset(dateTest, x_test, y_test, graph, ids_columns, ks, use_temporal_as_edges)

    # Assurez-vous que les ensembles ne sont pas vides
    assert len(Xst) > 0, "Le jeu de données d'entraînement est vide"
    assert len(XsV) > 0, "Le jeu de données de validation est vide"
    assert len(XsTe) > 0, "Le jeu de données de test est vide"

    # Création des datasets finaux
    train_dataset = InplaceGraphDataset(Xst, Yst, Est, len(Xst), device)
    val_dataset = InplaceGraphDataset(XsV, YsV, EsV, len(XsV), device)
    test_dataset = InplaceGraphDataset(XsTe, YsTe, EsTe, len(XsTe), device)

    return train_dataset, val_dataset, test_dataset

def create_test_loader(graph, df,
                       features_name,
                       device : torch.device,
                       use_temporal_as_edges : bool,
                       target_name,
                       ks :int):
    
    Xset, Yset = df[ids_columns + features_name].values, df[ids_columns + targets_columns + [target_name]].values

    X = []
    Y = []
    E = []

    if graph.graph_method == 'graph':
        graphId = np.unique(Xset[:, graph_id_index])
        for id in graphId:
            Xset_graph = Xset[Xset[:, graph_id_index] == id]
            Yset_graph = Yset[Yset[:, graph_id_index] == id]
            udates = np.unique(Xset_graph[:, date_index])
            for date in udates:
                if use_temporal_as_edges is None:
                    x, y = construct_time_series(date, Xset_graph, Yset_graph, ks, len(ids_columns))
                    if x is not None:
                        for i in range(x.shape[0]):
                            X.append(x[i])
                            Y.append(y[i])
                    continue
                elif use_temporal_as_edges:
                    x, y, e = construct_graph_set(graph, date, Xset_graph, Yset_graph, ks, len(ids_columns))
                else:
                    x, y, e = construct_graph_with_time_series(graph, date, Xset_graph, Yset_graph, ks, len(ids_columns))

                if x is None:
                    continue

                X.append(x)
                Y.append(y)
                E.append(e)
    else:
        graphId = np.unique(Xset[:, date_index])
        for date in graphId:
            if use_temporal_as_edges is None:
                x, y = construct_time_series(date, Xset, Yset, ks, len(ids_columns))
                if x is not None:
                    for i in range(x.shape[0]):
                        X.append(x[i])
                        Y.append(y[i])
                continue
            elif use_temporal_as_edges:
                x, y, e = construct_graph_set(graph, date, Xset, Yset, ks, len(ids_columns))
            else:
                x, y, e = construct_graph_with_time_series(graph, date, Xset, Yset, ks, len(ids_columns))

            if x is None:
                continue

            X.append(x)
            Y.append(y)
            E.append(e)

    dataset = InplaceGraphDataset(X, Y, E, len(X), device)
    
    if use_temporal_as_edges is None:
        loader = DataLoader(dataset, dataset.__len__(), False)
    else:
        loader = DataLoader(dataset, dataset.__len__(), False, collate_fn=graph_collate_fn)

    return loader

def load_x_from_pickle(date : int,
                       path : Path,
                       features_name_2D : list,
                       features : list) -> np.array:
    
    features_name_2D_full, _ = get_features_name_lists_2D(6, features)

    leni = len(features_name_2D)
    x_2D = read_object(f'X_{date}.pkl', path)

    if x_2D is None:
        return None
    new_x_2D = np.empty((leni, x_2D.shape[1], x_2D.shape[2]))
    
    for i, fet_2D in enumerate(features_name_2D):

        new_x_2D[i, :, :] = x_2D[features_name_2D_full.index(fet_2D), :, :]
        if False not in np.isnan(new_x_2D[i, :, :]):
            return None
        else:
            nan_mask = np.isnan(new_x_2D[i, :, :])
            new_x_2D[i, nan_mask] = np.nanmean(new_x_2D[i, :, :])

    return new_x_2D

def generate_image_y(y, y_raster):
    res = np.empty((y.shape[1], *y_raster.shape, y.shape[-1]))
    for i in range(y.shape[0]):
        graph = y[i][graph_id_index][0]
        node = y[i][id_index][0]
        longitude = y[i][longitude_index][0]
        latitude = y[i][latitude_index][0]
        departement = y[i][departement_index][0]
        mask = y_raster == node
        if mask.shape[0] == 0:
            logger.info(f'{node} not in {np.unique(y_raster)}')
        res[graph_id_index, mask, :] = graph
        res[id_index, mask, :] = node
        res[latitude_index, mask, :] = latitude
        res[longitude_index, mask, :] = longitude
        res[departement_index, mask, :] = departement
        for j in range(weight_index, y.shape[1]):
            for k in range(y.shape[2]):
                res[j, mask, k] = y[i, j, k]

    return res

def process_dept_raster(dept, graph, path, y, features_name_2D, ks, image_per_node, shape2D):
    """Process a department's raster and return processed X and Y data"""
    
    if image_per_node:
        X = np.empty((y.shape[0], len(features_name_2D), *shape2D[graph.scale], ks + 1))
        Y = y
    else:
        X = np.empty((len(features_name_2D), 64, 64, ks + 1))
        Y = np.empty((64, 64, y.shape[1], ks + 1))

    raster_dept = read_object(f'{int2name[dept]}rasterScale{graph.scale}_{graph.base}_{graph.graph_method}_node.pkl', path / 'raster')
    assert raster_dept is not None

    raster_dept_graph = read_object(f'{int2name[dept]}rasterScale{graph.scale}_{graph.base}_{graph.graph_method}.pkl', path / 'raster')

    unodes = np.unique(raster_dept)
    unodes = unodes[~np.isnan(unodes)]
    
    return X, Y, raster_dept, raster_dept_graph, unodes

def process_time_step(X, Y, dept, graph, ks, id, path, features_name_2D, features, y, raster_dept, raster_dept_graph, unodes, image_per_node, shape2D):
    """Process each time step and update X and Y arrays"""
    for k in range(ks + 1):
        date = int(id) - (ks - k)
        x_date = load_x_from_pickle(date, path / f'2D_database_{graph.scale}_{graph.base}_{graph.graph_method}' / int2name[dept], features_name_2D, features)
        if x_date is None:
            return None, None
        if x_date is None:
            x_date = np.zeros((len(features_name_2D), raster_dept.shape[0], raster_dept.shape[1]))
        
        if image_per_node:
            X = process_node_images(X, unodes, x_date, raster_dept, y, graph.scale, k, date, shape2D)
        else:
            X = process_dept_images(X, x_date, raster_dept, ks, k)

    if not image_per_node:
        y_dept = generate_image_y(y, raster_dept_graph)
        Y = process_dept_y(Y, y_dept, raster_dept_graph, ks)
    
    return X, Y

def process_node_images(X, unodes, x_date, raster_dept, y, scale, k, date, shape2D):
    """Process images for each node."""
    node2remove = []
    for node in unodes:
        if node not in np.unique(y[:, graph_id_index]):
            node2remove.append(node)
            continue
        mask = np.argwhere(raster_dept == node)
        x_node = extract_node_data(x_date, mask, node, raster_dept)
        X = update_node_images(X, x_node, mask, scale, k, shape2D, y, date, node)
    
    return X

def extract_node_data(x_date, mask, node, raster_dept):
    """Extract and mask data for a specific node."""
    x_date_cp = np.copy(x_date)
    x_date_cp[:, raster_dept != node] = np.nan
    minx, miny, maxx, maxy = np.min(mask[:, 0]), np.min(mask[:, 1]), np.max(mask[:, 0]), np.max(mask[:, 1])
    return x_date_cp[:, minx:maxx+1, miny:maxy+1]

def update_node_images(X, x_node, mask, scale, k, shape2D, y, date, node):
    """Resize and update node images in X array."""
    for band in range(x_node.shape[0]):
        x_band = x_node[band]
        if False not in np.isnan(x_band):
            x_band[np.isnan(x_band)] = -1
        else:
            x_band[np.isnan(x_band)] = np.nanmean(x_band)
        #index = np.argwhere((y[:, graph_id_index, 0] == node) & (y[:, date_index, :] == date))[:, 0]
        index = np.unique(np.argwhere((y[:, graph_id_index, 0] == node))[:, 0])
        
        X[index, band, :, :, k] = resize_no_dim(x_band, *shape2D[scale])
    
    return X

def process_dept_images(X, x_date, raster_dept, ks, k):
    """Process department images and update X."""
    for band in range(x_date.shape[0]):
        x_band = x_date[band]
        x_band[np.isnan(raster_dept)] = np.nanmean(x_band)
        X[band, :, :, k] = resize_no_dim(x_band, 64, 64)
    
    X[np.isnan(X)] = 0
    return X

def process_dept_y(Y, y_date, raster_dept, ks):
    """Process department labels and update Y."""
    for band in range(y_date.shape[0]):
        for k in range(ks + 1):
            y_band = y_date[band, :, :, k]
            y_band[np.isnan(raster_dept)] = 0
            Y[:, :, band, k] = resize_no_dim(y_band, 64, 64)
    
    Y[np.isnan(Y)] = 0
    return Y

def create_dataset_2D_2(graph, X_np, Y_np, ks, dates,
                        features_name_2D, features, path,
                        use_temporal_as_edges, image_per_node,
                        context):
    """Main function to create the dataset."""
    Xst, Yst, Est = [], [], []

    for id in dates:
        if use_temporal_as_edges is None:
            x, y = construct_time_series(id, X_np, Y_np, ks, len(ids_columns))
        elif use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, id, X_np, Y_np, ks, len(ids_columns))
        else:
            x, y, e = construct_graph_with_time_series(graph, id, X_np, Y_np, ks, len(ids_columns))
            
        if x is None:
            continue

        depts = np.unique(y[:, departement_index].astype(int))
        for dept in depts:

            X, Y, raster_dept, raster_dept_graph, unodes = process_dept_raster(dept, graph, path, y[y[:, departement_index, 0] == dept], features_name_2D, ks, image_per_node, shape2D)
            X, Y = process_time_step(X, Y, dept, graph, ks, id, path, features_name_2D, features, y[y[:, departement_index, 0] == dept], raster_dept, raster_dept_graph, unodes, image_per_node, shape2D)

            if X is None:
                continue

            sub_dir = 'image_per_node' if image_per_node else 'image_per_departement'

            if use_temporal_as_edges is None and image_per_node:
                for i in range(X.shape[0]):
                    if True:
                        save_object(X[i], f'X_{int(id)}_{dept}_{i}.pkl', path / f'2D_database_{graph.scale}_{graph.base}_{graph.graph_method}' / sub_dir / context)
                        Xst.append(f'X_{int(id)}_{dept}_{i}.pkl')
                        Yst.append(Y[i])
                    else:
                        Xst.append(X[i])
                        Yst.append(y[i])
            else:
                if True:
                    save_object(X, f'X_{int(id)}_{dept}.pkl', path / f'2D_database_{graph.scale}_{graph.base}_{graph.graph_method}' / sub_dir / context)
                    Xst.append(f'X_{int(id)}_{dept}.pkl')
                else:
                    Xst.append(X)
                Y[np.isnan(Y)] = 0
                Yst.append(Y)

        if 'e' in locals():
            Est.append(e)

    return Xst, Yst, Est

def create_dataset_2D(graph,
                    df_train,
                    df_val,
                    df_test,
                    path,
                    features_name_2D,
                    features,
                    target_name,
                    image_per_node,
                    use_temporal_as_edges : bool,
                    device,
                    ks : int):
    
    x_train, y_train = df_train[ids_columns].values, df_train[ids_columns + [target_name]].values

    x_val, y_val = df_val[ids_columns].values, df_val[ids_columns + [target_name]].values

    x_test, y_test = df_test[ids_columns].values, df_test[ids_columns + [target_name]].values

    dateTrain = np.sort(np.unique(y_train[np.argwhere(y_train[:, weight_index] > 0), date_index]))
    dateVal = np.sort(np.unique(y_val[np.argwhere(y_val[:, weight_index] > 0), date_index]))
    dateTest = np.sort(np.unique(y_test[np.argwhere(y_test[:, weight_index] > 0), date_index]))

    XsTe = []
    YsTe = []
    EsTe = []

    logger.info(f'Model configuration : image_per_node {image_per_node}, use_temporal_as_edges {use_temporal_as_edges}')

    logger.info('Creating train dataset')
    Xst, Yst, Est = create_dataset_2D_2(graph, x_train, y_train, ks, dateTrain,
                        features_name_2D, features, path, use_temporal_as_edges, image_per_node, context='train') 
    
    logger.info('Creating val dataset')
    # Val
    XsV, YsV, EsV = create_dataset_2D_2(graph, x_val, y_val, ks, dateVal,
                        features_name_2D, features, path, use_temporal_as_edges, image_per_node, context='val') 

    logger.info('Creating Test dataset')
    # Test
    XsTe, YsTe, EsTe = create_dataset_2D_2(graph, x_test, y_test, ks, dateTest,
                    features_name_2D, features, path, use_temporal_as_edges, image_per_node, context='test') 

    
    assert len(Xst) > 0
    assert len(XsV) > 0
    assert len(XsTe) > 0

    print(len(Xst), len(XsV), len(XsTe))

    sub_dir = 'image_per_node' if image_per_node else 'image_per_departement'
    if True:
        train_dataset = ReadGraphDataset_2D(Xst, Yst, Est, len(Xst), device, path / f'2D_database_{graph.scale}_{graph.base}_{graph.graph_method}' / sub_dir / 'train')
        val_dataset = ReadGraphDataset_2D(XsV, YsV, EsV, len(XsV), device, path / f'2D_database_{graph.scale}_{graph.base}_{graph.graph_method}' / sub_dir / 'val')
        test_dataset = ReadGraphDataset_2D(XsTe, YsTe, EsTe, len(XsTe), device, path / f'2D_database_{graph.scale}_{graph.base}_{graph.graph_method}' / sub_dir / 'test')
    else:
        train_dataset = InplaceGraphDataset(Xst, Yst, Est, len(Xst), device)
        val_dataset = InplaceGraphDataset(XsV, YsV, EsV, len(XsV), device)
        test_datset = InplaceGraphDataset(XsTe, YsTe, EsTe, len(XsTe), device)
    return train_dataset, val_dataset, test_dataset

class ModelTorch():
    def __init__(self, model_name, batch_size, lr, target_name, task_type,
                 features_name, ks, out_channels, dir_output, loss='mse', name='ModelTorch', device='cpu', non_fire_number='full'):
        self.model_name = model_name
        self.name = name
        self.loss = loss
        self.device = device
        self.model = None
        self.optimizer = None
        self.batch_size = batch_size
        self.target_name = target_name
        self.features_name = features_name
        self.ks = ks
        self.lr = lr
        self.out_channels = out_channels
        self.dir_output = dir_output
        self.task_type = task_type
        self.model_params = None
        self.non_fire_number = non_fire_number

    def compute_weights_and_target(self, labels, band, ids_columns, is_grap_or_node, graphs):
        weight_idx = ids_columns.index('weight')
        target_is_binary = self.target_name == 'binary'

        if len(labels.shape) == 3:
            weights = labels[:, weight_idx, -1]
            target = (labels[:, band, -1] > 0).long() if target_is_binary else labels[:, band, -1]

        elif len(labels.shape) == 5:
            weights = labels[:, :, :, weight_idx, -1]
            target = (labels[:, :, :, band, -1] > 0).long() if target_is_binary else labels[:, :, :, band, -1]

        elif len(labels.shape) == 4:
            weights = labels[:, :, :, weight_idx,]
            target = (labels[:, :, :, band] > 0).long() if target_is_binary else labels[:, :, :, band]

        else:
            weights = labels[:, weight_idx]
            target = (labels[:, band] > 0).long() if target_is_binary else labels[:, band]
        
        if is_grap_or_node:
            unique_elements = torch.unique(graphs, return_inverse=False, return_counts=False, sorted=True)
            first_indices = torch.tensor([torch.nonzero(graphs == u, as_tuple=True)[0][0] for u in unique_elements])
            weights = weights[first_indices]
            target = target[first_indices]

        return target, weights

    def compute_labels(self, labels, is_grap_or_node, graphs):
        if len(labels.shape) == 3:
            labels = labels[:, :, -1]
        elif len(labels.shape) == 5:
            labels = labels[:, :, :, :, -1]
        elif len(labels.shape) == 4:
            labels = labels
        else:
            labels = labels
        
        if is_grap_or_node:
            unique_elements = torch.unique(graphs, return_inverse=False, return_counts=False, sorted=True)
            first_indices = torch.tensor([torch.nonzero(graphs == u, as_tuple=True)[0][0] for u in unique_elements])
            labels = labels[first_indices]

        return labels

    def create_loader(self):
        pass

    def load_loader(self):
        pass

    def create_train_val_test_loader(self, df_train, df_val, df_test):
        pass

    def fit(self,):
        pass

    def predict(self, X):
        pass

    def _compute_loss(self, outputs, targets):
        pass

    def score(self, X, y, sample_weight=None):
        pass

    def plot_train_val_loss(self, epochs, train_loss_list, val_loss_list, dir_output):
        # Création de la figure et des axes
        plt.figure(figsize=(10, 6))

        # Tracé de la courbe de val_loss
        plt.plot(epochs, val_loss_list, label='Validation Loss', color='blue')

        # Ajout de la légende
        plt.legend()

        # Ajout des labels des axes
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        # Ajout d'un titre
        plt.title('Validation Loss over Epochs')
        plt.savefig(dir_output / 'Validation.png')
        plt.close('all')

        # Tracé de la courbe de train_loss
        plt.plot(epochs, train_loss_list, label='Training Loss', color='red')

        # Ajout de la légende
        plt.legend()

        # Ajout des labels des axes
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        # Ajout d'un titre
        plt.title('Training Loss over Epochs')
        plt.savefig(dir_output / 'Training.png')
        plt.close('all')

    def _load_model_from_path(self, path : Path, model : torch.nn.Module) -> None:
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True), strict=False)
        self.model = model

class ModelCNN(ModelTorch):
    def __init__(self, model_name, batch_size, lr, target_name, task_type, out_channels, dir_output, features_name, features, ks, loss, name, device, non_fire_number, path, image_per_node):
        super().__init__(model_name, batch_size, lr, target_name, task_type, features_name, ks, out_channels, dir_output, loss=loss, name=name, device=device, non_fire_number=non_fire_number)
        self.path = path
        self.features = features
        self.image_per_node = image_per_node

    def create_train_val_test_loader(self, graph, df_train, df_val, df_test):

        if self.non_fire_number != 'full':
            old_shape = df_train.shape
            if 'binary' in self.non_fire_number:
                vec = self.non_fire_number.split('-')
                try:
                    nb = int(vec[-1]) * len(df_train[df_train[self.target_name] > 0])
                except ValueError:
                    print(f'{self.non_fire_number} with undefined factor, set to 1 -> {len(df_train[df_train[self.target_name] > 0])}')
                    nb = len(df_train[df_train[self.target_name] > 0])

                # Separate the positive and zero classes based on y
                positive_mask = df_train[self.target_name] > 0
                non_fire_mask = df_train[self.target_name] == 0

                # Filtrer les données positives et non feu
                df_positive = df_train[positive_mask]
                df_non_fire = df_train[non_fire_mask]

                # Échantillonner les données non feu
                nb = min(len(df_non_fire), nb)
                sampled_indices = np.random.RandomState(42).choice(len(df_non_fire), nb, replace=False)
                df_non_fire_sampled = df_non_fire.iloc[sampled_indices]

                # Combiner les données positives et non feu échantillonnées
                df_combined = pd.concat([df_positive, df_non_fire_sampled])

                # Réinitialiser les index du DataFrame combiné
                df_combined.reset_index(drop=True, inplace=True)

                # Mettre à jour df_train pour l'entraînement
                df_train = df_combined

                print(f'Train mask df_train shape: {old_shape} -> {df_train.shape}')
        
        #if (self.dir_output / 'train_loader.pkl').is_file():
        if False:
            self.train_loader = read_object('train_loader.pkl', self.dir_output)
            self.val_loader = read_object('val_loader.pkl', self.dir_output)
            self.test_loader = read_object('test_loader.pkl', self.dir_output)
        else:
            train_dataset, val_dataset, test_dataset = create_dataset_2D(graph=graph,
                                                                df_train=df_train,
                                                                df_val=df_val,
                                                                df_test=df_test,
                                                                features_name_2D=self.features_name,
                                                                features=self.features,
                                                                target_name=self.target_name,
                                                                use_temporal_as_edges=None,
                                                                image_per_node=self.image_per_node,
                                                                device=self.device, ks=self.ks,
                                                                path=self.path)
        
            train_loader = DataLoader(train_dataset, batch_size, True,)
            val_loader = DataLoader(val_dataset, val_dataset.__len__(), False)
            test_loader = DataLoader(test_dataset, test_dataset.__len__(), False)

            save_object_torch(train_loader, 'train_loader.pkl', self.dir_output)
            save_object_torch(val_loader, 'val_loader.pkl', self.dir_output)
            save_object_torch(test_loader, 'test_loader.pkl', self.dir_output)

            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader

    def create_test_loader(self, graph, df):
        
        x_test, y_test = df[ids_columns].values, df[ids_columns + [self.target_name]].values

        dateTest = np.sort(np.unique(y_test[np.argwhere(y_test[:, weight_index] > 0), date_index]))
        
        XsTe, YsTe, EsTe = create_dataset_2D_2(graph, x_test, y_test, self.ks, dateTest,
                    self.features_name, self.features, self.path, None, self.image_per_node, context='test')
        
        sub_dir = 'image_per_node' if self.image_per_node else 'image_per_departement'
        test_dataset = ReadGraphDataset_2D(XsTe, YsTe, EsTe, len(XsTe), device, self.path / f'2D_database_{graph.scale}_{graph.base}_{graph.graph_method}' / sub_dir / 'test')
        
        loader = DataLoader(test_dataset, test_dataset.__len__(), False)

        return loader

    def launch_train_loader(self, loader, criterion, optimizer):
        
        self.model.train()
        for i, data in enumerate(loader, 0):

            inputs, labels, edges = data
            graphs = None

            band = -1

            try:
                target, weights = self.compute_weights_and_target(labels, band, ids_columns, self.model.is_graph_or_node, graphs)
            except Exception as e:
                target, weights = self.compute_weights_and_target(labels, band, ids_columns, False, graphs)

            #inputs_model = inputs[:, features]
            inputs_model = inputs
            output = self.model(inputs_model, edges)

            if self.task_type == 'regression':
                target = target.view(output.shape)
                weights = weights.view(output.shape)
                
                target = torch.masked_select(target, weights.gt(0))
                output = torch.masked_select(output, weights.gt(0))
                weights = torch.masked_select(weights, weights.gt(0))
                loss = criterion(output, target, weights)
            else:
                target = torch.masked_select(target, weights.gt(0))
                target = target.long()
                output = output[weights.gt(0)]
                weights = torch.masked_select(weights, weights.gt(0))
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss

    def launch_val_test_loader(self, loader, criterion):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():

            for i, data in enumerate(loader, 0):
                
                inputs, labels, edges = data
                graphs = None

                # Determine the index of the target variable in labels
                #if target_name == 'binary' or target_name == 'nbsinister':
                #    band = -2
                #else:
                band = -1

                try:
                    target, weights = self.compute_weights_and_target(labels, band, ids_columns, self.model.is_graph_or_node, graphs)
                except Exception as e:
                    target, weights = self.compute_weights_and_target(labels, band, ids_columns, False, graphs)

                inputs_model = inputs
                #inputs_model = inputs[:, features]
                output = self.model(inputs_model, edges)

                # Compute loss
                if self.task_type == 'regression':
                    target = target.view(output.shape)
                    weights = weights.view(output.shape)

                    # Mask out invalid weights
                    valid_mask = weights.gt(0)
                    target = torch.masked_select(target, valid_mask)
                    output = torch.masked_select(output, valid_mask)
                    weights = torch.masked_select(weights, valid_mask)
                    loss = criterion(output, target, weights)
                else:
                    valid_mask = weights.gt(0)
                    target = torch.masked_select(target, valid_mask)
                    target = target.long()
                    output = output[valid_mask]
                    weights = torch.masked_select(weights, valid_mask)
                    loss = criterion(output, target)

                total_loss += loss.item()

        return total_loss

    def func_epoch(self, train_loader, val_loader, optimizer, criterion):
        
        train_loss = self.launch_train_loader(train_loader, criterion, optimizer)

        if val_loader is not None:
            val_loss = self.launch_val_test_loader(val_loader, criterion)
        
        else:
            val_loss = train_loss

        return val_loss, train_loss
    
    def make_model(self, graph, custom_model_params):
        model, params = make_model(self.model_name, len(self.features_name), len(self.features_name),
                                graph, dropout, 'relu',
                                self.ks,
                                out_channels=self.out_channels,
                                task_type=self.task_type,
                                device=device, num_lstm_layers=num_lstm_layers,
                                custom_model_params=custom_model_params)
        if self.model_params is None:
            self.model_params = params
        return model, params

    def train(self, graph, PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params=None):
        """
        Train neural network model
        """

        if MLFLOW:
            existing_run = get_existing_run(f'{self.model_name}_')
            if existing_run:
                mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
            else:
                mlflow.start_run(run_name=f'{self.model_name}_', nested=True)

        assert self.train_loader is not None and self.val_loader is not None

        check_and_create_path(self.dir_output)

        criterion = get_loss_function(self.loss)

        self.model, _ = self.make_model(graph, custom_model_params)
        
        #if (dir_output / '100.pt').is_file():
        #    model.load_state_dict(torch.load((dir_output / '100.pt'), map_location=device, weights_only=True), strict=False)
            
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        BEST_VAL_LOSS = math.inf
        BEST_MODEL_PARAMS = None
        patience_cnt = 0

        val_loss_list = []
        train_loss_list = []
        epochs_list = []

        logger.info('Train model with')
        for epoch in tqdm(range(epochs)):
            val_loss, train_loss = self.func_epoch(self.train_loader, self.val_loader, optimizer, criterion)
            train_loss = train_loss.item()
            val_loss = round(val_loss, 3)
            train_loss = round(train_loss, 3)
            val_loss_list.append(val_loss)
            train_loss_list.append(train_loss)
            epochs_list.append(epoch)
            if val_loss < BEST_VAL_LOSS:
                BEST_VAL_LOSS = val_loss
                BEST_MODEL_PARAMS = self.model.state_dict()
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE_CNT:
                    logger.info(f'Loss has not increased for {patience_cnt} epochs. Last best val loss {BEST_VAL_LOSS}, current val loss {val_loss}')
                    save_object_torch(self.model.state_dict(), 'last.pt', self.dir_output)
                    save_object_torch(BEST_MODEL_PARAMS, 'best.pt', self.dir_output)
                    plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, self.dir_output)
                    if MLFLOW:
                        mlflow.end_run()
                    return
            if MLFLOW:
                mlflow.log_metric('loss', val_loss, step=epoch)
            if epoch % CHECKPOINT == 0:
                logger.info(f'epochs {epoch}, Val loss {val_loss}')
                logger.info(f'epochs {epoch}, Best val loss {BEST_VAL_LOSS}')
                save_object_torch(self.model.state_dict(), str(epoch)+'.pt', self.dir_output)

        logger.info(f'Last val loss {val_loss}')
        save_object_torch(self.model.state_dict(), 'last.pt', self.dir_output)
        save_object_torch(BEST_MODEL_PARAMS, 'best.pt', self.dir_output)
        self.plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, self.dir_output)

    def _predict_test_loader(self, X: DataLoader) -> torch.tensor:
            """
            Generates predictions using the model on the provided DataLoader, with optional autoregression.

            Note:
            - 'node_id' and 'date_id' are not included in features_name but can be found in labels[:, 0] and labels[:, 4].
            Parameters:
            - X_: DataLoader providing the test data.
            - features: Numpy array of feature indices to use.
            - device: Torch device to use for computations.
            - target_name: Name of the target variable.
            - autoRegression: Boolean indicating whether to use autoregression.
            - features_name: List mapping feature names to indices.
            - dataset: Pandas DataFrame containing 'node_id', 'date_id', and 'nbsinister' columns.

            Returns:
            - pred: Tensor containing the model's predictions.
            - y: Tensor containing the true labels.
            """
            assert self.model is not None
            self.model.eval()

            with torch.no_grad():
                pred = []
                y = []

                for i, data in enumerate(X, 0):
                    
                    inputs, orilabels, _ = data

                    orilabels = orilabels.to(device)
                    orilabels = orilabels[:, :, -1]

                    #labels = compute_labels(orilabels, self.model.is_graph_or_node, graphs)

                    inputs_model = inputs
                    output = self.model(inputs_model, None)

                    if output.shape[1] > 1:
                        output = torch.argmax(output, dim=1)

                    #output = output[weights.gt(0)]

                    pred.append(output)
                    y.append(orilabels)

                y = torch.cat(y, 0)
                pred = torch.cat(pred, 0)

                if self.target_name == 'binary' or self.target_name == 'risk':
                    pred = torch.round(pred, decimals=1)
                elif self.target_name == 'nbsinister':
                    pred = torch.round(pred, decimals=1)
                
                return pred, y
            
class ModelGNN(ModelTorch):
    def __init__(self, model_name, batch_size, lr, target_name, task_type, out_channels, dir_output, features_name, ks, loss, name, device, non_fire_number):
        super().__init__(model_name, batch_size, lr, target_name, task_type, features_name, ks, out_channels, dir_output, loss=loss, name=name, device=device, non_fire_number=non_fire_number)

    def create_train_val_test_loader(self, graph, df_train, df_val, df_test):

        if self.non_fire_number != 'full':
            old_shape = df_train.shape
            if 'binary' in self.non_fire_number:
                vec = self.non_fire_number.split('-')
                try:
                    nb = int(vec[-1]) * len(y[y > 0])
                except ValueError:
                    print(f'{self.non_fire_number} with undefined factor, set to 1 -> {len(df_train[df_train[self.target_name] > 0])}')
                    nb = len(df_train[df_train[self.target_name] > 0])

                # Separate the positive and zero classes based on y
                positive_mask = df_train[self.target_name] > 0
                non_fire_mask = df_train[self.target_name] == 0

                # Filtrer les données positives et non feu
                df_positive = df_train[positive_mask]
                df_non_fire = df_train[non_fire_mask]

                # Échantillonner les données non feu
                nb = min(len(df_non_fire), nb)
                sampled_indices = np.random.RandomState(42).choice(len(df_non_fire), nb, replace=False)
                df_non_fire_sampled = df_non_fire.iloc[sampled_indices]

                # Combiner les données positives et non feu échantillonnées
                df_combined = pd.concat([df_positive, df_non_fire_sampled])

                # Mettre à jour df_train pour l'entraînement
                df_train.loc[df_combined.index, 'weight'] = 0

                print(f'Train mask df_train shape: {old_shape} -> {df_train[df_train["weight"] > 0].shape}')
        
        #if (self.dir_output / 'train_loader.pkl').is_file():
        if False:
            self.train_loader = read_object('train_loader.pkl', self.dir_output)
            self.val_loader = read_object('val_loader.pkl', self.dir_output)
            self.test_loader = read_object('test_loader.pkl', self.dir_output)
        else:
            train_dataset, val_dataset, test_dataset = create_dataset(graph,
                                                                df_train,
                                                                df_val,
                                                                df_test,
                                                                self.features_name,
                                                                self.target_name,
                                                                False,
                                                                self.device, self.ks)
        
            self.train_loader = DataLoader(train_dataset, batch_size, True,)
            self.val_loader = DataLoader(val_dataset, val_dataset.__len__(), False)
            self.test_loader = DataLoader(test_dataset, test_dataset.__len__(), False)

        save_object_torch(self.train_loader, 'train_loader.pkl', self.dir_output)
        save_object_torch(self.val_loader, 'val_loader.pkl', self.dir_output)
        save_object_torch(self.test_loader, 'test_loader.pkl', self.dir_output)

    def create_test_loader(self, graph, df):
        loader = create_test_loader(graph, df,
                       self.features_name,
                       self.device,
                       False,
                       self.target_name,
                       self.ks)

        return loader

    def launch_train_loader(self, loader, criterion, optimizer):
        
        self.model.train()
        for i, data in enumerate(loader, 0):

            inputs, labels, edges = data
            graphs = None

            band = -1

            try:
                target, weights = self.compute_weights_and_target(labels, band, ids_columns, self.model.is_graph_or_node, graphs)
            except Exception as e:
                target, weights = self.compute_weights_and_target(labels, band, ids_columns, False, graphs)

            #inputs_model = inputs[:, features]
            inputs_model = inputs
            output = self.model(inputs_model, edges)

            if self.task_type == 'regression':
                target = target.view(output.shape)
                weights = weights.view(output.shape)
                
                target = torch.masked_select(target, weights.gt(0))
                output = torch.masked_select(output, weights.gt(0))
                weights = torch.masked_select(weights, weights.gt(0))
                loss = criterion(output, target, weights)
            else:
                target = torch.masked_select(target, weights.gt(0))
                target = target.long()
                output = output[weights.gt(0)]
                weights = torch.masked_select(weights, weights.gt(0))
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss

    def launch_val_test_loader(self, loader, criterion):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():

            for i, data in enumerate(loader, 0):
                
                inputs, labels, edges = data
                graphs = None

                # Determine the index of the target variable in labels
                #if target_name == 'binary' or target_name == 'nbsinister':
                #    band = -2
                #else:
                band = -1

                try:
                    target, weights = self.compute_weights_and_target(labels, band, ids_columns, self.model.is_graph_or_node, graphs)
                except Exception as e:
                    target, weights = self.compute_weights_and_target(labels, band, ids_columns, False, graphs)

                inputs_model = inputs
                #inputs_model = inputs[:, features]
                output = self.model(inputs_model, edges)

                # Compute loss
                if self.task_type == 'regression':
                    target = target.view(output.shape)
                    weights = weights.view(output.shape)

                    # Mask out invalid weights
                    valid_mask = weights.gt(0)
                    target = torch.masked_select(target, valid_mask)
                    output = torch.masked_select(output, valid_mask)
                    weights = torch.masked_select(weights, valid_mask)
                    loss = criterion(output, target, weights)
                else:
                    valid_mask = weights.gt(0)
                    target = torch.masked_select(target, valid_mask)
                    output = output[valid_mask]
                    target = target.long()
                    weights = torch.masked_select(weights, valid_mask)
                    loss = criterion(output, target)

                total_loss += loss.item()

        return total_loss

    def func_epoch(self, train_loader, val_loader, optimizer, criterion):
        
        train_loss = self.launch_train_loader(train_loader, criterion, optimizer)

        if val_loader is not None:
            val_loss = self.launch_val_test_loader(val_loader, criterion)
        
        else:
            val_loss = train_loss

        return val_loss, train_loss
    
    def make_model(self, graph, custom_model_params):
        model, params = make_model(self.model_name, len(self.features_name), len(self.features_name),
                                graph, dropout, 'relu',
                                self.ks,
                                out_channels=self.out_channels,
                                task_type=self.task_type,
                                device=device, num_lstm_layers=num_lstm_layers,
                                custom_model_params=custom_model_params)
        if self.model_params is None:
            self.model_params = params
        return model, params

    def train(self, graph, PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params=None):
        """
        Train neural network model
        """

        if MLFLOW:
            existing_run = get_existing_run(f'{self.model_name}_')
            if existing_run:
                mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
            else:
                mlflow.start_run(run_name=f'{self.model_name}_', nested=True)

        assert self.train_loader is not None and self.val_loader is not None

        check_and_create_path(self.dir_output)

        criterion = get_loss_function(self.loss)

        self.model, _ = self.make_model(graph, custom_model_params)
        
        #if (dir_output / '100.pt').is_file():
        #    model.load_state_dict(torch.load((dir_output / '100.pt'), map_location=device, weights_only=True), strict=False)
            
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        BEST_VAL_LOSS = math.inf
        BEST_MODEL_PARAMS = None
        patience_cnt = 0

        val_loss_list = []
        train_loss_list = []
        epochs_list = []

        logger.info('Train model with')
        for epoch in tqdm(range(epochs)):
            val_loss, train_loss = self.func_epoch(self.train_loader, self.val_loader, optimizer, criterion)
            train_loss = train_loss.item()
            val_loss = round(val_loss, 3)
            train_loss = round(train_loss, 3)
            val_loss_list.append(val_loss)
            train_loss_list.append(train_loss)
            epochs_list.append(epoch)
            if val_loss < BEST_VAL_LOSS:
                BEST_VAL_LOSS = val_loss
                BEST_MODEL_PARAMS = self.model.state_dict()
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE_CNT:
                    logger.info(f'Loss has not increased for {patience_cnt} epochs. Last best val loss {BEST_VAL_LOSS}, current val loss {val_loss}')
                    save_object_torch(self.model.state_dict(), 'last.pt', self.dir_output)
                    save_object_torch(BEST_MODEL_PARAMS, 'best.pt', self.dir_output)
                    plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, self.dir_output)
                    if MLFLOW:
                        mlflow.end_run()
                    return
            if MLFLOW:
                mlflow.log_metric('loss', val_loss, step=epoch)
            if epoch % CHECKPOINT == 0:
                logger.info(f'epochs {epoch}, Val loss {val_loss}')
                logger.info(f'epochs {epoch}, Best val loss {BEST_VAL_LOSS}')
                save_object_torch(self.model.state_dict(), str(epoch)+'.pt', self.dir_output)

        logger.info(f'Last val loss {val_loss}')
        save_object_torch(self.model.state_dict(), 'last.pt', self.dir_output)
        save_object_torch(BEST_MODEL_PARAMS, 'best.pt', self.dir_output)
        self.plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, self.dir_output)

    def _predict_test_loader(self, X: DataLoader) -> torch.tensor:
            """
            Generates predictions using the model on the provided DataLoader, with optional autoregression.

            Note:
            - 'node_id' and 'date_id' are not included in features_name but can be found in labels[:, 0] and labels[:, 4].
            Parameters:
            - X_: DataLoader providing the test data.
            - features: Numpy array of feature indices to use.
            - device: Torch device to use for computations.
            - target_name: Name of the target variable.
            - autoRegression: Boolean indicating whether to use autoregression.
            - features_name: List mapping feature names to indices.
            - dataset: Pandas DataFrame containing 'node_id', 'date_id', and 'nbsinister' columns.

            Returns:
            - pred: Tensor containing the model's predictions.
            - y: Tensor containing the true labels.
            """
            assert self.model is not None
            self.model.eval()

            with torch.no_grad():
                pred = []
                y = []

                for i, data in enumerate(X, 0):
                    
                    inputs, orilabels, edges, graphs = data

                    orilabels = orilabels.to(device)
                    orilabels = orilabels[:, :, -1]

                    #labels = compute_labels(orilabels, self.model.is_graph_or_node, graphs)

                    inputs_model = inputs
                    output = self.model(inputs_model, edges, graphs)

                    if output.shape[1] > 1:
                        output = torch.argmax(output, dim=1)

                    #output = output[weights.gt(0)]

                    pred.append(output)
                    y.append(orilabels)

                y = torch.cat(y, 0)
                pred = torch.cat(pred, 0)

                if self.target_name == 'binary' or self.target_name == 'risk':
                    pred = torch.round(pred, decimals=1)
                elif self.target_name == 'nbsinister':
                    pred = torch.round(pred, decimals=1)
                
                return pred, y

class Model_Torch(ModelTorch):
    def __init__(self, model_name, batch_size, lr, target_name, task_type, out_channels, dir_output, features_name, ks, loss, name, device, non_fire_number):
        super().__init__(model_name, batch_size, lr, target_name, task_type, features_name, ks, out_channels, dir_output, loss=loss, name=name, device=device, non_fire_number=non_fire_number)

    def create_train_val_test_loader(self, graph, df_train, df_val, df_test):

        if self.non_fire_number != 'full':
            old_shape = df_train.shape
            if 'binary' in self.non_fire_number:
                vec = self.non_fire_number.split('-')
                try:
                    nb = int(vec[-1]) * len(df_train[df_train[self.target_name] > 0])
                except:
                    print(f'{self.non_fire_number} with undefined factor, set to 1 -> {len(df_train[df_train[self.target_name] > 0])}')
                    nb = len(df_train[df_train[self.target_name] > 0])

                # Separate the positive and zero classes based on y
                positive_mask = df_train[self.target_name] > 0
                non_fire_mask = df_train[self.target_name] == 0

                # Filtrer les données positives et non feu
                df_positive = df_train[positive_mask]
                df_non_fire = df_train[non_fire_mask]

                # Échantillonner les données non feu
                nb = min(len(df_non_fire), nb)
                sampled_indices = np.random.RandomState(42).choice(len(df_non_fire), nb, replace=False)
                df_non_fire_sampled = df_non_fire.iloc[sampled_indices]

                # Combiner les données positives et non feu échantillonnées
                df_combined = pd.concat([df_positive, df_non_fire_sampled])

                # Réinitialiser les index du DataFrame combiné
                df_combined.reset_index(drop=True, inplace=True)

                # Mettre à jour df_train pour l'entraînement
                df_train = df_combined

                print(f'Train mask df_train shape: {old_shape} -> {df_train.shape}')
        
        #if (self.dir_output / 'train_loader.pkl').is_file():
        if False:
            self.train_loader = read_object('train_loader.pkl', self.dir_output)
            self.val_loader = read_object('val_loader.pkl', self.dir_output)
            self.test_loader = read_object('test_loader.pkl', self.dir_output)
        else:
            train_dataset, val_dataset, test_dataset = create_dataset(graph,
                                                                df_train,
                                                                df_val,
                                                                df_test,
                                                                self.features_name,
                                                                self.target_name,
                                                                None,
                                                                self.device, self.ks)
        
            self.train_loader = DataLoader(train_dataset, batch_size, True,)
            self.val_loader = DataLoader(val_dataset, val_dataset.__len__(), False)
            self.test_loader = DataLoader(test_dataset, test_dataset.__len__(), False)

        save_object_torch(self.train_loader, 'train_loader.pkl', self.dir_output)
        save_object_torch(self.val_loader, 'val_loader.pkl', self.dir_output)
        save_object_torch(self.test_loader, 'test_loader.pkl', self.dir_output)

    def create_test_loader(self, graph, df):
        loader = create_test_loader(graph, df,
                       self.features_name,
                       self.device,
                       None,
                       self.target_name,
                       self.ks)

        return loader

    def launch_train_loader(self, loader, criterion, optimizer):
        
        self.model.train()
        for i, data in enumerate(loader, 0):

            inputs, labels, _ = data
            graphs = None

            band = -1

            try:
                target, weights = self.compute_weights_and_target(labels, band, ids_columns, self.model.is_graph_or_node, graphs)
            except Exception as e:
                target, weights = self.compute_weights_and_target(labels, band, ids_columns, False, graphs)

            #inputs_model = inputs[:, features]
            inputs_model = inputs
            output = self.model(inputs_model)

            if self.task_type == 'regression':
                target = target.view(output.shape)
                weights = weights.view(output.shape)
                
                target = torch.masked_select(target, weights.gt(0))
                output = torch.masked_select(output, weights.gt(0))
                weights = torch.masked_select(weights, weights.gt(0))
                loss = criterion(output, target, weights)
            else:
                target = torch.masked_select(target, weights.gt(0))
                output = output[weights.gt(0)]
                target = target.long()
                weights = torch.masked_select(weights, weights.gt(0))
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss

    def launch_val_test_loader(self, loader, criterion):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():

            for i, data in enumerate(loader, 0):
                
                inputs, labels, _ = data
                graphs = None

                # Determine the index of the target variable in labels
                #if target_name == 'binary' or target_name == 'nbsinister':
                #    band = -2
                #else:
                band = -1

                try:
                    target, weights = self.compute_weights_and_target(labels, band, ids_columns, self.model.is_graph_or_node, graphs)
                except Exception as e:
                    target, weights = self.compute_weights_and_target(labels, band, ids_columns, False, graphs)

                inputs_model = inputs
                #inputs_model = inputs[:, features]
                output = self.model(inputs_model)

                # Compute loss
                if self.task_type == 'regression':
                    target = target.view(output.shape)
                    weights = weights.view(output.shape)

                    # Mask out invalid weights
                    valid_mask = weights.gt(0)
                    target = torch.masked_select(target, valid_mask)
                    output = torch.masked_select(output, valid_mask)
                    weights = torch.masked_select(weights, valid_mask)
                    loss = criterion(output, target, weights)
                else:
                    valid_mask = weights.gt(0)
                    target = torch.masked_select(target, valid_mask)
                    target = target.long()
                    output = output[valid_mask]
                    weights = torch.masked_select(weights, valid_mask)
                    loss = criterion(output, target)

                total_loss += loss.item()

        return total_loss

    def func_epoch(self, train_loader, val_loader, optimizer, criterion):
        
        train_loss = self.launch_train_loader(train_loader, criterion, optimizer)

        if val_loader is not None:
            val_loss = self.launch_val_test_loader(val_loader, criterion)
        
        else:
            val_loss = train_loss

        return val_loss, train_loss
    
    def make_model(self, graph, custom_model_params):
        model, params = make_model(self.model_name, len(self.features_name), len(self.features_name),
                                graph, dropout, 'relu',
                                self.ks,
                                out_channels=self.out_channels,
                                task_type=self.task_type,
                                device=device, num_lstm_layers=num_lstm_layers,
                                custom_model_params=custom_model_params)
        if self.model_params is None:
            self.model_params = params
        return model, params

    def train(self, graph, PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params=None):
        """
        Train neural network model
        """

        if MLFLOW:
            existing_run = get_existing_run(f'{self.model_name}_')
            if existing_run:
                mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
            else:
                mlflow.start_run(run_name=f'{self.model_name}_', nested=True)

        assert self.train_loader is not None and self.val_loader is not None

        check_and_create_path(self.dir_output)

        criterion = get_loss_function(self.loss)

        self.model, _ = self.make_model(graph, custom_model_params)
        
        #if (dir_output / '100.pt').is_file():
        #    model.load_state_dict(torch.load((dir_output / '100.pt'), map_location=device, weights_only=True), strict=False)
            
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        BEST_VAL_LOSS = math.inf
        BEST_MODEL_PARAMS = None
        patience_cnt = 0

        val_loss_list = []
        train_loss_list = []
        epochs_list = []

        logger.info('Train model with')
        for epoch in tqdm(range(epochs)):
            val_loss, train_loss = self.func_epoch(self.train_loader, self.val_loader, optimizer, criterion)
            train_loss = train_loss.item()
            val_loss = round(val_loss, 3)
            train_loss = round(train_loss, 3)
            val_loss_list.append(val_loss)
            train_loss_list.append(train_loss)
            epochs_list.append(epoch)
            if val_loss < BEST_VAL_LOSS:
                BEST_VAL_LOSS = val_loss
                BEST_MODEL_PARAMS = self.model.state_dict()
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE_CNT:
                    logger.info(f'Loss has not increased for {patience_cnt} epochs. Last best val loss {BEST_VAL_LOSS}, current val loss {val_loss}')
                    save_object_torch(self.model.state_dict(), 'last.pt', self.dir_output)
                    save_object_torch(BEST_MODEL_PARAMS, 'best.pt', self.dir_output)
                    plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, self.dir_output)
                    if MLFLOW:
                        mlflow.end_run()
                    return
            if MLFLOW:
                mlflow.log_metric('loss', val_loss, step=epoch)
            if epoch % CHECKPOINT == 0:
                logger.info(f'epochs {epoch}, Val loss {val_loss}')
                logger.info(f'epochs {epoch}, Best val loss {BEST_VAL_LOSS}')
                save_object_torch(self.model.state_dict(), str(epoch)+'.pt', self.dir_output)

        logger.info(f'Last val loss {val_loss}')
        save_object_torch(self.model.state_dict(), 'last.pt', self.dir_output)
        save_object_torch(BEST_MODEL_PARAMS, 'best.pt', self.dir_output)
        self.plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, self.dir_output)

    def _predict_test_loader(self, X: DataLoader) -> torch.tensor:
            """
            Generates predictions using the model on the provided DataLoader, with optional autoregression.

            Note:
            - 'node_id' and 'date_id' are not included in features_name but can be found in labels[:, 0] and labels[:, 4].
            Parameters:
            - X_: DataLoader providing the test data.
            - features: Numpy array of feature indices to use.
            - device: Torch device to use for computations.
            - target_name: Name of the target variable.
            - autoRegression: Boolean indicating whether to use autoregression.
            - features_name: List mapping feature names to indices.
            - dataset: Pandas DataFrame containing 'node_id', 'date_id', and 'nbsinister' columns.

            Returns:
            - pred: Tensor containing the model's predictions.
            - y: Tensor containing the true labels.
            """
            assert self.model is not None
            self.model.eval()

            with torch.no_grad():
                pred = []
                y = []

                for i, data in enumerate(X, 0):
                    
                    inputs, orilabels, _ = data

                    orilabels = orilabels.to(device)
                    orilabels = orilabels[:, :, -1]

                    inputs_model = inputs
                    output = self.model(inputs_model)

                    if output.shape[1] > 1:
                        output = torch.argmax(output, dim=1)

                    pred.append(output)
                    y.append(orilabels)

                y = torch.cat(y, 0)
                pred = torch.cat(pred, 0)

                if self.target_name == 'binary' or self.target_name == 'risk':
                    pred = torch.round(pred, decimals=1)
                elif self.target_name == 'nbsinister':
                    pred = torch.round(pred, decimals=1)
                
                return pred, y

"""class ModelHybrid(ModelTorch):
    def __init__(self, model, loss='mse', name='ModelGNN', device='cpu'):
        super().__init__(model, loss=loss, name=name, device=device)"""