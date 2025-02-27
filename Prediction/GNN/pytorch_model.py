from numpy import dtype
import torch_geometric
from torch_geometric.data import Dataset
from torch.utils.data import DataLoader
import torch
torch.set_printoptions(precision=3, sci_mode=False)

from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
from feature_engine.selection import SmartCorrelatedSelection

from GNN.discretization import *

from dgl import DGLGraph

from graph_builder import * 

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

class InplaceMeshGraphDataset(Dataset):
    def __init__(self, icospheres_graph_path : str, X : list, Y : list, edges : list, leni : int, device : torch.device) -> None:
        self.X = X
        self.Y = Y
        self.device = device
        self.edges = edges
        self.leni = leni
        self.icospheres_graph_path = icospheres_graph_path

    def __getitem__(self, index) -> tuple:
        x = self.X[index]
        y = self.Y[index]

        if len(self.edges) > 0:
            edges = self.edges[index]
        else:
            edges = []
        
        X, Y, E = torch.tensor(x, dtype=torch.float32, device=self.device), \
            torch.tensor(y, dtype=torch.float32, device=self.device), \
            torch.tensor(edges, dtype=torch.long, device=self.device),  \
            
        return X, Y, E, self.icospheres_graph_path

    def __len__(self) -> int:
        return self.leni
    
    def len(self):
        pass

    def get(self):
        pass

# Créez une classe de Dataset qui applique les transformations
class AugmentedInplaceGraphDataset(Dataset):
    def __init__(self, X, y, edges, transform=None, device=torch.device('cpu')):
        self.X = X
        self.y = y
        self.transform = transform
        self.device = device
        self.edges = edges

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        # Appliquez la transformation si elle est définie
        if self.transform:
            x, y = self.transform(x, y)

        if len(self.edges) != 0:
            edges = self.edges[idx]
        else:
            edges = []

        return torch.tensor(x, dtype=torch.float32, device=self.device), \
            torch.tensor(y, dtype=torch.float32, device=self.device), \
            torch.tensor(edges, dtype=torch.long, device=self.device)
    
    def len(self):
        pass

    def get(self):
        pass

class RandomFlipRotateAndCrop:
    def __init__(self, proba_flip, size_crop, max_angle):

        self.proba_flip = proba_flip
        self.size_crop = size_crop
        self.max_angle = max_angle

    def __call__(self, image, mask):
        # Random rotation angle
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)

        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, dtype=torch.float32)

        angle = random.uniform(-self.max_angle, self.max_angle)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

        # Random vertical flip
        if random.random() < self.proba_flip:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random horizontal flip
        if random.random() < self.proba_flip:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        return image, mask

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
    graph = DGLGraph((edge_index[0], edge_index[1]))
    return node_features, node_labels, graph, graph_labels_list

def graph_collate_fn_mesh(batch):
    #node_indice_list = []
    node_features_list = []
    node_labels_list = []
    graph_labels_list = []

    graph_list = []
    graph_mesh_list = []
    grid2mesh_list = []
    mesh2grid_list = []

    num_nodes_seen = 0

    last_graph_1 = None
    last_graph_2 = None
    last_graph_3 = None

    for graph_id, features_labels_graph_index_tuple in enumerate(batch):
        # Collecter les caractéristiques et les étiquettes des nœuds
        node_features_list.append(features_labels_graph_index_tuple[0])
        node_labels_list.append(features_labels_graph_index_tuple[1])

        icospheres_graph_path = features_labels_graph_index_tuple[3]
        
        """graph_mesh_grid2mesh_mesh2_grid = features_labels_graph_index_tuple[2]

        # Ajuster l'index des arêtes en fonction du nombre de nœuds vus
        if edge_index.shape[0] > 2:
            edge_index[0] += num_nodes_seen
            edge_index[1] += num_nodes_seen
            edge_index_list.append(edge_index)
        else:
            edge_index_list.append(edge_index + num_nodes_seen)
        
        # Ajouter l'ID du graphe pour chaque nœud
        graph_labels_list.append(torch.full((num_nodes,), graph_id, dtype=torch.long))  # Création d'un tensor d'IDs"""

        num_nodes = features_labels_graph_index_tuple[1].size(0)
        num_nodes_seen += num_nodes  # Mettre à jour le nombre de nœuds vus
        graph_labels_list.append(torch.full((num_nodes,), graph_id, dtype=torch.long))  # Création d'un tensor d'IDs

        latitudes = features_labels_graph_index_tuple[1][:, latitude_index]
        longtitudes = features_labels_graph_index_tuple[1][:, longitude_index]

        g_lat_lon_grid = torch.concat((latitudes, longtitudes), dim=1).to('cpu')
        graph_builder = GraphBuilder(icospheres_graph_path, g_lat_lon_grid, doPrint=False)

        graph_mesh = graph_builder.create_mesh_graph(last_graph_1)
        gridh2mesh, graph_mesh = graph_builder.create_g2m_graph(last_graph_2, graph_mesh)
        mesh2graph = graph_builder.create_m2g_graph(last_graph_3)

        #gridh2mesh.ndata['_ID'] = torch.arange(gridh2mesh.num_nodes(), dtype=torch.int32)
        #mesh2graph.ndata['_ID'] = torch.arange(mesh2graph.num_nodes(), dtype=torch.int32)
        
        graph_mesh_list.append(graph_mesh)
        grid2mesh_list.append(gridh2mesh)
        mesh2grid_list.append(mesh2graph)

        #last_graph_1 = deepcopy(graph_mesh)
        #last_graph_2 = deepcopy(gridh2mesh)
        #last_graph_3 = deepcopy(mesh2graph)

    # Merge the PPI graphs into a single graph with multiple connected components
    node_features = torch.cat(node_features_list, 0)
    node_labels = torch.cat(node_labels_list, 0)

    for g in graph_mesh_list:  # graphs est une liste de DGLGraph
        if '_ID' not in g.edata:
            g.edata['_ID'] = torch.arange(g.num_edges(), dtype=torch.int32)
        if '_ID' not in g.ndata:
            g.ndata['_ID'] = torch.arange(g.num_nodes(), dtype=torch.int32)

    graph_list.append(dgl.batch(graph_mesh_list))
    graph_list.append(dgl.batch(grid2mesh_list))
    graph_list.append(dgl.batch(mesh2grid_list))
    
    graph_labels_list = torch.cat(graph_labels_list, 0).to(device)

    return node_features, node_labels, graph_list, graph_labels_list

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

def construct_dataset(date_ids, x_data, y_data, graph, ids_columns, ks, use_temporal_as_edges):
    Xs, Ys, Es = [], [], []
    
    """if graph.graph_method == 'graph':
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
                    x, y, e = construct_graph_set(graph, date_id, x_data_graph, y_data_graph, ks, len(ids_columns), mesh=mesh)
                else:
                    x, y, e = construct_graph_with_time_series(graph, date_id, x_data_graph, y_data_graph, ks, len(ids_columns), mesh=mesh)

                if x is None:
                    continue

                if x.shape[0] == 0:
                    continue

                Xs.append(x)
                Ys.append(y)
                Es.append(e)
    
    else:"""
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

        if x.shape[0] == 0:
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
                    ks : int,
                    mesh=False,
                    mesh_file=''):
    
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

    if not mesh:
        # Création des datasets finaux
        train_dataset = InplaceGraphDataset(Xst, Yst, Est, len(Xst), device)
        val_dataset = InplaceGraphDataset(XsV, YsV, EsV, len(XsV), device)
        test_dataset = InplaceGraphDataset(XsTe, YsTe, EsTe, len(XsTe), device)
    else:
        train_dataset = InplaceMeshGraphDataset(mesh_file, Xst, Yst, Est, len(Xst), device)
        val_dataset = InplaceMeshGraphDataset(mesh_file, XsV, YsV, EsV, len(XsV), device)
        test_dataset = InplaceMeshGraphDataset(mesh_file, XsTe, YsTe, EsTe, len(XsTe), device)

    return train_dataset, val_dataset, test_dataset

def create_test_loader(graph, df,
                       features_name,
                       device : torch.device,
                       use_temporal_as_edges : bool,
                       target_name,
                       ks :int,
                       mesh=False,
                       mesh_file=''):
    
    Xset, Yset = df[ids_columns + features_name].values, df[ids_columns + targets_columns + [target_name]].values

    X = []
    Y = []
    E = []
    """if graph.graph_method == 'graph':
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

                if x.shape[0] == 0:
                    continue

                X.append(x)
                Y.append(y)
                E.append(e)
    else:"""
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

        if x.shape[0] == 0:
            continue

        X.append(x)
        Y.append(y)
        E.append(e)

    if not mesh:
        dataset = InplaceGraphDataset(X, Y, E, len(X), device)
        collate = graph_collate_fn
    else:
        dataset = InplaceMeshGraphDataset(mesh_file, X, Y, E, len(X), device)
        collate = graph_collate_fn_mesh
    
    if use_temporal_as_edges is None:
        loader = DataLoader(dataset, dataset.__len__(), False)
    else:
        loader = DataLoader(dataset, dataset.__len__(), False, collate_fn=collate)

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

        if fet_2D == 'Past_risk':
            x_past_risk = read_object(f'X_past_risk_{date}.pkl', path)
            new_x_2D[i, :, :] = x_past_risk
        else:
            new_x_2D[i, :, :] = x_2D[features_name_2D_full.index(fet_2D), :, :]
        if False not in np.isnan(new_x_2D[i, :, :]):
            return None
        else:
            nan_mask = np.isnan(new_x_2D[i, :, :])
            new_x_2D[i, nan_mask] = np.nanmean(new_x_2D[i, :, :])
        #   print(np.unique(np.isnan(new_x_2D)))
    #
    #  Remplacer les NaN dans une matrice entière
    #nan_mean = np.nanmean(new_x_2D, axis=(1, 2), keepdims=True)  # Moyenne par plan
    #new_x_2D = np.where(np.isnan(new_x_2D), nan_mean, new_x_2D)  # Remplacement conditionnel

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
        X = np.zeros((y.shape[0], len(features_name_2D), *shape2D[graph.scale], ks + 1))
        Y = y
    else:
        X = np.zeros((len(features_name_2D), 64, 64, ks + 1))
        Y = np.zeros((64, 64, y.shape[1], ks + 1))

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
    x_date_cp[:, raster_dept != node] = 0.0
    minx, miny, maxx, maxy = np.min(mask[:, 0]), np.min(mask[:, 1]), np.max(mask[:, 0]), np.max(mask[:, 1])
    res = x_date_cp[:, minx:maxx+1, miny:maxy+1]
    res[np.isnan(res)] = np.nanmean(res)
    return res

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
    
    X[np.isnan(X)] = 0.0
    return X

def process_dept_y(Y, y_date, raster_dept, ks):
    """Process department labels and update Y."""
    for band in range(y_date.shape[0]):
        for k in range(ks + 1):
            y_band = y_date[band, :, :, k]
            y_band[np.isnan(raster_dept)] = 0
            Y[:, :, band, k] = resize_no_dim(y_band, 64, 64)
    
    Y[np.isnan(Y)] = 0.0
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
                    #print(np.unique(np.isnan(X[i])))
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

    logger.info(f'{len(Xst)}, {len(XsV)}, {len(XsTe)}')

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
    def __init__(self, model_name, nbfeatures, batch_size, lr, target_name, task_type,
                 features_name, ks, out_channels, dir_log, loss='mse', name='ModelTorch', device='cpu', under_sampling='full', over_sampling='full'):
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
        self.dir_log = dir_log
        self.task_type = task_type
        self.model_params = None
        self.under_sampling = under_sampling
        self.over_sampling = over_sampling
        self.find_log = False
        self.nbfeatures = nbfeatures

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

    def split_dataset(self, dataset, nb, reset=True):
        # Separate the positive and zero classes based on y
        positive_mask = dataset[self.target_name] > 0
        non_fire_mask = dataset[self.target_name] == 0

        # Filtrer les données positives et non feu
        df_positive = dataset[positive_mask]
        df_non_fire = dataset[non_fire_mask]

        # Échantillonner les données non feu
        nb = min(len(df_non_fire), nb)

        sampled_indices = np.random.RandomState(42).choice(len(df_non_fire), nb, replace=False)
        df_non_fire_sampled = df_non_fire.iloc[sampled_indices]

        # Combiner les données positives et non feu échantillonnées
        df_combined = pd.concat([df_positive, df_non_fire_sampled])
        # Réinitialiser les index du DataFrame combiné
        if reset:
            df_combined.reset_index(drop=True, inplace=True)
        return df_combined
    
    def add_ordinal_class(self, X, y, limit):        
        pass
    
    def search_samples_proportion(self, graph, df_train, df_val, df_test, is_unknowed_risk, reset=True):
        
        if not is_unknowed_risk:
            test_percentage = np.arange(0.1, 1.05, 0.05)
        else:
            test_percentage = np.arange(0.0, 1.05, 0.05)

        under_prediction_score_scores = []
        over_prediction_score_scores = []
        data_log = None
        find_log = False

        if False:
            if (self.dir_log / 'unknowned_scores_per_percentage.pkl').is_file():
                data_log = read_object('unknowned_scores_per_percentage.pkl', self.dir_log)
        else:
            if (self.dir_log / 'test_percentage_scores.pkl').is_file():
                print(f'Load test_percentage_score')
                find_log = True
                data_log = read_object('test_percentage_scores.pkl', self.dir_log)

        print(f'data_log : {data_log}')
        if data_log is not None:
            test_percentage, under_prediction_score_scores, over_prediction_score_scores = data_log[0], data_log[1], data_log[2]
        else:
            y_ori = df_train[self.target_name].values 
            for tp in test_percentage:

                if not is_unknowed_risk:
                    nb = int(tp * y_ori[y_ori == 0].shape[0])
                else:
                    nb = int(tp * len(X[(X['potential_risk'] > 0) & (y_ori == 0)]))

                logger.info(f'Trained with {tp} -> {nb} sample of class 0')

                df_combined = self.split_dataset(df_train.copy(deep=True), nb, reset=reset)

                copy_model = deepcopy(self)
                copy_model.under_sampling = 'full'
                copy_model.create_train_val_test_loader(graph, df_combined, df_val, df_test)
                copy_model.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose=False)
                test_loader = copy_model.create_test_loader(graph, df_test)
                test_output, y = copy_model._predict_test_loader(test_loader)
                test_output = test_output.detach().cpu().numpy()

                y = y.detach().cpu().numpy()
                y = y[:, -1]

                under_prediction_score_value = under_prediction_score(y, test_output)
                over_prediction_score_value = over_prediction_score(y, test_output)
                
                under_prediction_score_scores.append(under_prediction_score_value)
                over_prediction_score_scores.append(over_prediction_score_value)

                print(f'Under achieved : {under_prediction_score_value}, Over achived {over_prediction_score_value}')

                if under_prediction_score_value > over_prediction_score_value:
                    break
        
        # Find the index where the two scores cross (i.e., where the difference changes sign)
        score_differences = np.array(under_prediction_score_scores) - np.array(over_prediction_score_scores)

        # If no crossing point is found, use the best point based on another criterion (e.g., minimum difference)
        index_max = np.argmin(np.abs(score_differences))
        best_tp = test_percentage[index_max]         

        if is_unknowed_risk:
            plt.figure(figsize=(15, 7))
            plt.plot(test_percentage, under_prediction_score_scores, label='under_prediction')
            plt.plot(test_percentage, over_prediction_score_scores, label='over_prediction')
            plt.xticks(test_percentage)
            plt.xlabel('Percentage of Unknowed sample')
            plt.ylabel('IOU Score')

            # Ajouter une ligne verticale pour le meilleur pourcentage (best_tp)
            plt.axvline(x=best_tp, color='r', linestyle='--', label=f'Best TP: {best_tp:.2f}')
            
            # Ajouter une légende
            plt.legend()

            # Sauvegarder et fermer la figure
            plt.savefig(self.dir_log / f'{self.name}_unknowned_scores_per_percentage.png')
            plt.close()
            save_object([test_percentage, under_prediction_score_scores, over_prediction_score_scores], 'unknowned_scores_per_percentage.pkl', self.dir_log)
        else:
            plt.figure(figsize=(15, 7))
            plt.plot(test_percentage, under_prediction_score_scores, label='under_prediction')
            plt.plot(test_percentage, over_prediction_score_scores, label='over_prediction')
            plt.xticks(test_percentage)
            plt.xlabel('Percentage of Binary sample')
            plt.ylabel('IOU Score')

            # Ajouter une ligne verticale pour le meilleur pourcentage (best_tp)
            plt.axvline(x=best_tp, color='r', linestyle='--', label=f'Best TP: {best_tp:.2f}')

            # Ajouter une légende
            plt.legend()

            # Sauvegarder et fermer la figure
            plt.savefig(self.dir_log / f'{self.name}_scores_per_percentage.png')
            plt.close()
            save_object([test_percentage, under_prediction_score_scores, over_prediction_score_scores], 'test_percentage_scores.pkl', self.dir_log)

        print(best_tp)
        return best_tp, find_log

    def search_samples_limit(self, X, y, X_val, y_val, X_test, y_test):
        pass

    def score(self, X, y, sample_weight=None):
        """
        Evaluate the model's performance for each ID.

        Parameters:
        - X_val: Validation data.
        - y_val: True labels.
        - id_val: List of IDs corresponding to validation data.

        Returns:
        - Mean score across all IDs.
        """
        predictions = self.predict(X)
        return self.score_with_prediction(predictions, y, sample_weight)

    def score_with_prediction(self, y_pred, y, sample_weight=None):
        
        return iou_score(y, y_pred)

    def _predict_test_loader(self, X: DataLoader, proba=False) -> torch.tensor:
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
                    output = self.model(inputs_model)

                    if output.shape[1] > 1 and not proba:
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

    def fit(self, graph, X, y, X_val, y_val, X_test, y_test, PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params=None):
        X[y.columns] = y.values
        X_val[y.columns] = y_val.values
        X_test[y.columns] = y_test.values
        self.create_train_val_test_loader(graph, X, X_val, X_test)
        try:
            if self.find_log:
                self._load_model_from_path(self.dir_log / 'best.pt', self.model)
                return
            else:
                self.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params=custom_model_params)
        except:
            self.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params=custom_model_params)

    def filtering_pred(self, df, predTensor, y, graph):
        
        y = y.detach().cpu().numpy()
        predTensor = predTensor.detach().cpu().numpy()

        # Extraire les paires de test_dataset_dept
        test_pairs = set(zip(df['date'], df['graph_id']))

        # Normaliser les valeurs dans YTensor
        date_values = [item for item in y[:, date_index]]
        graph_id_values = [item for item in y[:, graph_id_index]]

        # Filtrer les lignes de YTensor correspondant aux paires présentes dans test_dataset_dept
        filtered_indices = [
            i for i, (date, graph_id) in enumerate(zip(date_values, graph_id_values)) 
            if (date, graph_id) in test_pairs
            ]

        # Créer YTensor filtré
        y = y[filtered_indices]

        # Créer des paires et les convertir en set
        ytensor_pairs = set(zip(date_values, graph_id_values))

        # Filtrer les lignes en vérifiant si chaque couple (date, graph_id) appartient à ytensor_pairs
        df = df[
            df.apply(lambda row: (row['date'], row['graph_id']) in ytensor_pairs, axis=1)
        ].reset_index(drop=True)
        
        if graph.graph_method == 'graph':
            def keep_one_per_pair(dataset):
                # Supprime les doublons en gardant uniquement la première occurrence par paire (graph_id, date)
                return dataset.drop_duplicates(subset=['graph_id', 'date'], keep='first')
            
            def get_unique_pair_indices(array, graph_id_index, date_index):
                """
                Retourne les indices des lignes uniques basées sur les paires (graph_id, date).
                :param array: Liste de listes (tableau Python)
                :param graph_id_index: Index de la colonne `graph_id`
                :param date_index: Index de la colonne `date`
                :return: Liste des indices correspondant aux lignes uniques
                """
                seen_pairs = set()
                unique_indices = []
                for i, row in enumerate(array):
                    pair = (row[graph_id_index], row[date_index])
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        unique_indices.append(i)
                return unique_indices

            unique_indices = get_unique_pair_indices(y, graph_id_index=graph_id_index, date_index=date_index)
            #predTensor = predTensor[unique_indices]
            y = y[unique_indices]
            df = keep_one_per_pair(df)

        df.sort_values(['graph_id', 'date'], inplace=True)
        ind = np.lexsort((y[:,0], y[:,4]))
        y = y[ind]
        predTensor = predTensor[ind]
        
        if self.target_name == 'binary' or self.target_name == 'nbsinister':
            band = -2
        else:
            band = -1

        pred = np.full((predTensor.shape[0], 2), fill_value=np.nan)
        if name in ['Unet', 'ULSTM']:
            pred = np.full((y.shape[0], 2), fill_value=np.nan)
            pred_2D = predTensor
            Y_2D = y
            udates = np.unique(df['date'].values)
            ugraph = np.unique(df['graph_id'].values)
            for graph in ugraph:
                for date in udates:
                    mask_2D = np.argwhere((Y_2D[:, graph_id_index] == graph) & (Y_2D[:, date_index] == date))
                    mask = np.argwhere((y[:, graph_id_index] == graph) & (y[:, date_index] == date))
                    if mask.shape[0] == 0:
                        continue
                    pred[mask[:, 0], 0] = pred_2D[mask_2D[:, 0], band, mask_2D[:, 1], mask_2D[:, 2]]
        else:
            pred[:, 0] = predTensor
            pred[:, 1] = predTensor

        return pred

    def predict(self, df, graph=None):
        if graph is None:
            graph = self.graph

        if self.target_name not in list(df.columns):
            df[self.target_name] = 0

        loader = create_test_loader(graph, df,
                       self.features_name,
                       self.device,
                       None,
                       self.target_name,
                       self.ks)
        
        predTensor, YTensor = self._predict_test_loader(loader)
        pred = self.filtering_pred(df, predTensor, YTensor, graph)
        return pred
    
    def predict_proba(self, df, graph=None):
        if graph is None:
            graph = self.graph

        if self.target_name not in list(df.columns):
            df[self.target_name] = 0

        loader = create_test_loader(graph, df,
                       self.features_name,
                       self.device,
                       None,
                       self.target_name,
                       self.ks)
        
        predTensor, YTensor = self._predict_test_loader(loader, True)
        pred = self.filtering_pred(df, predTensor, YTensor, graph)
        return pred

    def plot_train_val_loss(self, epochs, train_loss_list, val_loss_list, dir_log):
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
        plt.savefig(dir_log / 'Validation.png')
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
        plt.savefig(dir_log / 'Training.png')
        plt.close('all')

    def _load_model_from_path(self, path : Path, model) -> None:
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True), strict=False)
        self.model = model

    def update_weight(self, weight):
        """
        Update the model's weights with the given state dictionary.

        Parameters:
        - weight (dict): State dictionary containing the new weights.
        """

        assert self.model is not None

        if not isinstance(weight, dict):
            raise ValueError("The provided weight must be a dictionary containing model parameters.")

        model_state_dict = self.model.state_dict()

        # Vérification que toutes les clés existent dans le modèle
        missing_keys = [key for key in weight.keys() if key not in model_state_dict]
        if missing_keys:
            raise KeyError(f"Some keys in the provided weights do not match the model's parameters: {missing_keys}")

        # Charger les poids dans le modèle
        self.model.load_state_dict(weight)

class ModelCNN(ModelTorch):
    def __init__(self, model_name, nbfeatures, batch_size, lr, target_name, task_type, out_channels, dir_log, features_name, features, ks, loss, name, device, under_sampling, over_sampling, path, image_per_node):
        super().__init__(model_name, nbfeatures, batch_size, lr, target_name, task_type, features_name, ks, out_channels, dir_log, loss=loss, name=name, device=device, under_sampling=under_sampling, over_sampling=over_sampling)
        self.path = path
        self.features = features
        self.image_per_node = image_per_node
        self.nbfeatures = nbfeatures
        
    def create_train_val_test_loader(self, graph, df_train, df_val, df_test):

        self.graph = graph

        importance_df = calculate_and_plot_feature_importance(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        importance_df = calculate_and_plot_feature_importance_shapley(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        features95, featuresAll = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)
        if self.nbfeatures != 'all':
            self.features_name = featuresAll[:int(self.nbfeatures)]
        else:
            self.features_name = featuresAll
        if self.under_sampling != 'full':
            y = df_train[self.target_name]
            old_shape = df_train.shape
            if 'binary' in self.under_sampling:
                vec = self.under_sampling.split('-')
                try:
                    nb = int(vec[-1]) * len(df_train[df_train[self.target_name] > 0])
                except:
                    logger.info(f'{self.under_sampling} with undefined factor, set to 1 -> {len(df_train[df_train[self.target_name] > 0])}')
                    nb = len(df_train[df_train[self.target_name] > 0])

                df_combined = self.split_dataset(df_train, nb)

                # Mettre à jour df_train pour l'entraînement
                df_train = df_combined

                logger.info(f'Train mask df_train shape: {old_shape} -> {df_train.shape}')

            elif self.under_sampling == 'search' or 'percentage' in self.under_sampling:
                    if self.under_sampling == 'search':
                        best_tp, find_log = self.search_samples_proportion(graph, df_train, df_val, df_test, False)
                        self.find_log = find_log
                    else:
                        vec = self.under_sampling.split('-')
                        try:
                            best_tp = float(vec[-1])
                        except ValueError:
                            logger.info(f'{self.under_sampling} with undefined factor, set to 0.3 -> {0.3 * len(y[y == 0])}')
                            best_tp = 0.3

                    nb = int(best_tp * len(y[y == 0]))

                    df_combined = self.split_dataset(df_train, nb)
                    df_train = df_combined
                    logger.info(f'Train mask df_train shape: {old_shape} -> {df_train.shape}')

        if self.over_sampling == 'full':
            pass
            
        else:
            raise ValueError(f'Unknow value of under_sampling -> {self.over_sampling}')
                
        if False:
            self.train_loader = read_object('train_loader.pkl', self.dir_log)
            self.val_loader = read_object('val_loader.pkl', self.dir_log)
            self.test_loader = read_object('test_loader.pkl', self.dir_log)
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
        
            train_loader = DataLoader(train_dataset, 16, True,)
            val_loader = DataLoader(val_dataset, 16, False)
            test_loader = DataLoader(test_dataset, 16, False)

            save_object_torch(train_loader, 'train_loader.pkl', self.dir_log)
            save_object_torch(val_loader, 'val_loader.pkl', self.dir_log)
            save_object_torch(test_loader, 'test_loader.pkl', self.dir_log)

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
                #output = torch.reshape(output, target.shape)
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
                    #output = torch.reshape(output, target.shape)
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

    def train(self, graph, PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params=None, new_model=True):
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

        check_and_create_path(self.dir_log)

        criterion = get_loss_function(self.loss)

        if new_model:
            self.model, _ = self.make_model(graph, custom_model_params)
        
        #if (dir_log / '100.pt').is_file():
        #    model.load_state_dict(torch.load((dir_log / '100.pt'), map_location=device, weights_only=True), strict=False)

        parameters = self.model.parameters()
        if has_method(criterion, 'get_learnable_parameters'):
            logger.info(f'Adding {self.loss} parameter')
            loss_parameters = criterion.get_learnable_parameters().values()
            parameters = list(parameters) + list(loss_parameters)

        optimizer = optim.Adam(parameters, lr=self.lr)

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
                    save_object_torch(self.model.state_dict(), 'last.pt', self.dir_log)
                    save_object_torch(BEST_MODEL_PARAMS, 'best.pt', self.dir_log)
                    plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, self.dir_log)
                    if MLFLOW:
                        mlflow.end_run()
                    return
            if MLFLOW:
                mlflow.log_metric('loss', val_loss, step=epoch)
            if epoch % CHECKPOINT == 0:
                logger.info(f'epochs {epoch}, Val loss {val_loss}')
                logger.info(f'epochs {epoch}, Best val loss {BEST_VAL_LOSS}')
                save_object_torch(self.model.state_dict(), str(epoch)+'.pt', self.dir_log)

        logger.info(f'Last val loss {val_loss}')
        save_object_torch(self.model.state_dict(), 'last.pt', self.dir_log)
        save_object_torch(BEST_MODEL_PARAMS, 'best.pt', self.dir_log)
        self.plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, self.dir_log)

        if 'loss_parameters' in locals():
            logger.info(f'################ Loss parameter ###################')
            logger.info(criterion.ratio_matrix)
            
class ModelGNN(ModelTorch):
    def __init__(self, graph_method, mesh, mesh_file, model_name, nbfeatures, batch_size, lr, target_name, task_type, out_channels, dir_log, features_name, ks, loss, name, device, under_sampling, over_sampling):
        super().__init__(model_name, nbfeatures, batch_size, lr, target_name, task_type, features_name, ks, out_channels, dir_log, loss=loss, name=name, device=device, under_sampling=under_sampling, over_sampling=over_sampling)
        self.mesh = mesh
        self.mesh_file = mesh_file
        self.graph_method = graph_method

    def create_train_val_test_loader(self, graph, df_train, df_val, df_test):

        self.graph = graph

        importance_df = calculate_and_plot_feature_importance(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        #importance_df = calculate_and_plot_feature_importance_shapley(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        features95, featuresAll = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)
        
        if self.nbfeatures != 'all':
            self.features_name = featuresAll[:int(self.nbfeatures)]
        else:
            self.features_name = featuresAll

        print(self.features_name)
        
        if self.under_sampling != 'full':
            y = df_train[self.target_name]
            old_shape = df_train.shape
            if 'binary' in self.under_sampling:
                vec = self.under_sampling.split('-')
                try:
                    nb = int(vec[-1]) * len(df_train[df_train[self.target_name] > 0])
                except:
                    logger.info(f'{self.under_sampling} with undefined factor, set to 1 -> {len(df_train[df_train[self.target_name] > 0])}')
                    nb = len(df_train[df_train[self.target_name] > 0])

                df_combined = self.split_dataset(df_train, nb, reset=False)

                # Mettre à jour df_train pour l'entraînement
                df_train.loc[df_combined.index, 'weight'] = 0

                logger.info(f'Train mask df_train shape: {old_shape} -> {df_train[df_train["weight"] > 0].shape}')

            elif self.under_sampling == 'search' or 'percentage' in self.under_sampling:
                    if self.under_sampling == 'search':
                        best_tp, find_log = self.search_samples_proportion(graph, df_train, df_val, df_test, False, False)
                        self.find_log = find_log
                    else:
                        vec = self.under_sampling.split('-')
                        try:
                            best_tp = float(vec[-1])
                        except ValueError:
                            logger.info(f'{self.under_sampling} with undefined factor, set to 0.3 -> {0.3 * len(y[y == 0])}')
                            best_tp = 0.3

                    nb = int(best_tp * len(y[y == 0]))

                    df_combined = self.split_dataset(df_train, nb, reset=False)
                    df_train.loc[df_combined.index, 'weight'] = 0
                    logger.info(f'Train mask df_train shape: {old_shape} -> {df_train[df_train["weight"] > 0].shape}')
        
        if self.over_sampling == 'full':
            pass

        elif self.over_sampling == 'smote':
            if self.task_type == 'classification':
                """y_negative = y[y == 0].shape[0]
                y_one = y_negative * 0.01
                y_two = y_negative * 0.01
                y_three = y_negative * 0.01
                y_four = y_negative * 0.01
                smote = SMOTE(random_state=42, sampling_strategy={0 : y_negative, 1 : y_one, 2 : y_two, 3 : y_three, 4 : y_four})"""
                smote = SMOTE(random_state=42, sampling_strategy='auto')
            elif self.task_type == 'binary':
                smote = SMOTE(random_state=42, sampling_strategy='auto')
            df_train = smote.fit_resample(df_train)

        else:
            raise ValueError(f'Unknow value of under_sampling -> {self.over_sampling}')
            
        #if (self.dir_log / 'train_loader.pkl').is_file():
        if False:
            self.train_loader = read_object('train_loader.pkl', self.dir_log)
            self.val_loader = read_object('val_loader.pkl', self.dir_log)
            self.test_loader = read_object('test_loader.pkl', self.dir_log)
        else:
            train_dataset, val_dataset, test_dataset = create_dataset(graph,
                                                                df_train,
                                                                df_val,
                                                                df_test,
                                                                self.features_name,
                                                                self.target_name,
                                                                False,
                                                                self.device, self.ks,
                                                                self.mesh,
                                                                self.mesh_file)
            
            if not self.mesh:            
                self.train_loader = DataLoader(train_dataset, batch_size, True, collate_fn=graph_collate_fn)
                self.val_loader = DataLoader(val_dataset, val_dataset.__len__(), False, collate_fn=graph_collate_fn)
                self.test_loader = DataLoader(test_dataset, test_dataset.__len__(), False, collate_fn=graph_collate_fn)
            else:
                self.train_loader = DataLoader(train_dataset, batch_size, True, collate_fn=graph_collate_fn_mesh)
                self.val_loader = DataLoader(val_dataset, val_dataset.__len__(), False, collate_fn=graph_collate_fn_mesh)
                self.test_loader = DataLoader(test_dataset, test_dataset.__len__(), False, collate_fn=graph_collate_fn_mesh)


        save_object_torch(self.train_loader, 'train_loader.pkl', self.dir_log)
        save_object_torch(self.val_loader, 'val_loader.pkl', self.dir_log)
        save_object_torch(self.test_loader, 'test_loader.pkl', self.dir_log)

    def create_test_loader(self, graph, df):
        loader = create_test_loader(graph, df,
                       self.features_name,
                       self.device,
                       False,
                       self.target_name,
                       self.ks,
                       self.mesh,
                       self.mesh_file)

        return loader

    def launch_train_loader(self, loader, criterion, optimizer):
        
        #if has_method(criterion, 'get_learnable_parameters'):
        #    criterion.eval()
            
        self.model.train()
        for i, data in enumerate(loader, 0):
            
            if not self.mesh:
                inputs, labels, graphs, graphs_id = data
            else:
                inputs, labels, DGLgraphs, graphs_id = data

            band = -1
            target, weights = self.compute_weights_and_target(labels, band, ids_columns, self.model.is_graph_or_node, graphs_id)

            inputs_model = inputs

            if not self.mesh:
                output = self.model(inputs_model, graphs)
            else:
                output = self.model(inputs_model, DGLgraphs[0], DGLgraphs[1], DGLgraphs[2])

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
                #output = torch.reshape(output, target.shape)
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss

    def launch_val_test_loader(self, loader, criterion, optimizer):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():

            for i, data in enumerate(loader, 0):
                
                if not self.mesh:
                    inputs, labels, graphs, graphs_id = data
                else:
                    inputs, labels, DGLgraphs, graphs_id = data

                # Determine the index of the target variable in labels
                #if target_name == 'binary' or target_name == 'nbsinister':
                #    band = -2
                #else:
                band = -1

                target, weights = self.compute_weights_and_target(labels, band, ids_columns, self.model.is_graph_or_node, graphs_id)

                inputs_model = inputs
               
                if not self.mesh:
                    output = self.model(inputs_model, graphs)
                else:
                    output = self.model(inputs_model, DGLgraphs[0], DGLgraphs[1], DGLgraphs[2])

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
                    #output = torch.reshape(output, target.shape)
                    loss = criterion(output, target)

                total_loss += loss.item()

        return total_loss

    def func_epoch(self, train_loader, val_loader, optimizer, optimizer_loss, criterion):
        
        train_loss = self.launch_train_loader(train_loader, criterion, optimizer)

        if val_loader is not None:
            val_loss = self.launch_val_test_loader(val_loader, criterion, optimizer_loss)
        
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

    def train(self, graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose=True, custom_model_params=None, new_model=True):
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

        check_and_create_path(self.dir_log)

        criterion = get_loss_function(self.loss)

        if new_model:
            self.model, _ = self.make_model(graph, custom_model_params)
            
        #if (dir_log / '100.pt').is_file():
        #    model.load_state_dict(torch.load((dir_log / '100.pt'), map_location=device, weights_only=True), strict=False)

        parameters = self.model.parameters()
        if has_method(criterion, 'get_learnable_parameters'):
            logger.info(f'Adding {self.loss} parameter')
            loss_parameters = criterion.get_learnable_parameters().values()
            #optimizer_loss = optim.Adam(parameters, lr=self.lr)
            parameters = list(parameters) + list(loss_parameters)
        else:
            optimizer_loss = None

        optimizer = optim.Adam(parameters, lr=self.lr)

        BEST_VAL_LOSS = math.inf
        BEST_MODEL_PARAMS = None
        patience_cnt = 0

        val_loss_list = []
        train_loss_list = []
        epochs_list = []

        logger.info('Train model with')
        for epoch in tqdm(range(epochs), disable=not verbose):
            val_loss, train_loss = self.func_epoch(self.train_loader, self.val_loader, optimizer, optimizer_loss, criterion)
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
                    save_object_torch(self.model.state_dict(), 'last.pt', self.dir_log)
                    save_object_torch(BEST_MODEL_PARAMS, 'best.pt', self.dir_log)
                    plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, self.dir_log)
                    if MLFLOW:
                        mlflow.end_run()
                    return
            if MLFLOW:
                mlflow.log_metric('loss', val_loss, step=epoch)
            if epoch % CHECKPOINT == 0 and verbose:
                logger.info(f'epochs {epoch}, Val loss {val_loss}')
                logger.info(f'epochs {epoch}, Best val loss {BEST_VAL_LOSS}')
                save_object_torch(self.model.state_dict(), str(epoch)+'.pt', self.dir_log)

        logger.info(f'Last val loss {val_loss}')
        save_object_torch(self.model.state_dict(), 'last.pt', self.dir_log)
        save_object_torch(BEST_MODEL_PARAMS, 'best.pt', self.dir_log)
        self.plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, self.dir_log)

    def _predict_test_loader(self, X: DataLoader, proba=False) -> torch.tensor:
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
                
                if not self.mesh:
                    inputs, orilabels, graphs, graphs_id = data
                else:
                    inputs, orilabels, DGLgraphs, graphs_id = data

                orilabels = orilabels.to(device)
                orilabels = orilabels[:, :, -1]

                #labels = compute_labels(orilabels, self.model.is_graph_or_node, graphs)

                inputs_model = inputs
                if not self.mesh:
                    output = self.model(inputs_model, graphs)
                else:
                    output = self.model(inputs_model, DGLgraphs[0], DGLgraphs[1], DGLgraphs[2])

                if output.shape[1] > 1 and not proba:
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
    def __init__(self, model_name, nbfeatures, batch_size, lr, target_name, task_type, out_channels, dir_log, features_name, ks, loss, name, device, under_sampling, over_sampling):
        super().__init__(model_name, nbfeatures, batch_size, lr, target_name, task_type, features_name, ks, out_channels, dir_log, loss=loss, name=name, device=device, under_sampling=under_sampling, over_sampling=over_sampling)

    def create_train_val_test_loader(self, graph, df_train, df_val, df_test):
        self.graph = graph

        ##################################### Select features #########################################
        importance_df = calculate_and_plot_feature_importance(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        #importance_df = calculate_and_plot_feature_importance_shapley(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        features95, featuresAll = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)

        if self.nbfeatures != 'all':
            self.features_name = featuresAll[:int(self.nbfeatures)]
        else:
            self.features_name = featuresAll
        
        ##################################### Define percentage of 0 samples #########################################
        if self.under_sampling != 'full':
            old_shape = df_train.shape
            y = df_train[self.target_name]
            if 'binary' in self.under_sampling:
                vec = self.under_sampling.split('-')
                try:
                    nb = int(vec[-1]) * len(df_train[df_train[self.target_name] > 0])
                except:
                    logger.info(f'{self.under_sampling} with undefined factor, set to 1 -> {len(df_train[df_train[self.target_name] > 0])}')
                    nb = len(df_train[df_train[self.target_name] > 0])

                df_combined = self.split_dataset(df_train, nb)

                # Mettre à jour df_train pour l'entraînement
                df_train = df_combined

                logger.info(f'Train mask df_train shape: {old_shape} -> {df_train.shape}')

            elif self.under_sampling == 'search' or 'percentage' in self.under_sampling:
                    if self.under_sampling == 'search':
                        best_tp, find_log = self.search_samples_proportion(graph, df_train, df_val, df_test, False)
                        self.find_log = find_log
                    else:
                        vec = self.under_sampling.split('-')
                        try:
                            best_tp = float(vec[-1])
                        except ValueError:
                            logger.info(f'{self.under_sampling} with undefined factor, set to 0.3 -> {0.3 * len(y[y == 0])}')
                            best_tp = 0.3

                    nb = int(best_tp * len(y[y == 0]))

                    df_combined = self.split_dataset(df_train, nb)
                    df_train = df_combined
                    logger.info(f'Train mask df_train shape: {old_shape} -> {df_train.shape}')

        if self.over_sampling == 'full':
            pass

        elif self.over_sampling == 'smote':
            if self.task_type == 'classification':
                """y_negative = y[y == 0].shape[0]
                y_one = y_negative * 0.01
                y_two = y_negative * 0.01
                y_three = y_negative * 0.01
                y_four = y_negative * 0.01
                smote = SMOTE(random_state=42, sampling_strategy={0 : y_negative, 1 : y_one, 2 : y_two, 3 : y_three, 4 : y_four})"""
                smote = SMOTE(random_state=42, sampling_strategy='auto')
            elif self.task_type == 'binary':
                smote = SMOTE(random_state=42, sampling_strategy='auto')
            df_train = smote.fit_resample(df_train)

        else:
            raise ValueError(f'Unknow value of under_sampling -> {self.over_sampling}')

        self.df_train = df_train
        self.df_test = df_test
        self.df_val = df_val

        ##################################### Create loader #########################################        
        #if (self.dir_log / 'train_loader.pkl').is_file():
        if False:
            self.train_loader = read_object('train_loader.pkl', self.dir_log)
            self.val_loader = read_object('val_loader.pkl', self.dir_log)
            self.test_loader = read_object('test_loader.pkl', self.dir_log)
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

        save_object_torch(self.train_loader, 'train_loader.pkl', self.dir_log)
        save_object_torch(self.val_loader, 'val_loader.pkl', self.dir_log)
        save_object_torch(self.test_loader, 'test_loader.pkl', self.dir_log)

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
                loss = criterion(output, target, weights, update_matrix=True)
            else:
                target = torch.masked_select(target, weights.gt(0))
                output = output[weights.gt(0)]
                target = target.long()
                weights = torch.masked_select(weights, weights.gt(0))
                loss = criterion(output, target, update_matrix=True)

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
            val_loss = train_loss.item()

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

    def train(self, graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose=True, custom_model_params=None, new_model=True):
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

        check_and_create_path(self.dir_log)

        criterion = get_loss_function(self.loss)

        if new_model:
            self.model, _ = self.make_model(graph, custom_model_params)

        #print(len(self.features_name))

        parameters = self.model.parameters()

        if has_method(criterion, 'get_learnable_parameters'):
            logger.info(f'Adding {self.loss} parameter')
            loss_parameters = criterion.get_learnable_parameters().values()
            parameters = list(parameters) + list(loss_parameters)

        optimizer = optim.Adam(parameters, lr=self.lr)

        BEST_VAL_LOSS = math.inf
        BEST_MODEL_PARAMS = None
        patience_cnt = 0

        val_loss_list = []
        train_loss_list = []
        epochs_list = []

        #disable = None if not verbose else False

        for epoch in tqdm(range(epochs), disable=not verbose):
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
                    save_object_torch(self.model.state_dict(), 'last.pt', self.dir_log)
                    save_object_torch(BEST_MODEL_PARAMS, 'best.pt', self.dir_log)
                    plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, self.dir_log)
                    if MLFLOW:
                        mlflow.end_run()
                    return
            if MLFLOW:
                mlflow.log_metric('loss', val_loss, step=epoch)
            if epoch % CHECKPOINT == 0 and verbose:
                logger.info(f'epochs {epoch}, Val loss {val_loss}')
                logger.info(f'epochs {epoch}, Best val loss {BEST_VAL_LOSS}')
                save_object_torch(self.model.state_dict(), str(epoch)+'.pt', self.dir_log)
                if has_method(criterion, 'get_learnable_parameters'):
                    logger.info(f'################ Loss parameter ###################')
                    mat = torch.round(input=criterion.ratio_matrix, decimals=2)
                    logger.info(mat)

        logger.info(f'Last val loss {val_loss}')
        save_object_torch(self.model.state_dict(), 'last.pt', self.dir_log)
        save_object_torch(BEST_MODEL_PARAMS, 'best.pt', self.dir_log)
        self.plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, self.dir_log)

class FederatedLearningModel(RegressorMixin, ClassifierMixin):
    def __init__(self, federated_model, features, federated_cluster='departement', loss='mse', 
                 name='FederatedModel', dir_log=Path('../'), under_sampling='full', over_sampling='full',
                 target_name='nbsinister', post_process=None, task_type='classification', 
                 aggregation_method='mean'):
        """
        Initialize the Federated Learning Model.

        Parameters:
        - federated_model: The base model to be used across all clusters.
        - federated_cluster: Column name used to identify clusters (default: 'departement').
        - aggregation_method: Method to aggregate local models ('mean', 'median', 'weighted', etc.).
        """
        super().__init__()
        self.features = features
        self.federated_cluster = federated_cluster
        self.name = name
        self.loss = loss
        self.dir_log = dir_log
        self.under_sampling = under_sampling
        self.over_sampling = over_sampling
        self.target_name = target_name
        self.post_process = post_process
        self.task_type = task_type
        self.aggregation_method = aggregation_method  # Méthode d'agrégation
        self.global_model = deepcopy(federated_model)  # Modèle global

    def fit(self, df_train, df_val, df_test, graph, args):
        """
        Train local models for each federated cluster, aggregate them into a global model, 
        and stop training once the global score does not improve for patience_count_global epochs.
        """
        self.global_model.graph = graph

        initiate_model, model_params = self.global_model.make_model(graph, custom_model_params=None)
        self.global_model.model = deepcopy(initiate_model)
        self.global_model.model_params = deepcopy(model_params)
        del initiate_model
        del model_params
        
        # Vérifier que la méthode d'agrégation est implémentée
        if self.aggregation_method not in ['mean', 'median', 'weighted']:
            raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' is not implemented.")

        # Récupération des paramètres d'entraînement
        global_epochs = args.get('global_epochs', 10)
        local_epochs = args.get('local_epochs', 5)
        patience_count_global = args.get('patience_count_global', 3)
        patience_count_local = args.get('patience_count_local', 2)

        if self.federated_cluster in np.unique(df_train.columns):
            clusters = df_train[self.federated_cluster].unique()
        else:
            raise NotImplementedError(f"Aggregation method '{self.federated_cluster}' is not implemented.")

        best_global_score = float('-inf')
        patience_counter = 0  # Compteur pour l'arrêt anticipé
        
        print(f"\n--- Training Federated Model for {global_epochs} global epochs ---")

        for epoch in range(global_epochs):
            print(f"\n--- Global Epoch {epoch + 1}/{global_epochs} ---")

            local_models = []
            local_weights = []

            for cluster in clusters:
                print(f"\nTraining local model for cluster: {cluster}")

                # Initialisation du modèle local
                local_model = deepcopy(self.global_model)

                # Création des datasets pour le cluster fédéré
                df_train_cluster, df_val_cluster, df_test_cluster = self.create_cluster_set(
                    df_train, df_val, df_test, cluster
                )

                if df_val_cluster.shape[0] == 0 or df_train_cluster.shape[0] == 0:
                    print(f'Skipping {cluster} due to empty dataset')
                    continue

                local_model.create_train_val_test_loader(graph, df_train_cluster, df_val_cluster, df_test_cluster)

                # Entraînement du modèle local
                local_model.train(graph, patience_count_local, CHECKPOINT, local_epochs, verbose=False, custom_model_params=None, new_model=False)
                
                local_models.append(local_model)

                # Stocker les poids des modèles locaux
                local_weights.append(deepcopy(local_model.model.state_dict()))

            # Agréger les modèles locaux dans le modèle global
            self.aggregate_models(local_weights)

            # Évaluer le modèle global
            global_score = self.global_model.score(df_val, df_val[self.target_name])
            print(f"\nGlobal Model Score after epoch {epoch + 1}: {global_score:.4f}")

            # Vérifier si le score s'est amélioré
            if global_score > best_global_score:
                best_global_score = global_score
                patience_counter = 0
            else:
                patience_counter += 1

            # Arrêt anticipé si le score global ne s'améliore plus
            if patience_counter >= patience_count_global:
                print("\nEarly stopping: Global model score did not improve.")
                break

        self.is_fitted_ = True
        print("\n--- Federated Learning Training Complete ---")

    def create_cluster_set(self, df_train, df_val, df_test, cluster):
        """
        Create training, validation, and test datasets for a given cluster.
        """
        if self.federated_cluster == 'departement':
            X_cluster = df_train[df_train[self.federated_cluster] == cluster].reset_index(drop=True)
            X_val_cluster = df_val[df_val[self.federated_cluster] == cluster].reset_index(drop=True)
            X_test_cluster = df_test[df_test[self.federated_cluster] == cluster].reset_index(drop=True)

            return X_cluster, X_val_cluster, X_test_cluster
        
        raise NotImplementedError(f"Federated clustering method '{self.federated_cluster}' is not implemented.")

    def aggregate_models(self, local_weights):
        """
        Aggregate local models into the global model using the chosen method.
        """
        print(f"\n--- Aggregating models using {self.aggregation_method} ---")

        param_keys = local_weights[0].keys()
        new_state_dict = {}

        for key in param_keys:
            stacked_params = torch.stack([weights[key] for weights in local_weights])

            if self.aggregation_method == 'mean':
                new_state_dict[key] = torch.mean(stacked_params, dim=0)
            elif self.aggregation_method == 'median':
                new_state_dict[key] = torch.median(stacked_params, dim=0)[0]
            elif self.aggregation_method == 'weighted':
                weights = torch.tensor([1 / len(local_weights)] * len(local_weights))  # Uniform weights
                new_state_dict[key] = torch.sum(stacked_params * weights[:, None, None], dim=0)

        # Mettre à jour les poids du modèle global
        self.global_model.update_weight(new_state_dict)
        print("\n--- Global Model Weights Updated ---")

    def predict(self, X):
        """
        Predict using the aggregated global model.
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Please train the model before predicting.")

        print(f'Predicting using Global Model')
        return self.global_model.predict(X)

    def predict_proba(self, X):
        """
        Predict probabilities using the aggregated global model.
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Please train the model before predicting.")

        if hasattr(self.global_model, "predict_proba"):
            return self.global_model.predict_proba(X)
        else:
            raise AttributeError("The global model does not support predict_proba.")

class ModelVotingPytorchAndSklearn(RegressorMixin, ClassifierMixin):
    def __init__(self, models, features, loss='mse', name='ModelVoting', dir_log=Path('../'), under_sampling='full', target_name='nbsinister', post_process=None, task_type='classification'):
        """
        Initialize the ModelVoting class.

        Parameters:
        - models: A list of base models to use (must follow the sklearn API).
        - name: The name of the model.
        - loss: Loss function to use ('logloss', 'hinge_loss', 'mse', 'rmse', etc.).
        """
        super().__init__()
        self.best_estimator_ = models  # Now a list of models
        self.features = features
        self.name = name
        self.loss = loss
        self.is_fitted_ = [False] * len(models)  # Keep track of fitted models
        self.features_per_model = []
        self.dir_log = dir_log
        self.post_process = post_process
        self.under_sampling = under_sampling
        self.target_name = target_name
        self.task_type = task_type

    def fit(self, X, y, X_val, y_val, X_test, y_test, args):
        """
        Train each model on the corresponding data.

        Parameters:
        - X_list: List of training data for each model.
        - y_list: List of labels for the training data for each model.
        - optimization: Optimization method to use ('grid' or 'bayes').
        - grid_params_list: List of parameters to optimize for each model.
        - fit_params_list: List of additional parameters for the fit function for each model.
        - cv_folds: Number of cross-validation folds.
        """

        ######## Pytorch params
        PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params_list = args['PATIENCE_CNT'], args['CHECKPOINT'], args['epochs'], args['custom_model_params_list']
        graph = args['graph']
        training_mode = args['training_mode']
        optimization = args['optimization']
        grid_params_list = args['grid_params_list']
        fit_params_list = args['fit_params_list']
        cv_folds = args['cv_folds']
        
        df_val = X_val.copy(deep=True)
        df_val[y_val.columns] = y_val
        self.cv_results_ = []
        self.is_fitted_ = [True] * len(self.best_estimator_)
        self.weights_for_model = []
        for i, model in enumerate(self.best_estimator_):
            model.dir_log = self.dir_log / '..' / model.name
            print(f'Fitting model -> {model.name}')

            #if issubclass(model, Model_Torch):
            model.fit(graph, X, y, X_val, y_val, X_test, y_test, PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params=None)
            #else:
            #    model.fit(graph, X, y[targets[i]], X_val, y_val[targets[i]], training_mode=training_mode, optimization=optimization, grid_params=grid_params_list[i], fit_params=fit_params_list[i], cv_folds=cv_folds)

            test_loader = model.create_test_loader(graph, df_val)
            test_output, y_test_val = model._predict_test_loader(test_loader)
            test_output = test_output.detach().cpu().numpy()
            y_test_val = y_test_val.detach().cpu().numpy()[:, -1]
                
            score_model = self.score_with_prediction(y_test_val, test_output)
            self.weights_for_model.append(score_model)

        self.weights_for_model = np.asarray(self.weights_for_model)
        # Affichage des poids et des modèles
        print("\n--- Final Model Weights ---")
        for model, weight in zip(self.best_estimator_, self.weights_for_model):
            print(f"Model: {model.name}, Weight: {weight:.4f}")

        # Plot des poids des modèles
        model_names = [model.name for model in self.best_estimator_]
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, self.weights_for_model, color='skyblue', edgecolor='black')
        plt.title('Model Weights', fontsize=16)
        plt.xlabel('Models', fontsize=14)
        plt.ylabel('Weights', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.tight_layout()
        plt.savefig(self.dir_log / 'weights_of_models.png')
        plt.close('all')

        save_object([model_names, self.weights_for_model], f'weights.pkl', self.dir_log)

    def predict_nbsinister(self, X, ids=None, preprocessor_ids=None, hard_or_soft='soft', weights_average=True):
        
        if self.target_name == 'nbsinister':
            return self.predict(X)
        else:
            assert self.post_process is not None
            predict = self.predict(X)
            return self.post_process.predict_nbsinister(predict, ids)
    
    def predict_risk(self, X, ids=None, preprocessor_ids=None, hard_or_soft='soft', weights_average=True):

        if self.task_type == 'classification':
            return self.predict(X)
        
        elif self.task_type == 'binary':
                assert self.post_process is not None
                predict = self.predict_proba(X)[:, 1]

                if isinstance(ids, pd.Series):
                    ids = ids.values
                if isinstance(preprocessor_ids, pd.Series):
                    preprocessor_ids = preprocessor_ids.values
    
                return self.post_process.predict_risk(predict, None, ids, preprocessor_ids)
        else:
            assert self.post_process is not None
            predict = self.predict(X)

            if isinstance(ids, pd.Series):
                ids = ids.values
            if isinstance(preprocessor_ids, pd.Series):
                preprocessor_ids = preprocessor_ids.values

            return self.post_process.predict_risk(predict, None, ids, preprocessor_ids)

    def predict(self, X, hard_or_soft='soft', weights_average=True):
        """
        Predict labels for input data using each model and aggregate the results.

        Parameters:
        - X_list: List of data to predict labels for.

        Returns:
        - Aggregated predicted labels.
        """
        print(f'Predict with {hard_or_soft} and weighs at {weights_average}')
        if hard_or_soft == 'hard':
            predictions = []
            for i, estimator in enumerate(self.best_estimator_):

                pred = estimator.predict(X)
                pred = pred.cpu().detach().numpy()
                predictions.append(pred)

            # Aggregate predictions
            aggregated_pred = self.aggregate_predictions(predictions, weights_average)
            return aggregated_pred
        else:
            aggregated_pred = self.predict_proba(X, weights_average)
            predictions = np.argmax(aggregated_pred, axis=1)
            return predictions

    def predict_proba(self, X, weights_average):
        """
        Predict probabilities for input data using each model and aggregate the results.

        Parameters:
        - X_list: List of data to predict probabilities for.
        
        Returns:
        - Aggregated predicted probabilities.
        """
        probas = []
        for i, estimator in enumerate(self.best_estimator_):
            X_ = X
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(X_)
                proba = proba.cpu().detach().numpy()
                probas.append(proba)
            else:
                raise AttributeError(f"The model at index {i} does not support predict_proba.")
            
        # Aggregate probabilities
        aggregated_proba = self.aggregate_probabilities(probas, weights_average)
        return aggregated_proba

    def aggregate_predictions(self, predictions_list, weights_average=True):
        """
        Aggregate predictions from multiple models with weights.

        Parameters:
        - predictions_list: List of predictions from each model.

        Returns:
        - Aggregated predictions.
        """
        predictions_array = np.array(predictions_list)
        if weights_average:
            weights_to_use = self.weights_for_model
        else:
            weights_to_use = np.ones_like(self.weights_for_model)

        if self.task_type == 'classification':
            # Weighted vote for classification
            unique_classes = np.arange(0, 5)
            weighted_votes = np.zeros((len(unique_classes), predictions_array.shape[1]))

            for i, cls in enumerate(unique_classes):
                mask = (predictions_array == cls)
                weighted_votes[i] = np.sum(mask * weights_to_use[:, None], axis=0)

            aggregated_pred = unique_classes[np.argmax(weighted_votes, axis=0)]
        else:
            # Weighted average for regression
            weighted_sum = np.sum(predictions_array * weights_to_use[:, None], axis=0)
            aggregated_pred = weighted_sum / np.sum(weights_to_use)
        
        return aggregated_pred

    def aggregate_probabilities(self, probas_list, weights_average=True):
        """
        Aggregate probabilities from multiple models with weights.

        Parameters:
        - probas_list: List of probability predictions from each model.

        Returns:
        - Aggregated probabilities.
        """
        probas_array = np.array(probas_list)
        if weights_average:
            weights_to_use = self.weights_for_model
        else:
            weights_to_use = np.ones_like(self.weights_for_model)
        
        # Weighted average for probabilities
        weighted_sum = np.sum(probas_array * weights_to_use[:, None, None], axis=0)
        aggregated_proba = weighted_sum / np.sum(weights_to_use)
        
        return aggregated_proba 

    def score(self, X, y, sample_weight=None):
        """
        Evaluate the model's performance for each ID.

        Parameters:
        - X_val: Validation data.
        - y_val: True labels.
        - id_val: List of IDs corresponding to validation data.

        Returns:
        - Mean score across all IDs.
        """
        predictions = self.predict(X)
        return self.score_with_prediction(predictions, y, sample_weight)
    
    def score_with_prediction(self, y_pred, y, sample_weight=None):
        
        return iou_score(y, y_pred)
    
class Model_susceptibility():
    def __init__(self, model_name, target, resolution, model_config, features_name, out_channels, task_type, ks, departements, train_departements, train_date, val_date, dir_log):
            self.model_name = model_name
            self.features_name = features_name
            self.out_channels = out_channels
            self.task_type = task_type
            self.ks = ks
            self.model_config = model_config
            self.resolution = resolution
            self.departements = departements
            self.train_date = train_date
            self.val_date = val_date
            self.train_departements = train_departements
            self.dir_log = dir_log
            self.target = target

    def susecptibility_map_individual_pixel_feature(self, dept, year, variables, target_value, sdate_year, edate_year, raster, dir_data, dir_output):

        if target_value is not None:
            target_value = target_value.flatten()
            
            risk_arg = np.argwhere((target_value > 0) & (~np.isnan(target_value)))[:, 0]
            mask_X = list(risk_arg)
            non_risk_arg = np.argwhere(target_value == 0)[:, 0]
            mask_X += list(np.random.choice(non_risk_arg, min(non_risk_arg.shape[0], risk_arg.shape[0]), replace=False))
            mask_X = np.asarray(mask_X).reshape(-1,1)

            y = list(target_value[mask_X[:, 0]])
        vec_x = []
        
        assert raster is not None
        for var in variables:
            if var in cems_variables:
                values = read_object(f'{var}raw.pkl', dir_data)
                assert values is not None
                values = values[:, :, allDates.index(sdate_year):allDates.index(edate_year)]
                values = np.mean(values, axis=2)
                save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var}', values, raster)
                values = values.flatten()
                vec_x.append(values[mask_X[:, 0]])

            elif var == 'population' or var == 'elevation':
                values = read_object(f'{var}.pkl', dir_data)
                assert values is not None
                values = values.reshape((values.shape[0], values.shape[1]))
                save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var}', values, raster)
                values = values.flatten()
                vec_x.append(values[mask_X[:, 0]])

            elif var == 'foret':
                values = read_object(f'{var}.pkl', dir_data)
                assert values is not None
                for i, var2 in enumerate(foret_variables):
                    values2 = values[i]
                    save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var2}', values2, raster)
                    values2 = values2.flatten()
                    vec_x.append(values2[mask_X[:, 0]])

            elif var == 'air':
                assert values is not None
                for i, var2 in enumerate(air_variables):
                    values = read_object(f'{var2}raw.pkl', dir_data)
                    assert values is not None
                    values = values[:, :, allDates.index(sdate_year):allDates.index(edate_year)]
                    values = np.mean(values, axis=2)
                    save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var2}', values, raster)
                    values = values.flatten()
                    vec_x.append(values[mask_X[:, 0]])

            elif var == 'sentinel':
                values = read_object(f'{var}.pkl', dir_data)
                assert values is not None
                for i, var2 in enumerate(sentinel_variables):
                    values2 = np.mean(values[i, :, :, allDates.index(sdate_year):allDates.index(edate_year)], axis=2)
                    save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var2}', values2, raster)
                    values2 = values2.flatten()
                    vec_x.append(values2[mask_X[:, 0]])

            elif var == 'vigicrues':
                assert values is not None
                for i, var2 in enumerate(vigicrues_variables):
                    values = read_object(f'vigicrues{var2}.pkl', dir_data)
                    assert values is not None
                    values = values[:, :, allDates.index(sdate_year):allDates.index(edate_year)]
                    save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'vigicrues{var2}', values, raster)
                    values = np.mean(values, axis=2)
                    values = values.flatten()
                    vec_x.append(values[mask_X[:, 0]])

            elif var == 'nappes':
                for i, var2 in enumerate(nappes_variables):
                    values = read_object(f'{var2}.pkl', dir_data)
                    assert values is not None
                    values = values[:, :, allDates.index(sdate_year):allDates.index(edate_year)]
                    save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var2}', values, raster)
                    values = np.mean(values, axis=2)
                    values = values.flatten()
                    vec_x.append(values[mask_X[:, 0]])

            elif var == 'osmnx':
                values = read_object(f'osmnx.pkl', dir_data)
                assert values is not None
                for i, var2 in enumerate(osmnx_variables):
                    values2 = values[i]
                    save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var2}', values2, raster)
                    values2 = values2.flatten()
                    vec_x.append(values2[mask_X[:, 0]])

            elif var == 'dynamic_world':
                values = read_object(f'{var}.pkl', dir_data)
                assert values is not None
                for i, var2 in enumerate(dynamic_world_variables):
                    values2 = np.mean(values[i, :, :, allDates.index(sdate_year):allDates.index(edate_year)], axis=2)
                    save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var2}', values2, raster)
                    values2 = values2.flatten()
                    vec_x.append(values2[mask_X[:, 0]])
            else:
                raise ValueError(f'Unknow variable {var}')
            
        vec_x = np.asarray(vec_x)
        if 'X' not in locals():
            X = vec_x
        else:
            X = np.concatenate((X, vec_x), axis=0)

        if target_value is not None:
            return X, y
        return X
    
    def susecptibility_map_all_image(self, dept, year, variables, target_value, sdate_year, edate_year, raster, dir_data, dir_output):
        
        height = 64
        if target_value is not None:
            y = np.copy(target_value)
            y[np.isnan(y)] = 0
            y = resize_no_dim(y, height, height)

        vec_x = []
        
        assert raster is not None
        for var in variables:
            if var in cems_variables:
                values = read_object(f'{var}raw.pkl', dir_data)
                assert values is not None
                values = values[:, :, allDates.index(sdate_year):allDates.index(edate_year)]
                values = np.mean(values, axis=2)
                save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var}', values, raster)
                vec_x.append(values)

            elif var == 'population' or var == 'elevation':
                values = read_object(f'{var}.pkl', dir_data)
                assert values is not None
                values = values.reshape((values.shape[0], values.shape[1]))
                save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var}', values, raster)
                vec_x.append(values)

            elif var == 'foret':
                values = read_object(f'{var}.pkl', dir_data)
                assert values is not None
                for i, var2 in enumerate(foret_variables):
                    values2 = values[i]
                    save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{foretint2str[var2]}', values2, raster)
                    vec_x.append(values2)

            elif var == 'cosia':
                values = read_object(f'{var}.pkl', dir_data)
                assert values is not None
                for i, var2 in enumerate(cosia_variables):
                    values2 = values[i]
                    save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var2}', values2, raster)
                    vec_x.append(values2)

            elif var == 'air':
                assert values is not None
                for i, var2 in enumerate(air_variables):
                    values = read_object(f'{var2}raw.pkl', dir_data)
                    assert values is not None
                    values = values[:, :, allDates.index(sdate_year):allDates.index(edate_year)]
                    values = np.mean(values, axis=2)
                    save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var2}', values, raster)
                    vec_x.append(values)

            elif var == 'sentinel':
                values = read_object(f'{var}.pkl', dir_data)
                assert values is not None
                for i, var2 in enumerate(sentinel_variables):
                    values2 = np.mean(values[i, :, :, allDates.index(sdate_year):allDates.index(edate_year)], axis=2)
                    save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var2}', values2, raster)
                    vec_x.append(values2)

            elif var == 'vigicrues':
                assert values is not None
                for i, var2 in enumerate(vigicrues_variables):
                    values = read_object(f'vigicrues{var2}.pkl', dir_data)
                    assert values is not None
                    values = values[:, :, allDates.index(sdate_year):allDates.index(edate_year)]
                    values = np.mean(values, axis=2)
                    save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'vigicrues{var2}', values, raster)
                    vec_x.append(values)

            elif var == 'nappes':
                for i, var2 in enumerate(nappes_variables):
                    values = read_object(f'{var2}.pkl', dir_data)
                    assert values is not None
                    values = values[:, :, allDates.index(sdate_year):allDates.index(edate_year)]
                    values = np.mean(values, axis=2)
                    save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var2}', values, raster)
                    vec_x.append(values)

            elif var == 'osmnx':
                values = read_object(f'osmnx.pkl', dir_data)
                assert values is not None
                for i, var2 in enumerate(osmnx_variables):
                    values2 = values[i]
                    save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{osmnxint2str[var2]}', values2, raster)
                    vec_x.append(values2)

            elif var == 'dynamic_world':
                values = read_object(f'{var}.pkl', dir_data)
                assert values is not None
                for i, var2 in enumerate(dynamic_world_variables):
                    values2 = np.mean(values[i, :, :, allDates.index(sdate_year):allDates.index(edate_year)], axis=2)
                    save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var2}', values2, raster)
                    vec_x.append(values2)
            else:
                raise ValueError(f'Unknow variable {var}')
            
        vec_x = np.asarray(vec_x)
        vec_x[np.isnan(vec_x)] = 0
        vec_x_2 = np.zeros((vec_x.shape[0], height, height))
        for band in range(vec_x.shape[0]):
            vec_x_2[band] = resize_no_dim(vec_x[band], height, height)
        if 'X' not in locals():
            X = vec_x_2
        else:
            X = np.concatenate((X, vec_x_2), axis=0)

        if target_value is not None:
            return X, y
        else:
            return X

    def create_numpy_data(self, root_target):
        y_train = []
        y_test = []
        y_val = []
        dept_test = []
        years_test = []

        dir_target = root_target / 'log' / self.resolution
        dir_target_bin = root_target / 'bin' / self.resolution
        dir_raster = root_target / 'raster' / self.resolution

        if not (self.dir_log / 'susecptibility_map_features' / self.model_config['type'] / 'y_train.pkl').is_file():

            for dept in self.departements:
                logger.info(f'{dept}')

                dir_data = rootDisk / 'csv' / dept / 'raster' / self.resolution
                raster = read_object(f'{dept}rasterScale0.pkl', dir_raster)
                assert raster is not None
                raster = raster[0]

                for year in years:

                    if self.target == 'risk':
                        target_value = read_object(f'{dept}Influence.pkl', dir_target)
                    elif self.target == 'nbsinister':
                        target_value = read_object(f'{dept}binScale0.pkl', dir_target_bin)
                
                    assert target_value is not None

                    sdate_year = f'{year}-06-01'
                    edate_year = f'{year}-10-01'
                    
                    if allDates[0] > edate_year:
                            continue
                    if allDates[-1] < sdate_year:
                            continue
                    
                    if sdate_year < allDates[0]:
                        sdate_year = allDates[0]

                    if edate_year > allDates[-1]:
                        edate_year = allDates[-1]

                    if dept == 'departement-69-rhone' and int(year) > 2022:
                        continue

                    if edate_year < self.train_date:
                        set_type = 'train'
                    elif edate_year < self.val_date:
                        set_type = 'val'
                    else:
                        set_type = 'test'

                    if dept not in self.train_departements:
                        set_type = 'test'
                        
                    target_value = target_value[:, :, allDates.index(sdate_year):allDates.index(edate_year)]

                    check_and_create_path(self.dir_log / 'susecptibility_map_features' / str(year))

                    # Calculer la somme de target_value le long de la troisième dimension (axis=2)
                    sum_values = np.sum(target_value, axis=2)

                    save_feature_image(self.dir_log / 'susecptibility_map_features' / str(year), dept, f'{self.target}', sum_values, raster)

                    if self.model_config['type'] in sklearn_model_list:
                        X_, y_ = self.susecptibility_map_individual_pixel_feature(dept, year, self.features_name, sum_values, sdate_year, edate_year, raster, dir_data, self.dir_log)
                        if set_type == 'train':
                            y_train += list(np.asarray([y_]))
                            if 'X_train' not in locals():
                                X_train = X_
                            else:
                                X_train = np.concatenate((X_train, X_), axis=0)
                        elif set_type == 'val':
                            y_val += list(np.asarray([y_]))
                            if 'X_val' not in locals():
                                X_val = X_
                            else:
                                X_val = np.concatenate((X_val, X_), axis=0)
                        elif set_type == 'test':
                            y_test += list(np.asarray([y_]))
                            if 'X_test' not in locals():
                                X_test = X_
                            else:
                                X_test = np.concatenate((X_test, X_), axis=0)
                    else:
                        X_, y_ = self.susecptibility_map_all_image(dept, year, self.features_name, sum_values, sdate_year, edate_year, raster, dir_data, self.dir_log)
                        X_ = X_[np.newaxis, :, :, :]
                        if set_type == 'train':
                            y_train += list(np.asarray([y_]))
                            if 'X_train' not in locals():
                                X_train = X_
                            else:
                                X_train = np.concatenate((X_train, X_), axis=0)
                        elif set_type == 'val':
                            y_val += list(np.asarray([y_]))
                            if 'X_val' not in locals():
                                X_val = X_
                            else:
                                X_val = np.concatenate((X_val, X_), axis=0)
                        elif set_type == 'test':
                            dept_test.append(dept)
                            years_test.append(year)
                            y_test += list(np.asarray([y_]))
                            if 'X_test' not in locals():
                                X_test = X_
                            else:
                                X_test = np.concatenate((X_test, X_), axis=0)

            y_train = np.asarray(y_train)
            y_test = np.asarray(y_test)
            y_val = np.asarray(y_val)
            X_train = np.asarray(X_train)
            X_val = np.asarray(X_val)
            X_test = np.asarray(X_test)

            save_object(y_train, 'y_train.pkl', self.dir_log / 'susecptibility_map_features' / self.model_config['type'])
            save_object(y_test, 'y_test.pkl', self.dir_log  / 'susecptibility_map_features' / self.model_config['type'])
            save_object(y_val, 'y_val.pkl', self.dir_log  / 'susecptibility_map_features' / self.model_config['type'])
            save_object(X_train, 'X_train.pkl', self.dir_log  / 'susecptibility_map_features' / self.model_config['type'])
            save_object(X_val, 'X_val.pkl', self.dir_log  / 'susecptibility_map_features' / self.model_config['type'])
            save_object(X_test, 'X_test.pkl', self.dir_log  / 'susecptibility_map_features' / self.model_config['type'])
            save_object(dept_test, 'dept_test.pkl', self.dir_log  / 'susecptibility_map_features' / self.model_config['type'])
            save_object(years_test, 'years_test.pkl', self.dir_log  / 'susecptibility_map_features' / self.model_config['type'])

        else:

            base_path = self.dir_log  / 'susecptibility_map_features' / self.model_config['type']
            y_train = read_object('y_train.pkl', base_path)
            y_test = read_object('y_test.pkl', base_path)
            y_val = read_object('y_val.pkl', base_path)
            X_train = read_object('X_train.pkl', base_path)
            X_val = read_object('X_val.pkl', base_path)
            X_test = read_object('X_test.pkl', base_path)
            dept_test = read_object('dept_test.pkl', base_path)
            years_test = read_object('years_test.pkl', base_path)

        return X_train, y_train, X_val, y_val, X_test, y_test, dept_test, years_test
    
    def create_model_and_train(self, graph):
        params = self.model_config['params']
        
        X_train, y_train, X_val, y_val, X_test, y_test, dept_test, years_test = self.create_numpy_data()
        assert y_train is not None

        logger.info('################# Susceptibility map dataset #########################')
        logger.info(f'positive : {y_train[y_train > 0].shape}, zero : {y_train[y_train == 0].shape}')

        if self.model_config['type'] in sklearn_model_list:
            self.model = get_model(model_type=self.model_config['type'], name=self.model_config['name'],
                                device=self.model_config['device'], task_type=self.model_config['task'], params=self.model_config['params'], loss=self.model_config['loss'])
            
            self.model.fit(X_train,
                            y_train,
                            X_val,
                            y_val,
                            X_test,
                            y_test,
                            'normal',
                            'skip',
                            grid_params = {},
                            fit_params = {})

            score = self.model.score(X_test, y_test, None)
            logger.info(f'Score obtained in susecptibility mapping {score}')

        else:
            train_loader, val_loader, test_loader = self.create_train_val_test_loader(X_train, y_train, X_val, y_val, X_test, y_test, dept_test, years_test)

            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader

            check_and_create_path(self.dir_log / 'susecptibility_map_features' / self.model_config['type'])

            features = np.arange(0, X_train.shape[1])

            self.susceptility_mapper_name = self.model_config['type']
            self.model = ModelCNN(model_name=model,
                                    nbfeatures=len(self.features_name),
                                    batch_size=batch_size,
                                    lr=params['lr'],
                                    target_name=self.target,
                                    out_channels=params['out_channels'],
                                    features_name=features,
                                    ks=params['k_days'],
                                    dir_log=self.dir_log / Path(f'check_{params["scaling"]}/{params["prefix"]}/{model}_{params["infos"]}'),
                                    name=f'{model}_{params["infos"]}',
                                    task_type=task_type,
                                    loss=loss,
                                    device=device,
                                    under_sampling='full',
                                    over_sampling='full')
            
            self.model.train_loader = train_loader
            self.model.test_loader = test_loader
            self.model.val_loader = val_loader
            if (self.dir_log / 'susecptibility_map_features' / self.model_config['type'] / 'best.pt').is_file():
                temp_model = self.model.make_model(graph, custom_model_params=self.model_config['params'])
                self.model._load_model_from_path(self.dir_log / 'susecptibility_map_features' / self.dir_log['type'] / 'best.pt', temp_model)
            else:
                self.model.train(params, PATIENCE_CNT=params['PATIENCE_CNT'], CHECKPOINT=params['CHECKPOINT'], epochs=params['epoch'], custom_model_params=self.model_config['params'])
    
    def create_train_val_test_loader(self, X_train, y_train, X_val, y_val, X_test, y_test, dept_test, years_test):
         # Initialisation des nouveaux tableaux avec des zéros
        T, H, W = y_train.shape
        new_y_train = np.zeros((T, H, W, 9), dtype=y_train.dtype)

        T, H, W = y_val.shape
        new_y_val = np.zeros((T, H, W, 9), dtype=y_val.dtype)

        T, H, W = y_test.shape
        new_y_test = np.zeros((T, H, W, 9), dtype=y_test.dtype)

        # Copie des données du tableau de base dans les bandes -1 et -2
        new_y_train[..., -1] = y_train
        new_y_train[..., -2] = y_train

        new_y_val[..., -1] = y_val
        new_y_val[..., -2] = y_val

        new_y_test[..., -1] = y_test
        new_y_test[..., -2] = y_test

        # Mise de la bande -4 à 1
        new_y_train[..., weight_index] = 1
        new_y_val[..., weight_index] = 1
        new_y_test[..., weight_index] = 1

        logger.info(f'{X_train.shape}, {X_val.shape}, {X_test.shape}')
        logger.info(f'{new_y_train.shape}, {new_y_val.shape}, {new_y_test.shape}')

        self.susceptibility_scaler = StandardScaler()
        self.susceptibility_scaler.fit(X_train.reshape(-1, X_train.shape[1]))
        X_train = self.susceptibility_scaler.transform(X_train.reshape(-1, X_train.shape[1])).reshape(X_train.shape)
        X_val = self.susceptibility_scaler.transform(X_val.reshape(-1, X_train.shape[1])).reshape(X_val.shape)
        X_test = self.susceptibility_scaler.transform(X_test.reshape(-1, X_train.shape[1])).reshape(X_test.shape)

        logger.info(f'{np.max(X_train)} {np.max(X_val)} {np.max(X_test)}')

        data_augmentation_transform = RandomFlipRotateAndCrop(proba_flip=0.5, size_crop=32, max_angle=180)

        # Appliquez les transformations aux ensembles d'entraînement et de validation
        """train_dataset = AugmentedInplaceGraphDataset(
            X_train, y_train, [], transform=data_augmentation_transform, device=device)
        val_dataset = AugmentedInplaceGraphDataset(
            X_val, y_val, [], transform=data_augmentation_transform, device=device)
        test_dataset = AugmentedInplaceGraphDataset(
            X_test, y_test, [], transform=None, device=device)"""  # Pas de data augmentation sur le test
        
        train_dataset = InplaceGraphDataset(
            X_train, new_y_train, [], leni=len(X_train), device=device)
        val_dataset = InplaceGraphDataset(
            X_val, new_y_val, [], leni=len(X_val), device=device)
        test_dataset = InplaceGraphDataset(
            X_test, new_y_test, [], leni=len(X_test), device=device)

        train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=val_dataset.__len__(), shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

        return train_loader, val_loader, test_loader
    
    def predict(self, X):
        if self.model_config['type'] in sklearn_model_list:
            return self.model.predict(X)
        else:
            return self.model._predict_test_loader(X)
    
    def make_model(self, custom_model_params):
        model, params = make_model(self.model_name, len(self.features_name), len(self.features_name),
                                None, dropout, 'relu',
                                self.ks,
                                out_channels=self.out_channels,
                                task_type=self.task_type,
                                device=device, num_lstm_layers=num_lstm_layers,
                                custom_model_params=custom_model_params)
        
        if self.model_params is None:
            self.model_params = params
        return model, params