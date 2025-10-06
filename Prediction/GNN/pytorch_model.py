from numpy import dtype
import numpy as np
import random
from sympy import false
from torch_geometric.data import Dataset
from torch.utils.data import DataLoader
import torch
from torch import optim

torch.set_printoptions(precision=3, sci_mode=False)

from PIL import Image
import torchvision.transforms.functional as TF

from copy import deepcopy
import itertools
from matplotlib import pyplot as plt
from GNN.discretization import *
from GNN.tools import (
    calculate_area_under_curve,
    under_prediction_score,
    over_prediction_score,
    iou_score,
    evaluate_metrics,
    calculate_ic95,
)
from GNN.config import graph_id_index, departement_index
from sklearn.metrics import f1_score, jaccard_score

import dgl

from GNN.graph_builder import *
from GNN.tools import check_and_create_path, save_object, read_object

from tqdm import tqdm

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

g = torch.Generator()
g.manual_seed(42)

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

class ReadGraphDataset_2D_from_xarray(Dataset):
    def __init__(self, X : list,
                 Y : list,
                 edges : list,
                 leni : int,
                 device : torch.device,
                 path : Path,
                 features,
                 features_1D,
                 kdays,
                 scale,
                 graph_method,
                 base,
                 target_path
                 ):
                
        self.X = X
        self.Y = np.asarray(Y)
        self.device = device
        self.edges = edges
        self.leni = leni
        self.path = path
        self.features = features
        self.features_1D = features_1D
        self.kdays = kdays
        self.scale = scale
        self.graph_method = graph_method
        self.base = base
        self.target_path = target_path

        self.datacubes = {}
        self.areas = {}

        depts = np.unique(self.Y[:, departement_index, -1])
        
        for dept in depts:
            logger.info(f'Loading features of {dept}')
            mask = self.Y[:, departement_index, -1] == dept
            dates = np.unique(self.Y[mask, date_index, :])
            
            datacube = read_object(
                f'datacube.pkl',
                self.path / int2name[dept] / 'raster' / '2x2'
            )

            datacube_mask = read_object(f'datacube_target_{int2name[dept]}_{self.scale}_{self.base}_{self.graph_method}.pkl', self.target_path)

            dates_str = [allDates[int(date)] for date in dates]

            datacube = datacube.sel(date=dates_str)

            for var in ['precipitationIndexN5', 'precipitationIndexN3', 'precipitationIndexN9']:
                n = int(var[-1])
                array = calculate_precipitation_index_image_full(datacube['prec24h'].values, A=0.1657, n=n)
                datacube[var] = (('latitude', 'longitude', 'date'), array)

            # Ne conserver que les variables 2D demandées et existantes
            existing_vars = set(datacube.data_vars)
            wanted_vars = [v for v in self.features if v in existing_vars]
            #missing = [v for v in self.features if v not in existing_vars]
            #if missing:
            #    logger.warning(f"[{int2name[dept]}] Variables 2D absentes ignorées: {missing}")

            # Sous-échantillonnage du Dataset xarray aux seules features 2D
            if wanted_vars:
                datacube = datacube[wanted_vars]
            else:
                logger.warning(f"[{int2name[dept]}] Aucune feature_2D trouvée; datacube vide en variables.")

            self.datacubes[dept] = datacube
            self.areas[dept] = datacube_mask['area']

    def get_area_coords(self, area_dataarray, area_id):
        mask = (area_dataarray == area_id)
        lat_coords = area_dataarray.latitude.values
        lon_coords = area_dataarray.longitude.values
        positions = mask.values.nonzero()
        
        if len(positions[0]) == 0:
            raise ValueError(f"Aucune position trouvée pour area_id={area_id}")

        lat_indices = positions[1]
        lon_indices = positions[2]

        #print(area_dataarray.values.shape)
        #print(positions)

        lat_start_idx = lat_indices.min()
        lat_end_idx = lat_indices.max() + 1

        lon_start_idx = lon_indices.min()
        lon_end_idx = lon_indices.max() + 1

        lat_start = lat_coords[lat_start_idx]
        lat_end = lat_coords[lat_end_idx - 1]
        lon_start = lon_coords[lon_start_idx]
        lon_end = lon_coords[lon_end_idx - 1]

        return lat_start, lat_end, lon_start, lon_end

    def __getitem__(self, index) -> tuple:
        y = self.Y[index]

        dept = y[departement_index][-1]
        area_id = y[graph_id_index][-1]
        dates = y[date_index][-1]

        dates = dates.astype(int)

        datacube = self.datacubes[dept]

        lat_start, lat_end, lon_start, lon_end = self.get_area_coords(
            self.areas[dept], area_id
        )

        feature_cubes = []

        for feat_idx, feat in enumerate(self.features):
            # Si la feature est une variable externe
            if feat in calendar_variables or \
                feat == "id_encoder" or \
                feat == "cluster_encoder" or \
                feat == 'Past_burnedarea' or \
                feat == 'Past_risk' or \
                'calendar' in feat:
                # On récupère depuis self.X
                feat_idx = self.features_1D.index(feat)
                values = self.X[index][feat_idx]  # shape attendue : (kdays,)
                values = np.array(values).reshape(1, 1, self.kdays + 1)
                values = np.tile(values, (shape2D[self.scale][0], shape2D[self.scale][1], 1))  # (16, 16, kdays)
            else:
                if feat == 'foret_encoder':
                    feat = 'forest_landcover'
                elif feat == 'corine_encoder':
                    feat = 'corine_landcover'
                elif feat == 'bdroute_encoder':
                    feat = 'route_landcover'

                da = datacube[feat]
                if "date" in da.dims:
                    if self.kdays > 0:
                        selected = da.sel(
                            latitude=slice(lat_start, lat_end),
                            longitude=slice(lon_start, lon_end),
                            date=slice(allDates[dates - self.kdays], allDates[dates])
                        ).values
                    else:
                        selected = da.sel(
                            latitude=slice(lat_start, lat_end),
                            longitude=slice(lon_start, lon_end),
                            date=allDates[dates]
                        ).values
                        selected = selected[:, :, None]

                    #selected = selected.transpose(1, 2, 0)
                else:
                    selected = da.sel(
                        latitude=slice(lat_start, lat_end),
                        longitude=slice(lon_start, lon_end)
                    ).values  # shape: (h, w)
                    selected = selected[:, :, None].repeat(self.kdays + 1, axis=2)

                nan_mask = np.isnan(selected)
                if np.any(nan_mask):
                    mean_val = np.nanmean(selected)
                    selected[nan_mask] = mean_val

                H, W, T = selected.shape
                resized = np.zeros((shape2D[self.scale][0], shape2D[self.scale][1], T), dtype=selected.dtype)

                for t in range(T):
                    resized[:, :, t] = cv2.resize(selected[:, :, t], (shape2D[self.scale][0], shape2D[self.scale][1]), interpolation=cv2.INTER_LINEAR)

                selected = resized
                values = selected

            feature_cubes.append(values)

        feature_tensor = torch.tensor(
            np.stack(feature_cubes, axis=2), dtype=torch.float32, device=self.device
        )
        #print('features_tensort', feature_tensor.shape)
        if len(self.edges) > 0:
            edges = self.edges[index]
        else:
            edges = []

        return feature_tensor, \
            torch.tensor(y, dtype=torch.float32, device=self.device), \
            torch.tensor(edges, dtype=torch.long, device=self.device)

    def __len__(self) -> int:
        return self.leni
    
    def len(self):
        pass

    def get(self):
        pass

class ReadGraphDataset2DOptim(Dataset):
    def __init__(self, X, Y, edges, leni, device, path, features, features_1D,
                 kdays, scale, graph_method, base, target_path):

        self.X = X
        self.Y = np.asarray(Y)
        self.device = device
        self.edges = edges
        self.leni = leni
        self.path = path
        self.features = features
        self.features_1D = features_1D
        self.kdays = kdays
        self.scale = scale
        self.graph_method = graph_method
        self.base = base
        self.target_path = target_path

        self.datacubes = {}
        self.areas = {}
        self.area_coords = {}   # cache des coordonnées

        # Map pour éviter .index() dans __getitem__
        self.features_1D_map = {feat: idx for idx, feat in enumerate(self.features_1D)}

        # Préparer uniquement le cache de zones (faible mémoire)
        depts = np.unique(self.Y[:, departement_index, -1])
        for dept in depts:
            logger.info(f'Loading features from {dept}')
            datacube = read_object(
                f'datacube.pkl',
                self.path / int2name[dept] / 'raster' / '2x2'
            )

            datacube_mask = read_object(
                f'datacube_target_{int2name[dept]}_{self.scale}_{self.base}_{self.graph_method}.pkl',
                self.target_path
            )

            self.datacubes[dept] = datacube  # pas converti en numpy
            self.areas[dept] = datacube_mask['area']

            self.area_coords[dept] = {}
            unique_ids = np.unique(self.areas[dept].values)
            for area_id in unique_ids:
                if np.isnan(area_id):
                    continue
                self.area_coords[dept][area_id] = self._compute_coords(self.areas[dept], area_id)

    def _compute_coords(self, area_dataarray, area_id):
        mask = (area_dataarray == area_id).values
        pos = np.nonzero(mask)
        lat_coords = area_dataarray.latitude.values
        lon_coords = area_dataarray.longitude.values

        lat_start_idx, lat_end_idx = pos[1].min(), pos[1].max() + 1
        lon_start_idx, lon_end_idx = pos[2].min(), pos[2].max() + 1

        return (lat_coords[lat_start_idx], lat_coords[lat_end_idx - 1],
                lon_coords[lon_start_idx], lon_coords[lon_end_idx - 1])

    def _resize_3d(self, arr, out_h, out_w):
        """Redimensionner un tableau (H, W, T) sans boucle Python."""
        t = arr.shape[2]
        resized = np.empty((out_h, out_w, t), dtype=arr.dtype)
        for i in range(t):
            resized[:, :, i] = cv2.resize(arr[:, :, i], (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return resized

    def __getitem__(self, index):
        y = self.Y[index]

        dept = y[departement_index][-1]
        area_id = y[graph_id_index][-1]
        date_int = int(y[date_index][-1])

        datacube = self.datacubes[dept]

        lat_start, lat_end, lon_start, lon_end = self.area_coords[dept][area_id]

        feature_cubes = []
        for feat in self.features:
            # Variables 1D -> répétition en 2D
            if feat in calendar_variables or feat in {"id_encoder", "cluster_encoder", "Past_burnedarea", "Past_risk"}:
                feat_idx = self.features_1D_map[feat]
                values = np.array(self.X[index][feat_idx]).reshape(1, 1, self.kdays + 1)
                values = np.tile(values, (shape2D[self.scale][0], shape2D[self.scale][1], 1))
            else:
                mapped_feat = {
                    'foret_encoder': 'forest_landcover',
                    'corine_encoder': 'corine_landcover',
                    'bdroute_encoder': 'route_landcover'
                }.get(feat, feat)

                da = datacube[mapped_feat]
                if "date" in da.dims:
                    if self.kdays > 0:
                        selected = da.sel(
                            latitude=slice(lat_start, lat_end),
                            longitude=slice(lon_start, lon_end),
                            date=slice(allDates[date_int - self.kdays], allDates[date_int])
                        ).values
                    else:
                        selected = da.sel(
                            latitude=slice(lat_start, lat_end),
                            longitude=slice(lon_start, lon_end),
                            date=allDates[date_int]
                        ).values
                        selected = selected[:, :, None]
                else:
                    selected = da.sel(
                        latitude=slice(lat_start, lat_end),
                        longitude=slice(lon_start, lon_end)
                    ).values[:, :, None].repeat(self.kdays + 1, axis=2)

                if np.any(np.isnan(selected)):
                    selected = np.nan_to_num(selected, nan=np.nanmean(selected))

                selected = self._resize_3d(selected, shape2D[self.scale][0], shape2D[self.scale][1])
                values = selected

            feature_cubes.append(values)

        feature_tensor = torch.tensor(np.stack(feature_cubes, axis=2), dtype=torch.float32, device=self.device)
        edges_tensor = torch.tensor(self.edges[index], dtype=torch.long, device=self.device) if len(self.edges) > 0 else []

        return feature_tensor, torch.tensor(y, dtype=torch.float32, device=self.device), edges_tensor

    def __len__(self):
        return self.leni

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

class InplaceMeshGraphDatasetInplace(Dataset):
    def __init__(self, X : list, Y : list, edges : list, leni : int, device : torch.device, graph_mesh, gridh2mesh, mesh2graph) -> None:
        self.X = X
        self.Y = Y
        self.device = device
        self.edges = edges
        self.leni = leni
            
        self.graph_mesh = graph_mesh
        self.grid2mesh = gridh2mesh
        self.mesh2grid = mesh2graph

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

        return X, Y, E, self.graph_mesh, self.grid2mesh, self.mesh2grid

    def __len__(self) -> int:
        return self.leni
    
    def len(self):
        pass

    def get(self):
        pass

class InplaceMulitpleGraphDataset(Dataset):
    def __init__(self, target_name : str, list_graph_file : list, X : list, Y : list, edges : list, leni : int, device : torch.device) -> None:
        self.X = X
        self.Y = Y
        
        self.device = device
        self.edges = edges
        self.leni = leni
        self.list_graph = [read_object(f, p) for (f, p) in list_graph_file]
        self.other_scale = [graph.scale for graph in self.list_graph]
        self.Y_other_graph = []
        
        for i, graph in enumerate(self.list_graph):
            if i == 0:
                continue
            p = list_graph_file[i][1]
            df_train_scale = read_object(f'df_train_full_{graph.scale}_0_{graph.base}_{graph.graph_method}.pkl', p / 'occurence_voting')
            df_val_scale = read_object(f'df_val_full_{graph.scale}_0_{graph.base}_{graph.graph_method}.pkl', p / 'occurence_voting')
            df_test_scale = read_object(f'df_test_full_{graph.scale}_0_{graph.base}_{graph.graph_method}.pkl', p / 'occurence_voting')

            df_scale = pd.concat((df_train_scale, df_val_scale, df_test_scale)).reset_index(drop=True)

            if df_scale not in np.unique(df_scale.columns):
                df_scale['scale'] = graph.scale
                if graph.scale == 'departement':
                    df_scale['scale'] = 10

            Y_scale = df_scale[ids_columns + targets_columns + [target_name]].values
            Y_scale = np.asarray(Y_scale, dtype=np.float32)
            self.Y_other_graph.append(Y_scale)

    def __getitem__(self, index) -> tuple:
        x = self.X[index]
        y = self.Y[index]

        date_Y = np.unique(y[:, date_index, -1])

        Y = [y]

        for i, y1 in enumerate(self.Y_other_graph):
            mask = np.isin(self.Y_other_graph[i][:, date_index], date_Y)
            new_Y = self.Y_other_graph[i][mask]
            new_Y = new_Y[:, :, np.newaxis]
            Y.append(new_Y)

        if len(self.edges) > 0:
            edges = self.edges[index]
        else:
            edges = []
        
        X, Y, E = torch.tensor(x, dtype=torch.float32, device=self.device), \
            Y, \
            torch.tensor(edges, dtype=torch.long, device=self.device),  \
            
        return X, Y, self.list_graph

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
    graph = dgl.graph((edge_index[0], edge_index[1]))

    return node_features, node_labels, graph, graph_labels_list

def match_indices(pos, sample, atol=1e-6):
    # Utilise un dtype stable
    pos = pos.to(dtype=torch.float64)
    sample = sample.to(dtype=torch.float64)

    D = torch.cdist(sample, pos)            # [M, N]
    close = D <= atol                       # [M, N] True si sample m est proche de pos n
    valid_pos = close.any(dim=0)            # masque côté POS (N)
    pos_idx_valid = torch.where(valid_pos)[0]
    #j = 7
    #print('shape D =', D.shape)                 # doit être [M, N]
    #print('col7 min =', D[:, j].min().item())
    #print('any(col7 <= atol) =', bool((D[:, j] <= atol).any()))

    valid_pos = (D <= atol).any(dim=0)          # masque côté POS (N)
    #print('valid_pos[7] =', bool(valid_pos[j]))

    # indices pos valides:
    idx_pos = torch.where(valid_pos)[0]
    #print('index valid (pos) =', idx_pos)

    return pos_idx_valid, valid_pos

def mesh_subgraph_from_graphbuilder(mesh_graph, mesh_from_g2m, mesh_from_m2g,
                                    how="intersection"):
    """
    Crée un sous-graphe de mesh_graph en utilisant les nœuds mesh impliqués
    dans g2m_graph et m2g_graph.
    
    - how: 'intersection' (par défaut) ou 'union' entre les deux bipartites.
    """
    mesh_from_g2m = torch.unique(mesh_from_g2m)
    mesh_from_m2g = torch.unique(mesh_from_m2g)

    if how == "union":
        mesh_ids = torch.unique(torch.cat([mesh_from_g2m, mesh_from_m2g], dim=0))
    elif how == "intersection":
        mesh_ids = torch.tensor(
            np.intersect1d(mesh_from_g2m.cpu().numpy(), mesh_from_m2g.cpu().numpy()),
            device=mesh_from_g2m.device, dtype=mesh_from_g2m.dtype
        )
    else:
        raise ValueError("how must be 'union' or 'intersection'")

    # Sous-graphe induit par ces nœuds mesh
    mesh_subg = dgl.node_subgraph(mesh_graph, mesh_ids, relabel_nodes=True)
    return mesh_subg

def select_sample_graph(cartesian_grid, graph, graph_type, atol=1e-6, shrink_mesh=True):
    """
    Sous-graphe hétéro gardant uniquement les nœuds 'grid' correspondant à cartesian_grid
    (match par position) et, optionnellement, les nœuds 'mesh' connectés à ces 'grid'.

    Returns
    -------
    subg : dgl.DGLHeteroGraph
        Le sous-graphe sélectionné.
    valid_sample_idx : torch.Tensor (1D, dtype=long, device=cartesian_grid.device)
        Indices (dans cartesian_grid) qui ont été validés (présents dans le sous-graphe),
        dans l'ordre d'apparition de cartesian_grid (sans doublons).
    """
    if graph_type not in {"g2m", "m2g"}:
        raise ValueError("graph_type must be 'g2m' or 'm2g'")

    idtype = graph.idtype  # ex: torch.int32
    device = graph.device  # même device que le graphe

    # 1) positions des nœuds GRID selon le type de graphe
    if graph_type == "g2m":
        pos_grid = graph.srcdata["pos"]            # grid positions
        etype = ("grid", "g2m", "mesh")
    else:  # "m2g"
        pos_grid = graph.dstdata["pos"]            # grid positions
        etype = ("mesh", "m2g", "grid")
    
    sample = cartesian_grid.to(dtype=pos_grid.dtype, device=pos_grid.device)
    
    #print('Original graph', pos_grid)

    # 2) IDs locaux des nœuds GRID qui matchent 'sample'
    #    match_indices doit renvoyer (idx_pos_grid, idx_sample) alignés
    grid_ids, sample_ids = match_indices(pos_grid, sample, atol=atol)

    #print('valid graph grid', pos_grid[grid_ids])

    if grid_ids.numel() == 0:
        empty = torch.tensor([], dtype=idtype, device=device)
        subg = dgl.node_subgraph(graph, {'grid': empty, 'mesh': empty})
        return subg

    grid_ids = grid_ids.to(dtype=idtype, device=device)

    # 3) IDs des nœuds MESH à garder
    u, v = graph.edges(etype=etype)
    if graph_type == "g2m":
        # u: grid, v: mesh
        mask_e = torch.isin(u, grid_ids)
        mesh_ids = torch.unique(v[mask_e])
    else:
        # u: mesh, v: grid
        mask_e = torch.isin(v, grid_ids)
        mesh_ids = torch.unique(u[mask_e])

    if not shrink_mesh:
        mesh_ids = torch.arange(graph.num_nodes('mesh'), device=device, dtype=idtype)
    else:
        mesh_ids = mesh_ids.to(dtype=idtype, device=device)

    # 4) Sous-graphe hétéro
    subg = dgl.node_subgraph(graph, {
        'grid': grid_ids,
        'mesh': mesh_ids,
    },
    relabel_nodes=True)

    #print('subgraph', subg.srcdata["pos"])

    return subg, mesh_ids

def graph_collate_fn_mesh(batch):
    node_features_list = []
    node_labels_list = []
    graph_labels_list = []

    graph_list = []
    graph_mesh_list = []
    grid2mesh_list = []
    mesh2grid_list = []

    num_nodes_seen = 0

    for graph_id, features_labels_graph_index_tuple in enumerate(batch):
        # Collecter les caractéristiques et les étiquettes des nœuds

        graph_mesh_ = features_labels_graph_index_tuple[3]
        gridh2mesh_ = features_labels_graph_index_tuple[4]
        mesh2graph_ = features_labels_graph_index_tuple[5]

        latitude_batch = features_labels_graph_index_tuple[1][:, latitude_index, -1].reshape(-1,1)
        longitude_batch = features_labels_graph_index_tuple[1][:, longitude_index, -1].reshape(-1,1)
        departement_batch = features_labels_graph_index_tuple[1][:, departement_index, -1].reshape(-1,)

        g_lat_lon_grid = torch.concat((latitude_batch, longitude_batch), dim=1).to('cpu')
        #print('Departement', torch.unique(departement_batch))
        cartesian_grid = latlon_points_to_xyz(g_lat_lon_grid.view(-1,2))
        #print('Original cartesian grid', cartesian_grid)
        ## Select nodes in grid2mesh, graph_mesh, and mesh2graph
        gridh2mesh, mesh_ids_g2m = select_sample_graph(cartesian_grid, gridh2mesh_, 'g2m')
        mesh2graph, mesh_ids_mg2 = select_sample_graph(cartesian_grid, mesh2graph_, 'm2g')

        #mesh2graph, gridh2mesh = restrict_mesh2graph_to_gridh2mesh(mesh2graph, gridh2mesh)

        def _check_non_empty_edges(g, etype, name):
                e = g.num_edges(etype)
                if e == 0:
                    print(f"[WARN] {name}: 0 edges pour etype {etype}.")
                return e

        _check_non_empty_edges(gridh2mesh, ("grid","g2m","mesh"), "gridh2mesh")
        _check_non_empty_edges(mesh2graph, ("mesh","m2g","grid"), "mesh2graph")

        graph_mesh = mesh_subgraph_from_graphbuilder(graph_mesh_, mesh_ids_g2m, mesh_ids_mg2, how="union")
        
        num_nodes = features_labels_graph_index_tuple[1].size(0)
        num_nodes_seen += num_nodes  # Mettre à jour le nombre de nœuds vus
        graph_labels_list.append(torch.full((num_nodes,), graph_id, dtype=torch.long))  # Création d'un tensor d'IDs

        #print("#############################################################################")

        #print(gridh2mesh_)
        #print(mesh2graph_)
        #print(graph_mesh)
        #exit(1)

        #print(torch.unique(features_labels_graph_index_tuple[1][:, departement_index, -1]))

        node_features_list.append(features_labels_graph_index_tuple[0])
        node_labels_list.append(features_labels_graph_index_tuple[1])

        #print(gridh2mesh)
        #print(mesh2graph)

        #print(torch.unique(features_labels_graph_index_tuple[1][:, departement_index, -1]))

        graph_mesh_list.append(graph_mesh)
        grid2mesh_list.append(gridh2mesh)
        mesh2grid_list.append(mesh2graph)
        
    # Merge the PPI graphs into a single graph with multiple connected components
    node_features = torch.cat(node_features_list, 0)
    node_labels = torch.cat(node_labels_list, 0)

    for g in graph_mesh_list:  # graphs est une liste de dgl.graph
        if '_ID' not in g.edata:
            g.edata['_ID'] = torch.arange(g.num_edges(), dtype=torch.int32)
        if '_ID' not in g.ndata:
            g.ndata['_ID'] = torch.arange(g.num_nodes(), dtype=torch.int32)

    graph_list.append(dgl.batch(graph_mesh_list))
    graph_list.append(dgl.batch(grid2mesh_list))
    graph_list.append(dgl.batch(mesh2grid_list))
    
    graph_labels_list = torch.cat(graph_labels_list, 0).to(device)
    return node_features, node_labels, graph_list, graph_labels_list

"""def graph_collate_fn_mesh(batch):
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
        
        #graph_mesh = features_labels_graph_index_tuple[4]
        #gridh2mesh = features_labels_graph_index_tuple[5]
        #mesh2graph = features_labels_graph_index_tuple[6]

        num_nodes = features_labels_graph_index_tuple[1].size(0)
        num_nodes_seen += num_nodes  # Mettre à jour le nombre de nœuds vus
        graph_labels_list.append(torch.full((num_nodes,), graph_id, dtype=torch.long))  # Création d'un tensor d'IDs

        latitudes = features_labels_graph_index_tuple[1][:, latitude_index, -1].reshape(-1,1)
        longitudes = features_labels_graph_index_tuple[1][:, longitude_index, -1].reshape(-1,1)

        g_lat_lon_grid = torch.concat((latitudes, longitudes), dim=1).to('cpu')
        g_lat_lon_grid = torch.unique(g_lat_lon_grid, dim=0)
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

    for g in graph_mesh_list:  # graphs est une liste de dgl.graph
        if '_ID' not in g.edata:
            g.edata['_ID'] = torch.arange(g.num_edges(), dtype=torch.int32)
        if '_ID' not in g.ndata:
            g.ndata['_ID'] = torch.arange(g.num_nodes(), dtype=torch.int32)

    graph_list.append(dgl.batch(graph_mesh_list))
    graph_list.append(dgl.batch(grid2mesh_list))
    graph_list.append(dgl.batch(mesh2grid_list))
    
    graph_labels_list = torch.cat(graph_labels_list, 0).to(device)
    return node_features, node_labels, graph_list, graph_labels_list"""

def graph_collate_fn_multiple_graph(batch):
    #node_indice_list = []
    node_features_list = []
    node_labels_list = []
    graph_labels_list = []  

    graph_list = []
    graph_scale_list = []
    decrease_scale = []
    increase_scale = []

    for graph_id, features_labels_graph_index_tuple in enumerate(batch):

        g_lat_lon_grid_scales = []
        for labels in features_labels_graph_index_tuple[1]:
            labels = np.asarray(labels, dtype=np.float32)
            if labels.ndim == 3:
                latitudes = torch.Tensor(labels[:, latitude_index, -1])
                longitudes = torch.Tensor(labels[:, longitude_index, -1])
            else:
                latitudes = torch.Tensor(labels[:, latitude_index])
                longitudes = torch.Tensor(labels[:, longitude_index])

            if latitudes.ndim == 1:
                latitudes = latitudes[:, None]
                longitudes = longitudes[:, None]

            g_lat_lon_grid = torch.concat((latitudes, longitudes), dim=1).to('cpu')
            #g_lat_lon_grid = torch.unique(g_lat_lon_grid, dim=0)
            g_lat_lon_grid_scales.append(g_lat_lon_grid)

        graph_builder = GraphBuilder2(g_lat_lon_grid_scales, features_labels_graph_index_tuple[1], features_labels_graph_index_tuple[2], date_index, id_index)

        current_graph_scale_list = graph_builder.graph_scale()
        increase_graph_scale_list, current_graph_scale_list = graph_builder.increase_scale(current_graph_scale_list)
        if increase_graph_scale_list is None:
            continue
        decrease_graph_scale_list = graph_builder.decrease_scale()
        
        for i, labels in enumerate(features_labels_graph_index_tuple[1]):
            node_graph = current_graph_scale_list[i]
            original_node_indices = node_graph.ndata[dgl.NID]

            if i == 0:
                node_features = features_labels_graph_index_tuple[0][original_node_indices]
                node_features_list.append(node_features)      
                labels = labels[original_node_indices]    

            if 'mask_zeros' in locals():
                if i > 0:
                    increase_graph = increase_graph_scale_list[i - 1]
                    decrease_graph = decrease_graph_scale_list[i - 1]
                    #print('###########################')
                    #print(increase_graph)

                    #print({ntype: increase_graph.nodes(ntype) for ntype in increase_graph.ntypes})

                    #increase_graph = dgl.node_subgraph(increase_graph, nodes={ntype: increase_graph.nodes(ntype) for ntype in increase_graph.ntypes})
                    #decrease_graph = dgl.node_subgraph(decrease_graph, nodes={ntype: decrease_graph.nodes(ntype) for ntype in decrease_graph.ntypes})

                    increase_graph_scale_list[i - 1] = increase_graph
                    decrease_graph_scale_list[i - 1] = decrease_graph

                    num_nodes = current_graph_scale_list[i -1].num_nodes()
                    labels = labels[original_node_indices]

                    if labels.ndim == 2:
                        labels = labels[None, :, :]

                    edge_type = list(increase_graph.canonical_etypes)[0]
                    src_nodes, dst_nodes = increase_graph.edges(etype=edge_type)

                    # Masque destination : destination avec -1,-1 == 0

                    dst_mask = (labels[dst_nodes, -1, -1] == 0)
                    #print(increase_graph)
                    #print(np.unique(src_nodes))
                    valid_src_mask = mask_zeros[src_nodes]

                    # Final: on garde les dst liés à des src masqués et eux-mêmes à 0
                    final_mask = valid_src_mask & dst_mask

                    labels[dst_nodes[final_mask], weight_index] = 0
            
            if labels.shape[2] > 1:
                labels = labels[:, :, -1]
            mask_zeros = (labels[:, weight_index, -1] == 0)

            labels = torch.Tensor(labels)
            node_labels_list.append(labels)

        #graph_scale_list = dgl.batch(graph_scale_list)
        #increase_graph_scale_list = dgl.batch(increase_graph_scale_list)
        #decrease_graph_scale_list = dgl.batch(decrease_graph_scale_list)

        graph_scale_list.append(current_graph_scale_list)
        increase_scale.append(increase_graph_scale_list)
        decrease_scale.append(decrease_graph_scale_list)

    # Merge the PPI graphs into a single graph with multiple connected components
    node_features = torch.cat(node_features_list, 0)
    node_labels = torch.cat(node_labels_list, 0).to(node_features.device)

    """for g in graph_mesh_list:  # graphs est une liste de dgl.graph
        if '_ID' not in g.edata:
            g.edata['_ID'] = torch.arange(g.num_edges(), dtype=torch.int32)
        if '_ID' not in g.ndata:
            g.ndata['_ID'] = torch.arange(g.num_nodes(), dtype=torch.int32)"""

    # Suppose graph_list, increase_scale, and decrease_scale are lists of lists of DGLGraphs
    
    batched_graph_list = [dgl.batch(graphs_at_i) for graphs_at_i in zip(*graph_scale_list)]
    batched_increase = [dgl.batch(graphs_at_i) for graphs_at_i in zip(*increase_scale)]
    batched_decrease = [dgl.batch(graphs_at_i) for graphs_at_i in zip(*decrease_scale)]

    graph_list.append(batched_graph_list)
    graph_list.append(batched_increase)
    graph_list.append(batched_decrease)
    
    #graph_labels_list = torch.cat(graph_labels_list, 0).to(device)
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

def construct_dataset(date_ids, x_data, y_data, graph, ids_columns, ks, use_temporal_as_edges, isNotmesh=False):
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
            if x is not None and isNotmesh:
                for i in range(x.shape[0]):
                    Xs.append(x[i])
                    Ys.append(y[i])
            elif x is not None:
                Xs.append(x)
                Ys.append(y)
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
                    graph_mesh=None,
                    gridh2mesh=None,
                    mesh2graph=None
                    ):
    
    x_train, y_train = df_train[ids_columns + features_name].values, df_train[ids_columns + targets_columns + [target_name]].values
    
    x_val, y_val = df_val[ids_columns + features_name].values, df_val[ids_columns + targets_columns + [target_name]].values

    x_test, y_test = df_test[ids_columns + features_name].values, df_test[ids_columns + targets_columns + [target_name]].values

    dateTrain = np.sort(np.unique(y_train[y_train[:, weight_index] > 0, date_index]))
    dateVal = np.sort(np.unique(y_val[y_val[:, weight_index] > 0, date_index]))
    dateTest = np.sort(np.unique(y_test[y_test[:, weight_index] > 0, date_index]))

    logger.info(f'{dateTrain.shape}, {dateVal.shape}, {dateTest.shape}')

    logger.info(f'Constructing train Dataset')
    Xst, Yst, Est = construct_dataset(dateTrain, x_train, y_train, graph, ids_columns, ks, use_temporal_as_edges, graph_mesh is None)

    logger.info(f'Constructing val Dataset')
    XsV, YsV, EsV = construct_dataset(dateVal, x_val, y_val, graph, ids_columns, ks, use_temporal_as_edges, graph_mesh is None)

    logger.info(f'Constructing test Dataset')
    XsTe, YsTe, EsTe = construct_dataset(dateTest, x_test, y_test, graph, ids_columns, ks, use_temporal_as_edges, graph_mesh is None)

    # Assurez-vous que les ensembles ne sont pas vides
    assert len(Xst) > 0, "Le jeu de données d'entraînement est vide"
    assert len(XsV) > 0, "Le jeu de données de validation est vide"
    assert len(XsTe) > 0, "Le jeu de données de test est vide"

    if graph_mesh is None:
        # Création des datasets finaux
        print('uzbdkazdkjzan')
        train_dataset = InplaceGraphDataset(Xst, Yst, Est, len(Xst), device)
        val_dataset = InplaceGraphDataset(XsV, YsV, EsV, len(XsV), device)
        test_dataset = InplaceGraphDataset(XsTe, YsTe, EsTe, len(XsTe), device)
    elif graph_mesh is not None:
        train_dataset = InplaceMeshGraphDatasetInplace(Xst, Yst, Est, len(Xst), device, graph_mesh, gridh2mesh, mesh2graph)
        val_dataset = InplaceMeshGraphDatasetInplace(XsV, YsV, EsV, len(XsV), device, graph_mesh, gridh2mesh, mesh2graph)
        test_dataset = InplaceMeshGraphDatasetInplace(XsTe, YsTe, EsTe, len(XsTe), device, graph_mesh, gridh2mesh, mesh2graph)
    #elif mesh == 'mygraph':
    #    train_dataset = InplaceMulitpleGraphDataset(target_name, mesh_file, Xst, Yst, Est, len(Xst), device)
    #    val_dataset = InplaceMulitpleGraphDataset(target_name, mesh_file, XsV, YsV, EsV, len(XsV), device)
    #    test_dataset = InplaceMulitpleGraphDataset(target_name, mesh_file, XsTe, YsTe, EsTe, len(XsTe), device)

    return train_dataset, val_dataset, test_dataset

def create_train_dataset(graph,
                    df_train,
                    features_name,
                    target_name,
                    use_temporal_as_edges : bool,
                    device,
                    ks : int,
                    graph_mesh=None,
                    gridh2mesh=None,
                    mesh2graph=None):

    x_train, y_train = df_train[ids_columns + features_name].values, df_train[ids_columns + targets_columns + [target_name]].values
    
    print('weight', df_train['weight'].unique())

    dateTrain = np.sort(np.unique(y_train[y_train[:, weight_index] > 0, date_index]))

    logger.info(f'{dateTrain.shape}')

    logger.info(f'Constructing train Dataset')
    Xst, Yst, Est = construct_dataset(dateTrain, x_train, y_train, graph, ids_columns, ks, use_temporal_as_edges, graph_mesh is None)

    # Assurez-vous que les ensembles ne sont pas vides
    assert len(Xst) > 0, "Le jeu de données d'entraînement est vide"

    if graph_mesh is None:
        # Création des datasets finaux
        print('uzbdkazdkjzan')
        train_dataset = InplaceGraphDataset(Xst, Yst, Est, len(Xst), device)
    elif graph_mesh is not None:
        train_dataset = InplaceMeshGraphDatasetInplace(Xst, Yst, Est, len(Xst), device, graph_mesh, gridh2mesh, mesh2graph)
    #elif mesh == 'mygraph':
    #    train_dataset = InplaceMulitpleGraphDataset(target_name, mesh_file, Xst, Yst, Est, len(Xst), device)

    return train_dataset

def create_test_val_dataset(graph,
                    df_val,
                    df_test,
                    features_name,
                    target_name,
                    use_temporal_as_edges : bool,
                    device,
                    ks : int,
                    graph_mesh=None,
                    gridh2mesh=None,
                    mesh2graph=None):
        
    x_val, y_val = df_val[ids_columns + features_name].values, df_val[ids_columns + targets_columns + [target_name]].values

    x_test, y_test = df_test[ids_columns + features_name].values, df_test[ids_columns + targets_columns + [target_name]].values

    dateVal = np.sort(np.unique(y_val[y_val[:, weight_index] > 0, date_index]))
    dateTest = np.sort(np.unique(y_test[y_test[:, weight_index] > 0, date_index]))

    logger.info(f'{dateVal.shape}, {dateTest.shape}')

    logger.info(f'Constructing val Dataset')
    XsV, YsV, EsV = construct_dataset(dateVal, x_val, y_val, graph, ids_columns, ks, use_temporal_as_edges, graph_mesh is None)

    logger.info(f'Constructing test Dataset')
    XsTe, YsTe, EsTe = construct_dataset(dateTest, x_test, y_test, graph, ids_columns, ks, use_temporal_as_edges, graph_mesh is None)

    # Assurez-vous que les ensembles ne sont pas vides
    assert len(XsV) > 0, "Le jeu de données de validation est vide"
    assert len(XsTe) > 0, "Le jeu de données de test est vide"

    if gridh2mesh is None:
        # Création des datasets finaux
        print('uzbdkazdkjzan')
        val_dataset = InplaceGraphDataset(XsV, YsV, EsV, len(XsV), device)
        test_dataset = InplaceGraphDataset(XsTe, YsTe, EsTe, len(XsTe), device)
    elif gridh2mesh is not None:
        val_dataset = InplaceMeshGraphDatasetInplace(XsV, YsV, EsV, len(XsV), device, graph_mesh, gridh2mesh, mesh2graph)
        test_dataset = InplaceMeshGraphDatasetInplace(XsTe, YsTe, EsTe, len(XsTe), device, graph_mesh, gridh2mesh, mesh2graph)
    #elif mesh == 'mygraph':
    #    val_dataset = InplaceMulitpleGraphDataset(target_name, mesh_file, XsV, YsV, EsV, len(XsV), device)
    #    test_dataset = InplaceMulitpleGraphDataset(target_name, mesh_file, XsTe, YsTe, EsTe, len(XsTe), device)

    return val_dataset, test_dataset

def get_numpy_data(graph, df,
                       features_name,
                       use_temporal_as_edges : bool,
                       ks :int):

    Xset = df[ids_columns + features_name].values

    X = []
    E = []
    Yset = None
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
            x, _ = construct_time_series(date, Xset, Yset, ks, len(ids_columns))
            if x is not None:
                for i in range(x.shape[0]):
                    X.append(x[i])
            continue
        elif use_temporal_as_edges:
            x, _, e = construct_graph_set(graph, date, Xset, Yset, ks, len(ids_columns))
        else:
            x, _, e = construct_graph_with_time_series(graph, date, Xset, Yset, ks, len(ids_columns))

        if x is None:
            continue

        if x.shape[0] == 0:
            continue

        X.append(x)
        if 'e' in locals():
            E.append(e)

    return np.asarray(X), np.asarray(E)

def create_test_loader(graph, df,
                       features_name,
                       device : torch.device,
                       use_temporal_as_edges : bool,
                       target_name,
                       ks :int,
                       graph_mesh=None,
                        gridh2mesh=None,
                        mesh2graph=None):
    
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

    if gridh2mesh is None:
        dataset = InplaceGraphDataset(X, Y, E, len(X), device)
        collate = graph_collate_fn
    elif gridh2mesh is not None:
        dataset = InplaceMeshGraphDatasetInplace(X, Y, E, len(X), device, graph_mesh, gridh2mesh, mesh2graph)
        collate = graph_collate_fn_mesh
    #elif mesh == 'mygraph':
    #    dataset = InplaceMulitpleGraphDataset(target_name, mesh_file, X, Y, E, len(X), device)
    #    collate = graph_collate_fn_multiple_graph
    else:
        raise ValueError(f'{mesh} is not a known mesh value')

    if use_temporal_as_edges is None:
        loader = DataLoader(dataset, dataset.__len__(), False, worker_init_fn=seed_worker,
            generator=g)
    else:
        loader = DataLoader(dataset, dataset.__len__(), False, collate_fn=collate,
                            worker_init_fn=seed_worker,
        generator=g)

    return loader

def load_x_from_pickle(date : int,
                       path : Path,
                       features_name_2D : list,
                       features : list,
                       features_1D,
                       raster : np.ndarray,
                       x_1d,
                       y_1d,
                       name_exp : str,
                       ) -> np.array:
    
    features_name_2D_full, _ = get_features_name_lists_2D(6, features)

    dir_encoder = path / '../../'

    encoder_osmnx = read_object(f'encoder_osmnx.pkl_{name_exp}', dir_encoder)
    encoder_foret = read_object(f'encoder_foret_{name_exp}.pkl', dir_encoder)
    encoder_argile = read_object(f'encoder_argile_{name_exp}.pkl', dir_encoder)
    encoder_cosia = read_object(f'encoder_cosia_{name_exp}.pkl', dir_encoder)

    leni = len(features_name_2D)
    if date < 0:
        return None
    x_2D = read_object(f'X_{date}.pkl', path)

    if x_2D is None:
        return None
    new_x_2D = np.empty((leni, x_2D.shape[1], x_2D.shape[2]))
    
    for i, fet_2D in enumerate(features_name_2D):
        
        #print(fet_2D, np.nanmax(x_2D[features_name_2D_full.index(fet_2D)]))
        """plt.imshow(x_2D[features_name_2D_full.index(fet_2D)])
        plt.colorbar()
        plt.savefig(f'{fet_2D}.png')
        plt.close('all')       """ 
        if fet_2D == 'Past_risk' or fet_2D == 'Past_burnedarea' or fet_2D == 'cluster_encoder' or fet_2D == 'id_encoder':
            unode = np.unique(raster) 
            for node in unode:
                mask = (raster == node)
                m1 = (y_1d[:, id_index] == node)
                if True not in m1:
                    new_x_2D[i, mask] = 0
                else:
                    new_x_2D[i, mask] = x_1d[m1, features_1D.index(fet_2D)]

        elif fet_2D == 'foret_encoder' in features:
            #logger.info('Foret landcover')
            assert encoder_foret is not None
            new_x_2D[i, :, :] = encoder_foret.transform(x_2D[features_name_2D_full.index(fet_2D), :, :].reshape(-1,1)).values.reshape((x_2D.shape[1], x_2D.shape[2]))

        elif fet_2D == 'highway_encoder' in features:
            #logger.info('OSMNX landcover')
            new_x_2D[i, :, :] = encoder_osmnx.transform(x_2D[features_name_2D_full.index(fet_2D), :, :].reshape(-1,1)).values.reshape((x_2D.shape[1], x_2D.shape[2]))

        elif fet_2D == 'argile_encoder' in features:
            #logger.info('OSMNX landcover')
            assert encoder_argile is not None
            new_x_2D[i, :, :] = encoder_argile.transform(x_2D[features_name_2D_full.index(fet_2D), :, :].reshape(-1,1)).values.reshape((x_2D.shape[1], x_2D.shape[2]))

        elif fet_2D == 'cosia_encoder' in features:
            #logger.info('OSMNX landcover')
            assert encoder_cosia is not None
            new_x_2D[i, :, :] = encoder_cosia.transform(x_2D[features_name_2D_full.index(fet_2D), :, :].reshape(-1,1)).values.reshape((x_2D.shape[1], x_2D.shape[2]))

        else:
            new_x_2D[i, :, :] = x_2D[features_name_2D_full.index(fet_2D), :, :]
        if False not in np.isnan(new_x_2D[i, :, :]):
            return None
        else:
            nan_mask = np.isnan(new_x_2D[i, :, :])
            new_x_2D[i, nan_mask] = np.nanmean(new_x_2D[i, :, :])
            #new_x_2D[i, nan_mask] = 0.0
        #   print(np.unique(np.isnan(new_x_2D)))
    #exit(1)
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

def process_time_step(X, Y, dept, graph, ks, id, path, features_name_2D, features, features_1D, x_1d, y_1d, raster_dept, raster_dept_graph, unodes, image_per_node, shape2D, name_exp):
    """Process each time step and update X and Y arrays"""
    for k in range(ks + 1):
        date = int(id) - (ks - k)
        x_date = load_x_from_pickle(date, path / f'2D_database' / int2name[dept], features_name_2D, features, features_1D, raster_dept, x_1d[:, :, k], y_1d[:, :, k], name_exp)

        if x_date is None:
            return None, None

        if x_date is None:
            x_date = np.zeros((len(features_name_2D), raster_dept.shape[0], raster_dept.shape[1]))
        
        if image_per_node:
            X = process_node_images(X, unodes, x_date, raster_dept, y_1d, graph.scale, k, date, shape2D)
        else:
            X = process_dept_images(X, x_date, raster_dept, ks, k)

    if not image_per_node:
        y_dept = generate_image_y(y_1d, raster_dept_graph)
        Y = process_dept_y(Y, y_dept, raster_dept_graph, ks)
    
    return X, Y

def process_node_images(X, unodes, x_date, raster_dept, y_1d, scale, k, date, shape2D):
    """Process images for each node."""
    node2remove = []
    for node in unodes:
        if node not in np.unique(y_1d[:, graph_id_index]):
            node2remove.append(node)
            continue
        mask = np.argwhere(raster_dept == node)
        x_node = extract_node_data(x_date, mask, node, raster_dept)
        X = update_node_images(X, x_node, mask, scale, k, shape2D, y_1d, date, node)
    
    return X

def extract_node_data(x_date, mask, node, raster_dept):
    """Extract and mask data for a specific node."""
    x_date_cp = np.copy(x_date)
    x_date_cp[:, raster_dept != node] = 0.0
    minx, miny, maxx, maxy = np.min(mask[:, 0]), np.min(mask[:, 1]), np.max(mask[:, 0]), np.max(mask[:, 1])
    res = x_date_cp[:, minx:maxx+1, miny:maxy+1]
    res[np.isnan(res)] = np.nanmean(res)
    return res

def update_node_images(X, x_node, mask, scale, k, shape2D, y_1d, date, node):
    """Resize and update node images in X array."""
    for band in range(x_node.shape[0]):
        x_band = x_node[band]
        if False not in np.isnan(x_band):
            x_band[np.isnan(x_band)] = -1
        else:
            x_band[np.isnan(x_band)] = np.nanmean(x_band)
        #index = np.argwhere((y[:, graph_id_index, 0] == node) & (y[:, date_index, :] == date))[:, 0]
        index = np.unique(np.argwhere((y_1d[:, graph_id_index, 0] == node))[:, 0])
        #print(np.unique(x_band))
        X[index, band, :, :, k] = resize_no_dim(x_band, *shape2D[scale])
        mask_nan = np.isnan(X[index, band, :, :, k])
        #X[index, band, mask_nan, k] = 0
    X[np.isnan(X)] = -1
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
                        features_name_2D, use_temporal_as_edges):
    """Main function to create the dataset."""
    Xst, Yst, Est = [], [], []
    leni = len(features_name_2D)
    for id in dates:
        if use_temporal_as_edges is None:
            x, y = construct_time_series(id, X_np, Y_np, ks, len(ids_columns))
        elif use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, id, X_np, Y_np, ks, len(ids_columns))
        else:
            x, y, e = construct_graph_with_time_series(graph, id, X_np, Y_np, ks, len(ids_columns))
        
        if x is None:
            continue
        
        """depts = np.unique(y[:, departement_index].astype(int))
        for dept in depts:
            y_dept = y[y[:, departement_index, 0] == dept]
            x_dept = x[y[:, departement_index, 0] == dept]
            new_x = []
            new_y = []

            sub_dir = f'image_per_node_{leni}' if image_per_node else f'image_per_departement_{leni}'

            for i in range(x_dept.shape[0]):
                cluster_id = y_dept[i, graph_id_index, -1]
                if use_temporal_as_edges is None and image_per_node:
                    is_file = (path / f'2D_database_{graph.scale}_{graph.base}_{graph.graph_method}' / sub_dir / context / f'X_{int(id)}_{dept}_{cluster_id}.pkl').is_file()
                else:
                    is_file = (path / f'2D_database_{graph.scale}_{graph.base}_{graph.graph_method}' / sub_dir / context / f'X_{int(id)}_{dept}.pkl').is_file()
                if not is_file:
                        new_x.append(x_dept[i])
                        new_y.append(y_dept[i])
                else:
                    Xst.append(f'X_{int(id)}_{dept}_{cluster_id}.pkl')
                    Yst.append(y_dept[i])
            
            #if len(new_y) == 0:
            #    continue
            
            #new_x = np.asarray(new_x)
            #new_y = np.asarray(new_y)

            new_y = np.copy(y_dept)
            new_x = np.copy(x_dept)
            
            X, Y, raster_dept, raster_dept_graph, unodes = process_dept_raster(dept, graph, path, new_y, features_name_2D, ks, image_per_node, shape2D)
            X, Y = process_time_step(X, Y, dept, graph, ks, id, path, features_name_2D, features, features_1D, new_x, new_y, raster_dept, raster_dept_graph, unodes, image_per_node, shape2D, name_exp)
            #print(X.shape)
            
            if X is None:
                continue

            if use_temporal_as_edges is None and image_per_node:
                for i in range(X.shape[0]):
                    #print(np.unique(np.isnan(X[i])))
                    if True:
                        cluster_id = Y[i][graph_id_index, -1]
                        save_object(X[i], f'X_{int(id)}_{dept}_{cluster_id}.pkl', path / f'2D_database' / sub_dir / context)
                        Xst.append(f'X_{int(id)}_{dept}_{cluster_id}.pkl')
                        Yst.append(Y[i])
                    else:
                        Xst.append(X[i])
                        Yst.append(y[i])
            else:
                if True:
                    save_object(X, f'X_{int(id)}_{dept}.pkl', path / f'2D_database' / sub_dir / context)
                    Xst.append(f'X_{int(id)}_{dept}.pkl')
                else:
                    Xst.append(X)
                Y[np.isnan(Y)] = 0
                Yst.append(Y)"""

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
                    features_1D,
                    target_name,
                    image_per_node,
                    use_temporal_as_edges : bool,
                    device,
                    ks : int,
                    name_exp):

    if df_train is not None:
        x_train, y_train = df_train[ids_columns + features_1D].values, df_train[ids_columns + [target_name]].values
        dateTrain = np.sort(np.unique(y_train[np.argwhere(y_train[:, weight_index] > 0), date_index]))

    if df_val is not None:
        x_val, y_val = df_val[ids_columns + features_1D].values, df_val[ids_columns + [target_name]].values
        dateVal = np.sort(np.unique(y_val[np.argwhere(y_val[:, weight_index] > 0), date_index]))

    if df_test is not None:
        x_test, y_test = df_test[ids_columns + features_1D].values, df_test[ids_columns + [target_name]].values
        dateTest = np.sort(np.unique(y_test[np.argwhere(y_test[:, weight_index] > 0), date_index]))

    XsTe = []
    YsTe = []
    EsTe = []
    
    logger.info(f'Model configuration : image_per_node {image_per_node}, use_temporal_as_edges {use_temporal_as_edges}')

    if df_train is not None:
        logger.info('Creating train dataset')
        Xst, Yst, Est = construct_dataset(dateTrain, x_train, y_train, graph, ids_columns, ks, use_temporal_as_edges, True) 
        assert len(Xst) > 0
    
    if df_val is not None:
        logger.info('Creating val dataset')
        XsV, YsV, EsV = construct_dataset(dateVal, x_val, y_val, graph, ids_columns, ks, use_temporal_as_edges, True)
        assert len(XsV) > 0

    if df_test is not None:
        logger.info('Creating Test dataset')
        XsTe, YsTe, EsTe = construct_dataset(dateTest, x_test, y_test, graph, ids_columns, ks, use_temporal_as_edges, True)

        assert len(XsTe) > 0

    #logger.info(f'{len(Xst)}, {len(XsV)}, {len(XsTe)}')
    train_dataset = None
    val_dataset = None
    test_dataset = None
    if True:
        if df_train is not None:
            train_dataset = ReadGraphDataset_2D_from_xarray(Xst, Yst, Est, len(Xst), device,
                                                            rootDisk / 'csv', features_name_2D, features_1D, ks,
                                                            graph.scale,
                                                            graph.graph_method,
                                                            graph.base,
                                                            path / 'datacube')
        
        if df_val is not None:
            val_dataset = ReadGraphDataset_2D_from_xarray(XsV, YsV, EsV, len(XsV), device,
                                                          rootDisk / 'csv', features_name_2D, features_1D, ks,
                                                        graph.scale,
                                                            graph.graph_method,
                                                            graph.base,
                                                            path / 'datacube')
        if df_test is not None:
            test_dataset = ReadGraphDataset_2D_from_xarray(XsTe, YsTe, EsTe, len(XsTe), device,
                                                           rootDisk / 'csv', features_name_2D, features_1D, ks,
                                                            graph.scale,
                                                            graph.graph_method,
                                                            graph.base,
                                                            path / 'datacube')
    else:
        train_dataset = InplaceGraphDataset(Xst, Yst, Est, len(Xst), device)
        val_dataset = InplaceGraphDataset(XsV, YsV, EsV, len(XsV), device)
        test_datset = InplaceGraphDataset(XsTe, YsTe, EsTe, len(XsTe), device)
    return train_dataset, val_dataset, test_dataset

class WrapperModel(torch.nn.Module):
    def __init__(self, original_model, F, T, edges, horizon=1):
        super().__init__()
        self.model = original_model
        self.F = F
        self.T = T
        self.edges = edges

        self.horizon = horizon

    def forward(self, x_flat):
        # reshape x_flat (B, F*T) vers (B, F, T)
        x_orig = x_flat.reshape(-1, self.F, self.T)
        return self.model(x_orig, self.edges)

class Training():
    def __init__(self, model_name, nbfeatures, batch_size, lr, target_name, task_type,
                 features_name, ks, out_channels, dir_log,
                 loss='mse', name='Training', device='cpu', under_sampling='full', over_sampling='full', n_run=1,
                 horizon=1):
        
        self.model_name = model_name
        self.name = name
        self.loss = loss
        self.device = device
        self.model = None
        self.optimizer = None
        self.batch_size = batch_size
        self.target_name = target_name
        self.features_name = [str(fet) for fet in features_name]
        self.ks = int(ks)
        self.lr = lr
        self.out_channels = out_channels
        self.dir_log = dir_log
        self.task_type = task_type
        self.model_params = None
        self.under_sampling = under_sampling
        self.over_sampling = over_sampling
        self.find_log = False
        self.nbfeatures = nbfeatures
        self.student_train = False
        self.use_temporal_as_edges = None
        self.n_run = n_run
        self.metrics = {}
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.constrastive = False
        self.use_prototypes = False
        self.prototype_weight = 1.0
        self.prototypes = None
        self.ALATraining = False
        self.area_parameters = None
        # Distillation tracking (best/worst losses per epoch)
        self.distill_best_log = []   # list of dicts: {epoch, graph_id, loss}
        self.distill_worst_log = []  # list of dicts: {epoch, graph_id, loss}
        self.criterion_params = []
        self._current_epoch = None
        self.seed = None
        self.horizon = horizon
        self.seed = None 

    def compute_weights_and_target(self, labels, band, ids_columns, is_grap_or_node, graphs):
        weight_idx = ids_columns.index('weight')
        target_is_binary = self.task_type == 'binary'

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
    
    def compute_single_loss(self, out, tar, wei, cluster_ids=None, tolong=False, criterion=None):
        if self.task_type == 'regression':
            tar = tar.view(out.shape[0])
            wei = wei.view(out.shape[0])

            tar = torch.masked_select(tar, wei.gt(0))
            out = out[wei.gt(0)]
            wei = torch.masked_select(wei, wei.gt(0))
        else:
            wei = wei.long()
            #print(torch.unique(tar))
            if not self.student_train: # works on probability
                tar = tar.long()

            tar = tar[wei.gt(0)]
            out = out[wei.gt(0)]

            if cluster_ids is not None:
                cluster_ids = cluster_ids[wei.gt(0)]

            wei = torch.masked_select(wei, wei.gt(0))

            if tolong:
                tar = tar.long()

            #if self.loss in ['kappa', 'cdw', 'mcewk']:
            #    tar = tar.to('cpu')
            #    out = out.to('cpu')

        if cluster_ids is not None:
            return criterion(out, tar, cluster_ids=cluster_ids)
        else:
            return criterion(out, tar)

    def loss_distill(
        self,
        output: torch.Tensor,          # [B, C] logits ou sorties du modèle (si compute_single_loss en a besoin)
        target: torch.Tensor,          # [B, ...] doit contenir les IDs de région en colonne graph_id_index
        weight: torch.Tensor,          # 
        label : torch.Tensor,
        hidden: torch.Tensor,          # [B, D] embeddings (features) par échantillon
        percent_less: float,           # ex: 0.10 pour 10% pires régions
        percent_high: float,           # ex: 0.10 pour 10% meilleures régions
        graph_id_index: int,           # index de la colonne dans target contenant l'ID de région
        lambda_kd: float = 1.0,        # poids du terme de distillation
        use_cosine: bool = True,       # True = 1 - cos, False = MSE
        tolong=False,
        cluster_ids=None,
        criterion=None
    ):
        """
        Calcule un terme de distillation d'embeddings des régions 'fortes' (meilleures) vers les 'faibles' (pires),
        en se basant sur la loss par région. Retourne:
        - kd_loss: le terme de distillation,
        - region_losses: dict {region_id: loss_scalar} (detach) pour inspection,
        - best_ids / worst_ids: listes d'IDs sélectionnés.

        On suppose l'existence d'une fonction globale:
            compute_single_loss(output_subset, target_subset) -> scalaire (Tensor)
        """

        assert 0 < percent_less <= 1 and 0 < percent_high <= 1, "percentages doivent être dans (0,1]"
        device = output.device
        region_ids = label[:, graph_id_index, -1]
        unique_ids = torch.unique(region_ids)

        # 1) Loss par région (pour le tri)
        region_losses = {}
        for rid in unique_ids:
            mask = (region_ids == rid)
            # IMPORTANT: compute_single_loss peut s'attendre à des shapes [N, C] / [N, ...]
            loss_i = self.compute_single_loss(output[mask], target[mask], weight[mask], cluster_ids, tolong, criterion)
            # on détache pour le tri (ne pas backprop à travers la sélection)
            region_losses[int(rid.item())] = loss_i.detach()

        # 2) Tri des régions par loss (croissant: meilleures d'abord)
        sorted_items = sorted(region_losses.items(), key=lambda kv: kv[1].item())
        n_regions = len(sorted_items)
        k_high = max(1, int(round(percent_high * n_regions)))
        k_low  = max(1, int(round(percent_less * n_regions)))

        best_ids  = [rid for rid, _ in sorted_items[:k_high]]           # meilleures (loss faible)
        worst_ids = [rid for rid, _ in sorted_items[-k_low:]]           # pires (loss élevée)

        # 3) Prototype enseignant = moyenne des embeddings des meilleures régions
        best_mask = torch.zeros_like(region_ids, dtype=torch.bool)
        for rid in best_ids:
            best_mask |= (region_ids == rid)

        # S'il n'y a pas d'échantillon (cas pathologique), on protège
        if best_mask.any():
            teacher_proto = hidden[best_mask].mean(dim=0, keepdim=True)  # [1, D]
        else:
            # fallback: moyenne globale
            teacher_proto = hidden.mean(dim=0, keepdim=True)

        # On "coupe" le gradient côté enseignant (on ne veut pas déplacer les meilleures)
        teacher_proto = teacher_proto.detach()

        # 4) Distillation: on pousse les embeddings des pires vers le prototype enseignant
        worst_mask = torch.zeros_like(region_ids, dtype=torch.bool)
        for rid in worst_ids:
            worst_mask |= (region_ids == rid)

        if worst_mask.any():
            student_emb = hidden[worst_mask]                 # [N_w, D]

            if use_cosine:
                # 1 - cos(sim)  (plus stable d'échelle que MSE)
                student = F.normalize(student_emb, dim=-1)
                teacher = F.normalize(teacher_proto, dim=-1)
                kd = 1.0 - (student @ teacher.T).squeeze(-1) # [N_w]
                kd_loss = kd.mean()
            else:
                # MSE sur embeddings non normalisés
                kd_loss = F.mse_loss(student_emb, teacher_proto.expand_as(student_emb))
        else:
            # aucune région "pire" sélectionnée
            kd_loss = torch.tensor(0.0, device=device)

        # 5) Pondération du terme de distillation
        kd_loss = lambda_kd * kd_loss

        return kd_loss, region_losses, best_ids, worst_ids

    def calculate_loss(self, criterion, output, target, weights, label, tolong=True):

        if 'cluster_ids' in required_params(criterion.forward):
            id_mask = label[:, criterion.id, -1]
            self.cluster_id_index = criterion.id
        else:
            id_mask = None
            
        base_loss = self.compute_single_loss(output, target, weights, id_mask, tolong, criterion)

        if 'area' in self.loss: # Calculate area loss (specify loss-area)
            area_mask = label[:, graph_id_index, -1]
            unique_ids = torch.unique(area_mask)
            values = []
            active_idx = []
            for aid in unique_ids:
                m = area_mask == aid
                if m.sum() == 0:
                    continue
                active_idx.append(int(aid))
                l = self.compute_single_loss(output[m], target[m], weights[m], None, tolong, criterion)
                #print(f'{aid}, {l}')
                values.append(l)
            if len(values) > 0:
                active_idx = torch.as_tensor(active_idx, dtype=torch.long)
                mask = torch.zeros_like(self.area_parameters, dtype=self.area_parameters.dtype)
                mask.index_fill_(0, active_idx, 1.0)
                vals_active = torch.as_tensor(values, device=self.area_parameters.device,
                              dtype=self.area_parameters.dtype)
                vals_full = torch.zeros_like(self.area_parameters)
                vals_full.index_copy_(0, active_idx, vals_active)
                ap_active = self.area_parameters * mask
                eps = 1e-8
                ap_active = torch.log(torch.nn.functional.softplus(ap_active))
                area_loss = (vals_full * ap_active).sum() / ap_active.sum().clamp_min(eps)
            else:
                area_loss = torch.as_tensor(0.0, device=output.device)

            if 'area-global' in self.loss:  # Calculate area * global (classic) loss  (specify loss-area-global)
                loss = area_loss + base_loss
                logger.info(f'area_loss : {area_loss}, {base_loss}, {loss}')
            else:
                loss = area_loss
        else:
            loss = base_loss

        return loss
    
    def calculate_contrastive_moon_loss(self, z, zprev, zglob, temperature=0.5):
        """
        Computes the MOON contrastive loss.

        Args:
            z       : Tensor of shape [batch_size, dim] from current local model.
            zprev   : Tensor of shape [batch_size, dim] from previous local model.
            zglob   : Tensor of shape [batch_size, dim] from global model.
            temperature (float): Temperature parameter τ for scaling similarities.

        Returns:
            loss (Tensor): Scalar contrastive loss for the batch.
        """
        # Normalize representations to compute cosine similarity
        z = F.normalize(z, dim=1)
        zprev = F.normalize(zprev, dim=1)
        zglob = F.normalize(zglob, dim=1)

        # Cosine similarities
        sim_pos = torch.sum(z * zglob, dim=1) / temperature  # similarity with global (positive)
        sim_neg = torch.sum(z * zprev, dim=1) / temperature   # similarity with previous (negative)

        # Contrastive loss per sample
        logits = torch.stack([sim_pos, sim_neg], dim=1)  # shape: [batch_size, 2]
        labels = torch.zeros(z.size(0), dtype=torch.long, device=z.device)  # positive is at index 0

        # Use cross-entropy to compute: -log( exp(sim_pos) / (exp(sim_pos) + exp(sim_neg)) )
        loss = F.cross_entropy(logits, labels)
        
        return loss

    def calculate_prototype_loss(self, hidden, target, prototypes):
        """Compute prototype alignment loss."""
        loss = 0.0
        classes = torch.unique(target)
        for cls in classes:
            cls_idx = int(cls.item())
            if prototypes is None or cls_idx not in prototypes:
                continue
            proto = prototypes[cls_idx].to(hidden.device)
            mask = target == cls
            if mask.sum() == 0:
                continue
            diff = hidden[mask] - proto
            loss += torch.mean(torch.norm(diff, dim=1))
        return loss

    def calculate_prototype_alignment_loss(self, hidden, target, prototypes):
        """
        Compute prototype alignment loss:
        Sum over classes of L2 distance squared between local and global prototypes.

        Arguments:
            hidden (Tensor): Embeddings of shape [batch_size, embedding_dim].
            target (Tensor): Class labels of shape [batch_size].
            prototypes (dict): {class_id: global_prototype_tensor}

        Returns:
            loss (Tensor): Scalar tensor representing the total alignment loss.
        """
        loss = 0.0
        classes = torch.unique(target)
        for cls in classes:
            cls_idx = int(cls.item())
            if prototypes is None or cls_idx not in prototypes:
                continue
            # Global prototype
            proto_global = prototypes[cls_idx].to(hidden.device)
            
            # Local prototype for class cls
            mask = target == cls
            if mask.sum() == 0:
                continue
            proto_local = hidden[mask].mean(dim=0)
            
            # L2 distance squared between local and global prototype
            diff = proto_local - proto_global
            loss += torch.sum(diff ** 2)
            
        return loss
    
    def launch_batch(self, data, criterion):
        inputs, labels, edges = data
        graphs = None

        if inputs.shape[0] == 1:
            return 0

        band = -1

        try:
            target, weights = self.compute_weights_and_target(labels, band, ids_columns, self.model.is_graph_or_node, graphs)
        except Exception as e:
            target, weights = self.compute_weights_and_target(labels, band, ids_columns, False, graphs)

        if self.loss not in ['kldivloss']: # works on probability
            target = target.long()
        
        output, logits, hidden = self.model(inputs, edges)

        loss = self.calculate_loss(criterion, logits, target, weights, labels)

        if self.student_train: # distallation traning
            criterion_teacher = self.get_loss('kldivloss')
            df_test = pd.DataFrame(inputs[:, :, -1], columns=self.features_name)
            df_test.columns = df_test.columns.astype(str)
            if self.top_model != 'task':
                teacher_logits = self.teacher.predict(df_test,
                                                            weights_average=self.weights_average,
                                                            top_model=self.top_model, id_col=(None, None),
                                                            prediction_type='RawFormulaVal')
            else:
                teacher_logits = self.teacher.predict_with_tasks(df_test,
                                                                 weights_average=self.weights_average,
                                                                 id_col=(None, None), proba='RawFormulaVal')
            
            teacher_logits = torch.Tensor(teacher_logits, device=inputs.device).to(torch.float32)

            T = torch.nn.functional.softplus(self.temperature_value) + 1e-6
            
            p_teacher = F.softmax(teacher_logits / T, dim=1)
            p_student = F.log_softmax(logits / T, dim=1)

            target = target / T

            kl_div_loss = self.calculate_loss(criterion_teacher, p_student, p_teacher, weights, labels, tolong=False) * (T * T)

            loss = self.alpha_value * kl_div_loss + (1 - self.alpha_value) * loss + 1e-3 * (torch.log(T) ** 2)
        
        if self.constrastive: # MOON federated training
            _, _, zprev = self.prev_model(inputs, edges)
            _, _, zglob = self.global_model(inputs, edges)
            loss_constrastive = self.calculate_contrastive_moon_loss(hidden, zprev, zglob, self.moon_temperature_value)
            loss = loss + self.smooth_value * loss_constrastive

        if self.use_prototypes and self.prototypes is not None:
            if not self.model.return_hidden:
                raise ValueError('Model must return hidden states for prototype training')
            
            proto_loss = self.calculate_prototype_alignment_loss(hidden, target, self.prototypes)
            loss = loss + self.prototype_weight * proto_loss

        if 'distillation' in self.loss:
            distill_loss, region_losses, best, worst = self.loss_distill(
                output, target, weights, labels, hidden, 0.10, 0.10, graph_id_index,
                lambda_kd=1, use_cosine=True, tolong=False, cluster_ids=None, criterion=criterion
            )
            loss = loss + distill_loss

            # Update per-epoch best/worst trackers using region_losses
            try:
                # region_losses is a dict {rid: tensor_loss}
                if isinstance(region_losses, dict) and len(region_losses) > 0:
                    # Best: smallest loss among reported best IDs
                    if len(best) > 0:
                        best_pair = min(((rid, region_losses[rid].item()) for rid in best if rid in region_losses),
                                        key=lambda kv: kv[1], default=None)
                        if best_pair is not None:
                            rid_b, loss_b = best_pair
                            if hasattr(self, '_epoch_distill_best'):
                                if loss_b < self._epoch_distill_best['loss']:
                                    self._epoch_distill_best['loss'] = float(loss_b)
                                    self._epoch_distill_best['graph_id'] = int(rid_b)
                    # Worst: largest loss among reported worst IDs
                    if len(worst) > 0:
                        worst_pair = max(((rid, region_losses[rid].item()) for rid in worst if rid in region_losses),
                                         key=lambda kv: kv[1], default=None)
                        if worst_pair is not None:
                            rid_w, loss_w = worst_pair
                            if hasattr(self, '_epoch_distill_worst'):
                                if loss_w > self._epoch_distill_worst['loss']:
                                    self._epoch_distill_worst['loss'] = float(loss_w)
                                    self._epoch_distill_worst['graph_id'] = int(rid_w)
            except Exception as _e:
                # Never break training because of logging
                pass

        if self.model_name in ['BayesianMLP', 'BayesianCNN', 'BayesianRNN']:
            loss += self.model.kl_loss()

        return loss
    
    def launch_train_loader(self, loader, criterion, optimizer):

        self.model.train()
        
        if has_method(criterion, 'get_learnable_parameters'):
            criterion.train()

        # Initialize per-epoch aggregation for distillation best/worst
        if 'distillation' in self.loss:
            self._epoch_distill_best = {'loss': float('inf'), 'graph_id': None}
            self._epoch_distill_worst = {'loss': float('-inf'), 'graph_id': None}
        
        for i, data in enumerate(loader, 0):

            loss = self.launch_batch(data, criterion)

            if isinstance(loss, int):
                continue
            
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
            
            if 'res_loss' in locals():
                res_loss += loss.item()
            else:
                res_loss = loss.item()

            if self.ALATraining:
                
                # Mises à jour SANS autograd
                with torch.no_grad():
                    # 1) update des weights
                    for p_t, p_prev, p_g, w in zip(
                            self.params_p, self.params_tp, self.params_gp, self.weights):
                        upd = w - self.eta * ((p_g - p_prev) * p_t.grad)
                        w.copy_(torch.clamp(upd, 0.0, 1.0))

                    if not self.ala_weight_only:
                        # 2) calcul des params interpolés
                        for p_t, p_prev, p_g, w in zip(
                                self.params_p, self.params_tp, self.params_gp, self.weights):
                            p_t.copy(p_t - self.eta * (p_g - p_prev) * (p_t.grad))

            if optimizer is not None:
                optimizer.step()

                #self.update_weight()
        # After finishing the epoch, persist the best/worst entries for this epoch
        if 'distillation' in self.loss:
            if getattr(self, '_epoch_distill_best', None) is not None and self._epoch_distill_best['graph_id'] is not None:
                self.distill_best_log.append({
                    'epoch': self._current_epoch if self._current_epoch is not None else -1,
                    'graph_id': int(self._epoch_distill_best['graph_id']),
                    'loss': float(self._epoch_distill_best['loss'])
                })
            if getattr(self, '_epoch_distill_worst', None) is not None and self._epoch_distill_worst['graph_id'] is not None:
                self.distill_worst_log.append({
                    'epoch': self._current_epoch if self._current_epoch is not None else -1,
                    'graph_id': int(self._epoch_distill_worst['graph_id']),
                    'loss': float(self._epoch_distill_worst['loss'])
                })

        if has_method(criterion, 'get_attribute'):
            params = criterion.get_attribute()
            dict_params = {'epoch': self._current_epoch}
            for par in params:
                name = par[0]
                value = par[1]
                dict_params[name] = copy.deepcopy(value.detach().cpu().numpy())
            
            self.criterion_params.append(dict_params)

        if self.student_train:
            self.distillation_log.append({'epoch' : self._current_epoch,
                                 'temperature' : copy.deepcopy(self.temperature_value.detach().cpu().numpy()),
                                  'alpha' : copy.deepcopy(self.alpha_value.detach().cpu().numpy())})

        return res_loss

    def launch_val_test_loader(self, loader, criterion, teacher=None):

        self.model.eval()

        if has_method(criterion, 'get_learnable_parameters'):
            criterion.eval()

        total_loss = 0.0

        with torch.no_grad():

            for i, data in enumerate(loader, 0):
                loss = self.launch_batch(data, criterion)

                total_loss += loss.item()
            
        if 'learnable-area' in self.loss:
            if hasattr(self, 'area_parameters_log'):
                self.area_parameters_log.append(self.area_parameters)
            else:
                self.area_parameters_log = []
                self.area_parameters_log.append(self.area_parameters)

        return total_loss
    
    def make_model(self, graph, custom_model_params):
        model, params = make_model(self.model_name, len(self.features_name), len(self.features_name),
                                graph, dropout, activation,
                                self.ks,
                                out_channels=self.out_channels,
                                task_type=self.task_type,
                                device=device, num_lstm_layers=num_lstm_layers,
                                custom_model_params=custom_model_params)
        if self.model_params is None:
            self.model_params = params
        return model, params
    
    def func_epoch(self, train_loader, val_loader, optimizer, criterion):

        train_loss = self.launch_train_loader(train_loader, criterion, optimizer)

        if val_loader is not None:
            val_loss = self.launch_val_test_loader(val_loader, criterion)
        else:
            val_loss = train_loss.item()

        return val_loss, train_loss

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

        criterion = self.get_loss(self.loss)

        if not isinstance(self, ModelCNN):
            static_idx, temporal_idx = get_static_temporal_idx(self.features_name)
        else:
            static_idx, temporal_idx = [0, 0]

        if self.model_name in ['SepGRUGNN']:
            if custom_model_params is None:
                custom_model_params = {'static_idx': static_idx, 'temporal_idx' : temporal_idx}
            else:
                custom_model_params.update({'static_idx': static_idx, 'temporal_idx' : temporal_idx})

        if new_model or self.model is None:
            self.model, _ = self.make_model(graph, custom_model_params)
        
        optimizer = self.get_optimizer(criterion)

        BEST_VAL_LOSS = math.inf
        BEST_MODEL_PARAMS = None
        best_epoch = 0
        patience_cnt = 0

        val_loss_list = []
        train_loss_list = []
        epochs_list = []

        #if (self.dir_log / 'best.pt').is_file():
        if False:
            self._load_model_from_path(self.dir_log / 'best.pt', self.model)
        else:
            for epoch in tqdm(range(epochs), disable=not verbose):
                # Expose current epoch to subroutines for logging
                self._current_epoch = epoch
                val_loss, train_loss = self.func_epoch(train_loader=self.train_loader, val_loader=self.val_loader,
                                                    optimizer=optimizer, criterion=criterion)

                val_loss_list.append(round(val_loss, 3))
                train_loss_list.append(round(train_loss, 3))
                epochs_list.append(epoch)
                if val_loss < BEST_VAL_LOSS:
                    BEST_VAL_LOSS = val_loss
                    BEST_MODEL_PARAMS = self.model.state_dict()
                    patience_cnt = 0
                    best_epoch = epoch
                else:
                    patience_cnt += 1
                    if patience_cnt >= PATIENCE_CNT:
                        logger.info(f'Loss has not increased for {patience_cnt} epochs. Last best val loss {BEST_VAL_LOSS}, current val loss {val_loss}')
                        save_object_torch(self.model.state_dict(), 'last.pt', self.dir_log)
                        save_object_torch(BEST_MODEL_PARAMS, 'best.pt', self.dir_log)
                        plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, self.dir_log)
                        if MLFLOW:
                            mlflow.end_run()
                        break
                if MLFLOW:
                    mlflow.log_metric('loss', val_loss, step=epoch)
                if epoch % CHECKPOINT == 0 and verbose:
                    logger.info(f'epochs {epoch}, Val loss {val_loss}')
                    logger.info(f'epochs {epoch}, Best val loss {BEST_VAL_LOSS}')
                    save_object_torch(self.model.state_dict(), str(epoch)+'.pt', self.dir_log)

            logger.info(f'Last val loss {val_loss}')
            save_object_torch(self.model.state_dict(), 'last.pt', self.dir_log)
            save_object_torch(BEST_MODEL_PARAMS, 'best.pt', self.dir_log)
            plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, self.dir_log)
        
        self.best_epoch = best_epoch
        logger.info(f'Best epoch {best_epoch}, Best val loss {BEST_VAL_LOSS}')
        ##################################### TEST #################################################

        test_output, y = self._predict_test_loader(self.test_loader, output_pdf='test')
        test_output = test_output.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        if np.any(y[:, -1] > 0) or np.any(test_output > 0):

            under_prediction_score_value = under_prediction_score(y[:, -1], test_output)
            over_prediction_score_value = over_prediction_score(y[:, -1], test_output)
            
            iou = iou_score(y[:, -1], test_output)
            f1 = f1_score((test_output > 0).astype(int), (y[:, -1] > 0).astype(int), zero_division=0)
            iou_area, f1_area = self.compute_area_score(test_output, y[:, -1], y[:, graph_id_index])

            print(f'Test -> Under achieved : {under_prediction_score_value}, Over achived {over_prediction_score_value}, IoU {iou}, f1 {f1}, IoU_area {iou_area}, f1_area {f1_area}')

            test_output, y = self._predict_test_loader(self.val_loader, output_pdf='Val')
            test_output = test_output.detach().cpu().numpy()
            
            y = y.detach().cpu().numpy()

            under_prediction_score_value = under_prediction_score(y[:, -1], test_output)
            over_prediction_score_value = over_prediction_score(y[:, -1], test_output)
            
            iou = iou_score(y[:, -1], test_output)
            f1 = f1_score((test_output > 0).astype(int), (y[:, -1] > 0).astype(int), zero_division=0)
            iou_area, f1_area = self.compute_area_score(test_output, y[:, -1], y[:, graph_id_index])

            print(f'Val {y.shape} -> Under achieved : {under_prediction_score_value}, Over achived {over_prediction_score_value}, IoU {iou} f1 {f1}, IoU_area {iou_area}, f1_area {f1_area}')

            plt.figure(figsize=(15,5))
            plt.plot(y[y[:, departement_index] == 13, -1])
            plt.plot(test_output[y[:, departement_index] == 13])
            plt.savefig(self.dir_log / 'test_13.png')

            plt.figure(figsize=(15,5))
            plt.plot(y[y[:, departement_index] == 6, -1])
            plt.plot(test_output[y[:, departement_index] == 6])
            plt.savefig(self.dir_log / 'test_6.png')

        if BEST_MODEL_PARAMS is not None:
            self.update_weight(BEST_MODEL_PARAMS)
        
        if 'learnable-area' in self.loss:
            ids = y[:, 0]                       # première colonne
            values = y[:, 1:]

            # Somme groupée par id
            unique_ids, inverse = np.unique(ids, return_inverse=True)
            sums = np.zeros((len(unique_ids), values.shape[1]), dtype=values.dtype)
            np.add.at(sums, inverse, values)
            
            self.plot_area_parameter(epochs_list, y[:, 0], sums[:, -1])
            save_object(self.area_parameters_log, 'area_parameters_log.pkl' ,self.dir_log)

        if has_method(criterion, 'plot_params'):
            if has_method(criterion, 'update_params'):
                criterion.update_params(self.criterion_params[self.best_epoch])
            criterion.plot_params(self.criterion_params, self.dir_log)

        # Save distillation best/worst logs and 3D plot at the end of training
        if 'distillation' in self.loss:
            try:
                self._save_distill_logs_and_plot()
            except Exception as _e:
                # Keep training flow robust even if plotting fails
                logger.info(f"Distillation log/plot skipped: {_e}")

        self.params = BEST_MODEL_PARAMS

    def _save_distill_logs_and_plot(self):
        """Persist best/worst per-epoch logs and save a 3D scatter plot.
        Axes: X=epoch, Y=loss, Z=graph_id. Two series: best (green) and worst (red).
        """
        # Persist raw logs
        logs = {
            'best': self.distill_best_log,
            'worst': self.distill_worst_log,
        }
        save_object(logs, 'distill_best_worst.pkl', self.dir_log)

        if len(self.distill_best_log) == 0 and len(self.distill_worst_log) == 0:
            return

        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

        # Prepare arrays for plotting
        bx = [d['epoch'] for d in self.distill_best_log]
        by = [d['loss'] for d in self.distill_best_log]
        bz = [d['graph_id'] for d in self.distill_best_log]

        wx = [d['epoch'] for d in self.distill_worst_log]
        wy = [d['loss'] for d in self.distill_worst_log]
        wz = [d['graph_id'] for d in self.distill_worst_log]

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        if len(bx) > 0:
            ax.scatter(bx, by, bz, c='green', marker='o', label='best')
        if len(wx) > 0:
            ax.scatter(wx, wy, wz, c='red', marker='^', label='worst')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_zlabel('Graph ID')
        ax.set_title('Distillation Best/Worst per Epoch')
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.dir_log / 'distill_best_worst_3d.png')
        plt.close('all')

    def plot_area_parameter(self, epochs_list, ids, sinisters):
        """
        Plot self.area_parameter (2D array: epochs x parameters) in 3D.
        X = epochs_list
        Y = parameter index
        Z = value of area_parameter
        """
    
        from mpl_toolkits.mplot3d import Axes3D

        sort_ids = np.argsort(sinisters)

        # Vérifier dimensions
        area_param = np.asarray([params.detach().numpy()[sort_ids] for params in self.area_parameters_log])  # doit être (epochs, n_params)
        
        logger.info(f'Last aera params -> {area_param[-1]}')

        # Créer grilles X et Y
        X = epochs_list
        Y = sinisters[sort_ids]
        X, Y = np.meshgrid(X, Y)
        
        # Z = valeurs de self.area_parameter transposées pour correspondre à la grille
        Z = area_param.T  
        
        # Plot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis')
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Parameter Index')
        ax.set_zlabel('Area Parameter Value')
        ax.set_title('Evolution of Area Parameter during Training')
        
        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.savefig(self.dir_log / 'area_parameters.png')
        plt.close()

    def split_dataset(self, dataset, nb, reset=True):
        # Separate the positive and zero classes based on y
        positive_mask = dataset[self.target_name] > 0
        non_fire_mask = dataset[self.target_name] == 0

        # Filtrer les données positives et non feu
        df_positive = dataset[positive_mask]
        df_non_fire = dataset[non_fire_mask]

        # Échantillonner les données non feu
        nb = min(len(df_non_fire), nb)

        if self.n_run == 1:
            seed = self.seed if self.seed is not None else 42
            sampled_indices = np.random.RandomState(seed).choice(len(df_non_fire), nb, replace=False)
        else:
            sampled_indices = np.random.RandomState().choice(len(df_non_fire), nb, replace=False)
            
        df_non_fire_sampled = df_non_fire.iloc[sampled_indices]

        # Combiner les données positives et non feu échantillonnées
        df_combined = pd.concat([df_positive, df_non_fire_sampled])
        # Réinitialiser les index du DataFrame combiné
        if reset:
            df_combined.reset_index(drop=True, inplace=True)
        return df_combined
    
    def add_ordinal_class(self, X, y, limit):        
        pass

    def calculcate_score(self, pred, y, id_mask=None):
        if id_mask is None:
            under_prediction_score_value = under_prediction_score(y, pred)
            over_prediction_score_value = over_prediction_score(y, pred)

            iou = iou_score(y, pred)
            return under_prediction_score_value, over_prediction_score_value, iou
        else:
            uids = np.unique(id_mask)
            under_prediction_score_value = []
            over_prediction_score_value = []
            iou = []
            
            for id in uids:
                mask = (id_mask == id)
                pred_mask = pred[mask]
                y_mask = y[mask]
                
                if np.any(y_mask > 0):

                    under_score = under_prediction_score(y_mask, pred_mask)
                    over_score = over_prediction_score(y_mask, pred_mask)
                    iou_val = iou_score(y_mask, pred_mask)

                    # Stocker les valeurs
                    under_prediction_score_value.append(under_score)
                    over_prediction_score_value.append(over_score)
                    iou.append(iou_val)

                    # Log propre
                    logger.info(
                        f'Id {id} -> under_prediction_score: {under_score}, over_prediction_score: {over_score}, iou: {iou_val}'
                    )
                
                else:
                     logger.info(
                        f'Id {id} -> No fire'
                    )

            under_prediction_score_value = np.trapz(under_prediction_score_value)
            over_prediction_score_value = np.trapz(over_prediction_score_value)
            iou = np.trapz(iou)

            return under_prediction_score_value, over_prediction_score_value, iou

    def compute_area_score(self, pred, y_true, graph_ids):
        unique_graphs = np.unique(graph_ids)
        graph_sums = {gid: y_true[graph_ids == gid].sum() for gid in unique_graphs}
        sorted_graphs = sorted(graph_sums, key=graph_sums.get, reverse=True)

        iou_scores = []
        f1_scores = []

        for gid in sorted_graphs:
            mask = graph_ids == gid
            y_g = y_true[mask]
            pred_g = pred[mask]
            if np.any(y_g > 0):
                iou = jaccard_score((y_g > 0).astype(int), (pred_g > 0).astype(int), zero_division=0)
                f1 = f1_score((y_g > 0).astype(int), (pred_g > 0).astype(int), zero_division=0)
                iou_scores.append(iou)
                f1_scores.append(f1)

        if len(iou_scores) == 0:
            return 0.0, 0.0

        max_area = np.trapz(np.ones(np.unique(graph_ids[y_true > 0]).shape[0]))
        IoU_area = calculate_area_under_curve(iou_scores)
        F1_area = calculate_area_under_curve(f1_scores)
        if max_area == 0:
            return 0, 0
        return IoU_area / max_area, F1_area / max_area

    def search_samples_proportion(self, graph, df_train, df_val, df_test, is_unknowed_risk, epochs, PATIENCE_CNT, CHECKPOINT, reset=True, custom_model_params=None, use_log=True):
        
        check_and_create_path(self.dir_log)

        if not is_unknowed_risk:
                test_percentage = np.round(np.arange(0.1, 1.05, 0.05), 2)
        else:
            test_percentage = np.arange(0.0, 1.05, 0.05)

        if 'MultiScale' in self.model_name:
            test_percentage = np.arange(0.5, 1.05, 0.05)

        under_prediction_score_scores = []
        over_prediction_score_scores = []
        iou_scores = []
        data_log = None
        find_log = False

        self.metrics['test_percentage'] = []
        self.metrics['under_prediction_scores'] = []
        self.metrics['over_predictio_scores'] = []
        self.metrics['iou_scores'] = []

        if use_log:
        #if False:
            if False:
                if (self.dir_log / 'unknowned_scores_per_percentage.pkl').is_file():
                    data_log = read_object('unknowned_scores_per_percentage.pkl', self.dir_log)
            else:
                if (self.dir_log / 'metrics.pkl').is_file():
                    print(f'Load metrics')
                    find_log = True
                    data_log = read_object('metrics.pkl', self.dir_log)
                else:
                    xs = [0, 10]
                    for x in xs:
                        other_model = f'{self.model_name}_search_full_{x}_all_one_{self.target_name}_{self.task_type}_{self.loss}'
                        print(f'{self.dir_log / ".."/ other_model / "metrics.pkl"}')
                        if (self.dir_log / '..'/ other_model / 'metrics.pkl').is_file():
                            data_log = read_object('metrics.pkl', self.dir_log / '..'/ other_model)
                        if data_log is not None:
                            break
                        
                    if data_log is None:
                        xs = [25]
                        for x in xs:
                            other_model = f'{self.model_name}_search_full_{self.ks}_{x}_one_{self.target_name}_{self.task_type}_{self.loss}'
                            print(f'{self.dir_log / ".."/ other_model / "metrics.pkl"}')
                            if (self.dir_log / '..'/ other_model / 'metrics.pkl').is_file():
                                data_log = read_object('metrics.pkl', self.dir_log / '..'/ other_model)
                            if data_log is not None:
                                break

        print(f'data_log : {data_log}')
        if data_log is not None:
            try:
                self.metrics = data_log
                #test_percentage = self.metrics['test_percentage']
                #under_prediction_score_scores = self.metrics['under_prediction_scores']
                #over_prediction_score_scores = self.metrics['over_prediction_scores']
            except Exception as e:
                print(e)
                self.metrics = {}
                data_log = None
                pass

        doSearch = True
        if data_log is not None: #and self.n_run == data_log['n_run']:
            test_percentage = np.asarray(self.metrics['test_percentage'])
            start_test = -1
            for i in range(0, len(test_percentage) - 1):
                start_test = i

                if test_percentage[i] in self.metrics.keys():
                    last_keys = test_percentage[i]
                    val_1 = np.mean(data_log[test_percentage[i]]['iou_val'])
                    val_2 = np.mean(data_log[test_percentage[i + 1]]['iou_val'])

                    std_1 = np.std(data_log[test_percentage[i]]['iou_val'])
                    std_2 = np.std(data_log[test_percentage[i + 1]]['iou_val'])

                    print('#########################"')
                    print(f'{test_percentage[i]} -> {val_1} -> {std_1}')
                    print(f'{test_percentage[i + 1]} -> {val_2} -> {std_2}')

                    try:
                        if  val_1 >  val_2 or ((val_1 == val_2) and (std_1 < std_2)):
                            print(f"Last score {val_1} current score {val_2}")
                            print(f"Last std {std_1} current std {std_2}")
                            doSearch = False
                            tp = test_percentage[i]
                            break
                    except Exception as e:
                        print(e)
                        doSearch = True
                        break

            start_test += 1

        else:
            start_test = 0
        
        if doSearch:
            last_score = -math.inf if start_test == 0 else np.mean(self.metrics[last_keys]['iou_val'])
            y_ori = df_train[self.target_name].values
            for i in range(start_test, test_percentage.shape[0]):
                tp = round(test_percentage[i], 2)

                if tp in self.metrics.keys():
                    continue
                
                df_train_copy = df_train.copy(deep=True)
                
                if not is_unknowed_risk:
                    nb = int(tp * y_ori[y_ori == 0].shape[0])
                else:
                    nb = int(tp * len(X[(X['potential_risk'] > 0) & (y_ori == 0)]))

                logger.info(f'Trained with {tp} -> {nb} sample of class 0')

                for run in range(self.n_run):

                    df_combined = self.split_dataset(df_train_copy, nb, reset=False)

                    # Mettre à jour df_train pour l'entraînement
                    df_train_copy['weight'] = 0
                    df_train_copy.loc[df_combined.index, 'weight'] = 1

                    copy_model = deepcopy(self)
                    copy_model.under_sampling = 'full'
                    copy_model.create_train_val_test_loader(graph, df_train_copy, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT, features_importance=False, custom_model_params=custom_model_params)
                    copy_model.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose=False, custom_model_params=custom_model_params)
                    
                    ############################# On set val ##############################
                    test_output, y = copy_model._predict_test_loader(copy_model.val_loader, output_pdf='Val')

                    prediction = test_output.detach().cpu().numpy()
                    
                    y = y.detach().cpu().numpy()
                    if 'MultiScale' in self.model_name:
                        id_mask = y[:, scale_index]
                    else:
                        id_mask = y[:, departement_index]
                        id_mask = None
                
                    dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
                    dff['departement'] = y[:, departement_index]
                    dff[self.target_name] = y[:, -1]
                    y = y[:, -1]

                    metrics_run = evaluate_metrics(dff, self.target_name, prediction)
                    metrics_run = round_floats(metrics_run)
                    under_prediction_score_value = under_prediction_score(y, prediction)
                    over_prediction_score_value = over_prediction_score(y, prediction)
                    update_metrics_as_arrays(self, tp, metrics_run, 'val')

                    ############################# On set test ##############################
                    test_output, y = copy_model._predict_test_loader(copy_model.test_loader, output_pdf='test')


                    prediction = test_output.detach().cpu().numpy()
                    y = y.detach().cpu().numpy()

                    if 'MultiScale' in self.model_name:
                        id_mask = y[:, scale_index]
                    else:
                        id_mask = y[:, departement_index]
                        id_mask = None
                
                    dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
                    dff['departement'] = y[:, departement_index]
                    dff[self.target_name] = y[:, -1]
                    y = y[:, -1]

                    metrics_run = evaluate_metrics(dff, self.target_name, prediction)
                    metrics_run = round_floats(metrics_run)
                    update_metrics_as_arrays(self, tp, metrics_run, 'test')
                
                self.metrics[tp] = add_ic95_to_dict(self.metrics[tp], None, "_ic95")

                iou = np.mean(self.metrics[tp]['iou_val'])
                std_iou = np.std(self.metrics[tp]['iou_val'])

                save_object(self.metrics, 'metrics.pkl', self.dir_log)
                
                print(f'Metrics achieved : {self.metrics[tp]}')

                if iou >= last_score:
                    last_score = iou
                else:
                    print(f'Last score {last_score} current score {iou}')
                    break
        
        keys_array = []
        eps = 1e-9  # tolérance pour considérer deux moyennes égales

        iou_means = []
        iou_stds = []
        for k in self.metrics.keys():
            if isinstance(k, float) and "iou_val" in self.metrics[k]:
                vals = np.asarray(self.metrics[k]["iou_val"], dtype=float)
                mean_iou = float(np.nanmean(vals)) if vals.size else -np.inf
                std_iou = np.std(vals)

                print(f"{k} -> mean={mean_iou:.6f}, std={std_iou:.6f}")
                keys_array.append(k)
                iou_means.append(mean_iou)
                iou_stds.append(std_iou)

        if not keys_array:
            raise ValueError("Aucune clé float avec 'iou_val' trouvée dans self.metrics.")

        # Sélection avec tie-break: max(mean), puis min(std)
        iou_means = np.array(iou_means, dtype=float)
        iou_stds  = np.array(iou_stds,  dtype=float)

        max_mean = np.nanmax(iou_means)
        candidates = np.where(np.isclose(iou_means, max_mean, atol=eps))[0]
        if candidates.size == 1:
            index_max = int(candidates[0])
        else:
            # parmi les ex aequo en moyenne, prendre le plus petit std
            index_max = int(candidates[np.nanargmin(iou_stds[candidates])])

        best_tp = keys_array[index_max]
        logger.info(f'Best tp {best_tp}')
        self.metrics['iou_score'] = iou_scores
        self.metrics['test_percentage'] = test_percentage
        self.metrics['under_prediction_scores'] = under_prediction_score_scores
        self.metrics['over_prediction_scores'] = over_prediction_score_scores
        self.metrics['best_tp'] = best_tp
        self.metrics['run'] = self.n_run

        logger.info(f'{self.metrics[best_tp]}')

        save_object(self.metrics, 'metrics.pkl', self.dir_log)

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
        predictions, y = self.predict(X, return_y=True)
        y = y[:, -1]
        return self.score_with_prediction(predictions, y, sample_weight)

    def score_with_prediction(self, y_pred, y, sample_weight=None):
        
        return iou_score(y, y_pred)

    def _predict_test_loader(self, X: DataLoader, prediction_type='Class', output_pdf="test") -> torch.tensor:
            assert self.model is not None
            self.model.eval()
            if len(self.criterion_params) > 0:
                criterion = self.get_loss(self.loss)
                criterion.update_params(self.criterion_params[self.best_epoch])
                criterion.eval()

            with torch.no_grad():
                pred = []
                y = []

                for i, data in enumerate(X, 0):
                    
                    inputs, orilabels, _ = data

                    orilabels = orilabels.to(device)
                    orilabels = orilabels[:, :, -1]

                    print(torch.unique(orilabels[:, weight_index]))

                    #labels = compute_labels(orilabels, self.model.is_graph_or_node, graphs)

                    inputs_model = inputs
                    output, logits, hidden = self.model(inputs_model)

                    if 'criterion' in locals() and hasattr(criterion, 'transform'):
                        params = {'inputs' : logits}
                        if 'cluster_ids' in required_params(criterion.transform):
                            cluster_ids = orilabels[:, self.cluster_id_index].long()
                            params['cluster_ids'] = cluster_ids
                        if 'output_pdf' in required_params(criterion.transform):
                            assert output_pdf is not None and self.dir_log is not None
                            params['output_pdf'] = output_pdf
                            params['dir_output'] = self.dir_log

                        output = criterion.transform(**params)

                    if prediction_type == 'Class':
                        if self.task_type == 'classification' or self.task_type == 'binary':
                            output = torch.argmax(output, dim=1)
                        elif self.task_type == 'regression' and output.ndim > 1 and output.shape[1] > 1:
                            print(torch.max(output[:, 0]))
                            print(torch.max(output[:, 1]))
                            print(torch.max(output[:, 2]))
                            print(torch.max(output[:, 3]))
                            print(torch.max(output[:, 4]))
                            output = torch.argmax(output, dim=1)
                    elif prediction_type == 'RawFormulaVal':
                        output = logits

                    #output = output[weights.gt(0)]

                    pred.append(output)
                    y.append(orilabels)

                y = torch.cat(y, 0)
                pred = torch.cat(pred, 0)

                #if self.task_type == 'regression' and prediction_type == 'Class':
                if pred.dtype != torch.long:
                    pred = torch.round(pred, decimals=1)

                return pred, y

    def fit(self, graph, X, y, X_val, y_val, X_test, y_test, PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params=None, use_log=True):
        
        X = X.set_index(ids_columns[:-1]).join(y.set_index(ids_columns[:-1])[targets_columns + [self.target_name]], on=ids_columns[:-1], how='left').reset_index()
        X_val = X_val.set_index(ids_columns[:-1]).join(y_val.set_index(ids_columns[:-1])[targets_columns + [self.target_name]], on=ids_columns[:-1], how='left').reset_index()
        X_test = X_test.set_index(ids_columns[:-1]).join(y_test.set_index(ids_columns[:-1])[targets_columns  + [self.target_name]], on=ids_columns[:-1], how='left').reset_index()
        #if (self.dir_log / 'last.pt').is_file():
        #    self.graph = graph
        #    self._load_model_from_path(self.dir_log / 'best.pt', self.model)
        #else:
        self.create_train_val_test_loader(graph, X, X_val, X_test, epochs, PATIENCE_CNT, CHECKPOINT, custom_model_params=custom_model_params, use_log=use_log)
        self.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params=custom_model_params)
            
    def filtering_pred(self, df, predTensor, y, graph, return_y = False):
        
        y = y.detach().cpu().numpy()
        predTensor = predTensor.detach().cpu().numpy()

        # Extraire les paires de test_dataset_dept
        test_pairs = set(zip(df['date'], df['graph_id'], df['scale']))

        # Normaliser les valeurs dans YTensor
        date_values = [item for item in y[:, date_index]]
        graph_id_values = [item for item in y[:, graph_id_index]]
        scale_values = [item for item in y[:, scale_index]]

        # Filtrer les lignes de YTensor correspondant aux paires présentes dans test_dataset_dept
        filtered_indices = [
            i for i, (date, graph_id, scale) in enumerate(zip(date_values, graph_id_values, scale_values)) 
            if (date, graph_id, scale) in test_pairs
            ]

        # Créer YTensor filtré
        y = y[filtered_indices]

        # Créer des paires et les convertir en set
        ytensor_pairs = set(zip(date_values, graph_id_values, scale_values))

        # Filtrer les lignes en vérifiant si chaque couple (date, graph_id) appartient à ytensor_pairs
        df = df[
            df.apply(lambda row: (row['date'], row['graph_id'], row['scale']) in ytensor_pairs, axis=1)
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

        df.sort_values(['graph_id', 'date', 'scale'], inplace=True)
        ind = np.lexsort((y[:, scale_index], y[:,0], y[:,4]))
        y = y[ind]
        predTensor = predTensor[ind]
        
        if self.target_name == 'binary' or self.target_name == 'nbsinister':
            band = -2
        else:
            band = -1

        pred = np.full((predTensor.shape[0], 1), fill_value=np.nan)
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
                    pred[mask[:, 0]] = pred_2D[mask_2D[:, 0], band, mask_2D[:, 1], mask_2D[:, 2]]
        else:
            pred = predTensor

        if return_y:
            return pred, y
        return pred

    def predict(self, df, graph=None, return_y=False, prediction_type='Class'):
        if graph is None and isinstance(self, ModelGNN):
            graph = self.graph

        if self.target_name not in list(df.columns):
            df[self.target_name] = 0

        loader = create_test_loader(graph, df,
                       self.features_name,
                       self.device,
                       None,
                       self.target_name,
                       self.ks)
        
        predTensor, YTensor = self._predict_test_loader(loader, prediction_type='Class')

        if return_y:
        #    pred, y = self.filtering_pred(df, predTensor, YTensor, graph, return_y=return_y)
            return predTensor, YTensor
        
        #pred = self.filtering_pred(df, predTensor, YTensor, graph, return_y=return_y)
        return predTensor
    
    def predict_proba(self, df, graph=None, return_y=False):
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
        if return_y:
            pred, y = self.filtering_pred(df, predTensor, YTensor, graph, return_y=return_y)
            return pred, y
        pred = self.filtering_pred(df, predTensor, YTensor, graph, return_y=return_y)
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
        model, _ = self.make_model(self.graph, None)
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
    
    def update_model(self, model):
        self.model = deepcopy(model)

    def get_loss(self, loss_name):
        loss_params = {'num_classes' : 5}
        return get_loss_function(loss_name, **loss_params)

    def get_learnable_parameters(self, criterion):
        """
        Retourne les paramètres apprenables (list/param groups) pour l'optimizer.
        Tous les objets doivent être nn.Parameter avec requires_grad=True.
        """
        params = list(self.model.parameters())

        # Ajouter paramètres spécifiques à la loss (s'ils existent)
        if has_method(criterion, 'get_learnable_parameters'):
            logger.info(f'Adding {self.loss} parameter(s)')
            # On s'assure que ce sont bien des nn.Parameter
            loss_params = []
            for p in criterion.get_learnable_parameters().values():
                if isinstance(p, torch.nn.Parameter):
                    loss_params.append(p)
                else:
                    # Convertir un tensor en Parameter si besoin
                    loss_params.append(torch.nn.Parameter(p, requires_grad=True))
            params.extend(loss_params)

        # Ajouter les area_parameters si la loss l'exige
        if 'learnable-area' in self.loss:
            logger.info("Adding learnable area parameters")
            # self.area_parameters est déjà nn.Parameter
            params.append(self.area_parameters)

        if self.student_train and self.temperature == 'seach':
            params.append(self.temperature_value)

        if self.student_train and self.alpha == 'seach':
            params.append(self.alpha_value)

        return params

    def get_optimizer(self, criterion,):
        parameters = self.get_learnable_parameters(criterion)
        optimizer = optim.Adam(parameters, lr=self.lr)
        return optimizer

    def shapley_additive_explanation(self, df, outname, dir_output, mode='bar', figsize=(50, 25), samples=None, samples_name=None):
        """
        Visualisation des valeurs SHAP pour expliquer les prédictions.
        :param df_set: DataFrame des caractéristiques d'entrée.
        :param outname: Nom de sortie pour le fichier d'image.
        :param dir_output: Répertoire où enregistrer les résultats.
        :param mode: Mode de visualisation ('bar' ou 'beeswarm').
        :param figsize: Taille de la figure.
        :param samples: Échantillons spécifiques à analyser.
        :param samples_name: Noms des échantillons à afficher.
        """
        if hasattr(self, 'use_temporal_as_edges'):
            use_temporal_as_edges = self.use_temporal_as_edges
        else:
            use_temporal_as_edges = None

        Xst, e = get_numpy_data(self.graph, df, self.features_name, use_temporal_as_edges, self.ks)
        Xst = torch.Tensor(Xst).to(self.device)

        B, F, T = Xst.shape

        Xst_flat = Xst.reshape((B, F*T))
        df_features = []
        # SHAP DeepExplainer avec wrapper du modèle
        explainer = shap.DeepExplainer(WrapperModel(self.model, F, T, e).to(self.device), Xst_flat)
        shap_values = explainer.shap_values(Xst_flat)

        n_classes = self.out_channels

        # Vérifier si la sortie SHAP est multi-classes
        if n_classes == 1:
            shap_values = shap_values[:, :, np.newaxis]
        
        shap_values = np.asarray(shap_values)
        shap_values = np.reshape(shap_values, (n_classes, B, F, T))
        shap_values = shap_values[:, :, :, -1]
        shap_values = np.moveaxis(shap_values, 0, 2)
        #shap_values = shap_values.values

        # Pour chaque classe, calculer et sauvegarder les résultats SHAP
        for class_idx in range(n_classes):
            # Calcul des valeurs SHAP moyennes et écarts-types
            shap_mean_abs = np.mean(np.abs(shap_values[:, :, class_idx]), axis=0)
            shap_std_abs = np.std(np.abs(shap_values[:, :, class_idx]), axis=0)
        
            df_shap = pd.DataFrame({
                "mean_abs_shap": shap_mean_abs,
                "stdev_abs_shap": shap_std_abs,
                "name": self.features_name
            }).sort_values("mean_abs_shap", ascending=False)

            df_shap['class'] = class_idx
            df_features.append(df_shap)

            # Visualisation globale (summary_plot) pour chaque classe
            plt.figure(figsize=figsize)
            """if mode == 'bar':
                shap.summary_plot(
                    shap_values[:, :, class_idx],
                    features=Xst_flat, 
                    feature_names=self.features_name,
                    plot_type='bar',
                    show=False
                )
            elif mode == 'beeswarm':
                #print(shap_values[:, :, class_idx].shape, df.values.shape, len(self.features_name))
                fig, ax = plt.subplots(figsize=(10, 6))

                # Générer le graphique SHAP pour une classe spécifique (class_idx)
                shap.summary_plot(
                    shap_values[:, :, class_idx],
                    features=Xst_flat,
                    feature_names=self.features_name,
                    show=False,
                    plot_type="dot",  # Vous pouvez choisir 'dot', 'bar', ou 'violin' comme type de plot
                    ax=ax
                )

                # Ajouter explicitement la colorbar
                plt.colorbar(ax.collections[0], ax=ax)
                """
                #plt.show()

            #print(dir_output / f"{outname}_class_{class_idx}_shapley.png")
            #plt.savefig(dir_output / f"{outname}_class_{class_idx}_shapley.png")
            #plt.close()

            # Visualisations spécifiques aux échantillons (force_plot)
            if samples is not None and samples_name is not None:

                for i, sample in enumerate(samples):
                    plt.figure(figsize=figsize)
                    shap.force_plot(
                        explainer.expected_value[class_idx],
                        shap_values[sample, :, class_idx],
                        features=df.iloc[sample].values,
                        feature_names=self.features_name,
                        matplotlib=True,
                        show=False
                    )

                    plt.savefig(
                        dir_output / f"{outname}_class_{class_idx}_{samples_name[i]}_shapley.png",
                        bbox_inches='tight'
                    )
                    plt.close()

        df_features = pd.concat(df_features)
        save_object(df_features, 'features_importance.pkl', dir_output)

############################################ Split training ##############################################################

class SplitTraining(Training):
    def __init__(self, federated_cluster, cut_layer_name, input_server_model, model_name,
                 nbfeatures, batch_size, lr, target_name, task_type, out_channels,
                 dir_log, features_name, ks, loss, name, device, under_sampling, over_sampling, n_run,
                 horizon=1):

        super().__init__(model_name, nbfeatures, batch_size, lr, target_name, task_type, features_name, ks,
                         out_channels, dir_log, loss=loss, name=name, device=device, under_sampling=under_sampling,
                         over_sampling=over_sampling, n_run=n_run, horizon=horizon)

        self.federated_cluster = federated_cluster
        self.cut_layer_name = cut_layer_name
        self.input_server_model = input_server_model

        self.horizon = horizon

    def create_client_model_upto_cut_layer(self, model):
        """
        Crée un sous-modèle client contenant les couches jusqu'à (non inclus) la couche de découpe.
        """
        from torch import nn

        layers = []
        for name, layer in model.named_children():
            if name == self.cut_layer_name:
                break
            layers.append(layer)

        client_model = nn.Sequential(*layers)
        return client_model

    def create_server_model_from_cut_layer(self, model):
        """
        Crée un sous-modèle serveur contenant les couches à partir de la couche de découpe (incluse).
        """
        from torch import nn

        start_adding = False
        layers = []

        for name, layer in model.named_children():
            if name == self.cut_layer_name:
                start_adding = True
            if start_adding:
                layers.append(layer)
        
        server_model = nn.Sequential(*layers)
        return server_model

    def initialize_clients(self, base_model, clusters, learning_rate=1e-3):
        client_models = {}
        client_optimizers = {}

        for cluster in clusters:
            model = deepcopy(base_model)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            client_models[cluster] = model
            client_optimizers[cluster] = optimizer
        
        return client_models, client_optimizers
    
    def initialize_server(self, model, learning_rate=1e-3):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        return model, optimizer

    def prepare_batch_data(self, df_train, graph, clusters, batch_size):
        batch_data = []

        for cluster in clusters:
            df_c = df_train[df_train[self.federated_cluster] == cluster]

            train_dataset = create_train_dataset(graph, df_c,
                                                 self.features_name,
                                                 self.target_name,
                                                 None, self.device, 
                                                 self.ks, False, '')

            loader = DataLoader(train_dataset, train_dataset.__len__(),
                                False, worker_init_fn=seed_worker,
                                generator=g)

            batch_data.append((cluster, loader))
            
        return batch_data
    
    def clients_forward(self, batch_data, client_models):
        activations = []
        inputs_for_backward = {}
        labels_list = []

        for cluster, batch in batch_data:
            model = client_models[cluster]
            model.train()
            for data in batch:
                inputs, labels, edges = data
                graphs = None

                if inputs.shape[0] == 1:
                    return 0

                band = -1
                
                try:
                    target, weights = self.compute_weights_and_target(labels, band, ids_columns, model.is_graph_or_node, graphs)
                except Exception as e:
                    target, weights = self.compute_weights_and_target(labels, band, ids_columns, False, graphs)
                
                if self.loss not in ['kldivloss']: # works on probability
                    target = target.long()
                
                output = model(inputs)

                output.retain_grad()

                activations.append(output)
                inputs_for_backward[cluster] = (inputs, output)
                labels_list.append(labels)

        return activations, inputs_for_backward, labels
    
    def server_forward_backward(self, server_model, server_optimizer, activations, labels, criterion):
        server_model.train()
        server_optimizer.zero_grad()

        concat = torch.cat(activations, dim=1)
        print(f'concat : {concat.shape}')
        output = server_model(concat)
        loss = criterion(output, labels)
        loss.backward()

        server_optimizer.step()
        return loss.item(), concat.grad

    def clients_backward_update(self, inputs_for_backward, grad_concat, client_models, client_optimizers):
        split_sizes = [out.shape[1] for _, out in inputs_for_backward.values()]
        grads = torch.split(grad_concat, split_sizes, dim=1)

        for (cluster, (X_batch, out)), grad in zip(inputs_for_backward.items(), grads):
            optimizer = client_optimizers[cluster]
            model = client_models[cluster]
            optimizer.zero_grad()
            out.backward(grad)
            optimizer.step()

    def train_split(self, df_train, df_val, df_test, graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose=True, custom_model_params=None):
        
        import torch
        from torch import nn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.test_loader = create_test_loader(graph, df_test,
                            self.features_name,
                            self.device,
                            False,
                            self.target_name,
                            self.ks,
                            False,
                            '')

        clusters = df_train[self.federated_cluster].unique()
        batch_size = self.batch_size
        
        lr = self.lr
        patience = PATIENCE_CNT

        model, _ = make_model(f'{self.model_name}CutClient', len(self.features_name), len(self.features_name),
                                graph, dropout, activation,
                                self.ks,
                                out_channels=self.out_channels,
                                task_type=self.task_type,
                                device=device, num_lstm_layers=num_lstm_layers,
                                custom_model_params=custom_model_params)

        client_models, client_optimizers = self.initialize_clients(model, clusters, lr)
        model, _ = make_model(f'{self.model_name}CutServer', self.input_server_model, len(self.features_name),
                                graph, dropout, activation,
                                self.ks,
                                out_channels=self.out_channels,
                                task_type=self.task_type,
                                device=device, num_lstm_layers=num_lstm_layers,
                                custom_model_params=custom_model_params)
        
        server_model, server_optimizer = self.initialize_server(model, lr)

        criterion = self.get_loss(self.loss)

        best_loss = float('inf')
        best_server_state = None
        patience_counter = 0
        
        batch_data = self.prepare_batch_data(df_train, graph, clusters, batch_size)
        val_batch_data = self.prepare_batch_data(df_val, graph, clusters, batch_size)
        
        val_num_batches = len(val_batch_data)
        num_batch = len(batch_data)

        for epoch in range(epochs):
            if verbose:
                logger.info(f"\nEpoch {epoch+1}/{epochs}")

            epoch_loss = 0

            activations, inputs_for_backward, labels = self.clients_forward(batch_data, client_models)
            loss_value, grad_concat = self.server_forward_backward(server_model, server_optimizer, activations, labels, criterion)
            self.clients_backward_update(inputs_for_backward, grad_concat, client_models, client_optimizers)
            epoch_loss += loss_value

            avg_loss = epoch_loss / num_batch

            # Compute validation loss for early stopping
            val_loss_total = 0
            activations, _, labels = self.clients_forward(val_batch_data, client_models)
            server_model.eval()
            with torch.no_grad():
                concat = torch.cat(activations, dim=1)
                output = server_model(concat)
                vloss = criterion(output, labels)
            val_loss_total += vloss.item()
            val_loss = val_loss_total / val_num_batches

            if epoch % CHECKPOINT and verbose:
                logger.info(f"Avg Epoch Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_server_state = deepcopy(server_model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience and verbose:
                    logger.info("Early stopping triggered.")
                    break

        if best_server_state is not None:
            server_model.load_state_dict(best_server_state)

        self.client_models = client_models
        self.model = server_model
        self.is_fitted_ = True

        test_output, y = self._predict_test_loader(self.test_loader)
        test_output = test_output.detach().cpu().numpy()

        y = y.detach().cpu().numpy()

        under_prediction_score_value = under_prediction_score(y[:, -1], test_output)
        over_prediction_score_value = over_prediction_score(y[:, -1], test_output)

        iou = iou_score(y[:, -1], test_output)
        f1 = f1_score((test_output > 0).astype(int), (y[:, -1] > 0).astype(int), zero_division=0)
        iou_area, f1_area = self.compute_area_score(test_output, y[:, -1], y[:, graph_id_index])

        logger.info(f'Test -> Under achieved : {under_prediction_score_value}, Over achived {over_prediction_score_value}, IoU {iou}, f1 {f1}, IoU_area {iou_area}, f1_area {f1_area}')

        test_output, y = self._predict_test_loader(self.val_loader)
        test_output = test_output.detach().cpu().numpy()
        
        y = y.detach().cpu().numpy()

        under_prediction_score_value = under_prediction_score(y[:, -1], test_output)
        over_prediction_score_value = over_prediction_score(y[:, -1], test_output)
        
        iou = iou_score(y[:, -1], test_output)
        f1 = f1_score((test_output > 0).astype(int), (y[:, -1] > 0).astype(int))
        iou_area, f1_area = self.compute_area_score(test_output, y[:, -1], y[:, graph_id_index])

        logger.info(f'Val -> Under achieved : {under_prediction_score_value}, Over achived {over_prediction_score_value}, IoU {iou} f1 {f1}, IoU_area {iou_area}, f1_area {f1_area}')

        plt.figure(figsize=(15,5))
        plt.plot(y[y[:, departement_index] == 13, -1])
        plt.plot(test_output[y[:, departement_index] == 13])
        plt.savefig(self.dir_log / 'test.png')
        plt.close('all')

        self.update_weight(server_model.state_dict())

    def _predict_test_loader(self, X: DataLoader, prediction_type='Class', output_pdf="test", proba=False) -> torch.tensor:

        """Generate predictions using the split learning setup."""

        try:
            if self.training_mode == 'normal':
                return super()._predict_test_loader(X, prediction_type=prediction_type, output_pdf=output_pdf)
        except:
                return super()._predict_test_loader(X, prediction_type=prediction_type, output_pdf=output_pdf)

        if not hasattr(self, "server_model") or not hasattr(self, "client_models"):
            raise ValueError("Model is not fitted. Please train the model before predicting.")

        self.server_model.eval()
        for model in self.client_models.values():
            model.eval()

        preds = []
        ys = []
        activations = []

        with torch.no_grad():
            for (cluster, data) in X:
                model = client = self.client_models.get(cluster)
                inputs, labels, edges = data
                labels = labels.to(self.device)
                labels_last = labels[:, :, -1]

                activation = client(inputs)
                activations.append(activation)

                ys.append(labels_last)

            output = self.model(activations)
            if output.shape[1] > 1 and not proba:
                output = torch.argmax(output, dim=1)

                preds.append(output.squeeze(0))

        pred_tensor = torch.stack(preds, 0)
        y_tensor = torch.stack(ys, 0)

        if self.target_name in ["binary", "risk", "nbsinister"]:
            pred_tensor = torch.round(pred_tensor, decimals=1)

        return pred_tensor, y_tensor

    def predict(self, df, graph=None, return_y=False):
        
        try:
            if self.training_mode == 'normal':
                return super().predict(df, graph=graph, return_y=return_y)
        except:
                return super().predict(df, graph=graph, return_y=return_y)
        
        if graph is None:
            graph = self.graph

        if self.target_name not in list(df.columns):
            df[self.target_name] = 0

        loader = self.prepare_batch_data(df, graph, df[self.federated_cluster].unique(), 1)

        pred_tensor, y_tensor = self._predict_test_loader(loader, output_pdf="test")

        if return_y:
            return pred_tensor, y_tensor

        return pred_tensor

    def predict_proba(self, df, graph=None, return_y=False):
        try:
            if self.training_mode == 'normal':
                return super().predict_proba(df, graph=graph, return_y=return_y)
        except:
                return super().predict_proba(df, graph=graph, return_y=return_y)
        
        if graph is None:
            graph = self.graph

        if self.target_name not in list(df.columns):
            df[self.target_name] = 0

        loader = create_test_loader(
            graph,
            df,
            self.features_name,
            self.device,
            None,
            self.target_name,
            self.ks,
        )

        pred_tensor, y_tensor = self._predict_test_loader(loader, True, output_pdf="test")

        if return_y:
            pred, y = self.filtering_pred(df, pred_tensor, y_tensor, graph, return_y=True)
            return pred, y

        pred = self.filtering_pred(df, pred_tensor, y_tensor, graph, return_y=False)
        return pred

    def search_samples_proportion_per_cluster(self, graph, df_train, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT):
        """Search optimal zero sample limits for each cluster.

        Parameters
        ----------
        graph : dgl.DGLGraph
            Graph used for training.
        df_train, df_val, df_test : pandas.DataFrame
            Datasets containing a ``self.federated_cluster`` column.
        args : dict, optional
            Additional arguments forwarded to :func:`train_split`.

        Returns
        -------
        dict
            Mapping cluster id to chosen zero count.
        float
            IoU score obtained with the best combination.
        """

        check_and_create_path(self.dir_log)
        self.metrics = read_object('metrics_cluster.pkl', self.dir_log) or {}
        scores = self.metrics.get('scores', {})

        clusters = df_train[self.federated_cluster].unique()
        best_score = self.metrics.get('best_score', -float("inf"))
        best_combination = self.metrics.get('best_combination')
        best_state = None
        sample_limits = np.arange(0.05, 1.0, 0.05)

        patience = 50
        i = 0
        for numb_combo, combo in enumerate(itertools.product(sample_limits, repeat=len(clusters))):
            logger.info(f'################ {numb_combo} ##################')
            metrics_combo = scores.get(combo, {
                'f1': [], 'iou': [], 'iou_val': [], 'prec': [],
                'recall': [], 'normalized_iou': [], 'normalized_f1': [],
                'under_prediction': [], 'over_prediction': []
            })

            if len(metrics_combo['iou']) >= self.n_run:
                iou_mean = float(np.mean(metrics_combo['iou_val']))
                if iou_mean > best_score:
                    best_score = iou_mean
                    best_combination = dict(zip(clusters, combo))
                continue
            
            for run in range(len(metrics_combo['iou']), self.n_run):
                df_parts = []
                for cluster, tp in zip(clusters, combo):
                    #logger.info(f'Config : {cluster}, {tp}')
                    df_cluster = df_train[df_train[self.federated_cluster] == cluster]
                    nb = int(len(df_cluster[df_cluster[self.target_name] == 0]) * tp)
                    sampled = self.split_dataset(df_cluster, nb, reset=False)
                    df_parts.append(sampled)

                df_train_split = pd.concat(df_parts).reset_index(drop=True)

                model_copy = deepcopy(self)
                model_copy.training_mode = 'normal'
                model_copy.under_sampling = 'full'
                model_copy.train_split(df_train_split, df_val, df_test, graph, epochs=epochs, PATIENCE_CNT=PATIENCE_CNT, CHECKPOINT=CHECKPOINT, verbose=False)

                pred_val, y_val = model_copy._predict_test_loader(model_copy.val_loader, output_pdf="val")
                y_val_np = y_val.detach().cpu().numpy()[:, -1]
                pred_val_np = pred_val.detach().cpu().numpy()
                metrics_val = evaluate_metrics(pd.DataFrame({self.target_name: y_val_np}), self.target_name, pred_val_np)
                metrics_combo['iou_val'].append(metrics_val['iou'])

                pred_test, y_test = model_copy._predict_test_loader(model_copy.test_loader, output_pdf="test")

                y_test_np = y_test.detach().cpu().numpy()[:, -1]
                pred_test_np = pred_test.detach().cpu().numpy()
                metrics_test = evaluate_metrics(pd.DataFrame({self.target_name: y_test_np}), self.target_name, pred_test_np)

                metrics_combo['iou'].append(metrics_test['iou'])
                metrics_combo['f1'].append(metrics_test['f1'])
                metrics_combo['recall'].append(metrics_test['recall'])
                metrics_combo['prec'].append(metrics_test['prec'])
                metrics_combo['normalized_iou'].append(metrics_test['normalized_iou'])
                metrics_combo['normalized_f1'].append(metrics_test['normalized_f1'])
                metrics_combo['under_prediction'].append(under_prediction_score(y_test_np, pred_test_np))
                metrics_combo['over_prediction'].append(over_prediction_score(y_test_np, pred_test_np))

                scores[combo] = metrics_combo
                self.metrics['scores'] = scores
                save_object(self.metrics, 'metrics_cluster.pkl', self.dir_log)

            if self.n_run == 1:
                metrics_combo['var_f1'] = 0
                metrics_combo['IC_f1'] = (0, 0)
                metrics_combo['var_iou'] = 0
                metrics_combo['IC_iou'] = (0, 0)
                metrics_combo['var_normalized_f1'] = 0
                metrics_combo['IC_normalized_f1'] = (0, 0)
                metrics_combo['var_normalized_iou'] = 0
                metrics_combo['IC_normalized_iou'] = (0, 0)
            else:
                metrics_combo['var_f1'] = np.var(metrics_combo['f1'])
                metrics_combo['IC_f1'] = calculate_ic95(metrics_combo['f1'])
                metrics_combo['var_iou'] = np.var(metrics_combo['iou'])
                metrics_combo['IC_iou'] = calculate_ic95(metrics_combo['iou'])
                metrics_combo['var_normalized_f1'] = np.var(metrics_combo['normalized_f1'])
                metrics_combo['IC_normalized_f1'] = calculate_ic95(metrics_combo['normalized_f1'])
                metrics_combo['var_normalized_iou'] = np.var(metrics_combo['normalized_iou'])
                metrics_combo['IC_normalized_iou'] = calculate_ic95(metrics_combo['normalized_iou'])

            iou_mean = float(np.mean(metrics_combo['iou_val']))
            if iou_mean > best_score:
                best_score = iou_mean
                best_combination = dict(zip(clusters, combo))
                best_state = deepcopy(model_copy.server_model.state_dict())
                i = 0
            else:
                i += 1
                if i == patience:
                    break

            scores[combo] = metrics_combo
            self.metrics['scores'] = scores
            self.metrics['best_score'] = best_score
            self.metrics['best_combination'] = best_combination
            save_object(self.metrics, 'metrics_cluster.pkl', self.dir_log)

        if best_state is not None:
            self.server_model.load_state_dict(best_state)

        self.metrics['scores'] = scores
        self.metrics['best_score'] = best_score
        self.metrics['best_combination'] = best_combination
        self.metrics['run'] = self.n_run
        save_object(self.metrics, 'metrics_cluster.pkl', self.dir_log)

        return best_combination

###################################################################### DUAL TRAINING ####################################################################################

class DualTraining:
    """Manage joint optimisation of two ``Training`` instances.

    The class orchestrates two sub-models: ``occ_model`` operates on the
    binarised dataset while ``num_model`` is restricted to samples with a
    positive label. Losses and parameters of both sub-models are combined so
    that a single optimisation step updates them simultaneously."""

    def __init__(self, target_name, occ_model: Training, num_model: Training, name, task_type: str, n_run : int = 1):
        self.occ_model = occ_model
        self.num_model = num_model
        self.name = name
        self.task_type = task_type
        self.n_run = n_run
        self.target_name = target_name

    def train(
        self,
        graph,
        PATIENCE_CNT,
        CHECKPOINT,
        epochs,
        verbose: bool = True,
        custom_model_params=None,
        new_model: bool = True,
    ):
        self.num_model.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose, custom_model_params, new_model)
        self.occ_model.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose, custom_model_params, new_model)

    # ------------------------------------------------------------------
    # Inference utilities
    # ------------------------------------------------------------------
    def create_test_loader(self, graph, df):
        """Create test loaders for both sub-models.

        The occurence model consumes the full dataframe while the numeric
        model will later be applied only on samples predicted positive. We
        therefore keep a reference to ``graph`` and ``df`` so that the
        numeric loader can be rebuilt after filtering.
        """

        # store for later use in ``_predict_test_loader``
        self._test_graph = graph
        self._test_df = df.reset_index(drop=True)

        # loader for the occurence model (covers all samples)
        self.test_loader = self.occ_model.create_test_loader(graph, df)
        self.occ_model.test_loader = self.test_loader
        return self.test_loader
    
    def create_train_val_test_loader(self, graph, dfs_train, dfs_val, dfs_test, eopchs, PATIENCE_CNT, CHECKPOINT, custom_model_params, features_importance, use_log):
        train_dataset, train_pos = dfs_train
        val_dataset, val_pos = dfs_val
        test_dataset, test_pos = dfs_test

        self.num_model.create_train_val_test_loader(
                graph,
                train_pos,
                val_pos,
                test_pos,
                epochs,
                PATIENCE_CNT,
                CHECKPOINT,
                custom_model_params=custom_model_params,
                features_importance=False,
                use_log=use_log,
            )
        
        self.num_model.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, True, custom_model_params=custom_model_params, new_model=True)

        self.metrics = {}
        tp = 'occ-based'
        
        for run in range(self.n_run):
            seed = int(random.random())
            self.occ_model.seed = seed
            self.occ_model.n_run = 1
            self.occ_model.create_train_val_test_loader(
                graph,
                train_dataset,
                val_dataset,
                test_dataset,
                epochs,
                PATIENCE_CNT,
                CHECKPOINT,
                custom_model_params=custom_model_params,
                features_importance=False,
                use_log=use_log,
            )
            self.occ_model.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, False, custom_model_params=custom_model_params, new_model=True)
            
            ############################# On set val ##############################
            test_output, y = self._predict_test_loader(self.occ_model.val_loader)
            prediction = test_output.detach().cpu().numpy()
            
            y = y.detach().cpu().numpy()
        
            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]

            metrics_run = evaluate_metrics(dff, self.target_name, prediction)
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'val')

            ############################# On set test ##############################
            test_output, y = self._predict_test_loader(self.occ_model.test_loader)
            prediction = test_output.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
        
            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]

            metrics_run = evaluate_metrics(dff, self.target_name, prediction)
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'test')
        
        self.metrics[tp] = add_ic95_to_dict(self.metrics[tp], None, "_ic95")
        self.metrics['best_tp'] = tp

    def _predict_test_loader(self, loader=None, prediction_type='Class', output_pdf=None) -> torch.tensor:
        """Run predictions combining the two sub-models.

        Parameters
        ----------
        loader : DataLoader, optional
            Loader used for the occurence model. If ``None``, the loader
            created by :func:`create_test_loader` is used.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Final numeric predictions and associated ground truth.
        """

        assert self.num_model is not None
        assert self.occ_model is not None
        self.num_model.model.eval()
        self.occ_model.model.eval()

        with torch.no_grad():
            pred = []
            y = []

            for i, data in enumerate(loader, 0):
                
                inputs, orilabels, _ = data

                orilabels = orilabels.to(device)
                orilabels = orilabels[:, :, -1]

                #labels = compute_labels(orilabels, self.model.is_graph_or_node, graphs)

                inputs_model = inputs
                output, logits, hidden = self.occ_model.model(inputs_model)

                output = torch.argmax(output, dim=1)
                output = output.to(torch.float32)

                #output = output[weights.gt(0)]

                if torch.all(output == 0):
                    continue

                output_num_mask = output > 0

                input_num = inputs_model[output_num_mask]

                output_num, logits, hidden = self.num_model.model(input_num)

                if hasattr(self.num_model, 'transform'):
                    if 'cluster_ids' in required_params(self.num_model.transform):
                        cluster_ids = orilabels[:, self.cluster_id_index].long()
                        output_num = self.num_model.transform(output, cluster_ids)
                    else:
                        output_num = self.num_model.transform(output)

                if prediction_type == 'Class':
                    if self.num_model.task_type == 'classification' or self.num_model.task_type == 'binary':
                        output_num = torch.argmax(output, dim=1)
                    elif self.num_model.task_type == 'regression' and  'egpd' in self.loss:
                        output_num = torch.argmax(output, dim=1)
                elif prediction_type == 'RawFormulaVal':
                    output_num = logits
                
                output[output_num_mask] = output_num
                
                pred.append(output)
                y.append(orilabels)
                
            y = torch.cat(y, 0)
            pred = torch.cat(pred, 0)

            if pred.dtype != torch.long:
                pred = torch.round(pred).long()

            return pred, y
        
        if loader is None:
            loader = self.occ_model.test_loader

        # 1) predict occurence on the full dataset
        occ_pred, y = self.occ_model._predict_test_loader(loader)
        occ_mask = occ_pred.reshape(-1) > 0
        
        # 2) build a loader for the numeric model restricted to positives
        df_pos = self._test_df.iloc[occ_mask.cpu().numpy()]
        if len(df_pos) > 0:
            num_loader = self.num_model.create_test_loader(self._test_graph, df_pos)
            num_pred, _ = self.num_model._predict_test_loader(num_loader)
            num_pred = num_pred.reshape(-1)
        else:
            num_pred = torch.tensor([], device=occ_pred.device)

        # 3) assemble final predictions over all samples
        final_pred = torch.zeros(len(self._test_df), device=occ_pred.device, dtype=num_pred.dtype if num_pred.numel() > 0 else torch.float32)
        if num_pred.numel() > 0:
            final_pred[occ_mask] = num_pred

        # ground truth of numeric target over all samples
        target_np = self._test_df[self.num_model.target_name].to_numpy()
        y_full = torch.as_tensor(target_np, dtype=final_pred.dtype, device=final_pred.device)

        return final_pred, y_full

    def search_samples_proportion(self, *args, **kwargs):
        """Delegate proportion search to the occurence model.

        This wrapper keeps the signature of :func:`Training.search_samples_proportion`
        for compatibility while relying on the occurence model implementation.
        """

        return self.occ_model.search_samples_proportion(*args, **kwargs)
    
################################################################# Base Models #############################################################
    
class ModelCNN(SplitTraining):
    def __init__(self, model_name, nbfeatures, batch_size, lr, target_name, task_type, out_channels, dir_log, features_name, features, features_1D,
                 ks, loss, name, device, under_sampling, over_sampling, path, image_per_node, n_run, training_mode='normal', federated_cluster='', cut_layer_name='', input_server_model=0,
                 **kwargs):

        super().__init__(federated_cluster=federated_cluster, cut_layer_name=cut_layer_name, input_server_model=input_server_model, model_name=model_name, nbfeatures=nbfeatures, batch_size=batch_size, lr=lr,
                         target_name=target_name, task_type=task_type, features_name=features_name, ks=ks,
                         out_channels=out_channels, dir_log=dir_log, loss=loss, name=name, device=device, under_sampling=under_sampling,
                         over_sampling=over_sampling, n_run=n_run, **kwargs)

        self.training_mode = training_mode
        self.path = path
        self.features = features
        self.features_1D = features_1D
        self.image_per_node = image_per_node
        self.nbfeatures = nbfeatures
        
    def create_train_val_test_loader(self, graph, df_train, df_val, df_test, epochs,
                                     PATIENCE_CNT, CHECKPOINT, features_importance=True,
                                     varying_time_variables=[], train_features=[], custom_model_params=None, name_exp=None):

        if features_importance:
            importance_df = calculate_and_plot_feature_importance(df_train[self.features_1D], df_train[self.target_name], self.features_1D, self.dir_log / '../importance', self.target_name)
            #importance_df = calculate_and_plot_feature_importance_shapley(df_train[self.features], df_train[self.target_name], self.features, self.dir_log / '../importance', self.target_name)
            features95, self.features_1D = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)
        
        if self.nbfeatures != 'all':
            self.nbfeatures = int(self.nbfeatures)
            varying_time_variables_2 = get_time_columns(varying_time_variables, self.ks, df_train.copy(deep=True), train_features)
            
            self.features_1D = [fet for fet in self.features_1D if fet in df_train.columns]
            
            self.features_1D = self.features_1D[:self.nbfeatures]

            print(self.features_1D)

            features_name_2D, newShape2D = get_features_name_lists_2D(6, train_features)
            self.features_name = get_features_selected_for_time_series_for_2D(self.features_1D, features_name_2D, varying_time_variables_2, self.nbfeatures)
            
            self.features_name = list(np.unique(self.features_name))
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

                df_combined = self.split_dataset(df_train, nb)

                # Mettre à jour df_train pour l'entraînement
                df_train = df_combined

                logger.info(f'Train mask df_train shape: {old_shape} -> {df_train.shape}')

            elif self.under_sampling == 'search' or 'percentage' in self.under_sampling:
                    if self.training_mode == 'splittraining':
                        best_combinaison = self.search_samples_proportion_per_cluster(graph, df_train, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT)
                        df_parts = []
                        for cluster, nb in best_combinaison:
                            df_cluster = df_train[df_train[self.federated_cluster] == cluster]
                            sampled = self.split_dataset(df_cluster, nb, reset=True)[cluster]
                            df_parts.append(sampled)

                        df_train = pd.concat(df_parts).reset_index(drop=True)
                    else:
                        if self.under_sampling == 'search':
                            best_tp, find_log = self.search_samples_proportion(graph, df_train, df_val, df_test, False, epochs, PATIENCE_CNT, CHECKPOINT)
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
                        df_train['weight'] = 0

                        # Mettre à jour df_train pour l'entraînement
                        df_train.loc[df_combined.index, 'weight'] = 1
                        logger.info(f'Train mask df_train shape: {old_shape} -> {df_train.shape}')
                
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
                                            features_1D=self.features_1D,
                                            target_name=self.target_name,
                                            use_temporal_as_edges=None,
                                            image_per_node=self.image_per_node,
                                            device=self.device, ks=self.ks,
                                            path=self.path,
                                            name_exp=name_exp)

            train_loader = DataLoader(train_dataset, 16, True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(val_dataset, 16, False, worker_init_fn=seed_worker, generator=g)
            test_loader = DataLoader(test_dataset, 16, False, worker_init_fn=seed_worker, generator=g)

            #save_object_torch(train_loader, 'train_loader.pkl', self.dir_log)
            #save_object_torch(val_loader, 'val_loader.pkl', self.dir_log)
            #save_object_torch(test_loader, 'test_loader.pkl', self.dir_log)

            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader

    def create_test_loader(self, graph, df):

        x_test, y_test = df[ids_columns + self.features_1D].values, df[ids_columns + [self.target_name]].values

        dateTest = np.sort(np.unique(y_test[np.argwhere(y_test[:, weight_index] > 0), date_index]))
        
        XsTe, YsTe, EsTe = create_dataset_2D_2(graph, x_test, y_test, self.ks, dateTest,
                    self.features_name, self.features, self.features_1D, False)
        
        sub_dir = f'image_per_node_{len(self.features_name)}' if self.image_per_node else f'image_per_departement_{len(self.features_name)}'
        test_dataset = ReadGraphDataset_2D_from_xarray(XsTe, YsTe, EsTe, len(XsTe), device, rootDisk / 'csv', self.features_name, self.features_1D, self.ks,
                                                graph.scale,
                                                graph.graph_method,
                                                graph.base,
                                                self.path / 'datacube')
        
        loader = DataLoader(test_dataset, test_dataset.__len__(), False)

        return loader

class ModelGNN(SplitTraining):
    def __init__(self, graph_method, mesh, mesh_file, model_name, nbfeatures, batch_size, lr, target_name, task_type,
                 out_channels, dir_log, features_name, ks, loss, name, device, under_sampling, over_sampling,
                 n_run, training_mode='normal', federated_cluster='', cut_layer_name='', input_server_model=0,
                 horizon=1):

        super().__init__(federated_cluster=federated_cluster, cut_layer_name=cut_layer_name, input_server_model=input_server_model, model_name=model_name, nbfeatures=nbfeatures, batch_size=batch_size, lr=lr, target_name=target_name, task_type=task_type, features_name=features_name, ks=ks,
                         out_channels=out_channels, dir_log=dir_log, loss=loss, name=name, device=device, under_sampling=under_sampling,
                         over_sampling=over_sampling, n_run=n_run, horizon=horizon)
        self.training_mode = training_mode
        self.mesh = mesh
        self.mesh_file = mesh_file
        self.graph_method = graph_method
        self.mesh2graph = None
        self.gridh2mesh = None
        self.graph_mesh = None
        self.horizon = horizon

    def create_train_val_test_loader(self, graph, df_train, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT, features_importance=True, custom_model_params=None, use_log=True):

        self.graph = graph
        if self.mesh and self.graph_mesh is None:
            
            df = pd.concat((df_train, df_val, df_test))

            latitudes = torch.Tensor(df['latitude'].values.reshape(-1,1))
            longitudes = torch.Tensor(df['longitude'].values.reshape(-1,1))
            departement = torch.Tensor(df['departement'].values.reshape(-1,))

            g_lat_lon_grid = torch.concat((latitudes, longitudes), dim=1).to('cpu')
            g_lat_lon_grid = torch.unique(g_lat_lon_grid, dim=0)
             
            graph_builder = GraphBuilder(self.mesh_file, g_lat_lon_grid, doPrint=False)

            self.graph_mesh = graph_builder.create_mesh_graph(None)
            self.gridh2mesh, graph_mesh = graph_builder.create_g2m_graph(None, self.graph_mesh)
            self.mesh2graph = graph_builder.create_m2g_graph(None)

            def _check_non_empty_edges(g, etype, name):
                e = g.num_edges(etype)
                if e == 0:
                    print(f"[WARN] {name}: 0 edges pour etype {etype}.")
                return e

            _check_non_empty_edges(self.gridh2mesh, ("grid","g2m","mesh"), "gridh2mesh")
            _check_non_empty_edges(self.mesh2graph, ("mesh","m2g","grid"), "mesh2graph")

            if '_ID' not in graph_mesh.edata:
                graph_mesh.edata['_ID'] = torch.arange(graph_mesh.num_edges(), dtype=torch.int32)
            if '_ID' not in graph_mesh.ndata:
                graph_mesh.ndata['_ID'] = torch.arange(graph_mesh.num_nodes(), dtype=torch.int32)

        if features_importance:
            importance_df = calculate_and_plot_feature_importance(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
            #importance_df = calculate_and_plot_feature_importance_shapley(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
            features95, featuresAll = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)
            
            #exit(1)
            if self.nbfeatures != 'all':
                self.features_name = featuresAll[:int(self.nbfeatures)]
            else:
                self.features_name = featuresAll
        
        if self.model_name in ['GRUGNN']:
            static_idx, temporal_idx = get_static_temporal_idx(self.features_name)
            logger.info(f'Num temporal features {len(temporal_idx)} num spatial features {len(static_idx)}')
            custom_model_params = {'static_idx': static_idx, 'temporal_idx' : temporal_idx}

        if self.val_loader is None:
            if self.mesh == 'mesh':
                val_dataset, test_dataset = create_test_val_dataset(graph,
                                                    df_val,
                                                    df_test,
                                                    self.features_name,
                                                    self.target_name,
                                                    None,
                                                    self.device, self.ks,
                                                    graph_mesh=self.graph_mesh,
                                                    gridh2mesh=self.gridh2mesh,
                                                    mesh2graph=self.mesh2graph
                                                    )
            else:
                val_dataset, test_dataset = create_test_val_dataset(graph,
                                                    df_val,
                                                    df_test,
                                                    self.features_name,
                                                    self.target_name,
                                                    False,
                                                    self.device, self.ks,
                                                    graph_mesh=self.graph_mesh,
                                                    gridh2mesh=self.gridh2mesh,
                                                    mesh2graph=self.mesh2graph)
                
            if not self.mesh or self.mesh == False:            
                self.val_loader = DataLoader(val_dataset, val_dataset.__len__(), False, collate_fn=graph_collate_fn)
                self.test_loader = DataLoader(test_dataset, test_dataset.__len__(), False, collate_fn=graph_collate_fn)
            
            elif self.mesh == 'mesh':
                self.val_loader = DataLoader(val_dataset, val_dataset.__len__(), False, collate_fn=graph_collate_fn_mesh)
                self.test_loader = DataLoader(test_dataset, test_dataset.__len__(), False, collate_fn=graph_collate_fn_mesh)

            elif self.mesh == 'mygraph':
                self.val_loader = DataLoader(val_dataset, val_dataset.__len__(), False, collate_fn=graph_collate_fn_multiple_graph)
                self.test_loader = DataLoader(test_dataset, test_dataset.__len__(), False, collate_fn=graph_collate_fn_multiple_graph)

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
                df_train['weight'] = 0
                # Mettre à jour df_train pour l'entraînement
                df_train.loc[df_combined.index, 'weight'] = 1

                logger.info(f'Train mask df_train shape: {old_shape} -> {df_train[df_train["weight"] > 0].shape}')
                
            elif self.under_sampling == 'search' or 'percentage' in self.under_sampling:
                    if self.training_mode == 'splittraining':
                        best_combinaison = self.search_samples_proportion_per_cluster(graph, df_train, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT)
                        df_parts = []
                        for cluster, nb in best_combinaison:
                            df_cluster = df_train[df_train[self.federated_cluster] == cluster]
                            sampled = self.split_dataset(df_cluster, nb, reset=True)[cluster]
                            df_parts.append(sampled)

                        df_train = pd.concat(df_parts).reset_index(drop=True)
                    else:
                        if self.under_sampling == 'search':
                            best_tp, find_log = self.search_samples_proportion(graph, df_train, df_val, df_test, False,
                                                                               epochs, PATIENCE_CNT, CHECKPOINT, False,
                                                                               custom_model_params=custom_model_params, use_log=use_log)
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
                        df_train['weight'] = 0

                        # Mettre à jour df_train pour l'entraînement
                        df_train.loc[df_combined.index, 'weight'] = 1
                        logger.info(f'Train mask df_train shape: {old_shape} -> {df_train[df_train["weight"] > 0].shape}')

        if False:
            self.train_loader = read_object('train_loader.pkl', self.dir_log)
            self.val_loader = read_object('val_loader.pkl', self.dir_log)
            self.test_loader = read_object('test_loader.pkl', self.dir_log)
        else:
            if self.mesh == 'mesh':
                train_dataset = create_train_dataset(graph,
                                                    df_train,
                                                    self.features_name,
                                                    self.target_name,
                                                    None,
                                                    self.device, self.ks,
                                                    graph_mesh=self.graph_mesh,
                                                    gridh2mesh=self.gridh2mesh,
                                                    mesh2graph=self.mesh2graph)
            else:
                train_dataset = create_train_dataset(graph,
                                                    df_train,
                                                    self.features_name,
                                                    self.target_name,
                                                    False,
                                                    self.device, self.ks,
                                                    graph_mesh=self.graph_mesh,
                                                    gridh2mesh=self.gridh2mesh,
                                                    mesh2graph=self.mesh2graph)

            if not self.mesh or self.mesh == False:            
                self.train_loader = DataLoader(train_dataset, self.batch_size, True, collate_fn=graph_collate_fn)
            
            elif self.mesh == 'mesh':
                self.train_loader = DataLoader(train_dataset, self.batch_size, True, collate_fn=graph_collate_fn_mesh)

            elif self.mesh == 'mygraph':
                self.train_loader = DataLoader(train_dataset, self.batch_size, True, collate_fn=graph_collate_fn_multiple_graph)

        #save_object_torch(self.train_loader, 'train_loader.pkl', self.dir_log)
        #save_object_torch(self.val_loader, 'val_loader.pkl', self.dir_log)
        #save_object_torch(self.test_loader, 'test_loader.pkl', self.dir_log)
        
    def create_test_loader(self, graph, df):
        loader = create_test_loader(graph, df,
                       self.features_name,
                       self.device,
                       False,
                       self.target_name,
                       self.ks,
                       self.graph_mesh,
                        self.gridh2mesh,
                        self.mesh2graph)

        return loader

    def launch_batch(self, data, criterion):
        
        if not self.mesh or self.mesh == False:
            inputs, labels, graphs, graphs = data
            output, logits, hidden = self.model(inputs, graphs)
        else:
            inputs, labels, DGLgraphs, graphs = data
            
            output, logits, hidden = self.model(inputs, DGLgraphs[0], DGLgraphs[1], DGLgraphs[2])
        
        if inputs.shape[0] == 1:
            return 0

        band = -1
        
        try:
            target, weights = self.compute_weights_and_target(labels, band, ids_columns, self.model.is_graph_or_node, graphs)
        except Exception as e:
            target, weights = self.compute_weights_and_target(labels, band, ids_columns, False, graphs)

        if self.loss not in ['kldivloss']: # works on probability
            target = target.long()
        
        loss = self.calculate_loss(criterion, logits, target, weights, labels)

        if self.student_train: # distallation traning
            criterion_teacher = self.get_loss('kldivloss')
            df_test = pd.DataFrame(inputs[:, :, -1], columns=self.features_name)
            df_test.columns = df_test.columns.astype(str)
            pred_teacher = self.teacher.predict_proba(df_test, weights_average=self.weights_average, top_model=self.top_model, id_col=(None, None))
            target = torch.Tensor(pred_teacher, device=inputs.device).to(torch.float32)
            target = target / self.temperature_value

            loss2 = self.calculate_loss(criterion_teacher, output, target, weights, labels, tolong=False)

            loss = self.alpha_value * loss2 + (1 - self.alpha_value) * loss
        
        if self.constrastive: # MOON federated training
            _, _, zprev = self.prev_model(inputs, graphs)
            _, _, zglob = self.global_model(inputs, graphs)
            loss_constrastive = self.calculate_contrastive_moon_loss(hidden, zprev, zglob, self.temperature_value)
            loss = loss + self.smooth_value * loss_constrastive

        if self.use_prototypes and self.prototypes is not None:
            
            proto_loss = self.calculate_prototype_alignment_loss(hidden, target, self.prototypes)
            loss = loss + self.prototype_weight * proto_loss

        if self.model_name in ['BayesianMLP', 'BayesianCNN', 'BayesianRNN']:
            loss += self.model.kl_loss()

        return loss

    def _predict_test_loader(self, X: DataLoader, prediction_type='Class', output_pdf='test') -> torch.tensor:
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
                    output, logits, hidden = self.model(inputs_model, graphs)
                else:
                    output, logits, hidden = self.model(inputs_model, DGLgraphs[0], DGLgraphs[1], DGLgraphs[2])

                if output.shape[1] > 1 and self.task_type != 'regression':
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

class Model_Torch(SplitTraining):
    def __init__(self, model_name, nbfeatures, batch_size, lr, target_name, task_type, out_channels,
                 dir_log, features_name, ks, loss, name, device, under_sampling, over_sampling, n_run,
                 training_mode='normal', federated_cluster='', cut_layer_name='', input_server_model=0,
                 horizon=1):

        #federated_cluster, model_name, nbfeatures, batch_size, lr, target_name, task_type, out_channels,
        #         dir_log, features_name, ks, loss, name, device, under_sampling, over_sampling, n_run
        
        super().__init__(federated_cluster=federated_cluster, cut_layer_name=cut_layer_name, input_server_model=input_server_model,
                         model_name=model_name, nbfeatures=nbfeatures, batch_size=batch_size, lr=lr,
                         target_name=target_name, task_type=task_type, features_name=features_name, ks=ks,
                         out_channels=out_channels, dir_log=dir_log, loss=loss, name=name, device=device, under_sampling=under_sampling,
                         over_sampling=over_sampling, n_run=n_run, horizon=horizon)

        self.training_mode = training_mode

        self.horizon = horizon

    def create_train_val_test_loader(self, graph, df_train, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT, features_importance=True, custom_model_params=None, use_log=True):
        self.graph = graph

        if 'learnable-area' in self.loss:
            area_parameters = np.sort(df_train['graph_id'].unique())
            self.area_parameters = torch.nn.Parameter(torch.Tensor(torch.ones_like(torch.tensor(area_parameters))))
        elif 'area' in self.loss:
            area_parameters = np.sort(df_train['graph_id'].unique())
            self.area_parameters = torch.ones_like(torch.tensor(area_parameters))
        
        if False:
            ##################################### Select features #########################################
            importance_df = calculate_and_plot_feature_importance(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
            #importance_df = calculate_and_plot_feature_importance_shapley(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
            features95, featuresAll = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)

            if self.nbfeatures != 'all':
                self.features_name = featuresAll[:int(self.nbfeatures)]
            else:
                self.features_name = featuresAll

        if self.val_loader is None:
            val_dataset, test_dataset = create_test_val_dataset(graph,
                                                                df_val,
                                                                df_test,
                                                                self.features_name,
                                                                self.target_name,
                                                                None,
                                                                self.device, self.ks,
                                                                graph_mesh=None,
                                                                gridh2mesh=None,
                                                                mesh2graph=None)
            
            val_loader = DataLoader(val_dataset, val_dataset.__len__(), False, worker_init_fn=seed_worker, generator=g)
            test_loader = DataLoader(test_dataset, test_dataset.__len__(), False, worker_init_fn=seed_worker, generator=g)
            self.val_loader = val_loader
            self.test_loader = test_loader
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
                    if self.training_mode == 'splittraining':
                        best_combinaison = self.search_samples_proportion_per_cluster(graph, df_train, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT)
                        df_parts = []
                        for cluster, nb in best_combinaison:
                            df_cluster = df_train[df_train[self.federated_cluster] == cluster]
                            sampled = self.split_dataset(df_cluster, nb, reset=True)[cluster]
                            df_parts.append(sampled)

                        df_train = pd.concat(df_parts).reset_index(drop=True)
                    else:
                        if self.under_sampling == 'search':
                            best_tp, find_log = self.search_samples_proportion(graph, df_train, df_val, df_test, is_unknowed_risk=False,
                                                                            epochs=epochs, PATIENCE_CNT=PATIENCE_CNT, CHECKPOINT=CHECKPOINT,
                                                                            custom_model_params=custom_model_params, use_log=use_log)
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
                        df_train['weight'] = 0

                        # Mettre à jour df_train pour l'entraînement
                        df_train.loc[df_combined.index, 'weight'] = 1
                        logger.info(f'Train mask df_train shape: {old_shape} -> {df_train.shape}')

        if 'smote' in self.over_sampling:
            
            if isinstance(self, ModelCNN) or isinstance(self, ModelGNN):
                raise ValueError(f'Smote is not adaptable to images or GNN')
            
            if self.ks > 0:
                raise ValueError(f'Smote is not adaptbale to time series')

            # Exemple : si ta cible est dans une colonne 'target'
            matching_ids_columns = [col for col in ids_columns if col != 'date']  # on exclut 'date' du matching

            # 2. Construire X (features + ids_columns), y (target)
            X_full = df_train[self.features_name + matching_ids_columns].copy()
            y = df_train[self.target_name].copy()

            smote_coef = int(self.over_sampling.split('-')[1])
            y_negative = y[y == 0].shape[0]
            
            y_one = min(y[y == 1].shape[0] * smote_coef, y_negative)
            y_two = min(y[y == 2].shape[0] * smote_coef, y_negative)
            y_three = min(y[y == 3].shape[0] * smote_coef, y_negative)
            y_four = min(y[y == 4].shape[0] * smote_coef, y_negative)

            if self.task_type in ['classification', 'ordinal-classification']:
                sampling_strategy = {
                    0: y_negative,
                    1: y_one,
                    2: y_two,
                    3: y_three,
                    4: y_four
                }
                smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
            
            elif self.task_type == 'binary':
                smote = SMOTE(random_state=42, sampling_strategy='auto')

            X_resampled, y_resampled = smote.fit_resample(X_full, y)

            df_train = X_resampled
            df_train[self.target_name] = y_resampled

            for uy in np.unique(df_train[self.target_name]):
                    print(f'Number of {uy} class : {df_train[df_train[self.target_name] == uy].shape}') 

        self.df_train = df_train
        self.df_test = df_test
        self.df_val = df_val

        print(df_train.shape, df_test.shape, df_val.shape)
        print(df_train[df_train['weight'] > 0].shape, df_val[df_val['weight'] > 0].shape)

        ##################################### Create loader #########################################        
        if False:
            train_dataset = read_object('train_dataset.pkl', self.dir_log)
            val_dataset = read_object('val_dataset.pkl', self.dir_log)
            test_dataset = read_object('test_dataset.pkl', self.dir_log)
        else:
            train_dataset = create_train_dataset(graph,
                                                df_train,
                                                self.features_name,
                                                self.target_name,
                                                None,
                                                self.device, self.ks,
                                                graph_mesh=None,
                                                gridh2mesh=None,
                                                mesh2graph=None)
    
            #save_object_torch(train_dataset, 'train_dataset.pkl', self.dir_log)
            #save_object_torch(val_dataset, 'val_dataset.pkl', self.dir_log)
            #save_object_torch(test_dataset, 'test_dataset.pkl', self.dir_log)

            train_loader = DataLoader(train_dataset, batch_size, True, worker_init_fn=seed_worker, generator=g)

            self.train_loader = train_loader

    def create_test_loader(self, graph, df):
        loader = create_test_loader(graph, df,
                       self.features_name,
                       self.device,
                       None,
                       self.target_name,
                       self.ks,
                        graph_mesh=None,
                        gridh2mesh=None,
                        mesh2graph=None)

        return loader

########################################## Federated Learning #########################################

class FederatedLearningModel(RegressorMixin, ClassifierMixin):
    def __init__(self, federated_model, features, federated_cluster='departement', loss='mse',
                 name='FederatedModel', dir_log=Path('../'), under_sampling='full', over_sampling='full',
                 target_name='nbsinister', post_process=None, task_type='classification',
                 aggregation_method='max', nbfeatures='all', n_run=1, horizon=1):
        """
        Initialize the Federated Learning Model.

        Parameters:
        - federated_model: The base model to be used across all clusters.
        - federated_cluster: Column name used to identify clusters (default: 'departement').
        - aggregation_method: Method to aggregate local models ('mean', 'median', 'weighted', etc.).
        """
        super().__init__()
        self.features_name = features
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
        self.nbfeatures = nbfeatures
        self.n_run = n_run

        self.horizon = horizon

    def fit(self, df_train, df_val, df_test, graph, args):
        """
        Train local models for each federated cluster, aggregate them into a global model, 
        and stop training once the global score does not improve for patience_count_global epochs.
        """

        #importance_df = calculate_and_plot_feature_importance(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        #importance_df = calculate_and_plot_feature_importance_shapley(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        #features95, featuresAll = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)
        
        #if self.nbfeatures != 'all':
        #    self.features_name = featuresAll[:int(self.nbfeatures)]
        #else:
            #features_name = featuresAll

        self.global_model.features_name = self.features_name
        self.global_model.nbfeatures = 'all'
        self.global_model.graph = graph

        initiate_model, model_params = self.global_model.make_model(graph, custom_model_params=None)
        
        # Vérifier que la méthode d'agrégation est implémentée
        if self.aggregation_method not in ['mean', 'median', 'weighted', 'max']:
            raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' is not implemented.")
        
        print(args)
        # Récupération des paramètres d'entraînement
        global_epochs = args.get('global_epochs')
        local_epochs = args.get('local_epochs')
        patience_count_global = args.get('patience_count_global')
        patience_count_local = args.get('patience_count_local')

        if self.federated_cluster in np.unique(df_train.columns):
            clusters = df_train[self.federated_cluster].unique()
        else:
            raise NotImplementedError(f"Aggregation method '{self.federated_cluster}' is not implemented.")
        
        print(f"\n--- Training Federated Model for {global_epochs} global epochs ---")
        
        self.metrics = {}

        tp = 'client-based'

        for run in range(self.n_run):

            self.global_model.model = deepcopy(initiate_model)
            self.global_model.model_params = deepcopy(model_params)
            local_models = {}
            
            seed = int(random.random())

            best_global_score = float('-inf')
            patience_counter = 0  # Compteur pour l'arrêt anticipé
            for epoch in range(global_epochs):
                print(f"\n--- Global Epoch {epoch + 1}/{global_epochs} ---")

                local_weights = []
                sample_counts = []
                
                for cluster in clusters:
                    print(f"\nTraining local model for cluster: {cluster}")

                    # Création des datasets pour le cluster fédéré
                    df_train_cluster, df_val_cluster, df_test_cluster = self.create_cluster_set(
                        df_train, df_val, df_test, cluster
                    )

                    if np.all(df_train[self.global_model.target_name].values == 0) or np.all(df_val[self.global_model.target_name].values == 0) or np.all(df_test[self.global_model.target_name].values == 0):
                        print(f'Skipping {cluster} due to no positif samples')
                        continue
                    
                    if df_val_cluster.shape[0] == 0 or df_train_cluster.shape[0] == 0:
                        print(f'Skipping {cluster} due to empty dataset')
                        continue

                    if cluster in local_models.keys():
                        local_model = local_models[cluster]
                        local_model.model.load_state_dict(self.global_model.model.state_dict())
                    else:
                        # Initialisation du modèle local
                        local_model = deepcopy(self.global_model)
                        local_model.seed = seed
                        local_model.name = f'{self.federated_cluster}_{cluster}_{self.global_model.name}'
                        local_model.dir_log = self.dir_log / local_model.name
                        if epoch == 0:
                            check_and_create_path(local_model.dir_log)
                        local_model.features_name = self.features_name
                        local_model.nbfeatures = 'all'

                    if epoch == 0:
                        local_model.create_train_val_test_loader(graph, df_train_cluster, df_val_cluster, df_test_cluster, local_epochs, patience_count_local, CHECKPOINT, False)
                        
                    # Entraînement du modèle local
                    local_model.train(graph, patience_count_local, CHECKPOINT, local_epochs, verbose=False, custom_model_params=None, new_model=False)
                    
                    local_models[cluster] = local_model

                    # Stocker les poids des modèles locaux
                    local_weights.append(deepcopy(local_model.model.state_dict()))
                    sample_counts.append(len(df_train_cluster))

                # Agréger les modèles locaux dans le modèle global
                self.aggregate_models(local_weights, sample_counts)

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
            
            loader = self.create_test_loader(graph, df_test)
            test_output, y = self._predict_test_loader(loader)
            test_output = test_output.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            ###################### TEST SET ########################
            loader = self.create_test_loader(graph, df_test)
            test_output, y = self._predict_test_loader(loader)

            test_output = test_output.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]
            
            metrics_run = evaluate_metrics(dff, self.target_name, test_output)
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'test')

            ###################### VAL SET ########################
            loader = self.create_test_loader(graph, df_val)
            test_output, y = self._predict_test_loader(loader)
            test_output = test_output.detach().cpu().numpy()
            
            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]
            
            metrics_run = evaluate_metrics(dff, self.target_name, test_output)
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'val')

        self.metrics['best_tp'] = tp

        self.is_fitted_ = True
        print("\n--- Federated Learning Training Complete ---")

    def create_cluster_set(self, df_train, df_val, df_test, cluster):
        """
        Create training, validation, and test datasets for a given cluster.
        """
        if self.federated_cluster in np.unique(df_train.columns):
            X_cluster = df_train[df_train[self.federated_cluster] == cluster].reset_index(drop=True)
            X_val_cluster = df_val[df_val[self.federated_cluster] == cluster].reset_index(drop=True)
            X_test_cluster = df_test[df_test[self.federated_cluster] == cluster].reset_index(drop=True)

            return X_cluster, X_val_cluster, X_test_cluster
        
        raise NotImplementedError(f"Federated clustering method '{self.federated_cluster}' is not implemented.")

    def aggregate_models(self, local_weights, sample_counts=None):
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
            elif self.aggregation_method == 'max':
                new_state_dict[key] = torch.max(stacked_params, dim=0)[0]
            elif self.aggregation_method == 'weighted':
                if sample_counts is not None and len(sample_counts) == len(local_weights):
                    weights = torch.tensor(sample_counts, dtype=torch.float32)
                    weights = weights / weights.sum()
                else:
                    weights = torch.tensor([1 / len(local_weights)] * len(local_weights), dtype=torch.float32)
                view_shape = [len(local_weights)] + [1] * (stacked_params.dim() - 1)
                new_state_dict[key] = torch.sum(stacked_params * weights.view(*view_shape), dim=0)

        # Mettre à jour les poids du modèle global
        self.global_model.update_weight(new_state_dict)
        print("\n--- Global Model Weights Updated ---")

    def predict(self, X, graph=None, return_y=False):
        """
        Predict using the aggregated global model.
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Please train the model before predicting.")

        print(f'Predicting using Global Model')
        return self.global_model.predict(X, graph, return_y)

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
        
    def make_model(self, graph, custom_model_params):
        return self.global_model.make_model(graph, custom_model_params)
    
    def create_test_loader(self, graph, df):
        return self.global_model.create_test_loader(graph, df)
    
    def _predict_test_loader(self, X):
        return self.global_model._predict_test_loader(X)

############################################ ALA Federated Model ##############################################################
class FederatedALA(FederatedLearningModel):
    """Federated learning strategy relying on the :class:`Training` class for local training."""

    def __init__(self, federated_model, eta, features, federated_cluster='departement', loss='mse',
                 name='FederatedModel', dir_log=Path('../'), under_sampling='full', over_sampling='full',
                 target_name='nbsinister', post_process=None, task_type='classification',
<<<<<<< HEAD
<<<<<<< HEAD
                 aggregation_method='max', nbfeatures='all', n_run=1, params_to_update=['linear2'], horizon=1):

=======
=======
>>>>>>> 5b18034 ([Update code])
                 aggregation_method='max', nbfeatures='all', n_run=1, params_to_update=['linear2']):
        
>>>>>>> 5b18034 ([Update code])
        super().__init__(federated_model=federated_model, features=features, federated_cluster=federated_cluster,
                         loss=loss, name=name, dir_log=dir_log, under_sampling=under_sampling,
                         over_sampling=over_sampling, target_name=target_name, post_process=post_process,
                         task_type=task_type, aggregation_method=aggregation_method, nbfeatures=nbfeatures,
                         n_run=n_run, horizon=horizon)
        self.eta = eta
        self.weight = 0.5
        self.params_to_update = params_to_update

<<<<<<< HEAD
<<<<<<< HEAD
        self.horizon = horizon

=======
>>>>>>> 5b18034 ([Update code])
=======
>>>>>>> 5b18034 ([Update code])
    def pick_params_by_name(self, model):
        names, params = [], []
        for n, p in model.named_parameters():
            for pick_parm in self.params_to_update:
                if pick_parm in n:
                    names.append(n); params.append(p)
        return names, params
    
    def pick_params_to_replace(self, model):
        names, params = [], []
        for n, p in model.named_parameters():
            add = True
            for pick_parm in self.params_to_update:
                if pick_parm in n:
                    add = False
                    break
            if add:
                names.append(n); params.append(p)
        return names, params

    def fedALA_params(self, local_model):
        with torch.no_grad():
            for p_t, p_g in zip(local_model.params_erase,
                                local_model.params_gp_erase,
                                ):
                p_t.copy_(p_g)

            for p_t, p_prev, p_g, w in zip(local_model.params_p,
                                        local_model.params_tp,
                                        local_model.params_gp,
                                        local_model.weights):
                # met tout sur le bon device/dtype
                p_prev = p_prev.to(local_model.device, dtype=p_t.dtype)
                p_g    = p_g.detach().to(local_model.device, dtype=p_t.dtype)
                w      = w.to(local_model.device, dtype=p_t.dtype)
                w = w.clamp_(0, 1)
                p_t.copy_(p_prev + (p_g - p_prev) * w)

    def fit(self, df_train, df_val, df_test, graph, args):
        """
        Train local models for each federated cluster, aggregate them into a global model, 
        and stop training once the global score does not improve for patience_count_global epochs.
        """

        importance_df = calculate_and_plot_feature_importance(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        #importance_df = calculate_and_plot_feature_importance_shapley(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        features95, featuresAll = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)
        
        #if self.nbfeatures != 'all':
        #    self.features_name = featuresAll[:int(self.nbfeatures)]
        #else:
        #    self.features_name = featuresAll

        self.global_model.features_name = self.features_name
        self.global_model.nbfeatures = 'all'
        self.global_model.graph = graph

        initiate_model, model_params = self.global_model.make_model(graph, custom_model_params=None)

        
        # Vérifier que la méthode d'agrégation est implémentée
        if self.aggregation_method not in ['mean', 'median', 'weighted', 'max']:
            raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' is not implemented.")
        
        # Récupération des paramètres d'entraînement
        global_epochs = args.get('global_epochs')
        local_epochs = args.get('local_epochs')
        patience_count_global = args.get('patience_count_global')
        patience_count_local = args.get('patience_count_local')

        if self.federated_cluster in np.unique(df_train.columns):
            clusters = df_train[self.federated_cluster].unique()
        else:
            raise NotImplementedError(f"Aggregation method '{self.federated_cluster}' is not implemented.")

        print(f"\n--- Training ALA Federated Model for {global_epochs} global epochs ---")

        tp = 'client-based'
        self.metrics = {}
        
        for run in range(self.n_run):
            
            self.model_params = deepcopy(model_params)
            self.global_model.model = deepcopy(initiate_model)
            self.global_model.model_params = deepcopy(model_params)
            self.global_model.eta = self.eta
            local_models = {}
            
            best_global_score = float('-inf')
            patience_counter = 0  # Compteur pour l'arrêt anticipé
            seed = int(random.random())

            for epoch in range(global_epochs):
                print(f"\n--- Global Epoch {epoch + 1}/{global_epochs} ---")

                local_weights = []
                sample_counts = []
                
                for cluster in clusters:
                    print(f"\nTraining local model for cluster: {cluster}")

                    # Création des datasets pour le cluster fédéré
                    df_train_cluster, df_val_cluster, df_test_cluster = self.create_cluster_set(
                        df_train, df_val, df_test, cluster
                    )

                    if np.all(df_train[self.global_model.target_name].values == 0) or np.all(df_val[self.global_model.target_name].values == 0) or np.all(df_test[self.global_model.target_name].values == 0):
                        print(f'Skipping {cluster} due to no positif samples')
                        continue

                    if df_val_cluster.shape[0] == 0 or df_train_cluster.shape[0] == 0:
                        print(f'Skipping {cluster} due to empty dataset')
                        continue

                    # Initialisation du modèle local
                    if epoch == 0:
                        local_model = deepcopy(self.global_model)
                        local_model.ALATraining = False
                        local_model.seed = seed
                        local_model.features_name = self.features_name
                        local_model.nbfeatures = 'all'
                        self.metrics[cluster] = local_model.metrics
                        local_model.name = f'{self.federated_cluster}_{cluster}_{self.global_model.name}'
                        local_model.dir_log = self.dir_log / local_model.name
                        local_model.create_train_val_test_loader(graph, df_train_cluster, df_val_cluster, df_test_cluster, local_epochs, patience_count_local, CHECKPOINT, False)
                        check_and_create_path(local_model.dir_log)
                    else:
                        local_model = local_models[cluster]
                        local_model.ALAtraining = True
                        local_model.ala_weight_only = False

                        local_model.global_model = self.global_model.model
                        _, params = self.pick_params_by_name(local_model.model)
                        _, params_g = self.pick_params_by_name(self.global_model.model)

                        _, params_erase = self.pick_params_to_replace(local_model.model)
                        _, params_gp_erase = self.pick_params_to_replace(self.global_model.model)

                        assert len(params) > 0
                        
                        local_model.params_p  = params
                        local_model.params_gp = params_g
                        local_model.params_erase = params_erase
                        local_model.params_gp_erase = params_gp_erase

                        # snapshot des poids locaux précédents (gelés)
                        local_model.params_tp = [p.detach().clone() for p in local_model.params_p]
                        
                        for param in local_model.params_tp:
                            param.requires_grad = False
                        
                        if epoch == 1:
                            local_model_log_params = deepcopy(local_model.model.state_dict())
                            w_t = torch.as_tensor(self.weight, device=local_model.device, dtype=local_model.params_p[0].dtype)
                            local_model.weights = [
                                w_t.expand_as(p).clone() for p in local_model.params_p
                            ]
                            local_model.ala_weight_only = True
                            local_model.train(graph, patience_count_local, CHECKPOINT, local_epochs, verbose=False, custom_model_params={'return_hidden' : True}, new_model=False)
                            local_model.model.load_state_dict(local_model_log_params)

                        # Fed ala params
                        self.fedALA_params(local_model)

                    # Entraînement du modèle local
                    local_model.train(graph, patience_count_local, CHECKPOINT, local_epochs if epoch < 1 else 1, verbose=False, custom_model_params={'return_hidden' : True}, new_model=False)
                    
                    local_models[cluster] = local_model
                    
                    # Stocker les poids des modèles locaux
                    local_weights.append(deepcopy(local_model.model.state_dict()))
                    sample_counts.append(len(df_train_cluster))

                # Agréger les modèles locaux dans le modèle global
                self.aggregate_models(local_weights, sample_counts)

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
            
            ###################### TEST SET ########################
            loader = self.create_test_loader(graph, df_test)
            test_output, y = self._predict_test_loader(loader)

            test_output = test_output.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]
            
            metrics_run = evaluate_metrics(dff, self.target_name, test_output)
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'test')

            ###################### VAL SET ########################
            loader = self.create_test_loader(graph, df_val)
            test_output, y = self._predict_test_loader(loader)
            test_output = test_output.detach().cpu().numpy()
            
            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]
            
            metrics_run = evaluate_metrics(dff, self.target_name, test_output)
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'val')

        self.metrics['best_tp'] = tp

        self.is_fitted_ = True
        print("\n--- ALA Federated Learning Training Complete ---")

############################################ MOON Federated Model ##############################################################

class MOONFederatedLearning(FederatedLearningModel):
<<<<<<< HEAD
<<<<<<< HEAD
    def __init__(self, federated_model, features, federated_cluster='departement', loss='mse',
                 name='MoonFederatedModel', dir_log=Path('../'), under_sampling='full', over_sampling='full',
                 target_name='nbsinister', post_process=None, task_type='classification',
                 aggregation_method='max', nbfeatures='all', n_run=1, temperature=1, smooth=0, horizon=1):

        super().__init__(federated_model=federated_model, features=features, federated_cluster=federated_cluster, loss=loss,
                         name=name, dir_log=dir_log, under_sampling=under_sampling, over_sampling=over_sampling,
                         target_name=target_name, post_process=post_process, task_type=task_type,
                         aggregation_method=aggregation_method, nbfeatures=nbfeatures, n_run=n_run, horizon=horizon)

        self.moon_temperature_value = temperature
        self.smooth_value = smooth

        self.horizon = horizon
=======
=======
>>>>>>> 5b18034 ([Update code])
    def __init__(self, federated_model, features, federated_cluster='departement', loss='mse', 
                 name='MoonFederatedModel', dir_log=Path('../'), under_sampling='full', over_sampling='full',
                 target_name='nbsinister', post_process=None, task_type='classification', 
                 aggregation_method='max', nbfeatures='all', n_run=1, temperature=1, smooth=0):
        
        super().__init__(federated_model=federated_model, features=features, federated_cluster=federated_cluster, loss=loss,
                         name=name, dir_log=dir_log, under_sampling=under_sampling, over_sampling=over_sampling,
                         target_name=target_name, post_process=post_process, task_type=task_type,
                         aggregation_method=aggregation_method, nbfeatures=nbfeatures, n_run=n_run)
        
        self.moon_temperature_value = temperature
        self.smooth_value = smooth
<<<<<<< HEAD
>>>>>>> 5b18034 ([Update code])
=======
>>>>>>> 5b18034 ([Update code])
    
    def fit(self, df_train, df_val, df_test, graph, args):
        """
        Train local models for each federated cluster, aggregate them into a global model, 
        and stop training once the global score does not improve for patience_count_global epochs.
        """

        #importance_df = calculate_and_plot_feature_importance(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        #importance_df = calculate_and_plot_feature_importance_shapley(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        #features95, featuresAll = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)
        
        #if self.nbfeatures != 'all':
        #    self.features_name = featuresAll[:int(self.nbfeatures)]
        #else:
        #    self.features_name = featuresAll

        self.global_model.features_name = self.features_name
        self.global_model.nbfeatures = 'all'
        self.global_model.graph = graph

        initiate_model, model_params = self.global_model.make_model(graph, custom_model_params={'return_hidden' : True})

        # Vérifier que la méthode d'agrégation est implémentée
        if self.aggregation_method not in ['mean', 'median', 'weighted', 'max']:
            raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' is not implemented.")

        # Récupération des paramètres d'entraînement
        global_epochs = args.get('global_epochs')
        local_epochs = args.get('local_epochs')
        patience_count_global = args.get('patience_count_global')
        patience_count_local = args.get('patience_count_local')

        if self.federated_cluster in np.unique(df_train.columns):
            clusters = df_train[self.federated_cluster].unique()
        else:
            raise NotImplementedError(f"Aggregation method '{self.federated_cluster}' is not implemented.")
        
        print(f"\n--- Training Federated Model for {global_epochs} global epochs ---")

        tp = 'client-based'
        
        self.metrics = {}
        for run in range(self.n_run):

            self.global_model.model = deepcopy(initiate_model)
            self.global_model.model_params = deepcopy(model_params)
            local_models = {}
            
            best_global_score = float('-inf')
            patience_counter = 0  # Compteur pour l'arrêt anticipé
            seed = int(random.random())

            for epoch in range(global_epochs):
                print(f"\n--- Global Epoch {epoch + 1}/{global_epochs} ---")

                local_weights = []
                sample_counts = []
                
                for cluster in clusters:
                    print(f"\nTraining local model for cluster: {cluster}")

                    # Création des datasets pour le cluster fédéré
                    df_train_cluster, df_val_cluster, df_test_cluster = self.create_cluster_set(
                        df_train, df_val, df_test, cluster
                    )

                    if np.all(df_train[self.global_model.target_name].values == 0) or np.all(df_val[self.global_model.target_name].values == 0) or np.all(df_test[self.global_model.target_name].values == 0):
                        print(f'Skipping {cluster} due to no positif samples')
                        continue

                    if df_val_cluster.shape[0] == 0 or df_train_cluster.shape[0] == 0:
                        print(f'Skipping {cluster} due to empty dataset')
                        continue

                    # Initialisation du modèle local
                    if cluster in local_models.keys():
                        local_model = local_models[cluster]
                        local_model.model.load_state_dict(self.global_model.model.state_dict())
                    else:
                        local_model = deepcopy(self.global_model)
                        local_model.seed = seed
                        local_model.name = f'{self.federated_cluster}_{cluster}_{self.global_model.name}'
                        local_model.dir_log = self.dir_log / local_model.name
                        print(local_model.dir_log)
                        local_model.global_model = self.global_model
                        local_model.moon_temperature_value = self.moon_temperature_value
                        local_model.smooth_value = self.smooth_value
                        local_model.constrastive = True

                    if epoch == 0:
                        local_model.prev_model = self.global_model.model
                        local_model.constrastive = False
                        check_and_create_path(local_model.dir_log)
                    else:
                        local_model.prev_model = deepcopy(local_models[cluster].model)
                    
                    local_model.global_model = deepcopy(self.global_model.model)

                    local_model.features_name = self.features_name
                    local_model.nbfeatures = 'all'

                    if epoch == 0:
                        local_model.create_train_val_test_loader(graph, df_train_cluster, df_val_cluster, df_test_cluster, local_epochs, patience_count_local, CHECKPOINT, False, custom_model_params={'return_hidden' : True})
                        self.metrics[cluster] = local_model.metrics

                    local_model.train(graph, patience_count_local, CHECKPOINT, local_epochs, verbose=False, custom_model_params=None, new_model=False)
                    
                    local_models[cluster] = local_model

                    # Stocker les poids des modèles locaux
                    local_weights.append(deepcopy(local_model.model.state_dict()))
                    sample_counts.append(len(df_train_cluster))

                # Agréger les modèles locaux dans le modèle global
                self.aggregate_models(local_weights, sample_counts)

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
            
            loader = self.create_test_loader(graph, df_test)
            test_output, y = self._predict_test_loader(loader)
            test_output = test_output.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]
            
            metrics_run = evaluate_metrics(dff, self.target_name, test_output)
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'test')
            
            loader = self.create_test_loader(graph, df_val)
            test_output, y = self._predict_test_loader(loader)
            test_output = test_output.detach().cpu().numpy()
            
            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]
            
            metrics_run = evaluate_metrics(dff, self.target_name, test_output)
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'val')

        self.is_fitted_ = True
        print("\n--- Federated Learning Training Complete ---")

############################################ Proto Federated Learning ############################################

class ProtoFederatedLearning(FederatedLearningModel):
    def __init__(self, federated_model, features, federated_cluster='departement', loss='mse',
                 name='ProtoFederatedModel', dir_log=Path('../'), under_sampling='full', over_sampling='full',
                 target_name='nbsinister', post_process=None, task_type='classification', nbfeatures='all', n_run=1, prototype_weight=1.0,
                 horizon=1):

        super().__init__(federated_model=federated_model, features=features, federated_cluster=federated_cluster, loss=loss,
                         name=name, dir_log=dir_log, under_sampling=under_sampling, over_sampling=over_sampling,
                         target_name=target_name, post_process=post_process, task_type=task_type,
                         aggregation_method='median', nbfeatures=nbfeatures, n_run=n_run, horizon=horizon)

        self.prototype_weight = prototype_weight

        self.horizon = horizon
        self.global_prototypes = {}

    def compute_local_prototypes(self, model):
        prototypes = {}
        counts = {}
        loader = model.train_loader
        model.model.eval()
        with torch.no_grad():
            for data in loader:
                inputs, labels, edges = data
                _, hidden = model.model(inputs, edges)
                try:
                    target, _ = model.compute_weights_and_target(labels, -1, ids_columns, model.model.is_graph_or_node, None)
                except:
                    target, _ = model.compute_weights_and_target(labels, -1, ids_columns, False, None)

                target = target.long()
                for cls in torch.unique(target):
                    cls_idx = int(cls.item())
                    mask = target == cls
                    if mask.sum() == 0:
                        continue
                    
                    vec = hidden[mask].mean(dim=0)
                    if cls_idx in prototypes:
                        prototypes[cls_idx] += vec * mask.sum()
                        counts[cls_idx] += mask.sum()
                    else:
                        prototypes[cls_idx] = vec * mask.sum()
                        counts[cls_idx] = mask.sum()

        for k in prototypes:
            prototypes[k] /= counts[k]
        return prototypes

    def aggregate_prototypes(self, prototypes_list):
        aggregated = {}
        counts = {}
        for proto in prototypes_list:
            for cls, vec in proto.items():
                if cls in aggregated:
                    aggregated[cls] += vec
                    counts[cls] += 1
                else:
                    aggregated[cls] = vec.clone()
                    counts[cls] = 1
        for cls in aggregated:
            aggregated[cls] /= counts[cls]
        self.global_prototypes = aggregated

    def fit(self, df_train, df_val, df_test, graph, args):
        #importance_df = calculate_and_plot_feature_importance(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        #features95, featuresAll = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)

        if self.nbfeatures != 'all':
            self.features_name = featuresAll[:int(self.nbfeatures)]
        else:
            self.features_name = self.features_name

        self.global_model.features_name = self.features_name
        self.global_model.nbfeatures = 'all'
        self.global_model.graph = graph

        initiate_model, model_params = self.global_model.make_model(graph, custom_model_params={'return_hidden' : True})
        self.global_model.model = deepcopy(initiate_model)
        self.global_model.model_params = deepcopy(model_params)
        self.model_params = deepcopy(model_params)
        del initiate_model
        del model_params

        if self.aggregation_method not in ['mean', 'median', 'weighted', 'max']:
            raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' is not implemented.")

        global_epochs = args.get('global_epochs')
        local_epochs = args.get('local_epochs')
        patience_count_global = args.get('patience_count_global')
        patience_count_local = args.get('patience_count_local')

        if self.federated_cluster in np.unique(df_train.columns):
            clusters = df_train[self.federated_cluster].unique()
        else:
            raise NotImplementedError(f"Aggregation method '{self.federated_cluster}' is not implemented.")

        best_global_score = float('-inf')
        patience_counter = 0
        self.local_models = {}

        print(f"\n--- Training Federated Model for {global_epochs} global epochs ---")

        for epoch in range(global_epochs):
            print(f"\n--- Global Epoch {epoch + 1}/{global_epochs} ---")

            local_weights = []
            sample_counts = []
            local_prototypes = []

            for cluster in clusters:
                print(f"\nTraining local model for cluster: {cluster}")

                # Création des datasets pour le cluster fédéré
                df_train_cluster, df_val_cluster, df_test_cluster = self.create_cluster_set(
                    df_train, df_val, df_test, cluster
                )

                if np.all(df_train[self.global_model.target_name].values == 0) or np.all(df_val[self.global_model.target_name].values == 0) or np.all(df_test[self.global_model.target_name].values == 0):
                    print(f'Skipping {cluster} due to no positif samples')
                    continue

                if df_val_cluster.shape[0] == 0 or df_train_cluster.shape[0] == 0:
                    print(f'Skipping {cluster} due to empty dataset')
                    continue
                
                if epoch == 0:
                    local_model = deepcopy(self.global_model)
                else:
                    local_model = self.local_models[cluster]

                local_model.name = f'{self.federated_cluster}_{cluster}_{self.global_model.name}'
                local_model.dir_log = self.global_model.dir_log / local_model.name
                if epoch == 0:
                    check_and_create_path(local_model.dir_log)

                local_model.features_name = self.features_name
                local_model.nbfeatures = 'all'
                local_model.use_prototypes = len(self.global_prototypes) > 0
                local_model.prototype_weight = self.prototype_weight
                local_model.prototypes = self.global_prototypes
                
                if epochs == 0:
                    local_model.create_train_val_test_loader(graph, df_train_cluster, df_val_cluster, df_test_cluster, local_epochs, patience_count_local, CHECKPOINT, False, custom_model_params={'return_hidden' : True})
                    self.metrics[cluster] = local_model.metrics

                local_model.train(graph, patience_count_local, CHECKPOINT, local_epochs, verbose=False, custom_model_params=None, new_model=False)
                
                self.local_models[cluster] = local_model
                local_weights.append(deepcopy(local_model.model.state_dict()))
                sample_counts.append(len(df_train_cluster))
                local_prototypes.append(self.compute_local_prototypes(local_model))

            self.aggregate_prototypes(local_prototypes)

            global_score = self.score(df_val, df_val[self.target_name])
            print(f"\nGlobal Model Score after epoch {epoch + 1}: {global_score:.4f}")

            if global_score > best_global_score:
                best_global_score = global_score
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience_count_global:
                print("\nEarly stopping: Global model score did not improve.")
                break

        self.is_fitted_ = True
        print("\n--- Federated Learning Training Complete ---")

    def create_cluster_set(self, df_train, df_val, df_test, cluster):
        """
        Create training, validation, and test datasets for a given cluster.
        """
        if self.federated_cluster in np.unique(df_train.columns):
            X_cluster = df_train[df_train[self.federated_cluster] == cluster].reset_index(drop=True)
            X_val_cluster = df_val[df_val[self.federated_cluster] == cluster].reset_index(drop=True)
            X_test_cluster = df_test[df_test[self.federated_cluster] == cluster].reset_index(drop=True)

            return X_cluster, X_val_cluster, X_test_cluster
        
        raise NotImplementedError(f"Federated clustering method '{self.federated_cluster}' is not implemented.")

    def score_with_prediction(self, y_pred, y, sample_weight=None):
        
        return iou_score(y, y_pred)
    
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
        predictions, y = self.predict(X, return_y=True)
        y = y[:, -1]
        return self.score_with_prediction(predictions, y, sample_weight)
    
    def predict(self, df, graph=None, return_y=False):
        """
        Prédit les classes via le modèle local et les prototypes globaux.
        """
        if graph is None:
            graph = self.global_model.graph

        if self.target_name not in df.columns:
            df = df.copy()
            df[self.target_name] = 0

        if self.federated_cluster not in np.unique(df.columns):
            logger.info(f'{self.federated_cluster} must be in dataframe for inference')
            exit(1)

        final_pred = []
        ys = []
        clusters_values = df[self.federated_cluster].values
        for cluster in np.unique(clusters_values):

            loader = self.create_test_loader(graph, df[df[self.federated_cluster] == cluster])

            pred, y = self._predict_test_loader(loader, prediction_type='Class', cluster=cluster)
            
            final_pred.append(pred)
            ys.append(y)

        final_pred = torch.concatenate(final_pred, dim=0)
        ys = torch.concatenate(ys, dim=0)

        return (final_pred, ys) if return_y else pred

    def predict_proba(self, df, graph=None, return_y=False):
        """
        Renvoie les probabilités de classe via softmax inversé sur les distances aux prototypes.
        """
        if graph is None:
            graph = self.global_model.graph
            
        if self.target_name not in df.columns:
            df = df.copy()
            df[self.target_name] = 0

        final_pred = []
        clusters_values = df[self.federated_cluster].values
        ys = []
        for cluster in np.unique(clusters_values):
            loader = self.create_test_loader(graph, df[df[self.federated_cluster] == cluster])

            pred, y = self._predict_test_loader(loader, prediction_type='Probability', cluster=cluster)

            final_pred.append(pred)
            ys.append(y)
        
        final_pred = torch.concatenate(final_pred, dim=0)
        ys = torch.concatenate(ys, dim=0)

        return (final_pred, ys) if return_y else pred

    def _predict_test_loader(self, loader, prediction_type='Class', cluster = None):
        """
        Effectue la prédiction sur un DataLoader en utilisant le modèle local
        + les prototypes globaux (style FedProto).
        """
        if cluster is None:
            logger.info(f'Need to provide the correspond cluster of the batch ({self.federated_cluster})')
        assert cluster is not None
        if cluster in self.local_models.keys():
            model = self.local_models[cluster].model
            model.eval()
        else:
            model = None
        preds = []
        targets = []
        
        with torch.no_grad():
            for data in loader:

                inputs, y, _ = data
                y = y[:, :, -1]

                batch_preds = []
                
                # Obtenir l'embedding via le modèle local
                if model is not None:
                    _, embeddings = model(inputs)
                else:
                    embeddings = torch.zeros(inputs.shape[0])

                # Comparer aux prototypes globaux
                for emb in embeddings:
                    distances = {
                        cls: torch.norm(emb - proto.to(embeddings.device))
                        for cls, proto in self.global_prototypes.items()
                    }
                    if prediction_type == 'Probability':
                        # Convertir les distances en "scores inversés"
                        inv = torch.tensor(
                            [-d.item() for d in distances.values()]
                        )
                        probs = torch.softmax(inv, dim=0)
                        batch_preds.append(probs)
                    else:
                        pred_class = min(distances, key=distances.get)
                        batch_preds.append(pred_class)

                preds.extend(batch_preds)
                targets.extend(y.cpu().tolist())

        if prediction_type == 'Probability':
            preds = torch.stack(preds)  # [N, num_classes]
        else:
            preds = torch.tensor(preds)

        return preds, torch.tensor(targets)

############################################ KNOWNLEDEG DISTILLATION ##############################################################

class ModelKnowledgeDistillation(Training):
    def __init__(self, temperature, alpha, distillation_training_mode, teacher_name, student_name, model_name, batch_size, lr, out_channels, dir_log, features_name, ks, loss, name, device,
                under_sampling, over_sampling, nbfeatures, weight_type, target_name, task_type, teacher_loss, horizon=1):

        super().__init__(f'{model_name}', nbfeatures, batch_size, lr, target_name, task_type, features_name, ks, \
        out_channels, dir_log, loss=loss, name=name, device=device, under_sampling=under_sampling, over_sampling=over_sampling, horizon=horizon)
        
        self.teacher_loss = teacher_loss
        self.distillation_training_mode = distillation_training_mode
        self.teacher_name = teacher_name
        self.student_name = student_name
        self.weight_type = weight_type
        self.temperature = temperature
        self.alpha = alpha
        self.temperature_value = float(temperature) if temperature != 'search' else None
        self.alpha_value = float(alpha) if alpha != 'search' else None
        self.student_train = True
        self.load_teacher = True
        self.distillation_log = []

        if 'group' in self.distillation_training_mode:
            self.model_list = []

        self.horizon = horizon

    def create_train_val_test_loader(self, graph, df_train, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT, features_importance=True, custom_model_params=None, use_log=True):
        self.graph = graph

        #if self.teacher_name in sklearn_model_list or 'xgboost' in self.target_name:
        #    print(self.dir_log / '..' / 'baseline' / self.teacher_name)
        #    self.teacher = read_object(f'{self.teacher_name}.pkl', self.dir_log / '..' / 'baseline' / self.teacher_name)
        #else:
        #    self.teacher = read_object(f'{self.teacher_name}', self.dir_log / self.teacher_name)
        
        check_and_create_path(self.dir_log)

        if self.load_teacher:
            filter_name, model_type, hard_or_soft, weights_average, top_model = self.teacher_name.split('-')
            self.hard_or_soft = hard_or_soft
            self.weights_average = weights_average
            self.top_model = top_model
            full_teacher_name = f'{filter_name}-{model_type}_{self.under_sampling}_{self.over_sampling}_0_{self.nbfeatures}_{self.weight_type}_{self.target_name}_{self.task_type}_{self.teacher_loss}'
            self.teacher = read_object(f'{full_teacher_name}.pkl', self.dir_log / '..' / 'baseline' / full_teacher_name)
            self.load_teacher = False

        if self.distillation_training_mode == 'normal':
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
                            best_tp, find_log = self.search_samples_proportion(graph, df_train, df_val, df_test, is_unknowed_risk=False,
                                                                               epochs=epochs, PATIENCE_CNT=PATIENCE_CNT, CHECKPOINT=CHECKPOINT,
                                                                               custom_model_params=custom_model_params, use_log=use_log)
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

        self.df_train = df_train
        self.df_test = df_test
        self.df_val = df_val

        if self.distillation_training_mode != 'normal':
            return

        ##################################### Create loader #########################################
        if False:
            train_dataset = read_object('train_dataset.pkl', self.dir_log)
            val_dataset = read_object('val_dataset.pkl', self.dir_log)
            test_dataset = read_object('test_dataset.pkl', self.dir_log)
        else:
            train_dataset, val_dataset, test_dataset = create_dataset(graph,
                                                                self.df_train,
                                                                df_val,
                                                                df_test,
                                                                self.features_name,
                                                                self.target_name,
                                                                None,
                                                                self.device, self.ks)

            #save_object_torch(train_dataset, 'train_dataset.pkl', self.dir_log)
            #save_object_torch(val_dataset, 'val_dataset.pkl', self.dir_log)
            #save_object_torch(test_dataset, 'test_dataset.pkl', self.dir_log)

            train_loader = DataLoader(train_dataset, batch_size, True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(val_dataset, val_dataset.__len__(), False, worker_init_fn=seed_worker, generator=g)
            test_loader = DataLoader(test_dataset, test_dataset.__len__(), False, worker_init_fn=seed_worker, generator=g)

            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader

    def create_train_val_test_loader_teacher(self, graph, df_train, df_val, df_test, teacher, features_importance=True):
        self.graph = graph

        #if self.teacher_name in sklearn_model_list or 'xgboost' in self.target_name:
        #    print(self.dir_log / '..' / 'baseline' / self.teacher_name)
        #    self.teacher = read_object(f'{self.teacher_name}.pkl', self.dir_log / '..' / 'baseline' / self.teacher_name)
        #else:
        #    self.teacher = read_object(f'{self.teacher_name}', self.dir_log / self.teacher_name)
        
        check_and_create_path(self.dir_log)
        print(teacher.dir_log)

        percentage = read_object('test_percentage_scores.pkl', teacher.dir_log)
        test_percentage, under_prediction_score_scores, over_prediction_score_scores = percentage[0], percentage[1], percentage[2]

        score_differences = np.array(under_prediction_score_scores) - np.array(over_prediction_score_scores)
        index_max = np.argmin(np.abs(score_differences))
        best_tp = test_percentage[index_max]
 
        nb = int(best_tp * len(df_train[df_train[teacher.target_name] == 0]))

        df_combined = self.split_dataset(df_train, nb)
        
        self.df_train = df_combined
        self.df_test = df_test
        self.df_val = df_val

        ##################################### Create loader #########################################
        if False:
            train_dataset = read_object('train_dataset.pkl', self.dir_log)
            val_dataset = read_object('val_dataset.pkl', self.dir_log)
            test_dataset = read_object('test_dataset.pkl', self.dir_log)
        else:
            train_dataset, val_dataset, test_dataset = create_dataset(graph,
                                                                self.df_train,
                                                                df_val,
                                                                df_test,
                                                                self.features_name,
                                                                self.target_name,
                                                                None,
                                                                self.device, self.ks)

            #save_object_torch(train_dataset, 'train_dataset.pkl', self.dir_log)
            #save_object_torch(val_dataset, 'val_dataset.pkl', self.dir_log)
            #save_object_torch(test_dataset, 'test_dataset.pkl', self.dir_log)

            train_loader = DataLoader(train_dataset, batch_size, True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(val_dataset, val_dataset.__len__(), False, worker_init_fn=seed_worker, generator=g)
            test_loader = DataLoader(test_dataset, test_dataset.__len__(), False, worker_init_fn=seed_worker, generator=g)

            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader

    def create_test_loader(self, graph, df):
        loader = create_test_loader(graph, df,
                       self.features_name,
                       self.device,
                       None,
                       self.target_name,
                       self.ks)

        return loader
    
    def search_temperature_alpha(self, graph, PATIENCE_CNT, CHECKPOINT, epoch, verbose=True, custom_model_params=None, new_model=True):
        """temperature_grid = [4.0, 6.0] if self.temperature == 'search' else [self.temperature_value]
        alpha_grid = [.0, .1, .2, .3, .4, .5,] if self.alpha == 'search' else [self.alpha_value]
        
        if False and (self.dir_log / 'log_parameters.pkl').is_file():
            parameters = read_object('log_parameters.pkl', self.dir_log)
            self.temperature_value = parameters['temperature']
            self.alpha_value = parameters['alpha']
        else:
            score_0 = 0
            patience_i = 0
            patience_c = 5
            top_temp = 0
            top_al = 0
            for temp in temperature_grid:
                for al in alpha_grid:
                    
                    logger.info(f'########### temperature {temp}, alpha {al} #############')

                    self.temperature_value = temp
                    self.alpha_value = al
                    self.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose, custom_model_params, new_model, search=False)
                    score = self.score(self.df_test, self.df_test[self.target_name])
                    if score > score_0:
                        logger.info(f'temperature {temp}, alpha {al} -> {score}')
                        score_0 = score
                        top_temp = temp
                        top_al = al
                    else:
                        patience_i += 1
                        if patience_i == patience_c:
                            break
            parameters = {'temperature' : top_temp, 'alpha' : top_al}
            self.alpha_value = top_al
            self.temperature_value = top_temp
            save_object(parameters, 'log_parameters.pkl', self.dir_log)"""
        
        self.temperature_value = torch.nn.Parameter(torch.tensor(6.0))
        self.alpha_value = torch.nn.Parameter(torch.tensor(0.1))
            
    def train(self, graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose=True, custom_model_params=None, new_model=True, search=True):

        if (self.temperature == 'search' or self.alpha == 'search') and search:
            self.search_temperature_alpha(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose, custom_model_params, new_model)
            
        if self.distillation_training_mode == 'iterate':
            sub_teachers = np.asarray([estimator for estimator in self.teacher.best_estimator_])
            weights2use = self.teacher.weights_for_model
            weights2use = np.asarray(weights2use)
            key = np.argsort(weights2use)
            key = np.flip(key)
            sub_teachers = sub_teachers[np.asarray(key)]
            score_0 = 0
            
            for i, sub_teacher in enumerate(sub_teachers):
                
                if i > 0:
                    model_params_log = self.model.state_dict()
                    new_model = False
                else:
                    new_model = True

                self.create_train_val_test_loader_teacher(graph, self.df_train, self.df_val, self.df_test, sub_teacher, features_importance=False)

                super().train(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose, custom_model_params, new_model)

                test_output, y = self._predict_test_loader(self.test_loader)
                test_output = test_output.detach().cpu().numpy()

                y = y.detach().cpu().numpy()

                plt.figure(figsize=(15,5))
                plt.plot(y[y[:, departement_index] == 1, -1])
                plt.plot(test_output[y[:, departement_index] == 1])
                plt.savefig(self.dir_log / 'test.png')

                score = self.score(self.df_test, self.df_test[self.target_name])
                logger.info(f'Teacher : {sub_teacher.name} -> {score}')
                teacher_score = sub_teacher.score(self.df_test, self.df_test[self.target_name])
                logger.info(f'Teacher score -> {teacher_score}')
                if score > score_0:
                    score_0 = score
                else:
                    break
                    self.update_weight(model_params_log)

        elif self.distillation_training_mode == 'normal':
            super().train(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose, custom_model_params, new_model)

        elif 'group' in self.distillation_training_mode:
            nbgroup = int(self.distillation_training_mode.split('-')[1])
            if nbgroup == 'search':
                score_0 = 0
                for sub_teacher in self.teacher.best_estimator_:
                    
                    model_params_log = self.model.model.state_dict()
                    super().train(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose, custom_model_params, new_model)

                    score = self.score(self.df_test, self.df_test[self.target_name])
                    if score > score_0:
                        score_0 = score
                    else:
                        self.update_weight(model_params_log)
                        self.model_list.append(deepcopy(self.model))
                        score_0 = 0
            else:
                nbgroup = int(nbgroup)
                score_0 = 0
                current_group = 0
                for sub_teacher in self.teacher.best_estimator_:
                    
                    model_params_log = self.model.model.state_dict()
                    super().train(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose, custom_model_params, new_model)
                    
                    score = self.score(self.df_test, self.df_test[self.target_name])
                    if score > score_0:
                        score_0 = score
                    else:
                        self.update_weight(model_params_log)
                        self.model_list.append(deepcopy(self.model))
                        score_0 = 0
                        current_group += 1
                        if current_group > nbgroup:
                            break

    def get_temperature_alpha(self):
        return torch.nn.functional.softplus(self.temperature_value), torch.nn.functional.sigmoid(self.alpha_value)                   

    def _save_temperature_alpha_plot(self):

        # Extraction directe (car distillation_log est un dict {epoch: {"kappa":..., "xi":...}})
        
        print(self.distillation_log)
        kappas = [distillation_log["temperature"] for distillation_log in self.distillation_log]
        xis = [distillation_log["xi"] for distillation_log in self.distillation_log]
        epochs = [distillation_log["epoch"] for distillation_log in self.distillation_log]

        # Sauvegarde pickle
        dist_to_save = {"epoch": epochs, "temperature": kappas, "xi": xis}
        save_object(dist_to_save, 'egpd_kappa_xi.pkl', self.dir_log)

        # temperature vs epoch
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, kappas, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('temperature')
        plt.title('temperature over epochs')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(self.dir_log / 'temperature_over_epochs.png')
        plt.close()

        # alpha vs epoch
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, xis, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('alpha')
        plt.title('alpha over epochs')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(self.dir_log / 'alpha_over_epochs.png')
        plt.close()

############################################ VOTING MODEL ##############################################################

class ModelVotingPytorchAndSklearn(RegressorMixin, ClassifierMixin):
    def __init__(self, models, features, loss='mse', name='ModelVoting', dir_log=Path('../'), under_sampling='full', target_name='nbsinister', post_process=None, task_type='classification', horizon=1):
        """
        Initialize the ModelVoting class.

        Parameters:
        - models: A list of base models to use (must follow the sklearn API).
        - name: The name of the model.
        - loss: Loss function to use ('logloss', 'hinge_loss', 'mse', 'rmse', etc.).
        """
        super().__init__()
        self.best_estimator_ = models  # Now a list of models
        self.feature_names = features
        self.name = name
        self.loss = loss
        self.is_fitted_ = [False] * len(models)  # Keep track of fitted models
        self.features_per_model = []
        self.dir_log = dir_log
        self.post_process = post_process
        self.under_sampling = under_sampling
        self.target_name = target_name
        self.task_type = task_type

<<<<<<< HEAD
<<<<<<< HEAD
        self.horizon = horizon

=======
>>>>>>> 5b18034 ([Update code])
=======
>>>>>>> 5b18034 ([Update code])
    def fit(self, X, y, X_val, y_val, X_test, y_test, args, use_log=True):
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
            model.fit(graph, X, y, X_val, y_val, X_test, y_test, PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params=None, use_log=use_log)
            #else:
            #    model.fit(graph, X, y[targets[i]], X_val, y_val[targets[i]], training_mode=training_mode, optimization=optimization, grid_params=grid_params_list[i], fit_params=fit_params_list[i], cv_folds=cv_folds)
            target_name_model = model.target_name
            model.target_name = self.target_name
            print(f'Change target name {target_name_model} to {model.target_name}')
            test_loader = model.create_test_loader(graph, df_val)
            test_output, y_test_val = model._predict_test_loader(test_loader)
            test_output = test_output.detach().cpu().numpy()
            y_test_val = y_test_val.detach().cpu().numpy()[:, -1]
                
            model.target_name = target_name_model

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
                predict = self.predict_proba(X, return_y=False)[:, 1]

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

    def predict_with_weight(self, X, hard_or_soft='soft', weights_average='weight', weights2use=[], top_model='all', prediction_type="Class"):
        
        models_list = np.asarray([estimator.name for estimator in self.best_estimator_])
        weights2use = np.asarray(weights2use)
        
        if hard_or_soft == 'hard' or prediction_type == 'RawFormulaVal':
            if top_model != 'all':
                top_model = int(top_model)
                key = np.argsort(weights2use)
                models_list = models_list[key]
                models_list = models_list[-top_model:]
                #weights2use = weights2use[np.asarray(key)]
                #weights2use = weights2use[-top_model:]
            else:
                key = np.arange(0, len(self.best_estimator_))

            models_to_mean = []
            predictions = []
            for i, estimator in enumerate(self.best_estimator_):
                if estimator.target_name == self.target_name:
                    pred, y = estimator.predict(X, return_y=True, prediction_type=prediction_type)
                if estimator.name not in models_list:
                    continue
                else:
                    if estimator.target_name != self.target_name:
                        pred = estimator.predict(X, return_y=False, prediction_type=prediction_type)

                    predictions.append(pred.detach().cpu().numpy())

                models_to_mean.append(key[i])

            try:
                weights2use = weights2use[models_to_mean]
            except:
                pass
            # Aggregate predictions
            aggregated_pred = self.aggregate_predictions(predictions, models_to_mean, weights2use)
            #print(aggregated_pred)
            #print(y)
            return aggregated_pred, y.detach().cpu().numpy()
        elif hard_or_soft == 'None':
            top_model = int(top_model)
            key = np.argsort(weights2use)
            idx = key[-top_model]
            estimator = self.best_estimator_[idx]
            if estimator.target_name == self.target_name:
                pred, y = estimator.predict(X, return_y=True)
                return pred.detach().cpu().numpy(), y.detach().cpu().numpy(),
            else:
                pred = estimator.predict(X, return_y=False)
                y = None
                for estimator in self.best_estimator_:
                    if estimator.target_name == self.target_name:
                        _, y = estimator.predict(X, return_y=True)
                        return pred.detach().cpu().numpy(), y.detach().cpu().numpy(),
        else:
            aggregated_pred, y = self.predict_proba_with_weights(X, weights_average=weights_average, top_model=top_model, weights2use=weights2use)
            predictions = np.argmax(aggregated_pred, axis=1)
            return predictions, y

    def predict_proba_with_weights(self, X, hard_or_soft='soft', weights_average='weight', top_model='all', weights2use=[], id_col=(None, None)):
        """
        Predict probabilities for input data using each model and aggregate the results.

        Parameters:
        - X_list: List of data to predict probabilities for.
        
        Returns:
        - Aggregated predicted probabilities.
        """
        models_list = np.asarray([estimator.name for estimator in self.best_estimator_])
        weights2use = np.asarray(weights2use)

        if top_model != 'all':
                top_model = int(top_model)
                key = np.argsort(weights2use)
                models_list = models_list[np.asarray(key)]
                models_list = models_list[-top_model:]
                #weights2use = weights2use[np.asarray(key)]
                #weights2use = weights2use[-top_model:]
        else:
            key = np.arange(0, len(self.best_estimator_))
        
        probas = []
        models_to_mean = []
        print(models_list)

        if hard_or_soft == 'None':
            top_model = int(top_model)
            idx = np.argsort(weights2use)[-top_model]
            estimator = self.best_estimator_[idx]
            if estimator.target_name == self.target_name:
                pred, y = estimator.predict_proba(X, return_y=True)
                return pred.detach().cpu().numpy(), y.detach().cpu().numpy()
            else:
                pred = estimator.predict(X, return_y=False)
                y = None
                for estimator in self.best_estimator_:
                    if estimator.target_name == self.target_name:
                        _, y = estimator.predict_proba(X, return_y=True)
                        return pred.detach().cpu().numpy(), y.detach().cpu().numpy()

        for i, estimator in enumerate(self.best_estimator_):
            X_ = X
            if estimator.target_name == self.target_name:
                proba, y = estimator.predict_proba(X, return_y=True)
            if estimator.name not in models_list:
                continue
            else:
                if estimator.target_name != self.target_name:
                    proba = estimator.predict_proba(X_, return_y=False)
            if proba.shape[1] != 5:
                continue
            #print(estimator.name, np.asarray(probas).shape)
            models_to_mean.append(key[i])
            probas.append(proba)
        try:
            weights2use = weights2use[models_to_mean]
        except:
            pass
        # Aggregate probabilities
        aggregated_proba = self.aggregate_probabilities(probas, models_to_mean, weights2use)
        return aggregated_proba, y
    
    def predict_with_tasks(
        self,
        X,
        hard_or_soft="soft",
        weights_average="weight",
        model_per_task=None,
        generalized_departement=None,
        id_col=(None, None),
        prediction_type='Class'
    ):
        """Predict with a specific ``top_model`` per task.

        Parameters
        ----------
        X : pandas.DataFrame
            Input dataframe.
        hard_or_soft : str
            Mode passed to :func:`predict_with_weight`.
        weights_average : str
            Weighting mode for aggregation.
        model_per_task : dict
            Mapping of task names to number of models to use.
        generalized_departement : list
            Departments for which to apply ``generalized_prediction`` task.
        id_col : tuple
            Id column information for weighted predictions.
        """

        if model_per_task is None:
            model_per_task = {}
        
        if model_per_task == 'default':
            model_per_task={'normal_predictions' : 4,
                                'generalized_prediction' : 12,
                                'class_value_2_predictions' : 12,
                                'class_value_3_predictions' : 20,
                                'class_value_4_predictions' : 20
                                }
            
            generalized_departement = [
                1.,  2.,  3.,  4.,  5.,  8.,  9., 10., 12., 14.,
                15., 16., 17., 18., 19., 21., 22., 23., 24., 25.,
                26., 27., 28., 29., 31., 32., 35., 36., 37., 38.,
                39., 41., 42., 43., 44., 45., 46., 47., 48., 49.,
                50., 51., 52., 53., 54., 55., 56., 57., 58., 59.,
                60., 61., 62., 63., 64., 65., 67., 68., 69., 70.,
                71., 72., 73., 74., 75., 76., 77., 78., 79., 80.,
                81., 82., 85., 86., 87., 88., 89., 90., 91., 92.,
                93., 94., 95.
            ]
        

        # Normal prediction for all samples
        top_model = model_per_task.get("normal_predictions", "all")
        if prediction_type == 'Class' or prediction_type == 'RawFormulaVal':
            predictions = self.predict_with_weight(
                X,
                hard_or_soft=hard_or_soft,
                weights_average=weights_average,
                weights2use=self.weights_for_model,
                top_model=top_model,
                prediction_type=prediction_type
            )
        else:
            predictions = self.predict_proba_with_weights(
                X,
                hard_or_soft=hard_or_soft,
                weights_average=weights_average,
                weights2use=self.weights_for_model,
                top_model=top_model,
                prediction_type=prediction_type
            )

        predictions = np.asarray(predictions)

        # Generalized prediction
        if (
            model_per_task.get("generalized_prediction") is not None
            and generalized_departement is not None
            and "departement" in X.columns
        ):
            mask = X["departement"].isin(generalized_departement).values
            if mask.any():
                if prediction_type == 'Class' or prediction_type == 'RawFormulaVal':
                    preds_gen  = self.predict_with_weight(
                        X[mask],
                        hard_or_soft=hard_or_soft,
                        weights_average=weights_average,
                        weights2use=self.weights_for_model,
                        top_model=model_per_task["generalized_prediction"],
                        prediction_type=prediction_type
                    )
                else:
                    preds_gen  = self.predict_proba_with_weights(
                        X[mask],
                        hard_or_soft=hard_or_soft,
                        weights_average=weights_average,
                        weights2use=self.weights_for_model,
                        top_model=model_per_task["generalized_prediction"],
                        prediction_type=prediction_type
                    )
                mask = np.isin(y[:, 4], generalized_departement)
                predictions[mask] = preds_gen

        for val in [2, 3, 4]:
            task_name = f"class_value_{val}_predictions"
            if task_name in model_per_task:
                if prediction_type == 'Class' or prediction_type == 'RawFormulaVal':
                    preds_cls = self.predict_with_weight(
                        X,
                        hard_or_soft=hard_or_soft,
                        weights_average=weights_average,
                        weights2use=self.weights_for_model,
                        top_model=model_per_task[task_name],
                        prediction_type=prediction_type
                    )
                    mask = (preds_cls >= val) | (predictions >= val)
                    if mask.any():
                        predictions[mask] = preds_cls[mask]
                else:
                    preds_cls = self.predict_proba_with_weights(
                        X,
                        hard_or_soft=hard_or_soft,
                        weights_average=weights_average,
                        weights2use=self.weights_for_model,
                        top_model=model_per_task[task_name],
                        prediction_type=prediction_type
                    )
                    # prendre les lignes où la classe la plus probable est >= val
                    mask = (np.argmax(preds_cls, axis=1) >= val) | (np.argmax(predictions, axis=1) >= val)
                    if mask.any():
                        predictions[mask] = preds_cls[mask]

        return predictions

    def predict(self, X, hard_or_soft='soft', weights_average='weight', top_model='all', id_col=(None, None), prediction_type="Class"):
        """
        Predict labels for input data using each model and aggregate the results.

        Parameters:
        - X_list: List of data to predict labels for.

        Returns:
        - Aggregated predicted labels.
        """

        if weights_average not in ['None', 'weight']:
            assert id_col[0] is not None and id_col[1] is not None
            vals = id_col[1]
            unique_ids = np.unique(vals)
            prediction = np.empty(X.shape[0], dtype=int)
            y = np.empty(X.shape[0], dtype=int)
            for id in unique_ids:
                print(f'Prediction for {id_col[0]} {id}')
                mask = (id_col[1] == id)
                prediction[mask], y[mask] = self.predict_with_weight(X[mask], hard_or_soft=hard_or_soft, weights_average='weight', weights2use=self.weights_id_model[id_col[0]][id], top_model=top_model, prediction_type=prediction_type)
            return prediction, y
        else:
            return self.predict_with_weight(X, hard_or_soft=hard_or_soft, weights_average='weight', weights2use=self.weights_for_model, top_model=top_model,  prediction_type=prediction_type)

        """print(f'Predict with {hard_or_soft} and weighs at {weights_average}')
        if hard_or_soft == 'hard':
            if top_model != 'all':
                top_model = int(top_model)
                key = np.argsort(self.weights_for_model)
                models_list = models_list[key]
                models_list = models_list[-top_model:]
            else:
                key = np.arange(0, len(self.best_estimator_))

            predictions = []
            for i, estimator in enumerate(self.best_estimator_):
                if estimator.name not in models_list:
                    continue
                else:
                    pred = estimator.predict(X)
                    predictions.append(pred)

                models_to_mean.append(key[i])

            # Aggregate predictions
            aggregated_pred = self.aggregate_predictions(predictions, models_to_mean, weights_average)
            return aggregated_pred
        else:
            aggregated_pred = self.predict_proba(X, weights_average, top_model)
            predictions = np.argmax(aggregated_pred, axis=1)
            return predictions"""

    def predict_proba(self, X, weights_average='weight', top_model='all', id_col=(None, None)):
        """
        Predict probabilities for input data using each model and aggregate the results.

        Parameters:
        - X_list: List of data to predict probabilities for.
        
        Returns:
        - Aggregated predicted probabilities.
        """

        if weights_average not in ['None', 'weight']:
            assert id_col[0] is not None and id_col[1] is not None
            vals = id_col[1]
            unique_ids = np.unique(vals)
            prediction = np.empty(X.shape[0], dtype=int)
            y = np.empty(X.shape[0], dtype=int)
            for id in unique_ids:
                print(f'Prediction for {id_col[0]} {id}')
                mask = (id_col[1] == id)
                prediction[mask], y[mask] = self.predict_proba_with_weights(X[mask], hard_or_soft='soft', weights_average='weight', weights2use=self.weights_id_model[id_col[0]][id], top_model=top_model)
            return prediction, y

        else:
            return self.predict_proba_with_weights(X, hard_or_soft='soft', weights_average='weight', weights2use=self.weights_for_model, top_model=top_model)

        """models_list = np.asarray([estimator.name for estimator in self.best_estimator_])

        if top_model != 'all':
                top_model = int(top_model)
                key = np.argsort(self.weights_for_model)
                models_list = models_list[np.asarray(key)]
                models_list = models_list[-top_model:]
        else:
            key = np.arange(0, len(self.best_estimator_))
        
        print(models_list)
        probas = []
        models_to_mean = []
        for i, estimator in enumerate(self.best_estimator_):
            if estimator.name not in models_list:
                continue
            X_ = X
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(X_)
                if proba.shape[1] != 5:
                    continue
                #print(estimator.name, np.asarray(probas).shape)
                models_to_mean.append(key[i])
                probas.append(proba)
            else:
                raise AttributeError(f"The model at index {i} does not support predict_proba.")
            
        # Aggregate probabilities
        aggregated_proba = self.aggregate_probabilities(probas, models_to_mean, weights_average)
        return aggregated_proba"""

    def aggregate_predictions(self, predictions_list, models_to_mean, weight2use=[], id_col=(None, None), prediction_type="RawFormulaVal"):
        """
        Aggregate predictions from multiple models with weights.

        Parameters:
        - predictions_list: List of predictions from each model.

        Returns:
        - Aggregated predictions.
        """
        
        if prediction_type == 'RawFormulaVal' or prediction_type == 'Probability':
            return self.aggregate_probabilities(predictions_list, models_to_mean, weight2use=weight2use, id_col=id_col)

        predictions_array = np.array(predictions_list)
        if len(weight2use) == 0 or weight2use is None:
            weight2use = np.ones_like(self.weights_for_model)[models_to_mean]

        if self.task_type == 'classification' or self.task_type == 'ordinal-classification':
            # Weighted vote for classification
            unique_classes = np.arange(0, 5)
            weighted_votes = np.zeros((len(unique_classes), predictions_array.shape[1]))

            for i, cls in enumerate(unique_classes):
                mask = (predictions_array == cls)
                weighted_votes[i] = np.sum(mask * weight2use.reshape(mask.shape[0], 1), axis=0)

            aggregated_pred = unique_classes[np.argmax(weighted_votes, axis=0)]
        else:
            # Weighted average for regression
            weighted_sum = np.sum(predictions_array * weight2use[:, None], axis=0)
            aggregated_pred = weighted_sum / np.sum(weight2use)
            #aggregated_pred = np.max(predictions_array * weight2use[:, None], axis=0)
        
        return aggregated_pred

    """def aggregate_predictions_id(self, predictions_array, models_to_mean, id_col=(None, None)):
        assert id_col[0] is not None and id_col[1] is not None
        id = id_col[0]
        vals = id_col[1]
        uvals = np.unique(vals)

        weight2use = np.zeros((len(models_to_mean), predictions_array.shape[1]))
        for val in uvals:
            mask = (vals == val)
            weight2use[:, mask] = self.weights_id_model[id][val][models_to_mean]
            if np.all(weight2use[:, mask] == 0):
                weight2use[:, mask] = self.weights_for_model[models_to_mean]

        unique_classes = np.arange(0, 5)
        weighted_votes = np.zeros((len(unique_classes), predictions_array.shape[1]))

        for i, cls in enumerate(unique_classes):
            mask = (predictions_array == cls)
            weighted_votes[i] = np.sum(mask * weight2use, axis=0)

        aggregated_pred = unique_classes[np.argmax(weighted_votes, axis=0)]
        return aggregated_pred"""   

    def aggregate_probabilities(self, probas_list, models_to_mean, weight2use=[], id_col=(None, None)):
        """
        Aggregate probabilities from multiple models with weights.

        Parameters:
        - probas_list: List of probability predictions from each model.

        Returns:
        - Aggregated probabilities.
        """
        probas_array = np.array(probas_list)
        if weight2use is None or len(weight2use) == 0:
            weight2use = np.ones_like(self.weights_for_model)[models_to_mean]
        
        # Weighted average for probabilities
        weighted_sum = np.sum(probas_array * weight2use[:, None, None], axis=0)
        aggregated_proba = weighted_sum / np.sum(weight2use)
        #aggregated_proba = np.max(probas_array * weight2use[:, None, None], axis=0)
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
    
class ModelPerID(RegressorMixin, ClassifierMixin):
    def __init__(self, model, dir_log, cluster="departement", horizon=1):
        self.base_model = model
        self.cluster_col = cluster
        self.models = {}
        self.is_fitted_ = False
        self.name = f'unique-{cluster}-{model.name}'

        self.horizon = horizon
        self.dir_log = dir_log

    def fit(self, df_train, df_val, df_test, graph, PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params, **args):
        """Train one model per cluster value."""
        self.models = {}
        values = df_train[self.cluster_col].unique()
        for val in values:
            train_subset = df_train[df_train[self.cluster_col] == val].reset_index(drop=True)
            val_subset = df_val[df_val[self.cluster_col] == val].reset_index(drop=True)
            test_subset = df_test[df_test[self.cluster_col] == val].reset_index(drop=True)
            model = deepcopy(self.base_model)
            model.dir_log = self.base_model.dir_log / f'unique-{val}-{self.base_model.name}'
            if hasattr(model, "name"):
                model.name = f"{val}_{model.name}"
            if hasattr(model, "fit"):
                args['PATIENCE_CNT'] = PATIENCE_CNT
                args['CHECKPOINT'] = CHECKPOINT
                args['epochs'] = epochs
                args['custom_model_params'] = custom_model_params
                model.fit(train_subset, val_subset, test_subset, graph, **args)
            else:
                raise ValueError("Base model must implement fit method")
            self.models[val] = model
        self.is_fitted_ = True

    def predict_proba(self, X, **kwargs):
        proba = None
        result = []
        for val, model in self.models.items():
            mask = X[self.cluster_col] == val
            if mask.any():
                preds = model.predict_proba(X[mask], **kwargs)
                if proba is None:
                    proba = np.zeros((len(X), preds.shape[1]))
                proba[mask] = preds
        return proba

    def predict(self, X, **kwargs):
        preds = np.zeros(len(X))
        for val, model in self.models.items():
            mask = X[self.cluster_col] == val
            if mask.any():
                preds[mask] = model.predict(X[mask], **kwargs)
        return preds

    def _predict_test_loader(self, loader):
        preds_list = []
        ys_list = []
        for val, model in self.models.items():
            pred, y = model._predict_test_loader(loader)
            preds_list.append(pred)
            ys_list.append(y)
        return torch.cat(preds_list, 0), torch.cat(ys_list, 0)

class Model_susceptibility():
    def __init__(self, model_name, target, resolution, model_config, features_name, out_channels, task_type, ks, departements, train_departements, train_date, val_date, dir_log, horizon=1):
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

            self.horizon = horizon
            self.model_params = None
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
    
    def create_model_and_train(self, graph, root_target):
        params = self.model_config['params']
        
        X_train, y_train, X_val, y_val, X_test, y_test, dept_test, years_test = self.create_numpy_data(root_target)
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
            self.model = ModelCNN(model_name=self.susceptility_mapper_name,
                                    nbfeatures='all',
                                    batch_size=batch_size,
                                    lr=params['lr'],
                                    target_name=self.target,
                                    out_channels=params['out_channels'],
                                    features_name=features,
                                    ks=params['k_days'],
                                    dir_log=self.dir_log / 'susecptibility_map_features' / self.model_config['type'],
                                    name=f'{self.susceptility_mapper_name}_{self.model_config["infos"]}',
                                    task_type=params['task_type'],
                                    loss=self.model_config['loss'],
                                    device=device,
                                    features=self.features_name,
                                    over_sampling='full',
                                    under_sampling='full',
                                    image_per_node='image_per_departement',
                                    path=self.dir_log
                                    )

            self.model.train_loader = train_loader
            self.model.test_loader = test_loader
            self.model.val_loader = val_loader
            if (self.dir_log / 'susecptibility_map_features' / self.model_config['type'] / 'best.pt').is_file():
                temp_model, _ = self.make_model(custom_model_params=self.model_config['params'])
                self.model._load_model_from_path(self.dir_log / 'susecptibility_map_features' / self.model_config['type'] / 'best.pt', temp_model)
            else:
                temp_model, _ = self.make_model(custom_model_params=self.model_config['params'])
                self.model.update_model(temp_model)
                self.model.train(params, PATIENCE_CNT=params['PATIENCE_CNT'], CHECKPOINT=params['CHECKPOINT'], epochs=params['epoch'], custom_model_params=self.model_config['params'], new_model=False)

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
        logger.info(f'{y_train.shape}, {y_val.shape}, {y_test.shape}')

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
        val_dataset = AugmentedIndef test_placeGraphDataset(
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
        model, params = make_model(self.model.model_name, len(self.features_name), len(self.features_name),
                                None, dropout, activation,
                                self.ks,
                                out_channels=self.out_channels,
                                task_type=self.task_type,
                                device=device, num_lstm_layers=num_lstm_layers,
                                custom_model_params=custom_model_params)
        
        if self.model_params is None:
            self.model_params = params

        return model, params

    def test_model(self, root_target):
        check_and_create_path(self.dir_log / 'test')
        name_i = 0
        with torch.no_grad():
            mae_func = torch.nn.L1Loss(reduce='none')
            mae = 0
            itest = 0
            for data in self.test_loader:
                X, y, _ = data
                logger.info(f'{torch.max(X)}')
                output = self.model.model(X)
                for b in range(y.shape[0]):
                    loss = mae_func(output[b, 0], y[b, :, :, -1])
                    
                    yb = y[b].detach().cpu().numpy()
                    outputb = output[b].detach().cpu().numpy()
                    
                    #logger.info(f'loss {departement}: {loss.item()} {np.max(outputb)}')

                    if MLFLOW:
                        existing_run = get_existing_run(f'susceptibility_{departement}_{self.model_name}')
                        if existing_run:
                            mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
                        else:
                            mlflow.start_run(run_name=f'susceptibility_{departement}_{self.model_name}', nested=True)
                        
                        mlflow.log_metric('MAE', loss.item())

                    mae += loss.item()

                    fig, ax = plt.subplots(1, 2, figsize=(15,5))
                    #raster = read_object(f'{names[b]}.pkl', dir_output / 'database')
                    #assert raster is not None
                    #raster = raster[0]

                    #output_image = resize_no_dim(outputb[0], raster.shape[0], raster.shape[1])
                    #y_image = resize_no_dim(yb[:,  :], raster.shape[0], raster.shape[1])

                    output_image = outputb[0]
                    y_image = yb[:, :, -1]

                    output_image[np.isnan(y_image)] = np.nan
                    #y_image[np.isnan(raster)] = np.nan

                    maxi = max(np.nanmax(output_image), np.nanmax(y_image))

                    im0 = ax[0].imshow(output_image, vmin=0, vmax=maxi)
                    ax[0].set_title('Prediction map')
                    cbar0 = plt.colorbar(im0, ax=ax[0], orientation='vertical')
                    cbar0.set_label('Feature Value')

                    im1 = ax[1].imshow(y_image, vmin=0, vmax=maxi)
                    ax[1].set_title('Ground truth')
                    cbar1 = plt.colorbar(im1, ax=ax[1], orientation='vertical')
                    cbar1.set_label('Feature Value')

                    plt.tight_layout()
                    plt.savefig(self.dir_log / 'test' / f'{name_i}.png')
                    plt.close('all')
                    
                    if MLFLOW:
                        mlflow.log_figure(fig, f'_{name_i}.png')
                        mlflow.end_run()

                    name_i += 1

                itest += y.shape[0]
                        
            logger.info(f'MAE on test set : {mae / itest}')
