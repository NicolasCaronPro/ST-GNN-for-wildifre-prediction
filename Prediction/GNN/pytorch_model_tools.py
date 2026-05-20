import numpy as np
import math
import logging
import optuna
import gc
import psutil
import tracemalloc
import resource
import os
import random
from torch_geometric.data import Dataset
from torch.utils.data import DataLoader
import torch
from torch import optim
import torch.nn.functional as F

torch.set_printoptions(precision=3, sci_mode=False)

from PIL import Image
import torchvision.transforms.functional as TF

from copy import deepcopy
import itertools
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from GNN.discretization import *
from GNN.tools import (
    calculate_area_under_curve,
    under_prediction_score,
    over_prediction_score,
    iou_score,
    evaluate_metrics,
    calculate_ic95,
    Scoring,
)
from GNN.config import graph_id_index, departement_index, date_index, logger
from sklearn.metrics import f1_score, jaccard_score

from GNN.graph_builder import *
from GNN.tools import check_and_create_path, save_object, read_object
from forecasting_models.pytorch.distillation_utils import RelationMLP, RelationAttention, Adapter, FitNet, multi_teacher_kd_loss, multi_teacher_kd_loss_global_weights, lht_loss, angle_triplet_loss, confidence_distillation_loss
from forecasting_models.pytorch.student_distillation import StudentMLP

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

def plot_score_per_epochs(score_per_epoch, dir_output, name):
    plt.figure(figsize=(15,5))
    scores = score_per_epoch['score']
    epochs = score_per_epoch['epoch']
    plt.plot(epochs, scores)
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.savefig(dir_output / f'{name}.png')
    plt.close('all')

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

def construct_dataset(date_ids, x_data, y_data, graph, ids_columns, ks, horizon, use_temporal_as_edges, isNotmesh=False, proportion_0_sample_with_positive_weight=1.0):
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
    print(ks, horizon)
    for id in date_ids:
        if use_temporal_as_edges is None:
            x, y = construct_time_series(id, x_data, y_data, ks, horizon, len(ids_columns), proportion_0_with_positive_weight=proportion_0_sample_with_positive_weight)
            if x is not None and isNotmesh:
                for i in range(x.shape[0]):
                    Xs.append(x[i])
                    Ys.append(y[i])
            elif x is not None:
                Xs.append(x)
                Ys.append(y)
            continue
        elif use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, id, x_data, y_data, ks, horizon, len(ids_columns), proportion_0_with_positive_weight=proportion_0_sample_with_positive_weight)
        else:
            x, y, e = construct_graph_with_time_series(graph, id, x_data, y_data, ks, horizon, len(ids_columns), proportion_0_with_positive_weight=proportion_0_sample_with_positive_weight)

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
                    horizon: int,
                    graph_mesh=None,
                    gridh2mesh=None,
                    mesh2graph=None,
                    proportion_0_with_positive_weight=1.0
                    ):
    
    x_train, y_train = df_train[ids_columns + features_name].values, df_train[ids_columns + targets_columns + [target_name]].values
    
    x_val, y_val = df_val[ids_columns + features_name].values, df_val[ids_columns + targets_columns + [target_name]].values

    x_test, y_test = df_test[ids_columns + features_name].values, df_test[ids_columns + targets_columns + [target_name]].values

    dateTrain = np.sort(np.unique(y_train[y_train[:, weight_index] > 0, date_index]))
    dateVal = np.sort(np.unique(y_val[y_val[:, weight_index] > 0, date_index]))
    dateTest = np.sort(np.unique(y_test[y_test[:, weight_index] > 0, date_index]))

    logger.info(f'{dateTrain.shape}, {dateVal.shape}, {dateTest.shape}')

    logger.info(f'Constructing train Dataset')
    Xst, Yst, Est = construct_dataset(dateTrain, x_train, y_train, graph, ids_columns, ks, horizon, use_temporal_as_edges, graph_mesh is None, proportion_0_with_positive_weight)

    logger.info(f'Constructing val Dataset')
    XsV, YsV, EsV = construct_dataset(dateVal, x_val, y_val, graph, ids_columns, ks, horizon, use_temporal_as_edges, graph_mesh is None, proportion_0_with_positive_weight)

    logger.info(f'Constructing test Dataset')
    XsTe, YsTe, EsTe = construct_dataset(dateTest, x_test, y_test, graph, ids_columns, ks, horizon, use_temporal_as_edges, graph_mesh is None, proportion_0_with_positive_weight)

    # Assurez-vous que les ensembles ne sont pas vides
    assert len(Xst) > 0, "Le jeu de données d'entraînement est vide"
    assert len(XsV) > 0, "Le jeu de données de validation est vide"
    assert len(XsTe) > 0, "Le jeu de données de test est vide"

    if graph_mesh is None:
        # Création des datasets finaux
        print('graph_mesh')
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
                    horizon:int,
                    graph_mesh=None,
                    gridh2mesh=None,
                    mesh2graph=None,
                    proportion_0_sample_with_positive_weight=1.0
                    ):

    x_train, y_train = df_train[ids_columns + features_name].values, df_train[ids_columns + targets_columns + [target_name]].values
    
    #print('weight', df_train['weight'].unique())

    dateTrain = np.sort(np.unique(y_train[y_train[:, weight_index] > 0, date_index]))

    logger.info(f'{dateTrain.shape}')

    logger.info(f'Constructing train Dataset')
    Xst, Yst, Est = construct_dataset(dateTrain, x_train, y_train, graph, ids_columns, ks, horizon, use_temporal_as_edges, graph_mesh is None, proportion_0_sample_with_positive_weight)

    # Assurez-vous que les ensembles ne sont pas vides
    assert len(Xst) > 0, "Le jeu de données d'entraînement est vide"

    if graph_mesh is None:
        # Création des datasets finaux
        print('graph_mesh')
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
                    horizon: int,
                    graph_mesh=None,
                    gridh2mesh=None,
                    mesh2graph=None,
                    proportion_0_sample_with_positive_weight=1.0
                    ):
        
    x_val, y_val = df_val[ids_columns + features_name].values, df_val[ids_columns + targets_columns + [target_name]].values

    x_test, y_test = df_test[ids_columns + features_name].values, df_test[ids_columns + targets_columns + [target_name]].values

    dateVal = np.sort(np.unique(y_val[y_val[:, weight_index] > 0, date_index]))
    dateTest = np.sort(np.unique(y_test[y_test[:, weight_index] > 0, date_index]))

    logger.info(f'{dateVal.shape}, {dateTest.shape}')

    logger.info(f'Constructing val Dataset')
    XsV, YsV, EsV = construct_dataset(dateVal, x_val, y_val, graph, ids_columns, ks, horizon, use_temporal_as_edges, graph_mesh is None)

    logger.info(f'Constructing test Dataset')
    XsTe, YsTe, EsTe = construct_dataset(dateTest, x_test, y_test, graph, ids_columns, ks, horizon, use_temporal_as_edges, graph_mesh is None)
    
    # Assurez-vous que les ensembles ne sont pas vides
    assert len(XsV) > 0, "Le jeu de données de validation est vide"
    if len(XsTe) == 0:
        XsTe, YsTe, EsTe = XsV, YsV, EsV

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
                       ks :int,
                       horizon:int,
                       ):

    Xset, Yset = df[ids_columns + features_name].values, df[ids_columns + targets_columns].values

    X = []
    E = []
    Y = []

    graphId = np.unique(Xset[:, date_index])
    for date in graphId:
        if use_temporal_as_edges is None:
            x, y = construct_time_series(date, Xset, Yset, ks, horizon, len(ids_columns), 1.0)
            if x is not None:
                for i in range(x.shape[0]):
                    X.append(x[i])
                    Y.append(y[i])
            continue
        elif use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, date, Xset, Yset, ks, horizon, len(ids_columns), 1.0)
        else:
            x, y, e = construct_graph_with_time_series(graph, date, Xset, Yset, ks, horizon, len(ids_columns), 1.0)

        if x is None:
            continue

        if x.shape[0] == 0:
            continue

        X.append(x)
        Y.append(y)
        if 'e' in locals():
            E.append(e)

    return np.asarray(X), np.asarray(Y), np.asarray(E)

def create_test_loader(graph, df,
                       features_name,
                       device : torch.device,
                       use_temporal_as_edges : bool,
                       target_name,
                       ks :int,
                       horizon:int,
                       graph_mesh=None,
                        gridh2mesh=None,
                        mesh2graph=None,
                        proportion_0_sample_witg_positive_weight=1.0):

    if 'DFE' not in df.columns:
        df['DFE'] = 0
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
            x, y = construct_time_series(date, Xset, Yset, ks, horizon, len(ids_columns), 1.0)
            if x is not None:
                for i in range(x.shape[0]):
                    X.append(x[i])
                    Y.append(y[i])
            continue
        elif use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, date, Xset, Yset, ks, horizon, len(ids_columns), 1.0)
        else:
            x, y, e = construct_graph_with_time_series(graph, date, Xset, Yset, ks, horizon,len(ids_columns), 1.0)

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

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def build_dataframe(
    inputs_horizon: torch.Tensor,   # shape: (X, F, T)
    labels: torch.Tensor,           # shape: (X, L, T)
    features_name,                  # list[str], len F
    ids_columns,                    # list[str]
    targets_columns,                # list[str]
    target_name                     # str
) -> pd.DataFrame:
    # --- vérifs de shapes ---
    X, F, T = inputs_horizon.shape
    X2, L, T2 = labels.shape
    assert X == X2,  f"X mismatch: {X} vs {X2}"
    assert T == T2,  f"T mismatch: {T} vs {T2}"
    expected_L = len(ids_columns) + len(targets_columns) + 1
    assert L == expected_L, f"L mismatch: got {L}, expected {expected_L}"

    # --- (X, F, T) -> (X, T, F) -> (X*T, F)
    inputs_np = to_numpy(inputs_horizon).transpose(0, 2, 1).reshape(-1, F)

    # --- (X, L, T) -> (X, T, L) -> (X*T, L)
    labels_np = to_numpy(labels).transpose(0, 2, 1).reshape(-1, L)

    # --- colonnes ---
    label_cols = list(ids_columns) + list(targets_columns) + [target_name]
    all_cols = list(map(str, features_name)) + list(map(str, label_cols))

    # --- DataFrame final (large) ---
    data = np.concatenate([inputs_np, labels_np], axis=1)
    df = pd.DataFrame(data, columns=all_cols)

    # Toujours des noms de colonnes str
    df.columns = df.columns.astype(str)
    return df

class WrapperModel(torch.nn.Module):
    def __init__(self, original_model, F, T, edges, y_background, horizon=0):
        super().__init__()
        self.model = original_model
        self.F = F
        self.T = T
        self.edges = edges
        self.y_background = y_background
        self.horizon = horizon

    def forward(self, x_flat):
        # reshape x_flat (B, F*T) vers (B, F, T)
        x_orig = x_flat.reshape(-1, self.F, self.T)

        logits, y = self.model._predict_tensor((x_orig, self.y_background, self.edges), prediction_type="RawFormulaVal", use_grad=True)

        # Select the desired horizon
        # logits shape is (B, Horizon+1, OutChannels)
        res = logits[:, self.horizon, :]
        
        loss_lower = self.model.loss.lower() if hasattr(self.model, 'loss') else ""
        task_type = self.model.task_type if hasattr(self.model, 'task_type') else ""
        out_channels = self.model.out_channels if hasattr(self.model, 'out_channels') else res.shape[-1]
        
        if "pdegpd" in loss_lower:
            pmf = self.model.criterion.pmf_all(res, from_logits=True)
            y_vals = torch.arange(pmf.size(-1), device=pmf.device, dtype=pmf.dtype)
            score = (pmf * y_vals).sum(dim=-1, keepdim=True)
            
        elif task_type == "classification":
            if out_channels == 1:
                score = torch.sigmoid(res)
            else:
                probs = torch.softmax(res, dim=-1)
                y_vals = torch.arange(probs.size(-1), device=probs.device, dtype=probs.dtype)
                score = (probs * y_vals).sum(dim=-1, keepdim=True)
            
        elif task_type == "regression" and out_channels == 1:
            score = res
        else:
            raise ValueError(f"SHAP explanation not defined for task_type={task_type}, out_channels={out_channels}, loss={loss_lower}")

        # Ensure output is strictly 2D (B, 1) for SHAP deep_pytorch compatibility
        if score.ndim == 1:
            score = score.unsqueeze(-1)
        elif score.ndim > 2:
            score = score.view(score.shape[0], -1)
            
        return score

class Training():
    def __init__(self, model_name, nbfeatures, batch_size, lr, delta_lr, patience_cnt_lr, target_name, task_type,
                 features_name, ks, out_channels, dir_log,
                 loss='mse', name='Training', device='cpu',
                 under_sampling='full', over_sampling='full', n_run=1,
                 horizon=0, post_process=None, loss_param_search=False):
        
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
        self.delta_lr = delta_lr
        self.patience_cnt_lr = patience_cnt_lr
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
        self.prox_term = False
        self.prox_value = 0.05
        self.use_prototypes = False
        self.prototype_weight = 1.0
        self.prototypes = None
        self.ALATraining = False
        self.area_parameters = None
        # Distillation tracking (best/worst losses per epoch)
        self.distill_best_log = []   # list of dicts: {epoch, graph_id, loss}
        self.distill_worst_log = []  # list of dicts: {epoch, graph_id, loss}
        self.criterion_params = []
        self.criterion = None
        self._current_epoch = None
        self.seed = None
        self.horizon = horizon
        self.apply_discretization = post_process is not None
        self.post_process = post_process
        self.loss_param_search = loss_param_search
        self.scoring = Scoring()

        if 'Past_risk' in self.features_name:
            self.id_past_risk = features_name.index('Past_risk')
        else:
            self.id_past_risk = None
        if 'Past_burnedarea' in self.features_name:
            self.id_past_ba = features_name.index('Past_burnedarea')
        else:
            self.id_past_ba = None

        self.prev_idx = []

        if self.task_type == "classification" or self.task_type == "corn":
            # Pour classification : les colonnes one-hot sont du type f"{colunm}_prev_<classe>"
            new_features = [f"{self.target_name}_prev_{i}" for i in range(self.out_channels)]
            self.prev_idx = [self.features_name.index(f) for f in new_features if f in self.features_name]
            print(self.prev_idx)

        elif self.task_type == "binary":
            # Pour binaire : on a colunm_prev_bin, colunm_prev_bin_0 et colunm_prev_bin_1
            new_features = [f"{self.target_name}_prev_bin"] + [f"{self.target_name}_prev_bin_{i}" for i in range(self.out_channels)]
            self.prev_idx = [self.features_name.index(f) for f in new_features if f in self.features_name]

        elif self.task_type == "regression" and self.target_name in ["nbsinister", "burnedarea"]:
            # Pour régression : une seule feature ajoutée
            new_features = [f"{self.target_name}_prev"]
            self.prev_idx = [self.features_name.index(f) for f in new_features if f in self.features_name]
        
        if len(self.prev_idx) == 0:
            self.prev_idx = None
        
        # History for loss components decomposition
        self.loss_components_history = {
            'loss_total': [],
            'loss_trans': [],     # Task loss (raw)
            'global_loss_trans': [],  # Global transition loss
            'entropy_pi': [],     # Entropy (raw)
            'entropy_weighted': [],  # Entropy contribution to loss
            'mu0_term': [],       # Mu0 term (already weighted)
            'dirichlet_reg': [],  # Dirichlet reg (raw)
            'dirichlet_weighted': [],  # Dirichlet contribution to loss
            'ce_loss': [],        # CE loss (raw)
            'ce_weighted': [],    # CE contribution to loss
            'epoch': [],
            
            # Detailed scaling stats
            'scale_min': [],
            'scale_mean': [],
            'scale_max': [],
            'diff_raw_mean': [],
            'diff_scaled_mean': [],
            'margin_mean': []
        }

    def log_memory(self, stage):
        """
        Log memory usage (RSS, MaxRSS, Tracemalloc).
        """
        try:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / 1024 / 1024
            
            # Resource (ru_maxrss is in KB on Linux)
            usage = resource.getrusage(resource.RUSAGE_SELF)
            maxrss_mb = usage.ru_maxrss / 1024
            
            # Tracemalloc
            try:
                current, peak = tracemalloc.get_traced_memory()
                current_mb = current / 1024 / 1024
                peak_mb = peak / 1024 / 1024
                trace_str = f" | Trace: {current_mb:.2f} MB (Peak: {peak_mb:.2f} MB)"
            except:
                trace_str = ""
            
            logger.info(f"[MEMORY] {stage} | RSS: {rss_mb:.2f} MB | MaxRSS: {maxrss_mb:.2f} MB{trace_str}")
        except Exception as e:
            logger.warning(f"Failed to log memory: {e}")

    def remove_graph(self):
        del self.graph
        
    def clean(self):
        try:
            del self.df_train
        except:
            pass
        try:
            del self.df_val
        except:
            pass
        try:
            del self.df_test
        except:
            pass
        try:
            del self.train_loader
        except:
            pass
        try:
            del self.val_loader
        except:
            pass
        try:
            del self.test_loader
        except:
            pass
    
    def free_memory(self):
        """
        Aggressively free memory by explicitly deleting heavy attributes.
        Used in search_samples_proportion to clean up deep-copied models.
        """
        # Delete dataloaders
        try:
            del self.train_loader
        except:
            pass
        try:
            del self.val_loader
        except:
            pass
        try:
            del self.test_loader
        except:
            pass
        
        # Delete dataframes
        try:
            del self.df_train
        except:
            pass
        try:
            del self.df_val
        except:
            pass
        try:
            del self.df_test
        except:
            pass
        
        # Delete graph
        try:
            del self.graph
        except:
            pass
        
        # Delete model and optimizer
        try:
            del self.optimizer
        except:
            pass
        try:
            del self.model
        except:
            pass

    def compute_weights_and_target(self, labels, band, ids_columns, is_grap_or_node, graphs, H):
        weight_idx = ids_columns.index('weight')
        target_is_binary = self.task_type == 'binary'

        if len(labels.shape) == 3:
            weights = labels[:, weight_idx, H]
            target = (labels[:, band, H] > 0).long() if target_is_binary else labels[:, band, H]

        elif len(labels.shape) == 5:
            weights = labels[:, :, :, weight_idx, H]
            target = (labels[:, :, :, band, H] > 0).long() if target_is_binary else labels[:, :, :, band, H]

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
    
    def compute_inputs(self, inputs, H, time_steps):
        if H + 1 == 0:
            if len(inputs.shape) == 3:
                inputs_horizon = inputs[:, :, -(self.ks + 1):]

            elif len(inputs.shape) == 5:
                inputs_horizon = inputs[:, :, :, :, -(self.ks + 1):]

            elif len(inputs.shape) == 4:
                inputs_horizon = inputs
            else:
                inputs_horizon = inputs
        
        else:
            if len(inputs.shape) == 3:
                inputs_horizon = inputs[:, :, H - self.ks:H + 1]

            elif len(inputs.shape) == 5:
                inputs_horizon = inputs[:, :, :, :, H - self.ks:H + 1]

            elif len(inputs.shape) == 4:
                inputs_horizon = inputs
            else:
                inputs_horizon = inputs

        if inputs_horizon.ndim % 2 == 0:
                inputs_horizon = inputs_horizon[:, :, None]
            
        return inputs_horizon
    
    def compute_single_loss(self, out, tar, wei, hidden, clusters_ids=None, tolong=False, areas=None, criterion=None, departement_ids=None, graph_ids=None, dates=None):
        if self.task_type == 'regression':
            tar = tar.view(out.shape[0])
            wei = wei.view(out.shape[0])

            tar = torch.masked_select(tar, wei.gt(0))
            out = out[wei.gt(0)]
            if hidden is not None:
                hidden = hidden[wei.gt(0)]
            if graph_ids is not None:
                graph_ids = graph_ids[wei.gt(0)]
            if dates is not None:
                dates = dates[wei.gt(0)]
            if clusters_ids is not None:
                clusters_ids = clusters_ids[wei.gt(0)]
            if departement_ids is not None:
                departement_ids = departement_ids[wei.gt(0)]
            if areas is not None:
                areas = areas[wei.gt(0)]
            wei = torch.masked_select(wei, wei.gt(0))
        else:
            wei = wei.long()

            tar = tar[wei.gt(0)]
            out = out[wei.gt(0)]

            if clusters_ids is not None:
                clusters_ids = clusters_ids[wei.gt(0)]
            if departement_ids is not None:
                departement_ids = departement_ids[wei.gt(0)]
            if areas is not None:
                areas = areas[wei.gt(0)]
            
            if hidden is not None:
                hidden = hidden[wei.gt(0)]
            
            if graph_ids is not None:
                graph_ids = graph_ids[wei.gt(0)]
            
            if dates is not None:
                dates = dates[wei.gt(0)]

            wei = torch.masked_select(wei, wei.gt(0))

            if tolong:
                tar = tar.long()
                
        additionnal_params = {}

        if clusters_ids is not None:
            additionnal_params['clusters_ids'] = clusters_ids
        
        if areas is not None:
            additionnal_params['areas'] = areas

        if departement_ids is not None:
            additionnal_params['departement_ids'] = departement_ids
            
        req_params = required_params(criterion.forward)
        
        if 'hidden' in req_params:
            additionnal_params['hidden'] = hidden
        
        if 'graph_ids' in req_params:
            additionnal_params['graph_ids'] = graph_ids
        
        if 'dates' in req_params:
            additionnal_params['dates'] = dates
            
        try:
            additionnal_params['sample_weight'] = wei
        
            return criterion(out, tar, **additionnal_params)
        except Exception as e:
            print(e)
            return criterion(out, tar)

    def calculate_loss(self, criterion, output, target, weights, hidden, label, tolong=True):
        
        departement_ids = None
        
        if 'clusters_ids' in required_params(criterion.forward):
            if hasattr(criterion, 'id') and criterion.id is not None:
                if criterion.id == -1 :
                    clusters_ids = torch.ones(target.shape[0], device=target.device)
                else:
                    clusters_ids = label[:, criterion.id, -1]
                    self.cluster_id_index = criterion.id
            else:
                # If id is not defined, use the whole batch as a single cluster
                clusters_ids = torch.zeros(target.shape[0], device=target.device)
        else:
            clusters_ids = None

        if 'departement_ids' in required_params(criterion.forward):
            departement_ids = label[:, departement_index, -1]

        if 'areas' in required_params(criterion.forward):
            areas = label[:, area_index, -1]

        else:
            areas = None

        if 'graph_ids' in required_params(criterion.forward):
            graph_ids = label[:, graph_id_index, -1]
        else:
            graph_ids = None
            
        if 'dates' in required_params(criterion.forward):
            dates = label[:, date_index, -1]
        else:
            dates = None
            
        base_loss = self.compute_single_loss(output, target, weights, hidden, clusters_ids, tolong, areas, criterion, departement_ids=departement_ids, graph_ids=graph_ids, dates=dates)
        
        if 'area' in self.loss and False: # Calculate area loss (specify loss-area)
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

            if 'area-global' in self.loss and False:  # Calculate area * global (classic) loss  (specify loss-area-global)
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
    
    def pick_params_by_name(self, model, names):
        names, params = [], []
        for n, p in model.named_parameters():
            for pick_parm in names:
                if pick_parm in n:
                    names.append(n); params.append(p)
        return names, params
    
    def calculate_prox_term(self, names):
        prox = 0.0

        model_params = self.pick_params_by_name(self.model, names)
        global_model_params = self.pick_params_by_name(self.global_model, names)

        for (name, p), (_, pg) in zip(model_params, global_model_params):
            if not p.requires_grad:
                continue

            if not name in names:
                continue

            prox = prox + (p - pg.detach()).pow(2).sum()

        return 0.5 * self.prox_value * prox

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
    
    def model_distillation_loss(self, loss, inputs_horizon, labels, target, logits, hiddens):
        """
        Compute knowledge distillation loss combining task loss and KL divergence.
        
        Formula: loss = alpha * task_loss + (1 - alpha) * T² * KL(student || teacher)
        
        Args:
            loss: Original task loss (cross-entropy)
            inputs_horizon: Input features
            labels: Ground truth labels
            logits: Student model logits
            hiddens: Student model hidden states
            
        Returns:
            Combined loss following Hinton et al. convention
        """
        criterion_teacher = self.get_loss('kldivloss', {})
        #df_test = build_dataframe(inputs_horizon, labels, self.features_name, ids_columns, targets_columns, self.target_name)
        # 

        teacher_logits = []
        teacher_feats = []
        weights = []

        # Teachers
        models_to_mean, _, _ = self.teacher.get_weights(self.top_model, return_self_model_idx=True)
        leni = len(models_to_mean)
        with torch.no_grad():
            for idx in models_to_mean:
                t_wrapper = self.teacher.best_estimator_[idx]
                assert t_wrapper.features_name == self.features_name
                    
                if t_wrapper.target_name == self.target_name:
                    continue

                t_model = t_wrapper.model
                t_model.eval()
                try:
                    _, t_log, t_feat = t_model(inputs_horizon, self.graph)
                except:
                    _, t_log, t_feat = t_model(inputs_horizon)
                teacher_logits.append(t_log)
                teacher_feats.append(t_feat)
                weights.append(self.teacher.weights_for_model[idx])

        T = self.temperature_value

        weights = np.asarray(weights)
        weights = weights / np.sum(weights)
        weights = torch.as_tensor(weights, device=logits.device, dtype=torch.float32)
        
        device = logits.device
            
        if self.distillation_training_mode == 'normal':
            kl_div_loss, _, _ = multi_teacher_kd_loss_global_weights(logits,  teacher_logits, weights, T=self.temperature_value)
            #print(f'kl_div_loss {kl_div_loss}, loss {loss}')
            loss =  (1 - self.alpha_value) * loss + (self.alpha_value) * kl_div_loss
            return loss
        
        elif self.distillation_training_mode == 'AdaptativeMLP':
            # AdaptativeMLP specific loss logic

            # Access intermediates stored in model
            # Handle DataParallel if necessary
            student_model = self.model
            
            # Adapter
            student_rep = hiddens[-1]

            weights = self.adapter(student_rep)
            
            # LKD
            T = self.temperature_value
            loss_kd, fused_soft, t_soft_all = multi_teacher_kd_loss(
                logits,
                teacher_logits,
                weights,
                T=T
            )

            # LHT
            n_group = student_model.n_group
            chunk_size = (len(teacher_feats) + n_group - 1) // n_group
            student_feats_expanded = []
            for i in range(len(teacher_feats)):
                group_idx = i // chunk_size
                if group_idx > self.model.n_group:
                    continue
                if group_idx >= n_group: group_idx = n_group - 1
                student_feats_expanded.append(hiddens[group_idx])
                
            loss_lht = lht_loss(teacher_feats, student_feats_expanded, self.fitnets)
            
            # LAngle
            student_soft = F.softmax(logits / T, dim=-1)
            loss_angle = angle_triplet_loss(fused_soft, student_soft)
            
            loss = loss + self.alpha_value * loss_kd + self.beta_value * loss_lht + self.gamma_value * loss_angle
            return loss

        elif self.distillation_training_mode == 'MATTKD':
            # Fusion teacher with attention layer
            t_l = torch.stack(teacher_feats, dim=1)
            super_teacher_logits = self.relation_att(t_l)
            kl_div_loss, _, _ = multi_teacher_kd_loss_global_weights(logits,  [super_teacher_logits], [1], T=self.temperature_value)

            # LHT
            n_group = self.model.n_group
            chunk_size = (len(teacher_feats) + n_group - 1) // n_group
            student_feats_expanded = []
            for i in range(len(teacher_feats)):
                group_idx = i // chunk_size
                if group_idx > self.model.n_group:
                    continue
                if group_idx >= n_group: group_idx = n_group - 1
                student_feats_expanded.append(hiddens[group_idx])
                
            loss_lht = lht_loss(teacher_feats, student_feats_expanded, self.fitnets)

            loss = loss + self.alpha_value * kl_div_loss + self.beta_value * loss_lht
            return loss

        elif self.distillation_training_mode == 'RelationMLP':
            # RelationMLP mode: Combined embedding and logits distillation
            # L = L_CE(y, student(x)) + α × ||f_S(x) - f_E(x)||² + β × KL(p_S(x) || p̄_T(x))

            kl_div_loss, _, _ = multi_teacher_kd_loss_global_weights(logits,  teacher_logits, weights, T=self.temperature_value)
            kl_div_loss
            # 3) Embedding distillation: L2 distance between student logits and ensemble embedding
            # Get ensemble embedding from RelationMLP (concatenates all teacher logits)
            t_l = torch.stack(teacher_feats, dim=1)
            ensemble_embedding = self.relation_mlp(t_l)  # [B, num_classes]
            
            # Student embedding: use logits directly (before softmax)
            if isinstance(hiddens, list):
                student_embedding = hiddens[-1]
            else:
                student_embedding = hiddens  # [B, num_classes]
            
            # Compute L2 loss (MSE)
            embedding_loss = F.mse_loss(student_embedding, ensemble_embedding)
            
            # 4) Combine all losses: L_CE + α × embedding_loss + β × kl_div_loss
            #print(f'kl_div_loss {kl_div_loss}, loss {loss}, embedding_loss {embedding_loss}')
            loss = loss + self.beta_value * embedding_loss + self.alpha_value * kl_div_loss
            return loss
        
        elif self.distillation_training_mode == 'RelationATT':
            # RelationATT mode: Combined embedding and logits distillation using Attention
            # L = L_CE(y, student(x)) + α × ||f_S(x) - f_E(x)||² + β × KL(p_S(x) || p̄_T(x))
            
            kl_div_loss, _, _ = multi_teacher_kd_loss_global_weights(logits,  teacher_logits, weights, T=self.temperature_value)
            
            # 2) Embedding distillation: L2 distance between student logits and ensemble embedding
            # Get ensemble embedding from RelationAttention
            t_l = torch.stack(teacher_feats, dim=1)
            ensemble_embedding = self.relation_att(t_l)  # [B, num_classes]

            # Student embedding: use logits directly (before softmax)
            if isinstance(hiddens, list):
                student_embedding = hiddens[-1]
            else:
                student_embedding = hiddens  # [B, num_classes]
            
            # Compute L2 loss (MSE)
            embedding_loss = F.mse_loss(student_embedding, ensemble_embedding)
            
            # 3) Combine all losses: L_CE + α × embedding_loss + β × kl_div_loss
            loss = loss + self.beta_value * embedding_loss + self.alpha_value * kl_div_loss
            return loss
        
        elif self.distillation_training_mode == 'Confidence':
            # Confidence distillation
            
            # Collect teacher classifiers
            teacher_classifiers = []
            for idx in models_to_mean:
                t_wrapper = self.teacher.best_estimator_[idx]
                if t_wrapper.target_name == self.target_name: continue
                
                # Assume teacher model has output_layer (Linear)
                # If not available, we might fail. 
                # For StudentMLP teachers it is available.
                if hasattr(t_wrapper.model, 'output_layer'):
                    teacher_classifiers.append(t_wrapper.model.output_layer)
                elif hasattr(t_wrapper.model, 'fc'): # ResNet style
                     teacher_classifiers.append(t_wrapper.model.fc)
                elif hasattr(t_wrapper.model, 'classifier'): 
                     teacher_classifiers.append(t_wrapper.model.classifier)
                else:
                    # Fallback or error? 
                    # For now let's assume it exists or use a dummy identity if we can't find it
                    # But we need it for w_inter calculation.
                    raise ValueError(f"Teacher model {t_wrapper.name} does not have a known classifier layer (output_layer, fc, classifier)")

            # Student feature (last hidden)
            if isinstance(hiddens, list):
                student_feat = hiddens[-1]
            else:
                student_feat = hiddens
                
            # Call confidence loss
            loss = loss + confidence_distillation_loss(
                student_logits=logits,
                student_feat=student_feat,
                teacher_logits_list=teacher_logits,
                teacher_feat_list=teacher_feats,
                labels=target,
                fitnets=self.fitnets,
                teacher_classifiers=teacher_classifiers,
                alpha=self.alpha_value,
                beta=self.beta_value,
                T=self.temperature_value
            )
            
            return loss
        
    def launch_batch(self, data, criterion, batch_type, do_update):
        inputs, labels, _ = data
        graphs = None
        
        # --- Verification of timeseries consistency on LABELS ---
        if labels.dim() == 3: # (B, C, T)
            B_l, C_l, T_l = labels.shape
            for b in range(B_l):
                # 1. Check graph_id consistency in labels
                gids_l = labels[b, graph_id_index, :]
                if not torch.all(gids_l == gids_l[0]):
                    raise ValueError(f"CRITICAL: graph_id inconsistency in labels for batch {b}. GIDs: {gids_l.cpu().numpy()}")
                
                # 2. Check date sequentiality in labels (step of 1)
                dates_l = labels[b, date_index, :]
                if not (dates_l[-1] == torch.max(dates_l)):
                    raise ValueError(f"CRITICAL: last date is not the maximum in labels for batch {b}. Dates: {dates_l.cpu().numpy()}")
                
                if T_l > 1:
                    diffs_l = torch.diff(dates_l)
                    if not torch.all(diffs_l == 1):
                        raise ValueError(f"CRITICAL: date sequence inconsistency in labels for batch {b}. Dates: {dates_l.cpu().numpy()}")
                
                # 3. Check weights: only the target window should have weight > 0
                weights_l = labels[b, weight_index, :]
                past_len = T_l - (self.horizon + 1)
                if past_len > 0:
                    if not torch.all(weights_l[:past_len] == 0):
                        raise ValueError(f"CRITICAL: weights non-zero in historical window for batch {b}. Weights: {weights_l.cpu().numpy()}")

        # --- Verification of dynamic variables variation in INPUTS ---
        from GNN.tools import get_static_temporal_idx
        _, dynamic_idx = get_static_temporal_idx(self.features_name)
        if len(dynamic_idx) > 0 and inputs.dim() == 3:
            B, C, T = inputs.shape
            if T > 1:
                for b in range(B):
                    # Check if all dynamic variables are constant (suspicious)
                    all_constant = True
                    for idx in dynamic_idx:
                        feat_values = inputs[b, idx, :]
                        if not torch.all(feat_values == feat_values[0]):
                            all_constant = False
                            break
                    if all_constant:
                         raise ValueError(f"CRITICAL: All temporal variables are constant in inputs for batch {b}. The timeseries might be incorrectly constructed (repeated days).")

        if torch.isnan(inputs).any():
            print(f">>> [DEBUG launch_batch] inputs contains NaN! Shape: {inputs.shape}")
        
        if inputs.shape[0] == 1:
            return 0, 0

        band = -1
        total_loss = 0
        
        hidden_past: List[torch.Tensor] = []  # contiendra des tenseurs (B, D)
        output_past: List[torch.Tensor] = []  # contiendra des tenseurs (B, D)

        is_tfn = self.model_name in ['TFN', 'itransformer']
        output_all = logits_all = hidden_all = None
        
        for H in range(self.horizon + 1):
            
            if hasattr(self.model, 'is_graph_or_node'):
                is_graph_or_node = self.model.is_graph_or_node
            else:
                is_graph_or_node = False

            target, weights = self.compute_weights_and_target(labels, band, ids_columns, is_graph_or_node, graphs,  -1 - (self.horizon - H))

            if self.loss not in ['kldivloss']: # works on probability
                target = target.long()

            if is_tfn:
                if H == 0:
                    # Appel unique du modèle — toutes les prédictions d'horizons retournées d'un coup
                    inputs_horizon = self.compute_inputs(inputs, -1 - self.horizon, "current")
                    output_all, logits_all, hidden_all = self.model(inputs_horizon, z_prev=None)
                    if do_update and has_method(criterion, 'update_after_batch'):
                        criterion.update_after_batch(logits_all[:, 0, :], target)
                # Extraction du slice correspondant à l'horizon H
                output = output_all[:, H, :]
                logits = logits_all[:, H, :]
                hidden = hidden_all[:, H, :]
            else:
                if H == 0:
                    inputs_horizon = self.compute_inputs(inputs, -1 - (self.horizon - H), "current")
                    # Store the current features for persistence
                    inputs_horizon_persistent = inputs_horizon.clone()
                    z_prev = None
                    output, logits, hidden = self.model(inputs_horizon, z_prev=None)
                    if do_update:
                        if has_method(criterion, 'update_after_batch'):
                            criterion.update_after_batch(logits, target)
                else:
                    # Use persistence (features from H=0)
                    inputs_horizon = inputs_horizon_persistent.clone()
                    if self.ks > 0:
                        # on prend les ks derniers états cachés déjà vus (detached to prevent BPTT from H>0 to H=0)
                        history = [h.detach() for h in hidden_past[-(self.ks + 1):]]
                        # empilement (B, D, L) avec L = len(history)
                        z_prev = torch.stack(history, dim=2)  # (B, D, L)

                        # padding à gauche si L < ks
                        L = z_prev.size(2)
                        if L < (self.ks + 1):
                            B, D = z_prev.size(0), z_prev.size(1)
                            pad = torch.zeros(
                                (B, D, self.ks + 1 - L),
                                device=z_prev.device,
                                dtype=z_prev.dtype
                            )
                            z_prev = torch.cat([pad, z_prev], dim=2)  # (B, D, ks)
                    else:
                        z_prev = hidden_past[-1]

                    if self.id_past_risk is not None:
                        inputs_horizon[:, self.id_past_risk, -H:] = 0
                    if self.id_past_ba is not None:
                        inputs_horizon[:, self.id_past_ba, -H:] = 0
                    if self.prev_idx is not None:
                        # Remplacer les valeurs de la feature par les prédictions passées (detached to prevent BPTT)
                        inputs_horizon[:, self.prev_idx, -H:] = torch.stack([o.detach() for o in output_past], dim=2)
    
                    output, logits, hidden = self.model(inputs_horizon, z_prev=z_prev)
            
            hidden_past.append(hidden)
            output_past.append(output)
            
            loss_res = self.calculate_loss(criterion, logits, target, weights, hidden, labels)
            
            #if torch.isnan(loss_res):
            #    print('####################################')
            #    print('output', torch.unique(output))
            #    print('logits', torch.unique(logits))
            #    print('target', torch.unique(target))

            if isinstance(loss_res, dict):
                loss = loss_res['total_loss']
            else:
                loss = loss_res
            
            if self.student_train: # distallation traning
                loss = self.model_distillation_loss(loss, inputs_horizon, labels, target, logits, hidden)
                
            if self.constrastive: # MOON federated training
                _, _, zprev = self.prev_model(inputs)
                _, _, zglob = self.global_model(inputs)
                loss_constrastive = self.calculate_contrastive_moon_loss(hidden, zprev, zglob, self.moon_temperature_value)
                loss = loss + self.smooth_value * loss_constrastive
            
            if self.prox_term:
                prox_term = self.calculate_prox_term(self.fed_prox_names)
                loss = loss + prox_term

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
            
            total_loss = total_loss + loss
                
        # Clean up intermediate tensors before returning
        try:
            del inputs, labels, target, weights
        except:
            pass
        try:
            del output, logits, hidden
        except:
            pass
        try:
            del inputs_horizon
        except:
            pass
        try:
            # Clean up lists of tensors
            for t in hidden_past:
                del t
            del hidden_past
        except:
            pass
        try:
            for t in output_past:
                del t
            del output_past
        except:
            pass
        try:
            del loss
        except:
            pass

        return total_loss, loss_res

    def launch_train_loader(self, loader, criterion, optimizer, do_update):

        self.model.train()
        
        if has_method(criterion, 'get_learnable_parameters'):
            criterion.train()

        # Initialize per-epoch aggregation for distillation best/worst
        if 'distillation' in self.loss:
            self._epoch_distill_best = {'loss': float('inf'), 'graph_id': None}
            self._epoch_distill_worst = {'loss': float('-inf'), 'graph_id': None}
        
        res_loss = 0.0
        res_loss_dict = {'l': 0.0}

        for i, data in enumerate(loader, 0):
            if i == 0:
                pass
                
            loss, loss_res = self.launch_batch(data, criterion, 'train', False)

            if isinstance(loss, int) or isinstance(loss, float):
                print(f'loss is does not required grad {loss}')
                continue
            
            #if optimizer is not None:
            #    optimizer.zero_grad()
            #    try:
            #        loss.backward()
                    # Clip gradients to prevent exploding gradients causing NaN weights
            #        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            #    except:
            #        continue

            try:
                optimizer.zero_grad()
                loss.backward()
            except Exception as e:
                continue
            
            if 'res_loss' in locals():
                res_loss += loss.item()
                if isinstance(loss_res, dict):
                    for key in loss_res:
                        val = loss_res[key]
                        if torch.is_tensor(val):
                            val = val.item()
                        if key in res_loss_dict:
                            res_loss_dict[key] += val
                        else:
                            res_loss_dict[key] = val
                else:
                    val = loss_res
                    if torch.is_tensor(val):
                        val = val.item()
                    res_loss_dict['l'] += val
            else:
                res_loss = loss.item()
                if isinstance(loss_res, dict):
                    res_loss_dict = {k: (v.item() if torch.is_tensor(v) else v) for k, v in loss_res.items()}
                else:
                    val = loss_res
                    if torch.is_tensor(val):
                        val = val.item()
                    res_loss_dict = {'l': val}

            if self.ALATraining:
                # Mises à jour SANS autograd
                with torch.no_grad():
                    # 1) update des weights
                    for p_t, p_prev, p_g, w in zip(
                            self.params_p, self.params_tp, self.params_gp, self.weights):
                        upd = w - self.eta * ((p_g - p_prev) * p_t.grad)
                        w.copy_(torch.clamp(upd, 0.0, 1.0))

                    #if not self.ala_weight_only:
                    #    # 2) calcul des params interpolés
                    #    for p_t, p_prev, p_g, w in zip(
                    #            self.params_p, self.params_tp, self.params_gp, self.weights):
                    #        p_t.copy(p_t - self.eta * (p_g - p_prev) * (p_t.grad))

            if optimizer is not None and not getattr(self, 'ala_weight_only', False):
                optimizer.step()

                #self.update_weight()
            
            # Clean up tensors from this batch
            try:
                del loss
            except:
                pass
            try:
                del loss_res
            except:
                pass
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
                if isinstance(value, torch.Tensor):
                    dict_params[name] = value.detach().cpu().numpy()
                else:
                    dict_params[name] = deepcopy(value)
            
            self.criterion_params.append(dict_params)

        if len(loader) > 0:
            res_loss /= len(loader)
            for k in res_loss_dict:
                if isinstance(res_loss_dict[k], (int, float, torch.Tensor)):
                    res_loss_dict[k] /= len(loader)

        return res_loss, res_loss_dict

    def launch_val_test_loader(self, loader, criterion, teacher=None):

        self.model.eval()

        if has_method(criterion, 'get_learnable_parameters'):
            criterion.eval()

        total_loss = 0.0

        total_loss_dict = {}

        with torch.no_grad():

            for i, data in enumerate(loader, 0):
                
                loss, loss_res = self.launch_batch(data, criterion, 'val', do_update=True)
                
                try:
                    if torch.isnan(loss):
                        continue
                except:
                    continue
                if loss is not None:
                    if torch.is_tensor(loss):
                        total_loss += loss.item()
                    else:
                        total_loss += loss
                else:
                    # Should not happen ideally, but safety first
                    pass

                if isinstance(loss_res, dict):
                    for key in loss_res:
                        val = loss_res[key]
                        if torch.is_tensor(val):
                            val = val.item()
                        if key not in total_loss_dict:
                            total_loss_dict[key] = 0
                        total_loss_dict[key] += val
                else:
                    val = loss_res
                    if torch.is_tensor(val):
                        val = val.item()
                    if 'l' not in total_loss_dict:
                        total_loss_dict['l'] = val
                    else:
                        total_loss_dict['l'] += val
                
                # Clean up tensors from this batch
                try:
                    del loss
                except:
                    pass
                try:
                    del loss_res
                except:
                    pass
            
        if 'learnable-area' in self.loss:
            if hasattr(self, 'area_parameters_log'):
                self.area_parameters_log.append(self.area_parameters.detach().cpu().numpy())
            else:
                self.area_parameters_log = []
                self.area_parameters_log.append(self.area_parameters.detach().cpu().numpy())

        if len(loader) > 0:
            total_loss /= len(loader)
            for k in total_loss_dict:
                total_loss_dict[k] /= len(loader)

        return total_loss, total_loss_dict
    
    def make_model(self, graph, custom_model_params):
        model, params = make_model(self.model_name, len(self.features_name), len(self.features_name),
                                graph, dropout, activation,
                                self.ks,
                                out_channels=self.out_channels,
                                task_type=self.task_type,
                                device=device, num_lstm_layers=num_lstm_layers,
                                custom_model_params=custom_model_params, horizon=self.horizon)

        if self.model_params is None:
            self.model_params = params

        return model, params

    def func_epoch(self, train_loader, val_loader, optimizer, criterion, do_update):

        train_loss, train_loss_dict = self.launch_train_loader(train_loader, criterion, optimizer, do_update)

        if val_loader is not None:
            val_loss, val_loss_dict = self.launch_val_test_loader(val_loader, criterion)
        else:
            val_loss = train_loss.item()
            val_loss_dict = train_loss_dict

        return val_loss, train_loss, val_loss_dict, train_loss_dict
    
    def get_class_freq(self, df_train):
        uclass = np.sort(df_train[self.target_name].unique())
        res = np.zeros_like(uclass)
        for i, cl in enumerate(uclass):
            res[i] = len(df_train[(df_train[self.target_name] == cl) & (df_train['weight'] > 0)])
        
        return self.compute_global_alpha(res)
    
    def compute_global_alpha(self, global_hist):
        freq = global_hist / global_hist.sum()
        alpha = 1 / np.sqrt(freq)
        alpha = alpha / alpha.sum()
        return alpha

    def get_class_hist_from_train(self, df_train, target_name: str):
        uclass = np.sort(df_train[target_name].unique())
        print(uclass)
        res = np.zeros_like(uclass, dtype=np.int64)
        for i, cl in enumerate(uclass):
            res[i] = len(df_train[(df_train[target_name] == cl) & (df_train["weight"] > 0)])
        return res  # shape (K,)

    def compute_corn_alpha_vector_from_hist(self, class_hist: np.ndarray):
        """
        class_hist: counts par classe (K,)
        Retour: alpha_vec (K-1,), alpha[i] = poids du POSITIF pour la tâche i (y>i | y>i-1)
        """
        class_hist = np.asarray(class_hist, dtype=np.float64)
        K = class_hist.shape[0]
        print(K)
        alpha_vec = np.zeros((K - 1,), dtype=np.float64)

        for i in range(K - 1):
            neg = class_hist[i]                 # y == i (dans le sous-ensemble conditionnel)
            pos = class_hist[i+1:].sum()        # y > i

            # Cas extrêmes
            if (neg + pos) <= 0:
                alpha_vec[i] = 0.5
                continue
            if pos <= 0:
                alpha_vec[i] = 0.0   # aucun positif => alpha_pos=0
                continue
            if neg <= 0:
                alpha_vec[i] = 1.0   # aucun négatif => alpha_pos=1
                continue

            a2 = self.compute_global_alpha(np.array([neg, pos], dtype=np.float64))
            alpha_pos = float(a2[1])  # index 1 = pos
            alpha_vec[i] = alpha_pos

        return alpha_vec

    def get_corn_alpha_from_train_df(self, df_train, target_name: str):
        hist = self.get_class_hist_from_train(df_train, target_name)
        alpha_vec = self.compute_corn_alpha_vector_from_hist(hist)
        return alpha_vec

    # ──────────────────────────────────────────────────────────────────────────
    # Shared scoring helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_raw_scores(self, y_true, y_pred, dates, zones):
        """
        Compute raw evaluation scores using evaluate_metrics.
        Returns the full evaluate_metrics dict, which already includes
        score_k1..k4, recall, iou, f1, etc.
        Missing or NaN score_k{n} values are replaced with 0.0.
        """
        metrics = self.scoring.evaluate_metrics(y_true, y_pred, dates=dates, zones=zones)
        for k in [1, 2, 3, 4]:
            key = f'score_k{k}'
            v = metrics.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                metrics[key] = 0.0
                
        key = 'score_min_class'
        v = metrics.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            metrics[key] = 0.0
        if 'recall' not in metrics or metrics['recall'] is None:
            try:
                from sklearn.metrics import recall_score as _rec
                metrics['recall'] = float(_rec(
                    (np.asarray(y_true) > 0).astype(int),
                    (np.asarray(y_pred) > 0).astype(int),
                    zero_division=0
                ))
            except Exception:
                metrics['recall'] = 0.0
        
        if 'f1' not in metrics or metrics['f1'] is None:
            try:
                from sklearn.metrics import recall_score as _rec
                metrics['f1'] = float(_rec(
                    (np.asarray(y_true) > 0).astype(int),
                    (np.asarray(y_pred) > 0).astype(int),
                    zero_division=0
                ))
            except Exception:
                metrics['f1'] = 0.0
                
        if 'prec' not in metrics or metrics['prec'] is None:
            try:
                from sklearn.metrics import recall_score as _rec
                metrics['prec'] = float(_rec(
                    (np.asarray(y_true) > 0).astype(int),
                    (np.asarray(y_pred) > 0).astype(int),
                    zero_division=0
                ))
            except Exception:
                metrics['prec'] = 0.0
                
        return metrics

    def _compute_geometric_agg(self, raw_dict):
        """
        Computes normalized scores (u_k) using self.reference_scores and returns
        the geometric mean (mapped back to -1, 1).
        Returns tuple: (agg_score, list_of_u_vals)
        """
        EPS = 1e-6
        ref = getattr(self, 'reference_scores', None)
        assert ref is not None
        s_ref_map = (ref.get('best_scores') or ref.get('ref_scores', {})) if ref is not None else {}
        
        def _u(sk, raw_key):
            skr = float(s_ref_map[raw_key])
            denom = max(abs(skr) + EPS, 0.1)
            # map tanh (-1, 1) to (0, 1) directly for the list of u_vals too
            return (np.tanh((sk - skr) / denom) + 1.0) / 2.0

        pairs = [(float(raw_dict[f'score_k{k}']), f'score_k{k}') for k in [1, 2, 3, 4]]
        pairs.append((float(raw_dict['recall']), 'recall'))
        pairs.append((float(raw_dict['score_min_class']), 'score_min_class'))
        
        u_vals = [_u(sk, key) for sk, key in pairs]
        
        # u_vals are already in (0, 1)
        U = np.exp(np.mean(np.log(np.array(u_vals) + EPS)))  # geometric mean in (0,1)
        agg = float(2.0 * U - 1.0)  # map back to (-1,1)
        
        return agg, u_vals

    def _compute_run_scores(self, y_true, y_pred, dates, zones):
        """
        Compute raw scores + normalise by self.reference_scores.

        - task_type == 'binary'  : agg = f1 score (binary detection).
        - other task types       : agg = geometric mean of normalised u_k scores.

        Returns dict: {score_k1..k4, recall, f1, agg, ...}.
        """
        raw = self._compute_raw_scores(y_true, y_pred, dates, zones)

        # ── Binary task: use F1 as the single optimisation target ──────────
        if getattr(self, 'task_type', None) == 'binary':
            result = {f'score_k{k}': raw.get(f'score_k{k}', 0.0) for k in [1, 2, 3, 4]}
            result['recall'] = raw.get('recall', 0.0)
            result['f1']     = raw.get('f1', 0.0)
            result['score_min_class'] = raw.get('score_min_class', 0.0)
            result['agg']    = result['f1']  # optimise on F1
            for k in range(5):
                if f'mu_{k}' in raw:
                    result[f'mu_{k}'] = raw[f'mu_{k}']
            for idx in range(50):
                if f'mu_dense_{idx}' in raw:
                    result[f'mu_dense_{idx}'] = raw[f'mu_dense_{idx}']
            return result

        # ── All other tasks: geometric aggregate ────────────────────────────
        agg, u_vals = self._compute_geometric_agg(raw)

        result = {f'score_k{k}': raw[f'score_k{k}'] for k in [1, 2, 3, 4]}
        result['recall'] = raw['recall']
        result['f1']     = raw.get('f1', 0.0)
        result['score_min_class'] = raw['score_min_class']
        result['agg'] = agg

        # Include mu values from splines
        for k in range(5):
            if f'mu_{k}' in raw:
                result[f'mu_{k}'] = raw[f'mu_{k}']
                
        # Include dense mu values from splines
        for idx in range(50):
            if f'mu_dense_{idx}' in raw:
                result[f'mu_dense_{idx}'] = raw[f'mu_dense_{idx}']

        # Individual normalised scores mapped to (0,1)
        for i, k in enumerate([1, 2, 3, 4]):
            result[f'u_k{k}'] = float(u_vals[i])
        result['u_recall'] = float(u_vals[4])
        result['u_score_min_class'] = float(u_vals[5])
        return result

    def define_reference_model(
        self,
        df_train,
        df_test=None,
        date_col: str  = 'date',
        zone_col: str  = 'graph_id',
        nbsinister_col: str = 'nbsinister',
        fwi_candidates: list = None,
        n_classes: int = 5,
        verbose: bool  = True,
    ):
        """
        Find the best FWI-based ordinal discretization as a reference baseline model.
        Uses self._compute_raw_scores internally for evaluation.
        Stores result in self.reference_scores and returns it.

        Returns dict:
            'best_config'    : {'fwi_col': str, 'quantile_name': str, 'quantiles': list, ...}
            'best_score'     : float  (mean_u of best config)
            'ref_scores'     : dict   {k: s_k_ref}  (uniform bins on first FWI col)
            'baseline_scores': dict   {k: s_k_baseline} (trivial all-0 predictor)
            'all_results'    : list[dict]
        """
        EPS = 1e-6
        df_eval = df_test if df_test is not None else df_train
        
        if 'DualTraining-num' in self.name:
            df_eval = df_eval[df_eval[self.target_name] > 0].copy()
            df_train = df_train[df_train[self.target_name] > 0].copy()
            n_classes = 4
        else:
            df_eval = df_eval.copy()
            n_classes = 5
            
        _DEFAULTS = ['fwi', 'fwi_mean', 'isi', 'bui', 'dc', 'ffmc', 'dmc', 'dailySeverityRating']
        if fwi_candidates is None:
            fwi_candidates = [c for c in _DEFAULTS if c in df_train.columns]
        if not fwi_candidates:
            if getattr(self, 'target_name', '') == 'DFE':
                if verbose:
                    print(f"[{getattr(self, 'target_name', 'DFE')}] No FWI candidate needed. Returning dummy reference.")
                self.reference_scores = {k: 0.001 for k in [1,2,3,4]}
                self.baseline_scores = {k: 0.001 for k in [1,2,3,4]}
                return {
                    'best_config': None,
                    'best_score': 0.0,
                    'ref_scores': self.reference_scores,
                    'baseline_scores': self.baseline_scores,
                    'all_results': []
                }
            raise ValueError("No FWI candidate column found in df_train.")

        for col in [date_col, zone_col]:
            if col not in df_train.columns:
                raise ValueError(f"Column '{col}' not found in evaluation dataframe.")
        if nbsinister_col not in df_train.columns or nbsinister_col not in df_eval.columns:
            raise ValueError(f"'{nbsinister_col}' not found in dataframe(s).")

        y_true = df_eval[nbsinister_col].fillna(0).values

        if verbose:
            import collections
            print(f"[ref_model] y_true stats: min={y_true.min():.2f} max={y_true.max():.2f} mean={y_true.mean():.2f} n={len(y_true)}")

        def _discretize(s_train, s_eval, quantiles):
            qs  = np.quantile(s_train.dropna(), quantiles[1:-1])
            return np.searchsorted(qs, s_eval.fillna(s_eval.median()).values
                                   ).clip(0, n_classes - 1).astype(int)

        def _scores_for(pred, df, reference=False):
            """Raw scores keyed by int k + 'recall' + 'score_min_class'."""
            # y_true: valeurs brutes de nbsinister (pas de quantile)
            y_true = df[nbsinister_col].fillna(0).values
            dates  = df[date_col].values
            zones  = df[zone_col].values
            raw = self.scoring.evaluate_metrics(y_true, pred, dates=dates, zones=zones, reference=reference)
            for k in [1, 2, 3, 4]:
                v = raw.get(f'score_k{k}')
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    raw[f'score_k{k}'] = 0.0
            if 'recall' not in raw or raw['recall'] is None:
                raw['recall'] = 0.0
            if 'score_min_class' not in raw or raw['score_min_class'] is None:
                raw['score_min_class'] = 0.0
            s = {f'score_k{k}': float(raw[f'score_k{k}']) for k in [1, 2, 3, 4]}
            s['recall'] = float(raw['recall'])
            s['score_min_class'] = float(raw['score_min_class'])
            return s

        self.sigma = np.std(y_true)
        self.scoring.set_sigma(self.sigma)

        # Baseline: trivial predictor (all class 0)
        s_baseline = _scores_for(np.zeros(len(y_true), dtype=int), df=df_eval, reference=False)
        if verbose:
            print(f"[ref_model] baseline: {s_baseline}")

        if n_classes == 5:
            _QUANTILE_GRID = [
                ("fwi_quantiles", [0.0, 0.50, 0.75, 0.95, 0.99, 1.0]),
                ("uniform",       np.linspace(0.0, 1.0, 6).tolist()),
                ("heavy_low",     [0.0, 0.50, 0.70, 0.83, 0.92, 1.0]),
                ("heavy_high",    [0.0, 0.08, 0.20, 0.40, 0.65, 1.0]),
                ("low_emphasis",  [0.0, 0.30, 0.55, 0.72, 0.87, 1.0]),
                ("high_emphasis", [0.0, 0.13, 0.28, 0.45, 0.70, 1.0]),
                ("balanced_low",  [0.0, 0.40, 0.60, 0.75, 0.88, 1.0]),
                ("extreme_tail",  [0.0, 0.60, 0.75, 0.85, 0.93, 1.0]),
                ("mild_tail",     [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]),
            ]
        elif n_classes == 4:
            _QUANTILE_GRID = [
                ("fwi_quantiles", [0.0, 0.50, 0.90, 0.98, 1.0]),
                ("uniform",       np.linspace(0.0, 1.0, 5).tolist()),
                ("heavy_low",     [0.0, 0.40, 0.70, 0.90, 1.0]),
                ("heavy_high",    [0.0, 0.15, 0.40, 0.75, 1.0]),
                ("low_emphasis",  [0.0, 0.35, 0.65, 0.85, 1.0]),
                ("high_emphasis", [0.0, 0.25, 0.55, 0.80, 1.0]),
                ("balanced_low",  [0.0, 0.50, 0.75, 0.90, 1.0]),
                ("extreme_tail",  [0.0, 0.75, 0.90, 0.96, 1.0]),
                ("mild_tail",     [0.0, 0.33, 0.66, 0.90, 1.0]),
            ]
        else:
            _QUANTILE_GRID = [("uniform", np.linspace(0.0, 1.0, n_classes + 1).tolist())]

        # ── Référence : colonne précalculée {target}-quantile-5-Class-Dept ──────
        # → y_ref_pred est la discrétisation ordinale officielle de la target (0..4)
        # C'est ce prédicteur qui définit pair_mean_deltas (étalon de normalisation).
        _ref_col = f"{self.target_name}-kmeans-5-Class-Dept"
        if _ref_col in df_train.columns:
            y_ref_pred = df_train[_ref_col].fillna(0).astype(int).values
            if n_classes == 4:
                # If target > 0, the original 5-class labels are [1, 2, 3, 4], map them to [0, 1, 2, 3]
                y_ref_pred = np.clip(y_ref_pred - 1, 0, n_classes - 1)
            else:
                y_ref_pred = np.clip(y_ref_pred, 0, n_classes - 1)
        else:
            # Fallback : quantilisation manuelle sur df_train si la colonne est absente
            if n_classes == 4:
                _qs = np.quantile(df_train[nbsinister_col].fillna(0).values, [0.50, 0.90, 0.98])
            else:
                _qs = np.quantile(df_train[nbsinister_col].fillna(0).values, [0.50, 0.75, 0.95, 0.99])
            y_ref_pred = np.searchsorted(
                _qs, df_train[nbsinister_col].fillna(0).values
            ).clip(0, n_classes - 1).astype(int)

        if verbose:
            import collections
            print(f"[ref_model] y_ref_pred col='{_ref_col}' dist: "
                  f"{dict(sorted(collections.Counter(y_ref_pred).items()))}")

        # Appel reference=True → peuple self.scoring.pair_mean_deltas uniquement
        # Le return value (scores de la TARGET) n'est pas utile → ignoré
        _scores_for(y_ref_pred, df_train, reference=True)
        if verbose:
            print(f"[ref_model] pair_mean_deltas (target): {self.scoring.pair_mean_deltas}")

        # ── Candidat initial : premier FWI × fwi_quantiles ─────────────────────
        # Définit le premier "meilleur candidat" → reference_scores['best_scores']
        # pour que _compute_geometric_agg soit opérationnel dès le début de la grille.
        _init_col    = fwi_candidates[0]
        _, _init_q   = _QUANTILE_GRID[0]   # fwi_quantiles
        if _init_col in df_train.columns:
            _init_pred   = _discretize(df_train[_init_col], df_eval[_init_col], _init_q)
            _init_scores = _scores_for(_init_pred, df_eval, reference=False)
        else:
            _init_scores = {f'score_k{k}': 0.0 for k in [1,2,3,4]}
            _init_scores.update({'recall': 0.0, 'score_min_class': 0.0})

        # Par définition, le candidat initial a agg=0 (centré sur lui-même)
        self.reference_scores = {
            'best_scores': _init_scores,
            'best_score':  0.0,
        }
        if verbose:
            print(f"[ref_model] Candidat initial ({_init_col} / fwi_quantiles): {_init_scores}")

        # ── Grille : chaque config est comparée au candidat initial ─────────────
        all_results = []
        for fwi_col in fwi_candidates:
            if fwi_col not in df_train.columns or fwi_col not in df_eval.columns:
                continue
            for qname, qbounds in _QUANTILE_GRID:
                pred   = _discretize(df_train[fwi_col], df_eval[fwi_col], qbounds)
                scores = _scores_for(pred, df_eval, reference=False)
                agg, _ = self._compute_geometric_agg(scores)
                cfg = {'fwi_col': fwi_col, 'quantile_name': qname,
                       'quantiles': qbounds, 'scores': scores, 'mean_u': float(agg)}
                all_results.append(cfg)
                if verbose:
                    k_nrm = [f"k{k}={scores.get(f'score_k{k}', 0.0):.4f}" for k in [1, 2, 3, 4]]
                    print(f"  {fwi_col:20s} | {qname:14s} | {' '.join(k_nrm)}  | agg={agg:.4f}")

        if not all_results:
            raise RuntimeError("Grid search produced no valid results.")

        # ── Élection du meilleur ─────────────────────────────────────────────────
        best = max(all_results, key=lambda r: r['mean_u'])
        if best['mean_u'] <= 0.0:
            if verbose:
                print(f"[ref_model] Warning: Best agg={best['mean_u']:.4f} <= 0. Falling back to initial candidate.")
            best = all_results[0]

        best_scores_final = best['scores']

        if verbose:
            lines = [
                f"\n{'─'*60}",
                f"[ref_model] BEST FWI CONFIG",
                f"  FWI col   : {best['fwi_col']}",
                f"  Quantiles : {best['quantile_name']}",
                f"  agg       : {best['mean_u']:.4f}",
                f"{'─'*60}",
                f"  {'Metric':<16} {'Best FWI':>10} {'Baseline':>10}",
                f"  {'─'*38}",
            ]
            for k in [1, 2, 3, 4]:
                key = f'score_k{k}'
                lines.append(f"  {key:<16} {best_scores_final.get(key, 0.0):>10.4f} "
                              f"{s_baseline.get(key, 0.0):>10.4f}")
            lines.append(f"  {'recall':<16} {best_scores_final.get('recall', 0.0):>10.4f} "
                          f"{s_baseline.get('recall', 0.0):>10.4f}")
            lines.append(f"  {'score_min_class':<16} {best_scores_final.get('score_min_class', 0.0):>10.4f} "
                          f"{s_baseline.get('score_min_class', 0.0):>10.4f}")
            lines.append(f"{'─'*60}")
            print('\n'.join(lines))

        result = {
            'best_config':     best,
            'best_score':      best['mean_u'],
            'best_scores':     best_scores_final,
            'baseline_scores': s_baseline,
            'all_results':     all_results,
        }
        self.reference_scores = result
        return result

    # ──────────────────────────────────────────────────────────────────────────

    def calculate_val_scores_and_compare(self, best_scores=None):
        """
        Calculate validation scores (evaluate_metrics + recall, normalised by reference)
        and compare two runs.

            u_k = clip((s_k - s_baseline) / (|s_k_ref - s_k_baseline| + ε), 0, 1)
            agg = mean_k(u_k)   k ∈ {1, 2, 3, 4, 'recall'}

        Returns:
            current_scores (dict) : {score_k1..k4, recall, agg}
            is_better      (bool) : True if current agg > best agg
            agg            (float): normalised aggregate score of current run
        """
        # ── 1. Predictions ────────────────────────────────────────────────────
        self.model.eval()
        with torch.no_grad():
            pred_tensor, y_tensor = self._predict_test_loader(self.val_loader)

        test_output = pred_tensor[:, 0]
        y           = y_tensor[:, :, 0]
        prediction  = test_output.detach().cpu().numpy()
        y_np        = y.detach().cpu().numpy()

        # ── Special case: DFE target ─────────────────────────────────────────
        if self.target_name == 'DFE':
            from forecasting_models.sklearn.score import iou_score
            iou = iou_score(y_np[:, -1], prediction)
            current_scores = {'iou_score': iou, 'agg': iou}
            del test_output, y, prediction, y_np, y_tensor, pred_tensor
            if best_scores is None:
                return current_scores, True, iou
            return current_scores, iou > best_scores.get('iou_score', -float('inf')), iou

        # ── 2. Build evaluation dataframe ────────────────────────────────────
        dff = pd.DataFrame(index=np.arange(y_np.shape[0]))
        dff['departement']    = y_np[:, departement_index]
        dff['date']           = y_np[:, date_index]
        dff['graph_id']       = y_np[:, graph_id_index]
        dff['weight']         = y_np[:, weight_index]
        dff[self.target_name] = y_np[:, -1]
        dff['_pred']          = prediction

        # ── Filter weight > 0 (match filter_prediction in test_dl_model) ────
        mask = dff['weight'] > 0
        dff  = dff[mask].reset_index(drop=True)
        prediction_filtered = dff['_pred'].values

        # ── 3. Compute normalised scores via shared helper ────────────────────
        current_scores = self._compute_run_scores(
            dff[self.target_name].values,
            prediction_filtered,
            dff['date'].values,
            dff['graph_id'].values,
        )

        # ── Clean up ─────────────────────────────────────────────────────────
        del dff, prediction, prediction_filtered, y_np, test_output, y, y_tensor, pred_tensor

        agg = current_scores['agg']

        # ── 4. First run ─────────────────────────────────────────────────────
        if best_scores is None:
            return current_scores, True, agg

        # ── 5. Compare ───────────────────────────────────────────────────────
        is_better = agg > float(best_scores.get('agg', -1.0))
        return current_scores, is_better, agg

    def train_run(self, graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose=True, custom_model_params=None, new_model=True, min_epochs=1, run_idx=0):

        """
        Train neural network model
        """
        self.score_per_epochs = {}
        if MLFLOW:
            existing_run = get_existing_run(f'{self.model_name}_')
            if existing_run:
                mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
            else:
                mlflow.start_run(run_name=f'{self.model_name}_', nested=True)

        assert self.train_loader is not None and self.val_loader is not None

        check_and_create_path(self.dir_log)

        loss_params = {}
        if 'cornfl' in self.loss:
            loss_params = {'alpha' : self.get_corn_alpha_from_train_df(self.df_train, self.target_name)}

        elif 'fl' in self.loss: # Use focal loss
            loss_params = {'alpha' : self.get_class_freq(self.df_train)}
            
        sigma = self.df_train[self.target_name].std()
        
        if 'ranknet' in loss_params:
            loss_params.update({
                "sigma" : sigma,
                "wmid": 0.1
            })
            
        if 'nbsinister' in self.target_name and 'ccllt' in self.loss and 'firemen' in self.dir_log.as_posix():
            print('Using optimal parameters for nbsinister-constrained regions')

            loss_params.update({
                  "sigma": sigma,
                "wmu0": 0.29,
                "wmid": 0.11,
                "wtrans": 1.5,
                "wcoverage": 2.76,
                "gainsfloor": 3.03,
                "wkdecay": "None",
                "taugate": 0.35,
                "gatetemp": 1.39,
                "massupdate": 0.12,
                "mumomentum": 0.55,
                "mulambdag": 3.0,
                "mulambdac": 0.49,
                "mulambdad": 0.83,
                "shift": 0.64,
                            })
                        
        elif 'ressource' in self.target_name and 'ccllt' in self.loss and 'firemen' in self.dir_log.as_posix():
            print('Using optimal parameters for ressource-constrained regions')
        
            loss_params.update({
              "wmu0": 0.73,
            "wmid": 0.09,
            "wtrans": 1.81,
            "wcoverage": 4.05,
            "gainsfloor": 3.84,
            "wkdecay": "power",
            "wkpower": 3.45,
            "taugate": 0.11,
            "gatetemp": 1.87,
            "massupdate": 0.35,
            "mumomentum": 0.02,
            "mulambdag": 2.17,
            "mulambdac": 0.33,
            "mulambdad": 1.73,
            "shift": 1.0,
                "sigma": sigma,
            })
            
        elif 'timeintervention' in self.target_name and 'ccllt' in self.loss and 'firemen' in self.dir_log.as_posix():
            print('Using optimal parameters for timeintervention-constrained regions')
            
            loss_params.update({
              "wmu0": 1.17,
                "wmid": 2.05,
                "wtrans": 3.37,
                "wcoverage": 2.85,
                "gainsfloor": 3.72,
                "wkdecay": "power",
                "wkpower": 5.52,
                "taugate": 0.54,
                "gatetemp": 0.37,
                "massupdate": 0.1,
                "mumomentum": 0.47,
                "mulambdag": 0.91,
                "mulambdac": 0.76,
                "mulambdad": 1.43,
                "shift": 0.95,
                "sigma": sigma,
            })
        
        elif 'nbsinister' in self.target_name and 'ccllt' in self.loss and 'bdiff' in self.dir_log.as_posix():
            print('Using optimal parameters for nbsinister-constrained regions')
            
            loss_params.update({
                  "gainsfloor": 3.0,
                "wkdecay": "None",
                "wkpower": 3.86,
                "wklambda": 1.15,
                "taugate": 0.34,
                "gatetemp": 0.88,
                "wmu0": 1.0,
                "massupdate": 0.41,
                "mumomentum": 0.93,
                "mulambdag": 1.0,
                "mulambdac": 0.0,
                "mulambdad": 1.0,
                "wmid": 1.0,
                "wtrans": 2.0,
                "wcoverage": 3.0,
                "sigma": sigma,
                    })
            
        elif 'burnedareaRoot' in self.target_name and 'ccllt' in self.loss and 'bdiff' in self.dir_log.as_posix():
            print('Using optimal parameters for burnedareaRoot-constrained regions')
            loss_params.update({
                    "gainsfloor": 3.0,
                    "wkdecay": "None",
                    "wkpower": 3.86,
                    "wklambda": 1.15,
                    "taugate": 0.34,
                    "gatetemp": 0.88,
                    "wmu0": 1.0,
                    "massupdate": 0.41,
                    "mumomentum": 0.93,
                    "mulambdag": 1.0,
                    "mulambdac": 0.0,
                    "mulambdad": 1.0,
                    "wmid": 1.0,
                    "wtrans": 2.0,
                    "wcoverage": 3.0,
                    "sigma": sigma,
                    })
    
        self.criterion = self.get_loss(self.loss, loss_params)

        if has_method(self.criterion, '_preprocess'):
            if 'id{departement}' in self.loss:
                self.criterion._preprocess(self.df_train[self.target_name].values, self.df_train['departement'].values, self.df_train['cluster-encoder'].values)
            elif 'id{node}' in self.loss:
                self.criterion._preprocess(self.df_train[self.target_name].values, self.df_train['graph_id'].values, self.df_train['cluster-encoder'].values)

        if has_method(self.criterion, 'calculate_class_coverage'):
            try:
                from GNN.config import cluster_encoder_index
            except ImportError:
                cluster_encoder_index = None

            cid = getattr(self.criterion, 'id', None)
            cluster_col = 'departement'
            if cid == departement_index:
                cluster_col = 'departement'
            elif cid == graph_id_index:
                cluster_col = 'graph_id'
            elif cluster_encoder_index is not None and cid == cluster_encoder_index:
                cluster_col = 'cluster-encoder'
            
            if cluster_col in self.df_train.columns:
                if self.target_name in ['nbsinister', 'ressource', 'timeintervention']:
                    self.criterion.calculate_class_coverage(self.df_train, cluster_col=cluster_col, target_col=f'{self.target_name}-kmeans-5-Class-Dept', dir_output=self.dir_log)
                else:    
                    self.criterion.calculate_class_coverage(self.df_train, cluster_col=cluster_col, target_col=self.target_name, dir_output=self.dir_log)
                
        static_idx, temporal_idx = get_static_temporal_idx(self.features_name)
        
        new_params = {'static_idx': static_idx, 'temporal_idx' : temporal_idx}
        
        if self.model_name == 'TFN':
            new_params = {'static_idx': static_idx, 'temporal_idx' : temporal_idx, 'd_static' : len(static_idx)}
                
        if custom_model_params is None:
            custom_model_params = new_params
        else:
            custom_model_params.update(new_params)

        # Fixer la seed pour reproduire (différente pour chaque run)
        trial_seed = 42 + run_idx
        import torch
        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed)
        random.seed(trial_seed)
        
        if new_model or self.model is None:
            self.model, _ = self.make_model(graph, custom_model_params)
        else:
            assert self.model is not None
        
        init_weight_sum = sum(p.sum().item() for p in self.model.parameters())
        print(f"[TRAIN] Initial model weights sum: {init_weight_sum}")
        
        optimizer = self.get_optimizer(self.criterion)

        BEST_VAL_LOSS = math.inf
        BEST_MODEL_PARAMS = None
        BEST_SCORES = None  # For score-based early stopping
        self.best_epoch = 0
        patience_cnt = 0
        current_patience_lr = 0

        val_loss_list = []
        train_loss_list = []
        epochs_list = []
        val_loss_dict_list = []
        train_loss_dict_list = []
        
        if (self.dir_log / 'best.pt').is_file():
            print(f"WARNING: Checkpoint found at {self.dir_log / 'best.pt'} but SKIPPING load due to hardcoded False.")
        
        #if (self.dir_log / 'best.pt').is_file():
        if False:
            self._load_model_from_path(self.dir_log / 'best.pt', self.model)
        else:
            for epoch in tqdm(range(epochs), disable=not verbose):
                # Expose current epoch to subroutines for logging
                self._current_epoch = epoch
                val_loss, train_loss, val_loss_dict, train_loss_dict = self.func_epoch(train_loader=self.train_loader, val_loader=self.val_loader,
                                                    optimizer=optimizer, criterion=self.criterion, do_update=True)

                val_loss_list.append(round(val_loss, 3))
                train_loss_list.append(round(train_loss, 3))
                val_loss_dict_list.append(val_loss_dict)
                train_loss_dict_list.append(train_loss_dict)
                epochs_list.append(epoch)
                
                # Score-based early stopping with Borda Count
                try:
                    current_scores, is_better, rank_sum = self.calculate_val_scores_and_compare(BEST_SCORES)

                    # Store scores for this epoch
                    self.score_per_epochs[epoch] = current_scores

                    if True: # HARDCODED OVERRIDE: early stopping on val_loss
                        is_better_loss = val_loss < BEST_VAL_LOSS
                        if is_better_loss:
                            prev_scores = BEST_SCORES
                            BEST_SCORES = current_scores
                            BEST_VAL_LOSS = val_loss
                            import copy
                            BEST_MODEL_PARAMS = copy.deepcopy(self.model.state_dict())
                            patience_cnt = 0
                            current_patience_lr = 0
                            self.best_epoch = epoch

                            logger.info(f"Epoch {epoch} [✓ NEW BEST LOSS] val_loss={val_loss:.4f} (agg={current_scores.get('agg', float('nan')):.4f})")
                        else:
                            patience_cnt += 1
                            if epoch % CHECKPOINT == 0:
                                logger.info(f"Epoch {epoch}  val_loss={val_loss:.4f} (agg={current_scores.get('agg', float('nan')):.4f})"
                                            f"  patience {patience_cnt}/{PATIENCE_CNT}")
                    elif is_better:
                        prev_scores = BEST_SCORES
                        BEST_SCORES = current_scores
                        BEST_VAL_LOSS = val_loss
                        import copy
                        BEST_MODEL_PARAMS = copy.deepcopy(self.model.state_dict())
                        patience_cnt = 0
                        current_patience_lr = 0
                        self.best_epoch = epoch

                        ref = getattr(self, 'reference_scores', None)
                        s_ref_map = (ref.get('best_scores') or ref.get('ref_scores', {})) if ref is not None else {}

                        # ── Log score table (new best only) ──────────────────────
                        _lines = [
                            f"Epoch {epoch} [✓ NEW BEST]  agg={current_scores.get('agg', float('nan')):.4f}",
                            f"  {'metric':<10} {'score':>8} {'u (0-1)':>9} {'ref score':>10} {'prev score':>11} {'prev u':>8}",
                            f"  {'─'*63}",
                        ]
                        if 'iou_score' in current_scores:
                            _lines.append(f"  {'iou':<10} {current_scores['iou_score']:>8.4f}")
                        else:
                            for _k in [1, 2, 3, 4]:
                                _s_cur  = current_scores.get(f'score_k{_k}', float('nan'))
                                _u_cur  = current_scores.get(f'u_k{_k}',    float('nan'))
                                _s_ref  = s_ref_map.get(f'score_k{_k}', float('nan'))
                                _s_prev = (prev_scores or {}).get(f'score_k{_k}', float('nan'))
                                _u_prev = (prev_scores or {}).get(f'u_k{_k}',    float('nan'))
                                _lines.append(f"  {'score_k'+str(_k):<10} {_s_cur:>8.4f} {_u_cur:>9.4f} {_s_ref:>10.4f} {_s_prev:>11.4f} {_u_prev:>8.4f}")
                            _rc   = current_scores.get('recall',   float('nan'))
                            _urc  = current_scores.get('u_recall', float('nan'))
                            _rc_ref = s_ref_map.get('recall', float('nan'))
                            _rcp  = (prev_scores or {}).get('recall',   float('nan'))
                            _urcp = (prev_scores or {}).get('u_recall', float('nan'))
                            _lines.append(f"  {'recall':<10} {_rc:>8.4f} {_urc:>9.4f} {_rc_ref:>10.4f} {_rcp:>11.4f} {_urcp:>8.4f}")
                            _smc  = current_scores.get('score_min_class',   float('nan'))
                            _usmc = current_scores.get('u_score_min_class', float('nan'))
                            _smc_ref = s_ref_map.get('score_min_class', float('nan'))
                            _smcp = (prev_scores or {}).get('score_min_class',   float('nan'))
                            _usmcp= (prev_scores or {}).get('u_score_min_class', float('nan'))
                            _lines.append(f"  {'score_min':<10} {_smc:>8.4f} {_usmc:>9.4f} {_smc_ref:>10.4f} {_smcp:>11.4f} {_usmcp:>8.4f}")
                            _lines.append(f"  {'─'*63}")
                            _lines.append(
                                f"  {'agg (min)':<10} {'':>8} {current_scores.get('agg', float('nan')):>9.4f}"
                                f" {'':>10} {(prev_scores or {}).get('agg', float('nan')):>11.4f}"
                            )
                        logger.info('\n'.join(_lines))
                        # ─────────────────────────────────────────────────────────
                    else:
                        patience_cnt += 1
                        if epoch % CHECKPOINT == 0:
                            logger.info(f"Epoch {epoch}  agg={current_scores.get('agg', float('nan')):.4f}"
                                        f"  patience {patience_cnt}/{PATIENCE_CNT}")
                except Exception as e:
                    import traceback
                    logger.warning(f'Score calculation failed ({e}):\n{traceback.format_exc()}\nFalling back to loss-based early stopping')
                    if val_loss < BEST_VAL_LOSS:
                        BEST_VAL_LOSS = val_loss
                        import copy
                        BEST_MODEL_PARAMS = copy.deepcopy(self.model.state_dict())
                        patience_cnt = 0
                        current_patience_lr = 0
                        self.best_epoch = epoch
                    else:
                        patience_cnt += 1
                    
                # Early stopping and LR decay logic - must run on every epoch
                if patience_cnt >= PATIENCE_CNT:
                    # Check if we can reduce LR (have we used all retries?)
                    # PATIENCE_CNT_LR is the number of allowed reductions/retries
                    if current_patience_lr >= self.patience_cnt_lr or self.patience_cnt_lr == 0:
                        logger.info(f'Loss has not increased for {patience_cnt} epochs AND max LR reductions ({self.patience_cnt_lr}) reached.')
                        logger.info(f'Last best val loss {BEST_VAL_LOSS}, current val loss {val_loss}')
                        save_object_torch(self.model.state_dict(), 'last.pt', self.dir_log)
                        save_object_torch(BEST_MODEL_PARAMS, 'best.pt', self.dir_log)
                        self.plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, self.dir_log)
                        if MLFLOW:
                            mlflow.end_run()
                        break
                    else:
                        # Reduce LR and reset patience_cnt
                        if self.delta_lr > 0:
                            current_patience_lr += 1
                            logger.info(f"Patience {PATIENCE_CNT} reached (Retry {current_patience_lr}/{self.patience_cnt_lr}). Decay LR by factor {self.delta_lr}.")
                            
                            current_lr = optimizer.param_groups[0]['lr']
                            new_lr = current_lr * (1 - self.delta_lr)
                            if new_lr <= 1e-9:
                                new_lr = 1e-9
                                logger.warn("Learning rate reached floor (1e-9).")
                            
                            logger.info(f"Reducing LR from {current_lr:.6f} to {new_lr:.6f}")
                            
                            # Define new optimizer with new LR (resets state/momentum as requested)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = new_lr
                            
                            # Reset patience_cnt to give model time to improve with new LR
                            patience_cnt = 0
                        else:
                            # No delta_lr defined, stop normal
                            logger.info(f'Loss has not increased for {patience_cnt} epochs. No delta_lr defined.')
                            save_object_torch(self.model.state_dict(), 'last.pt', self.dir_log)
                            save_object_torch(BEST_MODEL_PARAMS, 'best.pt', self.dir_log)
                            self.plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, self.dir_log)
                            if MLFLOW:
                                mlflow.end_run()
                            break
                if MLFLOW:
                    mlflow.log_metric('loss', val_loss, step=epoch)
                if epoch % CHECKPOINT == 0 and verbose:
                    curr_lr = optimizer.param_groups[0]['lr']
                    logger.info(f'Epoch {epoch}: Val loss {val_loss:.4f}, Train loss {train_loss:.4f}, Best val loss {BEST_VAL_LOSS:.4f}')
                    logger.info(f'    LR: {curr_lr:.6f} | Patience: {patience_cnt}/{PATIENCE_CNT} | Retry: {current_patience_lr}/{self.patience_cnt_lr}')
                    self.log_memory(f"Epoch {epoch} Checkpoint")
                    save_object_torch(self.model.state_dict(), str(epoch)+'.pt', self.dir_log)

            logger.info(f'Last val loss {val_loss}')
            save_object_torch(self.model.state_dict(), 'last.pt', self.dir_log)
            save_object_torch(BEST_MODEL_PARAMS, 'best.pt', self.dir_log)
            self.plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, self.dir_log)
            # Second call for the components dictionary
            self.plot_train_val_loss(epochs_list, train_loss_dict_list, val_loss_dict_list, self.dir_log)

        self.train_loss_dict_list = train_loss_dict_list
        self.val_loss_dict_list = val_loss_dict_list
        self.train_loss_list = train_loss_list
        self.val_loss_list = val_loss_list
        self.epochs_list = epochs_list

        if self.best_epoch == 0:
            print('WARNING: Best epoch is 0')
            print('Val loss', val_loss)
            print('Train loss', train_loss)
        logger.info(f'Best epoch {self.best_epoch}, Best val loss {BEST_VAL_LOSS}')

        if BEST_MODEL_PARAMS is not None:
            self.update_weight(BEST_MODEL_PARAMS)
            logger.info(f'Loaded best model (epoch {self.best_epoch}) for val/test plots.')

        ##################################### VAL #################################################
        test_output_, y_ = self._predict_test_loader(self.val_loader, output_pdf='test', calibrate=True)
        test_output_ = test_output_.detach().cpu().numpy()
        y_ = y_.detach().cpu().numpy()

        for H in range(self.horizon + 1):
            check_and_create_path(self.dir_log / f"H{H}")
            y = y_[:, :, -1 - (self.horizon - H)]
            test_output = test_output_[:, -1 - (self.horizon - H)]

            if np.any(y[:, -1] > 0) or np.any(test_output > 0):

                under_prediction_score_value = under_prediction_score(y[:, -1], test_output)
                over_prediction_score_value = over_prediction_score(y[:, -1], test_output)
                
                iou = iou_score(y[:, -1], test_output)
                f1 = f1_score((test_output > 0).astype(int), (y[:, -1] > 0).astype(int), zero_division=0)
                iou_area, f1_area = self.compute_area_score(test_output, y[:, -1], y[:, graph_id_index])

                _agg_val = BEST_SCORES.get('agg', float('nan')) if BEST_SCORES else float('nan')
                print(f'Horizon {H} -> Val -> Under achieved : {under_prediction_score_value}, Over achived {over_prediction_score_value}, IoU {iou}, f1 {f1}, IoU_area {iou_area}, f1_area {f1_area}, agg {_agg_val}')

                med_deps = [4, 5, 6, 7, 11, 13, 26, 30, 34, 48, 66, 83, 84]
                for gid in np.unique(y[:, graph_id_index]):
                    dep_for_gid = y[y[:, graph_id_index] == gid, departement_index]
                    if len(dep_for_gid) == 0:
                        continue
                    dep = int(dep_for_gid[0])
                    if dep not in med_deps:
                        continue
                    plt.figure(figsize=(15,5))
                    plt.plot(y[y[:, graph_id_index] == gid, -1], label="True")
                    plt.plot(test_output[y[:, graph_id_index] == gid], label="Pred")
                    plt.legend()
                    plt.title(f"Val Horizon {H} - Dep {dep} - Graph {int(gid)}")
                    plt.savefig(self.dir_log / f"H{H}" / f'val_dep{dep}_graph{int(gid)}.png')
                    plt.close('all')

        ##################################### Test #################################################
        test_output_, y_ = self._predict_test_loader(self.test_loader, output_pdf='test')
        test_output_ = test_output_.detach().cpu().numpy()
        y_ = y_.detach().cpu().numpy()
        
        for H in range(self.horizon + 1):

            y = y_[:, :, -1 - (self.horizon - H)]
            test_output = test_output_[:, -1 - (self.horizon - H)]
            
            under_prediction_score_value = under_prediction_score(y[:, -1], test_output)
            over_prediction_score_value = over_prediction_score(y[:, -1], test_output)
            
            iou = iou_score(y[:, -1], test_output)
            f1 = f1_score((test_output > 0).astype(int), (y[:, -1] > 0).astype(int), zero_division=0)
            iou_area, f1_area = self.compute_area_score(test_output, y[:, -1], y[:, graph_id_index])

            _agg_test = BEST_SCORES.get('agg', float('nan')) if BEST_SCORES else float('nan')
            print(f'Horizon {H} -> Test {y.shape} -> Under achieved : {under_prediction_score_value}, Over achived {over_prediction_score_value}, IoU {iou} f1 {f1}, IoU_area {iou_area}, f1_area {f1_area}, agg {_agg_test}')

            # Test plots per graph_id (Mediterranean only)
            med_deps = [4, 5, 6, 7, 11, 13, 26, 30, 34, 48, 66, 83, 84]
            for gid in np.unique(y[:, graph_id_index]):
                dep_for_gid = y[y[:, graph_id_index] == gid, departement_index]
                if len(dep_for_gid) == 0:
                    continue
                dep = int(dep_for_gid[0])
                if dep not in med_deps:
                    continue
                plt.figure(figsize=(15,5))
                plt.plot(y[y[:, graph_id_index] == gid, -1], label="True")
                plt.plot(test_output[y[:, graph_id_index] == gid], label="Pred")
                plt.legend()
                plt.title(f"Test Horizon {H} - Dep {dep} - Graph {int(gid)}")
                plt.savefig(self.dir_log / f"H{H}" / f'test_dep{dep}_graph{int(gid)}.png')
                plt.close('all')

        if 'learnable-area' in self.loss:
            ids = y[:, 0]                       # première colonne
            values = y[:, 1:]

            # Somme groupée par id
            unique_ids, inverse = np.unique(ids, return_inverse=True)
            sums = np.zeros((len(unique_ids), values.shape[1]), dtype=values.dtype)
            np.add.at(sums, inverse, values)
            
            self.plot_area_parameter(epochs_list, y[:, 0], sums[:, -1])
            save_object(self.area_parameters_log, 'area_parameters_log.pkl' ,self.dir_log)

        if has_method(self.criterion, 'plot_params'):
            print(f'Launch criterion params plot')
            self.criterion.plot_params(self.criterion_params, self.dir_log, best_epoch=self.best_epoch)
            
        if has_method(self.criterion, 'update_params'):
            print(f'Update criterion params with {self.criterion_params[self.best_epoch]}')
            self.criterion.update_params(self.criterion_params[self.best_epoch])

        """# --- LOG LOSS COMPONENTS ---
        # "Je veux les valeurs brutes, sans les multiplications par les lambda"
        if hasattr(self.criterion, 'epoch_stats'):
            est_g = self.criterion.epoch_stats.get('global', {})
            if self.criterion.epoch_stats:
                # We take the mean of the values collected during the epoch for the global component
                # Note: epoch_stats accumulates values at each batch.
                # Ideally we want the average over the epoch.
                
                # Helper to safely get mean
                def safe_mean(key):
                    vals = est_g.get(key, [])
                    if vals: 
                        return np.mean(vals)
                    
                    
                    # Fallback: aggregate from cluster stats if global is missing the key
                    # OR specific for 'loss_trans' if we want cluster average separate from global
                    all_vals = []
                    for k, v in self.criterion.epoch_stats.items():
                        if k == 'global': continue
                        if isinstance(v, dict) and key in v and v[key]:
                            all_vals.extend(v[key])
                    
                    if all_vals:
                        return np.mean(all_vals)
                        
                    return 0.0
                
                # Specific extraction for cluster average vs global
                # loss_trans (cluster avg)
                loss_trans_cluster = []
                for k, v in self.criterion.epoch_stats.items():
                     if k == 'global': continue
                     if isinstance(v, dict) and 'loss_trans' in v and v['loss_trans']:
                          loss_trans_cluster.extend(v['loss_trans'])
                l_trans = np.mean(loss_trans_cluster) if loss_trans_cluster else 0.0

                # global_loss_trans (from 'global' key)
                l_trans_glob = 0.0
                if 'global' in self.criterion.epoch_stats:
                     g_stats = self.criterion.epoch_stats['global']
                     if 'loss_trans' in g_stats and g_stats['loss_trans']:
                          l_trans_glob = np.mean(g_stats['loss_trans'])

                # Other components (averaged over clusters usually, or global if addglobal logic applies)
                # For entropy, dirichlet, ce, mu0 -> these are now computed per cluster (or global cluster).
                # So safe_mean (aggregating all) gives the average contribution per sample/cluster.
                l_ent = safe_mean('entropy_pi')
                l_ent_w = safe_mean('entropy_weighted')
                l_mu0 = safe_mean('mu0_term')
                l_dir = safe_mean('dirichlet_reg') # Now per cluster
                l_dir_w = safe_mean('dirichlet_weighted')
                l_ce  = safe_mean('ce_loss')
                l_ce_w = safe_mean('ce_weighted')
                l_total = safe_mean('loss_total')

                
                # Scaling stats
                s_min = safe_mean('scale_min')
                s_mean = safe_mean('scale_mean')
                s_max = safe_mean('scale_max')
                d_raw = safe_mean('diff_raw_mean')
                d_scaled = safe_mean('diff_scaled_mean')
                m_mean = safe_mean('margin_mean')

                self.loss_components_history['loss_total'].append(l_total)
                self.loss_components_history['loss_trans'].append(l_trans)
                self.loss_components_history['global_loss_trans'].append(l_trans_glob)
                self.loss_components_history['entropy_pi'].append(l_ent)
                self.loss_components_history['entropy_weighted'].append(l_ent_w)
                self.loss_components_history['mu0_term'].append(l_mu0)
                self.loss_components_history['dirichlet_reg'].append(l_dir)
                self.loss_components_history['dirichlet_weighted'].append(l_dir_w)
                self.loss_components_history['ce_loss'].append(l_ce)
                self.loss_components_history['ce_weighted'].append(l_ce_w)
                self.loss_components_history['epoch'].append(epochs_list[-1]) # Current epoch
                
                self.loss_components_history['scale_min'].append(s_min)
                self.loss_components_history['scale_mean'].append(s_mean)
                self.loss_components_history['scale_max'].append(s_max)
                self.loss_components_history['diff_raw_mean'].append(d_raw)
                self.loss_components_history['diff_scaled_mean'].append(d_scaled)
                self.loss_components_history['margin_mean'].append(m_mean)
                
                # Plot
                self.plot_loss_decomposition()
                self.plot_scaling_decomposition()"""
        
        # Plot score evolution
        try:
            self.plot_score_evolution()
        except Exception as e:
            logger.warning(f"Score evolution plot failed: {e}")
                
        # Save distillation best/worst logs and 3D plot at the end of training
        
        if 'distillation' in self.loss:
            try:
                self._save_distill_logs_and_plot()
            except Exception as _e:
                # Keep training flow robust even if plotting fails
                logger.info(f"Distillation log/plot skipped: {_e}")

        self.params = BEST_MODEL_PARAMS
        return self.score_per_epochs, self.criterion_params

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
        plt.close('all')

    def plot_loss_decomposition(self):
        """
        Plots the raw values of different loss components over epochs.
        Saves to 'loss_decomposition.png'.
        """
        if not self.loss_components_history['epoch']:
            return

        epochs = self.loss_components_history['epoch']
        
        # Prepare figure
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # We can plot everything on same axis or use twin axis if scales are very different.
        # Given "ordre de grandeur", maybe log scale or twin axis is better.
        # Let's try plotting raw values on linear scale first, but with different colors.
        
        # Plot weighted contributions (actual impact on total loss)
        ax1.plot(epochs, self.loss_components_history['loss_trans'], label='Transitional Loss (Cluster Avg)', color='blue')
        if any(v != 0 for v in self.loss_components_history.get('global_loss_trans', [])):
            ax1.plot(epochs, self.loss_components_history['global_loss_trans'], label='Global Transitional Loss', color='brown', linestyle=':')
        
        # Weighted components (contribution to total loss)
        if any(v != 0 for v in self.loss_components_history.get('entropy_weighted', [])):
            ax1.plot(epochs, self.loss_components_history['entropy_weighted'], label='Entropy (λ × H)', color='green', linestyle='--')
        
        ax1.plot(epochs, self.loss_components_history['mu0_term'], label='Mu0 (λ × μ₀)', color='orange', linestyle=':')
        
        if any(v != 0 for v in self.loss_components_history.get('dirichlet_weighted', [])):
            ax1.plot(epochs, self.loss_components_history['dirichlet_weighted'], label='Dirichlet (λ × R)', color='red', linestyle='-.')
        
        if any(v != 0 for v in self.loss_components_history.get('ce_weighted', [])):
            ax1.plot(epochs, self.loss_components_history['ce_weighted'], label='CE Loss (λ × CE)', color='purple', linestyle='-')
        
        ax1.plot(epochs, self.loss_components_history['loss_total'], label='Total Loss', color='black', linewidth=2, alpha=0.7)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Contribution to Loss')
        ax1.set_title('Loss Components (Weighted Contributions)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.dir_log / 'loss_decomposition.png')
        plt.close(fig)

    def plot_scaling_decomposition(self):
        """
        Plots the scaling and margin metrics over epochs.
        Saves to 'loss_scaling_decomposition.png'.
        """
        if not self.loss_components_history['epoch']:
            return

        epochs = self.loss_components_history['epoch']
        
        # Prepare figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Subplot 1: Scale stats
        ax1 = axes[0]
        ax1.plot(epochs, self.loss_components_history['scale_mean'], label='Scale Mean', color='blue')
        ax1.fill_between(epochs, self.loss_components_history['scale_min'], self.loss_components_history['scale_max'], color='blue', alpha=0.2, label='Min-Max Range')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Scale Value')
        ax1.set_title('Scale Statistics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Margins and Diffs
        ax2 = axes[1]
        ax2.plot(epochs, self.loss_components_history['diff_raw_mean'], label='Raw Diff Mean', color='orange')
        ax2.plot(epochs, self.loss_components_history['diff_scaled_mean'], label='Scaled Diff Mean (raw/scale)', color='purple', linestyle='--')
        ax2.plot(epochs, self.loss_components_history['margin_mean'], label='Margin Mean (gains)', color='green', linestyle='-.')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Value')
        ax2.set_title('Margins & Diffs')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.dir_log / 'loss_scaling_decomposition.png')
        plt.close(fig)
    
    def plot_score_evolution(self):
        """
        Plot the evolution of score_k1..k4, recall, agg over epochs.
        Creates a 3x2 subplot layout. Saves to 'score_evolution.png'.
        """
        if not self.score_per_epochs or len(self.score_per_epochs) == 0:
            logger.info("No score data to plot (score_per_epochs is empty)")
            return
        
        epochs = sorted(self.score_per_epochs.keys())
        def _get(key): return [self.score_per_epochs[e].get(key, float('nan')) for e in epochs]

        best_ep = getattr(self, 'best_epoch', None)

        if getattr(self, 'target_name', '') == 'DFE':
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            fig.suptitle('Score Evolution Over Epochs', fontsize=15)
            vals = _get('iou_score')
            if all(np.isnan(v) if isinstance(v, float) else False for v in vals):
                vals = _get('agg')
                
            ax.plot(epochs, vals, marker='P', color='black', linewidth=1.5, markersize=4)
            if best_ep is not None:
                ax.axvline(best_ep, color='r', linestyle='--', alpha=0.6, label=f'best ep={best_ep}')
                ax.legend(fontsize=7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('IoU / Agg')
            ax.set_title('IoU Score')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.dir_log / 'score_evolution.png')
            plt.close(fig)
            logger.info(f"Score evolution plot saved to {self.dir_log / 'score_evolution.png'}")
            return

        fig, axes = plt.subplots(3, 2, figsize=(13, 12))
        fig.suptitle('Score Evolution Over Epochs', fontsize=15)

        _panels = [
            (axes[0, 0], _get('score_k1'), 'Score K=1',  'blue',   'o'),
            (axes[0, 1], _get('score_k2'), 'Score K=2',  'green',  's'),
            (axes[1, 0], _get('score_k3'), 'Score K=3',  'orange', '^'),
            (axes[1, 1], _get('score_k4'), 'Score K=4',  'red',    'D'),
            (axes[2, 0], _get('recall'),   'Recall',     'purple', 'v'),
            (axes[2, 1], _get('agg'),      'Agg',        'black',  'P'),
        ]
        for ax, vals, title, col, mk in _panels:
            ax.plot(epochs, vals, marker=mk, color=col, linewidth=1.5, markersize=4)
            if best_ep is not None:
                ax.axvline(best_ep, color='r', linestyle='--', alpha=0.6, label=f'best ep={best_ep}')
                ax.legend(fontsize=7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.dir_log / 'score_evolution.png')
        plt.close(fig)
        logger.info(f"Score evolution plot saved to {self.dir_log / 'score_evolution.png'}")


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
            test_percentage = np.round(np.arange(0.1, 1.05, 0.1), 2)
        else:
            test_percentage = np.arange(0.0, 1.05, 0.05)

        if 'MultiScale' in self.model_name:
            test_percentage = np.arange(0.5, 1.05, 0.05)
            
        under_prediction_score_scores = []
        over_prediction_score_scores = []
        iou_scores = []
        data_log = None
        find_log = False

        self.metrics['iou_scores'] = []
 
        def _mean_u_agg(m_dict, suffix='_val'):
            # Build a dictionary looking like raw metric output
            mapped_dict = {}
            for k in [1, 2, 3, 4]:
                vals = np.atleast_1d(m_dict.get(f'score_k{k}{suffix}', [0.0]))
                sk = float(np.nanmean(vals)) if len(vals) else 0.0
                mapped_dict[f'score_k{k}'] = sk if not np.isnan(sk) else 0.0
            
            rv = np.atleast_1d(m_dict.get(f'recall{suffix}', [0.0]))
            sk_r = float(np.nanmean(rv)) if len(rv) else 0.0
            mapped_dict['recall'] = sk_r if not np.isnan(sk_r) else 0.0
            
            smcv = np.atleast_1d(m_dict.get(f'score_min_class{suffix}', [0.0]))
            sk_smc = float(np.nanmean(smcv)) if len(smcv) else 0.0
            mapped_dict['score_min_class'] = sk_smc if not np.isnan(sk_smc) else 0.0
            
            # We output `agg` (the true geometric mean of U converted back dynamically, or the direct index, depending on how agg works).
            agg, _ = self._compute_geometric_agg(mapped_dict)
            return float(agg)
        
        try:
            tracemalloc.start()
            self.log_memory("Start search_samples_proportion")
            
            if use_log:
                if False:
                    if (self.dir_log / 'unknowned_scores_per_percentage.pkl').is_file():
                        data_log = read_object('unknowned_scores_per_percentage.pkl', self.dir_log)
                else:
                    print(self.dir_log / 'metrics.pkl')
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
                                            
            tolerance = 0.1
            doSearch = False
            last_score = -math.inf
            start_test = 0
            
            if find_log and use_log:
                if data_log is not None and 'test_percentage' in data_log:
                    self.metrics = data_log
                    test_percentage = np.asarray(self.metrics['test_percentage'])
                    
                    doSearch = True
                    for i, tp_val in enumerate(test_percentage):
                        tp_val = round(tp_val, 2)
                        if tp_val in self.metrics:
                            current_agg = _mean_u_agg(self.metrics[tp_val], '_val')
                            if current_agg >= last_score - tolerance:
                                if current_agg > last_score:
                                    last_score = current_agg
                            else:
                                print(f'Stopping search: scores declining in logs (last_score={last_score:.4f}, current_agg={current_agg:.4f})')
                                doSearch = False
                                break
                        else:
                            start_test = i
                            doSearch = True
                            print(f'Resuming search from tp={tp_val} (first missing in data_log, index {i})')
                            break
                    else:
                        # All percentages found in data_log and they were "good enough"
                        # but we still stop because there is nothing left to search.
                        doSearch = False
                        print(f'All test_percentage values found in data_log → doSearch=False')
            else:
                doSearch = True
                if not use_log:
                    logger.info("Search disabled (use_log=False)")
                elif not find_log:
                    logger.info("Search disabled (No logs found and use_log=True)")
                    
            if doSearch:
                if start_test != 0:
                    last_keys = test_percentage[start_test - 1]

                # last_score initialized above during log check
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
                    self.log_memory(f"Before Run Loop (tp={tp})")
    
                    for run in range(self.n_run):
                        
                        df_combined = self.split_dataset(df_train_copy, nb, reset=False)
    
                        # Mettre à jour df_train pour l'entraînement
                        #df_train_copy["weight"] = df_combined["weight"].reindex(df_train_copy.index, fill_value=0)
                                                    
                        df_train_copy['weight'] = 0
                        #weight = egpd_trunc_discrete_weights(df_combined[self.target_name].values, df_combined['graph_id'].values)
                        df_train_copy.loc[df_combined.index, 'weight'] = 1
                        #df_train_copy.loc[df_combined.index, 'weight'] = weight
                        
                        # PREVENT DEEP COPY OF HEAVY OBJECTS
                        # We strip all data-related attributes from 'self' before deepcopy
                        # and restore them immediately after.
                        
                        # 1. Save references
                        ref_graph = getattr(self, 'graph', None)
                        ref_df_train = getattr(self, 'df_train', None)
                        ref_df_val = getattr(self, 'df_val', None)
                        ref_df_test = getattr(self, 'df_test', None)
                        ref_val_loader = getattr(self, 'val_loader', None)
                        ref_test_loader = getattr(self, 'test_loader', None)
                        ref_train_loader = getattr(self, 'train_loader', None)
                        ref_optimizer = getattr(self, 'optimizer', None)
                        ref_metrics = getattr(self, 'metrics', {})
                        
                        # 2. Unset attributes
                        self.graph = None
                        self.df_train = None
                        self.df_val = None
                        self.df_test = None
                        self.val_loader = None
                        self.test_loader = None
                        self.train_loader = None
                        self.optimizer = None
                        self.metrics = {}  # Empty dict to avoid copying full history
    
                        # 3. Deepcopy
                        copy_model = deepcopy(self)
                        
                        # 4. Restore attributes
                        self.graph = ref_graph
                        self.df_train = ref_df_train
                        self.df_val = ref_df_val
                        self.df_test = ref_df_test
                        self.val_loader = ref_val_loader
                        self.test_loader = ref_test_loader
                        self.train_loader = ref_train_loader
                        self.optimizer = ref_optimizer
                        self.metrics = ref_metrics
    
                        copy_model.under_sampling = 'full'
                        copy_model.horizon = 0
                        copy_model.create_train_val_test_loader(graph, df_train_copy, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT, features_importance=False, custom_model_params=custom_model_params)
                        # Restore parent's reference_scores: the FWI reference model must not be
                        # recomputed on subsampled data — it is always the one fitted on the full dataset.
                        copy_model.reference_scores = self.reference_scores
                        assert self.reference_scores is not None
                        copy_model.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose=False, custom_model_params=custom_model_params, n_runs=1)
                        
                        self.log_memory(f"After Train (tp={tp}, run={run})")
    
                        ############################# On set val ##############################
                        test_output, y = copy_model._predict_test_loader(copy_model.val_loader, output_pdf='Val', prediction_type='Class')
                        
                        test_output = test_output[:, 0]
                        y = y[:, :, 0]
                        
                        prediction = test_output.detach().cpu().numpy()
                        y = y.detach().cpu().numpy()
                        
                        if 'MultiScale' in self.model_name:
                            id_mask = y[:, scale_index]
                        else:
                            id_mask = y[:, departement_index]
                            id_mask = None
                            
                        dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
                        dff['departement'] = y[:, departement_index]
                        dff['date'] = y[:, date_index]
                        dff['graph_id'] = y[:, graph_id_index]
                        dff[self.target_name] = y[:, -1]
                        y = y[:, -1] > 0 if self.task_type == 'binary' else y[:, -1]
    
                        metrics_run = self._compute_raw_scores(
                            dff[self.target_name], prediction,
                            zones=dff['graph_id'].values, dates=dff['date'].values
                        )
                        metrics_run = round_floats(metrics_run)
                        under_prediction_score_value = under_prediction_score(y, prediction)
                        over_prediction_score_value = over_prediction_score(y, prediction)
                        update_metrics_as_arrays(self, tp, metrics_run, 'val')
                        
                        # Clean up validation dataframe and arrays
                        del dff, test_output, prediction, y, metrics_run
    
                        ############################# On set test ##############################
                        test_output, y = copy_model._predict_test_loader(copy_model.test_loader, output_pdf='Test', prediction_type='Class')
                        
                        test_output = test_output[:, 0]
                        y = y[:, :, 0]
                        
                        prediction = test_output.detach().cpu().numpy()
                        y = y.detach().cpu().numpy()
                        
                        if 'MultiScale' in self.model_name:
                            id_mask = y[:, scale_index]
                        else:
                            id_mask = y[:, departement_index]
                            id_mask = None
                    
                        dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
                        dff['departement'] = y[:, departement_index]
                        dff['date'] = y[:, date_index]
                        dff['graph_id'] = y[:, graph_id_index]
                        dff[self.target_name] = y[:, -1]
                        y = y[:, -1] > 0 if self.task_type == 'binary' else y[:, -1]
    
                        metrics_run = self._compute_raw_scores(
                            dff[self.target_name], prediction,
                            zones=dff['graph_id'].values, dates=dff['date'].values
                        )
                        metrics_run = round_floats(metrics_run)
                        under_prediction_score_value = under_prediction_score(y, prediction)
                        over_prediction_score_value = over_prediction_score(y, prediction)
                        update_metrics_as_arrays(self, tp, metrics_run, 'test')
                        
                        # Clean up test dataframe and arrays
                        del dff, test_output, prediction, y, metrics_run
                        
                        # Manually cleanup deepcopied model to free memory
                        try:
                            copy_model.free_memory()
                        except:
                            pass
                        del copy_model
                        
                        # Force garbage collection and clear CUDA cache
                        gc.collect()
                        gc.collect()  # Call twice for cyclic references
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()  # Wait for all ops to finish
                        
                        # Force Python to return memory to OS (Linux only)
                        try:
                            import ctypes
                            ctypes.CDLL('libc.so.6').malloc_trim(0)
                        except:
                            pass
                        
                        self.log_memory(f"After Cleanup (tp={tp}, run={run})")
                    
                    self.metrics[tp] = add_ic95_to_dict(self.metrics[tp], None, "_ic95")

                    # ── Per-tp score summary ─────────────────────────────────────────────
                    # Compute mean_u for this tp using the same shared helper logic
                    # mu_val and mu_test computed below using global _mean_u_agg

                    mu_val  = _mean_u_agg(self.metrics[tp], '_val')
                    mu_test = _mean_u_agg(self.metrics[tp], '_test')

                    if self.task_type != 'binary':
                        self.metrics[tp]['mean_u_val'] = mu_val
                        self.metrics[tp]['mean_u_test'] = mu_test
                    else:
                        self.metrics[tp]['mean_f1_val'] = self.metrics[tp]['f1_val']
                        self.metrics[tp]['mean_f1_test'] = self.metrics[tp]['f1_test']

                    def _fmt_score(m_dict, key):
                        vals = np.atleast_1d(m_dict.get(key, [float('nan')]))
                        return f"{np.nanmean(vals):7.4f}"
                        
                    header = f"  {'metric':<12} {'val':>8} {'test':>8}"
                    rows   = [header, "  " + "─" * 30]
                    for k in [1, 2, 3, 4]:
                        rows.append(f"  {'score_k'+str(k):<12}"
                                    f" {_fmt_score(self.metrics[tp], f'score_k{k}_val'):>8}"
                                    f" {_fmt_score(self.metrics[tp], f'score_k{k}_test'):>8}")
                    rows.append(f"  {'recall':<12}"
                                f" {_fmt_score(self.metrics[tp], 'recall_val'):>8}"
                                f" {_fmt_score(self.metrics[tp], 'recall_test'):>8}")
                    rows.append(f"  {'score_min_class':<12}"
                                f" {_fmt_score(self.metrics[tp], 'score_min_class_val'):>8}"
                                f" {_fmt_score(self.metrics[tp], 'score_min_class_test'):>8}")
                    rows.append(f"  {'mean_u':<12} {mu_val:>8.4f} {mu_test:>8.4f}")
                    logger.info(f"\n[search_tp={tp:.2f}]\n" + '\n'.join(rows))

                    # Track 'agg' (from mu_val) instead of 'score_val' for loop logging and early stopping
                    current_agg = mu_val

                    save_object(self.metrics, 'metrics.pkl', self.dir_log)

                    # REPLACED: Tolerance 0.0 because score scale is arbitrary/large and we want strict maximization
                    # CHANGED: We now want to scan ALL candidates for Rank-Based Selection. 
                    # So we update last_score for logging but DO NOT BREAK early.
                    
                    if current_agg >= last_score - tolerance:
                        if current_agg > last_score:
                            last_score = current_agg
                    else:
                        break
                        print(f'Last score {last_score} current score {current_agg} (Continuing search for Rank Selection)')
                        # break  <-- COMMENTED OUT TO TEST ALL CANDIDATES
            
            # --- REFERENCE-NORMALISED SELECTION ---
            # Select tp that maximises mean_u = mean_k u_k,  k ∈ {1,2,3,4,'recall', 'score_min_class'}
            # u_k = clip((s_k - s_baseline) / (|s_k_ref - s_k_baseline| + ε), 0, 1)

            tp_candidates = []
            mean_u_per_tp = {}
            
            for tp, metric_dict in self.metrics.items():
                if not isinstance(tp, float):
                    continue
                
                if self.task_type != 'binary':
                    # Build a mapped dict matching raw output
                    mapped_dict = {}
                    for k in [1, 2, 3, 4]:
                        key = f'score_k{k}_val'
                        vals = np.atleast_1d(metric_dict.get(key, [0.0]))
                        sk = float(np.nanmean(vals)) if len(vals) else 0.0
                        mapped_dict[f'score_k{k}'] = sk if not np.isnan(sk) else 0.0

                    recall_vals = np.atleast_1d(metric_dict.get('recall_val', [0.0]))
                    sk_recall = float(np.nanmean(recall_vals)) if len(recall_vals) else 0.0
                    mapped_dict['recall'] = sk_recall if not np.isnan(sk_recall) else 0.0

                    smc_vals = np.atleast_1d(metric_dict.get('score_min_class_val', [0.0]))
                    sk_smc = float(np.nanmean(smc_vals)) if len(smc_vals) else 0.0
                    mapped_dict['score_min_class'] = sk_smc if not np.isnan(sk_smc) else 0.0

                    #if getattr(self, 'loss', '') == 'bceloss':
                    #    iou_vals = np.atleast_1d(metric_dict.get('iou_val', [0.0]))
                    #    mean_u_per_tp[tp] = float(np.nanmean(iou_vals)) if len(iou_vals) else 0.0
                    #else:
                    agg, u_vals = self._compute_geometric_agg(mapped_dict)
                    # Consistent with early stopping behavior for the selection criterion:
                    mean_u_per_tp[tp] = float(agg)
                    
                    tp_candidates.append(tp)
                else:
                    mean_u_per_tp[tp] = float(np.nanmean(self.metrics[tp]['mean_f1_val']))
                    tp_candidates.append(tp)

            if not tp_candidates:
                logger.warning("No valid k-scores found for Reference Selection. Falling back to score_val maximization.")
                try:
                    best_tp = max(
                        (tp for tp in self.metrics if isinstance(tp, float) and 'score_val' in self.metrics[tp]),
                        key=lambda tp: float(np.nanmean(self.metrics[tp]['score_val']))
                    )
                except Exception as e:
                    raise ValueError(f"Reference Selection failed and Fallback failed: {e}")
            else:
                best_tp = max(tp_candidates, key=lambda tp: mean_u_per_tp[tp])

                # Logging
                logger.info("--- Reference-Normalised Selection Results ---")
                for tp in sorted(tp_candidates):
                    logger.info(f"tp={tp}: mean_u={float(mean_u_per_tp[tp]):.4f}")

            logger.info(f'Best tp {best_tp} (Rank-Based)')
            self.metrics['iou_score'] = iou_scores # Keep legacy key name or update? Let's keep data but variable name is misleading. It's actually score history list but variable iou_scores was empty anyway here
            self.metrics['test_percentage'] = test_percentage
            self.metrics['under_prediction_scores'] = under_prediction_score_scores
            self.metrics['over_prediction_scores'] = over_prediction_score_scores
            self.metrics['best_tp'] = best_tp
            self.metrics['run'] = self.n_run
    
            #logger.info(f'{self.metrics[best_tp]}')
    
            save_object(self.metrics, 'metrics.pkl', self.dir_log)
    
            return best_tp, find_log
        finally:
            try:
                tracemalloc.stop()
            except:
                pass

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
        predictions = predictions[:, 0]
        y = y[:, -1, 0]
        return self.score_with_prediction(predictions, y, sample_weight)
    
    def score_with_prediction(self, y_pred, y, sample_weight=None):
        
        return iou_score(y, y_pred)

    def _predict_test_loader(self, X: DataLoader, prediction_type='Class', output_pdf="test", calibrate=False) -> torch.tensor:
            assert self.model is not None
            self.model.eval()
            """if self.criterion is None:
                print(f'Model cannot predict')
                return None"""
            
            if hasattr(self, 'criterion'):
                criterion = self.criterion
            else:
                criterion = None
            
            with torch.no_grad():
                pred = []
                y = []

                for _, data in enumerate(X, 0):
                    
                    pred_horizon, labels_horizon = self._predict_tensor(data, prediction_type=prediction_type, output_pdf=output_pdf, calibrate=calibrate)
                    
                    pred.append(pred_horizon)
                    y.append(labels_horizon)
                    
                y = torch.cat(y, 0)
                pred = torch.cat(pred, 0)

                if self.task_type == 'regression' and prediction_type == 'Class' and self.apply_discretization:
                    pass
                    """for H in range(self.horizon + 1):
                        pred_h = pred[:, -1 - (self.horizon - H)].detach().cpu().numpy()
                        y_cluster = y[:, departement_index, -1 - (self.horizon - H)]
                        pred_h = self.post_process.predict(pred_h, pred_h, y_cluster)
                        pred[:, -1 - (self.horizon - H)] = torch.as_tensor(pred_h)
                        
                        y_h = y[:, -1, -1 - (self.horizon - H)].detach().cpu().numpy()
                        y_cluster = y[:, departement_index, -1 - (self.horizon - H)]
                        y_h = self.post_process.predict(y_h, y_h, y_cluster)
                        y[:, -1, -1 - (self.horizon - H)] = torch.as_tensor(y_h)
                        
                        print(np.unique(pred_h), np.unique(y_h))"""
                        
                elif prediction_type == 'Class' and pred.dtype != torch.long:
                #if pred.dtype != torch.long:
                    pred = torch.round(pred, decimals=1)
                    
            return pred, y
            
    def _predict_tensor(self, X, prediction_type='Class', output_pdf="test", calibrate=False, use_grad=False) -> torch.tensor:
        assert self.model is not None
        self.model.eval()

        """if self.criterion is None:
            print(f'Model cannot predict')
            return None"""
        
        if hasattr(self, 'criterion'):
            criterion = self.criterion
        else:
            criterion = None
                
        if use_grad:
            func = torch.enable_grad
        else:
            func = torch.no_grad
        with func():
                
            inputs, orilabels_, _ = X

            orilabels_ = orilabels_.to(self.device)
            pred_horizon = []
            labels_horizon = []

            hidden_past: List[torch.Tensor] = []  # contiendra des tenseurs (B, D)
            output_past: List[torch.Tensor] = []  # contiendra des tenseurs (B, D)
            
            is_tfn = self.model_name in ['TFN', 'itransformer']
            output_all = logits_all = hidden_all = None

            for H in range(self.horizon + 1):
                orilabels = orilabels_[:, :, -1 - (self.horizon - H)]
                orilabels[:, -1] = orilabels[:,  -1 ] > 0 if self.task_type == 'binary' else orilabels[:,  -1 ]
                
                if is_tfn:
                    if H == 0:
                        # Appel unique du modèle — toutes les prédictions d'horizons retournées d'un coup
                        inputs_horizon = self.compute_inputs(inputs, -1 - self.horizon, "current")
                        output_all, logits_all, hidden_all = self.model(inputs_horizon, z_prev=None)
                    # Extraction du slice correspondant à l'horizon H
                    output = output_all[:, H, :]
                    logits = logits_all[:, H, :]
                    hidden = hidden_all[:, H, :]
                else:
                    inputs_horizon = self.compute_inputs(inputs, -1 - (self.horizon - H), "current" if H == 0 else "futur")
                    
                    if H == 0:
                        z_prev = None
                    else:
                        if self.ks > 0:
                            # on prend les ks derniers états cachés déjà vus
                            history = hidden_past[-(self.ks + 1):]
                            # empilement (B, D, L) avec L = len(history)
                            z_prev = torch.stack(history, dim=2)  # (B, D, L)

                            # padding à gauche si L < ks
                            L = z_prev.size(2)
                            if L < (self.ks + 1):
                                B, D = z_prev.size(0), z_prev.size(1)
                                pad = torch.zeros(
                                    (B, D, self.ks + 1 - L),
                                    device=z_prev.device,
                                    dtype=z_prev.dtype
                                )
                                z_prev = torch.cat([pad, z_prev], dim=2)  # (B, D, ks)
                        else:
                            z_prev = hidden_past[-1]

                    if H == 0:
                        output, logits, hidden = self.model(inputs_horizon, z_prev=None)
                    else:
                        if self.id_past_risk is not None:
                            inputs_horizon[:, self.id_past_risk, -H:] = 0
                        if self.id_past_ba is not None:
                            inputs_horizon[:, self.id_past_ba, -H:] = 0
                        if self.prev_idx is not None:
                            inputs_horizon[:, self.prev_idx, -H:] = torch.stack(output_past, dim=2)
                        
                        output, logits, hidden = self.model(inputs_horizon, z_prev=z_prev)
                
                hidden_past.append(hidden)
                output_past.append(output)
                    
                if prediction_type != 'RawFormulaVal':
                    if 'criterion' in locals() and hasattr(criterion, 'calibrate') and calibrate:
                        if 'clusters_ids' in required_params(criterion.transform):
                            clusters_ids = orilabels[:, criterion.id].long()
                            calibration = criterion.calibrate(inputs=logits, y_true=orilabels[:, -1], score_fn=iou_score, clusters_ids=clusters_ids, dir_output=self.dir_log)
                        else:
                            calibration = criterion.calibrate(inputs=logits, y_true=orilabels[:, -1], score_fn=iou_score, dir_output=self.dir_log)
                        
                        self.calibration = calibration
                    
                    elif 'criterion' in locals() and hasattr(criterion, 'calibrate'):
                        assert hasattr(self, 'calibration')
                            
                    if 'criterion' in locals() and hasattr(criterion, 'transform'):
                        params = {'inputs' : logits}
                        if 'clusters_ids' in required_params(criterion.transform):
                            clusters_ids = orilabels[:, criterion.id].long()
                            params['clusters_ids'] = clusters_ids
                            
                        if 'output_pdf' in required_params(criterion.transform):
                            assert output_pdf is not None and self.dir_log is not None
                            params['output_pdf'] = output_pdf
                        
                        if 'dir_output' in required_params(criterion.transform):    
                            params['dir_output'] = self.dir_log
                                
                        if 'areas' in required_params(criterion.transform):
                            params['areas'] = orilabels[:, area_index]
                            
                        if 'p_thresh' in required_params(criterion.transform):
                            params['p_thresh'] = self.calibration

                        params['prediction_type'] = prediction_type
                        output = criterion.transform(**params)
                        
                if hasattr(criterion, 'score_to_class') and prediction_type == 'Class':
                    
                    self.ccllt_diff_params = {'graph_id': [], 'date': [], 'pred_bin': [], 'pred_argmax': []}
                    
                    clusters_ids = orilabels[:, criterion.id].long()
                    departement_ids = orilabels[:, departement_index].long()
                    
                    probs = output.detach().clone()
                    
                    pred_bin = criterion.score_to_class(
                        output,
                        clusters_ids=clusters_ids,
                        departement_ids=departement_ids
                    ).detach().cpu()

                    pred_argmax = probs.argmax(dim=1).detach().cpu()
                    
                    output = criterion.score_to_class(output, clusters_ids, departement_ids)
                        
                    diff_mask = (output.detach().cpu() != pred_argmax)
                    diff_mean = diff_mask.float().mean().item()
                    self.diff_bin_argmax = diff_mean
                    if self.metrics is None:
                        self.metrics = {}
                    self.metrics['diff_bin_argmax'] = self.diff_bin_argmax
                    
                    if diff_mask.any():
                        indices = torch.where(diff_mask)[0]
                        for idx in indices:
                            idx_item = idx.item()
                            self.ccllt_diff_params['graph_id'].append(orilabels[idx_item, graph_id_index].item())
                            self.ccllt_diff_params['date'].append(orilabels[idx_item, date_index].item())
                            self.ccllt_diff_params['pred_bin'].append(output[idx_item].item())
                            self.ccllt_diff_params['pred_argmax'].append(pred_argmax[idx_item].item())
                    
                    #output = pred_argmax
                    
                if prediction_type == 'Class':
                    
                    if self.task_type == 'classification' or self.task_type == 'binary' or self.task_type == 'corn':
                        output = torch.argmax(output, dim=1)
                        
                    elif self.task_type == 'uclassification':
                        output = torch.argmax(output[:, :-1], dim=1)
                        
                    elif self.task_type == 'regression' and output.ndim > 1 and output.shape[1] > 1:
                        output = torch.argmax(output, dim=1)

                elif prediction_type == 'RawFormulaVal':
                    output = logits
                    
                pred_horizon.append(output[:, None])
                labels_horizon.append(orilabels[:, :, None])

        pred = torch.cat(pred_horizon, dim=1)
        y = torch.cat(labels_horizon, dim=2)

        if self.task_type == 'regression' and prediction_type == 'Class' and self.apply_discretization:
            for H in range(self.horizon + 1):
                pred_h = pred[:, -1 - (self.horizon - H)].detach().cpu().numpy()
                y_cluster = y[:, departement_index, -1 - (self.horizon - H)]
                pred_h = self.post_process.predict(pred_h, pred_h, y_cluster)
                pred[:, -1 - (self.horizon - H)] = torch.as_tensor(pred_h)
                
                y_h = y[:, -1, -1 - (self.horizon - H)].detach().cpu().numpy()
                y_cluster = y[:, departement_index, -1 - (self.horizon - H)]
                y_h = self.post_process.predict(y_h, y_h, y_cluster)
                y[:, -1, -1 - (self.horizon - H)] = torch.as_tensor(y_h)
                
                print(np.unique(pred_h), np.unique(y_h))
                
        elif prediction_type == 'Class' and pred.dtype != torch.long:
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
        
    def train(self, graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose=True, custom_model_params=None, new_model=True, min_epochs=1, n_runs=1):

        logger.info(
            f"\n{'='*60}\n"
            f"  Training model : {self.name}\n"
            f"  task_type      : {getattr(self, 'task_type', 'N/A')}\n"
            f"  target         : {getattr(self, 'target_name', 'N/A')}\n"
            f"  epochs         : {epochs}  |  n_runs : {n_runs}\n"
            f"  train loader   : {len(self.train_loader)} batches\n"
            f"  val loader     : {len(self.val_loader)} batches\n"
            f"{'='*60}"
        )

        if self.loss_param_search:
            self.train_optuna(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose, custom_model_params, new_model, min_epochs)
            return

        original_dir_log = self.dir_log
        all_runs_scores = {}
        all_criterion_params = {}
        all_run_criteria = {}

        for r in range(n_runs):
            self.criterion_params = []
            logger.info(f"============= Starting RUN {r+1}/{n_runs} =============")
            self.dir_log = original_dir_log / f"run_{r}"
            check_and_create_path(self.dir_log)
            
            # For each run we must start from scratch 
            scores_evolution, criterion_params = self.train_run(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose, custom_model_params, True, min_epochs, run_idx=r)
            all_runs_scores[r] = scores_evolution
            all_criterion_params[r] = criterion_params
            all_run_criteria[r] = deepcopy(self.criterion)

        self.dir_log = original_dir_log
        
        # ── Find best run over all runs and load its model ─────────────────────
        best_run_idx = -1
        best_overall_agg = -float('inf')
        
        for r, scores_by_epoch in all_runs_scores.items():
            if scores_by_epoch:
                best_epoch_run = max(scores_by_epoch.keys(), key=lambda ep: scores_by_epoch[ep].get('agg', -100))
                best_scores = scores_by_epoch[best_epoch_run]
                if best_scores.get('agg', -100) > best_overall_agg:
                    best_overall_agg = best_scores.get('agg', -100)
                    best_run_idx = r
                    
        if best_run_idx != -1:
            logger.info(f"============= ALL RUNS COMPLETED. BEST RUN: {best_run_idx} (agg={best_overall_agg:.4f}) =============")
            # Load the best model into current self.model
            best_model_path = original_dir_log / f"run_{best_run_idx}" / "best.pt"
            self.criterion_params = all_criterion_params[best_run_idx]
            self.criterion = all_run_criteria.get(best_run_idx, self.criterion)
            
            if has_method(self.criterion, 'plot_params'):
                self.criterion.plot_params(self.criterion_params, self.dir_log, best_epoch=self.best_epoch)

            if best_model_path.is_file():
                import shutil
                self._load_model_from_path(best_model_path, self.model)

                # Assert that the weights are correctly loaded
                loaded_state_dict = torch.load(best_model_path, map_location=self.device, weights_only=True)
                for name, param in self.model.named_parameters():
                    if name in loaded_state_dict:
                        assert torch.allclose(param.data.cpu(), loaded_state_dict[name].cpu(), atol=1e-5), f"Model loading failed: weights do not match for {name}"
                
                shutil.copy(best_model_path, original_dir_log / "best.pt")
                logger.info(f"Loaded and saved best model from run {best_run_idx}.")
            last_model_path = original_dir_log / f"run_{best_run_idx}" / "last.pt"

            if last_model_path.is_file():
                import shutil
                shutil.copy(last_model_path, original_dir_log / "last.pt")
        else:
            logger.info("============= ALL RUNS COMPLETED =============")
            
        # Plot variance
        try:
            self.plot_runs_variance(all_runs_scores, n_runs)
        except Exception as e:
            logger.warning(f"Failed to plot runs variance: {e}")
            
    def plot_runs_variance(self, all_runs_scores, n_runs):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        is_dfe = getattr(self, 'target_name', '') == 'DFE'
        
        # Aggregate data
        if is_dfe:
            metrics_to_plot = ['agg', 'iou_score']
        else:
            metrics_to_plot = ['agg', 'score_high', 'score_low', 'score_k1', 'score_k2', 'score_k3', 'score_k4', 'recall', 'score_min_class']
            
        data = []
        mu_data = []
        mu_dense_data = []
        for r, scores_by_epoch in all_runs_scores.items():
            if scores_by_epoch:
                best_epoch_run = max(scores_by_epoch.keys(), key=lambda ep: scores_by_epoch[ep].get('agg', -100))
                best_scores = scores_by_epoch[best_epoch_run]
                
                row = {'run': r, 'best_epoch': best_epoch_run}
                for m in metrics_to_plot:
                    row[m] = best_scores.get(m, np.nan)
                data.append(row)
                
                if not is_dfe:
                    mu_row = {'run': r}
                    for k in range(5):
                        mu_row[k] = best_scores.get(f'mu_{k}', np.nan)
                    mu_data.append(mu_row)
                    
                    mu_dense_row = {'run': r}
                    for idx in range(50):
                        mu_dense_row[idx] = best_scores.get(f'mu_dense_{idx}', np.nan)
                    mu_dense_data.append(mu_dense_row)
                
        if not data:
            return
            
        import pandas as pd
        df = pd.DataFrame(data)
        
        if is_dfe:
            fig, ax_agg = plt.subplots(1, 1, figsize=(8, 6))
            sns.boxplot(data=df[['agg']], ax=ax_agg, palette=["#FF9999"])
            sns.stripplot(data=df[['agg']], ax=ax_agg, color='black', alpha=0.5, size=5)
            ax_agg.set_title('Variabilité du Score Agrégé (IoU)', fontsize=14, fontweight='bold')
            ax_agg.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.dir_log / 'runs_variance_evolution.png')
            plt.close(fig)
            return
        
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(18, 15))
        gs = GridSpec(3, 2, figure=fig)
        
        ax_agg = fig.add_subplot(gs[0, 0])
        sns.boxplot(data=df[['agg']], ax=ax_agg, palette=["#FF9999"])
        sns.stripplot(data=df[['agg']], ax=ax_agg, color='black', alpha=0.5, size=5)
        ax_agg.set_title('Variabilité du Score Agrégé (Agg)', fontsize=14, fontweight='bold')
        ax_agg.grid(True, alpha=0.3)
        
        ax_macro = fig.add_subplot(gs[0, 1])
        sns.boxplot(data=df[['score_high', 'score_low', 'score_min_class']], ax=ax_macro, palette="Set2")
        sns.stripplot(data=df[['score_high', 'score_low', 'score_min_class']], ax=ax_macro, color='black', alpha=0.5, size=4)
        ax_macro.set_title('Variabilité des Macros scores')
        ax_macro.grid(True, alpha=0.3)

        ax_k = fig.add_subplot(gs[1, 0])
        sns.boxplot(data=df[['score_k1', 'score_k2', 'score_k3', 'score_k4']], ax=ax_k, palette="Set3")
        sns.stripplot(data=df[['score_k1', 'score_k2', 'score_k3', 'score_k4']], ax=ax_k, color='black', alpha=0.5, size=4)
        ax_k.set_title('Variabilité des composantes k1 à k4')
        ax_k.grid(True, alpha=0.3)

        ax_rec = fig.add_subplot(gs[1, 1])
        sns.boxplot(data=df[['recall']], ax=ax_rec, palette="Pastel1")
        sns.stripplot(data=df[['recall']], ax=ax_rec, color='black', alpha=0.5, size=4)
        ax_rec.set_title('Variabilité du Recall')
        ax_rec.grid(True, alpha=0.3)

        # ─── Spline Functions subplot ───────────────────────────
        ax_spline = fig.add_subplot(gs[2, :])
        df_mu_dense = pd.DataFrame(mu_dense_data)
        
        # Plot individual runs continuously
        x_vals_dense = np.linspace(0, 4, 50)
        for idx, row in df_mu_dense.iterrows():
            y_vals_dense = [row[k] for k in range(50)]
            ax_spline.plot(x_vals_dense, y_vals_dense, color='gray', alpha=0.4, linewidth=1)
            
        # Plot median and variance area continuously
        if not df_mu_dense.empty:
            mu_median_dense = df_mu_dense[list(range(50))].median()
            mu_min_dense = df_mu_dense[list(range(50))].min()
            mu_max_dense = df_mu_dense[list(range(50))].max()
            
            ax_spline.plot(x_vals_dense, mu_median_dense, color='blue', linewidth=2, label='Médiane')
            ax_spline.fill_between(x_vals_dense, mu_min_dense, mu_max_dense, color='blue', alpha=0.15, label='Min/Max Variabilité')
            
        ax_spline.set_xticks([0, 1, 2, 3, 4])
        ax_spline.set_title('Variabilité de la fonction Spline (Prédiction Y moyenne μ par score)', fontsize=12, fontweight='bold')
        ax_spline.set_xlabel('Score de transition $x \in \{0, 1, 2, 3, 4\}$')
        ax_spline.set_ylabel('$\mu(x)$')
        ax_spline.legend(loc='best')
        ax_spline.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.dir_log / 'runs_variance_evolution.png')
        plt.close()

            
    def filtering_pred(self, df, predTensor, y, graph, return_y = False):
        
        y = y.detach().cpu().numpy()
        predTensor = predTensor.detach().cpu().numpy()

        # Extraire les paires de test_dataset_dept
        test_pairs = set(zip(df['date'], df['graph_id'], df['scale']))

        # Normaliser les valeurs dans YTensor
        date_values = [item for item in y[:, date_index, 0]]
        graph_id_values = [item for item in y[:, graph_id_index, 0]]
        scale_values = [item for item in y[:, scale_index, 0]]

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
        ind = np.lexsort((y[:, graph_id_index, 0], y[:, date_index, 0], y[:, scale_index, 0]))
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
        if graph is None:
            graph = self.graph

        if isinstance(df, pd.DataFrame):

            if self.target_name not in list(df.columns):
                df[self.target_name] = 0
            
            loader = create_test_loader(graph, df,
                        self.features_name,
                        self.device,
                        None,
                        self.target_name,
                        self.ks,
                        self.horizon)
            
            predTensor, YTensor = self._predict_test_loader(loader, prediction_type=prediction_type)
        
        else:
            predTensor, YTensor = self._predict_tensor(df, prediction_type=prediction_type)
            if return_y:
                return predTensor, YTensor
            else:
                return predTensor
            
        if return_y:
            return predTensor, YTensor
        
        return predTensor
    
    def predict_proba(self, df, graph=None, return_y=False, prediction_type="Proba"):
        if graph is None:
            graph = self.graph
            
        if prediction_type == 'Class':
            prediction_type = 'Proba'

        if isinstance(df, pd.DataFrame):
            
            if self.target_name not in list(df.columns):
                df[self.target_name] = 0

            loader = create_test_loader(graph, df,
                        self.features_name,
                        self.device,
                        None,
                        self.target_name,
                        self.ks,
                        self.horizon)
            
            predTensor, YTensor = self._predict_test_loader(loader, prediction_type=prediction_type)
        
        else:
            predTensor, YTensor = self._predict_tensor(df, prediction_type=prediction_type)
            if return_y:
                return predTensor, YTensor
            else:
                return predTensor
            
        if return_y:
            pred, y = self.filtering_pred(df, predTensor, YTensor, graph, return_y=return_y)
            return pred, y
        pred = self.filtering_pred(df, predTensor, YTensor, graph, return_y=return_y)
        return pred
    
    def plot_train_val_loss(self, epochs, train_loss_list, val_loss_list, dir_log):
        logger.info(f"Generating loss plots in {dir_log}...")
        if not train_loss_list or not val_loss_list:
            return

        if isinstance(train_loss_list[0], dict):
            keys = list(train_loss_list[0].keys())
            
            # We ignore non-loss keys like 'id' if they exist, and 'total_loss' for the combined components plot
            # We also ignore 'l' (legacy scalar loss) to avoid redundancy in the breakdown plot
            ignore_keys = ['id', 'C', 'l']
            component_keys = [k for k in keys if k not in ignore_keys and k != 'total_loss' and 'mean_' not in k]
            
            # 1) Overall Combined components plot (Training)
            if component_keys:
                plt.figure(figsize=(12, 7))
                for key in component_keys:
                    train_values = [float(epoch_dict.get(key, 0.0)) for epoch_dict in train_loss_list]
                    plt.plot(epochs, train_values, label=key)
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title('Training Loss Components (Breakdown)')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.savefig(dir_log / 'Training_Loss_Components_Combined.png')
                plt.close()

                # Overall Combined components plot (Validation)
                plt.figure(figsize=(12, 7))
                for key in component_keys:
                    val_values = [float(epoch_dict.get(key, 0.0)) for epoch_dict in val_loss_list]
                    plt.plot(epochs, val_values, label=key)
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title('Validation Loss Components (Breakdown)')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.savefig(dir_log / 'Validation_Loss_Components_Combined.png')
                plt.close()

            # 2) Individual plots for EVERY key in the dictionary (including total_loss and metrics)
            for key in keys:
                if key == 'id': continue
                
                train_values = [float(epoch_dict.get(key, 0.0)) for epoch_dict in train_loss_list]
                val_values = [float(epoch_dict.get(key, 0.0)) for epoch_dict in val_loss_list]

                plt.figure(figsize=(10, 6))
                plt.plot(epochs, train_values, label='Training', color='red', alpha=0.8)
                plt.plot(epochs, val_values, label='Validation', color='blue', alpha=0.8)
                plt.xlabel('Epochs')
                plt.ylabel(key)
                plt.title(f'{key} over Epochs')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.savefig(dir_log / f'loss_{key}_detailed.png')
                plt.close()

        else:
            # Standard logic for scalar loss lists
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_loss_list, label='Training Loss', color='red')
            plt.plot(epochs, val_loss_list, label='Validation Loss', color='blue')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Loss over Epochs')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(dir_log / 'Loss_Trajectory.png')
            plt.close()
        plt.close('all')

    def _load_model_from_path(self, path : Path, model) -> None:
        static_idx, temporal_idx = get_static_temporal_idx(self.features_name)
        new_params = {'static_idx': static_idx, 'temporal_idx' : temporal_idx}
        if self.model_name == 'TFN':
            new_params['d_static'] = len(static_idx)
        
        model, _ = self.make_model(self.graph, new_params)
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
        
    def get_loss(self, loss_name, loss_params):
        
        if 'ccllt' in loss_name or "ranknet" in loss_name or 'msetheta' in loss_name:
            loss_params['ndepartements'] = np.unique(self.udepts).shape[0]
        
        if 'ccllt' in loss_name and "bdiff" in self.dir_log.as_posix():
            loss_params['clustersequaldept'] = True
            
        if 'DualTraining-num' in self.name:
            loss_params.update({'num_classes' : 4})
        else:
            loss_params.update({'num_classes' : 5})
        return get_loss_function(loss_name, **loss_params)

    def get_learnable_parameters(self, criterion):
        """
        Retourne les paramètres apprenables (list/param groups) pour l'optimizer.
        Tous les objets doivent être nn.Parameter avec requires_grad=True.
        """
        params = list(self.model.parameters())

        print('has_method(criterion, get_learnable_parameters)', has_method(criterion, 'get_learnable_parameters'))

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

        # Ajouter les paramètres de distillation s'ils existent
        if hasattr(self, 'fitnets') and self.fitnets is not None:
            logger.info("Adding FitNets parameters")
            params.extend(list(self.fitnets.parameters()))
            
        if hasattr(self, 'relation_mlp') and self.relation_mlp is not None:
            logger.info("Adding RelationMLP parameters")
            params.extend(list(self.relation_mlp.parameters()))
            
        if hasattr(self, 'relation_att') and self.relation_att is not None:
            logger.info("Adding RelationAttention parameters")
            params.extend(list(self.relation_att.parameters()))
            
        if hasattr(self, 'adapter') and self.adapter is not None:
            logger.info("Adding Adapter parameters")
            params.extend(list(self.adapter.parameters()))

        # Ajouter les area_parameters si la loss l'exige
        if 'learnable-area' in self.loss:
            logger.info("Adding learnable area parameters")
            # self.area_parameters est déjà nn.Parameter
            params.append(self.area_parameters)

        if self.student_train and self.temperature == 'seach':
            params.append(self.temperature_value)
            
        if self.student_train and self.alpha == 'seach':
            params.append(self.alpha_value)
        
        # Add RelationMLP parameters for RelationMLP distillation mode
        if self.student_train and self.distillation_training_mode == 'RelationMLP':
            logger.info("Adding RelationMLP parameters to optimizer")
            params.extend(self.relation_mlp.parameters())
        
        return params
    
    def get_optimizer(self, criterion,):
        parameters = self.get_learnable_parameters(criterion)
        optimizer = optim.Adam(parameters, lr=self.lr)
        #optimizer = optim.SGD(parameters, lr=self.lr, momentum=0.9)
        return optimizer
    
    def shapley_additive_explanation(self, df, outname, dir_output, mode='beeswarm', figsize=(15, 25), samples=None, samples_name=None, horizon_shap=0, plot=True):
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

        # Utiliser un backend non-interactif pour éviter les erreurs Qt
        import matplotlib
        matplotlib.use('Agg')
        
        if hasattr(self, 'use_temporal_as_edges'):
            use_temporal_as_edges = self.use_temporal_as_edges
        else:
            use_temporal_as_edges = None

        Xst, y, e = get_numpy_data(self.graph, df, self.features_name, use_temporal_as_edges, self.ks, self.horizon)
        Xst = torch.Tensor(Xst).to(self.device)
        y_test = torch.Tensor(y).to(self.device)
        B, F, T = Xst.shape
        
        Xst_horizon = self.compute_inputs(Xst,  -1 - (self.horizon - horizon_shap), "current" if horizon_shap == 0 else "futur")
        Xst_flat = Xst.reshape((B, F*T))
        Xst_horizon_flat = Xst_horizon[:, :, -1]
        
        df_features = []
        
        """if self.under_sampling == 'search':
            y = self.df_train[self.target_name].values
            nb = int(self.metrics['best_tp'] * len(y[y == 0]))
            df_combined = self.split_dataset(self.df_train, nb, reset=False)
            self.df_train['weight'] = 0

            # Mettre à jour df_train pour l'entraînement
            self.df_train.loc[df_combined.index, 'weight'] = 1"""
            
        background_data_train = self.df_train
                        
        background_data_train, y_background, e = get_numpy_data(self.graph, background_data_train, self.features_name, use_temporal_as_edges, self.ks, self.horizon)
        background_data_train = torch.Tensor(background_data_train).to(self.device)
        y_background = torch.Tensor(y_background).to(self.device)
        B_train, F_train, T_train = background_data_train.shape

        background_data_train = background_data_train.reshape((B_train, F_train*T_train))
        wm = WrapperModel(self, F, T, e, y_background, horizon_shap).to(self.device)

        # SHAP DeepExplainer avec wrapper du modèle
        self.model.eval()
        if not hasattr(self, 'explainer') or self.explainer is None or not isinstance(self.explainer, dict):
            self.explainer = {}
            
        # Vérifier si l'explainer a été instancié avec l'ancien code multi-classe (num_outputs > 1) 
        # car on force désormais une sortie scalaire.
        if horizon_shap in self.explainer:
            if hasattr(self.explainer[horizon_shap], 'expected_value'):
                ev = self.explainer[horizon_shap].expected_value
                is_multi = isinstance(ev, list) or (isinstance(ev, np.ndarray) and ev.size > 1)
                if is_multi:
                    print(f"L'explainer en cache pour horizon {horizon_shap} est multi-classes. On le recrée pour ignorer le cache obsolète.")
                    del self.explainer[horizon_shap]
            
        if horizon_shap not in self.explainer:
            print(f"Initializing DeepExplainer for horizon {horizon_shap} with background data shape: {background_data_train.shape}")
            self.explainer[horizon_shap] = shap.DeepExplainer(wm, background_data_train)
        else:
            print(f'Explainer for horizon {horizon_shap} already exists')
            
        self.model.eval()
        wm.y_background = y_test
        shap_values = self.explainer[horizon_shap].shap_values(Xst_flat, check_additivity=False)
        
        n_classes = 1

        # Reformater proprement shap_values (n_classes, B, F*T) s'il s'agit d'une liste
        if isinstance(shap_values, list):
            shap_values = np.asarray(shap_values)
        else:
            shap_values = np.asarray(shap_values)
            if shap_values.ndim == 2:
                shap_values = shap_values[np.newaxis, :, :]

        # On passe de (n_classes, B, F*T) à (B, F*T, n_classes)
        shap_values = np.moveaxis(shap_values, 0, -1)

        expected_shape = (B, F, n_classes)
        try:
            shap_values = np.reshape(shap_values, expected_shape)
        except ValueError as e:
            # Fallback si T > 1 : on somme sur la dimension temporelle
            print(f"Fallback : la forme attendue était {expected_shape} mais on a T={T}.")
            shap_values = np.reshape(shap_values, (B, F, T, n_classes))
            shap_values = np.sum(shap_values, axis=2)
        
        # Retirer la dimension de classe car il n'y en a plus qu'une (score scalaire)
        shap_values = shap_values[:, :, 0]

        # Calcul des valeurs SHAP moyennes et écarts-types globalement
        shap_mean_abs = np.mean(np.abs(shap_values), axis=0)
        shap_std_abs = np.std(np.abs(shap_values), axis=0)
        shap_mean = np.mean(shap_values, axis=0)
        
        df_shap = pd.DataFrame({
            "mean_abs_shap": shap_mean_abs,
            "stdev_abs_shap": shap_std_abs,
            "mean_shap": shap_mean,
            "name": self.features_name
        }).sort_values("mean_abs_shap", ascending=False)
        
        df_features.append(df_shap)

        if plot:
            check_and_create_path(dir_output)
            # Visualisation globale (summary_plot)
            plt.figure(figsize=figsize)
            if mode == 'bar':
                shap.summary_plot(
                    shap_values,
                    features=Xst_horizon_flat,
                    feature_names=self.features_name,
                    plot_type='bar',
                    show=False
                )
            elif mode == 'beeswarm':
                shap.summary_plot(
                    shap_values,
                    features=Xst_horizon_flat,
                    feature_names=self.features_name,
                    show=False,
                    plot_type="dot"
                )

            print(f"Sauvegarde: {dir_output / f'{outname}_shapley.png'}")
            plt.savefig(dir_output / f"{outname}_shapley.png", bbox_inches='tight', dpi=100)
            plt.close('all')

            # Visualisations spécifiques aux échantillons (force_plot)
            if samples is not None and samples_name is not None:
                for i, sample in enumerate(samples):
                    plt.figure(figsize=figsize)
                    
                    expected_value = self.explainer[horizon_shap].expected_value
                    if isinstance(expected_value, (list, np.ndarray)):
                        expected_value = expected_value[0]
                        
                    shap.force_plot(
                        expected_value,
                        shap_values[sample, :],
                        features=df.iloc[sample].values,
                        feature_names=self.features_name,
                        matplotlib=True,
                        show=False
                    )

                    plt.savefig(
                        dir_output / f"{outname}_{samples_name[i]}_shapley.png",
                        bbox_inches='tight'
                    )
                    plt.close('all')
                    
        df_features = pd.concat(df_features)
        save_object(df_features, 'features_importance.pkl', dir_output)
        
        # Sauvegarder les valeurs SHAP ET l'explainer pour réutilisation ultérieure
        shap_data = {
            'shap_values': shap_values,  # Shape: (B, F)
            'expected_values': self.explainer[horizon_shap].expected_value,
            'feature_names': self.features_name,
            'n_classes': 1,
            'B': B,
            'F': F,
            'T': T,
            'Xst_flat': Xst_flat.cpu().numpy() if torch.is_tensor(Xst_flat) else Xst_flat,
            'horizon_shap': horizon_shap,
            'e': None  # Edges pour reconstruire le WrapperModel si nécessaire
        }
        save_object(shap_data, f'{outname}_shap_values.pkl', dir_output)
        
        # Sauvegarder l'explainer séparément (peut être volumineux)
        explainer_data = {
            'explainer': self.explainer[horizon_shap],
            'wrapper_model': WrapperModel(self, F, T, None, y_background, horizon_shap),
            'F': F,
            'T': T,
            'e': None,
            'y' : y_background,
            'horizon_shap': horizon_shap
        }
        #save_object(explainer_data, f'{outname}_shap_explainer.pkl', dir_output)
        save_object(self.explainer[horizon_shap], f'{outname}_shap_explainer.pkl', dir_output)
        print(f"SHAP values sauvegardées dans: {dir_output / f'{outname}_shap_values.pkl'}")
        print(f"SHAP explainer sauvegardé dans: {dir_output / f'{outname}_shap_explainer.pkl'}")

    def shapley_additive_explanation_sample(self, df_sample, explainer, outname, dir_output, 
                                           shap_data_file=None, sample_name=None, 
                                           figsize=(15, 10), generate_force_plot=True, plot=True,
                                           horizon=0):
        """
        Calcule et visualise les valeurs SHAP pour un échantillon spécifique.
        
        :param df_sample: DataFrame contenant un seul échantillon (1 ligne) ou index de l'échantillon dans df_test
        :param outname: Nom de sortie pour les fichiers
        :param dir_output: Répertoire de sortie
        :param shap_data_file: Chemin vers le fichier de SHAP values sauvegardé (optionnel)
        :param sample_name: Nom de l'échantillon pour les fichiers de sortie
        :param figsize: Taille des figures
        :param generate_force_plot: Si True, génère les force plots
        :param plot: Si True, génère les visualisations
        :return: Dictionary contenant les SHAP values pour cet échantillon
        """
        from pathlib import Path
        import pandas as pd
        
        # Priorité 1: Vérifier si self.explainer existe (explainer en mémoire)
        if hasattr(self, 'explainer') and self.explainer is not None and isinstance(self.explainer, dict) and horizon in self.explainer:
            if hasattr(self.explainer[horizon], 'expected_value'):
                ev = self.explainer[horizon].expected_value
                is_multi = isinstance(ev, list) or (isinstance(ev, np.ndarray) and ev.size > 1)
                if is_multi:
                    print(f"L'explainer en cache pour horizon {horizon} est multi-classes. Impossible d'utiliser le cache avec le nouveau WrapperModel. Veuillez recréer l'explainer.")
                    raise ValueError("Cache d'explainer SHAP incompatible (multi-classes) avec le nouveau modèle. Relancez d'abord shapley_additive_explanation global.")
                    
            print(f"Utilisation de self.explainer[{horizon}] (en mémoire)")
            
            # Préparer les données pour cet échantillon
            if hasattr(self, 'use_temporal_as_edges'):
                use_temporal_as_edges = self.use_temporal_as_edges
            else:
                use_temporal_as_edges = None
            
            Xst_sample, e = get_numpy_data(self.graph, df_sample, self.features_name, use_temporal_as_edges, self.ks, self.horizon)
            Xst_sample = torch.Tensor(Xst_sample).to(self.device)
            B, F, T = Xst_sample.shape
            Xst_sample_flat = Xst_sample.reshape((B, F*T))
            
            # Calculer les SHAP values pour cet échantillon
            
            # Activer le mode logits si le modèle est un WrapperModel
            sample_shap_values_raw = self.explainer[horizon].shap_values(Xst_sample_flat, check_additivity=False)
            
            n_classes = 1
            
            # Reformater proprement
            if isinstance(sample_shap_values_raw, list):
                sample_shap_values_raw = np.asarray(sample_shap_values_raw)
            else:
                sample_shap_values_raw = np.asarray(sample_shap_values_raw)
                if sample_shap_values_raw.ndim == 2:
                    sample_shap_values_raw = sample_shap_values_raw[np.newaxis, :, :]
            
            # shape est (n_classes, 1, F*T)
            sample_shap_values_raw = np.moveaxis(sample_shap_values_raw, 0, -1) # -> (1, F*T, n_classes)
            
            expected_shape = (1, F, n_classes)
            try:
                sample_shap_values_raw = np.reshape(sample_shap_values_raw, expected_shape)
            except ValueError as e:
                # Fallback proportionnel
                sample_shap_values_raw = np.reshape(sample_shap_values_raw, (1, F, T, n_classes))
                sample_shap_values_raw = np.sum(sample_shap_values_raw, axis=2)
            
            # Extraire pour cet échantillon
            sample_shap_values = sample_shap_values_raw[0, :, 0]  # Shape: (F,)
            sample_features = Xst_sample[:, :,  -1 - (self.horizon - horizon)].cpu().numpy()
            expected_values = self.explainer[horizon].expected_value
            if isinstance(expected_values, (list, np.ndarray)):
                expected_values = expected_values[0]
            feature_names = self.features_name
        else:
            print(f"Aucune SHAP value ni explainer pré-calculé trouvé.")
            raise FileNotFoundError(
                f"Impossible de trouver les fichiers SHAP nécessaires:\n"
                f"Veuillez d'abord exécuter shapley_additive_explanation() pour calculer et sauvegarder les valeurs SHAP."
            )
        
        # Générer les visualisations
        results = {
            'shap_values': sample_shap_values,
            'features': sample_features,
            'feature_names': feature_names,
            'plots_generated': []
        }
        
        if plot:
            # 1. Bar plot des valeurs SHAP pour cet échantillon
            plt.figure(figsize=figsize)
            
            # Créer un DataFrame pour faciliter la visualisation
            # Flatten sample_features to 1D if needed (it may have shape (1, F) or (F,))
            sample_features_flat = sample_features.flatten() if sample_features.ndim > 1 else sample_features
            
            shap_df = pd.DataFrame({
                'feature': feature_names,
                'shap_value': sample_shap_values,
                'feature_value': sample_features_flat[:len(feature_names)]
            })
            shap_df = shap_df.reindex(shap_df['shap_value'].abs().sort_values(ascending=False).index)
            
            # Limiter aux 10 features les plus importantes pour le bar plot
            shap_df_top10 = shap_df.head(10)
            
            # Bar plot
            colors = ['red' if x < 0 else 'blue' for x in shap_df_top10['shap_value']]
            plt.barh(range(len(shap_df_top10)), shap_df_top10['shap_value'], color=colors)
            plt.yticks(range(len(shap_df_top10)), shap_df_top10['feature'])
            plt.xlabel('SHAP value (Ordinal Impact)')
            plt.title(f'SHAP Values (Top 10) - {sample_name}')
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            plt.tight_layout()
            
            bar_plot_path = dir_output / f"{outname}_{sample_name}_shap_bar.png"
            plt.savefig(bar_plot_path, bbox_inches='tight', dpi=100)
            plt.close('all')
            results['plots_generated'].append(str(bar_plot_path))
            print(f"Sauvegardé: {bar_plot_path}")
            
            # 2. Waterfall plot (si SHAP le supporte)
            try:
                plt.figure(figsize=figsize)
                shap.plots._waterfall.waterfall_legacy(
                    expected_values,
                    sample_shap_values,
                    feature_names=feature_names,
                    max_display=20,
                    show=False
                )
                waterfall_path = dir_output / f"{outname}_{sample_name}_shap_waterfall.png"
                plt.savefig(waterfall_path, bbox_inches='tight', dpi=100)
                plt.close('all')
                results['plots_generated'].append(str(waterfall_path))
                print(f"Sauvegardé: {waterfall_path}")
            except Exception as e:
                print(f"Impossible de générer le waterfall plot: {e}")
            
            # 3. Force plot (optionnel)
            if generate_force_plot:
                try:
                    plt.figure(figsize=figsize)
                    # Arrondir les valeurs SHAP à 3 décimales pour meilleure visibilité
                    sample_shap_values_rounded = np.round(sample_shap_values, 5)
                    expected_values_rounded = np.round(expected_values, 3)
                    shap.force_plot(
                        expected_values_rounded,
                        sample_shap_values_rounded,
                        features=sample_features_flat[:len(feature_names)],
                        feature_names=feature_names,
                        matplotlib=True,
                        show=False
                    )
                    force_plot_path = dir_output / f"{outname}_{sample_name}_shap_force.png"
                    plt.savefig(force_plot_path, bbox_inches='tight', dpi=100)
                    plt.close('all')
                    results['plots_generated'].append(str(force_plot_path))
                    print(f"Sauvegardé: {force_plot_path}")
                except Exception as e:
                    print(f"Impossible de générer le force plot: {e}")
            
        # Sauvegarder les résultats pour cet échantillon
        save_object(results, f'{outname}_{sample_name}_shap_results.pkl', dir_output)
        print(f"\nRésultats sauvegardés: {dir_output / f'{outname}_{sample_name}_shap_results.pkl'}")
        print(f"Nombre de visualisations générées: {len(results['plots_generated'])}")
        
        return results

    def suggest_loss_params(self, trial, loss_name: str) -> dict:
        """
        Retourne un dict de paramètres Optuna pour la loss `loss_name`.
        Le dict est conçu pour être passé à: LossClass(**loss_params)

        - Si la loss n'a pas (ou inconnue) d'hyperparamètres explicites: retourne {}.
        - IMPORTANT: ne met pas ici les paramètres "num_classes", "id", "weight", etc.
        qui sont généralement gérés ailleurs dans ton code.
        """
        if loss_name is None:
            return {}

        name = str(loss_name).lower().strip()

        # Normalisation de quelques alias
        if name == "weightedcrossentropy-2":
            name = "weightedcrossentropy"

        params = {}

        # -------------------------
        # Losses sans params évidents -> {}
        # (ou importées d'ailleurs, donc on évite de casser)
        # -------------------------
        no_param_losses = {
            "poisson", "rmsle", "rmse", "mse", "huber", "logcosh", "tukeybiweight",
            "exponential", "ordidice", "dice",
            "kldivloss",
            "egpd", "degpd", "pdegpd", "bulktail", "pdegpdcluster", "degpdcluster", "egpdroot",
            "tailcdf", "tailcdfedges", "tailcdfall",
            "bulktailcdf", "bulktailcdfcluster", "tailcdfcluster",
            "gwdl",
            "cornloss", "cornfl",  # souvent: alpha/gamma gérés ailleurs chez toi, donc on laisse vide ici
            "bceloss",
        }
        if name in no_param_losses:
            return {}

        # -------------------------
        # CDWCELoss(num_classes, alpha=0.5, weight=None)
        # -------------------------
        if name == "cdw":
            # alpha est bien un paramètre du __init__ :contentReference[oaicite:1]{index=1}
            params["alpha"] = trial.suggest_float("cdw_alpha", 0.1, 3.0, log=True)
            return params

        # -------------------------
        # CEWKLoss(num_classes, C=1.0, C1=0.5, ...)
        # -------------------------
        if name == "cewk":
            # C et C1 existent bien :contentReference[oaicite:2]{index=2}
            params["C"]  = trial.suggest_float("cewk_C", 0.0, 1.0)
            params["C1"] = trial.suggest_float("cewk_C1", 0.0, 1.0)
            return params

        # -------------------------
        # MCEAndWKLoss(num_classes, C=..., learned=...)
        # DiceAndWKLoss / OrdinalDiceLossAndWKLoss / ForegroundDiceLossAndWKLoss
        # FocalLossAndWKLoss(num_classes, C=..., gamma=..., alpha=..., learned=...)
        # -------------------------
        if name in {"mcewk", "dwk", "odwk", "fdwk", "flwk"}:
            # Ces losses ont typiquement un mélange via C (0..1) 
            params["C"] = trial.suggest_float(f"{name}_C", 0.0, 1.0)

            if name == "flwk":
                # gamma/alpha existent dans FocalLossAndWKLoss :contentReference[oaicite:4]{index=4}
                params["gamma"] = trial.suggest_float("flwk_gamma", 0.5, 5.0, log=True)
                # alpha peut être float (dans ton implémentation) :contentReference[oaicite:5]{index=5}
                alpha_type = trial.suggest_categorical("flwk_alpha_type", ["scalar", "vector"])

                if alpha_type == "scalar":
                    params["alpha"] = trial.suggest_float("flwk_alpha_scalar", 0.1, 5.0, log=True)

                else:
                    num_classes = 5
                    vec = self.get_class_freq(self.df_train)
                    alpha_vec = [
                    trial.suggest_float(f"flwk_alpha_vec_{i}", vec[i], vec[i], log=True)
                    for i in range(num_classes)
                ]
                    params["alpha"] = alpha_vec

            return params

        # -------------------------
        # FocalWKInversionLoss(num_classes, A,B,C,gamma,alpha, inv_..., ...)
        # -------------------------
        if name == "flwki":
            # A,B,C,gamma,alpha + inv_* sont dans le __init__ :contentReference[oaicite:6]{index=6}
            # On contraint A,B,C à sommer ~1 via une parametrisation simple:
            a = trial.suggest_float("flwki_A", 0.0, 1.0)
            b = trial.suggest_float("flwki_B", 0.0, 1.0 - a)
            c = 1.0 - a - b
            params["A"] = a
            params["B"] = b
            params["C"] = c

            params["gamma"] = trial.suggest_float("flwki_gamma", 0.5, 5.0, log=True)
            params["alpha"] = trial.suggest_float("flwki_alpha", 0.1, 5.0, log=True)

            params["inv_margin"] = trial.suggest_float("flwki_inv_margin", 0.0, 2.0)
            params["inv_max_pairs_per_dep"] = trial.suggest_int("flwki_inv_max_pairs_per_dep", 128, 4096, log=True)
            params["inv_weight_by_distance"] = trial.suggest_categorical("flwki_inv_weight_by_distance", [True, False])

            return params

        # -------------------------
        # MonoticRiskLoss(num_classes, margin, beta_softmin, max_pairs_per_cluster, ...)
        # -------------------------
        if name == "monotonic":
            # margin/beta_softmin/max_pairs_per_cluster existent :contentReference[oaicite:7]{index=7}
            params["margin"] = trial.suggest_float("mono_margin", 0.0, 2.0)
            params["beta_softmin"] = trial.suggest_float("mono_beta_softmin", 1.0, 50.0, log=True)
            params["max_pairs_per_cluster"] = trial.suggest_int("mono_max_pairs_per_cluster", 16, 512, log=True)
            return params

        # -------------------------
        # OrdinalMonotonicLossNoCoverage / WithGains / CORNWithGains (même famille)
        #   - betasoftmin, tviolation, mushrinkalpha, gainsalpha, gainsalpha0, gainsfloorfrac, enforcegainmonotone
        #   - lambdamu0, lambdaentropy, wmed,wmin,wneg,wviol, lambdadir, diralpha, lambdace, lambdagl
        #   - cetype, alpha, gamma (pour la partie CORN focal) (dans CORNWithGains) :contentReference[oaicite:8]{index=8}
        # -------------------------
        if name in {"ordinalnocoverage", "ordinalnocoveragewithgains", "cornwithgains"}:
            
            if name in {"ordinalnocoveragewithgains", "cornwithgains"}:
                params["id"] = trial.suggest_categorical(
                    f"{name}_id",
                    [departement_index, graph_id_index]
                )

            # paramètres structurels de la surrogate monotone 
            b_soft = trial.suggest_float(f"{name}_betasoftmin", 1.0, 50.0, log=True)
            t_viol = trial.suggest_float(f"{name}_tviolation", 1e-4, 0.5, log=True)
            
            params["betasoftmin"] = max(b_soft, 1e-6)
            params["tviolation"] = max(t_viol, 1e-6)
            params["mushrinkalpha"] = trial.suggest_float(f"{name}_mushrinkalpha", 0.0, 10.0)

            # gains (marges) 
            params["gainsalpha"] = trial.suggest_float(f"{name}_gainsalpha", 0.0, 3.0)
            params["gainsalpha0"] = trial.suggest_float(f"{name}_gainsalpha0", 0.0, 3.0)
            params["gainsfloorfrac"] = trial.suggest_float(f"{name}_gainsfloorfrac", 0.0, 0.5)
            params["enforcegainmonotone"] = trial.suggest_categorical(f"{name}_enforcegainmonotone", [True, False])

            # pondérations internes des pénalités 
            params["wmed"]  = trial.suggest_float(f"{name}_wmed", 0.0, 2.0)
            params["wmin"]  = trial.suggest_float(f"{name}_wmin", 0.0, 2.0)
            params["wneg"]  = trial.suggest_float(f"{name}_wneg", 0.0, 2.0)
            params["wviol"] = trial.suggest_float(f"{name}_wviol", 0.0, 2.0)

            # termes additionnels :contentReference[oaicite:12]{index=12}
            params["lambdamu0"] = trial.suggest_float(f"{name}_lambdamu0", 0.0, 2.0)
            params["lambdaentropy"] = trial.suggest_float(f"{name}_lambdaentropy", 0.0, 1.0)
            params["lambdadir"] = trial.suggest_float(f"{name}_lambdadir", 0.0, 1.0)
            params["diralpha"] = trial.suggest_float(f"{name}_diralpha", 1.0, 2.0)

            params["lambdace"] = trial.suggest_float(f"{name}_lambdace", 0.0, 2.0)
            params["lambdagl"] = trial.suggest_float(f"{name}_lambdagl", 0.0, 2.0)

            # partie CORN focal possible dans CORNWithGains (cetype='cornfl', alpha, gamma) :contentReference[oaicite:13]{index=13}
            if name == "cornwithgains":
                params["cetype"] = trial.suggest_categorical("cornwithgains_cetype", ["cornfl", "corn"])
                alpha_type = trial.suggest_categorical("cornwithgains_alpha_type", ["scalar", "vector"])

                if alpha_type == "scalar":
                    params["alpha"] = trial.suggest_float("cornwithgains_alpha_scalar", 0.1, 5.0, log=True)

                else:
                    num_classes = 5
                    vec = self.get_corn_alpha_from_train_df(self.df_train, self.target_name)
                    alpha_vec = [
                    trial.suggest_float(f"cornwithgains_alpha_vec_{i}", vec[i], vec[i], log=True)
                    for i in range(num_classes - 1)
                ]
                    params["alpha"] = alpha_vec
                params["gamma"] = trial.suggest_float("cornwithgains_gamma", 0.5, 5.0, log=True)

            return params

        # -------------------------
        # OMMSE(lambda_mse, gainsalpha, gainsalpha0, gainsfloorfrac, enforcegainmonotone, mushrinkalpha)
        # -------------------------
        if name == "ommse":
            # paramètres visibles dans __init__ 
            params["lambda_mse"] = trial.suggest_float("ommse_lambda_mse", 0.1, 10.0, log=True)
            params["mushrinkalpha"] = trial.suggest_float("ommse_mushrinkalpha", 0.0, 10.0)
            params["gainsalpha"] = trial.suggest_float("ommse_gainsalpha", 0.0, 3.0)
            params["gainsalpha0"] = trial.suggest_float("ommse_gainsalpha0", 0.0, 3.0)
            params["gainsfloorfrac"] = trial.suggest_float("ommse_gainsfloorfrac", 0.0, 0.5)
            params["enforcegainmonotone"] = trial.suggest_categorical("ommse_enforcegainmonotone", [True, False])
            # addglobal existe aussi :contentReference[oaicite:15]{index=15}
            params["addglobal"] = trial.suggest_categorical("ommse_addglobal", [True, False])
            return params

        # -------------------------
        # CLMBinnedTransitionLoss (cllt)
        #   beta, t, wmed, wmin, wneg, gamma
        #   wkdecay, wkpower, wklambda, wkmin
        #   learngains, gainsfloor
        # -------------------------
        if name == "cllt":
            # Sharpness du softmin (beta grand => proche du vrai min)
            params["beta"] = trial.suggest_float("cllt_beta", 2.0, 50.0, log=True)

            # Seuil de softplus (t) – petite valeur, fine-tunable
            params["t"] = trial.suggest_float("cllt_t", 1e-3, 0.5, log=True)

            # Pondérations internes des trois pénalités par ordre de transition
            params["wmed"] = trial.suggest_float("cllt_wmed", 0.0, 3.0)
            params["wmin"] = trial.suggest_float("cllt_wmin", 0.0, 3.0)
            params["wneg"] = trial.suggest_float("cllt_wneg", 0.0, 3.0)

            # Sharpness de la ré-pondération des probs (gamma > 1 concentre vers les pics)
            params["gamma"] = trial.suggest_float("cllt_gamma", 1.0, 5.0, log=True)

            # Soft gate sur les probs : taugate = seuil, gatetemp = largeur de transition
            params["taugate"] = trial.suggest_float("cllt_taugate", 0.01, 0.5, log=True)
            params["gatetemp"] = trial.suggest_float("cllt_gatetemp", 0.005, 0.5, log=True)

            # Schedule de poids wk (transitions longue portée pénalisées moins)
            params["wkdecay"] = trial.suggest_categorical("cllt_wkdecay", ["power", "exp", "None"])
            if params["wkdecay"] == "power":
                params["wkpower"] = trial.suggest_float("cllt_wkpower", 0.5, 3.0)
            elif params["wkdecay"] == "exp":
                params["wklambda"] = trial.suggest_float("cllt_wklambda", 0.05, 2.0, log=True)
            params["wkmin"] = trial.suggest_float("cllt_wkmin", 1e-4, 0.1, log=True)

            # Gains learnables (cutpoints inter-bins) et leur floor
            params["learngains"] = trial.suggest_categorical("cllt_learngains", [True, False])
            if params["learngains"]:
                params["gainsfloor"] = trial.suggest_float("cllt_gainsfloor", 0.0, 2.0)

            # Focal loss terms
            params["wfocal"] = trial.suggest_float("cllt_wfocal", 0.0, 2.0)
            params["wmu0"] = trial.suggest_float("cllt_wmu0", 0.0, 2.0)
            params["fgamma"] = trial.suggest_float("cllt_fgamma", 0.5, 5.0, log=True)
            params["falpha"] = trial.suggest_float("cllt_falpha", 0.1, 0.9)

            return params

        # ─────────────────────────────────────────────────────────────────────
        # ClusterCLMBinnedTransitionLoss (ccllt)
        #   Extends cllt with cluster-specific hypers:
        #     scaleagg, weighttype, alphatype, mumomentum, mulambdag, mulambdac
        # ─────────────────────────────────────────────────────────────────────
        if "ccllt" in name:
            params["sigma"]       = self.df_train[self.target_name].std()
            #params["ndepartements"] = 3
            #params["num_classes"]   = 5

            params["wmu0"]        = trial.suggest_float("ccllt_wmu0", 0.0, 3.0, step=0.01)
            params["wmid"]        = trial.suggest_float("ccllt_wmid", 0.0, 5.0, step=0.01)
            params["wtrans"]      = trial.suggest_float("ccllt_wtrans", 0.0, 5.0, step=0.01)
            params["wcoverage"]   = trial.suggest_float("ccllt_wcoverage", 0.0, 5.0, step=0.01)
            params["gainsfloor"]  = trial.suggest_float("ccllt_gainsfloor", 0.5, 5.0, step=0.01)
            params["wkdecay"]     = trial.suggest_categorical("ccllt_wkdecay", ["power", "exp", "None"])
            if params["wkdecay"] == "power":
                params["wkpower"] = trial.suggest_float("ccllt_wkpower", 0.5, 6.0, step=0.01)
            elif params["wkdecay"] == "exp":
                params["wklambda"] = trial.suggest_float("ccllt_wklambda", 0.05, 3.0, log=True)
            params["taugate"]     = trial.suggest_float("ccllt_taugate", 0.01, 0.9, step=0.01)
            params["gatetemp"]    = trial.suggest_float("ccllt_gatetemp", 0.1, 2.0, step=0.01)
            params["massupdate"]  = trial.suggest_float("ccllt_massupdate", 0.0, 1.0, step=0.01)
            params["mumomentum"]  = trial.suggest_float("ccllt_mumomentum", 0.0, 1.0, step=0.01)
            params["mulambdag"]   = trial.suggest_float("ccllt_mulambdag", 0.0, 3.0, step=0.01)
            params["mulambdac"]   = trial.suggest_float("ccllt_mulambdac", 0.0, 3.0, step=0.01)
            params["mulambdad"]   = trial.suggest_float("ccllt_mulambdad", 0.0, 3.0, step=0.01)
            params["shift"]       = trial.suggest_float("ccllt_shift", 0.1, 1.0, step=0.01)

            return params
        
        # ─────────────────────────────────────────────────────────────────────
        # Si on arrive ici: loss inconnue ou pas câblée explicitement
        # ─────────────────────────────────────────────────────────────────────
        
        if "ranknet" in name:
            params["sigma"] = trial.suggest_float("ranknet_sigma", 0.1, 10.0, step=0.1)
            params["num_pairs_per_group"] = trial.suggest_categorical("ranknet_num_pairs_per_group", [None, 512, 1024, 2048])
            params["tie_epsilon"] = trial.suggest_float("ranknet_tie_epsilon", 0.0, 1.0, step=0.01)
            params["use_soft_targets"] = trial.suggest_categorical("ranknet_use_soft_targets", [True, False])
            
            if params["use_soft_targets"]:
                params["soft_target_temperature"] = trial.suggest_float("ranknet_soft_target_temperature", 0.1, 5.0, step=0.1)
                
            params["weight_by_delta"] = trial.suggest_categorical("ranknet_weight_by_delta", [True, False])
            
            if params["weight_by_delta"]:
                params["delta_power"] = trial.suggest_float("ranknet_delta_power", 0.5, 3.0, step=0.1)
                
            params["wrank"] = trial.suggest_float("ranknet_wrank", 0.1, 5.0, step=0.1)
            params["wmid"] = trial.suggest_float("ranknet_wmid", 0.0, 5.0, step=0.1)
            
            return params
            
        return {}
        
    def train_optuna(
        self,
        graph,
        PATIENCE_CNT,
        CHECKPOINT,
        epochs,
        verbose=True,
        custom_model_params=None,
        new_model=True,
        min_epochs=1,
        n_trials=500,
        warmup=5,
        enable_pruning=False,
    ):
        """
        Hyperparameter search with Optuna (multi-objective):
        - objective = (score_k1, score_k2, score_k3, score_k4) all maximize
        - selection at the end: choose trial that maximizes min(scores) (maximin)
        """
        import optuna
        import logging
        import numpy as np
        import random
        import math
        import torch

        optuna_logger = logging.getLogger("optuna")
        optuna_logger.setLevel(logging.INFO)

        # -------------------------
        # Preprocess data arrays once (stable across trials)
        # -------------------------
        y_train = self.df_train[self.target_name].values
        departement_ids = self.df_train["departement"].values if "departement" in self.df_train.columns else None
        node_ids = self.df_train["graph_id"].values if "graph_id" in self.df_train.columns else None
        similar_ids = self.df_train["cluster-encoder"].values if "cluster-encoder" in self.df_train.columns else None

        best_models_states = {}
        best_models_criterion_params = {}
        best_models_loss_params = {}
        best_models_best_scores = {}   # trial → full BEST_SCORES dict (k1..k4, recall, agg)

        def objective(trial):
            from copy import deepcopy
            
            # Seed will be set per-run inside the loop
            if 'g' in globals():
                pass

            logger.info(f"Starting Trial {trial.number}")

            # Suggest loss_name if self.loss is list, else fixed
            if isinstance(self.loss, list):
                loss_name = trial.suggest_categorical("loss_name", self.loss)
            else:
                loss_name = self.loss

            # Suggest parameters for this loss
            loss_params = self.suggest_loss_params(trial, loss_name)
            # Round float params to 2 decimal places for readability
            loss_params = {k: round(v, 2) if isinstance(v, float) else v
                           for k, v in loss_params.items()}

            # Instantiate loss
            try:
                criterion = self.get_loss(loss_name, loss_params)
                self.criterion = criterion  # expose to _predict_test_loader / _predict_tensor
            except Exception as e:
                logger.error(f"Failed to instantiate loss {loss_name} with params {loss_params}: {e}")
                raise optuna.exceptions.TrialPruned()

            # Preprocess if needed
            if hasattr(criterion, "_preprocess"):
                cid = getattr(criterion, "id", None) or loss_params.get("id", None)

                try:
                    if cid == departement_index and departement_ids is not None and similar_ids is not None:
                        criterion._preprocess(y_train, departement_ids, similar_ids)
                    elif cid == graph_id_index and node_ids is not None and similar_ids is not None:
                        criterion._preprocess(y_train, node_ids, similar_ids)
                    else:
                        # fallback minimal
                        criterion._preprocess(y_train)
                except Exception as e:
                    logger.warning(f"_preprocess failed for {loss_name} (id={cid}): {e}")

            if has_method(criterion, 'calculate_class_coverage'):
                try:
                    from GNN.config import cluster_encoder_index
                except ImportError:
                    cluster_encoder_index = None

                cid = getattr(criterion, "id", None) or loss_params.get("id", None)
                cluster_col = 'departement'
                if cid == departement_index:
                    cluster_col = 'departement'
                elif cid == graph_id_index:
                    cluster_col = 'graph_id'
                elif cluster_encoder_index is not None and cid == cluster_encoder_index:
                    cluster_col = 'cluster-encoder'
                
                if cluster_col in self.df_train.columns:
                    criterion.calculate_class_coverage(self.df_train, cluster_col=cluster_col, target_col=self.target_name, dir_output=self.dir_log)

            # Create model and optimizer

            # Log current trial params
            logger.info(f"Trial {trial.number} params: {trial.params}")

            # Create model and optimizer
        
            n_optuna_runs = 1
            optuna_runs_scores = {
                'score_k1': [], 'score_k2': [], 'score_k3': [], 'score_k4': [],
                'recall': [], 'score_min_class': [], 'iou_score': [], 'agg': []
            }
            
            trial_best_model_state = None
            trial_best_criterion_params_state = []
            best_run_agg = -1e9
            
            for optuna_run in range(n_optuna_runs):
                logger.info(f"Trial {trial.number}, Run {optuna_run+1}/{n_optuna_runs}")

                # Seed for reproducibility within run
                run_seed = 42 + (trial.number * 10) + optuna_run
                torch.manual_seed(run_seed)
                np.random.seed(run_seed)
                random.seed(run_seed)
                if 'g' in globals():
                    g.manual_seed(run_seed)

                # Handle custom_model_params (use argument if provided, else empty dict)
                # We use a local variable to avoid modifying the mutable default/argument in place if it's reused
                current_custom_params = custom_model_params.copy() if custom_model_params is not None else {}
    
                static_idx, temporal_idx = get_static_temporal_idx(self.features_name)
            
                new_params = {'static_idx': static_idx, 'temporal_idx' : temporal_idx}
    
                if self.model_name == 'TFN':
                    new_params = {'static_idx': static_idx, 'temporal_idx' : temporal_idx, 'd_static' : len(static_idx)}
                        
                current_custom_params.update(new_params)
    
                self.model, _ = self.make_model(graph, current_custom_params)
                
                init_weight_sum = sum(p.sum().item() for p in self.model.parameters())
                print(f"[OPTUNA][Run {optuna_run}] Initial model weights sum: {init_weight_sum}")
                
                optimizer = self.get_optimizer(criterion)
    
                # Track best per-objective — aligned with the saved model (BEST_SCORES epoch)
                # best_k is NOT accumulated across epochs independently here;
                # it is filled once we know the best epoch via is_better logic.
    
                # Early stopping init
                patience_cnt = 0
                current_patience_lr = 0
                BEST_SCORES = None
                BEST_VAL_LOSS = math.inf
                self.best_epoch = 0
                best_model_state = None
                best_criterion_params_state = []
                self.criterion_params = []
    
                for epoch in range(epochs):
                    self._current_epoch = epoch
    
                    # IMPORTANT: always train/update
                    val_loss, train_loss, val_loss_dict, train_loss_dict = self.func_epoch(
                        train_loader=self.train_loader,
                        val_loader=self.val_loader,
                        optimizer=optimizer,
                        criterion=criterion,
                        do_update=True,
                    )
    
                    if math.isnan(train_loss) or math.isnan(val_loss):
                        logger.warning(f"Trial {trial.number}, Run {optuna_run} pruned at epoch {epoch} because loss is NaN (train={train_loss}, val={val_loss})")
                        raise optuna.exceptions.TrialPruned()
    
                    # Compare against BEST_SCORES to check for improvement
                    current_scores, is_better, rank_sum = self.calculate_val_scores_and_compare(BEST_SCORES)
    
                    if epoch > min_epochs:
                        if val_loss < BEST_VAL_LOSS: # Early stopping calculation on loss and not original score
                            prev_scores = BEST_SCORES
                            BEST_SCORES = current_scores
                            BEST_VAL_LOSS = val_loss
                            patience_cnt = 0
                            current_patience_lr = 0
                            self.best_epoch = epoch
                            best_model_state = deepcopy(self.model.state_dict())
                            best_criterion_params_state = deepcopy(self.criterion_params)

                            ref = getattr(self, 'reference_scores', None)
                            s_ref_map = (ref.get('best_scores') or ref.get('ref_scores', {})) if ref is not None else {}

                            _lines = [
                                f"[T{trial.number}/R{optuna_run}] Epoch {epoch} [✓ NEW BEST LOSS]  agg={current_scores.get('agg', float('nan')):.4f}  val_loss={val_loss:.4f}",
                                f"  {'metric':<10} {'score':>8} {'u (0-1)':>9} {'ref score':>10} {'prev score':>11} {'prev u':>8}",
                                f"  {'─'*63}",
                            ]
                            if 'iou_score' in current_scores:
                                _lines.append(f"  {'iou':<10} {current_scores['iou_score']:>8.4f}")
                            else:
                                for _k in [1, 2, 3, 4]:
                                    _s_cur  = current_scores.get(f'score_k{_k}', float('nan'))
                                    _u_cur  = current_scores.get(f'u_k{_k}',    float('nan'))
                                    _s_ref  = s_ref_map.get(f'score_k{_k}', float('nan'))
                                    _s_prev = (prev_scores or {}).get(f'score_k{_k}', float('nan'))
                                    _u_prev = (prev_scores or {}).get(f'u_k{_k}',    float('nan'))
                                    _lines.append(f"  {'score_k'+str(_k):<10} {_s_cur:>8.4f} {_u_cur:>9.4f} {_s_ref:>10.4f} {_s_prev:>11.4f} {_u_prev:>8.4f}")
                                _rc   = current_scores.get('recall',   float('nan'))
                                _urc  = current_scores.get('u_recall', float('nan'))
                                _rc_ref = s_ref_map.get('recall', float('nan'))
                                _rcp  = (prev_scores or {}).get('recall',   float('nan'))
                                _urcp = (prev_scores or {}).get('u_recall', float('nan'))
                                _lines.append(f"  {'recall':<10} {_rc:>8.4f} {_urc:>9.4f} {_rc_ref:>10.4f} {_rcp:>11.4f} {_urcp:>8.4f}")
                                _smc  = current_scores.get('score_min_class',   float('nan'))
                                _usmc = current_scores.get('u_score_min_class', float('nan'))
                                _smc_ref = s_ref_map.get('score_min_class', float('nan'))
                                _smcp = (prev_scores or {}).get('score_min_class',   float('nan'))
                                _usmcp= (prev_scores or {}).get('u_score_min_class', float('nan'))
                                _lines.append(f"  {'score_min':<10} {_smc:>8.4f} {_usmc:>9.4f} {_smc_ref:>10.4f} {_smcp:>11.4f} {_usmcp:>8.4f}")
                                _lines.append(f"  {'─'*63}")
                                _lines.append(
                                    f"  {'agg (min)':<10} {'':<8} {current_scores.get('agg', float('nan')):>9.4f}"
                                    f" {'':<10} {(prev_scores or {}).get('agg', float('nan')):>11.4f}"
                                )
                            logger.info('\n'.join(_lines))
                        else:
                            patience_cnt += 1
                            if epoch % CHECKPOINT == 0:
                                logger.info(f"[T{trial.number}/R{optuna_run}] Epoch {epoch}  agg={current_scores.get('agg', float('nan')):.4f}"
                                            f"  patience {patience_cnt}/{PATIENCE_CNT}")
                    else:
                        if epoch == min_epochs:
                            BEST_SCORES = current_scores
                            best_model_state = deepcopy(self.model.state_dict())
                            best_criterion_params_state = deepcopy(self.criterion_params)
                        patience_cnt += 1
        
                    # Early stopping and LR decay logic - must run on every epoch
                    if patience_cnt >= PATIENCE_CNT:
                        break
                        # Check if we can reduce LR (have we used all retries?)
                        # PATIENCE_CNT_LR is the number of allowed reductions/retries
                        if current_patience_lr >= self.patience_cnt_lr:
                            logger.info(f'Loss has not increased for {patience_cnt} epochs AND max LR reductions ({self.patience_cnt_lr}) reached.')
                            logger.info(f'Last best val loss {BEST_VAL_LOSS}, current val loss {val_loss}')
                            break
                        else:
                            # Reduce LR and reset patience_cnt
                            if self.delta_lr > 0:
                                current_patience_lr += 1
                                logger.info(f"Patience {PATIENCE_CNT} reached (Retry {current_patience_lr}/{self.patience_cnt_lr}). Decay LR by factor {self.delta_lr}.")

                                current_lr = optimizer.param_groups[0]['lr']
                                new_lr = current_lr * (1 - self.delta_lr)
                                if new_lr <= 1e-9:
                                    new_lr = 1e-9
                                    logger.warning("Learning rate reached floor (1e-9).")

                                logger.info(f"Reducing LR from {current_lr:.6f} to {new_lr:.6f}")

                                for param_group in optimizer.param_groups:
                                    param_group['lr'] = new_lr

                                # Reset patience_cnt to give model time to improve with new LR
                                patience_cnt = 0
                            else:
                                # No delta_lr defined, stop normal
                                logger.info(f'Loss has not increased for {patience_cnt} epochs. No delta_lr defined.')
                                break

                    if epoch % CHECKPOINT == 0 and verbose:
                        curr_lr = optimizer.param_groups[0]['lr']
                        _cur_agg = (current_scores or {}).get('agg', float('nan'))
                        _best_agg = (BEST_SCORES or {}).get('agg', float('nan'))
                        logger.info(f'[T{trial.number}/R{optuna_run}] Ep {epoch}: '
                                    f'val={val_loss:.4f} train={train_loss:.4f} | '
                                    f'agg={_cur_agg:.4f} best_agg={_best_agg:.4f} (best_ep={self.best_epoch} LR={curr_lr:.2e} | Patience: {patience_cnt}/{PATIENCE_CNT} | LR-retry: {current_patience_lr}/{self.patience_cnt_lr})')

            if BEST_SCORES is not None:
                # Save run scores mapping to average them identically as update_metrics_as_arrays does
                for k in [1, 2, 3, 4]:
                    key = f'score_k{k}'
                    optuna_runs_scores[key].append(BEST_SCORES.get(key, 0.0))
                optuna_runs_scores['recall'].append(BEST_SCORES.get('recall', 0.0))
                optuna_runs_scores['score_min_class'].append(BEST_SCORES.get('score_min_class', 0.0))
                optuna_runs_scores['iou_score'].append(BEST_SCORES.get('iou_score', 0.0))
                optuna_runs_scores['agg'].append(BEST_SCORES.get('agg', -1e9)) # For backup sorting
            
            # Keep the best state out of the n_optuna_runs to represent this Trial's best model
            current_agg = BEST_SCORES.get('agg', -1e9) if BEST_SCORES is not None else -1e9

            # Pruning: report agg of the CURRENT epoch (not only best) so MedianPruner
            # can compare intermediate values across trials, but ONLY on the first run of the 5 runs
            if enable_pruning and optuna_run == 0:
                _current_agg = float((current_scores or BEST_SCORES or {}).get('agg', -1e9))
                trial.report(_current_agg, step=epoch)
                if epoch >= warmup and trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            if current_agg > best_run_agg:
                    best_run_agg = current_agg
                    from copy import deepcopy
                    if best_model_state is not None:
                        trial_best_model_state = deepcopy(best_model_state)
                        trial_best_criterion_params_state = deepcopy(best_criterion_params_state)
                    else:
                        trial_best_model_state = deepcopy(self.model.state_dict())
                        trial_best_criterion_params_state = deepcopy(self.criterion_params)
        
            # After 5 runs, save the absolute best model from them to this trial
            if trial_best_model_state is not None:
                best_models_states[trial.number] = trial_best_model_state
                best_models_criterion_params[trial.number] = trial_best_criterion_params_state
            
            from copy import deepcopy
            best_models_loss_params[trial.number] = deepcopy(loss_params)
            
            # --- Average the scores over the 5 runs ---
            # If nothing trained successfully, prune
            if not optuna_runs_scores['agg']:
                return -1e9

            if self.target_name == 'DFE':
                averaged_iou = float(np.mean(optuna_runs_scores['iou_score']))
                best_models_best_scores[trial.number] = {'iou_score': averaged_iou, 'agg': averaged_iou}
                return averaged_iou

            # Compute the geometric agg like search_samples_proportion
            mapped_dict = {
                'score_k1': float(np.mean(optuna_runs_scores['score_k1'])),
                'score_k2': float(np.mean(optuna_runs_scores['score_k2'])),
                'score_k3': float(np.mean(optuna_runs_scores['score_k3'])),
                'score_k4': float(np.mean(optuna_runs_scores['score_k4'])),
                'recall': float(np.mean(optuna_runs_scores['recall'])),
                'score_min_class': float(np.mean(optuna_runs_scores['score_min_class']))
            }
            avg_agg, _ = self._compute_geometric_agg(mapped_dict)
            avg_agg = float(avg_agg)
            
            # Store full scores for post-hoc plots
            best_models_best_scores[trial.number] = mapped_dict
            best_models_best_scores[trial.number]['agg'] = avg_agg

            return avg_agg

        # -------------------------
        # Single-objective study with pruning
        # -------------------------
        study_name = f"optuna_{self.model_name}_{self.loss}"
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=max(3, n_trials // 5),
            n_warmup_steps=warmup,
            interval_steps=1,
        ) if enable_pruning else optuna.pruners.NopPruner()
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=pruner,
        )

        logger.info(f"Starting Optuna search for {n_trials} trials (single-objective: agg)...")
        study.optimize(objective, n_trials=n_trials)

        logger.info("Optuna search completed!")
        logger.info(f"Best trial: {study.best_trial.number}  agg={study.best_trial.value:.4f}")

        # Single best trial
        best_trial = study.best_trial
        logger.info(f"Best trial params: {best_trial.params}")

        self.best_loss_params = best_trial.params
        save_object(best_trial.params, "optuna_best_params.pkl", self.dir_log)

        # Save ALL completed trials
        try:
            trials_data = []
            for t in study.get_trials(deepcopy=False):
                if t.state == optuna.trial.TrialState.COMPLETE:
                    trials_data.append({
                        "trial": t.number,
                        "value": t.value,   # scalar agg
                        "params": t.params
                    })
            save_object(trials_data, "optuna_all_trials.pkl", self.dir_log)
        except Exception as e:
            logger.warning(f"Failed to save optuna all trials: {e}")

        # Plot scores per trial (k1..k4, recall, agg)
        try:
            import matplotlib.pyplot as plt
            import os

            trials = study.get_trials(deepcopy=False)
            completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

            if len(completed_trials) > 0:
                tnums = [t.number for t in completed_trials]
                _score_keys = [('score_k1','k1'), ('score_k2','k2'),
                               ('score_k3','k3'), ('score_k4','k4'),
                               ('recall','recall'), ('score_min_class', 'min_class'), 
                               ('agg','agg')]
                _colors = ['steelblue','darkorange','green','red','purple', 'brown', 'black']

                n_plots = len(_score_keys)
                fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
                if n_plots == 1: axes = [axes]
                
                for ax, (sk, slabel), col in zip(axes, _score_keys, _colors):
                    vals = [best_models_best_scores.get(t.number, {}).get(sk, float('nan'))
                            for t in completed_trials]
                    _ls = '-' if sk != 'agg' else '--'
                    _lw = 2.0
                    ax.plot(tnums, vals, marker='o', markersize=6, label=slabel,
                            color=col, linestyle=_ls, linewidth=_lw)

                    # Mark best trial
                    ax.axvline(best_trial.number, color='r', linestyle=':', alpha=0.7, label=f'best (#{best_trial.number})')
                    
                    if sk == 'score_k1':
                        ax.set_title(f"Optuna scores per trial  [best=#{best_trial.number}, agg={study.best_trial.value:.4f}]")
                    
                    if sk == 'agg':
                        ax.set_xlabel("Trial")
                        
                    ax.set_ylabel(slabel)
                    ax.legend(fontsize=8, loc='best')
                    ax.grid(True, alpha=0.3)
                    
                plot_path = os.path.join(self.dir_log, "optuna_scores_per_trial.png")
                fig.tight_layout()
                fig.savefig(plot_path, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved scores per trial plot to {plot_path}")
        except Exception as e:
            logger.warning(f"Failed to plot optuna scores: {e}")

        # Plot param vs score for each loss parameter
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import os
            import numpy as np

            trials = study.get_trials(deepcopy=False)
            completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

            if len(completed_trials) > 1:


                # Collect all numeric params across completed trials
                all_param_names = set()
                for t in completed_trials:
                    for k, v in t.params.items():
                        if isinstance(v, (int, float)):
                            all_param_names.add(k)
                all_param_names = sorted(all_param_names)

                if all_param_names:
                    params_vs_score_dir = os.path.join(self.dir_log, "params_vs_score")
                    os.makedirs(params_vs_score_dir, exist_ok=True)

                    # Score array per trial: (n_completed, 6) → k1,k2,k3,k4,recall,agg
                    _skeys  = ['score_k1','score_k2','score_k3','score_k4','recall','agg']
                    _slabels = ['k1','k2','k3','k4','recall','agg']
                    trial_scores = np.array([
                        [best_models_best_scores.get(t.number, {}).get(sk, float('nan')) for sk in _skeys]
                        for t in completed_trials
                    ])   # (N, 6)

                    for param_name in all_param_names:
                        # Collect param values (NaN when trial didn't suggest it)
                        param_vals = np.array([
                            float(t.params[param_name]) if param_name in t.params else float("nan")
                            for t in completed_trials
                        ])

                        valid_mask = ~np.isnan(param_vals)
                        if valid_mask.sum() < 2:
                            continue  # not enough data

                        x = param_vals[valid_mask]
                        y_all = trial_scores[valid_mask]   # (M, 6)

                        n_scores = len(_skeys)
                        ncols = 3
                        nrows = (n_scores + ncols - 1) // ncols
                        fig, axes_grid = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharey=False)
                        axes = axes_grid.flatten()
                        _colors = ['steelblue','darkorange','green','red','purple','black']

                        for si, (ax, slabel, color) in enumerate(zip(axes, _slabels, _colors)):
                            y = y_all[:, si]
                            finite_mask = np.isfinite(y)
                            if finite_mask.sum() < 2:
                                ax.set_title(f"{slabel}\n(no data)")
                                continue

                            xf, yf = x[finite_mask], y[finite_mask]

                            # Scatter
                            ax.scatter(xf, yf, alpha=0.5, color=color, s=20)

                            # Trend: sort by x and compute rolling median (window = max(3, N//5))
                            order = np.argsort(xf)
                            xs, ys = xf[order], yf[order]
                            w = max(3, len(xs) // 5)
                            med_x, med_y = [], []
                            for i in range(len(xs)):
                                lo = max(0, i - w // 2)
                                hi = min(len(xs), i + w // 2 + 1)
                                med_x.append(xs[i])
                                med_y.append(np.median(ys[lo:hi]))
                            ax.plot(med_x, med_y, color=color, linewidth=1.5, alpha=0.8)

                            ax.set_xlabel(param_name, fontsize=8)
                            ax.set_ylabel("Score", fontsize=8)
                            ax.set_title(slabel, fontsize=9)
                            ax.grid(True, alpha=0.3)

                        fig.suptitle(f"Loss param: {param_name}", fontsize=11)
                        plt.tight_layout()
                        safe_name = param_name.replace("/", "_").replace("\\", "_")
                        plot_path = os.path.join(params_vs_score_dir, f"{safe_name}.png")
                        plt.savefig(plot_path, bbox_inches="tight")
                        plt.close(fig)


                    logger.info(f"Saved params vs score plots to {params_vs_score_dir}/")
        except Exception as e:
            import traceback
            logger.warning(f"Failed to plot params vs score: {e}\n{traceback.format_exc()}")

        try:
            from copy import deepcopy
            logger.info("Restoring best model from the selected maximin trial...")
            best_params = best_trial.params
            
            # Use current_custom_params to re-create exact same args as objective
            curr_custom_params = custom_model_params.copy() if custom_model_params is not None else {}
            static_idx, temporal_idx = get_static_temporal_idx(self.features_name)
            new_params = {'static_idx': static_idx, 'temporal_idx' : temporal_idx}
            if self.model_name == 'TFN':
                new_params['d_static'] = len(static_idx)
            curr_custom_params.update(new_params)
            
            # Remake model
            self.model, _ = self.make_model(graph, curr_custom_params)
            
            # Remake loss
            if isinstance(self.loss, list):
                loss_name = best_params.get("loss_name", self.loss[0])
            else:
                loss_name = self.loss
                
            real_loss_params = best_models_loss_params.get(best_trial.number, best_params)
            self.criterion = self.get_loss(loss_name, real_loss_params)
            
            # Load best weights
            if best_trial.number in best_models_states:
                self.model.load_state_dict(best_models_states[best_trial.number])
                logger.info("Successfully loaded best trial model state dict.")
            else:
                logger.warning(f"Could not find model state dict for trial {best_trial.number}, model uses random init.")
            
            # Load and plot best criterion params
            if best_trial.number in best_models_criterion_params:
                self.criterion_params = best_models_criterion_params[best_trial.number]
                if has_method(self.criterion, 'plot_params'):
                    self.criterion.plot_params(self.criterion_params, self.dir_log, best_epoch=self.best_epoch)
            
            # Save best.pt
            save_object_torch(self.model.state_dict(), 'best.pt', self.dir_log)
            logger.info("Saved best model to best.pt")
                
        except Exception as e:
            logger.warning(f"Failed to restore best model: {e}")
            real_loss_params = best_trial.params

        # Ensure the real parameters are captured for JSON saving
        if 'real_loss_params' not in locals():
            real_loss_params = best_models_loss_params.get(best_trial.number, best_trial.params)

        with open(os.path.join(self.dir_log, "best_loss_params.json"), "w") as f:
            json.dump(real_loss_params, f, indent=4)

        logger.info(f"====== BEST OPTUNA LOSS PARAMS ======\n{json.dumps(real_loss_params, indent=4)}")

        # -------------------------
        # Optuna Visualizations
        # -------------------------
        try:
            import optuna.visualization as vis
            
            # Optimization history
            fig = vis.plot_optimization_history(study)
            fig.write_html(os.path.join(self.dir_log, "optuna_optimization_history.html"))
            try:
                fig.write_image(os.path.join(self.dir_log, "optuna_optimization_history.png"))
            except:
                pass
            if verbose:
                fig.show()

            # Parameter importance
            fig = vis.plot_param_importances(study)
            fig.write_html(os.path.join(self.dir_log, "optuna_param_importances.html"))
            try:
                fig.write_image(os.path.join(self.dir_log, "optuna_param_importances.png"))
            except:
                pass
            if verbose:
                fig.show()

            # Parallel coordinate plot
            fig = vis.plot_parallel_coordinate(study)
            fig.write_html(os.path.join(self.dir_log, "optuna_parallel_coordinate.html"))
            try:
                fig.write_image(os.path.join(self.dir_log, "optuna_parallel_coordinate.png"))
            except:
                pass
            if verbose:
                fig.show()

            # Contour plot
            fig = vis.plot_contour(study)
            fig.write_html(os.path.join(self.dir_log, "optuna_contour.html"))
            try:
                fig.write_image(os.path.join(self.dir_log, "optuna_contour.png"))
            except:
                pass
            if verbose:
                fig.show()
                
            logger.info(f"Saved Optuna visualizations (HTML/PNG) to {self.dir_log}/")
        except Exception as e:
            logger.warning(f"Failed to generate Optuna visualizations: {e}")

        # Clean memory
        try:
            del best_models_states
            del best_models_criterion_params
            import gc
            gc.collect()
        except:
            pass

        return study

############################################ Split training ##############################################################

class SplitTraining(Training):
    def __init__(self, federated_cluster, cut_layer_name, input_server_model, model_name,
                 nbfeatures, batch_size, lr, delta_lr, patience_cnt_lr, target_name, task_type, out_channels,
                 dir_log, features_name, ks, loss, name, device, under_sampling, over_sampling, n_run,
                 horizon=0, post_process=None, loss_param_search=False):

        super().__init__(model_name, nbfeatures, batch_size, lr, delta_lr, patience_cnt_lr, target_name, task_type, features_name, ks,
                         out_channels, dir_log, loss=loss, name=name, device=device, under_sampling=under_sampling,
                         over_sampling=over_sampling, n_run=n_run, horizon=horizon, post_process=post_process, loss_param_search=loss_param_search)

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
                            self.horizon,
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

    def _predict_test_loader(self, X: DataLoader, prediction_type='Class', output_pdf="test", proba=False, calibrate=False) -> torch.tensor:

        """Generate predictions using the split learning setup."""

        try:
            if self.training_mode == 'normal':
                return super()._predict_test_loader(X, prediction_type=prediction_type, output_pdf=output_pdf, calibrate=calibrate)
        except:
                return super()._predict_test_loader(X, prediction_type=prediction_type, output_pdf=output_pdf, calibrate=calibrate)

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
                if getattr(self, 'task_type', None) == 'uclassification':
                    output = torch.argmax(output[:, :-1], dim=1)
                else:
                    output = torch.argmax(output, dim=1)

                preds.append(output.squeeze(0))

        pred_tensor = torch.stack(preds, 0)
        y_tensor = torch.stack(ys, 0)

        if self.target_name in ["binary", "risk", "nbsinister"]:
            pred_tensor = torch.round(pred_tensor, decimals=1)

        return pred_tensor, y_tensor

    def predict(self, df, graph=None, return_y=False, prediction_type="Class"):
        
        try:
            if self.training_mode == 'normal':
                return super().predict(df, graph=graph, return_y=return_y, prediction_type=prediction_type)
        except:
                return super().predict(df, graph=graph, return_y=return_y, prediction_type=prediction_type)
            
        if graph is None:
            graph = self.graph

        if self.target_name not in list(df.columns):
            df[self.target_name] = 0

        loader = self.prepare_batch_data(df, graph, df[self.federated_cluster].unique(), 1)

        pred_tensor, y_tensor = self._predict_test_loader(loader, output_pdf="test")

        if return_y:
            return pred_tensor, y_tensor

        return pred_tensor

    def predict_proba(self, df, graph=None, return_y=False, prediction_type='Proba'):
        try:
            if self.training_mode == 'normal':
                return super().predict_proba(df, graph=graph, return_y=return_y, prediction_type=prediction_type)
        except:
                return super().predict_proba(df, graph=graph, return_y=return_y, prediction_type=prediction_type)
        
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
            self.horizon
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
        graph : "Any"
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
                metrics_val = self.scoring.evaluate_metrics(y_val_np, pred_val_np, zones=y_val_np[:, graph_id_index], dates=y_val_np[:, date_index])
                metrics_combo['iou_val'].append(metrics_val['iou'])

                pred_test, y_test = model_copy._predict_test_loader(model_copy.test_loader, output_pdf="test")

                y_test_np = y_test.detach().cpu().numpy()[:, -1]
                pred_test_np = pred_test.detach().cpu().numpy()
                metrics_test = self.scoring.evaluate_metrics(y_test_np, pred_test_np, zones=y_test_np[:, graph_id_index], dates=y_test_np[:, date_index])

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
    positive label. Both sub-models are trained sequentially within the
    standard train() / create_train_val_test_loader() interface."""

    def __init__(self, target_name, occ_model: Training, num_model: Training, name, task_type: str,
                 n_run: int = 1, horizon=0, loss_param_search=False):
        self.occ_model = occ_model
        self.num_model = num_model
        self.name = name
        self.task_type = task_type
        self.n_run = n_run
        self.target_name = target_name
        self.horizon = horizon
        self.loss_param_search = loss_param_search
        self.metrics = {}
        self.scoring = Scoring()

    # ------------------------------------------------------------------
    # Loader creation  (does NOT train anything)
    # ------------------------------------------------------------------
    def create_train_val_test_loader(self, graph, dfs_train, dfs_val, dfs_test, epochs, PATIENCE_CNT, CHECKPOINT,
                                     custom_model_params=None, features_importance=False, use_log=True):
        train_dataset, train_pos = dfs_train
        val_dataset, val_pos = dfs_val
        test_dataset, test_pos = dfs_test

        self.num_model.create_train_val_test_loader(
            graph, train_pos, val_pos, test_pos,
            epochs, PATIENCE_CNT, CHECKPOINT,
            custom_model_params=custom_model_params,
            features_importance=False,
            use_log=use_log,
        )

        self.occ_model.create_train_val_test_loader(
            graph, train_dataset, val_dataset, test_dataset,
            epochs, PATIENCE_CNT, CHECKPOINT,
            custom_model_params=custom_model_params,
            features_importance=False,
            use_log=use_log,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, graph, PATIENCE_CNT, CHECKPOINT, epochs,
              verbose: bool = True, custom_model_params=None,
              new_model: bool = True, min_epochs: int = 1):
        """Train num_model first (positive-only), then occ_model over n_run seeds."""

        # ── 1. Train the numeric model (strictly on positive samples) ────────
        logger.info("============= DualTraining: training num_model =============")
        self.num_model.train(graph, PATIENCE_CNT, CHECKPOINT, epochs,
                             verbose, custom_model_params, new_model, min_epochs)

        # ── 2. Train occ_model with n_run seeds, evaluate, track scores ──────
        all_runs_scores = {}
        tp = 'occ-based'

        for run in range(self.n_run):
            logger.info(f"============= DualTraining: occ_model RUN {run + 1}/{self.n_run} =============")
            self.occ_model.seed = int(random.random())
            self.occ_model.n_run = 1

            scores_evolution, _ = self.occ_model.train_run(
                graph, PATIENCE_CNT, CHECKPOINT, epochs,
                verbose, custom_model_params, new_model, min_epochs, run_idx=run
            )
            all_runs_scores[run] = scores_evolution

            # ── Evaluate combined prediction on test set ────────────────────
            test_output, y = self._predict_test_loader(
                (self.occ_model.test_loader, self.num_model.test_loader)
            )
            prediction = test_output.detach().cpu().numpy()[:, 0]
            y_np = y.detach().cpu().numpy()[:, :, 0]

            dff = pd.DataFrame(index=np.arange(y_np.shape[0]))
            dff['departement'] = y_np[:, departement_index]
            dff['date'] = y_np[:, date_index]
            dff['graph_id'] = y_np[:, graph_id_index]
            dff[self.target_name] = y_np[:, -1]

            metrics_run = self.scoring.evaluate_metrics(
                dff[self.target_name], prediction,
                zones=dff['graph_id'], dates=dff['date']
            )
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'test')

        self.metrics[tp] = add_ic95_to_dict(self.metrics[tp], None, "_ic95")
        self.metrics['best_tp'] = tp

        # ── 3. Variance plots (mirrors Training.plot_runs_variance) ──────────
        try:
            self.occ_model.plot_runs_variance(all_runs_scores, self.n_run)
        except Exception as e:
            logger.warning(f"Failed to plot runs variance: {e}")

    # ------------------------------------------------------------------
    # Inference utilities
    # ------------------------------------------------------------------
    def _predict_test_loader(self, loader=None, prediction_type='Class', output_pdf=None, calibrate=False):
        occ_loader, num_loader = loader
        pred_occ, y_occ = self.occ_model._predict_test_loader(occ_loader, prediction_type, output_pdf, calibrate=calibrate)
        pred_num, y_num = self.num_model._predict_test_loader(num_loader, prediction_type, output_pdf, calibrate=calibrate)

        print('Size check ->', y_occ.shape, y_num.shape)

        pred_occ = torch.as_tensor(pred_occ, dtype=torch.float32)
        pred_num = torch.as_tensor(pred_num, dtype=torch.float32)
        for H in range(self.horizon + 1):
            pred_occ_horizon = pred_occ[:, H]
            pred_num_horizon = pred_num[:, H]

            occ_mask = pred_occ_horizon.reshape(-1) > 0
            y_occ_positive_samples = y_occ[occ_mask, :, H]

            selected_idx_num: List[torch.Tensor] = []
            selected_idx_occ: List[torch.Tensor] = []
            for i in range(y_occ_positive_samples.shape[0]):
                sample = y_occ_positive_samples[i]
                date = sample[date_index]
                graph_id = sample[graph_id_index]

                idx = torch.argwhere((y_num[:, date_index, H] == date) & (y_num[:, graph_id_index, H] == graph_id))
                if len(idx) > 0:
                    selected_idx_num += idx

                idx = torch.argwhere((y_occ[:, date_index, H] == date) & (y_occ[:, graph_id_index, H] == graph_id))
                if len(idx) > 0:
                    selected_idx_occ += idx

            print('Size idx check ->', len(selected_idx_num), len(selected_idx_occ))
            pred_occ[selected_idx_occ] = pred_num_horizon[selected_idx_num][..., None] + 1

        return pred_occ, y_num

    def create_test_loader(self, graph, df):
        loader_occ = self.occ_model.create_test_loader(graph, df)
        loader_num = self.num_model.create_test_loader(graph, df)
        return (loader_occ, loader_num)

    def search_samples_proportion(self, *args, **kwargs):
        """Delegate proportion search to the occurence model."""
        return self.occ_model.search_samples_proportion(*args, **kwargs)

    
class Distribution2Class:
    def __init__(self, target_name, distrib_model: Training, class_model: Training, name, task_type: str, n_run : int = 1, horizon=0):
        self.distrib_model = distrib_model
        self.class_model = class_model
        self.name = name
        self.task_type = task_type
        self.n_run = n_run
        self.target_name = target_name
        self.horizon = horizon
        
        print(distrib_model.dir_log / f'{distrib_model.name}.pkl')
        if (distrib_model.dir_log / f'{distrib_model.name}.pkl').is_file():
            self.distrib_model = read_object(f'{distrib_model.name}.pkl', distrib_model.dir_log)
            self.train_distrib_model = False
        else:
            self.train_distrib_model = True

    def create_train_val_test_loader(self, graph, df_train, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT,
                                     features_importance=True, custom_model_params=None, use_log=True):
        
        if self.train_distrib_model:
            print('################ Train distribution model ####################')
            self.distrib_model.create_train_val_test_loader(graph, df_train, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT,
                                     features_importance=features_importance, custom_model_params=custom_model_params, use_log=use_log)
            
            self.distrib_model.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose=True, custom_model_params=custom_model_params, new_model=True)

        print('################ Train class model ####################')
        ### Update df_train
        
        loader = self.distrib_model.create_test_loader(graph, df_train)
        output_train, y_train = self.distrib_model._predict_test_loader(loader, prediction_type='RawFormulaVal', output_pdf='train')
        output_val, y_val = self.distrib_model._predict_test_loader(self.distrib_model.val_loader, prediction_type='RawFormulaVal', output_pdf='Val')
        output_test, y_test = self.distrib_model._predict_test_loader(self.distrib_model.test_loader, prediction_type='RawFormulaVal', output_pdf='test')
        
        df_class_train = pd.DataFrame(index=np.arange(0, y_train.shape[0]))
        df_class_val = pd.DataFrame(index=np.arange(0, y_val.shape[0]))
        df_class_test = pd.DataFrame(index=np.arange(0, y_test.shape[0]))
        
        y_train = y_train[:, :, 0]
        y_val = y_val[:, :, 0]
        y_test = y_test[:, :, 0]
        
        output_train = output_train[:, :, 0]
        output_val = output_val[:, :, 0]
        output_test = output_test[:, :, 0]

        columns_y = ids_columns + targets_columns + [f'{self.distrib_model.target_name}']
        
        y_train = y_train.reshape(y_train.shape[0], -1)
        y_val = y_val.reshape(y_val.shape[0], -1)
        y_test = y_test.reshape(y_test.shape[0], -1)
        
        output_train = output_train.reshape(output_train.shape[0], -1)
        output_val = output_val.reshape(output_val.shape[0], -1)
        output_test = output_test.reshape(output_test.shape[0], -1)
                
        df_class_train[columns_y] = y_train
        df_class_val[columns_y] = y_val
        df_class_test[columns_y] = y_test
        
        df_class_train['weight'] = 1
        
        columns_x = [f'fet_{i}' for i in range(output_train.shape[-1])]
        
        df_class_train[columns_x] = output_train
        df_class_val[columns_x] = output_val
        df_class_test[columns_x] = output_test
        
        self.class_model.features_name = columns_x
        
        self.class_model.create_train_val_test_loader(graph, df_class_train, df_class_val, df_class_test, epochs, PATIENCE_CNT, CHECKPOINT, 
                        features_importance=features_importance, custom_model_params=custom_model_params, use_log=use_log)
        
        
        self.metrics = self.class_model.model
        
        self.class_model.graph = graph
        
        self.class_model.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params=custom_model_params)
    
    def _predict_test_loader(self, loader=None, prediction_type='Class', output_pdf=None, calibrate=False):
        pred_distrib, y_distrib = self.distrib_model._predict_test_loader(loader, prediction_type='RawFormulaVal', output_pdf=output_pdf, calibrate=calibrate)

        y_distrib = y_distrib[:, :, 0]
        pred_distrib = pred_distrib[:, :, 0]
        
        y_distrib = y_distrib.reshape(y_distrib.shape[0], -1)
        pred_distrib = pred_distrib.reshape(pred_distrib.shape[0], -1)
        
        df_class = pd.DataFrame(index=np.arange(0, y_distrib.shape[0]))
        
        columns_y = ids_columns + [f'{self.distrib_model.target_name}']
        columns_x = [f'fet_{i}' for i in pred_distrib.shape[-1]]
        
        df_class[columns_y] = y_distrib
        df_class[columns_x] = pred_distrib
        
        loader = self.class_model.create_test_loader(self.class_model.graph, df_class)
        
        res, y = self.class_model._predict_test_loader(loader, prediction_type, output_pdf, calibrate)
        
        return res, y
    
    def create_test_loader(self, graph, df):
        return self.distrib_model.create_test_loader(graph, df)
    
    def score(self, X, y, sample_weight=None):
        pred_distrib, y_distrib = self.distrib_model.predict(X, return_y=True)
        
        y_distrib = y_distrib[:, :, 0]
        pred_distrib = pred_distrib[:, :, 0]
        
        y_distrib = y_distrib.reshape(y_distrib.shape[0], -1)
        pred_distrib = pred_distrib.reshape(pred_distrib.shape[0], -1)
        
        df_class = pd.DataFrame(index=np.arange(0, y_distrib.shape[0]))
        
        columns_y = ids_columns + [f'{self.distrib_model.target_name}']
        columns_x = [f'fet_{i}' for i in pred_distrib.shape[-1]]
        
        df_class[columns_y] = y_distrib
        df_class[columns_x] = pred_distrib
        
        predictions, y = self.class_model.predict(df_class, return_y=True)
        predictions = predictions[:, 0]
        y = y[:, -1, 0]
        
        return self.class_model.score_with_prediction(predictions, y, sample_weight)