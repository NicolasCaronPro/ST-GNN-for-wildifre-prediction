from asyncio import tasks
from datetime import date
from genericpath import isfile
from threading import local
from cv2 import threshold
from numpy import dtype
from sympy import use
from torch_geometric.data import Dataset
from torch.utils.data import DataLoader
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
from GNN.discretization import *
from feature_engine.selection import SmartCorrelatedSelection

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

class ReadGraphDataset_hybrid(Dataset):
    def __init__(self, X_2D : list,
                 X_1D : list,
                 Y : list,
                 edges : list,
                 leni : int,
                 device : torch.device,
                 path : Path) -> None:
        
        self.X_2D = X_2D
        self.X_1D = X_1D
        self.Y = Y
        self.device = device
        self.edges = edges
        self.leni = leni
        self.path = path

    def __getitem__(self, index) -> tuple:

        x_2D = read_object(self.X_2D[index], self.path)
        x_1D = self.X_1D[index]
        #y = read_object(self.Y[index], self.path)
        y = self.Y[index]

        if len(self.edges) > 0:
            edges = self.edges[index]
        else:
            edges = None

        return torch.tensor(x_1D, dtype=torch.float32, device=self.device), \
            torch.tensor(x_2D, dtype=torch.float32, device=self.device), \
            torch.tensor(y, dtype=torch.float32, device=self.device), \
            torch.tensor(edges, dtype=torch.long, device=self.device)
        
    def __len__(self) -> int:
        return self.leni
    
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

#################################### NUMPY #############################################################
def get_train_val_test_set(graphScale, df, features_name, train_departements, prefix, dir_output, METHODS, args):
    # Input config
    maxDate = args.maxDate
    trainDate = args.trainDate
    doFet = args.featuresSelection == "True"
    nbfeatures = args.NbFeatures
    values_per_class = args.nbpoint
    scale = int(args.scale) if args.scale != 'departement' else args.scale
    dataset_name= args.dataset
    doPCA = args.pca == 'True'
    doKMEANS = args.KMEANS == 'True'
    ncluster = int(args.ncluster)
    shift = int(args.shift)
    k_days = int(args.k_days) # Size of the time series sequence use by DL models
    days_in_futur = int(args.days_in_futur) # The target time validation
    scaling = args.scaling
    name_exp = args.name
    sinister = args.sinister
    top_cluster = args.top_cluster
    graph_method = args.graph_method
    thresh_kmeans = args.thresh_kmeans

    ######################## Get departments and train departments #######################

    departements, train_departements = select_departments(dataset_name, sinister)

    if dataset_name == 'firemen2':
        dataset_name = 'firemen'

    save_object(features_name, 'features_name.pkl', dir_output)

    _, train_features, kmeans_features = get_features_for_sinister_prediction(dataset_name, sinister, name_exp == 'inference')
    features_selected_kmeans,_ = get_features_name_list(scale, kmeans_features, METHODS_KMEANS_TRAIN)
    
    prefix = f'full_{k_days}_{nbfeatures}_{scale}_{days_in_futur}_{graphScale.base}_{graphScale.graph_method}'

    if doKMEANS:
        prefix += f'_kmeans_{shift}_{thresh_kmeans}'

    if k_days > 0:
        new_fet_time = get_time_columns(varying_time_variables, k_days, train_features, features_name)

        ############################## Define target for experiment #############################
        features_name += new_fet_time
        
    if days_in_futur > 0:
        target_name_nbsinister = f'nbsinister_sum_'
        target_name_risk = 'risk_max_'
        target_name_class = 'class_risk_max_'

        if doKMEANS:
            target_name_nbsinister += f'{shift}_{thresh_kmeans}'
            target_name_risk += f'{shift}_{thresh_kmeans}'
            target_name_class += f'{shift}_{thresh_kmeans}'
        else:
            target_name_nbsinister += f'0_0'
            target_name_risk += f'0_0'
            target_name_class += f'0_0'
            weights_name_columns = [f'{wc}_0_0' for wc in weights_columns]

        target_name_nbsinister += f'_+{days_in_futur}'
        target_name_risk += f'_+{days_in_futur}'
        target_name_class += f'_+{days_in_futur}'

    else:
        if doKMEANS:
            target_name_nbsinister = f'nbsinister_{shift}_{thresh_kmeans}'
            target_name_risk = f'risk_{shift}_{thresh_kmeans}'
            target_name_class = f'class_risk_{shift}_{thresh_kmeans}'
        else:
            target_name_nbsinister = f'nbsinister_0_0'
            target_name_risk = f'risk_0_0'
            target_name_class = f'class_risk_0_0'
    
    weights_name_columns = ['weight']

    #####################################################################################################################

    df['nbsinister'] = df[target_name_nbsinister]
    df['risk'] = df[target_name_risk]
    df['class_risk'] = df[target_name_class]

    logger.info(f'Unique sinister -> {df["nbsinister"].unique()}')

    columns = np.unique(ids_columns + weights_name_columns + list(np.unique(list(features_selected_kmeans) + list(features_name))) + targets_columns + [col for col in df.columns if col.startswith('class_window')])
    df = df[columns]

    # Preprocess
    train_dataset, val_dataset, test_dataset, train_dataset_unscale, val_dataset_unscale, test_dataset_unscale = preprocess(df=df, scaling=scaling, maxDate=maxDate,
                                                    trainDate=trainDate, train_departements=train_departements,
                                                    departements = departements,
                                                    ks=k_days, dir_output=dir_output, prefix=prefix, features_name=features_name,
                                                    days_in_futur=days_in_futur,
                                                    futur_met=futur_met, ncluster=ncluster, graph=graphScale,
                                                    args=args)

    ############################################### APPLY PCA ################################
    if doPCA:
        prefix += '_pca'
        x_train = train_dataset[0]
        pca, components = train_pca(x_train, 0.99, dir_output / prefix, features_name)
        train_dataset = apply_pca(train_dataset[0], pca, components, features_name)
        val_dataset = apply_pca(val_dataset[0], pca, components, features_name)
        test_dataset = apply_pca(test_dataset[0], pca, components, features_name)
        train_features = ['pca_'+str(i) for i in range(components)]
        features_name, _ = get_features_name_list(0, train_features, METHODS_SPATIAL_TRAIN)
        
    ############################################### FEATURES SELECTION/ORDER #########################################

    features_importance_order = features_selection(doFet, train_dataset, dir_output / 'features_importance' / f'{values_per_class}_{k_days}_{scale}_{days_in_futur}_{graphScale.base}_{graphScale.graph_method}', features_name, len(features_name), 'nbsinister')
    
    if nbfeatures != 'all':
        features_name = features_importance_order[:, nbfeatures]
    else:
        features_name = features_importance_order

    ############################################### PLOTTING ####################################################
        
    realVspredict(test_dataset['risk'].values, test_dataset[ids_columns + targets_columns].values, -1, dir_output / prefix, 'raw')

    realVspredict(test_dataset['class_risk'].values, test_dataset[ids_columns + targets_columns].values, -3, dir_output / prefix, 'class')

    realVspredict(test_dataset['nbsinister'].values, test_dataset[ids_columns + targets_columns].values, -2, dir_output / prefix, 'nbsinister')
        
    logger.info(f'Train dates are between : {allDates[int(np.min(train_dataset["date"]))], allDates[int(np.max(train_dataset["date"]))]}')
    logger.info(f'Val dates are bewteen : {allDates[int(np.min(val_dataset["date"]))], allDates[int(np.max(val_dataset["date"]))]}')

    logger.info(f'Nb sinister in train set : {train_dataset["nbsinister"].sum()}')
    logger.info(f'Nb sinister in val set : {val_dataset["nbsinister"].sum()}')
    logger.info(f'Nb sinister in test set : {test_dataset["nbsinister"].sum()}')

    return train_dataset, val_dataset, test_dataset, train_dataset_unscale, val_dataset_unscale, test_dataset_unscale, prefix, features_name

#########################################################################################################
#                                                                                                       #
#                                         Preprocess                                                    #
#                                                                                                       #
#########################################################################################################

def preprocess(df: pd.DataFrame, scaling: str, maxDate: str, trainDate: str, train_departements: list, departements: list, ks: int,
               dir_output: Path, prefix: str, features_name: list, days_in_futur: int, futur_met: str, ncluster: int, graph,
               args : dict,
               save = True):
    
    scale = graph.scale

    global features

    old_shape = df.shape
    df = remove_nan_nodes(df, features_name)
    logger.info(f'Removing nan Features DataFrame shape : {old_shape} -> {df.shape}')

    old_shape = df.shape
    df = remove_none_target(df)
    logger.info(f'Removing nan Target DataFrame shape : {old_shape} -> {df.shape}')

    old_shape = df.shape
    df = remove_bad_period(df, PERIODES_A_IGNORER, departements, ks)
    logger.info(f'Removing bad period DataFrame shape : {old_shape} -> {df.shape}')

    if args.sinister == 'firepoint':
        old_shape = df.shape
        df = remove_non_fire_season(df, SAISON_FEUX, departements, ks)
        logger.info(f'Removing non fire season DataFrame shape : {old_shape} -> {df.shape}')
    
    assert df.shape[0] > 0
    
    # Sort by date
    df = df.sort_values('date')
    trainCode = [name2int[departement] for departement in train_departements]

    train_mask = (df['date'] < allDates.index(trainDate)) & (df['departement'].isin(trainCode))
    val_mask = (df['date'] >= allDates.index(trainDate) + ks) & (df['date'] < allDates.index(maxDate)) & (df['departement'].isin(trainCode))
    test_mask = ((df['date'] >= allDates.index(maxDate) + ks) & (df['departement'].isin(trainCode))) | (~df['departement'].isin(trainCode))

    train_dataset_unscale = df[train_mask].reset_index(drop=True).copy(deep=True)
    test_dataset_unscale = df[test_mask].reset_index(drop=True).copy(deep=True)
    val_dataset_unscale = df[val_mask].reset_index(drop=True).copy(deep=True)

    if save:
        save_object(df[train_mask], 'df_unscaled_train_'+prefix+'.pkl', dir_output)

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
        """for feature in features_name:
            if features_name.index(feature) == autoregressiveBand:
                logger.info(f'Avoid scale for Autoregressive feature')
                continue
            df[feature] = scaler(df[feature].values, df[train_mask][feature].values, concat=False)"""
        
        df[features_name] = scaler(df[features_name].values, df[train_mask][features_name].values, concat=False)

    logger.info(f'Max of the train set: {df[train_mask][features_name].max().max(), df[train_mask][features_name].max().idxmax()}, before scale : {df_no_scale[train_mask][features_name].max().max(), df_no_scale[train_mask][features_name].max().idxmax()}')
    logger.info(f'Max of the val set: {df[val_mask][features_name].max().max(), df[val_mask][features_name].max().idxmax()}, before scale : {df_no_scale[val_mask][features_name].max().max(), df_no_scale[val_mask][features_name].max().idxmax()}')
    logger.info(f'Max of the test set: {df[test_mask][features_name].max().max(), df[test_mask][features_name].max().idxmax()}, before scale : {df_no_scale[test_mask][features_name].max().max(), df_no_scale[test_mask][features_name].max().idxmax()}')
    logger.info(f'Min of the train set: {df[train_mask][features_name].min().min(), df[train_mask][features_name].min().idxmin()}, before scale : {df_no_scale[train_mask][features_name].min().min(), df_no_scale[train_mask][features_name].min().idxmin()}')
    logger.info(f'Min of the val set: {df[val_mask][features_name].max().min(), df[val_mask][features_name].max().idxmin()}, before scale : {df_no_scale[val_mask][features_name].min().min(), df_no_scale[val_mask][features_name].min().idxmin()}')
    logger.info(f'Min of the test set: {df[test_mask][features_name].min().min(), df[test_mask][features_name].min().idxmin()}, before scale : {df_no_scale[test_mask][features_name].min().min(), df_no_scale[test_mask][features_name].min().idxmin()}')
    logger.info(f'Size of the train set: {df[train_mask].shape[0]}, size of the val set: {df[val_mask].shape[0]}, size of the test set: {df[test_mask].shape[0]}')
    logger.info(f'Unique train dates: {np.unique(df[train_mask]["date"]).shape[0]}, val dates: {np.unique(df[val_mask]["date"]).shape[0]}, test dates: {np.unique(df[test_mask]["date"]).shape[0]}')
    
    logger.info(f'Train mask {df[train_mask].shape}')
    #logger.info(f'Train mask with weight > 0 {df[train_mask][df[train_mask]["weight_one"] > 0].shape}')
    logger.info(f'Val mask {df[val_mask].shape}')
    logger.info(f'Test mask {df[test_mask].shape}')

    logger.info(f'Unique train departments: {np.unique(df[train_mask]["departement"])}')
    logger.info(f'Unique val departments: {np.unique(df[val_mask]["departement"])}')
    logger.info(f'Unique test departments: {np.unique(df[test_mask]["departement"])}')

    if save:
        save_object(df[train_mask], 'df_train_'+prefix+'.pkl', dir_output)

    return df[train_mask].reset_index(drop=True), df[val_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True), train_dataset_unscale, val_dataset_unscale, test_dataset_unscale

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
                    use_temporal_as_edges : bool,
                    device,
                    ks : int):
    
    x_train, y_train = df_train[ids_columns + features_name].values, df_train[ids_columns + targets_columns].values
    
    x_val, y_val = df_val[ids_columns + features_name].values, df_val[ids_columns + targets_columns].values

    x_test, y_test = df_test[ids_columns + features_name].values, df_test[ids_columns + targets_columns].values

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

def load_x_from_pickle(date : int,
                       path : Path,
                       df_train : pd.DataFrame,
                       scaling : str,
                       features_name_2D : list,
                       features : list) -> np.array:
    
    features_name_2D_full, _ = get_features_name_lists_2D(6, features)
    leni = len(features_name_2D)
    x_2D = read_object(f'X_{date}.pkl', path)
    if x_2D is None:
        return None
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
        if fet_2D in calendar_variables or fet_2D in geo_variables or fet_2D in air_variables or \
            fet_2D.find('AutoRegression') != -1 or fet_2D in region_variables or fet_2D in historical_variables or fet_2D in cluster_encoder:
            var_1D = fet_2D
        else:
            var_1D = f'{fet_2D}_mean'
        new_x_2D[i, :, :] = x_2D[features_name_2D_full.index(fet_2D), :, :]
        if False not in np.isnan(new_x_2D[i, :, :]):
            return None
        else:
            nan_mask = np.isnan(new_x_2D[i, :, :])
            new_x_2D[i, nan_mask] = np.nanmean(new_x_2D[i, :, :])

        new_x_2D[i, :, :] = scaler(x_2D[features_name_2D_full.index(fet_2D), :, :].reshape(-1,1), df_train[var_1D].values, concat=False).reshape((x_2D.shape[1], x_2D.shape[2]))

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

def process_time_step(X, Y, dept, graph, ks, id, path, df_train, scaling, features_name_2D, features, y, raster_dept, raster_dept_graph, unodes, image_per_node, shape2D):
    """Process each time step and update X and Y arrays"""
    for k in range(ks + 1):
        date = int(id) - (ks - k)
        x_date = load_x_from_pickle(date, path / f'2D_database_{graph.scale}_{graph.base}_{graph.graph_method}' / int2name[dept], df_train, scaling, features_name_2D, features)
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

def create_dataset_2D_2(graph, X_np, Y_np, ks, dates, df_train, scaling,
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
            X, Y = process_time_step(X, Y, dept, graph, ks, id, path, df_train, scaling, features_name_2D, features, y[y[:, departement_index, 0] == dept], raster_dept, raster_dept_graph, unodes, image_per_node, shape2D)

            if X is None:
                continue

            sub_dir = 'image_per_node' if image_per_node else 'image_per_departement'

            if use_temporal_as_edges is None and image_per_node:
                for i in range(X.shape[0]):
                    if True:
                        save_object(X[i], f'X_{int(id)}_{dept}_{i}.pkl', path / f'2D_database_{graph.scale}_{graph.base}_{graph.graph_method}' / sub_dir / scaling / context)
                        Xst.append(f'X_{int(id)}_{dept}_{i}.pkl')
                        Yst.append(Y[i])
                    else:
                        Xst.append(X[i])
                        Yst.append(y[i])
            else:
                if True:
                    save_object(X, f'X_{int(id)}_{dept}.pkl', path / f'2D_database_{graph.scale}_{graph.base}_{graph.graph_method}' / sub_dir / scaling / context)
                    Xst.append(f'X_{int(id)}_{dept}.pkl')
                else:
                    Xst.append(X)
                Y[np.isnan(Y)] = 0
                Yst.append(Y)

        if 'e' in locals():
            Est.append(e)

    return Xst, Yst, Est

def create_dataset_hybrid(graph,
                    df_train,
                    df_val,
                    df_test,
                    scaling,
                    path,
                    features_name_2D,
                    features_name,
                    features,
                    image_per_node,
                    use_temporal_as_edges : bool,
                    scale,
                    device,
                    ks : int):
    
    x_train, y_train = df_train[ids_columns + features_name].values, df_train[ids_columns + targets_columns].values

    x_val, y_val = df_val[ids_columns + features_name].values, df_val[ids_columns + targets_columns].values

    x_test, y_test = df_test[ids_columns + features_name].values, df_test[ids_columns + targets_columns].values

    dateTrain = np.sort(np.unique(y_train[np.argwhere(y_train[:, weight_index] > 0), date_index]))
    dateVal = np.sort(np.unique(y_val[np.argwhere(y_val[:, weight_index] > 0), date_index]))
    dateTest = np.sort(np.unique(y_test[np.argwhere(y_test[:, weight_index] > 0), date_index]))
    
    logger.info(f'Model configuration : image_per_node {image_per_node}, use_temporal_as_edges {use_temporal_as_edges}')
 
    logger.info('Creating train dataset 2D')
    Xst_2D, _, _ = create_dataset_2D_2(graph, x_train, y_train, ks, dateTrain, df_train, scaling,
                        features_name_2D, features, path, use_temporal_as_edges, image_per_node, context='train') 
    
    logger.info('Creating val dataset 2D')
    # Val
    XsV_2D, _, _ = create_dataset_2D_2(graph, x_val, y_val, ks, dateVal, df_train, scaling,
                        features_name_2D, features, path, use_temporal_as_edges, image_per_node, context='val') 

    logger.info('Creating Test dataset 2D')
    # Test
    XsTe_2D, _, _ = create_dataset_2D_2(graph, x_test, y_test, ks, dateTest, df_train, scaling,
                    features_name_2D, features, path, use_temporal_as_edges, image_per_node, context='test')

    logger.info('Creating train dataset 1D')
    # Train
    train_dataset, val_dataset, test_dataset = create_dataset(graph,
                                                            df_train,
                                                            df_val,
                                                            df_test,
                                                            features_name,
                                                            use_temporal_as_edges,
                                                            device, ks)
    Xst = train_dataset.X
    Yst = train_dataset.Y
    Est = train_dataset.edges
    
    logger.info('Creating val dataset 1D')
    # Val
    XsV = val_dataset.X
    YsV = val_dataset.Y
    EsV = val_dataset.edges

    logger.info('Creating Test dataset 1D')
    # Test
    XsTe = test_dataset.X
    YsTe = test_dataset.Y
    EsTe = test_dataset.edges
    
    assert len(Xst) > 0
    assert len(XsV) > 0
    assert len(XsTe) > 0

    sub_dir = 'image_per_node' if image_per_node else 'image_per_departement'

    if is_pc:
        train_dataset = ReadGraphDataset_hybrid(Xst_2D, Xst, Yst, Est, len(Xst), device, path / f'2D_database_{graph.scale}_{graph.base}_{graph.graph_method}' / sub_dir / scaling / 'train')
        val_dataset = ReadGraphDataset_hybrid(XsV_2D, XsV, YsV, EsV, len(XsV), device, path / f'2D_database_{graph.scale}_{graph.base}_{graph.graph_method}' / sub_dir / scaling / 'val')
        testDatset = ReadGraphDataset_hybrid(XsTe_2D, XsTe, YsTe, EsTe, len(XsTe), device, path / f'2D_database_{graph.scale}_{graph.base}_{graph.graph_method}' / sub_dir / scaling / 'test')

    return train_dataset, val_dataset, testDatset

def create_dataset_2D(graph,
                    df_train,
                    df_val,
                    df_test,
                    scaling,
                    path,
                    features_name_2D,
                    features_name,
                    features,
                    image_per_node,
                    use_temporal_as_edges : bool,
                    device,
                    ks : int):
    
    x_train, y_train = df_train[ids_columns + features_name].values, df_train[ids_columns + targets_columns].values

    x_val, y_val = df_val[ids_columns + features_name].values, df_val[ids_columns + targets_columns].values

    x_test, y_test = df_test[ids_columns + features_name].values, df_test[ids_columns + targets_columns].values

    dateTrain = np.sort(np.unique(y_train[np.argwhere(y_train[:, weight_index] > 0), date_index]))
    dateVal = np.sort(np.unique(y_val[np.argwhere(y_val[:, weight_index] > 0), date_index]))
    dateTest = np.sort(np.unique(y_test[np.argwhere(y_test[:, weight_index] > 0), date_index]))

    XsTe = []
    YsTe = []
    EsTe = []

    logger.info(f'Model configuration : image_per_node {image_per_node}, use_temporal_as_edges {use_temporal_as_edges}')

    logger.info('Creating train dataset')
    Xst, Yst, Est = create_dataset_2D_2(graph, x_train, y_train, ks, dateTrain, df_train, scaling,
                        features_name_2D, features, path, use_temporal_as_edges, image_per_node, context='train') 
    
    logger.info('Creating val dataset')
    # Val
    XsV, YsV, EsV = create_dataset_2D_2(graph, x_val, y_val, ks, dateVal, df_train, scaling,
                        features_name_2D, features, path, use_temporal_as_edges, image_per_node, context='val') 

    logger.info('Creating Test dataset')
    # Test
    XsTe, YsTe, EsTe = create_dataset_2D_2(graph, x_test, y_test, ks, dateTest, df_train, scaling,
                    features_name_2D, features, path, use_temporal_as_edges, image_per_node, context='test') 

    
    assert len(Xst) > 0
    assert len(XsV) > 0
    assert len(XsTe) > 0

    print(len(Xst), len(XsV), len(XsTe))

    sub_dir = 'image_per_node' if image_per_node else 'image_per_departement'
    if True:
        train_dataset = ReadGraphDataset_2D(Xst, Yst, Est, len(Xst), device, path / f'2D_database_{graph.scale}_{graph.base}_{graph.graph_method}' / sub_dir / scaling / 'train')
        val_dataset = ReadGraphDataset_2D(XsV, YsV, EsV, len(XsV), device, path / f'2D_database_{graph.scale}_{graph.base}_{graph.graph_method}' / sub_dir / scaling / 'val')
        test_datset = ReadGraphDataset_2D(XsTe, YsTe, EsTe, len(XsTe), device, path / f'2D_database_{graph.scale}_{graph.base}_{graph.graph_method}' / sub_dir /  scaling / 'test')
    else:
        train_dataset = InplaceGraphDataset(Xst, Yst, Est, len(Xst), device)
        val_dataset = InplaceGraphDataset(XsV, YsV, EsV, len(XsV), device)
        test_datset = InplaceGraphDataset(XsTe, YsTe, EsTe, len(XsTe), device)
    return train_dataset, val_dataset, test_datset

def train_val_data_loader(graph,
                          df_train,
                          df_val,
                          df_test,
                          features_name,
                        use_temporal_as_edges : bool,
                        batch_size,
                        device,
                        ks : int):

    train_dataset, val_dataset, test_dataset = create_dataset(graph,
                                                            df_train,
                                                            df_val,
                                                            df_test,
                                                            features_name,
                                                            use_temporal_as_edges,
                                                            device, ks)
    
    if use_temporal_as_edges is None:
        train_loader = DataLoader(train_dataset, batch_size, True)
        val_loader = DataLoader(val_dataset, val_dataset.__len__(), False)
        test_loader = DataLoader(test_dataset, test_dataset.__len__(), False)
    else:
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
                            features : list,
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
                                                                features,
                                                                image_per_node,
                                                                use_temporal_as_edges,
                                                                device,
                                                                ks)
    
    if use_temporal_as_edges is None:
        train_loader = DataLoader(train_dataset, batch_size, True)
        val_loader = DataLoader(val_dataset, val_dataset.__len__(), False)
        test_loader = DataLoader(test_dataset, test_dataset.__len__(), False)
    else:
        train_loader = DataLoader(train_dataset, batch_size, True, collate_fn=graph_collate_fn)
        val_loader = DataLoader(val_dataset, val_dataset.__len__(), False, collate_fn=graph_collate_fn)
        test_loader = DataLoader(test_dataset, test_dataset.__len__(), False, collate_fn=graph_collate_fn)
    return train_loader, val_loader, test_loader

def train_val_data_loader_hybrid(graph,
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
                            features,
                            scale,
                            ks : int,
                            path : Path):
    
    train_dataset, val_dataset, test_dataset = create_dataset_hybrid(graph,
                                                                df_train,
                                                                df_val,
                                                                df_test,
                                                                scaling,
                                                                path,
                                                                features_name_2D,
                                                                features_name,
                                                                features,
                                                                image_per_node,
                                                                use_temporal_as_edges,
                                                                scale,
                                                                device,
                                                                ks)
    if use_temporal_as_edges is None:
        train_loader = DataLoader(train_dataset, batch_size, True)
        val_loader = DataLoader(val_dataset, val_dataset.__len__(), False)
        test_loader = DataLoader(test_dataset, test_dataset.__len__(), False)
    else:
        train_loader = DataLoader(train_dataset, batch_size, True, collate_fn=graph_collate_fn_hybrid)
        val_loader = DataLoader(val_dataset, val_dataset.__len__(), False, collate_fn=graph_collate_fn_hybrid)
        test_loader = DataLoader(test_dataset, test_dataset.__len__(), False, collate_fn=graph_collate_fn_hybrid)
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

    graphId = np.unique(Xset[:,date_index])

    batch = []

    i = 0
    for id in graphId:
        if use_temporal_as_edges is None:
            x, y = construct_time_series(graph, id, Xset, Yset, ks, len(ids_columns))
            batch.append([torch.tensor(x, dtype=torch.float32, device=device),
                            torch.tensor(y, dtype=torch.float32, device=device),
                            None])
        elif use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, id, Xset, Yset, ks, len(ids_columns))
        else:
            x, y, e = construct_graph_with_time_series(graph, id, Xset, Yset, ks, len(ids_columns))
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
                features_name : list,
                features : list,
                ks : int,
                scale : int,
                path : Path,
                test_name: str,
                ):
    """
    """

    x_test, y_test = df[ids_columns + features_name].values, df[ids_columns + targets_columns].values

    XsTe, YsTe, EsTe = create_dataset_2D_2(graph, x_test, y_test, ks, np.unique(x_test[:,date_index]), df_train, scaling,
                    features_name_2D, features, path, use_temporal_as_edges, image_per_node, test_name)
    
    if True:
        sub_dir = 'image_per_node' if image_per_node else 'image_per_departement'
        dataset = ReadGraphDataset_2D(XsTe, YsTe, EsTe, len(XsTe), device, path / f'2D_database_{graph.scale}_{graph.base}_{graph.graph_method}' / sub_dir / scaling / test_name)
    else:
        dataset = InplaceGraphDataset(XsTe, YsTe, EsTe, len(XsTe), device)

    if use_temporal_as_edges is None:
        loader = DataLoader(dataset, 1, False)
    else:
        loader = DataLoader(dataset, 1, False, collate_fn=graph_collate_fn)

    return loader

def create_test_loader_hybrid(graph,
                df_test : pd.DataFrame,
                df_train : pd.DataFrame,
                use_temporal_as_edges : bool,
                image_per_node : bool,
                device: torch.device,
                features,
                scaling : str,
                features_name_2D : list,
                features_name : list,
                ks : int,
                scale : int,
                path : Path,
                test_name: str,
                ):
   
    x_test, y_test = df_test[ids_columns + features_name].values, df_test[ids_columns + targets_columns].values

    dateTest = np.unique(y_test[:,date_index])
    
    logger.info(f'Model configuration : image_per_node {image_per_node}, use_temporal_as_edges {use_temporal_as_edges}')
 
    logger.info('Creating Test dataset 2D')

    # Test
    XsTe_2D, _, _ = create_dataset_2D_2(graph, x_test, y_test, ks, dateTest, df_train, scaling,
                    features_name_2D, features, path, use_temporal_as_edges, image_per_node, scale, context=test_name)

    X = []
    Y = []
    E = []

    i = 0
    for id in dateTest:
        if use_temporal_as_edges is None:
            x, y = construct_time_series(id, x_test, x_test, ks, len(ids_columns))
            for i in range(x.shape[0]):
                X.append(torch.tensor(x[i], dtype=torch.float32, device=device))
                Y.append(torch.tensor(y[i], dtype=torch.float32, device=device))
                continue
        elif use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, id, x_test, x_test, ks, len(ids_columns))
        else:
            x, y, e = construct_graph_with_time_series(graph, id, x_test, y_test, ks, len(ids_columns))
        if x is None:
            i += 1
            continue

        X.append(torch.tensor(x, dtype=torch.float32, device=device))
        Y.append(torch.tensor(y, dtype=torch.float32, device=device))
        E.append(torch.tensor(e, dtype=torch.long, device=device))

    train_dataset = InplaceGraphDataset(X, Y, E, len(X), device)
    sub_dir = 'image_per_node' if image_per_node else 'image_per_departement'

    testDataset = ReadGraphDataset_hybrid(XsTe_2D, train_dataset.X, train_dataset.Y, train_dataset.edges, train_dataset.__len__(),
                                          device, path / f'2D_database_{graph.scale}_{graph.base}_{graph.graph_method}' / sub_dir / scaling / test_name)
    
    loader = DataLoader(testDataset, 64, False, collate_fn=graph_collate_fn_hybrid)

    return loader

def create_test_loader(graph, df,
                       features_name,
                       device : torch.device,
                       use_temporal_as_edges : bool,
                       ks :int):
    
    Xset, Yset = df[ids_columns + features_name].values, df[ids_columns + targets_columns].values

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
                x, y, e = construct_graph_set(graph, id, Xset, Yset, ks, len(ids_columns))
            else:
                x, y, e = construct_graph_with_time_series(graph, id, Xset, Yset, ks, len(ids_columns))

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

def load_tensor_test(use_temporal_as_edges, graphScale, dir_output,
                     X, Y, device, encoding, features_name,
                     k_days, test, scaling, prefix, Rewrite):

    if not Rewrite:
        XTensor = read_object('XTensor_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
        if XTensor is not None:
            XTensor = XTensor.to(device)
            YTensor = read_object('YTensor_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output).to(device)
            ETensor = read_object('ETensor_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output).to(device)
            return XTensor, YTensor, ETensor

    XTensor, YTensor, ETensor = create_test(graph=graphScale, Xset=X, Yset=Y, features_name=features_name,
                                device=device, use_temporal_as_edges=use_temporal_as_edges, ks=k_days)

    save_object(XTensor, 'XTensor_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
    save_object(YTensor, 'YTensor_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)
    save_object(ETensor, 'ETensor_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'.pkl', dir_output)

    return XTensor, YTensor, ETensor

def load_loader_test(use_temporal_as_edges, graphScale, dir_output,
                     df, device, k_days, features_name,
                     test, scaling, encoding, prefix, Rewrite):
    loader = None
    if Rewrite:
        loader = create_test_loader(graph=graphScale, df=df, device=device, features_name=features_name,
                                    use_temporal_as_edges=use_temporal_as_edges, ks=k_days)

        save_object(loader, f'test_loader_{test}_{prefix}_{scaling}_{encoding}_{use_temporal_as_edges}.pkl', dir_output)
        loader = read_object(f'test_loader_{test}_{prefix}_{scaling}_{encoding}_{use_temporal_as_edges}.pkl', dir_output)
    else:
        loader = read_object(f'test_loader_{test}_{prefix}_{scaling}_{encoding}_{use_temporal_as_edges}.pkl', dir_output)
    if loader is None:
        loader = create_test_loader(graph=graphScale, df=df, device=device, features_name=features_name,
                                    use_temporal_as_edges=use_temporal_as_edges, ks=k_days)

        save_object(loader, f'test_loader_{test}_{prefix}_{scaling}_{encoding}_{use_temporal_as_edges}.pkl', dir_output)

    #path = Path('/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/firemen/firepoint/2x2/train/occurence_test_geometry')
    #loader = read_object(f'test_loader_binary_0_100_10_risk-watershed_weight_one_z-score_Catboost_True.pkl', path)

    return loader

def load_loader_test_hybrid(use_temporal_as_edges, image_per_node, scale, graphScale, dir_output,
                     df, df_train, device,
                     k_days, test, features_name, features, scaling, encoding, prefix, Rewrite):
    loader = None
    if Rewrite:
        loader = create_test_loader_hybrid(graph=graphScale,
                df_test=df,
                df_train=df_train,
                use_temporal_as_edges=use_temporal_as_edges,
                image_per_node=image_per_node,
                device=device,
                features=features,
                scaling=scaling,
                features_name_2D=features_name[1],
                features_name=features_name[0],
                ks=k_days,
                scale=scale,
                path=dir_output,
                test_name=test)

        save_object(loader, f'test_loader_{test}_{prefix}_{scaling}_{encoding}_{use_temporal_as_edges}.pkl', dir_output)
        loader = read_object(f'test_loader_{test}_{prefix}_{scaling}_{encoding}_{use_temporal_as_edges}.pkl', dir_output)
    else:
        loader = read_object(f'test_loader_{test}_{prefix}_{scaling}_{encoding}_{use_temporal_as_edges}.pkl', dir_output)
    if loader is None:
        loader = create_test_loader_hybrid(graph=graphScale,
                df_test=df,
                df_train=df_train,
                use_temporal_as_edges=use_temporal_as_edges,
                image_per_node=image_per_node,
                device=device,
                features=features,
                scaling=scaling,
                features_name_2D=features_name[1],
                features_name=features_name[0],
                ks=k_days,
                scale=scale,
                path=dir_output,
                test_name=test)

        save_object(loader, f'test_loader_{test}_{prefix}_{scaling}_{encoding}_{use_temporal_as_edges}.pkl', dir_output)

    return loader

def load_loader_test_2D(use_temporal_as_edges, image_per_node, scale, graphScale, dir_output,
                     df, df_train, device,
                     k_days, test, features_name_2D, features_name, features, scaling, encoding, prefix, Rewrite):
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
                                    features_name,
                                    features,
                                    k_days,
                                    scale,
                                    dir_output,
                                    test)

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
                                    features_name,
                                    features,
                                    k_days,
                                    scale,
                                    dir_output,
                                    test)

        save_object(loader, 'loader_'+test+'_'+prefix+'_'+scaling+'_'+encoding+'_'+str(use_temporal_as_edges)+'_2D.pkl', dir_output)

    return loader

def evaluate_pipeline(dir_train, prefix, df_test, pred, y, graph, test_departement, target_name, name, dir_output,
                      pred_min=None, pred_max=None, departement_scale = False):
    metrics = {}

    dir_predictor = dir_train / 'influenceClustering'
    if departement_scale:
        scale = 'Departement'
    else:
        scale = graph.scale

    ##################################### Create daily class ###########################################
    """graph_structure = graph.base
    for nameDep in test_departement:
        mask = np.argwhere((y[:, departement_index] == name2int[nameDep]))
        if mask.shape[0] == 0:
            continue
        if target_name == 'risk' and (dir_predictor / f'{nameDep}Predictor{scale}_{graph_structure}_{graph.graph_method}.pkl').is_file() and (dir_predictor / f'{nameDep}PredictorDepartement.pkl').is_file():
            if departement_scale:
                predictor = read_object(f'{nameDep}PredictorDepartement.pkl', dir_predictor)
            else:
                predictor = read_object(f'{nameDep}Predictor{scale}_{graph_structure}_{graph.graph_method}.pkl', dir_predictor)
        else:
            create_predictor(pred, name, nameDep, dir_predictor, target_name, graph, departement_scale)
            if departement_scale:
                predictor = read_object(f'{nameDep}Predictor{name}Departement.pkl', dir_predictor)
            else:
                predictor = read_object(f'{nameDep}Predictor{name}{scale}_{graph_structure}_{graph.graph_method}.pkl', dir_predictor)

        if target_name != 'binary':
            pred[mask[:,0], 1] = order_class(predictor, predictor.predict(pred[mask[:, 0], 0]))
           
            if pred_min is not None:
                pred_min[mask[:,0], 1] = order_class(predictor, predictor.predict(pred_min[mask[:, 0], 0]))
            if pred_max is not None:
                pred_max[mask[:,0], 1] = order_class(predictor, predictor.predict(pred_max[mask[:, 0], 0]))
        else:
            pred[mask[:,0], 1] = predictor.predict(pred[mask[:, 0], 0])
            if pred_min is not None:
                pred_min[mask[:,0], 1] = predictor.predict(pred_min[mask[:, 0], 0])
            if pred_max is not None:
                pred_max[mask[:,0], 1] = predictor.predict(pred_max[mask[:, 0], 0])"""

    ############################################## Get daily metrics #######################################################
    col_class = 'nbsinister-MinMaxClass'
    col_nbsininster = 'nbsinister'

    metrics = {}

    logger.info(f'###################### Analysis {target_name} #########################')
 
    """realVspredict(pred[:, 0], y, -1,
                dir_output / name, f'raw_{scale}',
                pred_min, pred_max)
            
    realVspredict(pred[:, 1], y, -3,
                dir_output / name, f'class_{scale}',
                pred_min, pred_max)

    realVspredict(pred[:, 0], y, -2,
                dir_output / name, f'nbfire_{scale}',
                pred_min, pred_max)"""
    
    realVspredict(pred[:, 0], y, -2,
                dir_output / name, f'{target_name}_{scale}',
                pred_min, pred_max)
    
    plot_and_save_roc_curve(df_test['nbsinister'].values > 0, pred[:, 0], dir_output / name, target_name, departement_scale)
    plot_and_save_pr_curve(df_test['nbsinister'].values > 0, pred[:, 0], dir_output / name, target_name, departement_scale)
    
    calibrated_curve(pred[:, 0], y, dir_output / name, 'calibration')
    calibrated_curve(pred[:, 1], y, dir_output / name, 'class_calibration')

    shapiro_wilk(pred[:,0], y[:,-1], dir_output / name, f'shapiro_wilk_{scale}')
 
    res = pd.DataFrame(columns=ids_columns + targets_columns + ['prediction', 'class', 'mae', 'rmse'], index=np.arange(pred.shape[0]))
    res[ids_columns + targets_columns] = df_test[ids_columns + targets_columns]
    res[target_name] = df_test[target_name]
    res['mae'] = np.sqrt((y[:,-4] * (pred[:,0] - y[:,-1]) ** 2))
    res['rmse'] = np.sqrt(((pred[:,0] - y[:,-1]) ** 2))
    
    #res = graph.compute_window_class(res, [1], 5, column='prediction', target_name=target_name, aggregate_funcs=['sum', 'mean', 'max', 'min', 'grad', 'std'], mode='test', dir_output=dir_train / 'class_window' / prefix / target_name)
    #df_test = graph.compute_window_class(df_test, [1], 5, column='nbsinister', target_name='nbsinister', aggregate_funcs=['max'], mode='test', dir_output=dir_train / 'class_window' / prefix / 'nbsinister')

    #res_class_window_cols = [col for col in res.columns if col.startswith('class_window')]
    #res_window = res.groupby(['graph_id', 'date'])[res_class_window_cols + ['prediction', 'nbsinister']].mean().reset_index()
    #test_class_window_cols = [col for col in df_test.columns if col.startswith('window_nbsinister')]
    #test_window = df_test.groupby(['graph_id', 'date', 'departement'])[test_class_window_cols + ['nbsinister', 'risk']].mean().reset_index()
    
    #test_ids = set(zip(res_window['graph_id'], res_window['date']))
    #test_window = test_window[test_window[['graph_id', 'date']].apply(tuple, axis=1).isin(test_ids)]
    df_test['saison'] = df_test['date'].apply(get_saison)
    res['saison'] = res['date'].apply(get_saison)

    ####################################### Sinister Evaluation ########################################

    logger.info(f'###################### Analysis {col_nbsininster} #########################')

    y_pred = pred[:, 0]
    if pred_max is not None:
        y_pred_max = pred_max[:, 0]
        y_pred_min = pred_min[:, 0]

    y_true = df_test[col_nbsininster].values
    
    apr = round(average_precision_score(y_true > 0, y_pred), 2)
    df_test['prediction'] = y_pred

    ks, _ = calculate_ks_continous(df_test, 'prediction', col_nbsininster, dir_output / name)

    r2 = r2_score(y_true, y_pred)

    metrics[f'nbsinister'] = df_test['nbsinister'].sum()
    logger.info(f'Number of sinister = {df_test["nbsinister"].sum()}')

    metrics[f'nb_{col_nbsininster}'] = df_test[col_nbsininster].sum()
    logger.info(f'Number of {col_nbsininster} = {df_test[col_nbsininster].sum()}')

    metrics[f'apr_{col_nbsininster}'] = apr
    logger.info(f'apr = {apr}')

    metrics[f'r2_{col_nbsininster}'] = r2
    logger.info(f'r2 = {r2}')

    metrics[f'KS_{col_nbsininster}'] = ks
    logger.info(f'KS = {ks}')
    
    """mae = round(mean_absolute_error(y_true, y_pred), 2)
    mse = round(mean_squared_error(y_true, y_pred), 2)
    mae_fire = round(mean_absolute_error(y_true[y_true > 0], y_pred[y_true > 0]), 2)
    mse_fire = round(mean_squared_error(y_true[y_true > 0], y_pred[y_true > 0]), 2)

    metrics[f'mae_raw_{col_for_dict}'] = mae
    metrics[f'mse_raw_{col_for_dict}'] = mse
    metrics[f'mae_fire_raw_{col_for_dict}'] = mae_fire
    metrics[f'mse_fire_raw_{col_for_dict}'] = mse_fire

    logger.info(f'mae_raw = {mae}')
    logger.info(f'mse_raw = {mse}')
    logger.info(f'mae_fire_raw = {mae_fire}')
    logger.info(f'mse_fire_raw = {mse_fire}')"""

    # Calcul des scores pour les signaux
    #iou_dict = calculate_signal_scores(y_pred, y_true, test_window['graph_id'].values, test_window['saison'].values)

    # Sauvegarder toutes les métriques calculées dans le dictionnaire metrics
    #for key, value in iou_dict.items():
    #    metric_key = f'{key}_raw_{col_for_dict}'  # Ajouter un suffixe basé sur col_for_dict
    #    metrics[metric_key] = value  # Ajouter au dictionnaire des métriques
    #    logger.info(f'{metric_key} = {value}')  # Afficher la métrique enregistrée

    y_true_temp = np.ones((y_pred.shape[0], y.shape[1]))
    y_true_temp[:, graph_id_index] = df_test['graph_id']
    y_true_temp[:, id_index] = df_test['graph_id']
    y_true_temp[:, departement_index] = df_test['departement']
    y_true_temp[:, date_index] = df_test['date']
    y_true_temp[:, -1] = y_true
    y_true_temp[:, -2] = df_test['nbsinister']
    y_true_temp[:, -3] = df_test[col_nbsininster]

    realVspredict(y_pred, y_true_temp, -1,
        dir_output / name, col_nbsininster,
        pred_min, pred_max)
    
    res['prediction'] = pred[:, 0]
    
    logger.info(f'###################### Analysis {col_class} #########################')

    y_pred = pred[:, 1]
    y_true = df_test[col_class]
    if pred_max is not None:
        y_pred_max = pred_max[:, 1]
        y_pred_min = pred_min[:, 1]
    else:
        y_pred_max = None
        y_pred_min = None

    all_class = np.unique(np.concatenate((df_test[col_class].values, y_pred)))
    all_class_label = [str(c) for c in all_class]

    #metrics = add_metrics(methods, i, pred, y, test_departement, target_name, graph, name, dir_train)
    plot_custom_confusion_matrix(df_test[col_class].values, y_pred, all_class_label, dir_output=dir_output / name, figsize=(15,8), normalize='true', filename=f'{scale}_confusion_matrix')
    plot_custom_confusion_matrix(df_test[col_class].values, y_pred, all_class_label, dir_output=dir_output / name, figsize=(15,8), normalize='all', filename=f'{scale}_confusion_matrix')
    plot_custom_confusion_matrix(df_test[col_class].values, y_pred, all_class_label, dir_output=dir_output / name, figsize=(15,8), normalize='pred', filename=f'{scale}_confusion_matrix')
    plot_custom_confusion_matrix(df_test[col_class].values, y_pred, all_class_label, dir_output=dir_output / name, figsize=(15,8), normalize=None, filename=f'{scale}_confusion_matrix')

    #mae = round(mean_absolute_error(y_true, y_pred), 2)
    #mse = round(mean_squared_error(y_true, y_pred), 2)
    #mae_fire = round(mean_absolute_error(y_true[y_true > 0], y_pred[y_true > 0]), 2)
    #mse_fire = round(mean_squared_error(y_true[y_true > 0], y_pred[y_true > 0]), 2)
    accuracy = round(accuracy_score(y_true, y_pred), 2)

    metrics[f'accuracy_{col_class}'] = accuracy
    """metrics[f'mae_{col_for_dict}'] = mae
    metrics[f'mse_{col_for_dict}'] = mse
    metrics[f'mae_fire_{col_for_dict}'] = mae_fire
    metrics[f'mse_fire_{col_for_dict}'] = mse_fire"""

    """logger.info(f'mae_discretization_{method} = {mae}')
    logger.info(f'mse_discretization_{method} = {mse}')
    logger.info(f'mae_fire_discretization_{method} = {mae_fire}')
    logger.info(f'mse_fire_discretization_{method} = {mse_fire}')"""
    logger.info(f'accuracy = {accuracy}')

    # Calcul des scores pour les signaux
    iou_dict = calculate_signal_scores(y_pred, y_true, df_test['graph_id'].values, df_test['saison'].values)

    # Sauvegarder toutes les métriques calculées dans le dictionnaire metrics
    for key, value in iou_dict.items():
        metric_key = f'{key}_{col_class}'  # Ajouter un suffixe basé sur col_for_dict
        metrics[metric_key] = value  # Ajouter au dictionnaire des métriques
        logger.info(f'{metric_key} = {value}')  # Afficher la métrique enregistrée

    y_true_temp = np.ones((y_pred.shape[0], y.shape[1]))
    y_true_temp[:, graph_id_index] = df_test['graph_id']
    y_true_temp[:, id_index] = df_test['graph_id']
    y_true_temp[:, departement_index] = df_test['departement']
    y_true_temp[:, date_index] = df_test['date']
    y_true_temp[:, -1] = y_true
    y_true_temp[:, -2] = df_test['nbsinister']
    y_true_temp[:, -3] = df_test[col_class]

    realVspredict(y_pred, y_true_temp, -1,
        dir_output / name, f'{col_class}',
        y_pred_min, y_pred_max)

    res[f'prediction_{col_class}'] = y_pred

    iou_vis(y_pred, y_true_temp, -1, dir_output / name, f'{col_class}')

    return metrics, res

def test_fire_index_model(args,
                           graphScale,
                          test_dataset_dept,                                                            
                          test_dataset_unscale_dept,
                           methods,
                           test_name,
                           prefix_train,
                           models,
                           dir_output,
                           encoding,
                           scaling,
                           test_departement,
                           dir_train): 
    res_scale = []
    res_departement = []
    
    scale = graphScale.scale
    metrics = {}
    metrics_dept = {}

    """################################ Ground Truth ############################################
    logger.info('#########################')
    logger.info(f'       GT              ')
    logger.info('#########################')

    y = test_dataset_dept[ids_columns + targets_columns].values
    pred = np.empty((y.shape[0], 2))
    pred[:, 0] = test_dataset_dept['risk'].values
    pred[:, 1] = test_dataset_dept['class_risk'].values

    if MLFLOW:
        existing_run = get_existing_run(f'{test_name}_GT_{prefix_train}')
        if existing_run:
            mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
        else:
            mlflow.start_run(run_name=f'{test_name}_GT_{prefix_train}', nested=True)

    metrics['GT'], res = evaluate_pipeline(dir_train, prefix_train, test_dataset_dept, pred, y, graphScale,
                                               methods, 0, test_departement, 'risk', 'GT', test_name,
                                               dir_output)
    i = 0
    if scale != 'departement':
            
        # Analyse departement prediction
        res_dept = res.groupby(['departement', 'date'])['risk'].sum().reset_index()
        res_dept['id'] = res_dept['departement']
        res_dept['graph_id'] = res_dept['departement']
        res_dept['latitude'] = res.groupby(['departement'])['latitude'].mean().reset_index()['latitude']
        res_dept['longitude'] = res.groupby(['departement'])['longitude'].mean().reset_index()['longitude']
        res_dept['nbsinister'] = res.groupby(['departement', 'date'])['nbsinister'].sum().reset_index()['nbsinister']
        res_dept['days_until_next_event'] = calculate_days_until_next_event(res_dept['id'].values, res_dept['date'].values, res_dept['nbsinister'].values)

        dir_predictor = dir_train / 'influenceClustering'
        for nameDep in test_departement:
            mask = np.argwhere(y[:, departement_index] == name2int[nameDep])
            if mask.shape[0] == 0:
                continue
            predictor = read_object(nameDep+'PredictorDepartement.pkl', dir_predictor)

        res_dept['class_risk'] = order_class(predictor, predictor.predict(res_dept['risk'].values, 0))
        res_dept['weight'] =  predictor.weight_array(predictor.predict(res_dept['risk'].values, 0).astype(int)).reshape(-1)

        res_dept['prediction'] = res.groupby(['departement', 'date']).sum().reset_index()['prediction']

        metrics_dept[f'GT_dept'], res_dept = evaluate_pipeline(dir_train, prefix_train, test_dataset_dept, res_dept[['prediction', 'class_risk']].values,
                                                                res_dept[ids_columns + targets_columns].values,
                                                                graphScale,
                                                                methods, i, test_departement, 'risk', 'GT', test_name,
                                                                dir_output)

        if MLFLOW:
            log_metrics_recursively(metrics_dept['GT_dept'], prefix='departement')
        
    if MLFLOW:
        log_metrics_recursively(metrics['GT'], prefix='')
        mlflow.log_params(args)
        mlflow.end_run()"""
    i = 1

    #################################### Traditionnal ##################################################

    for name, target_name in models:

        y = test_dataset_dept[ids_columns + targets_columns].values
        print(np.unique(y[:, graph_id_index]))
        logger.info('#########################')
        logger.info(f'      {name}            ')
        logger.info('#########################')

        if MLFLOW:
            existing_run = get_existing_run(f'{test_name}_{name}_{prefix_train}')
            if existing_run:
                mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
            else:
                mlflow.start_run(run_name=f'{test_name}_{name}_{prefix_train}', nested=True)

        features_selected = [name]
        pred = np.empty((y.shape[0], 2))
        if name == 'zero':
            pred[:, 0] = 0
        elif name == 'one':
            pred[:, 0] = 1
        else:
            pred[:, 0] = test_dataset_unscale_dept[name]

        if MLFLOW:
            mlflow.set_tag(f"Training", f"{name}")
            mlflow.log_param(f'relevant_feature_name', features_selected)
            mlflow.log_params(args)

        logger.info(f'pred min {np.nanmin(pred[:, 0])}, pred max : {np.nanmax(pred[:, 0])}')

        metrics[name], res = evaluate_pipeline(dir_train, prefix_train, test_dataset_dept, pred, y, graphScale,
                                               test_departement, target_name, name,
                                               dir_output)
        
        if MLFLOW:
            log_metrics_recursively(metrics[name], prefix='')
        
        save_object(res, name+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+test_name+'_pred.pkl', dir_output / name)

        if scale == 'departement':
            continue

        # Analyse departement prediction
        res_dept = res.groupby(['departement', 'date'])['risk'].sum().reset_index()
        res_dept['id'] = res_dept['departement']
        res_dept['graph_id'] = res_dept['departement']
        res_dept['latitude'] = res.groupby(['departement'])['latitude'].mean().reset_index()['latitude']
        res_dept['longitude'] = res.groupby(['departement'])['longitude'].mean().reset_index()['longitude']
        res_dept['nbsinister'] = res.groupby(['departement', 'date'])['nbsinister'].sum().reset_index()['nbsinister']
        res_dept['nbsinister_id'] = res.groupby(['departement', 'date'])['nbsinister'].sum().reset_index()['nbsinister']
        res_dept['days_until_next_event'] = calculate_days_until_next_event(res_dept['id'].values, res_dept['date'].values, res_dept['nbsinister'].values)

        """dir_predictor = dir_train / 'influenceClustering'
        for nameDep in test_departement:
            mask = np.argwhere(y[:, departement_index] == name2int[nameDep])
            if mask.shape[0] == 0:
                continue
            predictor = read_object(nameDep+'PredictorDepartement.pkl', dir_predictor)
            if predictor is None:
                predictor = read_object(f'{nameDep}Predictor{name}Departement.pkl', dir_predictor)
            break

        res_dept['class_risk'] = order_class(predictor, predictor.predict(res_dept['risk'].values, 0))
        res_dept['weight'] =  predictor.weight_array(predictor.predict(res_dept['risk'].values, 0).astype(int)).reshape(-1)

        res_dept['prediction'] = res.groupby(['departement', 'date']).mean().reset_index()['prediction']

        metrics_dept[f'{name}'], res_dept = evaluate_pipeline(dir_train, prefix_train, test_dataset_dept, res_dept[['prediction', 'class_risk']].values,
                                                                res_dept[ids_columns + targets_columns].values, graphScale,
                                                                methods, i, test_departement, target_name, name, test_name,
                                                                dir_output)

        if MLFLOW:
            log_metrics_recursively(metrics_dept[name], prefix='departement')
            
        save_object(res_dept, name+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+test_name+'_pred_dept.pkl', dir_output / name)"""

        res['model'] = name
        res_dept['model'] = name

        res_scale.append(res)
        res_departement.append(res_dept)

        i += 1

        if MLFLOW:
            mlflow.end_run()

    ########################################## Save metrics ################################ 
    outname = 'metrics'+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+test_name+'_tree.pkl'
    save_object(metrics, outname, dir_output)

    outname = 'metrics'+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+test_name+'_dept_tree.pkl'
    save_object(metrics_dept, outname, dir_output)

    return metrics, metrics_dept, res_scale, res_departement
    
def test_sklearn_api_model(args,
                           graphScale,
                          test_dataset_dept,
                          test_dataset_unscale_dept,
                           test_name,
                           prefix_train,
                           prefix_config,
                           models,
                           dir_output,
                           device,
                           encoding,
                           scaling,
                           test_departement,
                           dir_train,
                           name_exp,
                           dir_break_point,
                           doKMEANS):
    
    res_scale = []
    res_departement = []
    
    scale = graphScale.scale
    metrics = {}
    metrics_dept = {}

    i = 0

    #test_dataset_dept.sort_values(by=['id', 'date'], inplace=True, axis=1)

    #################################### Traditionnal ##################################################

    for name, target_name, autoRegression in models:

        y = test_dataset_dept[ids_columns + targets_columns].values
        model_dir = dir_train / name_exp / Path('check_'+scaling + '/' + prefix_train + '/baseline/' + name + '/')
        
        logger.info('#########################')
        logger.info(f'      {name}            ')
        logger.info('#########################')

        model = read_object(name+'.pkl', model_dir)
        quantile = name.find('quantile') != -1

        if model is None:
            continue

        if name.find('grid_search') != 1:
            logger.info(f'{model.get_params(deep=True)}')

        run = f'{test_name}_{name}_{prefix_train}'

        if MLFLOW:
            existing_run = get_existing_run(f'{run}')
            if existing_run:
                mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
            else:
                mlflow.start_run(run_name=f'{run}', nested=True)

        graphScale._set_model(model)

        features_selected = read_object('features.pkl', model_dir)

        pred, pred_max, pred_min = graphScale.predict_model_api_sklearn(test_dataset_dept, features_selected, target_name, autoRegression)

        if quantile:
            pred_max = np.empty((y.shape[0], 2))
            pred_min = np.empty((y.shape[0], 2))
            pred_min[:, 0] = preds[:, 0]
            pred[:, 0] = preds[:, 2]
            pred_max[:, 0] = preds[:, -1]

        if MLFLOW:
            #signature = infer_signature(test_dataset_dept[features_selected], pred[:, 0])
            mlflow.set_tag(f"Training", f"{name}")
            mlflow.log_param(f'relevant_feature_name', features_selected)
            mlflow.log_params(model.get_params(deep=True))
            mlflow.log_params({'scale' : args['scale']})
            #_ = mlflow.sklearn.log_model(
            #sk_model=model,
            #artifact_path=f'{name}',
            #signature=signature,
            #input_example=test_dataset_dept[features_selected],
            #registered_model_name=name,
            #)

        logger.info(f'pred min {np.nanmin(pred[:, 0])}, pred max : {np.nanmax(pred[:, 0])}')

        if doKMEANS:
            _, _ , kmeans_features = get_features_for_sinister_prediction(dataset_name=args['dataset'], sinister=args['sinister'], isInference=args['name'] == 'inference')
            df = pd.DataFrame(columns=ids_columns + targets_columns, index=np.arange(pred.shape[0]))
            df[ids_columns] = test_dataset_dept[ids_columns]
            df[targets_columns] = test_dataset_dept[targets_columns]
            features_selected_kmeans,_ = get_features_name_list(scale, kmeans_features, METHODS_KMEANS_TRAIN)
            df[features_selected_kmeans] = test_dataset_unscale_dept[features_selected_kmeans]
            df['prediction'] = pred[:, 0]
            shift = int(args['shift'])
            shift_list = np.arange(0, shift+1)
            prefix_kmeans = f'{args["nbpoint"]}_{scale}_{graphScale.base}_{graphScale.graph_method}'
            df = apply_kmeans_class_on_target(df.copy(deep=True), dir_train / 'check_none' / prefix_kmeans / 'kmeans', 'prediction', float(args['thresh_kmeans']), features_selected_kmeans, new_val=0, shifts=shift_list, mask_df=None)
            df = apply_kmeans_class_on_target(df.copy(deep=True), dir_train / 'check_none' / prefix_kmeans / 'kmeans', 'nbsinister', float(args['thresh_kmeans']), features_selected_kmeans, new_val=0, shifts=shift_list, mask_df=None)
            df = apply_kmeans_class_on_target(df.copy(deep=True), dir_train / 'check_none' / prefix_kmeans / 'kmeans', 'risk', float(args['thresh_kmeans']), features_selected_kmeans, new_val=0, shifts=shift_list, mask_df=None)
            
            test_dataset_dept['nbsinister'] = df[f"nbsinister_{args['shift']}_{float(args['thresh_kmeans'])}"].values
            test_dataset_dept['risk'] = df[f"risk_{args['shift']}_{float(args['thresh_kmeans'])}"].values
            pred[:, 0] = df[f"prediction_{args['shift']}_{float(args['thresh_kmeans'])}"].values
            y[:, risk_index] = df[f"risk_{args['shift']}_{float(args['thresh_kmeans'])}"].values
            y[:, nbsinister_index] = df[f"nbsinister_{args['shift']}_{float(args['thresh_kmeans'])}"].values

        metrics[run], res = evaluate_pipeline(dir_train, prefix_config, test_dataset_dept, pred, y, graphScale,
                                               test_departement, target_name, name,
                                               dir_output, pred_min = pred_min, pred_max = pred_max)
        
        if MLFLOW:
            log_metrics_recursively(metrics[name], prefix='')

        logger.info('Features importances')
        maxi = np.nanmax(test_dataset_dept['nbsinister'].values)
        samples = test_dataset_dept[test_dataset_dept['nbsinister'] == maxi].index
        samples_name = [
        f"{id_}_{allDates[int(date)]}" for id_, date in zip(test_dataset_dept.loc[samples, 'id'].values, test_dataset_dept.loc[samples, 'date'].values)
        ]
        try:
            model.shapley_additive_explanation(test_dataset_dept[features_selected], 'test', dir_output / name,  mode='beeswarm', figsize=(30,15), samples=samples, samples_name=samples_name)
        except:
            pass

        save_object(res, name+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+test_name+'_pred.pkl', dir_output / name)
    
        """if scale == 'departement':
            continue
        # Analyse departement prediction
        res_dept = res.groupby(['departement', 'date'])['risk'].sum().reset_index()
        res_dept['id'] = res_dept['departement']
        res_dept['graph_id'] = res_dept['departement']
        res_dept['latitude'] = res.groupby(['departement'])['latitude'].mean().reset_index()['latitude']
        res_dept['longitude'] = res.groupby(['departement'])['longitude'].mean().reset_index()['longitude']
        res_dept['nbsinister'] = res.groupby(['departement', 'date'])['nbsinister'].sum().reset_index()['nbsinister']
        res_dept['days_until_next_event'] = calculate_days_until_next_event(res_dept['id'].values, res_dept['date'].values, res_dept['nbsinister'].values)

        dir_predictor = dir_train / 'influenceClustering'
        for nameDep in test_departement:
            mask = np.argwhere(y[:, departement_index] == name2int[nameDep])
            if mask.shape[0] == 0:
                continue
            predictor = read_object(nameDep+'PredictorDepartement.pkl', dir_predictor)
            if predictor is None:
                predictor = read_object(f'{nameDep}Predictor{name}Departement.pkl', dir_predictor)
            break

        res_dept['class_risk'] = order_class(predictor, predictor.predict(res_dept['risk'].values, 0))
        res_dept['weight'] =  predictor.weight_array(predictor.predict(res_dept['risk'].values, 0).astype(int)).reshape(-1)

        if target_name == 'binary':
            res_dept['prediction'] = res.groupby(['departement', 'date']).apply(group_probability).reset_index()['prediction']
        else:
            res_dept['prediction'] = res.groupby(['departement', 'date']).sum().reset_index()['prediction']

        metrics_dept[f'{run}_departement'], res_dept = evaluate_pipeline(dir_train, prefix_train, res_dept[['prediction', 'class_risk']].values,
                                                                res_dept[ids_columns + targets_columns].values,
                                                                graphScale,
                                                                methods, i, test_departement, target_name, name, test_name,
                                                                dir_output, departement_scale=True)
        if MLFLOW:
            log_metrics_recursively(metrics_dept[name], prefix='departement')
            
        save_object(res_dept, name+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+test_name+'_pred_dept.pkl', dir_output / name)"""
        res['model'] = name
        #res_dept['model'] = name

        res_scale.append(res)
        #res_departement.append(res_dept)

        i += 1

        if MLFLOW:
            mlflow.end_run()

    ########################################## Save metrics ################################ 
    outname = 'metrics'+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+test_name+'_tree.pkl'
    save_object(metrics, outname, dir_output)

    outname = 'metrics'+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+test_name+'_dept_tree.pkl'
    save_object(metrics_dept, outname, dir_output)

    return metrics, metrics_dept, res_scale, res_departement
    
def test_dl_model(args,
                  graphScale, test_dataset_dept,
                          test_dataset_unscale_dept,
                          train_dataset,
                           methods,
                           test_name,
                           features_name,
                           prefix_train,
                           prefix_config,
                           models,
                           dir_output,
                           device,
                           k_days,
                           encoding,
                           scaling,
                           test_departement,
                           dir_train,
                           features,
                           dir_break_point,
                           name_exp,
                           doKMEANS,
                           ):
    
    res_scale = []
    res_departement = []
    
    scale = graphScale.scale
    metrics = {}
    metrics_dept = {}

    save_object(test_dataset_unscale_dept, f'test_dataset_unscale_{test_name}.pkl', dir_output)
    save_object(test_dataset_dept, f'test_dataset_{test_name}.pkl', dir_output)

    test_dataset_dept.sort_values(by=['graph_id', 'date'], inplace=True)
    i = 0

    #################################### GNN ###################################################
    for name, use_temporal_as_edges, target_name, autoRegression in models:
        model_name = name.split('_')[0]

        is_2D_model = model_name in models_2D
        if is_2D_model:
            if model_name in ['Zhang', 'ConvLSTM', 'Zhang3D']:
                image_per_node = True
            else:
                image_per_node = False

        if model_name in models_hybrid:
            image_per_node = True

        model_dir = dir_train / name_exp / f'check_{scaling}' / prefix_train / name

        logger.info('#########################')
        logger.info(f'       {name}          ')
        logger.info('#########################')

        features_selected_str = read_object('features_str.pkl', model_dir)
        features_selected = read_object('features.pkl', model_dir)

        if features_selected is None:
            logger.info(f'{model_dir}/features.pkl not found')
            continue

        if is_pc:
            temp_dir = 'Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN'
        else:
            temp_dir = 'GNN'

        if model_name in models_hybrid:
            test_loader = load_loader_test_hybrid(use_temporal_as_edges=use_temporal_as_edges,
                                                  image_per_node=image_per_node,
                                                  scale=scale, graphScale=graphScale, dir_output=rootDisk / temp_dir / dir_train,
                                                df=test_dataset_dept, df_train=train_dataset, device=device,
                                                k_days=k_days, test=test_name,
                                                features_name=features_selected_str,
                                                features=features,
                                                scaling=scaling, encoding=encoding, prefix=prefix, Rewrite=True)
        elif not is_2D_model:
            test_loader = load_loader_test(use_temporal_as_edges=use_temporal_as_edges, graphScale=graphScale, dir_output=dir_output,
                                        df=test_dataset_dept,
                                        device=device, k_days=k_days, test=test_name, features_name=features_selected_str,
                                        scaling=scaling, encoding=encoding, prefix=prefix, Rewrite=True)

        else:
            test_loader = load_loader_test_2D(use_temporal_as_edges=use_temporal_as_edges,
                                              image_per_node=image_per_node, graphScale=graphScale,
                                              dir_output=rootDisk / temp_dir / dir_train,
                                              df=test_dataset_unscale_dept,
                                            df_train=train_dataset, device=device, k_days=k_days, test=test_name,
                                            scaling=scaling, prefix=prefix, scale=scale,
                                            features_name_2D=features_selected_str, features=features, features_name=features_name,
                                            encoding=encoding,
                                            Rewrite=True)

        #print(test_dataset_dept[test_dataset_dept['date'] >= allDates.index('2024-01-01')]['weight'].unique())
        #print(allDates.index('2024-01-01'))

        if model_name in models_hybrid:
            model, model_params = make_model(model_name, len(features_selected[0]), len(features_selected[1]), graphScale, dropout, 'relu', k_days, target_name == 'binary', device, num_lstm_layers)
        else:
            model, model_params = make_model(model_name, len(features_selected), len(features_selected), graphScale, dropout, 'relu', k_days, target_name == 'binary', device, num_lstm_layers)
            
        if (model_dir / 'best.pt').is_file():
            graphScale._load_model_from_path(model_dir / 'best.pt', model, device, model_name)
        else:
            logger.info(f'{model_dir}/best.pt not found')
            continue

        run = f'{test_name}_{name}_{prefix_train}'

        if MLFLOW:
            existing_run = get_existing_run(f'{run}')
            if existing_run:
                mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
            else:
                mlflow.start_run(run_name=f'{run}', nested=True)

        predTensor, YTensor = graphScale._predict_test_loader(test_loader, features_selected, device=device, target_name=target_name, 
                                                            autoRegression=autoRegression, features_name=features_selected_str, dataset=test_dataset_dept)
        
        if target_name == 'binary' or target_name == 'nbsinister':
            band = -2
        else:
            band = -1

        pred = np.full((predTensor.shape[0], 2), fill_value=np.nan)
        if name in ['Unet', 'ULSTM']:
            pred = np.full((y.shape[0], 2), fill_value=np.nan)
            pred_2D = predTensor.detach().cpu().numpy()
            Y_2D = YTensor.detach().cpu().numpy()   
            udates = np.unique(test_dataset_dept['date'].values)
            ugraph = np.unique(test_dataset_dept['graph_id'].values)
            for graph in ugraph:
                for date in udates:
                    mask_2D = np.argwhere((Y_2D[:, graph_id_index] == graph) & (Y_2D[:, date_index] == date))
                    mask = np.argwhere((y[:, graph_id_index] == graph) & (y[:, date_index] == date))
                    if mask.shape[0] == 0:
                        continue
                    pred[mask[:, 0], 0] = pred_2D[mask_2D[:, 0], band, mask_2D[:, 1], mask_2D[:, 2]]
        else:
            pred[:, 0] = predTensor.detach().cpu().numpy()

        if MLFLOW:
            mlflow.set_tag(f"Testing", f"{name}")
            mlflow.log_param(f'relevant_feature_name', features)
            mlflow.log_param(f"p_coef' : {p_coef_value[name]}")
            mlflow.log_params(args)
            #model_params.update({'epochs' : epochs, 'learning_rate' : lr, 'patience_count': PATIENCE_CNT})
            #mlflow.log_params(model_params)
            """signature = infer_signature(test_dataset_dept[features], pred[:, 0])

            _ = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f'{name}',
            signature=signature,
            input_example=test_dataset_dept[features],
            registered_model_name=name,
            )"""

        if doKMEANS:
            _, _ , kmeans_features = get_features_for_sinister_prediction(dataset_name=args['dataset'], sinister=args['sinister'], isInference=args['name'] == 'inference')
            df = pd.DataFrame(columns=ids_columns + targets_columns, index=np.arange(pred.shape[0]))
            df[ids_columns] = test_dataset_dept[ids_columns]
            features_selected_kmeans,_ = get_features_name_list(scale, kmeans_features, METHODS_KMEANS_TRAIN)
            df[features_selected_kmeans] = test_dataset_unscale_dept[features_selected_kmeans]
            df[target_name] = pred[:, 0]
            df = apply_kmeans_class_on_target(df, dir_break_point=dir_break_point, target=target_name, tresh=tresh_kmeans,
                                            features_selected=features_selected_kmeans, new_val=0)

            pred[:, 0] = df[target_name]
        y = YTensor.detach().cpu().numpy()
        #y = test_dataset_dept[ids_columns + targets_columns].values
        #print(y.shape, YTensor.shape)
        metrics[run], res = evaluate_pipeline(dir_train, prefix_config, test_dataset_dept, pred, y, graphScale,
                                            methods, i, test_departement, target_name, name, test_name, dir_output)
        
        if MLFLOW:
            log_metrics_recursively(metrics[name], prefix='')

        logger.info('Features importances')
        maxi = np.nanmax(test_dataset_dept['nbsinister'].values)
        samples = test_dataset_dept[test_dataset_dept['nbsinister'] == maxi].index
        samples_name = [
        f"{id_}_{allDates[int(date)]}" for id_, date in zip(test_dataset_dept.loc[samples, 'id'].values, test_dataset_dept.loc[samples, 'date'].values)
        ]
    
        #top_features_ = shapley_additive_explanation_neural_network(model_name, test_dataset_dept[features_selected], 'test', dir_output / name, figsize=(30,15), samples=samples, samples_name=samples_name)
    
        """if top_features_ is not None:
            predict_and_plot_features(df_samples=test_dataset_unscale_dept[test_dataset_unscale_dept['nbsinister'] == maxi], variables=model.top_features_, dir_break_point=dir_break_point, dir_output=dir_output / name)
        else:
            logger.info('Could not produce features plot')"""
        
        if scale == 'departement':
            continue
        # Analyse departement prediction
        res_dept = res.groupby(['departement', 'date'])['risk'].sum().reset_index()
        res_dept['id'] = res_dept['departement']
        res_dept['graph_id'] = res_dept['departement']
        res_dept['latitude'] = res.groupby(['departement'])['latitude'].mean().reset_index()['latitude']
        res_dept['longitude'] = res.groupby(['departement'])['longitude'].mean().reset_index()['longitude']
        res_dept['nbsinister'] = res.groupby(['departement', 'date'])['nbsinister'].sum().reset_index()['nbsinister']
        res_dept['nbsinister_id'] = res.groupby(['departement', 'date'])['nbsinister'].sum().reset_index()['nbsinister']
        res_dept['days_until_next_event'] = calculate_days_until_next_event(res_dept['id'].values, res_dept['date'].values, res_dept['nbsinister'].values)

        dir_predictor = dir_train / 'influenceClustering'
        for nameDep in test_departement:
            mask = np.argwhere(y[:, departement_index] == name2int[nameDep])
            if mask.shape[0] == 0:
                continue
            predictor = read_object(nameDep+'PredictorDepartement.pkl', dir_predictor)
            if predictor is None:
                predictor = read_object(f'{nameDep}Predictor{name}Departement.pkl', dir_predictor)
            break

        res_dept['class_risk'] = order_class(predictor, predictor.predict(res_dept['risk'].values, 0))
        res_dept['weight'] =  predictor.weight_array(predictor.predict(res_dept['risk'].values, 0).astype(int)).reshape(-1)

        if target_name == 'binary':
            res_dept['prediction'] = res.groupby(['departement', 'date']).apply(group_probability).reset_index()['prediction']
        else:
            res_dept['prediction'] = res.groupby(['departement', 'date']).sum().reset_index()['prediction']

        """metrics_dept[f'{name}_departement'], res_dept = evaluate_pipeline(dir_train, prefix_train, test_dataset_dept, res_dept[['prediction', 'class_risk']].values,
                                                                res_dept[ids_columns + targets_columns].values,
                                                                graphScale,
                                                                methods, i, test_departement, target_name, name, test_name,
                                                                dir_output, departement_scale=True)"""
        metrics_dept[f'{run}_departement'] = {'mae': None}
        
        save_object(res, name+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+test_name+'_pred.pkl', dir_output / name)
        save_object(res_dept, name+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+test_name+'_dept_pred.pkl', dir_output / name)

        if MLFLOW:
            log_metrics_recursively(metrics_dept[f'{name}_departement'], prefix='departement')
            mlflow.end_run()

        res['model'] = name
        res_dept['model'] = name

        res_scale.append(res)
        res_departement.append(res_dept)
        
        i += 1

    ########################################## Save metrics ################################
    outname = 'metrics'+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+test_name+'_dl.pkl'
    if (dir_output / outname).is_file():
        log_metrics = read_object(outname, dir_output)
        metrics.update(log_metrics)

    save_object(metrics, outname, dir_output)

    outname = 'metrics'+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+test_name+'_dept_dl.pkl'
    if (dir_output / outname).is_file():
        log_metrics = read_object(outname, dir_output)
        metrics.update(log_metrics)
        
    save_object(metrics_dept, outname, dir_output)
    return metrics, metrics_dept, res_scale, res_departement

def test_simple_model(args,
                           graphScale,
                          test_dataset_dept,
                          test_dataset_unscale_dept,
                           methods,
                           test_name,
                           prefix_train,
                           dir_output,
                           encoding,
                           scaling,
                           test_departement,
                           dir_train,
                           dir_break_point,
                           doKMEANS
                           ):
    metrics = {}
    scale = graphScale.scale

    ################################ Ground Truth ############################################
    logger.info('#########################')
    logger.info(f'       GT              ')
    logger.info('#########################')

    y = test_dataset_dept[ids_columns + targets_columns].values
    pred = np.empty((y.shape[0], 2))
    pred[:, 0] = test_dataset_dept['risk'].values
    pred[:, 1] = test_dataset_dept['class_risk'].values

    if MLFLOW:
        existing_run = get_existing_run(f'{test_name}_GT_{prefix_train}')
        if existing_run:
            mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
        else:
            mlflow.start_run(run_name=f'{test_name}_GT_{prefix_train}', nested=True)

    metrics['GT'], res = evaluate_pipeline(dir_train, pred, y, graphScale,
                                               methods, 0, test_departement, 'risk', 'GT', test_name,
                                               dir_output)
                                
    if MLFLOW:
        log_metrics_recursively(metrics['GT'], prefix='')
        mlflow.log_params(args)
        mlflow.end_run()
    i = 1

    ################################ Persistance ############################################

    logger.info('#########################')
    logger.info(f'      pers_Y            ')
    logger.info('#########################')
    name = 'pers_Y'
    target_name = 'risk'
    pred = graphScale._predict_perference_with_Y(y, target_name)

    if doKMEANS:
        _, _ , kmeans_features = get_features_for_sinister_prediction(dataset_name=args['dataset'], sinister=args['sinister'], isInference=args['name'] == 'inference')
        df = pd.DataFrame(columns=ids_columns + targets_columns, index=np.arange(pred.shape[0]))
        df[ids_columns] = test_dataset_dept[ids_columns]
        features_selected_kmeans,_ = get_features_name_list(scale, kmeans_features, METHODS_KMEANS_TRAIN)
        df[features_selected_kmeans] = test_dataset_unscale_dept[features_selected_kmeans]
        df[target_name] = pred[:, 0]
        df = apply_kmeans_class_on_target(df, dir_break_point=dir_break_point, target=target_name, tresh=tresh_kmeans,
                                            features_selected=features_selected_kmeans, new_val=0)

        pred[:, 0] = df[target_name]


    metrics[name], res = evaluate_pipeline(dir_train, pred, y, scale, graphScale.base,
                                               methods, 0, test_departement, target_name, 'pers_Y', test_name,
                                               dir_output)
    
    res = np.empty((pred.shape[0], y.shape[1] + 3))
    res[:, :y.shape[1]] = y
    res[:,y.shape[1]] = pred
    res[:,y.shape[1]+1] = np.sqrt((y[:,-4] * (pred - y[:,-1]) ** 2))
    res[:,y.shape[1]+2] = np.sqrt(((pred - y[:,-1]) ** 2))

    save_object(res, name+'_pred.pkl', dir_output)

    logger.info('#########################')
    logger.info(f'      pers_Y_bin         ')
    logger.info('#########################')
    name = 'pers_Y_bin'
    target_name = 'binary'
    pred = graphScale._predict_perference_with_Y(y, target_name)

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
                                            methods, 0, test_departement, target_name, 'pers_Y_bin', test_name,
                                            dir_output)

    logger.info('#########################')
    logger.info(f'      pers_Y_sinis         ')
    logger.info('#########################')
    name = 'pers_Y_sinis'
    target_name = 'nbsinister'
    pred = graphScale._predict_perference_with_Y(y, target_name)

    if doKMEANS:
        df = pd.DataFrame(columns=ids_columns + targets_columns, index=np.arange(pred.shape[0]))
        df[ids_columns] = test_dataset_dept[ids_columns]
        features_selected_kmeans,_ = get_features_name_list(scale, kmeans_features, METHODS_KMEANS_TRAIN)
        df[features_selected_kmeans] = test_dataset_unscale_dept[features_selected_kmeans]
        df[target_name] = pred[:, 0]
        df = apply_kmeans_class_on_target(df, dir_break_point=dir_break_point, target=target_name, tresh=tresh_kmeans,
                                            features_selected=features_selected_kmeans, new_val=0)

    metrics[name], res = evaluate_pipeline(dir_train, pred, y, scale, graphScale.base,
                                            methods, 0, test_departement, target_name, 'pers_Y_sinis', test_name,
                                            dir_output)

    logger.info('#########################')
    logger.info(f'      pers_X         ')
    logger.info('#########################')
    name = 'pers_X'
    target_name = 'risk'
    pred = graphScale._predict_perference_with_X(test_dataset_dept, features_name)

    if doKMEANS:
        df = pd.DataFrame(columns=ids_columns + targets_columns, index=np.arange(pred.shape[0]))
        df[ids_columns] = test_dataset_dept[ids_columns]
        features_selected_kmeans,_ = get_features_name_list(scale, kmeans_features, METHODS_KMEANS_TRAIN)
        df[features_selected_kmeans] = test_dataset_unscale_dept[features_selected_kmeans]
        df[target_name] = pred[:, 0]
        df = apply_kmeans_class_on_target(df, dir_break_point=dir_break_point, target=target_name, tresh=tresh_kmeans,
                                            features_selected=features_selected_kmeans, new_val=0)

    metrics[name], res = evaluate_pipeline(dir_train, pred, y, scale, graphScale.base,
                                            methods, 0, test_departement, target_name, 'pers_X', test_name,
                                            dir_output)

    outname = 'metrics'+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+test_name+'_persistence.pkl'
    save_object(metrics, outname, dir_output)

def test_break_points_model(test_dataset_dept,
                           methods,
                           test_name,
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
            
    metrics[name], res = evaluate_pipeline(dir_train, pred, y, scale, methods, 0, test_departement, 'risk', name, test_name, dir_output)

    #save_object(res, name+'_'+prefix_train+'_'+str(scale)+'_'+scaling+'_'+encoding+'_'+test_name+'_pred.pkl', dir_output)
    save_object(res, name+'_pred.pkl', dir_output)
    outname = 'metrics'+'_'+prefix_train+'_'+scaling+'_'+encoding+'_'+test_name+'_bp.pkl'
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
            x, e = construct_graph_set(graph, id, X, None, ks, len(ids_columns))
        else:
            x, e = construct_graph_with_time_series(graph, id, x, None, ks, len(ids_columns))
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

def wrapped_train_deep_learning_1D(params):
    model = params['model']
    use_temporal_as_edges = params['use_temporal_as_edges']
    infos = params['infos']
    autoRegression = params['autoRegression']

    target_name, task_type, loss = infos.split('_')
    loss_name = loss

    logger.info(f'Fitting model {model}')
    logger.info('Try loading loader')

    train_loader = read_object(f'train_loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{use_temporal_as_edges}.pkl', params['dir_output'])
    val_loader = read_object(f'val_loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{use_temporal_as_edges}.pkl', params['dir_output'])
    test_loader = read_object(f'test_loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{use_temporal_as_edges}.pkl', params['dir_output'])

    if (train_loader is None or val_loader is None or test_loader is None) or params['Rewrite']:
        logger.info(f'Building loader, loading loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{use_temporal_as_edges}.pkl')
        train_loader, val_loader, test_loader = train_val_data_loader(
            graph=params['graphScale'],
            df_train=params['train_dataset'],
            df_val=params['val_dataset'],
            df_test=params['test_dataset'],
            features_name=params['features_selected_str'],
            batch_size=batch_size,
            device=params['device'],
            use_temporal_as_edges=use_temporal_as_edges,
            ks=params['k_days']
        )
        save_object(train_loader, f'train_loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{use_temporal_as_edges}.pkl', params['dir_output'])
        save_object(val_loader, f'val_loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{use_temporal_as_edges}.pkl', params['dir_output'])
        save_object(test_loader, f'test_loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{use_temporal_as_edges}.pkl', params['dir_output'])

    train_params = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "k_days": params['k_days'],
        "optimize_feature": params['optimize_feature'],
        "PATIENCE_CNT": params['PATIENCE_CNT'],
        "CHECKPOINT": params['CHECKPOINT'],
        "epochs": params['epochs'],
        "features_selected": params['features_selected'],
        "features_selected_str": params['features_selected_str'],
        "lr": params['lr'],
        'graph': params['graph'],
        "loss_name": loss_name,
        "modelname": model,
        "dir_output": params['dir_output'] / Path(f'check_{params["scaling"]}/{params["prefix"]}/{model}_{infos}'),
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
            graph=params['graph'],
            df_train=params['train_dataset_unscale'],
            df_val=params['val_dataset'],
            df_test=params['test_dataset'],
            use_temporal_as_edges=use_temporal_as_edges,
            image_per_node=image_per_node,
            batch_size=batch_size,
            device=params['device'],
            scaling=params['scaling'],
            features_name_2D=params['features_name_2D'],
            features_name=params['features_name_1D'],
            features=params['features'],
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
        "features_selected": params['features_selected_2D'],
        "features_selected_str": params['features_name_2D'],
        "lr": params['lr'],
        'graph': params['graph'],
        "loss_name": loss_name,
        "modelname": model,
        "dir_output": params['dir_output'] / Path(f'check_{params["scaling"]}/{params["prefix"]}/{model}_{infos}'),
        "target_name": target_name,
        'infos' : infos,
        "autoRegression": autoRegression,
        'k_days' : params['k_days']
    }

    train(train_params)

def wrapped_train_deep_learning_hybrid(params):
    model = params['model']
    use_temporal_as_edges = params['use_temporal_as_edges']
    infos = params['infos']
    autoRegression = params['autoRegression']
    image_per_node = params['image_per_node']

    target_name, task_type, loss = infos.split('_')
    loss_name = loss

    logger.info(f'Fitting model {model}')
    logger.info('Try loading loader')

    train_loader = read_object(f'train_loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{image_per_node}_{use_temporal_as_edges}_hybrid.pkl', params['dir_output'])
    val_loader = read_object(f'val_loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{image_per_node}_{use_temporal_as_edges}_hybrid.pkl', params['dir_output'])
    test_loader = read_object(f'test_loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{image_per_node}_{use_temporal_as_edges}_hybrid.pkl', params['dir_output'])

    if (train_loader is None or val_loader is None or test_loader is None) or params['Rewrite']:
        logger.info(f'Building loader, loading loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{image_per_node}_{use_temporal_as_edges}_hybrid.pkl')
        train_loader, val_loader, test_loader = train_val_data_loader_hybrid(
            graph=params['graphScale'],
            df_train=params['train_dataset_unscale'],
            df_val=params['val_dataset'],
            df_test=params['test_dataset'],
            use_temporal_as_edges=use_temporal_as_edges,
            image_per_node=image_per_node,
            scale=params['scale'],
            batch_size=batch_size,
            device=params['device'],
            scaling=params['scaling'],
            features_name=params['features_selected_str_1D'],
            features_name_2D=params['features_selected_str_2D'],
            features=params['features'],
            ks=params['k_days'],
            path=Path(params['name_dir'])
        )

        save_object(train_loader, f'train_loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{image_per_node}_{use_temporal_as_edges}_hybrid.pkl', params['dir_output'])
        save_object(val_loader, f'val_loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{image_per_node}_{use_temporal_as_edges}_hybrid.pkl', params['dir_output'])
        save_object(test_loader, f'test_loader_{params["prefix"]}_{params["scaling"]}_{params["encoding"]}_{image_per_node}_{use_temporal_as_edges}_hybrid.pkl', params['dir_output'])

    train_params = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "k_days": params['k_days'],
        "optimize_feature": params['optimize_feature'],
        "PATIENCE_CNT": params['PATIENCE_CNT'],
        "CHECKPOINT": params['CHECKPOINT'],
        "epochs": params['epochs'],
        "lr": params['lr'],
        "features_selected": params['features_selected'],
        "features_selected_str": (params['features_selected_str_1D'], params['features_selected_str_2D']),
        "loss_name": loss_name,
        "modelname": model,
        'scale' : params['scale'],
        'graph_method': params['graph_method'],
        "dir_output": params['dir_output'] / Path(f'check_{params["scaling"]}/{params["prefix"]}/{model}_{infos}'),
        "target_name": target_name,
        "autoRegression": autoRegression,
        'k_days' : params['k_days']
    }

    train(train_params)