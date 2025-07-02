from tools import *
from forecasting_models.pytorch.tools_2 import *
from forecasting_models.pytorch.loss import *
from forecasting_models.sklearn.sklearn_api_models_config import *

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import balanced_accuracy_score, classification_report
import pandas as pd
import os

from matplotlib.colors import ListedColormap
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from PixelEnv import *
from stable_baselines3 import PPO

import shap

# Fonction pour sélectionner la fonction de perte via son nom
def get_loss_function(loss_name, **loss_params):
    # Dictionnaire pour associer le nom de la fonction de perte à sa classe correspondante

    loss_dict = {
        "poisson": PoissonLoss(),
        "rmsle": RMSLELoss(),
        "rmse": RMSELoss(),
        "mse": MSELoss(),
        "huber": HuberLoss(),
        "logcosh": LogCoshLoss(),
        "tukeybiweight": TukeyBiweightLoss(),
        "exponential": ExponentialLoss(),
        'ordinal-dice' : OrdinalDiceLoss(),
        'dice' : DiceLoss2(),
        "weightedcrossentropy": WeightedCrossEntropyLoss(**loss_params),
        "weightedcrossentropy-2": WeightedCrossEntropyLoss(**loss_params),
        'kappa' : WKLoss(**loss_params),
        'cdw' : CDWCELoss(**loss_params),
        #'mcewk' : MCEAndWKLoss(**loss_params),
        #'kldivloss' : KLDivLoss(reduction='batchmean'),
    }
    loss_name = loss_name.lower()
    if loss_name in loss_dict:
        return loss_dict[loss_name]
    else:
        raise ValueError(f"Loss function '{loss_name}' not found in loss_dict.")
    
def get_model_from_string(model_type: str, config_string: str, **kwargs):
    """
    Parse la string et construit le bon modèle.

    Exemple :
    model_type = "GNN"
    config_string = "gnn_classic_classification_cross_entropy"

    -> retourne une instance de ModelGNN avec les bons paramètres
    """

    # Parsing de la string
    parts = config_string.strip().split("_")

    model_name, split_mode, task_type, training_type, loss = parts
    out_channels = 4
    if task_type == 'binary':
        out_channels = 2
    elif task_type == 'regression':
        out_channels = 1

    base_kwargs = dict(
        model_name=model_name,
        train_val_test_split_mode=split_mode,
        task_type=task_type,
        loss=loss,
        training_type=training_type,
        out_channels=out_channels
    )
    base_kwargs.update(kwargs)

    # Sélection du constructeur
    model_type = model_type.lower()
    if model_type == "gnn":
        return ModelGNN(**base_kwargs)
    elif model_type == "cnn":
        return ModelCNN(**base_kwargs)
    elif model_type == "normal":
        return Model_Torch(**base_kwargs)
    else:
        raise ValueError(f"Type de modèle inconnu : {model_type}")
    
# Classe mère
class Training():
    def __init__(
        self,
        model_name=None,
        features_name=None,
        out_channels=1,
        task_type='classification',
        device=None,
        loss='cross_entropy',
        batch_size=256,
        lr=1e-3,
        epochs=10,
        dropout=0.2,
        num_lstm_layers=1,
        ks=1,
        kernel_size=3,
        train_val_test_split_mode='classic',
        test_size=0.2,
        val_size=0.1,
        dir_output = Path('./'),
        spatial_mask=None,
        spatial_mask_name='',
        temporal_mask = None,
        temporal_mask_name = '',
        under_sampling='None',
        training_type='supervised',
        valid_transitions_dict=None
    ):
        self.model_name = model_name
        self.features_name = features_name or []
        self.out_channels = out_channels
        self.task_type = task_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loss = loss
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.num_lstm_layers = num_lstm_layers
        self.ks = ks
        self.kernel_size = kernel_size
        self.train_val_test_split_mode = train_val_test_split_mode

        self.model_params = None
        self.criterion = None
        self.model = None
        self.optimizer = None

        self.graph = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.test_size = test_size
        self.val_size = val_size

        self.dir_output = dir_output / f'{model_name}_{train_val_test_split_mode}_{spatial_mask_name}_{temporal_mask_name}_{task_type}_{training_type}_{loss}'
        self.log_name = f'{model_name}_{train_val_test_split_mode}_{spatial_mask_name}_{temporal_mask_name}_{task_type}_{loss}'
        check_and_create_path(self.dir_output)

        self.under_sampling = under_sampling

        self.spatial_mask = spatial_mask
        self.spatial_mask_name = spatial_mask_name

        self.temporal_mask = temporal_mask
        self.temporal_mask_name = temporal_mask_name

        self.valid_transitions_dict = valid_transitions_dict
        self.training_type = training_type
    
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

    def  make_model(self, graph, custom_model_params):
        model, params = make_model(
            self.model_name,
            len(self.features_name),
            len(self.features_name),
            graph,
            self.dropout,
            'relu',
            self.ks,
            out_channels=self.out_channels,
            task_type=self.task_type,
            device=self.device,
            num_lstm_layers=self.num_lstm_layers,
            custom_model_params=custom_model_params
        )
        if self.model_params is None:
            self.model_params = params
        return model, params
    
    def create_train_val_test_loader(X, y):
        pass

    def get_loss(self, loss_name):
        loss_params = {'num_classes' : 5}
        return get_loss_function(loss_name, **loss_params)

    def train(self, X, y, create_loader, custom_model_params):
        if create_loader:
            self.create_train_val_test_loader(X, y)
        
        self.criterion = self.get_loss(self.loss)
        BEST_MODEL_PARAMS = None
    
        if 'reinforcement' in self.training_type:
            print(custom_model_params)
            assert self.valid_transitions_dict is not None, f'Have to give a valid transition in custom_model_params'
            
            dl, _ = self.make_model(None, custom_model_params)
            reinforcement_type = self.training_type.split('-')[1]
            sample_env = self.create_env_fn(np.zeros((self.batch_size, len(self.features_name), self.ks + 1)), np.zeros((self.batch_size, 1)), self.valid_transitions_dict)

            if reinforcement_type == 'POO':
                self.model = PPO(
                    policy=CustomPolicy,
                    env=sample_env,
                    learning_rate=self.lr,
                    policy_kwargs={'model' : dl, 'features_dim' : self.out_channels},
                    verbose=0
                )
        else:
            self.model, _ = self.make_model(None, custom_model_params)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if (self.dir_output / 'best_params.pkl').is_file():
            BEST_MODEL_PARAMS = read_object('best_params.pkl', self.dir_output)
        else:
            BEST_MODEL_PARAMS = read_object('rein_model.pkl', self.dir_output)

        if BEST_MODEL_PARAMS is None:

            patience = custom_model_params.get('early_stopping_rounds', 10)
            count = 0
            best_loss = float('inf')

            for epoch in range(self.epochs):
                if self.training_type == 'supervised':
                    loss = self.launch_epoch(epoch)  # val_loss
                else:
                    print(f'epochs : {epoch}/{self.epochs}')
                    loss = self.launch_rl_epoch(custom_model_params)

                if loss < best_loss:
                    best_loss = loss
                    count = 0
                    if 'reinforcement' not in self.training_type:
                        BEST_MODEL_PARAMS = self.model.state_dict()
                    else:
                        #BEST_MODEL_PARAMS = copy.copy(self.model)
                        pass
                else:
                    count += 1
                    if count > patience:
                        print(f"Early stopping: no improvement for {patience} epochs.")
                        break
                
        if 'reinforcement' not in self.training_type:
            self.update_weight(BEST_MODEL_PARAMS)
            save_object(BEST_MODEL_PARAMS, 'best_params.pkl', self.dir_output)
        else:
           pass
            #save_object(BEST_MODEL_PARAMS, 'rein_model.pkl', self.dir_output)
            #self.model = copy.copy(BEST_MODEL_PARAMS)

    def launch_rl_epoch(self, custom_model_params):
        total_timesteps = custom_model_params.get('timesteps_per_epoch', 1)
        total_loss = 0
        val_batches = 0

        for X_batch, y_batch in self.train_loader:
            if X_batch.size(0) != self.batch_size:
                continue
            X_batch = X_batch.detach().cpu().numpy()
            y_batch = y_batch.detach().cpu().numpy()
            # entraînement PPO sur le bloc
            env = self.create_env_fn(X_batch, y_batch, self.valid_transitions_dict)
            self.model.set_env(env)
            self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)

        # Validation après tout le train_loader
        for X_batch, y_batch in self.val_loader:

            if X_batch.size(0) != self.batch_size:
                continue

            X_batch = X_batch.detach().cpu().numpy()
            y_batch = y_batch.detach().cpu().numpy()

            env = self.create_env_fn(X_batch, y_batch, self.valid_transitions_dict)
            obs = env.reset()
            actions = self.model.predict(obs, deterministic=True)[0]

            if X_batch.ndim == 2:
                previous_state = X_batch[-1, -1]
            elif X_batch.ndim == 3:
                previous_state = X_batch[:, -1, -1]
            elif X_batch.ndim == 4:
                previous_state = X_batch[:, :, -1, -1]
            elif X_batch.ndim == 5:
                previous_state = X_batch[:, :, :, -1, -1]
            else:
                pass
            
            output = np.array([
                self.valid_transitions_dict[previous_state[i]][actions[i]]
                for i in range(len(actions))
            ])
            
            loss = self.criterion(torch.Tensor(output), torch.Tensor(y_batch))

            total_loss += loss
            val_batches += 1

        total_loss = loss / val_batches
        print(f"[RL] Validation accuracy: {total_loss:.4f}")
        return total_loss

    def create_env_fn(self, X, y, valid_transitions):
        return create_vec_env_from_block(X, y, valid_transitions)

    def launch_epoch(self, epoch):
        print(f"Epoch {epoch+1}/{self.epochs}")
        self.train_epoch()
        val_loss = self.val_test_epoch(split='val')
        return val_loss
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(X_batch)
            loss = self.criterion(output, y_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        print(f"Train loss: {total_loss / len(self.train_loader):.4f}")
        
    def val_test_epoch(self, split='val'):
        self.model.eval()
        loader = self.val_loader if split == 'val' else self.test_loader
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                total_loss += loss.item()

        print(f"{split.capitalize()} loss: {total_loss / len(loader):.4f}")
        return total_loss
    
    def split_dataset(self, X, y, spatial_mask=None, temporal_mask=None):
        """
        Split les batchs X, y (torch.Tensor) en train/val/test.
        - X : (N, ...) torch.Tensor
        - y : (N, ...) torch.Tensor

        Retourne : X_train, X_val, X_test, y_train, y_val, y_test (tensors)
        """
        from sklearn.model_selection import train_test_split
        #print(X.shape)
        #X, y = remove_nan_pixels(X, y)

        N = X.shape[0]
        y_np = y.view(N, -1).float().mean(dim=1).cpu().numpy()  # pour stratify

        if self.train_val_test_split_mode == "temporal":
            assert N >= 3, f"Pas assez d'échantillons pour split temporal (N={N})"
            X_train = X[:N - 2]
            y_train = y[:N - 2]
            X_val = X[N - 2:N - 1]
            y_val = y[N - 2:N - 1]
            X_test = X[N - 1:]
            y_test = y[N - 1:]

        elif self.train_val_test_split_mode == "classic":
            idx = torch.arange(N).numpy()
            idx_train_val, idx_test = train_test_split(
                idx, test_size=self.test_size, random_state=42, stratify=y_np
            )
            y_train_val_np = y_np[idx_train_val]
            idx_train, idx_val = train_test_split(
                idx_train_val, test_size=self.val_size, random_state=42, stratify=y_train_val_np
            )

            X_train = X[idx_train]
            y_train = y[idx_train]
            X_val = X[idx_val]
            y_val = y[idx_val]
            X_test = X[idx_test]
            y_test = y[idx_test]

        elif self.train_val_test_split_mode == "spatial":
            assert spatial_mask is not None, "spatial_mask est requis pour le split spatial"
            deps = spatial_mask.cpu().numpy().flatten()
            unique_deps = np.unique(deps[~np.isnan(deps)])
            unique_deps = np.random.permutation(unique_deps)

            n_total = len(unique_deps)
            n_train = int(n_total * (1 - self.test_size - self.val_size))
            n_val = int(n_total * self.val_size)

            deps_train = unique_deps[:n_train]
            deps_val = unique_deps[n_train:n_train + n_val]
            deps_test = unique_deps[n_train + n_val:]

            deps = np.array(deps)
            mask_train = np.isin(deps, deps_train)
            mask_val = np.isin(deps, deps_val)
            mask_test = np.isin(deps, deps_test)

            X_train = X[mask_train]
            y_train = y[mask_train]
            X_val = X[mask_val]
            y_val = y[mask_val]
            X_test = X[mask_test]
            y_test = y[mask_test]

            full_mask_flat = np.full(self.spatial_mask.shape, np.nan)
            full_mask_flat[~np.isnan(self.spatial_mask)] = 0  # init

            # Application des splits (modification en place)
            full_mask_flat[np.isin(self.spatial_mask, deps_train)] = 1
            full_mask_flat[np.isin(self.spatial_mask, deps_val)] = 2
            full_mask_flat[np.isin(self.spatial_mask, deps_test)] = 3
            plt.figure(figsize=(10, 5))
            plt.imshow(full_mask_flat, cmap=plt.cm.get_cmap('tab10', 3), vmin=1, vmax=3)
            plt.title("Masques Train (1), Val (2), Test (3) par département")
            plt.colorbar(ticks=[1, 2, 3], label='Split')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(self.dir_output / f'{self.spatial_mask_name}_train_val_test.png')
            plt.close('all')

        elif self.train_val_test_split_mode == 'temporal':
            assert self.temporal_mask is not None and temporal_mask is not None, "temporal_mask est requis pour le split spatial"
            val_value = self.temporal_mask['val']
            test_value = self.temporal_mask['test']
            train_values = self.temporal_mask['train']

            mask_train = np.isin(temporal_mask, train_values)
            mask_val = np.isin(temporal_mask, val_value)
            mask_test = np.isin(temporal_mask, test_value)

            X_train = X[mask_train]
            y_train = y[mask_train]
            X_val = X[mask_val]
            y_val = y[mask_val]
            X_test = X[mask_test]
            y_test = y[mask_test]
        else:
            raise ValueError(f"Mode de split inconnu : {self.train_val_test_split_mode}")
        
        self.histogram(y_train, 'train')
        self.histogram(y_val, 'val')
        self.histogram(y_test, 'test')

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def histogram(self, y, name):
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        y_unique = np.sort(np.unique(y))
        values = []
        color = []
        for yi in y_unique:
            values.append(y[y == yi].shape[0])
            if yi == 0:
                color.append('blue')
            elif yi == 1:
                color.append('green')
            elif yi == 2:
                color.append('yellow')
            elif yi == 3:
                color.append('red')

        plt.bar(y_unique, values, label=y_unique, color=color)
        plt.xlabel('Class')
        plt.ylabel('Count')
        print(self.dir_output / f'historgam_{name}.png')
        plt.savefig(self.dir_output / f'historgam_{name}.png')
        plt.close('all')

    def test(self, loader, loader_name='test'):
        """
        Fait des prédictions sur le test set, affiche et enregistre un rapport de classification.
        Le fichier est sauvé dans self.dir_output avec nom basé sur les configs.
        """

        if 'reinforcement' not in self.training_type:
            self.model.eval()
        
        y_true = []
        y_pred = []
        x_test = []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                preds = self.predict(X_batch)

                x_test.append(X_batch)

                y_true.append(y_batch.cpu())
                y_pred.append(preds.cpu())

        y_true = torch.cat(y_true).view(-1).numpy()
        y_pred = torch.cat(y_pred).view(-1).numpy()
        x_test = torch.cat(x_test, dim=0)

        report = classification_report(y_true, y_pred, output_dict=True)
        score = self.score(y_pred, y_true)

        df_report = pd.DataFrame(report).transpose()
        df_report['test_name'] = loader_name
        
        # Affichage
        print(df_report)
        print(f'Score (bca) : {score}')
        # Sauvegarde
        df_report.to_csv(self.dir_output / f'{self.log_name}_{loader_name}.csv', index=False)
        print(f"Rapport de classification sauvegardé dans : {self.dir_output / self.log_name}")
        print(f'Compute shap values')

        if not 'reinforcement' in self.training_type:
            self.test_shap = self.compute_shap_values(x_test[:100])
            self.plot_shap_grid_torch(self.test_shap, x_test[:100], [0, 1, 2, 3], features_name=self.features_name, name=loader_name)

    def create_tensor(self):
        pass

    def plot_test(self, X, y, index, test_name):
        """
        Affiche et sauvegarde une figure avec les labels et prédictions du test set sous forme d'image.
        Les pixels non prédits sont laissés à NaN.
        
        Parameters:
            shape (tuple): (H, W) de l'image finale
        """
        if not 'reinforcement' in self.training_type:
            self.model.eval()
        
        H, W = y.shape
        y_true_img = np.full((H, W), np.nan)
        y_pred_img = np.full((H, W), np.nan)
        valid_indices = np.argwhere(~np.isnan(y))
        with torch.no_grad():
           # Reconstruction image
            for (i, j) in valid_indices:
                x_model = self.create_tensor(X, index, (i, j))

                if x_model.shape[0] == 0:
                    continue
                
                output = self.predict(x_model)

                y_pred_img[i, j] = output
                
            for (i, j) in valid_indices:
                y_true_img[i, j] = y[i, j]

        # Remise en forme (H, W)
        y_true_img = y_true_img.reshape(H, W)
        y_pred_img = y_pred_img.reshape(H, W)
        cmap = ListedColormap([
            "blue",   # 0 -> 0
            "green",  # 1 -> 0
            "yellow", # 0 -> 1
            "red"     # 1 -> 1
        ])

        vmin = min(np.nanmin(y_true_img), np.nanmin(y_pred_img))
        vmax = max(np.nanmax(y_true_img), np.nanmax(y_pred_img))

        # Affichage
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        im1 = axs[0].imshow(y_true_img, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[0].set_title("Test Ground Truth")

        im2 = axs[1].imshow(y_pred_img, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[1].set_title("Test Predictions")

        # Ajout d'une seule colorbar partagée
        #cbar = fig.colorbar(im2, ax=axs, orientation='vertical', shrink=0.6)
        #cbar.set_label("Valeurs")

        plt.suptitle(f"{self.model_name} - {self.train_val_test_split_mode}")
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Laisse de la place pour le suptitle

        output_path = self.dir_output / f"{self.log_name}_{test_name}_predictions.png"
        plt.savefig(output_path)
        print(f"Figure sauvegardée dans : {output_path}")
        plt.close('all')

    def score(self, y_pred, y):
        return balanced_accuracy_score(y, y_pred)

    def predict(self, X, y=None):
        #tesnort_test = self.create_tensor_array([X], 0)
        if 'reinforcement' not in self.training_type:
            proba = self.model(X)
            res = torch.argmax(proba, axis=1)
        else:
            actions = self.model.predict(X, deterministic=True)[0]
            if isinstance(actions, int):
                actions = [actions]
            
            if X.ndim == 2:
                previous_state = X[-1, -1]
            elif X.ndim == 3:
                previous_state = X[:, -1, -1]
            elif X.ndim == 4:
                previous_state = X[:, :, -1, -1]
            elif X.ndim == 5:
                previous_state = X[:, :, :, -1, -1]
            else:
                pass

            res = torch.Tensor([self.valid_transitions_dict[previous_state[i].item()][actions[i]] for i in range(len(actions))])
            if actions.shape[0] == 1:
                res = res[0]
    
        return res
    
    def select(self, X_train, y_train, percentage):
        # Trouver les indices où y > 0 (à garder tous)
        positive_indices = torch.where(y_train > 0)[0]

        # Indices où y == 0
        zero_indices = torch.where(y_train == 0)[0]

        # Nombre d'échantillons à sélectionner parmi les zéros
        num_zero_to_select = int(len(zero_indices) * percentage)

        # Sélection aléatoire parmi les zéros
        selected_zero_indices = zero_indices[torch.randperm(len(zero_indices))[:num_zero_to_select]]

        # Fusion des indices positifs + sélectionnés
        selected_indices = torch.cat([positive_indices, selected_zero_indices])

        # Mélange des indices
        shuffled_indices = selected_indices[torch.randperm(len(selected_indices))]

        # Sélection dans les tensors d'origine
        X_train_copy = X_train[shuffled_indices]
        y_train_copy = y_train[shuffled_indices]

        return X_train_copy, y_train_copy

    def search_zero_samples_proportion(self, X_train, y_train, X_val, y_val, X_test, y_test):
        test_percentage = np.arange(0.05, 1.05, 0.1)
        patience = 8
        scores = []
        best_score = -math.inf
        if (self.dir_output / 'scores.pkl').is_file():
            scores = read_object('scores.pkl', self.dir_output)
        else:
            for percentage in test_percentage:
                X_train_copy, y_train_copy = self.select(X_train, y_train, percentage)

                print(f"Pourcentage: {percentage:.2f} → {X_train_copy.shape}")

                model_copy = copy.deepcopy(self)

                model_copy.under_sampling = 'None'

                model_copy.train_loader = DataLoader(TensorDataset(X_train_copy, y_train_copy), batch_size=self.batch_size, shuffle=True)
                model_copy.val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=self.batch_size)
                model_copy.test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=self.batch_size)
                
                model_copy.train(None, None, False, {'early_stopping_rounds' : 15})

                pred_test = model_copy.predict(X_val)

                score = model_copy.score(pred_test.detach().cpu().numpy(), y_val.detach().cpu().numpy())
                if score > best_score:
                    best_score = score
                    patience_count = 0
                else:
                    patience_count += 1
                
                if patience_count == patience:
                    break

                del model_copy

                print(f'{percentage} -> {score}')
                scores.append(score)

        best_tp = test_percentage[np.argmax(scores)]

        plt.plot(test_percentage[:len(scores)], scores, label='bca')
        plt.savefig(self.dir_output / 'test_percentage_score.png')

        save_object(scores, f'scores.pkl', self.dir_output)

        X_train, y_train = self.select(X_train, y_train, best_tp)

        return X_train, y_train

    def create_tensor_array(self, X, index):
        pass

    def get_shape_values(self, X):
        pass

    def compute_shap_values(self, data, background_data=None, use_abs_output=False):
        """
        Calcule les SHAP values pour un modèle PyTorch.

        Paramètres :
            model (nn.Module) : modèle PyTorch entraîné
            data (torch.Tensor or numpy.ndarray) : données d’entrée à expliquer (shape: [N, F])
            background_data (torch.Tensor or np.ndarray) : données pour calcul des valeurs de base (shape: [M, F])
            task_type (str) : 'classification' ou 'regression'
            device (str) : 'cpu' ou 'cuda'
            use_abs_output (bool) : applique abs() sur les sorties (utile pour certains modèles régressifs)

        Retourne :
            shap.Explanation : objet SHAP contenant les valeurs pour chaque feature
        """
        #self.model.train()
        if 'reinforcement' not in self.training_type:
            self.model = self.model.to(torch.device('cpu'))
        else:
            return None
        
        # Si background non fourni, on utilise un échantillon aléatoire
        if background_data is None:
            if isinstance(data, torch.Tensor):
                background_data = data[:1]
            else:
                background_data = torch.tensor(data[:1]).float()

        if isinstance(data, torch.Tensor):
            data_input = data
        else:
            data_input = torch.tensor(data).float()

        # Wrapper modèle pour shap (permet d'utiliser uniquement les inputs nécessaires)
        def model_forward(x):
            x = torch.Tensor(x)
            out = self.model(x)
            if use_abs_output:
                out = torch.abs(out)
            return out.detach().cpu().numpy()

        # Utilise Explainer générique (TorchScript compatible)
        explainer = shap.DeepExplainer(self.model, background_data)
        shap_values = explainer.shap_values(data_input, check_additivity=False)

        #self.model.eval()

        return shap_values

    def plot_shap_grid_torch(self, shap_values, X, class_range, features_name, figsize=(8, 6), mode='beeswarm', name=''):
        """
        Affiche un plot SHAP (beeswarm ou bar) pour chaque classe pour un modèle PyTorch.

        Paramètres :
            shap_values (shap.Explanation): SHAP values après explainer(X)
            class_range (list): Indices des classes à afficher
            outname (str): Nom du fichier sans extension
            feature_names (list): Noms des features
            figsize (tuple): Taille de chaque plot
            mode (str): 'beeswarm' ou 'bar'
        """
        
        for class_index in class_range:
        # Extraire SHAP pour une classe
            """class_shap = shap.Explanation(
                values=shap_values.values[:, :, class_index],     # (N, F)
                base_values=shap_values.base_values[:, class_index],  # (N,)
                data=shap_values.data,                            # (N, F)
                feature_names=features_name
            )"""
            if self.model_name in ['ResNet']:
                class_shap = shap_values[:, :, :, :, -1, class_index]
                class_shap = np.mean(class_shap, axis=(2,3))
                X_ = X[:, :, :, :, -1]
                X_ = X_.detach().cpu().numpy()
                X_ = np.mean(X_, axis=(2,3))
            else:
                class_shap = shap_values[:, :, -1, class_index]
                X_ = X[:, :, -1]
            
            plt.figure(figsize=figsize)
            shap.summary_plot(class_shap, X_, features_name, show=False)
            #plt.title(f"Classe {class_index}", fontsize=14)
            save_path = self.dir_output / f"class_{class_index}_{name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ SHAP plot sauvegardé pour classe {class_index} : {save_path}")
            plt.close()

class ModelCNN(Training):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_train_val_test_loader(self, X, y):
        k = self.kernel_size
        ks = self.ks
        H, W = X[0].shape[-2:]
        all_patches = []
        all_labels = []
        all_spatial_mask = []
        all_temporal_mask = []
        for t in range(ks, len(X)):
            for i in range(0, H - k + 1):
                for j in range(0, W - k + 1):
                    if ks > 0:
                        patch = np.asarray([X[t - s - ks][:, i:i + k, j:j + k] for s in range(0, ks + 1, 1)])
                    else:
                        patch = X[t][:, i:i + k, j:j + k][:, :, :, None]
                    
                    patch = np.nan_to_num(patch, nan=0.0)
                    
                    # Pixel central du patch
                    center_i = i + k // 2
                    center_j = j + k // 2
                    label = y[t][center_i, center_j]
                    if not np.isnan(label):
                        all_patches.append(patch)
                        all_labels.append(label)
                        if self.spatial_mask is not None:
                            all_spatial_mask.append(self.spatial_mask[center_i, center_j])
                        if self.temporal_mask is not None:
                            all_temporal_mask.append(t)

        X_tensor = torch.tensor(np.stack(all_patches), dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(np.stack(all_labels), dtype=torch.long, device=self.device)
        all_spatial_mask = torch.Tensor(np.stack(all_spatial_mask))

        # split now
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_dataset(X_tensor, y_tensor, all_spatial_mask)

        if self.under_sampling == 'search':
            X_train, y_train = self.search_zero_samples_proportion(X_train, y_train, X_val, y_val, X_test, y_test)

        self.train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=self.batch_size)
        self.test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=self.batch_size)

    def create_test_loader(self, X, y):
        k = self.kernel_size
        ks = self.ks
        H, W = X[0].shape[-2:]

        all_patches = []
        all_labels = []
        all_spatial_mask = []
        for t in range(ks, len(X)):
            for i in range(0, H - k + 1):
                for j in range(0, W - k + 1):
                    if ks > 0:
                        patch = np.asarray(
                            [X[t - s - ks][:, i:i + k, j:j + k] for s in range(0, ks + 1, 1)]
                        )
                    else:
                        patch = X[t][:, i:i + k, j:j + k][:, :, :, None]
                    
                    patch = np.nan_to_num(patch, nan=0.0)
                    # Pixel central du patch
                    center_i = i + k // 2
                    center_j = j + k // 2
                    label = y[t][center_i, center_j]
                    if not np.isnan(label):
                        all_patches.append(patch)
                        all_labels.append(label)
                        if self.spatial_mask is not None:
                            all_spatial_mask.append(self.spatial_mask[center_i, center_j])

        X_tensor = torch.tensor(np.stack(all_patches), dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(np.stack(all_labels), dtype=torch.long, device=self.device)
        all_spatial_mask = torch.Tensor(np.stack(all_spatial_mask))

        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=1)
        return loader
    
    def create_tensor(self, X, index, pos):
        k = self.kernel_size
        ks = self.ks
        i, j = pos
        if ks > 0:
            patch = np.asarray(
                [X[index - s - ks][None:, i:i + k, j:j + k] for s in range(0, ks + 1, 1)]
            )
        else:
            patch = X[index][:, i:i + k, j:j + k][None, :, :, :, None]
        
        patch = np.nan_to_num(patch, nan=0.0)

        X_tensor = torch.tensor(patch, dtype=torch.float32, device=self.device)
        return X_tensor
    
    def create_tensor_array(self, X, index):
        k = self.kernel_size
        ks = self.ks
        all_data = []
        H, W = X[0].shape[-2:]
        for i in range(0, H - k + 1):
            for j in range(0, W - k + 1):
                if ks > 0:
                    patch = np.asarray(
                        [X[index - s - ks][None, :, i:i + k, j:j + k] for s in range(ks, -1, -1)]
                    )
                else:
                    patch = X[index][:, i:i + k, j:j + k][None, :, :, :, None]
            
                patch = np.nan_to_num(patch, nan=0.0)
                all_data.append(patch)

        X_tensor = torch.tensor(all_data, dtype=torch.float32, device=self.device)
        return X_tensor
    
class ModelGNN(Training):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_train_val_test_loader(self, X, y):
        ks = self.ks
        C, H, W = X[0].shape[1:]
        self.graph = self._create_pixel_graph(H, W)

        all_data, all_labels, all_spatial_mask, all_temporal_mask = [], [], [], []

        for X_seq, y_seq in zip(X, y):
            for t in range(ks, len(X_seq)):
                if ks > 0:
                    x_seq = np.asarray([X_seq[t - s - ks] for s in range(0, ks + 1, 1)])
                else:
                    x_seq = X_seq[t]

                x_flat = x_seq.reshape(x_seq.shape[0], -1).T  # (nb_pixels, features)
                label = y_seq[t].reshape(-1)                  # (nb_pixels,)

                # Masque des pixels valides (aucun nan dans x_flat)
                valid_mask = ~np.isnan(x_flat).any(axis=1)

                x_flat_clean = x_flat[valid_mask]
                label_clean = label[valid_mask]

                all_data.append(x_flat_clean)
                all_labels.append(label_clean)
                if self.spatial_mask is not None:
                    all_spatial_mask.append(self.spatial_mask.reshape(-1))
                if self.temporal_mask is not None:
                    all_temporal_mask.append(t)

        X_tensor = torch.tensor(np.stack(all_data), dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(np.stack(all_labels), dtype=torch.long, device=self.device)
        all_spatial_mask = torch.Tensor(np.stack(all_spatial_mask))

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_dataset(X_tensor, y_tensor, all_spatial_mask)

        self.train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=self.batch_size)
        self.test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=self.batch_size)

    def create_tensor(self, X, index, pos):
        ks = self.ks
        i, j = pos

        all_data = []
        all_spatial_mask = []
        if ks > 0:
            x_seq = np.concatenate([X[index - s - ks][:, i, j] for s in range(0, ks + 1, 1)], axis=0)
        else:
            x_seq = X[index]

        x_flat = x_seq.reshape(x_seq.shape[0], -1).T  # (nb_pixels, features)

        # Masque des pixels valides (aucun nan dans x_flat)
        valid_mask = ~np.isnan(x_flat).any(axis=1)

        x_flat_clean = x_flat[valid_mask]

        all_data.append(x_flat_clean)
        if self.spatial_mask is not None:
            all_spatial_mask.append(self.spatial_mask.reshape(-1))

        X_tensor = torch.tensor(np.stack(all_data), dtype=torch.float32, device=self.device)
        all_spatial_mask = torch.Tensor(np.stack(all_spatial_mask))

        return X_tensor

    def create_test_loader(self, X, y):
        k = self.kernel_size
        ks = self.ks
        H, W = X[0].shape[-2:]

        all_patches = []
        all_labels = []
        all_spatial_mask = []
        for t in range(ks, len(X)):
            for i in range(0, H - k + 1):
                for j in range(0, W - k + 1):
                    if ks > 0:
                        patch = np.asarray(
                            [X[t - s - ks][:, i:i + k, j:j + k] for s in range(0, ks + 1, 1)]
                        )
                    else:
                        patch = X[t][:, i:i + k, j:j + k][:, :, :, None]
                    
                    patch = np.nan_to_num(patch, nan=0.0)
                    # Pixel central du patch
                    center_i = i + k // 2
                    center_j = j + k // 2
                    label = y[t][center_i, center_j]
                    if not np.isnan(label):
                        all_patches.append(patch)
                        all_labels.append(label)
                        if self.spatial_mask is not None:
                            all_spatial_mask.append(self.spatial_mask[center_i, center_j])

        X_tensor = torch.tensor(np.stack(all_patches), dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(np.stack(all_labels), dtype=torch.long, device=self.device)
        all_spatial_mask = torch.Tensor(np.stack(all_spatial_mask))

        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=1)
        return loader
    
    def _create_pixel_graph(self, H, W):
        edges = []
        for i in range(H):
            for j in range(W):
                idx = i * W + j
                if i + 1 < H:
                    edges.append([idx, (i+1)*W + j])
                if j + 1 < W:
                    edges.append([idx, i*W + (j+1)])
        return torch.tensor(np.array(edges).T, dtype=torch.long)
    
    def create_tensor_array(self, X, index):
        k = self.kernel_size
        ks = self.ks
        all_data = []
        if ks > 0:
            x_seq = np.asarray([X[index - s - ks] for s in range(0, ks + 1, 1)])
        else:
            x_seq = X[index]

        x_flat = x_seq.reshape(x_seq.shape[0], -1).T  # (nb_pixels, features)

        # Masque des pixels valides (aucun nan dans x_flat)
        valid_mask = ~np.isnan(x_flat).any(axis=1)

        x_flat_clean = x_flat[valid_mask]

        all_data.append(x_flat_clean)

        X_tensor = torch.tensor(all_data, dtype=torch.float32, device=self.device)
        return X_tensor

class Model_Torch(Training):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_train_val_test_loader(self, X, y):
        ks = self.ks
        C, H, W = X[0].shape

        all_data, all_labels, all_spatial_mask, all_temporal_mask = [], [], [], []

        for t in range(ks, len(X)):
            if ks > 0:
                x_seq = np.asarray([X[t - s - ks] for s in range(0, ks + 1, 1)])
            else:
                x_seq = X[t][np.newaxis, :, :, :]

            x_flat = x_seq.reshape(x_seq.shape[0], x_seq.shape[1], -1).T  # (nb_pixels, features, ks+1)
            label = y[t].reshape(-1)                  # (nb_pixels,)
            
            # Masque des pixels valides (aucun nan dans x_flat)
            valid_mask = ~np.isnan(x_flat).any(axis=(1, 2))
            
            x_flat_clean = x_flat[valid_mask]
            label_clean = label[valid_mask]

            all_data.append(x_flat_clean)
            all_labels.append(label_clean)
            if self.spatial_mask is not None:
                all_spatial_mask.append(self.spatial_mask.reshape(-1)[valid_mask])
            if self.temporal_mask is not None:
                    all_temporal_mask.append(t)

        X_tensor = torch.tensor(np.stack(all_data), dtype=torch.float32, device=self.device).view(-1, C, ks + 1)
        y_tensor = torch.tensor(np.stack(all_labels), dtype=torch.long, device=self.device).view(-1)
        all_spatial_mask = torch.Tensor(np.stack(all_spatial_mask)).view(-1)
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_dataset(X_tensor, y_tensor, all_spatial_mask)

        if self.under_sampling == 'search':
            X_train, y_train = self.search_zero_samples_proportion(X_train, y_train, X_val, y_val, X_test, y_test)
        
        self.train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=self.batch_size)
        self.test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=self.batch_size)

    def create_loader(self, X, y):
        ks = self.ks
        C, H, W = X[0].shape[1:]

        all_data, all_labels, all_spatial_mask = [], [], []

        for X_seq, y_seq in zip(X, y):
            for t in range(ks, len(X_seq)):
                if ks > 0:
                    x_seq = np.asarray([X_seq[t - s - ks] for s in range(0, ks + 1, 1)])
                else:
                    x_seq = X_seq[t]

                x_flat = x_seq.reshape(x_seq.shape[0], x_seq.shape[1], -1).T  # (nb_pixels, features, ks+1)
                label = y_seq[t].reshape(-1)                  # (nb_pixels,)

                # Masque des pixels valides (aucun nan dans x_flat)
                valid_mask = ~np.isnan(x_flat).any(axis=1)

                x_flat_clean = x_flat[valid_mask]
                label_clean = label[valid_mask]

                all_data.append(x_flat_clean)
                all_labels.append(label_clean)
                if self.spatial_mask is not None:
                    all_spatial_mask.append(self.spatial_mask.reshape(-1))

        X_tensor = torch.tensor(np.stack(all_data), dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(np.stack(all_labels), dtype=torch.long, device=self.device)
        all_spatial_mask = torch.Tensor(np.stack(all_spatial_mask))

        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=1)
        return loader
    
    def create_tensor(self, X, index, pos):
        ks = self.ks
        i, j = pos

        if ks > 0:
            x_seq = np.asarray([X[index - s - ks][None, :, i, j] for s in range(0, ks + 1, 1)])
        else:
            x_seq = X[index][None, :, i, j]

        if x_seq.ndim == 2:
            x_seq = x_seq[np.newaxis, :, :]
            
        x_flat = np.moveaxis(x_seq, 0, 2)  # (nb_pixels, features, ks+1)

        valid_mask = ~np.isnan(x_flat).any(axis=(1, 2))

        x_flat_clean = x_flat[valid_mask]

        X_tensor = torch.tensor(x_flat_clean, dtype=torch.float32, device=self.device)

        return X_tensor
    
    """def create_tensor_array(self, X, index):
        k = self.kernel_size
        ks = self.ks
        if ks > 0:
            x_seq = np.asarray([X[index - s - ks] for s in range(0, ks + 1, 1)])
        else:
            x_seq = X[index]

        x_flat = x_seq.reshape(x_seq.shape[0], x_seq.shape[1], -1).T   # (nb_pixels, features)
        print('here', x_flat.shape, x_seq.shape)

        # Masque des pixels valides (aucun nan dans x_flat)
        valid_mask = ~np.isnan(x_flat).any(axis=(1, 2))

        x_flat_clean = x_flat[valid_mask]

        X_tensor = torch.tensor(x_flat_clean, dtype=torch.float32, device=self.device)
        
        print(X_tensor.shape)

        return torch.tensor"""