# coding: utf-8
"""Utility training class extracted from pytorch_model."""
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import shap
from pathlib import Path

from GNN.tools import calculate_area_under_curve
from GNN.config import graph_id_index
from GNN.tools import under_prediction_score, over_prediction_score, iou_score, evaluate_metrics, calculate_ic95
from tools import check_and_create_path, save_object, read_object
from forecasting_models.pytorch.tools_2 import get_loss_function

class Training:
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

    def calculate_loss(self, criterion, output, target, weights, label, tolong=True):
        def compute_single_loss(out, tar, wei, mask_id=None):
            if self.task_type == 'regression':
                tar = tar.view(out.shape)
                wei = wei.view(out.shape)
                tar = torch.masked_select(tar, wei.gt(0))
                out = torch.masked_select(out, wei.gt(0))
                wei = torch.masked_select(wei, wei.gt(0))
                return criterion(out, tar, wei)
            else:
                wei = wei.long()
                if not self.student_train:
                    tar = tar.long()
                tar = tar[wei.gt(0)]
                out = out[wei.gt(0)]
                if mask_id is not None:
                    mask_id = mask_id[wei.gt(0)]
                wei = torch.masked_select(wei, wei.gt(0))
                if tolong:
                    tar = tar.long()
                if self.loss in ['kappa', 'cdw', 'mcewk']:
                    tar = tar.to('cpu')
                    out = out.to('cpu')
                if mask_id is not None:
                    return criterion(out, tar, id_mask=mask_id)
                else:
                    return criterion(out, tar)

        if 'ID' in self.loss:
            id_mask = label[:, criterion.id, -1]
        else:
            id_mask = None

        base_loss = compute_single_loss(output, target, weights, id_mask)

        if 'area' in self.loss:
            area_mask = label[:, graph_id_index, -1]
            unique_ids = torch.unique(area_mask)
            values = []
            for aid in unique_ids:
                m = area_mask == aid
                if m.sum() == 0:
                    continue
                l = compute_single_loss(output[m], target[m], weights[m], None)
                values.append(l)
            if len(values) > 0:
                vals = torch.stack(values)
                area_score = torch.trapz(vals)
            else:
                area_score = torch.tensor(0.0, device=output.device)
            if 'area-global' in self.loss:
                loss = area_score * base_loss
            else:
                loss = area_score
        else:
            loss = base_loss
        return loss

    def make_model(self, graph, custom_model_params):
        model, params = make_model(self.model_name, len(self.features_name), len(self.features_name),
                                   graph, dropout, activation,
                                   self.ks,
                                   out_channels=self.out_channels,
                                   task_type=self.task_type,
                                   device=device, num_lstm_layers=num_lstm_layers,
                                   custom_model_params=custom_model_params)
        if getattr(self, 'model_params', None) is None:
            self.model_params = params
        return model, params

    def func_epoch(self, train_loader, val_loader, optimizer, criterion, criterion_val):
        train_loss = self.launch_train_loader(train_loader, criterion, optimizer)
        if val_loader is not None:
            val_loss = self.launch_val_test_loader(val_loader, criterion_val)
        else:
            val_loss = train_loss.item()
        return val_loss, train_loss

    def split_dataset(self, dataset, nb, reset=True):
        positive_mask = dataset[self.target_name] > 0
        non_fire_mask = dataset[self.target_name] == 0
        df_positive = dataset[positive_mask]
        df_non_fire = dataset[non_fire_mask]
        nb = min(len(df_non_fire), nb)
        if getattr(self, 'n_run', 1) == 1:
            sampled_indices = np.random.RandomState(42).choice(len(df_non_fire), nb, replace=False)
        else:
            sampled_indices = np.random.RandomState().choice(len(df_non_fire), nb, replace=False)
        df_non_fire_sampled = df_non_fire.iloc[sampled_indices]
        df_combined = pd.concat([df_positive, df_non_fire_sampled])
        if reset:
            df_combined.reset_index(drop=True, inplace=True)
        return df_combined

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
            for uid in uids:
                mask = (id_mask == uid)
                pred_mask = pred[mask]
                y_mask = y[mask]
                if np.any(y_mask > 0):
                    under_score = under_prediction_score(y_mask, pred_mask)
                    over_score = over_prediction_score(y_mask, pred_mask)
                    iou_val = iou_score(y_mask, pred_mask)
                    under_prediction_score_value.append(under_score)
                    over_prediction_score_value.append(over_score)
                    iou.append(iou_val)
                else:
                    pass
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
                iou = jaccard_score((y_g > 0).astype(int), (pred_g > 0).astype(int))
                f1 = f1_score((y_g > 0).astype(int), (pred_g > 0).astype(int))
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

    def search_samples_proportion(self, graph, df_train, df_val, df_test, is_unknowed_risk, reset=True, custom_model_params=None, use_log=True):
        check_and_create_path(self.dir_log)
        if not is_unknowed_risk:
            test_percentage = np.arange(0.05, 1.05, 0.05)
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
            if (self.dir_log / 'metrics.pkl').is_file():
                find_log = True
                data_log = read_object('metrics.pkl', self.dir_log)
        if data_log is not None:
            try:
                self.metrics = data_log
                under_prediction_score_scores = self.metrics['under_prediction_scores']
                over_prediction_score_scores = self.metrics['over_prediction_scores']
                iou_scores = self.metrics['iou_score']
            except Exception:
                self.metrics = {}
                data_log = None
        doSearch = True
        if data_log is not None:
            for i in range(0, len(iou_scores) - 1):
                try:
                    if data_log[test_percentage[i]]['iou_val'] > data_log[test_percentage[i + 1]]['iou_val']:
                        doSearch = False
                except Exception:
                    doSearch = True
                    break
            if doSearch:
                start_test = np.argmax(iou_scores)
        else:
            start_test = 0
        if doSearch:
            last_score = -math.inf if start_test == 0 else iou_scores[start_test - 1]
            y_ori = df_train[self.target_name].values
            for i in range(start_test, test_percentage.shape[0]):
                tp = test_percentage[i]
                if tp not in self.metrics:
                    self.metrics[tp] = {'f1': [], 'iou': [], 'iou_val': [], 'prec': [], 'recall': [], 'normalized_iou': [], 'normalized_f1': []}
                df_train_copy = df_train.copy(deep=True)
                if not is_unknowed_risk:
                    nb = int(tp * y_ori[y_ori == 0].shape[0])
                else:
                    nb = int(tp * len(X[(X['potential_risk'] > 0) & (y_ori == 0)]))
                df_combined = self.split_dataset(df_train_copy, nb, reset=False)
                df_train_copy['weight'] = 0
                df_train_copy.loc[df_combined.index, 'weight'] = 1
                copy_model = deepcopy(self)
                copy_model.under_sampling = 'full'
                copy_model.create_train_val_test_loader(graph, df_train_copy, df_val, df_test, features_importance=False, custom_model_params=custom_model_params)
                copy_model.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose=False, custom_model_params=custom_model_params)
                test_output, y = copy_model._predict_test_loader(copy_model.val_loader)
                prediction = test_output.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                metrics_run = evaluate_metrics(pd.DataFrame({self.target_name: y[:, -1]}), self.target_name, prediction)
                under_prediction_score_value = under_prediction_score(y[:, -1], prediction)
                over_prediction_score_value = over_prediction_score(y[:, -1], prediction)
                self.metrics[tp]['iou_val'].append(metrics_run['iou'])
                test_output, y = copy_model._predict_test_loader(copy_model.test_loader)
                prediction = test_output.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                metrics_run = evaluate_metrics(pd.DataFrame({self.target_name: y[:, -1]}), self.target_name, prediction)
                self.metrics[tp]['iou'].append(metrics_run['iou'])
                self.metrics[tp]['f1'].append(metrics_run['f1'])
                self.metrics[tp]['recall'].append(metrics_run['recall'])
                self.metrics[tp]['prec'].append(metrics_run['prec'])
                self.metrics[tp]['normalized_iou'].append(metrics_run['normalized_iou'])
                self.metrics[tp]['normalized_f1'].append(metrics_run['normalized_f1'])
                if iou_scores:
                    iou_scores[i] = metrics_run['iou']
                    under_prediction_score_scores[i] = under_prediction_score_value
                    over_prediction_score_scores[i] = over_prediction_score_value
                else:
                    iou_scores.append(metrics_run['iou'])
                    under_prediction_score_scores.append(under_prediction_score_value)
                    over_prediction_score_scores.append(over_prediction_score_value)
                save_object(self.metrics, 'metrics.pkl', self.dir_log)
                if metrics_run['iou'] > last_score:
                    last_score = metrics_run['iou']
                else:
                    break
        index_max = np.argmax(iou_scores)
        best_tp = test_percentage[index_max]
        self.metrics['iou_score'] = iou_scores
        self.metrics['test_percentage'] = test_percentage
        self.metrics['under_prediction_scores'] = under_prediction_score_scores
        self.metrics['over_prediction_scores'] = over_prediction_score_scores
        self.metrics['best_tp'] = best_tp
        self.metrics['run'] = self.n_run
        save_object(self.metrics, 'metrics.pkl', self.dir_log)
        if is_unknowed_risk:
            plt.figure(figsize=(15, 7))
            plt.plot(test_percentage[:len(under_prediction_score_scores)], under_prediction_score_scores, label='under_prediction')
            plt.plot(test_percentage[:len(under_prediction_score_scores)], over_prediction_score_scores, label='over_prediction')
            plt.plot(test_percentage[:len(under_prediction_score_scores)], iou_scores, label='Iou')
            plt.xticks(test_percentage)
            plt.xlabel('Percentage of Unknowed sample')
            plt.ylabel('IOU Score')
            plt.axvline(x=best_tp, color='r', linestyle='--', label=f'Best TP: {best_tp:.2f}')
            plt.legend()
            plt.savefig(self.dir_log / f'{self.name}_unknowned_scores_per_percentage.png')
            plt.close()
        else:
            plt.figure(figsize=(15, 7))
            plt.plot(test_percentage[:len(under_prediction_score_scores)], under_prediction_score_scores, label='under_prediction')
            plt.plot(test_percentage[:len(under_prediction_score_scores)], over_prediction_score_scores, label='over_prediction')
            plt.plot(test_percentage[:len(under_prediction_score_scores)], iou_scores, label='Iou')
            plt.xticks(test_percentage)
            plt.xlabel('Percentage of Binary sample')
            plt.ylabel('IOU Score')
            plt.axvline(x=best_tp, color='r', linestyle='--', label=f'Best TP: {best_tp:.2f}')
            plt.legend()
            plt.savefig(self.dir_log / f'{self.name}_scores_per_percentage.png')
            plt.close()
        return best_tp, find_log

    def plot_train_val_loss(self, epochs, train_loss_list, val_loss_list, dir_log):
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, val_loss_list, label='Validation Loss', color='blue')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation Loss over Epochs')
        plt.savefig(dir_log / 'Validation.png')
        plt.close('all')
        plt.plot(epochs, train_loss_list, label='Training Loss', color='red')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.savefig(dir_log / 'Training.png')
        plt.close('all')

    def update_weight(self, weight):
        assert self.model is not None
        if not isinstance(weight, dict):
            raise ValueError('The provided weight must be a dictionary containing model parameters.')
        model_state_dict = self.model.state_dict()
        missing_keys = [key for key in weight.keys() if key not in model_state_dict]
        if missing_keys:
            raise KeyError(f'Some keys in the provided weights do not match the model\'s parameters: {missing_keys}')
        self.model.load_state_dict(weight)

    def update_model(self, model):
        self.model = deepcopy(model)

    def get_loss(self, loss_name):
        loss_params = {'num_classes': 5}
        return get_loss_function(loss_name, **loss_params)

    def shapley_additive_explanation(self, df, outname, dir_output, mode='bar', figsize=(50, 25), samples=None, samples_name=None):
        from .pytorch_model import get_numpy_data, WrapperModel
        if hasattr(self, 'use_temporal_as_edges'):
            use_temporal_as_edges = self.use_temporal_as_edges
        else:
            use_temporal_as_edges = None
        Xst, e = get_numpy_data(self.graph, df, self.features_name, use_temporal_as_edges, self.ks)
        Xst = torch.Tensor(Xst).to(self.device)
        B, F_, T = Xst.shape
        Xst_flat = Xst.reshape((B, F_*T))
        df_features = []
        explainer = shap.DeepExplainer(WrapperModel(self.model, F_, T, e).to(self.device), Xst_flat)
        shap_values = explainer.shap_values(Xst_flat)
        n_classes = self.out_channels
        if n_classes == 1:
            shap_values = shap_values[:, :, np.newaxis]
        shap_values = np.asarray(shap_values)
        shap_values = np.reshape(shap_values, (n_classes, B, F_, T))
        shap_values = shap_values[:, :, :, -1]
        shap_values = np.moveaxis(shap_values, 0, 2)
        for class_idx in range(n_classes):
            shap_mean_abs = np.mean(np.abs(shap_values[:, :, class_idx]), axis=0)
            shap_std_abs = np.std(np.abs(shap_values[:, :, class_idx]), axis=0)
            df_shap = pd.DataFrame({
                'mean_abs_shap': shap_mean_abs,
                'stdev_abs_shap': shap_std_abs,
                'name': self.features_name
            }).sort_values('mean_abs_shap', ascending=False)
            df_shap['class'] = class_idx
            df_features.append(df_shap)
            plt.figure(figsize=figsize)
        df_features = pd.concat(df_features)
        save_object(df_features, 'features_importance.pkl', dir_output)
