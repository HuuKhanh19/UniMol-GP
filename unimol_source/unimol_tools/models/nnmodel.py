# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import os

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from ..utils import logger
from .loss import FocalLossWithLogits, GHMC_Loss, MAEwithNan, myCrossEntropyLoss
from .unimol import UniMolModel
from .unimolv2 import UniMolV2Model

NNMODEL_REGISTER = {
    'unimolv1': UniMolModel,
    'unimolv2': UniMolV2Model,
}

LOSS_RREGISTER = {
    'classification': myCrossEntropyLoss,
    'multiclass': myCrossEntropyLoss,
    'regression': nn.MSELoss(),
    'multilabel_classification': {
        'bce': nn.BCEWithLogitsLoss(),
        'ghm': GHMC_Loss(bins=10, alpha=0.5),
        'focal': FocalLossWithLogits,
    },
    'multilabel_regression': MAEwithNan,
}


def classification_activation(x):
    return F.softmax(x, dim=-1)[:, 1:]


def multiclass_activation(x):
    return F.softmax(x, dim=-1)


def regression_activation(x):
    return x


def multilabel_classification_activation(x):
    return F.sigmoid(x)


def multilabel_regression_activation(x):
    return x


ACTIVATION_FN = {
    'classification': classification_activation,
    'multiclass': multiclass_activation,
    'regression': regression_activation,
    'multilabel_classification': multilabel_classification_activation,
    'multilabel_regression': multilabel_regression_activation,
}
OUTPUT_DIM = {
    'classification': 2,
    'regression': 1,
}


class NNModel(object):
    """A :class:`NNModel` class is responsible for initializing the model"""

    def __init__(self, data, trainer, **params):
        """
        Initializes the neural network model with the given data and parameters.

        :param data: (dict) Contains the dataset information, including features and target scaling.
        :param trainer: (object) An instance of a training class, responsible for managing training processes.
        :param params: Various additional parameters used for model configuration.

        The model is configured based on the task type and specific parameters provided.
        """
        self.data = data
        self.num_classes = self.data['num_classes']
        self.target_scaler = self.data['target_scaler']
        self.features = data['unimol_input']
        self.model_name = params.get('model_name', 'unimolv1')
        self.data_type = params.get('data_type', 'molecule')
        self.loss_key = params.get('loss_key', None)
        self.trainer = trainer
        # self.splitter = self.trainer.splitter
        self.model_params = params.copy()
        self.task = params['task']
        if self.task in OUTPUT_DIM:
            self.model_params['output_dim'] = OUTPUT_DIM[self.task]
        elif self.task == 'multiclass':
            self.model_params['output_dim'] = self.data['multiclass_cnt']
        else:
            self.model_params['output_dim'] = self.num_classes
        self.model_params['device'] = self.trainer.device
        self.cv = dict()
        self.metrics = self.trainer.metrics
        if self.task == 'multilabel_classification':
            if self.loss_key is None:
                self.loss_key = 'focal'
            self.loss_func = LOSS_RREGISTER[self.task][self.loss_key]
        else:
            self.loss_func = LOSS_RREGISTER[self.task]
        self.activation_fn = ACTIVATION_FN[self.task]
        self.save_path = self.trainer.save_path
        self.trainer.set_seed(self.trainer.seed)
        self.model = self._init_model(**self.model_params)

    def _init_model(self, model_name, **params):
        """
        Initializes the neural network model based on the provided model name and parameters.

        :param model_name: (str) The name of the model to initialize.
        :param params: Additional parameters for model configuration.

        :return: An instance of the specified neural network model.
        :raises ValueError: If the model name is not recognized.
        """
        if self.task in ['regression', 'multilabel_regression']:
            params['pooler_dropout'] = 0
            logger.debug("set pooler_dropout to 0 for regression task")
        else:
            pass
        freeze_layers = params.get('freeze_layers', None)
        freeze_layers_reversed = params.get('freeze_layers_reversed', False)
        if model_name in NNMODEL_REGISTER:
            model = NNMODEL_REGISTER[model_name](**params)
            if isinstance(freeze_layers, str):
                freeze_layers = freeze_layers.replace(' ', '').split(',')
            if isinstance(freeze_layers, list):
                for layer_name, layer_param in model.named_parameters():
                    should_freeze = any(
                        layer_name.startswith(freeze_layer)
                        for freeze_layer in freeze_layers
                    )
                    layer_param.requires_grad = not (
                        freeze_layers_reversed ^ should_freeze
                    )
        else:
            raise ValueError('Unknown model: {}'.format(self.model_name))
        return model

    def collect_data(self, X, y, idx):
        """
        Collects and formats the training or validation data.

        :param X: (np.ndarray or dict) The input features, either as a numpy array or a dictionary of tensors.
        :param y: (np.ndarray) The target values as a numpy array.
        :param idx: Indices to select the specific data samples.

        :return: A tuple containing processed input data and target values.
        :raises ValueError: If X is neither a numpy array nor a dictionary.
        """
        assert isinstance(y, np.ndarray), 'y must be numpy array'
        if isinstance(X, np.ndarray):
            return torch.from_numpy(X[idx]).float(), torch.from_numpy(y[idx])
        elif isinstance(X, list):
            return {k: v[idx] for k, v in X.items()}, torch.from_numpy(y[idx])
        else:
            raise ValueError('X must be numpy array or dict')

    # def run(self):
    #     """
    #     Executes the training process of the model. This involves data preparation,
    #     model training, validation, and computing metrics for each fold in cross-validation.
    #     """
    #     logger.info("start training Uni-Mol:{}".format(self.model_name))
    #     X = np.asarray(self.features)
    #     y = np.asarray(self.data['target'])
    #     group = (
    #         np.asarray(self.data['group']) if self.data['group'] is not None else None
    #     )
    #     if self.task == 'classification':
    #         y_pred = np.zeros_like(y.reshape(y.shape[0], self.num_classes)).astype(
    #             float
    #         )
    #     else:
    #         y_pred = np.zeros((y.shape[0], self.model_params['output_dim']))
    #     for fold, (tr_idx, te_idx) in enumerate(self.data['split_nfolds']):
    #         X_train, y_train = X[tr_idx], y[tr_idx]
    #         X_valid, y_valid = X[te_idx], y[te_idx]
    #         traindataset = NNDataset(X_train, y_train)
    #         validdataset = NNDataset(X_valid, y_valid)
    #         if fold > 0:
    #             # need to initalize model for next fold training
    #             self.model = self._init_model(**self.model_params)

    #         # TODO: move the following code to model.load_pretrained_weights
    #         if self.model_params.get('load_model_dir', None) is not None:
    #             load_model_path = os.path.join(
    #                 self.model_params['load_model_dir'], f'model_{fold}.pth'
    #             )
    #             model_dict = torch.load(
    #                 load_model_path, map_location=self.model_params['device']
    #             )["model_state_dict"]
    #             if (
    #                 model_dict['classification_head.out_proj.weight'].shape[0]
    #                 != self.model.output_dim
    #             ):
    #                 current_model_dict = self.model.state_dict()
    #                 model_dict = {
    #                     k: v
    #                     for k, v in model_dict.items()
    #                     if k in current_model_dict
    #                     and 'classification_head.out_proj' not in k
    #                 }
    #                 current_model_dict.update(model_dict)
    #                 logger.info(
    #                     "The output_dim of the model is different from the loaded model, only load the common part of the model"
    #                 )
    #                 self.model.load_state_dict(model_dict, strict=False)
    #             else:
    #                 self.model.load_state_dict(model_dict)

    #             logger.info("load model success from {}".format(load_model_path))
    #         _y_pred = self.trainer.fit_predict(
    #             self.model,
    #             traindataset,
    #             validdataset,
    #             self.loss_func,
    #             self.activation_fn,
    #             self.save_path,
    #             fold,
    #             self.target_scaler,
    #         )
    #         y_pred[te_idx] = _y_pred

    #         if 'multiclass_cnt' in self.data:
    #             label_cnt = self.data['multiclass_cnt']
    #         else:
    #             label_cnt = None

    #         logger.info(
    #             "fold {0}, result {1}".format(
    #                 fold,
    #                 self.metrics.cal_metric(
    #                     self.data['target_scaler'].inverse_transform(y_valid),
    #                     self.data['target_scaler'].inverse_transform(_y_pred),
    #                     label_cnt=label_cnt,
    #                 ),
    #             )
    #         )

    #     self.cv['pred'] = y_pred
    #     self.cv['metric'] = self.metrics.cal_metric(
    #         self.data['target_scaler'].inverse_transform(y),
    #         self.data['target_scaler'].inverse_transform(self.cv['pred']),
    #     )
    #     self.dump(self.cv['pred'], self.save_path, 'cv.data')
    #     self.dump(self.cv['metric'], self.save_path, 'metric.result')
    #     logger.info("Uni-Mol metrics score: \n{}".format(self.cv['metric']))
    #     logger.info("Uni-Mol & Metric result saved!")

    def run(self):
        """
        Train the model with single train/valid split (no k-fold).
        Handles conformer-expanded features correctly.
        """
        logger.info("start training Uni-Mol:{}".format(self.model_name))
        X = np.asarray(self.features)
        y = np.asarray(self.data['target'])
        conf_per_mol = self._get_conf_per_mol()
 
        if conf_per_mol > 1:
            logger.info(f"Detected {conf_per_mol} conformers per molecule "
                        f"(features={len(X)}, molecules={len(y)})")
 
        # Use first (and only) split
        tr_idx, te_idx = self.data['split_nfolds'][0]
 
        # Map molecule indices to conformer indices
        tr_conf_idx = self._mol_idx_to_conf_idx(tr_idx, conf_per_mol)
        X_train = X[tr_conf_idx]
        y_train = np.repeat(y[tr_idx], conf_per_mol, axis=0) if conf_per_mol > 1 else y[tr_idx]
        traindataset = NNDataset(X_train, y_train)
 
        has_valid = hasattr(te_idx, '__len__') and len(te_idx) > 0
        if has_valid:
            te_conf_idx = self._mol_idx_to_conf_idx(te_idx, conf_per_mol)
            X_valid = X[te_conf_idx]
            y_valid = np.repeat(y[te_idx], conf_per_mol, axis=0) if conf_per_mol > 1 else y[te_idx]
        else:
            X_valid, y_valid = X_train, y_train
        validdataset = NNDataset(X_valid, y_valid)
 
        if self.task == 'classification':
            y_pred = np.zeros_like(
                y.reshape(y.shape[0], self.num_classes)
            ).astype(float)
        else:
            y_pred = np.zeros((y.shape[0], self.model_params['output_dim']))
 
        # Load pretrained weights for transfer learning
        if self.model_params.get('load_model_dir', None) is not None:
            load_model_path = os.path.join(
                self.model_params['load_model_dir'], 'model_0.pth'
            )
            if os.path.exists(load_model_path):
                model_dict = torch.load(
                    load_model_path, map_location=self.model_params['device']
                )["model_state_dict"]
                if (
                    model_dict['classification_head.out_proj.weight'].shape[0]
                    != self.model.output_dim
                ):
                    current_model_dict = self.model.state_dict()
                    model_dict = {
                        k: v
                        for k, v in model_dict.items()
                        if k in current_model_dict
                        and 'classification_head.out_proj' not in k
                    }
                    current_model_dict.update(model_dict)
                    logger.info("output_dim mismatch, only load common part")
                    self.model.load_state_dict(model_dict, strict=False)
                else:
                    self.model.load_state_dict(model_dict)
                logger.info("load model success from {}".format(load_model_path))
 
        # Train single fold
        _y_pred = self.trainer.fit_predict(
            self.model,
            traindataset,
            validdataset,
            self.loss_func,
            self.activation_fn,
            self.save_path,
            fold=0,
            target_scaler=self.target_scaler,
        )
 
        # Average conformer predictions back to per-molecule
        if has_valid:
            if conf_per_mol > 1:
                n_valid_mol = len(te_idx)
                _y_pred_mol = _y_pred.reshape(n_valid_mol, conf_per_mol, -1).mean(axis=1)
            else:
                _y_pred_mol = _y_pred
            y_pred[te_idx] = _y_pred_mol
 
        self.cv['pred'] = y_pred
 
        # Compute and log metrics
        if 'multiclass_cnt' in self.data:
            label_cnt = self.data['multiclass_cnt']
        else:
            label_cnt = None
 
        if has_valid:
            y_valid_mol = y[te_idx]
            metric_result = self.metrics.cal_metric(
                self.data['target_scaler'].inverse_transform(y_valid_mol),
                self.data['target_scaler'].inverse_transform(_y_pred_mol),
                label_cnt=label_cnt,
            )
            logger.info("fold 0, result {}".format(metric_result))
            self.cv['metric'] = metric_result
        else:
            self.cv['metric'] = self.metrics.cal_metric(
                self.data['target_scaler'].inverse_transform(y),
                self.data['target_scaler'].inverse_transform(y_pred),
            )
 
        self.dump(self.cv['pred'], self.save_path, 'cv.data')
        self.dump(self.cv['metric'], self.save_path, 'metric.result')
        logger.info("Uni-Mol metrics score: \n{}".format(self.cv['metric']))
        logger.info("Uni-Mol & Metric result saved!")

    def dump(self, data, dir, name):
        """
        Saves the specified data to a file.

        :param data: The data to be saved.
        :param dir: (str) The directory where the data will be saved.
        :param name: (str) The name of the file to save the data.
        """
        path = os.path.join(dir, name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        joblib.dump(data, path)
    
    def _get_conf_per_mol(self):
        """Detect conformer-per-molecule ratio."""
        n_features = len(self.features)
        n_molecules = len(self.data['target'])
        if n_features == n_molecules:
            return 1
        elif n_features > n_molecules and n_features % n_molecules == 0:
            return n_features // n_molecules
        else:
            logger.warning(
                f"Features ({n_features}) and targets ({n_molecules}) "
                f"size mismatch, assuming 1 conformer per molecule."
            )
            return 1
 
    def _mol_idx_to_conf_idx(self, mol_indices, conf_per_mol):
        """Map molecule indices to conformer indices in the flat features list."""
        if conf_per_mol == 1:
            return mol_indices
        conf_indices = []
        for i in mol_indices:
            conf_indices.extend(range(i * conf_per_mol, (i + 1) * conf_per_mol))
        return np.array(conf_indices, dtype=np.int64)

    # def evaluate(self, trainer=None, checkpoints_path=None):
    #     """
    #     Evaluates the model by making predictions on the test set and averaging the results.

    #     :param trainer: An optional trainer instance to use for prediction.
    #     :param checkpoints_path: (str) The path to the saved model checkpoints.
    #     """
    #     logger.info("start predict NNModel:{}".format(self.model_name))
    #     testdataset = NNDataset(self.features, np.asarray(self.data['target']))
    #     for fold in range(self.data['kfold']):
    #         _y_pred, _, __ = trainer.predict(
    #             self.model,
    #             testdataset,
    #             self.loss_func,
    #             self.activation_fn,
    #             self.save_path,
    #             fold,
    #             self.target_scaler,
    #             epoch=1,
    #             load_model=True,
    #         )
    #         if fold == 0:
    #             y_pred = np.zeros_like(_y_pred)
    #         y_pred += _y_pred
    #     y_pred /= self.data['kfold']
    #     self.cv['test_pred'] = y_pred

    def evaluate(self, trainer=None, checkpoints_path=None):
        """
        Evaluate model — load model_0.pth only, no k-fold ensemble.
        Handles conformer-expanded features.
        """
        logger.info("start predict NNModel:{}".format(self.model_name))
        target = np.asarray(self.data['target'])
        n_molecules = len(target)
        conf_per_mol = self._get_conf_per_mol()
 
        if conf_per_mol > 1:
            # Expand labels to match conformer count
            expanded_target = np.repeat(target, conf_per_mol, axis=0)
            testdataset = NNDataset(self.features, expanded_target)
        else:
            testdataset = NNDataset(self.features, target)
 
        y_pred, _, __ = trainer.predict(
            self.model,
            testdataset,
            self.loss_func,
            self.activation_fn,
            self.save_path,
            fold=0,
            target_scaler=self.target_scaler,
            epoch=1,
            load_model=True,
        )
 
        # Average conformer predictions back to per-molecule
        if conf_per_mol > 1:
            y_pred = y_pred.reshape(n_molecules, conf_per_mol, -1).mean(axis=1)
 
        self.cv['test_pred'] = y_pred

    def count_parameters(self, model):
        """
        Counts the number of trainable parameters in the model.

        :param model: The model whose parameters are to be counted.

        :return: (int) The number of trainable parameters.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def NNDataset(data, label=None):
    """
    Creates a dataset suitable for use with PyTorch models.

    :param data: The input data.
    :param label: Optional labels corresponding to the input data.

    :return: An instance of TorchDataset.
    """
    return TorchDataset(data, label)


class TorchDataset(Dataset):
    """
    A custom dataset class for PyTorch that handles data and labels. This class is compatible with PyTorch's Dataset interface
    and can be used with a DataLoader for efficient batch processing. It's designed to work with both numpy arrays and PyTorch tensors.
    """

    def __init__(self, data, label=None):
        """
        Initializes the dataset with data and labels.

        :param data: The input data.
        :param label: The target labels for the input data.
        """
        self.data = data
        self.label = label if label is not None else np.zeros((len(data), 1))

    def __getitem__(self, idx):
        """
        Retrieves the data item and its corresponding label at the specified index.

        :param idx: (int) The index of the data item to retrieve.

        :return: A tuple containing the data item and its label.
        """
        return self.data[idx], self.label[idx]

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        :return: (int) The size of the dataset.
        """
        return len(self.data)
