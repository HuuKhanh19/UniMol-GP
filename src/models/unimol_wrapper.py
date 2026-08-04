"""
UniMol v1 wrapper for Step 1 (gradient descent).

``UniMolWrapper`` adapts the project's flat params dict to unimol_tools'
MolTrain/MolPredict. ``Step1Trainer`` orchestrates train -> evaluate -> results.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score

from src.data.datasets import OUTPUT_DIR

#: Loggers unimol_tools writes through.
_UNIMOL_LOGGERS = ('Uni-Mol Tools', 'unimol', '')

#: Column names the unimol_tools CSV contract expects.
SMILES_COL = 'SMILES'
TARGET_COL = 'TARGET'
VALID_COL = 'VALID'


def _quiet_unimol_logger() -> None:
    """Drop timestamps/levels from unimol_tools output so logs stay readable."""
    fmt = logging.Formatter('%(message)s')
    for name in _UNIMOL_LOGGERS:
        for handler in logging.getLogger(name).handlers:
            handler.setFormatter(fmt)


class UniMolWrapper:
    """Train and evaluate a UniMol v1 model on a fixed train/valid/test split."""

    def __init__(self, params: dict[str, Any], save_path: str):
        self.p = params
        self.save_path = save_path
        self.task_type = params['task_type']

    def _write_train_csv(self, train_data: pd.DataFrame,
                         valid_data: pd.DataFrame,
                         smiles_column: str, target_column: str) -> str:
        """Write train+valid as one CSV, tagged so MolTrain keeps our split.

        The VALID column (0 = train, 1 = valid) is what
        ``MolTrain._override_split_with_valid_column`` reads to bypass the
        library's own splitter -- see README, "Note on unimol_source/".
        """
        frames = []
        for df, flag in ((train_data, 0), (valid_data, 1)):
            part = df[[smiles_column, target_column]].copy()
            part.columns = [SMILES_COL, TARGET_COL]
            part[VALID_COL] = flag
            frames.append(part)

        os.makedirs(self.save_path, exist_ok=True)
        csv_path = os.path.join(self.save_path, 'train_data.csv')
        pd.concat(frames, ignore_index=True).to_csv(csv_path, index=False)
        return csv_path

    def train(self, train_data: pd.DataFrame, valid_data: pd.DataFrame,
              smiles_column: str = 'smiles',
              target_column: str = 'target') -> dict[str, str]:
        from unimol_tools import MolTrain

        csv_path = self._write_train_csv(
            train_data, valid_data, smiles_column, target_column)

        p = self.p
        trainer = MolTrain(
            task=self.task_type,
            data_type='molecule',
            epochs=p['epochs'],
            batch_size=p['batch_size'],
            learning_rate=p['learning_rate'],
            early_stopping=p['patience'],
            metrics='mse' if self.task_type == 'regression' else 'auc',
            split='random',   # unused: the VALID column overrides the split
            kfold=1,
            save_path=self.save_path,
            remove_hs=p['remove_hs'],
            target_normalize=p['target_normalize'],
            max_norm=p['max_norm'],
            use_cuda=p['use_gpu'],
            use_amp=p['use_amp'],
            use_ddp=False,
            use_gpu=str(p['gpu_id']),
            model_name=p['model_name'],
            freeze_layers=p['freeze_layers'],
            smiles_col=SMILES_COL,
            target_cols=TARGET_COL,
        )
        # Not exposed as MolTrain kwargs; set on the resolved config instead.
        trainer.config.warmup_ratio = p['warmup_ratio']
        trainer.config.seed = p['random_seed']

        _quiet_unimol_logger()
        trainer.fit(csv_path)
        self.model = trainer
        return {'status': 'trained', 'save_path': self.save_path}

    def predict(self, data: pd.DataFrame,
                smiles_column: str = 'smiles') -> np.ndarray:
        from unimol_tools import MolPredict

        temp = data[[smiles_column]].copy()
        temp.columns = [SMILES_COL]
        csv_path = os.path.join(self.save_path, 'predict_data.csv')
        temp.to_csv(csv_path, index=False)

        preds = MolPredict(load_model=self.save_path).predict(csv_path)
        if isinstance(preds, dict):
            preds = preds.get('predict', preds)
        return np.asarray(preds).flatten()

    def evaluate(self, data: pd.DataFrame, smiles_column: str = 'smiles',
                 target_column: str = 'target') -> dict[str, float]:
        preds = self.predict(data, smiles_column)
        targets = data[target_column].values
        if self.task_type == 'regression':
            mse = mean_squared_error(targets, preds)
            return {'mse': float(mse), 'rmse': float(np.sqrt(mse))}
        return {'auc': float(roc_auc_score(targets, preds))}


class Step1Trainer:
    """Run the baseline end to end and collect metrics for all three splits."""

    def __init__(self, params: dict[str, Any], dataset_info: dict[str, Any],
                 experiment_name: str):
        self.params = params
        self.dataset_info = dataset_info
        self.task_type = dataset_info['task_type']
        self.metric = dataset_info['metric']

        self.output_dir = os.path.join(OUTPUT_DIR, experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.wrapper = UniMolWrapper(
            {**params, 'task_type': self.task_type}, save_path=self.output_dir)

    def run(self, train_df: pd.DataFrame, valid_df: pd.DataFrame,
            test_df: pd.DataFrame, smiles_column: str = 'smiles',
            target_column: str = 'target') -> dict[str, Any]:
        name = self.dataset_info['name']
        print(f"\n{'=' * 60}")
        print(f'Step 1: Baseline Training -- {name}')
        print(f'Task: {self.task_type} | Metric: {self.metric}')
        print(f"{'=' * 60}\n")

        print('Training...')
        self.wrapper.train(train_df, valid_df, smiles_column, target_column)

        print('\nEvaluating...')
        scores = {
            split: self.wrapper.evaluate(df, smiles_column, target_column)
            for split, df in (('train', train_df), ('valid', valid_df),
                              ('test', test_df))
        }

        print(f"\n{'=' * 60}")
        for split, score in scores.items():
            value = score.get(self.metric)
            shown = f'{value:.4f}' if value is not None else 'N/A'
            print(f'  {split.capitalize():<5} {self.metric}: {shown}')
        print(f"{'=' * 60}")

        return {
            'dataset': name,
            'task_type': self.task_type,
            'metric': self.metric,
            **scores,
            'model_path': self.output_dir,
        }
