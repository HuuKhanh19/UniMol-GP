"""
UniMol Model Wrapper for Step 1 (Gradient Descent).

Accepts a flat params dict. Passes ALL hyperparameters to MolTrain.
"""

import os, logging
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import mean_squared_error, roc_auc_score

from src.data.datasets import OUTPUT_DIR


def _quiet_unimol_logger():
    """Reduce UniMol log verbosity to message-only."""
    fmt = logging.Formatter('%(message)s')
    for name in ['Uni-Mol Tools', 'unimol', '']:
        lg = logging.getLogger(name)
        for h in lg.handlers:
            h.setFormatter(fmt)


class UniMolWrapper:
    """Thin wrapper around unimol_tools MolTrain / MolPredict."""

    def __init__(self, params: dict, save_path: str):
        self.p = params
        self.save_path = save_path
        self.task_type = params['task_type']

    def train(self, train_data, valid_data,
              smiles_column='smiles', target_column='target'):
        from unimol_tools import MolTrain

        # Combine train + valid with VALID flag
        train_df = train_data[[smiles_column, target_column]].copy()
        train_df.columns = ['SMILES', 'TARGET']
        train_df['VALID'] = 0

        valid_df = valid_data[[smiles_column, target_column]].copy()
        valid_df.columns = ['SMILES', 'TARGET']
        valid_df['VALID'] = 1

        combined = pd.concat([train_df, valid_df], ignore_index=True)
        os.makedirs(self.save_path, exist_ok=True)
        csv_path = os.path.join(self.save_path, 'train_data.csv')
        combined.to_csv(csv_path, index=False)

        p = self.p
        trainer = MolTrain(
            task=self.task_type,
            data_type='molecule',
            epochs=p['epochs'],
            batch_size=p['batch_size'],
            learning_rate=p['learning_rate'],
            early_stopping=p['patience'],
            metrics='mse' if self.task_type == 'regression' else 'auc',
            split='random',
            kfold=1,
            save_path=self.save_path,
            remove_hs=p.get('remove_hs', True),
            target_normalize=p.get('target_normalize', 'auto'),
            max_norm=p.get('max_norm', 5.0),
            use_cuda=p.get('use_gpu', True),
            use_amp=p.get('use_amp', True),
            use_ddp=False,
            model_name=p.get('model_name', 'unimolv1'),
            freeze_layers=p.get('freeze_layers', None),
            smiles_col='SMILES',
            target_cols='TARGET',
        )

        # Pass remaining hypers via config (not exposed in MolTrain __init__)
        trainer.config.n_confomer = p.get('n_confomer', 10)
        trainer.config.warmup_ratio = p.get('warmup_ratio', 0.03)
        trainer.config.seed = p.get('random_seed', 42)

        _quiet_unimol_logger()
        trainer.fit(csv_path)
        self.model = trainer
        return {"status": "trained", "save_path": self.save_path}

    def predict(self, data, smiles_column='smiles'):
        from unimol_tools import MolPredict

        temp = data[[smiles_column]].copy()
        temp.columns = ['SMILES']
        csv_path = os.path.join(self.save_path, 'predict_data.csv')
        temp.to_csv(csv_path, index=False)

        predictor = MolPredict(load_model=self.save_path)
        preds = predictor.predict(csv_path)
        if isinstance(preds, dict):
            preds = preds.get('predict', preds)
        return np.array(preds).flatten()

    def evaluate(self, data, smiles_column='smiles', target_column='target'):
        preds = self.predict(data, smiles_column)
        targets = data[target_column].values
        if self.task_type == 'regression':
            mse = mean_squared_error(targets, preds)
            return {'mse': mse, 'rmse': np.sqrt(mse)}
        else:
            return {'auc': roc_auc_score(targets, preds)}


class Step1Trainer:
    """Step 1: baseline UniMol training with gradient descent."""

    def __init__(self, params: dict, dataset_info: dict, experiment_name: str):
        self.params = params
        self.dataset_info = dataset_info
        self.task_type = dataset_info['task_type']
        self.metric = dataset_info['metric']

        self.output_dir = os.path.join(OUTPUT_DIR, experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)

        train_params = {**params, 'task_type': self.task_type}
        self.wrapper = UniMolWrapper(train_params, save_path=self.output_dir)

    def run(self, train_df, valid_df, test_df,
            smiles_column='smiles', target_column='target'):
        name = self.dataset_info['name']
        print(f"\n{'='*60}")
        print(f"Step 1: Baseline Training — {name}")
        print(f"Task: {self.task_type} | Metric: {self.metric}")
        print(f"{'='*60}\n")

        print("Training...")
        self.wrapper.train(train_df, valid_df, smiles_column, target_column)

        print("\nEvaluating...")
        train_r = self.wrapper.evaluate(train_df, smiles_column, target_column)
        valid_r = self.wrapper.evaluate(valid_df, smiles_column, target_column)
        test_r = self.wrapper.evaluate(test_df, smiles_column, target_column)

        results = {
            'dataset': name,
            'task_type': self.task_type,
            'metric': self.metric,
            'train': train_r,
            'valid': valid_r,
            'test': test_r,
            'model_path': self.output_dir,
        }

        m = self.metric
        print(f"\n{'='*60}")
        print(f"  Train {m}: {train_r.get(m, 'N/A'):.4f}")
        print(f"  Valid {m}: {valid_r.get(m, 'N/A'):.4f}")
        print(f"  Test  {m}: {test_r.get(m, 'N/A'):.4f}")
        print(f"{'='*60}")

        return results