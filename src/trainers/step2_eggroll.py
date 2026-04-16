"""
Step 2: UniMol + EGGROLL Evolution Strategies.

Same pretrained weights as Step 1 (fair comparison).
Replaces SGD with EGGROLL low-rank ES.
Full-batch fitness, n_confomer=1, remove_hs=True.
"""

import os, time, copy, json, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Callable
from sklearn.metrics import mean_squared_error, roc_auc_score

from ..optimizers.eggroll import EGGROLL, EGGROLLConfig
from ..data.datasets import OUTPUT_DIR


def _quiet_unimol_logger():
    fmt = logging.Formatter('%(message)s')
    for name in ['Uni-Mol Tools', 'unimol', '']:
        lg = logging.getLogger(name)
        for h in lg.handlers:
            h.setFormatter(fmt)


def prepare_data(data, smiles_column='smiles', target_column='target',
                 device=torch.device('cpu')):
    from unimol_tools.data.conformer import ConformerGen

    smiles_list = data[smiles_column].tolist()
    targets = torch.tensor(data[target_column].values, dtype=torch.float32)

    _quiet_unimol_logger()
    cg = ConformerGen(data_type='molecule', remove_hs=True, n_confomer=1)
    data_list, _ = cg.transform(smiles_list)

    max_len = max(d['src_tokens'].shape[0] for d in data_list)

    def pad1d(a, L, v=0):
        o = np.full(L, v, dtype=a.dtype); o[:len(a)] = a; return o
    def pad2d(a, L, v=0):
        o = np.full((L, L), v, dtype=a.dtype); o[:a.shape[0], :a.shape[1]] = a; return o
    def padcoord(a, L):
        o = np.zeros((L, 3), dtype=a.dtype); o[:a.shape[0]] = a; return o

    batch = {
        'src_tokens':    torch.tensor(np.stack([pad1d(d['src_tokens'], max_len) for d in data_list])).to(device),
        'src_distance':  torch.tensor(np.stack([pad2d(d['src_distance'], max_len) for d in data_list])).to(device),
        'src_coord':     torch.tensor(np.stack([padcoord(d['src_coord'], max_len) for d in data_list])).to(device),
        'src_edge_type': torch.tensor(np.stack([pad2d(d['src_edge_type'], max_len) for d in data_list])).to(device),
    }
    return batch, targets.to(device)


def make_fitness_fn(batch, targets, task_type, chunk_size=64):
    n = len(targets)

    def _forward(model):
        model.eval()
        parts = []
        with torch.no_grad():
            for i in range(0, n, chunk_size):
                j = min(i + chunk_size, n)
                out = model(
                    batch['src_tokens'][i:j], batch['src_distance'][i:j],
                    batch['src_coord'][i:j], batch['src_edge_type'][i:j],
                )
                parts.append(out.squeeze(-1))
        return torch.cat(parts, dim=0)

    if task_type == 'regression':
        def fitness_fn(model, _data):
            preds = _forward(model)
            return -(torch.mean((preds - targets) ** 2)).item()
    else:
        def fitness_fn(model, _data):
            preds = _forward(model)
            try:
                return roc_auc_score(targets.cpu().numpy(),
                                     torch.sigmoid(preds).cpu().numpy())
            except Exception:
                return 0.5
    return fitness_fn


def load_unimol_model(task_type, device):
    from unimol_tools.models import UniMolModel
    model = UniMolModel(output_dim=1, data_type='molecule', remove_hs=True)
    model = model.to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"UniMol loaded -- {total:,} params ({trainable:,} trainable)")
    return model


def evaluate_model(model, batch, targets, task_type, chunk_size=64):
    fn = make_fitness_fn(batch, targets, task_type, chunk_size)
    fitness = fn(model, None)
    if task_type == 'regression':
        mse = -fitness
        return {'mse': mse, 'rmse': np.sqrt(mse)}
    else:
        return {'auc': fitness}


class Step2Trainer:

    def __init__(self, params, dataset_info, experiment_name):
        self.p = params
        self.dataset_info = dataset_info
        self.task_type = dataset_info['task_type']
        self.metric = dataset_info['metric']

        self.output_dir = os.path.join(OUTPUT_DIR, experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)

        gpu_id = params.get('gpu_id', 0)
        use_gpu = params.get('use_gpu', True)
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_id}')
        else:
            self.device = torch.device('cpu')

        torch.manual_seed(params['random_seed'])
        np.random.seed(params['random_seed'])

    def run(self, train_df, valid_df, test_df,
            smiles_column='smiles', target_column='target'):
        p = self.p

        print(f"\n{'='*60}")
        print(f"Step 2: EGGROLL Training -- {self.dataset_info['name']}")
        print(f"Task: {self.task_type} | Metric: {self.metric}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        # 1. Load model
        print("Loading UniMol model (same pretrained weights as Step 1)...")
        _quiet_unimol_logger()
        model = load_unimol_model(self.task_type, self.device)

        # 2. Prepare data
        print("\nPreparing data (n_confomer=1, remove_hs=True)...")
        train_batch, y_train = prepare_data(train_df, smiles_column, target_column, self.device)
        print(f"  Train: {len(y_train)} molecules")
        valid_batch, y_valid = prepare_data(valid_df, smiles_column, target_column, self.device)
        print(f"  Valid: {len(y_valid)} molecules")
        test_batch, y_test = prepare_data(test_df, smiles_column, target_column, self.device)
        print(f"  Test:  {len(y_test)} molecules")

        # 3. EGGROLL optimizer
        chunk = p['eval_chunk_size']
        eggroll_cfg = EGGROLLConfig(
            population_size=p['population_size'],
            rank=p['rank'],
            sigma=p['sigma'],
            learning_rate=p['eggroll_lr'],
            num_generations=p['num_generations'],
            use_antithetic=p['use_antithetic'],
            normalize_fitness=p['normalize_fitness'],
            rank_transform=p['rank_transform'],
            centered_rank=p['centered_rank'],
            weight_decay=p['weight_decay'],
            lr_decay=p['lr_decay'],
            sigma_decay=p['sigma_decay'],
            seed=p['random_seed'],
        )
        print("\nInitializing EGGROLL optimizer...")
        eggroll = EGGROLL(model, eggroll_cfg, device=self.device)

        # 4. Fitness functions
        train_fitness_fn = make_fitness_fn(train_batch, y_train, self.task_type, chunk)
        valid_fitness_fn = make_fitness_fn(valid_batch, y_valid, self.task_type, chunk)

        # 5. Training loop
        patience = p['eggroll_patience']
        n_gens = p['num_generations']
        n_evals = p['population_size'] * (2 if p['use_antithetic'] else 1)
        print(f"\nTraining: {n_gens} generations, "
              f"{n_evals} evals/gen x {len(y_train)} samples (full-batch)")
        print(f"Early stopping patience: {patience}\n")

        best_valid = float('-inf')
        best_state = None
        patience_ctr = 0
        history = {'gen': [], 'train_fit': [], 'valid_fit': []}
        t0 = time.time()

        for gen in range(n_gens):
            gen_start = time.time()
            stats = eggroll.step(train_fitness_fn, None, verbose=False)
            valid_fit = valid_fitness_fn(model, None)
            gen_time = time.time() - gen_start

            history['gen'].append(gen)
            history['train_fit'].append(stats['mean_fitness'])
            history['valid_fit'].append(valid_fit)

            if valid_fit > best_valid:
                best_valid = valid_fit
                best_state = copy.deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1

            train_mse = -stats['mean_fitness']
            val_mse = -valid_fit if self.task_type == 'regression' else valid_fit
            best_mse = -best_valid if self.task_type == 'regression' else best_valid
            print(
                f"Gen [{gen+1}/{n_gens}] "
                f"train_mse: {train_mse:.4f}, "
                f"val_mse: {val_mse:.4f}, "
                f"best_mse: {best_mse:.4f}, "
                f"lr: {eggroll.current_lr:.6f}, "
                f"sigma: {eggroll.current_sigma:.6f} "
                f"[{gen_time:.1f}s]"
            )

            if patience_ctr >= patience:
                print(f"Early stopping at generation: {gen+1}")
                break

        total_time = time.time() - t0

        # 6. Restore best
        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"Restored best model (best val_mse: {-best_valid:.4f})")

        # 7. Evaluate
        print("\nEvaluating...")
        train_r = evaluate_model(model, train_batch, y_train, self.task_type, chunk)
        valid_r = evaluate_model(model, valid_batch, y_valid, self.task_type, chunk)
        test_r = evaluate_model(model, test_batch, y_test, self.task_type, chunk)

        m = self.metric
        print(f"\n{'='*60}")
        print(f"  Train {m}: {train_r.get(m, 'N/A'):.4f}")
        print(f"  Valid {m}: {valid_r.get(m, 'N/A'):.4f}")
        print(f"  Test  {m}: {test_r.get(m, 'N/A'):.4f}")
        print(f"{'='*60}")

        # 8. Save model
        model_path = os.path.join(self.output_dir, 'model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'params': self.p,
        }, model_path)
        print(f"Model saved -- {model_path}")

        return {
            'dataset': self.dataset_info['name'],
            'task_type': self.task_type,
            'metric': m,
            'method': 'EGGROLL',
            'train': train_r,
            'valid': valid_r,
            'test': test_r,
            'history': history,
            'model_path': self.output_dir,
            'training_time_s': total_time,
        }