"""
UniMol-GP Step 2 -- LoRA-subspace adaptation on a frozen pretrained backbone.

    Step 2.1  method='gd'  -> gradient descent on LoRA + head   (reference)
    Step 2.2  method='es'  -> EGGROLL / Evolution Strategies on LoRA + head

Strategy (max reuse, min risk)
------------------------------
We reuse the *entire* unimol_tools pipeline (DataHub featurisation stored in
``data['unimol_input']``, NNModel, Trainer loop, predict/metrics, EarlyStopper,
checkpointing) and change only two things:

  1. The model: ``LoRANNModel._init_model`` wraps every ``nn.Linear`` in the encoder
     with LoRA and freezes everything except LoRA A/B + the regression head.
  2. The optimisation:
       * GD (2.1): nothing else to do. The stock optimiser is
         ``Adam(model.parameters(), ...)`` and PyTorch only updates params that
         receive a gradient, i.e. LoRA + head. Same LR/warmup as the baseline.
       * ES (2.2): ``ESLoRATrainer._fit_predict_es`` runs an EGGROLL loop instead of
         backprop, reusing the same dataloader / decorate_batch / loss / predict /
         early-stopping / checkpoint.

After training we *merge* LoRA into the Linear weights and save a standard UniMol
checkpoint, so Step-1's evaluation path (MolPredict) loads it unchanged.

VERIFY-ON-YOUR-MACHINE (import paths / attributes -- adjust if your fork differs):
  * from unimol_tools.models.nnmodel import NNModel
  * from unimol_tools.tasks.trainer  import Trainer, NNDataLoader, EarlyStopper
  * from unimol_tools.train          import MolTrain
  * from unimol_tools.data           import DataHub
  * NNModel attrs used: ``.model`` (nn.Module), ``.cv['pred']`` (valid preds), ``.run()``.
  * Model attrs used: ``model.encoder``, ``model.classification_head``,
    ``model.batch_collate_fn``, ``model.load_pretrained_weights``.
Run Step 2.1 FIRST -- it validates all of the above with low risk.
"""

from __future__ import annotations

import os
import logging
import numpy as np
import pandas as pd
import torch

# --- unimol_tools internals (VERIFY paths) ---------------------------------
from unimol_tools.train import MolTrain
from unimol_tools.data import DataHub
from unimol_tools.models.nnmodel import NNModel
from unimol_tools.tasks.trainer import Trainer, NNDataLoader, EarlyStopper
from unimol_tools.utils import logger

# --- our modules ------------------------------------------------------------
from src.es import (
    inject_lora_, merge_lora_, setup_trainable_, trainable_params, count_params, EggrollES,
)
from src.models.unimol_wrapper import UniMolWrapper          # reuse predict()/evaluate()
from src.data.datasets import OUTPUT_DIR


def _quiet_unimol_logger():
    fmt = logging.Formatter('%(message)s')
    for name in ['Uni-Mol Tools', 'unimol', '']:
        lg = logging.getLogger(name)
        for h in lg.handlers:
            h.setFormatter(fmt)


def _cycle(loader):
    while True:
        for b in loader:
            yield b


# ===========================================================================
#  Model: inject LoRA + freeze backbone
# ===========================================================================
class LoRANNModel(NNModel):
    """NNModel that wraps the UniMol encoder with LoRA and freezes the backbone."""

    def _init_model(self, model_name, **params):
        model = super()._init_model(model_name, **params)      # builds UniMol + loads pretrained
        rank = int(params.get('lora_rank', 8))
        alpha = float(params.get('lora_alpha', 16.0))

        target = getattr(model, 'encoder', model)              # wrap ONLY the transformer encoder
        n_wrapped = inject_lora_(target, rank=rank, alpha=alpha)

        setup_trainable_(model, train_head=True, head_attr='classification_head')
        n_train = count_params(trainable_params(model))
        logger.info(f"[LoRA] wrapped {n_wrapped} Linear layers (rank={rank}, alpha={alpha}); "
                    f"trainable params = {n_train:,}")
        return model


# ===========================================================================
#  Trainer: ES branch (GD branch falls through to the stock backprop loop)
# ===========================================================================
class ESLoRATrainer(Trainer):
    def __init__(self, save_path=None, **config):
        # Pull our knobs out before the stock Trainer sees the config.
        self.es_method         = config.pop('es_method', 'gd')
        self.es_sigma          = float(config.pop('es_sigma', 1e-2))
        self.es_lr             = float(config.pop('es_lr', 1e-3))
        self.es_lr_decay       = float(config.pop('es_lr_decay', 1.0))   # cosine final-LR fraction; 1.0 = off
        self.es_popsize        = int(config.pop('es_popsize', 256))
        self.es_steps          = int(config.pop('es_steps', 600))
        self.es_log_every      = int(config.pop('es_log_every', 50))     # PRINT cadence (val is computed EVERY step)
        self.es_patience       = int(config.pop('es_patience', 400))     # early-stop patience IN STEPS
        self.es_weight_decay   = float(config.pop('es_weight_decay', 0.0))
        self.es_rank_transform = bool(config.pop('es_rank_transform', True))
        self.es_data_batch     = int(config.pop('es_data_batch', 0))     # 0 -> full-batch fitness
        config.pop('es_eval_every', None)                                # backward-compat: ignored
        config.pop('lora_rank', None)
        config.pop('lora_alpha', None)
        super().__init__(save_path=save_path, **config)

    def fit_predict_wo_ddp(self, model, train_dataset, valid_dataset, loss_func,
                           activation_fn, dump_dir, fold, target_scaler, feature_name=None):
        if self.es_method != 'es':
            # ---- Step 2.1: gradient descent on LoRA + head (backbone frozen) ----
            return super().fit_predict_wo_ddp(
                model, train_dataset, valid_dataset, loss_func,
                activation_fn, dump_dir, fold, target_scaler, feature_name)
        # ---- Step 2.2: EGGROLL / Evolution Strategies on LoRA + head ----
        return self._fit_predict_es(
            model, train_dataset, valid_dataset, loss_func,
            activation_fn, dump_dir, fold, target_scaler, feature_name)

    def _fit_predict_es(self, model, train_dataset, valid_dataset, loss_func,
                        activation_fn, dump_dir, fold, target_scaler, feature_name=None):
        model = model.to(self.device)
        bs = self.es_data_batch if self.es_data_batch > 0 else len(train_dataset)
        train_loader = NNDataLoader(
            feature_name=feature_name, dataset=train_dataset, batch_size=bs,
            shuffle=True, collate_fn=model.batch_collate_fn, drop_last=False,
        )

        params = trainable_params(model)
        es = EggrollES(
            params, sigma=self.es_sigma, lr=self.es_lr, popsize=self.es_popsize,
            weight_decay=self.es_weight_decay, rank_transform=self.es_rank_transform,
            device=str(self.device), seed=int(self.seed),
            lr_decay=self.es_lr_decay, total_steps=self.es_steps,
        )
        logger.info(f"[ES] d={es.d:,}  sigma={self.es_sigma}  lr={self.es_lr}  "
                    f"lr_decay={self.es_lr_decay}{' (cosine)' if self.es_lr_decay < 1.0 else ' (constant)'}  "
                    f"pop={self.es_popsize}  steps={self.es_steps}  "
                    f"data_batch={'full' if bs == len(train_dataset) else bs}  "
                    f"(val every step; log every {self.es_log_every}; patience {self.es_patience} steps)")

        # Keep the best model IN MEMORY (snapshot only the ~1M trainable LoRA+head params,
        # ~4 MB) instead of writing the full ~190 MB checkpoint to disk on every improvement.
        def snapshot():
            return [p.detach().cpu().clone() for p in params]

        def restore(snap):
            with torch.no_grad():
                for p, s in zip(params, snap):
                    p.copy_(s.to(p.device))

        model.eval()                                # deterministic fitness (no dropout noise)
        data_iter = _cycle(train_loader)
        last_train_loss = float('nan')
        best_score, best_step, best_snap, no_improve = float('inf'), 0, snapshot(), 0
        vmetric = self.metrics_str if isinstance(getattr(self, 'metrics_str', None), str) else 'metric'

        for step in range(1, self.es_steps + 1):
            batch = next(data_iter)
            net_input, net_target = self.decorate_batch(batch, feature_name)

            def forward_loss():
                with torch.no_grad():
                    if self.scaler and self.device.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            out = model(**net_input)
                            l = loss_func(out, net_target)
                    else:
                        out = model(**net_input)
                        l = loss_func(out, net_target)
                return float(l)

            last_train_loss = es.step(forward_loss)

            # --- validate EVERY step so the best checkpoint is never missed ---
            _, val_loss, metric_score = self.predict(
                model, valid_dataset, loss_func, activation_fn, dump_dir, fold,
                target_scaler, epoch=step, load_model=False, feature_name=feature_name)
            model.eval()                            # predict() may toggle modes; force eval back
            vmetric = list(metric_score.keys())[0]
            vscore = float(list(metric_score.values())[0])

            if vscore < best_score - 1e-9:
                best_score, best_step, best_snap, no_improve = vscore, step, snapshot(), 0
            else:
                no_improve += 1

            # --- but only PRINT sparsely (every es_log_every, plus last step / on stop) ---
            if step % self.es_log_every == 0 or step == self.es_steps or no_improve >= self.es_patience:
                logger.info(
                    f"[ES] step {step}/{self.es_steps}  train_loss(pop-mean): {last_train_loss:.4f}  "
                    f"val_loss: {np.mean(val_loss):.4f}  val_{vmetric}: {vscore:.4f}  "
                    f"(best {vmetric}: {best_score:.4f} @ step {best_step})  lr: {es.lr:.6f}")
            if no_improve >= self.es_patience:
                logger.info(f"[ES] early stop at step {step} "
                            f"(no val improvement for {self.es_patience} steps)")
                break

        # Restore the best trainable params (LoRA + head) into the model.
        restore(best_snap)
        logger.info(f"[ES] restored best model from step {best_step} (val {vmetric}: {best_score:.4f})")

        # Return valid preds of the best model (NNModel.run uses these for cv['pred']).
        y_preds, _, _ = self.predict(
            model, valid_dataset, loss_func, activation_fn, dump_dir, fold,
            target_scaler, epoch=0, load_model=False, feature_name=feature_name)
        return y_preds


# ===========================================================================
#  MolTrain that uses the LoRA model + ES/GD trainer
# ===========================================================================
class MolTrainLoRA(MolTrain):
    def __init__(self, *, method='gd', lora_rank=8, lora_alpha=16.0,
                 es_sigma=1e-2, es_lr=1e-3, es_lr_decay=1.0, es_popsize=256, es_steps=600,
                 es_log_every=50, es_patience=400, es_weight_decay=0.0,
                 es_rank_transform=True, es_data_batch=0, **kwargs):
        super().__init__(**kwargs)
        extra = dict(
            es_method=method, lora_rank=lora_rank, lora_alpha=lora_alpha,
            es_sigma=es_sigma, es_lr=es_lr, es_lr_decay=es_lr_decay,
            es_popsize=es_popsize, es_steps=es_steps,
            es_log_every=es_log_every, es_patience=es_patience,
            es_weight_decay=es_weight_decay, es_rank_transform=es_rank_transform,
            es_data_batch=es_data_batch,
        )
        for k, v in extra.items():
            self.config[k] = v

    def fit(self, data):
        # ---- identical to MolTrain.fit, but swaps in LoRANNModel + ESLoRATrainer ----
        self.datahub = DataHub(data=data, is_train=True, save_path=self.save_path, **self.config)
        self.data = self.datahub.data

        _overridden = False
        raw_df = self.data.get('raw_data', None)
        if raw_df is not None and hasattr(raw_df, 'columns') and 'VALID' in raw_df.columns:
            vf = raw_df['VALID'].values
            tr_idx = np.where(vf == 0)[0]; te_idx = np.where(vf == 1)[0]
            self.data['split_nfolds'] = [(tr_idx, te_idx)]; self.data['kfold'] = 1
            _overridden = True
            logger.info(f"Using external train/valid split: train={len(tr_idx)}, valid={len(te_idx)}")
        if not _overridden and isinstance(data, str) and os.path.exists(data):
            try:
                _df = pd.read_csv(data, usecols=['VALID']); vf = _df['VALID'].values
                if len(vf) == len(self.data['smiles']):
                    tr_idx = np.where(vf == 0)[0]; te_idx = np.where(vf == 1)[0]
                    self.data['split_nfolds'] = [(tr_idx, te_idx)]; self.data['kfold'] = 1
                    logger.info(f"Using external train/valid split (from CSV): "
                                f"train={len(tr_idx)}, valid={len(te_idx)}")
            except (ValueError, KeyError):
                pass

        self.update_and_save_config()
        self.trainer = ESLoRATrainer(save_path=self.save_path, **self.config)
        self.model = LoRANNModel(self.data, self.trainer, **self.config)
        self.model.run()

        scalar = self.data['target_scaler']
        y_pred = self.model.cv['pred']
        y_true = np.array(self.data['target'])
        if scalar is not None:
            y_pred = scalar.inverse_transform(y_pred)
            y_true = scalar.inverse_transform(y_true)
        self.cv_pred = y_pred
        return


# ===========================================================================
#  Wrapper (reuses UniMolWrapper.predict/evaluate; overrides train)
# ===========================================================================
class ESLoRAWrapper(UniMolWrapper):
    """Train with LoRA + (GD|ES); merge LoRA -> standard checkpoint -> reuse Step-1 eval."""

    def train(self, train_data, valid_data, smiles_column='smiles', target_column='target'):
        train_df = train_data[[smiles_column, target_column]].copy()
        train_df.columns = ['SMILES', 'TARGET']; train_df['VALID'] = 0
        valid_df = valid_data[[smiles_column, target_column]].copy()
        valid_df.columns = ['SMILES', 'TARGET']; valid_df['VALID'] = 1
        combined = pd.concat([train_df, valid_df], ignore_index=True)
        os.makedirs(self.save_path, exist_ok=True)
        csv_path = os.path.join(self.save_path, 'train_data.csv')
        combined.to_csv(csv_path, index=False)

        p = self.p
        gpu_id = p.get('gpu_id', 0)
        mt = MolTrainLoRA(
            task=self.task_type, data_type='molecule',
            epochs=p['epochs'], batch_size=p['batch_size'], learning_rate=p['learning_rate'],
            early_stopping=p['patience'],
            metrics='mse' if self.task_type == 'regression' else 'auc',
            split='random', kfold=1, save_path=self.save_path,
            remove_hs=p.get('remove_hs', True), target_normalize=p.get('target_normalize', 'auto'),
            max_norm=p.get('max_norm', 5.0), use_cuda=p.get('use_gpu', True),
            use_amp=p.get('use_amp', True), use_ddp=False, use_gpu=str(gpu_id),
            model_name=p.get('model_name', 'unimolv1'),
            freeze_layers=None, smiles_col='SMILES', target_cols='TARGET',
            # --- LoRA / ES ---
            method=p['method'], lora_rank=p['lora_rank'], lora_alpha=p['lora_alpha'],
            es_sigma=p['es_sigma'], es_lr=p['es_lr'], es_lr_decay=p['es_lr_decay'],
            es_popsize=p['es_popsize'],
            es_steps=p['es_steps'], es_log_every=p['es_log_every'], es_patience=p['es_patience'],
            es_weight_decay=p['es_weight_decay'], es_rank_transform=p['es_rank_transform'],
            es_data_batch=p['es_data_batch'],
        )
        mt.config.n_confomer = p.get('n_confomer', 1)
        mt.config.warmup_ratio = p.get('warmup_ratio', 0.03)
        mt.config.seed = p.get('random_seed', 42)
        _quiet_unimol_logger()
        mt.fit(csv_path)

        # Merge LoRA -> plain Linear so MolPredict (Step-1 eval path) can load it.
        model = mt.model.model
        target = getattr(model, 'encoder', model)
        n_merged = merge_lora_(target)
        torch.save({'model_state_dict': model.state_dict()},
                   os.path.join(self.save_path, 'model_0.pth'))
        logger.info(f"[LoRA] merged {n_merged} adapters into backbone; saved standard checkpoint.")
        self.model = mt
        return {"status": "trained", "save_path": self.save_path}


# ===========================================================================
#  Orchestrator (mirrors src.models.Step1Trainer)
# ===========================================================================
class Step2Trainer:
    def __init__(self, params: dict, dataset_info: dict, experiment_name: str):
        self.params = params
        self.dataset_info = dataset_info
        self.task_type = dataset_info['task_type']
        self.metric = dataset_info['metric']
        self.output_dir = os.path.join(OUTPUT_DIR, experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        train_params = {**params, 'task_type': self.task_type}
        self.wrapper = ESLoRAWrapper(train_params, save_path=self.output_dir)

    def run(self, train_df, valid_df, test_df,
            smiles_column='smiles', target_column='target'):
        name = self.dataset_info['name']
        method = self.params.get('method', 'gd')
        tag = 'Step 2.1 (GD on LoRA)' if method == 'gd' else 'Step 2.2 (EGGROLL on LoRA)'
        print(f"\n{'='*60}")
        print(f"{tag} -- {name}")
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
            'method': method,
            'train': train_r,
            'valid': valid_r,
            'test': test_r,
            'model_path': self.output_dir,
        }

        m = self.metric
        print(f"\n{'='*60}")
        print(f"  Train {m}: {train_r.get(m, float('nan')):.4f}")
        print(f"  Valid {m}: {valid_r.get(m, float('nan')):.4f}")
        print(f"  Test  {m}: {test_r.get(m, float('nan')):.4f}")
        print(f"{'='*60}")
        return results