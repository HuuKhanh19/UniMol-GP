"""
Step 2 Trainer: UniMol with EGGROLL Evolution Strategies

This module replaces gradient descent with EGGROLL for training UniMol models.
Trains end-to-end: UniMol encoder + MLP head.

UPDATED: Uses full-batch fitness evaluation for both training and validation.
Full-batch provides more stable gradient estimates than mini-batch for ES.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Callable
from sklearn.metrics import mean_squared_error, roc_auc_score
from dataclasses import dataclass, asdict
import copy
from tqdm import tqdm

from ..optimizers.eggroll import EGGROLL, EGGROLLConfig, compute_optimal_hyperparameters


@dataclass 
class Step2Config:
    """Configuration for Step 2 EGGROLL training."""
    
    # Dataset info
    dataset_name: str = "esol"
    task_type: str = "regression"  # 'regression' or 'classification'
    metric: str = "rmse"  # 'rmse', 'mse', or 'auc'
    
    # EGGROLL hyperparameters (TUNED for full-batch)
    population_size: int = 32       # N: number of perturbations (tuned)
    rank: int = 16                  # r: perturbation rank (N*r=512 for full-rank)
    sigma: float = 0.01             # σ: noise scale (KEEP SMALL for pretrained!)
    learning_rate: float = 0.1      # α: update step size (tuned)
    num_generations: int = 400      # Training generations
    
    # Training settings (FULL-BATCH)
    eval_chunk_size: int = 64       # Chunk size for forward pass (OOM prevention)
    use_antithetic: bool = True     # Mirrored sampling
    normalize_fitness: bool = True  # Normalize fitness scores (z-score fallback)
    rank_transform: bool = True     # ENABLED - prevents outlier domination
    centered_rank: bool = True      # Center ranks around 0 [-0.5, 0.5]
    weight_decay: float = 0.0       # L2 regularization
    lr_decay: float = 0.99          # LR decay per generation (tuned)
    sigma_decay: float = 0.99       # Sigma decay per generation (tuned)
    patience: int = 200             # Early stopping patience (tuned)
    
    # Paths
    step1_model_path: str = ""      # Path to Step 1 trained model (reference only)
    output_dir: str = "./experiments/step2_eggroll"
    
    # Hardware
    use_gpu: bool = True
    seed: int = 42


class UniMolEGGROLLWrapper:
    """
    Wrapper for UniMol model with EGGROLL training.
    
    This class:
    1. Loads a pretrained UniMol model (same starting point as Step 1)
    2. Handles fitness evaluation for regression/classification
    3. Manages EGGROLL optimization
    
    UPDATED: Uses full-batch evaluation for stable gradient estimates.
    """
    
    def __init__(
        self,
        config: Step2Config,
        model: Optional[nn.Module] = None
    ):
        """
        Initialize wrapper.
        
        Args:
            config: Step 2 configuration
            model: Optional pre-loaded PyTorch model (for testing)
        """
        self.config = config
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Set seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Load or use provided model
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = self._load_unimol_model()
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize EGGROLL optimizer (will be set during training)
        self.eggroll = None
        
    def _load_unimol_model(self) -> nn.Module:
        """
        Load UniMol model from pretrained weights (same starting point as Step 1).
        
        For fair comparison: Both GD (Step 1) and EGGROLL (Step 2) start from
        the same pretrained weights, not from Step 1 trained model.
        """
        try:
            from unimol_tools.models import UniMolModel
            
            # Determine output dimension based on task
            output_dim = 1  # Default for regression/binary classification
            
            # Create fresh model with pretrained weights
            task = 'regression' if self.config.task_type == 'regression' else 'classification'
            
            print(f"Creating UniMolModel from pretrained weights...")
            print(f"Task: {task}, Output dim: {output_dim}")
            
            model = UniMolModel(
                output_dim=output_dim,
                data_type='molecule',
                task=task
            )
            
            model = model.to(self.device)
            
            # Print model info
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Loaded UniMol model from pretrained weights!")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"NOTE: Starting from same pretrained weights as Step 1 (fair comparison)")
            
            return model
            
        except Exception as e:
            print(f"Error loading UniMol model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        smiles_column: str = "smiles",
        target_column: str = "target"
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Prepare data for UniMol model.
        
        Args:
            data: DataFrame with SMILES and targets
            smiles_column: SMILES column name
            target_column: Target column name
            
        Returns:
            batch_dict: Dict with src_tokens, src_distance, src_coord, src_edge_type
            targets: Target tensor
        """
        from unimol_tools.data.conformer import ConformerGen
        
        smiles_list = data[smiles_column].tolist()
        targets = torch.tensor(data[target_column].values, dtype=torch.float32)
        
        # Generate conformers
        cg = ConformerGen(data_type='molecule')
        data_list, _ = cg.transform(smiles_list)
        
        # Pad sequences to same length
        max_len = max(d['src_tokens'].shape[0] for d in data_list)
        
        # Pad functions
        def pad_1d(arr, max_len, pad_val=0):
            padded = np.full(max_len, pad_val, dtype=arr.dtype)
            padded[:len(arr)] = arr
            return padded
        
        def pad_2d(arr, max_len, pad_val=0):
            padded = np.full((max_len, max_len), pad_val, dtype=arr.dtype)
            padded[:arr.shape[0], :arr.shape[1]] = arr
            return padded
        
        def pad_coords(arr, max_len):
            padded = np.zeros((max_len, 3), dtype=arr.dtype)
            padded[:arr.shape[0]] = arr
            return padded
        
        # Stack all data
        src_tokens = torch.tensor(np.stack([pad_1d(d['src_tokens'], max_len) for d in data_list]))
        src_distance = torch.tensor(np.stack([pad_2d(d['src_distance'], max_len) for d in data_list]))
        src_coord = torch.tensor(np.stack([pad_coords(d['src_coord'], max_len) for d in data_list]))
        src_edge_type = torch.tensor(np.stack([pad_2d(d['src_edge_type'], max_len) for d in data_list]))
        
        batch_dict = {
            'src_tokens': src_tokens,
            'src_distance': src_distance,
            'src_coord': src_coord,
            'src_edge_type': src_edge_type
        }
        
        return batch_dict, targets
    
    def _create_full_batch_fitness_fn(
        self,
        batch_dict: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        chunk_size: int = 64
    ) -> Callable:
        """
        Create full-batch fitness function for EGGROLL training.
        
        Evaluates on ENTIRE dataset for stable gradient estimates.
        Uses chunked forward passes to prevent OOM.
        
        Args:
            batch_dict: Dict with all data tensors
            targets: All target values
            chunk_size: Chunk size for forward pass (OOM prevention)
            
        Returns:
            Fitness function that evaluates on full dataset
        """
        # Move to device
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        targets = targets.to(self.device)
        
        n_samples = len(targets)
        
        if self.config.task_type == 'regression':
            def fitness_fn(model, data):
                """Regression fitness: -MSE on full dataset."""
                model.eval()
                all_preds = []
                
                with torch.no_grad():
                    for i in range(0, n_samples, chunk_size):
                        end_idx = min(i + chunk_size, n_samples)
                        
                        output = model(
                            batch_dict['src_tokens'][i:end_idx],
                            batch_dict['src_distance'][i:end_idx],
                            batch_dict['src_coord'][i:end_idx],
                            batch_dict['src_edge_type'][i:end_idx]
                        )
                        all_preds.append(output.squeeze(-1))
                    
                    all_preds = torch.cat(all_preds, dim=0)
                    mse = torch.mean((all_preds - targets) ** 2)
                    return -mse.item()  # Negative because higher fitness is better
            
        else:  # classification
            def fitness_fn(model, data):
                """Classification fitness: AUC on full dataset."""
                model.eval()
                all_preds = []
                
                with torch.no_grad():
                    for i in range(0, n_samples, chunk_size):
                        end_idx = min(i + chunk_size, n_samples)
                        
                        output = model(
                            batch_dict['src_tokens'][i:end_idx],
                            batch_dict['src_distance'][i:end_idx],
                            batch_dict['src_coord'][i:end_idx],
                            batch_dict['src_edge_type'][i:end_idx]
                        )
                        all_preds.append(output.squeeze(-1))
                    
                    all_preds = torch.cat(all_preds, dim=0)
                    prob = torch.sigmoid(all_preds)
                    
                    try:
                        auc = roc_auc_score(
                            targets.cpu().numpy(), 
                            prob.cpu().numpy()
                        )
                        return auc  # AUC is already "higher is better"
                    except:
                        return 0.5
        
        return fitness_fn
    
    def train(
        self,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        smiles_column: str = "smiles",
        target_column: str = "target",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train model using EGGROLL with full-batch fitness evaluation.
        
        Full-batch provides more stable gradient estimates than mini-batch.
        """
        print(f"\n{'='*60}")
        print(f"Step 2: EGGROLL Training (Full-Batch)")
        print(f"Dataset: {self.config.dataset_name}")
        print(f"Task: {self.config.task_type}")
        print(f"{'='*60}\n")
        
        # Prepare data
        print("Preparing training data...")
        train_batch, y_train = self.prepare_data(train_data, smiles_column, target_column)
        print("Preparing validation data...")
        valid_batch, y_valid = self.prepare_data(valid_data, smiles_column, target_column)
        
        n_train = len(y_train)
        n_valid = len(y_valid)
        print(f"Train samples: {n_train}, Valid samples: {n_valid}")
        
        # Analyze model for optimal hyperparameters
        print("\nAnalyzing model architecture...")
        hp_info = compute_optimal_hyperparameters(self.model, target_full_rank=True)
        print(f"Max min(m,n) across layers: {hp_info['max_min_dim']}")
        print(f"Suggested configurations:")
        for key, val in hp_info['suggestions'].items():
            print(f"  {key}: N={val['population_size']}, r={val['rank']} - {val['note']}")
        
        current_Nr = self.config.population_size * self.config.rank
        if current_Nr < hp_info['max_min_dim']:
            print(f"\nWarning: Current N*r = {current_Nr} < {hp_info['max_min_dim']}")
        else:
            print(f"\nConfig OK: N*r = {current_Nr} >= {hp_info['max_min_dim']}")
        
        # Create EGGROLL config (no batch_size - full-batch mode)
        eggroll_config = EGGROLLConfig(
            population_size=self.config.population_size,
            rank=self.config.rank,
            sigma=self.config.sigma,
            learning_rate=self.config.learning_rate,
            num_generations=self.config.num_generations,
            use_antithetic=self.config.use_antithetic,
            normalize_fitness=self.config.normalize_fitness,
            rank_transform=self.config.rank_transform,
            centered_rank=self.config.centered_rank,
            weight_decay=self.config.weight_decay,
            lr_decay=self.config.lr_decay,
            sigma_decay=self.config.sigma_decay,
            seed=self.config.seed
        )
        
        # Initialize EGGROLL optimizer
        print("\nInitializing EGGROLL optimizer...")
        self.eggroll = EGGROLL(self.model, eggroll_config, device=self.device)
        
        # Create full-batch fitness functions
        chunk = self.config.eval_chunk_size
        print(f"Training fitness: FULL-BATCH ({n_train} samples, chunk_size={chunk})")
        print(f"Validation fitness: FULL-BATCH ({n_valid} samples, chunk_size={chunk})")
        
        train_fitness_fn = self._create_full_batch_fitness_fn(train_batch, y_train, chunk)
        valid_fitness_fn = self._create_full_batch_fitness_fn(valid_batch, y_valid, chunk)
        
        # Training history
        history = {
            'train_fitness': [],
            'valid_fitness': [],
            'generations': []
        }
        
        best_valid_fitness = float('-inf')
        best_model_state = None
        patience_counter = 0
        patience = self.config.patience
        
        # Start training
        n_evals = self.config.population_size * 2 if self.config.use_antithetic else self.config.population_size
        print(f"\nStarting EGGROLL training for {self.config.num_generations} generations...")
        print(f"Each generation: {n_evals} fitness evals × {n_train} samples (full-batch)")
        print(f"Early stopping patience: {patience}")
        print(f"Rank transform: {self.config.rank_transform}")
        print(f"LR decay: {self.config.lr_decay}, Sigma decay: {self.config.sigma_decay}")
        
        import time
        start_time = time.time()
        
        for gen in tqdm(range(self.config.num_generations), desc="Training"):
            # Perform EGGROLL step (full-batch)
            stats = self.eggroll.step(train_fitness_fn, None, verbose=False)
            
            # Evaluate on validation (also full-batch)
            valid_fitness = valid_fitness_fn(self.model, None)
            
            # Record history
            history['generations'].append(gen)
            history['train_fitness'].append(stats['mean_fitness'])
            history['valid_fitness'].append(valid_fitness)
            
            # Check for improvement
            if valid_fitness > best_valid_fitness:
                best_valid_fitness = valid_fitness
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at generation {gen}")
                break
            
            # Print progress
            if verbose and (gen + 1) % 10 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (gen + 1) * (self.config.num_generations - gen - 1)
                metric_name = 'AUC' if self.config.task_type == 'classification' else '-MSE'
                print(f"Gen {gen+1}: Train {metric_name}={stats['mean_fitness']:.4f}, "
                      f"Valid {metric_name}={valid_fitness:.4f}, "
                      f"Best={best_valid_fitness:.4f}, "
                      f"LR={self.eggroll.current_lr:.6f}, "
                      f"σ={self.eggroll.current_sigma:.6f}, "
                      f"ETA={eta/60:.1f}min")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Restored best model (valid fitness: {best_valid_fitness:.4f})")
        
        results = {
            'config': asdict(self.config),
            'history': history,
            'best_valid_fitness': best_valid_fitness,
            'final_generation': gen,
            'training_time_minutes': total_time / 60
        }
        
        return results
    
    def evaluate(
        self,
        data: pd.DataFrame,
        smiles_column: str = "smiles",
        target_column: str = "target"
    ) -> Dict[str, float]:
        """
        Evaluate model on data (full evaluation for accurate metrics).
        """
        # Prepare data
        batch_dict, targets = self.prepare_data(data, smiles_column, target_column)
        
        # Move to device
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        
        self.model.eval()
        all_preds = []
        n_samples = len(targets)
        chunk_size = self.config.eval_chunk_size
        
        with torch.no_grad():
            for i in range(0, n_samples, chunk_size):
                end_idx = min(i + chunk_size, n_samples)
                
                output = self.model(
                    batch_dict['src_tokens'][i:end_idx],
                    batch_dict['src_distance'][i:end_idx],
                    batch_dict['src_coord'][i:end_idx],
                    batch_dict['src_edge_type'][i:end_idx]
                )
                all_preds.append(output.squeeze(-1).cpu())
        
        all_preds = torch.cat(all_preds, dim=0).numpy()
        targets = targets.numpy()
        
        results = {}
        
        if self.config.task_type == 'regression':
            mse = mean_squared_error(targets, all_preds)
            rmse = np.sqrt(mse)
            results['mse'] = mse
            results['rmse'] = rmse
        else:
            prob = 1 / (1 + np.exp(-all_preds))
            auc = roc_auc_score(targets, prob)
            results['auc'] = auc
        
        return results
    
    def save(self, path: Optional[str] = None):
        """Save trained model."""
        save_path = path or os.path.join(self.config.output_dir, 'model.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': asdict(self.config),
            'eggroll_state': self.eggroll.state_dict() if self.eggroll else None
        }, save_path)
        print(f"Model saved to {save_path}")


class Step2Trainer:
    """
    Step 2 Trainer: EGGROLL Evolution Strategies.
    
    This replaces gradient descent with EGGROLL for UniMol training.
    Uses full-batch evaluation for stable gradient estimates.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        experiment_name: str = "step2_eggroll"
    ):
        """
        Initialize Step 2 trainer.
        
        Args:
            config: Configuration dictionary (from YAML)
            experiment_name: Name for this experiment
        """
        self.raw_config = config
        self.experiment_name = experiment_name
        
        # Create output directory
        output_dir = os.path.join(
            config['experiment']['output_dir'],
            experiment_name,
            config['dataset']['name']
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Get Step 1 model path (for reference, not used)
        step1_path = os.path.join(
            config['experiment']['output_dir'],
            'step1_baseline',
            config['dataset']['name']
        )
        
        # Create Step2Config with TUNED defaults
        ec = config.get('eggroll', {})
        
        self.step2_config = Step2Config(
            dataset_name=config['dataset']['name'],
            task_type=config['dataset']['task_type'],
            metric=config['dataset']['metric'],
            # TUNED hyperparameters
            population_size=ec.get('population_size', 32),      # was 128
            rank=ec.get('rank', 16),                            # was 4
            sigma=ec.get('sigma', 0.01),                        # keep small!
            learning_rate=ec.get('learning_rate', 0.1),         # was 0.05
            num_generations=ec.get('num_generations', 400),     # was 200
            # Full-batch settings
            eval_chunk_size=ec.get('eval_chunk_size', 64),      # NEW
            use_antithetic=ec.get('use_antithetic', True),
            normalize_fitness=ec.get('normalize_fitness', True),
            rank_transform=ec.get('rank_transform', True),      # was False
            centered_rank=ec.get('centered_rank', True),
            weight_decay=ec.get('weight_decay', 0.0),
            lr_decay=ec.get('lr_decay', 0.99),                  # was 1.0
            sigma_decay=ec.get('sigma_decay', 0.99),            # was 1.0
            patience=ec.get('patience', 200),                   # was 50
            step1_model_path=step1_path,
            output_dir=output_dir,
            use_gpu=config['unimol'].get('use_gpu', True),
            seed=config['data'].get('random_seed', 42)
        )
        
        self.output_dir = output_dir
        self.task_type = config['dataset']['task_type']
        self.metric_name = config['dataset']['metric']
        
        # Initialize wrapper (model will be loaded)
        self.wrapper = None
    
    def run(
        self,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        test_data: pd.DataFrame,
        smiles_column: str = "smiles",
        target_column: str = "measured"
    ) -> Dict[str, Any]:
        """
        Run complete EGGROLL training and evaluation.
        
        Args:
            train_data: Training DataFrame
            valid_data: Validation DataFrame
            test_data: Test DataFrame
            smiles_column: SMILES column name
            target_column: Target column name
            
        Returns:
            Results dictionary
        """
        # Initialize wrapper (loads model)
        self.wrapper = UniMolEGGROLLWrapper(self.step2_config)
        
        # Train
        print("Training with EGGROLL (Full-Batch)...")
        train_results = self.wrapper.train(
            train_data,
            valid_data,
            smiles_column=smiles_column,
            target_column=target_column
        )
        
        # Evaluate on all splits
        print("\nEvaluating...")
        train_eval = self.wrapper.evaluate(train_data, smiles_column, target_column)
        valid_eval = self.wrapper.evaluate(valid_data, smiles_column, target_column)
        test_eval = self.wrapper.evaluate(test_data, smiles_column, target_column)
        
        # Compile results
        results = {
            "dataset": self.step2_config.dataset_name,
            "task_type": self.task_type,
            "metric": self.metric_name,
            "method": "EGGROLL",
            "eggroll_config": asdict(self.step2_config),
            "train": train_eval,
            "valid": valid_eval,
            "test": test_eval,
            "training_history": train_results.get('history', {}),
            "model_path": self.output_dir
        }
        
        # Print results
        print(f"\n{'='*60}")
        print("Results:")
        print(f"  Train {self.metric_name.upper()}: {train_eval.get(self.metric_name, 'N/A'):.4f}")
        print(f"  Valid {self.metric_name.upper()}: {valid_eval.get(self.metric_name, 'N/A'):.4f}")
        print(f"  Test  {self.metric_name.upper()}: {test_eval.get(self.metric_name, 'N/A'):.4f}")
        print(f"{'='*60}\n")
        
        # Save results
        self._save_results(results)
        
        # Save model
        self.wrapper.save()
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to JSON file."""
        results_path = os.path.join(self.output_dir, "results.json")
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        results = convert_numpy(results)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_path}")