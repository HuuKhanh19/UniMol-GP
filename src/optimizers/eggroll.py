"""
EGGROLL: Evolution Strategies with Low-Rank Perturbations

Implementation based on:
"Evolution Strategies at the Hyperscale" (Sarkar et al., 2025)

Key concepts:
- Low-rank perturbations: E = (1/√r) * A @ B^T where r << min(m,n)
- Memory efficient: O(r(m+n)) instead of O(mn) per perturbation
- Full-rank updates: N*r >= min(m,n) ensures effective full-rank exploration
- Gaussian score approximation for faster convergence

This is a PyTorch port of the JAX EGGROLL implementation.

OPTIMIZATION NOTE (Paper Section 4.3):
    The update is computed WITHOUT materializing individual E_i matrices.
    Instead of:  E_all = A @ B^T  -> (N, m, n)   [O(N*m*n) memory]
    We use:      einsum('nir,njr->ij', f*A, B)    [O(N*r*(m+n)) memory]
    This preserves the low-rank memory advantage (up to 33x reduction).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import copy
from tqdm import tqdm


@dataclass
class EGGROLLConfig:
    """Configuration for EGGROLL optimizer."""
    
    # Core hyperparameters
    population_size: int = 32          # N: number of perturbations per step
    rank: int = 16                     # r: rank of perturbation matrices
    sigma: float = 0.01               # σ: noise scale (keep small for pretrained models)
    learning_rate: float = 0.1        # α: step size for parameter updates
    
    # Training settings
    num_generations: int = 400         # Number of evolution steps
    
    # Advanced settings
    use_antithetic: bool = True        # Use mirrored sampling (±E)
    normalize_fitness: bool = True     # Normalize fitness scores (z-score fallback)
    rank_transform: bool = True        # Use rank-based fitness shaping (more stable)
    centered_rank: bool = True         # Center ranks around 0 (for rank_transform)
    weight_decay: float = 0.0          # L2 regularization
    lr_decay: float = 0.99            # Learning rate decay per generation
    sigma_decay: float = 0.99         # Sigma decay per generation
    
    # Constraint: N*r >= min(m,n) for full-rank updates
    enforce_rank_constraint: bool = True
    
    # Random seed
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.population_size > 0, "Population size must be positive"
        assert self.rank > 0, "Rank must be positive"
        assert self.sigma > 0, "Sigma must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"


class LowRankPerturbation:
    """
    Low-rank perturbation for a parameter tensor.
    
    For a parameter W of shape (m, n), the perturbation is:
        E = (1/√r) * A @ B^T
    where A ∈ R^(m×r) and B ∈ R^(n×r)
    
    Memory per perturbation: O(r(m+n)) instead of O(mn)
    """
    
    def __init__(
        self, 
        shape: Tuple[int, ...], 
        rank: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ):
        self.shape = shape
        self.rank = rank
        self.device = device
        self.dtype = dtype
        
        # Handle different tensor dimensions
        if len(shape) == 1:
            self.m = shape[0]
            self.n = 1
            self.is_1d = True
        elif len(shape) == 2:
            self.m, self.n = shape
            self.is_1d = False
        else:
            # Higher dimensional (conv kernels etc): flatten to 2D
            self.m = shape[0]
            self.n = int(np.prod(shape[1:]))
            self.is_1d = False
            self.original_shape = shape
        
        # Effective rank (can't exceed min dimension)
        self.effective_rank = min(rank, self.m, self.n)
        
        # Scaling factor: 1/√r
        self.scale = 1.0 / np.sqrt(self.effective_rank)
    
    def sample(self, rng: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample low-rank factors A and B.
        
        Returns:
            A: Shape (m, r)
            B: Shape (n, r)
        """
        A = torch.randn(
            self.m, self.effective_rank, 
            generator=rng, device=self.device, dtype=self.dtype
        )
        B = torch.randn(
            self.n, self.effective_rank,
            generator=rng, device=self.device, dtype=self.dtype
        )
        return A, B
    
    def construct_perturbation(
        self, 
        A: torch.Tensor, 
        B: torch.Tensor
    ) -> torch.Tensor:
        """
        Construct full perturbation E = (1/√r) * A @ B^T
        
        Note: Only used during forward pass (apply/remove perturbation).
        The update computation uses einsum to avoid this materialization.
        """
        E = self.scale * torch.mm(A, B.t())  # (m, n)
        
        if self.is_1d:
            E = E.squeeze(-1)
        elif hasattr(self, 'original_shape'):
            E = E.view(self.original_shape)
        
        return E
    
    def compute_update(
        self,
        A_list: List[torch.Tensor],
        B_list: List[torch.Tensor],
        fitness_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute parameter update WITHOUT materializing full E matrices.
        
        Paper Section 4.3 optimization:
            Update = (scale/N) * Σ_i f_i * (A_i @ B_i^T)
                   = (scale/N) * einsum('nir,njr->ij', f*A, B)
        
        Memory: O(N * r * (m+n)) instead of O(N * m * n)
        
        For a layer with shape (512, 512), N=32, r=16:
            Old (bmm):    32 × 512 × 512 = 8.4M elements
            New (einsum): 32 × 16 × (512+512) = 524K elements  → 16× reduction
        """
        N = len(A_list)
        
        # Stack factors: (N, m, r) and (N, n, r)
        A_stack = torch.stack(A_list)  # (N, m, r)
        B_stack = torch.stack(B_list)  # (N, n, r)
        
        # Weight A by fitness: (N, 1, 1) * (N, m, r) -> (N, m, r)
        f = fitness_scores.view(N, 1, 1)
        weighted_A = f * A_stack
        
        # Compute update via einsum — never materializes (N, m, n)
        # Σ_i f_i * A_i @ B_i^T = einsum('nir,njr->ij', f*A, B)
        update = self.scale * torch.einsum('nir,njr->ij', weighted_A, B_stack) / N
        
        # Reshape to original parameter shape
        if self.is_1d:
            update = update.squeeze(-1)
        elif hasattr(self, 'original_shape'):
            update = update.view(self.original_shape)
        
        return update


class EGGROLL:
    """
    EGGROLL Optimizer: Evolution Strategies with Low-Rank Perturbations
    
    Replaces gradient-based training with evolution strategies.
    Applies low-rank perturbations to model parameters and updates
    based on fitness scores.
    
    Usage:
        config = EGGROLLConfig(population_size=32, rank=16, sigma=0.01)
        optimizer = EGGROLL(model, config)
        
        for gen in range(num_generations):
            stats = optimizer.step(fitness_fn)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: EGGROLLConfig,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        self.device = device or next(model.parameters()).device
        
        # Set random seed
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
        
        # Create RNG for reproducibility
        self.rng = torch.Generator(device=self.device)
        if config.seed is not None:
            self.rng.manual_seed(config.seed)
        
        # Setup trainable parameters and perturbation objects
        self.param_names = []
        self.param_shapes = {}
        self.perturbations = {}
        self._setup_parameters()
        
        # Current learning rate and sigma (for decay)
        self.current_lr = config.learning_rate
        self.current_sigma = config.sigma
        
        # Statistics
        self.generation = 0
        self.best_fitness = float('-inf')
        self.fitness_history = []
    
    def _setup_parameters(self):
        """Setup perturbation objects for each trainable parameter."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.param_names.append(name)
                self.param_shapes[name] = param.shape
                
                self.perturbations[name] = LowRankPerturbation(
                    shape=param.shape,
                    rank=self.config.rank,
                    device=self.device,
                    dtype=param.dtype
                )
                
                # Check rank constraint
                if self.config.enforce_rank_constraint:
                    pert = self.perturbations[name]
                    min_dim = min(pert.m, pert.n)
                    Nr = self.config.population_size * self.config.rank
                    
                    if Nr < min_dim:
                        print(f"Warning: '{name}' shape {param.shape}: "
                              f"N*r={Nr} < min(m,n)={min_dim}")
        
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"EGGROLL: {len(self.param_names)} param groups, "
              f"{total_params:,} trainable params")
        print(f"  N={self.config.population_size}, r={self.config.rank}, "
              f"σ={self.config.sigma}, α={self.config.learning_rate}")
        if self.config.rank_transform:
            print(f"  Fitness shaping: rank_transform (centered={self.config.centered_rank})")
        elif self.config.normalize_fitness:
            print(f"  Fitness shaping: z-score normalization")
    
    def _sample_perturbations(self) -> List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """Sample low-rank perturbation factors for all parameters."""
        N = self.config.population_size
        if self.config.use_antithetic:
            N = N // 2  # Will mirror each sample
        
        perturbation_samples = []
        for _ in range(N):
            sample = {}
            for name in self.param_names:
                A, B = self.perturbations[name].sample(self.rng)
                sample[name] = (A, B)
            perturbation_samples.append(sample)
        
        return perturbation_samples
    
    def _apply_perturbation(
        self,
        param_factors: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        sign: float = 1.0
    ):
        """Apply perturbation to model parameters in-place."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in param_factors:
                    A, B = param_factors[name]
                    E = self.perturbations[name].construct_perturbation(A, B)
                    param.add_(sign * self.current_sigma * E.to(param.device))
    
    def _remove_perturbation(
        self,
        param_factors: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        sign: float = 1.0
    ):
        """Remove previously applied perturbation."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in param_factors:
                    A, B = param_factors[name]
                    E = self.perturbations[name].construct_perturbation(A, B)
                    param.add_(-sign * self.current_sigma * E.to(param.device))
    
    def _evaluate_fitness(
        self,
        fitness_fn: Callable,
        data: Any,
        perturbation_samples: List[Dict],
    ) -> Tuple[torch.Tensor, List[Dict], List[float]]:
        """
        Evaluate fitness for all perturbations (with antithetic pairs).
        
        Returns:
            fitness_scores, all_factors, signs
        """
        fitness_scores = []
        all_factors = []
        signs = []
        
        self.model.eval()
        
        for factors in perturbation_samples:
            # Positive perturbation: μ + σE
            self._apply_perturbation(factors, sign=1.0)
            with torch.no_grad():
                fitness_pos = fitness_fn(self.model, data)
            self._remove_perturbation(factors, sign=1.0)
            
            fitness_scores.append(fitness_pos)
            all_factors.append(factors)
            signs.append(1.0)
            
            # Antithetic (negative) perturbation: μ - σE
            if self.config.use_antithetic:
                self._apply_perturbation(factors, sign=-1.0)
                with torch.no_grad():
                    fitness_neg = fitness_fn(self.model, data)
                self._remove_perturbation(factors, sign=-1.0)
                
                fitness_scores.append(fitness_neg)
                all_factors.append(factors)
                signs.append(-1.0)
        
        return torch.tensor(fitness_scores, device=self.device), all_factors, signs
    
    def _rank_transform(self, fitness: torch.Tensor) -> torch.Tensor:
        """
        Rank-based fitness shaping.
        
        More robust to outliers than z-score normalization.
        Converts fitness to ranks, then normalizes to [-0.5, 0.5] or [0, 1].
        """
        N = len(fitness)
        ranks = torch.zeros_like(fitness)
        sorted_indices = torch.argsort(fitness)
        ranks[sorted_indices] = torch.arange(N, device=fitness.device, dtype=fitness.dtype)
        
        if self.config.centered_rank:
            ranks = ranks / (N - 1) - 0.5  # [-0.5, 0.5]
        else:
            ranks = ranks / (N - 1)  # [0, 1]
        
        return ranks
    
    def _normalize_fitness(self, fitness: torch.Tensor) -> torch.Tensor:
        """Z-score normalization (fallback when rank_transform is off)."""
        std = fitness.std()
        if std > 1e-8:
            return (fitness - fitness.mean()) / std
        return fitness - fitness.mean()
    
    def _compute_updates(
        self,
        all_factors: List[Dict],
        signs: List[float],
        fitness_scores: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute parameter updates using einsum (memory-efficient).
        
        Update rule (Eq. 8): μ_{t+1} = μ_t + (α/N) * Σ_i E_i * f_i
        """
        updates = {}
        
        for name in self.param_names:
            A_list = []
            B_list = []
            
            for factors, sign in zip(all_factors, signs):
                A, B = factors[name]
                A_list.append(sign * A)  # Apply sign to A factor
                B_list.append(B)
            
            # Compute update via einsum (no full E materialization)
            updates[name] = self.perturbations[name].compute_update(
                A_list, B_list, fitness_scores
            )
        
        return updates
    
    def _apply_updates(self, updates: Dict[str, torch.Tensor]):
        """Apply computed updates to model parameters."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in updates:
                    update = updates[name].to(param.device)
                    param.add_(self.current_lr * update)
                    
                    if self.config.weight_decay > 0:
                        param.mul_(1 - self.config.weight_decay)
    
    def step(
        self,
        fitness_fn: Callable,
        data: Any = None,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Perform one EGGROLL update step.
        
        Args:
            fitness_fn: Function(model, data) -> fitness (higher is better)
            data: Data for fitness evaluation (can be None if fitness_fn captures data)
            verbose: Print progress
            
        Returns:
            Dict with generation statistics
        """
        # 1. Sample perturbations
        perturbation_samples = self._sample_perturbations()
        
        # 2. Evaluate fitness
        fitness_scores, all_factors, signs = self._evaluate_fitness(
            fitness_fn, data, perturbation_samples
        )
        
        # 3. Record raw statistics
        stats = {
            'generation': self.generation + 1,
            'mean_fitness': fitness_scores.mean().item(),
            'max_fitness': fitness_scores.max().item(),
            'min_fitness': fitness_scores.min().item(),
            'std_fitness': fitness_scores.std().item(),
        }
        
        if stats['max_fitness'] > self.best_fitness:
            self.best_fitness = stats['max_fitness']
        stats['best_fitness'] = self.best_fitness
        
        # 4. Shape fitness scores
        if self.config.rank_transform:
            fitness_scores = self._rank_transform(fitness_scores)
        elif self.config.normalize_fitness:
            fitness_scores = self._normalize_fitness(fitness_scores)
        
        # 5. Compute & apply updates (einsum-based, memory efficient)
        updates = self._compute_updates(all_factors, signs, fitness_scores)
        self._apply_updates(updates)
        
        # 6. Decay schedules
        self.current_lr *= self.config.lr_decay
        self.current_sigma *= self.config.sigma_decay
        
        self.generation += 1
        stats['learning_rate'] = self.current_lr
        stats['sigma'] = self.current_sigma
        self.fitness_history.append(stats)
        
        if verbose:
            print(f"Gen {self.generation}: "
                  f"mean={stats['mean_fitness']:.4f}, max={stats['max_fitness']:.4f}, "
                  f"best={self.best_fitness:.4f}")
        
        return stats
    
    def get_best_model(self) -> nn.Module:
        """Return current model."""
        return self.model
    
    def state_dict(self) -> Dict:
        """Get optimizer state for checkpointing."""
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'current_lr': self.current_lr,
            'current_sigma': self.current_sigma,
            'fitness_history': self.fitness_history,
            'rng_state': self.rng.get_state()
        }
    
    def load_state_dict(self, state: Dict):
        """Load optimizer state from checkpoint."""
        self.generation = state['generation']
        self.best_fitness = state['best_fitness']
        self.current_lr = state['current_lr']
        self.current_sigma = state['current_sigma']
        self.fitness_history = state['fitness_history']
        self.rng.set_state(state['rng_state'])


def compute_optimal_hyperparameters(
    model: nn.Module,
    target_full_rank: bool = True
) -> Dict[str, Any]:
    """
    Compute optimal EGGROLL hyperparameters for a model.
    
    Based on constraint: N * r >= min(m, n) for full-rank updates.
    """
    max_min_dim = 0
    param_info = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            shape = param.shape
            if len(shape) >= 2:
                m, n = shape[0], int(np.prod(shape[1:]))
            else:
                m, n = shape[0], 1
            
            min_dim = min(m, n)
            max_min_dim = max(max_min_dim, min_dim)
            param_info.append({
                'name': name, 'shape': tuple(shape),
                'm': m, 'n': n, 'min_dim': min_dim
            })
    
    suggestions = {}
    if target_full_rank:
        suggestions['option_1'] = {
            'rank': 1, 'population_size': max_min_dim,
            'note': 'Minimum rank, large population'
        }
        suggestions['option_2'] = {
            'rank': 4, 'population_size': int(np.ceil(max_min_dim / 4)),
            'note': 'Balanced rank and population'
        }
        suggestions['option_3'] = {
            'rank': 16, 'population_size': int(np.ceil(max_min_dim / 16)),
            'note': 'Higher rank, smaller population'
        }
    else:
        suggestions['default'] = {
            'rank': 4, 'population_size': 64,
            'note': 'Default (may not achieve full-rank updates)'
        }
    
    return {
        'max_min_dim': max_min_dim,
        'param_info': param_info,
        'suggestions': suggestions
    }