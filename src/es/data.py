"""Featurised molecules, targets and scaffold folds for the ES/GP loop.

Featurisation goes through ``unimol_tools.data.ConformerGen`` and batching
through ``UniMolModel.batch_collate_fn``, so molecules reach the encoder in
exactly the representation the stock training path produces. Nothing in
``unimol_source/`` is patched: this module imports it, it does not modify it.

Two batching details matter for the ES loop:

* **Length bucketing.** Attention memory and compute scale with ``S^2``, and a
  random batch of ESOL molecules pads to roughly the global maximum. Sorting a
  batch by atom count before tiling means each tile pads only to its own local
  maximum, which is most of the difference between fitting a population chunk in
  16 GB and not.
* **Fold-stratified sampling.** The ES fitness is a grouped-CV score computed
  *within* the batch, so every fold has to be represented in every batch or the
  CV degenerates.
"""

from __future__ import annotations

import numpy as np
import torch

from src.head.ridge import scaffold_folds


class MoleculeData:
    """Conformer features, scaled targets and Murcko scaffolds for one split.

    Args:
        smiles: SMILES strings.
        targets: raw target values.
        model: a ``UniMolModel``, used only for its collate function.
        remove_hs: must match the checkpoint the model loaded.
    """

    def __init__(self, smiles: list[str], targets: np.ndarray, model,
                 remove_hs: bool = False, seed: int = 42):
        from unimol_tools.data.conformer import ConformerGen

        self.smiles = list(smiles)
        self.model = model
        self.raw = np.asarray(targets, dtype=np.float64)
        self.inputs = ConformerGen(remove_hs=remove_hs, seed=seed).transform(self.smiles)
        self.n_atoms = np.array([len(d['src_tokens']) for d in self.inputs])

        self.y_mean = float(self.raw.mean())
        self.y_std = float(self.raw.std()) or 1.0
        self.scaled = (self.raw - self.y_mean) / self.y_std

    def __len__(self) -> int:
        return len(self.smiles)

    def rescale(self, y: np.ndarray | float) -> np.ndarray | float:
        """Map a standardised target (or RMSE) back to the original units."""
        return np.asarray(y) * self.y_std

    def targets(self, idx: np.ndarray, device) -> torch.Tensor:
        return torch.as_tensor(self.scaled[idx], device=device, dtype=torch.float32)

    def collate(self, idx: np.ndarray, device) -> dict[str, torch.Tensor]:
        """Pad and stack the given molecules into an encoder batch."""
        samples = [(self.inputs[i], float(self.scaled[i])) for i in idx]
        batch, _ = self.model.batch_collate_fn(samples)
        return {k: v.to(device) for k, v in batch.items()}

    # --- batching helpers ----------------------------------------------------

    def length_tiles(self, idx: np.ndarray, tile: int) -> list[np.ndarray]:
        """Split ``idx`` into length-homogeneous tiles.

        Returns tiles of positions *into ``idx``* (not into the dataset), so a
        caller can scatter per-tile results back into batch order.
        """
        order = np.argsort(self.n_atoms[idx], kind='stable')
        return [order[i:i + tile] for i in range(0, order.size, tile)]


def make_folds(smiles: list[str], n_folds: int, rng: np.random.Generator
               ) -> np.ndarray:
    """Assign training molecules to scaffold-grouped CV folds.

    Grouping by Murcko scaffold is what keeps the inner fitness honest: an
    ungrouped fold lets a scaffold appear on both sides of the split, so the
    search would be rewarded for memorising scaffolds -- precisely what the
    outer Bemis-Murcko test split punishes.
    """
    from src.data.splitters import generate_scaffold

    scaffolds = [generate_scaffold(s, include_chirality=True) for s in smiles]
    return scaffold_folds(scaffolds, n_folds, rng)


def stratified_batch(fold_of: np.ndarray, n_folds: int, size: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Draw ``size`` rows with every CV fold proportionally represented."""
    if size >= fold_of.size:
        return np.arange(fold_of.size)
    per = max(1, size // n_folds)
    picks = []
    for f in range(n_folds):
        pool = np.nonzero(fold_of == f)[0]
        if pool.size:
            picks.append(rng.choice(pool, size=min(per, pool.size), replace=False))
    return np.concatenate(picks)
