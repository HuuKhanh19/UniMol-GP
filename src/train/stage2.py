"""Stage 2: alternate EGGROLL on the backbone with GP on the symbolic head.

    warm-up GP  ->  [ ES phase  ->  GP phase  ->  evaluate ] x n_phases

Why alternating rather than nested: running GP inside every ES fitness
evaluation would make the fitness a *stochastic* function of the backbone -- two
nearby parameter vectors could yield entirely different trees -- which breaks
the local continuity that ES needs to estimate a gradient at all, quite apart
from costing N GP runs per step. Alternating keeps the trees fixed and shared
across the whole population within an ES step, so fitness differences are
attributable to the perturbation alone (common random numbers), and the GP phase
warm-starts from the previous population so the head drifts rather than jumps.

Everything the head needs is re-solved in closed form each phase: the ridge
merge every fitness evaluation, and the linear probe once per GP phase while the
backbone is momentarily frozen.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.es.data import MoleculeData, make_folds
from src.es.eggroll import EGGROLL, ESConfig, embed
from src.es.forward_unimol import SplitUniMol
from src.head.gp import CoevolutionGP, GPConfig, fit_final
from src.head.gp_head import GPHead
from src.head.ridge import fold_indices


@dataclass
class Stage2Config:
    """Phase schedule for the alternating search."""

    #: Generations of the initial GP phase, before ES has moved anything. The
    #: head has to be good enough for its fitness to say something about the
    #: backbone before ES starts listening to it.
    warm_gens: int = 300
    n_phases: int = 20
    es_steps: int = 100
    gp_gens: int = 40
    #: Phases without validation improvement before stopping.
    patience: int = 5
    log_every: int = 10
    seed: int = 0


class Stage2Trainer:
    """Co-adapt a UniMol backbone and a symbolic head."""

    def __init__(self, split: SplitUniMol, train: MoleculeData,
                 valid: MoleculeData, test: MoleculeData,
                 cfg: Stage2Config, gp_cfg: GPConfig, es_cfg: ESConfig,
                 device: torch.device, out_dir: str, metric: str = 'rmse'):
        self.split = split
        #: Reporting metric. The *search* fitness is squared error either way:
        #: on a 0/1 target that is the Brier score, a proper scoring rule, and
        #: keeping it squared is what lets the ridge merge stay closed-form.
        #: AUC is a ranking statistic over pairs -- it neither decomposes per
        #: molecule (which the ES fitness shaping needs) nor admits a
        #: closed-form fit.
        self.metric = metric
        self.higher_is_better = metric == 'auc'
        self.train, self.valid, self.test = train, valid, test
        self.cfg, self.gp_cfg, self.es_cfg = cfg, gp_cfg, es_cfg
        self.device = device
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        rng = np.random.default_rng(cfg.seed)
        self.fold_of = make_folds(train.smiles, es_cfg.n_folds, rng)
        self.folds = fold_indices(self.fold_of, es_cfg.n_folds, device)
        self.train_idx = np.arange(len(train))
        self.y_train = train.targets(self.train_idx, device)

        self.gp = CoevolutionGP(gp_cfg, split.embed_dim, rng, device)
        self.es = EGGROLL(split, train, self.fold_of, es_cfg, device)
        self.history: list[dict] = []
        worst = -float('inf') if self.higher_is_better else float('inf')
        self.best = {'valid_score': worst, 'phase': -1}
        self.best_head: GPHead | None = None
        self.phase = 0
        self._z_train: torch.Tensor | None = None

    # --- helpers -------------------------------------------------------------

    def _embed(self, data: MoleculeData) -> torch.Tensor:
        """Unperturbed CLS embeddings for a whole split."""
        return embed(self.split, data, np.arange(len(data)), self.device,
                     self.es_cfg.mol_tile)[0]

    def _embed_train(self) -> torch.Tensor:
        """Train embeddings, cached until ES next moves the backbone.

        The GP phase and the evaluation that follows it run back to back with
        the weights untouched, so without this the whole training split goes
        through the encoder twice per phase -- noticeable on Lipophilicity,
        where that is ~3400 molecules.
        """
        if self._z_train is None:
            self._z_train = self._embed(self.train)
        return self._z_train

    def _score(self, pred: torch.Tensor, data: MoleculeData) -> float:
        """Reporting score for one split, in the metric's own units."""
        if self.metric == 'auc':
            # A scaffold split can hand a small split a single class, and
            # roc_auc_score raises on that. 0.5 is the honest score there, and
            # far better than losing a run hours in.
            if np.unique(data.raw).size < 2:
                return 0.5
            return float(roc_auc_score(data.raw, pred.detach().cpu().numpy()))
        y = data.targets(np.arange(len(data)), self.device)
        return float(self.train.rescale(float((pred - y).pow(2).mean().sqrt())))

    def _better(self, candidate: float, incumbent: float) -> bool:
        return (candidate > incumbent if self.higher_is_better
                else candidate < incumbent)

    def _evaluate(self, head: GPHead) -> dict[str, float]:
        """Fit the head on train, then score all three splits."""
        z_train = self._embed_train()
        head = fit_final(head, z_train, self.y_train, self.gp.rho,
                         self.gp_cfg.probe_penalty_scale)
        out = {'metric': self.metric}
        for name, data in (('train', self.train), ('valid', self.valid),
                           ('test', self.test)):
            z = z_train if name == 'train' else self._embed(data)
            score = self._score(head.predict(z), data)
            # Both keys: `_score` is what tooling reads whatever the task is,
            # `_rmse`/`_auc` keeps the metric legible in the raw json.
            out[f'{name}_score'] = score
            out[f'{name}_{self.metric}'] = score
        out.update({f'share_{k}': v for k, v in head.contributions(z_train).items()})
        return out

    def _snapshot(self, tag: str, head: GPHead) -> None:
        """Persist the head plus the ES mean of every perturbed matrix."""
        head.save(os.path.join(self.out_dir, f'head_{tag}.npz'))
        torch.save(
            {f'{li}.{name}': self.split.mean((li, name)).cpu()
             for li, name in self.split.targets},
            os.path.join(self.out_dir, f'backbone_{tag}.pt'),
        )

    # --- phases --------------------------------------------------------------

    def _gp_phase(self, n_gens: int, label: str) -> GPHead:
        z = self._embed_train()
        if self.gp_cfg.replicas > 1:
            z = self._replicated(z)
        hist = self.gp.evolve(z, self.y_train, self.folds, n_gens,
                              log_every=self.cfg.log_every)
        if hist:
            last = hist[-1]
            print(f'    GP {label}: cv={last["cv_rmse"]:.4f} '
                  f'size={last["mean_size"]:.1f} rho={last["rho"]:.0e}')
        return self.gp.head(best=True)

    def _replicated(self, z_centre: torch.Tensor) -> torch.Tensor:
        """Stack embeddings from perturbed backbones around the current mean.

        Trees then have to work across the ES neighbourhood rather than for one
        exact parameter vector, which penalises a formula that only fits the
        current weights. Costs one extra suffix pass per replica and scales the
        CV work linearly, which is why it is off by default.
        """
        from src.es import perturb

        extra = self.gp_cfg.replicas - 1
        n_members = extra + (extra % 2)          # antithetic sampling needs pairs
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.cfg.seed * 7919 + self.phase)
        pert = perturb.sample(self.split, n_members, self.es_cfg.sigma, gen)
        z = embed(self.split, self.train, self.train_idx, self.device,
                  self.es_cfg.mol_tile, pert, n_members)
        return torch.cat([z_centre.unsqueeze(0), z[:extra]], dim=0)

    def _es_phase(self, n_steps: int) -> dict:
        stats = []
        for _ in range(n_steps):
            stats.append(self.es.step())
        self._z_train = None          # the backbone moved; the cache is stale
        mean_sec = float(np.mean([s['sec'] for s in stats]))
        last = stats[-1]
        print(f'    ES: cv(pop mean)={last["pop_cv_rmse_mean"]:.4f} '
              f'centre={self.es.centre_score():.4f} '
              f'|dW|={last["update_rel"]:.2e} {mean_sec:.2f}s/step')
        return {'sec_per_step': mean_sec, **last}

    # --- driver --------------------------------------------------------------

    def run(self) -> dict:
        cfg = self.cfg
        t0 = time.time()
        print(f'\n{"=" * 68}')
        print(f'Stage 2: EGGROLL + symbolic head  ({self.split.n_es_params():,} '
              f'ES params over the last {self.split.spec.n_layers} layers)')
        print(f'{"=" * 68}')

        print(f'\n  warm-up GP ({cfg.warm_gens} generations)')
        head = self._gp_phase(cfg.warm_gens, 'warm')
        metrics = self._evaluate(head)
        print(f'    valid={metrics["valid_score"]:.4f} '
              f'test={metrics["test_score"]:.4f}'
              f'  (trees carry {metrics.get("share_trees", 0):.0%} of the signal)')
        self.best = {**metrics, 'phase': 0}
        self.best_head = head
        self._snapshot('best', head)
        self.history.append({'phase': 0, 'stage': 'warm', **metrics})

        stale = 0
        for phase in range(1, cfg.n_phases + 1):
            self.phase = phase
            print(f'\n  phase {phase}/{cfg.n_phases}')
            self.es.set_head(self.gp.head(best=True), self.gp.rho)
            es_stats = self._es_phase(cfg.es_steps)

            head = self._gp_phase(cfg.gp_gens, f'phase {phase}')
            metrics = self._evaluate(head)
            print(f'    valid={metrics["valid_score"]:.4f} '
                  f'test={metrics["test_score"]:.4f}')

            self.history.append({'phase': phase, 'stage': 'alternate',
                                 **metrics, **es_stats})
            if self._better(metrics['valid_score'], self.best['valid_score']):
                self.best = {**metrics, 'phase': phase}
                self.best_head = head
                self._snapshot('best', head)
                stale = 0
                print('    * best so far')
            else:
                stale += 1
                if stale >= cfg.patience:
                    print(f'    early stop: {stale} phases without improvement')
                    break

        self._snapshot('last', head)
        # Report the head that actually won on validation, not whichever one the
        # loop happened to end on.
        winner = self.best_head or head
        result = {
            'metric': self.metric,
            'best': self.best,
            'history': self.history,
            'formulas': winner.formulas(),
            'tree_sizes': winner.sizes().tolist(),
            'es_params': self.split.n_es_params(),
            'elapsed_sec': time.time() - t0,
            'config': {
                'stage2': asdict(cfg),
                'gp': asdict(self.gp_cfg),
                'es': asdict(self.es_cfg),
            },
        }
        with open(os.path.join(self.out_dir, 'results.json'), 'w') as fh:
            json.dump(result, fh, indent=2, default=str)

        print(f'\n{"=" * 68}')
        print(f'  best phase {self.best["phase"]}: '
              f'valid={self.best["valid_score"]:.4f} '
              f'test={self.best["test_score"]:.4f} ({self.metric})')
        print(f'  elapsed {result["elapsed_sec"] / 60:.1f} min')
        print(f'{"=" * 68}\n')
        return result
