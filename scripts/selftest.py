"""Correctness checks for the pieces that are easy to get subtly wrong.

Run this before any training. Two of these checks are load-bearing:

``evaluator``  the batched lock-step interpreter is compared against a plain
               Python postfix interpreter, tree by tree. A stack-index slip here
               would silently evaluate the wrong formula and every downstream
               number would be meaningless but plausible.
``split``      ``SplitUniMol`` with sigma=0 must reproduce the stock
               ``UniMolModel`` CLS representation. The whole project rests on
               the rewritten suffix being the same function as the original, and
               the accumulated pair bias makes that a real thing to verify, not
               a formality.

    python scripts/selftest.py            # everything
    python scripts/selftest.py evaluator  # one check
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.head import ops as O
from src.head import ridge
from src.head.evaluator import evaluate
from src.head.genome import (
    Population,
    crossover,
    is_valid,
    mutate,
    random_population,
    subtree_sizes,
    to_infix,
)


def _reference_eval(pop: Population, i: int, x: np.ndarray) -> np.ndarray:
    """Plain Python postfix interpreter -- the ground truth for `evaluator`."""
    stack: list[np.ndarray] = []
    for t in range(int(pop.length[i])):
        op = int(pop.code[i, t])
        if op == O.NOP:
            continue
        if op == O.VAR:
            stack.append(x[:, int(pop.arg[i, t])].astype(np.float64))
        elif op == O.CONST:
            stack.append(np.full(x.shape[0], float(pop.const[i, t]), dtype=np.float64))
        elif O.ARITY[op] == 1:
            a = stack.pop()
            if op == O.TANH:
                r = np.tanh(a)
            elif op == O.EXP:
                r = np.exp(np.clip(a, -O.EXP_CLAMP, O.EXP_CLAMP))
            elif op == O.LOG:
                r = np.log(np.abs(a) + O.LOG_EPS)
            elif op == O.SQRT:
                r = np.sqrt(np.abs(a))
            else:
                r = a * a
            stack.append(np.clip(np.nan_to_num(r, posinf=O.OUT_CLAMP,
                                               neginf=-O.OUT_CLAMP),
                                 -O.OUT_CLAMP, O.OUT_CLAMP))
        else:
            b, a = stack.pop(), stack.pop()
            if op == O.ADD:
                r = a + b
            elif op == O.SUB:
                r = a - b
            elif op == O.MUL:
                r = a * b
            else:
                safe = np.where(np.abs(b) < O.DIV_EPS,
                                np.where(b < 0, -O.DIV_EPS, O.DIV_EPS), b)
                r = a / safe
            stack.append(np.clip(np.nan_to_num(r, posinf=O.OUT_CLAMP,
                                               neginf=-O.OUT_CLAMP),
                                 -O.OUT_CLAMP, O.OUT_CLAMP))
    return stack[-1]


def check_evaluator(device: str = 'cpu') -> None:
    rng = np.random.default_rng(0)
    cols = np.arange(24)
    pop = random_population(200, cols, rng, max_depth=5, min_depth=1)
    x = rng.normal(size=(97, 24)).astype(np.float32)

    got = evaluate(pop, torch.from_numpy(x).to(device)).double().cpu().numpy()
    worst = 0.0
    for i in range(pop.size):
        want = _reference_eval(pop, i, x)
        worst = max(worst, float(np.max(np.abs(got[i] - want))))
    assert worst < 1e-4, f'evaluator mismatch, max |diff| = {worst:.3e}'
    print(f'  evaluator      ok   (max |diff| = {worst:.2e}, {pop.size} trees)')


def check_genome() -> None:
    rng = np.random.default_rng(1)
    cols = np.arange(32)
    pop = random_population(400, cols, rng, max_depth=5, min_depth=1)
    assert is_valid(pop.code, pop.length).all(), 'initial population is not valid postfix'

    sizes = subtree_sizes(pop.code)
    rows = np.arange(pop.size)
    assert (sizes[rows, pop.length - 1] == pop.length).all(), \
        'root subtree size must equal tree length'

    for _ in range(6):
        kids = crossover(pop, pop[rng.permutation(pop.size)], rng)
        kids = mutate(kids, rng, cols)
        assert is_valid(kids.code, kids.length).all(), 'variation produced invalid trees'
        assert (kids.length <= pop.max_len).all(), 'variation overflowed max_len'
        assert np.isin(kids.arg[kids.code == O.VAR], cols).all(), \
            'variation escaped the tree region'
        pop = kids
    print(f'  genome         ok   (6 generations, mean size {pop.length.mean():.1f})')
    print(f'                      sample: {to_infix(pop, 0)[:100]}')


def check_ridge(device: str = 'cpu') -> None:
    rng = np.random.default_rng(2)
    n, k = 120, 6
    phi = rng.normal(size=(n, k))
    phi[:, 0] = 1.0
    beta = rng.normal(size=k)
    y = phi @ beta + 0.1 * rng.normal(size=n)
    fold_of = rng.integers(0, 4, size=n)

    phi_t = torch.tensor(phi, dtype=torch.float32, device=device).unsqueeze(0)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    folds = [torch.from_numpy(np.nonzero(fold_of == f)[0]).to(device) for f in range(4)]
    alpha = 1e-3 * n
    pen = ridge.penalty_vector(k, alpha, n_unpenalized=1)
    got = float(ridge.cv_score(phi_t, y_t, folds, pen)[0])

    # Reference: fit each fold with an explicit numpy solve.
    sse = 0.0
    pen_np = np.diag(pen)
    for f in range(4):
        te = fold_of == f
        tr = ~te
        a = phi[tr].T @ phi[tr] + pen_np
        b = np.linalg.solve(a, phi[tr].T @ y[tr])
        sse += float(((phi[te] @ b - y[te]) ** 2).sum())
    want = (sse / n) ** 0.5
    assert abs(got - want) < 1e-5, f'cv_score {got:.8f} != reference {want:.8f}'
    print(f'  ridge cv       ok   (rmse {got:.6f})')

    # A batch of identical designs must produce identical scores.
    batched = ridge.cv_score(phi_t.repeat(8, 1, 1), y_t, folds, pen)
    assert float(batched.std()) < 1e-6, 'batched solve is not member-independent'
    print('  ridge batched  ok')


def check_split(device: str = 'cuda', n_mol: int = 6) -> None:
    """sigma=0 through SplitUniMol must equal the stock UniMolModel CLS."""
    from unimol_tools.data import ConformerGen
    from unimol_tools.models.unimol import UniMolModel

    from src.es.forward_unimol import ESSpec, SplitUniMol

    smiles = [
        'CCO', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O', 'CCN(CC)CC',
        'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O', 'Clc1ccc(cc1)C(c1ccccc1)N1CCNCC1',
    ][:n_mol]

    torch.manual_seed(0)
    model = UniMolModel(output_dim=1, data_type='molecule', remove_hs=False)
    model = model.to(device).eval()
    inputs = ConformerGen(remove_hs=False).transform(smiles)
    batch, _ = model.batch_collate_fn([(d, 0.0) for d in inputs])
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        want = model(**batch, return_repr=True)['cls_repr']

    for n_layers in (1, 4, 15):
        split = SplitUniMol(model, ESSpec(n_layers=n_layers), dtype=torch.float32)
        with torch.no_grad():
            x0, bias0 = split.prefix(batch)
            got = split.suffix(x0, bias0, n_members=1, pert=None)[0]
        err = float((got - want).abs().max())
        rel = err / float(want.abs().max())
        assert rel < 1e-4, (
            f'suffix mismatch at n_layers={n_layers}: max |diff| = {err:.3e} '
            f'(relative {rel:.3e})'
        )
        print(f'  split L={n_layers:<2}      ok   (max |diff| = {err:.2e})')

    # A population of N identical unperturbed members must be identical too.
    split = SplitUniMol(model, ESSpec(n_layers=4), dtype=torch.float32)
    with torch.no_grad():
        x0, bias0 = split.prefix(batch)
        many = split.suffix(x0, bias0, n_members=4, pert=None)
    spread = float((many - many[0:1]).abs().max())
    assert spread < 1e-5, f'population members diverged without perturbation: {spread:.3e}'
    print('  split popN     ok')


CHECKS = {
    'genome': lambda dev: check_genome(),
    'evaluator': check_evaluator,
    'ridge': check_ridge,
    'split': check_split,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # No `choices=` here: with nargs='*' on a positional, argparse before 3.10
    # validates the empty default against choices and rejects its own default.
    ap.add_argument('checks', nargs='*', metavar='CHECK',
                    help=f'which checks to run, from {{{", ".join(CHECKS)}}} '
                         f'(default: all)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    unknown = [c for c in args.checks if c not in CHECKS]
    if unknown:
        ap.error(f'unknown check(s): {", ".join(unknown)}. '
                 f'Choose from: {", ".join(CHECKS)}')
    names = args.checks or list(CHECKS)
    print(f'self-test on {args.device}\n')
    failed = []
    for name in names:
        try:
            CHECKS[name](args.device)
        except AssertionError as exc:
            print(f'  {name:<14} FAIL {exc}')
            failed.append(name)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f'  {name:<14} ERROR {type(exc).__name__}: {exc}')
            failed.append(name)

    print()
    if failed:
        print(f'FAILED: {", ".join(failed)}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
