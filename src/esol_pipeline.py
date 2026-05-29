"""ESOL data pipeline + full-batch MSE fitness for EGGROLL fine-tuning (Task 5).

Turns molecules into the JAX arrays the UniMol v1 port consumes, and provides
the fitness used by the EGGROLL training loop.

Featurization is a faithful transcription of UniMol v1's `coords2unimol`
(`unimol_source/unimol_tools/data/conformer.py`):

    src_tokens   = [CLS] + atom-type indices + [SEP]            (L,)
    src_coord    = mean-centred coords, with zero rows for CLS/SEP   (L, 3)
    src_distance = pairwise Euclidean distance matrix of src_coord   (L, L)
    src_edge_type = src_tokens[:, None] * |dict| + src_tokens[None, :]  (L, L)

3D conformers come from RDKit (ETKDGv3 + MMFF), mirroring `inner_smi2coords`.

IMPORTANT -- dictionary consistency: the atom-type Dictionary used here MUST
be the exact one the pretrained checkpoint was trained with (token IDs index
`embed_tokens` rows). Use `Dictionary.load(official_mol_dict_path)` for a real
run. `build_test_dictionary` is for standalone testing only.

The fitness is full-batch and deterministic: every population member sees the
same molecule batch, predicts each one, and  f_i = -MSE_i.
"""
import csv
from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial import distance_matrix as _scipy_distance_matrix

from unimol_jax import UniMolV1, UniMolV1Config


# ==========================================================================
# Dictionary  (atom-type vocabulary; transcribed from unimol_tools)
# ==========================================================================
class Dictionary:
    """Atom-type vocabulary. Token IDs index the model's embed_tokens rows."""

    def __init__(self, bos="[CLS]", pad="[PAD]", eos="[SEP]", unk="[UNK]"):
        self.bos_word, self.unk_word = bos, unk
        self.pad_word, self.eos_word = pad, eos
        self.symbols, self.count, self.indices = [], [], {}
        self.specials = {bos, pad, eos, unk}

    def __len__(self):
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        if sym in self.indices:
            return self.indices[sym]
        return self.indices[self.unk_word]

    def add_symbol(self, word, n=1, overwrite=False, is_special=False):
        if is_special:
            self.specials.add(word)
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] += n
            return idx
        idx = len(self.symbols)
        self.indices[word] = idx
        self.symbols.append(word)
        self.count.append(n)
        return idx

    def bos(self):
        return self.index(self.bos_word)

    def pad(self):
        return self.index(self.pad_word)

    def eos(self):
        return self.index(self.eos_word)

    def unk(self):
        return self.index(self.unk_word)

    def add_from_file(self, f):
        if isinstance(f, str):
            with open(f, "r", encoding="utf-8") as fd:
                self.add_from_file(fd)
            return
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            word, *rest = line.split()
            count = int(rest[0]) if rest else 1
            self.add_symbol(word, n=count, overwrite=False)

    @classmethod
    def load(cls, path):
        """Load the official UniMol molecule dict file (one `symbol count`
        per line). The pretrained checkpoint REQUIRES this exact file."""
        d = cls()
        d.add_from_file(path)
        d.add_symbol("[MASK]", is_special=True)        # appended after load
        return d


# common organic elements -- a TEST vocabulary, NOT the official checkpoint dict
_TEST_ATOMS = ("C", "H", "O", "N", "S", "P", "F", "Cl", "Br", "I", "B", "Si")


def build_test_dictionary(atom_symbols=_TEST_ATOMS):
    """A small Dictionary for standalone testing. NOT checkpoint-compatible --
    a real run must use Dictionary.load(official mol.dict.txt)."""
    d = Dictionary()
    for sp in ("[CLS]", "[PAD]", "[SEP]", "[UNK]"):
        d.add_symbol(sp, is_special=True)
    for a in atom_symbols:
        d.add_symbol(a)
    d.add_symbol("[MASK]", is_special=True)
    return d


def esol_config(dictionary, **overrides):
    """A UniMolV1Config wired to a Dictionary (vocab_size / padding_idx)."""
    params = dict(vocab_size=len(dictionary), padding_idx=dictionary.pad())
    params.update(overrides)
    return UniMolV1Config(**params)


# ==========================================================================
# featurization
# ==========================================================================
def featurize_from_coords(atoms, coordinates, dictionary, max_atoms=256,
                          remove_hs=False, seed=42):
    """Atom symbols + 3D coords -> UniMol v1 feature dict (RDKit-free core).

    Faithful to `coords2unimol` / `inner_coords`.
    """
    atoms = list(atoms)
    coordinates = np.asarray(coordinates, dtype=np.float32)

    if remove_hs:                                       # inner_coords
        keep = [i for i, s in enumerate(atoms) if s != "H"]
        atoms = [atoms[i] for i in keep]
        coordinates = coordinates[keep]

    if len(atoms) > max_atoms:                          # deterministic crop
        rng = np.random.default_rng(seed=seed)
        idx = np.sort(rng.choice(len(atoms), size=max_atoms, replace=False))
        atoms = [atoms[i] for i in idx]
        coordinates = coordinates[idx]

    src_tokens = np.array(
        [dictionary.bos()]
        + [dictionary.index(a) for a in atoms]
        + [dictionary.eos()])

    src_coord = coordinates - coordinates.mean(axis=0)
    src_coord = np.concatenate(
        [np.zeros((1, 3), np.float32), src_coord, np.zeros((1, 3), np.float32)],
        axis=0)

    src_distance = _scipy_distance_matrix(src_coord, src_coord)
    src_edge_type = (src_tokens.reshape(-1, 1) * len(dictionary)
                     + src_tokens.reshape(1, -1))

    return dict(
        src_tokens=src_tokens.astype(np.int32),
        src_distance=src_distance.astype(np.float32),
        src_coord=src_coord.astype(np.float32),
        src_edge_type=src_edge_type.astype(np.int32),
    )


def smiles_to_atoms_coords(smiles, seed=42, remove_hs=False):
    """SMILES -> (atom symbols, 3D coords) via RDKit ETKDGv3 + MMFF.

    Mirrors `inner_smi2coords`: AddHs, ETKDGv3 embed (random-coords fallback),
    MMFF/UFF minimisation.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)

    ps = AllChem.ETKDGv3()
    ps.randomSeed = int(seed)
    cid = AllChem.EmbedMolecule(mol, ps)
    if cid < 0:                                          # random-coords retry
        ps.useRandomCoords = True
        ps.maxAttempts = 1000
        cid = AllChem.EmbedMolecule(mol, ps)
    if cid < 0:
        raise RuntimeError(f"conformer embedding failed for {smiles!r}")

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, confId=cid)
        else:
            AllChem.UFFOptimizeMolecule(mol, confId=cid)
    except Exception:
        pass

    conf = mol.GetConformer(cid)
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = conf.GetPositions().astype(np.float32)
    if remove_hs:
        keep = [i for i, s in enumerate(atoms) if s != "H"]
        atoms = [atoms[i] for i in keep]
        coords = coords[keep]
    return atoms, coords


def featurize_smiles(smiles, dictionary, max_atoms=256, remove_hs=False, seed=42):
    """SMILES -> UniMol v1 feature dict (conformer + featurization)."""
    atoms, coords = smiles_to_atoms_coords(smiles, seed=seed, remove_hs=remove_hs)
    # remove_hs already applied above; pass remove_hs=False to avoid double work
    return featurize_from_coords(atoms, coords, dictionary,
                                 max_atoms=max_atoms, remove_hs=False, seed=seed)


# ==========================================================================
# dataset loading
# ==========================================================================
def load_esol_csv(path, smiles_col="smiles",
                   target_col="measured log solubility in mols per litre"):
    """Read an ESOL-style CSV -> (smiles list, target array).

    Defaults match the MoleculeNet ESOL (Delaney) column names.
    """
    smiles, targets = [], []
    with open(path, newline="") as fd:
        reader = csv.DictReader(fd)
        if smiles_col not in reader.fieldnames:
            raise KeyError(f"column {smiles_col!r} not in CSV "
                           f"(have {reader.fieldnames})")
        if target_col not in reader.fieldnames:
            raise KeyError(f"column {target_col!r} not in CSV "
                           f"(have {reader.fieldnames})")
        for row in reader:
            smiles.append(row[smiles_col])
            targets.append(float(row[target_col]))
    return smiles, np.asarray(targets, dtype=np.float64)


class TargetScaler:
    """Standardise regression targets (z-score). MSE is better-conditioned in
    standardised space; keep the scaler to report errors in real units."""

    def __init__(self, targets):
        t = np.asarray(targets, dtype=np.float64)
        self.mean = float(t.mean())
        self.std = float(t.std())
        if self.std == 0.0:
            self.std = 1.0

    def transform(self, t):
        return (np.asarray(t, np.float64) - self.mean) / self.std

    def inverse(self, t):
        return np.asarray(t, np.float64) * self.std + self.mean


# ==========================================================================
# batching
# ==========================================================================
Batch = namedtuple("Batch", "src_tokens src_distance src_edge_type targets lengths")


def build_batch(features, targets, dictionary, dtype=jnp.float32, pad_to=None):
    """Pad a list of feature dicts to a common length -> a Batch of JAX arrays.

    Padding follows UniMol's collate: tokens / edge_type with padding_idx,
    distances with 0. The model derives its mask from (src_tokens == pad_idx),
    so padded positions are masked regardless of the other pad values.
    """
    pad_idx = dictionary.pad()
    B = len(features)
    L = max(f["src_tokens"].shape[0] for f in features)
    if pad_to is not None:
        L = max(L, pad_to)

    tok = np.full((B, L), pad_idx, dtype=np.int32)
    dist = np.zeros((B, L, L), dtype=np.float32)
    edge = np.full((B, L, L), pad_idx, dtype=np.int32)
    lengths = np.zeros(B, dtype=np.int32)

    for i, f in enumerate(features):
        n = f["src_tokens"].shape[0]
        lengths[i] = n
        tok[i, :n] = f["src_tokens"]
        dist[i, :n, :n] = f["src_distance"]
        edge[i, :n, :n] = f["src_edge_type"]

    return Batch(
        src_tokens=jnp.asarray(tok),
        src_distance=jnp.asarray(dist, dtype),
        src_edge_type=jnp.asarray(edge),
        targets=jnp.asarray(np.asarray(targets, np.float64), dtype),
        lengths=jnp.asarray(lengths),
    )


def featurize_dataset(smiles_list, targets, dictionary, max_atoms=256,
                      remove_hs=False, seed=42, dtype=jnp.float32,
                      standardize=True, scaler=None):
    """End-to-end: SMILES list -> (Batch, TargetScaler | None).

    Skips molecules whose conformer generation fails.

    If `scaler` is given it is REUSED (the test split must be standardised
    with the scaler fitted on the train split). Otherwise, when standardize
    is True, a fresh TargetScaler is fitted on these targets.
    """
    feats, ys = [], []
    for smi, y in zip(smiles_list, targets):
        try:
            feats.append(featurize_smiles(smi, dictionary, max_atoms=max_atoms,
                                          remove_hs=remove_hs, seed=seed))
            ys.append(y)
        except Exception as e:                           # robust to bad SMILES
            print(f"  [skip] {smi!r}: {e}")
    if not feats:
        raise RuntimeError("no molecules could be featurised")

    ys = np.asarray(ys, np.float64)
    if scaler is not None:                               # reuse train scaler
        ys = scaler.transform(ys)
    elif standardize:
        scaler = TargetScaler(ys)
        ys = scaler.transform(ys)
    return build_batch(feats, ys, dictionary, dtype=dtype), scaler


# ==========================================================================
# fitness  --  full-batch MSE,  f_i = -MSE_i
# ==========================================================================
def evaluate_clean(noiser, fnp, npp, fp, params, es_tree_key, batch):
    """Unperturbed evaluation (iterinfo=None): returns (predictions, MSE).

    This is the baseline -- the fitness of the current theta with no noise.
    """
    def one(tok, dist, edge):
        return UniMolV1.forward(noiser, fnp, npp, fp, params, es_tree_key,
                                None, tok, dist, edge)
    preds = jax.vmap(one)(batch.src_tokens, batch.src_distance,
                          batch.src_edge_type)
    mse = jnp.mean((preds - batch.targets) ** 2)
    return preds, mse


def evaluate_population(noiser, fnp, npp, fp, params, es_tree_key,
                        epoch, n_pop, batch, pop_chunk=None):
    """Full-batch MSE fitness for an EGGROLL population.

    Every population member i (iterinfo = (epoch, i)) predicts the SAME batch
    of molecules; fitness_i = -MSE_i. Antithetic +/- pairs are produced by the
    noiser from the thread_id parity.

    `pop_chunk` controls memory: with `pop_chunk=None` the whole population is
    vmapped at once (fine for small runs / tests). For a real run, a full
    UniMol-v1 population of 512+ vmapped at once OOMs a 16 GB GPU -- set
    `pop_chunk` to a small divisor of `n_pop` (e.g. 8-32); the population is
    then processed in chunks via `jax.lax.map` (chunks sequential, members
    within a chunk vmapped). `n_pop` must be divisible by `pop_chunk`.

    Returns (fitness, mse), each shape (n_pop,). `n_pop`/`pop_chunk` are
    static for jit.
    """
    def member(thread_id):
        def one(tok, dist, edge):
            return UniMolV1.forward(noiser, fnp, npp, fp, params, es_tree_key,
                                    (epoch, thread_id), tok, dist, edge)
        preds = jax.vmap(one)(batch.src_tokens, batch.src_distance,
                              batch.src_edge_type)
        mse = jnp.mean((preds - batch.targets) ** 2)
        return -mse, mse

    thread_ids = jnp.arange(n_pop)
    if pop_chunk is None or pop_chunk >= n_pop:
        return jax.vmap(member)(thread_ids)

    if n_pop % pop_chunk != 0:
        raise ValueError(f"n_pop ({n_pop}) must be divisible by "
                         f"pop_chunk ({pop_chunk})")
    chunks = thread_ids.reshape(n_pop // pop_chunk, pop_chunk)
    fitness, mse = jax.lax.map(jax.vmap(member), chunks)   # (n_chunks, pop_chunk)
    return fitness.reshape(-1), mse.reshape(-1)