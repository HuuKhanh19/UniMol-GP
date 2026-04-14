# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import os
import warnings

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.spatial import distance_matrix

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings(action='ignore')
from multiprocessing import Pool

from numba import njit
from tqdm import tqdm

from ..config import MODEL_CONFIG
from ..utils import logger
from ..weights import WEIGHT_DIR, weight_download
from .dictionary import Dictionary
# https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py
# allowable multiple choice node and edge features
allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OCTAHEDRAL",
        "CHI_SQUAREPLANAR",
        "CHI_OTHER",
    ],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
    "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "possible_bond_stereo_list": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "possible_is_conjugated_list": [False, True],
}


class ConformerGen(object):
    
    '''
    This class designed to generate conformers for molecules represented as SMILES strings using provided parameters and configurations. The `transform` method uses multiprocessing to speed up the conformer generation process.
    '''

    def __init__(self, **params):

        """
        Initializes the neural network model based on the provided model name and parameters.

        :param model_name: (str) The name of the model to initialize.
        :param params: Additional parameters for model configuration.

        :return: An instance of the specified neural network model.
        :raises ValueError: If the model name is not recognized.
        """
        self.fl=1
        self._init_features(**params)

    def _init_features(self, **params):
        """
        Initializes the features of the ConformerGen object based on provided parameters.

        :param params: Arbitrary keyword arguments for feature configuration.
                       These can include the random seed, maximum number of atoms, data type,
                       generation method, generation mode, and whether to remove hydrogens.
        """
        self.seed = params.get('seed', 42)
        self.max_atoms = params.get('max_atoms', 256)
        self.data_type = params.get('data_type', 'molecule')
        self.method = params.get('method', 'rdkit_random')
        self.mode = params.get('mode', 'fast')
        self.remove_hs = params.get('remove_hs', False)
        self.n_confomer = params.get('n_confomer', 10)
        if self.data_type == 'molecule':
            name = "no_h" if self.remove_hs else "all_h"
            name = self.data_type + '_' + name
            self.dict_name = MODEL_CONFIG['dict'][name]
        else:
            self.dict_name = MODEL_CONFIG['dict'][self.data_type]
        if not os.path.exists(os.path.join(WEIGHT_DIR, self.dict_name)):
            weight_download(self.dict_name, WEIGHT_DIR)
        self.dictionary = Dictionary.load(os.path.join(WEIGHT_DIR, self.dict_name))
        self.dictionary.add_symbol("[MASK]", is_special=True)
        if os.name == 'posix':
            self.multi_process = params.get('multi_process', True)
        else:
            self.multi_process = params.get('multi_process', False)
            if self.multi_process:
                logger.warning(
                    'Please use "if __name__ == "__main__":" to wrap the main function when using multi_process on Windows.'
                )

    def single_process(self, smiles):
    
        """
        Processes a single SMILES string to generate conformers using the specified method.

        :param smiles: (str) The SMILES string representing the molecule.
        :return: A unimolecular data representation (dictionary) of the molecule.
        :raises ValueError: If the conformer generation method is unrecognized.
        """
        if self.method == 'rdkit_random':
            atoms, coordinates, energies  = inner_smi2coords(
                smiles, seed=self.seed, mode=self.mode, n_confs=self.n_confomer,return_energy=True
            )
            # print(1)
            # print(atoms, coordinates)


            feat = coords2unimol(
                atoms,
                coordinates,
                self.dictionary,
                self.max_atoms,
                remove_hs=self.remove_hs,
                seed=self.seed
            )
            # print(2)
            for i in range(min(len(feat), len(energies))):
                feat[i]['energy'] = float(energies[i])

            # nếu thiếu conformer vì fail, vẫn an toàn
            for i in range(len(energies), len(feat)):
                feat[i]['energy'] = float('inf')
            return feat
        else:
            raise ValueError(
                'Unknown conformer generation method: {}'.format(self.method)
            )

    def transform_raw(self, atoms_list, coordinates_list):

        inputs = []
        for atoms, coordinates in zip(atoms_list, coordinates_list):
            inputs.append(
                coords2unimol(
                    atoms,
                    coordinates,
                    self.dictionary,
                    self.max_atoms,
                    remove_hs=self.remove_hs,
                    seed=self.seed
                )
            )
        return inputs

    def transform_mols(self, mols_list):
        inputs = []
        for mol in mols_list:
            atoms = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
            coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            inputs.append(
                coords2unimol(
                    atoms,
                    coordinates,
                    self.dictionary,
                    self.max_atoms,
                    remove_hs=self.remove_hs,
                )
            )
        return inputs

    def transform(self, smiles_list):
        logger.info('Start generating conformers...')
        if self.multi_process:
            pool = Pool(processes=min(8, os.cpu_count()))
            results = [item for item in tqdm(pool.imap(self.single_process, smiles_list))]
            pool.close()
        else:
            results = [self.single_process(smiles) for smiles in tqdm(smiles_list)]

        inputs = [feat for sublist in results for feat in sublist]
        mols = None

        failed_conf = [(item['src_coord'] == 0.0).all() for item in inputs]
        logger.info(
            'Succeeded in generating conformers for {:.2f}% of molecules.'.format(
                (1 - np.mean(failed_conf)) * 100
            )
        )

        k = self.n_confomer
        failed_conf_indices = [i for i, v in enumerate(failed_conf) if v]
        if len(failed_conf_indices) > 0:
            failed_mol_idx = sorted(set(i // k for i in failed_conf_indices))
            logger.info('Failed conformers indices: {}'.format(failed_conf_indices[:50]))
            logger.debug('Failed conformers SMILES: {}'.format([smiles_list[i] for i in failed_mol_idx[:50]]))

        failed_conf_3d = [(item['src_coord'][:, 2] == 0.0).all() for item in inputs]
        logger.info(
            'Succeeded in generating 3d conformers for {:.2f}% of molecules.'.format(
                (1 - np.mean(failed_conf_3d)) * 100
            )
        )

        failed_conf_3d_indices = [i for i, v in enumerate(failed_conf_3d) if v]
        if len(failed_conf_3d_indices) > 0:
            failed_mol_idx_3d = sorted(set(i // k for i in failed_conf_3d_indices))
            logger.info('Failed 3d conformers indices: {}'.format(failed_conf_3d_indices[:50]))
            logger.debug('Failed 3d conformers SMILES: {}'.format([smiles_list[i] for i in failed_mol_idx_3d[:50]]))

        return inputs, mols

def _minimize_energy(mol, conf_id=0):
    """Try MMFF, else UFF. Returns energy (float) or np.inf if fails."""
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=conf_id)
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
        if ff is None:
            return np.inf
        ff.Minimize()
        return float(ff.CalcEnergy())
    except Exception:
        return np.inf

def inner_smi2coords(
    smi, seed=42, mode='fast', optimize=True, n_confs=3, prune_conf=False, return_2d=False, return_energy=False
):
    """
    Robust SMILES->3D coords:
    - Replace '*' (dummy) atoms with H in a working copy for embedding
    - ETKDGv3 with retries (random coords, multi-conformer)
    - MMFF/UFF minimization, 2D fallback
    - Remove original '*' atoms from the final output (regardless of remove_hs)
    - Optionally remove hydrogens from the final output
    """
    # Parse SMILES (try to be robust with '*')
    
    coords = None
    conf_ids = []
    
    # Prepare ETKDGv3 params
    def _embed_with_params(m, n_confs=1, use_random=False, max_attempts=200, pruneRmsThresh=0.5):
        ps = AllChem.ETKDGv3()
        ps.randomSeed = seed
        ps.useRandomCoords = bool(use_random)
        ps.maxAttempts = int(max_attempts)
        ps.numThreads = 0 
        # check this code
        if prune_conf:
            ps.pruneRmsThresh = float(pruneRmsThresh)
        return list(AllChem.EmbedMultipleConfs(m, numConfs=int(n_confs), params=ps))
    try:
        work_mol_no_H = Chem.MolFromSmiles(smi)
        work_mol = AllChem.AddHs(work_mol_no_H)
    except Exception as e:
        print(f"An error with smi {smi}, {e}")
        return [None], [None]
    if work_mol is None:
        return [None], [None]
    
    if len(work_mol.GetAtoms()) > 400 or return_2d:
        print("large")
        # return 2D coords for very large molecules
        orig_atoms = [a.GetSymbol() for a in work_mol.GetAtoms()]
        keep_idx = [i for i, sym in enumerate(orig_atoms) if sym != '*']
        atoms = [sym for sym in orig_atoms if sym != '*']
        try:
            AllChem.Compute2DCoords(work_mol)
            conf = work_mol.GetConformer()
            coords2d = conf.GetPositions().astype(np.float32)
            coords = coords2d  # (N,3) with z=0
            
        except Exception:
            # Final fallback: zeros
            coords = np.zeros((work_mol.GetNumAtoms(), 3), dtype=np.float32)
        
        coordinates = coords[keep_idx]
        assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smi)
        return [atoms], [coordinates]
        

    # 1) quick single conformer
    conf_ids = _embed_with_params(work_mol, n_confs=n_confs, use_random=False, max_attempts=200)
    # print(f'Generated {len(conf_ids)} conformers for SMILES: {smi}')

    # 2) few conformers, same seed
    if len(conf_ids) == 0 and mode in ('heavy', 'fast'):
        conf_ids = _embed_with_params(work_mol, n_confs=n_confs, use_random=False, max_attempts=500)
    # 3) random coords fallback
    if len(conf_ids) == 0 and mode in ('heavy', 'fast'):
        conf_ids = _embed_with_params(work_mol, n_confs=n_confs, use_random=False,  max_attempts=800)
    # 4) random coords fallback
    if len(conf_ids) == 0 and mode in ('heavy', 'fast'):
        conf_ids = _embed_with_params(work_mol, n_confs=n_confs, use_random=True,  max_attempts=1000)
    # 5) random coords fallback, heavy mode only
    if len(conf_ids) == 0 and mode == 'heavy':
        conf_ids = _embed_with_params(work_mol, n_confs=n_confs, use_random=False, max_attempts=2000)
    # 6) random coords fallback, heavy mode only
    if len(conf_ids) == 0 and mode == 'heavy':
        conf_ids = _embed_with_params(work_mol, n_confs=n_confs, use_random=True, max_attempts=5000)

  
    all_confs_coords = []
    all_energies = []
    
    # Build atom symbols from ORIGINAL molecule (before capping),
    # so we know which were '*' and can drop them deterministically.
    orig_atoms = [a.GetSymbol() for a in work_mol.GetAtoms()]
    keep_idx = [i for i, sym in enumerate(orig_atoms) if sym != '*']
    atoms = [sym for sym in orig_atoms if sym != '*']
    
    
    for cid in conf_ids:
        coords = None
        e = np.inf
        if optimize:
            e = _minimize_energy(work_mol, conf_id=cid)
        try:
            conf = work_mol.GetConformer(int(cid))
            coords = conf.GetPositions().astype(np.float32)
        except Exception:
            coords = None

        # Fallback: 2D coords (Z=0)
        if coords is None:
            try:
                AllChem.Compute2DCoords(work_mol)
                conf = work_mol.GetConformer()
                coords2d = conf.GetPositions().astype(np.float32)
                coords = coords2d  # (N,3) with z=0
            except Exception:
                # Final fallback: zeros
                coords = np.zeros((work_mol.GetNumAtoms(), 3), dtype=np.float32)
        
        if coords.shape[0] < len(orig_atoms):
            # This shouldn’t happen with AddHs-before-embed; guard anyway.
            # pad = np.zeros((len(orig_atoms) - coords.shape[0], 3), dtype=np.float32)
            # coords = np.vstack([coords, pad])
            continue
                

        all_confs_coords.append(coords)
        all_energies.append(e)

    if len(all_confs_coords) == 0:
        # Final fallback: zeros
        coords = np.zeros((work_mol.GetNumAtoms(), 3), dtype=np.float32)
        all_confs_coords.append(coords)
        all_energies.append(np.inf)
    
    
    # Make sure shape matches: coordinates array should be at least the original core atoms count.
    all_confs_coords_new = []
    all_energies_new = []
    for coords, e in zip(all_confs_coords, all_energies):
        coordinates = coords[keep_idx]
        assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smi)
        all_confs_coords_new.append(coordinates)
        all_energies_new.append(e)
    
    assert len(atoms) == len(all_confs_coords_new[0]), "coordinates shape is not align with {}".format(smi)
    if return_energy:
        # (optional) đổi sang ΔE theo molecule để dùng weighting ổn hơn
        arr = np.array(all_energies_new, dtype=float)
        if np.isfinite(arr).any():
            arr = arr - np.nanmin(arr)
        return [atoms], all_confs_coords_new, arr.tolist()

    return [atoms], all_confs_coords_new

def inner_coords(atoms, coordinates, remove_hs=True):
    """
    Processes a list of atoms and their corresponding coordinates to remove hydrogen atoms if specified.
    This function takes a list of atom symbols and their corresponding coordinates and optionally removes hydrogen atoms from the output. It includes assertions to ensure the integrity of the data and uses numpy for efficient processing of the coordinates.

    :param atoms: (list) A list of atom symbols (e.g., ['C', 'H', 'O']).
    :param coordinates: (list of tuples or list of lists) Coordinates corresponding to each atom in the `atoms` list.
    :param remove_hs: (bool, optional) A flag to indicate whether hydrogen atoms should be removed from the output.
                      Defaults to True.

    :return: A tuple containing two elements; the filtered list of atom symbols and their corresponding coordinates.
             If `remove_hs` is False, the original lists are returned.

    :raises AssertionError: If the length of `atoms` list does not match the length of `coordinates` list.
    """
    assert len(atoms) == len(coordinates), "coordinates shape is not align atoms"
    coordinates = np.array(coordinates).astype(np.float32)
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[idx]
        assert len(atoms_no_h) == len(
            coordinates_no_h
        ), "coordinates shape is not align with atoms"
        return atoms_no_h, coordinates_no_h
    else:
        return atoms, coordinates


def coords2unimol(
    atoms,
    coordinates_list,
    dictionary,
    max_atoms=256,
    remove_hs=True,
    seed=42,
    **params
):
    """
    Supports single or multiple conformers.
    coordinates:
        - (N, 3)
        - (K, N, 3)
    """
    # print(atoms[0])
    if atoms[0] is None or coordinates_list[0] is None:
        print(atoms[0], coordinates_list[0])
        return [{
            'src_tokens': np.zeros((max_atoms + 2,), dtype=int),
            'src_distance': np.zeros((max_atoms + 2, max_atoms + 2), dtype=np.float32),
            'src_coord': np.zeros((max_atoms + 2, 3), dtype=np.float32),
            'src_edge_type': np.zeros((max_atoms + 2, max_atoms + 2), dtype=int),
        }]
    atoms_org = atoms[0]

    outputs = []
    idx = None 

    for coordinates in coordinates_list:

        # ---- single conformer ----
        atoms, coordinates = inner_coords(atoms_org, coordinates, remove_hs=remove_hs)

        if idx is None:
            # cropping
            if len(atoms) > max_atoms:
                rng = np.random.default_rng(seed=seed)

                idx = rng.choice(len(atoms), size=max_atoms, replace=False)
                idx = np.sort(idx)
            else:
                idx = np.arange(len(atoms))

        atoms = np.array(atoms)[idx]

        coordinates = coordinates[idx]

        coordinates = np.array(coordinates, dtype=np.float32)

        # tokens
        src_tokens = np.array(
            [dictionary.bos()]
            + [dictionary.index(atom) for atom in atoms]
            + [dictionary.eos()]
        )

        # normalize coords
        src_coord = coordinates - coordinates.mean(axis=0)
        src_coord = np.concatenate(
            [np.zeros((1, 3)), src_coord, np.zeros((1, 3))],
            axis=0
        )

        # distance matrix
        src_distance = distance_matrix(src_coord, src_coord)

        # edge type
        src_edge_type = (
            src_tokens.reshape(-1, 1) * len(dictionary)
            + src_tokens.reshape(1, -1)
        )

        outputs.append({ 
            'src_tokens': src_tokens.astype(int),
            'src_distance': src_distance.astype(np.float32),
            'src_coord': src_coord.astype(np.float32),
            'src_edge_type': src_edge_type.astype(int),
        })


    return outputs


class UniMolV2Feature(object):
    '''
    This class is responsible for generating features for molecules represented as SMILES strings. It uses the ConformerGen class to generate conformers for the molecules and converts the resulting atom symbols and coordinates into a unified molecular representation.
    '''

    def __init__(self, **params):
        """
        Initializes the neural network model based on the provided model name and parameters.

        :param model_name: (str) The name of the model to initialize.
        :param params: Additional parameters for model configuration.

        :return: An instance of the specified neural network model.
        :raises ValueError: If the model name is not recognized.
        """
        self._init_features(**params)

    def _init_features(self, **params):
        """
        Initializes the features of the UniMolV2Feature object based on provided parameters.

        :param params: Arbitrary keyword arguments for feature configuration.
                       These can include the random seed, maximum number of atoms, data type,
                       generation method, generation mode, and whether to remove hydrogens.
        """
        self.seed = params.get('seed', 42)
        self.max_atoms = params.get('max_atoms', 128)
        self.data_type = params.get('data_type', 'molecule')
        self.method = params.get('method', 'rdkit_random')
        self.mode = params.get('mode', 'fast')
        self.remove_hs = params.get('remove_hs', True)
        if os.name == 'posix':
            self.multi_process = params.get('multi_process', True)
        else:
            self.multi_process = params.get('multi_process', False)
            if self.multi_process:
                logger.warning(
                    'Please use "if __name__ == "__main__":" to wrap the main function when using multi_process on Windows.'
                )

    def single_process(self, smiles):
        """
        Processes a single SMILES string to generate conformers using the specified method.

        :param smiles: (str) The SMILES string representing the molecule.
        :return: A unimolecular data representation (dictionary) of the molecule.
        :raises ValueError: If the conformer generation method is unrecognized.
        """
        if self.method == 'rdkit_random':
            mol = inner_smi2coords(
                smiles,
                seed=self.seed,
                mode=self.mode,
                remove_hs=self.remove_hs,
                return_mol=True,
            )
            feat = mol2unimolv2(mol, self.max_atoms, remove_hs=self.remove_hs, seed=self.seed)
            return feat, mol
        else:
            raise ValueError(
                'Unknown conformer generation method: {}'.format(self.method)
            )

    def transform_raw(self, atoms_list, coordinates_list):

        inputs = []
        for atoms, coordinates in zip(atoms_list, coordinates_list):
            mol = create_mol_from_atoms_and_coords(atoms, coordinates)
            inputs.append(mol2unimolv2(mol, self.max_atoms, remove_hs=self.remove_hs))
        return inputs

    def transform_mols(self, mols_list):
        inputs = []
        for mol in mols_list:
            inputs.append(mol2unimolv2(mol, self.max_atoms, remove_hs=self.remove_hs))
        return inputs

    def transform(self, smiles_list):
        logger.info('Start generating conformers...')
        if self.multi_process:
            pool = Pool(processes=min(8, os.cpu_count()))
            results = [
                item for item in tqdm(pool.imap(self.single_process, smiles_list))
            ]
            pool.close()
        else:
            results = [self.single_process(smiles) for smiles in tqdm(smiles_list)]

        inputs, mols = zip(*results)
        inputs = list(inputs)
        mols = list(mols)

        failed_conf = [(item['src_coord'] == 0.0).all() for item in inputs]
        logger.info(
            'Succeeded in generating conformers for {:.2f}% of molecules.'.format(
                (1 - np.mean(failed_conf)) * 100
            )
        )
        failed_conf_indices = [
            index for index, value in enumerate(failed_conf) if value
        ]
        if len(failed_conf_indices) > 0:
            logger.info('Failed conformers indices: {}'.format(failed_conf_indices))
            logger.debug(
                'Failed conformers SMILES: {}'.format(
                    [smiles_list[index] for index in failed_conf_indices]
                )
            )

        failed_conf_3d = [(item['src_coord'][:, 2] == 0.0).all() for item in inputs]
        logger.info(
            'Succeeded in generating 3d conformers for {:.2f}% of molecules.'.format(
                (1 - np.mean(failed_conf_3d)) * 100
            )
        )
        failed_conf_3d_indices = [
            index for index, value in enumerate(failed_conf_3d) if value
        ]
        if len(failed_conf_3d_indices) > 0:
            logger.info(
                'Failed 3d conformers indices: {}'.format(failed_conf_3d_indices)
            )
            logger.debug(
                'Failed 3d conformers SMILES: {}'.format(
                    [smiles_list[index] for index in failed_conf_3d_indices]
                )
            )

        return inputs, mols


def create_mol_from_atoms_and_coords(atoms, coordinates):
    """
    Creates an RDKit molecule object from a list of atom symbols and their corresponding coordinates.

    :param atoms: (list) Atom symbols for the molecule.
    :param coordinates: (list) Atomic coordinates for the molecule.
    :return: RDKit molecule object.
    """
    mol = Chem.RWMol()
    atom_indices = []

    for atom in atoms:
        atom_idx = mol.AddAtom(Chem.Atom(atom))
        atom_indices.append(atom_idx)

    conf = Chem.Conformer(len(atoms))
    for i, coord in enumerate(coordinates):
        conf.SetAtomPosition(i, coord)

    mol.AddConformer(conf)
    Chem.SanitizeMol(mol)
    return mol


def mol2unimolv2(mol, max_atoms=128, remove_hs=True, seed=42, **params):
    """
    Converts atom symbols and coordinates into a unified molecular representation.

    :param mol: (rdkit.Chem.Mol) The molecule object containing atom symbols and coordinates.
    :param max_atoms: (int) The maximum number of atoms to consider for the molecule.
    :param remove_hs: (bool) Whether to remove hydrogen atoms from the representation. This must be True for UniMolV2.
    :param params: Additional parameters.

    :return: A batched data containing the molecular representation.
    """

    mol = AllChem.RemoveAllHs(mol)
    atoms = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)

    # cropping atoms and coordinates
    if len(atoms) > max_atoms:
        np.random.seed(seed)
        mask = np.zeros(len(atoms), dtype=bool)
        mask[:max_atoms] = True
        np.random.shuffle(mask)  # shuffle the mask
        atoms = atoms[mask]
        coordinates = coordinates[mask]
    else:
        mask = np.ones(len(atoms), dtype=bool)
    # tokens padding
    src_tokens = [AllChem.GetPeriodicTable().GetAtomicNumber(item) for item in atoms]
    src_coord = coordinates
    #
    node_attr, edge_index, edge_attr = get_graph(mol)
    feat = get_graph_features(edge_attr, edge_index, node_attr, drop_feat=0, mask=mask)
    feat['src_tokens'] = src_tokens
    feat['src_coord'] = src_coord
    return feat


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
        safe_index(allowable_features["possible_atomic_num_list"], atom.GetAtomicNum()),
        allowable_features["possible_chirality_list"].index(str(atom.GetChiralTag())),
        safe_index(allowable_features["possible_degree_list"], atom.GetTotalDegree()),
        safe_index(
            allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()
        ),
        safe_index(allowable_features["possible_numH_list"], atom.GetTotalNumHs()),
        safe_index(
            allowable_features["possible_number_radical_e_list"],
            atom.GetNumRadicalElectrons(),
        ),
        safe_index(
            allowable_features["possible_hybridization_list"],
            str(atom.GetHybridization()),
        ),
        allowable_features["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
        allowable_features["possible_is_in_ring_list"].index(atom.IsInRing()),
    ]
    return atom_feature


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
        safe_index(
            allowable_features["possible_bond_type_list"], str(bond.GetBondType())
        ),
        allowable_features["possible_bond_stereo_list"].index(str(bond.GetStereo())),
        allowable_features["possible_is_conjugated_list"].index(bond.GetIsConjugated()),
    ]
    return bond_feature


def get_graph(mol):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int32)
    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int32).T
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int32)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int32)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int32)
    return x, edge_index, edge_attr


def get_graph_features(edge_attr, edge_index, node_attr, drop_feat, mask):
    # atom_feat_sizes = [128] + [16 for _ in range(8)]
    atom_feat_sizes = [16 for _ in range(8)]
    edge_feat_sizes = [16, 16, 16]
    edge_attr, edge_index, x = edge_attr, edge_index, node_attr
    N = x.shape[0]

    # atom feature here
    atom_feat = convert_to_single_emb(x[:, 1:], atom_feat_sizes)

    # node adj matrix [N, N] bool
    adj = np.zeros([N, N], dtype=np.int32)
    adj[edge_index[0, :], edge_index[1, :]] = 1
    degree = adj.sum(axis=-1)

    # edge feature here
    if len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:, None]
    edge_feat = np.zeros([N, N, edge_attr.shape[-1]], dtype=np.int32)
    edge_feat[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr, edge_feat_sizes) + 1
    )
    shortest_path_result = floyd_warshall(adj)
    # max distance is 509
    if drop_feat:
        atom_feat[...] = 1
        edge_feat[...] = 1
        degree[...] = 1
        shortest_path_result[...] = 511
    else:
        atom_feat = atom_feat + 2
        edge_feat = edge_feat + 2
        degree = degree + 2
        shortest_path_result = shortest_path_result + 1

    # combine, plus 1 for padding
    feat = {}
    feat["atom_feat"] = atom_feat[mask]
    feat["atom_mask"] = np.ones(N, dtype=np.int64)[mask]
    feat["edge_feat"] = edge_feat[mask][:, mask]
    feat["shortest_path"] = shortest_path_result[mask][:, mask]
    feat["degree"] = degree.reshape(-1)[mask]
    # pair-type
    atoms = atom_feat[..., 0]
    pair_type = np.concatenate(
        [
            np.expand_dims(atoms, axis=(1, 2)).repeat(N, axis=1),
            np.expand_dims(atoms, axis=(0, 2)).repeat(N, axis=0),
        ],
        axis=-1,
    )
    pair_type = pair_type[mask][:, mask]
    feat["pair_type"] = convert_to_single_emb(pair_type, [128, 128])
    feat["attn_bias"] = np.zeros((mask.sum() + 1, mask.sum() + 1), dtype=np.float32)
    return feat


def convert_to_single_emb(x, sizes):
    assert x.shape[-1] == len(sizes)
    offset = 1
    for i in range(len(sizes)):
        assert (x[..., i] < sizes[i]).all()
        x[..., i] = x[..., i] + offset
        offset += sizes[i]
    return x


@njit
def floyd_warshall(M):
    (nrows, ncols) = M.shape
    assert nrows == ncols
    n = nrows
    # set unreachable nodes distance to 510
    for i in range(n):
        for j in range(n):
            if M[i, j] == 0:
                M[i, j] = 510

    for i in range(n):
        M[i, i] = 0

    # floyed algo
    for k in range(n):
        for i in range(n):
            for j in range(n):
                cost_ikkj = M[i, k] + M[k, j]
                if M[i, j] > cost_ikkj:
                    M[i, j] = cost_ikkj

    for i in range(n):
        for j in range(n):
            if M[i, j] >= 510:
                M[i, j] = 510
    return M

def _minimize_energy(mol, conf_id=0):
    """Try MMFF, else UFF. Returns energy (float) or np.inf if fails."""
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=conf_id)
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
        if ff is None:
            return np.inf
        ff.Minimize()
        return float(ff.CalcEnergy())
    except Exception:
        return np.inf