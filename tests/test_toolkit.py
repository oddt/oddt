import os
from collections import OrderedDict, deque

from six.moves.cPickle import loads, dumps
import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

import oddt
from oddt.spatial import rmsd
from oddt.toolkits.common import canonize_ring_path

test_data_dir = os.path.dirname(os.path.abspath(__file__))
xiap_receptor = os.path.join(test_data_dir, 'data', 'dude', 'xiap',
                             'receptor_rdkit.pdb')
xiap_actives = os.path.join(test_data_dir, 'data', 'dude', 'xiap',
                            'actives_docked.sdf')


def test_mol():
    """Test common molecule operations"""
    # Hydrogen manipulation in small molecules
    mol = oddt.toolkit.readstring('smi', 'c1ccccc1O')
    assert len(mol.atoms) == 7
    mol.addh()
    assert len(mol.atoms) == 13
    mol.removeh()
    mol.addh(only_polar=True)
    assert len(mol.atoms) == 8
    mol.removeh()
    assert len(mol.atoms) == 7

    # Hydrogen manipulation in proteins
    protein = next(oddt.toolkit.readfile('pdb', xiap_receptor))
    protein.protein = True

    res_atoms_n = [6, 10, 8, 8, 7, 11, 8, 7, 6, 8, 5, 8, 12, 9, 5, 11, 8,
                   11, 7, 11, 4, 7, 14, 8, 12, 6, 7, 8, 9, 9, 9, 8, 5, 11,
                   5, 4, 11, 12, 5, 8, 4, 9, 4, 8, 9, 7, 9, 6, 11, 10, 6,
                   4, 4, 4, 8, 7, 8, 14, 9, 7, 6, 9, 8, 7, 14, 9, 9, 10, 5,
                   9, 14, 12, 7, 4, 6, 9, 12, 8, 8, 9, 9, 9, 4, 9, 9, 12,
                   8, 8, 8, 8, 10, 8, 7, 10, 11, 12, 6, 7, 8, 11, 8, 9, 4,
                   8, 9, 7, 9, 6, 6, 4, 4, 4, 8, 7, 8, 14, 9, 7, 6, 9, 8,
                   7, 14, 9, 9, 10, 5, 9, 14, 12, 7, 4, 8, 10, 8, 7, 1, 1]
    res_atoms_n_addh = [12, 17, 17, 19, 14, 23, 14, 14, 11, 17, 10, 13, 21,
                        16, 10, 23, 19, 20, 14, 20, 7, 14, 24, 19, 21, 11,
                        16, 14, 21, 16, 17, 19, 10, 23, 10, 7, 20, 21, 10,
                        19, 7, 16, 7, 13, 21, 16, 21, 10, 20, 17, 10, 7, 7,
                        7, 19, 14, 13, 24, 21, 14, 11, 16, 13, 14, 24, 16,
                        17, 16, 10, 21, 24, 21, 14, 7, 10, 21, 21, 19, 19,
                        16, 17, 21, 7, 17, 16, 21, 19, 14, 14, 19, 17, 19,
                        14, 18, 25, 22, 11, 17, 21, 22, 21, 17, 7, 13, 21,
                        16, 21, 11, 11, 7, 7, 7, 19, 14, 13, 24, 21, 14,
                        11, 16, 13, 14, 24, 16, 17, 16, 10, 21, 24, 21, 14,
                        8, 20, 17, 19, 15, 1, 1]
    res_atoms_n_polarh = [9, 12, 9, 9, 7, 16, 11, 7, 8, 9, 6, 10, 14, 11,
                          6, 16, 9, 12, 9, 12, 5, 9, 16, 9, 14, 8, 8, 11,
                          12, 11, 12, 9, 6, 16, 6, 5, 12, 14, 6, 9, 5, 11,
                          5, 10, 12, 8, 12, 7, 12, 12, 7, 5, 5, 5, 9, 9,
                          10, 16, 12, 7, 8, 11, 10, 7, 16, 11, 12, 11, 6,
                          12, 16, 14, 7, 5, 7, 12, 14, 9, 9, 11, 12, 12, 5,
                          12, 11, 14, 9, 11, 11, 9, 12, 9, 9, 12, 17, 15,
                          8, 8, 10, 13, 10, 12, 5, 10, 12, 8, 12, 7, 8, 5,
                          5, 5, 9, 9, 10, 16, 12, 7, 8, 11, 10, 7, 16, 11,
                          12, 11, 6, 12, 16, 14, 7, 5, 10, 12, 9, 9, 1, 1]
    assert len(protein.atoms) == 1114
    assert len(protein.residues) == 138
    assert_array_equal([len(res.atoms) for res in protein.residues],
                       res_atoms_n)

    protein.addh()
    assert len(protein.atoms) == 2170
    assert len(protein.residues) == 138
    assert_array_equal([len(res.atoms) for res in protein.residues],
                       res_atoms_n_addh)

    protein.removeh()
    protein.addh(only_polar=True)
    assert len(protein.atoms) == 1356
    assert len(protein.residues) == 138
    assert_array_equal([len(res.atoms) for res in protein.residues],
                       res_atoms_n_polarh)

    protein.removeh()
    assert len(protein.atoms) == 1114
    assert len(protein.residues) == 138
    assert_array_equal([len(res.atoms) for res in protein.residues],
                       res_atoms_n)


def test_mol_calccharges():
    mol = oddt.toolkit.readstring('smi', 'c1ccccc1O')
    mol.addh()

    with pytest.raises(ValueError):
        mol.calccharges('mmff94aaaaaa')

    for m in ['gasteiger', 'mmff94']:
        mol.calccharges(m)
        assert (np.array(mol.charges) != 0.).any()

    protein = next(oddt.toolkit.readfile('pdb', xiap_receptor))
    protein.protein = True

    # for that protein mmff94 charges could not be generated
    with pytest.raises(Exception):
        protein.calccharges('mmff94')


def test_toolkit_hoh():
    """HOH residues splitting"""
    pdb_block = """ATOM      1  C1  GLY     1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  C2  GLY     1       0.000   0.000   0.000  1.00  0.00           C
ATOM      3  O1  GLY     1       0.000   0.000   0.000  1.00  0.00           O
ATOM      4  O2  GLY     1       0.000   0.000   0.000  1.00  0.00           O
ATOM      5  N1  GLY     1       0.000   0.000   0.000  1.00  0.00           N
ATOM      6  O3  HOH     2       0.000   0.000   0.000  1.00  0.00           O
ATOM      7  O4  HOH     3       0.000   0.000   0.000  1.00  0.00           O
ATOM      8  O5  HOH     4       0.000   0.000   0.000  1.00  0.00           O
"""
    protein = oddt.toolkit.readstring('pdb', pdb_block)
    protein.protein = True
    assert len(protein.residues) == 4

    protein.addh(only_polar=True)
    assert len(protein.residues) == 4

    protein.addh()
    assert len(protein.residues) == 4


def test_pickle():
    """Pickle molecules"""
    mols = list(oddt.toolkit.readfile('sdf', xiap_actives))
    pickled_mols = list(map(lambda x: loads(dumps(x)), mols))

    assert_array_equal(list(map(lambda x: x.title, mols)),
                       list(map(lambda x: x.title, pickled_mols)))

    assert_array_equal(list(map(lambda x: x.smiles, mols)),
                       list(map(lambda x: x.smiles, pickled_mols)))

    for mol, pickled_mol in zip(mols, pickled_mols):
        assert dict(mol.data) == dict(pickled_mol.data)

    # Test pickling of atom_dicts
    assert_array_equal(list(map(lambda x: x._atom_dict is None, mols)),
                       [True] * len(mols))
    mols_atom_dict = np.hstack(list(map(lambda x: x.atom_dict, mols)))
    assert_array_equal(list(map(lambda x: x._atom_dict is not None, mols)),
                       [True] * len(mols))
    pickled_mols = list(map(lambda x: loads(dumps(x)), mols))
    assert_array_equal(list(map(lambda x: x._atom_dict is not None, pickled_mols)),
                       [True] * len(mols))
    pickled_mols_atom_dict = np.hstack(list(map(lambda x: x._atom_dict, pickled_mols)))
    for name in mols[0].atom_dict.dtype.names:
        if issubclass(np.dtype(mols_atom_dict[name].dtype).type, np.number):
            assert_array_almost_equal(mols_atom_dict[name],
                                      pickled_mols_atom_dict[name])
        else:
            assert_array_equal(mols_atom_dict[name],
                               pickled_mols_atom_dict[name])

    # Lazy Mols
    mols = list(oddt.toolkit.readfile('sdf', xiap_actives, lazy=True))
    pickled_mols = list(map(lambda x: loads(dumps(x)), mols))

    assert_array_equal(list(map(lambda x: x._source is not None, pickled_mols)),
                       [True] * len(mols))

    assert_array_equal(list(map(lambda x: x.title, mols)),
                       list(map(lambda x: x.title, pickled_mols)))

    assert_array_equal(list(map(lambda x: x.smiles, mols)),
                       list(map(lambda x: x.smiles, pickled_mols)))

    for mol, pickled_mol in zip(mols, pickled_mols):
        assert dict(mol.data) == dict(pickled_mol.data)


def test_diverse_conformers():
    # FIXME: make toolkit a module so we can import from it
    diverse_conformers_generator = oddt.toolkit.diverse_conformers_generator

    mol = oddt.toolkit.readstring(
        'smi',
        'CN1CCN(S(=O)(C2=CC=C(OCC)C(C3=NC4=C(N(C)N=C4CCC)C(N3)=O)=C2)=O)CC1'
    )
    mol.make3D()

    if oddt.toolkit.backend == 'ob' and oddt.toolkit.__version__ < '2.4.0':
        with pytest.raises(NotImplementedError):
            diverse_conformers_generator(mol)
        return None  # skip test for older OB

    res = []
    for conf in diverse_conformers_generator(mol, seed=123456):
        res.append(rmsd(mol, conf))

    assert len(res) == 10
    if oddt.toolkit.backend == 'ob':
        assert_array_almost_equal(res, [0., 3.043712, 3.897143, 3.289482,
                                        3.066374, 2.909683, 2.913927,
                                        3.488244, 3.70603, 3.597467])
    # else:
    #     if oddt.toolkit.__version__ > '2016.03.9':
    #         assert_array_almost_equal(res, [1.237538, 2.346984, 0.900624,
    #                                         3.469511, 1.886213, 2.128909,
    #                                         2.852608, 1.312513, 1.291595,
    #                                         1.326843])
    #     else:
    #         assert_array_almost_equal(res, [3.08995, 2.846358, 3.021795,
    #                                         1.720319, 2.741972, 2.965332,
    #                                         2.925344, 2.930157, 2.934049,
    #                                         3.009545])

    # check all implemented methods
    if oddt.toolkit.backend == 'ob':
        methods = ['ga', 'confab']
    else:
        methods = ['dg', 'etkdg', 'kdg', 'etdg']
    for method in methods:
        assert len(diverse_conformers_generator(mol,
                                                seed=123456,
                                                n_conf=5,
                                                method=method)) == 5
        assert len(diverse_conformers_generator(mol,
                                                seed=123456,
                                                n_conf=10,
                                                method=method)) == 10
        assert len(diverse_conformers_generator(mol,
                                                seed=123456,
                                                n_conf=20,
                                                method=method)) == 20


def test_indices():
    """Test 0 and 1 based atom indices"""
    mol = oddt.toolkit.readstring('smi', 'CCc1cc(C)c(C)cc1-c1ccc(-c2cccc(C)c2)cc1')
    atom = mol.atoms[0]

    assert atom.idx0 == 0
    assert atom.idx1 == 1

    # the unmarked index is deprecated in ODDT
    with pytest.warns(DeprecationWarning):
        assert atom.idx == 1


def test_pickle_protein():
    """Pickle proteins"""
    # Proteins
    rec = next(oddt.toolkit.readfile('pdb', xiap_receptor))
    # generate atom_dict
    assert rec.atom_dict is not None

    assert rec._atom_dict is not None
    pickled_rec = loads(dumps(rec))
    assert pickled_rec.protein is False
    assert pickled_rec._atom_dict is not None

    rec.protein = True
    # setting protein property should clean atom_dict cache
    assert rec._atom_dict is None
    # generate atom_dict
    assert rec.atom_dict is not None

    pickled_rec = loads(dumps(rec))
    assert pickled_rec.protein is True
    assert pickled_rec._atom_dict is not None


if oddt.toolkit.backend == 'rdk':
    def test_badmol():
        """Propagate None's for bad molecules"""
        mol = oddt.toolkit.readstring('smi', 'c1cc2')
        assert mol is None


def test_dicts():
    """Test ODDT numpy structures, aka. dicts"""
    mols = list(oddt.toolkit.readfile('sdf', xiap_actives))
    list(map(lambda x: x.addh(only_polar=True), mols))

    skip_cols = ['radius', 'charge', 'id',
                 # following fields need to be standarized
                 'hybridization',
                 ]
    all_cols = [name for name in mols[0].atom_dict.dtype.names
                if name not in ['coords', 'neighbors', 'neighbors_id']]
    common_cols = [name for name in all_cols if name not in skip_cols]

    # Small molecules
    all_dicts = np.hstack([mol.atom_dict for mol in mols])
    all_dicts = all_dicts[all_dicts['atomicnum'] != 1]

    data = pd.DataFrame({name: all_dicts[name] for name in all_cols})
    data['mol_idx'] = [i
                       for i, mol in enumerate(mols)
                       for atom in mol
                       if atom.atomicnum != 1]

    # Save correct results
    # data[common_cols].to_csv(
    #     os.path.join(test_data_dir, 'data/results/xiap/mols_atom_dict.csv'),
    #     index=False)

    corr_data = pd.read_csv(os.path.join(test_data_dir, 'data', 'results',
                                         'xiap', 'mols_atom_dict.csv')
                            ).fillna('')

    for name in common_cols:
        if issubclass(np.dtype(data[name].dtype).type, np.number):
            mask = data[name] - corr_data[name] > 1e-6
            for i in np.argwhere(mask):
                print(i, data[name][i].values, corr_data[name][i].values,
                      mols[data['mol_idx'][int(i)]].write('smi'))
            assert_array_almost_equal(
                data[name],
                corr_data[name],
                err_msg='Mols atom_dict\'s collumn: "%s" is not equal' % name)
        else:
            mask = data[name] != corr_data[name]
            for i in np.argwhere(mask):
                print(i, data[name][i].values, corr_data[name][i].values,
                      mols[data['mol_idx'][int(i)]].write('smi'))
            assert_array_equal(
                data[name],
                corr_data[name],
                err_msg='Mols atom_dict\'s collumn: "%s" is not equal' % name)

    # Protein
    rec = next(oddt.toolkit.readfile('pdb', xiap_receptor))
    rec.protein = True
    rec.addh(only_polar=True)

    skip_cols = ['radius', 'charge', 'resid', 'id',
                 # following fields need to be standarized
                 'hybridization',
                 ]
    common_cols = [name for name in all_cols if name not in skip_cols]

    all_dicts = rec.atom_dict[rec.atom_dict['atomicnum'] != 1]

    data = pd.DataFrame({name: all_dicts[name] for name in all_cols})

    # Save correct results
    # data[common_cols].to_csv(
    #     os.path.join(test_data_dir, 'data/results/xiap/prot_atom_dict.csv'),
    #     index=False)

    corr_data = pd.read_csv(os.path.join(test_data_dir, 'data', 'results',
                                         'xiap', 'prot_atom_dict.csv')
                            ).fillna('')

    for name in common_cols:
        if issubclass(np.dtype(data[name].dtype).type, np.number):
            mask = data[name] - corr_data[name] > 1e-6
            for i in np.argwhere(mask):
                print(i,
                      data['atomtype'][i].values,
                      data['resname'][i].values,
                      data[name][i].values,
                      corr_data[name][i].values)
            assert_array_almost_equal(
                data[name],
                corr_data[name],
                err_msg='Protein atom_dict\'s collumn: "%s" is not equal' % name)
        else:
            mask = data[name] != corr_data[name]
            for i in np.argwhere(mask):
                print(i,
                      data['atomtype'][i].values,
                      data['resname'][i].values,
                      data[name][i].values,
                      corr_data[name][i].values)
            assert_array_equal(
                data[name],
                corr_data[name],
                err_msg='Protein atom_dict\'s collumn: "%s" is not equal' % name)


def test_ss():
    """Secondary structure assignment"""
    # Alpha Helix
    prot_file = os.path.join(test_data_dir, 'data', 'pdb', '1cos_helix.pdb')
    protein = next(oddt.toolkit.readfile('pdb', prot_file))
    protein.protein = True

    # print(protein.res_dict['resname'])
    # print(protein.res_dict['isalpha'])
    # print(protein.res_dict['isbeta'])

    isalpha = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
               18, 19, 20, 21, 22, 23, 24, 25, 26]

    assert len(protein.res_dict) == 29
    assert_array_equal(np.where(protein.res_dict['isalpha'])[0], isalpha)
    assert protein.res_dict['isalpha'].sum() == 27
    assert protein.res_dict['isbeta'].sum() == 0

    # Beta Sheet
    prot_file = os.path.join(test_data_dir, 'data', 'pdb', '1icl_sheet.pdb')
    protein = next(oddt.toolkit.readfile('pdb', prot_file))
    protein.protein = True

    # print(protein.res_dict['resname'])
    # print(protein.res_dict['isalpha'])
    # print(protein.res_dict['isbeta'])
    # print(protein.res_dict['isbeta'])
    # for mask_group in np.split(np.argwhere(protein.res_dict['isbeta']).flatten(),
    #                            np.argwhere(np.diff(np.argwhere(protein.res_dict['isbeta']).flatten()) != 1).flatten() + 1):
    #         print(mask_group + 1, protein.res_dict[mask_group]['resname'])

    isbeta = [2, 3, 4, 5, 10, 11, 12, 13]

    assert len(protein.res_dict) == 29
    assert_array_equal(np.where(protein.res_dict['isbeta'])[0], isbeta)
    assert protein.res_dict['isbeta'].sum() == 8
    assert protein.res_dict['isalpha'].sum() == 0

    # Protein test
    protein = next(oddt.toolkit.readfile('pdb', xiap_receptor))
    protein.protein = True

    # print(protein.res_dict['resname'])
    # print(protein.res_dict['isalpha'])
    # for mask_group in np.split(np.argwhere(protein.res_dict['isalpha']).flatten(),
    #                            np.argwhere(np.diff(np.argwhere(protein.res_dict['isalpha']).flatten()) != 1).flatten() + 1):
    #         print(mask_group + 1, protein.res_dict[mask_group]['resname'])

    # print(protein.res_dict['isbeta'])
    # for mask_group in np.split(np.argwhere(protein.res_dict['isbeta']).flatten(),
    #                            np.argwhere(np.diff(np.argwhere(protein.res_dict['isbeta']).flatten()) != 1).flatten() + 1):
    #         print(mask_group + 1, protein.res_dict[mask_group]['resname'])

    isalpha = [15, 16, 17, 18, 19, 20, 28, 29, 30, 31, 32, 33, 63, 64, 65, 66,
               67, 68, 69, 70, 75, 76, 77, 78, 79, 80, 83, 84, 85, 86, 87, 88,
               89, 90, 91, 121, 122, 123, 124, 125, 126, 127, 128]
    isbeta = [36, 37, 38, 45, 46, 47, 52, 53, 54]

    assert_array_equal(np.where(protein.res_dict['isalpha'])[0], isalpha)
    assert_array_equal(np.where(protein.res_dict['isbeta'])[0], isbeta)
    assert len(protein.res_dict) == 136
    assert protein.res_dict['isalpha'].sum() == 43
    assert protein.res_dict['isbeta'].sum() == 9
    assert (protein.res_dict['isalpha'] &
            protein.res_dict['isbeta']).sum() == 0  # Must be zero!
    assert (~protein.res_dict['isalpha'] &
            ~protein.res_dict['isbeta']).sum() == 84


def test_pdbqt():
    """RDKit PDBQT writer and reader"""
    mol = next(oddt.toolkit.readfile('sdf', xiap_actives))
    mol2 = oddt.toolkit.readstring('pdbqt', mol.write('pdbqt'))
    assert mol.title == mol2.title

    # test loop breaks in DFS algorithm
    mol = oddt.toolkit.readstring('smi', 'CCc1cc(C)c(C)cc1-c1ccc(-c2cccc(C)c2)cc1')
    mol.make3D()

    # roundtrip molecule with template
    mol2 = oddt.toolkit.readstring('pdbqt', mol.write('pdbqt'))
    mol.removeh()

    assert len(mol.atoms) == len(mol2.atoms)

    def nodes_size(block):
        out = OrderedDict()
        current_key = None
        for line in block.split('\n'):
            if line[:4] == 'ROOT' or line[:6] == 'BRANCH':
                current_key = line.strip()
                out[current_key] = 0
            elif line[:4] == 'ATOM':
                out[current_key] += 1
        return list(out.values())

    # check the branch order and size
    if oddt.toolkit.backend == 'ob':
        assert_array_equal(nodes_size(mol.write('pdbqt')),
                           [6, 8, 2, 7])
    else:
        assert_array_equal(nodes_size(mol.write('pdbqt')),
                           [8, 6, 7, 2])
    ligand_file = os.path.join(test_data_dir, 'data', 'dude', 'xiap',
                               'crystal_ligand.sdf')
    mol = next(oddt.toolkit.readfile('sdf', ligand_file))
    assert_array_equal(nodes_size(mol.write('pdbqt')),
                       [8, 3, 6, 6, 1, 6, 3, 2, 2])

    # roundtrip a disconnected fragments
    mol = oddt.toolkit.readstring('smi', 'c1ccccc1.c1ccccc1C')
    if oddt.toolkit.backend == 'ob':
        kwargs = {'opt': {'r': None}}
    else:
        kwargs = {'flexible': False}

    mol2 = oddt.toolkit.readstring('pdbqt', mol.write('pdbqt', **kwargs))
    assert len(mol.atoms) == len(mol2.atoms)

    mol2 = oddt.toolkit.readstring('pdbqt', mol.write('pdbqt'))
    assert len(mol.atoms) == len(mol2.atoms)


def test_residue_info():
    """Residue properties"""
    mol_file = os.path.join(test_data_dir, 'data', 'pdb', '3kwa_5Apocket.pdb')
    mol = next(oddt.toolkit.readfile('pdb', mol_file))
    assert len(mol.residues) == 19

    res = mol.residues[0]
    assert res.idx0 == 0
    assert res.number == 92
    assert res.chain == 'A'
    assert res.name == 'GLN'


def test_canonize_ring_path():
    """Test canonic paths"""
    path0 = list(range(6))
    path = deque(path0)
    path.rotate(3)

    assert canonize_ring_path(path) == path0
    path.reverse()
    assert canonize_ring_path(path) == path0

    with pytest.raises(ValueError):
        canonize_ring_path(tuple(range(6)))
