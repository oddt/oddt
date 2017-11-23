import os
from tempfile import NamedTemporaryFile
from collections import OrderedDict

from six.moves.cPickle import loads, dumps
import numpy as np
import pandas as pd

from nose.tools import nottest, assert_in, assert_not_in, assert_equal
from sklearn.utils.testing import (assert_true,
                                   assert_false,
                                   assert_dict_equal,
                                   assert_array_equal,
                                   assert_array_almost_equal)

import oddt

test_data_dir = os.path.dirname(os.path.abspath(__file__))


def test_mol():
    """Test common molecule operations"""
    # Hydrogen manipulation in small molecules
    mol = oddt.toolkit.readstring('smi', 'c1ccccc1O')
    assert_equal(len(mol.atoms), 7)
    mol.addh()
    assert_equal(len(mol.atoms), 13)
    mol.removeh()
    mol.addh(only_polar=True)
    assert_equal(len(mol.atoms), 8)
    mol.removeh()
    assert_equal(len(mol.atoms), 7)

    # Hydrogen manipulation in proteins
    protein = next(oddt.toolkit.readfile('pdb', os.path.join(test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb')))
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
    assert_equal(len(protein.atoms), 1114)
    assert_equal(len(protein.residues), 138)
    assert_array_equal([len(res.atoms) for res in protein.residues], res_atoms_n)

    protein.addh()
    assert_equal(len(protein.atoms), 2170)
    assert_equal(len(protein.residues), 138)
    assert_array_equal([len(res.atoms) for res in protein.residues], res_atoms_n_addh)

    protein.removeh()
    protein.addh(only_polar=True)
    assert_equal(len(protein.atoms), 1356)
    assert_equal(len(protein.residues), 138)
    assert_array_equal([len(res.atoms) for res in protein.residues], res_atoms_n_polarh)

    protein.removeh()
    assert_equal(len(protein.atoms), 1114)
    assert_equal(len(protein.residues), 138)
    assert_array_equal([len(res.atoms) for res in protein.residues], res_atoms_n)


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
    assert_equal(len(protein.residues), 4)

    protein.addh(only_polar=True)
    assert_equal(len(protein.residues), 4)

    protein.addh()
    assert_equal(len(protein.residues), 4)


def test_pickle():
    """Pickle molecules"""
    mols = list(oddt.toolkit.readfile('sdf',
                                      os.path.join(test_data_dir, 'data/dude/xiap/actives_docked.sdf')))
    pickled_mols = list(map(lambda x: loads(dumps(x)), mols))

    assert_array_equal(list(map(lambda x: x.title, mols)),
                       list(map(lambda x: x.title, pickled_mols)))

    assert_array_equal(list(map(lambda x: x.smiles, mols)),
                       list(map(lambda x: x.smiles, pickled_mols)))

    for mol, pickled_mol in zip(mols, pickled_mols):
        assert_dict_equal(dict(mol.data),
                          dict(pickled_mol.data))

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
            assert_array_almost_equal(mols_atom_dict[name], pickled_mols_atom_dict[name])
        else:
            assert_array_equal(mols_atom_dict[name], pickled_mols_atom_dict[name])

    # Lazy Mols
    mols = list(oddt.toolkit.readfile('sdf',
                                      os.path.join(test_data_dir, 'data/dude/xiap/actives_docked.sdf'),
                                      lazy=True))
    pickled_mols = list(map(lambda x: loads(dumps(x)), mols))

    assert_array_equal(list(map(lambda x: x._source is not None, pickled_mols)),
                       [True] * len(mols))

    assert_array_equal(list(map(lambda x: x.title, mols)),
                       list(map(lambda x: x.title, pickled_mols)))

    assert_array_equal(list(map(lambda x: x.smiles, mols)),
                       list(map(lambda x: x.smiles, pickled_mols)))

    for mol, pickled_mol in zip(mols, pickled_mols):
        assert_dict_equal(dict(mol.data),
                          dict(pickled_mol.data))


def test_pickle_protein():
    """Pickle proteins"""
    # Proteins
    rec = next(oddt.toolkit.readfile('pdb', os.path.join(test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb')))
    # generate atom_dict
    assert_false(rec.atom_dict is None)

    assert_false(rec._atom_dict is None)
    pickled_rec = loads(dumps(rec))
    assert_false(pickled_rec.protein)
    assert_false(pickled_rec._atom_dict is None)

    rec.protein = True
    # setting protein property should clean atom_dict cache
    assert_true(rec._atom_dict is None)
    # generate atom_dict
    assert_false(rec.atom_dict is None)

    pickled_rec = loads(dumps(rec))
    assert_true(pickled_rec.protein)
    assert_false(pickled_rec._atom_dict is None)


if oddt.toolkit.backend == 'rdk':
    def test_badmol():
        """Propagate None's for bad molecules"""
        mol = oddt.toolkit.readstring('smi', 'c1cc2')
        assert_equal(mol, None)


def test_dicts():
    """Test ODDT numpy structures, aka. dicts"""
    mols = list(oddt.toolkit.readfile('sdf', os.path.join(test_data_dir, 'data/dude/xiap/actives_docked.sdf')))
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
    # data.to_csv(os.path.join(test_data_dir, 'data/results/xiap/mols_atom_dict.csv'),
    #             index=False)

    corr_data = pd.read_csv(os.path.join(test_data_dir, 'data/results/xiap/mols_atom_dict.csv')).fillna('')

    for name in common_cols:
        if issubclass(np.dtype(data[name].dtype).type, np.number):
            mask = data[name] - corr_data[name] > 1e-6
            for i in np.argwhere(mask):
                print(i, data[name][i].values, corr_data[name][i].values,
                      mols[data['mol_idx'][int(i)]].write('smi'))
            assert_array_almost_equal(data[name],
                                      corr_data[name],
                                      err_msg='Mols atom_dict\'s collumn: "%s" is not equal' % name)
        else:
            mask = data[name] != corr_data[name]
            for i in np.argwhere(mask):
                print(i, data[name][i].values, corr_data[name][i].values,
                      mols[data['mol_idx'][int(i)]].write('smi'))
            assert_array_equal(data[name],
                               corr_data[name],
                               err_msg='Mols atom_dict\'s collumn: "%s" is not equal' % name)

    # Protein
    rec = next(oddt.toolkit.readfile('pdb', os.path.join(test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb')))
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
    # data.to_csv(os.path.join(test_data_dir, 'data/results/xiap/prot_atom_dict.csv'),
    #             index=False)

    corr_data = pd.read_csv(os.path.join(test_data_dir, 'data/results/xiap/prot_atom_dict.csv')).fillna('')

    for name in common_cols:
        if issubclass(np.dtype(data[name].dtype).type, np.number):
            mask = data[name] - corr_data[name] > 1e-6
            for i in np.argwhere(mask):
                print(i,
                      data['atomtype'][i].values,
                      data['resname'][i].values,
                      data[name][i].values,
                      corr_data[name][i].values)
            assert_array_almost_equal(data[name],
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
            assert_array_equal(data[name],
                               corr_data[name],
                               err_msg='Protein atom_dict\'s collumn: "%s" is not equal' % name)


def test_ss():
    """Secondary structure assignment"""
    # Alpha Helix
    protein = next(oddt.toolkit.readfile('pdb', os.path.join(test_data_dir, 'data/pdb/1cos_helix.pdb')))
    protein.protein = True

    # print(protein.res_dict['resname'])
    # print(protein.res_dict['isalpha'])
    # print(protein.res_dict['isbeta'])

    isalpha = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
               18, 19, 20, 21, 22, 23, 24, 25, 26]

    assert_equal(len(protein.res_dict), 29)
    assert_array_equal(np.where(protein.res_dict['isalpha'])[0], isalpha)
    assert_equal(protein.res_dict['isalpha'].sum(), 27)
    assert_equal(protein.res_dict['isbeta'].sum(), 0)

    # Beta Sheet
    protein = next(oddt.toolkit.readfile('pdb', os.path.join(test_data_dir, 'data/pdb/1icl_sheet.pdb')))
    protein.protein = True

    # print(protein.res_dict['resname'])
    # print(protein.res_dict['isalpha'])
    # print(protein.res_dict['isbeta'])
    # print(protein.res_dict['isbeta'])
    # for mask_group in np.split(np.argwhere(protein.res_dict['isbeta']).flatten(),
    #                            np.argwhere(np.diff(np.argwhere(protein.res_dict['isbeta']).flatten()) != 1).flatten() + 1):
    #         print(mask_group + 1, protein.res_dict[mask_group]['resname'])

    isbeta = [2, 3, 4, 5, 10, 11, 12, 13]

    assert_equal(len(protein.res_dict), 29)
    assert_array_equal(np.where(protein.res_dict['isbeta'])[0], isbeta)
    assert_equal(protein.res_dict['isbeta'].sum(), 8)
    assert_equal(protein.res_dict['isalpha'].sum(), 0)

    # Protein test
    protein = next(oddt.toolkit.readfile('pdb', os.path.join(test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb')))
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
    assert_equal(len(protein.res_dict), 136)
    assert_equal(protein.res_dict['isalpha'].sum(), 43)
    assert_equal(protein.res_dict['isbeta'].sum(), 9)
    assert_equal((protein.res_dict['isalpha'] & protein.res_dict['isbeta']).sum(), 0)  # Must be zero!
    assert_equal((~protein.res_dict['isalpha'] & ~protein.res_dict['isbeta']).sum(), 84)


def test_rdkit_pdbqt():
    """RDKit PDBQT writer and reader"""
    # test loop breaks in DFS algorithm
    mol = oddt.toolkit.readstring('smi', 'CCc1cc(C)c(C)cc1-c1ccc(-c2cccc(C)c2)cc1')
    mol.make3D()

    # roundtrip molecule with template
    mol2 = oddt.toolkit.readstring('pdbqt', mol.write('pdbqt'))
    mol.removeh()

    assert_equal(len(mol.atoms), len(mol2.atoms))

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
    mol = next(oddt.toolkit.readfile('sdf', os.path.join(test_data_dir, 'data/dude/xiap/crystal_ligand.sdf')))
    assert_array_equal(nodes_size(mol.write('pdbqt')),
                       [8, 3, 6, 6, 1, 6, 3, 2, 2])
