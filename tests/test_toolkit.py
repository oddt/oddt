import os
from tempfile import NamedTemporaryFile

from six.moves.cPickle import loads, dumps
import numpy as np
import pandas as pd

from nose.tools import nottest, assert_in, assert_not_in, assert_equal
from sklearn.utils.testing import (assert_true,
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


def test_pickle():
    """Pickle molecules"""
    mols = list(oddt.toolkit.readfile('sdf',
                                      os.path.join(test_data_dir, 'data/dude/xiap/actives_docked.sdf')))
    pickled_mols = list(map(lambda x: loads(dumps(x)), mols))

    assert_array_equal(list(map(lambda x: x.smiles, mols)),
                       list(map(lambda x: x.smiles, pickled_mols)))

    assert_array_equal(list(map(lambda x: dict(x.data), mols)),
                       list(map(lambda x: dict(x.data), pickled_mols)))

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

    assert_array_equal(list(map(lambda x: x.smiles, mols)),
                       list(map(lambda x: x.smiles, pickled_mols)))

    assert_array_equal(list(map(lambda x: dict(x.data), mols)),
                       list(map(lambda x: dict(x.data), pickled_mols)))


if oddt.toolkit.backend == 'rdk':
    def test_badmol():
        """Propagate None's for bad molecules"""
        mol = oddt.toolkit.readstring('smi', 'c1cc2')
        assert_equal(mol, None)


def test_dicts():
    """Test ODDT numpy structures, aka. dicts"""
    mols = list(oddt.toolkit.readfile('sdf', os.path.join(test_data_dir, 'data/dude/xiap/actives_docked.sdf')))
    list(map(lambda x: x.addh(only_polar=True), mols))

    skip_cols = ['radius', 'charge',
                 # following fields need to be standarized
                 'hybridization',
                 ]
    all_cols = [name for name in mols[0].atom_dict.dtype.names
                if name not in ['coords', 'neighbors']]
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

    skip_cols = ['radius', 'charge', 'resid',
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

    assert_equal(len(protein.res_dict), 29)
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

    assert_equal(len(protein.res_dict), 29)
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

    assert_equal(len(protein.res_dict), 136)
    assert_equal(protein.res_dict['isalpha'].sum(), 43)
    assert_equal(protein.res_dict['isbeta'].sum(), 9)
    assert_equal((protein.res_dict['isalpha'] & protein.res_dict['isbeta']).sum(), 0)  # Must be zero!
    assert_equal((~protein.res_dict['isalpha'] & ~protein.res_dict['isbeta']).sum(), 84)
