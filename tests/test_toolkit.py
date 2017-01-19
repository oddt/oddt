import os
from tempfile import NamedTemporaryFile

import numpy as np

from nose.tools import assert_in, assert_not_in, assert_equal
from sklearn.utils.testing import (assert_true,
                                   assert_array_equal,
                                   assert_array_almost_equal)

import oddt

test_data_dir = os.path.dirname(os.path.abspath(__file__))


def test_mol():
    """Test common molecule operations"""
    # Hydrogen manipulation
    mol = oddt.toolkit.readstring('smi', 'c1ccccc1')
    assert_equal(len(mol.atoms), 6)
    mol.addh()
    assert_equal(len(mol.atoms), 12)
    mol.removeh()
    assert_equal(len(mol.atoms), 6)


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
