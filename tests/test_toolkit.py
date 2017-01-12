import os
from tempfile import NamedTemporaryFile

import numpy as np

from nose.tools import assert_in, assert_not_in, assert_equal
from sklearn.utils.testing import (assert_true,
                                   assert_array_equal,
                                   assert_array_almost_equal)

import oddt

test_data_dir = os.path.dirname(os.path.abspath(__file__))


def test_ss():
    """Secondary structure assignment"""
    # Alpha Helix
    protein = next(oddt.toolkit.readfile('pdb', os.path.join(test_data_dir, 'data/pdb/1cos_helix.pdb')))
    protein.protein = True

    protein._dicts()
    print(protein.res_dict['resname'])
    print(protein.res_dict['isalpha'])
    print(protein.res_dict['isbeta'])

    assert_equal(len(protein.res_dict), 29)
    assert_equal(protein.res_dict['isalpha'].sum(), 27)
    assert_equal((~protein.res_dict['isalpha']).sum(), 2)
    assert_equal(protein.res_dict['isbeta'].sum(), 0)

    # Beta Sheet
    protein = next(oddt.toolkit.readfile('pdb', os.path.join(test_data_dir, 'data/pdb/1icl_sheet.pdb')))
    protein.protein = True

    print(protein.res_dict['resname'])
    print(protein.res_dict['isalpha'])
    print(protein.res_dict['isbeta'])

    assert_equal(len(protein.res_dict), 29)
    assert_equal(protein.res_dict['isbeta'].sum(), 10)
    assert_equal((~protein.res_dict['isbeta']).sum(), 19)
    assert_equal(protein.res_dict['isalpha'].sum(), 0)

    # Protein test
    protein = next(oddt.toolkit.readfile('pdb', os.path.join(test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb')))
    protein.protein = True

    print(protein.res_dict['resname'])
    print(protein.res_dict['isalpha'])
    print(protein.res_dict['isbeta'])
    print(protein.res_dict)

    assert_equal(len(protein.res_dict), 136)
    assert_equal(protein.res_dict['isalpha'].sum(), 52)
    assert_equal((~protein.res_dict['isalpha']).sum(), 84)
    assert_equal(protein.res_dict['isbeta'].sum(), 16)
    assert_equal((~protein.res_dict['isbeta']).sum(), 120)
    assert_equal((protein.res_dict['isalpha'] & protein.res_dict['isbeta']).sum(), 0)  # Must be zero!
    assert_equal((~protein.res_dict['isalpha'] & ~protein.res_dict['isbeta']).sum(), 68)
