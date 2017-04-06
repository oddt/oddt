import os

from nose.tools import assert_equal
from sklearn.utils.testing import assert_array_equal, assert_almost_equal

import oddt
from oddt.property import xlogp2_atom_contrib

test_data_dir = os.path.dirname(os.path.abspath(__file__))


def test_xlogp2():
    """Test XlogP results against original implementation"""
    mol = oddt.toolkit.readstring('smi', 'Oc1cc(Cl)ccc1Oc1ccc(Cl)cc1Cl')
    correct_xlogp = [0.082, -0.151, 0.337, 0.296, 0.663, 0.337, 0.337, -0.151,
                     0.435, -0.151, 0.337, 0.337, 0.296, 0.663, 0.337, 0.296,
                     0.663]

    predicted_xlogp = xlogp2_atom_contrib(mol, corrections=False)[mol.atom_dict['atomicnum'] != 1]
    assert_array_equal(correct_xlogp, predicted_xlogp)

    mol.removeh()
    predicted_xlogp = xlogp2_atom_contrib(mol, corrections=False)[mol.atom_dict['atomicnum'] != 1]
    assert_array_equal(correct_xlogp, predicted_xlogp)

    mol.addh()
    predicted_xlogp = xlogp2_atom_contrib(mol, corrections=False)[mol.atom_dict['atomicnum'] != 1]
    assert_array_equal(correct_xlogp, predicted_xlogp)

    mol.removeh()
    predicted_xlogp = xlogp2_atom_contrib(mol, corrections=False)[mol.atom_dict['atomicnum'] != 1]
    assert_array_equal(correct_xlogp, predicted_xlogp)

    mol = oddt.toolkit.readstring('smi', 'NC(N)c1ccc(C[C@@H](NC(=O)CNS(=O)(=O)c2ccc3ccccc3c2)C(=O)N2CCCCC2)cc1')
    correct_xlogp = [-0.534, -0.305, -0.534, 0.296, 0.337, 0.337, 0.296, -0.008,
                     -0.305, -0.096, -0.03, -0.399, -0.303, -0.112, -0.168,
                     -0.399, -0.399, 0.296, 0.337, 0.337, 0.296, 0.337, 0.337,
                     0.337, 0.337, 0.296, 0.337, -0.03, -0.399, 0.078, -0.137,
                     0.358, 0.358, 0.358, -0.137, 0.337, 0.337]

    predicted_xlogp = xlogp2_atom_contrib(mol, corrections=False)[mol.atom_dict['atomicnum'] != 1]
    assert_array_equal(correct_xlogp, predicted_xlogp)

    mol.removeh()
    predicted_xlogp = xlogp2_atom_contrib(mol, corrections=False)[mol.atom_dict['atomicnum'] != 1]
    assert_array_equal(correct_xlogp, predicted_xlogp)

    mol.addh()
    predicted_xlogp = xlogp2_atom_contrib(mol, corrections=False)[mol.atom_dict['atomicnum'] != 1]
    assert_array_equal(correct_xlogp, predicted_xlogp)

    mol.removeh()
    predicted_xlogp = xlogp2_atom_contrib(mol, corrections=False)[mol.atom_dict['atomicnum'] != 1]
    assert_array_equal(correct_xlogp, predicted_xlogp)

    predicted_xlogp = []
    for mol in oddt.toolkit.readfile('smi', os.path.join(test_data_dir, 'data/dude/xiap/actives_rdkit.smi')):
        mol.addh()
        predicted_xlogp.append(xlogp2_atom_contrib(mol, corrections=False).sum())

    correct_xlogp = [3.974, 3.974, 3.974, 3.974, 0.659, 0.659, 3.396, 0.891,
                     0.891, 2.086, 2.086, 2.516, 2.516, 0.621, 1.621, 1.319,
                     -0.037, 0.432, 2.210, 2.989, 1.598, 1.598, 1.598, 1.598,
                     4.498, 4.498, 1.598, 1.598, 1.598, 1.598, 1.996, 3.041,
                     3.041, -0.002, -0.002, 2.399, 2.815, 2.815, 0.585, 2.397,
                     2.846, 4.096, 1.017, 3.447, 2.061, 2.874, 2.330, 2.354,
                     3.051, 1.144, 0.941, 2.720, 0.633, 2.346, 2.700, 2.185,
                     3.580, 3.664, 2.039, 3.232, 4.367, 3.622, 2.248, 3.965,
                     1.299, 2.589, 4.313, 4.313, 4.313, 4.313, 2.754, 0.408,
                     -0.399, 0.583, 0.583, 0.362, 0.362, 0.362, 0.362, 1.476,
                     1.276, 2.779, 3.264, 0.585, 2.568, 2.568, 2.562, 2.276,
                     2.693, 3.533, 3.253, 1.956, 4.072, 4.311, 2.429, 2.785,
                     2.516, 3.169, 1.335, 0.280, -0.152, 1.972, 3.997, 3.997,
                     2.787, 3.143, 1.494, 1.118, 4.016, 3.770, 4.081, 1.017,
                     4.022, 4.430, 2.668, 2.668, 0.646, 2.874, 3.264, 1.709,
                     2.362, 2.907, 2.907, 2.907, 2.907, 3.169, 3.855, 4.068,
                     2.354, ]
    assert_almost_equal(correct_xlogp, predicted_xlogp, decimal=3)
