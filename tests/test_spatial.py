import os
from tempfile import NamedTemporaryFile

from nose.tools import assert_in, assert_not_in, assert_equal
from sklearn.utils.testing import (assert_true,
                                   assert_almost_equal,
                                   assert_array_equal,
                                   assert_array_almost_equal)
import numpy as np

import oddt
from oddt.spatial import (angle,
                          dihedral,
                          rmsd,
                          rotate)

test_data_dir = os.path.dirname(os.path.abspath(__file__))


def test_spatial():
    """Test spatial computations"""

    # Angles
    assert_array_almost_equal(angle(np.array((1, 0, 0)),
                                    np.array((0, 0, 0)),
                                    np.array((0, 1, 0))), 90)

    assert_array_almost_equal(angle(np.array((1, 0, 0)),
                                    np.array((0, 0, 0)),
                                    np.array((1, 1, 0))), 45)

    mol = oddt.toolkit.readstring('smi', 'c1ccccc1')
    mol.make3D()

    # Check benzene ring angle
    assert_array_almost_equal(angle(mol.coords[0],
                                    mol.coords[1],
                                    mol.coords[2]), 120, decimal=1)

    # Dihedrals
    assert_array_almost_equal(dihedral(np.array((1, 0, 0)),
                                       np.array((0, 0, 0)),
                                       np.array((0, 1, 0)),
                                       np.array((1, 1, 0))), 0)

    assert_array_almost_equal(dihedral(np.array((1, 0, 0)),
                                       np.array((0, 0, 0)),
                                       np.array((0, 1, 0)),
                                       np.array((1, 1, 1))), -45)

    # Check benzene ring dihedral
    assert_array_almost_equal(dihedral(mol.coords[0],
                                       mol.coords[1],
                                       mol.coords[2],
                                       mol.coords[3]), 0, decimal=1)

    mol = oddt.toolkit.readstring('smi', 'c1ccccc1')
    mol.make3D()
    mol2 = mol.clone

    # Test rotation
    assert_almost_equal(mol2.coords, rotate(mol2.coords, np.pi, np.pi, np.pi))

    # Rotate perpendicular to ring
    mol2.coords = rotate(mol2.coords, 0, 0, np.pi)

    # RMSD
    assert_almost_equal(rmsd(mol, mol2, method=None), 2.77, decimal=1)
    # Hungarian must be close to zero (RDKit is 0.3)
    assert_almost_equal(rmsd(mol, mol2, method='hungarian'), 0, decimal=0)

    # pick one molecule from docked poses
    mols = list(oddt.toolkit.readfile('sdf', os.path.join(test_data_dir, 'data/dude/xiap/actives_docked.sdf')))
    mols = list(filter(lambda x: x.title == '312335', mols))

    assert_array_almost_equal([rmsd(mols[0], mol) for mol in mols[1:]],
                              [4.753552, 2.501487, 2.7941732, 1.1281863, 0.74440968,
                               1.6256877, 4.762476, 2.7167852, 2.5504358, 1.9303833,
                               2.6200771, 3.1741529, 3.225431, 4.7784939, 4.8035369,
                               7.8962774, 2.2385094, 4.8625236, 3.2036853])

    assert_array_almost_equal([rmsd(mols[0], mol, method='hungarian') for mol in mols[1:]],
                              [2.5984519, 1.7295024, 1.1268076, 1.0285776, 0.73529714,
                               1.4094033, 2.5195069, 1.7449125, 1.5116163, 1.7796179,
                               2.6064286, 3.1576841, 3.2135022, 3.1675091, 2.7001681,
                               5.1263351, 2.0836117, 3.542397, 3.1873631])
