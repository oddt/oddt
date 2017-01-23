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
