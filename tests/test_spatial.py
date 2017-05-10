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
                          distance,
                          rotate)

test_data_dir = os.path.dirname(os.path.abspath(__file__))

ASPIRIN_SDF = """
     RDKit          3D

 13 13  0  0  0  0  0  0  0  0999 V2000
    3.3558   -0.4356   -1.0951 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0868   -0.6330   -0.3319 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0284   -0.9314    0.8534 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.0157   -0.4307   -1.1906 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.2079   -0.5332   -0.5260 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9020   -1.7350   -0.6775 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.1373   -1.8996   -0.0586 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.6805   -0.8641    0.6975 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.9933    0.3419    0.8273 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7523    0.5244    0.2125 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0600    1.8264    0.3368 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.9397    2.1527   -0.2811 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6931    2.6171    1.2333 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  2  0
  2  4  1  0
  4  5  1  0
  5  6  2  0
  6  7  1  0
  7  8  2  0
  8  9  1  0
  9 10  2  0
 10 11  1  0
 11 12  2  0
 11 13  1  0
 10  5  1  0
M  END

"""


def test_angles():
    """Test spatial computations - angles"""

    # Angles
    assert_array_almost_equal(angle(np.array((1, 0, 0)),
                                    np.array((0, 0, 0)),
                                    np.array((0, 1, 0))), 90)

    assert_array_almost_equal(angle(np.array((1, 0, 0)),
                                    np.array((0, 0, 0)),
                                    np.array((1, 1, 0))), 45)

    # Check benzene ring angle
    mol = oddt.toolkit.readstring('smi', 'c1ccccc1')
    mol.make3D()
    assert_array_almost_equal(angle(mol.coords[0],
                                    mol.coords[1],
                                    mol.coords[2]), 120, decimal=1)


def test_dihedral():
    """Test dihedrals"""
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
    mol = oddt.toolkit.readstring('smi', 'c1ccccc1')
    mol.make3D()
    assert_array_almost_equal(dihedral(mol.coords[0],
                                       mol.coords[1],
                                       mol.coords[2],
                                       mol.coords[3]), 0, decimal=1)


def test_distance():
    mol1 = oddt.toolkit.readstring('sdf', ASPIRIN_SDF)
    d = distance(mol1.coords, mol1.coords)
    n_atoms = len(mol1.coords)
    assert_equal(d.shape, (n_atoms, n_atoms))
    assert_array_equal(d[np.eye(len(mol1.coords), dtype=bool)], np.zeros(n_atoms))

    d = distance(mol1.coords, mol1.coords.mean(axis=0).reshape(1, 3))
    assert_equal(d.shape, (n_atoms, 1))
    ref_dist = [[3.556736951371501], [2.2058040428631056], [2.3896002745745415],
                [1.6231668718498249], [0.7772981740050453], [2.0694947503940004],
                [2.8600587871157184], [2.9014207091233857], [2.1850791695403564],
                [0.9413368403116871], [1.8581710293650173], [2.365629642108773],
                [2.975007440512798]]
    assert_array_almost_equal(d, ref_dist)


def test_spatial():
    """Test spatial misc computations"""
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
    # Minimized by symetry must close to zero
    assert_almost_equal(rmsd(mol, mol2, method='min_symmetry'), 0, decimal=0)

    # pick one molecule from docked poses
    mols = list(oddt.toolkit.readfile('sdf', os.path.join(test_data_dir, 'data/dude/xiap/actives_docked.sdf')))
    mols = list(filter(lambda x: x.title == '312335', mols))

    assert_array_almost_equal([rmsd(mols[0], mol) for mol in mols[1:]],
                              [4.753552, 2.501487, 2.7941732, 1.1281863, 0.74440968,
                               1.6256877, 4.762476, 2.7167852, 2.5504358, 1.9303833,
                               2.6200771, 3.1741529, 3.225431, 4.7784939, 4.8035369,
                               7.8962774, 2.2385094, 4.8625236, 3.2036853])

    assert_array_almost_equal([rmsd(mols[0], mol, method='hungarian') for mol in mols[1:]],
                              [0.90126, 1.073049, 1.053131, 1.028578, 0.735297, 1.409403,
                               0.539091, 1.329666, 1.088053, 1.779618, 2.606429, 3.157684,
                               3.213502, 0.812635, 1.290902, 2.521703, 2.083612, 1.832457,
                               3.187363])
