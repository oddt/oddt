import os

import pytest
from numpy.testing import (assert_almost_equal,
                           assert_array_equal,
                           assert_array_almost_equal)
import numpy as np

import oddt
from oddt.spatial import (angle,
                          dihedral,
                          rmsd,
                          distance,
                          rotate)
from .utils import shuffle_mol


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
    assert d.shape, (n_atoms == n_atoms)
    assert_array_equal(d[np.eye(len(mol1.coords), dtype=bool)], np.zeros(n_atoms))

    d = distance(mol1.coords, mol1.coords.mean(axis=0).reshape(1, 3))
    assert d.shape, (n_atoms == 1)
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


def test_rmsd():
    # pick one molecule from docked poses
    mols = list(oddt.toolkit.readfile('sdf', os.path.join(test_data_dir, 'data/dude/xiap/actives_docked.sdf')))
    mols = list(filter(lambda x: x.title == '312335', mols))

    res = {
        'method=None':
            [4.7536, 2.5015, 2.7942, 1.1282, 0.7444, 1.6257, 4.7625,
             2.7168, 2.5504, 1.9304, 2.6201, 3.1742, 3.2254, 4.7785,
             4.8035, 7.8963, 2.2385, 4.8625, 3.2037],
        'method=hungarian':
            [0.9013, 1.0730, 1.0531, 1.0286, 0.7353, 1.4094, 0.5391,
             1.3297, 1.0881, 1.7796, 2.6064, 3.1577, 3.2135, 0.8126,
             1.2909, 2.5217, 2.0836, 1.8325, 3.1874],
        'method=min_symmetry':
            [0.9013, 1.0732, 1.0797, 1.0492, 0.7444, 1.6257, 0.5391,
             1.5884, 1.0935, 1.9304, 2.6201, 3.1742, 3.2254, 1.1513,
             1.5206, 2.5361, 2.2385, 1.971, 3.2037],
        }

    kwargs_grid = [{'method': None},
                   {'method': 'hungarian'},
                   {'method': 'min_symmetry'}]
    for kwargs in kwargs_grid:
        res_key = '_'.join('%s=%s' % (k, v)
                           for k, v in sorted(kwargs.items()))
        assert_array_almost_equal([rmsd(mols[0], mol, **kwargs)
                                  for mol in mols[1:]],
                                  res[res_key], decimal=4)

    # test shuffled rmsd
    for _ in range(5):
        for kwargs in kwargs_grid:
            # dont use method=None in shuffled tests
            if kwargs['method'] is None:
                continue
            res_key = '_'.join('%s=%s' % (k, v)
                               for k, v in sorted(kwargs.items()))
            assert_array_almost_equal([rmsd(mols[0],
                                            shuffle_mol(mol),
                                            **kwargs)
                                       for mol in mols[1:]],
                                      res[res_key], decimal=4)


def test_rmsd_errors():
    mol = oddt.toolkit.readstring('smi', 'c1ccccc1')
    mol.make3D()
    mol.addh()
    mol2 = next(oddt.toolkit.readfile('sdf', os.path.join(test_data_dir, 'data/dude/xiap/actives_docked.sdf')))

    for method in [None, 'hungarian', 'min_symmetry']:
        with pytest.raises(ValueError, match='Unequal number of atoms'):
            rmsd(mol, mol2, method=method)

        for _ in range(5):
            with pytest.raises(ValueError, match='Unequal number of atoms'):
                rmsd(shuffle_mol(mol), shuffle_mol(mol2), method=method)
