import os
from types import SimpleNamespace

import pytest
from numpy.testing import assert_almost_equal, assert_array_equal, assert_array_almost_equal
import numpy as np
from scipy.spatial.transform import Rotation

import oddt
from oddt.spatial import angle, angle_2v, dihedral, rmsd, distance, distance_complex, rotate
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


def _reference_angle_2v(first, second):
    dot = (first * second).sum(axis=-1)
    norm = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    return np.degrees(np.arccos(np.clip(dot / norm, -1, 1)))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "first_shape,second_shape",
    [
        ((3,), (3,)),
        ((12, 3), (12, 3)),
        ((12, 1, 3), (12, 6, 3)),
        ((1, 6, 3), (12, 6, 3)),
        ((0, 1, 3), (0, 6, 3)),
        ((6,), (12, 1, 6)),
    ],
)
def test_angle_2v_broadcasting(dtype, first_shape, second_shape):
    rng = np.random.default_rng(42)
    first = rng.normal(size=first_shape).astype(dtype)
    second = rng.normal(size=second_shape).astype(dtype)
    expected = _reference_angle_2v(first, second)
    actual = angle_2v(first, second)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_angle_2v_noncontiguous(dtype):
    rng = np.random.default_rng(42)
    vectors = np.zeros(12, dtype=[("vectors", dtype, (6, 3)), ("padding", np.uint8, 11)])
    vectors["vectors"] = rng.normal(size=(12, 6, 3))
    first = vectors["vectors"][:, :1, :]
    second = vectors["vectors"][:, ::-1, :]
    assert not first.flags.c_contiguous
    assert not second.flags.c_contiguous
    np.testing.assert_allclose(angle_2v(first, second), _reference_angle_2v(first, second), rtol=2e-6, atol=2e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_angle_2v_degenerate_vectors(dtype):
    first = np.array([1, 0, 0], dtype=dtype)
    second = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 0], [np.nan, 0, 0]], dtype=dtype)
    with np.errstate(invalid="ignore", divide="ignore"):
        actual = angle_2v(first, second)
    np.testing.assert_allclose(actual, [0, 180, 90, np.nan, np.nan], equal_nan=True)


@pytest.mark.parametrize(
    "first_dtype,second_dtype",
    [(np.float16, np.float16), (np.float16, np.float32), (np.float32, np.float16)],
)
def test_angle_2v_float16_inputs(first_dtype, second_dtype):
    rng = np.random.default_rng(42)
    first = rng.normal(size=(256, 1, 3)).astype(first_dtype)
    second = rng.normal(size=(256, 6, 3)).astype(second_dtype)
    expected = _reference_angle_2v(first, second)
    actual = angle_2v(first, second)
    assert actual.dtype == expected.dtype
    assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "first_dtype,second_dtype",
    [(np.int8, np.int8), (np.uint8, np.uint8), (np.int64, np.int64), (bool, bool), (np.int8, np.float32)],
)
def test_angle_2v_nonfloating_inputs(first_dtype, second_dtype):
    first = np.array([[120, 120, 120], [1, 0, 0]], dtype=first_dtype)
    second = np.array([[120, 120, 120], [0, 1, 0]], dtype=second_dtype)
    expected = _reference_angle_2v(first, second)
    actual = angle_2v(first, second)
    assert actual.dtype == expected.dtype
    assert_array_equal(actual, expected)


def test_angles():
    """Test spatial computations - angles"""

    # Angles
    assert_array_almost_equal(angle(np.array((1, 0, 0)), np.array((0, 0, 0)), np.array((0, 1, 0))), 90)

    assert_array_almost_equal(angle(np.array((1, 0, 0)), np.array((0, 0, 0)), np.array((1, 1, 0))), 45)

    # Check benzene ring angle
    mol = oddt.toolkit.readstring("smi", "c1ccccc1")
    mol.make3D()
    assert_array_almost_equal(angle(mol.coords[0], mol.coords[1], mol.coords[2]), 120, decimal=1)


def test_dihedral():
    """Test dihedrals"""
    # Dihedrals
    assert_array_almost_equal(
        dihedral(np.array((1, 0, 0)), np.array((0, 0, 0)), np.array((0, 1, 0)), np.array((1, 1, 0))), 0
    )

    assert_array_almost_equal(
        dihedral(np.array((1, 0, 0)), np.array((0, 0, 0)), np.array((0, 1, 0)), np.array((1, 1, 1))), -45
    )

    # Check benzene ring dihedral
    mol = oddt.toolkit.readstring("smi", "c1ccccc1")
    mol.make3D()
    assert abs(dihedral(*mol.coords[:4])) < 2.0


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int32])
@pytest.mark.parametrize("count", [1, 12])
def test_dihedral_einsum(dtype, count):
    rng = np.random.default_rng(42)
    points = rng.uniform(-10, 10, (4, count, 3)).astype(dtype)[:, ::-1, :]
    first = (points[0] - points[1]) / np.linalg.norm(points[0] - points[1])
    second = (points[1] - points[2]) / np.linalg.norm(points[1] - points[2])
    third = (points[2] - points[3]) / np.linalg.norm(points[2] - points[3])
    normal_first = np.cross(first, second)
    normal_second = np.cross(second, third)
    expected = _reference_angle_2v(normal_first, normal_second)
    signed = ((normal_first / np.linalg.norm(normal_first)) * third).sum(axis=-1) > 0
    expected[signed] = -expected[signed]
    actual = dihedral(*points)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-5)


def test_distance():
    mol1 = oddt.toolkit.readstring("sdf", ASPIRIN_SDF)
    d = distance(mol1.coords, mol1.coords)
    n_atoms = len(mol1.coords)
    assert d.shape, n_atoms == n_atoms
    assert_array_equal(d[np.eye(len(mol1.coords), dtype=bool)], np.zeros(n_atoms))

    d = distance(mol1.coords, mol1.coords.mean(axis=0).reshape(1, 3))
    assert d.shape, n_atoms == 1
    ref_dist = [
        [3.556736951371501],
        [2.2058040428631056],
        [2.3896002745745415],
        [1.6231668718498249],
        [0.7772981740050453],
        [2.0694947503940004],
        [2.8600587871157184],
        [2.9014207091233857],
        [2.1850791695403564],
        [0.9413368403116871],
        [1.8581710293650173],
        [2.365629642108773],
        [2.975007440512798],
    ]
    assert_array_almost_equal(d, ref_dist)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int8, np.complex64])
@pytest.mark.parametrize(
    "first_shape,second_shape",
    [((3,), (3,)), ((2, 12, 3), (2, 1, 6, 3)), ((0, 3), (6, 3))],
)
def test_distance_complex_einsum(dtype, first_shape, second_shape):
    rng = np.random.default_rng(42)
    first = rng.uniform(-10, 10, first_shape[:-1] + (6,)).astype(dtype)[..., ::2]
    second = rng.uniform(-10, 10, second_shape[:-1] + (6,)).astype(dtype)[..., ::2]
    if np.dtype(dtype).kind == "c":
        first = first * (1 + 2j)
        second = second * (1 - 3j)
    expected = np.linalg.norm(first[..., np.newaxis, :] - second, axis=-1)
    actual = distance_complex(first, second)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-5)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int64])
@pytest.mark.parametrize("count", [1, 12])
def test_rotate_einsum(dtype, count):
    rng = np.random.default_rng(42)
    coords = rng.uniform(-10, 10, (count, 6)).astype(dtype)[:, ::2]
    original_coords = coords.copy()
    angles = (0.37, -0.82, 1.13)
    centroid = coords.mean(axis=0)
    matrix = Rotation.from_euler("xyz", angles).as_matrix()
    expected = ((coords - centroid)[:, np.newaxis, :] * matrix).sum(axis=-1) + centroid
    actual = rotate(coords, *angles)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert_array_equal(coords, original_coords)


def test_spatial():
    """Test spatial misc computations"""
    mol = oddt.toolkit.readstring("smi", "c1ccccc1")
    mol.make3D()
    mol2 = mol.clone
    # Test rotation
    assert_almost_equal(mol2.coords, rotate(mol2.coords, np.pi, np.pi, np.pi))

    # Rotate perpendicular to ring
    mol2.coords = rotate(mol2.coords, 0, 0, np.pi)

    # RMSD
    assert_almost_equal(rmsd(mol, mol2, method=None), 2.77, decimal=1)
    # Hungarian must be close to zero (RDKit is 0.3)
    assert_almost_equal(rmsd(mol, mol2, method="hungarian"), 0, decimal=0)
    # Minimized by symetry must close to zero
    assert_almost_equal(rmsd(mol, mol2, method="min_symmetry"), 0, decimal=0)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int8])
@pytest.mark.parametrize("normalize", [False, True])
def test_rmsd_einsum(dtype, normalize):
    rng = np.random.default_rng(42)
    first = rng.uniform(-10, 10, (12, 6)).astype(dtype)[:, ::2]
    second = rng.uniform(-10, 10, (12, 6)).astype(dtype)[:, ::2]
    reference = SimpleNamespace(coords=first)
    molecule = SimpleNamespace(coords=second, num_rotors=9)
    expected = np.sqrt(((second - first) ** 2).sum(axis=-1).mean())
    if normalize:
        expected /= np.sqrt(molecule.num_rotors)
    actual = rmsd(reference, molecule, ignore_h=False, normalize=normalize)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-5)


def test_rmsd():
    # pick one molecule from docked poses
    mols = list(oddt.toolkit.readfile("sdf", os.path.join(test_data_dir, "data/dude/xiap/actives_docked.sdf")))
    mols = list(filter(lambda x: x.title == "312335", mols))

    res = {
        "method=None": [
            4.7536,
            2.5015,
            2.7942,
            1.1282,
            0.7444,
            1.6257,
            4.7625,
            2.7168,
            2.5504,
            1.9304,
            2.6201,
            3.1742,
            3.2254,
            4.7785,
            4.8035,
            7.8963,
            2.2385,
            4.8625,
            3.2037,
        ],
        "method=hungarian": [
            0.9013,
            1.0730,
            1.0531,
            1.0286,
            0.7353,
            1.4094,
            0.5391,
            1.3297,
            1.0881,
            1.7796,
            2.6064,
            3.1577,
            3.2135,
            0.8126,
            1.2909,
            2.5217,
            2.0836,
            1.8325,
            3.1874,
        ],
        "method=min_symmetry": [
            0.9013,
            1.0732,
            1.0797,
            1.0492,
            0.7444,
            1.6257,
            0.5391,
            1.5884,
            1.0935,
            1.9304,
            2.6201,
            3.1742,
            3.2254,
            1.1513,
            1.5206,
            2.5361,
            2.2385,
            1.971,
            3.2037,
        ],
    }

    kwargs_grid = [{"method": None}, {"method": "hungarian"}, {"method": "min_symmetry"}]
    for kwargs in kwargs_grid:
        res_key = "_".join("%s=%s" % (k, v) for k, v in sorted(kwargs.items()))
        assert_array_almost_equal([rmsd(mols[0], mol, **kwargs) for mol in mols[1:]], res[res_key], decimal=4)

    # test shuffled rmsd
    for _ in range(5):
        for kwargs in kwargs_grid:
            # dont use method=None in shuffled tests
            if kwargs["method"] is None:
                continue
            res_key = "_".join("%s=%s" % (k, v) for k, v in sorted(kwargs.items()))
            assert_array_almost_equal(
                [rmsd(mols[0], shuffle_mol(mol), **kwargs) for mol in mols[1:]], res[res_key], decimal=4
            )


def test_rmsd_errors():
    mol = oddt.toolkit.readstring("smi", "c1ccccc1")
    mol.make3D()
    mol.addh()
    mol2 = next(oddt.toolkit.readfile("sdf", os.path.join(test_data_dir, "data/dude/xiap/actives_docked.sdf")))

    for method in [None, "hungarian", "min_symmetry"]:
        with pytest.raises(ValueError, match="Unequal number of atoms"):
            rmsd(mol, mol2, method=method)

        for _ in range(5):
            with pytest.raises(ValueError, match="Unequal number of atoms"):
                rmsd(shuffle_mol(mol), shuffle_mol(mol2), method=method)
