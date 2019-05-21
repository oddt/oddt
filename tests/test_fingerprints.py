import os
import numpy as np
from scipy.sparse import vstack as sparse_vstack
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           assert_almost_equal)

import pytest

import oddt
from oddt.fingerprints import (InteractionFingerprint,
                               SimpleInteractionFingerprint,
                               ECFP,
                               SPLIF,
                               similarity_SPLIF,
                               PLEC,
                               fold,
                               MIN_HASH_VALUE,
                               MAX_HASH_VALUE,
                               sparse_to_dense,
                               sparse_to_csr_matrix,
                               csr_matrix_to_sparse,
                               dense_to_sparse,
                               dice,
                               tanimoto,
                               ri_score,
                               b_factor)
from .utils import shuffle_mol


test_data_dir = os.path.dirname(os.path.abspath(__file__))

protein = next(oddt.toolkit.readfile('pdb', os.path.join(
    test_data_dir, 'data/pdbbind/10gs/10gs_pocket.pdb')))
protein.protein = True
protein.addh(only_polar=True)

ligand = next(oddt.toolkit.readfile('sdf', os.path.join(
    test_data_dir, 'data/pdbbind/10gs/10gs_ligand.sdf')))
ligand.addh(only_polar=True)


def test_folding():
    """FP Folding"""
    # Upper bound
    assert_array_equal(fold([MAX_HASH_VALUE], 1024), [1023])
    assert_array_equal(fold([MAX_HASH_VALUE], 1234567890), [1234567889])
    assert_array_equal(fold([MAX_HASH_VALUE], MAX_HASH_VALUE / 2),
                       [MAX_HASH_VALUE / 2 - 1])
    assert_array_equal(fold([MAX_HASH_VALUE], MAX_HASH_VALUE - 1),
                       [MAX_HASH_VALUE - 2])
    # Lower bound
    assert_array_equal(fold([MIN_HASH_VALUE], 1024), [0])
    assert_array_equal(fold([MIN_HASH_VALUE], 1234567890), [0])
    assert_array_equal(fold([MIN_HASH_VALUE], MAX_HASH_VALUE / 2), [0])
    assert_array_equal(fold([MIN_HASH_VALUE], MAX_HASH_VALUE - 1), [0])

    # Range check
    fp = np.arange(1, MAX_HASH_VALUE, 1e6, dtype=int)
    assert_array_equal(fold(fp, MAX_HASH_VALUE), fp - 1)


def test_sparse_densify():
    """FP densify"""
    sparse_fp = [0, 33, 49, 53, 107, 156, 161, 203, 215, 230, 251, 269, 299,
                 323, 331, 376, 389, 410, 427, 430, 450, 484, 538, 592, 593,
                 636, 646, 658, 698, 699, 702, 741, 753, 807, 850, 861, 882,
                 915, 915, 915, 969, 969, 1023]

    # count vectors
    dense = sparse_to_dense(sparse_fp, size=1024, count_bits=True)
    csr = sparse_to_csr_matrix(sparse_fp, size=1024, count_bits=True)
    assert_array_equal(dense.reshape(1, -1), csr.toarray())
    resparsed = dense_to_sparse(dense)
    resparsed_csr = csr_matrix_to_sparse(csr)
    assert_array_equal(sparse_fp, resparsed)
    assert_array_equal(sparse_fp, resparsed_csr)

    # bool vectors
    dense = sparse_to_dense(sparse_fp, size=1024, count_bits=False)
    csr = sparse_to_csr_matrix(sparse_fp, size=1024, count_bits=False)
    assert_array_equal(dense.reshape(1, -1), csr.toarray())
    resparsed = dense_to_sparse(dense)
    resparsed_csr = csr_matrix_to_sparse(csr)
    assert_array_equal(np.unique(sparse_fp), resparsed)
    assert_array_equal(np.unique(sparse_fp), resparsed_csr)

    # test stacking
    np.random.seed(0)
    sparse_fps = np.random.randint(0, 1024, size=(20, 100))
    dense = np.vstack(sparse_to_dense(fp, size=1024) for fp in sparse_fps)
    csr = sparse_vstack(sparse_to_csr_matrix(fp, size=1024)
                        for fp in sparse_fps)
    assert_array_equal(dense, csr.toarray())

    # test exceptions
    with pytest.raises(ValueError):
        csr_matrix_to_sparse(np.array([1, 2, 3]))


def test_InteractionFingerprint():
    """Interaction Fingerprint test"""
    if oddt.toolkit.backend == 'ob':
        IFP = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
               0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               1, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        IFP = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
               0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    assert_array_equal(IFP, InteractionFingerprint(ligand, protein))


def test_SimpleInteractionFingerprint():
    """Simple Interaction Fingerprint test """
    if oddt.toolkit.backend == 'ob':
        SIFP = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]
    else:
        SIFP = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]
    assert_array_equal(SIFP, SimpleInteractionFingerprint(ligand, protein))


def test_IFP_SIFP_Folding_cum_sum():
    """Checks, whether InteractionFingerprint and SimpleInteractionFingerprint outcomes matches"""
    IFP = np.sum(InteractionFingerprint(ligand, protein), axis=0)
    SIFP = np.sum(SimpleInteractionFingerprint(ligand, protein), axis=0)
    assert_array_equal(IFP, SIFP)


def test_similarity():
    """FP similarity"""
    mols = list(oddt.toolkit.readfile('sdf', os.path.join(
        test_data_dir, 'data/dude/xiap/actives_docked.sdf')))
    mols = list(filter(lambda x: x.title == '312335', mols))
    list(map(lambda x: x.addh(only_polar=True), mols))
    receptor = next(oddt.toolkit.readfile('pdb', os.path.join(
        test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb')))
    receptor.protein = True
    receptor.addh(only_polar=True)
    ref = SimpleInteractionFingerprint(mols[0], receptor)

    outcome = [dice(ref, SimpleInteractionFingerprint(
        mol, receptor)) for mol in mols[1:]]
    if oddt.toolkit.backend == 'ob':
        target_outcome = np.array([0.8, 0.625, 0.764706, 0.628571, 0.764706,
                                   0.611111, 0.787879, 0.6, 0.62069,
                                   0.6875, 0.555556, 0.727273, 0.642857,
                                   0.685714, 0.736842, 0.666667, 0.484848,
                                   0.533333, 0.588235])
    else:
        target_outcome = np.array([0.810811, 0.625, 0.777778, 0.611111,
                                   0.777778, 0.648649, 0.787879, 0.6, 0.6,
                                   0.666667, 0.578947, 0.742857, 0.62069,
                                   0.628571, 0.736842, 0.645161, 0.571429,
                                   0.580645, 0.628571])
    assert_array_almost_equal(outcome, target_outcome)

    outcome = [tanimoto(ref, SimpleInteractionFingerprint(
        mol, receptor)) for mol in mols[1:]]
    if oddt.toolkit.backend == 'ob':
        target_outcome = np.array([0.75, 0.5, 0.727273, 0.538462, 0.727273,
                                   0.727273, 0.8, 0.636364, 0.545455, 0.636364,
                                   0.636364, 0.636364, 0.7, 0.727273, 0.75,
                                   0.636364, 0.454545, 0.454545, 0.416667])
    else:
        target_outcome = np.array([0.75, 0.416667, 0.727273, 0.538462, 0.727273,
                                   0.727273, 0.7, 0.636364, 0.545455, 0.545455,
                                   0.636364, 0.636364, 0.6, 0.636364, 0.75,
                                   0.545455, 0.545455, 0.454545, 0.416667])
    assert_array_almost_equal(outcome, target_outcome)


def test_sparse_similarity():
    """Sparse similarity"""
    mol1 = oddt.toolkit.readstring(
        "smi", "CC1=C(C(=CC=C1)C)NC(=O)CN2CCN(CC2)CC(=O)N3CCC4=C(C3)C=CS4")
    mol2 = oddt.toolkit.readstring(
        "smi", "CC1=C(C(=CC=C1)O)NC(=O)CN2CCN(CC2)CC(=O)N3CCC4=C(C3)C=CS4")

    mol1_fp_dense = ECFP(mol1, depth=8, size=4096, sparse=False)
    mol2_fp_dense = ECFP(mol2, depth=8, size=4096, sparse=False)

    mol1_fp_sparse = ECFP(mol1, depth=8, size=4096, sparse=True)
    mol2_fp_sparse = ECFP(mol2, depth=8, size=4096, sparse=True)

    assert_almost_equal(dice(mol1_fp_sparse, mol2_fp_sparse, sparse=True),
                        dice(mol1_fp_dense, mol2_fp_dense))
    assert dice([], [], sparse=True) == 0.
    assert dice(np.zeros(10), np.zeros(10), sparse=False) == 0.
    assert_almost_equal(tanimoto(mol1_fp_sparse, mol2_fp_sparse, sparse=True),
                        tanimoto(mol1_fp_dense, mol2_fp_dense))
    assert tanimoto([], [], sparse=True) == 0.
    assert tanimoto(np.zeros(10), np.zeros(10), sparse=False) == 0.


def test_ecfp():
    """ECFP fingerprints"""
    mol1 = oddt.toolkit.readstring(
        "smi", "CC1=C(C(=CC=C1)C)NC(=O)CN2CCN(CC2)CC(=O)N3CCC4=C(C3)C=CS4")
    mol2 = oddt.toolkit.readstring(
        "smi", "CC1=C(C(=CC=C1)O)NC(=O)CN2CCN(CC2)CC(=O)N3CCC4=C(C3)C=CS4")

    mol1_fp = ECFP(mol1, depth=8, size=4096, sparse=False)
    mol2_fp = ECFP(mol2, depth=8, size=4096, sparse=False)

    ref1 = [2, 100, 176, 185, 200, 203, 359, 382, 447, 509, 518, 550, 572, 583,
            598, 606, 607, 684, 818, 821, 832, 861, 960, 992, 1006, 1019, 1042,
            1050, 1059, 1103, 1175, 1281, 1315, 1377, 1431, 1470, 1479, 1512,
            1577, 1588, 1598, 1620, 1633, 1647, 1663, 1723, 1749, 1751, 1775,
            1781, 1821, 1837, 1899, 1963, 1969, 1986, 2013, 2253, 2343, 2355,
            2368, 2435, 2547, 2654, 2657, 2702, 2722, 2725, 2803, 2816, 2853,
            2870, 2920, 2992, 3028, 3056, 3074, 3103, 3190, 3203, 3277, 3321,
            3362, 3377, 3383, 3401, 3512, 3546, 3552, 3585, 3593, 3617, 3674,
            3759, 3784, 3790, 3832, 3895, 3937, 3956, 3974, 4007, 4033]

    ref2 = [43, 100, 176, 200, 203, 231, 382, 396, 447, 490, 518, 583, 606,
            607, 650, 818, 821, 832, 840, 861, 907, 950, 960, 992, 1006, 1013,
            1019, 1042, 1050, 1059, 1103, 1104, 1112, 1175, 1281, 1293, 1315,
            1377, 1431, 1470, 1512, 1543, 1577, 1588, 1598, 1633, 1647, 1663,
            1723, 1749, 1751, 1757, 1759, 1775, 1781, 1821, 1837, 1880, 1963,
            1969, 1986, 2253, 2355, 2368, 2435, 2544, 2547, 2654, 2702, 2722,
            2725, 2726, 2799, 2816, 2853, 2870, 2920, 2992, 3028, 3074, 3190,
            3203, 3277, 3290, 3333, 3362, 3383, 3401, 3512, 3546, 3552, 3585,
            3593, 3617, 3640, 3660, 3674, 3759, 3784, 3790, 3805, 3832, 3856,
            3895, 3924, 3956, 3974, 3992, 4007, 4033]

    assert_array_equal(ref1, np.where(mol1_fp)[0])
    assert_array_equal(ref2, np.where(mol2_fp)[0])

    assert_almost_equal(dice(mol1_fp, mol2_fp), 0.69999999)
    assert_almost_equal(tanimoto(mol1_fp, mol2_fp), 0.63846153)

    # adding Hs should not change anything
    mol1.addh()
    mol2.addh()

    mol1_fp = ECFP(mol1, depth=8, size=4096, sparse=False)
    mol2_fp = ECFP(mol2, depth=8, size=4096, sparse=False)

    assert_array_equal(ref1, np.where(mol1_fp)[0])
    assert_array_equal(ref2, np.where(mol2_fp)[0])

    assert_almost_equal(dice(mol1_fp, mol2_fp), 0.69999999)
    assert_almost_equal(tanimoto(mol1_fp, mol2_fp), 0.63846153)

    # removig Hs should not change anything
    mol1.removeh()
    mol2.removeh()

    mol1_fp = ECFP(mol1, depth=8, size=4096, sparse=False)
    mol2_fp = ECFP(mol2, depth=8, size=4096, sparse=False)

    assert_array_equal(ref1, np.where(mol1_fp)[0])
    assert_array_equal(ref2, np.where(mol2_fp)[0])

    assert_almost_equal(dice(mol1_fp, mol2_fp), 0.69999999)
    assert_almost_equal(tanimoto(mol1_fp, mol2_fp), 0.63846153)


def test_fcfp():
    """FCFP fingerprints"""
    mol1 = oddt.toolkit.readstring(
        "smi", "CC1=C(C(=CC=C1)C)NC(=O)CN2CCN(CC2)CC(=O)N3CCC4=C(C3)C=CS4")
    mol2 = oddt.toolkit.readstring(
        "smi", "CC1=C(C(=CC=C1)O)NC(=O)CN2CCN(CC2)CC(=O)N3CCC4=C(C3)C=CS4")

    mol1_fp = ECFP(mol1, depth=8, size=4096,
                   sparse=False, use_pharm_features=True)
    mol2_fp = ECFP(mol2, depth=8, size=4096,
                   sparse=False, use_pharm_features=True)

    ref1 = [46, 111, 305, 310, 362, 384, 409, 451, 467, 548, 572, 595, 607,
            608, 620, 659, 691, 699, 724, 743, 752, 842, 926, 935, 974, 1037,
            1072, 1094, 1135, 1143, 1161, 1172, 1313, 1325, 1368, 1399, 1461,
            1486, 1488, 1492, 1603, 1619, 1648, 1665, 1666, 1838, 1887, 1900,
            1948, 1961, 1972, 1975, 1996, 2000, 2052, 2085, 2094, 2174, 2232,
            2236, 2368, 2382, 2383, 2402, 2483, 2492, 2527, 2593, 2616, 2706,
            2789, 2899, 2922, 2945, 2966, 3102, 3117, 3176, 3189, 3215, 3225,
            3297, 3326, 3349, 3373, 3513, 3525, 3535, 3601, 3619, 3780, 3820,
            3897, 3919, 3976, 3981, 4050, 4079, 4091]

    ref2 = [46, 111, 143, 172, 259, 305, 362, 409, 451, 467, 507, 518, 548,
            583, 595, 607, 608, 620, 639, 691, 693, 724, 752, 784, 825, 842,
            926, 1037, 1087, 1094, 1098, 1135, 1143, 1161, 1172, 1286, 1325,
            1368, 1371, 1395, 1399, 1461, 1486, 1488, 1492, 1565, 1619, 1648,
            1655, 1665, 1887, 1890, 1900, 1948, 1961, 1968, 1972, 1975, 1976,
            1996, 2000, 2007, 2094, 2125, 2174, 2232, 2236, 2368, 2382, 2383,
            2483, 2492, 2571, 2593, 2606, 2638, 2706, 2789, 2922, 2945, 2966,
            2986, 3030, 3100, 3102, 3117, 3227, 3326, 3350, 3373, 3406, 3419,
            3535, 3577, 3619, 3697, 3742, 3820, 3839, 3919, 3981, 4043, 4050,
            4079, 4091]

    assert_array_equal(ref1, np.where(mol1_fp)[0])
    assert_array_equal(ref2, np.where(mol2_fp)[0])

    assert_almost_equal(dice(mol1_fp, mol2_fp), 0.64074074)
    assert_almost_equal(tanimoto(mol1_fp, mol2_fp), 0.5)

    # adding Hs should not change anything
    mol1.addh()
    mol2.addh()

    assert_array_equal(ref1, np.where(mol1_fp)[0])
    assert_array_equal(ref2, np.where(mol2_fp)[0])

    assert_almost_equal(dice(mol1_fp, mol2_fp), 0.64074074)
    assert_almost_equal(tanimoto(mol1_fp, mol2_fp), 0.5)


def test_ecfp_invaraiants():
    """ECFP: test random reordering"""
    sildenafil = oddt.toolkit.readstring(
        "smi", "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12")

    params = {'depth': 4, 'size': 4096, 'sparse': True}
    fp = ECFP(sildenafil, **params)

    for n in range(10):
        sildenafil = shuffle_mol(sildenafil)
        assert_array_equal(fp, ECFP(sildenafil, **params))


def test_splif():
    """SPLIF fingerprints"""
    mols = list(oddt.toolkit.readfile('sdf', os.path.join(
        test_data_dir, 'data/dude/xiap/actives_docked.sdf')))
    mols = list(filter(lambda x: x.title == '312335', mols))
    list(map(lambda x: x.addh(only_polar=True), mols))
    receptor = next(oddt.toolkit.readfile('pdb', os.path.join(
        test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb')))
    receptor.protein = True
    receptor.addh(only_polar=True)
    splif = SPLIF(mols[0], receptor)
    reference = [6, 38, 49, 53, 53, 53, 70, 70, 81, 81, 81, 81, 165, 216, 219,
                 249, 330, 330, 333, 377, 380, 396, 396, 396, 423, 423, 479,
                 479, 498, 498, 498, 570, 592, 625, 638, 768, 768, 817, 818,
                 818, 818, 818, 858, 884, 888, 907, 930, 934, 935, 971, 1023,
                 1041, 1115, 1142, 1184, 1184, 1252, 1263, 1269, 1275, 1275,
                 1275, 1315, 1315, 1315, 1337, 1337, 1344, 1351, 1396, 1435,
                 1465, 1502, 1502, 1502, 1502, 1569, 1569, 1569, 1569, 1569,
                 1569, 1569, 1569, 1640, 1645, 1660, 1660, 1697, 1697, 1716,
                 1746, 1756, 1778, 1901, 1937, 1997, 2000, 2000, 2000, 2007,
                 2007, 2020, 2070, 2195, 2274, 2294, 2319, 2415, 2417, 2509,
                 2528, 2578, 2578, 2584, 2590, 2590, 2624, 2636, 2678, 2678,
                 2678, 2678, 2678, 2776, 2776, 2789, 2862, 2862, 2894, 2894,
                 2894, 2923, 2923, 3058, 3073, 3073, 3073, 3073, 3137, 3159,
                 3159, 3159, 3186, 3218, 3218, 3279, 3279, 3281, 3338, 3358,
                 3360, 3368, 3387, 3609, 3636, 3636, 3713, 3713, 3716, 3716,
                 3748, 3767, 3769, 3854, 3871, 3912, 3968, 3986, 3994, 3994,
                 4069]

    assert splif['hash'].shape == (172,)
    assert_array_equal(splif['ligand_coords'].shape, (172, 7, 3))
    assert_array_equal(splif['protein_coords'].shape, (172, 7, 3))
    assert_array_equal(reference, splif['hash'])


def test_splif_similarity():
    """SPLIF similarity"""
    mols = list(oddt.toolkit.readfile('sdf', os.path.join(
        test_data_dir, 'data/dude/xiap/actives_docked.sdf')))
    mols = list(filter(lambda x: x.title == '312335', mols))
    list(map(lambda x: x.addh(only_polar=True), mols))
    receptor = next(oddt.toolkit.readfile('pdb', os.path.join(
        test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb')))
    receptor.protein = True
    receptor.addh(only_polar=True)
    # print(outcome)
    ref = SPLIF(mols[0], receptor)
    outcome = [similarity_SPLIF(ref, SPLIF(mol, receptor)) for mol in mols[1:]]
    if oddt.toolkit.backend == 'ob':
        target_outcome = np.array([0.694,  0.639,  0.707,  0.657,  0.690,
                                   0.450,  0.702,  0.477,  0.576,  0.669,
                                   0.658,  0.714,  0.683,  0.585,  0.535,
                                   0.611,  0.554,  0.665,  0.697])
    else:
        target_outcome = np.array([0.694,  0.639,  0.707,  0.657,  0.690,
                                   0.467,  0.702,  0.485,  0.599,  0.669,
                                   0.658,  0.714,  0.683,  0.585,  0.535,
                                   0.611,  0.554,  0.665,  0.697])
    assert_array_almost_equal(outcome, target_outcome, decimal=3)


def test_plec():
    """PLEC fingerprints"""
    mols = list(oddt.toolkit.readfile('sdf', os.path.join(
        test_data_dir, 'data/dude/xiap/actives_docked.sdf')))
    mols = list(filter(lambda x: x.title == '312335', mols))
    list(map(lambda x: x.removeh(), mols))
    receptor = next(oddt.toolkit.readfile('pdb', os.path.join(
        test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb')))
    receptor.protein = True
    receptor.removeh()
    plec = PLEC(mols[0], receptor)
    reference = [80, 119, 120, 120, 120, 120, 137, 138, 155, 155, 155, 155,
                 155, 155, 155, 161, 199, 214, 214, 214, 226, 226, 233, 266,
                 282, 283, 283, 313, 313, 386, 386, 430, 431, 431, 432, 448,
                 581, 581, 643, 662, 684, 690, 729, 737, 741, 778, 778, 795,
                 799, 799, 812, 812, 876, 877, 894, 907, 924, 924, 925, 925,
                 935, 935, 935, 935, 935, 964, 964, 964, 993, 993, 996, 996,
                 1002, 1002, 1042, 1042, 1066, 1066, 1077, 1113, 1119, 1224,
                 1266, 1266, 1290, 1322, 1322, 1334, 1334, 1403, 1411, 1411,
                 1461, 1475, 1480, 1497, 1521, 1584, 1584, 1614, 1618, 1618,
                 1618, 1618, 1691, 1694, 1694, 1755, 1755, 1755, 1755, 1786,
                 1835, 1835, 1867, 1953, 1953, 1953, 1953, 1963, 1970, 1970,
                 1990, 1992, 1992, 1992, 2024, 2024, 2060, 2252, 2373, 2383,
                 2383, 2390, 2390, 2451, 2537, 2538, 2552, 2555, 2558, 2640,
                 2720, 2752, 2791, 2821, 2821, 2931, 2950, 2957, 2957, 2959,
                 2961, 2961, 2961, 2963, 2970, 2970, 2982, 3034, 3049, 3066,
                 3084, 3084, 3084, 3104, 3126, 3227, 3248, 3293, 3293, 3293,
                 3420, 3439, 3517, 3539, 3546, 3546, 3546, 3546, 3553, 3559,
                 3596, 3630, 3643, 3643, 3674, 3707, 3708, 3716, 3738, 3742,
                 3828, 3846, 3859, 3876, 3887, 3904, 3904, 3904, 3916, 3916,
                 3939, 3941, 3981, 3981, 3991, 3993, 4010, 4097, 4127, 4127,
                 4127, 4127, 4165, 4181, 4192, 4316, 4330, 4372, 4391, 4461,
                 4462, 4463, 4542, 4542, 4542, 4549, 4549, 4549, 4549, 4614,
                 4615, 4657, 4668, 4670, 4686, 4686, 4686, 4688, 4688, 4688,
                 4688, 4695, 4729, 4740, 4741, 4744, 4744, 4744, 4744, 4756,
                 4814, 4828, 4828, 4861, 4861, 4861, 4861, 4861, 4861, 4861,
                 4861, 4861, 4861, 4861, 4861, 4861, 4861, 4861, 4861, 4916,
                 4945, 4945, 5011, 5037, 5042, 5044, 5046, 5055, 5078, 5080,
                 5101, 5101, 5126, 5139, 5146, 5189, 5193, 5232, 5271, 5314,
                 5321, 5350, 5379, 5439, 5439, 5439, 5439, 5481, 5482, 5535,
                 5563, 5565, 5565, 5585, 5601, 5601, 5626, 5626, 5631, 5631,
                 5631, 5631, 5631, 5631, 5639, 5670, 5688, 5690, 5742, 5804,
                 5804, 5864, 5871, 5885, 5983, 5992, 6010, 6010, 6010, 6059,
                 6059, 6096, 6164, 6183, 6183, 6197, 6234, 6256, 6261, 6261,
                 6277, 6277, 6277, 6277, 6299, 6333, 6333, 6388, 6388, 6404,
                 6428, 6428, 6428, 6428, 6431, 6431, 6445, 6449, 6450, 6480,
                 6496, 6519, 6519, 6540, 6582, 6642, 6654, 6654, 6671, 6717,
                 6722, 6735, 6735, 6735, 6764, 6764, 6781, 6781, 6781, 6781,
                 6788, 6788, 6803, 6808, 6833, 6838, 6838, 6950, 6979, 6979,
                 6997, 7069, 7115, 7194, 7250, 7254, 7277, 7288, 7352, 7464,
                 7493, 7506, 7506, 7520, 7530, 7530, 7530, 7542, 7546, 7561,
                 7608, 7678, 7678, 7685, 7701, 7701, 7701, 7752, 7752, 7752,
                 7790, 7847, 7957, 7957, 7957, 7959, 8003, 8003, 8003, 8010,
                 8083, 8086, 8086, 8086, 8086, 8113, 8116, 8160, 8190, 8230,
                 8230, 8262, 8262, 8282, 8284, 8284, 8292, 8297, 8327, 8327,
                 8383, 8383, 8383, 8418, 8418, 8426, 8457, 8484, 8484, 8543,
                 8543, 8580, 8629, 8651, 8655, 8697, 8726, 8781, 8784, 8796,
                 8837, 8850, 8923, 9034, 9040, 9077, 9077, 9099, 9134, 9180,
                 9206, 9257, 9281, 9304, 9304, 9333, 9341, 9358, 9393, 9394,
                 9432, 9450, 9450, 9455, 9455, 9481, 9493, 9493, 9505, 9537,
                 9547, 9572, 9585, 9610, 9610, 9661, 9689, 9690, 9690, 9700,
                 9700, 9733, 9736, 9736, 9736, 9736, 9765, 9784, 9885, 9885,
                 9885, 9934, 9938, 9968, 9968, 10037, 10080, 10080, 10103,
                 10113, 10113, 10114, 10115, 10115, 10115, 10139, 10139, 10139,
                 10139, 10139, 10181, 10181, 10181, 10181, 10185, 10286, 10295,
                 10317, 10317, 10340, 10340, 10340, 10340, 10352, 10353, 10364,
                 10364, 10385, 10490, 10490, 10504, 10535, 10539, 10539, 10589,
                 10589, 10591, 10599, 10648, 10648, 10650, 10650, 10681, 10703,
                 10714, 10714, 10714, 10739, 10739, 10793, 10806, 10806, 10806,
                 10837, 10865, 10865, 10871, 10903, 10978, 10978, 11056, 11056,
                 11141, 11159, 11207, 11213, 11257, 11272, 11360, 11362, 11377,
                 11454, 11454, 11458, 11458, 11458, 11539, 11563, 11580, 11580,
                 11580, 11605, 11605, 11610, 11610, 11613, 11624, 11664, 11664,
                 11683, 11683, 11697, 11698, 11701, 11707, 11753, 11835, 11846,
                 11852, 11858, 11876, 11879, 11890, 11957, 11957, 12009, 12115,
                 12130, 12151, 12222, 12268, 12290, 12290, 12295, 12295, 12320,
                 12431, 12448, 12475, 12475, 12475, 12481, 12485, 12487, 12587,
                 12632, 12632, 12634, 12641, 12641, 12641, 12664, 12761, 12761,
                 12778, 12832, 12878, 12878, 12884, 12958, 12982, 12982, 12982,
                 12982, 12992, 13057, 13079, 13121, 13129, 13200, 13200, 13277,
                 13277, 13317, 13317, 13320, 13320, 13336, 13388, 13434, 13443,
                 13475, 13495, 13517, 13517, 13553, 13602, 13637, 13655, 13658,
                 13658, 13688, 13688, 13774, 13774, 13784, 13784, 13784, 13786,
                 13791, 13791, 13809, 13839, 13839, 13839, 13839, 13839, 13876,
                 13905, 13906, 13906, 13906, 13906, 13920, 13920, 13920, 13920,
                 13920, 13949, 13949, 14058, 14122, 14122, 14133, 14133, 14198,
                 14259, 14259, 14317, 14332, 14368, 14386, 14423, 14423, 14423,
                 14423, 14423, 14423, 14423, 14439, 14440, 14447, 14464, 14464,
                 14469, 14505, 14510, 14510, 14513, 14516, 14516, 14529, 14529,
                 14529, 14549, 14563, 14563, 14570, 14570, 14570, 14582, 14605,
                 14605, 14611, 14748, 14748, 14750, 14757, 14772, 14798, 14802,
                 14810, 14854, 14857, 14857, 14878, 14878, 14903, 14903, 14993,
                 14993, 14996, 15008, 15012, 15018, 15044, 15044, 15074, 15092,
                 15092, 15146, 15146, 15191, 15251, 15251, 15253, 15258, 15311,
                 15311, 15317, 15429, 15429, 15441, 15444, 15498, 15518, 15520,
                 15622, 15622, 15622, 15651, 15672, 15712, 15715, 15798, 15798,
                 15811, 15950, 15982, 15982, 15987, 16023, 16023, 16042, 16049,
                 16054, 16080, 16099, 16119, 16119, 16119, 16174, 16174, 16213,
                 16225, 16229, 16234, 16234, 16234, 16252, 16252, 16252, 16252,
                 16252, 16320, 16328, 16362, 16362]

    assert_array_equal(reference, plec)
    assert_array_equal(plec.shape, (860,))

    # Hydrogens should not impact the PLEC fingerprint
    list(map(lambda x: x.addh(only_polar=True), mols))
    receptor.addh(only_polar=True)
    plec = PLEC(mols[0], receptor)
    assert_array_equal(reference, plec, "Polar Hs break PLEC")

    list(map(lambda x: x.addh(), mols))
    receptor.addh()
    plec = PLEC(mols[0], receptor)
    assert_array_equal(reference, plec, "Non-polar Hs break PLEC")


def test_plec_binded_hoh():
    # if water coordinates metal in PDB and ligand is in contact with it, HOH
    # will pop up in metals environment, thus we cannot ignore HOHs in repr_dict

    if (oddt.toolkit.backend == 'ob' or
            (oddt.toolkit.backend == 'rdk' and
             oddt.toolkit.__version__ >= '2017.03')):
        ligand = next(oddt.toolkit.readfile('sdf', os.path.join(
            test_data_dir, 'data', 'pdb', '3kwa_ligand.sdf')))
        protein = next(oddt.toolkit.readfile('pdb', os.path.join(
            test_data_dir, 'data', 'pdb', '3kwa_5Apocket.pdb')))
        protein.protein = True

        assert len(PLEC(ligand, protein, ignore_hoh=True)) == 465
        assert len(PLEC(ligand, protein, ignore_hoh=False)) == 560


def test_plec_similarity():
    """PLEC similarity"""
    mols = list(oddt.toolkit.readfile('sdf', os.path.join(
        test_data_dir, 'data/dude/xiap/actives_docked.sdf')))
    mols = list(filter(lambda x: x.title == '312335', mols))
    list(map(lambda x: x.addh(only_polar=True), mols))
    receptor = next(oddt.toolkit.readfile('pdb', os.path.join(
        test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb')))
    receptor.protein = True
    receptor.addh(only_polar=True)
    reference_sparse = PLEC(mols[0], receptor)
    outcome_sparse = [dice(reference_sparse, PLEC(mol, receptor),
                           sparse=True) for mol in mols[1:]]
    target_outcome = np.array([0.833,  0.729,  0.849,  0.785,  0.821,
                               0.604,  0.868,  0.656, 0.712,  0.652,
                               0.699,  0.785,  0.736,  0.745,  0.661,
                               0.667, 0.555,  0.616,  0.714])
    reference_dense = PLEC(mols[0], receptor, sparse=False)
    outcome_dense = [dice(reference_dense, PLEC(mol, receptor, sparse=False),
                          sparse=False) for mol in mols[1:]]
    assert_array_almost_equal(outcome_sparse, target_outcome, decimal=2)
    assert_array_almost_equal(outcome_dense, target_outcome, decimal=2)


def test_ri_score():
    """Rigidity Index"""
    receptor = next(oddt.toolkit.readfile('pdb', os.path.join(
        test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb')))
    receptor.protein = True
    receptor.addh(only_polar=True)

    ligands = list(oddt.toolkit.readfile('sdf', os.path.join(
        test_data_dir, 'data/dude/xiap/actives_docked.sdf')))
    ligands = list(filter(lambda x: x.title == '312335', ligands))
    _ = list(map(lambda x: x.addh(only_polar=True), ligands))

    ri_score_target = np.array([
        4211.84, 4193.967, 4295.324, 4140.515, 4182.688, 4130.795, 4212.946,
        4119.207, 4261.942, 4146.171, 4175.418, 3810.425, 3695.924, 3702.532,
        4144.078, 4317.13, 3763.041, 4082.629, 4063.534, 3751.246])

    ri_score_computed = np.array(
        [ri_score(ligand, receptor) for ligand in ligands]).round(3)

    assert_array_equal(ri_score_target, ri_score_computed)


def test_b_factor():
    """Flexibility-Rigity Index"""
    receptor = next(oddt.toolkit.readfile('pdb', os.path.join(
        test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb')))
    receptor.protein = True
    receptor.addh(only_polar=True)

    ligands = list(oddt.toolkit.readfile('sdf', os.path.join(
        test_data_dir, 'data/dude/xiap/actives_docked.sdf')))
    ligands = list(filter(lambda x: x.title == '312335', ligands))
    _ = list(map(lambda x: x.addh(only_polar=True), ligands))

    b_factor_target = np.array([
        -0.055, -0.055, -0.056, -0.055, -0.055, -0.055, -0.055, -0.054,
        -0.056, -0.055, -0.055, -0.052, -0.05, -0.05, -0.055, -0.057,
        -0.051, -0.054, -0.054, -0.051])

    b_factor_computed = np.array(
        [b_factor(ligand, receptor) for ligand in ligands]).round(3)

    assert_array_equal(b_factor_target, b_factor_computed)
