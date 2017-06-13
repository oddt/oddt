import os
from random import shuffle
import numpy as np
from sklearn.utils.testing import (assert_array_equal,
                                   assert_array_almost_equal,
                                   assert_equal,
                                   assert_almost_equal)
import pandas as pd
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
                               dice,
                               tanimoto)

test_data_dir = os.path.dirname(os.path.abspath(__file__))

protein = next(oddt.toolkit.readfile('pdb', os.path.join(
    test_data_dir, 'data/pdbbind/10gs_pocket.pdb')))
protein.protein = True
protein.addh(only_polar=True)

ligand = next(oddt.toolkit.readfile('sdf', os.path.join(
    test_data_dir, 'data/pdbbind/10gs_ligand.sdf')))
ligand.addh(only_polar=True)


def shuffle_mol(mol):
    new_mol = mol.clone
    new_order = list(range(len(mol.atoms)))
    shuffle(new_order)
    if (hasattr(oddt.toolkits, 'ob') and
            isinstance(mol, oddt.toolkits.ob.Molecule)):
        new_mol.OBMol.RenumberAtoms([i + 1 for i in new_order])
    else:
        new_mol.Mol = oddt.toolkits.rdk.Chem.RenumberAtoms(new_mol.Mol, new_order)
    return new_mol


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


def test_InteractionFingerprint():
    """Interaction Fingerprint test"""
    if oddt.toolkit.backend == 'ob':
        IFP = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 1, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    else:
        IFP = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 1, 0, 0, 0, 0, 0,
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
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert_array_equal(IFP, InteractionFingerprint(ligand, protein))


def test_SimpleInteractionFingerprint():
    """Simple Interaction Fingerprint test """
    if oddt.toolkit.backend == 'ob':
        SIFP = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0])
    else:
        SIFP = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                         0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0])
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
    mol1 = oddt.toolkit.readstring("smi", "CC1=C(C(=CC=C1)C)NC(=O)CN2CCN(CC2)CC(=O)N3CCC4=C(C3)C=CS4")
    mol2 = oddt.toolkit.readstring("smi", "CC1=C(C(=CC=C1)O)NC(=O)CN2CCN(CC2)CC(=O)N3CCC4=C(C3)C=CS4")

    mol1_fp_dense = ECFP(mol1, depth=8, size=4096, sparse=False)
    mol2_fp_dense = ECFP(mol2, depth=8, size=4096, sparse=False)

    mol1_fp_sparse = ECFP(mol1, depth=8, size=4096, sparse=True)
    mol2_fp_sparse = ECFP(mol2, depth=8, size=4096, sparse=True)

    assert_almost_equal(dice(mol1_fp_sparse, mol2_fp_sparse, sparse=True),
                        dice(mol1_fp_dense, mol2_fp_dense))
    assert_equal(dice([], [], sparse=True), 0.)
    assert_equal(dice(np.zeros(10), np.zeros(10), sparse=False), 0.)
    assert_almost_equal(tanimoto(mol1_fp_sparse, mol2_fp_sparse, sparse=True),
                        tanimoto(mol1_fp_dense, mol2_fp_dense))
    assert_equal(tanimoto([], [], sparse=True), 0.)
    assert_equal(tanimoto(np.zeros(10), np.zeros(10), sparse=False), 0.)


def test_ecfp():
    """ECFP fingerprints"""
    mol1 = oddt.toolkit.readstring("smi", "CC1=C(C(=CC=C1)C)NC(=O)CN2CCN(CC2)CC(=O)N3CCC4=C(C3)C=CS4")
    mol2 = oddt.toolkit.readstring("smi", "CC1=C(C(=CC=C1)O)NC(=O)CN2CCN(CC2)CC(=O)N3CCC4=C(C3)C=CS4")

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
    mol1 = oddt.toolkit.readstring("smi", "CC1=C(C(=CC=C1)C)NC(=O)CN2CCN(CC2)CC(=O)N3CCC4=C(C3)C=CS4")
    mol2 = oddt.toolkit.readstring("smi", "CC1=C(C(=CC=C1)O)NC(=O)CN2CCN(CC2)CC(=O)N3CCC4=C(C3)C=CS4")

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
    sildenafil = oddt.toolkit.readstring("smi", "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12")

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
    reference = [38, 53, 53, 53, 70, 81, 81, 125, 125, 127, 216, 219, 242,
                 249, 262, 262, 279, 279, 330, 396, 423, 423, 424, 498,
                 498, 570, 592, 622, 625, 626, 676, 676, 817, 818, 818,
                 818, 818, 884, 888, 895, 907, 914, 935, 1030, 1041, 1082,
                 1082, 1115, 1142, 1184, 1184, 1191, 1198, 1263, 1269,
                 1275, 1275, 1275, 1283, 1315, 1315, 1315, 1328, 1328,
                 1344, 1391, 1396, 1435, 1465, 1494, 1502, 1502, 1502,
                 1502, 1506, 1569, 1569, 1569, 1569, 1569, 1569, 1617,
                 1640, 1645, 1697, 1697, 1746, 1796, 1937, 1979, 1997,
                 2000, 2007, 2007, 2020, 2150, 2178, 2195, 2224, 2228,
                 2415, 2417, 2417, 2484, 2509, 2511, 2578, 2578, 2578,
                 2624, 2636, 2665, 2678, 2736, 2776, 2776, 2789, 2857,
                 2862, 2862, 2894, 2923, 2944, 2944, 3035, 3058, 3073,
                 3073, 3073, 3073, 3079, 3137, 3159, 3159, 3166, 3218,
                 3218, 3279, 3279, 3281, 3338, 3353, 3360, 3368, 3387,
                 3605, 3605, 3609, 3615, 3620, 3636, 3650, 3688, 3688,
                 3713, 3713, 3716, 3716, 3729, 3732, 3769, 3843, 3854,
                 3871, 3912, 3927, 3986, 3994, 3994, 4069, 4087, 4087]

    assert_equal(splif['hash'].shape, (172,))
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
        target_outcome = np.array([0.751, 0.705, 0.76, 0.674, 0.745,
                                   0.45, 0.754, 0.477, 0.614, 0.737,
                                   0.727, 0.772, 0.747, 0.585, 0.535,
                                   0.681, 0.554, 0.736, 0.729])
    else:
        target_outcome = np.array([0.751, 0.705, 0.76, 0.674, 0.745,
                                   0.467, 0.754, 0.485, 0.637, 0.737,
                                   0.727, 0.772, 0.747, 0.585, 0.535,
                                   0.681, 0.554, 0.736, 0.729])
    assert_array_almost_equal(outcome, target_outcome, decimal=3)


def test_plec():
    """PLEC fingerprints"""
    mols = list(oddt.toolkit.readfile('sdf', os.path.join(
        test_data_dir, 'data/dude/xiap/actives_docked.sdf')))
    mols = list(filter(lambda x: x.title == '312335', mols))
    list(map(lambda x: x.addh(only_polar=True), mols))
    receptor = next(oddt.toolkit.readfile('pdb', os.path.join(
        test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb')))
    receptor.protein = True
    receptor.addh(only_polar=True)
    plec = PLEC(mols[0], receptor)
    reference = np.array([33,    50,    68,    80,    80,   120,   120,   120,   120,
                         126,   155,   155,   155,   155,   155,   155,   155,   180,
                         213,   213,   214,   214,   214,   226,   226,   282,   296,
                         313,   352,   380,   386,   386,   406,   431,   431,   432,
                         448,   509,   516,   528,   532,   532,   556,   556,   643,
                         729,   741,   778,   795,   812,   812,   827,   857,   877,
                         907,   924,   924,   935,   935,   935,   935,   935,   944,
                         964,   964,   993,   993,   996,   996,  1002,  1002,  1007,
                         1030,  1030,  1030,  1051,  1051,  1113,  1119,  1264,  1281,
                         1290,  1322,  1331,  1377,  1411,  1411,  1475,  1480,  1497,
                         1618,  1618,  1618,  1618,  1626,  1694,  1694,  1704,  1704,
                         1742,  1755,  1755,  1755,  1755,  1783,  1783,  1786,  1862,
                         1867,  1956,  1963,  1990,  1992,  1992,  2024,  2024,  2058,
                         2252,  2298,  2300,  2342,  2386,  2390,  2390,  2488,  2538,
                         2552,  2571,  2595,  2595,  2623,  2623,  2632,  2636,  2636,
                         2648,  2648,  2651,  2651,  2664,  2706,  2706,  2743,  2743,
                         2752,  2789,  2789,  2791,  2800,  2800,  2903,  2929,  2950,
                         2959,  2970,  2970,  2970,  2982,  3049,  3071,  3087,  3104,
                         3111,  3120,  3227,  3265,  3265,  3293,  3295,  3423,  3453,
                         3453,  3453,  3458,  3458,  3458,  3458,  3517,  3539,  3546,
                         3546,  3546,  3546,  3553,  3583,  3630,  3643,  3643,  3659,
                         3673,  3688,  3707,  3716,  3742,  3749,  3749,  3751,  3775,
                         3797,  3843,  3876,  3904,  3904,  3904,  3916,  3916,  3962,
                         3993,  4010,  4113,  4127,  4127,  4165,  4199,  4294,  4329,
                         4329,  4372,  4372,  4461,  4462,  4506,  4515,  4542,  4542,
                         4542,  4564,  4564,  4564,  4614,  4641,  4668,  4686,  4686,
                         4686,  4729,  4741,  4744,  4744,  4744,  4744,  4756,  4793,
                         4793,  4793,  4796,  4814,  4832,  4832,  4861,  4861,  4861,
                         4861,  4861,  4861,  4861,  4861,  4861,  4861,  4861,  4861,
                         4861,  4861,  4881,  4886,  4916,  4978,  4978,  5042,  5044,
                         5055,  5062,  5078,  5087,  5087,  5101,  5101,  5126,  5146,
                         5189,  5232,  5271,  5303,  5303,  5315,  5315,  5379,  5439,
                         5439,  5439,  5439,  5481,  5495,  5551,  5551,  5575,  5575,
                         5585,  5600,  5600,  5612,  5631,  5631,  5631,  5631,  5631,
                         5631,  5688,  5742,  5841,  5841,  5864,  5864,  5871,  5951,
                         5980,  5992,  6010,  6010,  6010,  6027,  6059,  6059,  6077,
                         6077,  6096,  6107,  6107,  6189,  6261,  6261,  6277,  6277,
                         6299,  6413,  6428,  6428,  6428,  6428,  6445,  6453,  6516,
                         6519,  6519,  6540,  6582,  6609,  6654,  6654,  6696,  6716,
                         6717,  6726,  6781,  6781,  6781,  6781,  6833,  6838,  6838,
                         6927,  6927,  6979,  6979,  6997,  7155,  7253,  7267,  7277,
                         7288,  7311,  7325,  7420,  7493,  7501,  7506,  7506,  7520,
                         7530,  7530,  7549,  7549,  7555,  7674,  7678,  7678,  7701,
                         7701,  7701,  7752,  7752,  7752,  7839,  7839,  7847,  7868,
                         7918,  7920,  7922,  7924,  7928,  7957,  7957,  7957,  8003,
                         8010,  8056,  8064,  8081,  8081,  8083,  8086,  8086,  8086,
                         8086,  8160,  8163,  8190,  8244,  8262,  8262,  8275,  8282,
                         8292,  8327,  8327,  8360,  8383,  8383,  8383,  8405,  8405,
                         8416,  8416,  8418,  8457,  8483,  8484,  8484,  8503,  8503,
                         8505,  8505,  8619,  8655,  8657,  8657,  8657,  8657,  8657,
                         8666,  8668,  8697,  8721,  8782,  8784,  8797,  8824,  8886,
                         8910,  8915,  8920,  8923,  8967,  8993,  9040,  9077,  9077,
                         9100,  9138,  9138,  9154,  9154,  9249,  9257,  9274,  9289,
                         9308,  9333,  9415,  9415,  9450,  9450,  9455,  9455,  9481,
                         9527,  9547,  9572,  9585,  9607,  9607,  9610,  9610,  9643,
                         9661,  9700,  9700,  9736,  9736,  9736,  9736,  9745,  9760,
                         9765,  9806,  9857,  9903,  9931,  9931,  9931,  9936,  9940,
                         9968,  9968, 10006, 10040, 10045, 10046, 10080, 10080, 10112,
                         10112, 10113, 10113, 10181, 10181, 10185, 10198, 10252, 10316,
                         10317, 10317, 10340, 10340, 10340, 10353, 10386, 10386, 10399,
                         10484, 10490, 10490, 10504, 10535, 10589, 10589, 10599, 10648,
                         10648, 10650, 10650, 10665, 10665, 10669, 10703, 10714, 10741,
                         10793, 10806, 10806, 10806, 10837, 10878, 10880, 10967, 10978,
                         10978, 10982, 11105, 11141, 11159, 11182, 11213, 11265, 11361,
                         11361, 11361, 11406, 11454, 11454, 11458, 11458, 11458, 11495,
                         11495, 11532, 11562, 11580, 11605, 11605, 11640, 11691, 11697,
                         11698, 11701, 11717, 11717, 11717, 11717, 11753, 11835, 11858,
                         11860, 11957, 11957, 11974, 12009, 12018, 12107, 12115, 12194,
                         12194, 12200, 12200, 12202, 12209, 12230, 12268, 12290, 12290,
                         12295, 12295, 12300, 12308, 12308, 12319, 12324, 12431, 12475,
                         12475, 12475, 12481, 12522, 12544, 12544, 12544, 12583, 12587,
                         12602, 12602, 12632, 12632, 12634, 12641, 12641, 12673, 12734,
                         12740, 12762, 12778, 12807, 12861, 12878, 12878, 12884, 12886,
                         12916, 12955, 12955, 12958, 12972, 12982, 12982, 13057, 13069,
                         13079, 13082, 13082, 13119, 13129, 13200, 13200, 13277, 13277,
                         13316, 13316, 13316, 13316, 13320, 13320, 13336, 13460, 13460,
                         13473, 13475, 13495, 13515, 13526, 13548, 13553, 13602, 13606,
                         13636, 13636, 13655, 13658, 13658, 13688, 13688, 13774, 13774,
                         13784, 13784, 13784, 13809, 13839, 13839, 13839, 13839, 13839,
                         13889, 13905, 13906, 13906, 13906, 13906, 13920, 13920, 13920,
                         13920, 13920, 13929, 13929, 13936, 13947, 13949, 13949, 14053,
                         14074, 14122, 14122, 14142, 14148, 14169, 14190, 14259, 14259,
                         14317, 14347, 14393, 14408, 14408, 14423, 14423, 14423, 14423,
                         14423, 14423, 14423, 14439, 14464, 14469, 14485, 14510, 14510,
                         14516, 14516, 14520, 14529, 14529, 14529, 14549, 14563, 14563,
                         14582, 14603, 14605, 14605, 14650, 14748, 14748, 14750, 14756,
                         14756, 14772, 14780, 14798, 14813, 14857, 14857, 14903, 14903,
                         14921, 14988, 14993, 14993, 15008, 15012, 15018, 15030, 15033,
                         15044, 15044, 15081, 15103, 15146, 15146, 15190, 15231, 15231,
                         15253, 15258, 15274, 15274, 15299, 15311, 15311, 15324, 15336,
                         15429, 15429, 15444, 15450, 15497, 15513, 15513, 15521, 15521,
                         15553, 15577, 15608, 15648, 15651, 15672, 15690, 15712, 15713,
                         15715, 15751, 15798, 15798, 15811, 15857, 15879, 15894, 15934,
                         15950, 15982, 15982, 16001, 16023, 16023, 16053, 16053, 16099,
                         16190, 16234, 16234, 16234, 16252, 16252, 16252, 16252, 16252,
                         16255, 16335, 16338, 16362, 16362])
    assert_array_equal(reference, plec)
    assert_array_equal(plec.shape, (860,))


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
