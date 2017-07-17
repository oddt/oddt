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
    assert_array_almost_equal(reference, splif['hash'], decimal=2)


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
