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
    assert_almost_equal(tanimoto(mol1_fp_sparse, mol2_fp_sparse, sparse=True),
                        tanimoto(mol1_fp_dense, mol2_fp_dense))


def test_ecfp():
    """ECFP fingerprints"""
    mol1 = oddt.toolkit.readstring("smi", "CC1=C(C(=CC=C1)C)NC(=O)CN2CCN(CC2)CC(=O)N3CCC4=C(C3)C=CS4")
    mol2 = oddt.toolkit.readstring("smi", "CC1=C(C(=CC=C1)O)NC(=O)CN2CCN(CC2)CC(=O)N3CCC4=C(C3)C=CS4")

    mol1_fp = ECFP(mol1, depth=8, size=4096, sparse=False)
    mol2_fp = ECFP(mol2, depth=8, size=4096, sparse=False)

    ref1 = [2, 100, 176, 185, 200, 203, 359, 382, 447, 509, 519, 550, 572, 583,
            598, 607, 684, 818, 822, 832, 862, 960, 992, 1007, 1019, 1043,
            1050, 1059, 1103, 1175, 1282, 1315, 1377, 1431, 1470, 1479, 1512,
            1578, 1588, 1598, 1620, 1633, 1647, 1664, 1723, 1749, 1751, 1776,
            1781, 1821, 1837, 1899, 1963, 1969, 1987, 2014, 2254, 2344, 2356,
            2368, 2369, 2435, 2547, 2654, 2657, 2703, 2723, 2726, 2804, 2816,
            2854, 2871, 2921, 2993, 3029, 3057, 3075, 3104, 3191, 3204, 3278,
            3322, 3363, 3377, 3383, 3402, 3513, 3547, 3553, 3586, 3593, 3618,
            3675, 3760, 3785, 3791, 3833, 3896, 3938, 3957, 3975, 4008, 4034]

    ref2 = [43, 100, 176, 200, 203, 231, 382, 396, 447, 490, 519, 583, 607,
            650, 818, 822, 832, 840, 862, 907, 950, 960, 992, 1007, 1013,
            1019, 1043, 1050, 1059, 1103, 1104, 1113, 1175, 1282, 1293, 1315,
            1377, 1431, 1470, 1512, 1544, 1578, 1588, 1598, 1633, 1647, 1664,
            1723, 1749, 1751, 1758, 1760, 1776, 1781, 1821, 1837, 1880, 1963,
            1969, 1987, 2254, 2356, 2368, 2369, 2435, 2545, 2547, 2654, 2703,
            2723, 2726, 2727, 2800, 2816, 2854, 2871, 2921, 2993, 3029, 3075,
            3191, 3204, 3278, 3291, 3334, 3363, 3383, 3402, 3513, 3547, 3553,
            3586, 3593, 3618, 3641, 3661, 3675, 3760, 3785, 3791, 3806, 3833,
            3857, 3896, 3925, 3957, 3975, 3993, 4008, 4034]

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

    ref1 = [
        46, 111, 305, 310, 362, 384, 410, 451, 467, 548, 572, 595, 607, 609,
        620, 659, 692, 699, 724, 743, 752, 842, 926, 936, 975, 1037, 1072,
        1094, 1136, 1143, 1161, 1172, 1313, 1325, 1368, 1399, 1462, 1486, 1488,
        1493, 1603, 1619, 1648, 1665, 1666, 1838, 1887, 1901, 1948, 1961, 1972,
        1975, 1997, 2001, 2052, 2085, 2095, 2174, 2233, 2236, 2368, 2382, 2383,
        2403, 2484, 2492, 2528, 2593, 2616, 2707, 2790, 2900, 2923, 2946, 2967,
        3103, 3118, 3176, 3190, 3216, 3225, 3297, 3327, 3350, 3374, 3514, 3525,
        3536, 3602, 3620, 3780, 3821, 3898, 3920, 3977, 3982, 4051, 4080, 4092
    ]
    ref2 = [
        46, 111, 143, 173, 259, 305, 362, 410, 451, 467, 507, 518, 548, 583,
        595, 607, 609, 620, 639, 692, 693, 724, 752, 784, 826, 842, 926, 1037,
        1087, 1094, 1098, 1136, 1143, 1161, 1172, 1286, 1325, 1368, 1371, 1395,
        1399, 1462, 1486, 1488, 1493, 1565, 1619, 1648, 1655, 1665, 1887, 1891,
        1901, 1948, 1961, 1968, 1972, 1975, 1977, 1997, 2001, 2007, 2095, 2125,
        2174, 2233, 2236, 2368, 2382, 2383, 2484, 2492, 2571, 2593, 2606, 2639,
        2707, 2790, 2923, 2946, 2967, 2987, 3030, 3101, 3103, 3118, 3228, 3327,
        3351, 3374, 3407, 3419, 3536, 3578, 3620, 3698, 3743, 3821, 3839, 3920,
        3982, 4044, 4051, 4080, 4092
    ]

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
    # splif = SPLIF(mols[0], receptor)
    # shaped_splif = np.hstack((splif['hash'].reshape((len(splif['hash']), 1)),
    # splif['ligand_coords'].reshape((len(splif['ligand_coords']),  15)),
    # splif['protein_coords'].reshape((len(splif['protein_coords']),  15))))
    # np.savetxt('test_splif.csv', np.nan_to_num(reshaped_splif), delimiter=',', fmt='%.3f')
    splif = SPLIF(mols[0], receptor)
    if oddt.toolkit.backend == 'ob':
        reference = np.array([38, 53, 53, 53, 70, 81, 81, 125, 125, 127, 216,
                              219, 242, 249, 262, 262, 279, 279, 330, 396, 423, 423,
                              424, 498, 498, 570, 592, 622, 625, 626, 676, 676, 817,
                              818, 818, 818, 818, 884, 888, 895, 907, 914, 935, 1030,
                              1041, 1082, 1082, 1115, 1143, 1184, 1184, 1191, 1199, 1264, 1269,
                              1275, 1275, 1275, 1283, 1316, 1316, 1316, 1328, 1328, 1344, 1392,
                              1396, 1435, 1466, 1495, 1502, 1502, 1502, 1502, 1506, 1569, 1569,
                              1569, 1569, 1569, 1569, 1617, 1641, 1645, 1697, 1697, 1747, 1796,
                              1938, 1979, 1997, 2000, 2007, 2007, 2021, 2150, 2179, 2196, 2225,
                              2229, 2415, 2417, 2417, 2485, 2510, 2511, 2579, 2579, 2579, 2625,
                              2637, 2666, 2678, 2737, 2776, 2776, 2790, 2857, 2863, 2863, 2895,
                              2924, 2945, 2945, 3035, 3058, 3074, 3074, 3074, 3074, 3080, 3138,
                              3160, 3160, 3167, 3219, 3219, 3279, 3279, 3282, 3339, 3353, 3361,
                              3369, 3388, 3606, 3606, 3610, 3616, 3621, 3637, 3651, 3689, 3689,
                              3714, 3714, 3717, 3717, 3730, 3732, 3770, 3844, 3855, 3872, 3913,
                              3928, 3987, 3995, 3995, 4070, 4088, 4088])
    else:
        reference = np.array([38, 53, 53, 53, 70, 81, 81, 125, 125, 127, 216,
                              219, 242, 249, 262, 262, 279, 279, 330, 396, 423, 423,
                              424, 498, 498, 570, 592, 622, 625, 626, 676, 676, 817,
                              818, 818, 818, 818, 884, 888, 895, 907, 914, 935, 1030,
                              1041, 1082, 1082, 1115, 1143, 1184, 1184, 1191, 1199, 1264, 1269,
                              1275, 1275, 1275, 1283, 1316, 1316, 1316, 1328, 1328, 1344, 1392,
                              1396, 1435, 1466, 1495, 1502, 1502, 1502, 1502, 1506, 1569, 1569,
                              1569, 1569, 1569, 1569, 1617, 1641, 1645, 1697, 1697, 1747, 1796,
                              1938, 1979, 1997, 2000, 2007, 2007, 2021, 2150, 2179, 2196, 2225,
                              2229, 2415, 2417, 2417, 2485, 2510, 2511, 2579, 2579, 2579, 2625,
                              2637, 2666, 2678, 2737, 2776, 2776, 2790, 2857, 2863, 2863, 2895,
                              2924, 2945, 2945, 3035, 3058, 3074, 3074, 3074, 3074, 3080, 3138,
                              3160, 3160, 3167, 3219, 3219, 3279, 3279, 3282, 3339, 3353, 3361,
                              3369, 3388, 3606, 3606, 3610, 3616, 3621, 3637, 3651, 3689, 3689,
                              3714, 3714, 3717, 3717, 3730, 3732, 3770, 3844, 3855, 3872, 3913,
                              3928, 3987, 3995, 3995, 4070, 4088, 4088])
    assert_equal(splif['hash'].shape, (172,))
    assert_array_equal(splif['ligand_coords'].shape, (172, 5, 3))
    assert_array_equal(splif['protein_coords'].shape, (172, 5, 3))
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
