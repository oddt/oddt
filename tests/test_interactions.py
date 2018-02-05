import os

from numpy.testing import assert_array_equal
import pytest

import oddt
from oddt.interactions import (close_contacts,
                               hbonds,
                               halogenbonds,
                               pi_stacking,
                               salt_bridges,
                               pi_cation,
                               hydrophobic_contacts)

test_data_dir = os.path.dirname(os.path.abspath(__file__))

mols = list(oddt.toolkit.readfile('sdf', os.path.join(test_data_dir, 'data/dude/xiap/actives_docked.sdf')))
list(map(lambda x: x.addh(only_polar=True), mols))

rec = next(oddt.toolkit.readfile('pdb', os.path.join(test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb')))
rec.protein = True
rec.addh(only_polar=True)


def test_close_contacts():
    """Close contacts test"""
    cc = [len(close_contacts(rec.atom_dict[rec.atom_dict['atomicnum'] != 1],
                             mol.atom_dict[mol.atom_dict['atomicnum'] != 1],
                             cutoff=3)[0]) for mol in mols]
    assert_array_equal(cc,
                       [5, 7, 6, 5, 3, 6, 5, 6, 6, 6, 5, 4, 7, 6, 6, 6, 7, 5,
                        6, 5, 5, 7, 4, 5, 6, 7, 6, 5, 7, 5, 6, 4, 5, 4, 3, 7,
                        6, 6, 3, 5, 4, 3, 1, 7, 3, 2, 4, 1, 2, 7, 4, 4, 6, 4,
                        6, 7, 7, 6, 6, 6, 5, 6, 5, 4, 4, 7, 3, 6, 6, 4, 7, 7,
                        4, 5, 4, 7, 3, 6, 6, 6, 5, 6, 4, 5, 4, 4, 6, 5, 5, 7,
                        6, 2, 6, 5, 1, 8, 6, 5, 7, 4])


@pytest.mark.skip
def test_hbonds():
    """H-Bonds test"""
    hbonds_count = [hbonds(rec, mol)[2].sum() for mol in mols]
    assert_array_equal(hbonds_count,
                       [6, 7, 5, 5, 6, 5, 6, 4, 6, 5, 4, 6, 6, 5, 8, 5, 6, 6,
                        6, 7, 6, 6, 5, 6, 7, 5, 5, 7, 6, 6, 7, 6, 6, 6, 6, 6,
                        6, 5, 5, 6, 4, 5, 5, 6, 6, 3, 5, 5, 4, 6, 4, 8, 6, 6,
                        6, 4, 6, 6, 6, 6, 7, 6, 7, 6, 6, 7, 6, 6, 6, 5, 4, 5,
                        5, 6, 6, 6, 6, 6, 6, 4, 7, 5, 6, 6, 5, 6, 6, 5, 6, 5,
                        6, 5, 5, 7, 7, 6, 8, 6, 4, 5])


@pytest.mark.skip
def test_halogenbonds():
    """Halogen-Bonds test"""
    halogenbonds_count = [len(halogenbonds(rec, mol)[2]) for mol in mols]
    print(halogenbonds_count)
    assert_array_equal(halogenbonds_count,
                       [])


@pytest.mark.skip
def test_pi_stacking():
    """Pi-stacking test"""
    pi_parallel_count = [pi_stacking(rec,
                                     mol,
                                     cutoff=8)[2].sum() for mol in mols]
    print(pi_parallel_count)
    # assert_array_equal(pi_parallel_count,
    #                    [])

    pi_perpendicular_count = [pi_stacking(rec,
                                          mol,
                                          cutoff=8)[3].sum() for mol in mols]
    print(pi_perpendicular_count)
    assert_array_equal(pi_perpendicular_count,
                       [])


@pytest.mark.skip
def test_salt_bridges():
    """Salt bridges test"""
    salt_bridges_count = [len(salt_bridges(rec, mol)[0]) for mol in mols]
    # print(salt_bridges_count)
    assert_array_equal(salt_bridges_count,
                       [6, 7, 5, 5, 6, 5, 6, 4, 6, 5, 4, 6, 6, 5, 8, 5, 6, 6,
                        6, 7, 6, 6, 5, 6, 7, 5, 5, 7, 6, 6, 7, 6, 6, 6, 6, 6,
                        6, 5, 5, 6, 4, 5, 5, 6, 6, 3, 5, 5, 4, 6, 4, 8, 6, 6,
                        6, 4, 6, 6, 6, 6, 7, 6, 7, 6, 6, 7, 6, 6, 6, 5, 4, 5,
                        5, 6, 6, 6, 6, 6, 6, 4, 7, 5, 6, 6, 5, 6, 6, 5, 6, 5,
                        6, 5, 5, 7, 7, 6, 8, 6, 4, 5])


@pytest.mark.skip
def test_pi_cation():
    """Pi-cation test"""
    pi_cation_count = [len(pi_cation(rec, mol)[2]) for mol in mols]
    # print(pi_cation_count)
    assert_array_equal(pi_cation_count,
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 2, 0, 0, 0, 0, 0])

    pi_cation_count = [len(pi_cation(mol, rec)[2]) for mol in mols]
    # print(pi_cation_count)
    assert_array_equal(pi_cation_count,
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        2, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 2, 1, 0,
                        2, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1,
                        0, 1, 0, 0, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0,
                        0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1,
                        0, 2, 0, 0, 0, 0, 1, 1, 0, 1])
    # Strict
    pi_cation_count = [pi_cation(mol, rec)[2].sum() for mol in mols]
    # print(pi_cation_count)
    assert_array_equal(pi_cation_count,
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def test_hyd_contacts():
    """Hydrophobic Contacts test"""
    hyd_contacts_count = [len(hydrophobic_contacts(rec, mol)[0]) for mol in mols]
    assert_array_equal(hyd_contacts_count,
                       [14, 10, 7, 14, 10, 13, 17, 14, 17, 12, 12, 10, 10, 11,
                        9, 8, 8, 4, 9, 16, 15, 6, 9, 8, 5, 5, 8, 11, 7, 10, 7,
                        13, 4, 13, 9, 9, 9, 4, 6, 16, 10, 13, 10, 9, 8, 9, 13,
                        15, 13, 9, 11, 9, 7, 10, 5, 3, 5, 7, 7, 10, 11, 7, 10,
                        20, 9, 6, 6, 3, 7, 7, 4, 7, 6, 2, 5, 6, 14, 9, 4, 6,
                        11, 10, 9, 6, 10, 8, 6, 5, 6, 11, 8, 16, 9, 9, 11, 6,
                        8, 5, 8, 15])
