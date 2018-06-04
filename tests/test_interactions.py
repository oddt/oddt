import os

from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest

import oddt
from oddt.interactions import (close_contacts,
                               hbonds,
                               distance,
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


def test_pi_stacking():
    """Pi-stacking test"""
    lig = next(oddt.toolkit.readfile('sdf', os.path.join(test_data_dir, 'data',
                                                         'pdbbind', '10gs',
                                                         '10gs_ligand.sdf')))
    rec = next(oddt.toolkit.readfile('pdb', os.path.join(test_data_dir, 'data',
                                                         'pdbbind', '10gs',
                                                         '10gs_pocket.pdb')))
    rec.protein = True
    ring, _, strict_parallel, strict_perpendicular = pi_stacking(rec, lig, cutoff=7.5, tolerance=60)

    lig_centroids = [[5.4701666, 6.1994996, 30.8313350],
                     [8.1811666, 2.5846664, 28.4028320]]

    lig_vectors = [[0.471341, 0.023758, 0.857421],
                   [0.772514, 0.518052, 1.354878]]

    rec_centroids = [[5.6579995, 2.2964999, 23.4626674],
                     [7.8634004, 7.7310004, 34.8283996],
                     [9.8471670, 8.5676660, 34.9915008],
                     [9.9951667, 3.7756664, 32.8191680],
                     [10.055333, -1.4720000, 17.2121658],
                     [14.519165, 1.8759999, 29.8346652],
                     [16.490833, 16.873500, 27.9169998],
                     [18.718666, 12.703166, 33.3141670],
                     [25.716165, 4.9741668, 31.8198337]]
    rec_vectors = [[0.197429, -0.044215, 0.937011],
                   [0.271271, 0.082978, 1.345218],
                   [0.335873, 0.659132, 1.307615],
                   [0.531174, 0.010709, -0.066511],
                   [0.535675, 0.512232, -0.519266],
                   [0.788169, 0.233635, -0.698541],
                   [1.097706, 0.017989, 1.071040],
                   [1.147590, 0.122895, 0.798543],
                   [1.347235, 0.516426, -0.461548]]

    centroids_dist = [[8.3406204, 5.5546951],
                      [4.9040379, 8.2385464],
                      [6.4863953, 9.0544131],
                      [5.5047319, 4.9206809],
                      [16.2897951, 12.0498984],
                      [10.0782127, 6.5362510],
                      [15.6167449, 16.5365460],
                      [14.9661240, 15.4124670],
                      [20.3071175, 18.0239224]]

    assert_array_almost_equal(distance(rec_centroids, lig_centroids), centroids_dist)

    assert len(lig.ring_dict) == 2
    assert_array_almost_equal(sorted(lig.ring_dict['centroid'].tolist()), lig_centroids, decimal=5)
    assert_array_almost_equal(sorted(lig.ring_dict['vector'].tolist()), lig_vectors, decimal=5)
    assert len(rec.ring_dict) == 9
    assert_array_almost_equal(sorted(rec.ring_dict['centroid'].tolist()), rec_centroids, decimal=5)
    assert_array_almost_equal(sorted(rec.ring_dict['vector'].tolist()), rec_vectors, decimal=5)

    assert len(ring) == 6
    assert strict_parallel.sum() == 3
    assert strict_perpendicular.sum() == 0
    resids = sorted(ring['resid'])
    # assert_array_equal(resids, [1, 2, 2, 17, 17, 58])
    # re-check indexing of residues
    assert_array_equal(rec.res_dict[resids]['resnum'], [7, 8, 8, 38, 38, 108])
    assert_array_equal(sorted(ring['resnum']), [7, 8, 8, 38, 38, 108])
    assert_array_equal(sorted(ring['resname']), ['PHE', 'PHE', 'TRP', 'TRP', 'TYR', 'TYR'])


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
