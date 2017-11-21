import os
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from nose.tools import nottest, assert_in, assert_not_in, assert_equal
from sklearn.utils.testing import (assert_true,
                                   assert_array_equal,
                                   assert_array_almost_equal)

import oddt
from oddt.spatial import rmsd
from oddt.virtualscreening import virtualscreening

test_data_dir = os.path.dirname(os.path.abspath(__file__))


def test_vs_scoring_vina():
    """VS scoring (Vina) tests"""
    vs = virtualscreening(n_cpu=1)
    vs.load_ligands('sdf', os.path.join(test_data_dir, 'data/dude/xiap/crystal_ligand.sdf'))
    vs.score(function='autodock_vina',
             protein=os.path.join(test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb'))
    mols = list(vs.fetch())
    assert_equal(len(mols), 1)
    mol_data = mols[0].data
    assert_in('vina_affinity', mol_data)
    assert_in('vina_gauss1', mol_data)
    assert_in('vina_gauss2', mol_data)
    assert_in('vina_hydrogen', mol_data)
    assert_in('vina_hydrophobic', mol_data)
    assert_in('vina_repulsion', mol_data)
    assert_equal(mol_data['vina_affinity'], '-3.57594')
    assert_equal(mol_data['vina_gauss1'], '63.01213')
    assert_equal(mol_data['vina_gauss2'], '999.07625')
    assert_equal(mol_data['vina_hydrogen'], '0.0')
    assert_equal(mol_data['vina_hydrophobic'], '26.12648')
    assert_equal(mol_data['vina_repulsion'], '3.63178')


def test_vs_docking():
    """VS docking (Vina) tests"""
    ref_mol = next(oddt.toolkit.readfile('sdf',
                                         os.path.join(test_data_dir,
                                                      'data/dude/xiap/crystal_ligand.sdf')))

    vs = virtualscreening(n_cpu=1)
    vs.load_ligands('sdf', os.path.join(test_data_dir, 'data/dude/xiap/crystal_ligand.sdf'))
    vs.dock(engine='autodock_vina',
            protein=os.path.join(test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb'),
            auto_ligand=os.path.join(test_data_dir, 'data/dude/xiap/crystal_ligand.sdf'),
            exhaustiveness=1,
            energy_range=5,
            num_modes=9,
            size=(20, 20, 20),
            seed=0)
    mols = list(vs.fetch())
    assert_equal(len(mols), 9)
    mol_data = mols[0].data
    assert_in('vina_affinity', mol_data)
    assert_in('vina_rmsd_lb', mol_data)
    assert_in('vina_rmsd_ub', mol_data)
    if oddt.toolkit.backend == 'ob':  # RDKit rewrite needed
        vina_scores = [-6.3, -6.0, -6.0, -5.9, -5.9, -5.8, -5.2, -4.2, -3.9]
    else:
        vina_scores = [-6.3, -6.0, -5.1, -3.9, -3.5, -3.5, -3.5, -3.3, -2.5]
    assert_array_equal([float(m.data['vina_affinity']) for m in mols], vina_scores)
    # assert_array_equal([mol.smiles for mol in mols], [ref_mol.smiles] * len(mols))
    # if oddt.toolkit.backend == 'ob':
    #     vina_rmsd = []
    # else:
    #     vina_rmsd = [4.572302, 4.381617, 5.924265, 6.800329, 6.516896, 6.219442,
    #                  6.690042,  7.074162,  7.505027]
    # assert_array_almost_equal([rmsd(ref_mol, mol) for mol in mols],
    #                           vina_rmsd)


if oddt.toolkit.backend == 'ob':  # RDKit rewrite needed
    def test_vs_filtering():
        """VS preset filtering tests"""
        vs = virtualscreening(n_cpu=-1)

        vs.load_ligands('sdf', os.path.join(test_data_dir, 'data/dude/xiap/actives_docked.sdf'))
        vs.apply_filter('ro5', soft_fail=1)
        assert_equal(len(list(vs.fetch())), 49)

        vs.load_ligands('sdf', os.path.join(test_data_dir, 'data/dude/xiap/actives_docked.sdf'))
        vs.apply_filter('ro3', soft_fail=2)
        assert_equal(len(list(vs.fetch())), 9)


def test_vs_pains():
    """VS PAINS filter tests"""
    vs = virtualscreening(n_cpu=-1)
    # TODO: add some failing molecules
    vs.load_ligands('sdf', os.path.join(test_data_dir, 'data/dude/xiap/actives_docked.sdf'))
    vs.apply_filter('pains', soft_fail=0)
    assert_equal(len(list(vs.fetch())), 100)
