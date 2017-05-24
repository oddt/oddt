import os
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from nose.tools import nottest, assert_in, assert_not_in, assert_equal
from sklearn.utils.testing import (assert_true,
                                   assert_array_equal,
                                   assert_array_almost_equal)

import oddt
from oddt.virtualscreening import virtualscreening

test_data_dir = os.path.dirname(os.path.abspath(__file__))


if oddt.toolkit.backend == 'ob':  # RDKit doesn't write PDBQT
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


if oddt.toolkit.backend == 'ob':  # RDKit doesn't write PDBQT
    def test_vs_docking():
        """VS docking (Vina) tests"""
        vs = virtualscreening(n_cpu=1)
        vs.load_ligands('sdf', os.path.join(test_data_dir, 'data/dude/xiap/crystal_ligand.sdf'))
        vs.dock(engine='autodock_vina',
                protein=os.path.join(test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb'),
                auto_ligand=os.path.join(test_data_dir, 'data/dude/xiap/crystal_ligand.sdf'),
                exhaustiveness=1,
                size=(20, 20, 20),
                seed=0)
        mols = list(vs.fetch())
        assert_equal(len(mols), 7)
        mol_data = mols[0].data
        assert_in('vina_affinity', mol_data)
        assert_in('vina_rmsd_lb', mol_data)
        assert_in('vina_rmsd_ub', mol_data)


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
