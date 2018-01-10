import os
from tempfile import mkdtemp, NamedTemporaryFile

from nose.tools import assert_in, assert_equal
from sklearn.utils.testing import (assert_array_equal,
                                   assert_array_almost_equal,
                                   assert_raises,
                                   assert_raises_regexp)

import pandas as pd

import oddt
from oddt.spatial import rmsd
from oddt.scoring import scorer
from oddt.scoring.functions import rfscore, nnscore
from oddt.virtualscreening import virtualscreening

test_data_dir = os.path.dirname(os.path.abspath(__file__))

# common file names
dude_data_dir = os.path.join(test_data_dir, 'data', 'dude', 'xiap')
xiap_crystal_ligand = os.path.join(dude_data_dir, 'crystal_ligand.sdf')
xiap_protein = os.path.join(dude_data_dir, 'receptor_rdkit.pdb')
xiap_actives_docked = os.path.join(dude_data_dir, 'actives_docked.sdf')


def test_vs_scoring_vina():
    """VS scoring (Vina) tests"""
    vs = virtualscreening(n_cpu=1)
    vs.load_ligands('sdf', xiap_crystal_ligand)
    vs.score(function='autodock_vina', protein=xiap_protein)
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
    vs = virtualscreening(n_cpu=1)
    vs.load_ligands('sdf', xiap_crystal_ligand)

    # bad docking engine
    assert_raises(ValueError, vs.dock, 'srina', 'prot.pdb')

    vs.dock(engine='autodock_vina',
            protein=xiap_protein,
            auto_ligand=xiap_crystal_ligand,
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
    if oddt.toolkit.backend == 'ob' and oddt.toolkit.__version__ < '2.4.0':
        vina_scores = [-6.3, -6.0, -6.0, -5.9, -5.9, -5.8, -5.2, -4.2, -3.9]
    else:
        vina_scores = [-6.3, -6.0, -5.1, -3.9, -3.5, -3.5, -3.5, -3.3, -2.5]

    # TODO: Fix problematic test
    # assert_array_equal([float(m.data['vina_affinity']) for m in mols], vina_scores)

    # verify the SMILES of molecules
    ref_mol = next(oddt.toolkit.readfile('sdf', xiap_crystal_ligand))

    if oddt.toolkit.backend == 'ob' and oddt.toolkit.__version__ < '2.4.0':
        # OB 2.3.2 will fail the following, since Hs are removed, etc.
        pass
    else:
        vina_rmsd = [8.247347, 5.316951, 7.964107, 7.445350, 8.127984, 7.465065,
                     8.486132, 7.943340, 7.762220]
        assert_array_equal([mol.smiles for mol in mols],
                           [ref_mol.smiles] * len(mols))

        # TODO: Fix problematic test
        # assert_array_almost_equal([rmsd(ref_mol, mol, method='min_symmetry')
        #                            for mol in mols], vina_rmsd)


def test_vs_docking_empty():
    vs = virtualscreening(n_cpu=1)
    vs.load_ligands('smi', os.path.join(dude_data_dir, 'actives_rdkit.smi'))

    vs.dock(engine='autodock_vina',
            protein=xiap_protein,
            auto_ligand=xiap_crystal_ligand,
            exhaustiveness=1,
            energy_range=5,
            num_modes=9,
            size=(20, 20, 20),
            seed=0)

    assert_raises_regexp(ValueError, 'has no 3D coordinates',
                         next, vs.fetch())


if oddt.toolkit.backend == 'ob':  # RDKit rewrite needed
    def test_vs_filtering():
        """VS preset filtering tests"""
        vs = virtualscreening(n_cpu=-1)

        vs.load_ligands('sdf', xiap_actives_docked)
        vs.apply_filter('ro5', soft_fail=1)
        assert_equal(len(list(vs.fetch())), 49)

        vs.load_ligands('sdf', xiap_actives_docked)
        vs.apply_filter('ro3', soft_fail=2)
        assert_equal(len(list(vs.fetch())), 9)


def test_vs_pains():
    """VS PAINS filter tests"""
    vs = virtualscreening(n_cpu=-1)
    # TODO: add some failing molecules
    vs.load_ligands('sdf', xiap_actives_docked)
    vs.apply_filter('pains', soft_fail=0)
    assert_equal(len(list(vs.fetch())), 100)


def test_vs_similarity():
    """VS similarity filter (USRs, IFPs) tests"""
    ref_mol = next(oddt.toolkit.readfile('sdf', xiap_crystal_ligand))
    receptor = next(oddt.toolkit.readfile('pdb', xiap_protein))

    # following toolkit differences is due to different Hs treatment
    vs = virtualscreening(n_cpu=-1)
    vs.load_ligands('sdf', xiap_actives_docked)
    vs.similarity('usr', cutoff=0.4, query=ref_mol)
    if oddt.toolkit.backend == 'ob':
        assert_equal(len(list(vs.fetch())), 11)
    else:
        assert_equal(len(list(vs.fetch())), 6)

    vs = virtualscreening(n_cpu=-1)
    vs.load_ligands('sdf', xiap_actives_docked)
    vs.similarity('usr_cat', cutoff=0.3, query=ref_mol)
    if oddt.toolkit.backend == 'ob':
        assert_equal(len(list(vs.fetch())), 16)
    else:
        assert_equal(len(list(vs.fetch())), 11)

    vs = virtualscreening(n_cpu=-1)
    vs.load_ligands('sdf', xiap_actives_docked)
    vs.similarity('electroshape', cutoff=0.45, query=ref_mol)
    if oddt.toolkit.backend == 'ob':
        assert_equal(len(list(vs.fetch())), 55)
    else:
        assert_equal(len(list(vs.fetch())), 89)

    vs = virtualscreening(n_cpu=1)
    vs.load_ligands('sdf', xiap_actives_docked)
    vs.similarity('ifp', cutoff=0.95, query=ref_mol, protein=receptor)
    if oddt.toolkit.backend == 'ob':
        assert_equal(len(list(vs.fetch())), 3)
    else:
        assert_equal(len(list(vs.fetch())), 6)

    vs = virtualscreening(n_cpu=1)
    vs.load_ligands('sdf', xiap_actives_docked)
    vs.similarity('sifp', cutoff=0.9, query=ref_mol, protein=receptor)
    if oddt.toolkit.backend == 'ob':
        assert_equal(len(list(vs.fetch())), 14)
    else:
        assert_equal(len(list(vs.fetch())), 21)

    # test wrong method error
    assert_raises(ValueError, vs.similarity, 'sift', query=ref_mol)


def test_vs_scoring():
    protein = next(oddt.toolkit.readfile('pdb', xiap_protein))
    protein.protein = True

    data_dir = os.path.join(test_data_dir, 'data')
    home_dir = mkdtemp()
    pdbbind_versions = (2007, 2013, 2016)

    pdbbind_dir = os.path.join(data_dir, 'pdbbind')
    for pdbbind_v in pdbbind_versions:
        version_dir = os.path.join(data_dir, 'v%s' % pdbbind_v)
        if not os.path.isdir(version_dir):
            os.symlink(pdbbind_dir, version_dir)

    filenames = []
    # train mocked SFs
    for model in [nnscore(n_jobs=1)] + [rfscore(version=v, n_jobs=1)
                                        for v in [1, 2, 3]]:
            model.gen_training_data(data_dir, pdbbind_versions=pdbbind_versions,
                                    home_dir=home_dir)
            filenames.append(model.train(home_dir=home_dir))
    vs = virtualscreening(n_cpu=1)
    vs.load_ligands('sdf', xiap_actives_docked)
    # error if no protein is fed
    assert_raises(ValueError, vs.score, 'nnscore')
    # bad sf name
    assert_raises(ValueError, vs.score, 'bad_sf', protein=protein)
    vs.score('nnscore', protein=xiap_protein)
    vs.score('nnscore_pdbbind2016', protein=protein)
    vs.score('rfscore_v1', protein=protein)
    vs.score('rfscore_v1_pdbbind2016', protein=protein)
    vs.score('rfscore_v2', protein=protein)
    vs.score('rfscore_v3', protein=protein)
    # use pickle directly
    vs.score(filenames[0], protein=protein)
    # pass SF object directly
    vs.score(scorer.load(filenames[0]), protein=protein)
    # pass wrong object (sum is not an instance of scorer)
    assert_raises(ValueError, vs.score, sum, protein=protein)

    mols = list(vs.fetch())

    assert_equal(len(mols), 100)
    mol_data = mols[0].data
    assert_in('nnscore', mol_data)
    assert_in('rfscore_v1', mol_data)
    assert_in('rfscore_v2', mol_data)
    assert_in('rfscore_v3', mol_data)

    vs = virtualscreening(n_cpu=1)
    vs.load_ligands('sdf', xiap_actives_docked)
    vs.score('nnscore', protein=protein)
    vs.score('rfscore_v1', protein=protein)
    vs.score('rfscore_v2', protein=protein)
    vs.score('rfscore_v3', protein=protein)
    with NamedTemporaryFile('w', suffix='.sdf') as molfile:
        with NamedTemporaryFile('w', suffix='.csv') as csvfile:
            vs.write('sdf', molfile.name, csv_filename=csvfile.name)
            data = pd.read_csv(csvfile.name)
            assert_in('nnscore', data.columns)
            assert_in('rfscore_v1', data.columns)
            assert_in('rfscore_v2', data.columns)
            assert_in('rfscore_v3', data.columns)

            mols = list(oddt.toolkit.readfile('sdf', molfile.name))
            assert_equal(len(mols), 100)

            vs.write_csv(csvfile.name, fields=['nnscore', 'rfscore_v1',
                                               'rfscore_v2', 'rfscore_v3'])
            data = pd.read_csv(csvfile.name)
            assert_equal(len(data.columns), 4)
            assert_in('nnscore', data.columns)
            assert_in('rfscore_v1', data.columns)
            assert_in('rfscore_v2', data.columns)
            assert_in('rfscore_v3', data.columns)

    # remove files
    for f in filenames:
        os.unlink(f)

    # remove symlinks
    for pdbbind_v in pdbbind_versions:
        version_dir = os.path.join(data_dir, 'v%s' % pdbbind_v)
        if os.path.islink(version_dir):
            os.unlink(version_dir)
