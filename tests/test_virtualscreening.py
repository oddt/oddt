import os
from tempfile import mkdtemp, NamedTemporaryFile

import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

import pandas as pd

import oddt
from oddt.utils import method_caller
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
    assert len(mols) == 1
    mol_data = mols[0].data
    assert 'vina_affinity' in mol_data
    assert 'vina_gauss1' in mol_data
    assert 'vina_gauss2' in mol_data
    assert 'vina_hydrogen' in mol_data
    assert 'vina_hydrophobic' in mol_data
    assert 'vina_repulsion' in mol_data
    assert mol_data['vina_affinity'] == '-3.57594'
    assert mol_data['vina_gauss1'] == '63.01213'
    assert mol_data['vina_gauss2'] == '999.07625'
    assert mol_data['vina_hydrogen'] == '0.0'
    assert mol_data['vina_hydrophobic'] == '26.12648'
    assert mol_data['vina_repulsion'] == '3.63178'


def test_vs_docking():
    """VS docking (Vina) tests"""
    vs = virtualscreening(n_cpu=1)
    vs.load_ligands('sdf', xiap_crystal_ligand)

    # bad docking engine
    with pytest.raises(ValueError):
        vs.dock('srina', 'prot.pdb')

    vs.dock(engine='autodock_vina',
            protein=xiap_protein,
            auto_ligand=xiap_crystal_ligand,
            exhaustiveness=1,
            energy_range=6,
            num_modes=7,
            size=(20, 20, 20),
            seed=0)
    mols = list(vs.fetch())
    assert len(mols) == 7
    mol_data = mols[0].data
    assert 'vina_affinity' in mol_data
    assert 'vina_rmsd_lb' in mol_data
    assert 'vina_rmsd_ub' in mol_data
    if oddt.toolkit.backend == 'ob':
        vina_scores = [-5.3, -4.0, -3.8, -3.7, -3.4, -3.4, -3.0]
    else:
        vina_scores = [-6.3, -6.0, -5.8, -5.8, -3.9, -3.0, -1.1]
    assert_array_equal([float(m.data['vina_affinity']) for m in mols], vina_scores)

    # verify the SMILES of molecules
    ref_mol = next(oddt.toolkit.readfile('sdf', xiap_crystal_ligand))

    if oddt.toolkit.backend == 'ob':
        # OB 2.3.2 will fail the following, since Hs are removed, etc.
        # OB 2.4 recognizes the smiles chirality wrong
        pass
    else:
        vina_rmsd = [8.153314, 5.32554, 8.514586, 8.510169, 9.060128, 8.995098,
                     8.626776]
        assert_array_equal([mol.smiles for mol in mols],
                           [ref_mol.smiles] * len(mols))

        assert_array_almost_equal([rmsd(ref_mol, mol, method='min_symmetry')
                                   for mol in mols], vina_rmsd)


def test_vs_empty():
    vs = virtualscreening(n_cpu=1)
    with pytest.raises(StopIteration, match='no molecules loaded'):
        vs.fetch()


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

    with pytest.raises(ValueError, match='has no 3D coordinates'):
        next(vs.fetch())


def test_vs_multithreading_fallback():
    vs = virtualscreening(n_cpu=8)
    vs.load_ligands('sdf', xiap_crystal_ligand)

    vs.score(function='autodock_vina', protein=xiap_protein)

    with pytest.warns(UserWarning, match='Falling back to sub-methods multithreading'):
        method_caller(vs, 'fetch')


if oddt.toolkit.backend == 'ob':  # RDKit rewrite needed
    def test_vs_filtering():
        """VS preset filtering tests"""
        vs = virtualscreening(n_cpu=1)

        vs.load_ligands('sdf', xiap_actives_docked)
        vs.apply_filter('ro5', soft_fail=1)
        assert len(list(vs.fetch())) == 49

        vs.load_ligands('sdf', xiap_actives_docked)
        vs.apply_filter('ro3', soft_fail=2)
        assert len(list(vs.fetch())) == 9


def test_vs_pains():
    """VS PAINS filter tests"""
    vs = virtualscreening(n_cpu=1)
    # TODO: add some failing molecules
    vs.load_ligands('sdf', xiap_actives_docked)
    vs.apply_filter('pains', soft_fail=0)
    assert len(list(vs.fetch())) == 100


def test_vs_similarity():
    """VS similarity filter (USRs, IFPs) tests"""
    ref_mol = next(oddt.toolkit.readfile('sdf', xiap_crystal_ligand))
    receptor = next(oddt.toolkit.readfile('pdb', xiap_protein))

    # following toolkit differences is due to different Hs treatment
    vs = virtualscreening(n_cpu=1, chunksize=10)
    vs.load_ligands('sdf', xiap_actives_docked)
    vs.similarity('usr', cutoff=0.4, query=ref_mol)
    if oddt.toolkit.backend == 'ob':
        assert len(list(vs.fetch())) == 11
    else:
        assert len(list(vs.fetch())) == 6

    vs = virtualscreening(n_cpu=1)
    vs.load_ligands('sdf', xiap_actives_docked)
    vs.similarity('usr_cat', cutoff=0.3, query=ref_mol)
    if oddt.toolkit.backend == 'ob':
        assert len(list(vs.fetch())) == 16
    else:
        assert len(list(vs.fetch())) == 11

    vs = virtualscreening(n_cpu=1)
    vs.load_ligands('sdf', xiap_actives_docked)
    vs.similarity('electroshape', cutoff=0.45, query=ref_mol)
    if oddt.toolkit.backend == 'ob':
        assert len(list(vs.fetch())) == 55
    else:
        assert len(list(vs.fetch())) == 95

    vs = virtualscreening(n_cpu=1)
    vs.load_ligands('sdf', xiap_actives_docked)
    vs.similarity('ifp', cutoff=0.95, query=ref_mol, protein=receptor)
    if oddt.toolkit.backend == 'ob':
        assert len(list(vs.fetch())) == 3
    else:
        assert len(list(vs.fetch())) == 6

    vs = virtualscreening(n_cpu=1)
    vs.load_ligands('sdf', xiap_actives_docked)
    vs.similarity('sifp', cutoff=0.9, query=ref_mol, protein=receptor)
    if oddt.toolkit.backend == 'ob':
        assert len(list(vs.fetch())) == 14
    else:
        assert len(list(vs.fetch())) == 21

    # test wrong method error
    with pytest.raises(ValueError):
        vs.similarity('sift', query=ref_mol)


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
    vs = virtualscreening(n_cpu=-1, chunksize=10)
    vs.load_ligands('sdf', xiap_actives_docked)
    # error if no protein is fed
    with pytest.raises(ValueError):
        vs.score('nnscore')
    # bad sf name
    with pytest.raises(ValueError):
        vs.score('bad_sf', protein=protein)
    vs.score('nnscore', protein=xiap_protein)
    vs.score('nnscore_pdbbind2016', protein=protein)
    vs.score('rfscore_v1', protein=protein)
    vs.score('rfscore_v1_pdbbind2016', protein=protein)
    vs.score('rfscore_v2', protein=protein)
    vs.score('rfscore_v3', protein=protein)
    vs.score('pleclinear', protein=protein)
    vs.score('pleclinear_p5_l1_s65536_pdbbind2016', protein=protein)
    # use pickle directly
    vs.score(filenames[0], protein=protein)
    # pass SF object directly
    vs.score(scorer.load(filenames[0]), protein=protein)
    # pass wrong object (sum is not an instance of scorer)
    with pytest.raises(ValueError):
        vs.score(sum, protein=protein)

    mols = list(vs.fetch())

    assert len(mols) == 100
    mol_data = mols[0].data
    assert 'nnscore' in mol_data
    assert 'rfscore_v1' in mol_data
    assert 'rfscore_v2' in mol_data
    assert 'rfscore_v3' in mol_data
    assert 'PLEClinear_p5_l1_s65536' in mol_data

    vs = virtualscreening(n_cpu=-1, chunksize=10)
    vs.load_ligands('sdf', xiap_actives_docked)
    vs.score('nnscore', protein=protein)
    vs.score('rfscore_v1', protein=protein)
    vs.score('rfscore_v2', protein=protein)
    vs.score('rfscore_v3', protein=protein)
    with NamedTemporaryFile('w', suffix='.sdf') as molfile:
        with NamedTemporaryFile('w', suffix='.csv') as csvfile:
            vs.write('sdf', molfile.name, csv_filename=csvfile.name)
            data = pd.read_csv(csvfile.name)
            assert 'nnscore' in data.columns
            assert 'rfscore_v1' in data.columns
            assert 'rfscore_v2' in data.columns
            assert 'rfscore_v3' in data.columns

            mols = list(oddt.toolkit.readfile('sdf', molfile.name))
            assert len(mols) == 100

            vs.write_csv(csvfile.name, fields=['nnscore', 'rfscore_v1',
                                               'rfscore_v2', 'rfscore_v3'])
            data = pd.read_csv(csvfile.name)
            assert len(data.columns) == 4
            assert len(data) == len(mols)
            assert 'nnscore' in data.columns
            assert 'rfscore_v1' in data.columns
            assert 'rfscore_v2' in data.columns
            assert 'rfscore_v3' in data.columns

    # remove files
    for f in filenames:
        os.unlink(f)

    # remove symlinks
    for pdbbind_v in pdbbind_versions:
        version_dir = os.path.join(data_dir, 'v%s' % pdbbind_v)
        if os.path.islink(version_dir):
            os.unlink(version_dir)
