import os
from types import GeneratorType

import numpy as np

from nose.tools import assert_almost_equal, assert_is_instance
from sklearn.utils.testing import assert_array_almost_equal

import oddt
from oddt.scoring import scorer
from oddt.scoring.descriptors import (autodock_vina_descriptor,
                                      fingerprints,
                                      oddt_vina_descriptor)
from oddt.scoring.models.classifiers import neuralnetwork
from oddt.scoring.functions import rfscore, nnscore

test_data_dir = os.path.dirname(os.path.abspath(__file__))
actives_sdf = os.path.join(test_data_dir, 'data', 'dude', 'xiap',
                           'actives_docked.sdf')
receptor_pdb = os.path.join(test_data_dir, 'data', 'dude', 'xiap',
                            'receptor_rdkit.pdb')
results = os.path.join(test_data_dir, 'data', 'results', 'xiap')


def test_scorer():
    np.random.seed(42)
    # toy example with made up values
    mols = list(oddt.toolkit.readfile('sdf', actives_sdf))

    values = [0]*5 + [1]*5
    test_values = [0, 0, 1, 1, 0]

    if oddt.toolkit.backend == 'ob':
        fp = 'fp2'
    else:
        fp = 'rdkit'

    simple_scorer = scorer(neuralnetwork(), fingerprints(fp))
    simple_scorer.fit(mols[:10], values)
    predictions = simple_scorer.predict(mols[10:15])
    assert_array_almost_equal(predictions, [0, 1, 0, 1, 0])

    score = simple_scorer.score(mols[10:15], test_values)
    assert_almost_equal(score, 0.6)

    scored_mols = [simple_scorer.predict_ligand(mol) for mol in mols[10:15]]
    single_predictions = [float(mol.data['score']) for mol in scored_mols]
    assert_array_almost_equal(predictions, single_predictions)

    scored_mols_gen = simple_scorer.predict_ligands(mols[10:15])
    assert_is_instance(scored_mols_gen, GeneratorType)
    gen_predictions = [float(mol.data['score']) for mol in scored_mols_gen]
    assert_array_almost_equal(predictions, gen_predictions)


if oddt.toolkit.backend == 'ob':  # RDKit doesn't write PDBQT
    def test_original_vina():
        """Check orignal Vina partial scores descriptor"""
        mols = list(oddt.toolkit.readfile('sdf', actives_sdf))
        list(map(lambda x: x.addh(), mols))

        rec = next(oddt.toolkit.readfile('pdb', receptor_pdb))
        rec.protein = True
        rec.addh()

        # Delete molecule which has differences in Acceptor-Donor def in RDK and OB
        del mols[65]

        vina_scores = ['vina_gauss1',
                       'vina_gauss2',
                       'vina_repulsion',
                       'vina_hydrophobic',
                       'vina_hydrogen']

        # save correct results (for future use)
        # np.savetxt(os.path.join(results, 'autodock_vina_scores.csv'),
        #            autodock_vina_descriptor(protein=rec,
        #                                     vina_scores=vina_scores).build(mols),
        #            fmt='%.16g',
        #            delimiter=',')
        autodock_vina_results_correct = np.loadtxt(
            os.path.join(results, 'autodock_vina_scores.csv'),
            delimiter=',',
            dtype=np.float64)
        autodock_vina_results = autodock_vina_descriptor(
            protein=rec,
            vina_scores=vina_scores).build(mols)
        assert_array_almost_equal(autodock_vina_results,
                                  autodock_vina_results_correct,
                                  decimal=4)


def test_internal_vina():
    """Compare internal vs orignal Vina partial scores"""
    mols = list(oddt.toolkit.readfile('sdf', actives_sdf))
    list(map(lambda x: x.addh(), mols))

    rec = next(oddt.toolkit.readfile('pdb', receptor_pdb))
    rec.protein = True
    rec.addh()

    # Delete molecule which has differences in Acceptor-Donor def in RDK and OB
    del mols[65]

    vina_scores = ['vina_gauss1',
                   'vina_gauss2',
                   'vina_repulsion',
                   'vina_hydrophobic',
                   'vina_hydrogen']
    autodock_vina_results = np.loadtxt(
        os.path.join(results, 'autodock_vina_scores.csv'),
        delimiter=',',
        dtype=np.float64)
    oddt_vina_results = oddt_vina_descriptor(
        protein=rec, vina_scores=vina_scores).build(mols)
    assert_array_almost_equal(oddt_vina_results, autodock_vina_results, decimal=4)


def test_rfscore():
    """Test RFScore v1-3 descriptors generators"""
    mols = list(oddt.toolkit.readfile('sdf', actives_sdf))
    list(map(lambda x: x.addh(), mols))

    rec = next(oddt.toolkit.readfile('pdb', receptor_pdb))
    rec.protein = True
    rec.addh()

    # Delete molecule which has differences in Acceptor-Donor def in RDK and OB
    del mols[65]

    for v in [1, 2, 3]:
        descs = rfscore(version=v, protein=rec).descriptor_generator.build(mols)
        # save correct results (for future use)
        # np.savetxt(os.path.join(results, 'rfscore_v%i_descs.csv' % v),
        #            descs,
        #            fmt='%.16g',
        #            delimiter=',')
        descs_correct = np.loadtxt(
            os.path.join(results, 'rfscore_v%i_descs.csv' % v),
            delimiter=',')

        # help debug errors
        for i in range(descs.shape[1]):
            mask = np.abs(descs[:, i] - descs_correct[:, i]) > 1e-4
            if mask.sum() > 1:
                print(i, np.vstack((descs[mask, i], descs_correct[mask, i])))

        assert_array_almost_equal(descs, descs_correct, decimal=4)


def test_nnscore():
    """Test NNScore descriptors generators"""
    mols = list(oddt.toolkit.readfile('sdf', actives_sdf))
    list(map(lambda x: x.addh(only_polar=True), mols))

    rec = next(oddt.toolkit.readfile('pdb', receptor_pdb))
    rec.protein = True
    rec.addh(only_polar=True)

    # Delete molecule which has differences in Acceptor-Donor def in RDK and OB
    del mols[65]

    gen = nnscore(protein=rec).descriptor_generator
    descs = gen.build(mols)
    # save correct results (for future use)
    # np.savetxt(os.path.join(results, 'nnscore_descs.csv'),
    #            descs,
    #            fmt='%.16g',
    #            delimiter=',')
    if oddt.toolkit.backend == 'ob':
        descs_correct = np.loadtxt(os.path.join(results, 'nnscore_descs_ob.csv'),
                                   delimiter=',')
    else:
        descs_correct = np.loadtxt(os.path.join(results, 'nnscore_descs_rdk.csv'),
                                   delimiter=',')

    # help debug errors
    for i in range(descs.shape[1]):
        mask = np.abs(descs[:, i] - descs_correct[:, i]) > 1e-4
        if mask.sum() > 1:
            print(i, gen.titles[i], mask.sum())
            print(np.vstack((descs[mask, i], descs_correct[mask, i])))

    assert_array_almost_equal(descs, descs_correct, decimal=4)
