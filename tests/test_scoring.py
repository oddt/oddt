import os
from types import GeneratorType
from tempfile import mkdtemp, NamedTemporaryFile

import numpy as np

from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest
from sklearn.metrics import r2_score

import oddt
from oddt.scoring import scorer, ensemble_descriptor, ensemble_model
from oddt.scoring.descriptors import (autodock_vina_descriptor,
                                      fingerprints,
                                      oddt_vina_descriptor)
from oddt.scoring.models.classifiers import neuralnetwork
from oddt.scoring.models import regressors
from oddt.scoring.functions import rfscore, nnscore, PLECscore

test_data_dir = os.path.dirname(os.path.abspath(__file__))
actives_sdf = os.path.join(test_data_dir, 'data', 'dude', 'xiap',
                           'actives_docked.sdf')
receptor_pdb = os.path.join(test_data_dir, 'data', 'dude', 'xiap',
                            'receptor_rdkit.pdb')
results = os.path.join(test_data_dir, 'data', 'results', 'xiap')


@pytest.mark.filterwarnings('ignore:Data with input dtype int64 was converted')
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
    assert isinstance(scored_mols_gen, GeneratorType)
    gen_predictions = [float(mol.data['score']) for mol in scored_mols_gen]
    assert_array_almost_equal(predictions, gen_predictions)


def test_ensemble_descriptor():
    mols = list(oddt.toolkit.readfile('sdf', actives_sdf))[:10]
    list(map(lambda x: x.addh(), mols))

    rec = next(oddt.toolkit.readfile('pdb', receptor_pdb))
    rec.protein = True
    rec.addh()

    desc1 = rfscore(version=1).descriptor_generator
    desc2 = oddt_vina_descriptor()
    ensemble = ensemble_descriptor((desc1, desc2))

    ensemble.set_protein(rec)
    assert len(ensemble) == len(desc1) + len(desc2)

    # set protein
    assert desc1.protein == rec
    assert desc2.protein == rec

    ensemble_scores = ensemble.build(mols)
    scores1 = desc1.build(mols)
    scores2 = desc2.build(mols)
    assert_array_almost_equal(ensemble_scores, np.hstack((scores1, scores2)))


def test_ensemble_model():
    X = np.vstack((np.arange(30, 10, -2, dtype='float64'),
                   np.arange(100, 90, -1, dtype='float64'))).T

    Y = np.arange(10, dtype='float64')

    rf = regressors.randomforest(random_state=42)
    nn = regressors.neuralnetwork(solver='lbfgs', random_state=42)
    ensemble = ensemble_model((rf, nn))

    # we do not need to fit underlying models, they change when we fit enseble
    ensemble.fit(X, Y)

    pred = ensemble.predict(X)
    mean_pred = np.vstack((rf.predict(X), nn.predict(X))).mean(axis=0)
    assert_array_almost_equal(pred, mean_pred)
    assert_almost_equal(ensemble.score(X, Y), r2_score(Y, pred))

    # ensemble of a single model should behave exactly like this model
    nn = neuralnetwork(solver='lbfgs', random_state=42)
    ensemble = ensemble_model((nn,))
    ensemble.fit(X, Y)
    assert_array_almost_equal(ensemble.predict(X), nn.predict(X))
    assert_almost_equal(ensemble.score(X, Y), nn.score(X, Y))


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


def test_rfscore_desc():
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


def test_nnscore_desc():
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


models = ([PLECscore(n_jobs=1, version=v, size=2048)
           for v in ['linear', 'nn', 'rf']] +
          [nnscore(n_jobs=1)] +
          [rfscore(version=v, n_jobs=1) for v in [1, 2, 3]])


@pytest.mark.parametrize('model', models)
def test_model_train(model):
    mols = list(oddt.toolkit.readfile('sdf', actives_sdf))[:10]
    list(map(lambda x: x.addh(), mols))

    rec = next(oddt.toolkit.readfile('pdb', receptor_pdb))
    rec.protein = True
    rec.addh()

    data_dir = os.path.join(test_data_dir, 'data')
    home_dir = mkdtemp()
    pdbbind_versions = (2007, 2013, 2016)

    pdbbind_dir = os.path.join(data_dir, 'pdbbind')
    for pdbbind_v in pdbbind_versions:
        version_dir = os.path.join(data_dir, 'v%s' % pdbbind_v)
        if not os.path.isdir(version_dir):
            os.symlink(pdbbind_dir, version_dir)

    with NamedTemporaryFile(suffix='.pickle') as f:
        model.gen_training_data(data_dir, pdbbind_versions=pdbbind_versions,
                                home_dir=home_dir)
        model.train(home_dir=home_dir, sf_pickle=f.name)
        model.set_protein(rec)
        # check if protein setting was successful
        assert model.protein == rec
        if hasattr(model.descriptor_generator, 'protein'):
            assert model.descriptor_generator.protein == rec

        preds = model.predict(mols)
        assert len(preds) == 10
        assert preds.dtype == np.float
        assert model.score(mols, preds) == 1.0
