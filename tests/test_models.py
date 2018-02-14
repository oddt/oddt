import pickle
import numpy as np

from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest

from oddt.scoring.models import classifiers, regressors


@pytest.mark.filterwarnings('ignore:Stochastic Optimizer')
@pytest.mark.parametrize('cls',
                         [classifiers.svm(probability=True),
                          classifiers.neuralnetwork(random_state=42)])
def test_classifiers(cls):
    # toy data
    X = np.concatenate((np.zeros((5, 2)), np.ones((5, 2))))
    Y = np.concatenate((np.ones(5), np.zeros(5)))

    np.random.seed(42)

    cls.fit(X, Y)

    assert_array_equal(cls.predict(X), Y)
    assert cls.score(X, Y) == 1.0

    prob = cls.predict_proba(X)
    assert_array_almost_equal(prob, [[0, 1]] * 5 + [[1, 0]] * 5, decimal=1)
    log_prob = cls.predict_log_proba(X)
    assert_array_almost_equal(np.log(prob), log_prob)

    pickled = pickle.dumps(cls)
    reloaded = pickle.loads(pickled)
    prob_reloaded = reloaded.predict_proba(X)
    assert_array_almost_equal(prob, prob_reloaded)


@pytest.mark.parametrize('reg',
                         [regressors.svm(C=10),
                          regressors.randomforest(random_state=42),
                          regressors.neuralnetwork(solver='lbfgs',
                                                   random_state=42,
                                                   hidden_layer_sizes=(20, 20)),
                          regressors.mlr()])
def test_regressors(reg):
    X = np.vstack((np.arange(30, 10, -2, dtype='float64'),
                   np.arange(100, 90, -1, dtype='float64'))).T

    Y = np.arange(10, dtype='float64')

    np.random.seed(42)

    reg.fit(X, Y)

    pred = reg.predict(X)
    assert (np.abs(pred.flatten() - Y) < 1).all()
    assert reg.score(X, Y) > 0.9

    pickled = pickle.dumps(reg)
    reloaded = pickle.loads(pickled)
    pred_reloaded = reloaded.predict(X)
    assert_array_almost_equal(pred, pred_reloaded)
