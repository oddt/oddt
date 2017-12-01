import numpy as np

from nose.tools import assert_equal, assert_greater, assert_true
from sklearn.utils.testing import assert_array_almost_equal, assert_array_equal

from oddt.scoring.models import classifiers, regressors


def test_classifiers():
    # toy data
    X = np.concatenate((np.zeros((5, 2)), np.ones((5, 2))))
    Y = np.concatenate((np.ones(5), np.zeros(5)))

    np.random.seed(42)

    for classifier in (classifiers.svm(probability=True),
                       classifiers.neuralnetwork(random_state=42)):
        classifier.fit(X, Y)

        assert_array_equal(classifier.predict(X), Y)
        assert_equal(classifier.score(X, Y), 1.0)

        prob = classifier.predict_proba(X)
        assert_array_almost_equal(prob, [[0, 1]] * 5 + [[1, 0]] * 5, decimal=1)
        log_prob = classifier.predict_log_proba(X)
        assert_array_almost_equal(np.log(prob), log_prob)


def test_regressors():
    X = np.vstack((np.arange(30, 10, -2, dtype='float64'),
                   np.arange(100, 90, -1, dtype='float64'))).T

    Y = np.arange(10, dtype='float64')

    np.random.seed(42)

    for regressor in (regressors.svm(C=10),
                      regressors.randomforest(random_state=42),
                      regressors.neuralnetwork(solver='lbfgs',
                                               random_state=42,
                                               hidden_layer_sizes=(20, 20)),
                      regressors.mlr()):

        regressor.fit(X, Y)

        assert_true((np.abs(regressor.predict(X).flatten() - Y) < 1).all())
        assert_greater(regressor.score(X, Y), 0.9)
