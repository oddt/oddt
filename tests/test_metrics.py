import numpy as np
from numpy.testing import assert_almost_equal

from oddt.metrics import (roc_auc, roc_log_auc, random_roc_log_auc,
                          enrichment_factor, rie, bedroc,
                          rmse, standard_deviation_error)


np.random.seed(42)

# Generate test data for classification
classes = np.array([0] * 90000 + [1] * 10000)
# poorly separated
poor_classes = np.random.rand(100000) * 100

# well separated
good_classes = np.concatenate([np.random.rand(90000) * 10 + 100,
                               np.random.rand(10000) * 10 + 1000])

# Generate test data for regression
values = np.arange(100000)
poor_values = np.random.rand(100000) * 100    # poorly predicted
good_values = np.arange(100000) + np.random.rand(100000)  # correctly predicted


def test_roc_auc():
    score = roc_auc(classes, poor_classes)
    assert score <= 0.55
    assert score >= 0.45

    assert roc_auc(classes, good_classes, ascending_score=True) == 0.0
    assert roc_auc(classes, good_classes, ascending_score=False) == 1.0


def test_roc_log_auc():
    random_score = random_roc_log_auc()
    score = roc_log_auc(classes, poor_classes)
    assert np.abs(score - random_score) < 0.01

    assert roc_log_auc(classes, good_classes, ascending_score=True) == 0
    assert roc_log_auc(classes, good_classes, ascending_score=False) == 1


def test_enrichment():
    order = sorted(range(len(poor_classes)), key=lambda k: poor_classes[k],
                   reverse=True)
    ef = enrichment_factor(classes[order], poor_classes[order])
    assert ef <= 1.5

    order = sorted(range(len(good_classes)), key=lambda k: good_classes[k],
                   reverse=True)
    ef = enrichment_factor(classes[order], good_classes[order])
    assert ef == 10

    ef = enrichment_factor(classes[order], good_classes[order],
                           kind='percentage')
    assert ef == 1


def test_rmse():
    assert rmse(values, poor_values) >= 30
    assert rmse(values, good_values) <= 1


def test_standard_deviation_error():
    assert standard_deviation_error(values, good_values) < 1.1
    assert standard_deviation_error(values, poor_values) > 2e4


def test_rie():
    order = sorted(range(len(poor_classes)), key=lambda k: poor_classes[k],
                   reverse=True)
    rie_score = rie(classes[order], poor_classes[order])
    assert rie_score <= 1.1

    order = sorted(range(len(good_classes)), key=lambda k: good_classes[k],
                   reverse=True)
    rie_score = rie(classes[order], good_classes[order])
    assert_almost_equal(rie_score, 8.646647185)


def test_bedroc():
    order = sorted(range(len(poor_classes)), key=lambda k: poor_classes[k],
                   reverse=True)
    bedroc_score = bedroc(classes[order], poor_classes[order])
    assert bedroc_score < 0.2

    order = sorted(range(len(good_classes)), key=lambda k: good_classes[k],
                   reverse=True)
    bedroc_score = bedroc(classes[order], good_classes[order])
    assert_almost_equal(bedroc_score, 1.0)
