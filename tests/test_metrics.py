from decimal import Decimal, localcontext

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from oddt.metrics import (
    roc_auc,
    roc_log_auc,
    random_roc_log_auc,
    enrichment_factor,
    rie,
    bedroc,
    rmse,
    standard_deviation_error,
)

np.random.seed(42)

# Generate test data for classification
classes = np.array([0] * 90000 + [1] * 10000)
# poorly separated
poor_classes = np.random.rand(100000) * 100

# well separated
good_classes = np.concatenate([np.random.rand(90000) * 10 + 100, np.random.rand(10000) * 10 + 1000])

# Generate test data for regression
values = np.arange(100000)
poor_values = np.random.rand(100000) * 100  # poorly predicted
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
    order = sorted(range(len(poor_classes)), key=lambda k: poor_classes[k], reverse=True)
    ef = enrichment_factor(classes[order], poor_classes[order])
    assert ef <= 1.5

    order = sorted(range(len(good_classes)), key=lambda k: good_classes[k], reverse=True)
    ef = enrichment_factor(classes[order], good_classes[order])
    assert ef == 10

    ef = enrichment_factor(classes[order], good_classes[order], kind="percentage")
    assert ef == 1


def test_rmse():
    assert rmse(values, poor_values) >= 30
    assert rmse(values, good_values) <= 1


def test_standard_deviation_error():
    assert standard_deviation_error(values, good_values) < 1.1
    assert standard_deviation_error(values, poor_values) > 2e4


def test_rie():
    order = sorted(range(len(poor_classes)), key=lambda k: poor_classes[k], reverse=True)
    rie_score = rie(classes[order], poor_classes[order])
    assert rie_score <= 1.1

    order = sorted(range(len(good_classes)), key=lambda k: good_classes[k], reverse=True)
    rie_score = rie(classes[order], good_classes[order])
    assert_almost_equal(rie_score, 8.646647185)


def test_bedroc():
    order = sorted(range(len(poor_classes)), key=lambda k: poor_classes[k], reverse=True)
    bedroc_score = bedroc(classes[order], poor_classes[order])
    assert bedroc_score < 0.2

    order = sorted(range(len(good_classes)), key=lambda k: good_classes[k], reverse=True)
    bedroc_score = bedroc(classes[order], good_classes[order])
    assert_almost_equal(bedroc_score, 1.0)


@pytest.mark.parametrize("metric", [roc_auc, roc_log_auc])
@pytest.mark.parametrize("dtype", [np.float64, np.uint8, np.uint64, np.int8, np.int64, bool])
@pytest.mark.parametrize("ascending_score", [False, True])
def test_roc_score_dtypes(metric, dtype, ascending_score):
    labels = np.array([0, 1])
    scores = np.array([1, 0], dtype=dtype)
    assert metric(labels, scores, ascending_score=ascending_score) == float(ascending_score)


@pytest.mark.parametrize("metric", [roc_auc, roc_log_auc])
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("ascending_score", [False, True])
def test_roc_extreme_integer_scores(metric, dtype, ascending_score):
    limits = np.iinfo(dtype)
    for scores in ([limits.min, limits.max], [limits.max - 1, limits.max]):
        assert metric([0, 1], np.array(scores, dtype=dtype), ascending_score=ascending_score) == float(
            not ascending_score
        )


@pytest.mark.parametrize("metric", [roc_auc, roc_log_auc])
@pytest.mark.parametrize("ascending_score", [False, True])
def test_roc_list_scores(metric, ascending_score):
    assert metric([0, 1], [1, 0], ascending_score=ascending_score) == float(ascending_score)


@pytest.mark.parametrize("metric", [enrichment_factor, rie, bedroc])
@pytest.mark.parametrize("pos_label", [None, 7, "active"])
def test_ranked_metric_list_inputs(metric, pos_label):
    positive = 1 if pos_label is None else pos_label
    negative = "inactive" if pos_label == "active" else 0
    labels = [positive, negative, positive, negative]
    scores = [4, 3, 2, 1]
    expected = metric(np.array(labels), np.array(scores), pos_label=pos_label)
    assert metric(labels, scores, pos_label=pos_label) == pytest.approx(expected)


@pytest.mark.parametrize("metric", [enrichment_factor, rie, bedroc])
@pytest.mark.parametrize(
    "labels, scores",
    [([], []), ([1, 0], [4]), ([[1], [0]], [4, 3]), (1, [4]), ([1, 0], [[4], [3]])],
)
def test_ranked_metric_invalid_inputs(metric, labels, scores):
    with pytest.raises(ValueError):
        metric(labels, scores)


@pytest.mark.parametrize("percentage", [0, -1, -50, 100.1, 200, np.nan, np.inf, -np.inf])
def test_enrichment_invalid_percentage(percentage):
    with pytest.raises(ValueError, match="percentage"):
        enrichment_factor([1, 1, 1, 1], [4, 3, 2, 1], percentage=percentage)


@pytest.mark.parametrize("kind", ["folds", "", None])
def test_enrichment_invalid_kind(kind):
    with pytest.raises(ValueError, match="kind"):
        enrichment_factor([1, 1, 0, 0], [4, 3, 2, 1], percentage=50, kind=kind)


@pytest.mark.parametrize("percentage", [0.1, 50, 100])
@pytest.mark.parametrize("kind", ["fold", "percentage"])
def test_enrichment_valid_percentage(percentage, kind):
    assert enrichment_factor([1, 1, 1, 1], [4, 3, 2, 1], percentage=percentage, kind=kind) == 1.0


def test_enrichment_no_positive_labels():
    with pytest.raises(ValueError, match="positive labels"):
        enrichment_factor([0, 0], [2, 1])


@pytest.mark.parametrize("log_min, log_max", [(0.001, 1.0), (0.001, 0.1), (0.01, 0.1), (0.1, 0.5)])
def test_roc_log_auc_partial_perfect_and_reversed(log_min, log_max):
    assert roc_log_auc([1, 0], [2, 1], ascending_score=False, log_min=log_min, log_max=log_max) == 1.0
    assert roc_log_auc([1, 0], [2, 1], ascending_score=True, log_min=log_min, log_max=log_max) == 0.0


@pytest.mark.parametrize("log_min, log_max, expected", [(0.25, 0.5, 0.5), (0.5, 1.0, 1.0), (0.25, 0.4, 0.5)])
def test_roc_log_auc_vertical_boundary(log_min, log_max, expected):
    result = roc_log_auc([1, 0, 1, 0], [4, 3, 2, 1], ascending_score=False, log_min=log_min, log_max=log_max)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("log_min, log_max", [(0.001, 1.0), (0.001, 0.1), (0.01, 0.1)])
def test_roc_log_auc_random_baseline(log_min, log_max):
    labels = np.tile([1, 0], 10000)
    scores = -np.arange(len(labels))
    result = roc_log_auc(labels, scores, ascending_score=False, log_min=log_min, log_max=log_max)
    baseline = random_roc_log_auc(log_min=log_min, log_max=log_max)
    assert result == pytest.approx(baseline, abs=1e-4)


@pytest.mark.parametrize("metric", [roc_log_auc, random_roc_log_auc])
@pytest.mark.parametrize(
    "log_min, log_max",
    [(0, 1), (-0.1, 1), (0.1, 0.1), (0.2, 0.1), (0.1, 1.1), (np.nan, 1), (0.1, np.nan), (np.inf, 1), (0.1, np.inf)],
)
def test_roc_log_auc_invalid_bounds(metric, log_min, log_max):
    with pytest.raises(ValueError, match="bounds"):
        if metric is roc_log_auc:
            metric([1, 0], [2, 1], log_min=log_min, log_max=log_max)
        else:
            metric(log_min=log_min, log_max=log_max)


@pytest.mark.parametrize("metric", [rie, bedroc])
@pytest.mark.parametrize("alpha", [1e-15, 1e-8, 20, 2000])
@pytest.mark.parametrize("label, expected", [(0, 0.0), (1, 1.0)])
def test_early_recognition_single_class(metric, alpha, label, expected):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert metric([label] * 4, [4, 3, 2, 1], alpha=alpha) == expected


@pytest.mark.parametrize("metric", [rie, bedroc])
@pytest.mark.parametrize("alpha", [0, -1, -20, np.nan, np.inf, -np.inf])
def test_early_recognition_invalid_alpha(metric, alpha):
    with pytest.raises(ValueError, match="alpha"):
        metric([1, 0], [2, 1], alpha=alpha)


@pytest.mark.parametrize("alpha", [np.nextafter(0.0, 1.0), 1e-15, 1e-8, 0.1, 20, 80, 2000, 1e308])
def test_bedroc_best_and_worst_rankings(alpha):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert bedroc([1, 1, 0, 0], [4, 3, 2, 1], alpha=alpha) == pytest.approx(1.0)
        assert bedroc([0, 0, 1, 1], [4, 3, 2, 1], alpha=alpha) == 0.0


@pytest.mark.parametrize("alpha", [1e-15, 1e-8, 0.1, 20, 80, 2000])
@pytest.mark.parametrize(
    "labels",
    [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 0]],
)
def test_early_recognition_decimal_reference(alpha, labels):
    with localcontext() as context:
        context.prec = 80
        samples = Decimal(len(labels))
        weights = [(-Decimal(str(alpha)) * Decimal(rank) / samples).exp() for rank in range(len(labels))]
        positive_count = sum(labels)
        observed = sum(weight for weight, label in zip(weights, labels) if label)
        best = sum(weights[:positive_count])
        worst = sum(weights[-positive_count:])
        expected_rie = float(observed / (Decimal(positive_count) / samples * sum(weights)))
        expected_bedroc = float((observed - worst) / (best - worst))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert rie(labels, [4, 3, 2, 1], alpha=alpha) == pytest.approx(expected_rie, rel=1e-12, abs=1e-14)
        assert bedroc(labels, [4, 3, 2, 1], alpha=alpha) == pytest.approx(expected_bedroc, rel=1e-12, abs=1e-14)


def test_early_recognition_subnormal_alpha():
    alpha = np.nextafter(0.0, 1.0)
    assert rie([1, 0, 1, 0], [4, 3, 2, 1], alpha=alpha) == 1.0
    assert bedroc([1, 0, 1, 0], [4, 3, 2, 1], alpha=alpha) == 0.75


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        ([0, 1, 2], [1, 1, 1], 1.0),
        ([4, 4, 4], [1, 1, 1], 0.0),
        ([4, 4, 4], [0, 1, 2], 0.0),
        ([1, 3, 5, 7], [0, 1, 2, 3], 0.0),
        ([0, 1, 4, 5], [0, 1, 2, 3], np.sqrt(0.8 / 3)),
        ([[0, 1, 2]], [[1, 1, 1]], 1.0),
    ],
)
def test_standard_deviation_error_degenerate_and_linear(y_true, y_pred, expected):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert standard_deviation_error(y_true, y_pred) == pytest.approx(expected)


@pytest.mark.parametrize("y_true, y_pred", [([], []), ([1], [1]), ([1, 2], [1]), ([1], [1, 2])])
def test_standard_deviation_error_invalid_samples(y_true, y_pred):
    with pytest.raises(ValueError):
        standard_deviation_error(y_true, y_pred)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("input_name", ["y_true", "y_pred"])
def test_standard_deviation_error_nonfinite_values(bad_value, input_name):
    inputs = dict(y_true=[0, 1, 2], y_pred=[0, 1, 2])
    inputs[input_name][0] = bad_value
    with pytest.raises(ValueError, match="finite"):
        standard_deviation_error(**inputs)
