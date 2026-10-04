"""Metrics for estimating performance of drug discovery methods implemented in
ODDT"""

from math import ceil
import numpy as np
from scipy.stats import linregress
from sklearn.metrics import roc_curve as roc, auc, mean_squared_error

__all__ = ["roc", "auc", "roc_auc", "roc_log_auc", "enrichment_factor", "random_roc_log_auc", "rmse", "rie", "bedroc"]


def _roc_scores(y_score, ascending_score):
    """Reverse integer rankings without overflow or loss of precision."""
    y_score = np.asarray(y_score)
    if not ascending_score:
        return y_score
    if y_score.dtype.kind in "biu":
        return np.bitwise_not(y_score)
    return -y_score


def _ranked_labels(y_true, y_score, pos_label):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("Labels and scores must be one-dimensional")
    if not len(y_true) or len(y_true) != len(y_score):
        raise ValueError("Labels and scores must have the same nonzero length")
    return y_true == (1 if pos_label is None else pos_label)


def _validate_log_bounds(log_min, log_max):
    if not np.isfinite(log_min) or not np.isfinite(log_max) or not 0 < log_min < log_max <= 1:
        raise ValueError("Logarithmic bounds must satisfy 0 < log_min < log_max <= 1")


def _validate_alpha(alpha):
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and greater than zero")


def roc_auc(y_true, y_score, pos_label=None, ascending_score=True):
    """Computes ROC AUC score

    Parameters
    ----------
    y_true : array, shape=[n_samples]
        True binary labels, in range {0,1} or {-1,1}. If positive label is
        different than 1, it must be explicitly defined.

    y_score : array, shape=[n_samples]
        Scores for tested series of samples

    pos_label: int
        Positive label of samples (if other than 1)

    ascending_score: bool (default=True)
        Indicates if your score is ascendig. Ascending score icreases with
        deacreasing activity. In other words it ascends on ranking list
        (where actives are on top).

    Returns
    -------
    roc_auc : float
        ROC AUC in range 0:1
    """
    y_score = _roc_scores(y_score, ascending_score)
    fpr, tpr, tresholds = roc(y_true, y_score, pos_label=pos_label)
    return auc(fpr, tpr)


def rmse(y_true, y_pred):
    """Compute Root Mean Squared Error (RMSE)

    Parameters
    ----------
    y_true : array-like of shape = [n_samples] or [n_samples, n_outputs]
        Ground truth (correct) target values.

    y_pred : array-like of shape = [n_samples] or [n_samples, n_outputs]
        Estimated target values.

    Returns
    -------
    rmse : float
        A positive floating point value (the best value is 0.0).
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def enrichment_factor(y_true, y_score, percentage=1, pos_label=None, kind="fold"):
    """Computes enrichment factor for given percentage, i.e. EF_1% is
    enrichment factor for first percent of given samples. This function assumes
    that results are already sorted and samples with best predictions are first.

    Parameters
    ----------
    y_true : array, shape=[n_samples]
        True binary labels, in range {0,1} or {-1,1}. If positive label is
        different than 1, it must be explicitly defined.

    y_score : array, shape=[n_samples]
        Scores for tested series of samples

    percentage : int or float
        The percentage for which EF is being calculated, in (0, 100].

    pos_label: int
        Positive label of samples (if other than 1)

    kind: 'fold' or 'percentage' (default='fold')
        Two kinds of enrichment factor: fold and percentage.
        Fold shows the increase over random distribution (1 is random, the
        higher EF the better enrichment). Percentage returns the fraction of
        positive labels within the top x% of dataset.

    Returns
    -------
    ef : float
        Fold enrichment over random, or a fraction in [0, 1] for percentage.
    """
    if not np.isfinite(percentage) or not 0 < percentage <= 100:
        raise ValueError("percentage must be finite and in (0, 100]")
    if kind not in ("fold", "percentage"):
        raise ValueError("kind must be 'fold' or 'percentage'")
    labels = _ranked_labels(y_true, y_score, pos_label)
    if not labels.any():
        raise ValueError("There are no positive labels. Double-check the pos_label")
    # calculate fraction of positve labels
    n_perc = int(ceil(percentage / 100.0 * len(labels)))
    out = labels[:n_perc].sum() / n_perc
    if kind == "fold":
        out /= labels.sum() / len(labels)
    return out


def roc_log_auc(y_true, y_score, pos_label=None, ascending_score=True, log_min=0.001, log_max=1.0):
    """Computes area under semi-log ROC.

    Parameters
    ----------
    y_true : array, shape=[n_samples]
        True binary labels, in range {0,1} or {-1,1}. If positive label is
        different than 1, it must be explicitly defined.

    y_score : array, shape=[n_samples]
        Scores for tested series of samples

    pos_label: int
        Positive label of samples (if other than 1)

    ascending_score: bool (default=True)
        Indicates if your score is ascendig. Ascending score icreases with
        deacreasing activity. In other words it ascends on ranking list
        (where actives are on top).

    log_min : float (default=0.001)
        Lower integration bound, strictly greater than zero and below log_max.

    log_max : float (default=1.)
        Upper integration bound, at most 1. ROC values are interpolated at
        both integration bounds.

    Returns
    -------
    auc : float
        Semi-log ROC AUC normalized to [0, 1] over the selected interval.
    """
    _validate_log_bounds(log_min, log_max)
    y_score = _roc_scores(y_score, ascending_score)
    fpr, tpr, t = roc(y_true, y_score, pos_label=pos_label)
    idx = (fpr > log_min) & (fpr <= log_max)
    bounded_fpr = np.concatenate(([log_min], fpr[idx], [log_max]))
    bounded_tpr = np.concatenate(([np.interp(log_min, fpr, tpr)], tpr[idx], [np.interp(log_max, fpr, tpr)]))
    log_fpr = (np.log(bounded_fpr) - np.log(log_min)) / (np.log(log_max) - np.log(log_min))
    return auc(log_fpr, bounded_tpr)


def random_roc_log_auc(log_min=0.001, log_max=1.0):
    """Computes area under semi-log ROC for random distribution.

    Parameters
    ----------
    log_min : float (default=0.001)
        Lower integration bound, strictly greater than zero and below log_max.

    log_max : float (default=1.)
        Upper integration bound, at most 1.

    Returns
    -------
    auc : float
        Normalized semi-log ROC AUC for a random distribution.
    """
    _validate_log_bounds(log_min, log_max)
    return (log_max - log_min) / (np.log(log_max) - np.log(log_min))


def standard_deviation_error(y_true, y_pred):
    """Standard Deviation (SD) error implemented as used by Li et. al for
    CASF-2013 (http://dx.doi.org/10.1021/ci500081m).

    Parameters
    ----------
    y_true : array-like
        True values.

    y_pred : array-like
        Prediction to be scored.

    Returns
    -------
    sd : float
        Standard deviation of residuals after fitting a line, using n - 1
        degrees of freedom. Constant predictions use the mean true value.

    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    if len(y_true) != len(y_pred):
        raise ValueError("True and predicted values must have the same length")
    if len(y_true) < 2:
        raise ValueError("At least two samples are required")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("True and predicted values must be finite")
    if np.all(y_true == y_true[0]):
        return 0.0
    if np.all(y_pred == y_pred[0]):
        residuals = y_true - y_true.mean()
    else:
        slope, intercept = linregress(y_pred, y_true)[:2]
        residuals = y_true - (slope * y_pred + intercept)
    return np.sqrt((residuals**2).sum() / (len(y_pred) - 1))


def rie(y_true, y_score, alpha=20, pos_label=None):
    """Computes Robust Initial Enhancement [1]_. This function assumes that results
    are already sorted and samples with best predictions are first.

    Parameters
    ----------
    y_true : array, shape=[n_samples]
        True binary labels, in range {0,1} or {-1,1}. If positive label is
        different than 1, it must be explicitly defined.

    y_score : array, shape=[n_samples]
        Scores for tested series of samples

    alpha: float
        Finite positive alpha. 1/Alpha should be proportional to the percentage
        in EF.

    pos_label: int
        Positive label of samples (if other than 1)

    Returns
    -------
    rie_score : float
            Robust Initial Enhancement. Returns 0 without positives and 1 when
            every sample is positive.

    References
    ----------
    .. [1] Sheridan, R. P.; Singh, S. B.; Fluder, E. M.; Kearsley, S. K.
           Protocols for bridging the peptide to nonpeptide gap in topological
           similarity searches. J. Chem. Inf. Comput. Sci. 2001, 41, 1395-1406.
           DOI: 10.1021/ci0100144

    """
    _validate_alpha(alpha)
    labels = _ranked_labels(y_true, y_score, pos_label)
    positive_count = int(labels.sum())
    if positive_count == 0:
        return 0.0
    if positive_count == len(labels) or alpha < np.finfo(float).eps:
        return 1.0
    positive_fraction = positive_count / len(labels)
    ranks = np.flatnonzero(labels) / len(labels)
    observed = np.exp(-alpha * ranks).sum()
    expected = positive_fraction * (-np.expm1(-alpha)) / (-np.expm1(-alpha / len(labels)))
    return float(observed / expected)


def bedroc(y_true, y_score, alpha=20.0, pos_label=None):
    """Computes Boltzmann-Enhanced Discrimination of Receiver Operating
    Characteristic [1]_.  This function assumes that results are already sorted
    and samples with best predictions are first.

    Parameters
    ----------
    y_true : array, shape=[n_samples]
        True binary labels, in range {0,1} or {-1,1}. If positive label is
        different than 1, it must be explicitly defined.

    y_score : array, shape=[n_samples]
        Scores for tested series of samples

    alpha: float
        Finite positive alpha. 1/Alpha should be proportional to the percentage
        in EF.

    pos_label: int
        Positive label of samples (if other than 1)

    Returns
    -------
    bedroc_score : float
        Boltzmann-Enhanced Discrimination of Receiver Operating Characteristic
        in [0, 1]. Returns 0 without positives and 1 when every sample is positive.

    References
    ----------
    .. [1] Truchon J-F, Bayly CI. Evaluating virtual screening methods: good
           and bad metrics for the "early recognition" problem.
           J Chem Inf Model. 2007;47: 488-508.
           DOI: 10.1021/ci600426e

    """
    _validate_alpha(alpha)
    labels = _ranked_labels(y_true, y_score, pos_label)
    positive_count = int(labels.sum())
    if positive_count == 0:
        return 0.0
    negative_count = len(labels) - positive_count
    if negative_count == 0:
        return 1.0
    ranks = np.flatnonzero(labels)
    rank_gaps = (negative_count + np.arange(positive_count) - ranks) / len(labels)
    negative_fraction = negative_count / len(labels)
    if alpha < np.finfo(float).eps:
        return float(rank_gaps.mean() / negative_fraction)
    weights = np.exp(-alpha * (ranks / len(labels)))
    observed_difference = (weights * (-np.expm1(-alpha * rank_gaps))).sum()
    best_sum = (-np.expm1(-alpha * (positive_count / len(labels)))) / (-np.expm1(-alpha / len(labels)))
    score = observed_difference / (best_sum * (-np.expm1(-alpha * negative_fraction)))
    return float(np.clip(score, 0, 1))
