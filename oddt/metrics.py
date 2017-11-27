"""Metrics for estimating performance of drug discovery methods implemented in ODDT"""
from __future__ import division

from math import ceil
import numpy as np
from scipy.stats import linregress
from sklearn.metrics import roc_curve as roc, auc, mean_squared_error

__all__ = ['roc', 'auc', 'roc_auc', 'roc_log_auc', 'enrichment_factor',
           'random_roc_log_auc', 'rmse']


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
        ef : float
            Enrichment Factor for given percenage in range 0:1
    """
    if ascending_score:
        y_score = -y_score
    fpr, tpr, tresholds = roc(y_true, y_score, pos_label=pos_label)
    return auc(fpr, tpr, reorder=False)


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


def enrichment_factor(y_true, y_score, percentage=1, pos_label=None, kind='fold'):
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
            The percentage for which EF is being calculated

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
            Enrichment Factor for given percenage in range 0:1
    """
    if pos_label is None:
        pos_label = 1
    labels = y_true == pos_label
    assert labels.sum() > 0, "There are no correct predicions. Double-check the pos_label"
    assert len(labels) > 0, "Sample size must be greater than 0"
    # calculate fraction of positve labels
    n_perc = int(ceil(float(percentage) / 100. * len(labels)))
    out = float(labels[:n_perc].sum()) / n_perc
    if kind == 'fold':
        out /= (float(labels.sum()) / len(labels))
    return out


def roc_log_auc(y_true, y_score, pos_label=None, ascending_score=True,
                log_min=0.001, log_max=1.):
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
            Minimum value for estimating AUC. Lower values will be clipped for
            numerical stability.

        log_max : float (default=1.)
            Maximum value for estimating AUC. Higher values will be ignored.

    Returns
    -------
        auc : float
            semi-log ROC AUC
    """
    if ascending_score:
        y_score = -y_score
    fpr, tpr, t = roc(y_true, y_score, pos_label=pos_label)
    fpr = fpr.clip(log_min)
    idx = (fpr <= log_max)
    log_fpr = 1 - np.log10(fpr[idx]) / np.log10(log_min)
    return auc(log_fpr, tpr[idx], reorder=False)


def random_roc_log_auc(log_min=0.001, log_max=1.):
    """Computes area under semi-log ROC for random distribution.

    Parameters
    ----------
        log_min : float (default=0.001)
            Minimum logarithm value for estimating AUC

        log_max : float (default=1.)
            Maximum logarithm value for estimating AUC.

    Returns
    -------
        auc : float
            semi-log ROC AUC for random distribution
    """
    return (log_max - log_min) / (np.log(10) * np.log10(log_max / log_min))


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
            semi-log ROC AUC for random distribution


    """
    y_true = np.atleast_1d(y_true).flatten()
    y_pred = np.atleast_1d(y_pred).flatten()
    A, B = linregress(y_true, y_pred)[:2]  # y = Ax + B
    y_ = A * y_pred + B
    sd = (((y_true - y_) ** 2).sum() / (len(y_pred) - 1)) ** 0.5
    return sd
