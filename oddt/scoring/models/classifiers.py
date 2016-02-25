from sklearn.ensemble import RandomForestClassifier as randomforest
from sklearn.svm import SVC as svm
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

from oddt.scoring.models.neuralnetwork import _ffnet_sklearned

__all__ = ['randomforest', 'svm', 'neuralnetwork']

class neuralnetwork(ClassifierMixin):
    def __init__(self, *args, **kwargs):
        """ Assemble Neural network using sklearn tools plus ffnet wrapper """
        self.pipeline = Pipeline([('empty_dims_remover', VarianceThreshold()),
                                  ('standard_scaler', MinMaxScaler()),
                                  ('neural_network', _ffnet_sklearned(*args, **kwargs))
                                 ])

    def fit(self, descs, target_values, **kwargs):
        self.pipeline.fit(descs, target_values, **kwargs)
        return self

    def predict(self, descs):
        return self.pipeline.predict(descs)

    def score(self, descs, target_values):
        return self.pipeline.score(descs, target_values)
