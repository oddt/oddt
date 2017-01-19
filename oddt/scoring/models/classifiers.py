from sklearn.ensemble import RandomForestClassifier as randomforest
from sklearn.svm import SVC
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPClassifier

__all__ = ['randomforest', 'svm', 'neuralnetwork']


class neuralnetwork(ClassifierMixin):
    def __init__(self, *args, **kwargs):
        """ Assemble Neural network using sklearn pipeline """
        # Cherrypick arguments for model. Exclude 'steps', which is pipeline argument
        local_kwargs = {key: kwargs.pop(key) for key in list(kwargs.keys())
                        if key != 'steps' and len(key.split('__', 1)) == 1}
        self.pipeline = Pipeline([('empty_dims_remover', VarianceThreshold()),
                                  ('scaler', StandardScaler()),
                                  ('neural_network', MLPClassifier(*args, **local_kwargs))
                                  ]).set_params(**kwargs)

    def get_params(self, deep=True):
        return self.pipeline.get_params(deep=deep)

    def set_params(self, **kwargs):
        return self.pipeline.set_params(**kwargs)

    def fit(self, descs, target_values, **kwargs):
        self.pipeline.fit(descs, target_values, **kwargs)
        return self

    def predict(self, descs):
        return self.pipeline.predict(descs)

    def predict_proba(self, descs):
        return self.pipeline.predict_proba(descs)

    def predict_log_proba(self, descs):
        return self.pipeline.predict_log_proba(descs)

    def score(self, descs, target_values):
        return self.pipeline.score(descs, target_values)


class svm(ClassifierMixin):
    def __init__(self, *args, **kwargs):
        """ Assemble a proper SVM classifier"""
        # Cherrypick arguments for model. Exclude 'steps', which is pipeline argument
        local_kwargs = {key: kwargs.pop(key) for key in list(kwargs.keys())
                        if key != 'steps' and len(key.split('__', 1)) == 1}
        self.pipeline = Pipeline([('empty_dims_remover', VarianceThreshold()),
                                  ('scaler', StandardScaler()),
                                  ('svm', SVC(*args, **local_kwargs))
                                  ]).set_params(**kwargs)

    def get_params(self, deep=True):
        return self.pipeline.get_params(deep=deep)

    def set_params(self, **kwargs):
        return self.pipeline.set_params(**kwargs)

    def fit(self, descs, target_values, **kwargs):
        self.pipeline.fit(descs, target_values, **kwargs)
        return self

    def predict(self, descs):
        return self.pipeline.predict(descs)

    def predict_proba(self, descs):
        return self.pipeline.predict_proba(descs)

    def predict_log_proba(self, descs):
        return self.pipeline.predict_log_proba(descs)

    def score(self, descs, target_values):
        return self.pipeline.score(descs, target_values)
