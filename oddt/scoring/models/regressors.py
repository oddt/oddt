"""Collection of regressors models"""

from sklearn.ensemble import RandomForestRegressor as randomforest
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression as mlr
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPRegressor
try:
    from sklearn.cross_decomposition import PLSRegression as pls
except ImportError:
    from sklearn.pls import PLSRegression as pls

__all__ = ['randomforest', 'svm', 'pls', 'neuralnetwork', 'mlr']


class neuralnetwork(RegressorMixin):
    def __init__(self, *args, **kwargs):
        """ Assemble Neural network using sklearn pipeline """
        # Cherrypick arguments for model. Exclude 'steps', which is pipeline argument
        local_kwargs = {key: kwargs.pop(key) for key in list(kwargs.keys())
                        if key != 'steps' and len(key.split('__', 1)) == 1}
        self.pipeline = Pipeline([('empty_dims_remover', VarianceThreshold()),
                                  ('scaler', StandardScaler()),
                                  ('neural_network', MLPRegressor(*args, **local_kwargs))
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

    def score(self, descs, target_values):
        return self.pipeline.score(descs, target_values)


class svm(RegressorMixin):
    def __init__(self, *args, **kwargs):
        """ Assemble a proper SVM using sklearn tools regressor """
        # Cherrypick arguments for model. Exclude 'steps', which is pipeline argument
        local_kwargs = {key: kwargs.pop(key) for key in list(kwargs.keys())
                        if key != 'steps' and len(key.split('__', 1)) == 1}
        self.pipeline = Pipeline([('empty_dims_remover', VarianceThreshold()),
                                  ('scaler', StandardScaler()),
                                  ('svm', SVR(*args, **local_kwargs))
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

    def score(self, descs, target_values):
        return self.pipeline.score(descs, target_values)
