"""Collection of regressors models"""

from sklearn.ensemble import RandomForestRegressor as randomforest
from sklearn.svm import SVR as svm
from sklearn.linear_model import LinearRegression as mlr
try:
    from sklearn.cross_decomposition import PLSRegression as pls
except ImportError:
    from sklearn.pls import PLSRegression as pls
    
from .neuralnetwork import neuralnetwork

__all__ = ['randomforest', 'svm', 'pls', 'neuralnetwork', 'mlr']
