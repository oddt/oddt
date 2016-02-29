# HACK import BFGS before ffnet, otherwise it will segfault when trying to use BFGS
from scipy.optimize import fmin_l_bfgs_b
fmin_l_bfgs_b
## FIX use ffnet for now, use sklearn in future
from ffnet import ffnet,mlgraph,tmlgraph
import numpy as np
from scipy.stats import linregress

class _ffnet_sklearned(object):
    def __init__(self, shape = None, full_conn=True, biases=True, random_weights=True):
        """
        shape: shape of a NN given as a tuple
        """
        self.shape = shape
        self.full_conn = full_conn
        self.biases = biases
        self.random_weights = random_weights
        self.shape = shape

    def get_params(self, deep=True):
        return {'shape': self.shape, 'full_conn': self.full_conn, 'biases': self.biases, 'random_weights': self.random_weights}

    def set_params(self, **args):
        self.__init__(**args)
        return self

    def fit(self, descs, target_values, train_alg='tnc', **kwargs):
        # setup neural network
        if self.full_conn:
            conec = tmlgraph(self.shape, self.biases)
        else:
            conec = mlgraph(self.shape, self.biases)
        self.model = ffnet(conec)
        if self.random_weights:
            self.model.randomweights()
        # train
        getattr(self.model, 'train_'+train_alg)(descs, target_values, **kwargs)
        return self

    def predict(self, descs):
        return np.squeeze(self.model.call(descs), axis=1)

    def score(self, X, y):
        return linregress(self.predict(X).flatten(), y)[2]**2
