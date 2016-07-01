## FIX use ffnet for now, use sklearn in future
try:
    from ffnet import ffnet,mlgraph,tmlgraph
except ImportError:
    pass
import numpy as np
from scipy.stats import linregress

from oddt import random_seed

class _ffnet_sklearned(object):
    def __init__(self, shape = None, full_conn=True, biases=True, random_state=None, random_weights=True, n_jobs=1):
        """
        shape: shape of a NN given as a tuple
        """
        self.shape = shape
        self.full_conn = full_conn
        self.biases = biases
        self.random_weights = random_weights
        self.random_state = random_state
        self.shape = shape
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {'shape': self.shape, 'full_conn': self.full_conn, 'biases': self.biases, 'random_state': self.random_state, 'random_weights': self.random_weights, 'n_jobs': self.n_jobs}

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
            if not self.random_state is None:
                random_seed(self.random_state)
            self.model.randomweights()
        # train
        getattr(self.model, 'train_'+train_alg)(descs, target_values, nproc='ncpu' if self.n_jobs < 1 else self.n_jobs, **kwargs)
        return self

    def predict(self, descs):
        return np.squeeze(self.model.call(descs), axis=1)

    def score(self, X, y):
        return linregress(self.predict(X).flatten(), y)[2]**2
