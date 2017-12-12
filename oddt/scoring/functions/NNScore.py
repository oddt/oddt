from __future__ import print_function
import sys
from os.path import dirname, isfile, join as path_join
import numpy as np
import warnings
from joblib import Parallel, delayed

from oddt import random_seed
from oddt.metrics import rmse
from oddt.scoring import scorer, ensemble_model, _parallel_helper
from oddt.scoring.descriptors.binana import binana_descriptor
from oddt.scoring.models.regressors import neuralnetwork

# numpy after pickling gives Runtime Warnings
warnings.simplefilter("ignore", RuntimeWarning)


class nnscore(scorer):
    def __init__(self, protein=None, n_jobs=-1):
        self.protein = protein
        self.n_jobs = n_jobs
        model = None
        decsriptors = binana_descriptor(protein)
        super(nnscore, self).__init__(model, decsriptors,
                                      score_title='nnscore')

    def gen_training_data(self,
                          pdbbind_dir,
                          pdbbind_versions=(2007, 2012, 2013, 2014, 2015, 2016),
                          home_dir=None):
        if home_dir is None:
            home_dir = dirname(__file__) + '/NNScore'
        filename = path_join(home_dir, 'nnscore_descs.csv')

        super(nnscore, self)._gen_pdbbind_desc(
            pdbbind_dir=pdbbind_dir,
            pdbbind_versions=pdbbind_versions,
            desc_path=filename
        )

    def train(self, home_dir=None, sf_pickle=None, pdbbind_version=2016):
        if not home_dir:
            home_dir = dirname(__file__) + '/NNScore'

        desc_path = path_join(home_dir, 'nnscore_descs.csv')

        super(nnscore, self)._load_pdbbind_desc(desc_path, pdbbind_version=2016)

        # number of network to sample; original implementation did 1000, but
        # 100 give results good enough.
        # TODO: allow user to specify number of nets?
        n = 1000
        # make nets reproducible
        random_seed(1)
        seeds = np.random.randint(123456789, size=n)
        trained_nets = (
            Parallel(n_jobs=self.n_jobs, verbose=10, pre_dispatch='all')(
                delayed(_parallel_helper)(
                    neuralnetwork((5,),
                                  random_state=seeds[i],
                                  activation='logistic',
                                  solver='lbfgs',
                                  max_iter=10000),
                    'fit',
                    self.train_descs,
                    self.train_target)
                for i in range(n)))
        # get 20 best
        trained_nets.sort(key=lambda n: n.score(self.test_descs,
                                                self.test_target.flatten()))
        self.model = ensemble_model(trained_nets[-20:])

        error = rmse(self.model.predict(self.test_descs), self.test_target)
        r2 = self.model.score(self.test_descs, self.test_target)
        r = np.sqrt(r2)
        print('Test set:',
              'R**2: %.4f' % r2,
              'R: %.4f' % r,
              'RMSE: %.4f' % error,
              sep='\t', file=sys.stderr)

        error = rmse(self.model.predict(self.train_descs), self.train_target)
        r2 = self.model.score(self.train_descs, self.train_target)
        r = np.sqrt(r2)
        print('Train set:',
              'R**2: %.4f' % r2,
              'R: %.4f' % r,
              'RMSE: %.4f' % error,
              sep='\t', file=sys.stderr)

        if sf_pickle is None:
            return self.save('NNScore_pdbbind%i.pickle' % (pdbbind_version))
        else:
            return self.save(sf_pickle)

    @classmethod
    def load(self, filename=None, pdbbind_version=2016):
        if filename is None:
            fname = 'NNScore_pdbbind%i.pickle' % (pdbbind_version)
            for f in [fname, path_join(dirname(__file__), fname)]:
                if isfile(f):
                    filename = f
                    break
            else:
                print('No pickle, training new scoring function.', file=sys.stderr)
                nn = nnscore()
                filename = nn.train(pdbbind_version=pdbbind_version)
        return scorer.load(filename)
