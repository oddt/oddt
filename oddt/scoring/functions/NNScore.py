from __future__ import print_function
import sys
from os.path import dirname, isfile, join as path_join
import numpy as np
import warnings
from joblib import Parallel, delayed

from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from oddt import random_seed
from oddt.utils import method_caller
from oddt.metrics import rmse, standard_deviation_error
from oddt.scoring import scorer, ensemble_model
from oddt.scoring.descriptors.binana import binana_descriptor
from oddt.scoring.models.regressors import neuralnetwork

# numpy after pickling gives Runtime Warnings
warnings.simplefilter("ignore", RuntimeWarning)


class nnscore(scorer):
    def __init__(self, protein=None, n_jobs=-1):
        """NNScore implementation [1]_. Based on Binana descriptors [2]_ and
        an ensemble of 20 best scored nerual networks with a hidden layer of
        5 nodes. The NNScore predicts binding affinity (pKi/d).

        Parameters
        ----------
        protein : oddt.toolkit.Molecule object
            Receptor for the scored ligands

        n_jobs: int (default=-1)
            Number of cores to use for scoring and training. By default (-1)
            all cores are allocated.

        References
        ----------
        .. [1] Durrant JD, McCammon JA. NNScore 2.0: a neural-network
            receptor-ligand scoring function. J Chem Inf Model. 2011;51:
            2897-2903. doi:10.1021/ci2003889

        .. [2] Durrant JD, McCammon JA. BINANA: a novel algorithm for
            ligand-binding characterization. J Mol Graph Model. 2011;29:
            888-893. doi:10.1016/j.jmgm.2011.01.004
        """
        self.protein = protein
        self.n_jobs = n_jobs
        model = None
        decsriptors = binana_descriptor(protein)
        super(nnscore, self).__init__(model, decsriptors,
                                      score_title='nnscore')

    def gen_training_data(self,
                          pdbbind_dir,
                          pdbbind_versions=(2007, 2012, 2013, 2014, 2015, 2016),
                          home_dir=None,
                          use_proteins=False):
        if home_dir is None:
            home_dir = dirname(__file__) + '/NNScore'
        filename = path_join(home_dir, 'nnscore_descs.csv')

        super(nnscore, self)._gen_pdbbind_desc(
            pdbbind_dir=pdbbind_dir,
            pdbbind_versions=pdbbind_versions,
            desc_path=filename,
            use_proteins=use_proteins
        )

    def train(self, home_dir=None, sf_pickle=None, pdbbind_version=2016):
        if not home_dir:
            home_dir = dirname(__file__) + '/NNScore'

        desc_path = path_join(home_dir, 'nnscore_descs.csv')

        super(nnscore, self)._load_pdbbind_desc(desc_path,
                                                pdbbind_version=pdbbind_version)

        # number of network to sample; original implementation did 1000, but
        # 100 give results good enough.
        # TODO: allow user to specify number of nets?
        n = 1000
        # make nets reproducible
        random_seed(1)
        seeds = np.random.randint(123456789, size=n)
        trained_nets = (
            Parallel(n_jobs=self.n_jobs, verbose=10, pre_dispatch='all')(
                delayed(method_caller)(
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

        sets = [
            ('Test', self.model.predict(self.test_descs), self.test_target),
            ('Train', self.model.predict(self.train_descs), self.train_target)]

        for name, pred, target in sets:
            print('%s set:' % name,
                  'R2_score: %.4f' % r2_score(target, pred),
                  'Rp: %.4f' % pearsonr(target, pred)[0],
                  'RMSE: %.4f' % rmse(target, pred),
                  'SD: %.4f' % standard_deviation_error(target, pred),
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
