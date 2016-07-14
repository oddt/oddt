from __future__ import print_function
import sys
import csv
from os.path import dirname, isfile
import numpy as np
from multiprocessing import Pool
import warnings
from joblib import Parallel, delayed

from oddt import toolkit, random_seed
from oddt.scoring import scorer, ensemble_model
from oddt.scoring.descriptors.binana import binana_descriptor
from oddt.scoring.models.regressors import neuralnetwork
from oddt.datasets import pdbbind

# numpy after pickling gives Runtime Warnings
warnings.simplefilter("ignore", RuntimeWarning)


# skip comments and merge multiple spaces
def _csv_file_filter(f):
    for row in open(f, 'rb'):
        if row[0] == '#':
            continue
        yield ' '.join(row.split())


def _parallel_helper(obj, methodname, *args, **kwargs):
    """Private helper to workaround Python 2 pickle limitations"""
    return getattr(obj, methodname)(*args, **kwargs)


class nnscore(scorer):
    def __init__(self, protein=None, n_jobs=-1, **kwargs):
        self.protein = protein
        self.n_jobs = n_jobs
        model = None
        decsriptors = binana_descriptor(protein)
        super(nnscore, self).__init__(model, decsriptors, score_title='nnscore')

    def gen_training_data(self, pdbbind_dir, pdbbind_version=2007, home_dir=None, sf_pickle=''):
        # build train and test
        pdbbind_db = pdbbind(pdbbind_dir, pdbbind_version)
        if not home_dir:
            home_dir = dirname(__file__) + '/NNScore'

        pdb_set = 'core'
        pdbbind_db.default_set = 'core'
        core_set = pdbbind_db.ids
        core_act = np.array(pdbbind_db.activities)
        result = Parallel(n_jobs=self.n_jobs)(delayed(_parallel_helper)(self.descriptor_generator, 'build', [pid.ligand], protein=pid.pocket) for pid in pdbbind_db if pid.pocket is not None)
        core_desc = np.vstack(result)

        pdbbind_db.default_set = 'refined'
        refined_set = [pid for pid in pdbbind_db.ids if pid not in core_set]
        refined_act = np.array([pdbbind_db.sets[pdbbind_db.default_set][pid] for pid in refined_set])
        result = Parallel(n_jobs=self.n_jobs)(delayed(_parallel_helper)(self.descriptor_generator, 'build', [pid.ligand], protein=pid.pocket) for pid in pdbbind_db if pid.pocket is not None and pid.id not in core_set)
        refined_desc = np.vstack(result)

        self.train_descs = refined_desc
        self.train_target = refined_act.flatten()
        self.test_descs = core_desc
        self.test_target = core_act.flatten()

        # save numpy arrays
        header = 'NNscore data generated using PDBBind v%s' % pdbbind_version
        np.savetxt(home_dir + '/train_descs_pdbbind%i.csv' % (pdbbind_version), self.train_descs, fmt='%.5g', delimiter=',', header=header)
        np.savetxt(home_dir + '/train_target_pdbbind%i.csv' % (pdbbind_version), self.train_target, fmt='%.2f', delimiter=',', header=header)
        np.savetxt(home_dir + '/test_descs_pdbbind%i.csv' % (pdbbind_version), self.test_descs, fmt='%.5g', delimiter=',', header=header)
        np.savetxt(home_dir + '/test_target_pdbbind%i.csv' % (pdbbind_version), self.test_target, fmt='%.2f', delimiter=',', header=header)

    def train(self, home_dir=None, sf_pickle='', pdbbind_version=2007):
        if not home_dir:
            home_dir = dirname(__file__) + '/NNScore'
        # load precomputed descriptors and target values
        self.train_descs = np.loadtxt(home_dir + '/train_descs_pdbbind%i.csv' % (pdbbind_version), delimiter=',', dtype=float)
        self.train_target = np.loadtxt(home_dir + '/train_target_pdbbind%i.csv' % (pdbbind_version), delimiter=',', dtype=float)
        self.test_descs = np.loadtxt(home_dir + '/test_descs_pdbbind%i.csv' % (pdbbind_version), delimiter=',', dtype=float)
        self.test_target = np.loadtxt(home_dir + '/test_target_pdbbind%i.csv' % (pdbbind_version), delimiter=',', dtype=float)

        n_dim = (~((self.train_descs == 0).all(axis=0) | (self.train_descs.min(axis=0) == self.train_descs.max(axis=0)))).sum()

        # number of network to sample; original implementation did 1000, but 100 give results good enough.
        n = 1000
        # make nets reproducible
        random_seed(1)
        seeds = np.random.randint(123456789, size=n)
        trained_nets = Parallel(n_jobs=self.n_jobs, verbose=10)(delayed(_parallel_helper)(neuralnetwork([n_dim, 5, 1], random_state=seeds[i]), 'fit', self.train_descs, self.train_target, neural_network__train_alg='tnc', neural_network__maxfun=10000) for i in range(n))
        # get 20 best
        best_idx = np.array([net.score(self.test_descs, self.test_target.flatten()) for net in trained_nets]).argsort()[::-1][:20]
        self.model = ensemble_model([trained_nets[i] for i in best_idx])

        r2 = self.model.score(self.test_descs, self.test_target)
        r = np.sqrt(r2)
        print('Test set: R**2:', r2, ' R:', r, file=sys.stderr)

        r2 = self.model.score(self.train_descs, self.train_target)
        r = np.sqrt(r2)
        print('Train set: R**2:', r2, ' R:', r, file=sys.stderr)

        if sf_pickle:
            return self.save(sf_pickle)
        else:
            return self.save('NNScore_pdbbind%i.pickle' % (pdbbind_version))

    @classmethod
    def load(self, filename='', pdbbind_version=2007):
        if not filename:
            for f in ['NNScore_pdbbind%i.pickle' % (pdbbind_version), dirname(__file__) + '/NNScore_pdbbind%i.pickle' % (pdbbind_version)]:
                if isfile(f):
                    filename = f
                    break
            else:
                print("No pickle, training new scoring function.", file=sys.stderr)
                nn = nnscore()
                filename = nn.train(pdbbind_version=pdbbind_version)
        return scorer.load(filename)
