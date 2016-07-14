from __future__ import print_function
import sys
import csv
from os.path import dirname, isfile, isdir
import numpy as np
from joblib import Parallel, delayed
import warnings
from sklearn.metrics import r2_score

try:
    import compiledtrees
except ImportError:
    compiledtrees = None

from oddt import toolkit, random_seed
from oddt.metrics import rmse
from oddt.scoring import scorer, ensemble_descriptor
from oddt.scoring.models.regressors import randomforest
from oddt.scoring.descriptors import close_contacts, oddt_vina_descriptor
from oddt.datasets import pdbbind

# numpy after pickling gives Runtime Warnings
warnings.simplefilter("ignore", RuntimeWarning)

# RF-Score settings
ligand_atomic_nums = [6, 7, 8, 9, 15, 16, 17, 35, 53]
protein_atomic_nums = [6, 7, 8, 16]
cutoff = 12


# define sub-function for paralelization
def _parallel_helper(*args, **kwargs):
    """Private helper to workaround Python 2 pickle limitations to paralelize methods"""
    obj, methodname = args[:2]
    new_args = args[2:]
    return getattr(obj, methodname)(*new_args, **kwargs)


# skip comments and merge multiple spaces
def _csv_file_filter(f):
    for row in open(f, 'rb'):
        if row[0] == '#':
            continue
        yield ' '.join(row.split())


class rfscore(scorer):
    def __init__(self, protein=None, n_jobs=-1, version=1, spr=0, **kwargs):
        self.protein = protein
        self.n_jobs = n_jobs
        self.version = version
        self.spr = spr
        if version == 1:
            cutoff = 12
            mtry = 6
            descriptors = close_contacts(protein, cutoff=cutoff, protein_types=protein_atomic_nums, ligand_types=ligand_atomic_nums)
        elif version == 2:
            cutoff = np.array([0, 2, 4, 6, 8, 10, 12])
            mtry = 14
            descriptors = close_contacts(protein, cutoff=cutoff, protein_types=protein_atomic_nums, ligand_types=ligand_atomic_nums)
        elif version == 3:
            cutoff = 12
            mtry = 6
            cc = close_contacts(protein, cutoff=cutoff, protein_types=protein_atomic_nums, ligand_types=ligand_atomic_nums)
            vina_scores = ['vina_gauss1',
                           'vina_gauss2',
                           'vina_repulsion',
                           'vina_hydrophobic',
                           'vina_hydrogen',
                           'vina_num_rotors']
            vina = oddt_vina_descriptor(protein, vina_scores=vina_scores)
            descriptors = ensemble_descriptor((vina, cc))
        model = randomforest(n_estimators=500,
                             oob_score=True,
                             n_jobs=n_jobs,
                             max_features=mtry,
                             bootstrap=True,
                             **kwargs)
        super(rfscore, self).__init__(model, descriptors, score_title='rfscore_v%i' % self.version)

    def gen_training_data(self, pdbbind_dir, pdbbind_version=2007, home_dir=None, sf_pickle=''):
        # build train and test
        pdbbind_db = pdbbind(pdbbind_dir, pdbbind_version, opt={'b': None})
        if not home_dir:
            home_dir = dirname(__file__) + '/RFScore'

        pdbbind_db.default_set = 'core'
        core_set = pdbbind_db.ids
        core_act = np.array(pdbbind_db.activities)
#         core_desc = np.vstack([self.descriptor_generator.build([pid.ligand], protein=pid.protein) for pid in pdbbind_db])
        result = Parallel(n_jobs=self.n_jobs)(delayed(_parallel_helper)(self.descriptor_generator, 'build', [pid.ligand], protein=pid.pocket) for pid in pdbbind_db if pid.pocket is not None)
        core_desc = np.vstack(result)

        pdbbind_db.default_set = 'refined'
        refined_set = [pid for pid in pdbbind_db.ids if pid not in core_set]
        refined_act = np.array([pdbbind_db.sets[pdbbind_db.default_set][pid] for pid in refined_set])
#         refined_desc = np.vstack([self.descriptor_generator.build([pid.ligand], protein=pid.protein) for pid in pdbbind_db])
        result = Parallel(n_jobs=self.n_jobs)(delayed(_parallel_helper)(self.descriptor_generator, 'build', [pid.ligand], protein=pid.pocket) for pid in pdbbind_db if pid.pocket is not None and pid.id not in core_set)
        refined_desc = np.vstack(result)

        self.train_descs = refined_desc
        self.train_target = refined_act
        self.test_descs = core_desc
        self.test_target = core_act

        # save numpy arrays
        header = 'RFScore data generated using PDBBind v%i' % pdbbind_version
        np.savetxt(home_dir + '/train_descs_v%i_pdbbind%i.csv' % (self.version, pdbbind_version), self.train_descs, fmt='%g', delimiter=',', header=header)
        np.savetxt(home_dir + '/train_target_pdbbind%i.csv' % pdbbind_version, self.train_target, fmt='%.2f', delimiter=',', header=header)
        np.savetxt(home_dir + '/test_descs_v%i_pdbbind%i.csv' % (self.version, pdbbind_version), self.test_descs, fmt='%g', delimiter=',', header=header)
        np.savetxt(home_dir + '/test_target_pdbbind%i.csv' % pdbbind_version, self.test_target, fmt='%.2f', delimiter=',', header=header)

    def train(self, home_dir=None, sf_pickle='', pdbbind_version=2007):
        if not home_dir:
            home_dir = dirname(__file__) + '/RFScore'
        # load precomputed descriptors and target values
        self.train_descs = np.loadtxt(home_dir + '/train_descs_v%i_pdbbind%i.csv' % (self.version, pdbbind_version), delimiter=',', dtype=float)
        self.train_target = np.loadtxt(home_dir + '/train_target_pdbbind%i.csv' % (pdbbind_version), delimiter=',', dtype=float)

        self.test_descs = np.loadtxt(home_dir + '/test_descs_v%i_pdbbind%i.csv' % (self.version, pdbbind_version), delimiter=',', dtype=float)
        self.test_target = np.loadtxt(home_dir + '/test_target_pdbbind%i.csv' % (pdbbind_version), delimiter=',', dtype=float)

        # remove sparse dimentions
        if self.spr > 0:
            self.mask = (self.train_descs > self.spr).any(axis=0)
            if self.mask.sum() > 0:
                self.train_descs = self.train_descs[:, self.mask]
                self.test_descs = self.test_descs[:, self.mask]

        # make nets reproducible
        random_seed(1)
        self.model.fit(self.train_descs, self.train_target)

        print("Training RFScore v%i on PDBBind v%i" % (self.version, pdbbind_version), file=sys.stderr)

        error = rmse(self.model.predict(self.test_descs), self.test_target)
        r2 = self.model.score(self.test_descs, self.test_target)
        r = np.sqrt(r2)
        print('Test set: R**2:', r2, ' R:', r, 'RMSE:', error, file=sys.stderr)

        error = rmse(self.model.predict(self.train_descs), self.train_target)
        r2 = self.model.score(self.train_descs, self.train_target)
        r = np.sqrt(r2)
        print('Train set: R**2:', r2, ' R:', r, 'RMSE:', error, file=sys.stderr)

        # compile trees
        if compiledtrees is not None:
            print("Compiling Random Forest using sklearn-compiledtrees", file=sys.stderr)
            self.model = compiledtrees.CompiledRegressionPredictor(self.model, n_jobs=self.n_jobs)

        if sf_pickle:
            return self.save(sf_pickle)
        else:
            return self.save('RFScore_v%i_pdbbind%i.pickle' % (self.version, pdbbind_version))

    @classmethod
    def load(self, filename='', version=1, pdbbind_version=2007):
        if not filename:
            for f in ['RFScore_v%i_pdbbind%i.pickle' % (version, pdbbind_version), dirname(__file__) + '/RFScore_v%i_pdbbind%i.pickle' % (version, pdbbind_version)]:
                if isfile(f):
                    filename = f
                    break
            else:
                print("No pickle, training new scoring function.", file=sys.stderr)
                rf = rfscore(version=version)
                filename = rf.train(sf_pickle=filename, pdbbind_version=pdbbind_version)
        return scorer.load(filename)
