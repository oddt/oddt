from __future__ import print_function
import sys
from os.path import dirname, isfile, join as path_join
import numpy as np

from scipy.stats import pearsonr
from sklearn.metrics import r2_score

import warnings

try:
    import compiledtrees
except ImportError:
    compiledtrees = None

from oddt import random_seed
from oddt.metrics import rmse
from oddt.scoring import scorer, ensemble_descriptor
from oddt.scoring.models.regressors import randomforest
from oddt.scoring.descriptors import (close_contacts_descriptor,
                                      oddt_vina_descriptor)


# numpy after pickling gives Runtime Warnings
warnings.simplefilter("ignore", RuntimeWarning)

# RF-Score settings
ligand_atomic_nums = [6, 7, 8, 9, 15, 16, 17, 35, 53]
protein_atomic_nums = [6, 7, 8, 16]
cutoff = 12


class rfscore(scorer):
    def __init__(self, protein=None, n_jobs=-1, version=1, spr=0, **kwargs):
        self.protein = protein
        self.n_jobs = n_jobs
        self.version = version
        self.spr = spr
        if version == 1:
            cutoff = 12
            mtry = 6
            descriptors = close_contacts_descriptor(
                protein,
                cutoff=cutoff,
                protein_types=protein_atomic_nums,
                ligand_types=ligand_atomic_nums)
        elif version == 2:
            cutoff = np.array([0, 2, 4, 6, 8, 10, 12])
            mtry = 14
            descriptors = close_contacts_descriptor(
                protein,
                cutoff=cutoff,
                protein_types=protein_atomic_nums,
                ligand_types=ligand_atomic_nums)
        elif version == 3:
            cutoff = 12
            mtry = 6
            cc = close_contacts_descriptor(
                protein,
                cutoff=cutoff,
                protein_types=protein_atomic_nums,
                ligand_types=ligand_atomic_nums)
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
                             min_samples_split=6,
                             **kwargs)
        super(rfscore, self).__init__(model, descriptors,
                                      score_title='rfscore_v%i' % self.version)

    def gen_training_data(self,
                          pdbbind_dir,
                          pdbbind_versions=(2007, 2012, 2013, 2014, 2015, 2016),
                          home_dir=None,
                          use_proteins=False):
        if home_dir is None:
            home_dir = dirname(__file__) + '/RFScore'
        filename = path_join(home_dir, 'rfscore_descs_v%i.csv' % self.version)

        super(rfscore, self)._gen_pdbbind_desc(
            pdbbind_dir=pdbbind_dir,
            pdbbind_versions=pdbbind_versions,
            desc_path=filename,
            use_proteins=use_proteins,
            opt={'b': None},
        )

    def train(self, home_dir=None, sf_pickle=None, pdbbind_version=2016):
        if not home_dir:
            home_dir = dirname(__file__) + '/RFScore'

        desc_path = path_join(home_dir, 'rfscore_descs_v%i.csv' % self.version)

        super(rfscore, self)._load_pdbbind_desc(desc_path,
                                                pdbbind_version=pdbbind_version)

        # remove sparse dimentions
        if self.spr > 0:
            self.mask = (self.train_descs > self.spr).any(axis=0)
            if self.mask.sum() > 0:
                self.train_descs = self.train_descs[:, self.mask]
                self.test_descs = self.test_descs[:, self.mask]

        # make nets reproducible
        random_seed(1)
        self.model.fit(self.train_descs, self.train_target)

        print('Training RFScore v%i on PDBBind v%i'
              % (self.version, pdbbind_version), file=sys.stderr)

        sets = [
            ('Test', self.model.predict(self.test_descs), self.test_target),
            ('Train', self.model.predict(self.train_descs), self.train_target),
            ('OOB', self.model.oob_prediction_, self.train_target)]

        for name, pred, target in sets:
            print('%s set:' % name,
                  'R2_score: %.4f' % r2_score(target, pred),
                  'Rp: %.4f' % pearsonr(target, pred)[0],
                  'RMSE: %.4f' % rmse(target, pred),
                  sep='\t', file=sys.stderr)

        # compile trees
        if compiledtrees is not None:
            try:
                print('Compiling Random Forest using sklearn-compiledtrees',
                      file=sys.stderr)
                self.model = compiledtrees.CompiledRegressionPredictor(
                    self.model, n_jobs=self.n_jobs)
            except Exception as e:
                print('Failed to compile Random Forest with exception: %s' % e,
                      file=sys.stderr)
                print('Continuing without compiled RF.', file=sys.stderr)

        if sf_pickle is None:
            return self.save('RFScore_v%i_pdbbind%i.pickle'
                             % (self.version, pdbbind_version))
        else:
            return self.save(sf_pickle)

    @classmethod
    def load(self, filename=None, version=1, pdbbind_version=2016):
        if filename is None:
            fname = 'RFScore_v%i_pdbbind%i.pickle' % (version, pdbbind_version)
            for f in [fname, path_join(dirname(__file__), fname)]:
                if isfile(f):
                    filename = f
                    break
            else:
                print('No pickle, training new scoring function.',
                      file=sys.stderr)
                rf = rfscore(version=version)
                filename = rf.train(sf_pickle=filename,
                                    pdbbind_version=pdbbind_version)
        return scorer.load(filename)
