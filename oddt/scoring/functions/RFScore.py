from __future__ import print_function
import sys
from os.path import dirname, isfile
import numpy as np
from joblib import Parallel, delayed
import warnings
import pandas as pd

try:
    import compiledtrees
except ImportError:
    compiledtrees = None

from oddt import random_seed
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


class rfscore(scorer):
    def __init__(self, protein=None, n_jobs=-1, version=1, spr=0, **kwargs):
        self.protein = protein
        self.n_jobs = n_jobs
        self.version = version
        self.spr = spr
        if version == 1:
            cutoff = 12
            mtry = 6
            descriptors = close_contacts(protein,
                                         cutoff=cutoff,
                                         protein_types=protein_atomic_nums,
                                         ligand_types=ligand_atomic_nums)
        elif version == 2:
            cutoff = np.array([0, 2, 4, 6, 8, 10, 12])
            mtry = 14
            descriptors = close_contacts(protein,
                                         cutoff=cutoff,
                                         protein_types=protein_atomic_nums,
                                         ligand_types=ligand_atomic_nums)
        elif version == 3:
            cutoff = 12
            mtry = 6
            cc = close_contacts(protein,
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
        super(rfscore, self).__init__(model, descriptors, score_title='rfscore_v%i' % self.version)

    def gen_training_data(self,
                          pdbbind_dir,
                          pdbbind_versions=(2007, 2012, 2013, 2014, 2015, 2016),
                          home_dir=None):
        pdbbind_versions = sorted(pdbbind_versions)

        # generate metadata
        df = []
        for pdbbind_version in pdbbind_versions:
            p = pdbbind('%s/v%i/' % (pdbbind_dir, pdbbind_version),
                        version=pdbbind_version,
                        opt={'b': None})
            # Core set
            tmp_df = pd.DataFrame({'pdbid': list(p.sets['core'].keys()),
                                   '%i_core' % pdbbind_version: list(p.sets['core'].values())})
            df = pd.merge(tmp_df, df, how='outer', on='pdbid') if len(df) else tmp_df

            # Refined Set
            tmp_df = pd.DataFrame({'pdbid': list(p.sets['refined'].keys()),
                                   '%i_refined' % pdbbind_version: list(p.sets['refined'].values())})
            df = pd.merge(tmp_df, df, how='outer', on='pdbid')

            # General Set
            general_name = 'general_PL' if pdbbind_version > 2007 else 'general'
            tmp_df = pd.DataFrame({'pdbid': list(p.sets[general_name].keys()),
                                   '%i_general' % pdbbind_version: list(p.sets[general_name].values())})
            df = pd.merge(tmp_df, df, how='outer', on='pdbid')

        df.sort_values('pdbid', inplace=True)
        tmp_act = df['%i_general' % pdbbind_versions[-1]].values
        df = df.set_index('pdbid').notnull()
        df['act'] = tmp_act
        # take non-empty and core + refined set
        df = df[df['act'].notnull() & df.filter(regex='.*_[refined,core]').any(axis=1)]

        # build descriptos
        pdbbind_db = pdbbind('%s/v%i/' % (pdbbind_dir, pdbbind_versions[-1]), version=pdbbind_versions[-1])
        if not home_dir:
            home_dir = dirname(__file__) + '/RFScore'

        result = Parallel(n_jobs=self.n_jobs,
                          verbose=1)(delayed(_parallel_helper)(self.descriptor_generator,
                                                               'build',
                                                               [pdbbind_db[pid].ligand],
                                                               protein=pdbbind_db[pid].pocket)
                                     for pid in df.index.values if pdbbind_db[pid].pocket is not None)
        descs = np.vstack(result)
        for i in range(len(self.descriptor_generator)):
            df[str(i)] = descs[:, i]
        df.to_csv(home_dir + '/rfscore_descs_v%i.csv' % self.version, float_format='%.5g')

    def train(self, home_dir=None, sf_pickle='', pdbbind_version=2016):
        if not home_dir:
            home_dir = dirname(__file__) + '/RFScore'

        # load precomputed descriptors and target values
        df = pd.read_csv(home_dir + '/rfscore_descs_v%i.csv' % self.version, index_col='pdbid')

        train_set = 'refined'
        test_set = 'core'
        self.train_descs = df[df['%i_%s' % (pdbbind_version, train_set)] & ~df['%i_%s' % (pdbbind_version, test_set)]][list(map(str, range(len(self.descriptor_generator))))].values
        self.train_target = df[df['%i_%s' % (pdbbind_version, train_set)] & ~df['%i_%s' % (pdbbind_version, test_set)]]['act'].values
        self.test_descs = df[df['%i_%s' % (pdbbind_version, test_set)]][list(map(str, range(len(self.descriptor_generator))))].values
        self.test_target = df[df['%i_%s' % (pdbbind_version, test_set)]]['act'].values

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
        print('Test set:',
              'R**2: %.4f' % r2,
              'R: %.4f' % r,
              'RMSE: %.4f' % error,
              sep='\t', file=sys.stderr)

        error = rmse(self.model.predict(self.train_descs), self.train_target)
        oob_error = rmse(self.model.oob_prediction_, self.train_target)
        r2 = self.model.score(self.train_descs, self.train_target)
        r = np.sqrt(r2)
        print('Train set:',
              'R**2: %.4f' % r2,
              'R: %.4f' % r,
              'RMSE: %.4f' % error,
              'OOB RMSE: %.4f' % oob_error,
              sep='\t', file=sys.stderr)

        # compile trees
        if compiledtrees is not None:
            try:
                print("Compiling Random Forest using sklearn-compiledtrees", file=sys.stderr)
                self.model = compiledtrees.CompiledRegressionPredictor(self.model, n_jobs=self.n_jobs)
            except Exception as e:
                print("Failed to compile Random Forest with exception: %s" % e, file=sys.stderr)
                print("Continuing without compiled RF.", file=sys.stderr)

        if sf_pickle:
            return self.save(sf_pickle)
        else:
            return self.save('RFScore_v%i_pdbbind%i.pickle' % (self.version, pdbbind_version))

    @classmethod
    def load(self, filename='', version=1, pdbbind_version=2016):
        if not filename:
            for f in ['RFScore_v%i_pdbbind%i.pickle' % (version, pdbbind_version),
                      dirname(__file__) + '/RFScore_v%i_pdbbind%i.pickle' % (version, pdbbind_version)]:
                if isfile(f):
                    filename = f
                    break
            else:
                print("No pickle, training new scoring function.", file=sys.stderr)
                rf = rfscore(version=version)
                filename = rf.train(sf_pickle=filename, pdbbind_version=pdbbind_version)
        return scorer.load(filename)
