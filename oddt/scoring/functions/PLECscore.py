from __future__ import print_function
import sys
from os.path import dirname, isfile, join as path_join
from functools import partial
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from sklearn import __version__ as sklearn_version
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from oddt.metrics import rmse, standard_deviation_error
from oddt.scoring import scorer
from oddt.fingerprints import PLEC, MAX_HASH_VALUE
from oddt.scoring.descriptors import universal_descriptor


class PLECscore(scorer):
    def __init__(self, protein=None, n_jobs=-1, version='linear',
                 depth_protein=5, depth_ligand=1, size=65536):
        """PLECscore - a novel scoring function based on PLEC fingerprints. The
        underlying model can be one of:
            * linear regression
            * neural network (dense, 200x200x200)
            * random forest (100 trees)
        The scoring function is trained on PDBbind v2016 database and even with
        linear model outperforms other machine-learning ones in terms of Pearson
        correlation coefficient on "core set". For details see PLEC publication.
        PLECscore predicts binding affinity (pKi/d).

        .. versionadded:: 0.6

        Parameters
        ----------
        protein : oddt.toolkit.Molecule object
            Receptor for the scored ligands

        n_jobs: int (default=-1)
            Number of cores to use for scoring and training. By default (-1)
            all cores are allocated.

        version: str (default='linear')
            A version of scoring function ('linear', 'nn' or 'rf') - which
            model should be used for the scoring function.

        depth_protein: int (default=5)
            The depth of ECFP environments generated on the protein side of
            interaction. By default 6 (0 to 5) environments are generated.

        depth_ligand: int (default=1)
            The depth of ECFP environments generated on the ligand side of
            interaction. By default 2 (0 to 1) environments are generated.

        size: int (default=65536)
            The final size of a folded PLEC fingerprint. This setting is not
            used to limit the data encoded in PLEC fingerprint (for that
            tune the depths), but only the final lenght. Setting it to too
            low value will lead to many collisions.

        """

        self.protein = protein
        self.n_jobs = n_jobs
        self.version = version
        self.depth_protein = depth_protein
        self.depth_ligand = depth_ligand
        self.size = size

        plec_func = partial(PLEC,
                            depth_ligand=depth_ligand,
                            depth_protein=depth_protein,
                            size=size,
                            count_bits=True,
                            sparse=True,
                            ignore_hoh=True)
        descriptors = universal_descriptor(plec_func, protein=protein,
                                           shape=size, sparse=True)

        if version == 'linear':
            # avoid deprecation warnings
            kwargs = {'fit_intercept': False,
                      'loss': 'huber',
                      'penalty': 'elasticnet',
                      'random_state': 0,
                      'verbose': 0,
                      'alpha': 1e-4,
                      'epsilon': 1e-1,
                      }
            if sklearn_version >= '0.19':
                kwargs['max_iter'] = 100
            else:
                kwargs['n_iter'] = 100
            model = SGDRegressor(**kwargs)
        elif version == 'nn':
            model = MLPRegressor((200, 200, 200),
                                 batch_size=10,
                                 random_state=0,
                                 verbose=0,
                                 solver='lbfgs')
        elif version == 'rf':
            model = RandomForestRegressor(n_estimators=100,
                                          n_jobs=n_jobs,
                                          verbose=0,
                                          oob_score=True,
                                          random_state=0)
        else:
            raise ValueError('The version "%s" is not supported by PLECscore'
                             % version)

        super(PLECscore, self).__init__(model, descriptors,
                                        score_title='PLEC%s_p%i_l%i_s%i' %
                                        (version, depth_protein, depth_ligand,
                                         size))

    def gen_training_data(self,
                          pdbbind_dir,
                          pdbbind_versions=(2016,),
                          home_dir=None,
                          use_proteins=True):
        if home_dir is None:
            home_dir = path_join(dirname(__file__), 'PLECscore')
        filename = path_join(home_dir, 'plecscore_descs_p%i_l%i.csv.gz' %
                             (self.depth_protein, self.depth_ligand))

        # The CSV will contain unfolded FP
        self.descriptor_generator.func.keywords['size'] = MAX_HASH_VALUE
        self.descriptor_generator.shape = MAX_HASH_VALUE

        super(PLECscore, self)._gen_pdbbind_desc(
            pdbbind_dir=pdbbind_dir,
            pdbbind_versions=pdbbind_versions,
            desc_path=filename,
            include_general_set=True,
            use_proteins=use_proteins,
        )

        # reset to the original size
        self.descriptor_generator.func.keywords['size'] = self.size
        self.descriptor_generator.shape = self.size

    def gen_json(self, home_dir=None, pdbbind_version=2016):
        if not home_dir:
            home_dir = path_join(dirname(__file__), 'PLECscore')

        if isinstance(self.model, SGDRegressor):
            attributes = ['coef_', 'intercept_', 't_']
        elif isinstance(self.model, MLPRegressor):
            attributes = ['loss_', 'coefs_', 'intercepts_', 'n_iter_',
                          'n_layers_', 'n_outputs_', 'out_activation_']

        out = {}
        for attr_name in attributes:
            attr = getattr(self.model, attr_name)
            # convert numpy arrays to list for json
            if isinstance(attr, np.ndarray):
                attr = attr.tolist()
            elif (isinstance(attr, (list, tuple)) and
                  isinstance(attr[0], np.ndarray)):
                attr = [x.tolist() for x in attr]
            out[attr_name] = attr

        json_path = path_join(home_dir, 'plecscore_%s_p%i_l%i_s%i_pdbbind%i.json' %
                              (self.version, self.depth_protein,
                               self.depth_ligand, self.size, pdbbind_version))

        with open(json_path, 'w') as json_f:
            json.dump(out, json_f, indent=2)
        return json_path

    def train(self, home_dir=None, sf_pickle=None, pdbbind_version=2016,
              ignore_json=False):
        if not home_dir:
            home_dir = path_join(dirname(__file__), 'PLECscore')
        desc_path = path_join(home_dir, 'plecscore_descs_p%i_l%i.csv.gz' %
                              (self.depth_protein, self.depth_ligand))

        json_path = path_join(
            home_dir, 'plecscore_%s_p%i_l%i_s%i_pdbbind%i.json' %
            (self.version, self.depth_protein,
             self.depth_ligand, self.size, pdbbind_version))

        if (self.version in ['linear'] and  # TODO: support other models
                isfile(json_path) and
                not ignore_json):
            print('Loading pretrained PLECscore %s with depths P%i L%i on '
                  'PDBBind v%i'
                  % (self.version, self.depth_protein, self.depth_ligand,
                     pdbbind_version), file=sys.stderr)
            with open(json_path) as json_f:
                json_data = json.load(json_f)
            for k, v in json_data.items():
                if isinstance(v, list):
                    if isinstance(v[0], list):
                        v = [np.array(x) for x in v]
                    else:
                        v = np.array(v)
                setattr(self.model, k, v)
        else:
            # blacklist core set 2013 and astex
            pdbids_blacklist = [
                '3ao4', '3i3b', '1uto', '1ps3', '1qi0', '3g2z', '3dxg', '3l7b',
                '3mfv', '3b3s', '3kgp', '3fk1', '3fcq', '3lka', '3udh', '4gqq',
                '3imc', '2xdl', '2ymd', '1lbk', '1bcu', '3zsx', '1f8d', '3muz',
                '2v00', '1loq', '3n7a', '2r23', '3nq3', '2hb1', '2w66', '1n2v',
                '3kwa', '3g2n', '4de2', '3ozt', '3b3w', '3cft', '3f3a', '2qmj',
                '3f80', '1a30', '1w3k', '3ivg', '2jdy', '3u9q', '3pxf', '2wbg',
                '1u33', '2x0y', '3mss', '1vso', '1q8t', '3acw', '3bpc', '3vd4',
                '3cj2', '2brb', '1p1q', '2vo5', '3d4z', '2gss', '2yge', '3gy4',
                '3zso', '3ov1', '1w4o', '1zea', '2zxd', '3ueu', '2qft', '1gpk',
                '1f8b', '2jdm', '3su5', '2wca', '3n86', '2x97', '1n1m', '1o5b',
                '2y5h', '3ehy', '4des', '3ebp', '1q8u', '4de1', '3huc', '3l4w',
                '2vl4', '3coy', '3f3c', '1os0', '3owj', '3bkk', '1yc1', '1hnn',
                '3vh9', '3bfu', '1w3l', '3k5v', '2qbr', '1lol', '10gs', '2j78',
                '1r5y', '2weg', '3uo4', '3jvs', '2yfe', '1sln', '2iwx', '2jdu',
                '4djv', '2xhm', '2xnb', '3s8o', '2zcr', '3oe5', '3gbb', '2d3u',
                '3uex', '4dew', '1xd0', '1z95', '2vot', '1oyt', '2ole', '3gcs',
                '1kel', '2vvn', '3kv2', '3pww', '3su2', '1f8c', '2xys', '3l4u',
                '2xb8', '2d1o', '2zjw', '3f3e', '2g70', '2zwz', '1u1b', '4g8m',
                '1o3f', '2x8z', '3cyx', '2cet', '3ag9', '2pq9', '3l3n', '1nvq',
                '2cbj', '2v7a', '1h23', '2qbp', '3b68', '2xbv', '2fvd', '2vw5',
                '3ejr', '3f17', '3nox', '1hfs', '1jyq', '2pcp', '3ge7', '2wtv',
                '2zcq', '2obf', '3e93', '2p4y', '3dd0', '3nw9', '3uri', '3gnw',
                '3su3', '2xy9', '1sqa', '3fv1', '2yki', '3g0w', '3pe2', '1e66',
                '1igj', '4tmn', '2zx6', '3myg', '4gid', '3utu', '1lor', '1mq6',
                '2x00', '2j62', '4djr', '1gm8', '1gpk', '1hnn', '1hp0', '1hq2',
                '1hvy', '1hwi', '1hww', '1ia1', '1j3j', '1jd0', '1jje', '1ke5',
                '1kzk', '1l2s', '1l7f', '1lpz', '1m2z', '1mmv', '1mzc', '1n1m',
                '1n2v', '1n46', '1nav', '1of1', '1of6', '1opk', '1oq5', '1owe',
                '1oyt', '1p2y', '1p62', '1pmn', '1q1g', '1q41', '1q4g', '1r1h',
                '1r55', '1r58', '1r9o', '1s19', '1s3v', '1sg0', '1sj0', '1sq5',
                '1sqn', '1t40', '1t46', '1t9b', '1tow', '1tt1', '1u1c', '1uml',
                '1unl', '1uou', '1v0p', '1v48', '1v4s', '1vcj', '1w1p', '1w2g',
                '1xm6', '1xoq', '1xoz', '1y6b', '1ygc', '1yqy', '1yv3', '1yvf',
                '1ywr', '1z95', '2bm2', '2br1', '2bsm']

            # use remote csv if it's not present
            if not isfile(desc_path):
                branch = 'master'  # define branch/commit
                desc_url = ('https://raw.githubusercontent.com/oddt/oddt/%s'
                            '/oddt/scoring/functions/PLECscore/'
                            'plecscore_descs_p%i_l%i.csv.gz' %
                            (branch, self.depth_protein, self.depth_ligand))

                warnings.warn('The CSV for PLEC P%i L%i is missing. Trying to '
                              'get it from ODDT GitHub.' % (self.depth_protein,
                                                            self.depth_ligand))

                # download and save CSV
                pd.read_csv(desc_url, index_col='pdbid').to_csv(
                    desc_path, compression='gzip')

            # set PLEC size to unfolded
            super(PLECscore, self)._load_pdbbind_desc(
                desc_path,
                train_set=('general', 'refined'),
                pdbbind_version=pdbbind_version,
                train_blacklist=pdbids_blacklist,
                fold_size=self.size,
                )

            print('Training PLECscore %s with depths P%i L%i on PDBBind v%i'
                  % (self.version, self.depth_protein, self.depth_ligand,
                     pdbbind_version), file=sys.stderr)

            self.model.fit(self.train_descs, self.train_target)

            sets = [
                ('Test', self.model.predict(self.test_descs), self.test_target),
                ('Train', self.model.predict(self.train_descs), self.train_target)]
            if self.version == 'rf':
                sets.append(('OOB', self.model.oob_prediction_, self.train_target))

            for name, pred, target in sets:
                print('%s set:' % name,
                      'R2_score: %.4f' % r2_score(target, pred),
                      'Rp: %.4f' % pearsonr(target, pred)[0],
                      'RMSE: %.4f' % rmse(target, pred),
                      'SD: %.4f' % standard_deviation_error(target, pred),
                      sep='\t', file=sys.stderr)

        if sf_pickle is None:
            return self.save('PLEC%s_p%i_l%i_pdbbind%i_s%i.pickle'
                             % (self.version, self.depth_protein,
                                self.depth_ligand, pdbbind_version, self.size))
        else:
            return self.save(sf_pickle)

    @classmethod
    def load(self, filename=None, version='linear', pdbbind_version=2016,
             depth_protein=5, depth_ligand=1, size=65536):
        if filename is None:
            # FIXME: it would be cool to have templates of names for a class
            fname = ('PLEC%s_p%i_l%i_pdbbind%i_s%i.pickle' %
                     (version, depth_protein, depth_ligand,
                      pdbbind_version, size))
            for f in [fname, path_join(dirname(__file__), fname)]:
                if isfile(f):
                    filename = f
                    break
            else:
                print('No pickle, training new scoring function.',
                      file=sys.stderr)
                sf = PLECscore(version=version)
                filename = sf.train(sf_pickle=filename,
                                    pdbbind_version=pdbbind_version)
        return scorer.load(filename)
