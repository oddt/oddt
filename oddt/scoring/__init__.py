from os.path import dirname, join as path_join
import gzip
from itertools import chain
from functools import partial

import six
from six.moves import cPickle as pickle

import numpy as np
from scipy.sparse import vstack as sparse_vstack
import pandas as pd

from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score, KFold
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import accuracy_score, r2_score

import oddt
from oddt.utils import method_caller
from oddt.datasets import pdbbind
from oddt.fingerprints import sparse_to_csr_matrix, csr_matrix_to_sparse, fold


def cross_validate(model, cv_set, cv_target, n=10, shuffle=True, n_jobs=1):
    """Perform cross validation of model using provided data

    Parameters
    ----------
    model: object
        Model to be tested

    cv_set: array-like of shape = [n_samples, n_features]
        Estimated target values.

    cv_target: array-like of shape = [n_samples] or [n_samples, n_outputs]
        Estimated target values.

    n: integer (default = 10)
        How many folds to be created from dataset

    shuffle: bool (default = True)
        Should data be shuffled before folding.

    n_jobs: integer (default = 1)
        How many CPUs to use during cross validation

    Returns
    -------
    r2: array of shape = [n]
        R^2 score for each of generated folds
    """
    if shuffle:
        cv = KFold(n_splits=n, shuffle=True)
    else:
        cv = n
    return cross_val_score(model, cv_set, cv_target, cv=cv, n_jobs=n_jobs)


# FIX ### If possible make ensemble scorer lazy, for now it consumes all ligands
class scorer(object):
    def __init__(self, model_instance, descriptor_generator_instance,
                 score_title='score'):
        """Scorer class is parent class for scoring functions.

        Parameters
        ----------
        model_instance: model
            Medel compatible with sklearn API (fit, predict and score
            methods)

        descriptor_generator_instance: array of descriptors
            Descriptor generator object

        score_title: string
            Title of score to be used.
        """
        self.model = model_instance
        self.descriptor_generator = descriptor_generator_instance
        self.score_title = score_title

    def _gen_pdbbind_desc(self,
                          pdbbind_dir,
                          pdbbind_versions=(2007, 2012, 2013, 2014, 2015, 2016),
                          desc_path=None,
                          include_general_set=False,
                          use_proteins=False,
                          **kwargs):
        pdbbind_versions = sorted(pdbbind_versions)
        opt = kwargs.get('opt', {})

        # generate metadata
        df = None
        for pdbbind_version in pdbbind_versions:
            p = pdbbind('%s/v%i/' % (pdbbind_dir, pdbbind_version),
                        version=pdbbind_version,
                        opt=opt)
            # Core set

            for set_name in p.pdbind_sets:
                if set_name == 'general_PL':
                    dataset_key = '%i_general' % pdbbind_version
                else:
                    dataset_key = '%i_%s' % (pdbbind_version, set_name)

                tmp_df = pd.DataFrame({
                    'pdbid': list(p.sets[set_name].keys()),
                    dataset_key: list(p.sets[set_name].values())
                })
                if df is not None:
                    df = pd.merge(tmp_df, df, how='outer', on='pdbid')
                else:
                    df = tmp_df

        df.sort_values('pdbid', inplace=True)
        tmp_act = df['%i_general' % pdbbind_versions[-1]].values
        df = df.set_index('pdbid').notnull()
        df['act'] = tmp_act
        # take non-empty and core + refined set
        df = df[df['act'].notnull() &
                (df.filter(regex='.*_[refined,core]').any(axis=1) |
                 include_general_set)]

        # build descriptos
        pdbbind_db = pdbbind('%s/v%i/' % (pdbbind_dir, pdbbind_versions[-1]),
                             version=pdbbind_versions[-1])
        if not desc_path:
            desc_path = path_join(dirname(__file__) + 'descs.csv')

        if self.n_jobs is None:
            n_jobs = -1
        else:
            n_jobs = self.n_jobs

        blacklist = []
        if use_proteins:
            # list of protein files that segfault OB 2.4.1
            blacklist = pdbbind_db.protein_blacklist[oddt.toolkit.backend]

        # check if PDBID exists or is blacklisted
        desc_idx = [pid for pid in df.index.values
                    if (pid not in blacklist and
                        getattr(pdbbind_db[pid], 'protein'
                                if use_proteins
                                else 'pocket') is not None)]

        result = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(method_caller)(
                self.descriptor_generator,
                'build',
                [pdbbind_db[pid].ligand],
                protein=getattr(pdbbind_db[pid], 'protein' if use_proteins
                                else 'pocket'))
            for pid in desc_idx)

        # sparse descs may have different shapes, dense are stored np.array
        sparse = (hasattr(self.descriptor_generator, 'sparse') and
                  self.descriptor_generator.sparse)

        if not sparse:
            result = np.vstack(result)

        # create dataframe with descriptors with pdbids as index
        df_desc = pd.DataFrame(result, index=desc_idx)
        df_desc.index.rename('pdbid', inplace=True)

        # for sparse features leave one column and cast explicitly to list
        if sparse:
            if len(df_desc.columns) > 1:
                raise Exception('There are more than one column in the '
                                'sparse descriptor table.')
            df_desc.columns = ['sparse']
            df_desc['sparse'] = df_desc['sparse'].map(
                lambda x: csr_matrix_to_sparse(x).tolist())

        compression = None
        if desc_path[-3:] == '.gz':
            compression = 'gzip'
        # DF are joined by index (pdbid) since some might be missing
        df.join(df_desc, how='inner').to_csv(desc_path,
                                             float_format='%.5g',
                                             compression=compression)

    def _load_pdbbind_desc(self, desc_path, pdbbind_version=2016,
                           train_set='refined', test_set='core',
                           train_blacklist=None, fold_size=None):
        """
        TODO: write the docs

        """

        df = pd.read_csv(desc_path, index_col='pdbid')

        # generate dense representation of sparse descriptor in CSV
        cols = list(map(str, range(len(self.descriptor_generator))))
        if 'sparse' in df.columns:
            # convert strings to np.arrays
            df['sparse'] = df['sparse'].map(
                lambda x: np.fromstring(x[1:-1], dtype=np.uint64, sep=','))
            cols = 'sparse'  # sparse array will have one column
            # fold only if necessary
            if fold_size:
                df['sparse'] = df['sparse'].map(lambda x: fold(x, fold_size))
            # convert to sparse csr_matrix
            df['sparse'] = df['sparse'].map(
                partial(sparse_to_csr_matrix,
                        size=len(self.descriptor_generator)))

        if isinstance(train_set, six.string_types):
            train_idx = df['%i_%s' % (pdbbind_version, train_set)]
        else:
            train_idx = df[['%i_%s' % (pdbbind_version, s)
                            for s in train_set]].any(axis=1)
        if train_blacklist:
            train_idx &= ~df.index.isin(train_blacklist)
        train_idx &= ~df['%i_%s' % (pdbbind_version, test_set)]

        # load sparse matrices as training is usually faster on them
        if 'sparse' in df.columns:
            self.train_descs = sparse_vstack(df.loc[train_idx, cols].values,
                                             format='csr')
        else:
            self.train_descs = df.loc[train_idx, cols].values
        self.train_target = df.loc[train_idx, 'act'].values

        test_idx = df['%i_%s' % (pdbbind_version, test_set)]
        if 'sparse' in df.columns:
            self.test_descs = sparse_vstack(df.loc[test_idx, cols].values,
                                            format='csr')
        else:
            self.test_descs = df.loc[test_idx, cols].values
        self.test_target = df.loc[test_idx, 'act'].values

    def fit(self, ligands, target, *args, **kwargs):
        """Trains model on supplied ligands and target values

        Parameters
        ----------
        ligands: array-like of ligands
            Molecules to featurize and feed into the model

        target: array-like of shape = [n_samples] or [n_samples, n_outputs]
            Ground truth (correct) target values.
        """
        self.train_descs = self.descriptor_generator.build(ligands)
        return self.model.fit(self.train_descs, target, *args, **kwargs)

    def predict(self, ligands, *args, **kwargs):
        """Predicts values (eg. affinity) for supplied ligands.

        Parameters
        ----------
        ligands: array-like of ligands
            Molecules to featurize and feed into the model

        Returns
        -------
        predicted: np.array or array of np.arrays of shape = [n_ligands]
            Predicted scores for ligands
        """
        descs = self.descriptor_generator.build(ligands)
        return self.model.predict(descs)

    def score(self, ligands, target, *args, **kwargs):
        """Methods estimates the quality of prediction using model's default
        score (accuracy for classification or R^2 for regression)

        Parameters
        ----------
        ligands: array-like of ligands
            Molecules to featurize and feed into the model

        target: array-like of shape = [n_samples] or [n_samples, n_outputs]
            Ground truth (correct) target values.

        Returns
        -------
        s: float
            Quality score (accuracy or R^2) for prediction
        """
        descs = self.descriptor_generator.build(ligands)
        return self.model.score(descs, target, *args, **kwargs)

    def predict_ligand(self, ligand):
        """Local method to score one ligand and update it's scores.

        Parameters
        ----------
        ligand: oddt.toolkit.Molecule object
            Ligand to be scored

        Returns
        -------
        ligand: oddt.toolkit.Molecule object
            Scored ligand with updated scores
        """
        score = self.predict([ligand])[0]
        ligand.data.update({self.score_title: score})
        return ligand

    def predict_ligands(self, ligands):
        """Method to score ligands in a lazy fashion.

        Parameters
        ----------
        ligands: iterable of oddt.toolkit.Molecule objects
            Ligands to be scored

        Returns
        -------
        ligand: iterator of oddt.toolkit.Molecule objects
            Scored ligands with updated scores
        """
        return (self.predict_ligand(lig) for lig in ligands)

    def set_protein(self, protein):
        """Proxy method to update protein in all relevant places.

        Parameters
        ----------
        protein: oddt.toolkit.Molecule object
            New default protein

        """
        self.protein = protein
        if hasattr(self.descriptor_generator, 'set_protein'):
            self.descriptor_generator.set_protein(protein)
        else:
            self.descriptor_generator.protein = protein

    def save(self, filename):
        """Saves scoring function to a pickle file.

        Parameters
        ----------
        filename: string
            Pickle filename
        """
        # FIXME: re-set protein after pickling
        self.set_protein(None)
        # return joblib.dump(self, filename, compress=9)[0]
        with gzip.open(filename, 'w+b', compresslevel=9) as f:
            pickle.dump(self, f, protocol=2)
        return filename

    @classmethod
    def load(self, filename):
        """Loads scoring function from a pickle file.

        Parameters
        ----------
        filename: string
            Pickle filename

        Returns
        -------
        sf: scorer-like object
            Scoring function object loaded from a pickle
        """
        # return joblib.load(filename)
        kwargs = {'encoding': 'latin1'} if six.PY3 else {}
        with gzip.open(filename, 'rb') as f:
            out = pickle.load(f, **kwargs)
        return out


class ensemble_model(object):
    def __init__(self, models):
        """Proxy class to build an ensemble of models with an API as one

        Parameters
        ----------
        models: array
            An array of models
        """
        self._models = models if len(models) else None
        if self._models is not None:
            if is_classifier(self._models[0]):
                check_type = is_classifier
                self._scoring_fun = accuracy_score
            elif is_regressor(self._models[0]):
                check_type = is_regressor
                self._scoring_fun = r2_score
            else:
                raise ValueError('Expected regressors or classifiers,'
                                 ' got %s instead' % type(self._models[0]))
            for model in self._models:
                if not check_type(model):
                    raise ValueError('Different types of models found, privide'
                                     ' either regressors or classifiers.')

    def fit(self, X, y, *args, **kwargs):
        for model in self._models:
            model.fit(X, y, *args, **kwargs)
        return self

    def predict(self, X, *args, **kwargs):
        return np.array([model.predict(X, *args, **kwargs)
                         for model in self._models]).mean(axis=0)

    def score(self, X, y, *args, **kwargs):
        return self._scoring_fun(y.flatten(),
                                 self.predict(X, *args, **kwargs).flatten())


class ensemble_descriptor(object):
    def __init__(self, descriptor_generators):
        """Proxy class to build an ensemble of destriptors with an API as one

        Parameters
        ----------
        models: array
            An array of models
        """
        self._desc_gens = (descriptor_generators if len(descriptor_generators)
                           else None)
        self.titles = list(chain(*(desc_gen.titles
                                   for desc_gen in self._desc_gens)))

    def build(self, mols, *args, **kwargs):
        desc = np.hstack(desc_gen.build(mols, *args, **kwargs)
                         for desc_gen in self._desc_gens)
        return desc

    def set_protein(self, protein):
        for desc in self._desc_gens:
            if hasattr(desc, 'set_protein'):
                desc.set_protein(protein)
            else:
                desc.protein = protein

    def __len__(self):
        """ Returns the dimensions of descriptors """
        return sum(len(desc) for desc in self._desc_gens)

    def __reduce__(self):
        return ensemble_descriptor, (self._desc_gens,)
