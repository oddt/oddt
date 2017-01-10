from itertools import chain

import numpy as np
from scipy.stats import linregress
from sklearn.model_selection import cross_val_score, KFold, train_test_split
import joblib
import gzip
import six
from six.moves import cPickle as pickle


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
        cv = KFold(len(cv_target), n_folds=n, shuffle=True)
    else:
        cv = n
    return cross_val_score(model, cv_set, cv_target, cv=cv, n_jobs=n_jobs)


# FIX ### If possible make ensemble scorer lazy, for now it consumes all ligands
class scorer(object):
    def __init__(self, model_instance, descriptor_generator_instance, score_title='score'):
        """Scorer class is parent class for scoring functions.

        Parameters
        ----------
            model_instance: model
                Medel compatible with sklearn API (fit, predict and score methods)

            descriptor_generator_instance: array of descriptors
                Descriptor generator object

            score_title: string
                Title of score to be used.
        """
        self.model = model_instance
        self.descriptor_generator = descriptor_generator_instance
        self.score_title = score_title

    def fit(self, ligands, target, *args, **kwargs):
        """Trains model on supplied ligands and target values

        Parameters
        ----------
            ligands: array-like of ligands
                Ground truth (correct) target values.

            target: array-like of shape = [n_samples] or [n_samples, n_outputs]
                Estimated target values.
        """
        self.train_descs = self.descriptor_generator.build(ligands)
        return self.model.fit(self.train_descs, target, *args, **kwargs)

    def predict(self, ligands, *args, **kwargs):
        """Predicts values (eg. affinity) for supplied ligands

        Parameters
        ----------
            ligands: array-like of ligands
                Ground truth (correct) target values.

            target: array-like of shape = [n_samples] or [n_samples, n_outputs]
                Estimated target values.

        Returns
        -------
            predicted: np.array or array of np.arrays of shape = [n_ligands]
                Predicted scores for ligands
        """
        descs = self.descriptor_generator.build(ligands)
        return self.model.predict(descs)

    def score(self, ligands, target, *args, **kwargs):
        """Methods estimates the quality of prediction as squared correlation coefficient (R^2)

        Parameters
        ----------
            ligands: array-like of ligands
                Ground truth (correct) target values.

            target: array-like of shape = [n_samples] or [n_samples, n_outputs]
                Estimated target values.

        Returns
        -------
            r2: float
                Squared correlation coefficient (R^2) for prediction
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
        """Method to score ligands lazily

        Parameters
        ----------
            ligands: iterable of oddt.toolkit.Molecule objects
                Ligands to be scored

        Returns
        -------
            ligand: iterator of oddt.toolkit.Molecule objects
                Scored ligands with updated scores
        """
        # make lazy calculation
        for lig in ligands:
            yield self.predict_ligand(lig)

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

    def fit(self, X, y, *args, **kwargs):
        for model in self._models:
            model.fit(X, y, *args, **kwargs)
        return self

    def predict(self, X, *args, **kwargs):
        return np.array([model.predict(X, *args, **kwargs) for model in self._models]).mean(axis=0)

    def score(self, X, y, *args, **kwargs):
        return linregress(self.predict(X, *args, **kwargs).flatten(), y.flatten())[2]**2


class ensemble_descriptor(object):
    def __init__(self, descriptor_generators):
        """Proxy class to build an ensemble of destriptors with an API as one

        Parameters
        ----------
            models: array
                An array of models
        """
        self._desc_gens = descriptor_generators if len(descriptor_generators) else None
        self.titles = list(chain(desc_gen.titles for desc_gen in self._desc_gens))

    def build(self, mols, *args, **kwargs):
        out = []
        for mol in mols:
            desc = np.hstack(desc_gen.build([mol], *args, **kwargs) for desc_gen in self._desc_gens)
            if len(out) == 0:
                out = np.zeros_like(desc)
            out = np.vstack((out, desc))
        return out[1:]

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
