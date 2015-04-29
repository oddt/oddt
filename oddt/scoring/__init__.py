import numpy as np
from scipy.stats import linregress
from sklearn.cross_validation import cross_val_score, KFold, train_test_split
from sklearn.externals import joblib as pickle

def cross_validate(model, cv_set, cv_target, n = 10, shuffle=True, n_jobs = 1):
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
    return cross_val_score(model, cv_set, cv_target, cv = cv, n_jobs = n_jobs)

### FIX ### If possible make ensemble scorer lazy, for now it consumes all ligands
class scorer(object):
    def __init__(self, model_instances, descriptor_generator_instances, score_title = 'score'):
        """Scorer class is parent class for scoring functions. It's capable of using multiple models and/or multiple descriptors.
        If multiple models and multiple descriptors are used they should be aligned, since no permutation of such is made.
        
        Parameters
        ----------
            model_instances: array of models
                An array of medels compatible with sklearn API (fit, predict and score methods)
            
            descriptor_generator_instances: array of descriptors
                An array of descriptor objects
            
            score_title: string
                Title of score to be used.
        """
        self.model = model_instances
        if type(model_instances) is list:
            self.single_model = False
        else:
            self.single_model = True
        
        self.descriptor_generator = descriptor_generator_instances
        if type(descriptor_generator_instances) is list:
            if len(descriptor_generator_instances) == len(model_instances):
                raise ValueError, "Length of models list doesn't equal descriptors list"
            self.single_descriptor = False
        else:
            self.single_descriptor = True
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
        if self.single_descriptor:
            self.train_descs = self.descriptor_generator.build(ligands)
        else:
            self.train_descs = [desc_gen.build(ligands) for desc_gen in self.descriptor_generator]
        self.train_target = target
        
        if self.single_model and self.single_descriptor:
            return model.fit(self.train_descs,target, *args, **kwargs)
        elif self.single_model and not self.single_descriptor:
            return [model.fit(desc,target, *args, **kwargs) for desc in self.train_descs]
        else:
            return [model.fit(self.train_descs[n],target, *args, **kwargs) for n, model in enumerate(self.model)]
    
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
        if self.single_model and self.single_descriptor:
            descs = self.descriptor_generator.build(ligands)
            return self.model.predict(descs)
        elif self.single_model and not self.single_descriptor:
            return [self.model.predict(descs, *args, **kwargs) for desc in self.train_descs]
        else:
            descs = [desc_gen.build(ligands) for desc_gen in self.descriptor_generator]
            return [model.predict(descs[n],target, *args, **kwargs) for n, model in enumerate(self.model)]
    
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
        if self.single_model and self.single_descriptor:
            descs = self.descriptor_generator.build(ligands)
            return self.model.score(descs, *args, **kwargs)
        elif self.single_model and not self.single_descriptor:
            return [self.model.score(descs, *args, **kwargs) for desc in self.train_descs]
        else:
            descs = [desc_gen.build(ligands) for desc_gen in self.descriptor_generator]
            return [model.score(descs[n],target, *args, **kwargs) for n, model in enumerate(self.model)]
    
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
        if self.single_descriptor:
            if hasattr(self.descriptor_generator, 'set_protein'):
                self.descriptor_generator.set_protein(protein)
            else:
                self.descriptor_generator.protein = protein
        else:
            for desc in self.descriptor_generator:
                if hasattr(desc, 'set_protein'):
                    desc.set_protein(protein)
                else:
                    desc.protein = protein
    
    def save(self, filename):
        """Saves scoring function to a pickle file.
        
        Parameters
        ----------
            filename: string
                Pickle filename
        """
        self.protein = None
        if self.single_descriptor:
            self.descriptor_generator.protein = None
        else:
            for desc in self.descriptor_generator:
                desc.protein = None
        return pickle.dump(self, filename, compress=9)[0]
    
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
        return pickle.load(filename)
    

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
