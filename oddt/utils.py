"""Common utilities for ODDT"""
from itertools import islice
from types import GeneratorType
import json

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
import oddt


def is_molecule(obj):
    """Check whether an object is an `oddt.toolkits.{rdk,ob}.Molecule` instance.
    """
    return is_openbabel_molecule(obj) or is_rdkit_molecule(obj)


def is_openbabel_molecule(obj):
    """Check whether an object is an `oddt.toolkits.ob.Molecule` instance."""
    return (hasattr(oddt.toolkits, 'ob') and
            isinstance(obj, oddt.toolkits.ob.Molecule))


def is_rdkit_molecule(obj):
    """Check whether an object is an `oddt.toolkits.rdk.Molecule` instance."""
    return (hasattr(oddt.toolkits, 'rdk') and
            isinstance(obj, oddt.toolkits.rdk.Molecule))


def check_molecule(mol,
                   force_protein=False,
                   force_coords=False,
                   non_zero_atoms=False):
    """Universal validator of molecule objects. Usage of positional arguments is
    allowed only for molecule object, otherwise it is prohibitted (i.e. the
    order of arguments **will** change). Desired properties of molecule are
    validated based on specified arguments. By default only the object type is
    checked. In case of discrepancy to the specification a `ValueError` is
    raised with appropriate message.

    Parameters
    ----------
        mol: oddt.toolkit.Molecule object
            Object to verify

        force_protein: bool (default=False)
            Force the molecule to be marked as protein (mol.protein).

        force_coords: bool (default=False)
            Force the molecule to have non-zero coordinates.

        non_zero_atoms: bool (default=False)
            Check if molecule has at least one atom.

    """
    # TODO 2to3 force only one positional argument by adding * to args
    if not is_molecule(mol):
        raise ValueError('Molecule object was expected, insted got: %s'
                         % str(mol))

    if force_protein and not mol.protein:
        raise ValueError('Molecule "%s" is not marked as a protein. Mark it by '
                         'setting the protein property to `True` (`mol.protein '
                         '= True)' % mol.title)

    if force_coords and (not mol.coords.any() or
                         np.isnan(mol.coords).any()):
        raise ValueError('Molecule "%s" has no 3D coordinates. All atoms are '
                         'located at (0, 0, 0).' % mol.title)

    if non_zero_atoms and len(mol.atoms) == 0:
        raise ValueError('Molecule "%s" has zero atoms.' % mol.title)


def compose_iter(iterable, funcs):
    """Chain functions and apply them to iterable, by exhausting the iterable.
    Functions are executed in the order from funcs."""
    for func in funcs:
        iterable = func(iterable)
    return list(iterable)


def chunker(iterable, chunksize=100):
    """Generate chunks from a generator object. If iterable is passed which is
    not a generator it will be converted to one with `iter()`."""
    # ensure it is a generator
    if not isinstance(iterable, GeneratorType):
        iterable = iter(iterable)
    chunk = list(islice(iterable, chunksize))
    while chunk:
        yield chunk
        chunk = list(islice(iterable, chunksize))


# TODO 2to3 remove it when support for Python 2.7 is dropped
def method_caller(obj, methodname, *args, **kwargs):
    """Helper function to workaround Python 2 pickle limitations to parallelize
    methods and generator objects"""
    return getattr(obj, methodname)(*args, **kwargs)


def model_to_dict(model):
    """Export a trained model to directory"""
    if isinstance(model, SGDRegressor):
        attributes = ['coef_', 'intercept_']
    elif isinstance(model, MLPRegressor):
        attributes = ['loss_', 'coefs_', 'intercepts_', 'n_iter_',
                      'n_layers_', 'n_outputs_', 'out_activation_']
    else:
        raise ValueError('Model type "%s" is not supported' %
                         model.__class__.__name__)

    out = {'class_name': model.__class__.__name__}
    for attr_name in attributes:
        attr = getattr(model, attr_name)
        # convert numpy arrays to list for json
        if isinstance(attr, np.ndarray):
            attr = attr.tolist()
        elif (isinstance(attr, (list, tuple)) and
              isinstance(attr[0], np.ndarray)):
            attr = [x.tolist() for x in attr]
        out[attr_name] = attr
    return out


def model_to_json(model, json_file=None):
    """Export a trained model to json"""
    if isinstance(model, list):
        out = [model_to_dict(m) for m in model]
    else:
        out = model_to_dict(model)
    if json_file:
        with open(json_file, 'w') as json_f:
            json.dump(out, json_f, indent=2)
        return json_file
    else:
        return json.dumps(out, indent=2)


def json_to_model(model, json_str=None, json_file=None):
    """Import a trained model from json to a new instance"""
    if json_str is None and json_file is None:
        raise ValueError('You need to provide either a json string or filepath')

    if json_file:
        with open(json_file) as json_f:
            json_data = json.load(json_f)
    else:
        json_data = json.loads(json_str)

    if not isinstance(model, list):
        model = [model]
        json_data = [json_data]

    for m, data in zip(model, json_data):
        if data['class_name'] != m.__class__.__name__:
            raise ValueError('It apears that supplied model ("%s") is not an '
                             'instance of a class in JSON ("%s")' %
                             (m.__class__.__name__, data['class_name']))
        for k, v in data.items():
            if k in ['class_name']:
                continue
            if isinstance(v, list):
                if isinstance(v[0], list):
                    v = [np.array(x) for x in v]
                else:
                    v = np.array(v)
            setattr(m, k, v)

    if len(model) == 1:
        return model[0]
    else:
        return model
