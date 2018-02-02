"""Common utils for testing purposes"""
from random import shuffle

import oddt
from oddt.utils import is_openbabel_molecule


def shuffle_mol(mol):
    """Randomly reorder molecule atoms and return a shuffled copy of input."""
    new_mol = mol.clone
    new_order = list(range(len(mol.atoms)))
    shuffle(new_order)
    if is_openbabel_molecule(mol):
        new_mol.OBMol.RenumberAtoms([i + 1 for i in new_order])
    else:
        new_mol.Mol = oddt.toolkits.rdk.Chem.RenumberAtoms(new_mol.Mol, new_order)
    return new_mol
