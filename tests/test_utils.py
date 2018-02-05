import os
import pytest

import oddt
from oddt.utils import check_molecule, chunker, compose_iter

test_data_dir = os.path.dirname(os.path.abspath(__file__))

# common file names
dude_data_dir = os.path.join(test_data_dir, 'data', 'dude', 'xiap')
xiap_crystal_ligand = os.path.join(dude_data_dir, 'crystal_ligand.sdf')
xiap_protein = os.path.join(dude_data_dir, 'receptor_rdkit.pdb')


def test_check_molecule():
    with pytest.raises(ValueError, match='Molecule object'):
        check_molecule([])

    ligand = next(oddt.toolkit.readfile('sdf', xiap_crystal_ligand))
    check_molecule(ligand)

    # force protein
    protein = next(oddt.toolkit.readfile('pdb', xiap_protein))
    with pytest.raises(ValueError, match='marked as a protein'):
        check_molecule(protein, force_protein=True)

    protein.protein = True
    check_molecule(protein, force_protein=True)

    # force coordinates
    mol = oddt.toolkit.readstring('smi', 'c1ccccc1')
    with pytest.raises(ValueError, match='3D coordinates'):
        check_molecule(mol, force_coords=True)

    mol.make3D()
    check_molecule(mol, force_coords=True)

    # with pytest.raises(ValueError, match='positional'):
    #     check_molecule(mol, True)

    mol = oddt.toolkit.readstring('sdf', '''mol_title
 handmade

  0  0  0  0  0  0  0  0  0  0999 V2000
M  END
                          ''')
    with pytest.raises(ValueError, match='has zero atoms'):
        check_molecule(mol, non_zero_atoms=True)


def test_func_composition():
    def double(x):
        return [i * 2 for i in x]

    def inc(x):
        return [i + 1 for i in x]

    assert compose_iter([1], funcs=[double, inc]) == [3]
    assert compose_iter([3], funcs=[double, inc]) == [7]
    assert compose_iter([10], funcs=[double, inc]) == [21]


def test_chunks():
    chunks = chunker('ABCDEFG', 2)
    assert list(chunks), [['A', 'B'], ['C', 'D'], ['E', 'F'] == ['G']]
