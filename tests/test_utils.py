import os
from sklearn.utils.testing import assert_raises_regexp

import oddt
from oddt.utils import check_molecule

test_data_dir = os.path.dirname(os.path.abspath(__file__))

# common file names
dude_data_dir = os.path.join(test_data_dir, 'data', 'dude', 'xiap')
xiap_crystal_ligand = os.path.join(dude_data_dir, 'crystal_ligand.sdf')
xiap_protein = os.path.join(dude_data_dir, 'receptor_rdkit.pdb')


def test_check_molecule():
    assert_raises_regexp(ValueError, 'Molecule object', check_molecule, [])

    ligand = next(oddt.toolkit.readfile('sdf', xiap_crystal_ligand))
    check_molecule(ligand)

    # force protein
    protein = next(oddt.toolkit.readfile('pdb', xiap_protein))
    assert_raises_regexp(ValueError,
                         'marked as a protein',
                         check_molecule,
                         protein,
                         force_protein=True)

    protein.protein = True
    check_molecule(protein, force_protein=True)

    # force coordinates
    mol = oddt.toolkit.readstring('smi', 'c1ccccc1')
    assert_raises_regexp(ValueError,
                         '3D coordinates',
                         check_molecule,
                         mol,
                         force_coords=True)
    mol.make3D()
    check_molecule(mol, force_coords=True)

    #assert_raises_regexp(ValueError, 'positional', check_molecule, mol, True)

    mol = oddt.toolkit.readstring('sdf', '''mol_title
 handmade

  0  0  0  0  0  0  0  0  0  0999 V2000
M  END
                          ''')
    assert_raises_regexp(ValueError,
                         'has zero atoms',
                         check_molecule,
                         mol,
                         non_zero_atoms=True)
