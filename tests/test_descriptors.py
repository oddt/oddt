import os

import oddt
from oddt.scoring import descriptors


test_data_dir = os.path.dirname(os.path.abspath(__file__))
actives_sdf = os.path.join(test_data_dir, 'data', 'dude', 'xiap',
                           'actives_docked.sdf')
receptor_pdb = os.path.join(test_data_dir, 'data', 'dude', 'xiap',
                            'receptor_rdkit.pdb')


def test_atoms_by_type():
    mol = next(oddt.toolkit.readfile('sdf', actives_sdf))

    for mode, types, num_atoms in (
        ('atomic_nums',
         (6, 7, 8),
         (33, 4, 4)),

        ('atom_types_sybyl',
         ('C.ar', 'C.2', 'C.3', 'N.3', 'N.am', 'O.3', 'O.2'),
         (12, 3, 18, 1, 3, 1, 3)),

        ('atom_types_ad4',
         ('A', 'C', 'N', 'OA'),
         (12, 21, 4, 4))):

        types_dict = descriptors.atoms_by_type(mol.atom_dict, types=types,
                                               mode=mode)
        for t, n in zip(types, num_atoms):
            assert t in types_dict
            assert len(types_dict[t]) == n


def test_close_contacts_descriptor():
    ligands = list(oddt.toolkit.readfile('sdf', actives_sdf))
    rec = next(oddt.toolkit.readfile('pdb', receptor_pdb))
    rec.protein = True
    rec.addh()

    for cutoff, num_contacts in ((4, 6816), ([4], 6816), ([2, 4], 6816),
                                 ([[1, 2], [3, 4]], 6304)):
        contacts_descriptor = descriptors.close_contacts_descriptor(
            cutoff=cutoff,
            ligand_types=[6, 7, 8],
            protein_types=[6, 7, 8])
        length = len(contacts_descriptor.cutoff) * 9
        assert len(contacts_descriptor) == length

        contacts = contacts_descriptor.build(ligands, protein=rec)
        assert contacts.shape, (len(ligands) == length)
        assert contacts.sum() == num_contacts
