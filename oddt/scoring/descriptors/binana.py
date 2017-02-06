"""Internal implementation of binana software
(http://nbcr.ucsd.edu/data/sw/hosted/binana/)

"""

import numpy as np
from oddt.scoring.descriptors import (atoms_by_type,
                                      close_contacts,
                                      oddt_vina_descriptor,
                                      autodock_vina_descriptor)
from oddt import interactions


class binana_descriptor(object):
    def __init__(self, protein=None):
        """ Descriptor build from binana script (as used in NNScore 2.0

        Parameters
        ----------
            protein: oddt.toolkit.Molecule object (default=None)
                Protein object to be used while generating descriptors.
        """
        self.protein = protein
        self.titles = []
        self.vina = oddt_vina_descriptor(protein, vina_scores=['vina_gauss1',
                                                               'vina_gauss2',
                                                               'vina_repulsion',
                                                               'vina_hydrophobic',
                                                               'vina_hydrogen'])
        self.titles += self.vina.titles
        # Close contacts descriptor generators
        cc_4_types = (('A', 'A'), ('A', 'C'), ('A', 'CL'), ('A', 'F'),
                      ('A', 'FE'), ('A', 'HD'), ('A', 'MG'), ('A', 'MN'),
                      ('A', 'N'), ('A', 'NA'), ('A', 'OA'), ('A', 'SA'),
                      ('A', 'ZN'), ('BR', 'C'), ('BR', 'HD'), ('BR', 'OA'),
                      ('C', 'C'), ('C', 'CL'), ('C', 'F'), ('C', 'HD'),
                      ('C', 'MG'), ('C', 'MN'), ('C', 'N'), ('C', 'NA'),
                      ('C', 'OA'), ('C', 'SA'), ('C', 'ZN'), ('CL', 'FE'),
                      ('CL', 'HD'), ('CL', 'MG'), ('CL', 'N'), ('CL', 'OA'),
                      ('CL', 'ZN'), ('F', 'HD'), ('F', 'N'), ('F', 'OA'),
                      ('F', 'SA'), ('FE', 'HD'), ('FE', 'N'), ('FE', 'OA'),
                      ('HD', 'HD'), ('HD', 'I'), ('HD', 'MG'), ('HD', 'MN'),
                      ('HD', 'N'), ('HD', 'NA'), ('HD', 'OA'), ('HD', 'P'),
                      ('HD', 'S'), ('HD', 'SA'), ('HD', 'ZN'), ('MG', 'NA'),
                      ('MG', 'OA'), ('MN', 'N'), ('MN', 'OA'), ('N', 'N'),
                      ('N', 'NA'), ('N', 'OA'), ('N', 'SA'), ('N', 'ZN'),
                      ('NA', 'OA'), ('NA', 'SA'), ('NA', 'ZN'), ('OA', 'OA'),
                      ('OA', 'SA'), ('OA', 'ZN'), ('S', 'ZN'), ('SA', 'ZN'),
                      ('A', 'BR'), ('A', 'I'), ('A', 'P'), ('A', 'S'),
                      ('BR', 'N'), ('BR', 'SA'), ('C', 'FE'), ('C', 'I'),
                      ('C', 'P'), ('C', 'S'), ('CL', 'MN'), ('CL', 'NA'),
                      ('CL', 'P'), ('CL', 'S'), ('CL', 'SA'), ('CU', 'HD'),
                      ('CU', 'N'), ('FE', 'NA'), ('FE', 'SA'), ('I', 'N'),
                      ('I', 'OA'), ('MG', 'N'), ('MG', 'P'), ('MG', 'S'),
                      ('MG', 'SA'), ('MN', 'NA'), ('MN', 'P'), ('MN', 'S'),
                      ('MN', 'SA'), ('N', 'P'), ('N', 'S'), ('NA', 'P'),
                      ('NA', 'S'), ('OA', 'P'), ('OA', 'S'), ('P', 'S'),
                      ('P', 'SA'), ('P', 'ZN'), ('S', 'SA'), ('SA', 'SA'),
                      ('A', 'CU'), ('C', 'CD'))
        cc_4_rec_types, cc_4_lig_types = zip(*cc_4_types)
        self.titles += ['cc_%s.%s_4' % (t1, t2) for t1, t2 in cc_4_types]
        self.cc_4 = cc_4_nn = close_contacts(protein,
                                             cutoff=4,
                                             protein_types=cc_4_rec_types,
                                             ligand_types=cc_4_lig_types,
                                             mode='atom_types_ad4',
                                             aligned_pairs=True)

        self.ele_types = (('A', 'A'), ('A', 'C'), ('A', 'CL'), ('A', 'F'),
                          ('A', 'FE'), ('A', 'HD'), ('A', 'MG'), ('A', 'MN'),
                          ('A', 'N'), ('A', 'NA'), ('A', 'OA'), ('A', 'SA'),
                          ('A', 'ZN'), ('BR', 'C'), ('BR', 'HD'), ('BR', 'OA'),
                          ('C', 'C'), ('C', 'CL'), ('C', 'F'), ('C', 'HD'),
                          ('C', 'MG'), ('C', 'MN'), ('C', 'N'), ('C', 'NA'),
                          ('C', 'OA'), ('C', 'SA'), ('C', 'ZN'), ('CL', 'FE'),
                          ('CL', 'HD'), ('CL', 'MG'), ('CL', 'N'), ('CL', 'OA'),
                          ('CL', 'ZN'), ('F', 'HD'), ('F', 'N'), ('F', 'OA'),
                          ('F', 'SA'), ('F', 'ZN'), ('FE', 'HD'), ('FE', 'N'),
                          ('FE', 'OA'), ('HD', 'HD'), ('HD', 'I'), ('HD', 'MG'),
                          ('HD', 'MN'), ('HD', 'N'), ('HD', 'NA'), ('HD', 'OA'),
                          ('HD', 'P'), ('HD', 'S'), ('HD', 'SA'), ('HD', 'ZN'),
                          ('MG', 'NA'), ('MG', 'OA'), ('MN', 'N'), ('MN', 'OA'),
                          ('N', 'N'), ('N', 'NA'), ('N', 'OA'), ('N', 'SA'),
                          ('N', 'ZN'), ('NA', 'OA'), ('NA', 'SA'), ('NA', 'ZN'),
                          ('OA', 'OA'), ('OA', 'SA'), ('OA', 'ZN'), ('S', 'ZN'),
                          ('SA', 'ZN'), ('A', 'BR'), ('A', 'I'), ('A', 'P'),
                          ('A', 'S'), ('BR', 'N'), ('BR', 'SA'), ('C', 'FE'),
                          ('C', 'I'), ('C', 'P'), ('C', 'S'), ('CL', 'MN'),
                          ('CL', 'NA'), ('CL', 'P'), ('CL', 'S'), ('CL', 'SA'),
                          ('CU', 'HD'), ('CU', 'N'), ('FE', 'NA'), ('FE', 'SA'),
                          ('I', 'N'), ('I', 'OA'), ('MG', 'N'), ('MG', 'P'),
                          ('MG', 'S'), ('MG', 'SA'), ('MN', 'NA'), ('MN', 'P'),
                          ('MN', 'S'), ('MN', 'SA'), ('N', 'P'), ('N', 'S'),
                          ('NA', 'P'), ('NA', 'S'), ('OA', 'P'), ('OA', 'S'),
                          ('P', 'S'), ('P', 'SA'), ('P', 'ZN'), ('S', 'SA'),
                          ('SA', 'SA'))
        self.titles += ['ele_%s.%s_4' % (t1, t2) for t1, t2 in self.ele_types]

        self.ligand_atom_types = ['A', 'BR', 'C', 'CL', 'F', 'HD', 'I', 'N', 'NA', 'OA', 'P', 'S', 'SA']
        self.titles += ['lig_%s' % t1 for t1 in self.ligand_atom_types]

        cc_25_types = [('A', 'A'), ('A', 'C'), ('A', 'CL'), ('A', 'F'),
                       ('A', 'FE'), ('A', 'HD'), ('A', 'MG'), ('A', 'MN'),
                       ('A', 'N'), ('A', 'NA'), ('A', 'OA'), ('A', 'SA'),
                       ('A', 'ZN'), ('BR', 'C'), ('BR', 'HD'), ('BR', 'OA'),
                       ('C', 'C'), ('C', 'CL'), ('C', 'F'), ('C', 'HD'),
                       ('C', 'MG'), ('C', 'MN'), ('C', 'N'), ('C', 'NA'),
                       ('C', 'OA'), ('C', 'SA'), ('C', 'ZN'), ('CD', 'OA'),
                       ('CL', 'FE'), ('CL', 'HD'), ('CL', 'MG'), ('CL', 'N'),
                       ('CL', 'OA'), ('CL', 'ZN'), ('F', 'HD'), ('F', 'N'),
                       ('F', 'OA'), ('F', 'SA'), ('F', 'ZN'), ('FE', 'HD'),
                       ('FE', 'N'), ('FE', 'OA'), ('HD', 'HD'), ('HD', 'I'),
                       ('HD', 'MG'), ('HD', 'MN'), ('HD', 'N'), ('HD', 'NA'),
                       ('HD', 'OA'), ('HD', 'P'), ('HD', 'S'), ('HD', 'SA'),
                       ('HD', 'ZN'), ('MG', 'NA'), ('MG', 'OA'), ('MN', 'N'),
                       ('MN', 'OA'), ('N', 'N'), ('N', 'NA'), ('N', 'OA'),
                       ('N', 'SA'), ('N', 'ZN'), ('NA', 'OA'), ('NA', 'SA'),
                       ('NA', 'ZN'), ('OA', 'OA'), ('OA', 'SA'), ('OA', 'ZN'),
                       ('S', 'ZN'), ('SA', 'ZN')]
        cc_25_rec_types, cc_25_lig_types = zip(*cc_25_types)
        self.cc_25 = close_contacts(protein,
                                    cutoff=2.5,
                                    protein_types=cc_25_rec_types,
                                    ligand_types=cc_25_lig_types,
                                    mode='atom_types_ad4',
                                    aligned_pairs=True)
        self.titles += ['cc_%s.%s_2.5' % (t1, t2) for t1, t2 in cc_25_types]
        # H-Bonds (<4A)
        self.titles += ['hb_4_mol_backbone_alpha',
                        'hb_4_mol_backbone_beta',
                        'hb_4_mol_backbone_other',
                        'hb_4_mol_sidechain_alpha',
                        'hb_4_mol_sidechain_beta',
                        'hb_4_mol_sidechain_other',
                        'hb_4_rec_backbone_alpha',
                        'hb_4_rec_backbone_beta',
                        'hb_4_rec_backbone_other',
                        'hb_4_rec_sidechain_alpha',
                        'hb_4_rec_sidechain_beta',
                        'hb_4_rec_sidechain_other']
        # Hydrophobic Contact <4A
        self.titles += ['hyd_4_backbone_alpha',
                        'hyd_4_backbone_beta',
                        'hyd_4_backbone_other',
                        'hyd_4_sidechain_alpha',
                        'hyd_4_sidechain_beta',
                        'hyd_4_sidechain_other',
                        'hyd_4_all']
        # Pi-stacking (<7.5A)
        self.titles += ['pi_stack_7.5_alpha',
                        'pi_stack_7.5_beta',
                        'pi_stack_7.5_other']
        # T-shaped Pi-Pi interaction
        self.titles += ['pi_t_7.5_alpha',
                        'pi_t_7.5_beta',
                        'pi_t_7.5_other']
        # Pi-cation (<6A)
        self.titles += ['pi_cat_mol_6_alpha',
                        'pi_cat_mol_6_beta',
                        'pi_cat_mol_6_other',
                        'pi_cat_rec_6_alpha',
                        'pi_cat_rec_6_beta',
                        'pi_cat_rec_6_other']
        # Active site flexibility (<4A)
        self.titles += ['as_flex_backbone_alpha',
                        'as_flex_backbone_beta',
                        'as_flex_backbone_other',
                        'as_flex_sidechain_alpha',
                        'as_flex_sidechain_beta',
                        'as_flex_sidechain_other',
                        'as_flex_all']
        # Salt bridges (<5.5)
        self.titles += ['salt_bridge_5.5_alpha',
                        'salt_bridge_5.5_beta',
                        'salt_bridge_5.5_other',
                        'salt_bridge_5.5_all']
        # Rotatable bonds
        self.titles += ['num_rotors']

        assert len(self.titles) == len(self)

    def set_protein(self, protein):
        """ One function to change all relevant proteins

        Parameters
        ----------
            protein: oddt.toolkit.Molecule object
                Protein object to be used while generating descriptors.
                Protein becomes new global and default protein.
        """
        self.protein = protein
        self.vina.set_protein(protein)
        self.cc_4.protein = protein
        self.cc_25.protein = protein

    def build(self, ligands, protein=None):
        """ Descriptor building method

        Parameters
        ----------
            ligands: array-like
                An array of generator of oddt.toolkit.Molecule objects for which the descriptor is computed

            protein: oddt.toolkit.Molecule object (default=None)
                Protein object to be used while generating descriptors.
                If none, then the default protein (from constructor) is used.
                Otherwise, protein becomes new global and default protein.

        Returns
        -------
            descs: numpy array, shape=[n_samples, 351]
                An array of binana descriptors, aligned with input ligands
        """
        if protein:
            self.set_protein(protein)
        else:
            protein = self.protein
        protein_dict = protein.atom_dict
        desc = None
        for mol in ligands:
            mol_dict = mol.atom_dict
            vec = np.array([], dtype=float)
            vec = tuple()
            # Vina
            # TODO: Asynchronous output from vina, push command to score and retrieve at the end?
            # TODO: Check if ligand has vina scores
            vec += tuple(self.vina.build(mol, single=True).flatten())

            # Close Contacts (<4A)
            vec += tuple(self.cc_4.build(mol, single=True).flatten())

            # Electrostatics (<4A)
            ele_rec_types, ele_lig_types = zip(*self.ele_types)
            ele_mol_atoms = atoms_by_type(mol_dict, ele_lig_types, 'atom_types_ad4')
            ele_rec_atoms = atoms_by_type(protein_dict, ele_rec_types, 'atom_types_ad4')
            ele = tuple()
            for r_t, m_t in self.ele_types:
                mol_ele_dict, rec_ele_dict = interactions.close_contacts(ele_mol_atoms[m_t], ele_rec_atoms[r_t], 4)
                if len(mol_ele_dict) and len(rec_ele_dict):
                    ele += (mol_ele_dict['charge'] *
                            rec_ele_dict['charge'] /
                            np.sqrt((mol_ele_dict['coords'] -
                                     rec_ele_dict['coords'])**2).sum(axis=-1) *
                            138.94238460104697e4).sum(),  # convert to J/mol
                else:
                    ele += 0,
            vec += tuple(ele)

            # Ligand Atom Types
            atoms = atoms_by_type(mol_dict, self.ligand_atom_types, 'atom_types_ad4')
            vec += tuple([len(atoms[t]) for t in self.ligand_atom_types])

            # Close Contacts (<2.5A)
            vec += tuple(self.cc_25.build(mol, single=True).flatten())

            # H-Bonds (<4A)
            hbond_mol, hbond_rec, strict = interactions.hbonds(mol, protein, 4)
            # Retain only strict hbonds
            hbond_mol = hbond_mol[strict]
            hbond_rec = hbond_rec[strict]
            backbone = hbond_rec['isbackbone']
            alpha = hbond_rec['isalpha']
            beta = hbond_rec['isbeta']
            other = ~alpha & ~beta
            donor_mol = hbond_mol['isdonor']
            donor_rec = hbond_rec['isdonor']
            hbond_vec = ((donor_mol & backbone & alpha).sum(),
                         (donor_mol & backbone & beta).sum(),
                         (donor_mol & backbone & other).sum(),
                         (donor_mol & ~backbone & alpha).sum(),
                         (donor_mol & ~backbone & beta).sum(),
                         (donor_mol & ~backbone & other).sum(),
                         (donor_rec & backbone & alpha).sum(),
                         (donor_rec & backbone & beta).sum(),
                         (donor_rec & backbone & other).sum(),
                         (donor_rec & ~backbone & alpha).sum(),
                         (donor_rec & ~backbone & beta).sum(),
                         (donor_rec & ~backbone & other).sum())
            vec += tuple(hbond_vec)

            # Hydrophobic contacts (<4A)
            hydrophobic = interactions.hydrophobic_contacts(mol, protein, 4)[1]
            backbone = hydrophobic['isbackbone']
            alpha = hydrophobic['isalpha']
            beta = hydrophobic['isbeta']
            other = ~alpha & ~beta
            hyd_vec = ((backbone & alpha).sum(),
                       (backbone & beta).sum(),
                       (backbone & other).sum(),
                       (~backbone & alpha).sum(),
                       (~backbone & beta).sum(),
                       (~backbone & other).sum(),
                       len(hydrophobic))
            vec += tuple(hyd_vec)

            # Pi-stacking (<7.5A)
            pi_mol, pi_rec, pi_paralel, pi_tshaped = interactions.pi_stacking(mol, protein, 7.5)
            alpha = pi_rec['isalpha'] & pi_paralel
            beta = pi_rec['isbeta'] & pi_paralel
            other = ~alpha & ~beta & pi_paralel
            pi_vec = (alpha.sum(), beta.sum(), other.sum())
            vec += tuple(pi_vec)

            # T-shaped Pi-Pi interaction
            alpha = pi_rec['isalpha'] & pi_tshaped
            beta = pi_rec['isbeta'] & pi_tshaped
            other = ~alpha & ~beta & pi_tshaped
            pi_t_vec = (alpha.sum(), beta.sum(), other.sum())

            # Pi-cation (<6A)
            pi_rec, cat_mol, strict = interactions.pi_cation(protein, mol, 6)
            alpha = pi_rec['isalpha'] & strict
            beta = pi_rec['isbeta'] & strict
            other = ~alpha & ~beta & strict
            pi_cat_vec = (alpha.sum(), beta.sum(), other.sum())

            pi_mol, cat_rec, strict = interactions.pi_cation(mol, protein, 6)
            alpha = cat_rec['isalpha'] & strict
            beta = cat_rec['isbeta'] & strict
            other = ~alpha & ~beta & strict
            pi_cat_vec += (alpha.sum(), beta.sum(), other.sum())

            vec += tuple(pi_cat_vec)

            # T-shape (perpendicular Pi's) (<7.5A)
            vec += tuple(pi_t_vec)

            # Active site flexibility (<4A)
            acitve_site = interactions.close_contacts(mol_dict[mol_dict['atomicnum'] != 1],
                                                      protein_dict[protein_dict['atomicnum'] != 1],
                                                      cutoff=4)[1]
            backbone = acitve_site['isbackbone']
            alpha = acitve_site['isalpha']
            beta = acitve_site['isbeta']
            other = ~alpha & ~beta
            as_flex = ((backbone & alpha).sum(),
                       (backbone & beta).sum(),
                       (backbone & other).sum(),
                       (~backbone & alpha).sum(),
                       (~backbone & beta).sum(),
                       (~backbone & other).sum(),
                       len(acitve_site))
            vec += tuple(as_flex)

            # Salt bridges (<5.5)
            salt_bridges = interactions.salt_bridges(mol, protein, 5.5)[1]
            vec += (salt_bridges['isalpha'].sum(),
                    salt_bridges['isbeta'].sum(),
                    (~salt_bridges['isalpha'] & ~salt_bridges['isbeta']).sum(),
                    len(salt_bridges))

            # Rotatable bonds
            vec += mol.num_rotors,

            if desc is None:
                desc = np.zeros(len(vec), dtype=float)
            desc = np.vstack((desc, np.array(vec, dtype=float)))

        return desc[1:]

    def __len__(self):
        """ Returns the dimensions of descriptors """
        return 350

    def __reduce__(self):
        return binana_descriptor, (self.protein,)
