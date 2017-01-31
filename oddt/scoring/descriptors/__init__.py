import numpy as np
from scipy.spatial.distance import cdist as distance

from oddt.docking import autodock_vina
from oddt.docking.internal import vina_docking

__all__ = ['close_contacts',
           'fingerprints',
           'autodock_vina_descriptor',
           'oddt_vina_descriptor']


def atoms_by_type(atom_dict, types, mode='atomic_nums'):
    """Returns atom dictionaries based on given criteria.
    Currently we have 3 types of atom selection criteria:
        * atomic numbers ['atomic_nums']
        * Sybyl Atom Types ['atom_types_sybyl']
        * AutoDock4 atom types ['atom_types_ad4'] (http://autodock.scripps.edu/faqs-help/faq/where-do-i-set-the-autodock-4-force-field-parameters)

    Parameters
    ----------
        atom_dict: oddt.toolkit.Molecule.atom_dict
            Atom dictionary as implemeted in oddt.toolkit.Molecule class

        types: array-like
            List of atom types/numbers wanted.

    Returns
    -------
        out: dictionary of shape=[len(types)]
            A dictionary of queried atom types (types are keys of the dictionary).
            Values are of oddt.toolkit.Molecule.atom_dict type.
    """
    if mode == 'atomic_nums':
        return {num: atom_dict[atom_dict['atomicnum'] == num] for num in set(types)}
    elif mode == 'atom_types_sybyl':
        return {t: atom_dict[atom_dict['atomtype'] == t] for t in set(types)}
    elif mode == 'atom_types_ad4':
        # all AD4 atom types are capitalized
        types = [t.upper() for t in types]
        out = {}
        for t in set(types):
            if t == 'HD':
                out[t] = atom_dict[(atom_dict['atomicnum'] == 1) & atom_dict['isdonorh']]
            elif t == 'C':
                out[t] = atom_dict[(atom_dict['atomicnum'] == 6) & ~atom_dict['isaromatic']]
            elif t == 'CD':  # not canonical AD4 type, although used by NNscore, with no description
                out[t] = atom_dict[(atom_dict['atomicnum'] == 6) & ~atom_dict['isdonor']]
            elif t == 'A':
                out[t] = atom_dict[(atom_dict['atomicnum'] == 6) & atom_dict['isaromatic']]
            elif t == 'N':
                out[t] = atom_dict[(atom_dict['atomicnum'] == 7) & ~atom_dict['isacceptor']]
            elif t == 'NA':
                out[t] = atom_dict[(atom_dict['atomicnum'] == 7) & atom_dict['isacceptor']]
            elif t == 'OA':
                out[t] = atom_dict[(atom_dict['atomicnum'] == 8) & atom_dict['isacceptor']]
            elif t == 'F':
                out[t] = atom_dict[atom_dict['atomicnum'] == 9]
            elif t == 'MG':
                out[t] = atom_dict[atom_dict['atomicnum'] == 12]
            elif t == 'P':
                out[t] = atom_dict[atom_dict['atomicnum'] == 15]
            elif t == 'SA':
                out[t] = atom_dict[(atom_dict['atomicnum'] == 16) & atom_dict['isacceptor']]
            elif t == 'S':
                out[t] = atom_dict[(atom_dict['atomicnum'] == 16) & ~atom_dict['isacceptor']]
            elif t == 'CL':
                out[t] = atom_dict[atom_dict['atomicnum'] == 17]
            elif t == 'CA':
                out[t] = atom_dict[atom_dict['atomicnum'] == 20]
            elif t == 'MN':
                out[t] = atom_dict[atom_dict['atomicnum'] == 25]
            elif t == 'FE':
                out[t] = atom_dict[atom_dict['atomicnum'] == 26]
            elif t == 'CU':
                out[t] = atom_dict[atom_dict['atomicnum'] == 29]
            elif t == 'ZN':
                out[t] = atom_dict[atom_dict['atomicnum'] == 30]
            elif t == 'BR':
                out[t] = atom_dict[atom_dict['atomicnum'] == 35]
            elif t == 'I':
                out[t] = atom_dict[atom_dict['atomicnum'] == 53]
            else:
                raise ValueError('Unsopported atom type: %s' % t)
        return out


class close_contacts(object):
    def __init__(self,
                 protein=None,
                 cutoff=4,
                 mode='atomic_nums',
                 ligand_types=None,
                 protein_types=None,
                 aligned_pairs=False):
        """Close contacts descriptor which tallies atoms of type X in certain cutoff from atoms of type Y.

        Parameters
        ----------
            protein: oddt.toolkit.Molecule or None (default=None)
                Default protein to use as reference

            cutoff: int or list, shape=[n,] or shape=[n,2] (default=4)
                Cutoff for atoms in Angstroms given as an integer or a list of ranges,
                eg. [0, 4, 8, 12] or [[0,4],[4,8],[8,12]].
                Upper bound is always inclusive, lower exclusive.

            mode: string (default='atomic_nums')
                Method of atoms selection, as used in `atoms_by_type`

            ligand_types: array
                List of ligand atom types to use

            protein_types: array
                List of protein atom types to use

            aligned_pairs: bool (default=False)
                Flag indicating should permutation of types should be done,
                otherwise the atoms are treated as aligned pairs.
        """
        if type(cutoff) in [int, float]:
            self.cutoff = np.array([cutoff])
        elif len(np.array(cutoff).shape) == 1:
            self.cutoff = np.vstack((np.array(cutoff)[:-1], np.array(cutoff)[1:])).T
        else:
            self.cutoff = np.array(cutoff)
        # for pickle save original value
        self.original_cutoff = cutoff

        self.protein = protein
        self.ligand_types = ligand_types
        self.protein_types = protein_types if protein_types else ligand_types
        self.aligned_pairs = aligned_pairs
        self.mode = mode

        # setup titles
        if len(self.cutoff) == 1:
            self.titles = ['%s.%s' % (str(p), str(l))
                           for p in self.protein_types for l in self.ligand_types
                           ]
        else:
            self.titles = ['%s.%s_%s-%s' % (str(p), str(l), str(c1), str(c2))
                           for p in self.protein_types
                           for l in self.ligand_types
                           for c1, c2 in self.cutoff
                           ]

    def build(self, ligands, protein=None, single=False):
        """Builds descriptors for series of ligands

        Parameters
        ----------
            ligands: iterable of oddt.toolkit.Molecules or oddt.toolkit.Molecule
                A list or iterable of ligands to build the descriptor or a single molecule.

            protein: oddt.toolkit.Molecule or None (default=None)
                Default protein to use as reference

            single: bool (default=False)
                Flag indicating if the ligand is single.

        """
        if protein:
            self.protein = protein
        if single and type(ligands) is not list:
            ligands = [ligands]
        out = np.zeros(len(self), dtype=int)
        for mol in ligands:
            mol_dict = atoms_by_type(mol.atom_dict, self.ligand_types, self.mode)
            if self.aligned_pairs:
                pairs = zip(self.ligand_types, self.protein_types)
            else:
                pairs = [(mol_type, prot_type) for mol_type in self.ligand_types for prot_type in self.protein_types]
            # desc = np.array([(distance(atoms_by_type(protein.atom_dict, [prot_type], self.mode)[prot_type]['coords'], atoms_by_type(mol.atom_dict, [mol_type], self.mode)[mol_type]['coords'])[..., np.newaxis] <= self.cutoff).sum(axis=(0,1)) for mol_type, prot_type in pairs], dtype=int).flatten()

            local_protein_dict = self.protein.atom_dict[(distance(self.protein.atom_dict['coords'], mol.atom_dict['coords']) <= self.cutoff.max()).any(axis=1)]
            prot_dict = atoms_by_type(local_protein_dict, self.protein_types, self.mode)
            desc = []
            for mol_type, prot_type in pairs:
                d = distance(prot_dict[prot_type]['coords'], mol_dict[mol_type]['coords'])[..., np.newaxis]
                if len(self.cutoff) > 1:
                    count = ((d > self.cutoff[..., 0]) & (d <= self.cutoff[..., 1])).sum(axis=(0, 1))
                    # count = ne.evaluate('(d > c0) & (d <= c1)', {'d': d, 'c0': cutoff[...,0], 'c1': self.cutoff[...,1]}).sum(axis=(0,1))
                else:
                    count = (d <= self.cutoff).sum()
                desc.append(count)
            desc = np.array(desc, dtype=int).flatten()
            out = np.vstack((out, desc))
        return out[1:]

    def __len__(self):
        """ Returns the dimensions of descriptors """
        if self.aligned_pairs:
            return len(self.ligand_types) * self.cutoff.shape[0]
        else:
            return len(self.ligand_types) * len(self.protein_types) * self.cutoff.shape[0]

    def __reduce__(self):
        return close_contacts, (self.protein,
                                self.original_cutoff,
                                self.mode,
                                self.ligand_types,
                                self.protein_types,
                                self.aligned_pairs)


class fingerprints(object):
    def __init__(self, fp='fp2', toolkit='ob'):
        self.fp = fp
        self.exchange = False
        # if toolkit == oddt.toolkit.backend:
        #    self.exchange = False
        # else:
        #    self.exchange = True
        #    self.target_toolkit = __import__('toolkits.'+toolkit)

    def _get_fingerprint(self, mol):
        if self.exchange:
            mol = self.target_toolkit.Molecule(mol)
        return mol.calcfp(self.fp).raw

    def build(self, mols, single=False):
        if single:
            mols = [mols]
        out = None

        for mol in mols:
            fp = self._get_fingerprint(mol)
            if out is None:
                out = np.zeros_like(fp)
            out = np.vstack((fp, out))
        return out[1:]

    def __reduce__(self):
        return fingerprints, ()


class autodock_vina_descriptor(object):
    def __init__(self, protein=None, vina_scores=None):
        self.protein = protein
        self.vina = autodock_vina(protein)
        self.vina_scores = vina_scores or ['vina_affinity',
                                           'vina_gauss1',
                                           'vina_gauss2',
                                           'vina_repulsion',
                                           'vina_hydrophobic',
                                           'vina_hydrogen']
        self.titles = self.vina_scores

    def set_protein(self, protein):
        self.protein = protein
        self.vina.set_protein(protein)

    def build(self, ligands, protein=None, single=False):
        if protein:
            self.set_protein(protein)
        else:
            protein = self.protein
        if ligands.__class__.__name__ == 'Molecule':
            ligands = [ligands]
        desc = None
        for mol in ligands:
            # Vina
            # TODO: Asynchronous output from vina, push command to score and retrieve at the end?
            # TODO: Check if ligand has vina scores
            scored_mol = self.vina.score(mol, single=True)[0].data
            vec = np.array(([scored_mol[key] for key in self.vina_scores]), dtype=np.float32).flatten()
            if desc is None:
                desc = vec
            else:
                desc = np.vstack((desc, vec))
        if len(desc.shape) == 1:
            desc = desc.reshape(1, -1)
        return desc

    def __len__(self):
        """ Returns the dimensions of descriptors """
        return len(self.vina_scores)

    def __reduce__(self):
        return autodock_vina_descriptor, (self.protein, self.vina_scores)


class oddt_vina_descriptor(object):
    def __init__(self, protein=None, vina_scores=None):
        self.protein = protein
        self.vina = vina_docking(protein)
        self.all_vina_scores = ['vina_affinity',
                                # inter-molecular interactions
                                'vina_gauss1',
                                'vina_gauss2',
                                'vina_repulsion',
                                'vina_hydrophobic',
                                'vina_hydrogen',
                                # intra-molecular interactions
                                'vina_intra_gauss1',
                                'vina_intra_gauss2',
                                'vina_intra_repulsion',
                                'vina_intra_hydrophobic',
                                'vina_intra_hydrogen',
                                'vina_num_rotors']
        self.vina_scores = vina_scores or self.all_vina_scores
        self.titles = self.vina_scores

    def set_protein(self, protein):
        self.protein = protein
        self.vina.set_protein(protein)

    def build(self, ligands, protein=None, single=False):
        if protein:
            self.set_protein(protein)
        else:
            protein = self.protein
        if ligands.__class__.__name__ == 'Molecule':
            ligands = [ligands]
        desc = None
        for mol in ligands:
            mol_keys = mol.data.keys()
            if any(x not in mol_keys for x in self.vina_scores):
                self.vina.set_ligand(mol)
                inter = self.vina.score_inter()
                intra = self.vina.score_intra()
                num_rotors = self.vina.num_rotors
                # could use self.vina.score(), but better to reuse variables
                affinity = ((inter * self.vina.weights[:5]).sum() /
                            (1 + self.vina.weights[5] * num_rotors))
                assert len(self.all_vina_scores) == len(inter) + len(intra) + 2
                score = dict(zip(self.all_vina_scores,
                                 np.hstack((affinity, inter, intra, num_rotors)).flatten()))
                mol.data.update(score)
            else:
                score = mol.data.to_dict()
            try:
                vec = np.array([score[s] for s in self.vina_scores], dtype=np.float32).flatten()
            except Exception as e:
                print(score, affinity, inter, intra, num_rotors)
                print(mol.title)
                raise e
            if desc is None:
                desc = vec
            else:
                desc = np.vstack((desc, vec))
        if len(desc.shape) == 1:
            desc = desc.reshape(1, -1)
        return desc

    def __len__(self):
        """ Returns the dimensions of descriptors """
        return len(self.vina_scores)

    def __reduce__(self):
        return oddt_vina_descriptor, (self.protein, self.vina_scores)
