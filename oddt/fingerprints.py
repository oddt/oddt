"""
    Module checks interactions between two molecules and
    creates interacion fingerprints.

"""

from itertools import chain
import numpy as np
import oddt
from oddt.interactions import (pi_stacking,
                               pi_cation,
                               hbond_acceptor_donor,
                               salt_bridge_plus_minus,
                               hydrophobic_contacts,
                               acceptor_metal)

__all__ = ['InteractionFingerprint',
           'SimpleInteractionFingerprint',
           'ECFP',
           'dice',
           'tanimoto']


def InteractionFingerprint(ligand, protein, strict=True):
    """
        Interaction fingerprint accomplished by converting the molecular
        interaction of ligand-protein into bit array according to
        the residue of choice and the interaction. For every residue
        (One row = one residue) there are eight bits which represent
        eight type of interactions:

        - (Column 0) hydrophobic contacts
        - (Column 1) aromatic face to face
        - (Column 2) aromatic edge to face
        - (Column 3) hydrogen bond (protein as hydrogen bond donor)
        - (Column 4) hydrogen bond (protein as hydrogen bond acceptor)
        - (Column 5) salt bridges (protein positively charged)
        - (Column 6) salt bridges (protein negatively charged)
        - (Column 7) salt bridges (ionic bond with metal ion)

        Parameters
        ----------
        ligand, protein : oddt.toolkit.Molecule object
            Molecules to check interactions between them
        strict : bool (deafult = True)
            If False, do not include condition, which informs whether atoms
            form 'strict' H-bond (pass all angular cutoffs).

        Returns
        -------
        InteractionFingerprint : numpy array
            Vector of calculated IFP (size = no residues * 8 type of interaction)

    """
    resids = np.unique(protein.atom_dict['resid'])
    IFP = np.zeros((len(resids), 8), dtype=np.uint8)

    # hydrophobic contacts (column = 0)
    hydrophobic = hydrophobic_contacts(protein, ligand)[0]['resid']
    np.add.at(IFP, [np.searchsorted(resids, hydrophobic), 0], 1)

    # aromatic face to face (Column = 1), aromatic edge to face (Column = 2)
    rings, _, strict_parallel, strict_perpendicular = pi_stacking(
        protein, ligand)
    np.add.at(IFP, [np.searchsorted(
        resids, rings[strict_parallel]['resid']), 1], 1)
    np.add.at(IFP, [np.searchsorted(
        resids, rings[strict_perpendicular]['resid']), 2], 1)

    # h-bonds, protein as a donor (Column = 3)
    _, donors, strict0 = hbond_acceptor_donor(ligand, protein)
    if strict is False:
        strict0 = None
    np.add.at(IFP, [np.searchsorted(resids, donors[strict0]['resid']), 3], 1)

    # h-bonds, protein as an acceptor (Column = 4)
    acceptors, _, strict1 = hbond_acceptor_donor(protein, ligand)
    if strict is False:
        strict1 = None
    np.add.at(IFP, [np.searchsorted(
        resids, acceptors[strict1]['resid']), 4], 1)

    # salt bridges, protein positively charged (Column = 5)
    plus, _ = salt_bridge_plus_minus(protein, ligand)
    np.add.at(IFP, [np.searchsorted(resids, plus['resid']), 5], 1)

    # salt bridges, protein negatively charged (Colum = 6)
    _, minus = salt_bridge_plus_minus(ligand, protein)
    np.add.at(IFP, [np.searchsorted(resids, minus['resid']), 6], 1)

    # salt bridges, ionic bond with metal ion (Column = 7)
    _, metal, strict2 = acceptor_metal(protein, ligand)
    if strict is False:
        strict2 = None
    np.add.at(IFP, [np.searchsorted(resids, metal[strict2]['resid']), 7], 1)

    return IFP.flatten()


def SimpleInteractionFingerprint(ligand, protein, strict=True):
    """
        Based on http://dx.doi.org/10.1016/j.csbj.2014.05.004
        Every IFP consists of 8 bits per amino acid (One row = one amino acid)
        and present eight type of interaction:

        - (Column 0) hydrophobic contacts
        - (Column 1) aromatic face to face
        - (Column 2) aromatic edge to face
        - (Column 3) hydrogen bond (protein as hydrogen bond donor)
        - (Column 4) hydrogen bond (protein as hydrogen bond acceptor)
        - (Column 5) salt bridges (protein positively charged)
        - (Column 6) salt bridges (protein negatively charged)
        - (Column 7) salt bridges (ionic bond with metal ion)

        Returns matrix, which is sorted acordingly to this pattern : 'ALA',
        'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU',
        'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', ''.
        The '' means cofactor. Index of amino acid in pattern coresponds
        to row in returned matrix.

        Parameters
        ----------
        ligand, protein : oddt.toolkit.Molecule object
            Molecules to check interactions beetwen them
        strict : bool (deafult = True)
            If False, do not include condition, which informs whether atoms
            form 'strict' H-bond (pass all angular cutoffs).

        Returns
        -------
        InteractionFingerprint : numpy array
            Vector of calculated IFP (size = 168)

    """

    amino_acids = np.array(['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',
                            'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
                            'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', ''],
                           dtype='<U3')

    IFP = np.zeros((len(amino_acids), 8), dtype=np.uint8)

    # hydrophobic (Column = 0)
    hydrophobic = hydrophobic_contacts(protein, ligand)[0]['resname']
    hydrophobic[~np.in1d(hydrophobic, amino_acids)] = ''
    np.add.at(IFP, [np.searchsorted(amino_acids, hydrophobic), 0], 1)

    # aromatic face to face (Column = 1), aromatic edge to face (Column = 2)
    rings, _, strict_parallel, strict_perpendicular = pi_stacking(
        protein, ligand)
    rings[strict_parallel]['resname'][~np.in1d(
        rings[strict_parallel]['resname'], amino_acids)] = ''
    np.add.at(IFP, [np.searchsorted(
        amino_acids, rings[strict_parallel]['resname']), 1], 1)
    rings[strict_parallel]['resname'][~np.in1d(rings[strict_perpendicular]
                                               ['resname'], amino_acids)] = ''
    np.add.at(IFP, [np.searchsorted(
        amino_acids, rings[strict_perpendicular]['resname']), 2], 1)

    # hbonds donated by the protein (Column = 3)
    _, donors, strict0 = hbond_acceptor_donor(ligand, protein)
    donors['resname'][~np.in1d(donors['resname'], amino_acids)] = ''
    if strict is False:
        strict0 = None
    np.add.at(IFP, [np.searchsorted(
        amino_acids, donors[strict0]['resname']), 3], 1)

    # hbonds donated by the ligand (Column = 4)
    acceptors, _, strict1 = hbond_acceptor_donor(protein, ligand)
    acceptors['resname'][~np.in1d(acceptors['resname'], amino_acids)] = ''
    if strict is False:
        strict1 = None
    np.add.at(IFP, [np.searchsorted(
        amino_acids, acceptors[strict1]['resname']), 4], 1)

    # ionic bond with protein cation(Column = 5)
    plus, _ = salt_bridge_plus_minus(protein, ligand)
    plus['resname'][~np.in1d(plus['resname'], amino_acids)] = ''
    np.add.at(IFP, [np.searchsorted(amino_acids, plus['resname']), 5], 1)

    # ionic bond with protein anion(Column = 6)
    _, minus = salt_bridge_plus_minus(ligand, protein)
    minus['resname'][~np.in1d(minus['resname'], amino_acids)] = ''
    np.add.at(IFP, [np.searchsorted(amino_acids, minus['resname']), 6], 1)

    # ionic bond with metal ion (Column = 7)
    _, metal, strict2 = acceptor_metal(protein, ligand)
    metal['resname'][~np.in1d(metal['resname'], amino_acids)] = ''
    if strict is False:
        strict2 = None
    np.add.at(IFP, [np.searchsorted(
        amino_acids, metal[strict2]['resname']), 7], 1)

    return IFP.flatten()


# ranges for hashing function
MIN_HASH_VALUE = 0
MAX_HASH_VALUE = 2 ** 32


def hash32(value):
    """Platform independend 32bit hashing method"""
    return hash(value) & 0xffffffff


def _ECFP_atom_repr(mol, idx, use_pharm_features=False):
    """Simple description of atoms used in ECFP/FCFP. Bonds are not described
    accounted for. Hydrogens are explicitly forbidden, they raise Exception.

    Reference:
    Rogers D, Hahn M. Extended-connectivity fingerprints. J Chem Inf Model.
    2010;50: 742–754. http://dx.doi.org/10.1021/ci100050t

    Parameters
    ----------
    mol : oddt.toolkit.Molecule object
        Input molecule for the FP calculations

    idx : int
        Root atom index (0-based).

    use_pharm_features : bool (default=False)
        Switch to use pharmacophoric features as atom representation instead of
        explicit atomic numbers etc.

    Returns
    -------
    atom_repr : tuple (size=6 or 7)
        Atom type desctiption or pharmacophoric features of atom.
    """
    if use_pharm_features:
        atom_dict = mol.atom_dict[idx]
        if atom_dict['atomicnum'] == 1:
            raise Exception('ECFP should not hash Hydrogens')
        return (int(atom_dict['isdonor']),
                int(atom_dict['isacceptor']),
                int(atom_dict['ishydrophobe']),
                int(atom_dict['isplus']),
                int(atom_dict['isminus']),
                int(atom_dict['isaromatic']))

    else:
        if oddt.toolkit.backend == 'ob':
            atom = mol.OBMol.GetAtom(idx + 1)
            if atom.GetAtomicNum() == 1:
                raise Exception('ECFP should not hash Hydrogens')
            return (atom.GetAtomicNum(),
                    atom.GetIsotope(),
                    atom.GetHvyValence(),
                    atom.ImplicitHydrogenCount() + atom.ExplicitHydrogenCount(),
                    atom.GetFormalCharge(),
                    int(atom.IsInRing()),
                    int(atom.IsAromatic()),)
        else:
            atom = mol.Mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() == 1:
                raise Exception('ECFP should not hash Hydrogens')
            return (atom.GetAtomicNum(),
                    atom.GetIsotope(),
                    atom.GetTotalDegree() - atom.GetTotalNumHs(includeNeighbors=True),
                    atom.GetTotalNumHs(includeNeighbors=True),
                    atom.GetFormalCharge(),
                    int(atom.IsInRing()),
                    int(atom.GetIsAromatic()),)


def _ECFP_atom_hash(mol, idx, depth=2, use_pharm_features=False):
    """Generate hashed environments for single atom up to certain depth
    (bond-wise). Hydrogens are ignored during neighbor lookup.

    Reference:
    Rogers D, Hahn M. Extended-connectivity fingerprints. J Chem Inf Model.
    2010;50: 742–754. http://dx.doi.org/10.1021/ci100050t

    Parameters
    ----------
    mol : oddt.toolkit.Molecule object
        Input molecule for the FP calculations

    idx : int
        Root atom index (0-based).

    depth : int (deafult = 2)
        The depth of the fingerprint, i.e. the number of bonds in Morgan
        algorithm. Note: For ECFP2: depth = 1, ECFP4: depth = 2, etc.

    use_pharm_features : bool (default=False)
        Switch to use pharmacophoric features as atom representation instead of
        explicit atomic numbers etc.

    Returns
    -------
    environment_hashes : list of ints
        Hashed environments for certain atom
    """
    atom_env = [[idx]]
    for r in range(1, depth + 1):
        prev_atom_env = atom_env[r - 1]
        if r > 2:  # prune visited atoms
            prev_atom_env = prev_atom_env[len(atom_env[r - 2]):]
        tmp = []
        for atom_idx in prev_atom_env:
            # Toolkit independent version (slower 30%)
            # for neighbor in mol.atoms[atom_idx].neighbors:
            #     if neighbor.atomicnum == 1:
            #         continue
            #     n_idx = neighbor.idx - 1  # atom.idx is 1-based!
            #     if (n_idx not in atom_env[r - 1] and n_idx not in tmp):
            #         tmp.append(n_idx)
            if oddt.toolkit.backend == 'ob':
                for neighbor in oddt.toolkit.OBAtomAtomIter(mol.OBMol.GetAtom(atom_idx + 1)):
                    if neighbor.GetAtomicNum() == 1:
                        continue
                    n_idx = neighbor.GetIdx() - 1
                    if (n_idx not in atom_env[r - 1] and n_idx not in tmp):
                        tmp.append(n_idx)
            else:
                for neighbor in mol.Mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                    if neighbor.GetAtomicNum() == 1:
                        continue
                    n_idx = neighbor.GetIdx()
                    if (n_idx not in atom_env[r - 1] and n_idx not in tmp):
                        tmp.append(n_idx)
        atom_env.append(atom_env[r - 1] + tmp)

    # Get atom representation only once, pull indices from largest env
    atom_repr = [_ECFP_atom_repr(mol, aidx, use_pharm_features=use_pharm_features)
                 for aidx in atom_env[-1]]
    # Get atom invariants
    out_hash = []
    for layer in atom_env:
        layer_invariant = tuple(sorted([a_repr for aidx, a_repr in zip(layer, atom_repr)]))
        out_hash.append(hash32(layer_invariant))
    return out_hash


def ECFP(mol, depth=2, size=4096, count_bits=True, sparse=True,
         use_pharm_features=False):
    """Extended connectivity fingerprints (ECFP) with an option to include
    atom features (FCPF). Depth of a fingerprint is counted as bond-steps, thus
    the depth for ECFP2 = 1, ECPF4 = 2, ECFP6 = 3, etc.

    Reference:
    Rogers D, Hahn M. Extended-connectivity fingerprints. J Chem Inf Model.
    2010;50: 742–754. http://dx.doi.org/10.1021/ci100050t

    Parameters
    ----------
    mol : oddt.toolkit.Molecule object
        Input molecule for the FP calculations

    depth : int (deafult = 2)
        The depth of the fingerprint, i.e. the number of bonds in Morgan
        algorithm. Note: For ECFP2: depth = 1, ECFP4: depth = 2, etc.

    size : int (default = 4096)
        Final size of fingerprint to which it is folded.

    count_bits : bool (default = True)
        Should the bits be counted or unique. In dense representation it
        translates to integer array (count_bits=True) or boolean array if False.

    sparse : bool (default=True)
        Should fingerprints be dense (contain all bits) or sparse (just the on
        bits).

    use_pharm_features : bool (default=False)
        Switch to use pharmacophoric features as atom representation instead of
        explicit atomic numbers etc.

    Returns
    -------
    fingerprint : numpy array
        Calsulated FP of fixed size (dense) or on bits indices (sparse). Dtype
        is either integer or boolean.
    """
    # Hash atom environments
    mol_hashed = []
    for idx, atom in enumerate(mol.atoms):
        if atom.atomicnum == 1:
            continue
        mol_hashed.append(_ECFP_atom_hash(mol, idx, depth=depth,
                                          use_pharm_features=use_pharm_features))
    mol_hashed = np.array(sorted(chain(*mol_hashed)))

    # folding
    mol_hashed = np.floor((mol_hashed.astype(np.float64) - MIN_HASH_VALUE) /
                          int(abs(MAX_HASH_VALUE - MIN_HASH_VALUE) / size))

    # cast to minimum unsigned integer dtype
    mol_hashed = mol_hashed.astype(np.min_scalar_type(size))

    if not count_bits:
        mol_hashed = np.unique(mol_hashed)

    # dense or sparse FP
    if not sparse:
        tmp = np.zeros(size, dtype=np.uint8 if count_bits else bool)
        np.add.at(tmp, mol_hashed, 1)
        mol_hashed = tmp

    return mol_hashed


def dice(a, b, sparse=False):
    """Calculates the Dice coefficient, the ratio of the bits in common to
        the arithmetic mean of the number of 'on' bits in the two fingerprints.
        Supports integer and boolean fingerprints.

        Parameters
        ----------
        a, b : numpy array
            Interaction fingerprints to check similarity between them.

        sparse : bool (default=False)
            Type of FPs to use. Defaults to dense form.

        Returns
        -------
        score : float
            Similarity between a, b.

    """
    if sparse:
        a_unique, a_counts = np.unique(a, return_counts=True)
        b_unique, b_counts = np.unique(b, return_counts=True)
        a_b_intersection = np.intersect1d(a_unique, b_unique, assume_unique=True)
        a_b = np.minimum(a_counts[np.in1d(a_unique, a_b_intersection)],
                         b_counts[np.in1d(b_unique, a_b_intersection)]).sum()
        return 2 * a_b.astype(float) / (len(a) + len(b))
    else:
        a_b = np.vstack((a, b)).min(axis=0).sum()
        return 2 * a_b.astype(float) / (a.sum() + b.sum())


def tanimoto(a, b, sparse=False):
    """
        Tanimoto coefficient, supports boolean fingerprints.
        Integer fingerprints are casted to boolean.

        Parameters
        ----------
        a, b : numpy array
            Interaction fingerprints to check similarity between them.

        sparse : bool (default=False)
            Type of FPs to use. Defaults to dense form.

        Returns
        -------
        score : float
            Similarity between a, b.

    """

    if sparse:
        a = np.unique(a)
        b = np.unique(b)
        a_b = float(len(np.intersect1d(a, b, assume_unique=True)))
        return a_b / (len(a) + len(b) - a_b)
    else:
        a = a.astype(bool)
        b = b.astype(bool)
        a_b = (a & b).sum().astype(float)
        return a_b / (a.sum() + b.sum() - a_b)
