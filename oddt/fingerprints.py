"""
    Module checks interactions between two molecules and
    creates interacion fingerprints.

"""

import numpy as np
from oddt.interactions import pi_stacking, pi_cation, \
    hbond_acceptor_donor, salt_bridge_plus_minus, hydrophobic_contacts, acceptor_metal

__all__ = ['InteractionFingerprint',
           'SimpleInteractionFingerprint', 'dice', 'tc']


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


def dice(a, b):
    """
        Calculates the Dice coefficient, the ratio of the bits in common to
        the arithmetic mean of the number of 'on' bits in the two fingerprints.
        Supports integer and boolean fingerprints.

        Parameters
        ----------
        a, b : numpy array
            Interaction fingerprints to check similarity between them.

        Returns
        -------
        score : float
            Similarity between a, b.

    """
    return 2 * np.vstack((a, b)).min(axis=0).sum().astype(float) / (a.sum() + b.sum())


def tanimoto(a, b):
    """
        Tanimoto coefficient, supports boolean fingerprints.
        Integer fingerprints are casted to boolean.

        Parameters
        ----------
        a, b : numpy array
            Interaction fingerprints to check similarity between them.

        Returns
        -------
        score : float
            Similarity between a, b.

    """
    a = a.astype(bool)
    b = b.astype(bool)
    a_b = (a * b).sum().astype(float)
    return a_b / (a.sum() + b.sum() - a_b)
