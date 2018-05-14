from __future__ import division, print_function
import os
from collections import OrderedDict
from itertools import combinations, chain
import sys
from io import BytesIO

from six.moves import urllib

from distutils.version import LooseVersion

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import rdkit
from rdkit import Chem
from rdkit.Chem.AllChem import ConstrainedEmbed
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField


METALS = (3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
          37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 56, 57,
          58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
          75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92, 93, 94,
          95, 96, 97, 98, 99, 100, 101, 102, 103)


def PathFromAtomList(mol, amap):
    out = []
    for i, j in combinations(amap, 2):
        bond = mol.GetBondBetweenAtoms(i, j)
        if bond:
            out.append(bond.GetIdx())
    return out


def AtomListToSubMol(mol, amap, includeConformer=False):
    """
    Parameters
    ----------
        mol: rdkit.Chem.rdchem.Mol
            Molecule
        amap: array-like
            List of atom indices (zero-based)
        includeConformer: bool (default=True)
            Toogle to include atoms coordinates in submolecule.

    Returns
    -------
        submol: rdkit.Chem.rdchem.RWMol
            Submol determined by specified atom list
    """
    if not isinstance(amap, list):
        amap = list(amap)
    submol = Chem.RWMol()
    for aix in amap:
        submol.AddAtom(mol.GetAtomWithIdx(aix))
    for i, j in combinations(amap, 2):
        bond = mol.GetBondBetweenAtoms(i, j)
        if bond:
            submol.AddBond(amap.index(i),
                           amap.index(j),
                           bond.GetBondType())
    if includeConformer:
        for conf in mol.GetConformers():
            new_conf = Chem.Conformer(len(amap))
            for i in range(len(amap)):
                new_conf.SetAtomPosition(i, conf.GetAtomPosition(amap[i]))
            submol.AddConformer(new_conf)
    return submol


def MolToTemplates(mol):
    """Prepare set of templates for a given PDB residue."""

    if mol.HasProp('_Name') and mol.GetProp('_Name') in ['DA', 'DG', 'DT', 'DC',
                                                         'A', 'G', 'T', 'C', 'U']:
        backbone = 'OP(=O)(O)OC'
    else:
        backbone = 'OC(=O)CN'

    match = mol.GetSubstructMatch(Chem.MolFromSmiles(backbone))
    mol2 = Chem.RWMol(mol)
    if match:
        mol2.RemoveAtom(match[0])

    Chem.SanitizeMol(mol2)
    mol2 = mol2.GetMol()
    return (mol, mol2)


def ReadTemplates(filename, resnames):
    """Load templates from file for specified residues"""

    template_mols = {}

    with open(filename) as f:
        for n, line in enumerate(f):
            data = line.split()
            # TODO: skip all residues that have 1 heavy atom
            if data[1] in resnames and data[1] != 'HOH':  # skip waters
                res = Chem.MolFromSmiles(data[0])
                res.SetProp('_Name', data[1])  # Needed for residue type lookup
                template_mols[data[1]] = MolToTemplates(res)

    return template_mols


def SimplifyMol(mol):
    """Change all bonds to single and discharge/dearomatize all atoms.
    The molecule is modified in-place (no copy is made).
    """
    for b in mol.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in mol.GetAtoms():
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)
    return mol


def UFFConstrainedOptimize(mol, moving_atoms=None, fixed_atoms=None,
                           cutoff=10.):
    """Minimize a molecule using UFF forcefield with a set of moving/fixed
    atoms. If both moving and fixed atoms are provided, fixed_atoms parameter
    will be ignored.  The minimization is done in-place (without copying
    molecule).

    Parameters
    ----------
        mol: rdkit.Chem.rdchem.Mol
            Molecule to be minimized.
        moving_atoms: array-like (default=None)
            Indices of freely moving atoms. If None, fixed atoms are assigned
            based on `fixed_atoms`. These two arguments are mutually exclusive.
        fixed_atoms: array-like (default=None)
            Indices of fixed atoms. If None, fixed atoms are assigned based on
            `moving_atoms`. These two arguments are mutually exclusive.
        cutoff: float (default=10.)
            Distance cutoff for the UFF minimization

    Returns
    -------
        mol: rdkit.Chem.rdchem.Mol
            Molecule with mimimized `moving_atoms`
    """
    if moving_atoms is None and fixed_atoms is None:
        raise ValueError('You must supply at least one set of moving/fixed '
                         'atoms.')

    all_atoms = set(range(mol.GetNumAtoms()))
    if moving_atoms is None:
        moving_atoms = list(all_atoms.difference(fixed_atoms))
    else:
        fixed_atoms = list(all_atoms.difference(moving_atoms))
    # extract submolecules containing atoms within cutoff
    pos = np.array(mol.GetConformer(-1).GetPositions())
    mask = (cdist(pos, pos[moving_atoms]) <= cutoff).any(axis=1)
    amap = np.where(mask)[0].tolist()
    # TODO: expand to whole residues?

    # TODO: above certain threshold its making a submolis redundant
    submol = AtomListToSubMol(mol, amap, includeConformer=True)
    # initialize ring info
    Chem.GetSSSR(submol)
    ff = UFFGetMoleculeForceField(submol, vdwThresh=cutoff,
                                  ignoreInterfragInteractions=False)
    for i in fixed_atoms:
        if i in amap:
            ff.AddFixedPoint(amap.index(i))
    ff.Initialize()
    ff.Minimize(energyTol=1e-2, forceTol=1e-2, maxIts=200)

    # get the positions backbone
    conf = mol.GetConformer(-1)
    for idx, pos in zip(amap, submol.GetConformer(-1).GetPositions()):
        conf.SetAtomPosition(idx, pos)
    return mol


def ExtractPocketAndLigand(mol, cutoff=12., expandResidues=True,
                           ligand_residue=None, ligand_residue_blacklist=None,
                           append_residues=None):
    """Function extracting a ligand (the largest HETATM residue) and the protein
    pocket within certain cutoff. The selection of pocket atoms can be expanded
    to contain whole residues. The single atom HETATM residues are attributed
    to pocket (metals and waters)

    Parameters
    ----------
        mol: rdkit.Chem.rdchem.Mol
            Molecule with a protein ligand complex
        cutoff: float (default=12.)
            Distance cutoff for the pocket atoms
        expandResidues: bool (default=True)
            Expand selection to whole residues within cutoff.
        ligand_residue: string (default None)
            Residue name which explicitly pint to a ligand(s).
        ligand_residue_blacklist: array-like, optional (default None)
            List of residues to ignore during ligand lookup.
        append_residues: array-like, optional (default None)
            List of residues to append to pocket, even if they are HETATM, such
            as MSE, ATP, AMP, ADP, etc.

    Returns
    -------
        pocket: rdkit.Chem.rdchem.RWMol
            Pocket constructed of protein residues/atoms around ligand
        ligand: rdkit.Chem.rdchem.RWMol
            Largest HETATM residue contained in input molecule
    """
    # Get heteroatom residues - connectivity still might be wrong, so GetFrags will fail
    # Use OrderDict, so that A chain is prefered first over B if ligands are equal
    hetatm_residues = OrderedDict()
    protein_residues = OrderedDict()
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        res_id = GetAtomResidueId(atom)
        if info.GetIsHeteroAtom():
            if res_id not in hetatm_residues:
                hetatm_residues[res_id] = []
            hetatm_residues[res_id].append(atom.GetIdx())
        else:
            if res_id not in protein_residues:
                protein_residues[res_id] = []
            protein_residues[res_id].append(atom.GetIdx())

    # check if desired ligand residue is present
    if ligand_residue is not None and ligand_residue not in hetatm_residues:
        ValueError('Threre is no residue named "%s" in the protein file' %
                   ligand_residue)

    for res_id in list(hetatm_residues.keys()):  # exhaust keys since we modify
        # Treat single atom residues (waters + metals) as pocket residues
        # Also append listed residues to protein
        if (len(hetatm_residues[res_id]) == 1 or
                append_residues is not None and res_id[1] in append_residues):
            protein_residues[res_id] = hetatm_residues[res_id]
            del hetatm_residues[res_id]
        # leave only the desired residues
        elif ligand_residue is not None and res_id[1] != ligand_residue:
            del hetatm_residues[res_id]
        # remove blacklisted residues
        elif (ligand_residue_blacklist is not None and
              res_id[1] in ligand_residue_blacklist):
                    del hetatm_residues[res_id]

    if len(hetatm_residues) == 0:
        raise ValueError('No ligands')

    # Take largest ligand
    ligand_key = sorted(hetatm_residues, key=lambda x: len(hetatm_residues[x]),
                        reverse=True)[0]
    ligand_amap = hetatm_residues[ligand_key]
    ligand = AtomListToSubMol(mol, ligand_amap, includeConformer=True)
    ligand_coords = np.array(ligand.GetConformer().GetPositions())

    # Get protein and waters
    blacklist_ids = list(chain(*hetatm_residues.values()))
    protein_amap = np.array([i for i in range(mol.GetNumAtoms())
                             if i not in blacklist_ids])
    protein_coords = np.array(mol.GetConformer().GetPositions())[protein_amap]

    # Pocket selection based on cutoff
    mask = (cdist(protein_coords, ligand_coords) <= cutoff).any(axis=1)
    # IDs of atoms within cutoff
    pocket_amap = protein_amap[np.where(mask)[0]].tolist()

    # Expand pocket's residues
    if expandResidues:
        pocket_residues = OrderedDict()
        for res_id in protein_residues.keys():
            if any(1 for res_aix in protein_residues[res_id]
                   if res_aix in pocket_amap):
                pocket_residues[res_id] = protein_residues[res_id]
        pocket_amap = list(chain(*pocket_residues.values()))

    # Create pocket mol, pocket_amap needs to be mapped to mol Idxs
    pocket = AtomListToSubMol(mol, pocket_amap, includeConformer=True)

    return pocket, ligand


def GetAtomResidueId(atom):
    """Return (residue number, residue name, chain id) for a given atom"""
    info = atom.GetPDBResidueInfo()
    res_id = (info.GetResidueNumber(), info.GetResidueName().strip(),
              info.GetChainId())
    return res_id


def GetResidues(mol, atom_list=None):
    """Create dictrionary that maps residues to atom IDs:
    (res number, res name, chain id) --> [atom1 idx, atom2 idx, ...]
    """

    residues = OrderedDict()

    if atom_list is None:
        atom_list = range(mol.GetNumAtoms())

    for aid in atom_list:
        res_id = GetAtomResidueId(mol.GetAtomWithIdx(aid))
        if res_id not in residues:
            residues[res_id] = []
        residues[res_id].append(aid)

    return residues


def PreparePDBResidue(protein, residue, amap, template):
    """
    Parameters
    ----------
        protein: rdkit.Chem.rdchem.RWMol
            Mol with whole protein. Note that it is modified in place.
        residue:
            Mol with residue only
        amap: list
            List mapping atom IDs in residue to atom IDs in whole protein
            (amap[i] = j means that i'th atom in residue corresponds to j'th
            atom in protein)
        template:
            Residue template
    Returns
    -------
        protein: rdkit.Chem.rdchem.RWMol
            Modified protein
        visited_bonds: list
            Bonds that match the template
        is_complete: bool
            Indicates whether all atoms in template were found in residue
    """

    # Catch residues which have less than 4 atoms (i.e. cannot have complete
    # backbone), and template has more atoms than that, or residues with
    # many missing atoms, which lead to low number of bonds (less than 3)
    if ((len(amap) < 4 or residue.GetNumBonds() < 3) and
            template.GetNumAtoms() > 4):
        raise ValueError('The residue "%s" has incomplete backbone'
                         % template.GetProp('_Name'),
                         Chem.MolToSmiles(template),
                         Chem.MolToSmiles(residue))

    # modify copies instead of original molecules
    template2 = Chem.Mol(template)
    residue2 = Chem.Mol(residue)

    visited_bonds = []

    # do the molecules match already?
    match = residue2.GetSubstructMatch(template2)
    if not match:  # no, they don't match
        residue2 = SimplifyMol(residue2)
        template2 = SimplifyMol(template2)
        # match is either tuple (if match was complete) or dict (if match
        # was partial)
        match = residue2.GetSubstructMatch(template2)

    # try inverse match
    if not match:
        inverse_match = template.GetSubstructMatch(residue)
        # if it failed try to match modified molecules (single bonds,
        # no charges, no aromatic atoms)
        if not inverse_match:
            inverse_match = template2.GetSubstructMatch(residue2)
        if inverse_match:
            match = (dict(zip(inverse_match, range(len(inverse_match)))))

    # do the molecules match now?
    if match:

        assert len(match) <= len(amap), \
            'matching is bigger than amap for %s' \
            '(%s / %s vs %s; %s atoms vs %s atoms)' % (
                template.GetProp('_Name'),
                Chem.MolToSmiles(template),
                Chem.MolToSmiles(template2),
                Chem.MolToSmiles(residue),
                residue.GetNumAtoms(),
                template.GetNumAtoms(),
        )

        # Convert matches to dict to support partial match, where keys
        # are not complete sequence, as in full match.
        if isinstance(match, (tuple, list)):
            match = dict(zip(range(len(match)), match))

        # apply matching: set bond properties
        for (atom1, atom2), (refatom1, refatom2) in \
            zip(combinations(match.values(), 2),
                combinations(match.keys(), 2)):

            b = template.GetBondBetweenAtoms(refatom1, refatom2)

            b2 = protein.GetBondBetweenAtoms(amap[atom1], amap[atom2])
            # remove extra bonds
            if b is None:
                if b2:  # this bond is not there
                    protein.RemoveBond(amap[atom1], amap[atom2])
                continue
            # add missing bonds
            if b2 is None:
                protein.AddBond(amap[atom1], amap[atom2])
                b2 = protein.GetBondBetweenAtoms(amap[atom1], amap[atom2])
            # set bond properties
            b2.SetBondType(b.GetBondType())
            b2.SetIsAromatic(b.GetIsAromatic())
            visited_bonds.append((amap[atom1], amap[atom2]))

        # apply matching: set atom properties
        for a in template.GetAtoms():
            if a.GetIdx() not in match:
                continue
            a2 = protein.GetAtomWithIdx(amap[match[a.GetIdx()]])
            a2.SetHybridization(a.GetHybridization())

            # partial match may not close ring, so set aromacity only if
            # atom is in ring
            if a2.IsInRing():
                a2.SetIsAromatic(a.GetIsAromatic())
            # TODO: check for connected Hs
            # n_hs = sum(n.GetAtomicNum() == 1 for n in a2.GetNeighbors())
            a2.SetNumExplicitHs(a.GetNumExplicitHs())
            a2.SetFormalCharge(a.GetFormalCharge())
            # Update computed properties for an atom
            a2.UpdatePropertyCache(strict=False)
        if len(match) < template.GetNumAtoms():
            is_complete = False
            # TODO: replace following with warning/logging
            # Get atom map of fixed fragment
            amap_frag = [amap[match[a.GetIdx()]]
                         for a in template.GetAtoms()
                         if a.GetIdx() in match]
            info = protein.GetAtomWithIdx(amap_frag[0]).GetPDBResidueInfo()
            print('Partial match. Probably incomplete sidechain.',
                  template.GetProp('_Name'),
                  Chem.MolToSmiles(template),
                  Chem.MolToSmiles(template2),
                  Chem.MolToSmiles(residue),
                  Chem.MolToSmiles(AtomListToSubMol(protein, amap_frag)),
                  info.GetResidueName(),
                  info.GetResidueNumber(),
                  info.GetChainId(),
                  sep='\t', file=sys.stderr)
        else:
            is_complete = True
    else:
        # most common missing sidechain AA
        msg = 'No matching found'
        raise ValueError(msg,
                         template.GetProp('_Name'),
                         Chem.MolToSmiles(template),
                         Chem.MolToSmiles(template2),
                         Chem.MolToSmiles(residue),
                         )

    return protein, visited_bonds, is_complete


def AddMissingAtoms(protein, residue, amap, template):
    """Add missing atoms to protein molecule only at the residue according to
    template.

    Parameters
    ----------
        protein: rdkit.Chem.rdchem.RWMol
            Mol with whole protein. Note that it is modified in place.
        residue:
            Mol with residue only
        amap: list
            List mapping atom IDs in residue to atom IDs in whole protein
            (amap[i] = j means that i'th atom in residue corresponds to j'th
            atom in protein)
        template:
            Residue template
    Returns
    -------
        protein: rdkit.Chem.rdchem.RWMol
            Modified protein
        visited_bonds: list
            Bonds that match the template
        is_complete: bool
            Indicates whether all atoms in template were found in residue
    """
    # TODO: try to better guess the types of atoms (if possible)

    # we need the match anyway and ConstrainedEmbed does not outputs it
    matched_atoms = template.GetSubstructMatch(residue)
    if matched_atoms:  # instead of catching ValueError
        fixed_residue = ConstrainedEmbed(template, residue)
    else:
        residue2 = SimplifyMol(Chem.Mol(residue))
        template2 = SimplifyMol(Chem.Mol(template))
        matched_atoms = template2.GetSubstructMatch(residue2)
        if matched_atoms:
            fixed_residue = ConstrainedEmbed(template2, residue2)
            # copy coordinates to molecule with appropriate bond orders
            fixed_residue2 = Chem.Mol(template)
            fixed_residue2.RemoveAllConformers()
            fixed_residue2.AddConformer(fixed_residue.GetConformer(-1))
            fixed_residue = fixed_residue2

            # initial minimization only on residue
            fixed_residue = UFFConstrainedOptimize(fixed_residue,
                                                   fixed_atoms=matched_atoms)
        else:
            raise ValueError('No matching found at missing atom stage.',
                             template.GetProp('_Name'),
                             Chem.MolToSmiles(template),
                             Chem.MolToSmiles(residue))

    new_atoms = []
    new_amap = []

    info = residue.GetAtomWithIdx(0).GetPDBResidueInfo()
    protein_conformer = protein.GetConformer()
    fixed_conformer = fixed_residue.GetConformer()

    for i in range(fixed_residue.GetNumAtoms()):
        if i not in matched_atoms:
            atom = fixed_residue.GetAtomWithIdx(i)
            # we need to generate atom names like 'H123', these are
            # "wrapped around" below when setting 'atomName' to '3H12'
            atom_symbol = atom.GetSymbol()
            name = (atom_symbol + str(i)[:4-len(atom_symbol)]).ljust(4)
            new_info = Chem.AtomPDBResidueInfo(
                atomName=name[-1:] + name[:-1],  # wrap around
                residueName=info.GetResidueName(),
                residueNumber=info.GetResidueNumber(),
                chainId=info.GetChainId(),
                insertionCode=info.GetInsertionCode(),
                isHeteroAtom=info.GetIsHeteroAtom()
            )

            atom.SetMonomerInfo(new_info)
            new_id = protein.AddAtom(atom)
            new_atoms.append(new_id)
            pos = fixed_conformer.GetAtomPosition(i)
            protein_conformer.SetAtomPosition(new_id, pos)
            new_amap.append(new_id)
        else:
            new_amap.append(amap[matched_atoms.index(i)])

    # add bonds in separate loop (we need all atoms added before that)
    for i in range(fixed_residue.GetNumAtoms()):
        if i not in matched_atoms:
            atom = fixed_residue.GetAtomWithIdx(i)
            for n in atom.GetNeighbors():
                ni = n.GetIdx()
                bond = fixed_residue.GetBondBetweenAtoms(i, ni)
                # for multiple missing atoms we may hit bonds multiple times
                new_bond = protein.GetBondBetweenAtoms(new_amap[i],
                                                       new_amap[ni])
                if new_bond is None:
                    protein.AddBond(new_amap[i], new_amap[ni])
                    new_bond = protein.GetBondBetweenAtoms(new_amap[i],
                                                           new_amap[ni])
                    new_bond.SetBondType(bond.GetBondType())

    if new_atoms:
        backbone_definitions = [
            # Phosphodiester Bond
            {'smarts': Chem.MolFromSmiles('O=P(O)OCC1OC(CC1O)'),
             'atom_types': {0: 'OP1', 1: 'P', 2: 'OP2', 3: 'O5\'', 4: 'C5\'',
                            5: 'C4\'', 9: 'C3\'', 10: 'O3\''},
             'bond_pair': ('O3\'', 'P')},
            # Peptide Bond
            {'smarts': Chem.MolFromSmiles('C(=O)CN'),
             'atom_types': {0: 'C', 1: 'O', 2: 'CA', 3: 'N'},
             'bond_pair': ('C', 'N')},
        ]
        info = residue.GetAtomWithIdx(0).GetPDBResidueInfo()
        res_num = info.GetResidueNumber()
        res_chain = info.GetChainId()

        for bond_def in backbone_definitions:
            backbone_match = fixed_residue.GetSubstructMatch(bond_def['smarts'])
            if backbone_match:
                for i in new_atoms:
                    if new_amap.index(i) in backbone_match:
                        atom = protein.GetAtomWithIdx(i)
                        match_idx = backbone_match.index(new_amap.index(i))
                        if match_idx not in bond_def['atom_types']:
                            # if atom type is not defined we can skip that atom
                            continue

                        # Set atom label if present in backbone definition
                        match_type = bond_def['atom_types'][match_idx]
                        atom.GetPDBResidueInfo().SetName(' ' + match_type.ljust(3))

                        # define upstream and downstream bonds
                        bonds = zip([bond_def['bond_pair'],
                                     reversed(bond_def['bond_pair'])],
                                    [1, -1])
                        for (a1, a2), diff in bonds:
                            if match_type == a1:
                                limit = max(-1, protein.GetNumAtoms() * diff)
                                for j in range(amap[0], limit, diff):
                                    info = (protein.GetAtomWithIdx(j)
                                            .GetPDBResidueInfo())
                                    res2_num = info.GetResidueNumber()
                                    res2_chain = info.GetChainId()
                                    if (res2_num == res_num + diff
                                            and res_chain == res2_chain):
                                        if info.GetName().strip() == a2:
                                            protein.AddBond(i, j, Chem.BondType.SINGLE)
                                            break
                                    elif (abs(res2_num - res_num) > 1
                                          or res_chain != res2_chain):
                                        break

    # run PreparePDBResidue to fix atom properies
    out = PreparePDBResidue(protein, fixed_residue, new_amap, template)
    return out + (new_atoms,)


def PreparePDBMol(mol,
                  removeHs=True,
                  removeHOHs=True,
                  residue_whitelist=None,
                  residue_blacklist=None,
                  remove_incomplete=False,
                  add_missing_atoms=False,
                  custom_templates=None,
                  replace_default_templates=False,
                  ):
    """Prepares protein molecule by:
        - Removing Hs by hard using atomic number [default=True]
        - Removes HOH [default=True]
        - Assign bond orders from smiles of PDB residues (over 24k templates)
        - Removes bonds to metals

    Parameters
    ----------
        mol: rdkit.Chem.rdchem.Mol
            Mol with whole protein.
        removeHs: bool, optional (default True)
            If True, hydrogens will be forcefully removed
        removeHOHs: bool, optional (default True)
            If True, remove waters using residue name
        residue_whitelist: array-like, optional (default None)
            List of residues to clean. If not specified, all residues
            present in the structure will be used.
        residue_blacklist: array-like, optional (default None)
            List of residues to ignore during cleaning. If not specified,
            all residues present in the structure will be cleaned.
        remove_incomplete: bool, optional (default False)
            If True, remove residues that do not fully match the template
        add_missing_atoms: bool (default=False)
            Switch to add missing atoms accordingly to template SMILES structure.
        custom_templates: str or dict, optional (default None)
            Custom templates for residues. Can be either path to SMILES file,
            or dictionary mapping names to SMILES or Mol objects
        replace_default_templates: bool, optional (default False)
            Indicates whether default default templates should be replaced by
            cusom ones. If False, default templates will be updated with custom
            ones. This argument is ignored if custom_templates is None.

    Returns
    -------
        new_mol: rdkit.Chem.rdchem.RWMol
            Modified protein
    """

    new_mol = Chem.RWMol(mol)
    if removeHs:
        for i in reversed(range(new_mol.GetNumAtoms())):
            atom = new_mol.GetAtomWithIdx(i)
            if atom.GetAtomicNum() == 1:
                new_mol.RemoveAtom(i)

    if removeHOHs:
        for i in reversed(range(new_mol.GetNumAtoms())):
            atom = new_mol.GetAtomWithIdx(i)
            if atom.GetPDBResidueInfo().GetResidueName() == 'HOH':
                new_mol.RemoveAtom(i)

    # list of unique residues and their atom indices
    unique_resname = set()
    residues_atom_map = GetResidues(new_mol)

    # create a list of residue mols with atom maps
    residues = []
    # residue_id == (res number, res name, chain id)
    for residue_id, amap in residues_atom_map.items():
        unique_resname.add(residue_id[1].strip())
        # skip waters
        if residue_id[1] != 'HOH':
            res = AtomListToSubMol(new_mol, amap, includeConformer=True)
            residues.append((residue_id, res, amap))

    # load cutom templates
    if custom_templates is not None:
        if isinstance(custom_templates, str):
            custom_mols = ReadTemplates(custom_templates, unique_resname)
        elif isinstance(custom_templates, dict):
            custom_mols = {}
            for resname, structure in custom_templates.items():
                if isinstance(structure, str):
                    structure = Chem.MolFromSmiles(structure)
                    structure.SetProp('_Name', resname)
                custom_mols[resname] = MolToTemplates(structure)
        else:
            raise TypeError('custom_templates should be file name on dict,'
                            ' %s was given' % type(custom_templates))

    if custom_templates is None or not replace_default_templates:
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'pdb_residue_templates.smi')
        template_mols = ReadTemplates(filename, unique_resname)
    else:
        template_mols = {}

    if custom_templates is not None:
        if replace_default_templates:
            template_mols = custom_mols
        else:
            template_mols.update(custom_mols)

    # Deal with residue lists
    if residue_whitelist is not None:
        unique_resname = set(residue_whitelist)
    if residue_blacklist is not None:
        unique_resname = unique_resname.difference(set(residue_blacklist))
    unique_resname = tuple(map(lambda x: x.strip().upper(), unique_resname))

    # reset B.O. using templates
    visited_bonds = []
    new_atoms = []
    atoms_to_del = []
    for ((resnum, resname, chainid), residue, amap) in residues:
        if resname not in unique_resname:
            continue
        if resname not in template_mols:
            raise ValueError('There is no template for residue "%s"' % resname)
        template_raw, template_chain = template_mols[resname]
        if residue.GetNumAtoms() > template_chain.GetNumAtoms():
            template = template_raw
        else:
            template = template_chain

        bonds = []
        atoms = []
        complete_match = False
        try:
            new_mol, bonds, complete_match = PreparePDBResidue(new_mol,
                                                               residue,
                                                               amap,
                                                               template)
            if add_missing_atoms and not complete_match:
                new_mol, bonds, complete_match, atoms = AddMissingAtoms(new_mol,
                                                                        residue,
                                                                        amap,
                                                                        template)
        except ValueError as e:
            print(resnum, resname, chainid, e, file=sys.stderr)
        finally:
            visited_bonds.extend(bonds)
            if remove_incomplete and not complete_match:
                atoms_to_del.extend(amap)
            else:
                new_atoms.extend(atoms)

    # HACK: remove not-visited bonds
    if visited_bonds:  # probably we dont want to delete all
        new_mol = Chem.RWMol(new_mol)
        visited_bonds = set(visited_bonds)
        bonds_queue = []
        for bond in new_mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            if (a1, a2) not in visited_bonds and (a2, a1) not in visited_bonds:
                bonds_queue.append((a1, a2))
        for a1_ix, a2_ix in bonds_queue:
            bond = new_mol.GetBondBetweenAtoms(a1_ix, a2_ix)
            a1 = new_mol.GetAtomWithIdx(a1_ix)
            a2 = new_mol.GetAtomWithIdx(a2_ix)
            # get residue number
            a1_num = a1.GetPDBResidueInfo().GetResidueNumber()
            a2_num = a2.GetPDBResidueInfo().GetResidueNumber()
            # get PDB atom names
            a1_name = a1.GetPDBResidueInfo().GetName().strip()
            a2_name = a2.GetPDBResidueInfo().GetName().strip()
            if (a1.GetAtomicNum() > 1 and
                a2.GetAtomicNum() > 1 and
                # don't remove bonds between residues in backbone
                # and sulphur bridges
                not (((a1_name, a2_name) in {('C', 'N'), ('N', 'C'),
                                             ('P', 'O3\''), ('O3\'', 'P')} and
                      abs(a1_num - a2_num) == 1) or  # peptide or DNA bond
                     (a1_name == 'SG' and a2_name == 'SG'))):  # sulphur bridge
                new_mol.RemoveBond(a1_ix, a2_ix)
            else:
                pass
    if atoms_to_del:
        new_mol = Chem.RWMol(new_mol)
        for idx in sorted(atoms_to_del, reverse=True):
            new_mol.RemoveAtom(idx)

    # minimize new atoms
    if new_atoms:
        old_new_mol = Chem.RWMol(new_mol)
        Chem.GetSSSR(new_mol)  # we need to update ring info
        new_mol = UFFConstrainedOptimize(new_mol, moving_atoms=new_atoms)
        print('RMS after minimization of added atoms (%i):' % len(new_atoms),
              Chem.rdMolAlign.AlignMol(new_mol, old_new_mol),
              file=sys.stderr)

    # if missing atoms were added we need to renumber them
    if add_missing_atoms and new_atoms:
        def atom_reorder_repr(i):
            """Generate keys for each atom during sort"""
            atom = new_mol.GetAtomWithIdx(i)
            info = atom.GetPDBResidueInfo()
            return (info.GetChainId(), info.GetResidueNumber(), i)
        order = list(range(new_mol.GetNumAtoms()))
        new_order = sorted(order, key=atom_reorder_repr)
        new_mol = Chem.RenumberAtoms(new_mol, new_order)
    return new_mol


def FetchAffinityTable(pdbids, affinity_types):

    pdb_report_url = 'https://www.rcsb.org/pdb/rest/customReport.csv'
    params = {'pdbids': ','.join(pdbids), 'service': 'wsfile', 'format': 'csv'}

    params['reportName'] = 'Ligands'
    ligand_data = urllib.parse.urlencode(params)
    ligand_data = ligand_data.encode()
    req = urllib.request.Request(pdb_report_url, ligand_data)
    response = urllib.request.urlopen(req)
    report = response.read()
    ligands = pd.read_csv(BytesIO(report))

    ligands = ligands.dropna(subset=['structureId', 'ligandId'])

    params['reportName'] = 'BindingAffinity'
    affinity_data = urllib.parse.urlencode(params)
    affinity_data = affinity_data.encode()
    req = urllib.request.Request(pdb_report_url, affinity_data)
    response = urllib.request.urlopen(req)
    report = response.read()
    affinity = pd.read_csv(BytesIO(report))

    affinity = affinity.rename(columns={'hetId': 'ligandId'})

    ligand_affinity = (
        pd.merge(ligands, affinity, sort=False)
        .drop_duplicates(subset=['structureId', 'ligandId'])
        .dropna(subset=affinity_types, how='all')
        .fillna('')
    )

    for affinity_type in affinity_types:
        ligand_affinity[affinity_type] = (
            ligand_affinity[affinity_type]
            .str
            .split(' ', expand=True)[0]
        )

    columns = ['structureId', 'ligandId', 'ligandFormula',
               'ligandMolecularWeight'] + affinity_types

    return ligand_affinity[columns]


def FetchStructure(pdbid, sanitize=False, removeHs=True):
    req = urllib.request.Request('https://files.rcsb.org/view/%s.pdb' % pdbid)
    response = urllib.request.urlopen(req)
    pdb_block = response.read().decode('utf-8')

    mol = Chem.MolFromPDBBlock(pdb_block, sanitize=sanitize, removeHs=removeHs)

    return mol


def IsResidueConnected(mol, atom_ids):

    residues = set()
    for aid in atom_ids:
        info = mol.GetAtomWithIdx(aid).GetPDBResidueInfo()
        residue = (info.GetResidueNumber(), info.GetResidueName().strip(),
                   info.GetChainId())
        residues.add(residue)
    if len(residues) > 1:
        raise ValueError('Atoms belong to multiple residues:' + str(residues))
    residue = residues.pop()

    to_check = set(atom_ids)
    visited_atoms = set()

    while len(to_check) > 0:
        aid = to_check.pop()
        assert aid not in visited_atoms

        visited_atoms.add(aid)
        atom = mol.GetAtomWithIdx(aid)

        for atom2 in atom.GetNeighbors():
            if atom2.GetIdx() in visited_atoms:
                continue

            info = atom2.GetPDBResidueInfo()
            if residue != (info.GetResidueNumber(),
                           info.GetResidueName().strip(),
                           info.GetChainId()):
                # we got to different residue so it is connected
                return True
            else:
                to_check.add(atom2.GetIdx())

    return False


def PrepareComplexes(pdbids, pocket_dist_cutoff=12., affinity_types=None):
    if affinity_types is None:
        affinity_types = ['Ki', 'Kd', 'EC50', 'IC50']

    affinity_table = FetchAffinityTable(pdbids, affinity_types)

    complexes = {}

    for pdbid, tab in affinity_table.groupby('structureId'):
        complexes[pdbid] = {}
        complex_mol = FetchStructure(pdbid)
        # we need to use fixer with rdkit < 2018
        complex_mol = PreparePDBMol(complex_mol)

        ligand_atoms = {res_name: {} for res_name in tab['ligandId']}
        for atom in complex_mol.GetAtoms():
            info = atom.GetPDBResidueInfo()
            res_name = info.GetResidueName().strip()
            if res_name not in ligand_atoms:
                continue

            res_id = (info.GetResidueNumber(), info.GetChainId())
            if res_id not in ligand_atoms[res_name]:
                ligand_atoms[res_name][res_id] = []
            ligand_atoms[res_name][res_id].append(atom.GetIdx())

        proper_ligands = []

        for res_name, atoms_ids in ligand_atoms.items():
            if not any(IsResidueConnected(complex_mol, atom_list)
                       for atom_list in atoms_ids.values()):
                proper_ligands.append(res_name)

        for res_name in proper_ligands:
            try:
                pocket, ligand = ExtractPocketAndLigand(
                    complex_mol,
                    cutoff=pocket_dist_cutoff,
                    ligand_residue=res_name)
            except:
                print('Cant get pocket and ligand for %s and %s'
                      % (pdbid, res_name))
                continue

            pocket = PreparePDBMol(pocket)
            flag = Chem.SanitizeMol(pocket)
            assert flag == Chem.SanitizeFlags.SANITIZE_NONE, \
                'Cannot sanitize pocket for for %s and %s' % (pdbid, res_name)

            flag = Chem.SanitizeMol(ligand)
            assert flag == Chem.SanitizeFlags.SANITIZE_NONE, \
                'Cannot sanitize ligand for for %s and %s' % (pdbid, res_name)

            affinity_values = (
                tab
                .query('ligandId == @res_name')
                [affinity_types]
                .iloc[0]
            )

            for affinity_type, value in zip(affinity_types, affinity_values):
                if len(value) == 0:
                    continue

                # parse values like ">1000" or "0.5-0.8"
                value = [float(v) for v in value.strip('<>~').split('-')]
                if len(value) == 1:
                    value = value[0]
                else:
                    # it's range, use its middle
                    assert len(value) == 2
                    value = sum(value) / 2
                ligand.SetProp(affinity_type, str(value))

            complexes[pdbid][res_name] = (pocket, ligand)
    return complexes
