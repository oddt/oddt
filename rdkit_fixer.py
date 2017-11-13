from __future__ import division, print_function
import os
from collections import OrderedDict
from itertools import combinations, chain
import sys
from distutils.version import LooseVersion

import numpy as np
from scipy.spatial.distance import cdist

import rdkit
from rdkit import Chem
from rdkit.Chem.AllChem import ConstrainedEmbed


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

    match = mol.GetSubstructMatch(Chem.MolFromSmiles('OC(=O)CN'))
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
    """Change all bonds to single and discharge/dearomatize all atoms"""
    for b in mol.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in mol.GetAtoms():
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)
    return mol


def ExtractPocketAndLigand(mol, cutoff=12., expandResidues=True, confId=-1,
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
        confId: int (default=-1)
            The conformer index for the pocket coordinates. By default the first
            one is used.
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
    # Use OrderDict, so that A chain is prefered (first over B if ligands are equal
    hetatm_residues = OrderedDict()
    protein_residues = OrderedDict()
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        res_id = (info.GetResidueNumber(), info.GetResidueName().strip(),
                  info.GetChainId())
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

    for res_id in list(hetatm_residues.keys()):  # exhaust keys, since we modify
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
    ligand_coords = np.array(ligand.GetConformer(-1).GetPositions())

    # Get protein and waters
    blacklist_ids = list(chain(*hetatm_residues.values()))
    protein_amap = np.array([i for i in range(mol.GetNumAtoms())
                             if i not in blacklist_ids])
    protein_coords = np.array(mol.GetConformer(-1).GetPositions())[protein_amap]

    # Pocket selection based on cutoff
    mask = (cdist(protein_coords, ligand_coords) <= cutoff).any(axis=1)
    pocket_amap = protein_amap[np.where(mask)[0]].tolist()  # ids strictly in within cutoff

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
    # TODO: outpur idxs of added atoms and validate if all of missing atoms were added
    # TODO: minimize all added atoms in context of full protein
    # TODO: add backbone peptide bonds, if they were missing
    # TODO: try to better guess the types of atoms (if possible)

    matched_atoms = template.GetSubstructMatch(residue)
    if matched_atoms:  # instead of catching ValueError
        fixed_residue = ConstrainedEmbed(template, residue)
    else:
        residue2 = SimplifyMol(Chem.Mol(residue))
        template2 = SimplifyMol(Chem.Mol(template))
        matched_atoms = template2.GetSubstructMatch(residue2)
        if matched_atoms:
            fixed_residue = ConstrainedEmbed(template2, residue2)
        else:
            raise ValueError('No matching found at missing atom stage.',
                             template.GetProp('_Name'),
                             Chem.MolToSmiles(template),
                             Chem.MolToSmiles(residue),
                             )

    new_amap = []
    for i in range(fixed_residue.GetNumAtoms()):
        if i not in matched_atoms:
            atom = fixed_residue.GetAtomWithIdx(i)
            info = residue.GetAtomWithIdx(0).GetPDBResidueInfo()
            new_info = Chem.AtomPDBResidueInfo(
                atomName=' %-3s' % atom.GetSymbol(),
                residueName=info.GetResidueName(),
                residueNumber=info.GetResidueNumber(),
                chainId=info.GetChainId(),
                insertionCode=info.GetInsertionCode(),
                isHeteroAtom=info.GetIsHeteroAtom()
            )

            atom.SetMonomerInfo(new_info)
            new_id = protein.AddAtom(atom)
            pos = fixed_residue.GetConformer().GetAtomPosition(i)
            protein.GetConformer().SetAtomPosition(new_id, pos)
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
                new_bond = protein.GetBondBetweenAtoms(new_amap[i], new_amap[ni])
                if new_bond is None:
                    protein.AddBond(new_amap[i], new_amap[ni])
                    new_bond = protein.GetBondBetweenAtoms(new_amap[i], new_amap[ni])
                    new_bond.SetBondType(bond.GetBondType())

    # run PreparePDBResidue again to fix atom properies
    return PreparePDBResidue(protein, fixed_residue, new_amap, template)


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

    if remove_incomplete and add_missing_atoms:
        raise ValueError('Arguments "remove_incomplete" and "add_missing_atoms"'
                         ' are mutually exclusive and cannot be both set to'
                         ' "True".')

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

    # (res number, res name, chain id) --> [atom1 idx, atom2 idx, ...]
    resiues_atom_map = OrderedDict()

    for atom in new_mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        res_id = (info.GetResidueNumber(), info.GetResidueName().strip(),
                  info.GetChainId())
        if res_id not in resiues_atom_map:
            resiues_atom_map[res_id] = []
        resiues_atom_map[res_id].append(atom.GetIdx())
        unique_resname.add(info.GetResidueName().strip())

    # create a list of residue mols with atom maps
    residues = []
    # residue_id == (res number, res name, chain id)
    for residue_id, amap in resiues_atom_map.items():
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
        complete_match = False
        try:
            new_mol, bonds, complete_match = PreparePDBResidue(new_mol,
                                                               residue,
                                                               amap,
                                                               template)
            if add_missing_atoms and not complete_match:
                new_mol, bonds, complete_match = AddMissingAtoms(new_mol,
                                                                 residue,
                                                                 amap,
                                                                 template)
        except ValueError as e:
            print(resnum, resname, chainid, e, file=sys.stderr)
        finally:
            visited_bonds.extend(bonds)
            if remove_incomplete and not complete_match:
                atoms_to_del.extend(amap)

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
                not ((a1_name == 'N' and
                      a2_name == 'C' and
                      abs(a1_num - a2_num) == 1) or  # peptide bond
                     (a1_name == 'C' and
                      a2_name == 'N' and
                      abs(a1_num - a2_num) == 1) or  # peptide bond
                     (a1_name == 'SG' and
                      a2_name == 'SG')  # sulphur bridge
                     )):
                new_mol.RemoveBond(a1_ix, a2_ix)
            else:
                pass
    if atoms_to_del:
        new_mol = Chem.RWMol(new_mol)
        for idx in sorted(atoms_to_del, reverse=True):
            new_mol.RemoveAtom(idx)

    # if missing atoms were added we need to renumber them
    if add_missing_atoms:
        def atom_reorder_repr(i):
            """Generate keys for each atom during sort"""
            atom = new_mol.GetAtomWithIdx(i)
            info = atom.GetPDBResidueInfo()
            return (info.GetChainId(), info.GetResidueNumber(), i)

        order = list(range(new_mol.GetNumAtoms()))
        new_order = sorted(order, key=atom_reorder_repr)
        if new_order != order:
            new_mol = Chem.RenumberAtoms(new_mol, new_order)
    return new_mol
