from collections import OrderedDict
from itertools import product

import rdkit
from rdkit import Chem
from rdkit.Chem.AllChem import ReplaceSubstructs
from rdkit.Chem import BondType


def AssignPDBResidueBondOrdersFromTemplate(refmol, mol):
    refmol2 = Chem.Mol(refmol)
    refmol3 = Chem.RWMol(refmol2) # copy of refmol without O TODO: remove that
    mol2 = Chem.Mol(mol)
    # The mol3 is needed due to a partial match.
    # Original mol is not modified and mol2 can get all bonds single
    mol3 = Chem.Mol(mol)

    # do the molecules match already?
    matches = mol2.GetSubstructMatches(refmol2)
    if not matches: # no, they don't match
        # check if bonds of mol are SINGLE
        for b in mol2.GetBonds():
            b.SetBondType(BondType.SINGLE)
            b.SetIsAromatic(False)
        # set the bonds of mol to SINGLE
        for b in refmol2.GetBonds():
            b.SetBondType(BondType.SINGLE)
            b.SetIsAromatic(False)
        # set atom charges to zero;
        for a in mol2.GetAtoms():
            a.SetFormalCharge(0)
            a.SetIsAromatic(False)
        for a in refmol2.GetAtoms():
            a.SetFormalCharge(0)
            a.SetIsAromatic(False)

        matches = mol2.GetSubstructMatches(refmol2)

    # do the molecules match now?
    if matches:
        mol3 = Chem.RWMol(mol3)
        for matching in matches:
            # apply matching: set bond properties
            for (atom1, atom2), (refatom1, refatom2) in zip(product(matching, repeat=2),
                                                            product(range(len(matching)), repeat=2)):
                b = refmol3.GetBondBetweenAtoms(refatom1, refatom2)
                b2 = mol3.GetBondBetweenAtoms(atom1, atom2)
                if b is None:
                    if b2: # this bond is not there
                        mol3.RemoveBond(atom1, atom2)
                    continue
                if b2 is None:
                    idx = mol3.AddBond(atom1, atom2)
                    b2 = mol3.GetBondBetweenAtoms(atom1, atom2)
                b2.SetBondType(b.GetBondType())
                b2.SetIsAromatic(b.GetIsAromatic())

            # apply matching: set atom properties
            for a in refmol3.GetAtoms():
                a2 = mol3.GetAtomWithIdx(matching[a.GetIdx()])
                a2.SetHybridization(a.GetHybridization())
                a2.SetIsAromatic(a.GetIsAromatic())
                a2.SetNumExplicitHs(a.GetNumExplicitHs())
                a2.SetFormalCharge(a.GetFormalCharge())

        mol3 = mol3.GetMol()
    else:
        raise ValueError("No matching found",
                         refmol.GetProp('_Name'),
                         Chem.MolToSmarts(refmol),
                         Chem.MolToSmarts(refmol2),
                         Chem.MolToSmarts(refmol3),
                        )
    if hasattr(mol2, '__sssAtoms'):
        mol2.__sssAtoms = None # we don't want all bonds highlighted
    return mol3


def PreparePDBMol(mol, removeHs=True, removeHOHs=True, residue_whitelist=None, residue_blacklist=None):
    """Prepares protein molecule by:
        - Removing Hs by hard using atomic number [default=True]
        - Removes HOH [default=True]
        - Assign bond orders from smiles of PDB residues (over 24k templates)
        - Removes bonds to metals
    """
    new_mol = Chem.RWMol(mol)

    # Remove Hs by hard, Chem.RemoveHs does not remove double bonded Hs
    for aix in reversed(range(new_mol.GetNumAtoms())):
        atom = new_mol.GetAtomWithIdx(aix)
        if removeHs and atom.GetAtomicNum() == 1:
            new_mol.RemoveAtom(aix)
        elif removeHOHs and atom.GetPDBResidueInfo().GetResidueName() == 'HOH':
            new_mol.RemoveAtom(aix)

    if residue_whitelist is None:
        # Get templates for all residues in molecules
        unique_resname = set([atom.GetPDBResidueInfo().GetResidueName() for atom in mol.GetAtoms()])
    else:
        unique_resname = set(residue_whitelist)
    if residue_blacklist is not None:
        unique_resname = unique_resname.difference(set(residue_blacklist))
    unique_resname = tuple(map(lambda x: x.strip().upper(), unique_resname))

    residue_mols = {}
    with open('pdbcodes_clean_smiles.csv') as f:
        backbone = Chem.MolFromSmarts('[#8]-[#6](=[#8])-[#6]-[#7]')
        perm_backbone = Chem.MolFromSmarts('[#8,#7]-[#6](=[#8])-[#6]-[#7]')
        for n, line in enumerate(f):
            if n == 0: continue
            data = line.split(',')
            if data[0] in unique_resname and data[0] != 'HOH': # skip waters
                res = Chem.MolFromSmiles(data[1])
                # TODO: might multiple matches occur?
                #res = ReplaceSubstructs(res, backbone, perm_backbone, replaceAll=True)[0]

                # remove oxygen for peptide
                # TODO: Remove that and treat the templates accordingly
                match = res.GetSubstructMatch(Chem.MolFromSmiles('OC(=O)CN'))
                if match:
                    res = Chem.RWMol(res)
                    res.RemoveAtom(match[0])
                    res = res.GetMol()

                res.SetProp('_Name', data[0])
                residue_mols[data[0]] = res

    # order residues by increasing size
    residue_mols = OrderedDict(sorted(residue_mols.items(), key=lambda x: x[1].GetNumAtoms()))

    # check if we have all templates
    for resname in unique_resname:
        if resname not in residue_mols and resname not in ['HOH']:
            raise ValueError('There is no template for residue "%s"' % resname)

    # reset B.O. using templates
    for resname in residue_mols.keys():
        template = residue_mols[resname]
        new_mol = AssignPDBResidueBondOrdersFromTemplate(template, new_mol)

    metals = (3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
               30, 31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
               50, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
               69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
               87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
               102, 103)
    new_mol = Chem.RWMol(new_mol)
    for atom in new_mol.GetAtoms():
        # HACK: termini oxygens get matched twice due to removal from templates
        # Terminus treatment
        if atom.GetPDBResidueInfo().GetName().strip() == 'OXT':
            atom.GetBonds()[0].SetBondType(BondType.SINGLE)

        # break bonds with metals
        if atom.GetAtomicNum() in metals:
            for n in atom.GetNeighbors():
                new_mol.RemoveBond(atom.GetIdx(), n.GetIdx())
    return new_mol.GetMol()
