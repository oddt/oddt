from collections import OrderedDict, defaultdict
from itertools import product, combinations

import rdkit
from rdkit import Chem
from rdkit.Chem.AllChem import ReplaceSubstructs
from rdkit.Chem import BondType


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


def AssignPDBResidueBondOrdersFromTemplate(protein, mol, amap, refmol):
    """
    protein: Mol with whole protein
    mol: Mol with residue only
    amap: atom map res->protein
    refmol: residue template
    """
    refmol2 = Chem.Mol(refmol)
    refmol3 = Chem.RWMol(refmol)  # copy of refmol without O TODO: remove that
    mol2 = Chem.Mol(mol)
    # The mol3 is needed due to a partial match.
    # Original mol is not modified and mol2 can get all bonds single
    mol3 = Chem.Mol(mol)

    visited_bonds = []

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
        protein = Chem.RWMol(protein)
        for matching in matches:
            # apply matching: set bond properties
            if len(matching) > len(amap):
                raise ValueError("Unequal amap and matching",
                                 refmol.GetProp('_Name'),
                                 Chem.MolToSmiles(refmol),
                                 Chem.MolToSmiles(refmol2),
                                 Chem.MolToSmiles(mol),
                                 mol.GetNumAtoms(),
                                 refmol.GetNumAtoms()
                                )
            for (atom1, atom2), (refatom1, refatom2) in zip(product(matching, repeat=2),
                                                            product(range(len(matching)), repeat=2)):
                b = refmol3.GetBondBetweenAtoms(refatom1, refatom2)
                b2 = protein.GetBondBetweenAtoms(amap[atom1], amap[atom2])
                if b is None:
                    if b2: # this bond is not there
                        protein.RemoveBond(amap[atom1], amap[atom2])
                    continue
                if b2 is None:
                    protein.AddBond(amap[atom1], amap[atom2])
                    b2 = protein.GetBondBetweenAtoms(amap[atom1], amap[atom2])
                b2.SetBondType(b.GetBondType())
                b2.SetIsAromatic(b.GetIsAromatic())
                visited_bonds.append(b2.GetIdx())

            # apply matching: set atom properties
            for a in refmol3.GetAtoms():
                a2 = protein.GetAtomWithIdx(amap[matching[a.GetIdx()]])
                a2.SetHybridization(a.GetHybridization())
                a2.SetIsAromatic(a.GetIsAromatic())
                # TODO: check for connected Hs
                # n_hs = sum(n.GetAtomicNum() == 1 for n in a2.GetNeighbors())
                # if a.GetNumExplicitHs() - n_hs < 0:
                #     raise ValueError("Hssss",
                #                      refmol.GetProp('_Name'),
                #                      Chem.MolToSmiles(refmol),
                #                      Chem.MolToSmiles(refmol2),
                #                      Chem.MolToSmiles(mol),
                #                      a.GetIdx(),
                #                      a.GetSmarts(),
                #                      a.GetNumExplicitHs(),
                #                      n_hs
                #                     )
                a2.SetNumExplicitHs(a.GetNumExplicitHs())
                a2.SetFormalCharge(a.GetFormalCharge())
    else:
        raise ValueError("No matching found",
                         refmol.GetProp('_Name'),
                         Chem.MolToSmiles(refmol),
                         Chem.MolToSmiles(refmol2),
                         Chem.MolToSmiles(mol),
                        )
    if hasattr(mol3, '__sssAtoms'):
        mol3.__sssAtoms = None # we don't want all bonds highlighted
    return protein, visited_bonds


def PreparePDBMol(mol,
                  removeHs=True,
                  removeHOHs=True,
                  residue_whitelist=None,
                  residue_blacklist=None,
                  disconnect_metals=True,
                  ):
    """Prepares protein molecule by:
        - Removing Hs by hard using atomic number [default=True]
        - Removes HOH [default=True]
        - Assign bond orders from smiles of PDB residues (over 24k templates)
        - Removes bonds to metals
    """
    new_mol = Chem.RWMol(mol)
    if removeHs:
        # new_mol = Chem.RWMol(Chem.RemoveHs(new_mol, sanitize=False))
        for i in reversed(range(new_mol.GetNumAtoms())):
            atom = new_mol.GetAtomWithIdx(i)
            if atom.GetAtomicNum() == 1:
                new_mol.RemoveAtom(i)

    if removeHOHs:
        for i in reversed(range(new_mol.GetNumAtoms())):
            atom = new_mol.GetAtomWithIdx(i)
            if atom.GetPDBResidueInfo().GetResidueName() == 'HOH':
                new_mol.RemoveAtom(i)
    # disconnect_metals and HOHs  for older versions of RDKit
    if rdkit.__version__.split(',')[0] < '2018':
        for i in reversed(range(new_mol.GetNumAtoms())):
            atom = new_mol.GetAtomWithIdx(i)
            atom_info = atom.GetPDBResidueInfo()
            if removeHOHs and atom_info.GetResidueName() == 'HOH':
                for n in atom.GetNeighbors():
                    n_info = n.GetPDBResidueInfo()
                    if n_info.GetResidueNumber() != atom_info.GetResidueNumber():
                        new_mol.RemoveBond(i, n.GetIdx())
            if atom.GetAtomicNum() in METALS:
                for n in atom.GetNeighbors():
                    new_mol.RemoveBond(i, n.GetIdx())


    # list of unique residues and their atom indices
    unique_resname = set()
    resiues_atom_map = defaultdict(list)
    for atom in new_mol.GetAtoms():
        # if atom.GetAtomicNum() > 1:
        info = atom.GetPDBResidueInfo()
        res_id = (info.GetResidueNumber(), info.GetResidueName(), info.GetChainId())
        resiues_atom_map[res_id].append(atom.GetIdx())
        unique_resname.add(info.GetResidueName())

    # create a list of residue mols with atom maps in both ways (mol->res and res->mol)
    residues = []
    for i, amap in resiues_atom_map.items():
        if len(amap) > 1:
            path = PathFromAtomList(new_mol, amap)
            mol_to_res_amap = {}
            res = Chem.PathToSubmol(new_mol, path, atomMap=mol_to_res_amap)
            res_to_mol_amap = sorted(mol_to_res_amap, key=mol_to_res_amap.get)
            residues.append((i, res, mol_to_res_amap, res_to_mol_amap))

    # load templates
    template_mols = {}
    with open('pdbcodes_clean_smiles.csv') as f:
        backbone = Chem.MolFromSmarts('[#8]-[#6](=[#8])-[#6]-[#7]')
        perm_backbone = Chem.MolFromSmarts('[#8,#7]-[#6](=[#8])-[#6]-[#7]')
        for n, line in enumerate(f):
            if n == 0: continue  # skip header
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

                res.SetProp('_Name', data[0])  # Needed for residue type lookup
                template_mols[data[0]] = res

    # Deal with residue lists
    if residue_whitelist is not None:
        unique_resname = set(residue_whitelist)
    if residue_blacklist is not None:
        unique_resname = unique_resname.difference(set(residue_blacklist))
    unique_resname = tuple(map(lambda x: x.strip().upper(), unique_resname))

    # reset B.O. using templates
    visited_bonds = []
    for ((resnum, resname, chainid), residue, mol_to_res_amap, res_to_mol_amap) in residues:
        if resname not in unique_resname:
            continue
        if resname not in template_mols:
            raise ValueError('There is no template for residue "%s"' % resname)
        template = template_mols[resname]
        try:
            new_mol, bonds = AssignPDBResidueBondOrdersFromTemplate(new_mol,
                                                                    residue,
                                                                    res_to_mol_amap,
                                                                    template)
        except ValueError as e:
            print(resnum, resname, chainid, e)
        visited_bonds.extend(bonds)

    # HACK: remove not-visited bonds
    if visited_bonds:  # probably we dont want to delete all
        new_mol = Chem.RWMol(new_mol)
        all_bonds = set(range(new_mol.GetNumBonds()))
        visited_bonds = set(visited_bonds)
        for bid in sorted(all_bonds.difference(visited_bonds), reverse=True):
            bond = new_mol.GetBondWithIdx(bid)
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            a1_num = a1.GetPDBResidueInfo().GetResidueNumber()
            a2_num = a2.GetPDBResidueInfo().GetResidueNumber()
            if (a1.GetAtomicNum() > 1 and
                a2.GetAtomicNum() > 1 and
                abs(a1_num - a2_num) > 1):  # FIXME: better avoid peptide bonds
                new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())

        # HACK: termini oxygens get matched twice due to removal from templates
        # TODO: remove by treatment of templates
        # Terminus treatment
        for atom in new_mol.GetAtoms():
            if atom.GetAtomicNum() == 8 and atom.GetPDBResidueInfo().GetName().strip() == 'OXT':
                bonds = atom.GetBonds()
                if len(bonds) > 0: # this should not happen at all
                    bonds[0].SetBondType(BondType.SINGLE)

    # be sure to recalculate stuff
    new_mol.UpdatePropertyCache()
    return new_mol
