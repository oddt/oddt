from __future__ import absolute_import, print_function
from itertools import chain
from math import isnan, isinf

from rdkit import Chem
from rdkit.Chem import AllChem

_metals = (3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
           30, 31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
           50, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
           69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
           87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
           102, 103)


def MolFromPDBBlock(molBlock,
                    sanitize=True,
                    removeHs=True,
                    flavor=0):
    mol = Chem.MolFromPDBBlock(molBlock,
                               sanitize=sanitize,
                               removeHs=removeHs,
                               flavor=flavor)
    if mol is None:
        return None
    # Adjust connectivity
    for atom in mol.GetAtoms():
        res = atom.GetPDBResidueInfo()
        if res is None:
            continue
        res_name = res.GetResidueName()
        atom_name = res.GetName().strip()

        # Fix missing double bonds in RDKit - double bonds
        if atom_name == 'O' and not res.GetIsHeteroAtom() and atom.GetDegree() == 1:
            atom.SetNoImplicit(True)
            atom.GetBonds()[0].SetBondType(Chem.BondType.DOUBLE)

        # Double bonds in sidechains
        if res_name in ['HID', 'HIE', 'HIP']:
            if atom_name == 'CD2':
                for bond in atom.GetBonds():
                    if bond.GetOtherAtom(atom).GetPDBResidueInfo().GetName().strip() == 'CG':
                        bond.SetBondType(Chem.BondType.DOUBLE)
                        break
            if res_name == 'HID':
                if atom_name == 'CE1':
                    for bond in atom.GetBonds():
                        if bond.GetOtherAtom(atom).GetPDBResidueInfo().GetName().strip() == 'ND1':
                            bond.SetBondType(Chem.BondType.DOUBLE)
                            break
            elif res_name in ['HIE', 'HIP']:
                if atom_name == 'CE1':
                    for bond in atom.GetBonds():
                        if bond.GetOtherAtom(atom).GetPDBResidueInfo().GetName().strip() == 'NE2':
                            bond.SetBondType(Chem.BondType.DOUBLE)
                            break

    if sanitize:
        result = Chem.SanitizeMol(mol)
        if result != 0:
            return None

    # Debug
    # for atom in mol.GetAtoms():
    #     res = atom.GetPDBResidueInfo()
    #     if res is None:
    #         continue
    #     res_name = res.GetResidueName()
    #     atom_name = res.GetName().strip()
    #     if atom_name in ['NE2', 'ND1'] and res_name in ['HID', 'HIE', 'HIS']:
    #         print(res_name,
    #               atom_name,
    #               atom.GetDegree(),
    #               atom.GetTotalValence(),
    #               atom.GetNumExplicitHs(),
    #               atom.GetNumImplicitHs(),
    #               sum(n.GetAtomicNum() == 1 for n in atom.GetNeighbors()),
    #               sep='\t')

    return mol


# Mol2 Atom typing
def _sybyl_atom_type(atom):
    """ Asign sybyl atom type
    Reference #1: http://www.tripos.com/mol2/atom_types.html
    Reference #2: http://chemyang.ccnu.edu.cn/ccb/server/AIMMS/mol2.pdf
    """
    sybyl = None
    atom_symbol = atom.GetSymbol()
    atomic_num = atom.GetAtomicNum()
    hyb = atom.GetHybridization()-1  # -1 since 1 = sp, 2 = sp1 etc
    hyb = min(hyb, 3)
    degree = atom.GetDegree()
    aromtic = atom.GetIsAromatic()

    # define groups for atom types
    guanidine = '[NX3,NX2]([!O,!S])!@C(!@[NX3,NX2]([!O,!S]))!@[NX3,NX2]([!O,!S])'  # strict
    # guanidine = '[NX3]([!O])([!O])!:C!:[NX3]([!O])([!O])' # corina compatible
    # guanidine = '[NX3]!@C(!@[NX3])!@[NX3,NX2]'
    # guanidine = '[NX3]C([NX3])=[NX2]'
    # guanidine = '[NX3H1,NX2,NX3H2]C(=[NH1])[NH2]' # previous
    #

    if atomic_num == 6:
        if aromtic:
            sybyl = 'C.ar'
        elif degree == 3 and _atom_matches_smarts(atom, guanidine):
            sybyl = 'C.cat'
        else:
            sybyl = '%s.%i' % (atom_symbol, hyb)
    elif atomic_num == 7:
        if aromtic:
            sybyl = 'N.ar'
        elif _atom_matches_smarts(atom, 'C(=[O,S])-N'):
            sybyl = 'N.am'
        elif degree == 3 and _atom_matches_smarts(atom, '[$(N!-*),$([NX3H1]-*!-*)]'):
            sybyl = 'N.pl3'
        elif _atom_matches_smarts(atom, guanidine):  # guanidine has N.pl3
            sybyl = 'N.pl3'
        elif degree == 4 or hyb == 3 and atom.GetFormalCharge():
            sybyl = 'N.4'
        else:
            sybyl = '%s.%i' % (atom_symbol, hyb)
    elif atomic_num == 8:
        # http://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
        if degree == 1 and _atom_matches_smarts(atom, '[CX3](=O)[OX1H0-]'):
            sybyl = 'O.co2'
        elif degree == 2 and not aromtic:  # Aromatic Os are sp2
            sybyl = 'O.3'
        else:
            sybyl = 'O.2'
    elif atomic_num == 16:
        # http://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
        if degree == 3 and _atom_matches_smarts(atom, '[$([#16X3]=[OX1]),$([#16X3+][OX1-])]'):
            sybyl = 'S.O'
        # https://github.com/rdkit/rdkit/blob/master/Data/FragmentDescriptors.csv
        elif _atom_matches_smarts(atom, 'S(=,-[OX1;+0,-1])(=,-[OX1;+0,-1])(-[#6])-[#6]'):
            sybyl = 'S.o2'
        else:
            sybyl = '%s.%i' % (atom_symbol, hyb)
    elif atomic_num == 15 and hyb == 3:
        sybyl = '%s.%i' % (atom_symbol, hyb)

    if not sybyl:
        sybyl = atom_symbol
    return sybyl


def _atom_matches_smarts(atom, smarts):
    idx = atom.GetIdx()
    patt = Chem.MolFromSmarts(smarts)
    for m in atom.GetOwningMol().GetSubstructMatches(patt):
        if idx in m:
            return True
    return False


def _amide_bond(bond):
    a1 = bond.GetBeginAtom()
    a2 = bond.GetEndAtom()
    if (a1.GetAtomicNum() == 6 and a2.GetAtomicNum() == 7 or
            a2.GetAtomicNum() == 6 and a1.GetAtomicNum() == 7):
        # https://github.com/rdkit/rdkit/blob/master/Data/FragmentDescriptors.csv
        patt = Chem.MolFromSmarts('C(=O)-N')
        for m in bond.GetOwningMol().GetSubstructMatches(patt):
            if a1.GetIdx() in m and a2.GetIdx() in m:
                return True
    return False


def MolFromPDBQTBlock(block, sanitize=True, removeHs=True, template=None):
    """Read PDBQT block to a RDKit Molecule

    Parameters
    ----------
        block: string
            Residue name which explicitly pint to a ligand(s).
        sanitize: bool (default=True)
            Should the sanitization be performed
        mol: rdkit.Chem.rdchem.Mol (default=None)
            A template used for assigning bond orders, as PDBQT does not encode
            them at all.

    Returns
    -------
        mol: rdkit.Chem.rdchem.Mol
            Molecule read from PDBQT
    """
    pdb_lines = []
    name = ''
    data = {}
    for line in block.split('\n'):
        # Get all know data from REMARK section
        if line[:12] == 'REMARK  Name':
            name = line[15:].strip()
        elif line[:18] == 'REMARK VINA RESULT':
            tmp = line[19:].split()
            data['vina_affinity'] = tmp[0]
            data['vina_rmsd_lb'] = tmp[1]
            data['vina_rmsd_ub'] = tmp[2]

        # no more data to collect
        if line[:4] != 'ATOM':
            continue

        pdb_line = line[:56]
        pdb_line += '1.00  0.00           '

        # Do proper atom type lookup
        atom_type = line[71:].split()[1]
        if atom_type == 'A':
            atom_type = 'C'
        elif atom_type[:1] == 'O':
            atom_type = 'O'
        elif atom_type[:1] == 'H':
            atom_type = 'H'
            if removeHs:
                continue
        elif atom_type == 'NA':
            atom_type = 'N'

        pdb_lines.append(pdb_line + atom_type)
    mol = Chem.MolFromPDBBlock('\n'.join(pdb_lines), sanitize=False)
    mol.SetProp('_Name', name)
    for k, v in data.items():
        mol.SetProp(str(k), str(v))
    if removeHs:
        mol = Chem.RemoveHs(mol, sanitize=sanitize)
    elif sanitize:
        Chem.SanitizeMol(mol)

    # reorder atoms using serial
    if not sanitize:
        Chem.GetSSSR(mol)
    new_order = sorted(range(mol.GetNumAtoms()),
                       key=lambda i: (mol.GetAtomWithIdx(i)
                                      .GetPDBResidueInfo()
                                      .GetSerialNumber()))
    mol = Chem.RenumberAtoms(mol, new_order)

    if template is not None:
        mol = AllChem.AssignBondOrdersFromTemplate(template, mol)

    return mol


def PDBQTAtomLines(mol, donors, acceptors):
    """Create a list with PDBQT atom lines for each atom in molecule"""

    atom_lines = [line.replace('HETATM', 'ATOM  ')
                  for line in Chem.MolToPDBBlock(mol).split('\n')
                  if line.startswith('HETATM') or line.startswith('ATOM')]

    pdbqt_lines = []
    for idx, atom in enumerate(mol.GetAtoms()):
        pdbqt_line = atom_lines[idx][:56]

        pdbqt_line += '0.00  0.00    '  # append empty vdW and ele
        # Get charge
        charge = None
        if atom.HasProp('_GasteigerCharge'):
            charge = float(atom.GetProp('_GasteigerCharge'))
        elif atom.GetAtomicNum() == 1:
            root_atom = atom.GetNeighbors()[0]
            if root_atom.HasProp('_GasteigerHCharge'):
                charge = float(root_atom.GetProp('_GasteigerHCharge'))
        if charge is None:
            raise ValueError('No charge information')
        if isnan(charge) or isinf(charge):  # FIXME: this should not happen, blame RDKit
            charge = 0.
        pdbqt_line += ('%.3f' % charge).rjust(6)

        # Get atom type
        # TODO: do proper atom typing here
        pdbqt_line += ' '
        atomicnum = atom.GetAtomicNum()
        if atomicnum == 6:
            if atom.GetIsAromatic():
                pdbqt_line += 'A'
            else:
                pdbqt_line += 'C'
        elif atomicnum == 7 and idx in acceptors:
            pdbqt_line += 'NA'
        elif atomicnum == 8 and idx in acceptors:
            pdbqt_line += 'OA'
        elif atomicnum == 1:
            if atom.GetNeighbors()[0].GetIdx() in donors:
                pdbqt_line += 'HD'
            else:
                pdbqt_line += 'H'
        else:
            pdbqt_line += atom.GetSymbol()
        pdbqt_lines.append(pdbqt_line)
    return pdbqt_lines


def MolToPDBQTBlock(mol, flexible=True, addHs=False, computeCharges=True):
    """Write RDKit Molecule to a PDBQT block

    Parameters
    ----------
        mol: rdkit.Chem.rdchem.Mol
            Molecule with a protein ligand complex
        flexible: bool (default=True)
            Should the molecule encode torsions. Ligands should be flexible,
            proteins in turn can be rigid.
        addHs: bool (default=False)
            The PDBQT format requires at least polar Hs on donors. By default Hs
            are added.
        computeCharges: bool (default=True)
            Partial charges are also required in PDBQT format. They are
            automatically computed by default. If the Hs are added the charges
            must and will be recomputed.

    Returns
    -------
        block: str
            String wit PDBQT encoded molecule
    """
    # make a copy of molecule
    mol = Chem.Mol(mol)

    # Identify donors and acceptors for atom typing
    # Acceptors
    patt = Chem.MolFromSmarts('[$([O;H1;v2]),'
                              '$([O;H0;v2;!$(O=N-*),'
                              '$([O;-;!$(*-N=O)]),'
                              '$([o;+0])]),'
                              '$([n;+0;!X3;!$([n;H1](cc)cc),'
                              '$([$([N;H0]#[C&v4])]),'
                              '$([N&v3;H0;$(Nc)])]),'
                              '$([F;$(F-[#6]);!$(FC[F,Cl,Br,I])])]')
    acceptors = list(map(lambda x: x[0],
                         mol.GetSubstructMatches(patt, maxMatches=mol.GetNumAtoms())))
    # Donors
    patt = Chem.MolFromSmarts('[$([N&!H0&v3,N&!H0&+1&v4,n&H1&+0,$([$([Nv3](-C)(-C)-C)]),'
                              '$([$(n[n;H1]),'
                              '$(nc[n;H1])])]),'
                              # Guanidine can be tautormeic - e.g. Arginine
                              '$([NX3,NX2]([!O,!S])!@C(!@[NX3,NX2]([!O,!S]))!@[NX3,NX2]([!O,!S])),'
                              '$([O,S;H1;+0])]')
    donors = list(map(lambda x: x[0],
                      mol.GetSubstructMatches(patt, maxMatches=mol.GetNumAtoms())))
    if addHs:
        mol = Chem.AddHs(mol, addCoords=True, onlyOnAtoms=donors, )
    if addHs or computeCharges:
        AllChem.ComputeGasteigerCharges(mol)

    atom_lines = PDBQTAtomLines(mol, donors, acceptors)
    assert len(atom_lines) == mol.GetNumAtoms()

    pdbqt_lines = []

    # vina scores
    if (mol.HasProp('vina_affinity') and mol.HasProp('vina_rmsd_lb') and
            mol.HasProp('vina_rmsd_lb')):
        pdbqt_lines.append('REMARK VINA RESULT:  ' +
                           ('%.1f' % float(mol.GetProp('vina_affinity'))).rjust(8) +
                           ('%.3f' % float(mol.GetProp('vina_rmsd_lb'))).rjust(11) +
                           ('%.3f' % float(mol.GetProp('vina_rmsd_ub'))).rjust(11))

    pdbqt_lines.append('REMARK  Name = ' +
                       (mol.GetProp('_Name') if mol.HasProp('_Name') else ''))
    if flexible:
        # Find rotatable bonds
        rot_bond = Chem.MolFromSmarts('[!$(*#*)&!D1&!$(C(F)(F)F)&'
                                      '!$(C(Cl)(Cl)Cl)&'
                                      '!$(C(Br)(Br)Br)&'
                                      '!$(C([CH3])([CH3])[CH3])&'
                                      '!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&'
                                      '!$([#7,O,S!D1]-!@[CD3]=[N,O,S])&'
                                      '!$([CD3](=[N+])-!@[#7!D1])&'
                                      '!$([#7!D1]-!@[CD3]=[N+])]-!@[!$(*#*)&'
                                      '!D1&!$(C(F)(F)F)&'
                                      '!$(C(Cl)(Cl)Cl)&'
                                      '!$(C(Br)(Br)Br)&'
                                      '!$(C([CH3])([CH3])[CH3])]')
        bond_atoms = list(mol.GetSubstructMatches(rot_bond))
        num_torsions = len(bond_atoms)

        # Active torsions header
        pdbqt_lines.append('REMARK  %i active torsions:' % num_torsions)
        pdbqt_lines.append('REMARK  status: (\'A\' for Active; \'I\' for Inactive)')
        for i, (a1, a2) in enumerate(bond_atoms):
            pdbqt_lines.append('REMARK%5.0i  A    between atoms: _%i  and  _%i'
                               % (i + 1, a1 + 1, a2 + 1))

        # Fragment molecule on bonds to ge rigid fragments
        frags = list(Chem.GetMolFrags(
            Chem.FragmentOnBonds(mol, [mol.GetBondBetweenAtoms(a1, a2).GetIdx()
                                       for a1, a2 in bond_atoms], addDummies=False)))
        # sort by the fragment size and the number of bonds (secondary)
        frags = sorted(frags,
                       key=lambda x: (len(x), sum(a1 in x or a2 in x
                                                  for a1, a2 in bond_atoms)),
                       reverse=True)
        # Start writting the lines with ROOT
        pdbqt_lines.append('ROOT')
        frag = frags.pop(0)
        for idx in frag:
            pdbqt_lines.append(atom_lines[idx])
        pdbqt_lines.append('ENDROOT')

        # Now build the tree of torsions usign DFS algorithm. Keep track of last
        # route with following variables to move down the tree and close branches
        branch_queue = []
        current_root = frag
        old_roots = [frag]

        visited_frags = []
        visited_bonds = []
        while len(frags) > len(visited_frags):
            end_branch = True
            for frag_num, frag in enumerate(frags):
                for bond_num, (a1, a2) in enumerate(bond_atoms):
                    if (frag_num not in visited_frags and
                        bond_num not in visited_bonds and
                        (a1 in current_root and a2 in frag or
                         a2 in current_root and a1 in frag)):
                        # direction of bonds is important
                        if a1 in current_root:
                            bond_dir = '%i %i' % (a1 + 1, a2 + 1)
                        else:
                            bond_dir = '%i %i' % (a2 + 1, a1 + 1)
                        pdbqt_lines.append('BRANCH %s' % bond_dir)
                        for idx in frag:
                            pdbqt_lines.append(atom_lines[idx])
                        branch_queue.append('ENDBRANCH %s' % bond_dir)

                        # Overwrite current root and stash previous one in queue
                        old_roots.append(current_root)
                        current_root = frag

                        # remove used elements from stack
                        visited_frags.append(frag_num)
                        visited_bonds.append(bond_num)

                        # mark that we dont want to end branch yet
                        end_branch = False
                        break
                    else:
                        continue
                    break

            if end_branch:
                if len(branch_queue) > 0:
                    pdbqt_lines.append(branch_queue.pop())
                if old_roots:
                    current_root = old_roots.pop()
                else:  # go to next disconnected fragment
                    next_frag = set(range(len(frag_num))).difference(visited_frags)[0]
                    current_root = frags[next_frag]
                    visited_frags.append(next_frag)
        # close opened branches if any is open
        while len(branch_queue):
            pdbqt_lines.append(branch_queue.pop())
        pdbqt_lines.append('TORSDOF %i' % num_torsions)
    else:
        pdbqt_lines.extend(atom_lines)

    return '\n'.join(pdbqt_lines)
