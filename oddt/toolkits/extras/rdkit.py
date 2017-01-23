from __future__ import absolute_import
from rdkit import Chem


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

    # define groups for atom types
    guanidine = '[NX3,NX2]([!O,!S])!@C(!@[NX3,NX2]([!O,!S]))!@[NX3,NX2]([!O,!S])'  # strict
    # guanidine = '[NX3]([!O])([!O])!:C!:[NX3]([!O])([!O])' # corina compatible
    # guanidine = '[NX3]!@C(!@[NX3])!@[NX3,NX2]'
    # guanidine = '[NX3]C([NX3])=[NX2]'
    # guanidine = '[NX3H1,NX2,NX3H2]C(=[NH1])[NH2]' # previous
    #

    if atomic_num == 6:
        if atom.GetIsAromatic():
            sybyl = 'C.ar'
        elif degree == 3 and _atom_matches_smarts(atom, guanidine):
            sybyl = 'C.cat'
        else:
            sybyl = '%s.%i' % (atom_symbol, hyb)
    elif atomic_num == 7:
        if atom.GetIsAromatic():
            sybyl = 'N.ar'
        elif _atom_matches_smarts(atom, 'C(=[O,S])-N'):
            sybyl = 'N.am'
        elif degree == 3 and _atom_matches_smarts(atom, '[$(N!-*),$([NX3H1]-*!-*)]'):
            sybyl = 'N.pl3'
        elif degree == 3 and _atom_matches_smarts(atom, guanidine):  # guanidine has N.pl3
            sybyl = 'N.pl3'
        elif degree == 4 or hyb == 3 and atom.GetFormalCharge():
            sybyl = 'N.4'
        else:
            sybyl = '%s.%i' % (atom_symbol, hyb)
    elif atomic_num == 8:
        # http://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
        if degree == 1 and _atom_matches_smarts(atom, '[CX3](=O)[OX1H0-]'):
            sybyl = 'O.co2'
        elif degree == 2:
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
    if a1.GetAtomicNum() == 6 and a2.GetAtomicNum() == 7 or a2.GetAtomicNum() == 6 and a1.GetAtomicNum() == 7:
        # https://github.com/rdkit/rdkit/blob/master/Data/FragmentDescriptors.csv
        patt = Chem.MolFromSmarts('C(=O)-N')
        for m in bond.GetOwningMol().GetSubstructMatches(patt):
            if a1.GetIdx() in m and a2.GetIdx() in m:
                return True
    return False
