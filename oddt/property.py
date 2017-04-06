"""
Module containg functions for prediction of molecular properies.
"""
from collections import OrderedDict
import numpy as np
import oddt


"""
Reference:
Wang R, Gao Y, Lai L. Calculating partition coefficient by atom-additive method.
Perspect Drug Discov Des. Kluwer Academic Publishers; 2000;19: 47-66.
https://dx.doi.org/10.1023/A:1008763405023
"""
XLOGP_SMARTS_1 = OrderedDict([
    # sp3 carbon
    ('*~[CX4]', [0.528]),  # C.3.unknown
    ('*~[CX4H3]', [0.528, 0.267]),  # C.3.h3.pi=0,1
    ('[#7,#8]~[CX4H3]', [-0.032]),  # C.3.h3.x
    ('*~[CX4H2,CX2]', [0.358, -0.008, -0.185]),  # C.3.h2.pi=0,1,2
    ('[#7,#8]~[CX4H2]', [-0.137, -0.303, -0.815]),  # C.3.h2.x.pi=0,1,2
    ('*~[CX4H1]', [0.127, -0.243, -0.499]),  # C.3.h.pi=0,1,2
    ('[#7,#8]~[CX4H]', [-0.205, -0.305, -0.709]),  # C.3.h.x.pi=0,1,2
    ('*~[CX4H0]', [-0.006, -0.570, -0.317]),  # C.3.pi=0,1,2
    ('[#7,#8]~[CX4H0]', [-0.316, -0.723]),  # C.3.x.pi=0,1

    # sp2 carbon
    ('*~[CX3]', [0.050]),
    ('*~[CX3H2]', [0.420]),  # C.2.h2           H    0.420
    ('*~[CX3H1]', [0.466, 0.136]),  # C.2.h.pi=0,1
    ('[#7X3,#8X2]~[CX3H1]', [0.001, -0.310]),  # C.2.h.x.pi=0,1
    ('*~[CX3H0]', [0.050, 0.013]),  # C.2.pi=0,1
    ('[#7,#8]~[CX3H0]', [-0.030, -0.027]),  # C.2.x.pi=0,1
    ('[#7!a,#8]~[CX3H0]~[#7!a,#8]', [0.005, -0.315]),  # C.2.x2.pi=0,1
    ('[#8,#7H0,H1]~[CX3H0]=[#7,#8]', [-0.030, -0.027]),  # Carboxyl
    ('[#7!a]-[CX3H0](-[#7!a])=[#7!a]', [-0.315]),  # C.cat => C.2.x2.pi=1

    # aromatic carbon
    ('*~c', [0.296]),  # C.ar.unknown
    ('*~[cH]', [0.337]),  # C.ar.h
    ('*~[cH]~[#7,#16]', [0.126]),  # C.ar.h.(X)
    ('*~[cH0]', [0.296]),  # C.ar
    ('*~[cH0]~[n,s]', [0.174]),  # C.ar.(X)
    ('*~[cH0]-,:[#7!a,#8]', [-0.151]),  # C.ar.x
    ('*~[cH0]-[nv3]', [-0.151]),  # C.ar.x
    ('[#7!a,#8]-,:[cH0]=,:[a#7,a#8,a#16]', [0.366]),  # C.ar.(X).x
    ('[nv3]-[cH0]=,:[a#7,a#8]', [0.366]),  # C.ar.(X).x


    # sp carbon
    ('*~[CX2]', [0.330]),  # C.1.unknown
    ('[!#7;!#8;!#1]~[CX2H1]', [0.209]),  # C.1.h
    ('*~[CX2H0]', [0.330]),  # C.1
    ('*=[CX2H0]=*', [2.073]),  # C.1.==

    # sp3 nitrogen
    ('*~[NX3,NX4+]', [0.159]),  # N.3.unknown
    ('*~[NX3H2,NX4H3+]', [-0.534, -0.329]),  # N.3.h2.pi=0,1
    ('[#7,#8]~[NX3H2,NX4H3+]', [-1.082]),  # N.3.h2.x
    ('*~[NX3H1,NX4H2+]', [-0.112, 0.166]),  # N.3.h.pi=0,1
    ('*~[#7X3H1a,#7X4H2+a]', [0.545]),  # N.3.h.ring
    ('[#7,#8]~[NX3H1,NX4H2+]', [0.324]),  # N.3.h.x
    ('[#7,#8]~[NX3H1R,NX4H2+R]', [0.153]),  # N.3.h.x.ring
    ('*~[NX3H0,NX4H0+]', [0.159,  0.761]),  # N.3.pi=0,1
    ('*~[NX3H0Ra,NX4H0+Ra]', [0.881]),  # N.3.ring
    ('[#7,#8]~[NX3H0,NX4H0+]', [-0.239]),  # N.3.x
    ('[#7,#8]~[NX3H0R,NX4H0+R]', [-0.010]),  # N.3.x.ring


    # amide nitrogen
    ('[CX3]([NX3])(=[O,S])', [0.078]),  # N.am.unknown
    ('[CX3]([NX3H2])(=[O,S])', [-0.646]),  # N.am.h2
    ('[CX3]([NX3H1])(=[O,S])', [-0.096]),  # N.am.h
    ('[CX3]([NX3H1]~[#7,#8])(=[O,S])', [-0.044]),  # N.am.h.x
    ('[CX3]([NX3H0])(=[O,S])', [0.078]),  # N.am
    ('[CX3]([NX3H0]~[#7,#8])(=[O,S])', [-0.118]),  # N.am.x

    # aromatic nitrogen
    ('*~n', [-0.493]),  # N.ar

    # sp2 nitrogen
    ('*~[NX2]', [0.007]),  # N.2.unknown
    ('*~[NX2]=[C,S]', [0.007, -0.275]),  # N.2.(=C).pi=0,1
    ('[#7,#8]~[NX2]=[C,S]', [0.366, 0.251]),  # N.2.(=C).x.pi=0,1
    ('N=[NX2,NX1]', [0.536]),  # N.2.(=N)
    ('N=[NX2]-*', [0.536]),  # N.2.(=N)
    ('[#7,#8]~[NX2]=N', [-0.597]),  # N.2.(=N).x
    ('*~[NX2](=O)', [0.427]),  # N.2.o
    ('*~[#7X3](=,:*)=,-[O,O-]', [1.178]),  # N.2.o2

    # sp nitrogen
    ('C#N', [-0.566]),  # N.1
    ('*=N=*', [-0.566]),  # N.1

    # sp3 oxygen
    ('*~[OX2H1]', [0.084]),  # O.3.unknown
    ('*~[OX2H1]', [-0.467, 0.082]),  # O.3.h.pi=0,1
    ('[#7,#8]~[OH1]', [-0.522]),  # O.3.h.x
    ('*~[#8X2H0]', [0.084, 0.435]),  # O.3.pi=0,1
    ('[#7,#8]~[#8X2H0]', [0.105]),  # O.3.x


    # sp2 oxygen
    ('*=[OX1]', [-0.399]),  # O.2
    ('*-[O-]', [-0.399]),  # O.2

    # sp3 sulfur
    ('[*][SH]', [0.419]),  # 76
    ('[*][SX2H0,SX4H0][*]', [0.255]),  # 77

    # sp2 sulfur
    ('[*]=[SX1]', [-0.148]),  # 78
    ('a:[#16X2]:a', [-0.148]),  # 78

    # sulfoxide sulfur
    ('[*][SX3](=O)-[*]', [-1.375]),  # 79

    # sulfone sulfur
    ('[*][SX4](=O)(=O)-[*]', [-0.168]),  # 80

    # phosphorus
    ('O=P([*])([*])[*]', [-0.447]),  # 81
    ('S=P([*])([*])[*]', [1.253]),  # 82

    # fluorine
    ('[*]F', [0.375, 0.202]),  # 83-84

    # chlorine
    ('[*]Cl', [0.512, 0.663]),  # 85-86

    # bromine
    ('[*]Br', [0.850, 0.839]),  # 87-88

    # iodine
    ('[*]I', [1.050, 1.050]),  # 89-90
])

"""
Reference:
Wang R, Gao Y, Lai L. Calculating partition coefficient by atom-additive method.
Perspect Drug Discov Des. Kluwer Academic Publishers; 2000;19: 47-66.
https://dx.doi.org/10.1023/A:1008763405023
"""
XLOGP_SMARTS_2 = [
    # Hydrophobic carbon
    {'smarts': '[C;!$([#6]~[!#6]);'
               '!$([#6]~[*]~[!#6]);'
               '!$([#6]~[*]~[*]~[!#6])]',
     'contrib_atoms': [0],
     'indicator': False,
     'coef': 0.211},
    # Internal H-bond
    {'smarts': '[O,N;!H0]-;!@*@,=*!@[O,N]',
     'contrib_atoms': [0, 3],
     'indicator': False,
     'coef': 0.429},
    {'smarts': '[O,N;!H0]-;!@*@,=*-*=;!@[O,N]',
     'contrib_atoms': [0, 4],
     'indicator': False,
     'coef': 0.429},
    {'smarts': '[O,N;!H0]-;!@*-*@,=*=;!@[O,N]',
     'contrib_atoms': [0, 4],
     'indicator': False,
     'coef': 0.429},
    # Halogen 1-3 pairs
    {'smarts': '[F,Cl,Br,I][*][F,Cl,Br,I]',
     'contrib_atoms': [0, 2],
     'indicator': False,
     'coef': 0.137},
    # Aromatic nitrogen 1-4 pair
    {'smarts': '[nX2r6]:*:*:[nX2r6]',
     'contrib_atoms': [0, 3],
     'indicator': False,
     'coef': 0.485},
    # Ortho sp3 oxygen pair
    {'smarts': '[OX2H0R0]-;!:;!@aa-;!:;!@[OX2H0R0]',
     'contrib_atoms': [0, 3],
     'indicator': False,
     'coef': -0.268},
    # Para donor pair
    {'smarts': '[O,N;!H0]-!:aaaa-!:[O,N;!H0]',
     'contrib_atoms': [0, 5],
     'indicator': False,
     'coef': -0.423},
    # sp2 oxygen 1-5 pair
    {'smarts': '[CX3](=O)-!:[*]-!:[CX3]=O',
     'contrib_atoms': [1, 4],
     'indicator': False,
     'coef': 0.580},
    # Indicator for alpha-amino acid
    {'smarts': '[NX3,NX4+][CX4H][*][CX3](=[OX1])[O,N]',
     'contrib_atoms': [0, 4],
     'indicator': True,
     'coef': -2.166},
    # Indicator for salicylic acid
    {'smarts': '[CX3](=[OX1])([O])-a:a-!:[OX1H]',
     'contrib_atoms': [1, 5],
     'indicator': True,
     'coef': 0.554},
    # Indicator for p-amino sulfonic acid
    {'smarts': '[SX4](=O)(=O)-c1ccc([NH2])cc1',
     'contrib_atoms': [0, 7],
     'indicator': True,
     'coef': -0.501},
]


def xlogp2_atom_contrib(mol, corrections=True):
    """
    Atoms contribution values taken from xlogp 2.0 publication. SMARTS patterns
    are in such orther that the described atom is always second. Values are
    sorted by increasing Pi bonds numbers.

    Reference:
    Wang R, Gao Y, Lai L. Calculating partition coefficient by atom-additive method.
    Perspect Drug Discov Des. Kluwer Academic Publishers; 2000;19: 47-66.
    https://dx.doi.org/10.1023/A:1008763405023
    """
    # count Pi bonds in n=2 environment
    pi_count = [sum(any(bond.order > 1 or bond.isaromatic
                        for bond in neighbor.bonds
                        if (bond.atoms[0].idx != atom.idx and
                            bond.atoms[1].idx != atom.idx))
                    for neighbor in atom.neighbors
                    if neighbor.atomicnum in [6, 7])
                if atom.atomicnum > 1 else 0
                for atom in mol]
    atom_contrib = np.zeros(len(pi_count))
    for smarts, contrib in XLOGP_SMARTS_1.items():
        matches = oddt.toolkit.Smarts(smarts).findall(mol)
        if matches:
            for match in matches:
                m = match[1]
                if oddt.toolkit.backend == 'ob':  # OB index is 1-based
                    m -= 1
                assert m >= 0
                atom_contrib[m] = contrib[pi_count[m]] if len(contrib) > pi_count[m] else contrib[-1]

    if corrections:
        for correction in XLOGP_SMARTS_2:
            matches = oddt.toolkit.Smarts(correction['smarts']).findall(mol)
            if matches:
                for match in matches:
                    for contrib_idx in correction['contrib_atoms']:
                        m = match[contrib_idx]
                        if oddt.toolkit.backend == 'ob':  # OB index is 1-based
                            m -= 1
                        assert m >= 0
                        atom_contrib[m] += correction['coef'] / float(len(correction['contrib_atoms']))
                        if correction['indicator']:
                            break

    return atom_contrib
