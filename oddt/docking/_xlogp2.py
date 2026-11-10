"""United-atom hydrophobic scales from Wang, Gao & Lai (2000), Table 1.

Numerical facts transcribed from the publication, not XLOGP implementation code:
https://www.pkumdl.cn/ligbuilder3/xlogp2.pdf, printed pp. 50--53.
These are atomic scales for X-Score HM, not a complete molecular logP predictor;
the molecular correction factors in Table 2 are intentionally not applied.
"""

import numpy as np

# Index zero is reserved for receptor metals (outside the organic atom table).
XLOGP2_CONTRIBUTIONS = np.array(
    [
        0.0,
        0.528,
        0.267,
        -0.032,
        0.358,
        -0.008,
        -0.185,
        0.137,
        -0.303,
        -0.815,
        0.127,
        -0.243,
        -0.499,
        -0.205,
        -0.305,
        -0.709,
        -0.006,
        -0.570,
        -0.317,
        -0.316,
        -0.723,
        0.420,
        0.466,
        0.136,
        0.001,
        -0.310,
        0.050,
        0.013,
        -0.030,
        -0.027,
        0.005,
        -0.315,
        0.337,
        0.126,
        0.296,
        -0.151,
        0.174,
        0.366,
        0.209,
        0.330,
        2.073,
        -0.534,
        -0.329,
        -1.082,
        -0.112,
        0.166,
        0.545,
        0.324,
        0.153,
        0.159,
        0.761,
        0.881,
        -0.239,
        -0.010,
        -0.646,
        -0.096,
        -0.044,
        0.078,
        -0.118,
        0.007,
        -0.275,
        0.366,
        0.251,
        0.536,
        -0.597,
        0.427,
        1.178,
        -0.493,
        -0.566,
        -0.467,
        0.082,
        -0.522,
        0.084,
        0.435,
        0.105,
        -0.399,
        0.419,
        0.255,
        -0.148,
        -1.375,
        -0.168,
        -0.477,
        1.253,
        0.375,
        0.202,
        0.512,
        0.663,
        0.850,
        0.839,
        1.050,
        1.109,
    ]
)


def bond_order(bond):
    """Normalize aromatic bonds, including Open Babel's Kekule bond orders."""
    aromatic = bond.Bond.GetIsAromatic() if hasattr(bond, "Bond") else bond.OBBond.IsAromatic()
    return 1.5 if aromatic else bond.order


def classify_xlogp2(molecule, indices):
    """Independent topology-based interpretation of the published 90 types.

    X denotes N/O, R other atoms, and pi counts adjacent pi systems through
    single bonds. Aromatic pyrrole-like nitrogen uses the conjugated-ring types.
    Charged groups are projected onto their neutral united-atom topology;
    every charged atom is reported, except internal nitro charge separation.
    No unspecified numerical fallback is used for an unsupported topology.
    """
    data = molecule.atom_dict
    z, hyb = data["atomicnum"], data["hybridization"]
    neighbors, bonds = [], []
    for atom in molecule.atoms:
        neighbors.append([n.idx0 for n in atom.neighbors if n.atomicnum != 1])
        bonds.append(
            [
                (next(a.idx0 for a in b.atoms if a.idx0 != atom.idx0), bond_order(b))
                for b in atom.bonds
                if all(a.atomicnum != 1 for a in b.atoms)
            ]
        )
    nitro = set()
    for i in indices:
        if z[i] == 7 and len([j for j in neighbors[i] if z[j] == 8]) == 2:
            nitro.add(int(i))
            nitro.update(j for j in neighbors[i] if z[j] == 8)
    types, charged = [], []
    for i in indices:
        ns = neighbors[i]
        attached = bonds[i]
        h = int(data["numhs"][i])
        pi = sum(
            bool(data["isaromatic"][j] or any(k != i and b >= 1.5 for k, b in bonds[j]))
            for j, order in attached
            if order == 1
        )
        x = sum(z[j] in (7, 8) for j, order in attached if order == 1)
        double = [j for j, order in attached if order == 2]
        triple = [j for j, order in attached if order == 3]
        t = None
        if data["ismetal"][i]:
            t = 0
        elif z[i] == 6:
            if data["isaromatic"][i]:
                ring_ns = [j for j, order in attached if data["isaromatic"][j] and order != 1]
                # Some backends Kekulize aromatic bonds. Aromatic neighbors are
                # the ring neighbors; external aromatic substituents use order 1.
                nitrogen = any(z[j] == 7 for j in ring_ns)
                if h:
                    t = 33 if nitrogen else 32
                else:
                    external_x = any(z[j] in (7, 8) for j in ns if j not in ring_ns)
                    t = (37 if external_x else 36) if nitrogen else (35 if external_x else 34)
            elif triple or hyb[i] == 1:
                t = 38 if h and triple and z[triple[0]] == 6 else 39 if triple else 40
            elif double or hyb[i] == 2:
                if h >= 2:
                    t = 21
                elif h == 1:
                    t = (24 if x else 22) + int(pi > 0)
                else:
                    t = (26 if x == 0 else 28 if x == 1 else 30) + int(pi > 0)
            elif hyb[i] == 3:
                if h == 3:
                    t = 3 if x else 1 + int(pi > 0)
                elif h == 2:
                    t = (7 if x else 4) + min(pi, 2)
                elif h == 1:
                    t = (13 if x else 10) + min(pi, 2)
                elif h == 0:
                    t = 19 + int(pi > 0) if x else 16 + min(pi, 2)
        elif z[i] == 7:
            amide = str(data["atomtype"][i]) == "N.am" or any(
                z[j] == 6 and any(order == 2 and z[k] in (8, 16) for k, order in bonds[j]) for j in ns
            )
            conjugated_ring = bool(data["isaromatic"][i] and not data["isacceptor"][i])
            if i in nitro:
                t = 66
            elif amide:
                other_x = sum(z[j] in (7, 8) for j in ns)
                t = 54 if h >= 2 else (56 if other_x else 55) if h else (58 if other_x else 57)
            elif data["isaromatic"][i] and not conjugated_ring:
                t = 67
            elif triple:
                t = 68
            elif any(z[j] == 8 for j in double):
                t = 65
            elif double:
                partner = double[0]
                t = 63 if z[partner] == 7 else 59 + int(pi > 0)
                if x:
                    t = 64 if z[partner] == 7 else 61 + int(pi > 0)
            else:
                # Neutral-topology projection for protonated amines/quaternary N.
                h = min(h, max(0, 3 - len(ns)))
                if h >= 2:
                    t = 43 if x else 41 + int(pi > 0)
                elif h == 1:
                    t = (48 if conjugated_ring else 47) if x else 46 if conjugated_ring else 44 + int(pi > 0)
                else:
                    t = (53 if conjugated_ring else 52) if x else 51 if conjugated_ring else 49 + int(pi > 0)
        elif z[i] == 8:
            if double or str(data["atomtype"][i]) == "O.co2":
                t = 75
            elif h or len(ns) == 1:  # Includes the neutral projection of alkoxide.
                t = 71 if x else 69 + int(pi > 0)
            else:
                t = 74 if x else 72 + int(pi > 0)
        elif z[i] == 16:
            oxygens = sum(z[j] == 8 for j in double)
            t = 80 if oxygens >= 2 else 79 if oxygens else 78 if double else 76 if h or len(ns) == 1 else 77
        elif z[i] == 15:
            if any(z[j] == 8 for j in double):
                t = 81
            elif any(z[j] == 16 for j in double):
                t = 82
        elif z[i] in (9, 17, 35, 53):
            t = {9: 83, 17: 85, 35: 87, 53: 89}[int(z[i])] + int(pi > 0)
        if t is None:
            raise ValueError(
                "No published XLOGP2 type for atom %d (%s); supply explicit atomic logP scales"
                % (i, data["atomtype"][i])
            )
        types.append(t)
        if data["formalcharge"][i] and i not in nitro and not data["ismetal"][i]:
            charged.append(int(i))
    types = np.asarray(types, dtype=int)
    return types, XLOGP2_CONTRIBUTIONS[types].copy(), charged
