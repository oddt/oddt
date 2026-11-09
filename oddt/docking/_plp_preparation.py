"""Atom typing and fixed molecular data for the independent PLP scorers."""

from itertools import permutations

import numpy as np
from scipy.spatial.distance import cdist

DONOR, ACCEPTOR, BOTH, NONPOLAR, METAL = range(5)
HYDROGEN, METAL_CONTACT, BURIED, NONPOLAR_CONTACT, REPULSIVE = range(5)

# Table 3.3 in Korb (2008): rows are ligand types, columns receptor types.
INTERACTION_TYPES = np.array([[4, 0, 0, 2, 4], [0, 4, 0, 2, 1], [0, 0, 0, 2, 1], [2, 2, 2, 3, 2]], dtype=int)

# Numerical SYBYL torsion examples published in the ADF QMMM Manual (2018),
# printed pp. 31--32, https://www.scm.com/doc.2018/QMMM/_downloads/QMMM.pdf.
# Its coefficient K is half the barrier V used in Korb Eq. 3.12.
# This is a documented subset, not the complete Tripos 5.2 parameter library.
SYBYL_TORSIONS = {
    ("*", "C.ar", "S.3", "*"): (2.0, 1, 3),
    ("*", "S.3", "S.3", "*"): (0.0, 1, 2),
    ("C.2", "C.2", "C.3", "*"): (0.252, -1, 3),
    ("C.3", "C.2", "C.3", "*"): (0.252, 1, 3),
    ("H", "C.2", "C.3", "*"): (0.548, 1, 3),
    ("*", "C.ar", "C.ar", "C.ar"): (4.7, -1, 2),
    ("*", "C.2", "C.3", "C.2"): (0.252, 1, 3),
    ("*", "C.2", "C.3", "C.3"): (0.252, 1, 3),
    ("*", "C.2", "C.3", "H"): (0.548, 1, 3),
    ("*", "C.3", "C.3", "H"): (0.64, 1, 3),
    ("O.2", "C.2", "C.3", "C.3"): (1.4, -1, 3),
    ("O.co2", "C.2", "C.3", "C.3"): (1.4, -1, 3),
    ("C.2", "C.3", "C.3", "C.2"): (0.08, 1, 3),
    ("C.2", "C.3", "C.3", "C.3"): (0.252, 1, 3),
}


def validate_torsion(value):
    array = np.asarray(value, dtype=float)
    if (
        array.shape != (3,)
        or not np.isfinite(array).all()
        or array[0] < 0
        or array[1] not in (-1, 1)
        or array[2] < 1
        or array[2] != int(array[2])
    ):
        raise ValueError("torsion parameters must be (nonnegative barrier, sign +/-1, positive integer period)")
    return tuple(array)


def atom_parameters(molecule, receptor=False):
    """ODDT donor/acceptor perception with the published ester/nitro exceptions."""
    atoms = molecule.atom_dict
    if np.any(atoms["atomicnum"] <= 0):
        raise ValueError("PLP does not support dummy or query atoms")
    if not receptor and np.any(atoms["ismetal"]):
        raise ValueError("PLP supports metal ions in the receptor, not in the ligand")
    neighbors = [np.array(sorted(n.idx0 for n in atom.neighbors), dtype=int) for atom in molecule.atoms]
    types = np.full(len(atoms), NONPOLAR, dtype=int)
    polar = np.isin(atoms["atomicnum"], (7, 8))
    donor = polar & atoms["isdonor"] & (atoms["numhs"] > 0)
    acceptor = polar & atoms["isacceptor"]
    negative = atoms["formalcharge"] < 0
    for atom in molecule.atoms:
        i = atom.idx0
        if atom.atomicnum == 8 and atoms["numhs"][i] == 0:
            # The bridging oxygen of an ester is excluded as an acceptor.
            for neighbor in atom.neighbors:
                if neighbor.atomicnum == 6 and any(
                    bond.order == 2 and any(a.atomicnum == 8 for a in bond.atoms) for bond in neighbor.bonds
                ):
                    if len([n for n in atom.neighbors if n.atomicnum != 1]) == 2:
                        acceptor[i] = False
        if atom.atomicnum == 7 and atom.formalcharge > 0:
            oxygens = [n.idx0 for n in atom.neighbors if n.atomicnum == 8]
            if len(oxygens) >= 2:
                negative[oxygens] = False
    types[donor], types[acceptor], types[donor & acceptor] = DONOR, ACCEPTOR, BOTH
    types[atoms["ismetal"]] = METAL
    parents = np.full(len(atoms), -1, dtype=int)
    ordinary, cho = [], []
    for atom in molecule.atoms:
        if atom.atomicnum != 1:
            continue
        i = atom.idx0
        if len(neighbors[i]) != 1:
            raise ValueError("PLP scoring requires each explicit hydrogen to have exactly one parent")
        parent = int(neighbors[i][0])
        parents[i] = parent
        if donor[parent]:
            ordinary.append(i)
        elif atoms["atomicnum"][parent] == 6 and any(
            atoms["atomicnum"][j] == 7 and atoms["isaromatic"][j] and acceptor[j] for j in neighbors[parent]
        ):
            cho.append(i)
    return dict(
        types=types,
        heavy=np.flatnonzero(atoms["atomicnum"] != 1),
        acceptors=np.flatnonzero(acceptor),
        neighbors=neighbors,
        donor_parent=parents,
        hydrogens=np.asarray(ordinary, dtype=int),
        cho_hydrogens=np.asarray(cho, dtype=int),
        negative=negative,
        positive=atoms["formalcharge"] > 0,
        atomicnum=atoms["atomicnum"].copy(),
        atomtypes=atoms["atomtype"].copy(),
        has_hydrogen=atoms["numhs"] > 0,
    )


def ligand_clashes(parameters, rotors):
    """Variable-fragment pairs at least three bonds apart (Table 3.5)."""
    heavy = parameters["heavy"]
    count = len(heavy)
    full_to_heavy = {atom: i for i, atom in enumerate(heavy)}
    adjacency = np.zeros((count, count), dtype=bool)
    for i, atom in enumerate(heavy):
        for j in parameters["neighbors"][atom]:
            if j in full_to_heavy:
                adjacency[i, full_to_heavy[j]] = True
    two = np.eye(count, dtype=bool) | adjacency | (adjacency @ adjacency)
    three = two | ((adjacency @ adjacency) @ adjacency)
    variable = np.zeros((count, count), dtype=bool)
    for rotor in rotors:
        mask = rotor["full_mask"][heavy]
        variable |= mask[:, None] != mask[None, :]
    rows, cols = np.nonzero(np.triu(variable & ~two, 1))
    exactly_three = three[rows, cols]
    z = parameters["atomicnum"][heavy]
    class1 = np.isin(z, (7, 8, 9, 17, 35))
    class2 = class1 & ~(np.isin(z, (7, 8)) & parameters["has_hydrogen"][heavy])
    class3 = ~class1
    both_class1 = class1[rows] & class1[cols]
    class23 = (class2[rows] & class3[cols]) | (class3[rows] & class2[cols])
    cutoff = np.where(exactly_three, 2.75, 3.0)
    cutoff[class23] = np.where(exactly_three[class23], 2.5, 2.75)
    cutoff[both_class1] = 2.5
    return heavy[rows], heavy[cols], cutoff


def ligand_torsions(parameters, rotors, rules, fallback):
    """Match numerical torsion data, reporting all uses of a generic fallback."""
    indices, coefficients, unmatched = [], [], []
    for rotor in rotors:
        _, first, second, _ = rotor["full_atoms"]
        for start in parameters["neighbors"][first]:
            if start == second:
                continue
            for end in parameters["neighbors"][second]:
                if end == first:
                    continue
                quad = (int(start), first, second, int(end))
                types = tuple(parameters["atomtypes"][list(quad)])
                selected, specificity = None, -1
                for pattern, value in rules.items():
                    if any(all(p == "*" or p == t for p, t in zip(pattern, order)) for order in (types, types[::-1])):
                        score = sum(p != "*" for p in pattern)
                        if score >= specificity:
                            selected, specificity = value, score
                if selected is None:
                    unmatched.append((quad, types))
                    if fallback is None:
                        raise ValueError("No published torsion parameters for %s; supply torsion_parameters" % (types,))
                    selected = fallback
                indices.append(quad)
                coefficients.append(selected)
    return (
        np.asarray(indices, dtype=int).reshape(-1, 4),
        np.asarray(coefficients, dtype=float).reshape(-1, 3),
        unmatched,
    )


def metal_sites(coordinates, parameters, cutoff):
    """Fit tetrahedral/octahedral vacant sites; keep undetermined metals radial."""
    tetrahedron = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    octahedron = np.vstack((np.eye(3), -np.eye(3)))
    result = []
    for metal in np.flatnonzero(parameters["types"] == METAL):
        candidates = np.flatnonzero(np.isin(parameters["atomicnum"], (7, 8, 16)))
        distances = cdist(coordinates[metal : metal + 1], coordinates[candidates])[0]
        attached = candidates[(distances > 0.1) & (distances <= cutoff)]
        vectors = coordinates[attached] - coordinates[metal]
        directions = vectors / np.linalg.norm(vectors, axis=1)[:, None] if len(vectors) else vectors
        description = dict(atom=int(metal), neighbors=attached, geometry="radial", sites=np.empty((0, 3)))
        if parameters["atomicnum"][metal] in (12, 20):
            result.append(description)
            continue
        if len(directions) < 2 or np.linalg.matrix_rank(directions, tol=1e-6) < 2:
            description["geometry"] = "radial_underdetermined"
            result.append(description)
            continue
        if len(directions) > 6:
            raise ValueError("ChemPLP metal-site fitting supports at most six receptor coordinating atoms")
        best = np.inf
        for name, vertices in (("tetrahedral", tetrahedron), ("octahedral", octahedron)):
            for assignment in permutations(range(len(vertices)), len(directions)):
                matched = vertices[list(assignment)]
                u, _, vt = np.linalg.svd(matched.T @ directions)
                rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
                error = float(np.mean(np.sum((matched @ rotation - directions) ** 2, axis=1)))
                if error < best - 1e-12:
                    best = error
                    free = [i for i in range(len(vertices)) if i not in assignment]
                    description.update(
                        geometry=name,
                        sites=coordinates[metal] + 2.2 * (vertices[free] @ rotation),
                        fit_rmsd=float(np.sqrt(error)),
                    )
        result.append(description)
    return result
