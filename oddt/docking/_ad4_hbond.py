"""Original direct-pair realization of AD4 hydrogen-bond geometry and valence.

AD4's angular/valence requirements are described by Huey et al. (2007),
https://doi.org/10.1002/jcc.20634, and the AutoDock4.2.6 manual. The explicit
lone-pair frames, cosine-squared product and global capacity assignment below
are our direct-pair adaptation, not AutoGrid's probe/grid algorithm or a copy
of its angular polynomials. All derivatives are analytic.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class _Vector:
    """A 3-vector and its small local Cartesian Jacobian."""

    def __init__(self, value, jacobian):
        self.value = value
        self.jacobian = jacobian

    def __add__(self, other):
        return _Vector(self.value + other.value, self.jacobian + other.jacobian)

    def __sub__(self, other):
        return self + other.scale(-1)

    def scale(self, scalar):
        return _Vector(self.value * scalar, self.jacobian * scalar)

    def unit(self):
        length = np.linalg.norm(self.value)
        if length < 1e-8:
            raise ValueError("AD4 angular scoring requires nondegenerate bonded geometry; supply a 3D conformation")
        unit = self.value / length
        jacobian = self.jacobian - np.einsum("i,j,jab->iab", unit, unit, self.jacobian)
        return _Vector(unit, jacobian / length)

    def cross(self, other):
        jacobian = np.cross(self.jacobian.transpose(1, 2, 0), other.value)
        jacobian += np.cross(self.value, other.jacobian.transpose(1, 2, 0))
        return _Vector(np.cross(self.value, other.value), jacobian.transpose(2, 0, 1))


def geometry_rules(molecule, types):
    """Local bonded frame rules and AD4 donor/acceptor capacities."""
    parents = np.full(len(types), -1, dtype=int)
    capacities = np.zeros(len(types), dtype=int)
    rules = {}
    for atom in molecule.atoms:
        index = atom.idx0
        if types[index] == "HD":
            parents[index] = atom.neighbors[0].idx0
        elif types[index] in ("NA", "OA", "SA"):
            capacities[index] = 1 if types[index] == "NA" else 2
            neighbors = sorted(atom.neighbors, key=lambda n: n.idx0)
            if not neighbors:
                rules[index] = ("spherical", (index,))
            elif types[index] == "NA":
                rules[index] = ("nitrogen", (index,) + tuple(n.idx0 for n in neighbors))
            elif len(neighbors) >= 2:
                rules[index] = ("tetrahedral", (index, neighbors[0].idx0, neighbors[1].idx0))
            else:
                parent = neighbors[0]
                plane = [n for n in parent.neighbors if n.idx0 != index and n.atomicnum != 1]
                if plane:
                    rules[index] = ("planar", (index, parent.idx0, min(n.idx0 for n in plane)))
                else:
                    # An isolated/linear terminal acceptor does not define a
                    # molecular plane. Use an axial lobe, never a world axis.
                    rules[index] = ("nitrogen", (index, parent.idx0))
    return dict(donor_parent=parents, acceptor_capacity=capacities, acceptor_rules=rules)


def acceptor_frames(coordinates, rules, gradient=True):
    """Lone-pair unit vectors with derivatives through all defining neighbors."""
    frames = {}
    for atom, (kind, indices) in rules.items():
        indices = np.asarray(indices, dtype=int)
        if kind == "spherical":
            frames[atom] = (None, indices, None)
            continue
        points = []
        for i, index in enumerate(indices):
            jacobian = np.zeros((3, len(indices), 3))
            jacobian[:, i] = np.eye(3)
            points.append(_Vector(coordinates[index], jacobian))
        center = points[0]
        bonds = [(point - center).unit() for point in points[1:]]
        if kind == "nitrogen":
            axis = bonds[0]
            for bond in bonds[1:]:
                axis = axis + bond
            if np.linalg.norm(axis.value) < 1e-6 and len(bonds) >= 2:
                normal = bonds[0].cross(bonds[1]).unit()
                directions = [normal, normal.scale(-1)]
            else:
                directions = [axis.scale(-1).unit()]
        elif kind == "tetrahedral":
            bisector = (bonds[0] + bonds[1]).unit()
            normal = bonds[0].cross(bonds[1]).unit()
            directions = [bisector.scale(-np.sqrt(1 / 3)) + normal.scale(sign * np.sqrt(2 / 3)) for sign in (-1, 1)]
        else:
            axis = bonds[0]
            # The third point defines the carbonyl/conjugated plane. Its
            # component parallel to the bond is removed by the cross products.
            normal = axis.cross(points[2] - points[1])
            if np.linalg.norm(normal.value) < 1e-8:
                directions = [axis.scale(-1)]
            else:
                tangent = normal.unit().cross(axis).unit()
                directions = [axis.scale(-0.5) + tangent.scale(sign * np.sqrt(3) / 2) for sign in (-1, 1)]
        frames[atom] = (
            np.array([v.value for v in directions]),
            indices,
            np.array([v.jacobian for v in directions]) if gradient else None,
        )
    return frames


def _normalize_rows(vectors):
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths < 1e-8):
        raise ValueError("AD4 angular scoring requires nonzero donor bonds and hydrogen–acceptor distances")
    return vectors / lengths[:, None], lengths


def _unit_pullback(unit, lengths, adjoint):
    return (adjoint - unit * np.sum(unit * adjoint, axis=1)[:, None]) / lengths[:, None]


def angular_factors(coordinates, donors, acceptors, parents, frames, gradient=True):
    """Cosine-squared donor and best lone-pair factors and sparse derivatives.

    Each sparse derivative is (contact index, atom index, 3-vector). Explicit
    neighbor forces are essential when the acceptor is on the flexible ligand.
    """
    count = len(donors)
    if not count:
        return np.empty(0), (np.empty(0, int), np.empty(0, int), np.empty((0, 3)))
    donor_parent = parents[donors]
    if np.any(donor_parent < 0):
        raise ValueError("AD4 directional donor hydrogen is missing its bonded parent")
    forward, forward_lengths = _normalize_rows(coordinates[donors] - coordinates[donor_parent])
    toward, toward_lengths = _normalize_rows(coordinates[acceptors] - coordinates[donors])
    cosines = np.einsum("ij,ij->i", forward, toward)
    positive = np.maximum(cosines, 0)
    donor_factor = positive**2
    acceptor_factor = np.ones(count)
    acceptor_derivatives = []
    for acceptor in np.unique(acceptors):
        contacts = np.flatnonzero(acceptors == acceptor)
        vectors, indices, jacobian = frames[int(acceptor)]
        if vectors is None:
            continue
        # Acceptor-to-H points opposite the H-to-acceptor donor vector.
        cosines_lp = -toward[contacts] @ vectors.T
        chosen = np.argmax(cosines_lp, axis=1)
        positive_lp = np.maximum(cosines_lp[np.arange(len(contacts)), chosen], 0)
        acceptor_factor[contacts] = positive_lp**2
        if gradient:
            v_adjoint = 2 * positive_lp[:, None] * vectors[chosen]
            g_hydrogen = _unit_pullback(-toward[contacts], toward_lengths[contacts], v_adjoint)
            acceptor_derivatives.append((contacts, donors[contacts], g_hydrogen))
            acceptor_derivatives.append((contacts, acceptors[contacts], -g_hydrogen))
            if jacobian is not None:
                lp_adjoint = -2 * positive_lp[:, None] * toward[contacts]
                neighbor_gradients = np.einsum("ni,nipq->npq", lp_adjoint, jacobian[chosen])
                for j, index in enumerate(indices):
                    acceptor_derivatives.append((contacts, np.full(len(contacts), index), neighbor_gradients[:, j]))
    factors = donor_factor * acceptor_factor
    if not gradient:
        return factors, None
    contact_ids = np.arange(count)
    g_forward = _unit_pullback(forward, forward_lengths, 2 * positive[:, None] * toward)
    g_toward = _unit_pullback(toward, toward_lengths, 2 * positive[:, None] * forward)
    derivatives = [
        (contact_ids, donor_parent, -g_forward * acceptor_factor[:, None]),
        (contact_ids, donors, (g_forward - g_toward) * acceptor_factor[:, None]),
        (contact_ids, acceptors, g_toward * acceptor_factor[:, None]),
    ]
    derivatives += [
        (contact, atom, value * donor_factor[contact, None]) for contact, atom, value in acceptor_derivatives
    ]
    return factors, tuple(np.concatenate([entry[i] for entry in derivatives]) for i in range(3))


def saturated_contacts(donors, acceptors, energies, capacities):
    """Minimum-energy capacitated matching: H donates once, N accepts once, O/S twice.

    Only negative contacts enter the assignment. All repulsions survive. The
    result is a continuous, piecewise differentiable minimum over feasible
    assignments; at exact ties the selected assignment supplies a subgradient.
    """
    keep = np.ones(len(energies), dtype=bool)
    attractive = np.flatnonzero(energies < 0)
    if not len(attractive):
        return keep
    d, a = donors[attractive], acceptors[attractive]
    unique_d, inverse_d = np.unique(d, return_inverse=True)
    unique_a = np.unique(a)
    slots = np.repeat(unique_a, capacities[unique_a])
    # One zero-cost dummy per donor allows every contact to remain unbound.
    costs = np.zeros((len(unique_d), len(slots) + len(unique_d)))
    for acceptor in unique_a:
        edges = np.flatnonzero(a == acceptor)
        columns = np.flatnonzero(slots == acceptor)
        costs[np.ix_(inverse_d[edges], columns)] = energies[attractive[edges], None]
    rows, columns = linear_sum_assignment(costs)
    assigned = (columns < len(slots)) & (costs[rows, columns] < 0)
    lookup = {(int(donor), int(acceptor)): index for index, donor, acceptor in zip(attractive, d, a)}
    keep[attractive] = False
    for row, column in zip(rows[assigned], columns[assigned]):
        keep[lookup[int(unique_d[row]), int(slots[column])]] = True
    return keep
