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
    """A 3-vector and an optional small local Cartesian Jacobian."""

    def __init__(self, value, jacobian):
        self.value = value
        self.jacobian = jacobian

    def __add__(self, other):
        jacobian = None if self.jacobian is None else self.jacobian + other.jacobian
        return _Vector(self.value + other.value, jacobian)

    def __sub__(self, other):
        return self + other.scale(-1)

    def scale(self, scalar):
        jacobian = None if self.jacobian is None else self.jacobian * scalar
        return _Vector(self.value * scalar, jacobian)

    def unit(self):
        length = np.linalg.norm(self.value)
        if length < 1e-8:
            raise ValueError("AD4 angular scoring requires nondegenerate bonded geometry; supply a 3D conformation")
        unit = self.value / length
        jacobian = None
        if self.jacobian is not None:
            jacobian = self.jacobian - np.einsum("i,j,jab->iab", unit, unit, self.jacobian)
            jacobian /= length
        return _Vector(unit, jacobian)

    def cross(self, other):
        jacobian = None
        if self.jacobian is not None:
            jacobian = np.cross(self.jacobian.transpose(1, 2, 0), other.value)
            jacobian += np.cross(self.value, other.jacobian.transpose(1, 2, 0))
            jacobian = jacobian.transpose(2, 0, 1)
        return _Vector(np.cross(self.value, other.value), jacobian)


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


def _frame_unit_rows(vectors):
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths < 1e-8):
        raise ValueError("AD4 angular scoring requires nondegenerate bonded geometry; supply a 3D conformation")
    return vectors / lengths[:, None], lengths


def _normalization_jacobians(unit, lengths):
    return (np.eye(3)[None] - np.einsum("ni,nj->nij", unit, unit)) / lengths[:, None, None]


def _tetrahedral_frames(coordinates, indices, gradient):
    """Batch two lone-pair lobes and their center/neighbor Jacobians."""
    points = coordinates[indices]
    first, first_lengths = _frame_unit_rows(points[:, 1] - points[:, 0])
    second, second_lengths = _frame_unit_rows(points[:, 2] - points[:, 0])
    bisector, bisector_lengths = _frame_unit_rows(first + second)
    normal, normal_lengths = _frame_unit_rows(np.cross(first, second))
    s, t = np.sqrt(1 / 3), np.sqrt(2 / 3)
    directions = np.stack((-s * bisector - t * normal, -s * bisector + t * normal), axis=1)
    if not gradient:
        return directions, None

    # Axes: frame, vector component, defining atom, Cartesian coordinate.
    first_jacobian = np.zeros((len(indices), 3, 3, 3))
    second_jacobian = np.zeros_like(first_jacobian)
    p_first = _normalization_jacobians(first, first_lengths)
    p_second = _normalization_jacobians(second, second_lengths)
    first_jacobian[:, :, 0, :], first_jacobian[:, :, 1, :] = -p_first, p_first
    second_jacobian[:, :, 0, :], second_jacobian[:, :, 2, :] = -p_second, p_second
    bisector_jacobian = np.einsum(
        "nij,njkl->nikl", _normalization_jacobians(bisector, bisector_lengths), first_jacobian + second_jacobian
    )
    cross_jacobian = np.cross(first_jacobian.transpose(0, 2, 3, 1), second[:, None, None, :])
    cross_jacobian += np.cross(first[:, None, None, :], second_jacobian.transpose(0, 2, 3, 1))
    normal_jacobian = np.einsum(
        "nij,njkl->nikl", _normalization_jacobians(normal, normal_lengths), cross_jacobian.transpose(0, 3, 1, 2)
    )
    jacobian = np.stack(
        (-s * bisector_jacobian - t * normal_jacobian, -s * bisector_jacobian + t * normal_jacobian), axis=1
    )
    return directions, jacobian


def acceptor_frames(coordinates, rules, gradient=True):
    """Lone-pair unit vectors with derivatives through all defining neighbors."""
    frames = {}
    tetrahedral = []
    for atom, (kind, indices) in rules.items():
        if kind == "tetrahedral":
            tetrahedral.append((atom, indices))
            continue
        indices = np.asarray(indices, dtype=int)
        if kind == "spherical":
            frames[atom] = (None, indices, None)
            continue
        points = []
        for i, index in enumerate(indices):
            jacobian = None
            if gradient:
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
    if tetrahedral:
        indices = np.array([indices for _, indices in tetrahedral], dtype=int)
        directions, jacobian = _tetrahedral_frames(coordinates, indices, gradient)
        for i, (atom, _) in enumerate(tetrahedral):
            frames[atom] = (directions[i], indices[i], jacobian[i] if gradient else None)
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
    # A nonpositive donor cosine makes both the product and its derivative
    # zero. Validate all donor geometry above, then omit these acceptor/force
    # calculations while retaining original contact indices in sparse output.
    contact_ids = np.flatnonzero(cosines > 0)
    active_acceptors = acceptors[contact_ids]
    acceptor_factor = np.ones(count)
    acceptor_derivatives = []
    for acceptor in np.unique(active_acceptors):
        contacts = contact_ids[active_acceptors == acceptor]
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
    g_forward = _unit_pullback(
        forward[contact_ids], forward_lengths[contact_ids], 2 * positive[contact_ids, None] * toward[contact_ids]
    )
    g_toward = _unit_pullback(
        toward[contact_ids], toward_lengths[contact_ids], 2 * positive[contact_ids, None] * forward[contact_ids]
    )
    active_factor = acceptor_factor[contact_ids, None]
    derivatives = [
        (contact_ids, donor_parent[contact_ids], -g_forward * active_factor),
        (contact_ids, donors[contact_ids], (g_forward - g_toward) * active_factor),
        (contact_ids, acceptors[contact_ids], g_toward * active_factor),
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
