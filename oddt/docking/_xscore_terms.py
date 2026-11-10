"""Independent X-Score (Wang et al., 2002) radial, angular and surface terms."""

from functools import lru_cache

import numpy as np
from scipy.spatial.distance import cdist


def vdw_term(distances, radii_sum):
    """Heavy-atom 8--4 Lennard-Jones potential, capped at +100 per pair (Eq. 2)."""
    power = np.minimum((radii_sum / np.maximum(distances, 1e-12)) ** 4, 1 + np.sqrt(101))
    return power**2 - 2 * power


def hydrophobic_pairs(distances, radii_sum):
    """Hydrophobic contact occupancy (Eq. 7)."""
    return np.clip((radii_sum + 2 - distances) / 1.5, 0, 1)


def root_angle(roots, centers, contacts):
    """Heavy-neighbor centroid angle factor: 0 at 60 degrees, 1 at 120."""
    first, second = roots - centers, contacts - centers
    lengths = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    cosine = np.divide(np.sum(first * second, axis=1), lengths, out=np.zeros(len(first)), where=lengths > 1e-12)
    value = np.clip((np.rad2deg(np.arccos(np.clip(cosine, -1, 1))) - 60) / 60, 0, 1)
    # Rootless ions/isolated atoms have no defined angle: isotropic adaptation.
    value[lengths <= 1e-12] = 1
    return value


def hydrogen_bonds(rec_coords, lig_coords, rec, lig, distances, radii_sum):
    """Root-angle H-bonds, with deterministic strongest-first role saturation.

    Each heavy atom pair can contribute once, even when both ends are donors
    and acceptors. Donor and acceptor capacities are counted separately. The
    publication specifies capacities, but not a unique assignment algorithm.
    """
    compatible = ((rec["donor"][:, None] > 0) & (lig["acceptor"][None, :] > 0)) | (
        (rec["acceptor"][:, None] > 0) & (lig["donor"][None, :] > 0)
    )
    rows, cols = np.nonzero(compatible & (distances < radii_sum) & (distances > 1e-10))
    if not len(rows):
        return 0.0, 0
    rroot = np.array([rec_coords[n].mean(axis=0) if len(n) else rec_coords[i] for i, n in enumerate(rec["neighbors"])])
    lroot = np.array([lig_coords[n].mean(axis=0) if len(n) else lig_coords[i] for i, n in enumerate(lig["neighbors"])])
    strength = np.clip((radii_sum[rows, cols] - distances[rows, cols]) / 0.7, 0, 1)
    strength *= root_angle(rroot[rows], rec_coords[rows], lig_coords[cols])
    strength *= root_angle(lroot[cols], lig_coords[cols], rec_coords[rows])
    offset = len(rec_coords)
    donors = np.r_[rec["donor"], lig["donor"]].copy()
    acceptors = np.r_[rec["acceptor"], lig["acceptor"]].copy()
    total, count = 0.0, 0
    for index in np.argsort(-strength, kind="stable"):
        if strength[index] <= 0:
            continue
        first, second = rows[index], cols[index] + offset
        for donor, acceptor in ((first, second), (second, first)):
            if donors[donor] > 0 and acceptors[acceptor] > 0:
                total += float(strength[index])
                count += 1
                donors[donor] -= 1
                acceptors[acceptor] -= 1
                break
    return total, count


@lru_cache(maxsize=64)
def sphere_points(count):
    """Independent equal-area Fibonacci sphere quadrature."""
    indices = np.arange(count) + 0.5
    z = 1 - 2 * indices / count
    angle = indices * (np.pi * (3 - np.sqrt(5)))
    radius = np.sqrt(1 - z**2)
    points = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    points.setflags(write=False)
    return points


def surface_frame(ligand, receptor):
    """Geometric quadrature frame that follows rigid transformations."""
    origin = ligand[0]
    axis = None
    for point in np.vstack((ligand[1:], receptor)):
        vector = point - origin
        if axis is None:
            norm = np.linalg.norm(vector)
            if norm > 1e-8:
                axis = vector / norm
        else:
            second = vector - np.dot(vector, axis) * axis
            norm = np.linalg.norm(second)
            if norm > 1e-8:
                second /= norm
                return np.column_stack((axis, second, np.cross(axis, second)))
    return np.eye(3)  # Isolated/collinear system; only quadrature accuracy applies.


def buried_surface(rec_coords, lig_coords, rec_radii, lig_radii, hydrophobic, probe, spacing):
    """Buried ligand SAS (Eq. 6), including H areas attributed to their parents."""
    rr, lr = rec_radii + probe, lig_radii + probe
    to_lig = cdist(lig_coords, lig_coords)
    to_rec = cdist(lig_coords, rec_coords)
    frame = surface_frame(lig_coords, rec_coords)
    area = 0.0
    for i in np.flatnonzero(hydrophobic):
        local = np.flatnonzero((to_lig[i] < lr[i] + lr) & (np.arange(len(lr)) != i))
        covered = (to_lig[i, local] < 1e-10) & (
            (lr[local] > lr[i] + 1e-10) | ((np.abs(lr[local] - lr[i]) < 1e-10) & (local < i))
        )
        if covered.any():
            continue
        protein = np.flatnonzero(to_rec[i] < lr[i] + rr)
        if not len(protein):
            continue
        count = max(12, int(np.ceil(4 * np.pi * lr[i] ** 2 / spacing**2)))
        points = lig_coords[i] + lr[i] * (sphere_points(count) @ frame.T)
        visible = np.all(cdist(points, lig_coords[local]) >= lr[local] - 1e-10, axis=1)
        points = points[visible]
        buried = np.any(cdist(points, rec_coords[protein]) < rr[protein], axis=1)
        area += float(buried.sum() * 4 * np.pi * lr[i] ** 2 / count)
    return area
