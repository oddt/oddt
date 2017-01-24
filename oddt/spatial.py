"""Spatial functions included in ODDT
Mainly used by other modules, but can be accessed directly.
"""

from math import sin, cos
from six import PY3
import numpy as np
from scipy.spatial.distance import cdist as distance
# for Hungarian algorithm, in future use scipy.optimize.linear_sum_assignment (in scipy 0.17+)
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    from sklearn.utils.linear_assignment_ import linear_assignment

    def linear_sum_assignment(M):
        out = linear_assignment(M)
        return out[:, 0], out[:, 1]

__all__ = ['angle',
           'angle_2v',
           'dihedral',
           'distance',
           'rmsd',
           'rotate']


def angle(p1, p2, p3):
    """Returns an angle from a series of 3 points (point #2 is centroid).
    Angle is returned in degrees.

    Parameters
    ----------
    p1,p2,p3 : numpy arrays, shape = [n_points, n_dimensions]
        Triplets of points in n-dimensional space, aligned in rows.

    Returns
    -------
    angles : numpy array, shape = [n_points]
        Series of angles in degrees
    """
    v1 = p1 - p2
    v2 = p3 - p2
    return angle_2v(v1, v2)


def angle_2v(v1, v2):
    """Returns an angle between two vecors.Angle is returned in degrees.

    Parameters
    ----------
    v1,v2 : numpy arrays, shape = [n_vectors, n_dimensions]
        Pairs of vectors in n-dimensional space, aligned in rows.

    Returns
    -------
    angles : numpy array, shape = [n_vectors]
        Series of angles in degrees
    """
    dot = (v1 * v2).sum(axis=-1)  # better than np.dot(v1, v2), multiple vectors can be applied
    norm = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    return np.degrees(np.arccos(np.clip(dot/norm, -1, 1)))


def dihedral(p1, p2, p3, p4):
    """Returns an dihedral angle from a series of 4 points.
    Dihedral is returned in degrees.
    Function distingishes clockwise and antyclockwise dihedrals.

    Parameters
    ----------
    p1, p2, p3, p4 : numpy arrays, shape = [n_points, n_dimensions]
        Quadruplets of points in n-dimensional space, aligned in rows.

    Returns
    -------
    angles : numpy array, shape = [n_points]
        Series of angles in degrees
    """
    v12 = (p1 - p2)/np.linalg.norm(p1 - p2)
    v23 = (p2 - p3)/np.linalg.norm(p2 - p3)
    v34 = (p3 - p4)/np.linalg.norm(p3 - p4)
    c1 = np.cross(v12, v23)
    c2 = np.cross(v23, v34)
    out = angle_2v(c1, c2)
    # check clockwise and anticlockwise
    n1 = c1 / np.linalg.norm(c1)
    mask = (n1 * v34).sum(axis=-1) > 0
    if len(mask.shape) == 0:
        if mask:
            out = -out
    else:
        out[mask] = -out[mask]
    return out


def rmsd(ref, mol, ignore_h=True, method=None, normalize=False):
    """Computes root mean square deviation (RMSD) between two molecules
    (including or excluding Hydrogens). No symmetry checks are performed.

    Parameters
    ----------
    ref : oddt.toolkit.Molecule object
        Reference molecule for the RMSD calculation

    mol : oddt.toolkit.Molecule object
        Query molecule for RMSD calculation

    ignore_h : bool (default=False)
        Flag indicating to ignore Hydrogen atoms while performing RMSD calculation

    method : str (default=None)
        The method to be used for atom asignment between ref and mol.
        None means that direct matching is applied, which is the default behavior.
        Available methods:
            - canonize - match heavy atoms using OB canonical ordering (it forces ignoring H's)
            - hungarian - minimize RMSD using Hungarian algorithm

    normalize : bool (default=False)
        Normalize RMSD by square root of rot. bonds

    Returns
    -------
    rmsd : float
        RMSD between two molecules
    """

    if method == 'canonize':
        ref_atoms = ref.coords[ref.canonic_order]
        mol_atoms = mol.coords[mol.canonic_order]
    elif method == 'hungarian':
        mol_map = []
        ref_map = []
        for a_type in np.unique(mol.atom_dict['atomtype']):
            if a_type != 'H' or not ignore_h:
                mol_idx = np.argwhere(mol.atom_dict['atomtype'] == a_type).flatten()
                ref_idx = np.argwhere(ref.atom_dict['atomtype'] == a_type).flatten()
                if len(mol_idx) == 1:
                    mol_map.append(mol_idx)
                    ref_map.append(ref_idx)
                    continue
                M = distance(mol.atom_dict['coords'][ref_idx], ref.atom_dict['coords'][ref_idx])
                M = M - M.min(axis=0) - M.min(axis=1).reshape(-1, 1)
                tmp_mol, tmp_ref = linear_sum_assignment(M)
                mol_map.append(mol_idx[tmp_mol])
                ref_map.append(ref_idx[tmp_ref])
        mol_atoms = mol.atom_dict['coords'][np.hstack(mol_map)]
        ref_atoms = ref.atom_dict['coords'][np.hstack(ref_map)]
    elif ignore_h:
        hvy_map = np.argwhere(mol.atom_dict['atomicnum'] != 1).flatten()
        mol_atoms = mol.coords[hvy_map]
        ref_atoms = ref.coords[hvy_map]
    else:
        mol_atoms = mol.coords
        ref_atoms = ref.coords
    if mol_atoms.shape == ref_atoms.shape:
        rmsd = np.sqrt(((mol_atoms - ref_atoms)**2).sum(axis=-1).mean())
        if normalize:
            rmsd /= np.sqrt(mol.num_rotors)
        return rmsd
    # at this point raise an exception
    raise Exception('Unequal number of atoms in molecules')


def distance_complex(x, y):
    """ Computes distance between points, similar to distance(cdist),
    with major difference - allows higher dimmentions of input (cdist supports 2).
    But it's 2-6 times slower, so use distance unless you have to nest it with a for loop.
    """
    return np.sqrt(((x[..., np.newaxis, :] - y)**2).sum(axis=-1))


def rotate(coords, alpha, beta, gamma):
    """Rotate coords by cerain angle in X, Y, Z. Angles are specified in radians.

    Parameters
    ----------
    coords : numpy arrays, shape = [n_points, 3]
        Coordinates in 3-dimensional space.

    alpha, beta, gamma: float
        Angles to rotate the coordinates along X, Y and Z axis.
        Angles are specified in radians.

    Returns
    -------
    new_coords : numpy arrays, shape = [n_points, 3]
        Rorated coordinates in 3-dimensional space.
    """
    centroid = coords.mean(axis=0)
    coords = coords - centroid

    sin_alpha = sin(alpha)
    cos_alpha = cos(alpha)
    sin_beta = sin(beta)
    cos_beta = cos(beta)
    sin_gamma = sin(gamma)
    cos_gamma = cos(gamma)

    rot_matrix = np.array([[cos_beta * cos_gamma,
                            sin_alpha * sin_beta * cos_gamma - cos_alpha * sin_gamma,
                            cos_alpha * sin_beta * cos_gamma + sin_alpha * sin_gamma],
                           [cos_beta * sin_gamma,
                            sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma,
                            cos_alpha * sin_beta * sin_gamma - sin_alpha * cos_gamma],
                           [-sin_beta,
                            sin_alpha * cos_beta,
                            cos_alpha * cos_beta]])

    return (coords[:, np.newaxis, :] * rot_matrix).sum(axis=2) + centroid
