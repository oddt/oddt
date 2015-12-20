"""Spatial functions included in ODDT
Mainly used by other modules, but can be accessed directly.
"""

from math import sin, cos
import math
import numpy as np
from scipy.spatial.distance import cdist
# for Hungarian algorithm, in future use scipy.optimize.linear_sum_assignment (in scipy 0.17+)
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    from sklearn.utils.linear_assignment_ import linear_assignment

    def linear_sum_assignment(M):
        out = linear_assignment(M)
        return out[:, 0], out[:, 1]

from numba import autojit, double

import oddt

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
            - min_symmetry - makes multiple molecule-molecule matches and finds minimal RMSD (the slowest)

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
                assert len(mol_idx) == len(ref_idx), 'Unequal no. atoms of type: %s' % a_type
                if len(mol_idx) == 1:
                    mol_map.append(mol_idx)
                    ref_map.append(ref_idx)
                    continue
                M = distance(mol.atom_dict['coords'][mol_idx], ref.atom_dict['coords'][ref_idx])
                M = M - M.min(axis=0) - M.min(axis=1).reshape(-1, 1)
                tmp_mol, tmp_ref = linear_sum_assignment(M)
                mol_map.append(mol_idx[tmp_mol])
                ref_map.append(ref_idx[tmp_ref])
        mol_atoms = mol.atom_dict['coords'][np.hstack(mol_map)]
        ref_atoms = ref.atom_dict['coords'][np.hstack(ref_map)]
    elif method == 'min_symmetry':
        min_rmsd = None
        ref_atoms = ref.coords[ref.atom_dict['atomicnum'] != 1]
        mol_atoms = mol.coords[mol.atom_dict['atomicnum'] != 1]
        for match in oddt.toolkit.Smarts(ref).findall(mol, unique=False):
            match = np.array(match, dtype=int)
            if oddt.toolkit.backend == 'ob':
                match -= 1
            rmsd = np.sqrt(((mol_atoms - ref_atoms[match])**2).sum(axis=-1).mean())
            if min_rmsd is None or rmsd < min_rmsd:
                min_rmsd = rmsd
        return min_rmsd
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


def distance(x, y):
    """Computes distance between each pair of points from x and y.

    Parameters
    ----------
    x : numpy arrays, shape = [n_x, 3]
        Array of poinds in 3D

    y : numpy arrays, shape = [n_y, 3]
        Array of poinds in 3D

    Returns
    -------
    dist_matrix : numpy arrays, shape = [n_x, n_y]
        Distance matrix
    """
    return cdist(x, y)


def distance_complex(x, y):
    """ Computes distance between points, similar to distance(cdist),
    with major difference - allows higher dimmentions of input (cdist supports 2).
    distance is purely float64 and can de slightly more precise.

    Parameters
    ----------
    x : numpy arrays, shape = [..., 3]
        Array of poinds in 3D

    y : numpy arrays, shape = [..., 3]
        Array of poinds in 3D

    Returns
    -------
    dist_matrix : numpy arrays
        Distance matrix
    """
    return np.linalg.norm(x[..., np.newaxis, :] - y, axis=-1)


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


# Experimental Numba support goes below
# Numba helper functions that replace numpy
@autojit
def numba_cross(vec1, vec2):
    """ Calculate the cross product of two 3d vectors. """
    n = len(vec1)
    result = np.zeros((n, 3))
    for i in range(n):
        a1, a2, a3 = double(vec1[i][0]), double(vec1[i][1]), double(vec1[i][2])
        b1, b2, b3 = double(vec2[i][0]), double(vec2[i][1]), double(vec2[i][2])

        result[i][0] = a2 * b3 - a3 * b2
        result[i][1] = a3 * b1 - a1 * b3
        result[i][2] = a1 * b2 - a2 * b1
    return result


@autojit
def numba_dot(vec1, vec2):
    """ Calculate the dot product of two vectors. """
    result = np.zeros((vec1.shape[0], vec2.shape[1]))
    for i in range(vec1.shape[0]):
        for j in range(vec2.shape[1]):
            for d in range(vec1.shape[1]):
                result[i][j] += vec1[i][d] * vec2[d][j]
    return result


@autojit
def numba_norm(vec):
    """ Calculate the norm of a vector. """
    M, N = vec.shape
    result = np.zeros(M)
    for i in range(M):
        for d in range(N):
            result[i] += vec[i][d]**2
        result[i] = math.sqrt(result[i])
    return result


@autojit
def numba_normalize(vec):
    """ Calculate the normalized vector (norm: one). """
    return vec / numba_norm(vec).reshape(-1, 1)


# Numba versions of ODDT's functions
@autojit
def numba_angle_2v(v1, v2):
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
    # doesn't work with broadcasting (check if ndims = 3)
    v1_M, v1_N = v1.shape
    v2_M, v2_N = v2.shape
    result = np.zeros(v1_M)
    result_norm = numba_norm(v1)*numba_norm(v2)
    for i in range(v1_M):
        for d in range(v1_N):
            result[i] += v1[i][d]*v2[i][d]
        result[i] /= result_norm[i]
        # clip values due to rounding
        if result[i] > 1:
            result[i] = 1
        elif result[i] < -1:
            result[i] = -1
        result[i] = math.degrees(math.acos(result[i]))
    return result


@autojit
def numba_angle(p1, p2, p3):
    v1 = p1-p2
    v2 = p3-p2
    return numba_angle_2v(v1, v2)


@autojit
def numba_dihedral(p1, p2, p3, p4):
    # BUG! works for series (2dim), fix for points (1dim)
    v12 = (p1 - p2)/numba_norm(p1-p2).reshape(-1, 1)
    v23 = (p2 - p3)/numba_norm(p2-p3).reshape(-1, 1)
    v34 = (p3 - p4)/numba_norm(p3-p4).reshape(-1, 1)
    c1 = numba_cross(v12, v23)
    c2 = numba_cross(v23, v34)
    out = numba_angle_2v(c1, c2)
    # check clockwise and anticlockwise
    n1 = c1/numba_norm(c1).reshape(-1, 1)

    # not numba save
    mask = (n1*v34).sum(axis=-1) > 0
    if len(mask.shape) == 0:
        if mask:
            out = -out
    else:
        out[mask] = -out[mask]
    return out


@autojit
def numba_distance(X, Y):
    X_M = X.shape[0]
    X_N = X.shape[1]
    Y_M = Y.shape[0]
    Y_N = Y.shape[1]
    if X_N == Y_N:
        N = X_N
    else:
        raise Exception('Wrong dims')
    D = np.empty((X_M, Y_M), dtype=np.float)
    for i in range(X_M):
        for j in range(Y_M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - Y[j, k]
                d += tmp * tmp
            D[i, j] = math.sqrt(d)
    return D


# def init():
#     """ call all functions once to compile them """
#     vec1, vec2 = np.random.random((10,3)), np.random.random((10,3))
#     numba_cross(vec1, vec2)
#     numba_norm(vec1)
#     numba_normalize(vec1)
# init()
