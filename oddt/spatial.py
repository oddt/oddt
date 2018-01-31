"""Spatial functions included in ODDT
Mainly used by other modules, but can be accessed directly.
"""

from math import sin, cos

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

import oddt
from oddt.utils import is_openbabel_molecule

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
    # better than np.dot(v1, v2), multiple vectors can be applied
    dot = (v1 * v2).sum(axis=-1)
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
        Flag indicating to ignore Hydrogen atoms while performing RMSD
        calculation. This toggle works only with 'hungarian' method and without
        sorting (method=None).

    method : str (default=None)
        The method to be used for atom asignment between ref and mol.
        None means that direct matching is applied, which is the default
        behavior.
        Available methods:
            - canonize - match heavy atoms using canonical ordering (it forces
            ignoring H's)
            - hungarian - minimize RMSD using Hungarian algorithm
            - min_symmetry - makes multiple molecule-molecule matches and finds
            minimal RMSD (the slowest). Hydrogens are ignored.

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
                if len(mol_idx) != len(ref_idx):
                    raise ValueError('Unequal number of atoms type: %s' % a_type)
                if len(mol_idx) == 1:
                    mol_map.append(mol_idx)
                    ref_map.append(ref_idx)
                    continue
                M = distance(mol.atom_dict['coords'][mol_idx],
                             ref.atom_dict['coords'][ref_idx])
                M = M - M.min(axis=0) - M.min(axis=1).reshape(-1, 1)
                tmp_mol, tmp_ref = linear_sum_assignment(M)
                mol_map.append(mol_idx[tmp_mol])
                ref_map.append(ref_idx[tmp_ref])
        mol_atoms = mol.atom_dict['coords'][np.hstack(mol_map)]
        ref_atoms = ref.atom_dict['coords'][np.hstack(ref_map)]
    elif method == 'min_symmetry':
        min_rmsd = None
        ref_atoms = ref.atom_dict[ref.atom_dict['atomicnum'] != 1]['coords']
        mol_atoms = mol.atom_dict[mol.atom_dict['atomicnum'] != 1]['coords']
        # safety swith to check if number of heavy atoms match
        if ref_atoms.shape == mol_atoms.shape:
            # match mol to ref, generate all matches to find best RMSD
            matches = oddt.toolkit.Smarts(ref).findall(mol, unique=False)
            if not matches:
                raise ValueError('Could not find any match between molecules.')
            # calculate RMSD between all matches and retain the smallest
            for match in matches:
                match = np.array(match, dtype=int)
                if is_openbabel_molecule(mol):
                    match -= 1  # OB has 1-based indices
                tmp_dict = mol.atom_dict[match]
                mol_atoms = tmp_dict[tmp_dict['atomicnum'] != 1]['coords']
                # following should not happen, although safety check is left
                if mol_atoms.shape != ref_atoms.shape:
                    raise Exception('Molecular match got wrong number of atoms.')
                rmsd = np.sqrt(((mol_atoms - ref_atoms)**2).sum(axis=-1).mean())
                if min_rmsd is None or rmsd < min_rmsd:
                    min_rmsd = rmsd
            return min_rmsd
    elif ignore_h:
        mol_atoms = mol.coords[mol.atom_dict['atomicnum'] != 1]
        ref_atoms = ref.coords[ref.atom_dict['atomicnum'] != 1]
    else:
        mol_atoms = mol.coords
        ref_atoms = ref.coords
    if mol_atoms.shape == ref_atoms.shape:
        rmsd = np.sqrt(((mol_atoms - ref_atoms)**2).sum(axis=-1).mean())
        if normalize:
            rmsd /= np.sqrt(mol.num_rotors)
        return rmsd
    # at this point raise an exception
    raise ValueError('Unequal number of atoms in molecules (%i and %i)'
                     % (len(mol_atoms), len(ref_atoms)))


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
