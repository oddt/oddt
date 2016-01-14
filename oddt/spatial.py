"""Spatial functions included in ODDT
Mainly used by other modules, but can be accessed directly.
"""

from math import sin, cos
import numpy as np
from scipy.spatial.distance import cdist as distance

__all__ = ['angle', 'angle_2v', 'dihedral', 'distance']

# angle functions
def angle(p1,p2,p3):
    """Returns an angle from a series of 3 points (point #2 is centroid).Angle is returned in degrees.

    Parameters
    ----------
    p1,p2,p3 : numpy arrays, shape = [n_points, n_dimensions]
        Triplets of points in n-dimensional space, aligned in rows.

    Returns
    -------
    angles : numpy array, shape = [n_points]
        Series of angles in degrees
    """
    v1 = p1-p2
    v2 = p3-p2
    return angle_2v(v1,v2)

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
    dot = (v1*v2).sum(axis=-1) # better than np.dot(v1, v2), multiple vectors can be applied
    norm = np.linalg.norm(v1, axis=-1)* np.linalg.norm(v2, axis=-1)
    return np.degrees(np.arccos(np.clip(dot/norm, -1, 1)))

def dihedral(p1,p2,p3,p4):
    """Returns an dihedral angle from a series of 4 points. Dihedral is returned in degrees.
    Function distingishes clockwise and antyclockwise dihedrals.

    Parameters
    ----------
    p1,p2,p3,p4 : numpy arrays, shape = [n_points, n_dimensions]
        Quadruplets of points in n-dimensional space, aligned in rows.

    Returns
    -------
    angles : numpy array, shape = [n_points]
        Series of angles in degrees
    """
    v12 = (p1-p2)/np.linalg.norm(p1-p2)
    v23 = (p2-p3)/np.linalg.norm(p2-p3)
    v34 = (p3-p4)/np.linalg.norm(p3-p4)
    c1 = np.cross(v12, v23)
    c2 = np.cross(v23, v34)
    out = angle_2v(c1, c2)
    # check clockwise and anticlockwise
    n1 = c1/np.linalg.norm(c1)
    mask = (n1*v34).sum(axis=-1) > 0
    if len(mask.shape) == 0:
        if mask:
            out = -out
    else:
        out[mask] = -out[mask]
    return out

def rmsd(ref, mol, ignore_h = True, canonize = False, normalize = False):
    """Computes root mean square deviation (RMSD) between two molecules (including or excluding Hydrogens). No symmetry checks are performed.

    Parameters
    ----------
    ref : oddt.toolkit.Molecule object
        Reference molecule for the RMSD calculation

    mol : oddt.toolkit.Molecule object
        Query molecule for RMSD calculation

    ignore_h : bool (default=False)
        Flag indicating to ignore Hydrogen atoms while performing RMSD calculation

    canonize : bool (default=False)
        Match heavy atoms using OB canonical ordering

    normalize : bool (default=False)
        Normalize RMSD by square root of rot. bonds

    Returns
    -------
    rmsd : float
        RMSD between two molecules
    """
    if ignore_h:
        if canonize:
            ref_hvy = ref.coords[ref.canonic_order]
            mol_hvy = mol.coords[mol.canonic_order]
        else:
            hvy_map = np.array([atom.idx-1 for atom in mol if atom.atomicnum != 1])
            mol_hvy = mol.coords[hvy_map]
            ref_hvy = ref.coords[hvy_map]
        if mol_hvy.shape == ref_hvy.shape:
            rmsd = np.sqrt(((mol_hvy - ref_hvy)**2).sum(axis=-1).mean())
            if normalize:
                rmsd /= np.sqrt(mol.num_rotors)
            return rmsd
    else:
        if mol.coords.shape == ref.coords.shape:
            rmsd = np.sqrt(((mol.coords - ref.coords)**2).sum(axis=-1).mean())
            if normalize:
                    rmsd /= np.sqrt(mol.num_rotors)
            return rmsd
    # at this point raise an exception
    raise Exception('Unequal number of atoms in molecules')

def distance_complex(x, y):
    """ Computes distance between points, similar to distance(cdist), with major difference - allows higher dimmentions of input (cdist supports 2). But it's 2-6 times slower, so use distance unless you have to nest it wit a for loop."""
    return np.sqrt(((x[...,np.newaxis,:]-y)**2).sum(axis=-1))

def rotate(coords, alpha, beta, gamma):
    """Returns an angle from a series of 3 points (point #2 is centroid).Angle is returned in degrees.

    Parameters
    ----------
    coords : numpy arrays, shape = [n_points, 3]
        Coordinates in 3-dimensional space.

    alpha, beta, gamma: float
        Angles to rotate the coordinates along X,Y and Z axis. Angles are specified in radians.

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

    rot_matrix = np.array([[cos_beta*cos_gamma, sin_alpha*sin_beta*cos_gamma-cos_alpha*sin_gamma, cos_alpha*sin_beta*cos_gamma+sin_alpha*sin_gamma],
                           [cos_beta*sin_gamma, sin_alpha*sin_beta*sin_gamma+cos_alpha*cos_gamma, cos_alpha*sin_beta*sin_gamma-sin_alpha*cos_gamma],
                           [-sin_beta,          sin_alpha*cos_beta,                               cos_alpha*cos_beta]])

    return (coords[:,np.newaxis,:] * rot_matrix).sum(axis=2) + centroid
