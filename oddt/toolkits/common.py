"""Code common to all toolkits"""
from collections import deque
import numpy as np

from oddt.spatial import dihedral, distance


def detect_secondary_structure(res_dict):
    """Detect alpha helices and beta sheets in res_dict by phi and psi angles"""
    first = res_dict[:-1]
    second = res_dict[1:]
    psi = dihedral(first['N'], first['CA'], first['C'], second['N'])
    phi = dihedral(first['C'], second['N'], second['CA'], second['C'])
    d = second['id'] - first['id']

    # Alpha helices
    res_mask_alpha = (((phi > -145) & (phi < -35) &
                       (psi > -70) & (psi < 50) & (d == 1)))  # alpha
    res_mask_alpha = np.union1d(np.argwhere(res_mask_alpha),
                                np.argwhere(res_mask_alpha))

    # Ignore groups smaller than 3
    for mask_group in np.split(res_mask_alpha, np.argwhere(np.diff(res_mask_alpha) != 1).flatten() + 1):
        if len(mask_group) >= 3:
            res_dict['isalpha'][mask_group] = True

    # Alpha helices have to form H-Bonds
    hbond_dist_mask = np.abs(res_dict[res_dict['isalpha']]['resnum'] -
                             res_dict[res_dict['isalpha']]['resnum'][:, np.newaxis]) >= 3
    hbond_mask = distance(res_dict[res_dict['isalpha']]['N'],
                          res_dict[res_dict['isalpha']]['O']) < 3.5
    p_mask = ((hbond_mask & hbond_dist_mask).any(axis=0) |
              (hbond_mask & hbond_dist_mask).any(axis=1))
    res_dict['isalpha'][np.argwhere(res_dict['isalpha']).flatten()[~p_mask]] = False

    # Ignore groups smaller than 3
    res_mask_alpha = np.argwhere(res_dict['isalpha']).flatten()
    for mask_group in np.split(res_mask_alpha, np.argwhere(np.diff(res_mask_alpha) != 1).flatten() + 1):
        if 0 < len(mask_group) < 3:
            res_dict['isalpha'][mask_group] = False

    # Beta sheets
    res_mask_beta = (((phi >= -180) & (phi < -40) &
                      (psi <= 180) & (psi > 90) & (d == 1)) |
                     ((phi >= -180) & (phi < -70) &
                      (psi <= -165) & (d == 1)))  # beta
    res_mask_beta = np.union1d(np.argwhere(res_mask_beta),
                               np.argwhere(res_mask_beta))

    # Ignore groups smaller than 3
    for mask_group in np.split(res_mask_beta, np.argwhere(np.diff(res_mask_beta) != 1).flatten() + 1):
        if len(mask_group) >= 3:
            res_dict['isbeta'][mask_group] = True

    # Beta strands have to be alongside eachother
    res_dist_mask = np.abs(res_dict[res_dict['isbeta']]['resnum'] -
                           res_dict[res_dict['isbeta']]['resnum'][:, np.newaxis]) >= 4
    hbond_mask = distance(res_dict[res_dict['isbeta']]['N'],
                          res_dict[res_dict['isbeta']]['O']) < 3.5
    ca_mask = distance(res_dict[res_dict['isbeta']]['CA'],
                       res_dict[res_dict['isbeta']]['CA']) < 4.5
    p_mask = ((hbond_mask & res_dist_mask).any(axis=0) |
              (hbond_mask & res_dist_mask).any(axis=1) |
              (ca_mask & res_dist_mask).any(axis=0))
    res_dict['isbeta'][np.argwhere(res_dict['isbeta']).flatten()[~p_mask]] = False

    # Ignore groups smaller than 3
    res_mask_beta = np.argwhere(res_dict['isbeta']).flatten()
    for mask_group in np.split(res_mask_beta, np.argwhere(np.diff(res_mask_beta) != 1).flatten() + 1):
        if 0 < len(mask_group) < 3:
            res_dict['isbeta'][mask_group] = False

    return res_dict


def canonize_ring_path(path):
    """Make a canonic path - list of consecutive atom IDXs bonded in a ring
    sorted in an uniform fasion.
        1) Move the smallest index to position 0
        2) Look for the smallest first step (delta IDX)
        3) Ff -1 is smallest, inverse the path and move min IDX to position 0

    Parameters
    ----------
    path : list of integers
        A list of consecutive atom indices in a ring

    Returns
    -------
    canonic_path : list of integers
        Sorted list of atoms
    """
    if isinstance(path, deque):
        path_deque = path
        path = list(path)
    elif isinstance(path, list):
        path_deque = deque(path)
    else:
        raise ValueError('Path must be a list or deque.')
    # FIXME: Py2 deque does not have deque.index()
    path_deque.rotate(-path.index(min(path)))
    if path_deque[1] - path_deque[0] > path_deque[-1] - path_deque[0]:
        path_deque.reverse()
        path_deque.rotate(1)
    return list(path_deque)
