"""Code common to all toolkits"""
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
    hbond_dist_mask = np.abs(res_dict[res_dict['isalpha']]['id'] -
                             res_dict[res_dict['isalpha']]['id'][:, np.newaxis]) >= 3
    hbond_mask = distance(res_dict[res_dict['isalpha']]['N'],
                          res_dict[res_dict['isalpha']]['O']) < 3.5
    p_mask = ((hbond_mask & hbond_dist_mask).any(axis=0) |
              (hbond_mask & hbond_dist_mask).any(axis=1))
    res_dict['isalpha'][np.argwhere(res_dict['isalpha']).flatten()[~p_mask]] = False
    # Ignore groups smaller than 3
    res_mask_alpha = np.argwhere(res_dict['isalpha']).flatten()
    for mask_group in np.split(res_mask_alpha, np.argwhere(np.diff(res_mask_alpha) != 1).flatten() + 1):
        if len(mask_group) < 3:
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
    res_dist_mask = np.abs(res_dict[res_dict['isbeta']]['id'] -
                           res_dict[res_dict['isbeta']]['id'][:, np.newaxis]) >= 4
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
        if len(mask_group) < 3:
            res_dict['isbeta'][mask_group] = False

    return res_dict
