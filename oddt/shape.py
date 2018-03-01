from __future__ import print_function
import sys
import numpy as np
from numpy.linalg import norm
from scipy.stats import moment
from scipy.special import cbrt


def common_usr(molecule, ctd=None, cst=None, fct=None, ftf=None, atoms_type=None):
    """Function used in USR and USRCAT function

    Parameters
    ----------
    molecule : oddt.toolkit.Molecule
        Molecule to compute USR shape descriptor

    ctd : numpy array or None (default = None)
        Coordinates of the molecular centroid
        If 'None', the point is calculated

    cst : numpy array or None (default = None)
        Coordinates of the closest atom to the molecular centroid
        If 'None', the point is calculated

    fct : numpy array or None (default = None)
        Coordinates of the farthest atom to the molecular centroid
        If 'None', the point is calculated

    ftf : numpy array or None (default = None)
        Coordinates of the farthest atom
        to the farthest atom to the molecular centroid
        If 'None', the point is calculated

    atoms_type : str or None (default None)
        Type of atoms to be selected from atom_dict
        If 'None', all atoms are used to calculate shape descriptor

    Returns
    -------
    shape_descriptor : numpy array, shape = (12)
        Array describing shape of molecule
    """
    if atoms_type is None:
        atoms = molecule.atom_dict['coords']
    else:
        if atoms_type == 'ishydrophobe':
            mask = (molecule.atom_dict['ishalogen'] |
                    molecule.atom_dict['ishydrophobe'] |
                    (molecule.atom_dict['atomicnum'] == 16))
        else:
            mask = molecule.atom_dict[atoms_type]
        atoms = molecule.atom_dict[mask]['coords']

    if len(atoms) == 0:
        return np.zeros(12), ((0., 0., 0.),) * 4

    if ctd is None:
        ctd = atoms.mean(0)
    distances_ctd = norm(atoms - ctd, axis=1)

    if cst is None:
        cst = atoms[distances_ctd.argmin()]
    distances_cst = norm(atoms - cst, axis=1)

    if fct is None:
        fct = atoms[distances_ctd.argmax()]
    distances_fct = norm(atoms - fct, axis=1)

    if ftf is None:
        ftf = atoms[distances_fct.argmax()]
    distances_ftf = norm(atoms - ftf, axis=1)

    distances_list = [distances_ctd, distances_cst, distances_fct, distances_ftf]

    shape_descriptor = np.zeros(12)

    for i, distances in enumerate(distances_list):
        shape_descriptor[i * 3 + 0] = np.mean(distances)
        shape_descriptor[i * 3 + 1] = np.var(distances)
        shape_descriptor[i * 3 + 2] = moment(distances, moment=3)

    return shape_descriptor, (ctd, cst, fct, ftf)


def usr(molecule):
    """Computes USR shape descriptor based on
    Ballester PJ, Richards WG (2007). Ultrafast shape recognition to search
    compound databases for similar molecular shapes. Journal of
    computational chemistry, 28(10):1711-23.
    http://dx.doi.org/10.1002/jcc.20681

    Parameters
    ----------
    molecule : oddt.toolkit.Molecule
        Molecule to compute USR shape descriptor

    Returns
    -------
    shape_descriptor : numpy array, shape = (12)
        Array describing shape of molecule
    """
    return common_usr(molecule)[0]


def usr_cat(molecule):
    """Computes USRCAT shape descriptor based on
    Adrian M Schreyer, Tom Blundell (2012). USRCAT: real-time ultrafast
    shape recognition with pharmacophoric constraints. Journal of
    Cheminformatics, 2012 4:27.
    http://dx.doi.org/10.1186/1758-2946-4-27

    Parameters
    ----------
    molecule : oddt.toolkit.Molecule
        Molecule to compute USRCAT shape descriptor

    Returns
    -------
    shape_descriptor : numpy array, shape = (60)
        Array describing shape of molecule
    """
    all_atoms_shape, points = common_usr(molecule)
    ctd, cst, fct, ftf = points
    hydrophobic_shape = common_usr(
        molecule, ctd, cst, fct, ftf, 'ishydrophobe')[0]
    aromatic_shape = common_usr(molecule, ctd, cst, fct, ftf, 'isaromatic')[0]
    acceptor_shape = common_usr(molecule, ctd, cst, fct, ftf, 'isacceptor')[0]
    donor_shape = common_usr(molecule, ctd, cst, fct, ftf, 'isdonor')[0]

    cat_shape = np.hstack((all_atoms_shape, hydrophobic_shape,
                           aromatic_shape, acceptor_shape, donor_shape))

    return np.nan_to_num(cat_shape)


def electroshape(mol):
    """Computes shape descriptor based on
    Armstrong, M. S. et al. ElectroShape: fast molecular similarity
    calculations incorporating shape, chirality and electrostatics.
    J Comput Aided Mol Des 24, 789-801 (2010).
    http://dx.doi.org/doi:10.1007/s10822-010-9374-0

    Aside from spatial coordinates, atoms' charges are also used
    as the fourth dimension to describe shape of the molecule.

    Parameters
    ----------
    mol : oddt.toolkit.Molecule
        Molecule to compute Electroshape descriptor

    Returns
    -------
    shape_descriptor : numpy array, shape = (15)
                       Array describing shape of molecule
    """
    if (mol.atom_dict['coords'] == 0).all():
        raise Exception('Molecule needs 3D coordinates')

    if np.isnan(mol.atom_dict['charge']).any():
        print('Nan values in charge values of molecule ' + mol.title, file=sys.stderr)

    charge = np.nan_to_num(mol.atom_dict['charge'])

    mi = 25  # scaling factor converting electron charges to Angstroms

    four_dimensions = np.column_stack((mol.atom_dict['coords'], charge * mi))

    c1 = four_dimensions.mean(0)  # geometric centre of the molecule
    distances_c1 = norm(four_dimensions - c1, axis=1)

    c2 = four_dimensions[distances_c1.argmax()]  # atom position furthest from c1
    distances_c2 = norm(four_dimensions - c2, axis=1)

    c3 = four_dimensions[distances_c2.argmax()]  # atom position furthest from c2
    distances_c3 = norm(four_dimensions - c3, axis=1)

    vector_a = c2 - c1
    vector_b = c3 - c1
    vector_as = vector_a[:3]    # spatial parts of these vectors -
    vector_bs = vector_b[:3]    # the first three coordinates
    vector_c = ((norm(vector_a) /
                (2 * norm(np.cross(vector_as, vector_bs))))
                * np.cross(vector_as, vector_bs))

    vector_c1s = c1[:3]

    max_charge = np.array(np.amax(charge) * mi)
    min_charge = np.array(np.amin(charge) * mi)

    c4 = np.append(vector_c1s + vector_c, max_charge)
    c5 = np.append(vector_c1s + vector_c, min_charge)

    distances_c4 = norm(four_dimensions - c4, axis=1)
    distances_c5 = norm(four_dimensions - c5, axis=1)

    distances_list = [distances_c1, distances_c2, distances_c3,
                      distances_c4, distances_c5]

    shape_descriptor = np.zeros(15)

    i = 0
    for distances in distances_list:
        mean = np.mean(distances)
        shape_descriptor[0 + i] = mean
        shape_descriptor[1 + i] = np.std(distances)
        shape_descriptor[2 + i] = cbrt(np.sum(((distances - mean) ** 3) / distances.size))
        i += 3

    return shape_descriptor


def usr_similarity(mol1_shape, mol2_shape, ow=1., hw=1., rw=1., aw=1., dw=1.):
    """Computes similarity between molecules

    Parameters
    ----------
    mol1_shape : numpy array
        USR shape descriptor

    mol2_shape : numpy array
        USR shape descriptor

    ow : float (default = 1.)
        Scaling factor for all atoms
        Only used for USRCAT, ignored for other types

    hw : float (default = 1.)
        Scaling factor for hydrophobic atoms
        Only used for USRCAT, ignored for other types

    rw : float (default = 1.)
        Scaling factor for aromatic atoms
        Only used for USRCAT, ignored for other types

    aw : float (default = 1.)
        Scaling factor for acceptors
        Only used for USRCAT, ignored for other types

    dw : float (default = 1.)
        Scaling factor for donors
        Only used for USRCAT, ignored for other types

    Returns
    -------
    similarity : float from 0 to 1
        Similarity between shapes of molecules,
        1 indicates identical molecules
    """
    if mol1_shape.shape[0] == 12 and mol2_shape.shape[0] == 12:
        sim = 1. / (1. + (1. / 12) * np.sum(np.fabs(mol1_shape - mol2_shape)))
    elif mol1_shape.shape[0] == 60 and mol2_shape.shape[0] == 60:
        w = np.array([ow, hw, rw, aw, dw])
        # Normalize weights
        w = w / w.sum()
        shape_diff = np.abs(mol1_shape - mol2_shape).reshape(-1, 12)
        sim = 1. / (1 + (w * (1. / 12) * shape_diff.sum(axis=1)).sum())
    elif mol1_shape.shape[0] == 15 and mol2_shape.shape[0] == 15:
        sim = 1. / (1 + (1. / 15) * np.sum(np.fabs(mol1_shape - mol2_shape)))
    else:
        raise Exception('Given vectors are not valid USR shape descriptors '
                        'or come from different methods. Correct vector lengths'
                        'are: 12 for USR, 60 for USRCAT, 15 for Electroshape')

    return sim
