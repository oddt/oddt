"""This module generates and does computation with molecular surfaces.
"""

from __future__ import division
from numbers import Number
from distutils.version import LooseVersion
import warnings

import oddt.toolkits
import numpy as np
from scipy.spatial import cKDTree

try:
    from skimage.morphology import ball, binary_closing
    from skimage import __version__ as skimage_version
    if LooseVersion(skimage_version) >= LooseVersion('0.13'):
        from skimage.measure import marching_cubes_lewiner as marching_cubes
    else:
        from skimage.measure import marching_cubes
except ImportError as e:
    warnings.warn('scikit-image could not be imported and is required for'
                  'generating molecular surfaces.')
    skimage = None


def generate_surface_marching_cubes(molecule, remove_hoh=False, scaling=1.,
                                    probe_radius=1.4):
    """Generates a molecular surface mesh using the marching_cubes
    method from scikit-image. Ignores hydrogens present in the molecule.

    Parameters
    ----------
    molecule : oddt.toolkit.Molecule object
        Molecule for which the surface will be generated

    remove_hoh : bool (default = False)
        If True, remove waters from the molecule before generating the surface.
        Requires molecule.protein to be set to True.

    scaling : float (default = 1.0)
        Expands the grid in which computation is done by a factor of scaling.
        Results in a more accurate representation of the surface, but increases
        computation time.

    probe_radius : float (default = 1.4)
        Radius of a ball used to patch up holes inside the molecule
        resulting from some molecular distances being larger
        (usually in protein). Basically reduces the surface to one
        accesible by other molecules of radius smaller than probe_radius.

    Returns
    -------
    verts : numpy array
        Spatial coordinates for mesh vertices.

    faces : numpy array
        Faces are defined by referencing vertices from verts.
    """
    # Input validation
    if not isinstance(molecule, oddt.toolkit.Molecule):
        raise TypeError('molecule needs to be of type oddt.toolkit.Molecule')
    if not (isinstance(probe_radius, Number) and probe_radius >= 0):
        raise ValueError('probe_radius needs to be a positive number')

    # Removing waters and hydrogens
    atom_dict = molecule.atom_dict
    atom_dict = atom_dict[atom_dict['atomicnum'] != 1]
    if remove_hoh:
        if molecule.protein is not True:
            raise ValueError('Residue names are needed for water removal, '
                             'molecule.protein property must be set to True')
        no_hoh = atom_dict['resname'] != 'HOH'
        atom_dict = atom_dict[no_hoh]

    # Take a molecule's coordinates and atom radii and scale if necessary
    coords = atom_dict['coords'] * scaling
    radii = atom_dict['radius'] * scaling

    # More input validation
    if radii.min() < 1:
        raise ValueError('Scaling times the radius of the smallest atom must '
                         'be larger than 1')
    # Create a ball for each atom in the molecule
    ball_dict = {radius: ball(radius, dtype=bool) for radius in set(radii)}
    ball_radii = np.array([ball_dict[radius].shape[0] for radius in radii])

    # Transform the coordinates because the grid starts at (0, 0 ,0)
    min_coords = np.min(coords, axis=0)
    max_rad = np.max(ball_radii, axis=0)
    adjusted = np.round(coords - min_coords + max_rad * 5).astype(np.int64)
    offset = adjusted[0] - coords[0]

    # Calculate boundries in the grid for each ball.
    ball_coord_min = (adjusted.T - np.floor(ball_radii / 2).astype(np.int64)).T
    ball_coord_max = (ball_coord_min.T + ball_radii).T

    # Create the grid
    grid = np.zeros(shape=ball_coord_max.max(axis=0) + int(8 * scaling), dtype=bool)

    # Place balls in grid
    for radius, coord_min, coord_max in zip(radii, ball_coord_min, ball_coord_max):
        grid[coord_min[0]:coord_max[0],
             coord_min[1]:coord_max[1],
             coord_min[2]:coord_max[2]] += ball_dict[radius]
    spacing = (1 / scaling,) * 3

    # Hole-filling with morphological closing
    grid = binary_closing(grid, ball(probe_radius * 2 * scaling))

    # Marching cubes
    verts, faces = marching_cubes(grid, level=0, spacing=spacing)[:2]

    # Verts already scaled by the marching cubes function (spacing parameter)
    # Only need to scale the offset
    # Results in skimage version lower than 0.11 are offset by 1 in each direction
    if LooseVersion(skimage_version) < LooseVersion('0.11'):
        verts += 1 / scaling
    return verts - offset / scaling, faces


def find_surface_residues(molecule, max_dist=None, scaling=1.):
    """Finds residues close to the molecular surface using
    generate_surface_marching_cubes. Ignores hydrogens and
    waters present in the molecule.

    Parameters
    ----------
    molecule : oddt.toolkit.Molecule
        Molecule to find surface residues in.

    max_dist : array_like, numeric or None (default = None)
        Maximum distance from the surface where residues would
        still be considered close. If None, compares distances
        to radii of respective atoms.

    scaling : float (default = 1.0)
        Expands the grid in which computation is done by
        generate_surface_marching_cubes by a factor of scaling.
        Results in a more accurate representation of the surface,
        and therefore more accurate computation of distances
        but increases computation time.

    Returns
    -------
    atom_dict : numpy array
        An atom_dict containing only the surface residues
        from the original molecule.
    """
    # Input validation
    if not isinstance(molecule, oddt.toolkit.Molecule):
        raise TypeError('molecule needs to be of type oddt.toolkit.Molecule')

    # Copy the atom_dict, remove waters
    atom_dict = molecule.atom_dict
    mask = (atom_dict['resname'] != 'HOH') & (atom_dict['atomicnum'] != 1)
    atom_dict = atom_dict[mask]
    coords = atom_dict['coords']
    if max_dist is None:
        max_dist = atom_dict['radius']

    # More input validation
    elif isinstance(max_dist, Number):
        max_dist = np.repeat(max_dist, coords.shape[0])
    else:
        max_dist = np.array(max_dist)
    if not np.issubdtype(max_dist.dtype, np.number):
        raise ValueError('max_dist has to be a number or an '
                         'array_like object containing numbers')
    if coords.shape[0] != len(max_dist):
        raise ValueError('max_dist doesn\'t match coords\' length')

    # Marching cubes
    verts, _ = generate_surface_marching_cubes(molecule, remove_hoh=True,
                                               scaling=scaling, probe_radius=1.4)

    # Calculate distances between atoms and the surface
    tree_verts = cKDTree(verts)
    mask = [bool(tree_verts.query_ball_point(point, radius))
            for point, radius in zip(coords, max_dist)]
    return atom_dict[np.array(mask)]
