import os
from distutils.version import LooseVersion

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage import __version__ as skimage_version

import oddt
from oddt.surface import (generate_surface_marching_cubes,
                          find_surface_residues)

test_data_dir = os.path.dirname(os.path.abspath(__file__))
protein = next(oddt.toolkit.readfile('pdb', os.path.join(
    test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb')))
protein.protein = True
protein.addh(only_polar=True)


def test_generate_surface_marching_cubes():
    """Tests generating surfaces"""
    verts1, faces1 = generate_surface_marching_cubes(protein, scaling=1., probe_radius=1.4, remove_hoh=False)
    verts2, faces2 = generate_surface_marching_cubes(protein, scaling=2., probe_radius=1.4, remove_hoh=False)
    verts3, faces3 = generate_surface_marching_cubes(protein, scaling=1., probe_radius=1.4, remove_hoh=True)
    verts4, faces4 = generate_surface_marching_cubes(protein, scaling=1., probe_radius=0, remove_hoh=True)

    # Higher scaling should result in a higher number of vertices
    assert len(verts2) > len(verts1), ('Higher scaling should result in '
                                       'a higher number of vertices')

    # versions of skimage older than 0.12 use a slightly different version of the marching cubes algorithm
    # producing slightly different results
    if LooseVersion(skimage_version) >= LooseVersion('0.13'):
        if oddt.toolkit.backend == 'ob':
            ref_vert_shape_1 = (9040, 3)
            ref_face_shape_1 = (18094, 3)
            ref_vert_shape_2 = (35950, 3)
            ref_face_shape_2 = (71926, 3)
            ref_vert_shape_3 = (9040, 3)
            ref_face_shape_3 = (18094, 3)
            ref_vert_shape_4 = (14881, 3)
            ref_face_shape_4 = (30468, 3)
        else:
            ref_vert_shape_1 = (9044, 3)
            ref_face_shape_1 = (18102, 3)
            ref_vert_shape_2 = (35788, 3)
            ref_face_shape_2 = (71578, 3)
            ref_vert_shape_3 = (9044, 3)
            ref_face_shape_3 = (18102, 3)
            ref_vert_shape_4 = (15035, 3)
            ref_face_shape_4 = (30848, 3)
    else:
        if oddt.toolkit.backend == 'ob':
            ref_vert_shape_1 = (5923, 3)
            ref_face_shape_1 = (11862, 3)
            ref_vert_shape_2 = (20819, 3)
            ref_face_shape_2 = (41634, 3)
            ref_vert_shape_3 = (5923, 3)
            ref_face_shape_3 = (11862, 3)
            ref_vert_shape_4 = (10263, 3)
            ref_face_shape_4 = (21658, 3)
        else:
            ref_vert_shape_1 = (5916, 3)
            ref_face_shape_1 = (11848, 3)
            ref_vert_shape_2 = (20845, 3)
            ref_face_shape_2 = (41686, 3)
            ref_vert_shape_3 = (5916, 3)
            ref_face_shape_3 = (11848, 3)
            ref_vert_shape_4 = (10243, 3)
            ref_face_shape_4 = (21686, 3)

    assert ref_vert_shape_1 == verts1.shape
    assert ref_face_shape_1 == faces1.shape

    assert ref_vert_shape_2 == verts2.shape
    assert ref_face_shape_2 == faces2.shape

    assert ref_vert_shape_3 == verts3.shape
    assert ref_face_shape_3 == faces3.shape

    assert ref_vert_shape_4 == verts4.shape
    assert ref_face_shape_4 == faces4.shape

    with pytest.raises(TypeError):
        generate_surface_marching_cubes(molecule=1)
    with pytest.raises(ValueError):
        generate_surface_marching_cubes(molecule=protein, probe_radius=-1)
    with pytest.raises(ValueError):
        generate_surface_marching_cubes(molecule=protein, scaling=0.1)


def test_find_surface_residues():
    """Tests finding residues on the surface"""
    atom_dict_0 = find_surface_residues(protein, max_dist=0, scaling=1)
    atom_dict_1 = find_surface_residues(protein, max_dist=2, scaling=1)
    atom_dict_2 = find_surface_residues(protein, max_dist=3, scaling=1)
    atom_dict_3 = find_surface_residues(protein, max_dist=None, scaling=1)
    atom_dict_4 = find_surface_residues(protein, max_dist=None, scaling=2)

    assert atom_dict_0.size == 0
    assert len(atom_dict_1) > len(atom_dict_0), ('Increasing max_dist should '
                                                 'result in more/equal number '
                                                 'of atoms found')

    assert len(atom_dict_2) >= len(atom_dict_1), ('Increasing max_dist should '
                                                  'result in more/equal number '
                                                  'of atoms found')
    assert_array_equal(np.intersect1d(atom_dict_1['id'], atom_dict_2['id']), atom_dict_1['id'])

    if oddt.toolkit.backend == 'ob':
        ref_len_1 = 762
        ref_len_2 = 968
        ref_len_3 = 654
        ref_len_4 = 379
    else:
        ref_len_1 = 759
        ref_len_2 = 966
        ref_len_3 = 735
        ref_len_4 = 489

    assert len(atom_dict_1) == ref_len_1
    assert len(atom_dict_2) == ref_len_2
    assert len(atom_dict_3) == ref_len_3
    assert len(atom_dict_4) == ref_len_4

    # Adding hydrogen atoms should have no effect on the result
    protein.addh()
    atom_dict_addh = find_surface_residues(protein, max_dist=2, scaling=1)
    assert_array_equal(atom_dict_addh['id'], atom_dict_1['id'])

    with pytest.raises(TypeError):
        find_surface_residues(molecule=1)
    with pytest.raises(ValueError):
        find_surface_residues(molecule=protein, max_dist='a')
    with pytest.raises(ValueError):
        find_surface_residues(molecule=protein, max_dist=[1, 1, 1])
