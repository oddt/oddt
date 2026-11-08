"""Independent geometry, capacity and Cartesian-derivative checks for AD4 H-bonds."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.spatial.transform import Rotation

from oddt.docking._ad4_hbond import acceptor_frames, angular_factors, saturated_contacts
from oddt.docking.ad4 import AD4_PARAMETERS
from .test_docking_scoring import ad4_pair


def angular_system(kind):
    coords = [[0.0, 0.0, 0.0], [1.0, 0.2, 0.1], [3.0, 0.8, -0.2]]
    neighbors = {
        "nitrogen": [[4.0, 1.0, -0.1]],
        "tetrahedral": [[3.7, 1.5, -0.2], [3.7, 0.2, 0.0]],
        "planar": [[4.0, 0.8, -0.2], [4.0, 1.8, -0.1]],
        "spherical": [],
    }[kind]
    coords = np.array(coords + neighbors)
    rules = {2: (kind, tuple(range(2, len(coords))))}
    parents = np.full(len(coords), -1)
    parents[1] = 0
    return coords, rules, parents


def angular_value_gradient(coords, rules, parents):
    frames = acceptor_frames(coords, rules)
    value, (contact, atom, derivatives) = angular_factors(coords, np.array([1]), np.array([2]), parents, frames)
    gradient = np.zeros_like(coords)
    np.add.at(gradient, atom, derivatives)
    return value[0], gradient


@pytest.mark.parametrize("kind", ["nitrogen", "tetrahedral", "planar", "spherical"])
def test_angular_derivatives_include_frame_neighbors(kind):
    coords, rules, parents = angular_system(kind)
    value, actual = angular_value_gradient(coords, rules, parents)
    assert 0 < value < 1
    numeric = np.zeros_like(coords)
    for atom in range(len(coords)):
        for axis in range(3):
            delta = np.zeros_like(coords)
            delta[atom, axis] = 1e-6
            numeric[atom, axis] = (
                angular_value_gradient(coords + delta, rules, parents)[0]
                - angular_value_gradient(coords - delta, rules, parents)[0]
            ) / 2e-6
    assert_allclose(actual, numeric, rtol=1e-5, atol=1e-7)
    assert_allclose(actual.sum(axis=0), 0, atol=1e-12)
    if kind != "spherical":
        assert np.linalg.norm(actual[3:]) > 0
    assert_allclose(np.cross(coords, actual).sum(axis=0), 0, atol=1e-12)


@pytest.mark.parametrize("kind", ["nitrogen", "tetrahedral", "planar", "spherical"])
def test_angular_rigid_transform_invariance(kind):
    coords, rules, parents = angular_system(kind)
    value, gradient = angular_value_gradient(coords, rules, parents)
    rotation = Rotation.from_rotvec([0.3, -0.8, 0.2]).as_matrix()
    changed_value, changed_gradient = angular_value_gradient(coords @ rotation.T + [5, -7, 3], rules, parents)
    assert changed_value == pytest.approx(value, abs=1e-12)
    assert_allclose(changed_gradient, gradient @ rotation.T, atol=1e-12)


def test_donor_angle_known_values_and_wrong_side():
    # A spherical acceptor isolates cos^2(theta) of the donor angle.
    coords = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.9, 0, 0]])
    rules = {2: ("spherical", (2,))}
    parents = np.array([-1, 0, -1])
    assert angular_value_gradient(coords, rules, parents)[0] == pytest.approx(1)
    coords[2] = [1 + 1.9 / np.sqrt(2), 1.9 / np.sqrt(2), 0]
    assert angular_value_gradient(coords, rules, parents)[0] == pytest.approx(0.5)
    coords[2] = [0.5, 1.0, 0]
    value, gradient = angular_value_gradient(coords, rules, parents)
    assert value == 0
    assert_allclose(gradient, 0)


def test_lone_pair_orientations():
    coords = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1.0, 0]])
    planar, _, _ = acceptor_frames(coords, {0: ("planar", (0, 1, 2))})[0]
    assert_allclose(planar[:, 0], -0.5)
    assert_allclose(np.sort(planar[:, 1]), [-np.sqrt(3) / 2, np.sqrt(3) / 2])
    assert_allclose(planar[:, 2], 0)
    a = 1 / np.sqrt(3)
    coords[1:] = [[a, np.sqrt(2 / 3), 0], [a, -np.sqrt(2 / 3), 0]]
    tetra, _, _ = acceptor_frames(coords, {0: ("tetrahedral", (0, 1, 2))})[0]
    assert_allclose(tetra @ tetra.T, [[1, -1 / 3], [-1 / 3, 1]], atol=1e-12)
    assert_allclose(tetra @ coords[1:].T, -np.ones((2, 2)) / 3, atol=1e-12)


def test_capacity_matching_finds_global_optimum():
    donors = np.array([0, 0, 1])
    acceptors = np.array([2, 3, 2])
    capacities = np.array([0, 0, 1, 1])
    energies = np.array([-5.0, -4.0, -4.0])
    keep = saturated_contacts(donors, acceptors, energies, capacities)
    assert_array_equal(keep, [False, True, True])  # -8, rather than greedy -5.
    for order in ([2, 0, 1], [1, 2, 0]):
        chosen = saturated_contacts(donors[order], acceptors[order], energies[order], capacities)
        assert energies[order][chosen].sum() == -8


def test_oxygen_capacity_and_repulsion_retention():
    keep = saturated_contacts(
        np.array([0, 1, 2, 3]), np.array([4, 4, 4, 4]), np.array([-5.0, -4.0, -3.0, 7.0]), np.array([0, 0, 0, 0, 2])
    )
    assert_array_equal(keep, [True, True, False, True])
    empty = saturated_contacts(np.array([], int), np.array([], int), np.array([]), np.array([], int))
    assert empty.shape == (0,)


def prepared_parameters(types, parents, capacities):
    return dict(
        types=np.array(types),
        values=np.array([AD4_PARAMETERS[t] for t in types]),
        active=np.ones(len(types), bool),
        charges=np.zeros(len(types)),
        donor_parent=np.array(parents),
        acceptor_capacity=np.array(capacities),
        acceptor_rules={},
    )


def test_saturation_is_joint_across_inter_and_intraligand_bonds():
    engine = ad4_pair()
    engine.weights[:] = [0, 1, 0, 0, 0]
    engine.hbond_saturation = True
    engine.rec_coordinates = np.array([[3.2, 0.0, 0.0]])
    engine.rec_parameters = prepared_parameters(["NA"], [-1], [1])
    engine.scoring_coordinates = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.9, 0, 0]])
    engine.lig_parameters = prepared_parameters(["N", "HD", "NA"], [-1, 0, -1], [0, 0, 1])
    engine._distant = np.zeros((3, 3), bool)
    engine._distant[1, 2] = True
    assert engine.score_inter()[1] == 0  # The intraligand 1.9 A contact wins.
    assert engine.score_intra()[1] == pytest.approx(-5)
    assert engine.energy() == pytest.approx(-5)
    _, gradient = engine.energy_and_gradient(engine.scoring_coordinates)
    assert_allclose(gradient, 0, atol=1e-12)
    engine.hbond_saturation = False
    assert engine.energy() < -5


def test_wrong_angle_does_not_remove_repulsion():
    engine = ad4_pair()
    engine.weights[:] = [0, 1, 0, 0, 0]
    engine.directional_hb = True
    engine.rec_coordinates = np.array([[2.9, 0.0, 0.0]])
    engine.rec_parameters = prepared_parameters(["NA"], [-1], [1])
    engine._rec_frames = {0: (None, np.array([0]), None)}
    engine.scoring_coordinates = np.array([[2.0, 0, 0], [1.0, 0, 0]])
    engine.lig_parameters = prepared_parameters(["N", "HD"], [-1, 0], [0, 0])
    engine._distant = np.zeros((2, 2), bool)
    assert engine.energy() == 0  # Incorrect donor direction suppresses attraction.
    engine.rec_coordinates[0, 0] = 1.5
    directed = engine.energy()
    assert directed > 0
    engine.directional_hb = False
    assert engine.energy() == directed


def test_degenerate_bonded_frame_fails_clearly():
    with pytest.raises(ValueError, match="nondegenerate"):
        acceptor_frames(np.zeros((2, 3)), {0: ("nitrogen", (0, 1))})


def test_saturated_hunting_cap_transforms_the_complete_energy():
    engine = ad4_pair()
    engine.weights[:] = [1, 0, 0, 0, 0]
    engine.hbond_saturation = True
    engine.scoring_coordinates = np.array([[2.8, 0.0, 0.0], [4.5, 0.0, 0.0]])
    engine.lig_parameters = prepared_parameters(["C", "C"], [-1, -1], [0, 0])
    engine._distant = np.zeros((2, 2), bool)
    raw, raw_gradient = engine.energy_and_gradient(engine.scoring_coordinates)
    assert raw > 0
    capped, capped_gradient = engine.energy_and_gradient(engine.scoring_coordinates, 10)
    factor = 10 / (10 + raw)
    assert capped == pytest.approx(raw * factor)
    assert_allclose(capped_gradient, raw_gradient * factor**2)
    assert engine.energy_and_gradient(engine.scoring_coordinates, np.inf)[0] == raw


def test_hydrogen_bond_options_are_validated():
    from oddt.docking import ad4_scoring

    for options in ({"directional_hb": 1}, {"hbond_saturation": "yes"}, {"weights": [1, -1, 1, 1, 0]}):
        with pytest.raises(ValueError):
            ad4_scoring(None, **options)
    scorer = ad4_scoring(None, directional_hb=False, hbond_saturation=False)
    assert not scorer.directional_hb and not scorer.hbond_saturation
