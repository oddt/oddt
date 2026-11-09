"""Published PLP/ChemPLP values, geometry derivatives and search integration."""

import pickle

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.spatial.transform import Rotation

import oddt
from oddt import docking
from oddt.docking import OddtPLPScoring, OddtChemPLPScoring
from oddt.docking._plp_preparation import SYBYL_TORSIONS, atom_parameters, ligand_clashes, ligand_torsions, metal_sites
from oddt.docking._plp_terms import angle_factor, plp_potential, repulsive_potential, torsional_potential
from .test_docking import docking_pair  # noqa: F401


def finite_difference(function, coords, step=1e-6):
    gradient = np.zeros_like(coords)
    for index in np.ndindex(coords.shape):
        delta = np.zeros_like(coords)
        delta[index] = step
        gradient[index] = (function(coords + delta) - function(coords - delta)) / (2 * step)
    return gradient


def parameters(types, atomicnum, bonds=(), positive=(), negative=(), hydrogens=(), cho=()):
    """Explicit synthetic topology, independent of toolkit chemical perception."""
    types, atomicnum = np.array(types), np.array(atomicnum)
    count = len(types)
    neighbors = [[] for _ in range(count)]
    for first, second in bonds:
        neighbors[first].append(second)
        neighbors[second].append(first)
    parents = np.full(count, -1)
    for hydrogen in np.flatnonzero(atomicnum == 1):
        parents[hydrogen] = neighbors[hydrogen][0]
    pos, neg = np.zeros(count, bool), np.zeros(count, bool)
    pos[list(positive)], neg[list(negative)] = True, True
    return dict(
        types=types,
        atomicnum=atomicnum,
        heavy=np.flatnonzero(atomicnum != 1),
        neighbors=[np.array(n, int) for n in neighbors],
        donor_parent=parents,
        acceptors=np.flatnonzero(np.isin(types, [1, 2])),
        positive=pos,
        negative=neg,
        hydrogens=np.array(hydrogens, int),
        cho_hydrogens=np.array(cho, int),
        has_hydrogen=np.array([any(atomicnum[j] == 1 for j in n) for n in neighbors]),
    )


def contact_engine(rec, lig, rp, lp):
    engine = OddtChemPLPScoring(None)
    engine.rec_coordinates = np.array(rec, float)
    engine.scoring_coordinates = np.array(lig, float)
    engine.rec_parameters, engine.lig_parameters = rp, lp
    engine._clash_pairs = (np.array([], int), np.array([], int), np.array([]))
    engine._torsion_indices = np.empty((0, 4), int)
    engine._torsion_values = np.empty((0, 3))
    return engine


@pytest.mark.parametrize(
    "knots,depth", [((2.3, 2.6, 3.1, 3.4), -2), ((1.4, 2.2, 2.6, 2.8), -4), ((3.4, 3.6, 4.5, 5.5), -0.4)]
)
def test_published_plp_radial_values_and_slopes(knots, depth):
    a, b, c, d = knots
    values, _ = plp_potential(np.array([0, a, b, c, d, d + 1]), *knots, depth)
    assert_allclose(values, [20, 0, depth, depth, 0, 0])
    r = np.array([a / 2, (a + b) / 2, (b + c) / 2, (c + d) / 2, d + 1])
    value, actual = plp_potential(r, *knots, depth)
    assert_allclose(value, [10, depth / 2, depth, depth / 2, 0])
    assert_allclose(actual, finite_difference(lambda x: plp_potential(x, *knots, depth)[0].sum(), r), atol=1e-7)


def test_published_repulsion_and_cutoff():
    values, _ = repulsive_potential(np.array([0, 1.6, 3.2, 4.1, 5, 6]), 0.5)
    assert_allclose(values, [10, 5.025, 0.05, 0.025, 0, 0])


def test_all_contact_types_use_published_interaction_matrix():
    from oddt.docking._plp_preparation import INTERACTION_TYPES

    assert_array_equal(INTERACTION_TYPES, [[4, 0, 0, 2, 4], [0, 4, 0, 2, 1], [0, 0, 0, 2, 1], [2, 2, 2, 3, 2]])


def test_angle_derivatives_and_rigid_transform():
    coords = np.array([[-0.7, 0.6, -0.1], [0, 0, 0], [2.1, 0.1, 0.3]])

    def evaluate(x):
        value, gradients = angle_factor(*x, 30, 80)
        return value, np.array(gradients)

    value, actual = evaluate(coords)
    assert 0 < value < 1
    assert_allclose(actual, finite_difference(lambda x: evaluate(x)[0], coords), atol=1e-8)
    rotation = Rotation.from_rotvec([0.4, -0.8, 0.1]).as_matrix()
    transformed, gradient = evaluate(coords @ rotation.T + [8, 3, -7])
    assert transformed == pytest.approx(value)
    assert_allclose(gradient, actual @ rotation.T, atol=1e-12)
    assert_allclose(actual.sum(axis=0), 0, atol=1e-12)


@pytest.mark.parametrize("period", [1, 2, 3, 6])
@pytest.mark.parametrize("height", [0, 0.7])
def test_torsional_cartesian_derivatives_including_planar(period, height):
    coords = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1.1, -1, height]], float)
    indices = np.array([[0, 1, 2, 3]])

    def evaluate(x):
        return torsional_potential(x, indices, np.array([2.0]), np.array([1]), np.array([period]))

    value, actual = evaluate(coords)
    assert_allclose(actual, finite_difference(lambda x: evaluate(x)[0], coords), atol=1e-8)
    assert_allclose(actual.sum(axis=0), 0, atol=1e-12)
    assert_allclose(np.cross(coords, actual).sum(axis=0), 0, atol=1e-12)
    if height == 0:
        assert value == pytest.approx(1 + (-1) ** period)


@pytest.mark.parametrize("charged,weak,expected", [(False, False, -3), (True, False, -6), (False, True, -3)])
def test_published_directional_hydrogen_well_depth(charged, weak, expected):
    rp = parameters(
        [0, 3],
        [7, 1],
        [(0, 1)],
        positive=[0] if charged else [],
        hydrogens=[] if weak else [1],
        cho=[1] if weak else [],
    )
    lp = parameters([1, 3], [8, 6], [(0, 1)], negative=[0] if charged else [])
    radius = 2.35 if weak else 1.85
    engine = contact_engine([[-1, 0, 0], [0, 0, 0]], [[radius, 0, 0], [radius + 1, 0, 0]], rp, lp)
    full = np.vstack((engine.rec_coordinates, engine.scoring_coordinates))
    terms, gradient = engine._hydrogen_bonds(full, True)
    assert terms[7 if weak else 6 if charged else 5] == pytest.approx(expected)
    assert terms.sum() == pytest.approx(expected)
    assert_allclose(gradient, 0, atol=1e-12)


@pytest.mark.parametrize("reverse", [False, True])
def test_hydrogen_gradient_includes_donor_and_acceptor_neighbors(reverse):
    donor = parameters([0, 3], [7, 1], [(0, 1)], hydrogens=[1])
    acceptor = parameters([1, 3], [8, 6], [(0, 1)])
    dc, ac = [[-0.7, 0.7, 0.1], [0, 0, 0]], [[2.2, 0, 0], [2.2, 0.9, 0.1]]
    engine = contact_engine(
        ac if reverse else dc, dc if reverse else ac, acceptor if reverse else donor, donor if reverse else acceptor
    )

    def evaluate(x):
        return engine._hydrogen_bonds(np.vstack((engine.rec_coordinates, x)), True)

    terms, actual = evaluate(engine.scoring_coordinates)
    assert terms.sum() < 0
    assert np.linalg.norm(actual[1]) > 0
    assert_allclose(actual, finite_difference(lambda x: evaluate(x)[0].sum(), engine.scoring_coordinates), atol=1e-8)


def test_only_best_acceptor_per_donor_hydrogen_is_counted():
    engine = contact_engine(
        [[-1, 0, 0], [0, 0, 0]],
        [[1.85, 0, 0], [2.3, 0, 0]],
        parameters([0, 3], [7, 1], [(0, 1)], hydrogens=[1]),
        parameters([1, 1], [8, 8]),
    )
    terms, _ = engine._hydrogen_bonds(np.vstack((engine.rec_coordinates, engine.scoring_coordinates)), True)
    assert terms.sum() == pytest.approx(-3)


@pytest.mark.parametrize("donor_charged,acceptor_charged", [(True, False), (False, True)])
def test_one_charged_partner_retains_ordinary_hydrogen_weight(donor_charged, acceptor_charged):
    rp = parameters([0, 3], [7, 1], [(0, 1)], positive=[0] if donor_charged else [], hydrogens=[1])
    lp = parameters([1], [8], negative=[0] if acceptor_charged else [])
    engine = contact_engine([[-1, 0, 0], [0, 0, 0]], [[1.85, 0, 0]], rp, lp)
    terms, _ = engine._hydrogen_bonds(np.vstack((engine.rec_coordinates, engine.scoring_coordinates)), True)
    assert terms[5] == pytest.approx(-3)
    assert terms[6] == 0


@pytest.mark.parametrize("negative,expected", [(False, -6), (True, -12)])
def test_published_magnesium_well_depth(negative, expected):
    rp = parameters([4], [12])
    lp = parameters([1, 3], [8, 6], [(0, 1)], negative=[0] if negative else [])
    engine = contact_engine([[0, 0, 0]], [[2.4, 0, 0], [3.4, 0, 0]], rp, lp)
    engine.metal_geometries_ = metal_sites(engine.rec_coordinates, rp, 2.8)
    terms, gradient = engine._metal_bonds(np.vstack((engine.rec_coordinates, engine.scoring_coordinates)), True)
    assert terms[9 if negative else 8] == pytest.approx(expected)
    assert_allclose(gradient, 0)


def test_radial_metal_neighbor_and_distance_gradients():
    engine = contact_engine(
        [[0, 0, 0]], [[2.8, 0, 0], [2.9, 1, 0.1]], parameters([4], [12]), parameters([1, 3], [8, 6], [(0, 1)])
    )
    engine.metal_geometries_ = metal_sites(engine.rec_coordinates, engine.rec_parameters, 2.8)

    def evaluate(x):
        return engine._metal_bonds(np.vstack((engine.rec_coordinates, x)), True)

    terms, actual = evaluate(engine.scoring_coordinates)
    assert terms.sum() < 0
    assert np.linalg.norm(actual[1]) > 0
    assert_allclose(actual, finite_difference(lambda x: evaluate(x)[0].sum(), engine.scoring_coordinates), atol=1e-8)


def test_fitted_zinc_site_is_covariant_and_saturates():
    tetra = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    coords = np.vstack((np.zeros(3), 2.2 * tetra[:3]))
    rp = parameters([4, 1, 1, 1], [30, 7, 7, 7])
    description = metal_sites(coords, rp, 2.8)[0]
    assert description["geometry"] == "tetrahedral"
    assert_allclose(description["sites"], [2.2 * tetra[3]], atol=1e-12)
    rotation = Rotation.from_rotvec([0.7, -0.3, 1]).as_matrix()
    transformed = metal_sites(coords @ rotation.T + [3, 7, -2], rp, 2.8)[0]
    assert_allclose(transformed["sites"], description["sites"] @ rotation.T + [3, 7, -2], atol=1e-12)
    # Two ligating atoms compete for the same single vacant site.
    engine = contact_engine(coords, [2.2 * tetra[3], 2.3 * tetra[3]], rp, parameters([1, 1], [8, 8]))
    engine.metal_geometries_ = [description]
    terms, _ = engine._metal_bonds(np.vstack((coords, engine.scoring_coordinates)), True)
    assert terms.sum() == pytest.approx(-6)
    # A point in the site-distance ramp exercises the fitted-site derivative.
    x = np.array([2.9 * tetra[3]])
    engine.lig_parameters = parameters([1], [8])
    engine.scoring_coordinates = x
    engine._contacts = None
    evaluate = lambda y: engine._metal_bonds(np.vstack((coords, y)), True)
    assert_allclose(evaluate(x)[1], finite_difference(lambda y: evaluate(y)[0].sum(), x), atol=1e-8)


def test_planar_coordination_leaves_two_octahedral_sites_and_reports_missing_geometry():
    coords = np.array([[0, 0, 0], [2.2, 0, 0], [-2.2, 0, 0], [0, 2.2, 0], [0, -2.2, 0]])
    rp = parameters([4, 1, 1, 1, 1], [26, 7, 7, 7, 7])
    site = metal_sites(coords, rp, 2.8)[0]
    assert site["geometry"] == "octahedral"
    assert_allclose(np.sort(site["sites"][:, 2]), [-2.2, 2.2], atol=1e-12)
    site = metal_sites(coords[:2], parameters([4, 1], [30, 7]), 2.8)[0]
    assert site["geometry"] == "radial_underdetermined"


def test_clash_pair_exclusions_and_published_cutoffs():
    p = parameters([3, 3, 3, 3, 1], [6, 6, 6, 6, 8], [(0, 1), (1, 2), (2, 3), (3, 4)])
    rotor = {"full_mask": np.array([False, False, True, True, True])}
    rows, cols, cutoff = ligand_clashes(p, [rotor])
    assert dict(zip(zip(rows, cols), cutoff)) == {(0, 3): 2.75, (0, 4): 2.75, (1, 4): 2.5}
    assert len(ligand_clashes(p, [])[0]) == 0
    engine = contact_engine([[10, 0, 0]], [[0, 0, 0], [1, 0, 0]], parameters([3], [6]), parameters([3, 3], [6, 6]))
    engine._clash_pairs = (np.array([0]), np.array([1]), np.array([2.5]))
    value, gradient = engine._internal(engine.scoring_coordinates, True)
    assert value[0] == pytest.approx(42)
    assert_allclose(
        gradient,
        finite_difference(lambda x: engine._internal(x, False)[0].sum(), engine.scoring_coordinates),
        atol=1e-8,
    )


def test_chemical_perception_ester_nitro_and_cho():
    molecule = oddt.toolkit.readstring("smi", "COC(=O)C.[O-][N+](=O)c1ncccc1")
    molecule.addh()
    p = atom_parameters(molecule)
    bridging = [a.idx0 for a in molecule.atoms if a.atomicnum == 8 and len(p["neighbors"][a.idx0]) == 2]
    assert len(bridging) == 1 and bridging[0] not in p["acceptors"]
    assert not p["negative"].any()  # Nitro oxygens are neutral for charge weighting.
    assert len(p["cho_hydrogens"]) > 0
    assert not set(p["cho_hydrogens"]) & set(p["hydrogens"])
    with pytest.raises(ValueError, match="metal ions in the receptor"):
        atom_parameters(oddt.toolkit.readstring("smi", "[Zn+2]"))


def test_torsion_coverage_strict_lookup_and_overrides(docking_pair):
    protein, ligand, box = docking_pair
    engine = OddtPLPScoring(protein, ligand, box=box)
    assert engine.torsion_fallbacks_
    with pytest.raises(ValueError, match="No published torsion parameters"):
        OddtPLPScoring(None, ligand, torsion_fallback=None)
    strict = OddtPLPScoring(None, ligand, torsion_fallback=None, torsion_parameters={("*", "*", "*", "*"): (1, -1, 2)})
    assert not strict.torsion_fallbacks_
    assert len(strict._torsion_indices) >= len(strict.rotors)
    assert_allclose(pickle.loads(pickle.dumps(engine)).energy(), engine.energy())


def test_published_torsion_matching_direction_and_override_priority():
    p = parameters([3] * 4, [6] * 4, [(0, 1), (1, 2), (2, 3)])
    p["atomtypes"] = np.array(["C.2", "C.3", "C.3", "C.3"])
    rotor = {"full_atoms": (0, 1, 2, 3)}
    indices, values, fallback = ligand_torsions(p, [rotor], SYBYL_TORSIONS, None)
    assert_array_equal(indices, [[0, 1, 2, 3]])
    assert_allclose(values, [[0.252, 1, 3]])
    assert not fallback
    p["atomtypes"] = p["atomtypes"][::-1]
    assert_allclose(ligand_torsions(p, [rotor], SYBYL_TORSIONS, None)[1], values)
    custom = dict(SYBYL_TORSIONS)
    custom[("C.2", "C.3", "C.3", "C.3")] = (1, -1, 2)
    assert_allclose(ligand_torsions(p, [rotor], custom, None)[1], [[1, -1, 2]])


@pytest.mark.parametrize("scorer", [OddtPLPScoring, OddtChemPLPScoring])
@pytest.mark.parametrize("cap", [None, 10.0, np.inf])
@pytest.mark.parametrize("boundary", [False, True])
@pytest.mark.parametrize("docking_pair", ["4bfq", "4lb3"], indirect=True)
def test_flexible_pose_gradients_and_input_preservation(docking_pair, scorer, cap, boundary):
    protein, ligand, box = docking_pair
    original = protein.coords.copy(), ligand.coords.copy()
    engine = scorer(protein, ligand, box=box, box_size=2)
    x = np.linspace(0.13, 0.67, len(engine.lig.x0))
    if boundary:
        x[0] = 5
    value, gradient = engine._objective(x, cap)
    assert np.isfinite(value)
    assert_allclose(gradient, finite_difference(lambda y: engine._objective(y, cap)[0], x), rtol=1e-5, atol=2e-4)
    assert_array_equal(protein.coords, original[0])
    assert_array_equal(ligand.coords, original[1])
    assert sum(engine.score_components().values()) == pytest.approx(engine.energy())
    coords = engine.lig.mutate(x, all_atoms=True)
    engine.set_coords(coords)
    assert engine.energy() == pytest.approx(engine.energy(coords))


@pytest.mark.parametrize("model", ["PLP", "ChemPLP"])
@pytest.mark.parametrize("search", ["GA", "MC", "DE"])
def test_search_composition_metadata_and_seeded_repeatability(docking_pair, model, search):
    protein, ligand, box = docking_pair
    cls = getattr(docking, "Oddt" + model + search)
    kwargs = {"population_size": 4, "generations": 0} if search != "MC" else {"exhaustiveness": 1, "steps": 4}
    maxiter = 60 if search == "MC" else 30
    if search == "DE":
        kwargs["local_maxiter"] = 5
    observations = []
    engine = cls(protein, ligand, box=box, sample_callback=lambda result, cap: observations.append(result.fun))
    poses = engine.dock(**kwargs, seed=42, maxiter=maxiter, num_modes=2, energy_range=1000, progress=True)
    assert poses and observations
    assert type(pickle.loads(pickle.dumps(cls(None)))) is cls
    first = [pose.coords.copy() for pose in poses]
    prefix = model.lower()
    for pose, result in zip(poses, engine.results_):
        assert pose.data["docking_scorer"] == prefix
        assert pose.data["docking_score_units"] == "empirical score units"
        assert float(pose.data[prefix + "_score"]) == pytest.approx(engine.score(engine._result_coords(result)))
        assert float(pose.data[prefix + "_torsion_fallback_terms"]) == len(engine.torsion_fallbacks_)
        assert np.all(pose.coords[engine._heavy_mask] >= box[0] - 1e-4)
        assert np.all(pose.coords[engine._heavy_mask] <= box[1] + 1e-4)
    repeated = engine.dock(**kwargs, seed=42, maxiter=maxiter, num_modes=2, energy_range=1000)
    assert_allclose([pose.coords for pose in repeated], first, atol=1e-10)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"parameters": {"bad": 1}}, "Unknown"),
        ({"parameters": {"torsion": -1}}, "nonnegative"),
        ({"metal_cutoff": 0}, "positive"),
        ({"torsion_fallback": (1, 1, 0)}, "positive integer"),
        ({"torsion_parameters": {"****": (1, 1, 3)}}, "four SYBYL"),
    ],
)
def test_invalid_configuration(options, match):
    with pytest.raises(ValueError, match=match):
        OddtPLPScoring(None, **options)
