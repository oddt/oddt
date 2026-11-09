"""Published pair values, analytic forces, preparation and search composition."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import oddt
from oddt.docking import (
    ad4_scoring,
    ad4_ga,
    ad4_mc,
    ad4_de,
    vinardo_scoring,
    vinardo_ga,
    vinardo_mc,
    vinardo_de,
)
from oddt.docking.ad4 import AD4_PARAMETERS
from .test_docking import DATA, docking_pair  # noqa: F401


def ad4_pair(first="C", second="C", r=4, charges=(0, 0), smoothing=0):
    """One independent pair, without molecule preparation or covalent bonds."""
    engine = ad4_scoring.__new__(ad4_scoring)
    engine.weights = np.array([0.1662, 0.1209, 0.1406, 0.1322, 0.2983])
    engine.smoothing = smoothing
    engine.num_rotors = 0
    engine.unbound_energy = None
    engine.directional_hb = False
    engine.hbond_saturation = False
    engine._pair_cache = {}
    engine.scoring_coordinates = np.array([[r, 0.0, 0.0]])
    engine.rec_coordinates = np.zeros((1, 3))
    engine._distant = np.zeros((1, 1), dtype=bool)
    for attribute, name, charge in (("rec_parameters", first, charges[0]), ("lig_parameters", second, charges[1])):
        setattr(
            engine,
            attribute,
            {
                "types": np.array([name]),
                "values": np.array([AD4_PARAMETERS[name]]),
                "active": np.ones(1, dtype=bool),
                "charges": np.array([charge]),
            },
        )
    return engine


@pytest.mark.parametrize(
    "first,second,r,term,minimum",
    [
        ("C", "C", 4.0, 0, -0.15),
        ("N", "N", 3.5, 0, -0.16),
        ("HD", "OA", 1.9, 1, -5.0),
        ("NA", "HD", 1.9, 1, -5.0),
        ("HD", "SA", 2.5, 1, -1.0),
    ],
)
def test_ad4_published_well_depth(first, second, r, term, minimum):
    engine = ad4_pair(first, second, r)
    terms = engine.score_inter()
    assert terms[term] == pytest.approx(minimum)
    assert terms[1 - term] == 0  # HB replaces vdW for donor-H/acceptor pairs.
    weights = engine.weights.copy()
    engine.weights[:] = 0
    engine.weights[term] = weights[term]
    _, gradient = engine.energy_and_gradient(engine.scoring_coordinates)
    assert_allclose(gradient, 0, atol=1e-12)


def test_ad4_screening_desolvation_and_long_range():
    engine = ad4_pair(r=4, charges=(1, -1))
    terms = engine.score_inter()
    # Mehler–Solmajer equation, evaluated independently with 45-digit decimals.
    assert terms[2] == pytest.approx(-332.06371 / (4 * 18.58106685376879))
    # ASP_C=-.00143, V_C=33.5103, QASP=.01097, sigma=3.5 A.
    assert terms[3] == pytest.approx(2 * (-0.00143 + 0.01097) * 33.5103 * np.exp(-16 / 24.5))
    engine.scoring_coordinates[0, 0] = 12
    terms = engine.score_inter()
    assert_allclose(terms[:2], 0)
    assert terms[2] < 0 and terms[3] > 0


@pytest.mark.parametrize("r", [0.05, 0.2, 1.3, 1.82, 2.31, 3.1, 4.4, 9.5])
@pytest.mark.parametrize("cap", [None, 10.0])
def test_ad4_pair_derivative(r, cap):
    engine = ad4_pair("HD", "OA", r, charges=(0.3, -0.5), smoothing=0.5)
    coords = engine.scoring_coordinates.copy()
    value, actual = engine.energy_and_gradient(coords, cap)
    delta = np.array([[1e-6, 0, 0]])
    numeric = (
        engine.energy_and_gradient(coords + delta, cap)[0] - engine.energy_and_gradient(coords - delta, cap)[0]
    ) / 2e-6
    assert np.isfinite(value)
    assert_allclose(actual[0, 0], numeric, rtol=2e-5, atol=1e-6)
    assert_allclose(actual[0, 1:], 0)


def test_ad4_smoothing_is_a_flat_well():
    engine = ad4_pair(smoothing=0.5)
    engine.weights[:] = [1, 0, 0, 0, 0]
    for r in (3.8, 4, 4.2):
        coords = np.array([[r, 0, 0.0]])
        value, gradient = engine.energy_and_gradient(coords)
        assert value == pytest.approx(-0.15)
        assert_allclose(gradient, 0, atol=1e-12)


@pytest.mark.parametrize("scorer", [vinardo_scoring, ad4_scoring])
@pytest.mark.parametrize("cap", [None, 10.0, np.inf])
@pytest.mark.parametrize("boundary", [False, True])
@pytest.mark.parametrize("docking_pair", ["4bfq", "4lb3"], indirect=True)
def test_new_scorer_pose_derivative(docking_pair, scorer, cap, boundary):
    protein, ligand, box = docking_pair
    engine = scorer(protein, ligand, box=box, box_size=2)
    x = np.linspace(0.13, 0.67, len(engine.lig.x0))
    if boundary:
        x[0] = 5
    value, actual = engine._objective(x, cap)
    numeric = np.empty_like(x)
    for i in range(len(x)):
        delta = np.zeros_like(x)
        delta[i] = 1e-5
        numeric[i] = (engine._objective(x + delta, cap)[0] - engine._objective(x - delta, cap)[0]) / 2e-5
    assert np.isfinite(value)
    assert_allclose(actual, numeric, rtol=1e-5, atol=2e-4)


def test_vinardo_published_radii_and_pair_terms(docking_pair):
    protein, ligand, _ = docking_pair
    engine = vinardo_scoring(protein, ligand)
    atoms = engine.rec_dict
    for element, aromatic, radius in ((6, False, 2), (6, True, 1.9), (7, False, 1.7), (8, False, 1.6)):
        mask = atoms["atomicnum"] == element
        if element == 6:
            mask &= atoms["isaromatic"] == aromatic
        assert mask.any()
        assert_allclose(atoms["radius"][mask], radius)
    # Isolate two hydrophobic carbons: surface gap 1 A gives Gaussian exp(-1/.64)
    # and hydrophobic occupancy .6, with no repulsion or H bond.
    engine.rec_dict = atoms[:1].copy()
    engine.lig_dict = engine.lig_dict[:1].copy()
    for data in (engine.rec_dict, engine.lig_dict):
        for field in ("isdonor", "isacceptor", "ismetal", "ishalogen"):
            data[field] = False
        data["ishydrophobe"] = True
        data["radius"] = 2
    engine.rec_dict["coords"] = 0
    assert_allclose(engine.score_inter(np.array([[5.0, 0, 0]])), [np.exp(-1 / 0.64), 0, 0.6, 0])
    engine.rec_dict["isdonor"] = True
    engine.lig_dict["isacceptor"] = True
    assert engine.score_inter(np.array([[3.7, 0, 0]]))[3] == pytest.approx(0.5, abs=1e-6)


def test_ad4_preparation_preserves_inputs_and_conserves_charge(docking_pair):
    protein, ligand, box = docking_pair
    original_p, original_l = protein.coords.copy(), ligand.coords.copy()
    ligand.title = "AD4 input"
    ligand.data["source"] = "test"
    engine = ad4_scoring(protein, ligand, box=box)
    assert_array_equal(protein.coords, original_p)
    assert_array_equal(ligand.coords, original_l)
    assert engine._ligand.title == ligand.title
    assert engine._ligand.data["source"] == "test"
    p = engine.lig_parameters
    assert "HD" in p["types"]
    assert "A" in p["types"]
    assert p["charges"].sum() == pytest.approx(p["unmerged_charges"].sum(), abs=2e-5)
    assert np.all(p["charges"][~p["active"]] == 0)
    _, forces = engine.energy_and_gradient(engine.scoring_coordinates)
    assert_allclose(forces[~p["active"]], 0)
    assert np.linalg.norm(forces[p["types"] == "HD"]) > 0
    terms = engine.score_components()
    assert sum(terms.values()) == pytest.approx(engine.energy())
    assert engine.score() == pytest.approx(
        engine.weights[:4] @ engine.score_inter() + engine.weights[4] * engine.num_rotors
    )
    engine.unbound_energy = -2.5
    assert engine.score() == pytest.approx(engine.energy() + engine.weights[4] * engine.num_rotors + 2.5)
    changed = engine.scoring_coordinates + 0.1
    engine.set_coords(changed)
    assert_allclose(engine.scoring_coordinates, changed)
    with pytest.raises(ValueError, match="every explicit atom"):
        engine.set_coords(changed[engine._heavy_mask])


def test_ad4_pair_cache_reuses_coefficients_across_poses(docking_pair):
    protein, ligand, _ = docking_pair
    engine = ad4_scoring(protein, ligand)
    engine.energy_and_gradient(engine.scoring_coordinates)
    inter, intra = engine._pair_cache[False], engine._pair_cache[True]
    # Coordinates and scoring options remain dynamic after the cache is warm.
    engine.set_coords(engine.scoring_coordinates + [0.4, -0.2, 0.1])
    engine.smoothing = 0.2
    engine.weights = np.array([0.2, 0.1, 0.15, 0.12, 0.3])
    fresh = ad4_scoring(protein, ligand, smoothing=engine.smoothing, weights=engine.weights)
    coords = engine.scoring_coordinates
    for cap in (None, 10.0, np.inf):
        value, gradient = engine.energy_and_gradient(coords, cap)
        expected, expected_gradient = fresh.energy_and_gradient(coords, cap)
        assert_allclose(value, expected, rtol=1e-12, atol=1e-12)
        assert_allclose(gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    assert engine.score() == pytest.approx(fresh.score(coords), rel=1e-12)
    assert engine._pair_cache[False] is inter
    assert engine._pair_cache[True] is intra


def test_ad4_pair_cache_rebuilds_after_receptor_change(docking_pair):
    protein, ligand, _ = docking_pair
    engine = ad4_scoring(protein, ligand)
    coords = engine.scoring_coordinates.copy()
    original_value, original_gradient = engine.energy_and_gradient(coords)
    intra = engine._pair_cache[True]
    # The ligand also serves as a replacement receptor with different typing
    # and atom count; no protein-specific preparation is required by AD4.
    engine.set_protein(ligand)
    assert False not in engine._pair_cache
    assert engine._pair_cache[True] is intra
    fresh = ad4_scoring(ligand, ligand)
    value, gradient = engine.energy_and_gradient(coords)
    expected, expected_gradient = fresh.energy_and_gradient(coords)
    assert_allclose(value, expected, rtol=1e-12, atol=1e-12)
    assert_allclose(gradient, expected_gradient, rtol=1e-12, atol=1e-12)

    engine.set_protein(None)
    assert False not in engine._pair_cache
    with pytest.raises(ValueError, match="receptor is required"):
        engine.energy_and_gradient(coords)
    assert_allclose(engine.score_intra(), ad4_scoring(None, ligand).score_intra(), rtol=1e-12, atol=1e-12)
    assert engine._pair_cache[True] is intra

    engine.set_protein(protein)
    value, gradient = engine.energy_and_gradient(coords)
    assert_allclose(value, original_value, rtol=1e-12, atol=1e-12)
    assert_allclose(gradient, original_gradient, rtol=1e-12, atol=1e-12)


def test_ad4_pair_cache_rebuilds_after_ligand_change(docking_pair):
    protein, ligand, _ = docking_pair
    engine = ad4_scoring(protein)
    engine.set_ligand(ligand)
    engine.energy_and_gradient(engine.scoring_coordinates)
    replacement = next(oddt.toolkit.readfile("sdf", str(DATA / "4lb3_ligand.sdf")))
    engine.set_ligand(replacement)
    assert not engine._pair_cache
    fresh = ad4_scoring(protein, replacement)
    coords = engine.scoring_coordinates
    for cap in (None, 10.0, np.inf):
        value, gradient = engine.energy_and_gradient(coords, cap)
        expected, expected_gradient = fresh.energy_and_gradient(coords, cap)
        assert_allclose(value, expected, rtol=1e-12, atol=1e-12)
        assert_allclose(gradient, expected_gradient, rtol=1e-12, atol=1e-12)


def test_ad4_supplied_charges_are_preserved(docking_pair):
    protein, ligand, _ = docking_pair
    protein.addh()
    protein.calccharges()
    ligand.calccharges()
    original = ligand.charges.copy()
    engine = ad4_scoring(protein, ligand, charge_model=None)
    assert engine.lig_parameters["charges"].sum() == pytest.approx(original.sum())
    polar = engine.lig_parameters["types"] == "HD"
    assert_allclose(engine.lig_parameters["charges"][polar], original[polar])
    assert_array_equal(ligand.charges, original)
    ligand.removeh()
    with pytest.raises(ValueError, match="explicit polar hydrogens"):
        engine.set_ligand(ligand)


@pytest.mark.parametrize(
    "engine_class,kwargs",
    [
        (vinardo_ga, {"population_size": 4, "generations": 1}),
        (vinardo_mc, {"exhaustiveness": 1, "steps": 3}),
        (vinardo_de, {"population_size": 4, "generations": 1, "local_maxiter": 2}),
        (ad4_ga, {"population_size": 4, "generations": 1}),
        (ad4_mc, {"exhaustiveness": 1, "steps": 3}),
        (ad4_de, {"population_size": 4, "generations": 1, "local_maxiter": 2}),
    ],
)
def test_new_scorers_compose_with_every_search(docking_pair, engine_class, kwargs):
    protein, ligand, box = docking_pair
    original = ligand.coords.copy()
    engine = engine_class(protein, ligand, box=box)
    poses = engine.dock(**kwargs, maxiter=30, seed=42, num_modes=2)
    assert 1 <= len(poses) <= 2
    energies = []
    for pose, result in zip(poses, engine.results_):
        coords = engine._result_coords(result)
        energy = engine.energy(coords)
        energies.append(energy)
        assert float(pose.data["docking_energy"]) == pytest.approx(energy)
        assert float(pose.data["docking_score"]) == pytest.approx(engine.score(coords))
        assert pose.data["docking_scorer"] == engine.scoring_name
        assert float(pose.data[engine.scoring_name + "_affinity"]) == pytest.approx(result.score)
        assert_allclose(pose.coords[pose.atom_dict["atomicnum"] != 1], result.coords, atol=2e-5)
        if engine.score_all_atoms:
            assert pose.data["ad4_model"] == "directional_saturated"
            assert int(pose.data["ad4_directional_hb"]) == 1
            assert int(pose.data["ad4_hbond_saturation"]) == 1
            assert_allclose(pose.charges, engine.lig_parameters["charges"])
            rescored = ad4_scoring(None, pose, charge_model=None)
            assert_allclose(rescored.lig_parameters["charges"], engine.lig_parameters["charges"])
    assert energies == sorted(energies)
    assert_array_equal(ligand.coords, original)


@pytest.mark.parametrize("scorer", [vinardo_scoring, ad4_scoring])
def test_new_scorer_validation(docking_pair, scorer):
    protein, ligand, _ = docking_pair
    for weights in ([1, 2], [0, 0, 0, 0, -1], [0, 0, 0, np.nan, 0]):
        with pytest.raises(ValueError, match="weights"):
            scorer(protein, ligand, weights=weights)
    engine = scorer(protein, ligand)
    with pytest.raises(ValueError, match="positive"):
        engine.energy_and_gradient(engine.scoring_coordinates if scorer is ad4_scoring else engine.lig.c0, 0)
    if scorer is ad4_scoring:
        for kwargs in ({"charge_model": "unknown"}, {"smoothing": -1}, {"unbound_energy": np.inf}):
            with pytest.raises(ValueError):
                scorer(protein, ligand, **kwargs)
