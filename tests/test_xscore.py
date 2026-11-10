"""Published X-Score terms, independent surface geometry and rescoring APIs."""

import pickle

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

import oddt
from oddt.docking import OddtXScoreScoring
from oddt.docking._xlogp2 import XLOGP2_CONTRIBUTIONS, classify_xlogp2
from oddt.docking._xscore_terms import buried_surface, hydrophobic_pairs, hydrogen_bonds, root_angle, vdw_term
from oddt.docking.xscore import XSCORE_COEFFICIENTS, rotor_count
from .test_docking import docking_pair  # noqa: F401


def prepared(smiles):
    molecule = oddt.toolkit.readstring("smi", smiles)
    molecule.addh()
    return molecule


@pytest.mark.parametrize(
    "smiles,atom,expected",
    [
        ("CCC", 0, 1),
        ("CCC", 1, 4),
        ("CC=C", 0, 2),
        ("CCO", 1, 7),
        ("CCO", 2, 69),
        ("CO", 0, 3),
        ("CC(C)C", 1, 10),
        ("CC(C)(C)C", 1, 16),
        ("C=C", 0, 21),
        ("CC=C", 1, 22),
        ("CC=O", 1, 22),
        ("NC=O", 1, 24),
        ("CC(=O)C", 1, 26),
        ("CC(=O)N", 1, 28),
        ("COC(=O)N", 2, 30),
        ("c1ccccc1", 0, 32),
        ("c1ncccc1", 0, 33),
        ("Cc1ccccc1", 1, 34),
        ("Oc1ccccc1", 1, 35),
        ("Cc1ncccc1", 1, 36),
        ("Oc1ncccc1", 1, 37),
        ("C#C", 0, 38),
        ("CC#N", 1, 39),
        ("C=C=C", 1, 40),
        ("CN", 1, 41),
        ("Nc1ccccc1", 0, 42),
        ("NN", 0, 43),
        ("CNC", 1, 44),
        ("CN(C)C", 1, 49),
        ("CC(=O)N", 3, 54),
        ("CC(=O)NC", 3, 55),
        ("CC(=O)N(C)C", 3, 57),
        ("CC=NCC", 2, 59),
        ("C[N+](=O)[O-]", 1, 66),
        ("c1ncccc1", 1, 67),
        ("CC#N", 2, 68),
        ("Oc1ccccc1", 0, 70),
        ("NO", 1, 71),
        ("COC", 1, 72),
        ("COC=O", 1, 73),
        ("CC=O", 2, 75),
        ("CS", 1, 76),
        ("CSC", 1, 77),
        ("CC=S", 2, 78),
        ("CS(=O)C", 1, 79),
        ("CS(=O)(=O)C", 1, 80),
        ("CP(=O)(C)C", 1, 81),
        ("CP(=S)(C)C", 1, 82),
        ("CF", 1, 83),
        ("Fc1ccccc1", 0, 84),
        ("CCl", 1, 85),
        ("Clc1ccccc1", 0, 86),
        ("CBr", 1, 87),
        ("Brc1ccccc1", 0, 88),
        ("CI", 1, 89),
        ("Ic1ccccc1", 0, 90),
    ],
)
def test_published_united_atom_classification(smiles, atom, expected):
    molecule = prepared(smiles)
    indices = np.flatnonzero(molecule.atom_dict["atomicnum"] != 1)
    types, scales, _ = classify_xlogp2(molecule, indices)
    assert types[atom] == expected
    assert scales[atom] == XLOGP2_CONTRIBUTIONS[expected]


def test_published_atomic_values_charge_diagnostics_and_custom_scales(docking_pair):
    assert len(XLOGP2_CONTRIBUTIONS) == 91
    assert_allclose(XLOGP2_CONTRIBUTIONS[[1, 7, 32, 66, 75, 90]], [0.528, 0.137, 0.337, 1.178, -0.399, 1.109])
    molecule = prepared("C[NH3+]")
    types, _, charged = classify_xlogp2(molecule, np.array([0, 1]))
    assert types[1] == 41 and charged == [1]
    molecule = prepared("C[N+](=O)[O-]")
    _, _, charged = classify_xlogp2(molecule, np.arange(4))
    assert not charged  # Internal nitro charge separation has a published type.
    protein, ligand, _ = docking_pair
    scales = np.zeros(np.count_nonzero(ligand.atom_dict["atomicnum"] != 1))
    engine = OddtXScoreScoring(protein, ligand, models="hm", ligand_logp=scales)
    assert engine.raw_terms()["hydrophobic_match"] == 0
    assert_array_equal(engine.lig_parameters["xlogp2_types"], -np.ones(len(scales)))


def test_vdw_published_8_4_well_depth_and_clipping():
    assert_allclose(vdw_term(np.array([0.0, 1.0, 3.8]), np.full(3, 3.8)), [100, 100, -1])
    assert vdw_term(np.array([7.6]), np.array([3.8]))[0] == pytest.approx(1 / 256 - 2 / 16)
    assert vdw_term(np.array([1e6]), np.array([3.8]))[0] < 0  # No invented short cutoff.


def test_hydrophobic_contacts_published_knots():
    assert_allclose(hydrophobic_pairs(np.array([3.8, 4.3, 5.05, 5.8, 6.8]), np.full(5, 3.8)), [1, 1, 0.5, 0, 0])


@pytest.mark.parametrize(
    "distance,environment,expected", [(6.0, -0.499, 0.528), (6.0, -0.5, 0), (6.0, -0.501, 0), (6.001, -0.6, 0.528)]
)
def test_hm_published_environment_boundary_and_empty_sum(distance, environment, expected):
    engine = OddtXScoreScoring(models="hm")
    for attribute, scale in [("rec_parameters", environment), ("lig_parameters", 0.528)]:
        setattr(
            engine,
            attribute,
            dict(
                heavy=np.array([0]),
                radii=np.array([1.9]),
                donor=np.array([0]),
                acceptor=np.array([0]),
                neighbors=[np.array([], int)],
                hydrophobic=np.array([True]),
                logp=np.array([scale]),
            ),
        )
    engine.rec_coordinates = np.zeros((1, 3))
    engine.scoring_coordinates = np.array([[distance, 0, 0.0]])
    engine.rotor_count_ = 0.0
    assert engine.raw_terms()["hydrophobic_match"] == pytest.approx(expected)


def test_hydrogen_root_angle_reference_values():
    degrees = np.array([0, 60, 90, 120, 180])
    roots = np.repeat([[1.0, 0, 0]], 5, axis=0)
    contacts = np.column_stack((np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees)), np.zeros(5)))
    assert_allclose(root_angle(roots, np.zeros((5, 3)), contacts), [0, 0, 0.5, 1, 1], atol=1e-15)
    assert root_angle(np.zeros((1, 3)), np.zeros((1, 3)), np.ones((1, 3)))[0] == 1


def hbond_parameters(donor, acceptor, neighbors=None):
    return dict(
        donor=np.array(donor),
        acceptor=np.array(acceptor),
        neighbors=(
            [np.array(n, int) for n in neighbors] if neighbors is not None else [np.array([], int) for _ in donor]
        ),
    )


@pytest.mark.parametrize("distance,expected", [(2.8, 1), (3.15, 0.5), (3.5, 0)])
def test_hydrogen_bond_published_distance_factor(distance, expected):
    rc, lc = np.array([[-1.0, 0, 0], [0, 0, 0]]), np.array([[distance, 0, 0], [distance + 1, 0, 0]])
    rec = hbond_parameters([0, 1], [0, 0], [[1], [0]])
    lig = hbond_parameters([0, 0], [1, 0], [[1], [0]])
    value, count = hydrogen_bonds(rc, lc, rec, lig, cdist(rc, lc), np.full((2, 2), 3.5))
    assert value == pytest.approx(expected)
    assert count == int(expected > 0)


def test_hydrogen_bond_capacity_and_dual_role_pair_is_counted_once():
    rc = np.zeros((1, 3))
    lc = np.array([[2.8, 0, 0], [0, 2.8, 0], [0, 0, 2.8]])
    rec = hbond_parameters([2], [0])
    lig = hbond_parameters([0] * 3, [1] * 3)
    value, count = hydrogen_bonds(rc, lc, rec, lig, cdist(rc, lc), np.full((1, 3), 3.5))
    assert value == count == 2
    rec = hbond_parameters([1], [2])
    lig = hbond_parameters([1], [2])
    value, count = hydrogen_bonds(rc, lc[:1], rec, lig, cdist(rc, lc[:1]), np.array([[3.5]]))
    assert value == count == 1


def test_hydrogen_neighbor_centroid_and_wrong_side():
    rec = hbond_parameters([0, 0, 1], [0, 0, 0], [[2], [2], [0, 1]])
    lig = hbond_parameters([0, 0], [1, 0], [[1], [0]])
    rc = np.array([[-1, -0.5, 0], [-1, 0.5, 0], [0, 0, 0.0]])
    lc = np.array([[2.8, 0, 0.0], [3.8, 0, 0]])
    value, _ = hydrogen_bonds(rc, lc, rec, lig, cdist(rc, lc), np.full((3, 2), 3.5))
    assert value == pytest.approx(1)
    lc[:, 0] *= -1
    assert hydrogen_bonds(rc, lc, rec, lig, cdist(rc, lc), np.full((3, 2), 3.5))[0] == 0


def test_sas_analytic_spherical_cap_and_ligand_occlusion():
    # Unit ligand sphere centered at origin; a unit receptor sphere one unit
    # away buries a spherical cap of height 1/2, area pi.
    rec = np.array([[1.0, 0, 0]])
    lig = np.array([[0.0, 0, 0]])
    area = buried_surface(rec, lig, np.ones(1), np.ones(1), np.ones(1, bool), 0, 0.04)
    assert area == pytest.approx(np.pi, abs=0.015)
    assert buried_surface(rec + 10, lig, np.ones(1), np.ones(1), np.ones(1, bool), 0, 0.1) == 0
    assert buried_surface(rec, lig, np.ones(1), np.ones(1), np.zeros(1, bool), 0, 0.1) == 0
    # An enclosing ligand sphere makes the inner hydrophobic atom inaccessible.
    assert buried_surface(rec, np.zeros((2, 3)), np.ones(1), np.array([1.0, 2.0]), np.array([True, False]), 0, 0.1) == 0


def test_sas_rigid_transform_covariance():
    rec = np.array([[1.0, 0.1, -0.2], [-1.0, 0.4, 0.2]])
    lig = np.array([[0, 0, 0.0], [0.0, 1.0, 0.2], [0.2, 0.3, 1.0]])
    radii = np.array([0.9, 1.1, 0.8])
    rotation = Rotation.from_rotvec([0.2, -0.8, 0.7]).as_matrix()
    before = buried_surface(rec, lig, np.ones(2), radii, np.ones(3, bool), 0.2, 0.15)
    after = buried_surface(
        rec @ rotation.T + [3, -7, 2], lig @ rotation.T + [3, -7, 2], np.ones(2), radii, np.ones(3, bool), 0.2, 0.15
    )
    assert after == pytest.approx(before, abs=1e-12)


@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("CCCC", 1),
        ("CCCCC", 2),
        ("CCC(CC)CC", 2),
        ("C1CCCCC1", 0),
        ("CCC=O", 1),
        ("c1ccccc1-c1ccccc1", 0),
        ("CC(=O)NC", 0),
    ],
)
def test_published_rotor_rule_and_crossed_rotors(smiles, expected):
    assert rotor_count(prepared(smiles)) == expected


@pytest.mark.parametrize("docking_pair", ["4bfq", "4lb3"], indirect=True)
def test_consensus_and_components_on_real_complexes(docking_pair):
    protein, ligand, _ = docking_pair
    before = protein.coords.copy(), ligand.coords.copy()
    engine = OddtXScoreScoring(protein, ligand)
    raw, variants = engine.raw_terms(), engine.score_variants()
    assert set(variants) == {"hp", "hm", "hs"}
    assert raw["vdw"] < 0 and raw["hydrophobic_surface"] > 0
    assert engine.score() == pytest.approx(np.mean(list(variants.values())))
    assert sum(engine.score_components().values()) == pytest.approx(engine.score())
    for model, hydro in [("hp", "hydrophobic_pair"), ("hm", "hydrophobic_match"), ("hs", "hydrophobic_surface")]:
        coefficients = XSCORE_COEFFICIENTS[model]
        expected = sum(
            c * value for c, value in zip(coefficients, [raw["vdw"], raw["hbond"], raw[hydro], raw["rotor"], 1])
        )
        assert variants[model] == pytest.approx(expected)
    distant = engine.scoring_coordinates + [100, 100, 100]
    assert engine.score() > engine.score(distant)
    engine.set_coords(distant)
    assert engine.score() == pytest.approx(engine.score(distant))
    assert_array_equal(protein.coords, before[0])
    assert_array_equal(ligand.coords, before[1])
    restored = pickle.loads(pickle.dumps(engine))
    assert restored.score() == pytest.approx(engine.score())


def test_variants_custom_coefficients_and_surface_skip(docking_pair, monkeypatch):
    import oddt.docking.xscore as xscore

    protein, ligand, _ = docking_pair
    monkeypatch.setattr(xscore, "buried_surface", lambda *a: pytest.fail("disabled HS must not calculate surface"))
    engine = OddtXScoreScoring(protein, ligand, models=["hp", "hm"])
    assert engine.raw_terms()["hydrophobic_surface"] == 0
    assert set(engine.score_variants()) == {"hp", "hm"}
    engine = OddtXScoreScoring(protein, ligand, models="hp", coefficients={"hp": (0, 0, 0, 0, 7.25)})
    assert engine.score() == 7.25
    assert sum(engine.score_components().values()) == 7.25


def test_water_removal_is_private(docking_pair):
    protein, ligand, _ = docking_pair
    removed = OddtXScoreScoring(protein, ligand)
    retained = OddtXScoreScoring(protein, ligand, remove_water=False)
    p = removed._receptor.atom_dict
    assert not np.isin(p["resname"][removed.rec_parameters["heavy"]], ["HOH", "WAT", "H2O"]).any()
    assert len(removed.rec_parameters["heavy"]) <= len(retained.rec_parameters["heavy"])


@pytest.mark.parametrize("target", ["10gs", "1imx", "1k9q", "1nlo", "1nlp", "4yef"])
def test_native_pdbbind_fixture_coverage(target):
    from pathlib import Path

    directory = Path(__file__).parent / "data/pdbbind" / target
    protein = next(oddt.toolkit.readfile("pdb", str(directory / (target + "_pocket.pdb"))))
    protein.protein = True
    ligand = next(oddt.toolkit.readfile("sdf", str(directory / (target + "_ligand.sdf"))))
    engine = OddtXScoreScoring(protein, ligand)
    values = engine.score_variants()
    assert len(values) == 3 and np.isfinite(list(values.values())).all()
    assert engine.raw_terms()["hydrophobic_surface"] >= 0


def test_scored_copies_streaming_ranking_and_stale_metadata(docking_pair):
    protein, ligand, _ = docking_pair
    ligand.title = "native"
    ligand.data.update({"source": "input", "xscore_old": 123})
    distant = ligand.clone
    distant.data.update(dict(ligand.data))
    distant.coords = distant.coords + [100, 100, 100]
    engine = OddtXScoreScoring(protein)
    ranked = engine.rescore([distant, ligand])
    assert ranked[0].title == "native"
    assert float(ranked[0].data["xscore_score"]) > float(ranked[1].data["xscore_score"])
    assert all(p.data["xscore_units"] == "pKd" and p.data["source"] == "input" for p in ranked)
    assert "xscore_old" not in ranked[0].data
    assert int(ligand.data["xscore_old"]) == 123
    assert "xscore_score" not in ligand.data
    result = list(engine.predict_ligands([distant, ligand]))
    assert float(result[0].data["xscore_score"]) < float(result[1].data["xscore_score"])


@pytest.mark.parametrize("selector", ["xscore", "xscore_hp", "xscore_hm", "xscore_hs", "object"])
def test_virtual_screening_integration(docking_pair, selector):
    from oddt.virtualscreening import virtualscreening

    protein, ligand, _ = docking_pair
    vs = virtualscreening(n_cpu=1)
    vs._mol_feed = iter([ligand])
    function = OddtXScoreScoring(models="hp") if selector == "object" else selector
    vs.score(function, protein=protein)
    poses = list(vs.fetch())
    assert len(poses) == 1
    assert "xscore_score" in poses[0].data


@pytest.mark.parametrize(
    "options",
    [
        dict(models=[]),
        dict(models=["hp", "hp"]),
        dict(models="bad"),
        dict(surface_spacing=0),
        dict(probe_radius=-1),
        dict(hydrogen_radius=np.inf),
        dict(coefficients={"bad": (0,) * 5}),
        dict(coefficients={"hp": (0,) * 4}),
        dict(coefficients={"hs": [np.nan] * 5}),
    ],
)
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        OddtXScoreScoring(**options)


def test_missing_inputs_bad_coordinates_and_unsupported_elements(docking_pair):
    protein, ligand, _ = docking_pair
    with pytest.raises(ValueError, match="ligand is required"):
        OddtXScoreScoring(protein).score()
    with pytest.raises(ValueError, match="receptor is required"):
        OddtXScoreScoring(lig=ligand).score()
    engine = OddtXScoreScoring(protein, ligand)
    original_score = engine.score()
    with pytest.raises(ValueError, match="every prepared ligand atom"):
        engine.score(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="atomic logP scales"):
        OddtXScoreScoring(protein, ligand, ligand_logp=[0])
    with pytest.raises(ValueError, match="Unsupported X-Score element"):
        engine.set_ligand(prepared("C[Si](C)(C)C"))
    with pytest.raises(ValueError, match="ligand metal ions"):
        engine.set_ligand(prepared("[Zn+2]"))
    assert engine.score() == pytest.approx(original_score)
    assert not hasattr(engine, "dock")
