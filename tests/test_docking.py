"""Internal genetic docking, using crystallographic test complexes."""

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import oddt
from oddt.docking import vina_docking
from oddt.spatial import distance, rmsd, rotate

DATA = Path(__file__).parent / "data" / "pdb"


@pytest.fixture
def docking_pair(request):
    name = getattr(request, "param", "4bfq")
    protein = next(oddt.toolkit.readfile("pdb", str(DATA / (name + "_pocket.pdb"))))
    protein.protein = True
    ligand = next(oddt.toolkit.readfile("sdf", str(DATA / (name + "_ligand.sdf"))))
    ligand.addh()
    heavy = ligand.atom_dict["atomicnum"] != 1
    center = ligand.coords[heavy].mean(axis=0)
    return protein, ligand, np.array([center - 8, center + 8])


def test_pose_transform_is_deterministic_and_preserves_geometry(docking_pair):
    protein, ligand, box = docking_pair
    engine = vina_docking(protein, ligand, box=box, box_size=2)
    x = np.linspace(0.13, 0.67, len(engine.lig.x0))
    expected = engine.lig.mutate(x, force=True)
    for i in range(len(x)):
        perturbed = x.copy()
        perturbed[i] += 0.01
        engine.lig.mutate(perturbed)
        assert_allclose(engine.lig.mutate(x), expected, atol=1e-12)
    full = engine.lig.mutate(x, all_atoms=True)
    assert_allclose(full[engine._heavy_mask], expected, atol=1e-10)
    assert_allclose(expected.mean(axis=0), engine.lig.c0.mean(axis=0) + 2 * x[:3])
    assert expected.dtype == np.float64
    for bond in ligand.bonds:
        a, b = (atom.idx0 for atom in bond.atoms)
        assert_allclose(
            np.linalg.norm(full[a] - full[b]), np.linalg.norm(ligand.coords[a] - ligand.coords[b]), atol=1e-5
        )
    assert all(rotor["mask"].sum() <= len(engine.lig.c0) / 2 for rotor in engine.rotors)


@pytest.mark.parametrize("boundary", [False, True])
@pytest.mark.parametrize("docking_pair", ["4bfq", "4lb3"], indirect=True)
def test_objective_gradient(docking_pair, boundary):
    protein, ligand, box = docking_pair
    engine = vina_docking(protein, ligand, box=box, box_size=2)
    x = np.linspace(0.13, 0.67, len(engine.lig.x0))
    if boundary:
        x[0] = 5
    energy, actual = engine._objective(x)
    numeric = np.empty_like(x)
    for i in range(len(x)):
        delta = np.zeros_like(x)
        delta[i] = 1e-5
        numeric[i] = (engine._objective(x + delta)[0] - engine._objective(x - delta)[0]) / 2e-5
    assert_allclose(actual, numeric, rtol=1e-5, atol=2e-4)
    if not boundary:
        assert energy == pytest.approx(engine.weighted_total(engine.lig.mutate(x)))


def test_intraligand_pairs_are_counted_once(docking_pair):
    protein, ligand, box = docking_pair
    engine = vina_docking(protein, ligand, box=box)
    coords = engine.lig.c0
    radii = engine.lig_dict["radius"]
    r = distance(coords, coords)
    mask = np.triu(engine.lig_distant_members, k=1) & (r < 8)
    d = (r - radii[:, None] - radii[None, :])[mask]
    terms = engine.score_intra()
    assert terms[0] == pytest.approx(np.exp(-((d / 0.5) ** 2)).sum())
    assert terms[2] == pytest.approx((np.minimum(d, 0) ** 2).sum())
    assert not np.any(np.diag(engine.lig_distant_members))


def test_lbfgs_improves_a_perturbed_pose(docking_pair):
    protein, ligand, box = docking_pair
    engine = vina_docking(protein, ligand, box=box)
    x = np.linspace(0.13, 0.67, len(engine.lig.x0))
    initial = engine._objective(x)[0]
    result = engine.optimize(x, maxiter=60)
    assert result.fun < initial - 1
    assert result.nit > 0
    assert result.coords.dtype == np.float64
    assert result.fun == pytest.approx(engine.weighted_total(result.coords))
    assert result.affinity == pytest.approx(engine.score(result.coords))
    assert_allclose(result.coords, engine.lig.mutate(result.x))


def test_conjugated_amide_stays_rigid():
    ligand = next(oddt.toolkit.readfile("sdf", str(DATA / "4lb3_ligand.sdf")))
    ligand.addh()
    engine = vina_docking(None, ligand)
    # The carbonyl C-N bond (10, 9) is conjugated. The six remaining
    # heavy-atom torsions retain flexibility on either side of the amide.
    assert len(engine.rotors) == 6
    assert frozenset((9, 10)) not in {frozenset(rotor["full_atoms"][1:3]) for rotor in engine.rotors}
    atoms = [8, 9, 10, 11]  # C-N-C=O, including both amide substituents.
    original = distance(ligand.coords[atoms], ligand.coords[atoms])
    rng = np.random.default_rng(4)
    for _ in range(5):
        x = rng.uniform(-np.pi, np.pi, len(engine.lig.x0))
        coords = engine.lig.mutate(x, all_atoms=True)
        assert_allclose(distance(coords[atoms], coords[atoms]), original, atol=1e-6)


def test_genetic_redocking_4lb3():
    protein = next(oddt.toolkit.readfile("pdb", str(DATA / "4lb3_pocket.pdb")))
    protein.protein = True
    reference = next(oddt.toolkit.readfile("sdf", str(DATA / "4lb3_ligand.sdf")))
    reference.addh()
    center = reference.coords[reference.atom_dict["atomicnum"] != 1].mean(axis=0)
    box = np.array([center - 8, center + 8])
    ligand = reference.clone
    ligand.title = "displaced 4LB3"
    ligand.data.update({"source": "4LB3", "vina_gauss1": -999})
    ligand.coords = rotate(ligand.coords, 1.2, -0.7, 2.1) + [12, -7, 8]
    original = ligand.coords.copy()
    receptor_coords = protein.coords.copy()
    assert rmsd(reference, ligand) > 10
    engine = vina_docking(protein, ligand, box=box)
    poses = engine.dock(population_size=24, generations=8, maxiter=60, seed=42, num_modes=5)
    assert 1 <= len(poses) <= 5
    assert rmsd(reference, poses[0]) < 1
    assert len(engine.history_) == 9
    assert np.all(np.diff(engine.history_) <= 0)
    assert_array_equal(ligand.coords, original)
    assert_array_equal(protein.coords, receptor_coords)
    totals = [float(pose.data["vina_total"]) for pose in poses]
    assert totals == sorted(totals)
    for i, (pose, result) in enumerate(zip(poses, engine.results_)):
        heavy_coords = pose.coords[pose.atom_dict["atomicnum"] != 1]
        assert_allclose(heavy_coords, result.coords, atol=1e-5)
        assert np.all(heavy_coords >= box[0] - 1e-4)
        assert np.all(heavy_coords <= box[1] + 1e-4)
        assert pose.title == ligand.title
        assert pose.data["source"] == "4LB3"
        assert float(pose.data["vina_gauss1"]) == pytest.approx(engine.score_inter(result.coords)[0])
        assert float(pose.data["vina_affinity"]) == pytest.approx(engine.score(result.coords))
        assert float(pose.data["vina_total"]) == pytest.approx(
            float(pose.data["vina_inter"]) + float(pose.data["vina_intra"])
        )
        assert totals[i] <= totals[0] + 3
        assert all(rmsd(pose, other) >= 1 for other in poses[:i])


def test_rigid_docking_is_reproducible_without_changing_global_rng():
    protein = next(oddt.toolkit.readfile("pdb", str(DATA / "4ljh_pocket.pdb")))
    protein.protein = True
    ligand = next(oddt.toolkit.readfile("sdf", str(DATA / "4ljh_ligand.sdf")))
    ligand.data.update({"vina_rmsd_lb": -999})
    center = ligand.coords.mean(axis=0)
    engine = vina_docking(protein, ligand, box=[center - 10, center + 10])
    assert not engine.rotors
    state = np.random.get_state()
    poses = engine.dock(population_size=4, generations=1, maxiter=5, seed=7)
    again = engine.dock(population_size=4, generations=1, maxiter=5, seed=7)
    for first, second in zip(poses, again):
        assert_array_equal(first.coords, second.coords)
        assert "vina_rmsd_lb" not in first.data
    assert len(poses) == len(again)
    after = np.random.get_state()
    assert state[0] == after[0]
    assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"population_size": 3},
        {"generations": -1},
        {"generations": 1.5},
        {"num_modes": 0},
        {"maxiter": 0},
        {"mutation_rate": 1.1},
        {"min_rmsd": -1},
        {"energy_range": np.nan},
    ],
)
def test_invalid_search_parameters(docking_pair, kwargs):
    protein, ligand, box = docking_pair
    with pytest.raises(ValueError):
        vina_docking(protein, ligand, box=box).dock(**kwargs)


def test_missing_search_inputs_and_invalid_box(docking_pair):
    protein, ligand, box = docking_pair
    for engine in (vina_docking(None, ligand, box), vina_docking(protein, box=box), vina_docking(protein, ligand)):
        with pytest.raises(ValueError):
            engine.dock()
    for invalid in (np.zeros((2, 3)), [0, 1], [[0, 0, 0], [1, np.inf, 1]]):
        with pytest.raises(ValueError, match="box"):
            vina_docking(protein, ligand, box=invalid)
    engine = vina_docking(protein, ligand, box=box, weights=np.ones(6))
    assert_array_equal(engine.weights, np.ones(6))
    with pytest.raises(ValueError, match="x0"):
        engine.optimize([0])


def test_no_feasible_pose_in_tiny_box(docking_pair):
    protein, ligand, box = docking_pair
    center = box.mean(axis=0)
    engine = vina_docking(protein, ligand, box=[center - 0.01, center + 0.01])
    with pytest.raises(RuntimeError, match="No pose fits"):
        engine.dock(population_size=4, generations=0, maxiter=1, seed=1)
    assert engine.results_ == []
