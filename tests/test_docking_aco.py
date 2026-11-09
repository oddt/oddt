"""Continuous ant-colony kernels, archive learning and docking integration."""

import pickle
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.optimize import OptimizeResult
from scipy.spatial.transform import Rotation

from oddt import docking
from oddt.docking import AntColonyMixin, OddtVinaACO
from oddt.docking.search import _acor_sample, _acor_weights, _acor_widths
from .test_docking import QuadraticScoring, docking_pair, progress_output  # noqa: F401


def test_gaussian_rank_weights_and_numerical_limits():
    actual = _acor_weights(4, 0.25)
    expected = np.exp(-0.5 * np.arange(4) ** 2)
    assert_allclose(actual, expected / expected.sum())
    assert np.all(np.diff(actual) < 0)
    assert_array_equal(_acor_weights(4, 1e-300), [1, 0, 0, 0])
    assert_allclose(_acor_weights(4, 1e100), np.full(4, 0.25))


def test_archive_widths_use_so3_and_shortest_torsion_arcs():
    poses = np.array([[0, 0, 0, 0, 0, np.pi - 0.05, np.pi - 0.1], [2, 4, 6, 0, 0, -np.pi + 0.05, -np.pi + 0.1]], float)
    bounds = np.array([[-10.0] * 3, [10.0] * 3])
    actual = _acor_widths(poses, 0, 0.5, bounds, 0)
    assert_allclose(actual[:3], [1, 2, 3])
    assert_allclose(actual[3:6], np.full(3, 0.05 / np.sqrt(3)))
    assert actual[6] == pytest.approx(0.1)
    # Equivalent Euler representations must have the same SO(3) distances.
    changed = poses.copy()
    changed[1, 3:6] += 2 * np.pi
    assert_allclose(_acor_widths(changed, 0, 0.5, bounds, 0), actual, atol=1e-14)
    collapsed = np.repeat(poses[:1], 3, axis=0)
    floors = _acor_widths(collapsed, 0, 0.85, bounds, 0.02)
    assert_allclose(floors[:3], [0.4] * 3)
    assert_allclose(floors[3:6], np.full(3, 0.04 * np.pi / np.sqrt(3)))
    assert floors[6] == pytest.approx(0.04 * np.pi)


def test_ant_sampling_reflects_translation_and_composes_orientation():
    poses = np.array([[0.9, -0.9, 0, 0.2, -0.5, np.pi - 0.01, np.pi - 0.01]] * 2)
    bounds = np.array([[-1.0] * 3, [1.0] * 3])
    step = np.array([5.0, -5.0, 2.0, 0.2, -0.1, 0.3, 0.1])
    sigma = _acor_widths(poses, 0, 0.85, bounds, 0.1)

    class Generator:
        def choice(self, count, p):
            return 0

        def normal(self, size):
            return step / sigma

    candidate = _acor_sample(poses, [0.75, 0.25], 0.85, bounds, 0.1, Generator())
    assert_allclose(candidate[:3], [0.1, -0.1, 0], atol=1e-12)
    expected = Rotation.from_rotvec(step[3:6]) * Rotation.from_euler("xyz", poses[0, 3:6])
    assert_allclose(Rotation.from_euler("xyz", candidate[3:6]).as_matrix(), expected.as_matrix(), atol=1e-12)
    assert candidate[6] == pytest.approx(-np.pi + 0.09)


class ArchiveHarness(AntColonyMixin):
    """No local optimizer: improvements here must come from colony learning."""

    def __init__(self, invalid=False):
        self.lig = SimpleNamespace(x0=np.zeros(7))
        self.bounds = np.array([[-2.0] * 3, [2.0] * 3])
        self.calls = 0
        self.invalid = invalid

    def _search_bounds(self):
        return self.bounds

    def _random_conformation(self, bounds, rng):
        pose = rng.uniform(-np.pi, np.pi, 7)
        pose[:3] = rng.uniform(*bounds)
        pose[3:6] = Rotation.random(random_state=rng).as_euler("xyz")
        return pose

    def optimize(self, x, maxiter):
        self.calls += 1
        angle = Rotation.from_euler("xyz", x[3:6]).magnitude()
        value = float(np.sum((x[:3] - [0.4, -0.2, 0.6]) ** 2) + angle**2 + 1 - np.cos(x[6] - 2.8))
        if self.invalid and self.calls % 3 == 0:
            value = np.nan
        return OptimizeResult(x=x.copy(), fun=value, nfev=1)

    def _add_minimum(self, candidates, result, min_rmsd, capacity):
        if np.isfinite(result.fun):
            candidates.append(result)
            candidates.sort(key=lambda result: result.fun)
            del candidates[capacity:]

    def _finish_docking(self, candidates, *args):
        return candidates


def test_colony_learns_without_local_optimizer_and_memory_is_bounded():
    engine = ArchiveHarness()
    results = engine.dock(ants=12, iterations=80, archive_size=24, min_sigma=0.002, seed=42)
    assert results[0].fun < 0.02
    assert engine.history_[-1] < engine.history_[0] / 100
    assert np.all(np.diff(engine.history_) <= 0)
    assert len(engine.history_) == 81
    assert len(engine.pheromone_archive_) == len(engine.pheromone_weights_) == 24
    assert len(results) <= 24
    values = [result.fun for result in engine.pheromone_archive_]
    assert values == sorted(values)
    assert engine.calls == 24 + 12 * 80
    stats = engine.search_stats_[0]
    assert stats["local_minimizations"] == stats["objective_evaluations"] == engine.calls
    assert stats["kernel_samples"] + stats["random_samples"] == 12 * 80
    assert stats["archive_replacements"] > 0


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_random_immigrants_and_nonfinite_candidates(fraction):
    engine = ArchiveHarness(invalid=True)
    results = engine.dock(ants=4, iterations=2, archive_size=6, random_fraction=fraction, seed=7)
    assert len(engine.pheromone_archive_) == 6
    assert all(np.isfinite(result.fun) for result in engine.pheromone_archive_ + results)
    stats = engine.search_stats_[0]
    assert stats["random_samples"] == (8 if fraction else 0)
    assert stats["kernel_samples"] == (0 if fraction else 8)


def test_colony_samples_from_frozen_archive(monkeypatch):
    import oddt.docking.search as search

    archives = []
    sample = search._acor_sample

    def capture(poses, *args):
        archives.append(poses.copy())
        return sample(poses, *args)

    monkeypatch.setattr(search, "_acor_sample", capture)
    ArchiveHarness().dock(ants=3, iterations=2, archive_size=4, random_fraction=0, seed=7)
    assert len(archives) == 6
    for iteration in (0, 3):
        assert_array_equal(archives[iteration], archives[iteration + 1])
        assert_array_equal(archives[iteration], archives[iteration + 2])
    assert not np.array_equal(archives[0], archives[3])


@pytest.mark.parametrize("model", ["Vina", "Vinardo", "AD4", "PLP", "ChemPLP"])
def test_all_scoring_models_compose_and_preserve_inputs(docking_pair, model):
    protein, ligand, box = docking_pair
    original = protein.coords.copy(), ligand.coords.copy()
    cls = getattr(docking, "Oddt" + model + "ACO")
    observations = []
    engine = cls(protein, ligand, box=box, sample_callback=lambda result, cap: observations.append((result.fun, cap)))
    poses = engine.dock(ants=2, iterations=1, archive_size=4, maxiter=40, seed=42, num_modes=3, energy_range=1000)
    assert poses
    assert len(observations) == 6
    assert all(cap is None for _, cap in observations)
    assert engine.rotors  # Flexible 4BFQ ligand; perception differs by backend.
    assert_array_equal(protein.coords, original[0])
    assert_array_equal(ligand.coords, original[1])
    for pose, result in zip(poses, engine.results_):
        heavy = pose.atom_dict["atomicnum"] != 1
        assert np.all(pose.coords[heavy] >= box[0] - 1e-4)
        assert np.all(pose.coords[heavy] <= box[1] + 1e-4)
        score_key = model.lower() + ("_score" if "PLP" in model else "_affinity")
        assert float(pose.data[score_key]) == pytest.approx(engine.score(engine._result_coords(result)))
    assert isinstance(engine, AntColonyMixin)
    assert type(pickle.loads(pickle.dumps(cls(None)))) is cls
    assert cls.__name__ in docking.__all__


def test_progress_seed_rng_state_and_zero_iterations(docking_pair, progress_output):
    protein, ligand, box = docking_pair
    engine = OddtVinaACO(protein, ligand, box=box)
    state = np.random.get_state()
    options = dict(ants=2, iterations=2, archive_size=4, maxiter=3, seed=7, num_modes=2)
    quiet = engine.dock(**options)
    history, stats = engine.history_, engine.search_stats_
    visible = engine.dock(**options, progress=True)
    assert engine.history_ == history
    assert engine.search_stats_ == stats
    assert_array_equal([pose.coords for pose in quiet], [pose.coords for pose in visible])
    bars, _ = progress_output
    assert bars[0].n == bars[0].total == 8
    assert bars[0].disable
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    assert_array_equal(state[1], after[1])
    engine.dock(archive_size=4, iterations=0, maxiter=40, seed=7)
    assert len(engine.history_) == 1
    assert engine.search_stats_[0]["local_minimizations"] == 4


def test_independent_scorer_uses_search_energy_and_neutral_metadata(docking_pair):
    class Engine(AntColonyMixin, QuadraticScoring):
        pass

    protein, ligand, box = docking_pair
    engine = Engine(protein, ligand, box=box)
    poses = engine.dock(archive_size=4, ants=2, iterations=1, maxiter=30, seed=7)
    values = [float(pose.data["docking_energy"]) for pose in poses]
    assert values == sorted(values)
    assert all(pose.data["docking_scorer"] == "quadratic" for pose in poses)
    assert all(float(pose.data["docking_score"]) == pytest.approx(-value) for pose, value in zip(poses, values))


def test_progress_closes_when_no_finite_initial_poses(docking_pair, monkeypatch, progress_output):
    protein, ligand, box = docking_pair
    engine = OddtVinaACO(protein, ligand, box=box)
    monkeypatch.setattr(engine, "optimize", lambda *a, **k: OptimizeResult(fun=np.nan, nfev=1))
    with pytest.raises(RuntimeError, match="two finite initial poses"):
        engine.dock(archive_size=4, progress=True)
    bars, _ = progress_output
    assert bars[0].n == 4 and bars[0].disable


@pytest.mark.parametrize(
    "options",
    [
        dict(ants=0),
        dict(iterations=-1),
        dict(archive_size=1),
        dict(q=0),
        dict(xi=np.inf),
        dict(min_sigma=-0.1),
        dict(random_fraction=1.1),
        dict(num_modes=0),
        dict(maxiter=0),
        dict(min_rmsd=-1),
        dict(energy_range=-1),
        dict(ants=True),
    ],
)
def test_invalid_parameters(docking_pair, options):
    protein, ligand, box = docking_pair
    with pytest.raises(ValueError):
        OddtVinaACO(protein, ligand, box=box).dock(**options)
