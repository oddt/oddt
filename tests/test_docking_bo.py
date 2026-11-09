"""Bayesian search acquisition, periodic geometry and scorer integration."""

import pickle

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.optimize import OptimizeResult
from scipy.spatial.transform import Rotation

from oddt import docking
from oddt.docking import BayesianOptimizationMixin, OddtVinaBO
from oddt.docking.search import _bo_features, _bo_local_trial, _expected_improvement
from .test_docking import QuadraticScoring, docking_pair, progress_output  # noqa: F401
from .test_docking_aco import ArchiveHarness


def test_expected_improvement_known_values_and_zero_variance():
    actual = _expected_improvement(np.array([0.0, 1.0, -1.0, 1.0]), np.array([1.0, 1.0, 0.0, 0.0]), 0, 0)
    assert_allclose(actual, [1 / np.sqrt(2 * np.pi), 0.0833154705876863, 1, 0])
    assert _expected_improvement(np.array([0.0]), np.array([2.0]), 0, 0)[0] > actual[0]
    assert _expected_improvement(np.array([0.0]), np.array([1.0]), 0, 0.2)[0] < actual[0]


def test_embedding_periodicity_rotation_equivalence_and_group_normalization():
    bounds = np.array([[-2.0] * 3, [2.0] * 3])
    poses = np.zeros((4, 8))
    poses[1, :3] = 4  # Opposite translation corners have distance one.
    poses[2, 3] = np.pi
    poses[3, 6:] = np.pi
    features = _bo_features(poses, bounds)
    assert_allclose(np.linalg.norm(features[1:] - features[0], axis=1), 1)
    changed = poses.copy()
    changed[:, 3:] += 2 * np.pi
    assert_allclose(_bo_features(changed, bounds), features, atol=1e-15)
    # A nontrivial equivalent Euler branch, beyond adding full turns.
    first = np.array([[0, 0, 0, 0.3, -0.2, 0.7, np.pi - 0.01]])
    second = first.copy()
    second[0, 3:6] = [0.3 + np.pi, np.pi + 0.2, 0.7 + np.pi]
    assert_allclose(_bo_features(first, bounds), _bo_features(second, bounds), atol=1e-15)
    second = first.copy()
    second[0, 6] = -np.pi + 0.01
    assert np.linalg.norm(_bo_features(second, bounds) - _bo_features(first, bounds)) == pytest.approx(np.sin(0.01))
    assert _bo_features(np.zeros((3, 6)), bounds).shape == (3, 12)  # Rigid ligand.


def test_local_candidate_bounds_rotation_composition_and_torsion_wrap():
    center = np.array([0.9, -0.9, 0, 0.2, -0.5, np.pi - 0.01, np.pi - 0.01])
    before = center.copy()
    bounds = np.array([[-1.0] * 3, [1.0] * 3])
    normal = np.array([5, -5, 2, 0.2, -0.1, 0.3, 0.1])

    class Generator:
        def normal(self, size):
            return normal

    candidate = _bo_local_trial(center, bounds, 0.5, Generator())
    assert_allclose(candidate[:3], [0.1, -0.1, 0], atol=1e-12)
    expected = Rotation.from_rotvec(normal[3:6] * 0.5 * np.pi) * Rotation.from_euler("xyz", center[3:6])
    assert_allclose(Rotation.from_euler("xyz", candidate[3:6]).as_matrix(), expected.as_matrix(), atol=1e-12)
    assert candidate[6] == pytest.approx(-np.pi + 0.1 * np.pi - 0.01)
    assert_array_equal(center, before)


class PeriodicHarness(BayesianOptimizationMixin, ArchiveHarness):
    """One periodic basin objective, without local minimization."""

    def __init__(self):
        super().__init__()
        self.evaluations = []

    def _random_conformation(self, bounds, rng):
        x = np.zeros(7)
        x[6] = rng.uniform(-np.pi, np.pi)
        return x

    def _bo_candidates(self, bounds, rng, count, *args):
        return np.array([self._random_conformation(bounds, rng) for _ in range(count)])

    def optimize(self, x, maxiter):
        self.calls += 1
        value = float(1 - np.cos(x[6] - 2.8))
        self.evaluations.append((x.copy(), value))
        return OptimizeResult(x=x.copy(), fun=value, nfev=1)


def test_surrogate_learns_periodic_objective_without_local_minimizer():
    engine = PeriodicHarness()
    results = engine.dock(n_initial=4, iterations=18, candidate_pool=128, random_fraction=0, seed=42)
    assert results[0].fun < 0.002
    assert results[0].fun < engine.history_[0] / 10
    assert np.all(np.diff(engine.history_) <= 0)
    assert len(engine.history_) == 19
    stats = engine.search_stats_[0]
    assert stats["acquisition_samples"] == 18
    assert stats["local_minimizations"] == stats["objective_evaluations"] == 22
    assert stats["surrogate_fits"] == 19
    # Final posterior and raw observations include the final evaluation.
    assert len(engine.surrogate_.X_train_) == len(engine.bo_observations_["energy"]) == 22
    assert_array_equal(engine.bo_observations_["x"][-1], engine.evaluations[-1][0])


def test_memory_keeps_best_plus_recent_and_random_mode_avoids_acquisition():
    engine = PeriodicHarness()
    engine.dock(n_initial=6, iterations=8, max_samples=4, random_fraction=1, seed=42)
    observation = engine.bo_observations_
    assert len(observation["energy"]) == len(engine.surrogate_.X_train_) == 4
    assert observation["energy"].min() == min(y for _, y in engine.evaluations)
    assert observation["energy"][-1] == engine.evaluations[-1][1]
    stats = engine.search_stats_[0]
    assert stats["acquisition_samples"] == 0 and stats["random_samples"] == 8
    assert stats["surrogate_fits"] == 1


def test_observations_pair_raw_start_with_post_refinement_energy(monkeypatch):
    engine = PeriodicHarness()
    starts = []

    def optimize(x, maxiter):
        starts.append(x.copy())
        # Mock a many-to-one basin map: all starts relax to the same pose.
        return OptimizeResult(x=np.zeros(7), fun=0.0, nfev=3)

    monkeypatch.setattr(engine, "optimize", optimize)
    engine.dock(n_initial=4, iterations=0, seed=42)
    assert_array_equal(engine.bo_observations_["x"], starts)
    assert_array_equal(engine.bo_observations_["energy"], np.zeros(4))
    assert engine.surrogate_target_scale_ == 1
    assert np.isfinite(engine.surrogate_.predict(_bo_features(np.array(starts), engine.bounds))).all()


def test_duplicate_candidates_fall_back_and_duplicate_observations_are_stable(monkeypatch):
    engine = PeriodicHarness()
    monkeypatch.setattr(engine, "_random_conformation", lambda *args: np.zeros(7))
    engine.dock(n_initial=2, iterations=1, candidate_pool=4, random_fraction=0, noise=1e-20, seed=42)
    assert engine.search_stats_[0]["duplicate_fallbacks"] == 1
    assert engine.search_stats_[0]["random_samples"] == 1
    assert np.isfinite(engine.surrogate_.L_).all()


def test_zero_expected_improvement_uses_highest_uncertainty(monkeypatch):
    import oddt.docking.search as search
    from sklearn.gaussian_process import GaussianProcessRegressor

    engine = PeriodicHarness()
    pool = np.zeros((3, 7))
    pool[:, 6] = [0.1, 0.2, 0.3]
    monkeypatch.setattr(engine, "_bo_candidates", lambda *args: pool)
    monkeypatch.setattr(GaussianProcessRegressor, "predict", lambda *a, **k: (np.zeros(3), np.array([0.2, 0.8, 0.4])))
    monkeypatch.setattr(search, "_expected_improvement", lambda *args: np.zeros(3))
    engine.dock(n_initial=2, iterations=1, candidate_pool=3, random_fraction=0, seed=42)
    assert_array_equal(engine.bo_observations_["x"][-1], pool[1])


@pytest.mark.parametrize("model", ["Vina", "Vinardo", "AD4", "PLP", "ChemPLP"])
def test_all_scorers_compose_and_preserve_inputs(docking_pair, model):
    protein, ligand, box = docking_pair
    before = protein.coords.copy(), ligand.coords.copy()
    cls = getattr(docking, "Oddt" + model + "BO")
    observations = []
    engine = cls(protein, ligand, box=box, sample_callback=lambda result, cap: observations.append((result.fun, cap)))
    poses = engine.dock(
        n_initial=4, iterations=2, candidate_pool=16, maxiter=40, seed=42, num_modes=3, energy_range=1000
    )
    assert poses and engine.rotors
    assert len(observations) == 6 and all(cap is None for _, cap in observations)
    for pose, result in zip(poses, engine.results_):
        heavy = pose.atom_dict["atomicnum"] != 1
        assert np.all(pose.coords[heavy] >= box[0] - 1e-4)
        assert np.all(pose.coords[heavy] <= box[1] + 1e-4)
        key = model.lower() + ("_score" if "PLP" in model else "_affinity")
        assert float(pose.data[key]) == pytest.approx(engine.score(engine._result_coords(result)))
    assert_array_equal(protein.coords, before[0])
    assert_array_equal(ligand.coords, before[1])
    assert isinstance(engine, BayesianOptimizationMixin)
    assert cls.__name__ in docking.__all__
    assert type(pickle.loads(pickle.dumps(cls(None)))) is cls
    assert_allclose(
        pickle.loads(pickle.dumps(engine.surrogate_)).predict(engine.surrogate_.X_train_),
        engine.surrogate_.predict(engine.surrogate_.X_train_),
    )


def test_progress_seed_rng_state_and_zero_iterations(docking_pair, progress_output):
    protein, ligand, box = docking_pair
    engine = OddtVinaBO(protein, ligand, box=box)
    state = np.random.get_state()
    options = dict(n_initial=4, iterations=2, candidate_pool=16, maxiter=40, seed=7, num_modes=2)
    quiet = engine.dock(**options)
    history, stats = engine.history_, engine.search_stats_
    visible = engine.dock(**options, progress=True)
    assert engine.history_ == history and engine.search_stats_ == stats
    assert_array_equal([p.coords for p in quiet], [p.coords for p in visible])
    bars, _ = progress_output
    assert bars[0].n == bars[0].total == 6 and bars[0].disable
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    assert_array_equal(state[1], after[1])
    engine.dock(n_initial=4, iterations=0, maxiter=40, seed=7)
    assert len(engine.history_) == 1
    assert engine.search_stats_[0]["local_minimizations"] == 4


def test_independent_scorer_uses_energy_rather_than_reported_score(docking_pair):
    class Engine(BayesianOptimizationMixin, QuadraticScoring):
        pass

    protein, ligand, box = docking_pair
    engine = Engine(protein, ligand, box=box)
    poses = engine.dock(n_initial=4, iterations=2, candidate_pool=16, maxiter=30, seed=7)
    values = [float(p.data["docking_energy"]) for p in poses]
    assert values == sorted(values)
    assert all(float(p.data["docking_score"]) == pytest.approx(-v) for p, v in zip(poses, values))
    assert all(p.data["docking_scorer"] == "quadratic" for p in poses)


def test_nonfinite_results_are_excluded_and_progress_closes_on_failure(docking_pair, monkeypatch, progress_output):
    protein, ligand, box = docking_pair
    engine = OddtVinaBO(protein, ligand, box=box)
    monkeypatch.setattr(engine, "optimize", lambda *a, **k: OptimizeResult(fun=np.nan, nfev=1))
    with pytest.raises(RuntimeError, match="two finite initial poses"):
        engine.dock(n_initial=4, progress=True)
    bars, _ = progress_output
    assert bars[0].n == 4 and bars[0].disable


@pytest.mark.parametrize(
    "options",
    [
        dict(n_initial=1),
        dict(iterations=-1),
        dict(candidate_pool=1),
        dict(max_samples=1),
        dict(length_scale=0),
        dict(noise=np.inf),
        dict(exploration=-1),
        dict(local_scale=0),
        dict(local_scale=2),
        dict(local_fraction=1.1),
        dict(random_fraction=-0.1),
        dict(archive_size=0),
        dict(num_modes=0),
        dict(maxiter=0),
        dict(min_rmsd=-1),
        dict(energy_range=-1),
        dict(n_initial=True),
    ],
)
def test_invalid_configuration(docking_pair, options):
    protein, ligand, box = docking_pair
    with pytest.raises(ValueError):
        OddtVinaBO(protein, ligand, box=box).dock(**options)
