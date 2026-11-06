"""Global pose-search mixins, independent of the pair-scoring implementation.

A scoring base supplies ``_search_bounds``, ``_random_conformation``,
``optimize`` (returning x, fun, coords, affinity and nfev), ``_add_minimum`` and
``_finish_docking``. Pose geometry is exposed as ``lig.x0``, ``lig.mutate(x)``,
``rotors`` and ``box_size``. Both searches use the same local minimizer and
output selection supplied by that base.
"""

import math

import numpy as np
from scipy.spatial.transform import Rotation


def _metropolis_accept(old_energy, new_energy, temperature, rng):
    """Accept downhill moves and thermally sample uphill local minima."""
    if not np.isfinite(new_energy):
        return False
    if new_energy < old_energy:
        return True
    return rng.random() < math.exp((old_energy - new_energy) / temperature)


class MonteCarloMixin:
    """Vina-inspired iterated local search with Metropolis acceptance."""

    def _mutate_conf(self, x, rng, amplitude):
        """Perturb one entity, following Vina's mutate_conf move distribution."""
        candidate = x.copy()
        entity = rng.integers(2 + len(self.rotors))
        if entity < 2:
            while True:
                displacement = rng.uniform(-1.0, 1.0, 3)
                if np.dot(displacement, displacement) < 1:
                    break
            if entity == 0:
                candidate[:3] += amplitude * displacement / self.box_size
            else:
                coords = self.lig.mutate(x)
                radius = np.sqrt(np.mean(np.sum((coords - coords.mean(axis=0)) ** 2, axis=1)))
                if radius > 1e-12:
                    increment = Rotation.from_rotvec(amplitude / radius * displacement)
                    candidate[3:6] = (increment * Rotation.from_euler("xyz", x[3:6])).as_euler("xyz")
        else:
            candidate[entity + 4] = rng.uniform(-np.pi, np.pi)
        return candidate

    def dock(
        self,
        exhaustiveness=8,
        steps=250,
        num_modes=9,
        seed=None,
        maxiter=50,
        temperature=1.2,
        mutation_amplitude=2.0,
        min_rmsd=1.0,
        energy_range=3.0,
    ):
        """Search poses with iterated local search and Metropolis acceptance.

        Parameters
        ----------
        exhaustiveness : int, optional
            Number of independent randomized Monte Carlo trajectories, run
            serially. This has the same meaning as Vina's independent runs.
        steps : int, optional
            Number of mutation/local-minimization proposals per trajectory.
            This explicit budget replaces Vina's much larger step heuristic.
        num_modes : int, optional
            Maximum number of distinct poses to return.
        seed : int or None, optional
            Seed for an isolated NumPy random generator.
        maxiter : int, optional
            Maximum L-BFGS-B iterations per local minimization.
        temperature : float, optional
            Positive Metropolis energy scale; Vina's default is 1.2.
        mutation_amplitude : float, optional
            Positive displacement scale in Angstroms; Vina's default is 2.
            Angular displacement is scaled by the ligand's gyration radius.
        min_rmsd : float, optional
            Minimum heavy-atom RMSD between returned poses, in Angstroms.
            Atom order is used directly, without alignment or symmetry matching.
        energy_range : float, optional
            Maximum total pair-energy difference from the best feasible pose.

        Returns
        -------
        poses : list of oddt.toolkit.Molecule
            Independent ligand copies, ordered by total pair energy. Hydrogens
            follow their parent fragments. Data fields are ``vina_affinity``
            (torsion-corrected inter energy), ``vina_inter``, ``vina_intra`` and
            ``vina_total``. Inputs are unchanged. Up to ``num_modes`` poses are
            returned; all heavy atoms lie inside the box to within 0.0001 A.

        Notes
        -----
        Each trajectory starts with random position, uniform orientation and
        random torsions. Each step mutates exactly one entity: translation,
        orientation, or a single torsion. Proposals undergo L-BFGS-B with Vina's
        positive-energy soft cap of 10, followed by Metropolis acceptance.
        Accepted promising minima receive uncapped refinement before saving.
        As in Vina, the refined coordinates and energy continue the trajectory.
        Subsequent proposals use the hunt cap again; saved/output poses use
        uncapped energies rather than Vina's finite high refinement cap.
        ``results_`` contains selected OptimizeResults; ``history_`` records the
        best uncapped restrained energy after every proposal. ``search_stats_``
        reports accepted moves, uphill moves and refinements per trajectory.
        Scoring, box restraints and SciPy's L-BFGS-B remain independent of Vina.
        """
        bounds = self._search_bounds()
        for name, value, minimum in (
            ("exhaustiveness", exhaustiveness, 1),
            ("steps", steps, 1),
            ("num_modes", num_modes, 1),
            ("maxiter", maxiter, 1),
        ):
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError("%s must be an integer >= %d" % (name, minimum))
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if not np.isfinite(mutation_amplitude) or mutation_amplitude <= 0:
            raise ValueError("mutation_amplitude must be positive and finite")
        if not np.isfinite(min_rmsd) or min_rmsd < 0 or not np.isfinite(energy_range) or energy_range < 0:
            raise ValueError("min_rmsd and energy_range must be finite and nonnegative")
        self.results_ = []
        self.history_ = []
        self.search_stats_ = []
        archive = []
        best_energy = np.inf
        # Independent streams keep a trajectory reproducible when increasing
        # exhaustiveness, without modifying NumPy's process-global RNG.
        for stream in np.random.SeedSequence(seed).spawn(exhaustiveness):
            rng = np.random.default_rng(stream)
            current = self._random_conformation(bounds, rng)
            current_energy = np.inf
            minima = []
            chain_best = np.inf
            stats = {
                "proposals": steps,
                "accepted": 0,
                "uphill_accepted": 0,
                "refinements": 0,
                "local_minimizations": 0,
                "objective_evaluations": 0,
            }
            for step in range(steps):
                proposal = self._mutate_conf(current, rng, mutation_amplitude)
                candidate = self.optimize(proposal, maxiter=maxiter, energy_cap=10.0)
                stats["local_minimizations"] += 1
                stats["objective_evaluations"] += candidate.nfev
                if _metropolis_accept(current_energy, candidate.fun, temperature, rng):
                    stats["accepted"] += 1
                    stats["uphill_accepted"] += int(step > 0 and candidate.fun > current_energy)
                    current = candidate.x.copy()
                    current_energy = candidate.fun
                    # As in Vina, refine accepted promising states and save a
                    # bounded set of distinct minima, not rejected proposals.
                    if current_energy < chain_best or len(minima) < num_modes:
                        # Use the infinite-cap limit of the hunting evaluator so
                        # refinement preserves its floating-point reduction order.
                        refined = self.optimize(current, maxiter=maxiter, energy_cap=np.inf)
                        stats["refinements"] += 1
                        stats["local_minimizations"] += 1
                        stats["objective_evaluations"] += refined.nfev
                        current = refined.x.copy()
                        current_energy = refined.fun
                        chain_best = min(chain_best, refined.fun)
                        best_energy = min(best_energy, refined.fun)
                        self._add_minimum(minima, refined, min_rmsd, num_modes)
                self.history_.append(best_energy)
            stats["best_energy"] = chain_best
            self.search_stats_.append(stats)
            archive.extend(minima)

        return self._finish_docking(archive, num_modes, min_rmsd, energy_range)


class GeneticAlgorithmMixin:
    """Elitist genetic search with local refinement, crossover and mutation."""

    def dock(
        self,
        population_size=24,
        generations=10,
        num_modes=9,
        seed=None,
        maxiter=50,
        mutation_rate=0.2,
        min_rmsd=1.0,
        energy_range=3.0,
    ):
        """Search randomized poses with tournament selection and L-BFGS-B.

        ``population_size`` includes elites and must be at least four.
        Each generation preserves the best eighth of the population, uses
        three-way tournaments and per-gene crossover, and introduces 10%
        random immigrants. Mutations use 15% of the box width for translation
        and 0.6 radians for angles; ``mutation_rate`` is the per-gene probability.
        At least one gene is mutated when the rate is positive. Angles wrap
        along the shortest arc during crossover. ``generations=0`` evaluates
        only the initial random population.

        ``history_`` records the best restrained objective initially and after
        each generation. ``search_stats_`` records the actual number of local
        minimizations and objective evaluations. Pose selection and scoring
        are supplied by the scoring base, identically to Monte Carlo search.
        """
        bounds = self._search_bounds()
        for name, value, minimum in (
            ("population_size", population_size, 4),
            ("generations", generations, 0),
            ("num_modes", num_modes, 1),
            ("maxiter", maxiter, 1),
        ):
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError("%s must be an integer >= %d" % (name, minimum))
        if not np.isfinite(mutation_rate) or not 0 <= mutation_rate <= 1:
            raise ValueError("mutation_rate must be finite and between zero and one")
        if not np.isfinite(min_rmsd) or min_rmsd < 0 or not np.isfinite(energy_range) or energy_range < 0:
            raise ValueError("min_rmsd and energy_range must be finite and nonnegative")
        self.results_ = []
        self.search_stats_ = []
        rng = np.random.default_rng(seed)
        dimensions = len(self.lig.x0)
        stats = {"local_minimizations": 0, "objective_evaluations": 0}

        def refine(x):
            result = self.optimize(x, maxiter=maxiter)
            stats["local_minimizations"] += 1
            stats["objective_evaluations"] += result.nfev
            return result

        def tournament():
            indices = rng.integers(len(population), size=3)
            return min((population[index] for index in indices), key=lambda result: result.fun).x

        population = [refine(self._random_conformation(bounds, rng)) for _ in range(population_size)]
        archive = list(population)
        self.history_ = [min(result.fun for result in population)]
        elite_count = max(1, population_size // 8)
        scales = np.r_[0.15 * (bounds[1] - bounds[0]), np.full(dimensions - 3, 0.6)]
        for generation in range(generations):
            offspring = sorted(population, key=lambda result: result.fun)[:elite_count]
            while len(offspring) < population_size:
                if rng.random() < 0.1:
                    child = self._random_conformation(bounds, rng)
                else:
                    first, second = tournament(), tournament()
                    difference = second - first
                    difference[3:] = (difference[3:] + np.pi) % (2 * np.pi) - np.pi
                    child = first + rng.random(dimensions) * difference
                    mutated = rng.random(dimensions) < mutation_rate
                    if mutation_rate > 0:
                        mutated[rng.integers(dimensions)] = True
                    child += mutated * rng.normal(size=dimensions) * scales
                    child[:3] = np.clip(child[:3], bounds[0], bounds[1])
                    child[3:] = (child[3:] + np.pi) % (2 * np.pi) - np.pi
                result = refine(child)
                offspring.append(result)
                archive.append(result)
            population = offspring
            self.history_.append(min(result.fun for result in population))
        stats["best_energy"] = self.history_[-1]
        self.search_stats_.append(stats)
        return self._finish_docking(archive, num_modes, min_rmsd, energy_range)
