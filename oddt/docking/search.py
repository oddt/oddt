"""Global pose-search mixins, independent of the pair-scoring implementation.

A scoring base supplies ``_search_bounds``, ``_random_conformation``,
``optimize`` (returning x, fun, coords and nfev), ``_add_minimum`` and
``_finish_docking``. Pose geometry is exposed as ``lig.x0``, ``lig.mutate(x)``,
``rotors`` and ``box_size``. All searches use the same local minimizer and
output selection supplied by that base. ``DockingBase`` implements this
interface for models supplying unrestrained energy and Cartesian gradients.
"""

from contextlib import nullcontext
import math

import numpy as np
from scipy.spatial.transform import Rotation


def _progress_bar(enabled, **kwargs):
    """Import the optional display dependency only when requested."""
    if not enabled:
        return nullcontext()
    try:
        from tqdm.auto import tqdm
    except ImportError as error:
        raise ImportError("Progress bars require tqdm; install oddt[progress] or tqdm") from error
    return tqdm(dynamic_ncols=True, **kwargs)


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
        temperature=None,
        mutation_amplitude=2.0,
        min_rmsd=1.0,
        energy_range=None,
        progress=False,
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
        temperature : float or None, optional
            Positive Metropolis energy scale. None uses the scoring base's
            ``default_mc_temperature`` (1.2 for Vina).
        mutation_amplitude : float, optional
            Positive displacement scale in Angstroms; Vina's default is 2.
            Angular displacement is scaled by the ligand's gyration radius.
        min_rmsd : float, optional
            Minimum heavy-atom RMSD between returned poses, in Angstroms.
            Atom order is used directly, without alignment or symmetry matching.
        energy_range : float or None, optional
            Maximum unrestrained energy difference from the best feasible pose.
            None uses ``default_energy_range`` (3.0 for Vina).
        progress : bool, optional
            Show a tqdm progress bar over all trajectories' proposals.
            Disabled by default; requires the optional tqdm dependency.

        Returns
        -------
        poses : list of oddt.toolkit.Molecule
            Independent ligand copies, ordered by unrestrained energy. Hydrogens
            follow their parent fragments. For Vina, data fields are ``vina_affinity``
            (torsion-corrected inter energy), ``vina_inter``, ``vina_intra`` and
            ``vina_total``. Inputs are unchanged. Up to ``num_modes`` poses are
            returned; all heavy atoms lie inside the box to within 0.0001 A.

        Notes
        -----
        Each trajectory starts with random position, uniform orientation and
        random torsions. Each step mutates exactly one entity: translation,
        orientation, or a single torsion. With Vina scoring, proposals undergo
        L-BFGS-B with a positive-energy soft cap of 10, then Metropolis acceptance.
        Accepted promising minima receive uncapped refinement before saving.
        As in Vina, the refined coordinates and energy continue the trajectory.
        Subsequent proposals use the hunt cap again; saved/output poses use
        uncapped energies rather than Vina's finite high refinement cap.
        ``results_`` contains selected OptimizeResults; ``history_`` records the
        best uncapped restrained energy after every proposal. ``search_stats_``
        reports accepted moves, uphill moves and refinements per trajectory.
        Scoring, box restraints and SciPy's L-BFGS-B remain independent of Vina.
        Other bases select caps with ``mc_hunt_energy_cap`` and
        ``mc_refinement_energy_cap``; DockingBase defaults to uncapped searches.
        """
        bounds = self._search_bounds()
        temperature = getattr(self, "default_mc_temperature", 1.2) if temperature is None else temperature
        energy_range = getattr(self, "default_energy_range", 3.0) if energy_range is None else energy_range
        hunt_cap = getattr(self, "mc_hunt_energy_cap", 10.0)
        refinement_cap = getattr(self, "mc_refinement_energy_cap", np.inf)
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
        with _progress_bar(progress, total=exhaustiveness * steps, desc="MC docking", unit="proposal") as bar:
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
                    candidate = self.optimize(proposal, maxiter=maxiter, energy_cap=hunt_cap)
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
                            # Vina uses the infinite-cap hunting evaluator to
                            # preserve reduction order; other models choose their cap.
                            refined = self.optimize(current, maxiter=maxiter, energy_cap=refinement_cap)
                            stats["refinements"] += 1
                            stats["local_minimizations"] += 1
                            stats["objective_evaluations"] += refined.nfev
                            current = refined.x.copy()
                            current_energy = refined.fun
                            chain_best = min(chain_best, refined.fun)
                            best_energy = min(best_energy, refined.fun)
                            self._add_minimum(minima, refined, min_rmsd, num_modes)
                    self.history_.append(best_energy)
                    if bar is not None:
                        bar.update()
                stats["best_energy"] = chain_best
                self.search_stats_.append(stats)
                archive.extend(minima)

        return self._finish_docking(archive, num_modes, min_rmsd, energy_range)


class GeneticAlgorithmMixin:
    """Elitist genetic search with local refinement, crossover and mutation."""

    def dock(
        self,
        population_size=100,
        generations=10,
        num_modes=9,
        seed=None,
        maxiter=50,
        mutation_rate=0.2,
        min_rmsd=1.0,
        energy_range=None,
        progress=False,
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
        ``progress=True`` displays a tqdm bar counting local minimizations,
        including the initial population and newly generated offspring.
        Progress is disabled by default and requires the optional tqdm dependency.
        ``energy_range=None`` uses the scoring base's default energy window.
        """
        bounds = self._search_bounds()
        energy_range = getattr(self, "default_energy_range", 3.0) if energy_range is None else energy_range
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
            if bar is not None:
                bar.update()
            return result

        def tournament():
            indices = rng.integers(len(population), size=3)
            return min((population[index] for index in indices), key=lambda result: result.fun).x

        elite_count = max(1, population_size // 8)
        total = population_size + generations * (population_size - elite_count)
        with _progress_bar(progress, total=total, desc="GA docking", unit="minimization") as bar:
            population = [refine(self._random_conformation(bounds, rng)) for _ in range(population_size)]
            archive = list(population)
            self.history_ = [min(result.fun for result in population)]
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


def _acor_weights(size, q):
    """Normalized Gaussian rank weights, Socha & Dorigo (2008), Eqs. 7--8."""
    with np.errstate(over="ignore", under="ignore"):
        scaled = np.arange(size, dtype=float) / (q * size)
        weights = np.exp(-0.5 * scaled**2)
    return weights / weights.sum()


def _acor_widths(poses, selected, xi, bounds, min_sigma):
    """Eq. 9 widths, with shortest torsion arcs and isotropic SO(3) kernels."""
    center = poses[selected]
    delta = poses - center
    delta[:, 6:] = (delta[:, 6:] + np.pi) % (2 * np.pi) - np.pi
    widths = xi * np.abs(delta).sum(axis=0) / (len(poses) - 1)
    rotations = Rotation.from_euler("xyz", poses[:, 3:6])
    angles = (rotations * rotations[selected].inv()).magnitude()
    # An isotropic three-dimensional tangent Gaussian has RMS angle sqrt(3)*sigma.
    widths[3:6] = xi * angles.sum() / ((len(poses) - 1) * np.sqrt(3))
    scales = np.r_[bounds[1] - bounds[0], np.full(3, 2 * np.pi / np.sqrt(3)), np.full(len(center) - 6, 2 * np.pi)]
    return np.maximum(widths, min_sigma * scales)


def _acor_sample(poses, weights, xi, bounds, min_sigma, rng):
    """Choose one archive kernel for the entire pose and construct one ant."""
    selected = rng.choice(len(poses), p=weights)
    center = poses[selected]
    sigma = _acor_widths(poses, selected, xi, bounds, min_sigma)
    displacement = rng.normal(size=len(center)) * sigma
    candidate = center + displacement
    width = bounds[1] - bounds[0]
    reflected = (candidate[:3] - bounds[0]) % (2 * width)
    candidate[:3] = bounds[0] + np.minimum(reflected, 2 * width - reflected)
    candidate[3:6] = (Rotation.from_rotvec(displacement[3:6]) * Rotation.from_euler("xyz", center[3:6])).as_euler("xyz")
    candidate[6:] = (candidate[6:] + np.pi) % (2 * np.pi) - np.pi
    return candidate


class AntColonyMixin:
    """Continuous ant-colony pose search with a ranked Gaussian-kernel archive.

    Independently implements the archive/rank/width equations of Socha & Dorigo
    (2008), doi:10.1016/j.ejor.2006.06.046. Rotation-space kernels, wrapped
    torsions, reflected bounds, exploration floors, random immigrants and local
    minimization adapt ACO_R to docking. This is not the PLANTS search algorithm.
    """

    def dock(
        self,
        ants=16,
        iterations=20,
        archive_size=32,
        q=0.2,
        xi=0.85,
        min_sigma=0.02,
        random_fraction=0.1,
        num_modes=9,
        seed=None,
        maxiter=50,
        min_rmsd=1.0,
        energy_range=None,
        progress=False,
    ):
        """Sample and refine ants from a continuous pheromone distribution.

        Parameters
        ----------
        ants : int, optional
            Poses sampled per iteration, at least one. Default is 16.
        iterations : int, optional
            Archive updates, at least zero. Default is 20; zero evaluates only
            the initial randomized archive.
        archive_size : int, optional
            Ranked pheromone memory size, at least two. Default is 32.
        q : float, optional
            Positive Gaussian rank width, default 0.2. Smaller values favor
            the best archive members more strongly.
        xi : float, optional
            Positive multiplier of mean archive distances, default 0.85.
            Larger values slow concentration of the sampling distribution.
        min_sigma : float, optional
            Minimum kernel width as a fraction of the translation box width
            or full angular period, in [0, 1]. Default is 0.02. For orientation
            this sets the RMS tangent angle rather than a single-axis width.
        random_fraction : float, optional
            Probability of an independently randomized ant, in [0, 1].
            Default is 0.1. Zero disables random immigrants.
        num_modes : int, optional
            Maximum number of returned pose clusters, at least one.
        seed : int or None, optional
            Seed for an isolated NumPy generator; global RNG state is preserved.
        maxiter : int, optional
            Maximum uncapped L-BFGS-B iterations for every pose, at least one.
        min_rmsd : float, optional
            Nonnegative heavy-atom RMSD for output clustering, in Angstroms.
        energy_range : float or None, optional
            Nonnegative total-energy window in the scorer's units. None uses
            the scorer's default. The box restraint is excluded from filtering.
        progress : bool, optional
            Show optional tqdm progress over all local minimizations.

        Returns
        -------
        poses : list of oddt.toolkit.Molecule
            Feasible pose copies, ordered by the scorer's unrestrained energy.
            Inputs and their score metadata are preserved.

        Notes
        -----
        Starts with archive_size independent random poses, with uniform SO(3)
        orientation. Each ant chooses one rank-weighted archive member for all
        pose variables. Translation uses Gaussian kernels reflected at box
        bounds; orientation uses an isotropic rotation-vector increment; torsion
        widths and samples use shortest periodic arcs. The complete colony is
        sampled from the previous iteration's frozen archive. After uncapped
        local refinement, the best archive_size finite results survive.

        Exploration floors and immigrants counter archive collapse. The ranked
        model may contain similar poses; a separate RMSD-clustered pool retains
        up to max(archive_size, num_modes) feasible output candidates. Both pools
        are bounded. The default budget is 352 local minimizations, run serially.

        ``history_`` records the best restrained objective initially and after each
        iteration. ``search_stats_`` records actual work and archive replacements.
        ``pheromone_archive_`` and ``pheromone_weights_`` expose the final ranked model.
        No reference coordinates or reference RMSD guide the search.
        """
        bounds = self._search_bounds()
        energy_range = getattr(self, "default_energy_range", 3.0) if energy_range is None else energy_range
        for name, value, minimum in (
            ("ants", ants, 1),
            ("iterations", iterations, 0),
            ("archive_size", archive_size, 2),
            ("num_modes", num_modes, 1),
            ("maxiter", maxiter, 1),
        ):
            if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < minimum:
                raise ValueError("%s must be an integer >= %d" % (name, minimum))
        for name, value in (("q", q), ("xi", xi)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % name)
        for name, value in (("min_sigma", min_sigma), ("random_fraction", random_fraction)):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("%s must be finite and between zero and one" % name)
        if not np.isfinite(min_rmsd) or min_rmsd < 0 or not np.isfinite(energy_range) or energy_range < 0:
            raise ValueError("min_rmsd and energy_range must be finite and nonnegative")
        rng = np.random.default_rng(seed)
        self.results_, self.history_ = [], []
        self.pheromone_archive_, self.pheromone_weights_ = [], np.empty(0)
        stats = dict.fromkeys(
            (
                "local_minimizations",
                "objective_evaluations",
                "kernel_samples",
                "random_samples",
                "archive_replacements",
            ),
            0,
        )
        self.search_stats_ = [stats]
        candidates = []

        def refine(x):
            result = self.optimize(x, maxiter=maxiter)
            stats["local_minimizations"] += 1
            stats["objective_evaluations"] += int(result.nfev)
            self._add_minimum(candidates, result, min_rmsd, max(archive_size, num_modes))
            if bar is not None:
                bar.update()
            return result

        total = archive_size + ants * iterations
        with _progress_bar(progress, total=total, desc="ACO docking", unit="minimization") as bar:
            initial = [refine(self._random_conformation(bounds, rng)) for _ in range(archive_size)]
            memory = sorted((result for result in initial if np.isfinite(result.fun)), key=lambda result: result.fun)
            if len(memory) < 2:
                raise RuntimeError(
                    "ACO requires at least two finite initial poses; check scoring or increase archive_size"
                )
            self.history_.append(float(memory[0].fun))
            for iteration in range(iterations):
                poses = np.array([result.x for result in memory])
                weights = _acor_weights(len(memory), q)
                colony = []
                for ant in range(ants):
                    if rng.random() < random_fraction:
                        x = self._random_conformation(bounds, rng)
                        stats["random_samples"] += 1
                    else:
                        x = _acor_sample(poses, weights, xi, bounds, min_sigma, rng)
                        stats["kernel_samples"] += 1
                    result = refine(x)
                    if np.isfinite(result.fun):
                        colony.append(result)
                combined = [(result, False) for result in memory] + [(result, True) for result in colony]
                combined.sort(key=lambda entry: entry[0].fun)
                retained = combined[:archive_size]
                stats["archive_replacements"] += sum(new for _, new in retained)
                memory = [result for result, _ in retained]
                self.history_.append(float(memory[0].fun))
            self.pheromone_archive_ = memory
            self.pheromone_weights_ = _acor_weights(len(memory), q)
        stats["best_energy"] = self.history_[-1]
        return self._finish_docking(candidates, num_modes, min_rmsd, energy_range)


def _de_trial(current, pbest, first, second, scale, crossover, bounds, rng):
    """Current-to-pbest/1 on translations, SO(3), and periodic torsions.

    Orientation crosses over as one block. Relative rotations are expressed
    in the common world frame, so Euler branch cuts do not create large moves.
    """
    mutant = current.copy()
    mutant[:3] += scale * (pbest[:3] - current[:3] + first[:3] - second[:3])
    width = bounds[1] - bounds[0]
    reflected = (mutant[:3] - bounds[0]) % (2 * width)
    mutant[:3] = bounds[0] + np.minimum(reflected, 2 * width - reflected)
    rotations = Rotation.from_euler("xyz", np.array([current[3:6], pbest[3:6], first[3:6], second[3:6]]))
    increment = scale * (
        (rotations[1] * rotations[0].inv()).as_rotvec() + (rotations[2] * rotations[3].inv()).as_rotvec()
    )
    mutant[3:6] = (Rotation.from_rotvec(increment) * rotations[0]).as_euler("xyz")

    def wrap(value):
        return (value + np.pi) % (2 * np.pi) - np.pi

    mutant[6:] = wrap(current[6:] + scale * (wrap(pbest[6:] - current[6:]) + wrap(first[6:] - second[6:])))
    blocks = rng.random(len(current) - 2) < crossover
    blocks[rng.integers(len(blocks))] = True
    mask = np.r_[blocks[:3], np.repeat(blocks[3], 3), blocks[4:]]
    return np.where(mask, mutant, current)


class DifferentialEvolutionMixin:
    """Adaptive differential evolution with local search and diversity restarts."""

    def dock(
        self,
        population_size=32,
        generations=20,
        num_modes=9,
        seed=None,
        maxiter=50,
        local_maxiter=10,
        pbest_fraction=0.2,
        history_size=6,
        diversity_rmsd=1.0,
        archive_size=64,
        stagnation_generations=5,
        restart_fraction=0.25,
        min_rmsd=1.0,
        energy_range=None,
        progress=False,
    ):
        """Search with adaptive current-to-pbest/1 differential evolution.

        ``population_size >= 4`` poses evolve for ``generations >= 0`` rounds.
        Translation differences reflect at box bounds; orientation differences
        use rotation vectors and torsions use shortest arcs. Crossover treats
        orientation as a single block and always selects at least one block.
        ``pbest_fraction`` selects the best fraction of parents as attractors.

        Mutation scale and crossover probability are drawn from success-history
        memories of length ``history_size`` (initially 0.5). Energy improvements
        weight a Lehmer mean for scale and an arithmetic mean for crossover.
        A bounded donor archive retains displaced parents. This is a hybrid
        inspired by SHADE, rather than an implementation of canonical SHADE.

        Every proposal receives ``min(local_maxiter, maxiter)`` L-BFGS-B
        iterations. Improving proposals and initial/restarted members receive
        up to ``maxiter`` further iterations. A proposal within ``diversity_rmsd``
        Angstroms of an existing member competes with that nearest member;
        otherwise it competes with its parent. At each generation's end, up to
        ``restart_fraction`` of members that are crowded or have not improved
        for ``stagnation_generations`` rounds are randomized. The best member
        is protected. Set ``restart_fraction=0`` to disable restarts.

        A separate feasible pose archive holds at most ``max(archive_size,
        num_modes)`` RMSD clusters. Its remaining short-search poses receive
        full refinement before common output selection. All local searches use
        the same uncapped scoring objective. Diversity uses direct heavy-atom
        RMSD without alignment, independently of the benchmark's symmetry RMSD.

        ``history_`` contains the best objective initially, after each round,
        and after final refinement. ``search_stats_`` records search effort,
        successful replacements and restarts. ``parameter_memory_`` exposes
        the final scale/crossover memories. ``progress=True`` enables optional
        tqdm output counting initialization, generations and final refinement.
        ``energy_range=None`` uses the scoring base's default energy window.
        """
        bounds = self._search_bounds()
        energy_range = getattr(self, "default_energy_range", 3.0) if energy_range is None else energy_range
        for name, value, minimum in (
            ("population_size", population_size, 4),
            ("generations", generations, 0),
            ("num_modes", num_modes, 1),
            ("maxiter", maxiter, 1),
            ("local_maxiter", local_maxiter, 1),
            ("history_size", history_size, 1),
            ("archive_size", archive_size, 1),
            ("stagnation_generations", stagnation_generations, 1),
        ):
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError("%s must be an integer >= %d" % (name, minimum))
        for name, value, positive in (
            ("pbest_fraction", pbest_fraction, True),
            ("restart_fraction", restart_fraction, False),
        ):
            if not np.isfinite(value) or not (0 < value <= 1 if positive else 0 <= value <= 1):
                raise ValueError("%s must be finite and %s" % (name, "in (0, 1]" if positive else "in [0, 1]"))
        for name, value in (("diversity_rmsd", diversity_rmsd), ("min_rmsd", min_rmsd), ("energy_range", energy_range)):
            if not np.isfinite(value) or value < 0:
                raise ValueError("%s must be finite and nonnegative" % name)

        rng = np.random.default_rng(seed)
        self.results_ = []
        self.history_ = []
        stats = dict.fromkeys(
            (
                "proposals",
                "accepted",
                "short_minimizations",
                "refinements",
                "restarts",
                "local_minimizations",
                "objective_evaluations",
            ),
            0,
        )
        self.search_stats_ = [stats]
        memory_f = np.full(history_size, 0.5)
        memory_cr = np.full(history_size, 0.5)
        memory_slot = 0
        donors, archive = [], []
        ages = np.zeros(population_size, dtype=int)
        best_energy = np.inf
        capacity = max(archive_size, num_modes)

        def minimize(x, full=False):
            nonlocal best_energy
            result = self.optimize(x, maxiter=maxiter if full else min(local_maxiter, maxiter))
            result.de_fully_refined = full
            stats["refinements" if full else "short_minimizations"] += 1
            stats["local_minimizations"] += 1
            stats["objective_evaluations"] += int(result.nfev)
            if np.isfinite(result.fun):
                best_energy = min(best_energy, float(result.fun))
                self._add_minimum(archive, result, min_rmsd, capacity)
            return result

        def random_member():
            short = minimize(self._random_conformation(bounds, rng))
            return minimize(short.x, full=True)

        def distances(candidate, members):
            return np.sqrt(
                np.mean(
                    np.sum((np.array([member.coords for member in members]) - candidate.coords) ** 2, axis=-1), axis=-1
                )
            )

        with _progress_bar(progress, total=generations + 2, desc="DE docking", unit="stage") as bar:
            population = [random_member() for _ in range(population_size)]
            self.history_.append(best_energy)
            if bar is not None:
                bar.update()
            for generation in range(generations):
                # Freeze population parents for a round; replacements and
                # niche competition use the live population.
                parents = list(population)
                ranked = sorted(range(population_size), key=lambda i: parents[i].fun)
                pbest_count = max(2, int(np.ceil(pbest_fraction * population_size)))
                successes = []
                ages += 1
                for i, parent in enumerate(parents):
                    slot = rng.integers(history_size)
                    scale = memory_f[slot] + 0.1 * rng.standard_cauchy()
                    while scale <= 0:
                        scale = memory_f[slot] + 0.1 * rng.standard_cauchy()
                    scale = min(float(scale), 1.0)
                    crossover = float(np.clip(rng.normal(memory_cr[slot], 0.1), 0, 1))
                    pbest = parents[rng.choice(ranked[:pbest_count])]
                    first_index = rng.choice([j for j in range(population_size) if j != i])
                    pool = [member.x for j, member in enumerate(parents) if j not in (i, first_index)] + donors
                    second = pool[rng.integers(len(pool))]
                    trial = _de_trial(parent.x, pbest.x, parents[first_index].x, second, scale, crossover, bounds, rng)
                    candidate = minimize(trial)
                    stats["proposals"] += 1
                    if not np.isfinite(candidate.fun):
                        continue
                    rmsds = distances(candidate, population)
                    nearest = int(np.argmin(rmsds))
                    competitor = nearest if rmsds[nearest] < diversity_rmsd else i
                    if candidate.fun >= population[competitor].fun:
                        continue
                    candidate = minimize(candidate.x, full=True)
                    # Refinement may move into a different niche.
                    rmsds = distances(candidate, population)
                    nearest = int(np.argmin(rmsds))
                    competitor = nearest if rmsds[nearest] < diversity_rmsd else i
                    improvement = population[competitor].fun - candidate.fun
                    if not np.isfinite(candidate.fun) or improvement <= 0:
                        continue
                    donors.append(population[competitor].x.copy())
                    if len(donors) > population_size:
                        donors.pop(int(rng.integers(len(donors))))
                    population[competitor] = candidate
                    ages[competitor] = 0
                    stats["accepted"] += 1
                    if np.isfinite(improvement):
                        successes.append((scale, crossover, improvement))
                if successes:
                    sf, scr, weights = np.array(successes).T
                    weights /= weights.sum()
                    memory_f[memory_slot] = np.sum(weights * sf**2) / np.sum(weights * sf)
                    memory_cr[memory_slot] = np.sum(weights * scr)
                    memory_slot = (memory_slot + 1) % history_size

                ranked = sorted(range(population_size), key=lambda i: population[i].fun)
                representatives, crowded = [], set()
                for i in ranked:
                    if representatives and np.min(distances(population[i], representatives)) < diversity_rmsd:
                        crowded.add(i)
                    else:
                        representatives.append(population[i])
                restartable = [i for i in reversed(ranked[1:]) if i in crowded or ages[i] >= stagnation_generations]
                for i in restartable[: int(np.ceil(restart_fraction * population_size))]:
                    population[i] = random_member()
                    ages[i] = 0
                    stats["restarts"] += 1
                self.history_.append(best_energy)
                if bar is not None:
                    bar.update()

            final_archive = []
            for result in list(archive):
                if not result.de_fully_refined:
                    result = minimize(result.x, full=True)
                self._add_minimum(final_archive, result, min_rmsd, capacity)
            self.history_.append(best_energy)
            if bar is not None:
                bar.update()
        self.parameter_memory_ = {"scale": memory_f.copy(), "crossover": memory_cr.copy()}
        stats["best_energy"] = best_energy
        return self._finish_docking(final_archive, num_modes, min_rmsd, energy_range)
