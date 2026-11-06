"""ODDT's internal docking/scoring engines"""

import numpy as np
import math
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from oddt.spatial import distance, rotate


def get_children(molecule, mother, restricted):
    atoms = np.zeros(len(molecule.atoms), dtype=bool)
    atoms[mother] = True
    d = 1  # Pass first
    prev = 0
    while d > 0:
        atoms[
            [
                n.idx0
                for i in np.nonzero(atoms)[0]
                if i != restricted
                for n in molecule.atoms[i].neighbors
                if n.idx0 != restricted
            ]
        ] = True
        d = atoms.sum() - prev
        prev = atoms.sum()
    return atoms


def get_close_neighbors(molecule, a_idx, num_bonds=1):
    atoms = np.zeros(len(molecule.atoms), dtype=bool)
    atoms[a_idx] = True
    for i in range(num_bonds):
        atoms[[n.idx0 for i in np.nonzero(atoms)[0] for n in molecule.atoms[i].neighbors]] = True
    return atoms


def change_dihedral(coords, a1, a2, a3, a4, target_angle, rot_mask):
    sin = math.sin(target_angle)
    cos = math.cos(target_angle)
    t = 1 - cos
    v0 = coords[a2] - coords[a3]
    v = (v0) / np.linalg.norm(v0)
    rot_matrix = np.array(
        [
            [t * v[0] * v[0] + cos, t * v[0] * v[1] + sin * v[2], t * v[0] * v[2] - sin * v[1]],
            [t * v[0] * v[1] - sin * v[2], t * v[1] * v[1] + cos, t * v[1] * v[2] + sin * v[0]],
            [t * v[0] * v[2] + sin * v[1], t * v[1] * v[2] - sin * v[0], t * v[2] * v[2] + cos],
        ]
    )

    centroid = coords[a3]
    coords = coords - centroid
    # Old and slower version
    # coords[rot_mask] = (coords[rot_mask,np.newaxis,:] * rot_matrix).sum(axis=2)
    coords[rot_mask] = np.einsum("ij,jk->ik", coords[rot_mask], rot_matrix)
    return coords + centroid


def num_rotors_pdbqt(lig):
    i = 0
    for atom in lig.atoms:
        if atom.atomicnum == 1:
            continue
        num_local_rot = sum(int(b.isrotor) for b in atom.bonds)
        if num_local_rot == 0:
            pass
        elif num_local_rot == 1:
            i += 0.5
        elif num_local_rot == 2:
            i += 1.0
        elif num_local_rot >= 3:
            i += 0.5
    return i


def _is_search_rotor(bond):
    """Keep conjugated carbonyl-heteroatom bonds rigid during pose search."""
    if not bond.isrotor or bond.order != 1:
        return False
    first, second = bond.atoms
    for carbon, heteroatom in ((first, second), (second, first)):
        if carbon.atomicnum == 6 and heteroatom.atomicnum in (7, 8, 16):
            for neighbor_bond in carbon.bonds:
                if neighbor_bond.order == 2 and any(atom.atomicnum in (7, 8, 16) for atom in neighbor_bond.atoms):
                    return False
    return True


def _metropolis_accept(old_energy, new_energy, temperature, rng):
    """Accept downhill moves and thermally sample uphill local minima."""
    if not np.isfinite(new_energy):
        return False
    if new_energy < old_energy:
        return True
    return rng.random() < math.exp((old_energy - new_energy) / temperature)


class vina_docking(object):
    """Vina-inspired scoring and Monte Carlo docking against a rigid receptor.

    Parameters
    ----------
    rec : oddt.toolkit.Molecule or None
        Receptor with coordinates and protein atom typing.
    lig : oddt.toolkit.Molecule or None, optional
        Ligand with an existing three-dimensional conformation. Ring geometry
        and conjugated carbonyl-heteroatom bonds are retained; translation,
        orientation and other rotatable bonds are searched.
    box : array-like, shape (2, 3), optional
        Lower and upper Cartesian corners of the search box, in Angstroms.
        Required for docking and local optimization, but not for scoring.
    box_size : float, optional
        Angstroms per unit of the three translation genes.
    weights : array-like, shape (6,), optional
        Five Vina pair-potential weights and the rotatable-bond penalty.

    Notes
    -----
    This is an independent search implementation, not AutoDock Vina. It uses
    ODDT atom typing and direct pair scoring rather than Vina affinity grids.
    """

    def __init__(self, rec, lig=None, box=None, box_size=1.0, weights=None):
        if not np.isfinite(box_size) or box_size <= 0:
            raise ValueError("box_size must be positive and finite")
        self.box_size = float(box_size)
        self.rec_dict = None
        self.lig_dict = None
        self.set_protein(rec)
        if lig is not None:
            self.set_ligand(lig)
        self.set_box(box)
        self.weights = np.asarray(
            (-0.0356, -0.00516, 0.840, -0.0351, -0.587, 0.0585) if weights is None else weights, dtype=float
        )
        if self.weights.shape != (6,) or not np.isfinite(self.weights).all() or self.weights[5] < 0:
            raise ValueError("weights must contain six finite values with a nonnegative rotor penalty")
        self.mask_inter = {}
        self.mask_intra = {}

    def set_box(self, box):
        if box is not None:
            box = np.asarray(box, dtype=float)
            if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[0] >= box[1]):
                raise ValueError("box must contain finite lower and upper corners with positive dimensions")
            self.box = box.copy()
        else:
            self.box = None

    def set_protein(self, rec):
        if rec is None:
            self.rec_dict = None
            self.mask_inter = {}
        else:
            self.rec_dict = rec.atom_dict[rec.atom_dict["atomicnum"] != 1].copy()
            self.rec_dict = self.correct_radius(self.rec_dict)
            self.mask_inter = {}

    def set_ligand(self, lig):
        lig_hvy_mask = lig.atom_dict["atomicnum"] != 1
        if not lig_hvy_mask.any() or not np.isfinite(lig.coords).all():
            raise ValueError("ligand must contain heavy atoms with finite three-dimensional coordinates")
        self._ligand = lig.clone
        self._ligand.title = lig.title
        self._ligand.data.update(dict(lig.data))
        self._heavy_mask = lig_hvy_mask.copy()
        self.lig_dict = lig.atom_dict[lig_hvy_mask].copy()
        self.lig_dict = self.correct_radius(self.lig_dict)
        self.num_rotors = num_rotors_pdbqt(lig)
        self.mask_inter = {}
        self.mask_intra = {}

        # Find distant members (min 3 consecutive bonds)
        mask = np.vstack([~get_close_neighbors(lig, i, num_bonds=3) for i in range(len(lig.atoms))])
        mask = mask[lig_hvy_mask[np.newaxis, :] * lig_hvy_mask[:, np.newaxis]]
        self.lig_distant_members = mask.reshape(lig_hvy_mask.sum(), lig_hvy_mask.sum())

        # prepare rotors dictionary
        self.rotors = []
        for b in lig.bonds:
            if _is_search_rotor(b):
                a2 = int(b.atoms[0].idx0)
                a3 = int(b.atoms[1].idx0)
                for n in b.atoms[0].neighbors:
                    if a3 != int(n.idx0) and n.atomicnum != 1:
                        a1 = int(n.idx0)
                        break
                for n in b.atoms[1].neighbors:
                    if a2 != int(n.idx0) and n.atomicnum != 1:
                        a4 = int(n.idx0)
                        break
                full_atoms = (a1, a2, a3, a4)
                full_mask = get_children(lig, a3, a2)
                rot_mask = full_mask[lig_hvy_mask]
                # translate atom indicies to lig_dict indicies (heavy only)
                a1 = np.argwhere(self.lig_dict["id"] == a1).flatten()[0]
                a2 = np.argwhere(self.lig_dict["id"] == a2).flatten()[0]
                a3 = np.argwhere(self.lig_dict["id"] == a3).flatten()[0]
                a4 = np.argwhere(self.lig_dict["id"] == a4).flatten()[0]
                # rotate smaller part of the molecule
                if rot_mask.sum() > len(rot_mask) / 2:
                    rot_mask = ~rot_mask
                    full_mask = ~full_mask
                    a4, a3, a2, a1 = a1, a2, a3, a4
                    full_atoms = full_atoms[::-1]
                self.rotors.append(
                    {"atoms": (a1, a2, a3, a4), "mask": rot_mask, "full_atoms": full_atoms, "full_mask": full_mask}
                )

        # Setup cached ligand coords
        self.lig = vina_ligand(self.lig_dict["coords"].copy(), len(self.rotors), self, self.box_size)

    def set_coords(self, coords):
        """Set the default heavy-atom scoring coordinates.

        Use ``set_ligand`` to replace the reference conformation for searching.
        """
        coords = np.asarray(coords, dtype=float)
        if coords.shape != self.lig_dict["coords"].shape or not np.isfinite(coords).all():
            raise ValueError("coords must provide finite coordinates for every ligand heavy atom")
        self.lig_dict["coords"] = coords

    def score(self, coords=None):
        return (self.score_inter(coords) * self.weights[:5]).sum() / (1 + self.weights[5] * self.num_rotors)

    def weighted_total(self, coords=None):
        return (self.score_total(coords) * self.weights[:5]).sum()

    def score_total(self, coords=None):
        return self.score_inter(coords) + self.score_intra(coords)

    def weighted_inter(self, coords=None):
        return (self.score_inter(coords) * self.weights[:5]).sum()

    def weighted_intra(self, coords=None):
        return (self.score_intra(coords) * self.weights[:5]).sum()

    def score_inter(self, coords=None):
        return self._pair_score(coords)[0]

    def score_intra(self, coords=None):
        return self._pair_score(coords, intra=True)[0]

    def _pair_score(self, coords=None, intra=False, gradient=False, energy_cap=None):
        """Evaluate the five pair terms and optionally their Cartesian gradient."""
        if coords is None:
            coords = self.lig_dict["coords"]
        first = self.lig_dict if intra else self.rec_dict
        if first is None:
            raise ValueError("a receptor is required for intermolecular scoring")
        first_coords = coords if intra else first["coords"]
        r = distance(first_coords, coords)
        mask = r < 8
        if intra:
            # Count each nonbonded pair once, excluding 1-2, 1-3 and 1-4 pairs.
            mask &= np.triu(self.lig_distant_members, k=1)
        rows, cols = np.nonzero(mask)
        d = r[mask] - first["radius"][rows] - self.lig_dict["radius"][cols]
        cache = self.mask_intra if intra else self.mask_inter
        if "hyd" not in cache:
            cache["hyd"] = (first["ishydrophobe"] | first["ishalogen"])[:, None] & (
                self.lig_dict["ishydrophobe"] | self.lig_dict["ishalogen"]
            )[None, :]
        if "hb" not in cache:
            cache["hb"] = ((first["isdonor"] | first["ismetal"])[:, None] & self.lig_dict["isacceptor"][None, :]) | (
                first["isacceptor"][:, None] & (self.lig_dict["isdonor"] | self.lig_dict["ismetal"])[None, :]
            )
        hyd = cache["hyd"][mask]
        hb = cache["hb"][mask]
        gauss1 = np.exp(-((d / 0.5) ** 2))
        gauss2 = np.exp(-(((d - 3) / 2) ** 2))
        pair_terms = np.array(
            [
                gauss1,
                gauss2,
                np.minimum(d, 0) ** 2,
                np.clip(1.5 - d, 0, 1) * hyd,
                np.clip(d / -0.7, 0, 1) * hb,
            ]
        )
        factors = np.ones(len(d))
        if energy_cap is not None:
            energies = (pair_terms * self.weights[:5, None]).sum(axis=0)
            if not intra:
                # Vina softens the summed receptor energy of each ligand atom;
                # intraligand interactions are softened individually per pair.
                energies = np.bincount(cols, weights=energies, minlength=len(coords))[cols]
            factors = energy_cap / (energy_cap + np.maximum(energies, 0))
        terms = (pair_terms * factors).sum(axis=1)
        if not gradient:
            return terms, None
        slope = (
            self.weights[0] * -8 * d * gauss1
            + self.weights[1] * -0.5 * (d - 3) * gauss2
            + self.weights[2] * 2 * np.minimum(d, 0)
            - self.weights[3] * (hyd & (d > 0.5) & (d < 1.5))
            - self.weights[4] / 0.7 * (hb & (d > -0.7) & (d < 0))
        )
        slope *= factors**2
        pair_gradient = (coords[cols] - first_coords[rows]) * (slope / np.maximum(r[mask], 1e-12))[:, None]
        cartesian = np.zeros_like(coords, dtype=float)
        np.add.at(cartesian, cols, pair_gradient)
        if intra:
            np.add.at(cartesian, rows, -pair_gradient)
        return terms, cartesian

    def _search_bounds(self):
        if self.rec_dict is None or not len(self.rec_dict):
            raise ValueError("a receptor with heavy atoms is required for docking")
        if self.lig_dict is None:
            raise ValueError("a ligand is required for docking")
        if self.box is None:
            raise ValueError("a search box is required for docking")
        if any(
            np.linalg.norm(self.lig.c0[rotor["atoms"][1]] - self.lig.c0[rotor["atoms"][2]]) < 1e-8
            for rotor in self.rotors
        ):
            raise ValueError("rotatable bonds must have nonzero length; supply a three-dimensional conformer")
        return (self.box - self.lig.c0.mean(axis=0)) / self.box_size

    def _objective(self, x, energy_cap=None):
        """Total pair energy plus a box restraint, with a pose-space gradient."""
        coords = self.lig.mutate(x)
        inter, grad_inter = self._pair_score(coords, gradient=True, energy_cap=energy_cap)
        intra, grad_intra = self._pair_score(coords, intra=True, gradient=True, energy_cap=energy_cap)
        # A stiff wall confines all heavy atoms, not just the ligand centroid.
        outside = coords - np.clip(coords, self.box[0], self.box[1])
        energy = np.dot(inter + intra, self.weights[:5]) + 1e4 * (outside**2).sum()
        cartesian = grad_inter + grad_intra + 2e4 * outside
        gradient = np.empty_like(x)
        gradient[:3] = cartesian.sum(axis=0) * self.box_size
        # The XYZ Euler transform is Rz(gamma) Ry(beta) Rx(alpha). Project
        # Cartesian torque onto each rotation axis in the final coordinate frame.
        beta, gamma = x[4:6]
        axes = np.array(
            [
                [math.cos(beta) * math.cos(gamma), math.cos(beta) * math.sin(gamma), -math.sin(beta)],
                [-math.sin(gamma), math.cos(gamma), 0],
                [0, 0, 1],
            ]
        )
        torque = np.cross(coords - coords.mean(axis=0), cartesian).sum(axis=0)
        gradient[3:6] = axes @ torque
        # Each rotor moves a connected fragment around its bond. Removing the
        # mean force accounts for recentering the ligand after torsion changes.
        centered_gradient = cartesian - cartesian.mean(axis=0)
        for i, rotor in enumerate(self.rotors, start=6):
            _, a2, a3, _ = rotor["atoms"]
            axis = coords[a2] - coords[a3]
            axis /= np.linalg.norm(axis)
            mask = rotor["mask"]
            torque = np.cross(coords[mask] - coords[a3], centered_gradient[mask]).sum(axis=0)
            gradient[i] = np.dot(axis, torque)
        return float(energy), gradient

    def optimize(self, x0=None, maxiter=100, energy_cap=None):
        """Refine one pose using :func:`scipy.optimize.minimize` and L-BFGS-B.

        Parameters
        ----------
        x0 : array-like, optional
            Translation genes, XYZ Euler angles, and torsion offsets (radians),
            in that order. The default is the input pose, with its centroid
            clipped to the box. Translation units are ``box_size``.
        maxiter : int, optional
            Maximum number of L-BFGS-B iterations.
        energy_cap : float or None, optional
            Soften positive atom/pair energies using ``E * cap / (cap + E)``
            during Monte Carlo hunting. The default leaves energies unchanged.
            Box restraints are never softened.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            Optimizer diagnostics, pose genes ``x``, heavy-atom ``coords``,
            and torsion-corrected intermolecular ``affinity``. ``fun`` includes
            intra-ligand energy and the box restraint. An iteration limit or
            line-search failure can still leave a useful improved pose.
        """
        bounds = self._search_bounds()
        if not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
            raise ValueError("maxiter must be a positive integer")
        if energy_cap is not None and (not np.isfinite(energy_cap) or energy_cap <= 0):
            raise ValueError("energy_cap must be positive and finite, or None")
        x0 = self.lig.x0.copy() if x0 is None else np.asarray(x0, dtype=float).copy()
        if x0.shape != self.lig.x0.shape or not np.isfinite(x0).all():
            raise ValueError("x0 must contain six rigid-body genes and one gene per rotatable bond")
        x0[:3] = np.clip(x0[:3], bounds[0], bounds[1])
        initial_energy, initial_gradient = self._objective(x0, energy_cap)
        result = minimize(
            self._objective,
            x0,
            method="L-BFGS-B",
            jac=True,
            args=(energy_cap,),
            bounds=list(zip(bounds[0], bounds[1])) + [(None, None)] * (len(x0) - 3),
            options={"maxiter": int(maxiter), "ftol": 1e-9, "gtol": 1e-4, "maxls": 30},
        )
        if not np.isfinite(result.fun) or result.fun > initial_energy:
            result.x = x0
            result.fun = initial_energy
            result.jac = initial_gradient
            result.success = False
            result.message = "Local search did not improve the starting pose; retained starting pose."
        result.coords = self.lig.mutate(result.x)
        result.affinity = float(self.score(result.coords))
        return result

    def _random_conformation(self, bounds, rng):
        x = rng.uniform(-np.pi, np.pi, len(self.lig.x0))
        x[:3] = rng.uniform(bounds[0], bounds[1])
        x[3:6] = Rotation.random(random_state=rng).as_euler("xyz")
        return x

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

    def _add_minimum(self, minima, result, min_rmsd, capacity):
        """Keep a bounded set of distinct feasible, fully refined minima."""
        if (
            not np.isfinite(result.fun)
            or not np.all(result.coords >= self.box[0] - 1e-4)
            or not np.all(result.coords <= self.box[1] + 1e-4)
        ):
            return
        if minima:
            rmsds = [np.sqrt(np.mean(np.sum((result.coords - other.coords) ** 2, axis=1))) for other in minima]
            nearest = int(np.argmin(rmsds))
            if rmsds[nearest] < min_rmsd:
                if result.fun >= minima[nearest].fun:
                    return
                minima.pop(nearest)
        minima.append(result)
        minima.sort(key=lambda item: item.fun)
        del minima[capacity:]

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
            stats = {"proposals": steps, "accepted": 0, "uphill_accepted": 0, "refinements": 0}
            for step in range(steps):
                proposal = self._mutate_conf(current, rng, mutation_amplitude)
                candidate = self.optimize(proposal, maxiter=maxiter, energy_cap=10.0)
                if _metropolis_accept(current_energy, candidate.fun, temperature, rng):
                    stats["accepted"] += 1
                    stats["uphill_accepted"] += int(step > 0 and candidate.fun > current_energy)
                    current = candidate.x.copy()
                    current_energy = candidate.fun
                    # As in Vina, refine accepted promising states and save a
                    # bounded set of distinct minima, not rejected proposals.
                    if current_energy < chain_best or len(minima) < num_modes:
                        refined = self.optimize(current, maxiter=maxiter)
                        stats["refinements"] += 1
                        current = refined.x.copy()
                        current_energy = refined.fun
                        chain_best = min(chain_best, refined.fun)
                        best_energy = min(best_energy, refined.fun)
                        self._add_minimum(minima, refined, min_rmsd, num_modes)
                self.history_.append(best_energy)
            stats["best_energy"] = chain_best
            self.search_stats_.append(stats)
            archive.extend(minima)

        feasible = [
            result
            for result in archive
            if np.isfinite(result.fun)
            and np.all(result.coords >= self.box[0] - 1e-4)
            and np.all(result.coords <= self.box[1] + 1e-4)
        ]
        if not feasible:
            raise RuntimeError("No pose fits inside the search box; enlarge the box or increase the search budget")
        # Rank by physical energy, excluding the numerical boundary restraint.
        feasible.sort(key=lambda result: self.weighted_total(result.coords))
        best_energy = self.weighted_total(feasible[0].coords)
        self.results_ = []
        poses = []
        for result in feasible:
            total = self.weighted_total(result.coords)
            if total > best_energy + energy_range:
                break
            if any(
                np.sqrt(np.mean(np.sum((result.coords - other.coords) ** 2, axis=1))) < min_rmsd
                for other in self.results_
            ):
                continue
            pose = self._ligand.clone
            pose.title = self._ligand.title
            # Recompute score descriptors rather than inheriting values attached
            # to the input conformation by an earlier scoring operation.
            for key in list(pose.data.keys()):
                if key.startswith("vina_"):
                    del pose.data[key]
            pose.data.update({key: value for key, value in self._ligand.data.items() if not key.startswith("vina_")})
            pose.coords = self.lig.mutate(result.x, all_atoms=True)
            terms = ("gauss1", "gauss2", "repulsion", "hydrophobic", "hydrogen")
            pose.data.update(dict(zip(("vina_" + term for term in terms), self.score_inter(result.coords))))
            pose.data.update(dict(zip(("vina_intra_" + term for term in terms), self.score_intra(result.coords))))
            pose.data.update(
                {
                    "vina_affinity": result.affinity,
                    "vina_inter": float(self.weighted_inter(result.coords)),
                    "vina_intra": float(self.weighted_intra(result.coords)),
                    "vina_total": float(total),
                    "vina_num_rotors": self.num_rotors,
                }
            )
            poses.append(pose)
            self.results_.append(result)
            if len(poses) == num_modes:
                break
        return poses

    def correct_radius(self, atom_dict):
        vina_r = {
            6: 1.9,
            7: 1.8,
            8: 1.7,
            9: 1.5,
            15: 2.1,
            16: 2.0,
            17: 1.8,
            35: 2.0,
            53: 2.2,
        }
        for a, r in vina_r.items():
            atom_dict["radius"][atom_dict["atomicnum"] == a] = r
        # metals - 1.2 A
        atom_dict["radius"][atom_dict["ismetal"]] = 1.2
        return atom_dict


class vina_ligand(object):
    def __init__(self, c0, num_rotors, engine, box_size=1):
        # Atom dictionaries store float32 coordinates. Optimization must retain
        # sub-float32 perturbations when computing pose derivatives.
        self.c0 = np.array(c0, dtype=float, copy=True)
        self.x0 = np.zeros(6 + num_rotors)
        self.c1 = self.c0.copy()
        self.x1 = np.zeros_like(self.x0)
        self.engine = engine
        self.box_size = box_size

    def mutate(self, x2, force=False, all_atoms=False):
        """Map absolute genes to coordinates, independent of evaluation order.

        ``force`` is retained for compatibility. Incremental Euler rotations
        are not equivalent to evaluating the absolute orientation, so every
        transform starts from the reference coordinates.
        """
        return self._full_mutate(np.asarray(x2, dtype=float), all_atoms=all_atoms)

    def _full_mutate(self, x, all_atoms=False):
        c = np.array(self.engine._ligand.coords, dtype=float) if all_atoms else self.c0.copy()
        heavy = self.engine._heavy_mask if all_atoms else np.ones(len(c), dtype=bool)
        for angle, rotor in zip(x[6:], self.engine.rotors):
            atoms = rotor["full_atoms"] if all_atoms else rotor["atoms"]
            mask = rotor["full_mask"] if all_atoms else rotor["mask"]
            c = change_dihedral(c, *atoms, angle, mask)
        c = rotate(c, *x[3:6])
        # Torsion changes must not move the translation anchor.
        c += self.c0.mean(axis=0) + x[:3] * self.box_size - c[heavy].mean(axis=0)
        if not all_atoms:
            self.c1 = c.copy()
            self.x1 = x.copy()
        return c
