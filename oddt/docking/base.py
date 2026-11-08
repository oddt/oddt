"""Scorer-independent ligand geometry, local optimization and pose selection."""

import math

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from oddt.spatial import rotate


def _cross_rows(first, second):
    """Cross products of Nx3 arrays without NumPy's general axis dispatch."""
    result = np.empty_like(first)
    result[:, 0] = first[:, 1] * second[:, 2] - first[:, 2] * second[:, 1]
    result[:, 1] = first[:, 2] * second[:, 0] - first[:, 0] * second[:, 2]
    result[:, 2] = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    return result


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


class DockingBase:
    """Pose geometry and L-BFGS-B refinement, independent of a scoring model.

    Parameters
    ----------
    rec : oddt.toolkit.Molecule or None
        Rigid receptor. Set ``protein=True`` before computing protein typing.
    lig : oddt.toolkit.Molecule or None, optional
        Ligand with finite three-dimensional coordinates.
    box : array-like, shape (2, 3), optional
        Lower and upper Cartesian box corners in Angstroms.
    box_size : float, optional
        Angstroms per translation gene; must be positive.
    sample_callback : callable or None, optional
        Read-only observer called as ``sample_callback(result, energy_cap)``
        after each local search. Exceptions propagate.

    Notes
    -----
    Subclasses implement ``energy_and_gradient(coords, energy_cap=None)`` for
    heavy-atom Cartesian coordinates, excluding the shared box restraint.
    ``energy`` defines the unrestrained energy used for output ranking;
    ``score`` optionally defines a distinct reported score. Named weighted
    contributions may be supplied by ``score_components``. Hydrogens follow
    their fragments in output poses, but are not scored by this interface.

    Override ``prepare_atom_dict`` for scorer-specific typing/parameters and
    ``_pose_metadata`` or ``score_metadata_prefixes`` for specialized outputs.
    Default metadata uses ``docking_*`` fields. Monte Carlo hunting/refinement
    caps default to None; score models may set their own caps, temperature and
    output energy window through the class attributes below.
    """

    scoring_name = "custom"
    score_metadata_prefixes = ("docking_",)
    default_mc_temperature = 1.2
    default_energy_range = 3.0
    mc_hunt_energy_cap = None
    mc_refinement_energy_cap = None

    def __init__(self, rec, lig=None, box=None, box_size=1.0, sample_callback=None):
        if sample_callback is not None and not callable(sample_callback):
            raise ValueError("sample_callback must be callable or None")
        self.sample_callback = sample_callback
        if not np.isfinite(box_size) or box_size <= 0:
            raise ValueError("box_size must be positive and finite")
        self.box_size = float(box_size)
        self.rec_dict = None
        self.lig_dict = None
        self.set_protein(rec)
        if lig is not None:
            self.set_ligand(lig)
        self.set_box(box)

    def prepare_atom_dict(self, atom_dict):
        """Prepare a copied heavy-atom dictionary for this scoring model.

        Preserve atom order, ids and coordinates used by the pose geometry.
        """
        return atom_dict

    def energy_and_gradient(self, coords, energy_cap=None):
        """Return physical energy and an Nx3 heavy-atom Cartesian gradient.

        This excludes box restraints. None requests the exact objective;
        infinity is also uncapped. A scorer may implement finite positive
        hunting caps, or reject them when unsupported. The returned gradient
        must differentiate the returned energy, including any soft cap.
        """
        raise NotImplementedError("A scoring model must implement energy_and_gradient")

    def energy(self, coords=None):
        """Return unrestrained energy for ranking and output filtering."""
        if coords is None:
            coords = self.lig_dict["coords"]
        return float(self.energy_and_gradient(coords)[0])

    def score(self, coords=None):
        """Return the reported score; defaults to unrestrained energy."""
        return self.energy(coords)

    def score_components(self, coords=None):
        """Return optional named weighted energy contributions as a mapping."""
        return {}

    def _pose_metadata(self, result, total):
        """Produce scorer-neutral output fields for a selected pose."""
        data = {
            "docking_scorer": self.scoring_name,
            "docking_energy": float(total),
            "docking_score": float(result.score if "score" in result else result.affinity),
        }
        data.update(
            {"docking_term_" + name: float(value) for name, value in self.score_components(result.coords).items()}
        )
        return data

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
        else:
            self.rec_dict = rec.atom_dict[rec.atom_dict["atomicnum"] != 1].copy()
            self.rec_dict = self.prepare_atom_dict(self.rec_dict)

    def set_ligand(self, lig):
        lig_hvy_mask = lig.atom_dict["atomicnum"] != 1
        if not lig_hvy_mask.any() or not np.isfinite(lig.coords).all():
            raise ValueError("ligand must contain heavy atoms with finite three-dimensional coordinates")
        self._ligand = lig.clone
        self._ligand.title = lig.title
        self._ligand.data.update(dict(lig.data))
        self._heavy_mask = lig_hvy_mask.copy()
        self.lig_dict = lig.atom_dict[lig_hvy_mask].copy()
        self.lig_dict = self.prepare_atom_dict(self.lig_dict)

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
        self.lig = LigandPose(self.lig_dict["coords"].copy(), len(self.rotors), self, self.box_size)

    def set_coords(self, coords):
        """Set the default heavy-atom scoring coordinates.

        Use ``set_ligand`` to replace the reference conformation for searching.
        """
        coords = np.asarray(coords, dtype=float)
        if coords.shape != self.lig_dict["coords"].shape or not np.isfinite(coords).all():
            raise ValueError("coords must provide finite coordinates for every ligand heavy atom")
        self.lig_dict["coords"] = coords

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
        """Scorer energy plus a box restraint, with a pose-space gradient."""
        coords = self.lig.mutate(x)
        energy, cartesian = self.energy_and_gradient(coords, energy_cap=energy_cap)
        # A stiff wall confines all heavy atoms, not just the ligand centroid.
        outside = coords - np.clip(coords, self.box[0], self.box[1])
        energy = energy + 1e4 * (outside**2).sum()
        cartesian = cartesian + 2e4 * outside
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
        torque = _cross_rows(coords - coords.mean(axis=0), cartesian).sum(axis=0)
        gradient[3:6] = axes @ torque
        # Each rotor moves a connected fragment around its bond. Removing the
        # mean force accounts for recentering the ligand after torsion changes.
        centered_gradient = cartesian - cartesian.mean(axis=0)
        for i, rotor in enumerate(self.rotors, start=6):
            _, a2, a3, _ = rotor["atoms"]
            axis = coords[a2] - coords[a3]
            axis /= np.linalg.norm(axis)
            mask = rotor["mask"]
            torque = _cross_rows(coords[mask] - coords[a3], centered_gradient[mask]).sum(axis=0)
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
        energy_cap : positive float or None, optional
            Scorer-defined hunting cap. None requests the exact objective;
            infinity requests uncapped refinement. A model may reject finite
            caps. Box restraints are never softened.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            Optimizer diagnostics, pose genes ``x``, heavy-atom ``coords``,
            and model-defined ``score`` (also available as the compatibility
            alias ``affinity``). ``fun`` includes the box restraint. An iteration limit or
            line-search failure can still leave a useful improved pose.
        """
        bounds = self._search_bounds()
        if not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
            raise ValueError("maxiter must be a positive integer")
        if energy_cap is not None and (np.isnan(energy_cap) or energy_cap <= 0):
            raise ValueError("energy_cap must be positive (including infinity), or None")
        x0 = self.lig.x0.copy() if x0 is None else np.asarray(x0, dtype=float).copy()
        if x0.shape != self.lig.x0.shape or not np.isfinite(x0).all():
            raise ValueError("x0 must contain six rigid-body genes and one gene per rotatable bond")
        x0[:3] = np.clip(x0[:3], bounds[0], bounds[1])
        initial_energy, initial_gradient = self._objective(x0, energy_cap)

        def objective(x):
            # SciPy starts at x0 too; retain its first value for the fallback
            # without evaluating that pose twice for every local search.
            if np.array_equal(x, x0):
                return initial_energy, initial_gradient
            return self._objective(x, energy_cap)

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            jac=True,
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
        result.score = result.affinity
        if self.sample_callback is not None:
            self.sample_callback(result, energy_cap)
        return result

    def _random_conformation(self, bounds, rng):
        x = rng.uniform(-np.pi, np.pi, len(self.lig.x0))
        x[:3] = rng.uniform(bounds[0], bounds[1])
        x[3:6] = Rotation.random(random_state=rng).as_euler("xyz")
        return x

    def _add_minimum(self, minima, result, min_rmsd, capacity):
        """Keep a bounded set of distinct feasible local-search results."""
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

    def _finish_docking(self, archive, num_modes, min_rmsd, energy_range):
        """Rank, cluster and materialize poses using the shared scoring model."""
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
        feasible.sort(key=lambda result: self.energy(result.coords))
        best_energy = self.energy(feasible[0].coords)
        self.results_ = []
        poses = []
        for result in feasible:
            total = self.energy(result.coords)
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
                if key.startswith(self.score_metadata_prefixes):
                    del pose.data[key]
            pose.data.update(
                {
                    key: value
                    for key, value in self._ligand.data.items()
                    if not key.startswith(self.score_metadata_prefixes)
                }
            )
            pose.coords = self.lig.mutate(result.x, all_atoms=True)
            pose.data.update(self._pose_metadata(result, total))
            poses.append(pose)
            self.results_.append(result)
            if len(poses) == num_modes:
                break
        return poses


class LigandPose:
    """Absolute heavy-atom pose transforms with attached output hydrogens."""

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
