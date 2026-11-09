"""Vina-specific pair potentials and compatible public docking engines."""

import numpy as np

from oddt.spatial import distance
from ._compat import deprecated_class_alias as _deprecated_class_alias
from .base import (
    DockingBase,
    LigandPose,
    _cross_rows,
    _is_search_rotor,
    get_children,
    get_close_neighbors,
    change_dihedral,
)
from .search import AntColonyMixin, DifferentialEvolutionMixin, GeneticAlgorithmMixin, MonteCarloMixin


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


class OddtVinaScoring(DockingBase):
    """Vina scoring, pose geometry and L-BFGS-B refinement against a rigid receptor.

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
    sample_callback : callable or None, optional
        Read-only observer called as ``sample_callback(result, energy_cap)``
        after each local search, including proposals later discarded. The
        result contains heavy-atom coordinates; copy them to retain a snapshot.
        Callback exceptions propagate. Disabled by default.

    Notes
    -----
    This is an independent search implementation, not AutoDock Vina. It uses
    ODDT atom typing and direct pair scoring rather than Vina affinity grids.
    """

    scoring_name = "vina"
    score_metadata_prefixes = ("docking_", "vina_")
    mc_hunt_energy_cap = 10.0
    mc_refinement_energy_cap = np.inf
    term_names = ("gauss1", "gauss2", "repulsion", "hydrophobic", "hydrogen")

    def __init__(self, rec, lig=None, box=None, box_size=1.0, weights=None, sample_callback=None):
        super().__init__(rec, lig, box, box_size, sample_callback=sample_callback)
        self.weights = np.asarray(
            (-0.0356, -0.00516, 0.840, -0.0351, -0.587, 0.0585) if weights is None else weights, dtype=float
        )
        if self.weights.shape != (6,) or not np.isfinite(self.weights).all() or self.weights[5] < 0:
            raise ValueError("weights must contain six finite values with a nonnegative rotor penalty")
        self.mask_inter = {}
        self.mask_intra = {}

    def prepare_atom_dict(self, atom_dict):
        return self.correct_radius(atom_dict)

    def set_protein(self, rec):
        super().set_protein(rec)
        self.mask_inter = {}

    def set_ligand(self, lig):
        super().set_ligand(lig)
        self.num_rotors = num_rotors_pdbqt(lig)
        self.mask_inter = {}
        self.mask_intra = {}

    def energy(self, coords=None):
        return float(self.weighted_total(coords))

    def energy_and_gradient(self, coords, energy_cap=None):
        inter, grad_inter = self._pair_score(coords, gradient=True, energy_cap=energy_cap)
        intra, grad_intra = self._pair_score(coords, intra=True, gradient=True, energy_cap=energy_cap)
        return np.dot(inter + intra, self.weights[:5]), grad_inter + grad_intra

    def score_components(self, coords=None):
        inter = self.score_inter(coords) * self.weights[:5]
        intra = self.score_intra(coords) * self.weights[:5]
        return dict(
            zip(("inter_" + term for term in self.term_names), map(float, inter)),
            **dict(zip(("intra_" + term for term in self.term_names), map(float, intra))),
        )

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
            if "distant" not in self.mask_intra:
                self.mask_intra["distant"] = np.triu(self.lig_distant_members, k=1)
            mask &= self.mask_intra["distant"]
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
        if energy_cap is None:
            # Retain the original scorer's reduction order. Summing masked
            # contacts with inserted zeros changes floating-point roundoff and
            # can send otherwise identical stochastic searches to other minima.
            terms = np.array(
                [
                    gauss1.sum(),
                    gauss2.sum(),
                    (np.minimum(d, 0) ** 2).sum(),
                    np.clip(1.5 - d[hyd], 0, 1).sum(),
                    np.clip(d[hb] / -0.7, 0, 1).sum(),
                ]
            )
        else:
            pair_terms = np.array(
                [
                    gauss1,
                    gauss2,
                    np.minimum(d, 0) ** 2,
                    np.clip(1.5 - d, 0, 1) * hyd,
                    np.clip(d / -0.7, 0, 1) * hb,
                ]
            )
            if np.isinf(energy_cap):
                factors = np.ones(len(d))
            else:
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
        if energy_cap is not None:
            slope *= factors**2
        pair_gradient = (coords[cols] - first_coords[rows]) * (slope / np.maximum(r[mask], 1e-12))[:, None]
        cartesian = np.zeros_like(coords, dtype=float)
        # Each column is a ligand atom. bincount accumulates the same ordered
        # additions in compiled code without indexed ufunc dispatch per pair.
        for axis in range(3):
            cartesian[:, axis] = np.bincount(cols, weights=pair_gradient[:, axis], minlength=len(coords))
        if intra:
            np.add.at(cartesian, rows, -pair_gradient)
        return terms, cartesian

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

    def _pose_metadata(self, result, total):
        data = {}
        terms = ("gauss1", "gauss2", "repulsion", "hydrophobic", "hydrogen")
        data.update(dict(zip(("vina_" + term for term in terms), self.score_inter(result.coords))))
        data.update(dict(zip(("vina_intra_" + term for term in terms), self.score_intra(result.coords))))
        data.update(
            {
                "vina_affinity": result.affinity,
                "vina_inter": float(self.weighted_inter(result.coords)),
                "vina_intra": float(self.weighted_intra(result.coords)),
                "vina_total": float(total),
                "vina_num_rotors": self.num_rotors,
            }
        )
        return data


class OddtVinaGA(GeneticAlgorithmMixin, OddtVinaScoring):
    """Vina scoring with a genetic algorithm and L-BFGS-B local refinement."""


class OddtVinaMC(MonteCarloMixin, OddtVinaScoring):
    """Vina scoring with Monte Carlo basin hopping and L-BFGS-B refinement."""


class OddtVinaDE(DifferentialEvolutionMixin, OddtVinaScoring):
    """Vina scoring with adaptive differential evolution and L-BFGS-B."""


class OddtVinaACO(AntColonyMixin, OddtVinaScoring):
    """Vina scoring with continuous ant-colony search and L-BFGS-B."""


# The default docking engine continues to use Monte Carlo search.
OddtVinaDocking = OddtVinaMC

_DEPRECATED_ALIASES = {
    "vina_scoring": "OddtVinaScoring",
    "vina_docking": "OddtVinaDocking",
    "vina_ligand": "LigandPose",
}
_DEPRECATED_ALIASES.update({name[4:]: name for name in _DEPRECATED_ALIASES.values() if name.startswith("Oddt")})


def __getattr__(name):
    return _deprecated_class_alias(name, globals(), _DEPRECATED_ALIASES)


def __dir__():
    return sorted(set(globals()) | _DEPRECATED_ALIASES.keys())


__all__ = [
    "OddtVinaScoring",
    "OddtVinaDocking",
    "OddtVinaGA",
    "OddtVinaMC",
    "OddtVinaDE",
    "OddtVinaACO",
    "num_rotors_pdbqt",
] + list(_DEPRECATED_ALIASES)
