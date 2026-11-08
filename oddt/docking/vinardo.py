"""Independent Vinardo equations from Quiroga & Villar (2016), Tables 1–3.

Publication: https://doi.org/10.1371/journal.pone.0155183
No implementation code from Vinardo, Smina or AutoDock Vina is used here.
"""

import numpy as np
from scipy.spatial.distance import cdist

from .base import DockingBase
from .internal import num_rotors_pdbqt
from .search import DifferentialEvolutionMixin, GeneticAlgorithmMixin, MonteCarloMixin


class vinardo_scoring(DockingBase):
    """Vinardo pair scoring with analytic gradients and ODDT atom typing.

    Parameters
    ----------
    rec : oddt.toolkit.Molecule or None
        Rigid receptor with appropriate protonation and protein typing.
    lig : oddt.toolkit.Molecule or None, optional
        Ligand with three-dimensional coordinates.
    box : array-like, shape (2, 3), optional
        Lower and upper Cartesian search corners in Angstroms.
    box_size : float, optional
        Angstroms per translation gene.
    weights : array-like, shape (5,), optional
        Gaussian, repulsion, hydrophobic, hydrogen bond and torsion weights.
        Defaults to (-0.045, 0.8, -0.035, -0.6, 0.05846).
    sample_callback : callable or None, optional
        Observer called after each local optimization.

    Notes
    -----
    Implements the published radial potentials, including the distinct aromatic
    carbon radius. Uses direct pairs within 8 Angstroms, rather than grids.
    Search energy includes intraligand pairs more than three bonds apart;
    reported affinity is intermolecular energy / (1 + w_tors * N_rot).
    """

    scoring_name = "vinardo"
    score_metadata_prefixes = ("docking_", "vinardo_")
    mc_hunt_energy_cap = 10.0
    term_names = ("gaussian", "repulsion", "hydrophobic", "hydrogen")

    def __init__(self, rec, lig=None, box=None, box_size=1.0, weights=None, sample_callback=None):
        self.weights = np.asarray((-0.045, 0.8, -0.035, -0.6, 0.05846) if weights is None else weights, dtype=float)
        if self.weights.shape != (5,) or not np.isfinite(self.weights).all() or self.weights[-1] < 0:
            raise ValueError("weights must contain five finite values with a nonnegative torsion weight")
        super().__init__(rec, lig, box, box_size, sample_callback=sample_callback)

    def prepare_atom_dict(self, atom_dict):
        radii = {6: 2.0, 7: 1.7, 8: 1.6, 9: 1.5, 15: 2.1, 16: 2.0, 17: 1.8, 35: 2.0, 53: 2.2}
        supported = np.isin(atom_dict["atomicnum"], list(radii)) | atom_dict["ismetal"]
        if not supported.all():
            raise ValueError("unsupported Vinardo element(s): %s" % np.unique(atom_dict["atomicnum"][~supported]))
        for element, radius in radii.items():
            atom_dict["radius"][atom_dict["atomicnum"] == element] = radius
        atom_dict["radius"][(atom_dict["atomicnum"] == 6) & atom_dict["isaromatic"]] = 1.9
        atom_dict["radius"][atom_dict["ismetal"]] = 1.2
        return atom_dict

    def set_ligand(self, lig):
        super().set_ligand(lig)
        self.num_rotors = num_rotors_pdbqt(lig)

    def _pairs(self, coords, intra=False, gradient=False, energy_cap=None):
        if energy_cap is not None and (np.isnan(energy_cap) or energy_cap <= 0):
            raise ValueError("energy_cap must be positive or None")
        coords = self.lig_dict["coords"].astype(float) if coords is None else np.asarray(coords, dtype=float)
        if coords.shape != self.lig_dict["coords"].shape or not np.isfinite(coords).all():
            raise ValueError("coords must contain finite coordinates for all ligand heavy atoms")
        first = self.lig_dict if intra else self.rec_dict
        if first is None:
            raise ValueError("a receptor is required for intermolecular scoring")
        first_coords = coords if intra else first["coords"]
        distances = cdist(first_coords, coords)
        mask = distances < 8.0
        if intra:
            mask &= np.triu(self.lig_distant_members, 1)
        rows, cols = np.nonzero(mask)
        r = distances[rows, cols]
        d = r - first["radius"][rows] - self.lig_dict["radius"][cols]
        hyd = (first["ishydrophobe"][rows] | first["ishalogen"][rows]) & (
            self.lig_dict["ishydrophobe"][cols] | self.lig_dict["ishalogen"][cols]
        )
        hb = ((first["isdonor"][rows] | first["ismetal"][rows]) & self.lig_dict["isacceptor"][cols]) | (
            first["isacceptor"][rows] & (self.lig_dict["isdonor"][cols] | self.lig_dict["ismetal"][cols])
        )
        gaussian = np.exp(-((d / 0.8) ** 2))
        terms = np.array(
            [gaussian, np.minimum(d, 0) ** 2, hyd * np.clip(1 - d / 2.5, 0, 1), hb * np.clip(-d / 0.6, 0, 1)]
        )
        factors = np.ones(len(r))
        if energy_cap is not None and np.isfinite(energy_cap):
            energies = self.weights[:4] @ terms
            if not intra:
                energies = np.bincount(cols, weights=energies, minlength=len(coords))[cols]
            factors = energy_cap / (energy_cap + np.maximum(energies, 0))
        total = (terms * factors).sum(axis=1)
        if not gradient:
            return total, None
        slope = (
            self.weights[0] * (-2 / 0.8**2) * d * gaussian
            + self.weights[1] * 2 * np.minimum(d, 0)
            - self.weights[2] / 2.5 * (hyd & (d > 0) & (d < 2.5))
            - self.weights[3] / 0.6 * (hb & (d > -0.6) & (d < 0))
        ) * factors**2
        forces = (coords[cols] - first_coords[rows]) * (slope / np.maximum(r, 1e-12))[:, None]
        cartesian = np.zeros_like(coords)
        for axis in range(3):
            cartesian[:, axis] = np.bincount(cols, weights=forces[:, axis], minlength=len(coords))
        if intra:
            np.add.at(cartesian, rows, -forces)
        return total, cartesian

    def energy_and_gradient(self, coords, energy_cap=None):
        inter, g_inter = self._pairs(coords, gradient=True, energy_cap=energy_cap)
        intra, g_intra = self._pairs(coords, intra=True, gradient=True, energy_cap=energy_cap)
        return float(self.weights[:4] @ (inter + intra)), g_inter + g_intra

    def score_inter(self, coords=None):
        """Return the four unweighted intermolecular terms."""
        return self._pairs(coords)[0]

    def score_intra(self, coords=None):
        """Return the four unweighted intraligand terms."""
        return self._pairs(coords, intra=True)[0]

    def energy(self, coords=None):
        return float(self.weights[:4] @ (self.score_inter(coords) + self.score_intra(coords)))

    def score(self, coords=None):
        return float(self.weights[:4] @ self.score_inter(coords) / (1 + self.weights[4] * self.num_rotors))

    def score_components(self, coords=None):
        return {
            prefix + name: float(value)
            for prefix, terms in (("inter_", self.score_inter(coords)), ("intra_", self.score_intra(coords)))
            for name, value in zip(self.term_names, terms * self.weights[:4])
        }

    def _pose_metadata(self, result, total):
        data = super()._pose_metadata(result, total)
        data.update(
            vinardo_affinity=float(result.score), vinardo_total=float(total), vinardo_num_rotors=self.num_rotors
        )
        return data


class vinardo_ga(GeneticAlgorithmMixin, vinardo_scoring):
    """Vinardo scoring with genetic search and L-BFGS-B refinement."""


class vinardo_mc(MonteCarloMixin, vinardo_scoring):
    """Vinardo scoring with Monte Carlo search and L-BFGS-B refinement."""


class vinardo_de(DifferentialEvolutionMixin, vinardo_scoring):
    """Vinardo scoring with adaptive differential evolution and L-BFGS-B."""
