"""Independent, direct-pair AD4 radial scoring with analytic derivatives.

Equations: Huey et al. (2007), https://doi.org/10.1002/jcc.20634.
Numerical atom parameters and bound-state weights: AutoDock4.2.6 User Guide,
pp. 34–35 (printed page numbers). Dielectric: AutoDock3 User Guide, section 9.
These are transcribed scientific parameters, not upstream implementation code.

This radial model omits AutoGrid's angular H-bond factors and saturation.
It is not a numerical replica of AutoDock4 or its grid interpolation.
"""

import numpy as np
from scipy.spatial.distance import cdist

from .base import DockingBase, get_close_neighbors
from .search import DifferentialEvolutionMixin, GeneticAlgorithmMixin, MonteCarloMixin

# R_ii (diameter), epsilon_ii, volume, atomic solvation parameter,
# optimum H-bond distance, H-bond well depth. Units: Angstrom, kcal/mol.
# Published numerical facts from the AD4.2.6 manual, not a copied parameter file.
AD4_PARAMETERS = {
    "H": (2.00, 0.020, 0.0000, 0.00051, 0.0, 0.0),
    "HD": (2.00, 0.020, 0.0000, 0.00051, 0.0, 0.0),
    "C": (4.00, 0.150, 33.5103, -0.00143, 0.0, 0.0),
    "A": (4.00, 0.150, 33.5103, -0.00052, 0.0, 0.0),
    "N": (3.50, 0.160, 22.4493, -0.00162, 0.0, 0.0),
    "NA": (3.50, 0.160, 22.4493, -0.00162, 1.9, 5.0),
    "OA": (3.20, 0.200, 17.1573, -0.00251, 1.9, 5.0),
    "F": (3.09, 0.080, 15.4480, -0.00110, 0.0, 0.0),
    "Mg": (1.30, 0.875, 1.5600, -0.00110, 0.0, 0.0),
    "P": (4.20, 0.200, 38.7924, -0.00110, 0.0, 0.0),
    "S": (4.00, 0.200, 33.5103, -0.00214, 0.0, 0.0),
    "SA": (4.00, 0.200, 33.5103, -0.00214, 2.5, 1.0),
    "Cl": (4.09, 0.276, 35.8235, -0.00110, 0.0, 0.0),
    "Ca": (1.98, 0.550, 2.7700, -0.00110, 0.0, 0.0),
    "Mn": (1.30, 0.875, 2.1400, -0.00110, 0.0, 0.0),
    "Fe": (1.30, 0.010, 1.8400, -0.00110, 0.0, 0.0),
    "Zn": (1.48, 0.550, 1.7000, -0.00110, 0.0, 0.0),
    "Br": (4.33, 0.389, 42.5661, -0.00110, 0.0, 0.0),
    "I": (4.72, 0.550, 55.0585, -0.00110, 0.0, 0.0),
}


def _parameters(molecule, charges):
    """Assign united-atom types and merge charges of carbon-bound hydrogens."""
    atoms = molecule.atom_dict
    types = np.full(len(atoms), "", dtype="U2")
    elements = {
        1: "H",
        6: "C",
        7: "N",
        8: "OA",
        9: "F",
        12: "Mg",
        15: "P",
        16: "S",
        17: "Cl",
        20: "Ca",
        25: "Mn",
        26: "Fe",
        30: "Zn",
        35: "Br",
        53: "I",
    }
    for element, name in elements.items():
        types[atoms["atomicnum"] == element] = name
    if np.any(types == ""):
        raise ValueError("unsupported AD4 element(s): %s" % np.unique(atoms["atomicnum"][types == ""]))
    types[(atoms["atomicnum"] == 6) & atoms["isaromatic"]] = "A"
    types[(atoms["atomicnum"] == 7) & atoms["isacceptor"]] = "NA"
    types[(atoms["atomicnum"] == 16) & atoms["isacceptor"]] = "SA"
    charges = np.asarray(charges, dtype=float).copy()
    if charges.shape != (len(atoms),) or not np.isfinite(charges).all():
        raise ValueError(
            "AD4 requires finite partial charges for every atom; prepare charges and use charge_model=None"
        )
    unmerged_charges = charges.copy()
    active = atoms["atomicnum"] != 1
    for atom in molecule.atoms:
        if atom.atomicnum != 1:
            continue
        neighbors = atom.neighbors
        if len(neighbors) != 1:
            raise ValueError("AD4 hydrogen atoms must have exactly one bonded neighbor")
        parent = neighbors[0]
        if parent.atomicnum in (7, 8, 16):
            types[atom.idx0] = "HD"
            active[atom.idx0] = True
        else:
            charges[parent.idx0] += charges[atom.idx0]
            charges[atom.idx0] = 0
    return {
        "types": types,
        "values": np.array([AD4_PARAMETERS[t] for t in types]),
        "charges": charges,
        "unmerged_charges": unmerged_charges,
        "active": active,
    }


class ad4_scoring(DockingBase):
    """AD4 radial pair potentials, united-atom charges and bound-state scoring.

    Parameters
    ----------
    rec : oddt.toolkit.Molecule or None
        Rigid receptor with correct protonation and protein typing.
    lig : oddt.toolkit.Molecule or None, optional
        Three-dimensional ligand. Hydrogens are explicit in scoring coordinates.
    box : array-like, shape (2, 3), optional
        Cartesian lower and upper search corners, in Angstroms.
    box_size : float, optional
        Angstroms per translation gene.
    weights : array-like, shape (5,), optional
        vdW, hydrogen bond, electrostatic, desolvation and torsion coefficients.
        Defaults to the AD4.2 bound-state calibration.
    charge_model : {'gasteiger', None}, optional
        Default adds hydrogens and calculates Gasteiger charges on private copies.
        None preserves supplied partial charges and requires explicit polar H;
        charges of omitted nonpolar H must already be merged into their parents.
    smoothing : float, optional
        Full width of the radial vdW/H-bond minimum filter, default 0.5 Angstrom.
        Set to zero for the unsmoothed published pair equations.
    unbound_energy : float or None, optional
        Fixed unbound intraligand reference. None uses the bound conformation,
        so intraligand energy cancels from reported affinity, but not search.
    sample_callback : callable or None, optional
        Observer called after each local optimization.

    Notes
    -----
    Independently implements 12-6 dispersion/repulsion, radial 12-10 H-bonding,
    Mehler–Solmajer screened electrostatics, charge-dependent Gaussian
    desolvation (sigma=3.5 Angstrom) and additive torsional entropy. Distances
    below 0.1 Angstrom are clamped; vdW/H-bonds use an 8 Angstrom cutoff,
    electrostatics/desolvation have no cutoff. Nonpolar H charges are merged
    into their parents. Unsupported elements and nonfinite charges fail.

    This is a radial AD4 variant: AutoGrid's angular H-bond factors, saturation
    and grid interpolation are omitted. Rings, conjugated bonds and terminal
    polar-H torsions retain their input geometry. The torsion count includes
    searched heavy-atom bonds. Affinities are not AutoDock4-equivalent.
    """

    scoring_name = "ad4"
    score_all_atoms = True
    score_metadata_prefixes = ("docking_", "ad4_")
    mc_hunt_energy_cap = 10.0
    term_names = ("vdw", "hydrogen", "electrostatic", "desolvation")

    def __init__(
        self,
        rec,
        lig=None,
        box=None,
        box_size=1.0,
        weights=None,
        charge_model="gasteiger",
        smoothing=0.5,
        unbound_energy=None,
        sample_callback=None,
    ):
        self.weights = np.asarray((0.1662, 0.1209, 0.1406, 0.1322, 0.2983) if weights is None else weights, dtype=float)
        if self.weights.shape != (5,) or not np.isfinite(self.weights).all() or self.weights[-1] < 0:
            raise ValueError("weights must contain five finite values with a nonnegative torsion weight")
        if charge_model not in ("gasteiger", None):
            raise ValueError("charge_model must be 'gasteiger' or None")
        if not np.isfinite(smoothing) or smoothing < 0:
            raise ValueError("smoothing must be nonnegative and finite")
        if unbound_energy is not None and not np.isfinite(unbound_energy):
            raise ValueError("unbound_energy must be finite or None")
        self.charge_model = charge_model
        self.smoothing = float(smoothing)
        self.unbound_energy = unbound_energy
        super().__init__(rec, lig, box, box_size, sample_callback=sample_callback)

    def _prepare(self, molecule):
        # RDKit's binary clone does not preserve arbitrary atom properties;
        # capture supplied charges before cloning. Never modify user molecules.
        supplied = molecule.charges.copy() if self.charge_model is None else None
        prepared = molecule.clone
        prepared.protein = molecule.protein
        prepared.title = molecule.title
        prepared.data.update(dict(molecule.data))
        if self.charge_model is not None:
            prepared.addh()
            prepared.calccharges(self.charge_model)
            if hasattr(prepared, "Mol"):
                charges = np.array([a.GetDoubleProp("_GasteigerCharge") for a in prepared.Mol.GetAtoms()])
            else:
                charges = prepared.charges
        else:
            probe = prepared.clone
            probe.protein = prepared.protein
            probe.addh(only_polar=True)
            if len(probe.atoms) != len(prepared.atoms):
                raise ValueError("charge_model=None requires explicit polar hydrogens; add H before assigning charges")
            charges = supplied
        if not np.isfinite(prepared.coords).all():
            raise ValueError("AD4 requires finite three-dimensional coordinates")
        return prepared, _parameters(prepared, charges)

    def set_protein(self, rec):
        if rec is None:
            self.rec_parameters = None
            super().set_protein(None)
        else:
            prepared, self.rec_parameters = self._prepare(rec)
            self.rec_coordinates = np.array(prepared.coords, dtype=float)
            super().set_protein(prepared)

    def set_ligand(self, lig):
        prepared, self.lig_parameters = self._prepare(lig)
        super().set_ligand(prepared)
        self.scoring_coordinates = np.array(prepared.coords, dtype=float)
        self.num_rotors = len(self.rotors)
        distant = np.vstack([~get_close_neighbors(prepared, i, num_bonds=3) for i in range(len(prepared.atoms))])
        self._distant = np.triu(distant, 1)

    def set_coords(self, coords):
        """Set default scoring coordinates for every explicit ligand atom."""
        coords = self._coords(coords)
        self.scoring_coordinates = coords.copy()
        self.lig_dict["coords"] = coords[self._heavy_mask]

    def _coords(self, coords):
        coords = self.scoring_coordinates if coords is None else np.asarray(coords, dtype=float)
        if coords.shape != self.scoring_coordinates.shape or not np.isfinite(coords).all():
            raise ValueError("AD4 coords must include finite coordinates for every explicit atom, including hydrogens")
        return coords

    def _pairs(self, coords=None, intra=False, gradient=False, energy_cap=None):
        if energy_cap is not None and (np.isnan(energy_cap) or energy_cap <= 0):
            raise ValueError("energy_cap must be positive or None")
        coords = self._coords(coords)
        first = self.lig_parameters if intra else self.rec_parameters
        if first is None:
            raise ValueError("a receptor is required for intermolecular scoring")
        first_coords = coords if intra else self.rec_coordinates
        first_ids = np.flatnonzero(first["active"])
        ligand_ids = np.flatnonzero(self.lig_parameters["active"])
        distances = cdist(first_coords[first_ids], coords[ligand_ids])
        mask = self._distant[np.ix_(first_ids, ligand_ids)] if intra else np.ones(distances.shape, dtype=bool)
        row, col = np.nonzero(mask)
        rows, cols = first_ids[row], ligand_ids[col]
        raw_r = distances[row, col]
        r = np.maximum(raw_r, 0.1)
        p, q = first["values"][rows], self.lig_parameters["values"][cols]
        q1, q2 = first["charges"][rows], self.lig_parameters["charges"][cols]
        t1, t2 = first["types"][rows], self.lig_parameters["types"][cols]
        hb = ((t1 == "HD") & (q[:, 5] > 0)) | ((t2 == "HD") & (p[:, 5] > 0))
        optimum = np.where(hb, p[:, 4] + q[:, 4], (p[:, 0] + q[:, 0]) / 2)
        depth = np.where(hb, p[:, 5] + q[:, 5], np.sqrt(p[:, 1] * q[:, 1]))
        # Minimum over [r - smoothing/2, r + smoothing/2]: move toward the
        # potential's known minimum, or use its flat well if it is in the interval.
        offset = r - optimum
        smoothed_r = optimum + np.sign(offset) * np.maximum(np.abs(offset) - self.smoothing / 2, 0)
        smoothed_r = np.maximum(smoothed_r, 0.1)
        ratio = optimum / smoothed_r
        power = np.where(hb, 10, 6)
        # General n-m well of depth epsilon at R: m/(n-m)*(R/r)^n - n/(n-m)*(R/r)^m.
        a = power / (12 - power)
        b = 12 / (12 - power)
        radial = depth * (a * ratio**12 - b * ratio**power)
        radial_slope = depth * (-12 * a * ratio**12 + power * b * ratio**power) / smoothed_r
        radial_slope *= (np.abs(offset) > self.smoothing / 2) & (smoothed_r > 0.1)
        short = r < 8
        radial *= short
        radial_slope *= short
        # Sigmoidal distance-dependent dielectric, with a solvent limit of 78.4.
        dielectric_b = 78.4 + 8.5525
        exponential = 7.7839 * np.exp(-0.003627 * dielectric_b * r)
        dielectric = -8.5525 + dielectric_b / (1 + exponential)
        d_dielectric = 0.003627 * dielectric_b**2 * exponential / (1 + exponential) ** 2
        dielectric_valid = dielectric > 1
        dielectric = np.maximum(dielectric, 1)
        d_dielectric *= dielectric_valid
        electrostatic = 332.06371 * q1 * q2 / (dielectric * r)
        electrostatic_slope = -electrostatic * (1 / r + d_dielectric / dielectric)
        solvation = (p[:, 3] + 0.01097 * np.abs(q1)) * q[:, 2] + (q[:, 3] + 0.01097 * np.abs(q2)) * p[:, 2]
        desolvation = solvation * np.exp(-(r**2) / (2 * 3.5**2))
        desolvation_slope = -r / 3.5**2 * desolvation
        terms = np.array([radial * ~hb, radial * hb, electrostatic, desolvation])
        slopes = np.array([radial_slope * ~hb, radial_slope * hb, electrostatic_slope, desolvation_slope])
        factors = np.ones(len(r))
        if energy_cap is not None and np.isfinite(energy_cap):
            pair_energies = self.weights[:4] @ terms
            if not intra:
                pair_energies = np.bincount(cols, weights=pair_energies, minlength=len(coords))[cols]
            factors = energy_cap / (energy_cap + np.maximum(pair_energies, 0))
        total = (terms * factors).sum(axis=1)
        if not gradient:
            return total, None
        slope = (self.weights[:4] @ slopes) * factors**2 * (raw_r > 0.1)
        forces = (coords[cols] - first_coords[rows]) * (slope / np.maximum(raw_r, 1e-12))[:, None]
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
        """Return unweighted vdW, H-bond, electrostatic and desolvation terms."""
        return self._pairs(coords)[0]

    def score_intra(self, coords=None):
        """Return unweighted intraligand terms, excluding 1-2, 1-3 and 1-4."""
        return self._pairs(coords, intra=True)[0]

    def energy(self, coords=None):
        return float(self.weights[:4] @ (self.score_inter(coords) + self.score_intra(coords)))

    def score(self, coords=None):
        score = self.weights[:4] @ self.score_inter(coords) + self.weights[4] * self.num_rotors
        if self.unbound_energy is not None:
            score += self.weights[:4] @ self.score_intra(coords) - self.unbound_energy
        return float(score)

    def score_components(self, coords=None):
        return {
            prefix + name: float(value)
            for prefix, terms in (("inter_", self.score_inter(coords)), ("intra_", self.score_intra(coords)))
            for name, value in zip(self.term_names, terms * self.weights[:4])
        }

    def _pose_metadata(self, result, total):
        data = super()._pose_metadata(result, total)
        data.update(
            ad4_affinity=float(result.score),
            ad4_total=float(total),
            ad4_num_rotors=self.num_rotors,
            ad4_torsion=float(self.weights[4] * self.num_rotors),
            ad4_model="radial",
            ad4_charge_model=self.charge_model or "supplied",
            ad4_smoothing=self.smoothing,
            ad4_unbound_model="bound" if self.unbound_energy is None else "fixed",
        )
        if self.unbound_energy is not None:
            data["ad4_unbound_energy"] = float(self.unbound_energy)
        return data

    def _finish_docking(self, archive, num_modes, min_rmsd, energy_range):
        poses = super()._finish_docking(archive, num_modes, min_rmsd, energy_range)
        # Molecular cloning can discard atom properties. Keep the actual united-
        # atom charges on the returned molecules so supplied-charge rescoring is
        # possible without recalculating a different preparation.
        for pose in poses:
            for atom, charge in zip(pose.atoms, self.lig_parameters["charges"]):
                if hasattr(pose, "Mol"):
                    for name in ("_MMFF94Charge", "_GasteigerCharge"):
                        atom.Atom.ClearProp(name)
                    atom.Atom.SetDoubleProp("_TriposPartialCharge", float(charge))
                else:
                    atom.OBAtom.SetPartialCharge(float(charge))
            if not hasattr(pose, "Mol"):
                pose.OBMol.SetPartialChargesPerceived(True)
            pose._charges = None
        return poses


class ad4_ga(GeneticAlgorithmMixin, ad4_scoring):
    """AD4 radial scoring with genetic search and L-BFGS-B refinement."""


class ad4_mc(MonteCarloMixin, ad4_scoring):
    """AD4 radial scoring with Monte Carlo search and L-BFGS-B refinement."""


class ad4_de(DifferentialEvolutionMixin, ad4_scoring):
    """AD4 radial scoring with adaptive differential evolution and L-BFGS-B."""
