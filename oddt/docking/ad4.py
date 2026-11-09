"""Independent, direct-pair AD4 scoring with analytic derivatives.

Equations: Huey et al. (2007), https://doi.org/10.1002/jcc.20634.
Numerical atom parameters and bound-state weights: AutoDock4.2.6 User Guide,
pp. 34–35 (printed page numbers). Dielectric: AutoDock3 User Guide, section 9.
These are transcribed scientific parameters, not upstream implementation code.

Directionality and valence use an explicit lone-pair, direct-pair adaptation;
see ``_ad4_hbond``. It is not a numerical replica of AutoGrid's angular
polynomials, probe saturation or grid interpolation.
"""

import numpy as np
from scipy.spatial.distance import cdist

from ._compat import deprecated_class_alias as _deprecated_class_alias
from .base import DockingBase, get_close_neighbors
from ._ad4_hbond import acceptor_frames, angular_factors, geometry_rules, saturated_contacts
from .search import AntColonyMixin, DifferentialEvolutionMixin, GeneticAlgorithmMixin, MonteCarloMixin

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
        **geometry_rules(molecule, types),
    }


class OddtAD4Scoring(DockingBase):
    """AD4 pair terms, directional/saturable H-bonds and bound-state scoring.

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
    directional_hb : bool, optional
        Apply cosine-squared donor and explicit lone-pair angular factors.
        Default True. False retains radial hydrogen-bond strengths.
    hbond_saturation : bool, optional
        Default True limits each donor H to one favorable bond, nitrogen to
        one accepted bond and oxygen/sulfur to two, jointly across inter- and
        intraligand contacts. Repulsions are retained.

    Notes
    -----
    Independently implements 12-6 dispersion/repulsion, directional 12-10 H-bonding,
    Mehler–Solmajer screened electrostatics, charge-dependent Gaussian
    desolvation (sigma=3.5 Angstrom) and additive torsional entropy. Distances
    below 0.1 Angstrom are clamped; vdW/H-bonds use an 8 Angstrom cutoff,
    electrostatics/desolvation have no cutoff. Nonpolar H charges are merged
    into their parents. Unsupported elements and nonfinite charges fail.

    Explicit lone-pair frames and minimum-energy capacity matching adapt AD4's
    angular/valence requirements to direct pairs, rather than reproducing
    AutoGrid's angular polynomials, probe saturation or grid interpolation.
    Matching is piecewise differentiable; exact ties use a selected subgradient.
    Set both H-bond options to False for the previous radial model.
    Rings, conjugated bonds and terminal
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
        directional_hb=True,
        hbond_saturation=True,
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
        for name, value in (("directional_hb", directional_hb), ("hbond_saturation", hbond_saturation)):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(name + " must be a boolean")
        if hbond_saturation and self.weights[1] < 0:
            raise ValueError("hbond_saturation requires a nonnegative hydrogen-bond weight")
        self.charge_model = charge_model
        self.smoothing = float(smoothing)
        self.unbound_energy = unbound_energy
        self.directional_hb = bool(directional_hb)
        self.hbond_saturation = bool(hbond_saturation)
        self._pair_cache = {}
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
        self._pair_cache.pop(False, None)
        if rec is None:
            self.rec_parameters = None
            super().set_protein(None)
        else:
            prepared, self.rec_parameters = self._prepare(rec)
            self.rec_coordinates = np.array(prepared.coords, dtype=float)
            self._rec_frames = (
                acceptor_frames(self.rec_coordinates, self.rec_parameters["acceptor_rules"], gradient=False)
                if self.directional_hb
                else {}
            )
            super().set_protein(prepared)

    def set_ligand(self, lig):
        self._pair_cache.clear()
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

    def _pair_parameters(self, intra=False):
        """Cache pose-independent pairs, typing and coefficients until preparation changes."""
        first = self.lig_parameters if intra else self.rec_parameters
        if first is None:
            raise ValueError("a receptor is required for intermolecular scoring")
        if intra in self._pair_cache:
            return self._pair_cache[intra]
        first_ids = np.flatnonzero(first["active"])
        ligand_ids = np.flatnonzero(self.lig_parameters["active"])
        mask = (
            self._distant[np.ix_(first_ids, ligand_ids)]
            if intra
            else np.ones((len(first_ids), len(ligand_ids)), dtype=bool)
        )
        row, col = np.nonzero(mask)
        rows, cols = first_ids[row], ligand_ids[col]
        p, q = first["values"][rows], self.lig_parameters["values"][cols]
        q1, q2 = first["charges"][rows], self.lig_parameters["charges"][cols]
        t1, t2 = first["types"][rows], self.lig_parameters["types"][cols]
        hb = ((t1 == "HD") & (q[:, 5] > 0)) | ((t2 == "HD") & (p[:, 5] > 0))
        optimum = np.where(hb, p[:, 4] + q[:, 4], (p[:, 0] + q[:, 0]) / 2)
        depth = np.where(hb, p[:, 5] + q[:, 5], np.sqrt(p[:, 1] * q[:, 1]))
        power = np.where(hb, 10, 6)
        solvation = (p[:, 3] + 0.01097 * np.abs(q1)) * q[:, 2] + (q[:, 3] + 0.01097 * np.abs(q2)) * p[:, 2]
        self._pair_cache[intra] = dict(
            first_ids=first_ids,
            ligand_ids=ligand_ids,
            row=row,
            col=col,
            rows=rows,
            cols=cols,
            hb=hb,
            optimum=optimum,
            depth=depth,
            power=power,
            a=power / (12 - power),
            b=12 / (12 - power),
            electrostatic=332.06371 * q1 * q2,
            solvation=solvation,
            first_types=t1,
        )
        return self._pair_cache[intra]

    def _pair_data(self, coords, intra=False):
        pairs = self._pair_parameters(intra)
        first_coords = coords if intra else self.rec_coordinates
        distances = cdist(first_coords[pairs["first_ids"]], coords[pairs["ligand_ids"]])
        raw_r = distances[pairs["row"], pairs["col"]]
        r = np.maximum(raw_r, 0.1)
        hb, optimum, depth = pairs["hb"], pairs["optimum"], pairs["depth"]
        # Minimum over [r - smoothing/2, r + smoothing/2]: move toward the
        # potential's known minimum, or use its flat well if it is in the interval.
        offset = r - optimum
        smoothed_r = optimum + np.sign(offset) * np.maximum(np.abs(offset) - self.smoothing / 2, 0)
        smoothed_r = np.maximum(smoothed_r, 0.1)
        ratio = optimum / smoothed_r
        power = pairs["power"]
        # General n-m well of depth epsilon at R: m/(n-m)*(R/r)^n - n/(n-m)*(R/r)^m.
        a, b = pairs["a"], pairs["b"]
        high, low = ratio**12, ratio**power
        radial = depth * (a * high - b * low)
        radial_slope = depth * (-12 * a * high + power * b * low) / smoothed_r
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
        electrostatic = pairs["electrostatic"] / (dielectric * r)
        electrostatic_slope = -electrostatic * (1 / r + d_dielectric / dielectric)
        desolvation = pairs["solvation"] * np.exp(-(r**2) / (2 * 3.5**2))
        desolvation_slope = -r / 3.5**2 * desolvation
        terms = np.array([radial * ~hb, radial * hb, electrostatic, desolvation])
        slopes = np.array([radial_slope * ~hb, radial_slope * hb, electrostatic_slope, desolvation_slope])
        return dict(
            rows=pairs["rows"],
            cols=pairs["cols"],
            r=raw_r,
            terms=terms,
            slopes=slopes,
            hb=hb,
            first_coords=first_coords,
            first_types=pairs["first_types"],
            intra=intra,
        )

    def _evaluate(self, coords=None, gradient=False, energy_cap=None, receptor=True):
        """Apply angular/valence terms before caps, then differentiate the result."""
        if energy_cap is not None and (np.isnan(energy_cap) or energy_cap <= 0):
            raise ValueError("energy_cap must be positive or None")
        coords = self._coords(coords)
        data = ([self._pair_data(coords)] if receptor else []) + [self._pair_data(coords, intra=True)]
        offset = len(self.rec_coordinates) if receptor else 0
        full_coords = np.vstack((self.rec_coordinates, coords)) if receptor else coords
        donors, acceptors, radial_hb, contacts = [], [], [], []
        for item in data:
            indices = np.flatnonzero(item["hb"] & (item["terms"][1] < 0))
            first_ids = item["rows"][indices] + (offset if item["intra"] else 0)
            ligand_ids = item["cols"][indices] + offset
            first_donor = item["first_types"][indices] == "HD"
            donors.append(np.where(first_donor, first_ids, ligand_ids))
            acceptors.append(np.where(first_donor, ligand_ids, first_ids))
            radial_hb.append(item["terms"][1, indices].copy())
            contacts.append(indices)
        donors, acceptors = np.concatenate(donors), np.concatenate(acceptors)
        radial_hb = np.concatenate(radial_hb)
        angular = np.ones(len(donors))
        sparse = None
        if self.directional_hb and len(donors):
            frames = acceptor_frames(coords, self.lig_parameters["acceptor_rules"], gradient=gradient)
            frames = {
                index + offset: (vectors, indices + offset, jacobian)
                for index, (vectors, indices, jacobian) in frames.items()
            }
            if receptor:
                frames.update(self._rec_frames)
            ligand_parents = self.lig_parameters["donor_parent"]
            parents = np.where(ligand_parents < 0, -1, ligand_parents + offset)
            if receptor:
                parents = np.concatenate((self.rec_parameters["donor_parent"], parents))
            angular, sparse = angular_factors(full_coords, donors, acceptors, parents, frames, gradient=gradient)
        keep = np.ones(len(donors), dtype=bool)
        if self.hbond_saturation and len(donors):
            capacities = self.lig_parameters["acceptor_capacity"]
            if receptor:
                capacities = np.concatenate((self.rec_parameters["acceptor_capacity"], capacities))
            keep = saturated_contacts(donors, acceptors, radial_hb * angular, capacities)
        coefficients = np.zeros(len(donors))
        outputs, position = [], 0
        for item, indices in zip(data, contacts):
            end = position + len(indices)
            factors_hb = angular[position:end] * keep[position:end]
            item["terms"][1, indices] *= factors_hb
            item["slopes"][1, indices] *= factors_hb
            factors = np.ones(len(item["r"]))
            if energy_cap is not None and np.isfinite(energy_cap) and not self.hbond_saturation:
                pair_energies = self.weights[:4] @ item["terms"]
                if not item["intra"]:
                    pair_energies = np.bincount(item["cols"], weights=pair_energies, minlength=len(coords))[
                        item["cols"]
                    ]
                factors = energy_cap / (energy_cap + np.maximum(pair_energies, 0))
            terms = (item["terms"] * factors).sum(axis=1)
            cartesian = None
            if gradient:
                slope = (self.weights[:4] @ item["slopes"]) * factors**2 * (item["r"] > 0.1)
                rows, cols = item["rows"], item["cols"]
                forces = (coords[cols] - item["first_coords"][rows]) * (slope / np.maximum(item["r"], 1e-12))[:, None]
                cartesian = np.zeros_like(coords)
                for axis in range(3):
                    cartesian[:, axis] = np.bincount(cols, weights=forces[:, axis], minlength=len(coords))
                if item["intra"]:
                    np.add.at(cartesian, rows, -forces)
                coefficients[position:end] = (
                    self.weights[1] * radial_hb[position:end] * keep[position:end] * factors[indices] ** 2
                )
            outputs.append([terms, cartesian])
            position = end
        if gradient and sparse is not None:
            contact, atom, derivatives = sparse
            position = 0
            for output, indices in zip(outputs, contacts):
                end = position + len(indices)
                mask = (contact >= position) & (contact < end) & (atom >= offset)
                np.add.at(output[1], atom[mask] - offset, derivatives[mask] * coefficients[contact[mask], None])
                position = end
        if self.hbond_saturation and energy_cap is not None and np.isfinite(energy_cap):
            # A monotone cap of the complete energy preserves continuity when
            # equally good capacity assignments exchange inter/intraligand
            # contacts. Separate per-atom caps would give those assignments
            # different energies and introduce jumps at the matching boundary.
            energy = float(self.weights[:4] @ sum(output[0] for output in outputs))
            factor = energy_cap / (energy_cap + max(energy, 0))
            for output in outputs:
                output[0] *= factor
                if gradient:
                    output[1] *= factor**2
        return outputs

    def _pairs(self, coords=None, intra=False, gradient=False, energy_cap=None):
        receptor = self.rec_parameters is not None
        if not intra and not receptor:
            raise ValueError("a receptor is required for intermolecular scoring")
        outputs = self._evaluate(coords, gradient, energy_cap, receptor=receptor)
        return outputs[-1 if intra else 0]

    def energy_and_gradient(self, coords, energy_cap=None):
        (inter, g_inter), (intra, g_intra) = self._evaluate(coords, gradient=True, energy_cap=energy_cap)
        return float(self.weights[:4] @ (inter + intra)), g_inter + g_intra

    def score_inter(self, coords=None):
        """Return unweighted vdW, H-bond, electrostatic and desolvation terms."""
        return self._pairs(coords)[0]

    def score_intra(self, coords=None):
        """Return unweighted intraligand terms, excluding 1-2, 1-3 and 1-4."""
        return self._pairs(coords, intra=True)[0]

    def energy(self, coords=None):
        inter, intra = self._evaluate(coords)
        return float(self.weights[:4] @ (inter[0] + intra[0]))

    def score(self, coords=None):
        inter, intra = self._evaluate(coords)
        score = self.weights[:4] @ inter[0] + self.weights[4] * self.num_rotors
        if self.unbound_energy is not None:
            score += self.weights[:4] @ intra[0] - self.unbound_energy
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
            ad4_model=("directional" if self.directional_hb else "radial")
            + ("_saturated" if self.hbond_saturation else ""),
            ad4_directional_hb=int(self.directional_hb),
            ad4_hbond_saturation=int(self.hbond_saturation),
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


class OddtAD4GA(GeneticAlgorithmMixin, OddtAD4Scoring):
    """Directional, saturable AD4 scoring with genetic search and L-BFGS-B."""


class OddtAD4MC(MonteCarloMixin, OddtAD4Scoring):
    """Directional, saturable AD4 scoring with Monte Carlo search and L-BFGS-B."""


class OddtAD4DE(DifferentialEvolutionMixin, OddtAD4Scoring):
    """Directional, saturable AD4 scoring with adaptive DE and L-BFGS-B."""


class OddtAD4ACO(AntColonyMixin, OddtAD4Scoring):
    """Directional, saturable AD4 scoring with continuous ACO and L-BFGS-B."""
