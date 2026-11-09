"""Independent PLANTS-inspired PLP M5 and ChemPLP M3 scoring.

Scientific reference: O. Korb (2008), thesis, Section 3.3.3, equations
3.5--3.20, Tables 3.2--3.5, 3.8 (M3) and 3.9 (M5):
https://nbn-resolving.org/urn:nbn:de:bsz:352-opus-64838
See also Korb, Stuetzle & Exner (2009), doi:10.1021/ci800298z.
No PLANTS, GOLD or other docking implementation code is used.

ODDT perception, rigid receptor/terminal polar H, documented partial SYBYL
torsion coverage and the handling of undetermined metal sites differ from
PLANTS. Scores are empirical score units, not binding energies in kcal/mol.
"""

import numpy as np
from scipy.spatial.distance import cdist

from .base import DockingBase
from ._plp_preparation import (
    INTERACTION_TYPES,
    SYBYL_TORSIONS,
    atom_parameters,
    ligand_clashes,
    ligand_torsions,
    metal_sites,
    validate_torsion,
)
from ._plp_terms import angle_factor, plp_potential, product_factors, ramp, repulsive_potential, torsional_potential
from .search import AntColonyMixin, DifferentialEvolutionMixin, GeneticAlgorithmMixin, MonteCarloMixin

__all__ = [
    "OddtPLPScoring",
    "OddtChemPLPScoring",
    "OddtPLPGA",
    "OddtPLPMC",
    "OddtPLPDE",
    "OddtPLPACO",
    "OddtChemPLPGA",
    "OddtChemPLPMC",
    "OddtChemPLPDE",
    "OddtChemPLPACO",
]

_RADIAL_NAMES = ("plp_hydrogen", "plp_metal", "plp_buried", "plp_nonpolar", "plp_repulsive")
_RADIAL_POINTS = np.array([[2.3, 2.6, 3.1, 3.4], [1.4, 2.2, 2.6, 2.8], [3.4, 3.6, 4.5, 5.5], [3.4, 3.6, 4.5, 5.5]])
_PLP_PARAMETERS = dict(zip(_RADIAL_NAMES, (-2.0, -4.0, -0.05, -0.4, 0.5)), clash=1.0, torsion=1.0)
_CHEMPLP_PARAMETERS = dict(
    zip(_RADIAL_NAMES, (-1.0, -1.0, -0.1, -0.4, 1.0)),
    clash=1.0,
    torsion=2.0,
    hydrogen=-3.0,
    charged_hydrogen_factor=2.0,
    cho=-3.0,
    metal=-6.0,
    charged_metal_factor=2.0,
)


class OddtPLPScoring(DockingBase):
    """PLP contact, ligand clash and cosine-torsion scoring with analytic gradients.

    Parameters
    ----------
    rec : oddt.toolkit.Molecule or None
        Rigid, appropriately protonated receptor; mark proteins with protein=True.
    lig : oddt.toolkit.Molecule or None, optional
        Three-dimensional ligand. Hydrogens are added to private copies.
    box : array-like, shape (2, 3), optional
        Lower and upper Cartesian search corners, in Angstroms.
    box_size : float, optional
        Angstroms per translation gene.
    parameters : mapping or None, optional
        Overrides for the named parameters in ``default_parameters``. Defaults
        are PLP M5 (ChemPLP M3 in the subclass), including clash and torsion weights.
    torsion_parameters : mapping or None, optional
        Additional SYBYL atom-type quadruples mapped to (barrier V, sign, period).
        Wildcards are '*'; more specific matches win, with later entries breaking
        ties. Both directions are matched. Extends the documented published subset.
    torsion_fallback : tuple or None, optional
        Default (0.2, 1, 3) is a generic cosine barrier, not a complete Tripos
        parameterization. None requires explicit coverage for every torsion.
        ``torsion_fallbacks_`` lists every quadruple using this approximation.
    metal_cutoff : float, optional
        Receptor N/O/S coordination-neighbor cutoff in Angstroms, default 2.8.
        Used to fit ChemPLP coordination sites; PLP itself uses radial contacts.
    sample_callback : callable or None, optional
        Read-only observer called after each local minimization.

    Notes
    -----
    Evaluates the published contact/clash equations directly and counts each
    intra-ligand clash pair once. Torsions include all outer-atom combinations,
    including hydrogens, around the searched heavy-atom rotors. Rings, conjugated
    bonds and terminal polar H retain their supplied geometry. The SYBYL torsion
    subset is transcribed from the ADF QMMM Manual (2018), printed pp. 31--32;
    additional parameters or a strict lookup can be supplied explicitly.

    Coordinates include all explicit atoms, as for OddtAD4Scoring. Only heavy
    atoms participate in PLP contacts/clashes. ``score_inter`` and ``score_intra``
    return already weighted contributions. Lower scores are preferred; the
    reported score is the total contact plus internal strain score. No affinity
    conversion or historical -20 search offset is applied.
    """

    scoring_name = "plp"
    parameterization = "PLP M5"
    score_units = "empirical score units"
    directional_scoring = False
    score_all_atoms = True
    score_metadata_prefixes = ("docking_", "plp_")
    default_parameters = _PLP_PARAMETERS
    inter_term_names = _RADIAL_NAMES
    intra_term_names = ("clash", "torsion")
    term_names = inter_term_names + intra_term_names

    def __init__(
        self,
        rec,
        lig=None,
        box=None,
        box_size=1.0,
        parameters=None,
        torsion_parameters=None,
        torsion_fallback=(0.2, 1, 3),
        metal_cutoff=2.8,
        sample_callback=None,
    ):
        self.parameters = dict(self.default_parameters)
        if parameters is not None:
            unknown = set(parameters) - self.parameters.keys()
            if unknown:
                raise ValueError("Unknown PLP parameter(s): " + ", ".join(sorted(unknown)))
            self.parameters.update(parameters)
        for name, value in self.parameters.items():
            if not np.isscalar(value) or not np.isfinite(value):
                raise ValueError("PLP parameters must be finite scalar values")
            attractive = name in self.inter_term_names and name != "plp_repulsive"
            if (attractive and value > 0) or (not attractive and value < 0):
                raise ValueError("PLP attraction parameters must be nonpositive; other parameters nonnegative")
            self.parameters[name] = float(value)
        if not np.isfinite(metal_cutoff) or metal_cutoff <= 0:
            raise ValueError("metal_cutoff must be positive and finite")
        self.metal_cutoff = float(metal_cutoff)
        self.torsion_parameters = dict(SYBYL_TORSIONS)
        if torsion_parameters is not None:
            for pattern, value in torsion_parameters.items():
                if isinstance(pattern, str) or len(pattern) != 4 or not all(isinstance(p, str) for p in pattern):
                    raise ValueError("torsion parameter keys must be four SYBYL atom-type strings")
                self.torsion_parameters[tuple(pattern)] = validate_torsion(value)
        self.torsion_fallback = None if torsion_fallback is None else validate_torsion(torsion_fallback)
        self.torsion_fallbacks_ = []
        self._pair_cache = None
        self._contacts = None
        self.rec_parameters = None
        self.lig_parameters = None
        self.metal_geometries_ = []
        super().__init__(rec, lig, box, box_size, sample_callback=sample_callback)

    @staticmethod
    def _prepare(molecule):
        prepared = molecule.clone
        prepared.protein = molecule.protein
        prepared.title = molecule.title
        prepared.data.update(dict(molecule.data))
        prepared.addh()
        if not np.isfinite(prepared.coords).all():
            raise ValueError("PLP requires finite three-dimensional coordinates")
        return prepared

    def set_protein(self, rec):
        self._pair_cache = self._contacts = None
        if rec is None:
            self.rec_parameters = None
            self.metal_geometries_ = []
            super().set_protein(None)
            return
        prepared = self._prepare(rec)
        self.rec_coordinates = np.asarray(prepared.coords, dtype=float).copy()
        self.rec_parameters = atom_parameters(prepared, receptor=True)
        self.metal_geometries_ = (
            metal_sites(self.rec_coordinates, self.rec_parameters, self.metal_cutoff)
            if self.directional_scoring
            else []
        )
        super().set_protein(prepared)

    def set_ligand(self, lig):
        self._pair_cache = self._contacts = None
        prepared = self._prepare(lig)
        self.lig_parameters = atom_parameters(prepared)
        super().set_ligand(prepared)
        self.scoring_coordinates = np.asarray(prepared.coords, dtype=float).copy()
        self._clash_pairs = ligand_clashes(self.lig_parameters, self.rotors)
        self._torsion_indices, self._torsion_values, self.torsion_fallbacks_ = ligand_torsions(
            self.lig_parameters,
            self.rotors,
            self.torsion_parameters,
            self.torsion_fallback,
        )

    def _coords(self, coords):
        if self.lig_parameters is None:
            raise ValueError("a ligand is required for PLP scoring")
        coords = self.scoring_coordinates if coords is None else np.asarray(coords, dtype=float)
        if coords.shape != self.scoring_coordinates.shape or not np.isfinite(coords).all():
            raise ValueError("PLP coords must contain finite coordinates for every explicit ligand atom")
        return coords

    def set_coords(self, coords):
        coords = self._coords(coords)
        self.scoring_coordinates = coords.copy()
        self.lig_dict["coords"] = coords[self._heavy_mask]

    def _radial(self, coords, gradient):
        terms = np.zeros(len(self.inter_term_names))
        cartesian = np.zeros_like(coords) if gradient else None
        if self.rec_parameters is None:
            return terms, cartesian
        if self._pair_cache is None:
            rec = self.rec_parameters["heavy"]
            lig = self.lig_parameters["heavy"]
            kinds = INTERACTION_TYPES[
                self.lig_parameters["types"][lig][None, :], self.rec_parameters["types"][rec][:, None]
            ]
            self._pair_cache = rec, lig, kinds
        rec, lig, kinds = self._pair_cache
        distances = cdist(self.rec_coordinates[rec], coords[lig])
        rows, cols = np.nonzero(distances < 5.5)
        r, kind = distances[rows, cols], kinds[rows, cols]
        values, slopes = np.zeros(len(r)), np.zeros(len(r))
        for i, name in enumerate(_RADIAL_NAMES):
            mask = kind == i
            if i == 4:
                values[mask], slopes[mask] = repulsive_potential(r[mask], self.parameters[name])
            else:
                values[mask], slopes[mask] = plp_potential(r[mask], *_RADIAL_POINTS[i], self.parameters[name])
        terms[:5] = np.bincount(kind, weights=values, minlength=5)
        if gradient:
            forces = (coords[lig[cols]] - self.rec_coordinates[rec[rows]]) * (slopes / np.maximum(r, 1e-12))[:, None]
            np.add.at(cartesian, lig[cols], forces)
        return terms, cartesian

    def _internal(self, coords, gradient):
        rows, cols, cutoff = self._clash_pairs
        delta = coords[cols] - coords[rows]
        squared = np.sum(delta**2, axis=1)
        clash = float(50 * self.parameters["clash"] * np.maximum(1 - squared / cutoff**2, 0).sum())
        values = self._torsion_values
        torsion, cartesian = torsional_potential(
            coords,
            self._torsion_indices,
            values[:, 0],
            values[:, 1],
            values[:, 2].astype(int),
            gradient,
        )
        weight = self.parameters["torsion"]
        if gradient:
            cartesian *= weight
            forces = delta * np.where(squared < cutoff**2, -100 * self.parameters["clash"] / cutoff**2, 0)[:, None]
            np.add.at(cartesian, cols, forces)
            np.add.at(cartesian, rows, -forces)
        return np.array([clash, weight * torsion]), cartesian

    def _evaluate(self, coords=None, gradient=False):
        coords = self._coords(coords)
        inter, forces = self._radial(coords, gradient)
        intra, internal_forces = self._internal(coords, gradient)
        if gradient:
            forces += internal_forces
        return inter, intra, forces

    def energy_and_gradient(self, coords, energy_cap=None):
        if energy_cap is not None and (np.isnan(energy_cap) or energy_cap <= 0):
            raise ValueError("energy_cap must be positive or None")
        inter, intra, forces = self._evaluate(coords, gradient=True)
        total = float(inter.sum() + intra.sum())
        if energy_cap is not None and np.isfinite(energy_cap) and total > 0:
            factor = energy_cap / (energy_cap + total)
            return total * factor, forces * factor**2
        return total, forces

    def energy(self, coords=None):
        inter, intra, _ = self._evaluate(coords)
        return float(inter.sum() + intra.sum())

    def score(self, coords=None):
        return self.energy(coords)

    def score_inter(self, coords=None):
        """Return weighted intermolecular contributions in inter_term_names order."""
        if self.rec_parameters is None:
            raise ValueError("a receptor is required for intermolecular scoring")
        return self._evaluate(coords)[0]

    def score_intra(self, coords=None):
        """Return the weighted ligand clash and torsion contributions."""
        return self._internal(self._coords(coords), gradient=False)[0]

    def score_components(self, coords=None):
        inter, intra, _ = self._evaluate(coords)
        return {name: float(value) for name, value in zip(self.term_names, np.r_[inter, intra])}

    def _pose_metadata(self, result, total):
        data = super()._pose_metadata(result, total)
        prefix = self.scoring_name + "_"
        data.update(
            {
                prefix + "score": float(result.score),
                prefix + "total": float(total),
                prefix + "model": self.parameterization,
                "docking_score_units": self.score_units,
                prefix + "torsion_fallback_terms": len(self.torsion_fallbacks_),
            }
        )
        return data


class OddtChemPLPScoring(OddtPLPScoring):
    """ChemPLP M3 with directional, charged, CH--O and metal-site interactions.

    Uses the same constructor as OddtPLPScoring. Donor hydrogen/acceptor
    neighbor angles follow Korb equations 3.14--3.16. A donor hydrogen can
    contribute only its best interaction. Mg/Ca use radial metal interactions;
    other receptor metals use vacant tetrahedral/octahedral sites fitted to
    N/O/S neighbors within metal_cutoff. Insufficient or collinear coordination
    neighbors use the orientation-independent Mg/Ca form, reported explicitly
    in ``metal_geometries_``. Each fitted site accepts one best interaction.
    All geometry derivatives include the defining ligand neighbors.

    The receptor and explicit water molecules are fixed. Flexible protein terms,
    displaceable-water occupancy and PLANTS' terminal-H search are not modeled.
    """

    scoring_name = "chemplp"
    parameterization = "ChemPLP M3"
    directional_scoring = True
    score_metadata_prefixes = ("docking_", "chemplp_")
    default_parameters = _CHEMPLP_PARAMETERS
    inter_term_names = _RADIAL_NAMES + ("hydrogen", "charged_hydrogen", "cho", "metal", "charged_metal")
    term_names = inter_term_names + OddtPLPScoring.intra_term_names

    def _contact_topology(self):
        if self._contacts is not None:
            return self._contacts
        rec, lig = self.rec_parameters, self.lig_parameters
        offset = len(self.rec_coordinates)
        neighbors = [indices[rec["types"][indices] != 4] for indices in rec["neighbors"]]
        neighbors += [indices + offset for indices in lig["neighbors"]]
        parents = np.r_[rec["donor_parent"], np.where(lig["donor_parent"] >= 0, lig["donor_parent"] + offset, -1)]
        positive, negative = np.r_[rec["positive"], lig["positive"]], np.r_[rec["negative"], lig["negative"]]
        groups = []
        for key, weak in (("hydrogens", False), ("cho_hydrogens", True)):
            hp, hl = rec[key], lig[key] + offset
            ap, al = rec["acceptors"], lig["acceptors"]
            if weak:
                ap = ap[rec["atomicnum"][ap] == 8]
                al = al[lig["atomicnum"][al] == 8]
            al = al + offset
            hs = np.r_[np.repeat(hp, len(al)), np.repeat(hl, len(ap))]
            acc = np.r_[np.tile(al, len(hp)), np.tile(ap, len(hl))]
            groups.append((hs, acc, weak))
        self._contacts = offset, neighbors, parents, positive, negative, groups
        return self._contacts

    @staticmethod
    def _add_sparse(cartesian, offset, ids, forces):
        ids = np.asarray(ids)
        mask = ids >= offset
        np.add.at(cartesian, ids[mask] - offset, forces[mask])

    def _hydrogen_bonds(self, full, gradient):
        offset, neighbors, parents, positive, negative, groups = self._contact_topology()
        terms = np.zeros(len(self.inter_term_names))
        forces = np.zeros_like(full[offset:]) if gradient else None
        for hydrogens, acceptors, weak in groups:
            difference = full[acceptors] - full[hydrogens]
            distances = np.linalg.norm(difference, axis=1)
            deviation = distances - (2.35 if weak else 1.85)
            radial, slopes = ramp(np.abs(deviation), 0.25, 0.65)
            active = radial > 0
            hs, acc = hydrogens[active], acceptors[active]
            radial, slopes = radial[active], slopes[active] * np.sign(deviation[active])
            vectors = difference[active] / distances[active, None]
            components = np.full(len(hs), 7 if weak else 5, dtype=int)
            charged = positive[parents[hs]] & negative[acc]
            if not weak:
                components[charged] = 6
            weights = np.full(len(hs), self.parameters["cho" if weak else "hydrogen"])
            if not weak:
                weights[charged] *= self.parameters["charged_hydrogen_factor"]
            values = np.zeros(len(hs))
            sparse = []
            for acceptor in np.unique(acc):
                contacts = np.flatnonzero(acc == acceptor)
                hydrogen = hs[contacts]
                parent = parents[hydrogen]
                alpha, ga = angle_factor(
                    full[parent], full[hydrogen], full[acc[contacts]], 50 if weak else 30, 100 if weak else 80, gradient
                )
                factors = [radial[contacts], alpha]
                local = []
                if gradient:
                    local = [
                        [
                            (hydrogen, -vectors[contacts] * slopes[contacts, None]),
                            (acc[contacts], vectors[contacts] * slopes[contacts, None]),
                        ],
                        [(parent, ga[0]), (hydrogen, ga[1]), (acc[contacts], ga[2])],
                    ]
                for neighbor in neighbors[acceptor]:
                    beta, gb = angle_factor(
                        full[np.full(len(contacts), neighbor)], full[acc[contacts]], full[hydrogen], 80, 100, gradient
                    )
                    factors.append(beta)
                    if gradient:
                        local.append(
                            [(np.full(len(contacts), neighbor), gb[0]), (acc[contacts], gb[1]), (hydrogen, gb[2])]
                        )
                product, partials = product_factors(factors)
                values[contacts] = product * weights[contacts]
                if gradient:
                    for partial, entries in zip(partials, local):
                        for ids, derivative in entries:
                            sparse.append((contacts, ids, derivative * (weights[contacts] * partial)[:, None]))
            order = np.lexsort((np.arange(len(hs)), values, hs))
            chosen = order[np.r_[True, hs[order][1:] != hs[order][:-1]]] if len(hs) else order
            selected = np.zeros(len(hs), dtype=bool)
            selected[chosen[values[chosen] < 0]] = True
            terms += np.bincount(components[selected], weights=values[selected], minlength=len(terms))
            if gradient:
                for contacts, ids, derivative in sparse:
                    keep = selected[contacts]
                    self._add_sparse(forces, offset, ids[keep], derivative[keep])
        return terms, forces

    def _metal_bonds(self, full, gradient):
        offset, neighbors, _, _, negative, _ = self._contact_topology()
        terms = np.zeros(len(self.inter_term_names))
        forces = np.zeros_like(full[offset:]) if gradient else None
        acceptors = self.lig_parameters["acceptors"] + offset
        for description in self.metal_geometries_:
            metal = description["atom"]
            radial_mode = description["geometry"].startswith("radial")
            sites = full[metal : metal + 1] if radial_mode else description["sites"]
            for point in sites:
                difference = full[acceptors] - point
                distances = np.linalg.norm(difference, axis=1)
                radial, slopes = ramp(distances, 2.6 if radial_mode else 0.6, 3.0 if radial_mode else 0.8)
                active = (radial > 0) & (np.linalg.norm(full[acceptors] - full[metal], axis=1) > 1e-10)
                ids = np.flatnonzero(active)
                values, derivatives = [], []
                for index in ids:
                    acceptor = acceptors[index]
                    factors = [radial[index]]
                    local = [[(acceptor, slopes[index] * difference[index] / max(distances[index], 1e-12))]]
                    for neighbor in neighbors[acceptor]:
                        beta, gb = angle_factor(full[neighbor], full[acceptor], full[metal], 80, 90, gradient)
                        factors.append(beta)
                        if gradient:
                            local.append([(neighbor, gb[0]), (acceptor, gb[1])])
                    product, partials = product_factors(factors)
                    weight = self.parameters["metal"]
                    if negative[acceptor]:
                        weight *= self.parameters["charged_metal_factor"]
                    values.append(float(weight * product))
                    derivatives.append(
                        [
                            (atom, weight * partial * derivative)
                            for partial, entries in zip(partials, local)
                            for atom, derivative in entries
                        ]
                        if gradient
                        else []
                    )
                chosen = range(len(ids)) if radial_mode else ([int(np.argmin(values))] if len(ids) else [])
                for choice in chosen:
                    if values[choice] >= 0:
                        continue
                    acceptor = acceptors[ids[choice]]
                    terms[9 if negative[acceptor] else 8] += values[choice]
                    if gradient:
                        for atom, derivative in derivatives[choice]:
                            if atom >= offset:
                                forces[atom - offset] += derivative
        return terms, forces

    def _evaluate(self, coords=None, gradient=False):
        inter, intra, forces = super()._evaluate(coords, gradient)
        if self.rec_parameters is not None:
            full = np.vstack((self.rec_coordinates, self._coords(coords)))
            for evaluate in (self._hydrogen_bonds, self._metal_bonds):
                terms, extra = evaluate(full, gradient)
                inter += terms
                if gradient:
                    forces += extra
        return inter, intra, forces


class OddtPLPGA(GeneticAlgorithmMixin, OddtPLPScoring):
    """PLP scoring with genetic search and L-BFGS-B refinement."""


class OddtPLPMC(MonteCarloMixin, OddtPLPScoring):
    """PLP scoring with Monte Carlo search and L-BFGS-B refinement."""


class OddtPLPDE(DifferentialEvolutionMixin, OddtPLPScoring):
    """PLP scoring with adaptive differential evolution and L-BFGS-B."""


class OddtChemPLPGA(GeneticAlgorithmMixin, OddtChemPLPScoring):
    """ChemPLP scoring with genetic search and L-BFGS-B refinement."""


class OddtChemPLPMC(MonteCarloMixin, OddtChemPLPScoring):
    """ChemPLP scoring with Monte Carlo search and L-BFGS-B refinement."""


class OddtChemPLPDE(DifferentialEvolutionMixin, OddtChemPLPScoring):
    """ChemPLP scoring with adaptive differential evolution and L-BFGS-B."""


class OddtPLPACO(AntColonyMixin, OddtPLPScoring):
    """PLP scoring with continuous ant-colony search and L-BFGS-B."""


class OddtChemPLPACO(AntColonyMixin, OddtChemPLPScoring):
    """ChemPLP scoring with continuous ant-colony search and L-BFGS-B."""
