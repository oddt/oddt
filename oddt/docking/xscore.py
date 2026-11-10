"""Independent post-docking X-Score, using the published 2002 calibration.

Equations 2--12 and Table 1 in Wang, Lai & Wang (2002), JCAMD 16:11--26,
https://doi.org/10.1023/A:1016357811882. Atomic HM scales: Wang, Gao & Lai
(2000), Perspectives in Drug Discovery and Design 19:47--66, Table 1.
No X-Score/XLOGP implementation code or parameter files are used.
"""

import numpy as np
from scipy.spatial.distance import cdist

from ._xlogp2 import bond_order, classify_xlogp2
from ._xscore_terms import buried_surface, hydrogen_bonds, hydrophobic_pairs, vdw_term

__all__ = ["OddtXScoreScoring"]

# Table 1: VDW, HB, hydrophobic term, rotor, intercept; pKd units.
XSCORE_COEFFICIENTS = {
    "hp": (-0.00096, 0.412, 0.0373, -0.100, 2.78),
    "hm": (-0.00214, 0.311, 0.602, -0.169, 3.10),
    "hs": (-0.00201, 0.307, 0.00710, -0.159, 2.69),
}
XSCORE_RADII = {6: 1.9, 7: 1.8, 8: 1.7, 9: 1.5, 15: 2.1, 16: 2.0, 17: 1.8, 35: 2.0, 53: 2.2}


def _parameters(molecule, remove_water, hydrogen_radius, logp, need_logp, receptor):
    data = molecule.atom_dict
    heavy = data["atomicnum"] != 1
    if remove_water:
        heavy &= ~np.isin(data["resname"], ["HOH", "WAT", "H2O"])
    indices = np.flatnonzero(heavy)
    if not len(indices):
        raise ValueError("X-Score requires at least one retained heavy atom")
    if np.any(data["atomicnum"][indices] <= 0):
        raise ValueError("X-Score does not support dummy/query atoms")
    if not receptor and data["ismetal"][indices].any():
        raise ValueError("X-Score supports receptor metals, not ligand metal ions")
    lookup = {int(atom): i for i, atom in enumerate(indices)}
    neighbors = [
        np.array([lookup[n.idx0] for n in molecule.atoms[int(i)].neighbors if n.idx0 in lookup], int) for i in indices
    ]
    z = data["atomicnum"][indices]
    unsupported = ~np.isin(z, list(XSCORE_RADII)) & ~data["ismetal"][indices]
    if unsupported.any():
        raise ValueError("Unsupported X-Score element(s): %s" % np.unique(z[unsupported]))
    radii = np.array([XSCORE_RADII.get(int(value), 1.2) for value in z])
    donor = np.where(np.isin(z, [7, 8]), data["numhs"][indices], 0).astype(int)
    for j in np.flatnonzero(data["ismetal"][indices]):
        # Coordination capacities are an explicit adaptation: not given in the paper.
        donor[j] = {30: 4, 20: 8}.get(int(z[j]), 6)
    acceptor = data["isacceptor"][indices] & ((z == 8) | ((z == 7) & (data["hybridization"][indices] <= 2)))
    capacity = np.where(acceptor, np.where(z == 8, 2, 1) + (data["formalcharge"][indices] < 0), 0)
    hydrophobic = np.isin(z, [9, 17, 35, 53])
    for j in np.flatnonzero(z == 6):
        hydrophobic[j] = all(z[n] == 6 for n in neighbors[j])
    if logp is not None:
        scales = np.asarray(logp, float)
        if scales.shape != (len(indices),) or not np.isfinite(scales).all():
            raise ValueError("atomic logP scales must give one finite value per retained heavy atom")
        types, charged = np.full(len(indices), -1, int), []
        scales = scales.copy()
    elif need_logp:
        types, scales, charged = classify_xlogp2(molecule, indices)
    else:
        types, scales, charged = np.full(len(indices), -1, int), np.zeros(len(indices)), []
    # All explicit atoms contribute occlusion to SAS; only surfaces attached to
    # hydrophobic heavy atoms contribute area. H area belongs to its parent.
    surface_indices, surface_radii, surface_hydrophobic = [], [], []
    for atom in molecule.atoms:
        i = atom.idx0
        if i in lookup:
            j = lookup[i]
            surface_indices.append(i)
            surface_radii.append(radii[j])
            surface_hydrophobic.append(hydrophobic[j])
        elif atom.atomicnum == 1:
            parents = [n.idx0 for n in atom.neighbors if n.idx0 in lookup]
            if parents:
                if len(parents) != 1:
                    raise ValueError("each X-Score surface hydrogen must have exactly one heavy parent")
                surface_indices.append(i)
                surface_radii.append(hydrogen_radius)
                surface_hydrophobic.append(hydrophobic[lookup[parents[0]]])
    return dict(
        heavy=indices,
        radii=radii,
        neighbors=neighbors,
        donor=donor,
        acceptor=capacity,
        hydrophobic=hydrophobic,
        logp=scales,
        xlogp2_types=types,
        charged_atoms=charged,
        surface_indices=np.array(surface_indices, int),
        surface_radii=np.array(surface_radii),
        surface_hydrophobic=np.array(surface_hydrophobic, bool),
    )


def rotor_count(molecule):
    """Eq. 5: acyclic sp3--sp3/sp2 rotors, with the crossed-rotor correction."""
    data = molecule.atom_dict
    adjacency = [set(n.idx0 for n in atom.neighbors if n.atomicnum != 1) for atom in molecule.atoms]
    counts = np.zeros(len(data), int)
    for bond in molecule.bonds:
        a, b = (atom.idx0 for atom in bond.atoms)
        if bond_order(bond) != 1 or len(adjacency[a]) < 2 or len(adjacency[b]) < 2:
            continue
        ha, hb = data["hybridization"][[a, b]]
        if not ((ha == 3 and hb in (2, 3)) or (hb == 3 and ha in (2, 3))):
            continue
        visited, queue = {a}, [a]
        while queue:
            current = queue.pop()
            for nxt in adjacency[current] - visited:
                if {current, nxt} == {a, b}:
                    continue
                visited.add(nxt)
                queue.append(nxt)
        if b not in visited:
            counts[[a, b]] += 1
    return float(np.where(counts > 2, 0.5, 0.5 * counts).sum())


class OddtXScoreScoring:
    """Post-docking HPScore, HMScore and HSScore consensus, in pKd units.

    Parameters
    ----------
    rec : oddt.toolkit.Molecule or None, optional
        Receptor in the ligand coordinate frame. Set protein=True for proteins.
    lig : oddt.toolkit.Molecule or None, optional
        Three-dimensional ligand. Private copies are hydrogenated.
    models : str or sequence of str, optional
        'consensus' (default) or a selection of 'hp', 'hm', 'hs'. Scores average
        the enabled variants. Higher pKd is preferred.
    coefficients : mapping or None, optional
        Override model rows (VDW, HB, hydrophobic, rotor, intercept). Defaults
        are Table 1 of the 2002 paper, not the later recalibrated X-Score releases.
    surface_spacing : float, optional
        Positive SAS quadrature spacing in Angstroms, default 0.5.
    probe_radius : float, optional
        Nonnegative solvent-probe radius, default 1.5 Angstroms.
    hydrogen_radius : float, optional
        Positive SAS hydrogen radius, default 1.2 Angstroms. The paper does not
        specify this radius; it is an exposed surface-preparation adaptation.
    remove_water : bool, optional
        Remove HOH/WAT/H2O receptor residues and their H, default True.
    receptor_logp : array-like or None, optional
        Override atomic HM scales, in retained receptor heavy-atom order.
    ligand_logp : array-like or None, optional
        Override atomic HM scales, in ligand heavy-atom order. Must match each
        ligand supplied subsequently; omit for automatic XLOGP2 typing.

    Notes
    -----
    This is a rescoring API, not a DockingBase/search engine. HS uses hard SAS
    occlusion and HM a hard 6-A environment/threshold test; no smooth search
    surrogate or analytic gradient is substituted. Coordinates passed explicitly
    include all atoms in the prepared ligand, available as scoring_coordinates.

    Atomic HM values come from all 90 published XLOGP2 types, without molecular
    correction factors. Charged topologies are projected to the neutral lookup
    scheme; affected atoms are listed in ``xlogp2_charged_atoms_``. Metals contribute
    zero to HM. Unknown organic topologies raise instead of using a numeric
    fallback. ODDT perception can differ from X-Score's original preparation.

    H-bonds use heavy-neighbor centroid roots, separate donor-H/lone-pair
    capacities, and deterministic strongest-first saturation. Rootless or
    coincident-root atoms use an isotropic angle. Metal donor capacities are
    Zn 4, Ca 8, other metals 6. These choices are explicit interpretations where
    the publication does not specify assignment or missing-root details.

    HS uses independent Fibonacci quadrature at approximately the published
    0.5-A spacing, with explicit H areas attributed to hydrophobic parents.
    Other retained ligand/protein atoms occlude those surfaces. The quadrature
    frame follows the molecular geometry. No upstream code is reused, and
    numerical identity with X-Score binaries is not claimed.
    """

    scoring_name = "xscore"
    score_units = "pKd"
    parameterization = "Wang2002_Table1_XLOGP2000_Table1"
    default_coefficients = XSCORE_COEFFICIENTS

    def __init__(
        self,
        rec=None,
        lig=None,
        models="consensus",
        coefficients=None,
        surface_spacing=0.5,
        probe_radius=1.5,
        hydrogen_radius=1.2,
        remove_water=True,
        receptor_logp=None,
        ligand_logp=None,
    ):
        self.models = (
            (("hp", "hm", "hs") if models == "consensus" else (models,)) if isinstance(models, str) else tuple(models)
        )
        if not self.models or len(set(self.models)) != len(self.models) or set(self.models) - set(XSCORE_COEFFICIENTS):
            raise ValueError("models must select unique variants from 'hp', 'hm', 'hs'")
        self.coefficients = {name: np.array(row) for name, row in XSCORE_COEFFICIENTS.items()}
        if coefficients is not None:
            if set(coefficients) - set(self.coefficients):
                raise ValueError("unknown X-Score coefficient model")
            for name, values in coefficients.items():
                array = np.asarray(values, float)
                if array.shape != (5,) or not np.isfinite(array).all():
                    raise ValueError("each coefficient row must contain five finite values")
                self.coefficients[name] = array.copy()
        for name, value, allow_zero in (
            ("surface_spacing", surface_spacing, False),
            ("probe_radius", probe_radius, True),
            ("hydrogen_radius", hydrogen_radius, False),
        ):
            if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
                raise ValueError(name + " must be finite and " + ("nonnegative" if allow_zero else "positive"))
        self.surface_spacing, self.probe_radius, self.hydrogen_radius = map(
            float, (surface_spacing, probe_radius, hydrogen_radius)
        )
        self.remove_water = bool(remove_water)
        self.receptor_logp = None if receptor_logp is None else np.asarray(receptor_logp, float).copy()
        self.ligand_logp = None if ligand_logp is None else np.asarray(ligand_logp, float).copy()
        self._receptor = self._ligand = None
        self.rec_parameters = self.lig_parameters = None
        self.xlogp2_charged_atoms_ = {"receptor": [], "ligand": []}
        self.set_protein(rec)
        if lig is not None:
            self.set_ligand(lig)

    @staticmethod
    def _prepare(molecule):
        prepared = molecule.clone
        prepared.protein, prepared.title = molecule.protein, molecule.title
        prepared.data.update(dict(molecule.data))
        prepared.addh()
        if not np.isfinite(prepared.coords).all():
            raise ValueError("X-Score requires finite three-dimensional coordinates")
        return prepared

    def set_protein(self, rec):
        """Prepare a private receptor copy, or clear it with None."""
        if rec is None:
            self._receptor = self.rec_parameters = None
            self.xlogp2_charged_atoms_["receptor"] = []
            return
        prepared = self._prepare(rec)
        parameters = _parameters(
            prepared, self.remove_water, self.hydrogen_radius, self.receptor_logp, "hm" in self.models, True
        )
        self._receptor, self.rec_parameters = prepared, parameters
        self.rec_coordinates = np.asarray(prepared.coords, float).copy()
        self.xlogp2_charged_atoms_["receptor"] = list(parameters["charged_atoms"])

    def set_ligand(self, lig):
        """Prepare a private ligand copy and its topology-dependent rotor term."""
        prepared = self._prepare(lig)
        parameters = _parameters(prepared, False, self.hydrogen_radius, self.ligand_logp, "hm" in self.models, False)
        rotors = rotor_count(prepared)
        self._ligand, self.lig_parameters = prepared, parameters
        self.scoring_coordinates = np.asarray(prepared.coords, float).copy()
        self.rotor_count_ = rotors
        self.xlogp2_charged_atoms_["ligand"] = list(parameters["charged_atoms"])

    def _coords(self, coords):
        if self.lig_parameters is None:
            raise ValueError("a ligand is required for X-Score")
        result = self.scoring_coordinates if coords is None else np.asarray(coords, float)
        if result.shape != self.scoring_coordinates.shape or not np.isfinite(result).all():
            raise ValueError("coords must provide finite coordinates for every prepared ligand atom")
        return result

    def set_coords(self, coords):
        """Set all-atom scoring coordinates without changing the topology."""
        self.scoring_coordinates = self._coords(coords).copy()

    def raw_terms(self, coords=None):
        """Return unweighted terms; disabled hydrophobic variants have zero terms."""
        coords = self._coords(coords)
        if self.rec_parameters is None:
            raise ValueError("a receptor is required for X-Score")
        rec, lig = self.rec_parameters, self.lig_parameters
        rc, lc = self.rec_coordinates[rec["heavy"]], coords[lig["heavy"]]
        distances = cdist(rc, lc)
        radii_sum = rec["radii"][:, None] + lig["radii"][None, :]
        hb, _ = hydrogen_bonds(rc, lc, rec, lig, distances, radii_sum)
        terms = dict(
            vdw=float(vdw_term(distances, radii_sum).sum()),
            hbond=hb,
            hydrophobic_pair=0.0,
            hydrophobic_match=0.0,
            hydrophobic_surface=0.0,
            rotor=self.rotor_count_,
        )
        if "hp" in self.models:
            mask = rec["hydrophobic"][:, None] & lig["hydrophobic"][None, :]
            terms["hydrophobic_pair"] = float(hydrophobic_pairs(distances, radii_sum)[mask].sum())
        if "hm" in self.models:
            environment = np.sum((distances <= 6) * rec["logp"][:, None], axis=0)
            terms["hydrophobic_match"] = float(lig["logp"][lig["hydrophobic"] & (environment > -0.5)].sum())
        if "hs" in self.models:
            terms["hydrophobic_surface"] = buried_surface(
                self.rec_coordinates[rec["surface_indices"]],
                coords[lig["surface_indices"]],
                rec["surface_radii"],
                lig["surface_radii"],
                lig["surface_hydrophobic"],
                self.probe_radius,
                self.surface_spacing,
            )
        return terms

    def _variant_scores(self, terms):
        names = {"hp": "hydrophobic_pair", "hm": "hydrophobic_match", "hs": "hydrophobic_surface"}
        return {
            model: float(
                self.coefficients[model]
                @ np.array([terms["vdw"], terms["hbond"], terms[names[model]], terms["rotor"], 1])
            )
            for model in self.models
        }

    def score_variants(self, coords=None):
        """Return enabled HP/HM/HS scores in pKd units; higher is better."""
        return self._variant_scores(self.raw_terms(coords))

    def score_components(self, coords=None):
        """Return weighted consensus contributions, including the intercept."""
        terms = self.raw_terms(coords)
        rows = np.array([self.coefficients[model] for model in self.models])
        count = len(rows)
        result = {name: float(rows[:, i].mean() * terms[name]) for name, i in [("vdw", 0), ("hbond", 1), ("rotor", 3)]}
        result["constant"] = float(rows[:, 4].mean())
        for model, name in [("hp", "hydrophobic_pair"), ("hm", "hydrophobic_match"), ("hs", "hydrophobic_surface")]:
            result[name] = float(self.coefficients[model][2] * terms[name] / count) if model in self.models else 0.0
        return result

    def score(self, coords=None):
        """Return the mean enabled-variant pKd score; higher is better."""
        return float(np.mean(list(self.score_variants(coords).values())))

    def score_ligand(self, ligand):
        """Return a prepared scored copy; preserve input coordinates and metadata."""
        self.set_ligand(ligand)
        terms = self.raw_terms()
        variants = self._variant_scores(terms)
        result = self._ligand.clone
        result.title = ligand.title
        result.data.update({key: value for key, value in ligand.data.items() if not key.startswith("xscore_")})
        for key in list(result.data.keys()):
            if key.startswith("xscore_"):
                del result.data[key]
        result.data.update(
            dict(
                xscore_score=float(np.mean(list(variants.values()))),
                xscore_units=self.score_units,
                xscore_model=self.parameterization,
                xscore_variants=",".join(self.models),
                xscore_charged_ligand_atoms=len(self.xlogp2_charged_atoms_["ligand"]),
            )
        )
        result.data.update({"xscore_" + model: value for model, value in variants.items()})
        result.data.update({"xscore_term_" + name: value for name, value in terms.items()})
        return result

    def predict_ligands(self, ligands):
        """Lazily yield scored copies, retaining input order for VS pipelines."""
        for ligand in ligands:
            yield self.score_ligand(ligand)

    def rescore(self, ligands):
        """Return scored copies ranked by descending pKd, with stable ties."""
        return sorted(
            self.predict_ligands(ligands), key=lambda ligand: float(ligand.data["xscore_score"]), reverse=True
        )
