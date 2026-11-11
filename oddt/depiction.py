"""Draw two-dimensional ligand maps with labelled protein contact sites."""

from collections import defaultdict
from itertools import combinations
import json
import math
import re
import xml.etree.ElementTree as ET

import numpy as np

__all__ = ["draw_interactions"]

SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)


def _indices(molecule, records):
    """Resolve detector records before changing any drawing coordinates."""
    result = []
    for record in records:
        fields = record.dtype.names or ()
        if "centroid" in fields:
            rings = [
                ring
                for ring in molecule.Mol.GetRingInfo().AtomRings()
                if all(molecule.Mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
                and np.allclose(molecule.coords[list(ring)].mean(axis=0), record["centroid"], atol=0.001)
            ]
            if len(rings) != 1:
                raise ValueError("A ring record must match exactly one aromatic ring in its source molecule")
            result.append(tuple(rings[0]))
        elif "id" in fields:
            index = int(record["id"])
            if not 0 <= index < molecule.Mol.GetNumAtoms():
                raise ValueError("Contact atom ID is outside its source molecule")
            result.append((index,))
        else:
            raise ValueError("Contacts must be ODDT atom_dict or ring_dict records")
    return result


def _residue(atom):
    info = atom.GetPDBResidueInfo()
    if info is None:
        raise ValueError("Protein contact atoms must have PDB residue information")
    return (
        info.GetResidueName().strip(),
        info.GetChainId().strip(),
        info.GetResidueNumber(),
        info.GetInsertionCode().strip(),
    )


def _crosses(first, second):
    """Allow a shared endpoint, but reject crossings, overlap, and T junctions."""
    a, b = np.asarray(first, dtype=float)
    c, d = np.asarray(second, dtype=float)
    cross = lambda x, y: x[0] * y[1] - x[1] * y[0]
    ab, cd = b - a, d - c
    denominator = cross(ab, cd)
    if abs(denominator) < 1e-8:
        if abs(cross(c - a, ab)) > 1e-8 or np.dot(ab, ab) < 1e-8:
            return False
        interval = sorted((np.dot(c - a, ab), np.dot(d - a, ab)))
        return min(np.dot(ab, ab), interval[1]) - max(0, interval[0]) > 1e-6
    t, u = cross(c - a, cd) / denominator, cross(c - a, ab) / denominator
    if not (-1e-6 <= t <= 1 + 1e-6 and -1e-6 <= u <= 1 + 1e-6):
        return False
    return not ((abs(t) < 1e-6 or abs(t - 1) < 1e-6) and (abs(u) < 1e-6 or abs(u - 1) < 1e-6))


def _hits_box(line, box):
    left, bottom, right, top = box
    a, b = line
    if any(left <= point[0] <= right and bottom <= point[1] <= top for point in (a, b)):
        return True
    corners = [(left, bottom), (right, bottom), (right, top), (left, top)]
    return any(_crosses(line, (corners[i], corners[(i + 1) % 4])) for i in range(4))


def _fit_plane(xyz, contact_points):
    """Fit the least-squares plane, using contact points for degenerate ligands."""
    origin = xyz.mean(axis=0)
    centered = xyz - origin
    fit = centered
    if np.linalg.matrix_rank(centered, tol=1e-8) < 2 and len(contact_points):
        fit = np.concatenate((centered, np.asarray(contact_points) - origin))
        fit -= fit.mean(axis=0)
    _, _, vectors = np.linalg.svd(fit, full_matrices=True)
    axes = vectors[:2].copy()
    # Resolve SVD's arbitrary axis signs without depending on random state.
    for axis in axes:
        if axis[np.argmax(np.abs(axis))] < 0:
            axis *= -1
    return origin, axes


def _ligand_layouts(molecule, origin, axes):
    """Keep legal chemical geometry while sampling layouts close to the bound pose."""
    from rdkit import Chem
    from rdkit.Chem import rdDepictor

    xyz = molecule.GetConformer().GetPositions()
    projection = (xyz - origin) @ axes.T
    projection -= projection.mean(axis=0)
    lower = np.tril_indices(len(xyz), -1)
    distances = [np.linalg.norm(points[lower[0]] - points[lower[1]], axis=1) for points in (projection, xyz)]
    variants = [(None, None, seed) for seed in (42, 43, 44, 45)]
    if len(xyz) >= 3:
        variants += [(dist, weight, seed) for dist in distances for weight in (0.25, 0.6, 0.9) for seed in (42, 43)]
    layouts = []
    seen = set()
    for dist, weight, seed in variants:
        copied = Chem.Mol(molecule)
        if dist is None:
            rdDepictor.Compute2DCoords(copied, nFlipsPerSample=3, nSample=150, sampleSeed=seed, forceRDKit=True)
        else:
            rdDepictor.Compute2DCoordsMimicDistmat(
                copied, dist, weightDistMat=weight, nSample=200, sampleSeed=seed, forceRDKit=True
            )
        xy = copied.GetConformer().GetPositions()[:, :2]
        xy -= xy.mean(axis=0)
        u, _, v = np.linalg.svd(xy.T @ projection)
        xy = xy @ (u @ v)
        signature = tuple(np.round(xy, 5).ravel())
        if signature in seen:
            continue
        seen.add(signature)
        error = float(np.sqrt(np.mean(np.sum((xy - projection) ** 2, axis=1))))
        # Shared centering is retained when projecting the contact directions.
        offset = (xy.min(axis=0) + xy.max(axis=0)) / 2
        layouts.append((error, xy - offset))
    return sorted(layouts, key=lambda item: item[0])


def _visible_line(start, target, ring, xy, bonds):
    """Stop outside an atom label or at the near edge of a contacted ring."""
    vector = target - start
    length = np.linalg.norm(vector)
    if length < 1.5:
        return None
    unit = vector / length
    finish = target - unit * 0.45
    if len(ring) > 1:
        ring = set(ring)
        entries = []
        for first, second in bonds:
            if first not in ring or second not in ring:
                continue
            a, b = xy[first], xy[second]
            matrix = np.column_stack((vector, a - b))
            if abs(np.linalg.det(matrix)) < 1e-9:
                continue
            t, u = np.linalg.solve(matrix, a - start)
            if 0 <= t <= 1 and 0 <= u <= 1:
                entries.append(t)
        if not entries:
            return None
        finish = start + unit * (min(entries) * length - 0.22)
    line = (start + unit * 0.85, finish)
    if np.dot(line[1] - line[0], unit) <= 0.1:
        return None
    if any(_crosses(line, (xy[first], xy[second])) for first, second in bonds):
        return None
    # Avoid grazing a non-target atom's symbol even without an exact bond hit.
    delta = line[1] - line[0]
    t = np.clip(((xy - line[0]) @ delta) / np.dot(delta, delta), 0, 1)
    distance = np.linalg.norm(xy - line[0] - t[:, None] * delta, axis=1)
    if any(d < 0.32 for i, d in enumerate(distance) if i not in ring):
        return None
    return line


def _place_sites(xy, targets, groups, legend, bonds, endpoint_ids, preferred):
    """Assign each site its own visible angle, preferring its projected 3D side."""
    keys = sorted({key for key, _ in groups})
    legend_point = np.array([0.0, xy[:, 1].min() - 1.8])
    legend_box = (-len(legend) * 0.37, legend_point[1] - 0.45, len(legend) * 0.37, legend_point[1] + 1.0)
    if not keys:
        return {}, [], [(legend, legend_point)]
    half = np.ptp(xy, axis=0) / 2
    candidates = {}
    for key in keys:
        indices = [i for i, (residue, _) in enumerate(groups) if residue == key]
        target_center = np.mean([targets[i] for i in indices], axis=0)
        direction = preferred[key]
        direction_norm = np.linalg.norm(direction)
        options = []
        for expansion in (1.0, 1.3, 1.7):
            for angle in np.linspace(0, 2 * math.pi, 96, endpoint=False):
                position = (half + 4.5) * expansion * np.array([math.cos(angle), math.sin(angle)])
                lines = {}
                for index in indices:
                    line = _visible_line(position, targets[index], endpoint_ids[index], xy, bonds)
                    if line is None or _hits_box(line, legend_box):
                        break
                    lines[index] = line
                else:
                    if any(_crosses(a, b) for a, b in combinations(lines.values(), 2)):
                        continue
                    label = f"{key[0]} {key[1] or '–'}:{key[2]}{key[3]}"
                    label_point = position + position / np.linalg.norm(position) * 2.1
                    box = (
                        label_point[0] - len(label) * 0.37,
                        label_point[1] - 0.45,
                        label_point[0] + len(label) * 0.37,
                        label_point[1] + 1.0,
                    )
                    if any(_hits_box(line, box) for line in lines.values()):
                        continue
                    delta = position - target_center
                    angular = (
                        1 - np.dot(delta, direction) / (np.linalg.norm(delta) * direction_norm)
                        if direction_norm > 1e-6
                        else 0
                    )
                    score = 4 * angular + 0.1 * sum(np.linalg.norm(b - a) for a, b in lines.values())
                    options.append((float(score), position, lines, (label, label_point), box))
        if not options:
            raise ValueError("No unobstructed straight contact for a residue")
        candidates[key] = sorted(options, key=lambda option: option[0])
    order = sorted(keys, key=lambda key: len(candidates[key]))
    chosen = {}
    visits = 0

    def assign(depth):
        nonlocal visits
        if depth == len(order):
            return True
        key = order[depth]
        for option in candidates[key]:
            visits += 1
            if visits > 20000:
                return False
            _, position, lines, label, box = option
            if not (
                box[2] < legend_box[0] or legend_box[2] < box[0] or box[3] < legend_box[1] or legend_box[3] < box[1]
            ):
                continue
            conflict = False
            for _, other_position, other_lines, _, other_box in chosen.values():
                if (
                    np.linalg.norm(position - other_position) < 2.5
                    or not (
                        box[2] < other_box[0] or other_box[2] < box[0] or box[3] < other_box[1] or other_box[3] < box[1]
                    )
                    or any(_crosses(a, b) for a in lines.values() for b in other_lines.values())
                    or any(_hits_box(line, other_box) for line in lines.values())
                    or any(_hits_box(line, box) for line in other_lines.values())
                ):
                    conflict = True
                    break
            if not conflict:
                chosen[key] = option
                if assign(depth + 1):
                    return True
                del chosen[key]
        return False

    if not assign(0):
        raise ValueError("Could not place contacts without ligand or label crossings")
    sites = {key: chosen[key][1] for key in keys}
    lines = [chosen[key][2][index] for index, (key, _) in enumerate(groups)]
    labels = [(legend, legend_point), *[chosen[key][3] for key in keys]]
    return sites, lines, labels


def draw_interactions(protein, ligand, protein_atoms, ligand_atoms, size=(1000, 720), color="#0f766e", legend="Ligand"):
    """Draw a ligand surrounded by labelled protein sites and straight contacts.

    Parameters
    ----------
    protein, ligand : oddt.toolkits.rdk.Molecule
        RDKit-backed ODDT molecules in the same 3D coordinate frame. Protein
        atoms must retain PDB residue information. Inputs are not modified.
    protein_atoms, ligand_atoms : numpy.ndarray
        Aligned atom_dict or ring_dict records from an interaction detector,
        in protein/ligand order. Apply a strict mask before drawing if needed.
        Contacts to the same residue and ligand atom/ring share one line;
        all original pairs and 3D distances remain in the SVG metadata.
    size : tuple of int, optional
        SVG width and height in pixels; each dimension must be at least 400.
    color : str, optional
        Six-digit hexadecimal color for the interaction lines.
    legend : str, optional
        Label beneath the ligand.

    Returns
    -------
    str
        Standalone SVG with a chemical ligand depiction, labelled protein
        pseudoatoms, symbolic surface patches, and straight, uncrossed contacts.

    Raises
    ------
    TypeError
        If either molecule is not an RDKit-backed ODDT molecule.
    ValueError
        If inputs are incompatible, or straight contacts cannot be laid out
        without crossings. Select fewer contacts for dense interaction graphs.

    Notes
    -----
    Protein sites are grouped by residue name, chain, number and insertion code.
    Their surface silhouettes are symbolic, not calculated molecular surfaces.
    The ligand's least-squares best-fit plane guides conformer selection and
    orientation, while projected protein contact directions guide site placement.
    Chemical drawing geometry is retained rather than flattening 3D coordinates
    directly. Straight lines avoid ligand bonds and unrelated atom symbols.
    For ring contacts a line stops outside the near ring edge, and a separate
    dot marks the true depicted centroid. The layout remains schematic;
    reported distances always come from the original 3D coordinates.
    """
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Geometry import Point2D

    if not all(isinstance(getattr(molecule, "Mol", None), Chem.Mol) for molecule in (protein, ligand)):
        raise TypeError("draw_interactions requires RDKit-backed ODDT molecules")
    if not ligand.Mol.GetNumAtoms():
        raise ValueError("The ligand must contain at least one atom")
    if not all(
        m.Mol.GetNumConformers() and m.Mol.GetConformer().Is3D() and np.isfinite(m.coords).all()
        for m in (protein, ligand)
    ):
        raise ValueError("Both molecules must have finite bound 3D coordinates")
    if len(size) != 2 or any(not isinstance(n, int) or n < 400 for n in size):
        raise ValueError("size must contain two integer dimensions of at least 400 pixels")
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        raise ValueError("color must be a six-digit hexadecimal color")
    if len(protein_atoms) != len(ligand_atoms):
        raise ValueError("Protein and ligand contact arrays must have the same length")
    protein_ids, ligand_ids = _indices(protein, protein_atoms), _indices(ligand, ligand_atoms)
    if any(
        m.Mol.GetAtomWithIdx(i).GetAtomicNum() == 1
        for m, endpoints in ((protein, protein_ids), (ligand, ligand_ids))
        for ids in endpoints
        for i in ids
    ):
        raise ValueError("Contact endpoints must be heavy atoms or aromatic rings")

    central = Chem.Mol(ligand.Mol)
    for atom in central.GetAtoms():
        atom.SetIntProp("_oddt_source_index", atom.GetIdx())
    central = Chem.RemoveHs(central)
    lookup = {a.GetIntProp("_oddt_source_index"): a.GetIdx() for a in central.GetAtoms()}
    groups = defaultdict(list)
    contacts = []
    for index, (first, second) in enumerate(zip(protein_ids, ligand_ids)):
        key = _residue(protein.Mol.GetAtomWithIdx(first[0]))
        if any(_residue(protein.Mol.GetAtomWithIdx(i)) != key for i in first):
            raise ValueError("A protein endpoint cannot span multiple residues")
        groups[(key, second)].append(index)
        contacts.append(
            {
                "protein_atoms": first,
                "ligand_atoms": second,
                "distance": float(
                    np.linalg.norm(protein.coords[list(first)].mean(axis=0) - ligand.coords[list(second)].mean(axis=0))
                ),
            }
        )
    xyz = central.GetConformer().GetPositions()
    contact_points = [protein.coords[list(first)].mean(axis=0) for first in protein_ids]
    origin, axes = _fit_plane(xyz, contact_points)
    preferred = {
        key: np.mean(
            [
                (protein.coords[list(protein_ids[i])].mean(axis=0) - ligand.coords[list(ligand_ids[i])].mean(axis=0))
                @ axes.T
                for (residue, _), indices in groups.items()
                if residue == key
                for i in indices
            ],
            axis=0,
        )
        for key in {residue for residue, _ in groups}
    }
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in central.GetBonds()]
    endpoint_ids = [tuple(lookup[i] for i in second) for _, second in groups]
    for fit_error, xy in _ligand_layouts(central, origin, axes):
        targets = [xy[list(ids)].mean(axis=0) for ids in endpoint_ids]
        try:
            sites, lines, labels = _place_sites(xy, targets, groups, legend, bonds, endpoint_ids, preferred)
        except ValueError:
            continue
        break
    else:
        raise ValueError("Could not place straight contacts without crossing the ligand; select fewer contacts")
    for index, (x, y) in enumerate(xy):
        central.GetConformer().SetAtomPosition(index, (float(x), float(y), 0.0))
    central.GetConformer().Set3D(False)
    all_points = np.concatenate(
        [xy, np.array([point for _, point in labels]), np.array(list(sites.values())).reshape(-1, 2)]
    )
    low, high = all_points.min(axis=0) - 3, all_points.max(axis=0) + 3
    # Long residue labels need horizontal space as well as their anchor points.
    for text, point in labels:
        low = np.minimum(low, point - [len(text) * 0.4 + 1, 1.5])
        high = np.maximum(high, point + [len(text) * 0.4 + 1, 1.5])
    prepared = rdMolDraw2D.PrepareMolForDrawing(central, addChiralHs=False)
    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.prepareMolsBeforeDrawing = False
    font_size = max(8, round(17 * min(size[0] / 1000, size[1] / 720)))
    options.fixedFontSize = font_size
    options.bondLineWidth = 1.7
    drawer.SetScale(*size, Point2D(*low), Point2D(*high))
    drawer.DrawMolecule(prepared)
    drawer.FinishDrawing()
    root = ET.fromstring(drawer.GetDrawingText())
    root.set("role", "img")
    root.set("aria-label", "2D ligand interaction map surrounded by labelled protein sites")
    root.set("class", "oddt-interaction-map")
    element = lambda parent, name, **attrs: ET.SubElement(
        parent, f"{{{SVG}}}{name}", {k.replace("_", "-"): str(v) for k, v in attrs.items()}
    )
    background = ET.Element(f"{{{SVG}}}rect", {"width": "100%", "height": "100%", "fill": "#ffffff"})
    root.insert(0, background)
    element(root, "title").text = f"{legend}: protein–ligand contacts"
    element(root, "desc").text = (
        "Labelled protein pseudoatoms surround the ligand. Surface silhouettes are symbolic. Straight dashed lines group contacts by residue and ligand atom/ring; distances in tooltips come from 3D coordinates."
    )
    to_pixel = lambda p: np.array([drawer.GetDrawCoords(Point2D(*p)).x, drawer.GetDrawCoords(Point2D(*p)).y])
    metadata = {
        "atom_positions": {"ligand": {str(i): to_pixel(xy[j]).round(6).tolist() for i, j in lookup.items()}},
        "ligand_bonds": [
            [
                central.GetAtomWithIdx(a).GetIntProp("_oddt_source_index"),
                central.GetAtomWithIdx(b).GetIntProp("_oddt_source_index"),
            ]
            for a, b in bonds
        ],
        "projection": {
            "origin": origin.tolist(),
            "axes": axes.tolist(),
            "fit_rmsd": fit_error,
            "rms_out_of_plane": float(np.sqrt(np.mean(((xyz - origin) @ np.cross(*axes)) ** 2))),
        },
        "contacts": contacts,
    }
    element(root, "metadata").text = json.dumps(metadata, separators=(",", ":"))
    layer = ET.Element(f"{{{SVG}}}g", {"class": "interaction-contacts"})
    root.insert(1, layer)
    for (key, second), (start, finish), indices in zip(groups, lines, groups.values()):
        first_point = to_pixel(start)
        last_point = to_pixel(finish)
        group = element(layer, "g", data_contact=indices[0], data_contacts=" ".join(map(str, indices)))
        name = f"{key[0]} {key[1] or '–'}:{key[2]}{key[3]}"
        element(group, "title").text = (
            name + " · " + ", ".join(f"{contacts[i]['distance']:.2f} Å (3D)" for i in indices)
        )
        element(
            group,
            "line",
            x1=f"{first_point[0]:.3f}",
            y1=f"{first_point[1]:.3f}",
            x2=f"{last_point[0]:.3f}",
            y2=f"{last_point[1]:.3f}",
            stroke=color,
            stroke_width="1.8",
            stroke_dasharray="5 4",
        )
        if len(second) > 1:
            x, y = to_pixel(xy[[lookup[i] for i in second]].mean(axis=0))
            element(group, "circle", cx=f"{x:.3f}", cy=f"{y:.3f}", r="3", fill=color)
    for key, point in sites.items():
        target = np.mean([target for target, (residue, _) in zip(targets, groups) if residue == key], axis=0)
        direction = math.atan2(*(target - point)[::-1])
        angles = np.linspace(direction - math.pi / 3, direction + math.pi / 3, 13)
        inner = [point + 0.85 * np.array([math.cos(a), math.sin(a)]) for a in angles]
        outer = [
            point + (1.15 + 0.08 * math.sin(i * math.pi / 2)) * np.array([math.cos(a), math.sin(a)])
            for i, a in enumerate(angles)
        ]
        patch = [to_pixel(p) for p in [*inner, *outer[::-1]]]
        outline = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in patch) + " Z"
        site = element(root, "g", data_protein_site=f"{key[0]} {key[1] or '–'}:{key[2]}{key[3]}")
        element(
            site,
            "path",
            d=outline,
            fill="#d8e1e9",
            stroke="#718096",
            stroke_width="1.1",
            **{"class": "protein-surface"},
        )
        x, y = to_pixel(point)
        element(site, "circle", cx=f"{x:.2f}", cy=f"{y:.2f}", r="2.8", fill="#64748b", **{"class": "protein-site"})
    for index, (label, point) in enumerate(labels):
        x, y = to_pixel(point)
        node = element(
            root,
            "text",
            x=f"{x:.2f}",
            y=f"{y:.2f}",
            text_anchor="middle",
            font_family="sans-serif",
            font_size=font_size,
            font_weight="600",
            fill="#1f2937",
            data_residue="ligand" if index == 0 else label,
            **{"class": "residue-label"},
        )
        node.text = label
    return ET.tostring(root, encoding="unicode")
