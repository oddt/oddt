"""Validate the chemistry and geometry of exported 2D interaction maps."""

from itertools import combinations
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

pytest.importorskip("rdkit")
from rdkit import Chem
from rdkit.Chem import AllChem

from oddt.depiction import draw_interactions
from oddt.toolkits import rdk

ASSETS = Path(__file__).parents[1] / "docs/_static/interactions"
NS = {"svg": "http://www.w3.org/2000/svg"}
KINDS = [entry["id"] for entry in json.loads((ASSETS / "manifest.json").read_text())]


def load_example(kind):
    path = ASSETS / kind
    scene = json.loads((path / "scene.json").read_text())
    protein = next(rdk.readfile("pdb", str(path / scene["protein_file"])))
    protein.protein = True
    ligand = next(rdk.readfile("sdf", str(path / scene["ligand_file"])))
    for molecule in (protein, ligand):
        molecule.addh(only_polar=True)
    records = []
    for side, molecule in (("start", protein), ("end", ligand)):
        selected = []
        for contact in scene["contacts"]:
            end = contact[side]
            if end["kind"] == "atom":
                selected.append(molecule.atom_dict[end["indices"][0]])
            else:
                selected.append(
                    next(r for r in molecule.ring_dict if np.allclose(r["centroid"], end["position"], atol=0.001))
                )
        records.append(np.array(selected))
    return scene, protein, ligand, records


def segments_intersect(a, b, c, d):
    """Independently check exported segments, including touching and overlap."""
    delta1, delta2 = b - a, d - c
    matrix = np.column_stack((delta1, -delta2))
    if abs(np.linalg.det(matrix)) > 1e-7:
        t, u = np.linalg.solve(matrix, c - a)
        return -1e-7 <= t <= 1 + 1e-7 and -1e-7 <= u <= 1 + 1e-7
    if abs(delta1[0] * (c - a)[1] - delta1[1] * (c - a)[0]) > 1e-4:
        return False
    axis = int(np.argmax(np.abs(delta1)))
    first, second = sorted((a[axis], b[axis])), sorted((c[axis], d[axis]))
    return min(first[1], second[1]) - max(first[0], second[0]) > 1e-4


def assert_no_crossovers(root):
    assert not root.findall(".//svg:polyline", NS)
    lines = [
        np.array(
            [[float(line.attrib["x1"]), float(line.attrib["y1"])], [float(line.attrib["x2"]), float(line.attrib["y2"])]]
        )
        for line in root.findall(".//svg:line", NS)
    ]
    for (a, b), (c, d) in combinations(lines, 2):
        assert not segments_intersect(a, b, c, d), (a, b, c, d)


@pytest.mark.parametrize("kind", KINDS)
def test_exported_atlas_maps(kind):
    """Check actual straight lines, residue labels, and complete contact data."""
    scene = json.loads((ASSETS / kind / "scene.json").read_text())
    root = ET.parse(ASSETS / kind / scene["depiction_file"]).getroot()
    assert_no_crossovers(root)
    metadata = json.loads(root.find("svg:metadata", NS).text)
    positions = metadata["atom_positions"]["ligand"]
    ligand = next(Chem.SDMolSupplier(str(ASSETS / kind / scene["ligand_file"]), removeHs=False))
    expected_bonds = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in ligand.GetBonds()]
    assert metadata["ligand_bonds"] == expected_bonds
    groups = root.findall(".//svg:g[@data-contact]", NS)
    grouped = sorted(int(i) for group in groups for i in group.attrib["data-contacts"].split())
    assert grouped == list(range(len(scene["contacts"])))
    expected_labels = set()
    for contact, original in zip(scene["contacts"], metadata["contacts"]):
        residue = contact["start"]["residue"]
        expected_labels.add(f"{residue['name']} {residue['chain'] or '–'}:{residue['number']}{residue['insertion']}")
        assert original["protein_atoms"] == contact["start"]["indices"]
        assert original["ligand_atoms"] == contact["end"]["indices"]
        assert original["distance"] == pytest.approx(contact["distance"], abs=1e-4)
    for group in groups:
        line = group.find("svg:line", NS)
        start = np.array([float(line.attrib["x1"]), float(line.attrib["y1"])])
        finish = np.array([float(line.attrib["x2"]), float(line.attrib["y2"])])
        for a, b in expected_bonds:
            assert not segments_intersect(start, finish, np.array(positions[str(a)]), np.array(positions[str(b)])), (
                kind,
                group.attrib["data-contacts"],
                a,
                b,
            )
        indices = list(map(int, group.attrib["data-contacts"].split()))
        contacts = [scene["contacts"][i] for i in indices]
        assert len({tuple(c["end"]["indices"]) for c in contacts}) == 1
        assert len({tuple(c["start"]["residue"].values()) for c in contacts}) == 1
        for contact in contacts:
            assert f"{contact['distance']:.2f} Å (3D)" in group.find("svg:title", NS).text
        end = contacts[0]["end"]
        if end["kind"] == "ring":
            center = np.mean([positions[str(i)] for i in end["indices"]], axis=0)
            circle = group.find("svg:circle", NS)
            np.testing.assert_allclose([float(circle.attrib["cx"]), float(circle.attrib["cy"])], center, atol=0.001)
            ring = np.array([positions[str(i)] for i in end["indices"]])
            crossings = 0
            for a, b in zip(ring, np.roll(ring, -1, axis=0)):
                if (a[1] > finish[1]) != (b[1] > finish[1]):
                    x = a[0] + (finish[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                    crossings += x > finish[0]
            assert crossings % 2 == 0, "Ring contact line must stop outside the near ring edge"
    labels = {
        node.text for node in root.findall("svg:text[@data-residue]", NS) if node.attrib["data-residue"] != "ligand"
    }
    assert labels == expected_labels
    assert {
        node.attrib["data-protein-site"] for node in root.findall("svg:g[@data-protein-site]", NS)
    } == expected_labels
    assert len(root.findall('.//svg:path[@class="protein-surface"]', NS)) == len(expected_labels)


def test_draw_interactions_preserves_inputs_and_is_deterministic():
    _, protein, ligand, records = load_example("hydrogen-bond")
    before = [Chem.MolToMolBlock(m.Mol) for m in (protein, ligand)]
    svg = draw_interactions(protein, ligand, *records, legend="MK1 <bound>")
    assert svg == draw_interactions(protein, ligand, *records, legend="MK1 <bound>")
    assert before == [Chem.MolToMolBlock(m.Mol) for m in (protein, ligand)]
    root = ET.fromstring(svg)
    assert_no_crossovers(root)
    assert root.find("svg:text[@data-residue='ligand']", NS).text == "MK1 <bound>"


def test_best_fit_plane_uses_bound_ligand_heavy_atoms():
    root = ET.parse(ASSETS / "hydrogen-bond/interaction.svg").getroot()
    projection = json.loads(root.find("svg:metadata", NS).text)["projection"]
    ligand = next(Chem.SDMolSupplier(str(ASSETS / "hydrogen-bond/ligand.sdf"), removeHs=True))
    xyz = ligand.GetConformer().GetPositions()
    origin = np.array(projection["origin"])
    axes = np.array(projection["axes"])
    np.testing.assert_allclose(origin, xyz.mean(axis=0), atol=1e-8)
    np.testing.assert_allclose(axes @ axes.T, np.eye(2), atol=1e-8)
    normal = np.cross(*axes)
    errors = (xyz - origin) @ normal
    eigenvalues = np.linalg.eigvalsh((xyz - origin).T @ (xyz - origin) / len(xyz))
    assert np.mean(errors**2) == pytest.approx(eigenvalues[0], abs=1e-8)
    assert projection["rms_out_of_plane"] == pytest.approx(np.sqrt(eigenvalues[0]), abs=1e-8)


def test_empty_and_duplicate_contacts():
    _, protein, ligand, records = load_example("hydrogen-bond")
    empty = ET.fromstring(draw_interactions(protein, ligand, records[0][:0], records[1][:0]))
    assert not empty.findall(".//svg:line", NS)
    duplicate = ET.fromstring(
        draw_interactions(protein, ligand, np.repeat(records[0][:1], 2), np.repeat(records[1][:1], 2))
    )
    assert len(duplicate.findall(".//svg:line", NS)) == 1
    assert len(json.loads(duplicate.find("svg:metadata", NS).text)["contacts"]) == 2


@pytest.mark.parametrize("smiles,opposite", [("F/C=C/F", True), ("F/C=C\\F", False)])
def test_ligand_double_bond_stereochemistry(smiles, opposite):
    """E/Z geometry must survive copying and hydrogen hiding in the drawing."""
    _, protein, _, records = load_example("hydrogen-bond")
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    root = ET.fromstring(draw_interactions(protein, rdk.Molecule(mol), records[0][:0], records[1][:0]))
    positions = json.loads(root.find("svg:metadata", NS).text)["atom_positions"]["ligand"]
    first, left, right, last = [np.array(positions[str(i)]) for i in range(4)]
    axis = right - left
    side1 = axis[0] * (first - left)[1] - axis[1] * (first - left)[0]
    side2 = axis[0] * (last - right)[1] - axis[1] * (last - right)[0]
    assert bool(side1 * side2 < 0) == opposite


def test_invalid_inputs():
    _, protein, ligand, records = load_example("hydrogen-bond")
    with pytest.raises(ValueError, match="same length"):
        draw_interactions(protein, ligand, records[0][:-1], records[1])
    with pytest.raises(TypeError, match="RDKit-backed"):
        draw_interactions(SimpleNamespace(Mol=None), ligand, *records)
    with pytest.raises(ValueError, match="six-digit"):
        draw_interactions(protein, ligand, *records, color="red")
    with pytest.raises(ValueError, match="dimensions"):
        draw_interactions(protein, ligand, *records, size=(10, 10))
    bad_atom = records[0][:1].copy()
    bad_atom["id"] = protein.Mol.GetNumAtoms()
    with pytest.raises(ValueError, match="outside"):
        draw_interactions(protein, ligand, bad_atom, records[1][:1])
    without_pose = rdk.Molecule(Chem.Mol(ligand.Mol))
    without_pose.Mol.RemoveAllConformers()
    with pytest.raises(ValueError, match="3D coordinates"):
        draw_interactions(protein, without_pose, *records)
