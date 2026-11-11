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


def assert_atom_view_endpoints(root):
    """Every visible segment must point at its actual pair of chemical endpoints."""
    metadata = json.loads(root.find("svg:metadata", NS).text)
    bond_lengths = [
        np.linalg.norm(np.array(metadata["atom_positions"][side][str(a)]) - metadata["atom_positions"][side][str(b)])
        for side in ("ligand", "protein")
        for a, b in metadata[f"{side}_bonds"]
    ]
    scale = np.median(bond_lengths) / 1.5 if bond_lengths else None
    for group in root.findall(".//svg:g[@data-contact]", NS):
        contacts = [metadata["contacts"][int(i)] for i in group.attrib["data-contacts"].split()]
        assert len({tuple(c["protein_atoms"]) for c in contacts}) == 1
        assert len({tuple(c["ligand_atoms"]) for c in contacts}) == 1
        contact = contacts[0]
        protein, ligand = [
            np.mean([metadata["atom_positions"][side][str(i)] for i in contact[f"{side}_atoms"]], axis=0)
            for side in ("protein", "ligand")
        ]
        direction = ligand - protein
        length = np.linalg.norm(direction)
        unit = direction / length
        distances = []
        for line in group.findall("svg:line", NS):
            for index in (1, 2):
                point = np.array([float(line.attrib[f"x{index}"]), float(line.attrib[f"y{index}"])])
                delta = point - protein
                assert abs(delta[0] * unit[1] - delta[1] * unit[0]) < 0.002
                distances.append(np.dot(delta, unit))
        assert 0 < distances[0] < distances[-1] < length
        assert all(first < second for first, second in zip(distances, distances[1:]))
        if scale is not None:
            if len(contact["protein_atoms"]) == 1:
                assert distances[0] == pytest.approx(0.45 * scale, abs=0.002)
            if len(contact["ligand_atoms"]) == 1:
                assert length - distances[-1] == pytest.approx(0.45 * scale, abs=0.002)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("show_protein_atoms", [False, True])
def test_exported_atlas_maps(kind, show_protein_atoms):
    """Check actual straight lines, residue labels, and complete contact data."""
    scene = json.loads((ASSETS / kind / "scene.json").read_text())
    filename = scene["depiction_atoms_file" if show_protein_atoms else "depiction_file"]
    root = ET.parse(ASSETS / kind / filename).getroot()
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
        for line in group.findall("svg:line", NS):
            start = np.array([float(line.attrib["x1"]), float(line.attrib["y1"])])
            finish = np.array([float(line.attrib["x2"]), float(line.attrib["y2"])])
            for a, b in expected_bonds:
                assert not segments_intersect(
                    start, finish, np.array(positions[str(a)]), np.array(positions[str(b)])
                ), (
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
    if show_protein_atoms:
        assert_atom_view_endpoints(root)
        protein = Chem.MolFromPDBFile(str(ASSETS / kind / scene["protein_file"]), removeHs=False)
        expected_atoms = {
            atom.GetIdx()
            for atom in protein.GetAtoms()
            if atom.GetAtomicNum() > 1
            and (
                f"{atom.GetPDBResidueInfo().GetResidueName().strip()} "
                f"{atom.GetPDBResidueInfo().GetChainId().strip() or '–'}:"
                f"{atom.GetPDBResidueInfo().GetResidueNumber()}{atom.GetPDBResidueInfo().GetInsertionCode().strip()}"
            )
            in expected_labels
        }
        positions = metadata["atom_positions"]["protein"]
        assert set(map(int, positions)) == expected_atoms
        expected_bonds = {
            tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
            for bond in protein.GetBonds()
            if bond.GetBeginAtomIdx() in expected_atoms and bond.GetEndAtomIdx() in expected_atoms
        }
        assert {tuple(sorted(pair)) for pair in metadata["protein_bonds"]} == expected_bonds
        for group in groups:
            for line in group.findall("svg:line", NS):
                start, finish = [np.array([float(line.attrib[f"x{i}"]), float(line.attrib[f"y{i}"])]) for i in (1, 2)]
                for first, second in expected_bonds:
                    assert not segments_intersect(
                        start, finish, np.array(positions[str(first)]), np.array(positions[str(second)])
                    )
        assert not root.findall('.//svg:path[@class="protein-surface"]', NS)
        assert all(node.findall("svg:path", NS) for node in root.findall("svg:g[@data-protein-site]", NS))
    else:
        assert len(root.findall('.//svg:path[@class="protein-surface"]', NS)) == len(expected_labels)


@pytest.mark.parametrize("show_protein_atoms", [False, True])
def test_draw_interactions_preserves_inputs_and_is_deterministic(show_protein_atoms):
    _, protein, ligand, records = load_example("hydrogen-bond")
    before = [Chem.MolToMolBlock(m.Mol) for m in (protein, ligand)]
    svg = draw_interactions(protein, ligand, *records, legend="MK1 <bound>", show_protein_atoms=show_protein_atoms)
    assert svg == draw_interactions(
        protein, ligand, *records, legend="MK1 <bound>", show_protein_atoms=show_protein_atoms
    )
    assert before == [Chem.MolToMolBlock(m.Mol) for m in (protein, ligand)]
    root = ET.fromstring(svg)
    assert_no_crossovers(root)
    if show_protein_atoms:
        assert_atom_view_endpoints(root)
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


def test_residue_atom_mode_handles_metals_and_no_contacts():
    _, protein, ligand, records = load_example("acceptor-metal")
    root = ET.fromstring(draw_interactions(protein, ligand, *records, show_protein_atoms=True))
    metadata = json.loads(root.find("svg:metadata", NS).text)
    assert set(metadata["atom_positions"]["protein"]) == {str(records[0][0]["id"])}
    assert protein.Mol.GetAtomWithIdx(int(records[0][0]["id"])).GetSymbol() == "Zn"
    assert not metadata["protein_bonds"]
    empty = ET.fromstring(draw_interactions(protein, ligand, records[0][:0], records[1][:0], show_protein_atoms=True))
    assert not empty.findall(".//svg:line", NS)
    assert json.loads(empty.find("svg:metadata", NS).text)["atom_positions"]["protein"] == {}


def test_pocket_backbone_cuts_are_peptide_continuations():
    """A cropped amide N has one H; side-chain NH2 groups must stay NH2."""
    from oddt.depiction import _protein_fragments

    _, protein, _, records = load_example("hydrogen-bond")
    before = Chem.MolToMolBlock(protein.Mol)
    groups = {}
    for record in records[0]:
        atom = protein.Mol.GetAtomWithIdx(int(record["id"]))
        info = atom.GetPDBResidueInfo()
        key = (
            info.GetResidueName().strip(),
            info.GetChainId().strip(),
            info.GetResidueNumber(),
            info.GetInsertionCode().strip(),
        )
        groups[(key, (), (int(record["id"]),))] = []
    fragments = _protein_fragments(protein, groups)
    for fragment in fragments.values():
        for source_id, local_id in fragment["lookup"].items():
            atom = fragment["molecule"].GetAtomWithIdx(local_id)
            original = protein.Mol.GetAtomWithIdx(source_id)
            name = original.GetPDBResidueInfo().GetName().strip()
            if name == "N":
                assert atom.GetTotalNumHs() == 1
                assert atom.GetDegree() == 2
            elif name == "C":
                assert atom.GetTotalNumHs() == 0
                assert atom.GetDegree() == 3
            elif name in ("NH1", "NH2"):
                assert atom.GetTotalNumHs() == original.GetTotalNumHs(includeNeighbors=True)
    assert Chem.MolToMolBlock(protein.Mol) == before


def test_explicit_protein_termini_are_retained():
    from oddt.depiction import _protein_fragments

    molecule = Chem.MolFromSmiles("[NH3+][C@@H](C)C(=O)[O-]")
    for atom, name in zip(molecule.GetAtoms(), ("N", "CA", "CB", "C", "O", "OXT")):
        info = Chem.AtomPDBResidueInfo()
        info.SetName(name)
        info.SetResidueName("ALA")
        info.SetChainId("A")
        info.SetResidueNumber(1)
        atom.SetMonomerInfo(info)
    fragments = _protein_fragments(rdk.Molecule(molecule), {(("ALA", "A", 1, ""), (), (0,)): []})
    fragment = fragments[("ALA", "A", 1, "")]["molecule"]
    assert Chem.MolToSmiles(fragment) == Chem.MolToSmiles(molecule)
    assert fragment.GetAtomWithIdx(0).GetTotalNumHs() == 3
    assert not any(atom.GetAtomicNum() == 0 for atom in fragment.GetAtoms())


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
    with pytest.raises(TypeError, match="boolean"):
        draw_interactions(protein, ligand, *records, show_protein_atoms="yes")
    bad_atom = records[0][:1].copy()
    bad_atom["id"] = protein.Mol.GetNumAtoms()
    with pytest.raises(ValueError, match="outside"):
        draw_interactions(protein, ligand, bad_atom, records[1][:1])
    without_pose = rdk.Molecule(Chem.Mol(ligand.Mol))
    without_pose.Mol.RemoveAllConformers()
    with pytest.raises(ValueError, match="3D coordinates"):
        draw_interactions(protein, without_pose, *records)
