"""Regenerate the deposited-coordinate examples used by the interaction atlas.

Run from a checkout with the documentation environment installed:
    python docs/generate_interaction_examples.py

Inputs are cached under docs/_build; normal Sphinx builds use the committed
assets and never download structures or run this script.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
from urllib.request import urlopen

import numpy as np
from rdkit import Chem, rdBase
from scipy.spatial.distance import cdist

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

import oddt
from oddt import interactions
from oddt.depiction import draw_interactions
from oddt.toolkits import rdk

EXAMPLES = [
    dict(
        id="hydrogen-bond",
        title="Hydrogen bonds",
        pdb_id="1HSG",
        component="MK1",
        chain="B",
        number=902,
        complex="HIV protease · indinavir",
        color="#b45309",
        description="Conventional N/O hydrogen bonds between indinavir and the protease pocket. Only contacts passing ODDT's strict angular mask are shown.",
    ),
    dict(
        id="hydrophobic",
        title="Hydrophobic contacts",
        pdb_id="4BFQ",
        component="083",
        chain="A",
        number=301,
        complex="Acetylcholine-binding protein · ligand 083",
        color="#0f766e",
        description="Representative hydrophobic atom pairs around ligand 083. The closest pair per protein residue is shown, for up to eight residues; other copies of the ligand are omitted.",
    ),
    dict(
        id="halogen-bond",
        title="Halogen bonds",
        pdb_id="1P5E",
        component="TBS",
        chain="A",
        number=301,
        complex="CDK2 · tetrabromobenzotriazole",
        color="#7c3aed",
        description="Bromine–acceptor contacts in the CDK2 pocket, filtered by ODDT's strict halogen-bond geometry.",
    ),
    dict(
        id="salt-bridge",
        title="Salt bridges",
        pdb_id="3PTB",
        component="BEN",
        chain="A",
        number=1,
        complex="Trypsin · benzamidine",
        color="#be185d",
        description="The benzamidine amidinium group contacts Asp189. ODDT's default ionizable-group flags are used; the exact-formal-charge filters are not enabled.",
    ),
    dict(
        id="pi-stacking",
        title="π stacking",
        pdb_id="1ACJ",
        component="THA",
        chain="A",
        number=999,
        complex="Acetylcholinesterase · tacrine",
        color="#2563eb",
        description="Aromatic ring contacts passing ODDT's parallel or perpendicular stacking mask. Dashed lines join ring centroids, not atoms.",
    ),
    dict(
        id="pi-cation",
        title="π–cation contacts",
        pdb_id="1ACL",
        component="DME",
        chain="A",
        number=999,
        complex="Acetylcholinesterase · decamethonium",
        color="#c2410c",
        description="Protein aromatic rings contact the explicitly charged quaternary ammonium groups of decamethonium. Lines join ring centroids to cation atoms.",
    ),
    dict(
        id="acceptor-metal",
        title="Acceptor–metal coordination",
        pdb_id="1HFC",
        component="PLH",
        chain="A",
        number=280,
        complex="Fibroblast collagenase · hydroxamate inhibitor",
        color="#0369a1",
        description="Ligand acceptor atoms coordinate a pocket zinc ion and pass ODDT's strict angular mask. Nearby deposited metal ions are retained in the receptor.",
    ),
    dict(
        id="pi-metal",
        title="π–metal contacts",
        pdb_id="1LPI",
        component="NA",
        chain="A",
        number=200,
        complex="Lysozyme · sodium",
        color="#6d28d9",
        description="The sodium ion contacts a protein tryptophan π system. Here the protein supplies the ring and the ligand-side molecule is the metal ion.",
    ),
]


def fetch(url, cache):
    """Cache source bytes; their checksums are recorded in every generated scene."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=60) as response:
            cache.write_bytes(response.read())
    data = cache.read_bytes()
    return data.decode("utf-8"), {"url": url, "sha256": hashlib.sha256(data).hexdigest()}


def ccd_loop(text, category):
    """Read a CCD atom/bond loop, including quoted atom names, without extra deps."""
    if category + "." not in text:
        return []
    singleton = {}
    for line in text.splitlines():
        if line.startswith(category + "."):
            values = shlex.split(line)
            if len(values) == 2:
                singleton[values[0].split(".", 1)[1]] = values[1]
    if singleton:
        return [singleton]
    lines = iter(text.splitlines())
    for line in lines:
        if line.strip() != "loop_":
            continue
        headers = []
        for line in lines:
            if line.startswith("_"):
                headers.append(line.strip())
            else:
                break
        values = []
        while line.strip() and not line.lstrip().startswith("#"):
            values.extend(shlex.split(line))
            line = next(lines, "#")
        if headers and headers[0].startswith(category + "."):
            if len(values) % len(headers):
                raise ValueError(f"Incomplete {category} loop")
            names = [header.split(".", 1)[1] for header in headers]
            return [dict(zip(names, values[i : i + len(headers)])) for i in range(0, len(values), len(headers))]
    return []


def deposited_atoms(text):
    """First model, blank/A alternatives, and deposited heavy atoms only."""
    result = []
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith(("ATOM  ", "HETATM")) and line[16] in " A":
            if line[76:78].strip() not in ("H", "D"):
                result.append(line)
    return result


def residue_key(line):
    return line[17:20].strip(), line[21], int(line[22:26]), line[26]


def ligand_from_ccd(lines, ccd):
    """Use deposited coordinates and CCD connectivity, H counts, and charges."""
    atom_rows = ccd_loop(ccd, "_chem_comp_atom")
    bond_rows = ccd_loop(ccd, "_chem_comp_bond")
    chemistry = {row["atom_id"]: row for row in atom_rows}
    names = [line[12:16].strip() for line in lines]
    if set(names) != {row["atom_id"] for row in atom_rows if row["type_symbol"] not in ("H", "D")}:
        raise ValueError("Example requires a complete deposited heavy-atom ligand")
    hydrogens = {name: 0 for name in names}
    for row in bond_rows:
        for first, second in ((row["atom_id_1"], row["atom_id_2"]), (row["atom_id_2"], row["atom_id_1"])):
            if first in hydrogens and chemistry[second]["type_symbol"] in ("H", "D"):
                hydrogens[first] += 1
    molecule = Chem.RWMol()
    for name in names:
        atom = Chem.Atom(chemistry[name]["type_symbol"].title())
        atom.SetFormalCharge(int(chemistry[name]["charge"]))
        atom.SetNumExplicitHs(hydrogens[name])
        molecule.AddAtom(atom)
    indices = {name: index for index, name in enumerate(names)}
    orders = {
        "SING": Chem.BondType.SINGLE,
        "DOUB": Chem.BondType.DOUBLE,
        "TRIP": Chem.BondType.TRIPLE,
        "AROM": Chem.BondType.AROMATIC,
    }
    for row in bond_rows:
        first, second = row["atom_id_1"], row["atom_id_2"]
        if first in indices and second in indices:
            molecule.AddBond(indices[first], indices[second], orders[row["value_order"]])
    molecule = molecule.GetMol()
    Chem.SanitizeMol(molecule)
    conformer = Chem.Conformer(len(names))
    conformer.Set3D(True)
    for index, line in enumerate(lines):
        conformer.SetAtomPosition(index, tuple(float(line[i : i + 8]) for i in (30, 38, 46)))
    molecule.AddConformer(conformer)
    return molecule, names


def pocket_block(lines, ligand, retain_metals=False):
    """Whole protein residues within 6 Å; omit other hetero compounds and waters."""
    protein = [line for line in lines if line.startswith("ATOM  ")]
    xyz = np.array([[float(line[i : i + 8]) for i in (30, 38, 46)] for line in protein])
    near = cdist(xyz, ligand.GetConformer().GetPositions()).min(axis=1) <= 6.0
    residues = {residue_key(line) for line, keep in zip(protein, near) if keep}
    selected = [line for line in protein if residue_key(line) in residues]
    if retain_metals:
        for line in lines:
            if line.startswith("HETATM") and line[76:78].strip() in ("ZN", "CA"):
                position = np.array([float(line[i : i + 8]) for i in (30, 38, 46)])
                if np.linalg.norm(ligand.GetConformer().GetPositions() - position, axis=1).min() <= 6.0:
                    selected.append(line)
    # Preserve deposited atom names/serials; normalize the selected alternate to blank.
    return "\n".join(line[:16] + " " + line[17:] for line in selected) + "\nEND\n"


def prepare(protein_pdb, ligand_sdf):
    """Fixed RDKit backend: no minimization or movement of deposited heavy atoms."""
    protein = rdk.readstring("pdb", protein_pdb)
    protein.protein = True
    protein.addh(only_polar=True)
    ligand = rdk.readstring("sdf", ligand_sdf)
    ligand.addh(only_polar=True)
    return protein, ligand


def endpoint(molecule, record, role, scene):
    is_ring = "centroid" in record.dtype.names
    position = record["centroid"] if is_ring else record["coords"]
    if is_ring:
        rings = [
            ring
            for ring in molecule.Mol.GetRingInfo().AtomRings()
            if all(molecule.Mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        ]
        indices = next(
            list(ring) for ring in rings if np.allclose(molecule.coords[list(ring)].mean(axis=0), position, atol=0.001)
        )
    else:
        indices = [int(record["id"])]
    if role == "protein":
        info = molecule.Mol.GetAtomWithIdx(indices[0]).GetPDBResidueInfo()
        residue = dict(
            name=info.GetResidueName().strip(),
            chain=info.GetChainId(),
            number=info.GetResidueNumber(),
            insertion=info.GetInsertionCode().strip(),
        )
        atom_names = [molecule.Mol.GetAtomWithIdx(i).GetPDBResidueInfo().GetName().strip() for i in indices]
    else:
        residue = dict(name=scene["component"], chain=scene["chain"], number=scene["number"], insertion="")
        atom_names = [scene["ligand_atom_names"][i] for i in indices]
    ring_number = None
    if is_ring:
        if role == "protein":
            residue_rings = []
            for ring in rings:
                ring_info = molecule.Mol.GetAtomWithIdx(ring[0]).GetPDBResidueInfo()
                if (ring_info.GetChainId(), ring_info.GetResidueNumber(), ring_info.GetInsertionCode().strip()) == (
                    residue["chain"],
                    residue["number"],
                    residue["insertion"],
                ):
                    residue_rings.append(ring)
        else:
            residue_rings = rings
        ring_number = residue_rings.index(tuple(indices)) + 1
    atom_label = f"ring {ring_number}" if is_ring else atom_names[0]
    label = f"{residue['name']} {residue['chain']}:{residue['number']}{residue['insertion']} {atom_label}"
    result = dict(
        role=role,
        kind="ring" if is_ring else "atom",
        indices=indices,
        atom_names=atom_names,
        residue=residue,
        label=label,
        position=[round(float(value), 6) for value in position],
    )
    if is_ring:
        result.update(ring_number=ring_number, ring_size=len(indices))
    return result


def calculate_contacts(scene, protein, ligand):
    """Return the actual ODDT-selected contacts and reproducible analysis settings."""
    kind = scene["id"]
    geometries = None
    if kind == "hydrogen-bond":
        first, second, mask = interactions.hbonds(protein, ligand, mol1_exact=True, mol2_exact=True)
        mask &= np.isin(first["atomicnum"], [7, 8]) & np.isin(second["atomicnum"], [7, 8])
        settings = dict(
            function="hbonds",
            cutoff=3.5,
            tolerance=30,
            mol1_exact=True,
            mol2_exact=True,
            selection="Unique strict N/O atom pairs only",
        )
    elif kind == "hydrophobic":
        first, second = interactions.hydrophobic_contacts(protein, ligand)
        mask = np.ones(len(first), dtype=bool)
        settings = dict(
            function="hydrophobic_contacts",
            cutoff=4.0,
            selection="Closest atom pair per protein residue, up to eight residues",
        )
    elif kind == "halogen-bond":
        first, second, mask = interactions.halogenbonds(protein, ligand)
        settings = dict(function="halogenbonds", cutoff=4.0, tolerance=30, selection="Strict contacts")
    elif kind == "salt-bridge":
        first, second = interactions.salt_bridges(protein, ligand)
        mask = np.ones(len(first), dtype=bool)
        settings = dict(function="salt_bridges", cutoff=4.0, mol1_exact=False, mol2_exact=False)
    elif kind == "pi-stacking":
        first, second, parallel, perpendicular = interactions.pi_stacking(protein, ligand)
        mask = parallel | perpendicular
        geometries = np.where(parallel, "parallel", "perpendicular")
        settings = dict(function="pi_stacking", cutoff=5.0, tolerance=30, selection="Parallel or perpendicular mask")
    elif kind == "pi-cation":
        first, second, mask = interactions.pi_cation(protein, ligand, cation_exact=True)
        settings = dict(function="pi_cation", cutoff=5.0, tolerance=30, cation_exact=True)
    elif kind == "acceptor-metal":
        second, first, mask = interactions.acceptor_metal(ligand, protein)
        settings = dict(function="acceptor_metal", cutoff=4.0, tolerance=30, argument_order="ligand, protein")
    elif kind == "pi-metal":
        first, second, mask = interactions.pi_metal(protein, ligand)
        settings = dict(function="pi_metal", cutoff=5.0, tolerance=30, argument_order="protein, metal ligand")
    else:
        raise ValueError(kind)
    result = []
    for row in np.flatnonzero(mask):
        start = endpoint(protein, first[row], "protein", scene)
        end = endpoint(ligand, second[row], "ligand", scene)
        separation = float(np.linalg.norm(np.array(start["position"]) - end["position"]))
        contact = dict(start=start, end=end, distance=round(separation, 6))
        if geometries is not None:
            contact["geometry"] = str(geometries[row])
        result.append(contact)
    result.sort(key=lambda contact: contact["distance"])
    # hbonds() searches both directions; dual donor/acceptor features can return
    # the same physical atom pair twice. Render a single edge for that pair.
    unique_pairs = {}
    for contact in result:
        pair = (tuple(contact["start"]["indices"]), tuple(contact["end"]["indices"]))
        unique_pairs.setdefault(pair, contact)
    result = list(unique_pairs.values())
    if kind == "hydrophobic":
        unique = {}
        for contact in result:
            residue = contact["start"]["residue"]
            unique.setdefault((residue["chain"], residue["number"], residue["insertion"]), contact)
        result = list(unique.values())[:8]
    return result, settings, len(first)


def write_depiction(scene, protein, ligand, directory):
    """Draw the stored atom/ring contacts without rerunning contact selection."""
    records = []
    for side, molecule in (("start", protein), ("end", ligand)):
        selected = []
        for contact in scene["contacts"]:
            end = contact[side]
            if end["kind"] == "atom":
                selected.append(molecule.atom_dict[end["indices"][0]])
            else:
                selected.append(
                    next(
                        ring
                        for ring in molecule.ring_dict
                        if np.allclose(ring["centroid"], end["position"], atol=0.001)
                    )
                )
        records.append(np.array(selected))
    for key, filename, show_atoms in (
        ("depiction_file", "interaction.svg", False),
        ("depiction_atoms_file", "interaction-atoms.svg", True),
    ):
        svg = draw_interactions(
            protein, ligand, *records, color=scene["color"], legend=scene["component"], show_protein_atoms=show_atoms
        )
        scene[key] = filename
        scene["assets_sha256"][filename] = hashlib.sha256(svg.encode()).hexdigest()
        (directory / filename).write_text(svg)


def regenerate_depictions(output):
    """Regenerate only the SVG maps, using committed local structures and contacts."""
    manifest = json.loads((output / "manifest.json").read_text())
    for entry in manifest:
        path = output / entry["scene"]
        scene = json.loads(path.read_text())
        protein, ligand = prepare(
            (path.parent / scene["protein_file"]).read_text(), (path.parent / scene["ligand_file"]).read_text()
        )
        write_depiction(scene, protein, ligand, path.parent)
        path.write_text(json.dumps(scene, indent=2, ensure_ascii=False) + "\n")
        print(f"{scene['title']}: {scene['pdb_id']}, {len(scene['contacts'])} 2D contacts", flush=True)


def generate(cache, output):
    manifest = []
    for example in EXAMPLES:
        scene = dict(example)
        pdb_id, component = scene["pdb_id"], scene["component"]
        pdb, pdb_source = fetch(f"https://files.rcsb.org/download/{pdb_id}.pdb", cache / f"{pdb_id}.pdb")
        ccd, ccd_source = fetch(f"https://files.rcsb.org/ligands/download/{component}.cif", cache / f"{component}.cif")
        lines = deposited_atoms(pdb)
        ligand_lines = [
            line for line in lines if residue_key(line) == (component, scene["chain"], scene["number"], " ")
        ]
        ligand, scene["ligand_atom_names"] = ligand_from_ccd(ligand_lines, ccd)
        ligand.SetProp("_Name", f"{pdb_id} {component} {scene['chain']}:{scene['number']}")
        ligand_sdf = Chem.MolToMolBlock(ligand) + "\n$$$$\n"
        protein_pdb = pocket_block(lines, ligand, scene["id"] == "acceptor-metal")
        protein, prepared_ligand = prepare(protein_pdb, ligand_sdf)
        contacts, settings, candidates = calculate_contacts(scene, protein, prepared_ligand)
        if not contacts:
            raise ValueError(f"No verified {scene['id']} in {pdb_id} ({candidates} candidates)")
        scene.update(
            contacts=contacts,
            analysis=settings,
            candidates=candidates,
            sources=[pdb_source, ccd_source],
            backend="rdk",
            oddt_version=oddt.__version__,
            rdkit_version=rdBase.rdkitVersion,
            pocket_radius_angstrom=6.0,
            protein_file="protein.pdb",
            ligand_file="ligand.sdf",
            assets_sha256={
                "protein.pdb": hashlib.sha256(protein_pdb.encode()).hexdigest(),
                "ligand.sdf": hashlib.sha256(ligand_sdf.encode()).hexdigest(),
            },
        )
        directory = output / scene["id"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "protein.pdb").write_text(protein_pdb)
        (directory / "ligand.sdf").write_text(ligand_sdf)
        write_depiction(scene, protein, prepared_ligand, directory)
        (directory / "scene.json").write_text(json.dumps(scene, indent=2, ensure_ascii=False) + "\n")
        manifest.append(
            {key: scene[key] for key in ("id", "title", "pdb_id", "complex", "color")}
            | {"scene": f"{scene['id']}/scene.json", "contact_count": len(contacts)}
        )
        print(f"{scene['title']}: {pdb_id}, {len(contacts)} displayed / {candidates} candidates", flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=REPOSITORY / "docs/_build/interaction-sources")
    parser.add_argument("--output", type=Path, default=REPOSITORY / "docs/_static/interactions")
    parser.add_argument(
        "--depictions-only", action="store_true", help="Redraw 2D SVGs from local assets without downloads"
    )
    args = parser.parse_args()
    if args.depictions_only:
        regenerate_depictions(args.output)
    else:
        generate(args.cache, args.output)
