"""Compare Monte Carlo docking and Vina on the local PDBbind test complexes.

Run from the repository root with ``python -m asv_bench.docking_pdbbind``.
This is a standalone accuracy/runtime experiment, not an ASV timing benchmark.
Results are checkpointed after every run and can be resumed with ``--resume``.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def discover_targets(directory):
    return sorted(
        folder.name
        for folder in directory.iterdir()
        if (folder / (folder.name + "_pocket.pdb")).is_file() and (folder / (folder.name + "_ligand.sdf")).is_file()
    )


def prepare(directory, target):
    import oddt
    from oddt.docking import vina_docking
    from oddt.spatial import rotate

    protein = next(oddt.toolkit.readfile("pdb", str(directory / target / (target + "_pocket.pdb"))))
    protein.protein = True
    protein.addh()
    reference = next(oddt.toolkit.readfile("sdf", str(directory / target / (target + "_ligand.sdf"))))
    reference.addh()
    heavy = reference.atom_dict["atomicnum"] != 1
    coords = reference.coords[heavy]
    center = coords.mean(axis=0)
    # Keep at least 4 A between every reference heavy atom and the box wall,
    # using the same >=16 A rule for every target and both engines.
    size = np.maximum(16.0, 2 * np.max(np.abs(coords - center), axis=0) + 8.0)
    ligand = reference.clone
    ligand.coords = rotate(ligand.coords, 1.2, -0.7, 2.1) + [12, -7, 8]
    probe = vina_docking(protein, ligand)
    details = {
        "receptor_heavy_atoms": int(len(probe.rec_dict)),
        "ligand_heavy_atoms": int(len(coords)),
        "searched_torsions": int(len(probe.rotors)),
        "center": center.tolist(),
        "size": size.tolist(),
    }
    return protein, reference, ligand, center, size, details


def pose_metrics(reference, poses):
    """Measure direct and graph-symmetry RMSD without fitting coordinates."""
    reference = reference.clone
    reference.removeh()
    # The benchmark uses RDKit. Exclude hydrogen permutations and require
    # stereochemistry when enumerating chemical graph automorphisms.
    matches = reference.Mol.GetSubstructMatches(reference.Mol, uniquify=False, useChirality=True, maxMatches=10000)
    if not matches or len(matches) == 10000:
        raise ValueError("Could not enumerate all heavy-atom graph automorphisms")
    mappings = np.asarray(matches)
    direct = []
    symmetry = []
    for pose in poses:
        pose = pose.clone
        pose.removeh()
        coords = np.asarray(pose.coords, dtype=float)
        if coords.shape != reference.coords.shape:
            raise ValueError("Docked pose has a different heavy-atom count")
        difference = coords - reference.coords
        direct.append(float(np.sqrt(np.mean(np.sum(difference**2, axis=-1)))))
        difference = coords[mappings] - reference.coords
        symmetry.append(float(np.sqrt(np.min(np.mean(np.sum(difference**2, axis=-1), axis=-1)))))
    return {
        "pose_direct_rmsds": direct,
        "pose_symmetry_rmsds": symmetry,
        "symmetry_mappings": len(matches),
        "top_direct_rmsd": direct[0],
        "best_direct_rmsd": min(direct),
        "top_symmetry_rmsd": symmetry[0],
        "best_symmetry_rmsd": min(symmetry),
        "top_recovered_2A": symmetry[0] < 2.0,
        "best_recovered_2A": min(symmetry) < 2.0,
    }


def read_vina_poses(ligand, pdbqt):
    """Transfer heavy coordinates by original PDBQT serial, preserving topology."""
    heavy_serials = np.flatnonzero(ligand.atom_dict["atomicnum"] != 1) + 1
    template = ligand.clone
    template.removeh()
    poses = []
    for block in pdbqt.split("ENDMDL"):
        records = [line for line in block.splitlines() if line.startswith(("ATOM  ", "HETATM"))]
        if not records:
            continue
        coords = {int(line[6:11]): [float(line[30:38]), float(line[38:46]), float(line[46:54])] for line in records}
        if not set(heavy_serials).issubset(coords):
            raise ValueError("Vina output is missing original heavy-atom serials")
        pose = template.clone
        pose.coords = np.array([coords[int(serial)] for serial in heavy_serials])
        affinity = next(line[19:].split()[0] for line in block.splitlines() if line.startswith("REMARK VINA RESULT"))
        pose.data.update({"vina_affinity": float(affinity)})
        poses.append(pose)
    return poses


def dock_vina(protein, ligand, center, size, seed):
    from vina import Vina

    # Use the installed engine directly: the general RDKit wrapper copies whole
    # conformers and fails when PDBQT reading removes some explicit hydrogens.
    with TemporaryDirectory(prefix="oddt-pdbbind-vina-") as directory:
        receptor_file = str(Path(directory) / "receptor.pdbqt")
        ligand_file = str(Path(directory) / "ligand.pdbqt")
        # The writer's torsion SMARTS depend on atom degree. Nonpolar explicit
        # hydrogens can make terminal methyl bonds look rotatable. Start from
        # implicit hydrogens and let the writer add only polar donor hydrogens.
        receptor = protein.clone
        receptor.removeh()
        prepared_ligand = ligand.clone
        prepared_ligand.removeh()
        receptor.write("pdbqt", receptor_file, overwrite=True, flexible=False, addHs=True)
        prepared_ligand.write("pdbqt", ligand_file, overwrite=True, flexible=True, addHs=True)
        pdbqt = Path(ligand_file).read_text()
        torsions = int(next(line.split()[1] for line in pdbqt.splitlines() if line.startswith("TORSDOF")))
        engine = Vina(sf_name="vina", cpu=1, seed=seed, verbosity=0)
        engine.set_receptor(receptor_file)
        engine.set_ligand_from_file(ligand_file)
        engine.compute_vina_maps(center=center.tolist(), box_size=size.tolist())
        engine.dock(exhaustiveness=8, n_poses=5)
        return read_vina_poses(prepared_ligand, engine.poses(n_poses=5, energy_range=3)), torsions


def run_one(directory, target, seed, engine_name):
    from oddt.docking import vina_docking
    from oddt.spatial import rmsd

    row = {
        "target": target,
        "seed": seed,
        "engine": engine_name,
        "status": "error",
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        protein, reference, ligand, center, size, details = prepare(directory, target)
        row.update(details)
        row["input_rmsd"] = float(rmsd(reference, ligand))
        start = time.perf_counter()
        try:
            if engine_name == "internal":
                engine = vina_docking(protein, ligand, box=np.array([center - size / 2, center + size / 2]))
                poses = engine.dock(exhaustiveness=8, steps=250, maxiter=60, seed=seed, num_modes=5)
                row["search_stats"] = engine.search_stats_
            else:
                poses, row["vina_torsions"] = dock_vina(protein, ligand, center, size, seed)
        finally:
            row["elapsed_seconds"] = time.perf_counter() - start
        if not poses:
            raise RuntimeError("Docking returned no poses")
        row["num_poses"] = len(poses)
        row["top_affinity"] = float(poses[0].data["vina_affinity"])
        row.update(pose_metrics(reference, poses))
        row["status"] = "ok"
    except Exception as error:
        row["error"] = "%s: %s" % (type(error).__name__, error)
    return row


def save(report, output):
    report["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    columns = (
        "target",
        "seed",
        "engine",
        "status",
        "elapsed_seconds",
        "ligand_heavy_atoms",
        "searched_torsions",
        "vina_torsions",
        "receptor_heavy_atoms",
        "num_poses",
        "top_direct_rmsd",
        "best_direct_rmsd",
        "top_symmetry_rmsd",
        "best_symmetry_rmsd",
        "top_recovered_2A",
        "best_recovered_2A",
        "error",
    )
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["results"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "tests/data/pdbbind")
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 42])
    parser.add_argument("--engines", nargs="+", choices=["internal", "vina"], default=["internal", "vina"])
    parser.add_argument("--output", type=Path, default=ROOT / "docs/_static/benchmarks/pdbbind_docking.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.environ["ODDT_TOOLKIT"] = "rdk"
    import oddt

    targets = args.targets or discover_targets(args.data)
    if not targets or set(targets) - set(discover_targets(args.data)):
        parser.error("Every target must contain a ligand SDF and pocket PDB")
    protocol = {
        "targets": targets,
        "seeds": args.seeds,
        "engines": args.engines,
        "backend": "rdk",
        "receptor": "PDBbind pocket files, as provided, with added hydrogens",
        "preparation": "Add hydrogens without minimizing; preserve input formal charges, ring and conjugated-bond geometry",
        "box": "Reference heavy-atom centroid; each axis max(16 A, twice maximal absolute centered extent + 8 A)",
        "input_pose": "XYZ Euler rotation (1.2, -0.7, 2.1) radians, then displacement (12, -7, 8) A",
        "internal": {
            "algorithm": "monte_carlo",
            "exhaustiveness": 8,
            "steps": 250,
            "maxiter": 60,
            "temperature": 1.2,
            "mutation_amplitude": 2.0,
            "hunt_energy_cap": 10.0,
        },
        "vina": {
            "exhaustiveness": 8,
            "n_cpu": 1,
            "adapter": "Python API; heavy coordinates mapped by PDBQT serial",
            "preparation": "Remove explicit hydrogens, then PDBQT addHs=True for polar donor hydrogens only",
        },
        "both": {"num_modes": 5, "energy_range": 3},
        "timing": "Engine construction and dock(), including Vina grids and I/O; excludes shared preparation and evaluation",
        "primary_metric": "Heavy-atom graph-symmetry RMSD without alignment, chirality preserved; recovery < 2 A",
        "limits": "Small local fixture set; truncated pockets and supplied ring/conjugated geometry; not a PDBbind-wide benchmark",
    }
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "rdkit", "vina")}
    source_files = [
        Path(__file__),
        Path(oddt.__file__).parent / "docking/internal.py",
        Path(oddt.__file__).parent / "docking/AutodockVina.py",
    ]
    source_files.extend(
        args.data / target / (target + suffix) for target in targets for suffix in ("_ligand.sdf", "_pocket.pdb")
    )
    checksums = {
        os.path.relpath(path.resolve(), ROOT): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files
    }
    if args.resume and args.output.is_file():
        report = json.loads(args.output.read_text())
        if report["protocol"] != protocol or report["versions"] != versions or report["sha256"] != checksums:
            parser.error("Saved protocol, versions or source checksums differ; use a new output path")
    else:
        report = {
            "protocol": protocol,
            "versions": versions,
            "sha256": checksums,
            "platform": {
                "system": platform.platform(),
                "machine": platform.machine(),
                "python": sys.version,
                "logical_cpus": os.cpu_count(),
            },
            "results": [],
        }
    completed = {(row["target"], row["seed"], row["engine"]) for row in report["results"]}
    for target in targets:
        for seed in args.seeds:
            for name in args.engines:
                if (target, seed, name) in completed:
                    continue
                print("START %s seed=%d %s" % (target, seed, name), flush=True)
                row = run_one(args.data, target, seed, name)
                report["results"].append(row)
                save(report, args.output)
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
