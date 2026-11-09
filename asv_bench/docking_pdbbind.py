"""Compare GA, Monte Carlo, differential evolution and Vina on local complexes.

Run from the repository root with ``python -m asv_bench.docking_pdbbind``.
This is a standalone accuracy/runtime experiment, not an ASV timing benchmark.
Results are checkpointed after every run and can be resumed with ``--resume``.
"""

import argparse
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

from .docking_benchmark_data import RUN_OUTPUT

ROOT = Path(__file__).resolve().parents[1]
MAX_SYMMETRY_MATCHES = 10_000
RMSD_BATCH_SIZE = 256
ALGORITHMS = {"ga": "genetic", "mc": "monte_carlo", "de": "differential_evolution"}
SCORERS = {"vina": "OddtVina", "vinardo": "OddtVinardo", "ad4": "OddtAD4", "plp": "OddtPLP", "chemplp": "OddtChemPLP"}
ENGINES = [
    *ALGORITHMS,
    "vina",
    *(scorer + "_" + search for scorer in SCORERS if scorer != "vina" for search in ALGORITHMS),
]


def _engine_class(scoring, search=None):
    """Resolve ODDT-prefixed classes while retaining the historical benchmark IDs."""
    from oddt import docking

    prefix = SCORERS[scoring]
    return getattr(docking, prefix + ("Scoring" if search is None else search.upper()))


def engine_spec(name):
    """Return scoring model and search; retain historical Vina engine ids."""
    if name == "vina":
        return "vina", None
    if name in ALGORITHMS:
        return "vina", name
    if name in ENGINES:
        return tuple(name.rsplit("_", 1))
    raise ValueError("Unknown engine: " + name)


def scoring_settings(engine):
    """Record the actual scorer's parameters, independent of search settings."""
    data = {"name": engine.scoring_name, "coordinate_atoms": "all" if engine.score_all_atoms else "heavy"}
    if hasattr(engine, "weights"):
        data["weights"] = engine.weights.tolist()
        data["term_order"] = [*engine.term_names, "torsion"]
    if engine.scoring_name == "ad4":
        data.update(
            charge_model=engine.charge_model,
            directional_hb=engine.directional_hb,
            hbond_saturation=engine.hbond_saturation,
            smoothing=engine.smoothing,
            unbound_energy=engine.unbound_energy,
            preparation="Private copies; add H, calculate Gasteiger charges, merge nonpolar H charges",
            angular_model="Explicit lone-pair cosine-squared direct-pair adaptation",
            saturation_model="Joint inter/intraligand capacitated matching",
            terminal_polar_h_torsions="Fixed input geometry",
        )
    elif engine.scoring_name in ("plp", "chemplp"):
        data.update(
            parameters=dict(engine.parameters),
            term_order=list(engine.term_names),
            parameterization=engine.parameterization,
            units=engine.score_units,
            preparation="Private hydrogenated copies; ODDT typing with ester/nitro exceptions",
            torsion_parameters=[
                dict(types=list(types), values=list(values)) for types, values in engine.torsion_parameters.items()
            ],
            torsion_fallback=engine.torsion_fallback,
            torsion_fallback_terms=len(engine.torsion_fallbacks_),
            metal_cutoff=engine.metal_cutoff,
            metal_geometries=[dict(atom=site["atom"], geometry=site["geometry"]) for site in engine.metal_geometries_],
            terminal_polar_h_torsions="Fixed input geometry",
        )
    else:
        data["preparation"] = "ODDT atom typing on shared hydrogenated input; heavy atoms scored"
    return data


def search_settings(
    ga_population=100,
    ga_generations=8,
    mc_exhaustiveness=8,
    mc_steps=250,
    maxiter=60,
    de_population=32,
    de_generations=20,
    de_local_maxiter=10,
):
    """Single source of settings for both execution and report provenance."""
    return {
        "ga": {
            "algorithm": "genetic",
            "population_size": ga_population,
            "generations": ga_generations,
            "maxiter": maxiter,
            "mutation_rate": 0.2,
        },
        "mc": {
            "algorithm": "monte_carlo",
            "exhaustiveness": mc_exhaustiveness,
            "steps": mc_steps,
            "maxiter": maxiter,
            "temperature": 1.2,
            "mutation_amplitude": 2.0,
        },
        "de": {
            "algorithm": "differential_evolution",
            "population_size": de_population,
            "generations": de_generations,
            "maxiter": maxiter,
            "local_maxiter": de_local_maxiter,
            "pbest_fraction": 0.2,
            "history_size": 6,
            "diversity_rmsd": 1.0,
            "archive_size": 64,
            "stagnation_generations": 5,
            "restart_fraction": 0.25,
        },
    }


def discover_targets(directory):
    return sorted(
        folder.name
        for folder in directory.iterdir()
        if (folder / (folder.name + "_pocket.pdb")).is_file() and (folder / (folder.name + "_ligand.sdf")).is_file()
    )


def prepare(directory, target):
    import oddt
    from oddt.docking import OddtVinaScoring
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
    # using the same >=16 A rule for every target and all engines.
    size = np.maximum(16.0, 2 * np.max(np.abs(coords - center), axis=0) + 8.0)
    ligand = reference.clone
    ligand.coords = rotate(ligand.coords, 1.2, -0.7, 2.1) + [12, -7, 8]
    probe = OddtVinaScoring(protein, ligand)
    details = {
        "receptor_heavy_atoms": int(len(probe.rec_dict)),
        "ligand_heavy_atoms": int(len(coords)),
        "searched_torsions": int(len(probe.rotors)),
        "center": center.tolist(),
        "size": size.tolist(),
    }
    return protein, reference, ligand, center, size, details


def _symmetry_graph(molecule):
    """Use literal chemical attributes, not SDF substructure-query predicates."""
    from rdkit import Chem

    source = molecule.Mol
    if not any(atom.HasQuery() for atom in source.GetAtoms()) and not any(
        bond.HasQuery() for bond in source.GetBonds()
    ):
        return source
    graph = Chem.RWMol()
    for atom in source.GetAtoms():
        graph.AddAtom(Chem.Atom(atom))  # Copy chemical attributes into a plain atom.
    for bond in source.GetBonds():
        graph.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
        copied = graph.GetBondWithIdx(graph.GetNumBonds() - 1)
        copied.SetIsAromatic(bond.GetIsAromatic())
        copied.SetBondDir(bond.GetBondDir())
    for original, copied in zip(source.GetBonds(), graph.GetBonds()):
        if original.GetStereoAtoms():
            copied.SetStereoAtoms(*original.GetStereoAtoms())
        copied.SetStereo(original.GetStereo())
    graph = graph.GetMol()
    Chem.SanitizeMol(graph)
    return graph


def pose_metrics(reference, poses):
    """Measure direct and graph-symmetry RMSD without fitting coordinates."""
    coordinates = []
    for pose in poses:
        pose = pose.clone
        pose.removeh()
        coordinates.append(np.asarray(pose.coords, dtype=float))
    return coordinate_metrics(reference, coordinates)


def coordinate_metrics(reference, coordinates):
    """Evaluate heavy-coordinate snapshots, enumerating graph symmetry once."""
    reference = reference.clone
    reference.removeh()
    # The benchmark uses RDKit. Exclude hydrogen permutations and require
    # stereochemistry when enumerating chemical graph automorphisms.
    graph = _symmetry_graph(reference)
    matches = graph.GetSubstructMatches(graph, uniquify=False, useChirality=True, maxMatches=MAX_SYMMETRY_MATCHES)
    if not matches or len(matches) == MAX_SYMMETRY_MATCHES:
        raise ValueError("Could not enumerate all heavy-atom graph automorphisms")
    mappings = np.asarray(matches)
    direct = []
    symmetry = []
    for coords in coordinates:
        coords = np.asarray(coords, dtype=float)
        if coords.shape != reference.coords.shape:
            raise ValueError("Docked pose has a different heavy-atom count")
        difference = coords - reference.coords
        direct.append(float(np.sqrt(np.mean(np.sum(difference**2, axis=-1)))))
        # Large peptides can have tens of thousands of automorphisms. Bound
        # the coordinate temporary while retaining an exact minimum over all.
        best_squared = np.inf
        for start in range(0, len(mappings), RMSD_BATCH_SIZE):
            difference = coords[mappings[start : start + RMSD_BATCH_SIZE]] - reference.coords
            best_squared = min(best_squared, np.min(np.mean(np.sum(difference**2, axis=-1), axis=-1)))
        symmetry.append(float(np.sqrt(best_squared)))
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


def search_diagnostics(protein, reference, box, snapshots, maxiter, top_pose, scoring="vina"):
    """Post-search oracles distinguish sampled poses from final energy ranking.

    Reference coordinates are used only after dock() has finished. No native
    pose or RMSD feeds back into a search or its output selection.
    """
    feasible = [
        (coords, uncapped)
        for coords, uncapped in snapshots
        if np.isfinite(coords).all() and np.all(coords >= box[0] - 1e-4) and np.all(coords <= box[1] + 1e-4)
    ]
    result = {"local_search_results": len(snapshots), "feasible_results": len(feasible)}
    for label, coords in (
        ("sampled", [coords for coords, _ in feasible]),
        ("uncapped", [coords for coords, uncapped in feasible if uncapped]),
    ):
        result[label + "_results"] = len(coords)
        result[label + "_best_symmetry_rmsd"] = (
            coordinate_metrics(reference, coords)["best_symmetry_rmsd"] if coords else None
        )
    scorer = _engine_class(scoring)(protein, reference, box=box)
    native_energy = float(scorer.energy())
    native = scorer.optimize(maxiter=maxiter)
    native_feasible = bool(np.all(native.coords >= box[0] - 1e-4) and np.all(native.coords <= box[1] + 1e-4))
    result["reference"] = {
        "initial_total": native_energy,
        "minimized_total": float(scorer.energy(scorer._result_coords(native))),
        "minimized_symmetry_rmsd": coordinate_metrics(reference, [native.coords])["top_symmetry_rmsd"],
        "minimized_feasible": native_feasible,
    }
    result["returned_top_total"] = float(top_pose.data[scoring + "_total"])
    result["scoring"] = scoring
    return result


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


def run_one(directory, target, seed, engine_name, settings=None, progress=False, diagnostics=False):
    from oddt.spatial import rmsd

    row = {
        "target": target,
        "seed": seed,
        "engine": engine_name,
        "status": "error",
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        scoring, search = engine_spec(engine_name)
        row.update(scoring=scoring, algorithm=ALGORITHMS.get(search, "external_vina"))
        protein, reference, ligand, center, size, details = prepare(directory, target)
        row.update(details)
        row["input_rmsd"] = float(rmsd(reference, ligand))
        box = np.array([center - size / 2, center + size / 2])
        snapshots = []

        def observe(result, cap):
            snapshots.append((result.coords.copy(), cap is None or np.isinf(cap)))

        start = time.perf_counter()
        try:
            if search is not None:
                configuration = dict((settings or search_settings())[search])
                configuration.pop("algorithm")
                engine_class = _engine_class(scoring, search)
                observer = {"sample_callback": observe} if diagnostics else {}
                engine = engine_class(protein, ligand, box=box, **observer)
                if hasattr(engine, "scoring_name"):
                    row["scoring_parameters"] = scoring_settings(engine)
                    row["searched_torsions"] = len(engine.rotors)
                    if engine.score_all_atoms:
                        row["prepared_ligand_atoms"] = len(engine.scoring_coordinates)
                    if scoring == "ad4":
                        row["ligand_polar_hydrogens"] = int((engine.lig_parameters["types"] == "HD").sum())
                        row["ligand_charge"] = float(engine.lig_parameters["charges"].sum())
                poses = engine.dock(**configuration, seed=seed, num_modes=5, energy_range=3.0, progress=progress)
                row["search_stats"] = engine.search_stats_
                for metric in ("local_minimizations", "objective_evaluations"):
                    row[metric] = sum(stats[metric] for stats in engine.search_stats_)
            elif engine_name == "vina":
                poses, row["vina_torsions"] = dock_vina(protein, ligand, center, size, seed)
            else:
                raise ValueError("Unknown engine: " + engine_name)
        finally:
            row["elapsed_seconds"] = time.perf_counter() - start
        if not poses:
            raise RuntimeError("Docking returned no poses")
        row["num_poses"] = len(poses)
        if scoring in ("plp", "chemplp"):
            row["top_score"] = float(poses[0].data[scoring + "_score"])
            row["score_units"] = "empirical score units"
        else:
            row["top_affinity"] = float(poses[0].data[scoring + "_affinity"])
            row["top_score"] = row["top_affinity"]
            row["score_units"] = "kcal/mol"
        row.update(pose_metrics(reference, poses))
        if diagnostics:
            row["diagnostics"] = (
                search_diagnostics(
                    protein, reference, box, snapshots, configuration["maxiter"], poses[0], scoring=scoring
                )
                if engine_name != "vina"
                else {"unavailable": "Vina API does not expose local-search results"}
            )
        row["status"] = "ok"
    except Exception as error:
        row["error"] = "%s: %s" % (type(error).__name__, error)
    return row


def save(report, output):
    """Checkpoint raw results; the archive owns the shared CSV export."""
    report["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "tests/data/pdbbind")
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 42])
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=["ga", "mc", "vina"])
    parser.add_argument("--ga-population", type=int, default=100)
    parser.add_argument("--ga-generations", type=int, default=8)
    parser.add_argument("--mc-exhaustiveness", type=int, default=8)
    parser.add_argument("--mc-steps", type=int, default=250)
    parser.add_argument("--de-population", type=int, default=32)
    parser.add_argument("--de-generations", type=int, default=20)
    parser.add_argument("--de-local-maxiter", type=int, default=10)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--output", type=Path, default=RUN_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true", help="Show tqdm progress bars for internal searches")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Record sampled-pose and minimized-reference diagnostics after search",
    )
    args = parser.parse_args()
    os.environ["ODDT_TOOLKIT"] = "rdk"
    import oddt

    targets = args.targets or discover_targets(args.data)
    if not targets or set(targets) - set(discover_targets(args.data)):
        parser.error("Every target must contain a ligand SDF and pocket PDB")
    if (
        len(set(args.targets or targets)) != len(targets)
        or len(set(args.seeds)) != len(args.seeds)
        or len(set(args.engines)) != len(args.engines)
    ):
        parser.error("Targets, seeds and engines must be unique")
    if (
        args.ga_population < 4
        or args.ga_generations < 0
        or args.de_population < 4
        or args.de_generations < 0
        or min(args.mc_exhaustiveness, args.mc_steps, args.maxiter, args.de_local_maxiter) < 1
    ):
        parser.error("Invalid search budget")
    settings = search_settings(
        args.ga_population,
        args.ga_generations,
        args.mc_exhaustiveness,
        args.mc_steps,
        args.maxiter,
        args.de_population,
        args.de_generations,
        args.de_local_maxiter,
    )
    protocol = {
        "targets": targets,
        "seeds": args.seeds,
        "engines": args.engines,
        "backend": "rdk",
        "receptor": "PDBbind pocket files, as provided, with added hydrogens",
        "preparation": "Add hydrogens without minimizing; preserve input formal charges, ring and conjugated-bond geometry",
        "box": "Reference heavy-atom centroid; each axis max(16 A, twice maximal absolute centered extent + 8 A)",
        "input_pose": "XYZ Euler rotation (1.2, -0.7, 2.1) radians, then displacement (12, -7, 8) A",
        "ga": settings["ga"],
        "mc": dict(settings["mc"], hunt_energy_cap=10.0, refinement_energy_cap="infinity (uncapped limit)"),
        "de": settings["de"],
        "vina": {
            "exhaustiveness": 8,
            "n_cpu": 1,
            "adapter": "Python API; heavy coordinates mapped by PDBQT serial",
            "preparation": "Remove explicit hydrogens, then PDBQT addHs=True for polar donor hydrogens only",
        },
        "all": {"num_modes": 5, "energy_range": 3},
        "timing": "Engine construction and dock(), including Vina grids and I/O; excludes shared preparation and evaluation",
        "diagnostics": {
            "enabled": args.diagnostics,
            "timing": "Snapshot copies are included; RMSDs and reference minimization are evaluated after the timer",
            "sampled": "All feasible local-search results, including discarded proposals and short/capped searches",
            "uncapped": "Subset optimized with energy_cap=None or infinity; includes DE short searches",
        },
        "primary_metric": "Heavy-atom graph-symmetry RMSD without alignment, chirality preserved; recovery < 2 A",
        "symmetry_enumeration": {
            "max_matches": MAX_SYMMETRY_MATCHES,
            "rmsd_batch_size": RMSD_BATCH_SIZE,
            "query_records": "Literal atom/bond attributes; preserve atom order and stereochemistry",
        },
        "limits": "Small local fixture set; truncated pockets and supplied ring/conjugated geometry; not a PDBbind-wide benchmark",
    }
    protocol["scoring_models"] = {
        scoring: scoring_settings(_engine_class(scoring)(None))
        for scoring in sorted({engine_spec(name)[0] for name in args.engines if name != "vina"})
    }
    for name in args.engines:
        scoring, search = engine_spec(name)
        if search is not None and name not in ALGORITHMS:
            scorer_class = _engine_class(scoring)
            protocol[name] = dict(settings[search], scoring=scoring)
            if search == "mc":
                protocol[name].update(
                    hunt_energy_cap=scorer_class.mc_hunt_energy_cap,
                    refinement_energy_cap=scorer_class.mc_refinement_energy_cap,
                )
    packages = ["numpy", "scipy", "rdkit"] + (["vina"] if "vina" in args.engines else [])
    versions = {name: importlib.metadata.version(name) for name in packages}
    source_files = [
        Path(__file__),
        Path(__file__).with_name("docking_benchmark_data.py"),
        Path(oddt.__file__).parent / "docking/internal.py",
        Path(oddt.__file__).parent / "docking/base.py",
        Path(oddt.__file__).parent / "docking/search.py",
        Path(oddt.__file__).parent / "docking/__init__.py",
        Path(oddt.__file__).parent / "docking/AutodockVina.py",
        Path(oddt.__file__).parent / "docking/vinardo.py",
        Path(oddt.__file__).parent / "docking/ad4.py",
        Path(oddt.__file__).parent / "docking/_ad4_hbond.py",
        Path(oddt.__file__).parent / "docking/plp.py",
        Path(oddt.__file__).parent / "docking/_plp_terms.py",
        Path(oddt.__file__).parent / "docking/_plp_preparation.py",
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
                row = run_one(
                    args.data, target, seed, name, settings, progress=args.progress, diagnostics=args.diagnostics
                )
                report["results"].append(row)
                save(report, args.output)
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
