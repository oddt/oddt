"""Store published docking experiments in one JSON archive and derived CSV.

Add a completed run with ``python -m asv_bench.docking_benchmark_data
--input benchmark-results/docking.json --experiment my_experiment``.
Experiments retain independent protocols, versions, checksums and results.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/_static/benchmarks/docking.json"
RUN_OUTPUT = ROOT / "benchmark-results/docking.json"
RESULT_COLUMNS = (
    "target",
    "seed",
    "engine",
    "scoring",
    "algorithm",
    "status",
    "elapsed_seconds",
    "top_affinity",
    "top_score",
    "score_units",
    "local_minimizations",
    "objective_evaluations",
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


def load_report(path, experiment=None):
    """Read a standalone report or select one archived PDBbind experiment."""
    data = json.loads(Path(path).read_text())
    if "experiments" not in data:
        if experiment is not None:
            raise ValueError("--experiment requires a consolidated archive")
        return data
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported docking archive schema")
    name = experiment or "baseline"
    if name not in data["experiments"]:
        raise ValueError("Unknown experiment %r; choose from %s" % (name, ", ".join(data["experiments"])))
    return data["experiments"][name]


def write_csv(data, path):
    """Export per-run metrics, identifying experiments when writing an archive.

    Validation and profiling have different schemas and remain in JSON only.
    """
    archived = "experiments" in data
    columns = ("experiment", *RESULT_COLUMNS) if archived else RESULT_COLUMNS
    reports = data["experiments"].items() if archived else [(None, data)]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for name, report in reports:
            for row in report["results"]:
                writer.writerow(dict(row, experiment=name) if archived else row)


def save_archive(data, path=ARCHIVE):
    """Write the archive and regenerate its CSV from the same records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    write_csv(data, path.with_suffix(".csv"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--input", type=Path, required=True, help="Completed standalone PDBbind report")
    parser.add_argument("--experiment", required=True, help="Unique experiment name in the archive")
    args = parser.parse_args()
    # Import here to keep the shared I/O module independent of the renderer.
    from .docking_pdbbind_report import summarize

    data = (
        json.loads(args.archive.read_text())
        if args.archive.exists()
        else {
            "schema_version": 1,
            "experiments": {},
            "validation": {},
            "source_files": {},
        }
    )
    if data.get("schema_version") != 1 or "experiments" not in data:
        parser.error("Unsupported docking archive schema")
    if (
        not args.experiment.strip()
        or args.experiment in data["experiments"]
        or args.experiment in data.get("source_files", {})
    ):
        parser.error("Use a new, nonempty experiment name; existing experiments are preserved")
    source = args.input.read_bytes()
    report = json.loads(source)
    if "experiments" in report:
        parser.error("--input must be a standalone report")
    try:
        summarize(report)  # Reject incomplete or duplicated target/seed/engine grids.
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    data["experiments"][args.experiment] = report
    data["source_files"][args.experiment] = {
        "file": args.input.name,
        "sha256": hashlib.sha256(source).hexdigest(),
        "section": "experiments",
    }
    save_archive(data, args.archive)


if __name__ == "__main__":
    main()
