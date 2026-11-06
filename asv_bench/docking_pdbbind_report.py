"""Render the completed PDBbind comparison as tables and a scientific figure."""

import argparse
import json
from pathlib import Path

import numpy as np

from .docking_pdbbind import ROOT


def summarize(report):
    """Reject incomplete experiments and count failed runs as recovery misses."""
    protocol = report["protocol"]
    expected = {
        (target, seed, engine)
        for target in protocol["targets"]
        for seed in protocol["seeds"]
        for engine in protocol["engines"]
    }
    keys = [(row["target"], row["seed"], row["engine"]) for row in report["results"]]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Incomplete or duplicated benchmark: %d of %d unique runs" % (len(set(keys)), len(expected)))
    result = {}
    for target in [*protocol["targets"], "all"]:
        for engine in protocol["engines"]:
            runs = [
                row
                for row in report["results"]
                if row["engine"] == engine and (target == "all" or row["target"] == target)
            ]
            successful = [row for row in runs if row["status"] == "ok"]
            result[target, engine] = {
                "attempts": len(runs),
                "completed": len(successful),
                "top_recovered": sum(row["top_recovered_2A"] for row in successful),
                "best_recovered": sum(row["best_recovered_2A"] for row in successful),
                "median_rmsd": (
                    float(np.median([row["top_symmetry_rmsd"] for row in successful])) if successful else None
                ),
                "median_seconds": (
                    float(np.median([row["elapsed_seconds"] for row in successful])) if successful else None
                ),
            }
    return result


def table(title, headers, rows):
    lines = [".. list-table:: " + title, "   :header-rows: 1", ""]
    for row in [headers, *rows]:
        lines.extend(("   * - " if i == 0 else "     - ") + str(value) for i, value in enumerate(row))
    return "\n".join(lines) + "\n"


def number(value, precision=2):
    return "--" if value is None else ("%.*f" % (precision, value))


def render(report, page, figure):
    summary = summarize(report)
    if report["protocol"].get("internal", {}).get("algorithm") != "monte_carlo":
        raise ValueError("The Monte Carlo report cannot label results from another search algorithm")
    if set(report["protocol"]["engines"]) != {"internal", "vina"}:
        raise ValueError("The comparison report requires both internal and Vina runs")
    targets = report["protocol"]["targets"]
    seed_count = len(report["protocol"]["seeds"])
    names = {"internal": "Internal MC", "vina": "Vina " + report["versions"]["vina"]}
    overall = []
    for engine in report["protocol"]["engines"]:
        stats = summary["all", engine]
        overall.append(
            [
                names[engine],
                "%d/%d" % (stats["top_recovered"], stats["attempts"]),
                "%d/%d" % (stats["best_recovered"], stats["attempts"]),
                number(stats["median_rmsd"]),
                number(stats["median_seconds"], 1),
                stats["attempts"] - stats["completed"],
            ]
        )
    accuracy = []
    timing = []
    recovery = []
    for target in targets:
        runs = [row for row in report["results"] if row["target"] == target]
        details = runs[0]
        vina_torsions = next((row["vina_torsions"] for row in runs if "vina_torsions" in row), "--")
        internal, vina = (summary[target, name] for name in ("internal", "vina"))
        accuracy.append(
            [
                target.upper(),
                details.get("ligand_heavy_atoms", "--"),
                details.get("searched_torsions", "--"),
                vina_torsions,
                number(internal["median_rmsd"]),
                number(vina["median_rmsd"]),
            ]
        )
        recovery.append(
            [target.upper()]
            + [
                "%d/%d" % (summary[target, engine][metric], summary[target, engine]["attempts"])
                for engine, metric in (
                    ("internal", "top_recovered"),
                    ("vina", "top_recovered"),
                    ("internal", "best_recovered"),
                    ("vina", "best_recovered"),
                )
            ]
        )
        timing.append([target.upper(), number(internal["median_seconds"], 1), number(vina["median_seconds"], 1)])
    parts = [
        "Results\n-------\n",
        "RMSDs below are symmetry-aware, in Angstroms. Times are seconds. Per-target values are medians over %d seeds.\n"
        % seed_count,
        table(
            "Across all target/seed runs",
            ["Engine", "Top <2 A", "Any returned <2 A", "Median top RMSD", "Median time", "Errors"],
            overall,
        ),
        table(
            "Pose recovery by complex",
            ["Complex", "Heavy atoms", "MC torsions", "Vina torsions", "MC RMSD", "Vina RMSD"],
            accuracy,
        ),
        table(
            "Recovered runs by complex (RMSD <2 A)",
            ["Complex", "MC top", "Vina top", "MC any returned", "Vina any returned"],
            recovery,
        ),
        table("Median runtime by complex", ["Complex", "Internal MC (s)", "Vina (s)"], timing),
        ".. figure:: ../_static/benchmarks/pdbbind_docking.svg\n   :alt: Per-run accuracy and runtime for both engines across the selected PDBbind complexes.\n\n   Dots show individual seeds; larger markers show medians. The dashed line marks 2 Angstrom RMSD. Runtime uses a logarithmic scale.\n",
        "Download the :download:`per-run CSV <../_static/benchmarks/pdbbind_docking.csv>` or the :download:`full JSON report <../_static/benchmarks/pdbbind_docking.json>`. The JSON records the protocol, source/input checksums, per-pose RMSDs and Monte Carlo acceptance statistics.\n",
    ]
    text = page.read_text()
    start = text.index(".. benchmark-results-start")
    end = text.index(".. benchmark-results-end", start)
    page.write_text(text[:start] + ".. benchmark-results-start\n\n" + "\n".join(parts) + "\n" + text[end:])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for engine, offset, color, marker in (("internal", -0.13, "#087f8c", "o"), ("vina", 0.13, "#bd4f00", "s")):
        for column, (metric, aggregate, label) in enumerate(
            (
                ("top_symmetry_rmsd", "median_rmsd", "Top-pose symmetry RMSD (Å)"),
                ("elapsed_seconds", "median_seconds", "Runtime (s)"),
            )
        ):
            for i, target in enumerate(targets):
                values = [
                    row[metric]
                    for row in report["results"]
                    if row["target"] == target and row["engine"] == engine and row["status"] == "ok"
                ]
                axes[column].scatter(np.full(len(values), i + offset), values, c=color, s=20, alpha=0.35, marker=marker)
                median = summary[target, engine][aggregate]
                if median is not None:
                    axes[column].scatter(
                        i + offset, median, c=color, s=65, marker=marker, label=names[engine] if i == 0 else None
                    )
            axes[column].set_ylabel(label)
    for axis in axes:
        axis.set_xticks(range(len(targets)), [target.upper() for target in targets], rotation=30)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    axes[0].axhline(2, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_ylim(bottom=0)
    axes[1].set_yscale("log")
    fig.suptitle("PDBbind pocket redocking · %d seeds per complex · fixed settings" % seed_count)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure)
    fig.savefig(figure.with_suffix(".png"), dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "docs/_static/benchmarks/pdbbind_docking.json")
    parser.add_argument("--page", type=Path, default=ROOT / "docs/user-guide/docking-benchmarks.rst")
    parser.add_argument("--figure", type=Path, default=ROOT / "docs/_static/benchmarks/pdbbind_docking.svg")
    args = parser.parse_args()
    render(json.loads(args.input.read_text()), args.page, args.figure)


if __name__ == "__main__":
    main()
