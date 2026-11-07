"""Render the completed PDBbind comparison as tables and a scientific figure."""

import argparse
import json
import os
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


def render(report, page, figure, source=None):
    summary = summarize(report)
    engines = report["protocol"]["engines"]
    if not {"ga", "mc"}.issubset(engines) or set(engines) - {"ga", "mc", "vina"}:
        raise ValueError("The comparison report requires GA and MC runs, optionally Vina")
    for engine, algorithm in (("ga", "genetic"), ("mc", "monte_carlo")):
        if report["protocol"].get(engine, {}).get("algorithm") != algorithm:
            raise ValueError("Engine %s cannot label results from another search algorithm" % engine)
    targets = report["protocol"]["targets"]
    seed_count = len(report["protocol"]["seeds"])
    seed_word = "seed" if seed_count == 1 else "seeds"
    names = {"ga": "Internal GA", "mc": "Internal MC", "vina": "Vina " + report["versions"].get("vina", "")}
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
    work = []
    for target in targets:
        runs = [row for row in report["results"] if row["target"] == target]
        details = runs[0]
        vina_torsions = next((row["vina_torsions"] for row in runs if "vina_torsions" in row), "--")
        accuracy.append(
            [
                target.upper(),
                details.get("ligand_heavy_atoms", "--"),
                details.get("searched_torsions", "--"),
            ]
            + ([vina_torsions] if "vina" in engines else [])
            + [number(summary[target, engine]["median_rmsd"]) for engine in engines]
        )
        recovery.append(
            [target.upper()]
            + [
                "%d/%d" % (summary[target, engine][metric], summary[target, engine]["attempts"])
                for metric in ("top_recovered", "best_recovered")
                for engine in engines
            ]
        )
        timing.append([target.upper()] + [number(summary[target, engine]["median_seconds"], 1) for engine in engines])
        for engine in engines:
            successful = [row for row in runs if row["engine"] == engine and row["status"] == "ok"]
            counts = []
            for metric in ("local_minimizations", "objective_evaluations"):
                values = [row[metric] for row in successful if metric in row]
                counts.append(number(float(np.median(values)), 0) if values else "--")
            work.append([target.upper(), names[engine], *counts])
    figure_link = os.path.relpath(figure, page.parent).replace(os.sep, "/")
    source = figure.with_suffix(".json") if source is None else source
    download_stem = str(Path(os.path.relpath(source, page.parent)).with_suffix("")).replace(os.sep, "/")
    ga, mc = (report["protocol"][name] for name in ("ga", "mc"))
    parts = [
        "Results\n-------\n",
        "This report uses seeds %s and engines %s. GA: population %s, generations %s, maxiter %s. "
        "MC: %s trajectories, %s proposals each, maxiter %s. These budgets are not matched energy-evaluation counts.\n"
        % (
            ", ".join(map(str, report["protocol"]["seeds"])),
            ", ".join(names[e] for e in engines),
            ga.get("population_size", "--"),
            ga.get("generations", "--"),
            ga.get("maxiter", "--"),
            mc.get("exhaustiveness", "--"),
            mc.get("steps", "--"),
            mc.get("maxiter", "--"),
        ),
        (
            "A single-seed report is a validation probe and cannot establish search robustness across seeds.\n"
            if seed_count == 1
            else ""
        ),
        "RMSDs below are symmetry-aware, in Angstroms. Times are seconds. Per-target values are medians over %d %s.\n"
        % (seed_count, seed_word),
        table(
            "Across all target/seed runs",
            ["Engine", "Top <2 A", "Any returned <2 A", "Median top RMSD", "Median time", "Errors"],
            overall,
        ),
        table(
            "Pose recovery by complex",
            ["Complex", "Heavy atoms", "Internal torsions"]
            + (["Vina torsions"] if "vina" in engines else [])
            + [names[e] + " RMSD" for e in engines],
            accuracy,
        ),
        table(
            "Recovered runs by complex (RMSD <2 A)",
            ["Complex"] + [names[e] + " " + metric for metric in ("top", "any returned") for e in engines],
            recovery,
        ),
        table("Median runtime by complex", ["Complex"] + [names[e] + " (s)" for e in engines], timing),
        table(
            "Median search work (Vina does not expose these counters)",
            ["Complex", "Engine", "Local minimizations", "Objective evaluations"],
            work,
        ),
        ".. figure:: %s\n   :alt: Per-run accuracy and runtime across search engines.\n\n   Dots show individual seeds; larger markers show medians. The dashed line marks 2 Angstrom RMSD. Runtime uses a logarithmic scale.\n"
        % figure_link,
        "Download the :download:`per-run CSV <%s.csv>` or :download:`full JSON report <%s.json>`. The JSON records the protocol, source/input checksums, per-pose RMSDs and search statistics.\n"
        % (download_stem, download_stem),
    ]
    text = page.read_text()
    start = text.index(".. benchmark-results-start")
    end = text.index(".. benchmark-results-end", start)
    page.write_text(text[:start] + ".. benchmark-results-start\n\n" + "\n".join(parts) + "\n" + text[end:])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    styles = {"ga": ("#5746a3", "^"), "mc": ("#087f8c", "o"), "vina": ("#bd4f00", "s")}
    for engine, offset in zip(engines, np.linspace(-0.22, 0.22, len(engines))):
        color, marker = styles[engine]
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
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=len(engines), frameon=False, fontsize=9)
    axes[0].axhline(2, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_ylim(bottom=0)
    axes[1].set_yscale("log")
    fig.suptitle("PDBbind pocket redocking · %d %s per complex · fixed settings" % (seed_count, seed_word))
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
    render(json.loads(args.input.read_text()), args.page, args.figure, source=args.input)


if __name__ == "__main__":
    main()
