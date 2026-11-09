"""Render experiment tables and one scientific overview of the shared archive."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .docking_pdbbind import ROOT, ALGORITHMS, ENGINES, engine_spec

MODEL_NAMES = {"vina": "Vina score", "vinardo": "Vinardo", "ad4": "AD4", "plp": "PLP", "chemplp": "ChemPLP"}
from .docking_benchmark_data import ARCHIVE, load_report


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


def compare_reports(report, reference, experiment):
    """Compose a comparison view without duplicating archived measurements."""
    summarize(report)
    summarize(reference)
    protocol, baseline = report["protocol"], reference["protocol"]
    for key in ("targets", "seeds", "backend"):
        if protocol.get(key) != baseline.get(key):
            raise ValueError("Comparison protocols differ: " + key)
    if set(protocol["engines"]) & set(baseline["engines"]):
        raise ValueError("Comparison engine ids overlap")
    for package in ("numpy", "scipy", "rdkit"):
        if report["versions"].get(package) != reference["versions"].get(package):
            raise ValueError("Comparison numerical versions differ: " + package)
    inputs = {path: digest for path, digest in report.get("sha256", {}).items() if path.startswith("tests/data/")}
    old_inputs = {
        path: digest for path, digest in reference.get("sha256", {}).items() if path.startswith("tests/data/")
    }
    if not inputs or inputs != old_inputs:
        raise ValueError("Comparison input checksums differ or are missing")
    for name in protocol["engines"]:
        _, search = engine_spec(name)
        for old_name in baseline["engines"]:
            if search is not None and engine_spec(old_name)[1] == search:
                current = {key: value for key, value in protocol[name].items() if key != "scoring"}
                previous = {key: value for key, value in baseline[old_name].items() if key != "scoring"}
                if current != previous:
                    raise ValueError("Comparison search budgets differ: " + search)
    merged = deepcopy(report)
    merged["protocol"]["engines"] = [*baseline["engines"], *protocol["engines"]]
    for name in baseline["engines"]:
        merged["protocol"][name] = deepcopy(baseline[name])
    merged["results"] = [*deepcopy(reference["results"]), *merged["results"]]
    merged["comparison_reference"] = {
        "experiment": experiment,
        "report_sha256": hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest(),
        "platform": deepcopy(reference.get("platform")),
    }
    return merged


def table(title, headers, rows):
    lines = [".. list-table:: " + title, "   :header-rows: 1", ""]
    for row in [headers, *rows]:
        lines.extend(("   * - " if i == 0 else "     - ") + str(value) for i, value in enumerate(row))
    return "\n".join(lines) + "\n"


def number(value, precision=2):
    return "--" if value is None else ("%.*f" % (precision, value))


def validate_engines(report):
    """Keep engine labels consistent with the recorded search/scoring models."""
    engines = report["protocol"]["engines"]
    if not engines or set(engines) - set(ENGINES):
        raise ValueError("The comparison report requires supported scoring/search engine ids")
    for engine in engines:
        scoring, search = engine_spec(engine)
        config = report["protocol"].get(engine, {})
        if search is not None and config.get("algorithm") != ALGORITHMS[search]:
            raise ValueError("Engine %s cannot label results from another search algorithm" % engine)
        if engine not in (*ALGORITHMS, "vina") and config.get("scoring") != scoring:
            raise ValueError("Engine %s cannot label results from another scoring model" % engine)
    return engines


def render(report, page, figure=None, source=None, marker="benchmark-results", title="Results"):
    summary = summarize(report)
    engines = validate_engines(report)
    targets = report["protocol"]["targets"]
    seed_count = len(report["protocol"]["seeds"])
    seed_word = "seed" if seed_count == 1 else "seeds"
    names = {
        "ga": "Internal GA",
        "mc": "Internal MC",
        "de": "Internal DE",
        "aco": "Internal ACO",
        "bo": "Internal BO",
        "vina": "Vina " + report["versions"].get("vina", ""),
    }
    names.update(
        {
            name: MODEL_NAMES[engine_spec(name)[0]] + " " + engine_spec(name)[1].upper()
            for name in engines
            if name not in names
        }
    )
    if len({engine_spec(name)[0] for name in engines}) > 1:
        for name in engines:
            if name in ALGORITHMS:
                names[name] = "Vina scoring " + name.upper()
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
    figure_link = os.path.relpath(figure, page.parent).replace(os.sep, "/") if figure is not None else None
    source = ARCHIVE if source is None else Path(source)
    download_stem = str(Path(os.path.relpath(source, page.parent)).with_suffix("")).replace(os.sep, "/")
    budgets = []
    for engine in engines:
        config = report["protocol"].get(engine, {})
        _, search = engine_spec(engine)
        if search in ("ga", "de"):
            budgets.append(
                "%s: population %s, generations %s, maxiter %s%s."
                % (
                    names[engine],
                    config.get("population_size", "--"),
                    config.get("generations", "--"),
                    config.get("maxiter", "--"),
                    ", short local maxiter %s" % config.get("local_maxiter", "--") if search == "de" else "",
                )
            )
        elif search == "aco":
            budgets.append(
                "%s: archive %s, %s ants, iterations %s, maxiter %s."
                % (
                    names[engine],
                    config.get("archive_size", "--"),
                    config.get("ants", "--"),
                    config.get("iterations", "--"),
                    config.get("maxiter", "--"),
                )
            )
        elif search == "bo":
            budgets.append(
                "%s: initial %s, iterations %s, candidates %s, GP samples %s, maxiter %s."
                % (
                    names[engine],
                    config.get("n_initial", "--"),
                    config.get("iterations", "--"),
                    config.get("candidate_pool", "--"),
                    config.get("max_samples", "--"),
                    config.get("maxiter", "--"),
                )
            )
        elif search == "mc":
            budgets.append(
                "%s: %s trajectories, %s proposals each, maxiter %s."
                % (
                    names[engine],
                    config.get("exhaustiveness", "--"),
                    config.get("steps", "--"),
                    config.get("maxiter", "--"),
                )
            )
    diagnostic_rows = []
    for row in report["results"]:
        diagnostic = row.get("diagnostics", {})
        if row["status"] == "ok" and "sampled_best_symmetry_rmsd" in diagnostic:
            diagnostic_rows.append(
                [
                    row["target"].upper(),
                    row["seed"],
                    names[row["engine"]],
                    number(diagnostic["sampled_best_symmetry_rmsd"]),
                    number(diagnostic["uncapped_best_symmetry_rmsd"]),
                    number(row["best_symmetry_rmsd"]),
                ]
            )
    parts = [
        title + "\n" + "-" * len(title) + "\n",
        "This report uses seeds %s and engines %s. %s These budgets are not matched energy-evaluation counts.\n"
        % (
            ", ".join(map(str, report["protocol"]["seeds"])),
            ", ".join(names[e] for e in engines),
            " ".join(budgets),
        ),
        (
            "A single-seed report is a validation probe and cannot establish search robustness across seeds.\n"
            if seed_count == 1
            else ""
        ),
        (
            "The Vina-scoring reference is reused from the archived ``%s`` experiment, not rerun. "
            "Target/seed grids, search budgets, numerical package versions and input checksums match. "
            "Its timing measurements are historical; source versions remain recorded separately in the archive.\n"
            % report["comparison_reference"]["experiment"]
            if "comparison_reference" in report
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
            "Median search work" + (" (Vina does not expose these counters)" if "vina" in engines else ""),
            ["Complex", "Engine", "Local minimizations", "Objective evaluations"],
            work,
        ),
        (
            table(
                "Sampled versus returned poses (per run, symmetry RMSD in Angstroms)",
                ["Complex", "Seed", "Engine", "Closest sampled", "Closest uncapped", "Best returned"],
                diagnostic_rows,
            )
            + "\nSampled results include discarded proposals and short/capped local searches. "
            "The uncapped subset includes short searches. Reference minimization diagnostics and "
            "same-scorer total energies are available in the JSON. These diagnostics do not guide the search.\n"
            if diagnostic_rows
            else ""
        ),
        (
            ".. figure:: %s\n   :alt: Per-run accuracy and runtime across docking engines.\n\n   Dots show individual seeds; larger markers show medians. The dashed line marks 2 Angstrom RMSD. Runtime uses a logarithmic scale.\n"
            % figure_link
            if figure is not None
            else ""
        ),
        (
            "Download the :download:`per-run CSV <%s.csv>` or :download:`full JSON report <%s.json>`. "
            % (download_stem, download_stem)
            if source.with_suffix(".csv").is_file()
            else "Download the :download:`full JSON report <%s.json>`. " % download_stem
        )
        + "The JSON records the protocol, source/input checksums, per-pose RMSDs and search statistics.\n",
    ]
    text = page.read_text()
    start = text.index(".. " + marker + "-start")
    end = text.index(".. " + marker + "-end", start)
    page.write_text(text[:start] + ".. " + marker + "-start\n\n" + "\n".join(parts) + "\n" + text[end:])
    if figure is None:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    styles = {
        "ga": ("#5746a3", "^"),
        "mc": ("#087f8c", "o"),
        "de": ("#43852c", "D"),
        "aco": ("#a23b72", "P"),
        "bo": ("#b38b00", "X"),
        "vina": ("#bd4f00", "s"),
    }
    styles.update(
        {
            name: (
                "#247c91" if name.startswith("vinardo_") else "#43852c",
                {"ga": "^", "mc": "o", "de": "D", "aco": "P", "bo": "X"}[engine_spec(name)[1]],
            )
            for name in engines
            if name not in styles
        }
    )
    offsets = np.linspace(-0.22, 0.22, len(engines)) if len(engines) > 1 else [0]
    for engine, offset in zip(engines, offsets):
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
        axis.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=min(3, len(engines)), frameon=False, fontsize=9
        )
    axes[0].axhline(2, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_ylim(bottom=0)
    axes[1].set_yscale("log")
    fig.suptitle("PDBbind pocket redocking · %d %s per complex · fixed settings" % (seed_count, seed_word))
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure)
    plt.close(fig)


def overview_data(archive):
    """Summarize each archived engine separately, retaining its seed denominator."""
    if archive.get("schema_version") != 1 or not archive.get("experiments"):
        raise ValueError("The overview requires a nonempty consolidated archive")
    targets = list(dict.fromkeys(t for r in archive["experiments"].values() for t in r["protocol"]["targets"]))
    rows = []
    for experiment, report in archive["experiments"].items():
        summary = summarize(report)
        protocol = report["protocol"]
        seeds = len(protocol["seeds"])
        for engine in validate_engines(report):
            scoring, search = engine_spec(engine)
            if search is None:
                label = "Vina " + report["versions"].get("vina", "")
            else:
                model = MODEL_NAMES[scoring]
                population = protocol.get(engine, {}).get("population_size", "") if search in ("ga", "de") else ""
                label = "%s %s%s" % (model, search.upper(), population)
            notes = "%d %s" % (seeds, "seed" if seeds == 1 else "seeds")
            if protocol.get("diagnostics", {}).get("enabled"):
                notes += ", diagnostics"
            rows.append(
                {
                    "experiment": experiment,
                    "engine": engine,
                    "label": "%s: %s (%s)" % (experiment, label, notes),
                    "targets": {t: summary.get((t, engine)) for t in targets},
                    "overall": summary["all", engine],
                }
            )
    return targets, rows


def render_overview(archive, figure):
    """Write one image covering all experiments without pooling their settings."""
    targets, rows = overview_data(archive)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.patches import Rectangle

    with plt.rc_context({"svg.fonttype": "none"}):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, max(5.2, 0.55 * len(rows) + 2)),
            gridspec_kw={"width_ratios": [1, 1, 0.45]},
            constrained_layout=True,
        )
        for column, (metric, title, precision) in enumerate(
            (("median_rmsd", "Top-pose symmetry RMSD (Å)", 2), ("median_seconds", "Docking time (s, log scale)", 1))
        ):
            values = np.array(
                [[r["targets"][t][metric] if r["targets"][t] else np.nan for t in targets] for r in rows],
                dtype=float,
            )
            finite = values[np.isfinite(values)]
            if column == 1:
                if np.any(finite <= 0):
                    raise ValueError("Docking times must be positive for a logarithmic scale")
                low, high = (float(finite.min()), float(finite.max())) if finite.size else (1, 10)
                norm = LogNorm(low, max(high, low * 1.01))
            else:
                norm = Normalize(0, max(2, float(finite.max()) if finite.size else 2))
            axis = axes[column]
            cmap = plt.get_cmap("YlGnBu").with_extremes(bad="#eeeeee")
            im = axis.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, norm=norm)
            axis.set_title(title, pad=12)
            axis.set_xticks(range(len(targets)), [t.upper() for t in targets])
            if column == 0:
                axis.set_yticks(range(len(rows)), [r["label"] for r in rows])
            else:
                axis.set_yticks([])
            axis.tick_params(length=0, labelsize=10)
            if column == 0:
                axis.tick_params(axis="y", pad=10)
            for i in range(len(rows)):
                for j in range(len(targets)):
                    value = values[i, j]
                    color = "white" if np.isfinite(value) and norm(value) > 0.6 else "#16212a"
                    axis.text(
                        j,
                        i,
                        number(value, precision) if np.isfinite(value) else "--",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color=color,
                    )
                    if column == 0 and np.isfinite(value) and value < 2:
                        axis.add_patch(
                            Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor="#26734b", linewidth=2)
                        )
            axis.spines[:].set_visible(False)
            fig.colorbar(im, ax=axis, orientation="horizontal", fraction=0.06, pad=0.08)
        axis = axes[2]
        axis.set_xlim(0, 2)
        axis.set_ylim(len(rows) - 0.5, -0.5)
        axis.set_title("Recovered <2 Å", pad=33)
        axis.set_xticks([0.4, 1.5], ["Top", "Any returned"])
        axis.xaxis.tick_top()
        axis.tick_params(length=0, labelsize=10)
        axis.set_yticks([])
        axis.spines[:].set_visible(False)
        for i, row in enumerate(rows):
            stats = row["overall"]
            for x, key in ((0.4, "top_recovered"), (1.5, "best_recovered")):
                axis.text(x, i, "%d/%d" % (stats[key], stats["attempts"]), ha="center", va="center", fontsize=11)
        fig.suptitle("PDBbind docking benchmark archive", fontsize=18)
        fig.supxlabel(
            "Per-complex medians; budgets differ by row. Outlines: median RMSD <2 Å. Failed runs count as recovery misses.",
            fontsize=10,
        )
        figure = Path(figure)
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure, dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ARCHIVE)
    parser.add_argument(
        "--experiment", help="Archived experiment to render (default: baseline); omit for standalone reports"
    )
    parser.add_argument(
        "--compare-experiment", help="Compatible archived reference experiment to display alongside results"
    )
    parser.add_argument("--marker", default="benchmark-results", help="RST start/end marker prefix")
    parser.add_argument("--title", default="Results", help="Generated report section heading")
    parser.add_argument("--page", type=Path, default=ROOT / "docs/user-guide/docking-benchmarks.rst")
    parser.add_argument("--figure", type=Path, help="Optional image export; only the requested format is written")
    parser.add_argument(
        "--overview", action="store_true", help="Render the archive as one overview image (default: docking.svg)"
    )
    args = parser.parse_args()
    if args.overview:
        if args.experiment or args.compare_experiment:
            parser.error("--overview summarizes every archived experiment; omit experiment selectors")
        render_overview(json.loads(args.input.read_text()), args.figure or args.input.with_suffix(".svg"))
        return
    report = load_report(args.input, args.experiment)
    if args.compare_experiment:
        report = compare_reports(report, load_report(args.input, args.compare_experiment), args.compare_experiment)
    render(report, args.page, args.figure, source=args.input, marker=args.marker, title=args.title)


if __name__ == "__main__":
    main()
