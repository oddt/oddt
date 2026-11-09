"""Verify the pose metrics used by the PDBbind comparison."""

import numpy as np
import pytest

from asv_bench.docking_pdbbind import dock_vina, pose_metrics, read_vina_poses
from asv_bench.docking_pdbbind_report import render, summarize


@pytest.fixture
def benzene():
    rdk = pytest.importorskip("oddt.toolkits.rdk")
    from rdkit import Chem

    raw = Chem.MolFromSmiles("c1ccccc1")
    conformer = Chem.Conformer(6)
    angles = np.arange(6) * np.pi / 3
    for i, angle in enumerate(angles):
        conformer.SetAtomPosition(i, (float(np.cos(angle)), float(np.sin(angle)), 0.0))
    raw.AddConformer(conformer)
    molecule = rdk.Molecule(raw)
    molecule.addh()
    return molecule


def test_benchmark_rmsd_does_not_align_poses(benzene):
    shifted = benzene.clone
    shifted.coords = shifted.coords + [3, 0, 0]
    metrics = pose_metrics(benzene, [shifted, benzene.clone])
    assert metrics["top_direct_rmsd"] == pytest.approx(3)
    assert metrics["top_symmetry_rmsd"] == pytest.approx(3)
    assert metrics["best_symmetry_rmsd"] == pytest.approx(0, abs=1e-7)
    assert metrics["top_recovered_2A"] is False
    assert metrics["best_recovered_2A"] is True


def test_benchmark_rmsd_recognizes_graph_symmetry(benzene):
    permuted = benzene.clone
    coords = permuted.coords.copy()
    heavy = permuted.atom_dict["atomicnum"] != 1
    coords[heavy] = np.roll(coords[heavy], 1, axis=0)
    permuted.coords = coords
    metrics = pose_metrics(benzene, [permuted])
    assert metrics["top_direct_rmsd"] == pytest.approx(1)
    assert metrics["top_symmetry_rmsd"] == pytest.approx(0, abs=1e-7)
    assert metrics["symmetry_mappings"] == 12


def test_benchmark_handles_query_atoms_and_bonds_in_1nlp():
    from pathlib import Path
    from rdkit import Chem
    from oddt.toolkits import rdk
    from asv_bench.docking_pdbbind import _symmetry_graph

    path = Path(__file__).parent / "data/pdbbind/1nlp/1nlp_ligand.sdf"
    reference = next(rdk.readfile("sdf", str(path)))
    reference.removeh()
    assert any(atom.HasQuery() for atom in reference.Mol.GetAtoms())
    assert any(bond.HasQuery() for bond in reference.Mol.GetBonds())
    graph = _symmetry_graph(reference)
    assert Chem.FindMolChiralCenters(graph) == Chem.FindMolChiralCenters(reference.Mol)
    assert [(a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsotope()) for a in graph.GetAtoms()] == [
        (a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsotope()) for a in reference.Mol.GetAtoms()
    ]
    metrics = pose_metrics(reference, [reference.clone])
    assert metrics["symmetry_mappings"] == 16
    assert metrics["top_symmetry_rmsd"] == pytest.approx(0, abs=1e-7)


def test_vina_reader_preserves_heavy_atom_order_with_interleaved_hydrogens(benzene):
    from rdkit import Chem
    from oddt.toolkits import rdk

    # Existing conformer copying cannot handle an output with fewer hydrogens.
    # Serial-based transfer must also work when heavy serials are nonconsecutive.
    order = [index for pair in zip(range(6), range(6, 12)) for index in pair]
    molecule = rdk.Molecule(Chem.RenumberAtoms(benzene.Mol, order))
    pdbqt = molecule.write("pdbqt")
    records = [line for line in pdbqt.splitlines() if line.startswith("ATOM") and line.split()[-1] not in ("H", "HD")]
    # Reverse file order to make coordinate transfer depend on serials.
    block = "MODEL 1\nREMARK VINA RESULT: -5.1 0.0 0.0\n" + "\n".join(records[::-1]) + "\nENDMDL\n"
    poses = read_vina_poses(molecule, block)
    assert len(poses) == 1
    assert float(poses[0].data["vina_affinity"]) == pytest.approx(-5.1)
    heavy = molecule.atom_dict["atomicnum"] != 1
    np.testing.assert_allclose(poses[0].coords, molecule.coords[heavy], atol=0.001)


def test_vina_preparation_retains_polar_hydrogens_only(benzene, monkeypatch):
    from pathlib import Path
    from rdkit import Chem
    from oddt.toolkits import rdk

    vina = pytest.importorskip("vina")
    raw = Chem.MolFromSmiles("CO")
    conformer = Chem.Conformer(2)
    conformer.SetAtomPosition(0, (0, 0, 0))
    conformer.SetAtomPosition(1, (1.4, 0, 0))
    raw.AddConformer(conformer)
    ligand = rdk.Molecule(raw)
    ligand.addh()

    class PreparedVina:
        def __init__(self, **kwargs):
            pass

        def set_receptor(self, path):
            assert Path(path).is_file()

        def set_ligand_from_file(self, path):
            self.pdbqt = Path(path).read_text()
            types = [line.split()[-1] for line in self.pdbqt.splitlines() if line.startswith("ATOM")]
            assert types.count("HD") == 1
            assert "H" not in types
            assert len(types) == 3  # Carbon, oxygen and the hydroxyl hydrogen.

        def compute_vina_maps(self, **kwargs):
            pass

        def dock(self, **kwargs):
            pass

        def poses(self, **kwargs):
            return "MODEL 1\nREMARK VINA RESULT: -1.0 0.0 0.0\n" + self.pdbqt + "\nENDMDL\n"

    monkeypatch.setattr(vina, "Vina", PreparedVina)
    poses, torsions = dock_vina(benzene, ligand, np.zeros(3), np.full(3, 16), seed=1)
    assert torsions == 0
    np.testing.assert_allclose(poses[0].coords, ligand.coords[:2], atol=0.001)


@pytest.fixture
def small_report():
    return {
        "protocol": {
            "targets": ["example"],
            "seeds": [1, 7],
            "engines": ["ga", "mc", "vina"],
            "ga": {"algorithm": "genetic"},
            "mc": {"algorithm": "monte_carlo"},
        },
        "versions": {"vina": "1.2.7"},
        "results": [
            {
                "target": "example",
                "seed": 1,
                "engine": "mc",
                "status": "ok",
                "top_symmetry_rmsd": 0.8,
                "elapsed_seconds": 10,
                "top_recovered_2A": True,
                "best_recovered_2A": True,
            },
            {"target": "example", "seed": 7, "engine": "mc", "status": "error"},
            {
                "target": "example",
                "seed": 1,
                "engine": "ga",
                "status": "ok",
                "top_symmetry_rmsd": 1.0,
                "elapsed_seconds": 5,
                "top_recovered_2A": True,
                "best_recovered_2A": True,
                "local_minimizations": 192,
                "objective_evaluations": 500,
            },
            {"target": "example", "seed": 7, "engine": "ga", "status": "error"},
            {
                "target": "example",
                "seed": 1,
                "engine": "vina",
                "status": "ok",
                "top_symmetry_rmsd": 0.7,
                "elapsed_seconds": 40,
                "top_recovered_2A": True,
                "best_recovered_2A": True,
            },
            {
                "target": "example",
                "seed": 7,
                "engine": "vina",
                "status": "ok",
                "top_symmetry_rmsd": 2.4,
                "elapsed_seconds": 60,
                "top_recovered_2A": False,
                "best_recovered_2A": True,
            },
        ],
    }


def test_report_counts_failed_runs_as_misses(small_report):
    stats = summarize(small_report)
    assert stats["all", "mc"]["attempts"] == 2
    assert stats["all", "mc"]["completed"] == 1
    assert stats["all", "mc"]["top_recovered"] == 1
    assert stats["all", "mc"]["median_seconds"] == 10
    assert stats["all", "ga"]["median_seconds"] == 5
    assert stats["all", "vina"]["top_recovered"] == 1
    assert stats["all", "vina"]["best_recovered"] == 2
    assert stats["all", "vina"]["median_rmsd"] == pytest.approx(1.55)
    assert stats["all", "vina"]["median_seconds"] == 50


@pytest.mark.parametrize("duplicate", [False, True])
def test_report_rejects_partial_or_duplicate_runs(small_report, duplicate):
    if duplicate:
        small_report["results"].append(small_report["results"][0])
    else:
        small_report["results"].pop()
    with pytest.raises(ValueError, match="Incomplete or duplicated"):
        summarize(small_report)


def test_report_does_not_relabel_genetic_results_as_monte_carlo(small_report, tmp_path):
    small_report["protocol"]["mc"] = {"algorithm": "genetic"}
    with pytest.raises(ValueError, match="cannot label results from another search algorithm"):
        render(small_report, tmp_path / "report.rst", tmp_path / "figure.svg")


@pytest.mark.parametrize("name", ["ga", "mc", "de", "aco"])
@pytest.mark.parametrize("progress", [False, True])
def test_benchmark_runs_search_with_the_recorded_budget(benzene, monkeypatch, tmp_path, name, progress):
    from asv_bench import docking_pdbbind as benchmark
    from oddt import docking

    center = benzene.coords[:6].mean(axis=0)
    size = np.full(3, 16.0)
    details = {"searched_torsions": 0, "ligand_heavy_atoms": 6}
    monkeypatch.setattr(
        benchmark, "prepare", lambda directory, target: (benzene, benzene, benzene, center, size, details)
    )

    class Engine:
        def __init__(self, protein, ligand, box):
            np.testing.assert_allclose(box, [center - size / 2, center + size / 2])
            self.search_stats_ = [{"local_minimizations": 200, "objective_evaluations": 1000}]

        def dock(self, **kwargs):
            expected = dict(benchmark.search_settings()[name])
            expected.pop("algorithm")
            assert kwargs == dict(expected, seed=7, num_modes=5, energy_range=3.0, progress=progress)
            pose = benzene.clone
            pose.data.update({"vina_affinity": -1})
            return [pose]

    monkeypatch.setattr(docking, "OddtVina" + name.upper(), Engine)
    row = benchmark.run_one(tmp_path, "example", 7, name, progress=progress)
    assert row["status"] == "ok", row.get("error")
    assert row["top_recovered_2A"]
    assert row["local_minimizations"] == 200
    assert row["objective_evaluations"] == 1000


def test_benchmark_cli_uses_larger_population_and_passes_progress(monkeypatch, tmp_path):
    import json
    import sys
    from asv_bench import docking_pdbbind as benchmark

    monkeypatch.setenv("ODDT_TOOLKIT", "rdk")
    folder = tmp_path / "example"
    folder.mkdir()
    for suffix in ("_ligand.sdf", "_pocket.pdb"):
        (folder / ("example" + suffix)).touch()
    output = tmp_path / "results.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "docking_pdbbind",
            "--data",
            str(tmp_path),
            "--engines",
            "ga",
            "mc",
            "de",
            "aco",
            "--de-population",
            "12",
            "--de-generations",
            "2",
            "--de-local-maxiter",
            "4",
            "--aco-ants",
            "4",
            "--aco-iterations",
            "2",
            "--aco-archive",
            "8",
            "--seeds",
            "7",
            "--progress",
            "--output",
            str(output),
        ],
    )
    calls = []

    def run_one(directory, target, seed, name, settings, progress, diagnostics):
        assert diagnostics is False
        calls.append((name, settings, progress))
        return dict(target=target, seed=seed, engine=name, status="ok")

    monkeypatch.setattr(benchmark, "run_one", run_one)
    benchmark.main()
    report = json.loads(output.read_text())
    assert report["protocol"]["ga"]["population_size"] == 100
    assert (
        report["protocol"]["de"]
        == benchmark.search_settings(de_population=12, de_generations=2, de_local_maxiter=4)["de"]
    )
    assert [call[0] for call in calls] == ["ga", "mc", "de", "aco"]
    expected = benchmark.search_settings(
        de_population=12, de_generations=2, de_local_maxiter=4, aco_ants=4, aco_iterations=2, aco_archive=8
    )
    assert report["protocol"]["aco"] == expected["aco"]
    assert all(settings == expected and progress for _, settings, progress in calls)
    assert not output.with_suffix(".csv").exists()


def test_report_renders_both_searches_and_vina(small_report, tmp_path):
    import json

    page = tmp_path / "report.rst"
    page.write_text("Results\n=======\n\n.. benchmark-results-start\n\n.. benchmark-results-end\n")
    figure = tmp_path / "figure.svg"
    source = tmp_path / "raw.json"
    source.write_text(json.dumps(small_report))
    render(small_report, page, source=source)
    assert not figure.exists()
    assert "per-run CSV" not in page.read_text()  # No link to a nonexistent standalone CSV.
    render(small_report, page, figure, source=source)
    text = page.read_text()
    for label in ("Internal GA", "Internal MC", "Vina 1.2.7", "Local minimizations", "Objective evaluations"):
        assert label in text
    assert figure.stat().st_size > 1000
    assert not figure.with_suffix(".png").exists()


def test_report_renders_de_only_and_checks_algorithm(small_report, tmp_path):
    protocol = small_report["protocol"]
    protocol["engines"] = ["de"]
    protocol["de"] = {"algorithm": "differential_evolution", "population_size": 32, "generations": 20}
    small_report["results"] = [dict(row, engine="de") for row in small_report["results"] if row["engine"] == "ga"]
    small_report["results"][0].update(
        {
            "best_symmetry_rmsd": 0.8,
            "diagnostics": {"sampled_best_symmetry_rmsd": 0.2, "uncapped_best_symmetry_rmsd": 0.3},
        }
    )
    page = tmp_path / "de.rst"
    page.write_text(".. benchmark-results-start\n\n.. benchmark-results-end\n")
    render(small_report, page, tmp_path / "de.svg")
    assert "Internal DE: population 32, generations 20" in page.read_text()
    assert "Closest sampled" in page.read_text()
    assert "0.20" in page.read_text()
    protocol["de"]["algorithm"] = "genetic"
    with pytest.raises(ValueError, match="cannot label"):
        render(small_report, page, tmp_path / "de.svg")


@pytest.mark.parametrize("scoring", ["vina", "vinardo", "ad4", "plp", "chemplp"])
def test_report_renders_ant_colony_budget_and_model(small_report, tmp_path, scoring):
    from asv_bench.docking_pdbbind import search_settings

    name = "aco" if scoring == "vina" else scoring + "_aco"
    small_report["protocol"]["engines"] = [name]
    small_report["protocol"][name] = dict(search_settings()["aco"], scoring=scoring)
    small_report["results"] = [dict(row, engine=name) for row in small_report["results"] if row["engine"] == "ga"]
    page = tmp_path / "aco.rst"
    page.write_text(".. benchmark-results-start\n\n.. benchmark-results-end\n")
    render(small_report, page, tmp_path / "aco.svg")
    assert "ACO: archive 32, 16 ants, iterations 20, maxiter 60" in page.read_text()
    assert "ant_colony" == small_report["protocol"][name]["algorithm"]


def test_diagnostics_include_discarded_and_capped_poses_but_exclude_outside_box(benzene):
    from asv_bench.docking_pdbbind import search_diagnostics

    coords = np.asarray(benzene.coords[:6], dtype=float)
    box = np.array([[-8.0] * 3, [8.0] * 3])
    top = benzene.clone
    top.data.update({"vina_total": 123})
    diagnostic = search_diagnostics(
        benzene,
        benzene,
        box,
        [(coords, False), (coords + [3, 0, 0], True), (coords + [20, 0, 0], True)],
        maxiter=2,
        top_pose=top,
    )
    assert diagnostic["local_search_results"] == 3
    assert diagnostic["feasible_results"] == 2
    assert diagnostic["sampled_best_symmetry_rmsd"] == pytest.approx(0, abs=1e-7)
    assert diagnostic["uncapped_best_symmetry_rmsd"] == pytest.approx(3)
    assert diagnostic["returned_top_total"] == 123
    assert diagnostic["reference"]["minimized_total"] <= diagnostic["reference"]["initial_total"]


def test_benchmark_diagnostics_observe_search_and_run_after_timer(benzene, monkeypatch, tmp_path):
    from scipy.optimize import OptimizeResult
    from asv_bench import docking_pdbbind as benchmark
    from oddt import docking

    events = []
    monkeypatch.setattr(
        benchmark, "prepare", lambda *args: (benzene, benzene, benzene, np.zeros(3), np.full(3, 16), {})
    )
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: events.append("timer") or len(events))

    class Engine:
        def __init__(self, protein, ligand, box, sample_callback):
            self.observe = sample_callback
            self.search_stats_ = [{"local_minimizations": 1, "objective_evaluations": 1}]

        def dock(self, **kwargs):
            events.append("dock")
            self.observe(OptimizeResult(coords=benzene.coords[:6].copy()), None)
            pose = benzene.clone
            pose.data.update({"vina_affinity": -1})
            return [pose]

    def diagnostics(protein, reference, box, snapshots, maxiter, top_pose, scoring="vina"):
        assert scoring == "vina"
        events.append("diagnostics")
        assert len(snapshots) == 1 and snapshots[0][1]
        return {"observed": True}

    monkeypatch.setattr(docking, "OddtVinaDE", Engine)
    monkeypatch.setattr(benchmark, "search_diagnostics", diagnostics)
    row = benchmark.run_one(tmp_path, "example", 1, "de", diagnostics=True)
    assert row["status"] == "ok", row.get("error")
    assert row["diagnostics"] == {"observed": True}
    assert events == ["timer", "dock", "timer", "diagnostics"]


def test_published_archive_retains_protocols_provenance_and_csv_rows():
    import csv
    import json
    from asv_bench.docking_benchmark_data import ARCHIVE, RESULT_COLUMNS

    archive = json.loads(ARCHIVE.read_text())
    assert archive["schema_version"] == 1
    assert {"baseline", "ga48", "ga100", "de_probe"} <= archive["experiments"].keys()
    assert {"4lb3", "mc_performance"} <= archive["validation"].keys()
    with ARCHIVE.with_suffix(".csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == sum(len(report["results"]) for report in archive["experiments"].values())
    for name, report in archive["experiments"].items():
        summarize(report)  # Each experiment remains independently complete.
        expected = [
            {key: str(row[key]) if row.get(key) is not None else "" for key in RESULT_COLUMNS}
            for row in report["results"]
        ]
        # Older published exports predate optional score/unit columns.
        actual = [{key: row.get(key, "") for key in RESULT_COLUMNS} for row in rows if row["experiment"] == name]
        assert actual == expected
    source_files = {(entry["file"], entry["sha256"]) for entry in archive["source_files"].values()}
    ga100 = archive["experiments"]["ga100"]
    for name, digest in ga100["comparison_baselines"].items():
        assert (name, digest) in source_files
    control = archive["experiments"]["ga48"]["baseline_control"]
    assert (control["baseline_file"], control["baseline_sha256"]) in source_files


def test_loader_selects_archive_experiments_and_supports_standalone(small_report, tmp_path):
    import json
    from asv_bench.docking_benchmark_data import load_report

    path = tmp_path / "archive.json"
    variant = dict(small_report, updated_at_utc="variant")
    path.write_text(json.dumps({"schema_version": 1, "experiments": {"baseline": small_report, "variant": variant}}))
    assert load_report(path) == small_report
    assert load_report(path, "variant") == variant
    with pytest.raises(ValueError, match="Unknown experiment"):
        load_report(path, "absent")
    path.write_text(json.dumps(small_report))
    assert load_report(path) == small_report
    with pytest.raises(ValueError, match="consolidated archive"):
        load_report(path, "variant")


def test_archive_cli_adds_completed_report_and_preserves_existing_names(small_report, monkeypatch, tmp_path):
    import csv
    import json
    import sys
    from asv_bench.docking_benchmark_data import main

    source = tmp_path / "run.json"
    source.write_text(json.dumps(small_report))
    archive = tmp_path / "published.json"
    monkeypatch.setattr(
        sys, "argv", ["archive", "--input", str(source), "--archive", str(archive), "--experiment", "new"]
    )
    main()
    data = json.loads(archive.read_text())
    assert data["experiments"]["new"] == small_report
    assert data["source_files"]["new"]["file"] == "run.json"
    with archive.with_suffix(".csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(small_report["results"])
    assert all(row["experiment"] == "new" for row in rows)
    before = archive.read_bytes()
    with pytest.raises(SystemExit):
        main()
    assert archive.read_bytes() == before


def test_archive_cli_rejects_partial_report_without_writing(small_report, monkeypatch, tmp_path):
    import json
    import sys
    from asv_bench.docking_benchmark_data import main

    small_report["results"].pop()
    source = tmp_path / "partial.json"
    source.write_text(json.dumps(small_report))
    archive = tmp_path / "published.json"
    monkeypatch.setattr(
        sys, "argv", ["archive", "--input", str(source), "--archive", str(archive), "--experiment", "partial"]
    )
    with pytest.raises(SystemExit):
        main()
    assert not archive.exists()
    assert not archive.with_suffix(".csv").exists()


def test_archive_cli_preserves_validation_provenance(small_report, monkeypatch, tmp_path):
    import json
    import sys
    from asv_bench.docking_benchmark_data import main

    source = tmp_path / "run.json"
    source.write_text(json.dumps(small_report))
    archive = tmp_path / "published.json"
    archive.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiments": {},
                "validation": {"4lb3": {}},
                "source_files": {"4lb3": {"file": "validation.json", "sha256": "original", "section": "validation"}},
            }
        )
    )
    monkeypatch.setattr(
        sys, "argv", ["archive", "--input", str(source), "--archive", str(archive), "--experiment", "4lb3"]
    )
    before = archive.read_bytes()
    with pytest.raises(SystemExit):
        main()
    assert archive.read_bytes() == before


@pytest.mark.parametrize("scoring", ["ad4", "vinardo", "plp", "chemplp"])
@pytest.mark.parametrize("search", ["ga", "mc", "de", "aco"])
def test_benchmark_selects_scoring_and_search_independently(benzene, monkeypatch, tmp_path, scoring, search):
    from asv_bench import docking_pdbbind as benchmark
    from oddt import docking

    monkeypatch.setattr(
        benchmark, "prepare", lambda *args: (benzene, benzene, benzene, np.zeros(3), np.full(3, 16.0), {})
    )

    class Engine:
        def __init__(self, protein, ligand, box):
            self.search_stats_ = [{"local_minimizations": 1, "objective_evaluations": 2}]

        def dock(self, **kwargs):
            config = dict(benchmark.search_settings()[search])
            config.pop("algorithm")
            assert kwargs == dict(config, seed=7, num_modes=5, energy_range=3.0, progress=False)
            pose = benzene.clone
            pose.data[scoring + ("_score" if scoring in ("plp", "chemplp") else "_affinity")] = -3.5
            return [pose]

    prefix = benchmark.SCORERS[scoring]
    monkeypatch.setattr(docking, prefix + search.upper(), Engine)
    row = benchmark.run_one(tmp_path, "example", 7, scoring + "_" + search)
    assert row["status"] == "ok", row.get("error")
    assert row["scoring"] == scoring
    assert row["algorithm"] == benchmark.ALGORITHMS[search]
    assert row["top_score"] == -3.5
    if scoring in ("plp", "chemplp"):
        assert "top_affinity" not in row
        assert row["score_units"] == "empirical score units"
    else:
        assert row["top_affinity"] == -3.5


def test_report_renders_new_scorers_without_relabeling(small_report, tmp_path):
    from copy import deepcopy

    report = deepcopy(small_report)
    models = ("ad4", "vinardo", "plp", "chemplp")
    report["protocol"]["engines"] = [model + "_ga" for model in models]
    for scoring in models:
        report["protocol"][scoring + "_ga"] = dict(
            algorithm="genetic", scoring=scoring, population_size=100, generations=8, maxiter=60
        )
    source = [row for row in small_report["results"] if row["engine"] == "ga"]
    report["results"] = [dict(row, engine=scoring + "_ga") for scoring in models for row in source]
    page = tmp_path / "scoring.rst"
    page.write_text(".. benchmark-results-start\n\n.. benchmark-results-end\n")
    render(report, page, tmp_path / "scoring.svg")
    assert "AD4 GA" in page.read_text() and "Vinardo GA" in page.read_text()
    assert "PLP GA" in page.read_text() and "ChemPLP GA" in page.read_text()
    report["protocol"]["ad4_ga"]["scoring"] = "vina"
    with pytest.raises(ValueError, match="another scoring model"):
        render(report, page, tmp_path / "scoring.svg")


@pytest.mark.parametrize("scoring", ["ad4", "vinardo", "plp", "chemplp"])
def test_reference_diagnostics_use_the_selected_scorer(benzene, scoring):
    from asv_bench.docking_pdbbind import search_diagnostics

    top = benzene.clone
    top.data[scoring + "_total"] = 123
    result = search_diagnostics(benzene, benzene, np.array([[-8.0] * 3, [8.0] * 3]), [], 2, top, scoring=scoring)
    assert result["scoring"] == scoring
    assert result["returned_top_total"] == 123
    assert result["reference"]["minimized_total"] <= result["reference"]["initial_total"]


@pytest.mark.parametrize("scoring", ["plp", "chemplp"])
def test_plp_benchmark_records_actual_parameters_and_score_units(benzene, monkeypatch, tmp_path, scoring):
    import json
    from asv_bench import docking_pdbbind as benchmark

    monkeypatch.setattr(
        benchmark, "prepare", lambda *args: (benzene, benzene, benzene, np.zeros(3), np.full(3, 16.0), {})
    )
    settings = benchmark.search_settings(ga_population=4, ga_generations=0, maxiter=3)
    row = benchmark.run_one(tmp_path, "example", 42, scoring + "_ga", settings=settings, diagnostics=True)
    assert row["status"] == "ok", row.get("error")
    assert "top_affinity" not in row
    assert row["score_units"] == "empirical score units"
    parameters = row["scoring_parameters"]
    assert parameters["coordinate_atoms"] == "all"
    assert parameters["parameterization"] == ("PLP M5" if scoring == "plp" else "ChemPLP M3")
    assert parameters["parameters"]["torsion"] == (1 if scoring == "plp" else 2)
    assert parameters["torsion_fallback"] == (0.2, 1, 3)
    assert row["diagnostics"]["scoring"] == scoring
    json.dumps(row, allow_nan=False)


def test_scoring_comparison_checks_inputs_budgets_and_preserves_reports(small_report):
    from copy import deepcopy
    from asv_bench.docking_pdbbind_report import compare_reports

    primary = deepcopy(small_report)
    primary["protocol"].update(
        engines=["ad4_ga"], backend="rdk", ad4_ga={"algorithm": "genetic", "scoring": "ad4", "population_size": 100}
    )
    primary["results"] = [dict(row, engine="ad4_ga") for row in primary["results"] if row["engine"] == "ga"]
    primary["versions"] = {"numpy": "example", "scipy": "example", "rdkit": "example"}
    primary["sha256"] = {"tests/data/example.sdf": "checksum"}
    reference = deepcopy(primary)
    reference["protocol"]["engines"] = ["ga"]
    reference["protocol"]["ga"] = {"algorithm": "genetic", "population_size": 100}
    reference["results"] = [dict(row, engine="ga") for row in primary["results"]]
    before_primary, before_reference = deepcopy(primary), deepcopy(reference)
    combined = compare_reports(primary, reference, "ga100")
    assert combined["protocol"]["engines"] == ["ga", "ad4_ga"]
    assert combined["comparison_reference"]["experiment"] == "ga100"
    assert primary == before_primary and reference == before_reference
    summarize(combined)
    reference["protocol"]["ga"]["population_size"] = 24
    with pytest.raises(ValueError, match="budgets differ"):
        compare_reports(primary, reference, "ga24")
    reference["protocol"]["ga"]["population_size"] = 100
    reference["sha256"]["tests/data/example.sdf"] = "different"
    with pytest.raises(ValueError, match="checksums differ"):
        compare_reports(primary, reference, "different")


def test_archive_overview_keeps_experiments_and_failed_run_denominators(small_report):
    from copy import deepcopy
    from asv_bench.docking_pdbbind_report import overview_data

    other = deepcopy(small_report)
    other["protocol"].update(engines=["ga"], seeds=[42], ga={"algorithm": "genetic", "population_size": 100})
    other["results"] = [dict(small_report["results"][2], seed=42, top_symmetry_rmsd=8, top_recovered_2A=False)]
    archive = {"schema_version": 1, "experiments": {"baseline": small_report, "ga100": other}}
    original = deepcopy(archive)
    targets, rows = overview_data(archive)
    assert targets == ["example"]
    assert sum(r["overall"]["attempts"] for r in rows) == 7
    baseline = next(r for r in rows if r["experiment"] == "baseline" and r["engine"] == "ga")
    assert baseline["targets"]["example"]["median_rmsd"] == 1
    assert baseline["overall"]["top_recovered"] == 1
    assert baseline["overall"]["attempts"] == 2  # Failed seed remains in the denominator.
    newer = next(r for r in rows if r["experiment"] == "ga100")
    assert newer["targets"]["example"]["median_rmsd"] == 8
    assert newer["overall"]["attempts"] == 1
    assert "GA100 (1 seed)" in newer["label"]
    assert archive == original


def test_overview_cli_exports_one_image_without_changing_data(small_report, tmp_path, monkeypatch):
    import json
    import sys
    from asv_bench.docking_pdbbind_report import main

    source = tmp_path / "docking.json"
    original = json.dumps({"schema_version": 1, "experiments": {"baseline": small_report}})
    source.write_text(original)
    monkeypatch.setattr(sys, "argv", ["report", "--overview", "--input", str(source)])
    main()
    svg = source.with_suffix(".svg").read_text()
    assert "baseline: Vina score GA (2 seeds)" in svg
    assert "EXAMPLE" in svg
    assert "1/2" in svg
    assert source.read_text() == original
    assert not source.with_suffix(".png").exists()
    assert not source.with_suffix(".csv").exists()
