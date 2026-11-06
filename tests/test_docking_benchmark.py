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
        "protocol": {"targets": ["example"], "seeds": [1, 7], "engines": ["internal", "vina"]},
        "results": [
            {
                "target": "example",
                "seed": 1,
                "engine": "internal",
                "status": "ok",
                "top_symmetry_rmsd": 0.8,
                "elapsed_seconds": 10,
                "top_recovered_2A": True,
                "best_recovered_2A": True,
            },
            {"target": "example", "seed": 7, "engine": "internal", "status": "error"},
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
    assert stats["all", "internal"]["attempts"] == 2
    assert stats["all", "internal"]["completed"] == 1
    assert stats["all", "internal"]["top_recovered"] == 1
    assert stats["all", "internal"]["median_seconds"] == 10
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
    small_report["protocol"]["internal"] = {"algorithm": "genetic"}
    with pytest.raises(ValueError, match="cannot label results from another search algorithm"):
        render(small_report, tmp_path / "report.rst", tmp_path / "figure.svg")


def test_benchmark_runs_monte_carlo_with_the_recorded_budget(benzene, monkeypatch, tmp_path):
    from asv_bench import docking_pdbbind as benchmark
    from oddt import docking

    center = benzene.coords[:6].mean(axis=0)
    size = np.full(3, 16.0)
    details = {"searched_torsions": 0, "ligand_heavy_atoms": 6}
    monkeypatch.setattr(
        benchmark, "prepare", lambda directory, target: (benzene, benzene, benzene, center, size, details)
    )

    class MonteCarloEngine:
        def __init__(self, protein, ligand, box):
            np.testing.assert_allclose(box, [center - size / 2, center + size / 2])
            self.search_stats_ = [{"proposals": 250, "accepted": 50}]

        def dock(self, **kwargs):
            assert kwargs == {"exhaustiveness": 8, "steps": 250, "maxiter": 60, "seed": 7, "num_modes": 5}
            pose = benzene.clone
            pose.data.update({"vina_affinity": -1})
            return [pose]

    monkeypatch.setattr(docking, "vina_docking", MonteCarloEngine)
    row = benchmark.run_one(tmp_path, "example", 7, "internal")
    assert row["status"] == "ok", row.get("error")
    assert row["top_recovered_2A"]
    assert row["search_stats"] == [{"proposals": 250, "accepted": 50}]
