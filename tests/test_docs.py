"""Checks for the version selection and links used by the Pages build."""

import importlib.util
from pathlib import Path
import posixpath
import runpy
import shutil
import textwrap
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("build_versions", Path(__file__).parents[1] / "docs/build_versions.py")
build_versions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_versions)


def test_release_tags():
    tags = ["0.6", "0.7", "0.7.1", "v0.8", "0.10", "1.0rc1", "1.0", "notes", "latest", "release/2.0"]
    assert build_versions.release_tags(tags) == ["1.0", "1.0rc1", "0.10", "v0.8", "0.7.1", "0.7"]


def test_no_matching_release_tags():
    assert build_versions.release_tags(["0.6", "0.7rc1", "not-a-version"]) == []


@pytest.mark.parametrize("current", ["latest", "0.7", "v0.8"])
def test_version_links_reach_site_root_and_release_directories(current):
    directory = "" if current == "latest" else current
    links = build_versions.version_links(current, ["v0.8", "0.7"])
    destinations = {item["name"]: posixpath.normpath(posixpath.join(directory, item["path"])) for item in links}
    assert destinations == {"latest": "index", "v0.8": "v0.8/index", "0.7": "0.7/index"}


@pytest.mark.parametrize(
    "output_format, theme, remove_contents",
    [("html", "furo", True), ("latex", "furo", False), ("html", "classic", False)],
)
def test_legacy_inline_contents(output_format, theme, remove_contents, monkeypatch):
    nodes = pytest.importorskip("docutils.nodes")
    core = pytest.importorskip("docutils.core")
    monkeypatch.setenv("ODDT_DOCS_VERSION", "latest")
    config = runpy.run_path(str(Path(__file__).parents[1] / "docs/conf.py"))
    document = core.publish_doctree(
        "Guide\n=====\n\n.. contents::\n    :depth: 2\n\n"
        "Installation\n------------\n\nInstall the toolkit.\n\n"
        ".. topic:: Overview\n\n    Keep this overview.\n"
    )
    app = SimpleNamespace(builder=SimpleNamespace(format=output_format), config=SimpleNamespace(html_theme=theme))
    config["remove_inline_contents"](app, document, "index")
    contents = [node for node in document.findall(nodes.topic) if "contents" in node["classes"]]
    assert bool(contents) == (not remove_contents)
    assert "Install the toolkit." in document.astext()
    assert "Keep this overview." in document.astext()
    remaining_ids = {
        target
        for element in document.findall(nodes.Element)
        for target in element.get("ids", [])
    }
    for title in document.findall(nodes.title):
        assert not title.get("refid") or title["refid"] in remaining_ids


@pytest.mark.parametrize(
    "document",
    [
        "installation.rst",
        "user-guide/molecules.rst",
        "user-guide/interactions.rst",
        "user-guide/fingerprints.rst",
        "user-guide/shape.rst",
    ],
)
def test_user_guide_python_examples(document, tmp_path, monkeypatch):
    """Execute each page's Python examples in reading order with real fixtures."""
    repository = Path(__file__).resolve().parents[1]
    data = tmp_path / "tests/data/pdb"
    data.mkdir(parents=True)
    for name in ("4lb3_pocket.pdb", "4lb3_ligand.sdf"):
        shutil.copyfile(repository / "tests/data/pdb" / name, data / name)
    monkeypatch.chdir(tmp_path)
    path = repository / "docs" / document
    lines = path.read_text().splitlines(keepends=True)
    namespace = {"__name__": "__main__"}
    examples = 0
    index = 0
    while index < len(lines):
        if lines[index].rstrip() != ".. code-block:: python":
            index += 1
            continue
        start = index + 1
        index = start
        while index < len(lines) and (
            not lines[index].strip() or lines[index].startswith("   ")
        ):
            index += 1
        code = textwrap.dedent("".join(lines[start:index]))
        # Preserve source line numbers in failures so the displayed guide is actionable.
        exec(compile("\n" * start + code, str(path), "exec"), namespace)
        examples += 1
    assert examples, f"No Python examples found in {document}"
