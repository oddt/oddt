"""Build the current docs and every release tag from 0.7 into one Pages site."""

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

from packaging.version import InvalidVersion, Version


def release_tags(tags):
    """Select version tags numerically, retaining their original directory names."""
    releases = []
    for tag in tags:
        try:
            version = Version(tag)
        except InvalidVersion:
            continue
        if version >= Version("0.7"):
            releases.append((version, tag))
    return [tag for _, tag in sorted(releases, reverse=True)]


def version_links(current, tags):
    """Return Sphinx document paths relative to this version's documentation root."""
    prefix = "" if current == "latest" else "../"
    return [
        {"name": "latest", "path": prefix + "index"},
        *({"name": tag, "path": prefix + tag + "/index"} for tag in tags),
    ]


def build_docs(repo, output):
    tags = release_tags(subprocess.check_output(["git", "tag", "--list"], cwd=repo, text=True).splitlines())
    print("Building documentation: " + ", ".join(["latest", *tags]), flush=True)
    output.mkdir(parents=True, exist_ok=True)

    # Each build gets its own source tree and Python process, so autodoc cannot
    # reuse the current package when documenting a historical release.
    with tempfile.TemporaryDirectory(prefix="oddt-docs-") as temporary:
        for name in ["latest", *tags]:
            source = Path(temporary) / name
            source.mkdir()
            if name == "latest":
                for directory in ("docs", "oddt"):
                    shutil.copytree(
                        repo / directory,
                        source / directory,
                        ignore=shutil.ignore_patterns("_build", "__pycache__"),
                    )
            else:
                archive = subprocess.check_output(["git", "archive", "refs/tags/" + name, "docs", "oddt"], cwd=repo)
                with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
                    tree.extractall(source, filter="data")

            environment = dict(
                os.environ,
                ODDT_DOCS_SOURCE=str(source),
                ODDT_DOCS_VERSION=name,
                ODDT_DOCS_VERSIONS=json.dumps(version_links(name, tags)),
            )
            environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
            target = output if name == "latest" else output / name
            print(f"Building {name} from {source}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sphinx.ext.apidoc",
                    "--force",
                    "--output-dir",
                    str(source / "docs/rst"),
                    str(source / "oddt"),
                ],
                cwd=source,
                env=environment,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sphinx",
                    "--builder",
                    "html",
                    "--conf-dir",
                    str(repo / "docs"),
                    str(source / "docs"),
                    str(target),
                ],
                cwd=source,
                env=environment,
                check=True,
            )

    (output / ".nojekyll").touch()


if __name__ == "__main__":
    repository = Path(__file__).resolve().parent.parent
    build_docs(repository, repository / "docs/_build/html")
