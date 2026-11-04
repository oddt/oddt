# Open Drug Discovery Toolkit (ODDT)

**Cheminformatics, molecular modeling, and virtual screening in Python.**

[![Tests](https://github.com/oddt/oddt/actions/workflows/tests.yml/badge.svg)](https://github.com/oddt/oddt/actions/workflows/tests.yml)
[![Documentation](https://github.com/oddt/oddt/actions/workflows/docs.yml/badge.svg)](https://oddt.github.io/oddt/)
[![PyPI](https://img.shields.io/pypi/v/oddt.svg)](https://pypi.org/project/oddt/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](./LICENSE)

ODDT brings RDKit and Open Babel together through a common molecule interface,
with NumPy arrays for molecular data and tools for protein–ligand analysis,
fingerprints, docking, and scoring.

[Documentation](https://oddt.github.io/oddt/) ·
[API reference](https://oddt.github.io/oddt/rst/modules.html) ·
[Installation](#installation) · [Quickstart](#quickstart) ·
[Contributing](#contributing) · [Citation](#citation)

## What you can do

| Workflow | Capabilities |
| --- | --- |
| [Molecular data](./oddt/toolkits) | Read and write molecules with RDKit or Open Babel; work with NumPy atom, residue, and ring arrays. |
| [Interaction analysis](./oddt/interactions.py) | Detect hydrogen and halogen bonds, hydrophobic contacts, salt bridges, π interactions, and metal coordination. |
| [Fingerprints](./oddt/fingerprints.py) | Generate IFP, SIFP, SPLIF, ECFP, and PLEC representations. |
| [Shape comparison](./oddt/shape.py) | Compare molecules with USR, USRCAT, and Electroshape. |
| [Virtual screening](./oddt/virtualscreening.py) | Filter, search, dock with AutoDock Vina, and rescore compounds in Python or from the command line. |
| [Scoring](./oddt/scoring/functions) | Use RFScore, NNScore, and PLECscore, or train scoring models. |

## Installation

### Current development version

The current branch requires **Python 3.12+** and at least one chemistry backend:
**RDKit** or **Open Babel**. CI tests Python 3.12 and 3.14 with both backends.

With [Miniforge](https://conda-forge.org/download/) or an existing conda installation,
create the environment supplied with the repository:

```sh
git clone https://github.com/oddt/oddt.git
cd oddt
conda env create -n oddt -f environment.yml
conda activate oddt
conda install -c conda-forge pip setuptools wheel
python -m pip install --no-build-isolation --no-deps .
```

The [environment](./environment.yml) includes Python 3.12, RDKit, Open Babel,
Vina, scientific dependencies, and test tools. See [requirements.txt](./requirements.txt)
for the Python package dependencies; scikit-image is also used for molecular surfaces.

<details>
<summary>Installing a published release</summary>

Published releases are available on [PyPI](https://pypi.org/project/oddt/) and
[conda-forge](https://anaconda.org/conda-forge/oddt). Their Python and dependency
requirements can differ from this branch; use the matching version in the
[documentation](https://oddt.github.io/oddt/).

With a chemistry backend already installed:

```sh
python -m pip install oddt
```

Or install the conda-forge package in a compatible conda environment:

```sh
conda install -c conda-forge oddt
```

</details>

## Quickstart

Given a protein structure in `protein.pdb` and a ligand pose in `ligand.sdf`
in the same coordinate frame, find hydrogen bonds and calculate an interaction
fingerprint:

```python
import oddt
from oddt.fingerprints import InteractionFingerprint
from oddt.interactions import hbonds

protein = next(oddt.toolkit.readfile("pdb", "protein.pdb"))
protein.protein = True
protein.addh(only_polar=True)

ligand = next(oddt.toolkit.readfile("sdf", "ligand.sdf"))
ligand.addh(only_polar=True)

protein_atoms, ligand_atoms, strict = hbonds(protein, ligand)
fingerprint = InteractionFingerprint(ligand, protein)

print("Strict hydrogen bonds:", strict.sum())
print("Fingerprint features:", fingerprint.size)
```

The `strict` mask identifies atom pairs that meet both distance and angular
criteria. Setting `protein.protein = True` enables residue information used by
the interaction fingerprint.

The Python API selects Open Babel when both backends are installed. Choose a
backend explicitly by setting `ODDT_TOOLKIT` before starting Python:

```sh
ODDT_TOOLKIT=rdk python your_script.py
ODDT_TOOLKIT=ob python your_script.py
```

### Command line

Convert an SDF library to SMILES using RDKit:

```sh
oddt_cli --toolkit rdk ligands.sdf -o smi -O ligands.smi
```

Run `oddt_cli --help` for docking, similarity search, scoring, and output options.

## Documentation

Read the [user guide](https://oddt.github.io/oddt/) and
[API reference](https://oddt.github.io/oddt/rst/modules.html).
The site root tracks the default branch as **latest**; the sidebar's **Versions**
menu opens documentation for release tags from 0.7 onward. Each tagged version
uses its own documentation and Python source with the shared Furo theme.

### Build and preview locally

From the repository root, with the environment above active:

```sh
python -m pip install -r docs/requirements.txt
git fetch --tags
python docs/build_versions.py
python -m http.server 8000 --bind 127.0.0.1 --directory docs/_build/html
```

Open <http://127.0.0.1:8000/> for `latest`, or
<http://127.0.0.1:8000/0.7/> for version 0.7.

<details>
<summary>Publishing and theme maintenance</summary>

The [documentation workflow](./.github/workflows/docs.yml) builds documentation on
pull requests and publishes updates from the default branch and version tag pushes.
PR builds provide a downloadable `github-pages` artifact; hosted PR previews are
not configured.

To enable deployment, set **Settings → Pages → Build and deployment → Source** to
**GitHub Actions**. Push to the default branch or run **Documentation** from the
**Actions** tab on that branch. Fork URLs follow
`https://<owner>.github.io/<repository>/`.

Theme settings and historical compatibility handling live in
[docs/conf.py](./docs/conf.py), project styles in
[docs/_static/custom.css](./docs/_static/custom.css), and sidebar templates in
[docs/_templates](./docs/_templates). Historical releases before 0.8 mock pandas
during autodoc to handle the removed `pandas.Panel` API. Furo builds use the
contents sidebar instead of legacy inline contents lists.

For a single-version build, `make -C docs html` is also available.
`make -C docs latexpdf` produces a PDF when LaTeX is installed.

</details>

## Contributing

Report bugs and request features in [GitHub Issues](https://github.com/oddt/oddt/issues),
or open a [pull request](https://github.com/oddt/oddt/pulls) with a focused change
and tests for new behavior. The [mailing list](https://groups.google.com/g/oddt)
is also available for discussion.

To work on ODDT, use the repository environment and install the checkout in
editable mode, then test each chemistry backend:

```sh
python -m pip install --no-build-isolation --no-deps -e .
ODDT_TOOLKIT=rdk python -m pytest tests
ODDT_TOOLKIT=ob python -m pytest tests
```

The [test workflow](./.github/workflows/tests.yml) covers Python 3.12 and 3.14,
both chemistry backends, and the Vina 1.2 Python API and Vina 1.1.2 executable.
See the [changelog](./CHANGELOG.md) for release history.

<details>
<summary>Docking backend details</summary>

The repository environment installs Vina 1.2.7+ alongside RDKit and Open Babel
from conda-forge. ODDT uses the Vina Python API for docking and affinity, and
reconstructs unweighted interaction terms with its internal scorer.

Vina 1.1.2 remains supported through its command-line executable and native
scoring path. Pass its binary path as `executable` to select it explicitly when
the Python API is also installed. Native tests are skipped only when neither
the Python package nor an executable is available.

</details>

## Citation

If you use ODDT in your research, please cite:

Wójcikowski, M., Zielenkiewicz, P., & Siedlecki, P. (2015).
**Open Drug Discovery Toolkit (ODDT): a new open-source player in the drug discovery field.**
*Journal of Cheminformatics*, 7, 26.
[doi:10.1186/s13321-015-0078-2](https://doi.org/10.1186/s13321-015-0078-2)

## License

ODDT is distributed under the [BSD 3-Clause license](./LICENSE).
