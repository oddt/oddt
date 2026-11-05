.. _installation-guide:

Installation
============

ODDT needs Python and at least one chemistry backend: RDKit or Open Babel.
The current development branch requires Python 3.12 or newer. Published releases
can have different requirements; select the matching documentation version.

Install the development environment
-----------------------------------

The repository supplies a conda environment with both backends, Vina,
scientific dependencies, and test tools. From a terminal:

.. code-block:: bash

   git clone https://github.com/oddt/oddt.git
   cd oddt
   conda env create -n oddt -f environment.yml
   conda activate oddt
   conda install -c conda-forge pip setuptools wheel
   python -m pip install --no-build-isolation --no-deps .

Use ``-e .`` instead of ``.`` in the last command for an editable installation.
The numerical and machine-learning dependencies include NumPy, SciPy,
scikit-learn, joblib, pandas, and packaging. Molecular surface calculations
also use scikit-image.

The package declares NumPy >=1.13, SciPy >=0.19, scikit-learn >=0.18,
joblib >=0.10, pandas >=0.19.2, and packaging >=20. The supplied conda
environment selects builds compatible with its Python version and includes
RDKit >=2026.03.1 and Open Babel >=3.2.1. Install at least one backend;
the other scientific dependencies do not provide a chemistry backend.

With the dependencies already installed, you can also install the development
branch directly:

.. code-block:: bash

   python -m pip install --no-build-isolation --no-deps \
       git+https://github.com/oddt/oddt.git@master

A downloaded release source archive can be unpacked and installed with the
same ``python -m pip install --no-build-isolation --no-deps .`` command from
its source directory. This replaces the older ``python setup.py install``
workflow.

Install a published release
---------------------------

With a chemistry backend already installed:

.. code-block:: bash

   python -m pip install oddt

Or use conda-forge:

.. code-block:: bash

   conda install -c conda-forge oddt

Check the selected release's Python and backend requirements before creating
its environment. A package manager can install an older compatible release.

Choose a chemistry backend
--------------------------

ODDT's Python API chooses Open Babel when both backends are importable.
To select a backend explicitly, set ``ODDT_TOOLKIT`` before importing ODDT:

.. code-block:: bash

   ODDT_TOOLKIT=rdk python your_script.py
   ODDT_TOOLKIT=ob python your_script.py

In an interactive session, set the environment variable before the first
``import oddt``, or restart the kernel after changing it. Atom typing,
protonation perception, and residue handling can differ between backends.
Keep the backend fixed when comparing results.

Verify the installation
-----------------------

.. code-block:: python

   import oddt

   molecule = oddt.toolkit.readstring('smi', 'c1ccccc1 benzene')
   print('Backend:', oddt.toolkit.backend)
   print('Title:', molecule.title)
   print('Atoms:', len(molecule.atoms))

The molecule has six atoms before explicit hydrogens are added. A SMILES
string provides connectivity; it does not provide a protein-bound 3D pose.

If no backend is available, install RDKit or Open Babel in the same environment
as the Python interpreter running your script. An ``import oddt`` alone does
not establish that a usable backend was found.

Next steps
----------

Start with :doc:`user-guide/molecules` to load structures and inspect molecular
arrays, or go directly to :doc:`user-guide/interactions` for a protein–ligand
walkthrough.
