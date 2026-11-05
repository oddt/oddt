Development and contributing
============================

Report bugs and request features in
`GitHub Issues <https://github.com/oddt/oddt/issues>`_, or open a focused
pull request with a reproducer and checks for the changed behavior.

Set up a checkout
-----------------

Follow :doc:`installation` and install the package in editable mode:

.. code-block:: bash

   python -m pip install --no-build-isolation --no-deps -e .

Run tests with the backend affected by your change:

.. code-block:: bash

   ODDT_TOOLKIT=rdk python -m pytest tests/test_interactions.py tests/test_fingerprints.py
   ODDT_TOOLKIT=ob python -m pytest tests/test_interactions.py tests/test_fingerprints.py

Use ``python -m pytest tests`` for the full suite. Tests that depend on an
external docking backend need that backend installed.

Work with molecular indices
---------------------------

Use ``atom.idx0`` for zero-based indexing and ``atom.idx1`` for one-based
indexing, rather than the deprecated ``atom.idx``. ODDT's molecular array
IDs are zero-based. Backend-native APIs can use different conventions.
See :doc:`user-guide/molecules` for array fields and cache behavior.

Build the documentation
-----------------------

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   python docs/build_versions.py

The output is written to ``docs/_build/html``. The site contains the current
documentation at its root and release tags from 0.7 in version subdirectories.
Each historical build reads the source and API from its tag and uses the
current shared theme, templates, and static assets.

Serve it locally:

.. code-block:: bash

   python -m http.server 8765 --directory docs/_build/html

Open ``http://localhost:8765/``. To build only the current source manually,
``make -C docs html`` is also available. Keep user-guide examples executable,
cross-reference public API symbols, and check navigation in both latest and
historical documentation before changing shared templates.

Update the 3D interaction examples
----------------------------------

The interaction atlas uses committed pocket/ligand files and precomputed
ODDT contacts. Sphinx builds do not download or reanalyze PDB entries. To
regenerate the examples with the RDKit backend:

.. code-block:: bash

   python docs/generate_interaction_examples.py
   python -m pytest tests/test_docs.py
   python docs/build_versions.py

The generator caches official PDB and Chemical Component Dictionary inputs
under ``docs/_build/interaction-sources`` and records their SHA-256 checksums.
It preserves deposited heavy-atom coordinates, assigns ligand chemistry from
the CCD, and uses ODDT's detectors to select the displayed contacts. Review
the generated structures, detector settings, and software versions when
updating an example. Every interaction type must have a different PDB ID.

The browser uses the pinned, BSD-licensed 3Dmol.js bundle under
``docs/_static/vendor``. Keep its license and dependency notices when updating
the renderer. Check switching between all eight examples, contact selection,
labels, surface toggling, and mobile layout. The atlas belongs to latest;
historical releases retain their tagged guide content.
