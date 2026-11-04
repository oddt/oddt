Command-line workflows
======================

The installed ``oddt_cli`` command exposes the
:class:`~oddt.virtualscreening.virtualscreening` pipeline. It accepts molecular
input files and combines conversion, filtering, similarity searching, docking,
and scoring. From the root of an uninstalled source checkout, use
``PYTHONPATH=. python bin/oddt_cli`` in place of ``oddt_cli``.

.. code-block:: bash

   oddt_cli --help

Convert molecular files
-----------------------

This example can be run from the repository root:

.. code-block:: bash

   oddt_cli tests/data/pdb/4lb3_ligand.sdf --toolkit rdk -o smi -O ligand.smi

Use ``--toolkit rdk`` or ``--toolkit ob`` explicitly. Input formats are
normally inferred from the filename; ``-i`` overrides that inference.
``-o`` selects an output format and ``-O`` selects an output file.
Supplying ``-o`` without ``-O`` writes molecular output to standard output.

Filter a library
----------------

The remaining commands are workflow templates: supply your prepared input
files and install the docking/scoring dependencies needed by the operation.

.. code-block:: bash

   oddt_cli ligands.sdf --toolkit ob --filter pains -O filtered.sdf

Repeated ``--filter`` arguments add pipeline filters. Inspect the installed
command's help for available presets and options.

Search for similar binding modes
--------------------------------

.. code-block:: bash

   oddt_cli poses.sdf --toolkit rdk --receptor protein.pdb \
       --query reference.sdf --similarity ifp --cutoff 0.7 -O similar.sdf

IFP/SIFP searches require protein and ligand poses in the same coordinate frame.
USR, USRCAT, and Electroshape searches compare molecular shape instead.
See :doc:`fingerprints` and :doc:`shape` for their interpretation.

Dock and rescore
----------------

Define a docking box from a bound reference ligand:

.. code-block:: bash

   oddt_cli ligands.sdf --toolkit ob --dock autodock_vina \
       --receptor protein.pdb --auto_ligand reference.sdf \
       --exhaustiveness 20 --seed 1 -O docked.sdf

Or provide the box center and dimensions explicitly:

.. code-block:: bash

   oddt_cli ligands.sdf --toolkit ob --dock autodock_vina \
       --receptor protein.pdb --center '(1, 2, 0.5)' --size '(8, 8, 8)' \
       --exhaustiveness 20 -O docked.sdf

The docking backend must be available and the receptor/ligands prepared for
it. Scoring models also need their trained model resources:

.. code-block:: bash

   oddt_cli docked.sdf --toolkit rdk --receptor protein.pdb \
       --score rfscore_v1 --score rfscore_v2 --score rfscore_v3 -O scored.sdf

Use ``--score_file`` for a model saved through ODDT's scoring interface:

.. code-block:: bash

   oddt_cli docked.sdf --toolkit rdk --receptor protein.pdb \
       --score_file model.pickle -O rescored.sdf

For custom Python pipelines, see :mod:`oddt.virtualscreening`.
