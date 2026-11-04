.. _usage-guide:

User guide
==========

These guides introduce the common molecule interface, protein–ligand analysis,
and ways to compare molecules and binding modes. Each page links to the
corresponding API reference for the complete parameter list.

.. toctree::
   :maxdepth: 1

   molecules
   interactions
   fingerprints
   shape
   cli

Choose a starting point
-----------------------

* :doc:`molecules`: read and write structures, choose atom indices, and use
  NumPy molecular arrays.
* :doc:`interactions`: prepare a complex, detect contacts, apply geometry
  filters, and export a residue-level report.
* :doc:`fingerprints`: choose IFP, SIFP, ECFP, PLEC, or SPLIF; compare binding
  modes and build feature matrices.
* :doc:`shape`: compare 3D conformers with USR, USRCAT, and Electroshape.
* :doc:`cli`: convert files and assemble filtering, docking, and scoring
  workflows from the command line.

Run the examples
----------------

Install ODDT as described in :doc:`../installation`. The molecule,
interaction, fingerprint, and shape examples use the small 4LB3 example
distributed in the source repository. Run them from the repository root:

.. code-block:: bash

   git clone https://github.com/oddt/oddt.git
   cd oddt

The input files are ``tests/data/pdb/4lb3_pocket.pdb`` and
``tests/data/pdb/4lb3_ligand.sdf``. They already share a coordinate frame.
They contain an aldose reductase pocket and its inhibitor from
`PDB 4LB3 <https://www.rcsb.org/structure/4LB3>`_.
Follow Python blocks on each page in order in one session. Replace those
paths with your prepared protein and ligand files for your own work.
