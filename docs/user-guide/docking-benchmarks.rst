PDBbind docking comparison
==========================

This experiment compares ODDT's internal Monte Carlo docking engine with installed
AutoDock Vina 1.2.7 on all six PDBbind complexes distributed with the tests:
10GS, 1IMX, 1K9Q, 1NLO, 1NLP and 4YEF. The earlier :doc:`docking` example
was developed on 4LB3; these six complexes provide a separate check.

These fixtures are a small, selected set with large and flexible ligands.
They are not a representative sample of the full PDBbind database. In
particular, 1NLO and 1NLP have identical protein SEQRES entries in these files.
The supplied PDBbind index lists 1K9Q, 1NLO and 1NLP as NMR structures;
the supplied ligand coordinates serve as the reference in every case.
The measurements describe this experiment rather than general docking accuracy.

.. list-table:: Structural coverage
   :header-rows: 1

   * - PDB entry
     - Receptor
     - Ligand class
   * - `10GS <https://www.rcsb.org/structure/10GS>`_
     - Glutathione S-transferase P1-1
     - Modified glutathione inhibitor
   * - `1IMX <https://www.rcsb.org/structure/1IMX>`_
     - Insulin-like growth factor 1
     - CPQ detergent
   * - `1K9Q <https://www.rcsb.org/structure/1K9Q>`_
     - YAP65 WW domain
     - N-octyl peptide
   * - `1NLO <https://www.rcsb.org/structure/1NLO>`_
     - Src SH3 domain
     - NL1 peptide analogue
   * - `1NLP <https://www.rcsb.org/structure/1NLP>`_
     - Src SH3 domain
     - NL2 peptide analogue
   * - `4YEF <https://www.rcsb.org/structure/4YEF>`_
     - AMPK beta1 carbohydrate-binding module
     - Cyclodextrin carbohydrate

The names above follow the linked PDB entries; calculations use the bundled
PDBbind coordinates. Thus, the six complexes cover five receptor identities
and include peptide analogues, a detergent and a carbohydrate.

Protocol
--------

Both engines use the same RDKit-loaded ligand and receptor pocket. Pocket
files are used consistently because a full protein is unavailable for 10GS.
Hydrogens are added without optimizing the heavy-atom coordinates or changing
the supplied formal charges. Waters and other atoms in the supplied pockets
are retained. No target-specific scoring weights or search settings are tuned.

The reference heavy-atom centroid defines the box center. Each box dimension
is the larger of 16 Angstroms and twice the largest absolute displacement from
the center along that axis, plus 8 Angstroms. This gives the reference ligand
at least 4 Angstroms of padding on each side, including the larger ligands.
Both engines receive the same box for a given target.

The input ligand is rotated by XYZ Euler angles (1.2, -0.7, 2.1) radians and
translated by (12, -7, 8) Angstroms. Each search randomizes ligand position,
orientation and permitted torsions. Ring and conjugated-bond geometry comes
from the supplied conformation. The experiment therefore tests redocking
within those constraints rather than generating new ring conformations.

Each target is run with seeds 1, 7 and 42. The internal engine uses eight
independent Monte Carlo trajectories, 250 proposals per trajectory, and at
most 60 L-BFGS-B iterations per local minimization. Its temperature is 1.2,
mutation amplitude is 2 Angstroms, and hunt energy cap is 10. Accepted
promising minima receive uncapped local refinement.
Vina uses exhaustiveness 8 and one CPU. These parameters specify
different search budgets; they are not a matched number of energy evaluations.
Both engines return at most five poses with an energy-range setting of 3,
interpreted using each engine's own energy convention.
The saved ``searched_torsions`` field counts internal-engine heavy-atom search
genes on both kinds of row; Vina uses its own PDBQT torsion definitions.
The table reports these counts separately. Vina can include torsions that
move only polar hydrogens, while the internal engine searches heavy-atom
torsions. The two engines also use different atom typing and scoring
implementations. The comparison measures their complete docking workflows;
it does not isolate the effect of the search algorithm.

Runtime includes engine construction and docking, including Vina's PDBQT
writing and interaction-grid generation. Shared input loading, hydrogen
addition and pose evaluation are excluded. Runs execute sequentially on the
same Apple M2 machine (eight physical/logical cores). Vina uses one CPU;
the internal search also runs serially. The Vina adapter maps original PDBQT heavy-atom serials back
to the source ligand graph, avoiding hydrogen-count differences in file reading.
Vina PDBQT input is generated from implicit hydrogens with polar donor
hydrogens added explicitly. Nonpolar hydrogens are omitted before Vina torsion
perception, preventing terminal methyl groups from gaining extra torsions.
The internal engine uses the fully hydrogenated source and scores heavy atoms.
See the `Vina preparation tutorial
<https://autodock-vina.readthedocs.io/en/stable/docking_basic.html>`_ for its
recommended input workflow; this experiment uses ODDT's PDBQT writer.

Metrics
-------

The primary accuracy measure is symmetry-aware heavy-atom RMSD in the
protein coordinate frame, with no alignment. Graph automorphisms preserve
atom identity, connectivity and specified stereochemistry. Hydrogens are
removed before symmetry matching. Direct atom-order RMSD is also saved.

Recovery means RMSD below 2 Angstroms. Top-pose recovery evaluates the
highest-ranked returned pose. Best-returned recovery asks whether any of the
up to five returned poses is below the threshold; this can reveal cases where
a native-like pose was found but ranked below another pose. Failed runs count
as misses in recovery rates. Timing and RMSD medians use completed runs only.

.. benchmark-results-start

The Monte Carlo comparison has not yet completed. The interrupted genetic
search measurements are archived separately in
:download:`GA partial JSON <../_static/benchmarks/pdbbind_docking_ga_partial.json>`
and :download:`GA partial CSV <../_static/benchmarks/pdbbind_docking_ga_partial.csv>`;
they are not results for the current Monte Carlo implementation.

.. benchmark-results-end

Reproduce the comparison
------------------------

Install the RDKit backend and Vina Python package, then run from the repository
root:

.. code-block:: bash

   python -m asv_bench.docking_pdbbind

The runner writes a JSON report containing settings, package versions, source
checksums, input checksums, per-pose RMSDs and run status, together with a CSV
of per-run metrics. Each completed run is checkpointed. To resume an interrupted
run with unchanged sources and settings:

.. code-block:: bash

   python -m asv_bench.docking_pdbbind --resume

After all runs finish, regenerate the summary tables and figure with:

.. code-block:: bash

   python -m asv_bench.docking_pdbbind_report

Use ``--targets``, ``--seeds`` or ``--engines`` for a smaller experiment and
``--output`` for a separate report. Timings depend on the machine and numerical
environment. A different seed or broader search can change the resulting poses.
