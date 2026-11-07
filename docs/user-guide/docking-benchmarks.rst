PDBbind docking comparison
==========================

This runner compares ODDT's internal genetic and Monte Carlo engines, with
optional installed AutoDock Vina 1.2.7, on all six PDBbind complexes distributed with the tests:
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

Default protocol
----------------

The settings in this section are the runner defaults. Generated results state
the actual seeds, engines and search budgets used, which may be smaller for
a validation probe.

All engines use the same RDKit-loaded ligand and receptor pocket. Pocket
files are used consistently because a full protein is unavailable for 10GS.
Hydrogens are added without optimizing the heavy-atom coordinates or changing
the supplied formal charges. Waters and other atoms in the supplied pockets
are retained. No target-specific scoring weights or search settings are tuned.

The reference heavy-atom centroid defines the box center. Each box dimension
is the larger of 16 Angstroms and twice the largest absolute displacement from
the center along that axis, plus 8 Angstroms. This gives the reference ligand
at least 4 Angstroms of padding on each side, including the larger ligands.
All engines receive the same box for a given target.

The input ligand is rotated by XYZ Euler angles (1.2, -0.7, 2.1) radians and
translated by (12, -7, 8) Angstroms. Each search randomizes ligand position,
orientation and permitted torsions. Ring and conjugated-bond geometry comes
from the supplied conformation. The experiment therefore tests redocking
within those constraints rather than generating new ring conformations.

By default each target is run with seeds 1, 7 and 42. GA now uses population 100, eight
generations, mutation rate 0.2 and at most 60 L-BFGS-B iterations per local
minimization. With twelve elites retained per generation this is 804 local
minimizations. MC uses eight
independent Monte Carlo trajectories, 250 proposals per trajectory, and at
most 60 L-BFGS-B iterations per local minimization. Its temperature is 1.2,
mutation amplitude is 2 Angstroms, and hunt energy cap is 10. Accepted
promising minima receive uncapped local refinement.
Vina uses exhaustiveness 8 and one CPU. These parameters specify
different search budgets; they are not a matched number of energy evaluations.
All engines return at most five poses with an energy-range setting of 3,
interpreted using each engine's own energy convention.
The saved ``searched_torsions`` field counts internal-engine heavy-atom search
genes on all kinds of row; Vina uses its own PDBQT torsion definitions.
The table reports these counts separately. Vina can include torsions that
move only polar hydrogens, while the internal engine searches heavy-atom
torsions. The two engines also use different atom typing and scoring
implementations. GA and MC share the scoring base, local minimizer and output
selection, so their comparison changes only the global search and its budget.
Each internal run records actual local minimizations and objective evaluations;
compare these counters alongside time. The comparison with external Vina
measures complete docking workflows, including its separate scoring implementation.

Use ``--ga-population 24`` or ``--ga-population 48`` to select an earlier budget.
The completed baseline below records population 24 explicitly. To evaluate population 100
without overwriting it, run:

.. code-block:: console

   python -m asv_bench.docking_pdbbind --engines ga --ga-population 100 --output docs/_static/benchmarks/pdbbind_ga100.json

Both internal engines support optional tqdm progress bars. Add ``--progress``
to show them; the default quiet mode is used for timing comparisons. Install
``oddt[progress]`` or ``tqdm`` to enable this option. The display flag does not
change the recorded search budget or pose results.

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
The evaluation graph converts SDF query records to literal atom/bond
attributes while retaining atom indices, charges, isotopes and stereochemistry.
This handles the query-typed 1NLP fixture, whose literal graph has 16
automorphisms.

Recovery means RMSD below 2 Angstroms. Top-pose recovery evaluates the
highest-ranked returned pose. Best-returned recovery asks whether any of the
up to five returned poses is below the threshold; this can reveal cases where
a native-like pose was found but ranked below another pose. Failed runs count
as misses in recovery rates. Timing and RMSD medians use completed runs only.

The earlier interrupted search measurements are archived separately in
:download:`GA partial JSON <../_static/benchmarks/pdbbind_docking_ga_partial.json>`
and :download:`GA partial CSV <../_static/benchmarks/pdbbind_docking_ga_partial.csv>`,
and :download:`MC partial JSON <../_static/benchmarks/pdbbind_docking_mc_partial.json>`
and :download:`MC partial CSV <../_static/benchmarks/pdbbind_docking_mc_partial.csv>`.
The MC archive contains 21 completed runs. These are historical partial
experiments, separate from the full comparison below. The earlier single-seed
GA/MC probe, using 20 MC proposals per trajectory, is retained as
:download:`probe JSON <../_static/benchmarks/pdbbind_ga_mc_probe.json>` and
:download:`probe CSV <../_static/benchmarks/pdbbind_ga_mc_probe.csv>`.

.. benchmark-results-start

Results
-------

This report uses seeds 1, 7, 42 and engines Internal GA, Internal MC, Vina 1.2.7. GA: population 24, generations 8, maxiter 60. MC: 8 trajectories, 250 proposals each, maxiter 60. These budgets are not matched energy-evaluation counts.


RMSDs below are symmetry-aware, in Angstroms. Times are seconds. Per-target values are medians over 3 seeds.

.. list-table:: Across all target/seed runs
   :header-rows: 1

   * - Engine
     - Top <2 A
     - Any returned <2 A
     - Median top RMSD
     - Median time
     - Errors
   * - Internal GA
     - 3/18
     - 3/18
     - 8.19
     - 28.6
     - 0
   * - Internal MC
     - 2/18
     - 2/18
     - 8.52
     - 363.1
     - 0
   * - Vina 1.2.7
     - 3/18
     - 4/18
     - 7.74
     - 555.7
     - 0

.. list-table:: Pose recovery by complex
   :header-rows: 1

   * - Complex
     - Heavy atoms
     - Internal torsions
     - Vina torsions
     - Internal GA RMSD
     - Internal MC RMSD
     - Vina 1.2.7 RMSD
   * - 10GS
     - 33
     - 13
     - 14
     - 4.43
     - 5.36
     - 5.26
   * - 1IMX
     - 60
     - 22
     - 34
     - 8.86
     - 13.54
     - 16.24
   * - 1K9Q
     - 46
     - 16
     - 17
     - 13.03
     - 8.14
     - 13.28
   * - 1NLO
     - 84
     - 24
     - 25
     - 7.53
     - 15.25
     - 7.17
   * - 1NLP
     - 76
     - 16
     - 16
     - 15.04
     - 13.92
     - 10.06
   * - 4YEF
     - 77
     - 7
     - 28
     - 1.06
     - 8.78
     - 1.13

.. list-table:: Recovered runs by complex (RMSD <2 A)
   :header-rows: 1

   * - Complex
     - Internal GA top
     - Internal MC top
     - Vina 1.2.7 top
     - Internal GA any returned
     - Internal MC any returned
     - Vina 1.2.7 any returned
   * - 10GS
     - 0/3
     - 1/3
     - 0/3
     - 0/3
     - 1/3
     - 1/3
   * - 1IMX
     - 0/3
     - 0/3
     - 0/3
     - 0/3
     - 0/3
     - 0/3
   * - 1K9Q
     - 0/3
     - 0/3
     - 0/3
     - 0/3
     - 0/3
     - 0/3
   * - 1NLO
     - 0/3
     - 0/3
     - 0/3
     - 0/3
     - 0/3
     - 0/3
   * - 1NLP
     - 0/3
     - 0/3
     - 0/3
     - 0/3
     - 0/3
     - 0/3
   * - 4YEF
     - 3/3
     - 1/3
     - 3/3
     - 3/3
     - 1/3
     - 3/3

.. list-table:: Median runtime by complex
   :header-rows: 1

   * - Complex
     - Internal GA (s)
     - Internal MC (s)
     - Vina 1.2.7 (s)
   * - 10GS
     - 21.6
     - 271.4
     - 97.0
   * - 1IMX
     - 27.5
     - 375.5
     - 726.9
   * - 1K9Q
     - 23.1
     - 308.2
     - 211.5
   * - 1NLO
     - 42.7
     - 526.5
     - 1304.8
   * - 1NLP
     - 34.7
     - 468.1
     - 766.4
   * - 4YEF
     - 29.7
     - 356.9
     - 402.5

.. list-table:: Median search work (Vina does not expose these counters)
   :header-rows: 1

   * - Complex
     - Engine
     - Local minimizations
     - Objective evaluations
   * - 10GS
     - Internal GA
     - 192
     - 30318
   * - 10GS
     - Internal MC
     - 2198
     - 341460
   * - 10GS
     - Vina 1.2.7
     - --
     - --
   * - 1IMX
     - Internal GA
     - 192
     - 32146
   * - 1IMX
     - Internal MC
     - 2198
     - 353881
   * - 1IMX
     - Vina 1.2.7
     - --
     - --
   * - 1K9Q
     - Internal GA
     - 192
     - 36093
   * - 1K9Q
     - Internal MC
     - 2226
     - 353854
   * - 1K9Q
     - Vina 1.2.7
     - --
     - --
   * - 1NLO
     - Internal GA
     - 192
     - 35353
   * - 1NLO
     - Internal MC
     - 2326
     - 364153
   * - 1NLO
     - Vina 1.2.7
     - --
     - --
   * - 1NLP
     - Internal GA
     - 192
     - 36330
   * - 1NLP
     - Internal MC
     - 2329
     - 355292
   * - 1NLP
     - Vina 1.2.7
     - --
     - --
   * - 4YEF
     - Internal GA
     - 192
     - 34284
   * - 4YEF
     - Internal MC
     - 3085
     - 379234
   * - 4YEF
     - Vina 1.2.7
     - --
     - --

.. figure:: ../_static/benchmarks/pdbbind_docking.svg
   :alt: Per-run accuracy and runtime across search engines.

   Dots show individual seeds; larger markers show medians. The dashed line marks 2 Angstrom RMSD. Runtime uses a logarithmic scale.

Download the :download:`per-run CSV <../_static/benchmarks/pdbbind_docking.csv>` or :download:`full JSON report <../_static/benchmarks/pdbbind_docking.json>`. The JSON records the protocol, source/input checksums, per-pose RMSDs and search statistics.

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

The default engine list is ``ga mc vina``. Use ``--engines ga mc`` to compare
only the two searches on shared scoring. Search budgets can be varied with
``--ga-population``, ``--ga-generations``, ``--mc-exhaustiveness``,
``--mc-steps`` and ``--maxiter``; the exact values are saved in the protocol.
For example, a shorter six-complex comparison with one seed is:

.. code-block:: bash

   python -m asv_bench.docking_pdbbind --engines ga mc --seeds 42 --mc-steps 20 --output docs/_static/benchmarks/pdbbind_ga_mc_probe.json
   python -m asv_bench.docking_pdbbind_report --input docs/_static/benchmarks/pdbbind_ga_mc_probe.json --figure docs/_static/benchmarks/pdbbind_ga_mc_probe.svg

This reduces MC's search budget and is not equivalent to the default experiment.
Use ``--targets``, ``--seeds`` or ``--engines`` for a smaller experiment and
``--output`` for a separate report. Timings depend on the machine and numerical
environment. A different seed or broader search can change the resulting poses.

.. ga-population-results-start

GA population comparison
------------------------

Separate 18-run experiments compare GA populations 24, 48 and 100 on the same six complexes and seeds 1, 7 and 42. All use eight generations, maxiter 60 and up to five returned poses. Runs are sequential on the same machine, with progress disabled. The population-24 measurements come from the completed baseline above.

Population 100 performs 804 local minimizations per run, compared with 384 at population 48 and 192 at population 24. All 18 population-100 runs completed without errors. Median runtime is 106.5 seconds, 2.11 times the population-48 median.

Population 100 improves per-complex median RMSD on 3 of six complexes relative to population 48 and 4 of six relative to population 24. Overall median top-pose RMSD improves from 8.26 to 7.48 Angstroms relative to population 48. Top-pose and best-returned recovery below 2 Angstroms remain 3/18 for all three populations. More search effort improves some poses without recovering additional cases here.

.. list-table:: Across the same 18 target/seed runs
   :header-rows: 1

   * - GA population
     - Local minimizations/run
     - Top <2 A
     - Any returned <2 A
     - Median top RMSD
     - Median time (s)
   * - 24
     - 192
     - 3/18
     - 3/18
     - 8.19
     - 28.6
   * - 48
     - 384
     - 3/18
     - 3/18
     - 8.26
     - 50.4
   * - 100
     - 804
     - 3/18
     - 3/18
     - 7.48
     - 106.5

Per-target values are medians over three seeds. RMSDs are symmetry-aware, in Angstroms; times are seconds.

.. list-table:: Top-pose RMSD by complex and population
   :header-rows: 1

   * - Complex
     - Population 24
     - Population 48
     - Population 100
   * - 10GS
     - 4.43
     - 6.22
     - 7.60
   * - 1IMX
     - 8.86
     - 9.75
     - 6.33
   * - 1K9Q
     - 13.03
     - 9.29
     - 12.71
   * - 1NLO
     - 7.53
     - 7.92
     - 7.30
   * - 1NLP
     - 15.04
     - 12.99
     - 7.51
   * - 4YEF
     - 1.06
     - 0.95
     - 1.14

.. list-table:: Runtime by complex and population (seconds)
   :header-rows: 1

   * - Complex
     - Population 24
     - Population 48
     - Population 100
   * - 10GS
     - 21.6
     - 38.7
     - 80.9
   * - 1IMX
     - 27.5
     - 53.6
     - 114.0
   * - 1K9Q
     - 23.1
     - 45.1
     - 93.3
   * - 1NLO
     - 42.7
     - 77.6
     - 157.0
   * - 1NLP
     - 34.7
     - 54.2
     - 116.4
   * - 4YEF
     - 29.7
     - 44.7
     - 93.3

The population-100 experiment has unchanged inputs, numerical package versions and search settings apart from population. Source checks confirm that the GA population defaults are the only engine and runner changes from the population-48 experiment. The :download:`population-100 JSON <../_static/benchmarks/pdbbind_ga100.json>` records source/input checksums, baseline checksums, per-pose RMSDs and search statistics; :download:`CSV <../_static/benchmarks/pdbbind_ga100.csv>` contains the per-run measurements.

An earlier population-24 control on 10GS, seed 42, reproduced every returned RMSD, top affinity and search counter exactly after adding progress support. It is retained in the :download:`population-48 JSON <../_static/benchmarks/pdbbind_ga48.json>`, with its :download:`CSV <../_static/benchmarks/pdbbind_ga48.csv>` measurements.

.. ga-population-results-end
