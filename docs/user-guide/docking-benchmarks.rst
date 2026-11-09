PDBbind docking comparison
==========================

This runner supports Vina, Vinardo, AD4, PLP and ChemPLP scoring with ODDT's internal
genetic, Monte Carlo, adaptive differential evolution, ant-colony and Bayesian searches, with
optional installed AutoDock Vina 1.2.7, on all six PDBbind complexes distributed with the tests:
10GS, 1IMX, 1K9Q, 1NLO, 1NLP and 4YEF. The earlier :doc:`docking` example
was developed on 4LB3; these six complexes provide a separate check.

PLP/ChemPLP can be selected with ``--engines plp_ga chemplp_ga``;
``plp_mc``, ``plp_de``, ``chemplp_mc`` and ``chemplp_de`` are also available.
ACO selectors are ``aco`` (Vina scoring), ``vinardo_aco``, ``ad4_aco``,
``plp_aco`` and ``chemplp_aco``. Existing archived measurements predate ACO.
Bayesian selectors are ``bo`` (Vina scoring), ``vinardo_bo``, ``ad4_bo``,
``plp_bo`` and ``chemplp_bo``. Existing archived measurements also predate BO.
The archived measurements below predate these two scorers. No PLP/ChemPLP
accuracy or speed comparison is implied by those measurements. See
:doc:`docking` for parameter coverage and preparation differences.

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
mutation amplitude is 2 Angstroms, and the Vina-scoring hunt energy cap is 10. Accepted
promising minima receive uncapped local refinement.
Vina uses exhaustiveness 8 and one CPU. These parameters specify
different search budgets; they are not a matched number of energy evaluations.
All engines return at most five poses with an energy-range setting of 3,
interpreted using each engine's own energy convention.
PLP and ChemPLP use uncapped MC minimization and empirical score units.
Their ``temperature=1.2`` and ``energy_range=3`` defaults have not been
calibrated against Vina. Their rows save ``top_score`` and ``score_units``;
``top_affinity`` is left empty. All scorer parameters, torsion fallback counts
and metal-geometry adaptations are recorded in the JSON provenance.
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

   python -m asv_bench.docking_pdbbind --engines ga --ga-population 100 --output benchmark-results/ga100.json

All internal engines support optional tqdm progress bars. Add ``--progress``
to show them; the default quiet mode is used for timing comparisons. Install
``oddt[progress]`` or ``tqdm`` to enable this option. The display flag does not
change the recorded search budget or pose results.

DE and search diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~

DE is available explicitly through ``--engines de``; the default engine list
remains GA, MC and Vina. It shares the internal scoring base and final pose
selection. Defaults are 32 members, 20 generations, ten short L-BFGS-B
iterations, and further refinement up to the runner's common ``--maxiter 60``.
Success-history adaptation, RMSD niche competition and restarts are described
in :doc:`docking`. Customize the budget with ``--de-population``,
``--de-generations`` and ``--de-local-maxiter``. For example:

.. code-block:: console

   python -m asv_bench.docking_pdbbind --engines ga mc de vina --diagnostics --output benchmark-results/four_engines.json

``--diagnostics`` records the closest symmetry RMSD among all feasible
local-search results, including discarded proposals, and separately among
uncapped results. The latter includes DE's short local searches, which may
not have converged. These oracle measurements describe sampling, independently
of the returned-pose energy ranking. They never affect search decisions.
The JSON also records reference total energy before and after local
minimization, its resulting RMSD and feasibility, and the top returned pose's
total energy under the same internal scoring function. Comparing these
values helps identify scoring and reference-geometry limitations. External
Vina does not expose local-search snapshots through its Python API.

Ant-colony budget
~~~~~~~~~~~~~~~~~

ACO defaults to a 32-pose pheromone archive, 16 ants per iteration and 20
iterations, with full uncapped local refinement up to ``--maxiter 60``.
That is 352 local minimizations, including initialization. Its rank width
is 0.2, kernel-distance multiplier 0.85, minimum width fraction 0.02 and
random-immigrant probability 0.1. The runner records these parameters and
all source checksums. This budget is not matched to GA, MC or DE.

``--aco-ants``, ``--aco-iterations`` and ``--aco-archive`` customize the
budget. Progress and sampling diagnostics use the same interfaces as
the other internal searches. For example:

.. code-block:: console

   python -m asv_bench.docking_pdbbind --engines plp_aco chemplp_aco --aco-ants 16 --aco-iterations 20 --aco-archive 32 --seeds 42 --diagnostics --output benchmark-results/aco.json

Bayesian search budget
~~~~~~~~~~~~~~~~~~~~~~

BO defaults to 24 initial poses and 80 acquisition/random evaluations,
each refined up to ``--maxiter 60``, for 104 local minimizations. Each
acquisition uses a 512-candidate pool and at most 128 GP training samples.
The Matérn-5/2 length scale is fixed at 0.5, normalized observation variance
is 1e-6, improvement margin is 0.01, local candidate fraction is 0.75,
local proposal scale is 0.15 and random-evaluation probability is 0.1.
See :doc:`docking` for pose features, target normalization and memory rules.

Use ``--bo-initial``, ``--bo-iterations``, ``--bo-candidates`` and
``--bo-max-samples`` to change the budget. The JSON records the actual
settings, scikit-learn version and source checksums. Timing includes GP
fitting and acquisition selection as well as scoring, and the default
evaluation budget is not matched to the other search methods. For example:

.. code-block:: console

   python -m asv_bench.docking_pdbbind --engines bo chemplp_bo --bo-initial 24 --bo-iterations 80 --bo-candidates 512 --bo-max-samples 128 --seeds 42 --diagnostics --output benchmark-results/bo.json

Snapshot-copy overhead is included in diagnostic run times. RMSD evaluation
and reference minimization happen after timing. Use the same diagnostic flag
for timing comparisons and a new output path for each protocol. Full DE
versus Vina superiority requires a wider comparison across seeds and matched
search effort; the historical results below predate DE.

Runtime
~~~~~~~

Runtime includes engine construction and docking, including Vina's PDBQT
writing and interaction-grid generation. Shared input loading, hydrogen
addition and pose evaluation are excluded. Runs execute sequentially on the
same Apple M2 machine (eight physical/logical cores). Vina uses one CPU;
the internal search also runs serially. The Vina adapter maps original PDBQT heavy-atom serials back
to the source ligand graph, avoiding hydrogen-count differences in file reading.
Vina PDBQT input is generated from implicit hydrogens with polar donor
hydrogens added explicitly. Nonpolar hydrogens are omitted before Vina torsion
perception, preventing terminal methyl groups from gaining extra torsions.
The Vina and Vinardo internal engines use the fully hydrogenated source and
score heavy atoms. AD4 prepares private copies with Gasteiger charges, merges
nonpolar-H charges and scores heavy atoms plus explicit polar H. Its
directionality and joint H-bond saturation are enabled by default; see
:doc:`docking` for the direct-pair model and its differences from AutoGrid.
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

Published data
--------------

All published measurements are stored in one
:download:`JSON archive <../_static/benchmarks/docking.json>` and one derived
:download:`per-run CSV <../_static/benchmarks/docking.csv>`. The JSON's
``experiments`` entries retain independent protocols, package versions,
source/input checksums, per-pose RMSDs and counters:

* ``baseline``: 54 GA/MC/Vina runs, including GA population 24.
* ``ga48``: 18 GA population-48 runs, plus the baseline reproducibility control.
* ``ga100``: 18 GA population-100 runs and comparison provenance.
* ``de_probe``: six DE runs with sampling and reference diagnostics.
* ``scoring_ga100``: 36 Vinardo/AD4 GA population-100 runs, using the
  archived ``ga100`` experiment as the Vina-scoring comparison reference.

* ``ad4_optimized_ga100``: 18 AD4 GA population-100 runs after pair caching and
  hydrogen-bond geometry optimizations, with the original search budget.

The CSV has 150 rows and an ``experiment`` column. Filter by that column
before comparing runs; budgets and diagnostic settings differ across
experiments. The JSON also retains ``validation.4lb3`` and
``validation.mc_performance``. These functional and profiling records have
different schemas and are not included in the PDBbind CSV.

``source_files`` records the original filenames and SHA-256 hashes. Those
filenames also appear in historical baseline references inside the unchanged
reports; resolve them through this index. Superseded partial GA/MC runs,
the preliminary GA/MC probe and older MC-only 4LB3 measurements were removed.

.. figure:: ../_static/benchmarks/docking.svg
   :alt: Median docking accuracy, runtime and recovery for every archived experiment.

   One overview covers all 150 runs. Each row retains its experiment, engine
   and seed count; search budgets differ between rows. Cells show per-complex
   medians. Outlines mark median RMSD below 2 Angstroms; the right columns
   count recovered runs, including failed runs as misses. DE is a single-seed
   probe with diagnostics. Timing colors use a logarithmic scale.

The archive, its derived CSV and this SVG are the three published benchmark
assets. Regenerate the overview with
``python -m asv_bench.docking_pdbbind_report --overview``. Optional image
exports use ``--figure path.svg`` or ``--figure path.png`` and create only
the requested format.

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

Download the :download:`per-run CSV <../_static/benchmarks/docking.csv>` or :download:`JSON archive <../_static/benchmarks/docking.json>`. These results are in experiment ``baseline``.

.. benchmark-results-end

Reproduce the comparison
------------------------

Install the RDKit backend and Vina Python package, then run from the repository
root:

.. code-block:: bash

   python -m asv_bench.docking_pdbbind

The runner writes ``benchmark-results/docking.json`` with settings, package
versions, source/input checksums, per-pose RMSDs and run status. It checkpoints
after each completed run. This local directory is ignored by Git; publishing
a completed experiment to the archive generates the shared CSV. To resume an
interrupted run with unchanged sources and settings:

.. code-block:: bash

   python -m asv_bench.docking_pdbbind --resume

After all runs finish, add the completed report to the archive using a unique
experiment name, then generate its tables and refresh the shared overview:

.. code-block:: bash

   python -m asv_bench.docking_benchmark_data --input benchmark-results/docking.json --experiment new_comparison
   python -m asv_bench.docking_pdbbind_report --experiment new_comparison
   python -m asv_bench.docking_pdbbind_report --overview

The archive command rejects incomplete or duplicated run grids and preserves
existing experiment names. It regenerates the consolidated CSV automatically.
To regenerate the published baseline without running docking, use
``python -m asv_bench.docking_pdbbind_report --experiment baseline``.
The renderer also accepts a standalone report through ``--input``. Table
updates create no image exports unless ``--figure`` is supplied.

The default engine list is ``ga mc vina``. Use ``--engines ga mc`` to compare
only the two searches on shared scoring. Search budgets can be varied with
``--ga-population``, ``--ga-generations``, ``--mc-exhaustiveness``,
``--mc-steps``, ``--de-population``, ``--de-generations``, ``--de-local-maxiter``
and ``--maxiter``; the exact values are saved in the protocol.
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

The population-100 experiment has unchanged inputs, numerical package versions and search settings apart from population. Source checks confirm that the GA population defaults are the only engine and runner changes from the population-48 experiment. The ``ga100`` experiment in the :download:`JSON archive <../_static/benchmarks/docking.json>` records source/input checksums, baseline checksums, per-pose RMSDs and search statistics; filter the :download:`CSV <../_static/benchmarks/docking.csv>` by ``experiment=ga100`` for the per-run measurements.

An earlier population-24 control on 10GS, seed 42, reproduced every returned RMSD, top affinity and search counter exactly after adding progress support. It is retained under ``experiments.ga48.baseline_control`` in the same JSON archive.

.. ga-population-results-end

DE validation probe
-------------------

The adaptive DE implementation was checked on all six complexes using seed
42, population 32, 20 generations, ten short iterations and at most 60
further refinement iterations. Diagnostics were enabled and progress was off.
All six runs completed. Top and best-returned recovery were both 1/6
(4YEF); the median top symmetry RMSD was 8.98 Angstroms and median docking
time was 46.8 seconds. This single-seed probe does not establish an accuracy
improvement over GA, MC or Vina. The historical comparisons above use
different budgets and have no snapshot-copy overhead.

.. list-table:: DE probe: symmetry RMSD in Angstroms and runtime in seconds
   :header-rows: 1

   * - Complex
     - Top returned
     - Best returned
     - Closest sampled
     - Time (s)
   * - 10GS
     - 6.22
     - 5.52
     - 3.75
     - 37.3
   * - 1IMX
     - 9.00
     - 6.43
     - 4.94
     - 50.8
   * - 1K9Q
     - 12.52
     - 7.44
     - 4.59
     - 45.1
   * - 1NLO
     - 8.97
     - 8.89
     - 4.07
     - 63.5
   * - 1NLP
     - 14.52
     - 10.41
     - 5.07
     - 48.4
   * - 4YEF
     - 1.48
     - 1.00
     - 0.87
     - 38.8

The closest sampled result was still above 2 Angstroms on each of the five
missed complexes. For 10GS the minimized reference remained close to the
input (0.70 Angstroms), with total score -15.84 versus DE's -13.81: DE missed
a better-scoring reference basin. For 1K9Q, DE's distant top pose scored
-10.67 versus -9.19 for the minimized reference (1.46 Angstroms from input).
Here the internal score favored the distant pose over that reference minimum.
These examples show why search and scoring need separate diagnostics.

The ``de_probe`` experiment in the :download:`JSON archive <../_static/benchmarks/docking.json>`
contains protocol, source/input checksums, search counters and reference diagnostics.
Filter the :download:`per-run CSV <../_static/benchmarks/docking.csv>` by
``experiment=de_probe`` for these measurements.
Reproduce it with:

.. code-block:: console

   python -m asv_bench.docking_pdbbind --engines de --seeds 42 --diagnostics --output benchmark-results/de_probe.json

AD4 and Vinardo scoring comparison
----------------------------------

The AD4 measurements in this section precede the performance optimizations
reported in the final section below.

This comparison holds the genetic search fixed at population 100, eight
generations and at most 60 local L-BFGS-B iterations, with seeds 1, 7 and 42
on all six fixtures. Each internal engine performs 804 local minimizations.
Scorer-specific forces and preparation differ. AD4 uses directional,
saturable hydrogen bonds; terminal polar-H torsions retain their input
geometry. No target-specific parameters are adjusted.

AD4 here is ODDT's independently implemented direct-pair adaptation. All
three scoring models use the internal genetic search; the earlier external
Vina comparison remains a separate experiment.

The comparison view includes the archived Vina-scoring ``ga100`` experiment
as a historical reference. The renderer verifies identical target/seed grids,
GA settings, numerical package versions and input-file checksums before
combining the view. Original measurements remain in their own archive entries;
the reference rows are not duplicated into the new experiment. Absolute
binding-score values are not calibrated identically across these models, so
the comparison uses pose recovery, RMSD and runtime.

On these six fixtures, all three models recover only 4YEF, in every seed:
top and any-returned recovery are both 3/18. Vinardo has lower recorded
median top RMSD and runtime than the historical Vina-scoring GA100 reference.
AD4 gives the closest top poses on 4YEF (median 0.80 Angstroms), but has the
highest overall median RMSD and runtime. These fixtures do not establish
general scorer superiority.

Reproduce the new runs, publish the complete experiment and update this
section with:

.. code-block:: console

   python -m asv_bench.docking_pdbbind --engines vinardo_ga ad4_ga --ga-population 100 --ga-generations 8 --maxiter 60 --seeds 1 7 42 --output benchmark-results/scoring_ga100.json --resume
   python -m asv_bench.docking_benchmark_data --input benchmark-results/scoring_ga100.json --experiment scoring_ga100
   python -m asv_bench.docking_pdbbind_report --experiment scoring_ga100 --compare-experiment ga100 --marker scoring-results --title "GA100 scoring comparison"
   python -m asv_bench.docking_pdbbind_report --overview

.. scoring-results-start

GA100 scoring comparison
------------------------

This report uses seeds 1, 7, 42 and engines Vina scoring GA, Vinardo GA, AD4 GA. Vina scoring GA: population 100, generations 8, maxiter 60. Vinardo GA: population 100, generations 8, maxiter 60. AD4 GA: population 100, generations 8, maxiter 60. These budgets are not matched energy-evaluation counts.


The Vina-scoring reference is reused from the archived ``ga100`` experiment, not rerun. Target/seed grids, search budgets, numerical package versions and input checksums match. Its timing measurements are historical; source versions remain recorded separately in the archive.

RMSDs below are symmetry-aware, in Angstroms. Times are seconds. Per-target values are medians over 3 seeds.

.. list-table:: Across all target/seed runs
   :header-rows: 1

   * - Engine
     - Top <2 A
     - Any returned <2 A
     - Median top RMSD
     - Median time
     - Errors
   * - Vina scoring GA
     - 3/18
     - 3/18
     - 7.48
     - 106.5
     - 0
   * - Vinardo GA
     - 3/18
     - 3/18
     - 6.36
     - 87.1
     - 0
   * - AD4 GA
     - 3/18
     - 3/18
     - 8.25
     - 744.8
     - 0

.. list-table:: Pose recovery by complex
   :header-rows: 1

   * - Complex
     - Heavy atoms
     - Internal torsions
     - Vina scoring GA RMSD
     - Vinardo GA RMSD
     - AD4 GA RMSD
   * - 10GS
     - 33
     - 13
     - 7.60
     - 2.47
     - 8.07
   * - 1IMX
     - 60
     - 22
     - 6.33
     - 4.45
     - 6.94
   * - 1K9Q
     - 46
     - 16
     - 12.71
     - 6.82
     - 12.68
   * - 1NLO
     - 84
     - 24
     - 7.30
     - 7.08
     - 10.47
   * - 1NLP
     - 76
     - 16
     - 7.51
     - 6.11
     - 12.36
   * - 4YEF
     - 77
     - 7
     - 1.14
     - 1.33
     - 0.80

.. list-table:: Recovered runs by complex (RMSD <2 A)
   :header-rows: 1

   * - Complex
     - Vina scoring GA top
     - Vinardo GA top
     - AD4 GA top
     - Vina scoring GA any returned
     - Vinardo GA any returned
     - AD4 GA any returned
   * - 10GS
     - 0/3
     - 0/3
     - 0/3
     - 0/3
     - 0/3
     - 0/3
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
     - 3/3
     - 3/3
     - 3/3
     - 3/3
     - 3/3

.. list-table:: Median runtime by complex
   :header-rows: 1

   * - Complex
     - Vina scoring GA (s)
     - Vinardo GA (s)
     - AD4 GA (s)
   * - 10GS
     - 80.9
     - 58.4
     - 562.1
   * - 1IMX
     - 114.0
     - 101.7
     - 798.2
   * - 1K9Q
     - 93.3
     - 76.5
     - 404.9
   * - 1NLO
     - 157.0
     - 162.0
     - 923.6
   * - 1NLP
     - 116.4
     - 99.9
     - 711.9
   * - 4YEF
     - 93.3
     - 70.4
     - 1076.2

.. list-table:: Median search work
   :header-rows: 1

   * - Complex
     - Engine
     - Local minimizations
     - Objective evaluations
   * - 10GS
     - Vina scoring GA
     - 804
     - 126816
   * - 10GS
     - Vinardo GA
     - 804
     - 86908
   * - 10GS
     - AD4 GA
     - 804
     - 73903
   * - 1IMX
     - Vina scoring GA
     - 804
     - 137668
   * - 1IMX
     - Vinardo GA
     - 804
     - 117233
   * - 1IMX
     - AD4 GA
     - 804
     - 92272
   * - 1K9Q
     - Vina scoring GA
     - 804
     - 147518
   * - 1K9Q
     - Vinardo GA
     - 804
     - 115324
   * - 1K9Q
     - AD4 GA
     - 804
     - 88684
   * - 1NLO
     - Vina scoring GA
     - 804
     - 154660
   * - 1NLO
     - Vinardo GA
     - 804
     - 144086
   * - 1NLO
     - AD4 GA
     - 804
     - 101362
   * - 1NLP
     - Vina scoring GA
     - 804
     - 147567
   * - 1NLP
     - Vinardo GA
     - 804
     - 116468
   * - 1NLP
     - AD4 GA
     - 804
     - 94345
   * - 4YEF
     - Vina scoring GA
     - 804
     - 148575
   * - 4YEF
     - Vinardo GA
     - 804
     - 106054
   * - 4YEF
     - AD4 GA
     - 804
     - 77940


Download the :download:`per-run CSV <../_static/benchmarks/docking.csv>` or :download:`full JSON report <../_static/benchmarks/docking.json>`. The JSON records the protocol, source/input checksums, per-pose RMSDs and search statistics.

.. scoring-results-end

.. ad4-optimization-start

AD4 performance after optimization
----------------------------------

This reruns all six fixtures and seeds 1, 7 and 42 with AD4 GA population 100, eight generations and maxiter 60. Runs are sequential on the same machine, with progress and diagnostics disabled. The pair cache, batched tetrahedral lone-pair frames, zero-donor angular pruning and energy-only Jacobian shortcut preserve the scoring equations. Input files, numerical package versions, scoring parameters, search code and budgets match the historical ``scoring_ga100`` AD4 runs. The previous results remain in that archive entry; new measurements are in ``ad4_optimized_ga100``.

All 18 optimized runs completed without errors. The ratio of overall median
runtimes is 1.67x; the median of the 18 paired runtime ratios is 1.62x. These
are full docking timings, including preparation specific to the engine and
output scoring. Top and any-returned recovery remain 3/18, all on 4YEF.
Overall median top-pose RMSD increases from 8.25 to 9.31 Angstroms.

.. list-table:: AD4 across the same 18 target/seed runs
   :header-rows: 1

   * - Version
     - Median time (s)
     - Median top RMSD (A)
     - Top <2 A
     - Any returned <2 A
   * - Original AD4 GA100
     - 744.8
     - 8.25
     - 3/18
     - 3/18
   * - Optimized AD4 GA100
     - 444.8
     - 9.31
     - 3/18
     - 3/18

.. list-table:: AD4 by complex: medians over three seeds
   :header-rows: 1

   * - Complex
     - Original time (s)
     - Optimized time (s)
     - Time ratio
     - Original top RMSD (A)
     - Optimized top RMSD (A)
   * - 10GS
     - 562.1
     - 343.5
     - 1.64x
     - 8.07
     - 7.23
   * - 1IMX
     - 798.2
     - 443.7
     - 1.80x
     - 6.94
     - 11.40
   * - 1K9Q
     - 404.9
     - 264.9
     - 1.53x
     - 12.68
     - 12.38
   * - 1NLO
     - 923.6
     - 576.4
     - 1.60x
     - 10.47
     - 10.47
   * - 1NLP
     - 711.9
     - 445.8
     - 1.60x
     - 12.36
     - 7.69
   * - 4YEF
     - 1076.2
     - 538.4
     - 2.00x
     - 0.80
     - 0.84

Objective-evaluation counts match in 0/18 paired runs, and all returned symmetry RMSDs match exactly in 1/18. The new archive entry includes each run's counters and per-pose RMSDs so changes in search work can be distinguished from evaluation cost. The small floating-point differences introduced by batched derivatives can affect the path of a long search.

Filter the :download:`shared CSV <../_static/benchmarks/docking.csv>` by ``experiment=ad4_optimized_ga100`` or inspect the :download:`JSON archive <../_static/benchmarks/docking.json>` for provenance and individual results.

Commands used for this experiment are below. Use a new output path and
experiment name to collect another measurement; existing archive entries
are preserved.

.. code-block:: console

   python -m asv_bench.docking_pdbbind --engines ad4_ga --ga-population 100 --ga-generations 8 --maxiter 60 --seeds 1 7 42 --output benchmark-results/ad4_optimized_ga100.json --resume
   python -m asv_bench.docking_benchmark_data --input benchmark-results/ad4_optimized_ga100.json --experiment ad4_optimized_ga100
   python -m asv_bench.docking_pdbbind_report --overview

.. ad4-optimization-end
