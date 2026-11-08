Internal docking engines
========================

:class:`oddt.docking.vina_ga`, :class:`oddt.docking.vina_mc` and
:class:`oddt.docking.vina_de` search ligand
translation, orientation and rotatable bonds against a rigid receptor. They
share Vina-inspired scoring and SciPy's L-BFGS-B local minimizer, with genetic,
Monte Carlo and adaptive differential evolution global search respectively.
They require NumPy, SciPy and an ODDT molecular
backend; no AutoDock Vina installation is needed.
The existing ``vina_docking`` name remains an alias for ``vina_mc``.

Prepare a receptor with appropriate protonation and set ``protein = True``
before computing atom properties. Supply a ligand with a three-dimensional
conformation. Ring conformations remain fixed during the search.
Conjugated carbonyl-heteroatom bonds, including amides and esters, also retain
their input geometry. For 4LB3 this leaves six searched heavy-atom torsions;
the earlier seven-torsion implementation incorrectly allowed the amide to rotate.

Redock a test complex
---------------------

Run this example from the repository root. The 4LB3 pocket and crystal ligand
are distributed with the tests. The reference defines the box and measures
recovery; all search positions, orientations and torsions are randomized.

.. code-block:: python

   import numpy as np
   import oddt
   from oddt.docking import vina_ga, vina_mc, vina_de
   from oddt.spatial import rmsd, rotate

   protein = next(oddt.toolkit.readfile("pdb", "tests/data/pdb/4lb3_pocket.pdb"))
   protein.protein = True
   reference = next(oddt.toolkit.readfile("sdf", "tests/data/pdb/4lb3_ligand.sdf"))
   reference.addh()
   heavy = reference.atom_dict["atomicnum"] != 1
   center = reference.coords[heavy].mean(axis=0)

   ligand = reference.clone
   ligand.coords = rotate(ligand.coords, 1.2, -0.7, 2.1) + [12, -7, 8]
   box = np.array([center - 8, center + 8])
   engine = vina_ga(protein, ligand, box=box)
   poses = engine.dock(population_size=100, generations=8, maxiter=60, seed=42, num_modes=5, progress=True)
   print("RMSD:", rmsd(reference, poses[0]))
   print("Affinity:", poses[0].data["vina_affinity"])
   poses[0].write("sdf", "docked.sdf", overwrite=True)

``box`` contains the lower and upper Cartesian corners in Angstroms.
All ligand heavy atoms must fit inside it. Returned poses are independent
molecule copies; the input receptor and ligand are preserved, including
explicit hydrogens, which move with their fragments.

Use the same inputs and scoring with Monte Carlo:

.. code-block:: python

   engine = vina_mc(protein, ligand, box=box)
   poses = engine.dock(exhaustiveness=8, steps=250, maxiter=60, seed=42, num_modes=5)

Or use adaptive differential evolution:

.. code-block:: python

   engine = vina_de(protein, ligand, box=box)
   poses = engine.dock(population_size=32, generations=20, local_maxiter=10,
                       maxiter=60, seed=42, num_modes=5, progress=True)

Increase ``population_size``/``generations`` for GA or DE, or
``exhaustiveness``/``steps`` for MC, for more extensive searching. ``maxiter``
controls each local minimization. A fixed ``seed`` makes runs reproducible within the same numerical
environment. Multiple seeds help assess search variability.

All engines accept ``progress=True`` to display a tqdm progress bar in a
terminal or notebook. Install the optional dependency with
``pip install "oddt[progress]"`` (or ``pip install tqdm``). Progress is off by
default and does not change random sampling or search results. GA counts
local minimizations, including its initial population; MC counts proposals
across all trajectories, advancing after any associated refinement finishes.
DE counts initialization, each generation, and final archive refinement.

Scoring base and search mixins
------------------------------

``vina_scoring`` supplies pair terms, analytic pose gradients, geometry,
L-BFGS-B refinement and final pose selection. ``GeneticAlgorithmMixin``,
``MonteCarloMixin`` and ``DifferentialEvolutionMixin`` supply only their global
search. The three public engines
are compositions of these classes; they use identical atom typing, search
torsions, restraints, energy ranking and clustering.

.. code-block:: python

   from oddt.docking import vina_scoring, GeneticAlgorithmMixin

   class MyEngine(GeneticAlgorithmMixin, vina_scoring):
       pass

   scorer = vina_scoring(protein, ligand)  # No box needed for scoring.
   print(scorer.score())

A custom scoring base must provide the pose/optimizer interface documented in
``oddt.docking.search``. Its ``optimize(x, maxiter, energy_cap)`` must accept
MC's hunting cap of 10 and the uncapped limit ``np.inf``; GA and DE use the default
uncapped objective. A box is required for ``optimize()`` and ``dock()``.

Genetic search
--------------

The initial population randomizes position, uniform orientation and permitted
torsions. Each generation retains the best eighth of the population, then
uses three-way tournament selection, per-gene crossover along the shortest
angular arc, Gaussian mutations and 10% random immigrants. Every new pose is
locally minimized. ``mutation_rate`` defaults to 0.2; positive rates mutate
at least one gene. The default population is 100. With the benchmark's eight
generations this performs 804 local minimizations, including the initial
population. Populations of 24 and 48 performed 192 and 384 respectively; larger populations
increase search effort but do not guarantee better pose recovery.

Monte Carlo search
------------------

Each of the ``exhaustiveness`` independent trajectories starts with randomized
position, orientation and torsions. Every step chooses one entity uniformly:
the translation, the orientation, or one torsion. Translation moves are drawn
from a sphere with radius ``mutation_amplitude`` (2 Angstroms by default).
Orientation changes use a rotation vector scaled by the ligand's gyration
radius. Torsion moves replace one angle with a uniform value in [-pi, pi].

L-BFGS-B refines each proposal before the Metropolis decision. Lower energies
are accepted; uphill moves are accepted with probability
``exp((old_energy - new_energy) / temperature)``, with default temperature 1.2.
Hunting softens positive energies using ``E * 10 / (10 + E)``. Promising
accepted poses receive uncapped refinement and enter a bounded collection of
distinct minima. Rejected proposals do not become the next starting pose.

These move rules follow Vina's `Monte Carlo implementation
<https://github.com/ccsb-scripps/AutoDock-Vina/blob/v1.2.7/src/lib/monte_carlo.cpp>`_
and `mutation implementation
<https://github.com/ccsb-scripps/AutoDock-Vina/blob/v1.2.7/src/lib/mutate.cpp>`_.
As in Vina, accepted refinement updates the running coordinates and energy;
the next proposal again uses the hunt cap. Saved minima use uncapped
refinement rather than Vina's finite high cap.
It uses the ligand centroid as the rotation anchor, SciPy L-BFGS-B, direct
pair scoring and explicit ``steps`` rather than Vina's grid/BFGS implementation
and automatic step heuristic. It is an independent approximation of the search.

Adaptive differential evolution
-------------------------------

DE uses current-to-pbest/1 mutation: each parent moves toward a randomly
selected member of the best ``pbest_fraction`` of the population, plus a
scaled difference between two donors. Displaced parents enter a donor
archive bounded by the population size. Translation reflects at the search
bounds, torsions use shortest angular differences, and orientations compose
relative rotations. Orientation crosses over as one block.

Successful energy improvements adapt the mutation scale and crossover
probability through a six-slot history. The scale uses a weighted Lehmer
mean and crossover uses a weighted arithmetic mean. This hybrid takes
inspiration from SHADE; its local search and molecular geometry operations
differ from canonical SHADE.

The defaults are 32 members, 20 generations and ten short local iterations.
Initial members, restarted members and improving trials receive further
refinement up to ``maxiter`` iterations. ``local_maxiter`` is capped at
``maxiter``. Proposals within ``diversity_rmsd=1`` Angstrom of another member
compete with that member, retaining the better energy in that region.
Crowded members and members stale for five generations can be restarted;
at most a quarter of the population (rounded up) restarts per generation,
protecting the best member. Set ``restart_fraction=0`` to disable restarts.

The flexible 4LB3 regression test uses population 32, 20 generations, ten
short iterations, ``maxiter=60`` and seed 42. From the displaced input above,
DE returned a top pose at 0.62 Angstrom direct RMSD in 28.4 seconds with
RDKit, and 0.50 Angstrom in 30.7 seconds with OpenBabel, on Apple M2.
These times cover ``dock()``; both runs preserve hydrogens and all six
searched torsions. The wider fixture probe is in :doc:`docking-benchmarks`.

A separate archive keeps up to ``max(archive_size, num_modes)`` feasible pose
clusters (default ``archive_size=64``). Remaining short-search archive members
receive full refinement before output ranking. These extra local searches
are counted in the statistics, so ``population_size * generations`` alone
does not measure the total search cost. DE remains experimental; improving
the scoring objective does not guarantee improved native-pose recovery.

Scores and diagnostics
----------------------

The five Vina-inspired pair terms are Gaussian attraction (two terms),
repulsion, hydrophobic contacts and hydrogen bonds. The optimization objective
includes receptor interactions, ligand internal energy and a box restraint.
Internal pairs separated by more than three bonds are counted once each.
This corrects the earlier internal scorer's double counting of intra-ligand
terms; its intermolecular terms remain unchanged.

Poses are ordered by ``vina_total = vina_inter + vina_intra``. The
``vina_affinity`` field is the intermolecular score divided by the Vina
rotatable-bond penalty. The ``energy_range`` filter uses total energy;
``min_rmsd`` removes poses close to an already selected pose using direct
heavy-atom matching without alignment or symmetry correction.

``engine.history_`` records the best restrained objective after each MC
proposal, initially and after each GA generation, or initially, after each
DE generation and after final archive refinement. All engines report
``local_minimizations`` and SciPy ``objective_evaluations`` in
``engine.search_stats_``. MC also reports accepted moves, accepted uphill
moves and uncapped refinements separately for each trajectory.
DE reports proposals, accepted replacements, restarts, short searches and
full refinements. ``engine.parameter_memory_`` contains its final mutation
scale and crossover memories.
``engine.results_`` holds SciPy optimizer results for the returned poses,
including iteration counts, termination status and pose genes. A local search
that reaches its iteration limit may still produce an improved candidate.
Use ``engine.optimize(maxiter=100)`` to refine only the supplied pose.

For an optional read-only view of every local-search result, including
discarded proposals, pass ``sample_callback=observer`` to the constructor.
The observer receives ``(result, energy_cap)``. Copy ``result.coords`` to
retain heavy coordinates. ``energy_cap`` is 10 for MC hunting and ``None``
or infinity for uncapped searches. Callback exceptions propagate.
The :doc:`docking-benchmarks` runner uses this hook for optional post-search
sampling diagnostics without exposing reference coordinates to the search.

MC performance
--------------

Profiling a 10GS trajectory (one chain, 30 proposals, seed 42, up to 60 local
iterations) identified pose-gradient torque calculations and pair-gradient
accumulation as hot paths. The shared base now uses specialized three-vector
cross products, compiled ``bincount`` accumulation, a cached intraligand pair
mask and reuse of the initial objective in each L-BFGS-B call.

Three alternating, unprofiled before/after pairs on the same machine reduced
median ``dock()`` time from 4.48 to 3.70 seconds: 1.21 times faster, or 17.3%
less runtime. The selected genes, coordinates, energies and entire search
history were identical. The ``validation.mc_performance`` entry in the
:download:`benchmark archive <../_static/benchmarks/docking.json>` records
every timing and settings.
Both sides use identical scoring and reduction order. Direct scoring retains
the original term reductions; MC refinement uses the infinite-cap limit of
its hunting evaluator, preserving its numerical reduction order too.
This short probe measures implementation cost, not docking accuracy; it does
not recover 10GS's native pose. The same optimizations also benefit GA.

At the benchmark defaults MC makes 2,000 proposals plus accepted-pose
refinements, whereas GA now performs 804 local minimizations. The implementation
speedup does not remove this budget difference. Compare measured objective
evaluations as well as accuracy and elapsed time when choosing a search.

4LB3 validation
---------------

Both algorithms start from a displaced and rotated ligand, randomize all
permitted search coordinates, and use a 16 Angstrom cubic box, seed 42,
at most 60 L-BFGS-B iterations per minimization, and up to five returned modes.
GA uses population 24 and eight generations; MC uses eight trajectories and
250 proposals per trajectory. The regression checks direct heavy-atom RMSD without alignment,
input preservation, box containment, pose clustering and fresh score metadata.

.. mc-validation-start

Both backends passed all 61 focused docking, benchmark and scoring tests.
The top-ranked 4LB3 results were:

.. list-table:: Flexible redocking, seed 42
   :header-rows: 1

   * - Backend
     - GA top RMSD
     - MC top RMSD
   * - RDKit
     - 0.32 Angstrom
     - 0.32 Angstrom
   * - Open Babel
     - 0.62 Angstrom
     - 0.56 Angstrom

Both retain the requirement of RMSD below 1 Angstrom. The
:download:`benchmark archive <../_static/benchmarks/docking.json>`, under
``validation.4lb3``, records settings, versions and source/input checksums.
These functional suites ran
concurrently, so their durations are not used as performance measurements.
See the isolated MC timing probe above and the sequential PDBbind comparison
below for runtime measurements. This combined validation supersedes the
earlier MC-only measurements.

.. mc-validation-end

The engine keeps conjugated bonds rigid and projects Cartesian forces onto
translation, rotation and torsion coordinates analytically. Its ODDT atom typing,
scores and rankings can differ from AutoDock Vina. One successful redocking
case does not establish accuracy across targets, ligands or protonation states.

See :doc:`docking-benchmarks` for the completed six-complex comparison of
GA, MC and external Vina across three seeds.
