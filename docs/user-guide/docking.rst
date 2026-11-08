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

``vinardo_scoring`` and ``ad4_scoring`` provide two additional independently
implemented models. Each has ``_ga``, ``_mc`` and ``_de`` engines, for example
``vinardo_ga`` and ``ad4_mc``. AD4 is a **radial direct-pair variant**, with the
limitations and preparation requirements described below.

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

``DockingBase`` supplies ligand geometry, Cartesian-to-pose gradient
projection, box restraints, L-BFGS-B refinement and final pose selection.
``LigandPose`` evaluates absolute translation, orientation and torsion genes,
including attached hydrogens for output. ``vina_scoring`` extends the base
with Vina atom radii, pair terms, energy gradients, torsional correction and
the existing ``vina_*`` output fields. Its public scoring methods and import
paths remain available, including the ``vina_docking`` alias.

``GeneticAlgorithmMixin``, ``MonteCarloMixin`` and
``DifferentialEvolutionMixin`` supply the global searches. All three Vina
engines compose their search with ``vina_scoring``, sharing the same typing,
restraints and output selection.

.. code-block:: python

   from oddt.docking import vina_scoring, GeneticAlgorithmMixin

   class MyEngine(GeneticAlgorithmMixin, vina_scoring):
       pass

   scorer = vina_scoring(protein, ligand)  # No box needed for scoring.
   print(scorer.score())

Vinardo scoring
~~~~~~~~~~~~~~~

.. code-block:: python

   from oddt.docking import vinardo_scoring, vinardo_ga

   scorer = vinardo_scoring(protein, ligand)
   print(scorer.score(), scorer.score_components())
   engine = vinardo_ga(protein, ligand, box=box)
   poses = engine.dock(population_size=100, generations=8, seed=42)
   print(poses[0].data["vinardo_affinity"])

The implementation follows the equations and Tables 1–3 in
`Quiroga and Villar (2016) <https://doi.org/10.1371/journal.pone.0155183>`_.
It has one Gaussian (width 0.8 Angstrom, weight -0.045), quadratic overlap
repulsion (0.8), hydrophobic attraction (-0.035, surface-gap transition from
0 to 2.5 Angstrom), and hydrogen bonding (-0.6, transition from -0.6 to 0
Angstrom). Carbon radii are 2.0 Angstrom for aliphatic and 1.9 for aromatic
atoms; nitrogen and oxygen radii are 1.7 and 1.6 respectively. Other supported
elements retain the Vina radii. Pairs use an 8 Angstrom center-distance cutoff.
Unsupported elements raise an error. Typing uses ODDT's donor, acceptor and
hydrophobic flags, so results need not match other programs' preparation.

Search energy is the weighted sum of intermolecular and intraligand terms,
excluding intraligand pairs separated by three or fewer bonds. Reported
affinity is the intermolecular energy divided by ``1 + 0.05846 * N_rot``,
using the same PDBQT-style rotor count as the existing Vina scorer.
``weights`` accepts four pair weights followed by the torsional coefficient.
``score_inter()`` and ``score_intra()`` return four unweighted terms in the
order Gaussian, repulsion, hydrophobic and hydrogen bond.

AD4 radial scoring
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from oddt.docking import ad4_scoring, ad4_mc

   scorer = ad4_scoring(protein, ligand)  # Private copies: add H and Gasteiger charges.
   print(scorer.score(), scorer.score_components())
   engine = ad4_mc(protein, ligand, box=box)
   poses = engine.dock(exhaustiveness=8, steps=250, seed=42)
   print(poses[0].data["ad4_affinity"])

The equations are independently written from
`Huey et al. (2007) <https://doi.org/10.1002/jcc.20634>`_; atom parameters and
bound-state weights are numerical specifications in the
`AutoDock4.2.6 User Guide, pp. 34–35
<https://autodock.scripps.edu/wp-content/uploads/sites/56/2022/04/AutoDock4.2.6_UserGuide.pdf>`_.
No upstream implementation code is included in either new scorer.

AD4 includes 12-6 van der Waals interactions, radial 12-10 hydrogen bonding,
Mehler–Solmajer distance-dependent dielectric electrostatics and
charge-dependent Gaussian desolvation. The Gaussian sigma is 3.5 Angstrom,
as published in the paper. The five default weights are 0.1662, 0.1209,
0.1406, 0.1322 and 0.2983 for vdW, H-bond, electrostatics, desolvation and
torsional entropy. H-bond pairs replace the vdW term. The optional
``smoothing=0.5`` takes the minimum vdW/H-bond potential within a full-width
0.5 Angstrom interval; ``smoothing=0`` evaluates the unsmoothed equations.
These short-range terms have an 8 Angstrom cutoff. Electrostatics and
desolvation have no cutoff. Distances below 0.1 Angstrom are clamped.

This implementation **omits AutoGrid's angular H-bond factors, H-bond
saturation and grid interpolation**. It is useful as an AD4 radial model
with independent search, but its affinities are not numerically equivalent
to AutoDock4. Rings, conjugated bonds and terminal polar-H torsions retain
their supplied geometry; the additive torsion count includes searched
heavy-atom bonds. These distinctions affect both scores and pose recovery.

By default, both molecules are cloned, missing hydrogens are added and
Gasteiger charges are calculated on those private copies. Correct protonation,
tautomer choice and missing heavy atoms remain the caller's responsibility.
Nonpolar H charges are merged into their bonded parent. Heavy atoms and
explicit polar H participate in scoring; inactive H have zero force. Output
molecules include added ligand hydrogens, while the inputs are preserved.
Unsupported elements or nonfinite partial charges raise an error. The parameter
table supports H, C, N, O, F, Mg, P, S, Cl, Ca, Mn, Fe, Zn, Br and I.

Use ``charge_model=None`` to preserve supplied partial charges. Explicit
polar hydrogens must already be present, and any omitted nonpolar-H charges
must already be merged. This option is also appropriate for prepared metal
sites for which Gasteiger charge calculation fails. When calling scoring
methods with ``coords``, AD4 requires coordinates for **every atom of the
prepared ligand**, in order, including H. The default coordinates are
available as ``scoring_coordinates``; after local search obtain the full
coordinates with ``engine.lig.mutate(result.x, all_atoms=True)``.

Search minimizes intermolecular plus intraligand energy. By default the
unbound reference is the bound conformation, so intraligand energy cancels
from the reported affinity: ``inter + 0.2983 * N_rot``. A fixed
``unbound_energy`` instead reports ``inter + intra - unbound_energy + torsion``.
Output ranking still uses the search energy, as with the other scorers.
``score_inter()`` and ``score_intra()`` return four unweighted terms in the
order vdW, H-bond, electrostatic and desolvation.

Both scorers emit ``docking_scorer``, ``docking_energy``, ``docking_score``
and weighted ``docking_term_inter_*`` / ``docking_term_intra_*`` fields.
They also emit ``vinardo_affinity`` / ``ad4_affinity``, model-prefixed totals
and rotor counts. AD4 records ``ad4_model=radial``, charge preparation,
smoothing and unbound-reference settings. Energies are in kcal/mol; these
empirical estimates should not be compared across scoring models as if
they were the same calibrated observable.

Scorer validation
~~~~~~~~~~~~~~~~~

Tests check published pair minima, independent dielectric reference values,
charge conservation, analytic gradients on 4BFQ and 4LB3 (including box
restraints and capped MC energies), and composition with all three searches
on both RDKit and OpenBabel. Saved Vina energy, gradient, optimization and
seeded search controls remain exactly unchanged on both backends.

A single RDKit 4LB3 redocking validation used the displaced flexible ligand
from the example above, GA population 24, eight generations, 60 local
iterations, seed 42 and five output modes:

================  ==================  =======================
Scorer            Top-pose RMSD (Å)   Best returned RMSD (Å)
================  ==================  =======================
Vinardo           0.564               0.564
AD4 radial        5.781               1.126
================  ==================  =======================

This checks integration on one complex, not a broad comparison of scoring
accuracy. AD4 radial found a near-native candidate but ranked another pose
first. Its omitted H-bond directionality and preparation differences remain
material limitations.

Implementing another scorer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subclass ``DockingBase`` and implement
``energy_and_gradient(coords, energy_cap=None)``. It receives an ``N x 3``
array of heavy-atom coordinates and returns an unrestrained energy and its
Cartesian gradient with the same shape. Include the model's intermolecular
and intramolecular contributions. The base adds box restraints and projects
forces onto the pose genes. A scorer can set ``score_all_atoms=True`` to
receive every explicit ligand atom, as AD4 does. Its gradient must have the
same shape; inactive atoms may have zero force. Rotation and torsion
projection retain the heavy-atom centroid as the translation anchor, and
box restraints and RMSD clustering still use heavy atoms. Such scorers
must supply compatible full coordinates for default scoring and metadata.

The minimal composition is:

.. code-block:: python

   from oddt.docking import DockingBase, GeneticAlgorithmMixin

   class MyScoring(DockingBase):
       scoring_name = "my_model"

       def energy_and_gradient(self, coords, energy_cap=None):
           # Supply this model's scalar energy and Cartesian derivatives.
           return my_potential(self.rec_dict, self.lig_dict, coords, energy_cap)

   class MyEngine(GeneticAlgorithmMixin, MyScoring):
       pass

``energy(coords)`` defines output ranking and defaults to the energy from
that hook. Override it to provide an efficient evaluation without gradients.
``score(coords)`` defines the separately reported score and defaults to
``energy``. Search acceptance and output ranking minimize energy, regardless
of the reported score's convention. ``score_components(coords)`` may return
any number of named, weighted contributions. ``prepare_atom_dict`` can
parameterize copied atom dictionaries while preserving atom order, ids and
coordinates. Model-specific caches can be cleared by overriding the receptor
and ligand setters and calling their base methods.

Generic outputs use ``docking_scorer``, ``docking_energy``, ``docking_score``
and ``docking_term_<name>`` fields. ``_pose_metadata`` and
``score_metadata_prefixes`` customize annotation and removal of stale fields.
Vina retains its existing unweighted term fields and torsion-corrected
``vina_affinity`` convention. Optimizer results expose the reported value as
``score``, with ``affinity`` retained as a compatibility alias.

MC obtains hunting and refinement caps from ``mc_hunt_energy_cap`` and
``mc_refinement_energy_cap``. The generic defaults are ``None``; Vina uses
10 and infinity, preserving its two numerical evaluation paths. A scorer
that accepts a finite cap must differentiate the capped energy consistently.
``None`` and infinity represent uncapped evaluations. GA and DE use ``None``.
``temperature=None`` selects ``default_mc_temperature``; ``energy_range=None``
selects ``default_energy_range`` for all searches. Their Vina values remain
1.2 and 3.0. Explicit arguments override these scorer defaults. A box is
required for ``optimize()`` and ``dock()``, but not ordinary scoring.

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
