Internal Monte Carlo docking
============================

:class:`oddt.docking.vina_docking` searches ligand translation, orientation and
rotatable bonds against a rigid receptor. It combines Vina-inspired Monte Carlo
basin hopping with SciPy's L-BFGS-B local minimizer. It requires NumPy, SciPy and an ODDT molecular
backend; no AutoDock Vina installation is needed.

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
   from oddt.docking import vina_docking
   from oddt.spatial import rmsd, rotate

   protein = next(oddt.toolkit.readfile("pdb", "tests/data/pdb/4lb3_pocket.pdb"))
   protein.protein = True
   reference = next(oddt.toolkit.readfile("sdf", "tests/data/pdb/4lb3_ligand.sdf"))
   reference.addh()
   heavy = reference.atom_dict["atomicnum"] != 1
   center = reference.coords[heavy].mean(axis=0)

   ligand = reference.clone
   ligand.coords = rotate(ligand.coords, 1.2, -0.7, 2.1) + [12, -7, 8]
   engine = vina_docking(protein, ligand, box=np.array([center - 8, center + 8]))
   poses = engine.dock(exhaustiveness=8, steps=250, maxiter=60, seed=42, num_modes=5)
   print("RMSD:", rmsd(reference, poses[0]))
   print("Affinity:", poses[0].data["vina_affinity"])
   poses[0].write("sdf", "docked.sdf", overwrite=True)

``box`` contains the lower and upper Cartesian corners in Angstroms.
All ligand heavy atoms must fit inside it. Returned poses are independent
molecule copies; the input receptor and ligand are preserved, including
explicit hydrogens, which move with their fragments.

Increase ``exhaustiveness``, ``steps`` or ``maxiter`` for more extensive
searching. A fixed ``seed`` makes runs reproducible within the same numerical
environment. Multiple seeds help assess search variability.

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

``engine.history_`` records the best uncapped restrained objective after each
proposal. ``engine.search_stats_`` reports accepted moves, accepted uphill
moves and uncapped refinements separately for each trajectory.
``engine.results_`` holds SciPy optimizer results for the returned poses,
including iteration counts, termination status and pose genes. A local search
that reaches its iteration limit may still produce an improved candidate.
Use ``engine.optimize(maxiter=100)`` to refine only the supplied pose.

4LB3 validation
---------------

The regression starts from a displaced and rotated ligand, randomizes all
permitted search coordinates, and uses a 16 Angstrom cubic box, seed 42,
eight trajectories, 250 proposals per trajectory, at most 60 L-BFGS-B
iterations per minimization, and up to five returned modes. It checks direct heavy-atom RMSD without alignment,
input preservation, box containment, pose clustering and fresh score metadata.

.. mc-validation-start

Both backends passed the 41 focused docking, benchmark and scoring tests.
The top-ranked 4LB3 results on an Apple M2 machine were:

.. list-table:: Monte Carlo redocking, seed 42
   :header-rows: 1

   * - Backend
     - Top RMSD
     - Docking time
   * - RDKit
     - 0.32 Angstrom
     - 235 s
   * - Open Babel
     - 0.56 Angstrom
     - 239 s

Times measure ``dock()`` only, excluding shared preparation and engine
construction. The regression retains its requirement of RMSD below 1 Angstrom.
The :download:`validation JSON <../_static/benchmarks/4lb3_monte_carlo.json>`
records measured values, settings, versions and source/input checksums.

.. mc-validation-end

The engine keeps conjugated bonds rigid and projects Cartesian forces onto
translation, rotation and torsion coordinates analytically. Its ODDT atom typing,
scores and rankings can differ from AutoDock Vina. One successful redocking
case does not establish accuracy across targets, ligands or protonation states.

See :doc:`docking-benchmarks` for a comparison on the six local PDBbind
complexes using the same settings across targets and three seeds per engine.
