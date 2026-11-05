Internal genetic docking
========================

:class:`oddt.docking.vina_docking` searches ligand translation, orientation and
rotatable bonds against a rigid receptor. It combines a genetic algorithm with
SciPy's L-BFGS-B local minimizer. It requires NumPy, SciPy and an ODDT molecular
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
   poses = engine.dock(population_size=24, generations=8, maxiter=60, seed=42)
   print("RMSD:", rmsd(reference, poses[0]))
   print("Affinity:", poses[0].data["vina_affinity"])
   poses[0].write("sdf", "docked.sdf", overwrite=True)

``box`` contains the lower and upper Cartesian corners in Angstroms.
All ligand heavy atoms must fit inside it. Returned poses are independent
molecule copies; the input receptor and ligand are preserved, including
explicit hydrogens, which move with their fragments.

Increase ``population_size``, ``generations`` or ``maxiter`` for more extensive
searching. A fixed ``seed`` makes runs reproducible within the same numerical
environment. Multiple seeds help assess search variability.

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

``engine.history_`` records the best restrained objective per generation.
``engine.results_`` holds SciPy optimizer results for the returned poses,
including iteration counts, termination status and pose genes. A local search
that reaches its iteration limit may still produce an improved candidate.
Use ``engine.optimize(maxiter=100)`` to refine only the supplied pose.

4LB3 validation
---------------

The following single-seed measurements used the same 16 Angstrom cubic box,
seed 42, and a serial search on the development machine. The genetic search
used a population of 24, eight generations and at most 60 L-BFGS-B iterations
per individual. AutoDock Vina 1.2.7 used one CPU and exhaustiveness 8; its time
includes map generation and file handling. These are different search budgets.
RMSD uses direct heavy-atom matching to the crystal pose, without alignment.

=============================== =========== ===========
Engine                          Time        Top RMSD
=============================== =========== ===========
Original genetic search         72 s        1.69 A
Revised search, Open Babel      about 20 s  0.62 A
Revised search, RDKit           about 20 s  0.32 A
AutoDock Vina 1.2.7             23 s        0.30 A
=============================== =========== ===========

The revision keeps conjugated bonds rigid and projects Cartesian forces onto
translation, rotation and torsion coordinates analytically. This replaces the
finite-difference pose derivatives. The regression test requires the top 4LB3
pose to be within 1 Angstrom of the crystal pose. Additional Open Babel runs
with seeds 1 and 7 gave 0.57 and 1.32 Angstrom top-pose RMSD, respectively;
the successful seed-42 result does not guarantee recovery in every run.

This engine uses ODDT atom typing and direct pair calculations. Its genetic
search, scores and rankings can differ from AutoDock Vina. The 4LB3 regression
checks one successful redocking case. An exploratory run of the original
implementation on 4BFQ instead ranked a pose about 7 Angstroms from the
crystal pose first, despite improving the score. Broader benchmarks are needed
to assess accuracy across targets, flexible ligands and protonation states.
