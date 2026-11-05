Protein–ligand interactions
===========================

:mod:`oddt.interactions` detects atom or ring pairs using molecular feature
flags, distance cutoffs, and, for directional interactions, angular filters.
It returns the participating records so you can inspect individual contacts,
aggregate them by residue, or build your own report.

For a compact vector representation of a binding mode, continue with
:doc:`fingerprints`.

.. _interaction-atlas:

Explore interactions in 3D
--------------------------

Choose an interaction type, then select **Load interactive 3D**. Each type
uses a **different experimental PDB structure**. Drag to rotate, scroll or
pinch to zoom, and select a contact below the view to focus its atoms or
rings. Rotation and zoom buttons also work with a keyboard.

The ligand is shown as thick sticks (or a sphere for a single ion), the
protein pocket as thin lines, and contacting residues as sticks. Colored
dashes mark ODDT-classified contacts. A dot marks a ring centroid;
ring-contact distances are measured from that centroid. Enable **Pocket
surface** to add a translucent van der Waals envelope of the displayed
protein atoms. The selected contact has atom/residue labels and a distance.

.. raw:: html

   <div class="oddt-interaction-atlas"
        data-base-url="../_static/interactions/"
        data-library-url="../_static/vendor/3Dmol-2.5.5.min.js">
     <p>Enable JavaScript and WebGL to explore the structures interactively.
        The examples and downloadable data are listed below.</p>
   </div>
   <script defer src="../_static/interaction-viewer.js"></script>

.. list-table:: Eight types, eight structures
   :header-rows: 1
   :widths: 25 15 60

   * - Interaction
     - PDB entry
     - Complex / downloadable contact data
   * - Hydrogen bonds
     - `1HSG <https://www.rcsb.org/structure/1HSG>`_
     - HIV protease–indinavir
       (:download:`contacts <../_static/interactions/hydrogen-bond/scene.json>`).
   * - Hydrophobic contacts
     - `4BFQ <https://www.rcsb.org/structure/4BFQ>`_
     - Acetylcholine-binding protein–083
       (:download:`contacts <../_static/interactions/hydrophobic/scene.json>`).
   * - Halogen bonds
     - `1P5E <https://www.rcsb.org/structure/1P5E>`_
     - CDK2–tetrabromobenzotriazole
       (:download:`contacts <../_static/interactions/halogen-bond/scene.json>`).
   * - Salt bridges
     - `3PTB <https://www.rcsb.org/structure/3PTB>`_
     - Trypsin–benzamidine
       (:download:`contacts <../_static/interactions/salt-bridge/scene.json>`).
   * - π stacking
     - `1ACJ <https://www.rcsb.org/structure/1ACJ>`_
     - Acetylcholinesterase–tacrine
       (:download:`contacts <../_static/interactions/pi-stacking/scene.json>`).
   * - π–cation contacts
     - `1ACL <https://www.rcsb.org/structure/1ACL>`_
     - Acetylcholinesterase–decamethonium
       (:download:`contacts <../_static/interactions/pi-cation/scene.json>`).
   * - Acceptor–metal coordination
     - `1HFC <https://www.rcsb.org/structure/1HFC>`_
     - Fibroblast collagenase–hydroxamate inhibitor
       (:download:`contacts <../_static/interactions/acceptor-metal/scene.json>`).
   * - π–metal contacts
     - `1LPI <https://www.rcsb.org/structure/1LPI>`_
     - Lysozyme–sodium, with the protein supplying the aromatic ring
       (:download:`contacts <../_static/interactions/pi-metal/scene.json>`).

The examples retain deposited heavy-atom coordinates and complete protein
residues within 6 Å of the selected ligand. They use the first model and
blank/A alternate conformations. Ligand connectivity, hydrogen counts, and
formal charges come from the PDB Chemical Component Dictionary; polar
hydrogens are added with RDKit through ODDT. Other hetero compounds, waters,
and crystallographic symmetry mates are omitted; the coordination example
retains nearby zinc/calcium ions. No coordinates are minimized or docked.

Displayed directional contacts pass their strict angular masks. Hydrogen
bonds are restricted to N/O pairs; hydrophobic contacts are summarized by
the closest pair per protein residue. Salt bridges use ionizable-group
flags, while π–cation contacts require an explicit positive formal charge.
Fused aromatic rings can contribute more than one centroid contact. These
are numbered within each residue or ligand so they can be distinguished.
Repeated atom pairs are displayed once. These are examples of ODDT's
classifications, not measurements of bond energies.

The viewer's downloads include the pocket PDB, bound ligand SDF, contacts,
detector settings, source URLs/checksums, and software versions. Rendering
and data are bundled with the docs; no live PDB or CDN connection is needed.
The Python walkthrough below continues to use 4LB3 as one shared example.

Prepare a bound complex
-----------------------

These examples use the bundled 4LB3 pocket and ligand. Run the blocks in order
from the repository root, as described in :doc:`index`:

.. code-block:: python

   from pathlib import Path
   import numpy as np
   import oddt
   from oddt.interactions import hbonds

   data_dir = Path('tests/data/pdb')
   protein = next(oddt.toolkit.readfile(
       'pdb', str(data_dir / '4lb3_pocket.pdb')
   ))
   protein.protein = True
   protein.addh(only_polar=True)

   ligand_path = data_dir / '4lb3_ligand.sdf'
   ligand = next(oddt.toolkit.readfile('sdf', str(ligand_path)))
   ligand.addh(only_polar=True)

   assert np.isfinite(protein.coords).all()
   assert np.isfinite(ligand.coords).all()

The protein and ligand must have 3D coordinates in the same frame, with
distances expressed in Å. A separately generated ligand conformer is not a
bound pose. Setting ``protein.protein = True`` enables residue-aware
perception; do this before accessing its molecular arrays.

Adding hydrogens does not choose the correct protonation or tautomer for a
particular pH. Prepare those states, bond orders, missing atoms, and alternate
conformations before analysis. Decide whether to retain waters, metals,
cofactors, and crystallographic partners in the supplied molecules: these
functions analyze the records you give them.

Hydrogen bonds: candidates and strict contacts
----------------------------------------------

.. code-block:: python

   protein_atoms, ligand_atoms, strict = hbonds(
       protein, ligand,
       cutoff=3.5,
       tolerance=30,
       mol1_exact=True,
       mol2_exact=True,
   )
   print('Distance-qualified candidates:', len(protein_atoms))
   print('Strict hydrogen bonds:', int(strict.sum()))
   assert len(protein_atoms) == len(ligand_atoms) == len(strict)
   assert strict.any()

The three arrays are aligned: row ``i`` in each refers to the same contact.
The first two arrays contain atom records, rather than atom objects.
``strict`` is a Boolean mask that says whether both angular checks pass.
A false entry still meets the distance and feature-selection criteria.

``hbonds(protein, ligand)`` searches both donor/acceptor directions but
keeps the returned atom arrays in the input molecule order. In contrast,
:func:`~oddt.interactions.hbond_acceptor_donor` is directional: its first
argument supplies acceptors and its second supplies donors.

The ``mol1_exact`` and ``mol2_exact`` flags require donor atoms in the
respective molecules to have attached hydrogens. They do not require explicit
formal charges. With the default false values, ODDT can include atoms perceived
as possible donors without requiring the prepared state to carry a hydrogen.

Select strict contacts before reporting distances:

.. code-block:: python

   strict_protein = protein_atoms[strict]
   strict_ligand = ligand_atoms[strict]
   distances = np.linalg.norm(
       strict_protein['coords'] - strict_ligand['coords'], axis=1
   )
   for p_atom, l_atom, distance in zip(
       strict_protein, strict_ligand, distances
   ):
       print(
           str(p_atom['resname']), int(p_atom['resnum']),
           'protein atom', int(p_atom['id']),
           'ligand atom', int(l_atom['id']),
           f'{distance:.2f} Å',
       )

Strict is a geometry classification under the chosen feature definitions.
It is not a binding-energy calculation or independent evidence of an
interaction. Inspect atom types and prepared chemical states when interpreting
unusual acceptors or donors; those feature assignments can differ by backend.

Count contacts by residue
-------------------------

An atom-pair count is different from a count of interacting residues. One
residue can contribute several atom pairs:

.. code-block:: python

   residue_ids, pair_counts = np.unique(
       strict_protein['resid'], return_counts=True
   )
   for residue_id, count in zip(residue_ids, pair_counts):
       atom = strict_protein[strict_protein['resid'] == residue_id][0]
       print(
           int(residue_id), str(atom['resname']),
           int(atom['resnum']), int(count),
       )

Group by ``resid`` within the loaded molecule, rather than only by
``resnum``, because residue numbers can repeat across chains. Retain a
chain-aware mapping when combining reports from independently loaded proteins.
See :doc:`molecules` for molecular-array fields and atom-index conventions.

Export a contact table
----------------------

.. code-block:: python

   import csv

   rows = [
       {
           'protein_resid': int(p_atom['resid']),
           'resname': str(p_atom['resname']),
           'resnum': int(p_atom['resnum']),
           'protein_atom_id': int(p_atom['id']),
           'ligand_atom_id': int(l_atom['id']),
           'distance_angstrom': round(float(distance), 3),
       }
       for p_atom, l_atom, distance in zip(
           strict_protein, strict_ligand, distances
       )
   ]
   fields = [
       'protein_resid', 'resname', 'resnum',
       'protein_atom_id', 'ligand_atom_id', 'distance_angstrom',
   ]
   with open('hydrogen-bonds.csv', 'w', newline='') as output:
       writer = csv.DictWriter(output, fieldnames=fields)
       writer.writeheader()
       writer.writerows(rows)
   assert len(rows) == int(strict.sum())

This writes a header even when there are no strict contacts. Atom IDs in this
table are ODDT's zero-based IDs for the prepared molecules, not PDB serials.

Other interaction types
-----------------------

Default cutoffs are summarized below. Atom-pair functions return two aligned
arrays; directional functions must be called with the roles shown.

.. list-table::
   :header-rows: 1
   :widths: 26 12 24 38

   * - Function
     - Cutoff (Å)
     - Returns
     - Interpretation
   * - :func:`~oddt.interactions.hbonds`
     - 3.5
     - Two atom arrays, strict mask
     - Both hydrogen-bond directions.
   * - :func:`~oddt.interactions.halogenbonds`
     - 4.0
     - Two atom arrays, strict mask
     - Both acceptor/halogen directions.
   * - :func:`~oddt.interactions.hydrophobic_contacts`
     - 4.0
     - Two atom arrays
     - Contacts between perceived hydrophobic atoms.
   * - :func:`~oddt.interactions.salt_bridges`
     - 4.0
     - Two atom arrays
     - Oppositely ionizable groups; optional exact-charge filters.
   * - :func:`~oddt.interactions.pi_stacking`
     - 5.0
     - Two ring arrays, parallel mask, perpendicular mask
     - Ring-centroid distance and ring geometry.
   * - :func:`~oddt.interactions.pi_cation`
     - 5.0
     - Ring array, cation array, strict mask
     - Rings in argument 1, cations in argument 2.
   * - :func:`~oddt.interactions.acceptor_metal`
     - 4.0
     - Acceptor array, metal array, strict mask
     - Acceptors in argument 1, metals in argument 2.
   * - :func:`~oddt.interactions.pi_metal`
     - 5.0
     - Ring array, metal array, strict mask
     - Rings in argument 1, metals in argument 2.

Directional geometry functions use a default ``tolerance=30`` degrees.
The exact angular rules depend on the interaction; the API reference describes
their individual parameters.

Hydrophobic contacts
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from oddt.interactions import hydrophobic_contacts

   hydrophobic_protein, hydrophobic_ligand = hydrophobic_contacts(
       protein, ligand, cutoff=4.0
   )
   print('Hydrophobic atom pairs:', len(hydrophobic_protein))
   print('Interacting residues:', np.unique(
       hydrophobic_protein['resid']
   ).size)

There is no additional strict mask. To make a less crowded map, select the
closest pair per residue while retaining the complete results separately:

.. code-block:: python

   closest = {}
   for p_atom, l_atom in zip(hydrophobic_protein, hydrophobic_ligand):
       distance = float(np.linalg.norm(p_atom['coords'] - l_atom['coords']))
       residue_id = int(p_atom['resid'])
       if residue_id not in closest or distance < closest[residue_id][0]:
           closest[residue_id] = (
               distance, int(p_atom['id']), int(l_atom['id'])
           )
   assert len(closest) <= len(hydrophobic_protein)

These representative pairs are a reporting choice; the unaggregated atom-pair
counts are what the residue interaction fingerprints use.

Halogen bonds and π stacking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from oddt.interactions import halogenbonds, pi_stacking

   p_halogen, l_halogen, halogen_strict = halogenbonds(protein, ligand)
   print('Strict halogen bonds:', int(halogen_strict.sum()))

   p_rings, l_rings, parallel, perpendicular = pi_stacking(protein, ligand)
   print('Parallel stacking:', int(parallel.sum()))
   print('Perpendicular stacking:', int(perpendicular.sum()))
   assert len(p_rings) == len(l_rings) == len(parallel) == len(perpendicular)

π-stacking results contain ring records with ``centroid`` and
``vector`` fields, not individual atom records. Use the appropriate
geometry mask before selecting those rings.

Salt bridges, cations, and metals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from oddt.interactions import (
       salt_bridges, pi_cation, acceptor_metal, pi_metal,
   )

   p_ions, l_ions = salt_bridges(
       protein, ligand, mol1_exact=True, mol2_exact=True
   )
   print('Pairs with explicit opposite formal charges:', len(p_ions))

   ligand_rings, protein_cations, cation_strict = pi_cation(
       ligand, protein, cation_exact=True
   )
   ligand_acceptors, protein_metals, metal_strict = acceptor_metal(
       ligand, protein
   )
   ligand_metal_rings, ring_metals, pi_metal_strict = pi_metal(
       ligand, protein
   )

Salt-bridge detection normally uses the ``isplus`` and ``isminus``
feature flags. The exact flags additionally require positive or negative
formal charges. They do not assign those charges. Likewise,
``cation_exact=True`` requires a positive formal charge for π–cation
contacts.

For protein rings contacting a ligand cation, call
``pi_cation(protein, ligand)`` separately. Reverse the arguments to the
metal functions if metals reside in the ligand-side molecule. The ligand in
this example need not produce any contacts for these categories.

Define a distance-only contact search
-------------------------------------

:func:`~oddt.interactions.close_contacts` accepts structured arrays directly.
It does not apply chemical feature selection for you:

.. code-block:: python

   from oddt.interactions import close_contacts

   p_heavy = protein.atom_dict[protein.atom_dict['atomicnum'] != 1]
   l_heavy = ligand.atom_dict[ligand.atom_dict['atomicnum'] != 1]
   p_close, l_close = close_contacts(
       p_heavy, l_heavy, cutoff=4.0, cutoff_low=0.0
   )
   assert len(p_close) == len(l_close)

The accepted interval is ``(cutoff_low, cutoff]``. A close atom pair alone
does not establish a hydrogen bond, halogen bond, or hydrophobic contact.

Process a pose library
----------------------

Reuse the prepared protein and stream ligands in the same coordinate frame:

.. code-block:: python

   pose_summary = []
   for pose in oddt.toolkit.readfile('sdf', str(ligand_path)):
       pose.addh(only_polar=True)
       _, _, pose_strict = hbonds(
           protein, pose, mol1_exact=True, mol2_exact=True
       )
       pose_summary.append((pose.title, int(pose_strict.sum())))
   assert pose_summary

Keep backend, preparation, cutoffs, and residue mapping fixed when comparing
poses. Save these settings alongside exported tables or fingerprints.
For periodic simulation data, make the complex whole and choose the intended
periodic image before calling these Cartesian-coordinate functions.
