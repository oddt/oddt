Molecules and molecular arrays
==============================

ODDT exposes a common interface over RDKit and Open Babel. Select the backend
before importing ODDT, as described in :doc:`../installation`. For details,
see :mod:`oddt.toolkits.rdk` and :mod:`oddt.toolkits.ob`.

Read a structure
----------------

.. code-block:: python

   from pathlib import Path
   import oddt

   data_dir = Path('tests/data/pdb')
   ligand_path = data_dir / '4lb3_ligand.sdf'
   ligand = next(oddt.toolkit.readfile('sdf', str(ligand_path)))
   print(ligand.title)
   print(ligand.coords.shape)

``readfile`` returns an iterator, so ``next`` selects its first molecule.
Iterate over the reader to process a multi-record SDF without collecting the
whole library in memory:

.. code-block:: python

   titles = [
       molecule.title
       for molecule in oddt.toolkit.readfile('sdf', str(ligand_path))
   ]
   print(titles)

A molecule can also be constructed from a string:

.. code-block:: python

   benzene = oddt.toolkit.readstring('smi', 'c1ccccc1 benzene')
   print(benzene.title)
   assert len(benzene.atoms) == 6

SMILES defines molecular connectivity. Use a structure containing the bound
3D coordinates for contact calculations. Generating a conformer from SMILES
does not place that molecule in a protein's binding site.

Atoms, bonds, and residues
--------------------------

Use explicit index conventions when moving between ODDT arrays and backend
objects:

.. code-block:: python

   first_atom = next(iter(ligand.atoms))
   print(first_atom.idx0, first_atom.idx1, first_atom.atomicnum)
   assert first_atom.idx1 == first_atom.idx0 + 1

   for bond in ligand.bonds:
       print([atom.idx0 for atom in bond.atoms], bond.order)

``atom.idx0`` is zero-based; ``atom.idx1`` is one-based. The legacy
``atom.idx`` property is deprecated. The ``id`` field in ``atom_dict``
uses the zero-based convention. These IDs belong to the current molecule;
do not assume they are PDB serial numbers or portable across backends.

Mark protein structures before requesting residue-aware data:

.. code-block:: python

   protein = next(oddt.toolkit.readfile(
       'pdb', str(data_dir / '4lb3_pocket.pdb')
   ))
   protein.protein = True
   print([(residue.name, len(residue.atoms)) for residue in protein.residues][:5])

Molecular arrays
----------------

Despite their names, ``atom_dict``, ``ring_dict``, and ``res_dict``
are NumPy structured arrays. Select a named field to get a vector, or apply a
Boolean mask to select records:

.. code-block:: python

   import numpy as np

   atoms = ligand.atom_dict
   acceptors = atoms[atoms['isacceptor']]
   heavy_atoms = atoms[atoms['atomicnum'] != 1]
   print(acceptors['id'])
   assert heavy_atoms['coords'].shape[1] == 3
   assert np.isfinite(heavy_atoms['coords']).all()

.. _atom-dict-guide:

``atom_dict``
~~~~~~~~~~~~~~~~

Useful fields include:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Fields
     - Meaning
   * - ``id``, ``atomicnum``, ``atomtype``
     - Zero-based atom ID, atomic number, and backend-perceived atom type.
   * - ``coords``, ``radius``, ``charge``
     - Coordinates in Å, atomic radius, and partial charge.
   * - ``formalcharge``, ``numhs``, ``hybridization``
     - Formal charge, attached hydrogen count, and hybridization.
   * - ``neighbors_id``, ``neighbors``
     - Neighbor IDs and coordinates; unused coordinate slots can contain NaNs.
   * - ``resid``, ``resnum``, ``resname``, ``isbackbone``
     - Internal residue ID, structure residue number, name, and backbone flag.
   * - ``isacceptor``, ``isdonor``, ``isdonorh``
     - Hydrogen-bond feature flags.
   * - ``ishydrophobe``, ``isaromatic``, ``ishalogen``, ``ismetal``
     - Hydrophobic, aromatic, halogen, and metal flags.
   * - ``isplus``, ``isminus``, ``isalpha``, ``isbeta``
     - Ionizable-group and secondary-structure flags.

``resnum`` can repeat across chains. Use ``resid`` to group records
within a loaded protein, and retain a separate chain/residue mapping when
comparing independently prepared structures.

.. _ring-dict-guide:

``ring_dict``
~~~~~~~~~~~~~~~~

Each aromatic ring record provides ``centroid`` and ``vector`` (a ring
normal), plus residue and secondary-structure fields. Interaction functions
use these geometric records for π interactions.

.. _res-dict-guide:

``res_dict``
~~~~~~~~~~~~~~

Protein residue records include ``id``, ``resnum``, ``resname``,
backbone coordinates ``N``, ``CA``, ``C``, ``O``, and secondary
structure flags. This array is built around amino-acid backbone information;
it need not contain every hetero residue present in ``atom_dict``.

The arrays are cached. Atom and ring arrays are read-only; make a copy for
analysis that modifies fields. Use ODDT's molecule methods and coordinate
setter when changing a molecule so its caches are refreshed.

Write or copy a molecule
------------------------

.. code-block:: python

   ligand.write('sdf', 'ligand-copy.sdf', overwrite=True)
   copied = next(oddt.toolkit.readfile('sdf', 'ligand-copy.sdf'))
   assert len(copied.atoms) == len(ligand.atoms)

   translated = ligand.clone
   translated.coords = translated.coords + np.array([1.0, 0.0, 0.0])
   assert np.allclose(translated.coords, ligand.coords + [1.0, 0.0, 0.0])

Translating a ligand changes its pose relative to an unchanged protein.
See :doc:`interactions` for preparation choices and :doc:`fingerprints`
for comparing those poses.
