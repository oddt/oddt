Molecules and molecular arrays
==============================

ODDT exposes a common interface over RDKit and Open Babel. Select the backend
before importing ODDT, as described in :doc:`../installation`. For details,
see :mod:`oddt.toolkits.rdk` and :mod:`oddt.toolkits.ob`.

The interface follows the Pybel/Cinfony style, with additional handles for
numerical work: NumPy coordinates, indexable atom and bond collections,
residue access, and cached molecular arrays. Existing Pybel-style workflows
provide a useful starting point; check the selected backend's API for the
methods and file formats it supports.

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

   for molecule in oddt.toolkit.readfile('sdf', str(ligand_path)):
       print(molecule.title)

Use ``list(oddt.toolkit.readfile('sdf', filename))`` when you deliberately
want to keep all records in memory.

A molecule can also be constructed from a string:

.. code-block:: python

   benzene = oddt.toolkit.readstring('smi', 'c1ccccc1 benzene')
   print(benzene.title)
   assert len(benzene.atoms) == 6

SMILES defines molecular connectivity. Use a structure containing the bound
3D coordinates for contact calculations. Generating a conformer from SMILES
does not place that molecule in a protein's binding site.

Lazy loading and process workers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For supported formats, ``lazy=True`` defers conversion of a text record to a
backend molecule until a property requiring that molecule is accessed. Both
backends support this for SDF, MOL2, and PDB. The reader still reads the file;
parsing is deferred, and no background worker is started automatically.

.. code-block:: python

   import pickle

   lazy_ligand = next(oddt.toolkit.readfile(
       'sdf', str(ligand_path), lazy=True
   ))
   restored_ligand = pickle.loads(pickle.dumps(lazy_ligand))
   assert len(restored_ligand.atoms) == len(ligand.atoms)

ODDT molecules support pickling, including the Open Babel wrapper. A parsed
Open Babel molecule is serialized through MOL2; an unparsed lazy molecule
retains its source record. This lets process workers receive lazy records and
perform parsing and analysis locally. Accessing ``atoms``, ``coords``, or
``title`` before dispatch can trigger parsing in the parent process.
Lazy loading is opt-in (``lazy=False`` by default), and support depends on
the backend and input format.

For large libraries, stream records into bounded batches. Send only the
coordinates and features needed by a numerical worker when that avoids
serializing a full molecule and its caches. Initialize a fixed receptor once
per worker, and account for process startup and data transfer when measuring
parallel performance.

Atoms, bonds, and residues
--------------------------

Use explicit index conventions when moving between ODDT arrays and backend
objects:

.. code-block:: python

   first_atom = next(iter(ligand.atoms))
   print(first_atom.idx0, first_atom.idx1, first_atom.atomicnum)
   assert first_atom.idx1 == first_atom.idx0 + 1

   assert ligand.atoms[0].idx0 == first_atom.idx0
   for atom in ligand.atoms:
       print(atom.idx0, atom.atomicnum)

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

   for residue in protein.residues:
       for atom in residue.atoms:
           assert atom.idx0 >= 0

Atom and bond collections support iteration and indexing without first
building a Python list of every wrapped object. Residues expose their atoms
through ``residue.atoms``.

.. _molecular-arrays:
.. _numpy-dictionaries:

NumPy dictionaries for high-performance analysis
------------------------------------------------

Despite their names, ``atom_dict``, ``ring_dict``, and ``res_dict``
are NumPy structured arrays: each row describes one entity, and each named
field contains a typed scalar or fixed-size array. They provide a common
representation across the two chemistry backends. Bonds are accessed through
``molecule.bonds``; there is no corresponding ``bond_dict``.

The first access builds the molecular arrays and perceives features such as
donors, acceptors, and aromatic rings. ODDT caches the result on the molecule
so subsequent calculations can reuse it. This moves backend-specific work out
of repeated numerical calculations. Prepare the molecule before building its
arrays, and reuse a prepared protein across ligand poses.

Select a named field to get an array view, or apply a Boolean mask to select
records:

.. code-block:: python

   import numpy as np

   atoms = ligand.atom_dict
   assert atoms is ligand.atom_dict
   assert np.shares_memory(atoms['coords'], atoms)
   acceptors = atoms[atoms['isacceptor']]
   heavy_atoms = atoms[atoms['atomicnum'] > 1]
   print(acceptors['id'])
   assert heavy_atoms['coords'].shape[1] == 3
   assert np.isfinite(heavy_atoms['coords']).all()

.. _atom-dict-guide:

``atom_dict``
~~~~~~~~~~~~~~~~

The current schema is below. Shapes are **per record**: for ``n`` atoms,
``atoms['coords']`` has shape ``(n, 3)``. ``U5`` and ``U3`` are Unicode
strings of up to five and three characters.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Fields
     - dtype / shape
     - Meaning
   * - ``id``
     - ``uint32``
     - Zero-based atom ID.
   * - ``coords``
     - ``float32 (3,)``
     - Cartesian coordinates in Å.
   * - ``radius``, ``charge``
     - ``float32`` each
     - Atomic radius and partial charge.
   * - ``atomicnum``, ``hybridization``, ``formalcharge``
     - ``int8`` each
     - Atomic number, hybridization, and formal charge.
   * - ``atomtype``
     - ``U5``
     - Sybyl-style atom type, perceived by the backend.
   * - ``numhs``
     - ``uint8``
     - Number of attached hydrogens.
   * - ``neighbors_id``
     - ``int16 (6,)``
     - IDs in the stored neighbor slots.
   * - ``neighbors``
     - ``float32 (6, 3)``
     - Neighbor coordinates; unused slots contain NaNs.
   * - ``resid``, ``resnum``
     - ``int16`` each
     - Internal residue ID and structure residue number.
   * - ``resname``
     - ``U3``
     - Three-character residue name.
   * - ``isbackbone``
     - ``bool``
     - Backbone membership.
   * - ``isacceptor``, ``isdonor``, ``isdonorh``
     - ``bool`` each
     - Hydrogen-bond feature flags.
   * - ``ishydrophobe``, ``isaromatic``, ``ishalogen``, ``ismetal``
     - ``bool`` each
     - Hydrophobic, aromatic, halogen, and metal flags.
   * - ``isplus``, ``isminus``
     - ``bool`` each
     - Positively/negatively ionizable-group flags.
   * - ``isalpha``, ``isbeta``
     - ``bool`` each
     - Alpha-helix and beta-strand membership.

``resnum`` can repeat across chains. Use ``resid`` to group records
within a loaded protein, and retain a separate chain/residue mapping when
comparing independently prepared structures.
Feature flags reflect the backend's perception. In particular, ``isplus``
and ``isminus`` describe ionizable groups; use ``formalcharge`` for an
explicit formal-charge selection. ``charge`` is a partial charge and requires
an appropriate charge model for calculations that depend on it.

.. _ring-dict-guide:

``ring_dict``
~~~~~~~~~~~~~~~~

Each record describes an aromatic ring. Interaction functions use these
geometric records for π interactions.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Fields
     - dtype / shape
     - Meaning
   * - ``centroid``, ``vector``
     - ``float32 (3,)`` each
     - Ring center and normal vector (not necessarily unit length).
   * - ``resid``, ``resnum``
     - ``int16`` each
     - Internal residue ID and structure residue number.
   * - ``resname``
     - ``U3``
     - Residue name.
   * - ``isalpha``, ``isbeta``
     - ``bool`` each
     - Secondary-structure flags.

.. _res-dict-guide:

``res_dict``
~~~~~~~~~~~~~~

This array is built around amino-acid backbone information; it need not
contain every hetero residue present in ``atom_dict``. Set
``molecule.protein = True`` before requesting it.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Fields
     - dtype / shape
     - Meaning
   * - ``id``, ``resnum``
     - ``int16`` each
     - Internal residue ID and structure residue number.
   * - ``resname``
     - ``U3``
     - Residue name.
   * - ``N``, ``CA``, ``C``, ``O``
     - ``float32 (3,)`` each
     - Backbone atom coordinates in Å.
   * - ``isalpha``, ``isbeta``
     - ``bool`` each
     - Secondary-structure flags.

Inspect ``array.dtype.names``, ``array.dtype.fields``, and ``array.shape``
when constructing a numerical pipeline, especially when reading a different
ODDT version. For example:

.. code-block:: python

   print(atoms.dtype.names)
   assert atoms.dtype.fields['coords'][0].shape == (3,)
   neighbor_valid = np.isfinite(atoms['neighbors']).all(axis=-1)
   assert neighbor_valid.shape == atoms['neighbors_id'].shape

Select and reduce without Python atom loops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine feature masks with ``&``, ``|``, and ``~``. Parenthesize comparisons;
Python's ``and`` and ``or`` do not combine array masks element by element.

.. code-block:: python

   polar_mask = (
       (atoms['atomicnum'] > 1)
       & (atoms['isacceptor'] | atoms['isdonor'])
   )
   polar_atom_ids = atoms['id'][polar_mask]
   print('Polar heavy atoms:', np.count_nonzero(polar_mask))

   center = heavy_atoms['coords'].mean(axis=0, dtype=np.float64)
   offsets = heavy_atoms['coords'] - center
   squared_radii = np.einsum('ij,ij->i', offsets, offsets)
   radius_of_gyration = np.sqrt(squared_radii.mean())
   assert np.isfinite(radius_of_gyration)

Here the ``(3,)`` center broadcasts over ``(n, 3)`` coordinates. This computes
an unweighted geometric radius of gyration; mass weighting would require
atomic masses. Masks, arithmetic, and reductions operate on numerical arrays
without repeatedly constructing Python atom wrappers. ``np.vectorize`` still
calls a Python function element by element; use array operations for this
kind of calculation.

Calculate protein–ligand distances in bulk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build coordinate matrices once for repeated numerical work. A field view
from a structured array is usually strided because other fields occur
between successive atoms. Converting just the required coordinates to a
contiguous matrix can help numerical libraries and avoids copying every field.

.. code-block:: python

   from scipy.spatial.distance import cdist

   protein_heavy = protein.atom_dict[protein.atom_dict['atomicnum'] > 1]
   protein_xyz = np.ascontiguousarray(
       protein_heavy['coords'], dtype=np.float64
   )
   ligand_xyz = np.ascontiguousarray(
       heavy_atoms['coords'], dtype=np.float64
   )
   assert protein_xyz.flags.c_contiguous
   cutoff = 4.0  # Å
   distance_squared = cdist(protein_xyz, ligand_xyz, metric='sqeuclidean')
   protein_rows, ligand_rows = np.nonzero(distance_squared <= cutoff**2)
   contact_distances = np.sqrt(distance_squared[protein_rows, ligand_rows])
   protein_atom_ids = protein_heavy['id'][protein_rows]
   ligand_atom_ids = heavy_atoms['id'][ligand_rows]
   assert np.all(contact_distances <= cutoff)

``cdist`` calculates all atom-pair distances in compiled code. Comparing
squared distances avoids square roots for pairs that fail the cutoff. Rows
index the **selected arrays**, so map through their ``id`` fields to report
original atom IDs. These are distance contacts; hydrogen bonds and other
chemical interactions also need feature and geometry rules from
:doc:`interactions`.

Aggregate contacts by residue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Map internal residue IDs to compact group indices once, then use NumPy
reductions to build counts and minimum distances:

.. code-block:: python

   residue_ids, first_rows, groups = np.unique(
       protein_heavy['resid'], return_index=True, return_inverse=True
   )
   pair_counts = np.bincount(
       groups[protein_rows], minlength=len(residue_ids)
   )
   minimum_squared = np.full(len(residue_ids), np.inf)
   np.minimum.at(minimum_squared, groups, distance_squared.min(axis=1))
   labels = protein_heavy[first_rows]
   for row in np.flatnonzero(pair_counts):
       print(
           int(residue_ids[row]), str(labels['resname'][row]),
           int(labels['resnum'][row]), int(pair_counts[row]),
           float(np.sqrt(minimum_squared[row])),
       )
   assert pair_counts.sum() == len(contact_distances)

The count is the number of atom pairs, so one protein atom can contribute
several times. The result arrays retain all residues; the report prints only
those with contacts. The only Python loop formats the report; grouping and
aggregation use numerical kernels. ``np.minimum.at`` handles repeated group
indices correctly, unlike an assignment through repeated advanced indices.

Bound memory for large systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An ``n × m`` float64 distance matrix occupies ``8 × n × m`` bytes, before
masks and results. Broadcasting ``x[:, None, :] - y[None, :, :]`` also creates
an ``n × m × 3`` displacement array. For large receptors or many poses,
process a block at a time and retain only the required summary:

.. code-block:: python

   def contact_counts_in_blocks(receptor_xyz, pose_xyz, cutoff=4.0,
                               block_size=256):
       counts = np.zeros(len(receptor_xyz), dtype=np.int64)
       for start in range(0, len(receptor_xyz), block_size):
           stop = start + block_size
           block = cdist(receptor_xyz[start:stop], pose_xyz,
                         metric='sqeuclidean')
           counts[start:stop] = np.count_nonzero(block <= cutoff**2, axis=1)
       return counts

   blocked_counts = contact_counts_in_blocks(protein_xyz, ligand_xyz)
   assert np.array_equal(
       blocked_counts,
       np.bincount(protein_rows, minlength=len(protein_xyz)),
   )

The distance working set scales with ``block_size × m`` instead of ``n × m``.
The small example above retains the earlier dense matrix to check agreement;
a production pipeline can use the block calculation alone. Blocking bounds
temporary memory while still evaluating every atom pair.

Use a spatial index when only nearby atoms are needed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A fixed protein can be indexed once and queried for each ligand pose. A
nearest-neighbor query avoids materializing a full distance matrix:

.. code-block:: python

   from scipy.spatial import cKDTree

   receptor_tree = cKDTree(protein_xyz, copy_data=True)
   nearest_distances, nearest_rows = receptor_tree.query(ligand_xyz, k=1)
   assert np.allclose(
       nearest_distances, np.sqrt(distance_squared.min(axis=0))
   )
   nearest_protein_ids = protein_heavy['id'][nearest_rows]

   nearby_rows = receptor_tree.query_ball_point(ligand_xyz, r=cutoff)
   assert sum(map(len, nearby_rows)) == len(contact_distances)

``nearest_rows`` indexes the protein coordinate matrix; ``nearby_rows``
contains all protein neighbors for each ligand atom. Radius-query output
still grows with the number of matching pairs. Reuse the tree while the
protein coordinates stay fixed, and rebuild it after a receptor geometry
change. Compare this approach with blocked distances on representative data;
tree construction and conversion costs matter for small systems.

Cache, layout, and parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The arrays are cached. Atom and ring arrays are read-only; make a copy for
analysis that modifies fields. Treat all cached arrays as analysis inputs.
Changing a copied array does not update the backend molecule.

* Named-field access and basic slices return views. Boolean indexing makes
  a copy. Selecting coordinates before applying a mask, as in
  ``atoms['coords'][mask]``, avoids copying all fields in each selected record.
* Keep small stored dtypes when appropriate; promote coordinates or
  accumulations when your calculation needs float64 arithmetic. Promotion
  cannot recover precision already lost in float32 storage. Narrow integer
  fields such as ``resid`` and ``neighbors_id`` also have finite ranges.
* ODDT's coordinate setter refreshes the coordinate and atom-array caches.
  If ring or residue geometry has already been cached, use a fresh molecule
  for the transformed structure before a geometry-dependent analysis.
  Previously saved array references remain snapshots of the old geometry.
* Build and reuse arrays outside the timed numerical kernel. Measure parsing,
  feature perception, conversion, and computation separately before choosing
  an optimization. Small systems may be dominated by setup work.
* Vectorization reduces Python overhead; it does not automatically distribute
  a calculation across CPU cores or GPUs. For independent molecules, batch
  work across processes and avoid copying a large receptor per ligand. For
  Numba, Cython, or accelerator kernels, extract the supported numerical
  fields into appropriately typed matrices and include transfer costs in
  performance measurements.

See NumPy's `structured arrays
<https://numpy.org/doc/stable/user/basics.rec.html>`_ and `broadcasting
<https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ documentation
for memory-layout details, and SciPy's
`cKDTree.query <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html>`_
for spatial-query options.

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
