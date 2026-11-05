Molecular and interaction fingerprints
======================================

:mod:`oddt.fingerprints` provides residue-based interaction counts, molecular
environment hashes, and protein–ligand contact-environment fingerprints.
Choose the representation according to what you want to compare.

Choose a fingerprint
---------------------

.. list-table::
   :header-rows: 1
   :widths: 15 30 30 25

   * - Function
     - Encodes
     - Representation
     - Typical use
   * - :func:`~oddt.fingerprints.InteractionFingerprint`
     - Eight interaction channels per protein residue.
     - Dense counts, length 8 × residue count.
     - Compare poses against a fixed residue mapping.
   * - :func:`~oddt.fingerprints.SimpleInteractionFingerprint`
     - The same channels grouped by residue type.
     - Dense counts, length 168.
     - Compare coarse interaction profiles without residue positions.
   * - :func:`~oddt.fingerprints.ECFP`
     - Circular environments in one molecule's graph.
     - Sparse indices by default; optional dense counts or presence bits.
     - Compare molecular connectivity without a protein or a bound pose.
   * - :func:`~oddt.fingerprints.PLEC`
     - Paired ligand/protein environments around close atom pairs.
     - Sparse indices by default; optional dense counts or presence bits.
     - Encode a protein–ligand pose using local chemical environments.
   * - :func:`~oddt.fingerprints.SPLIF`
     - Hashed contact environments and their coordinates.
     - Structured records.
     - Compare contact chemistry and spatial agreement together.

Except for ECFP, these fingerprints require bound 3D poses. Read
:doc:`interactions` for preparation, feature perception, and geometric filters.
Protein–ligand fingerprint functions take ``(ligand, protein)``, in that
order.

Load an example complex
-----------------------

Run the blocks in order from the repository root:

.. code-block:: python

   from pathlib import Path
   import numpy as np
   import oddt
   from oddt.fingerprints import (
       InteractionFingerprint, SimpleInteractionFingerprint,
       ECFP, PLEC, SPLIF, similarity_SPLIF, dice, tanimoto,
   )

   data_dir = Path('tests/data/pdb')
   protein_path = data_dir / '4lb3_pocket.pdb'
   ligand_path = data_dir / '4lb3_ligand.sdf'
   protein = next(oddt.toolkit.readfile('pdb', str(protein_path)))
   protein.protein = True
   protein.addh(only_polar=True)
   ligand = next(oddt.toolkit.readfile('sdf', str(ligand_path)))
   ligand.addh(only_polar=True)

   query_pose = ligand.clone
   query_pose.coords = query_pose.coords + np.array([0.5, 0.0, 0.0])

The shifted ligand is a second pose against the same unchanged protein. It is
used to illustrate comparisons, not as a docking or pose-optimization method.

IFP: inspect interaction counts by residue
------------------------------------------

.. code-block:: python

   ifp = InteractionFingerprint(ligand, protein, strict=True)
   query_ifp = InteractionFingerprint(query_pose, protein, strict=True)
   residue_order = np.unique(protein.atom_dict['resid'])
   matrix = ifp.reshape(len(residue_order), 8)
   assert ifp.size == len(residue_order) * 8
   assert np.issubdtype(ifp.dtype, np.unsignedinteger)

   for residue_id, counts in zip(residue_order, matrix):
       if counts.any():
           atom = protein.atom_dict[
               protein.atom_dict['resid'] == residue_id
           ][0]
           print(
               int(residue_id), str(atom['resname']),
               int(atom['resnum']), counts.tolist(),
           )

Rows follow the sorted unique ``resid`` values in the protein's atom
array. Do not infer their ordering from ``protein.res_dict``, which may
omit hetero residues. The eight columns have this fixed order:

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Column
     - Counted interaction
   * - 0
     - Hydrophobic atom pairs.
   * - 1
     - Parallel, face-to-face aromatic ring pairs.
   * - 2
     - Perpendicular, edge-to-face aromatic ring pairs.
   * - 3
     - Hydrogen bonds donated by the protein.
   * - 4
     - Hydrogen bonds accepted by the protein.
   * - 5
     - Salt bridges with a positively ionizable protein group.
   * - 6
     - Salt bridges with a negatively ionizable protein group.
   * - 7
     - Ligand acceptor–protein metal contacts.

These are counts, not Boolean bits: several atom or ring pairs can contribute
to one residue/channel. The output uses the smallest unsigned integer dtype
that holds its largest count. Convert explicitly if you want presence only:

.. code-block:: python

   ifp_presence = ifp > 0
   assert ifp_presence.dtype == np.bool_
   print('Count-aware Dice:', dice(ifp, query_ifp))
   print('Presence-only Tanimoto:', tanimoto(ifp, query_ifp))

With ``strict=True``, the hydrogen-bond and acceptor–metal channels apply
their angular masks. ``strict=False`` relaxes those channels to their
distance-qualified candidates; π-stacking channels still apply their geometry
masks. Cutoffs come from the underlying interaction functions and are not
parameters of IFP itself.

Halogen bonds, π–cation contacts, and π–metal contacts are not separate IFP
channels. Use :mod:`oddt.interactions` when you need those individual contacts.
Equal vector lengths alone do not establish comparable residue identities:
use the same prepared protein, or establish a common residue ordering first.

SIFP: aggregate by amino-acid type
----------------------------------

.. code-block:: python

   sifp = SimpleInteractionFingerprint(ligand, protein)
   query_sifp = SimpleInteractionFingerprint(query_pose, protein)
   amino_acids = [
       '', 'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN',
       'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
       'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
   ]
   type_counts = sifp.reshape(21, 8)
   assert sifp.shape == (168,)
   assert len(amino_acids) == len(type_counts)
   print('SIFP Dice:', dice(sifp, query_sifp))

SIFP uses the same eight channels, summing over residues with the same name.
The first row, labeled ``''``, collects residue names outside the twenty
standard amino acids, including cofactors. This fixed shape permits a coarse
comparison across proteins, but loses residue position and sequence context:
two different binding modes can have the same SIFP.

ECFP: compare molecular connectivity
-------------------------------------

ECFP works without a protein or 3D coordinates:

.. code-block:: python

   aspirin = oddt.toolkit.readstring('smi', 'CC(=O)Oc1ccccc1C(=O)O')
   salicylate = oddt.toolkit.readstring('smi', 'Oc1ccccc1C(=O)O')
   ecfp_a = ECFP(aspirin, depth=2, size=4096)
   ecfp_b = ECFP(salicylate, depth=2, size=4096)
   print('Sparse count Dice:', dice(ecfp_a, ecfp_b, sparse=True))
   print('Sparse presence Tanimoto:', tanimoto(ecfp_a, ecfp_b, sparse=True))
   assert np.isclose(dice(ecfp_a, ecfp_a, sparse=True), 1.0)

The default ``depth=2`` expands environments through two graph bonds,
corresponding to the radius commonly associated with ECFP4. ODDT uses its own
atom-environment hashing and folding implementation; do not assume its
identifiers are interchangeable with RDKit's native Morgan fingerprints.

Set ``use_pharm_features=True`` to hash donor, acceptor, hydrophobic,
ionizable-group, and aromatic features instead of the standard atom invariants:

.. code-block:: python

   feature_ecfp = ECFP(
       aspirin, depth=2, size=4096, use_pharm_features=True
   )
   assert feature_ecfp.ndim == 1

Both variants ignore explicit hydrogen atoms as fingerprint roots and graph
neighbors, but perceived hydrogen counts and chemical features can still
depend on molecular preparation.

Sparse indices, dense counts, and presence bits
-----------------------------------------------

For ECFP and PLEC, ``sparse=True`` means a one-dimensional array of active
feature indices, not a SciPy sparse matrix. With the default
``count_bits=True``, repeated indices represent feature multiplicity.
``len(fp)`` is therefore not the folded feature-space size.

.. code-block:: python

   from oddt.fingerprints import (
       sparse_to_dense, dense_to_sparse, sparse_to_csr_matrix,
   )

   ecfp_counts = ECFP(aspirin, depth=2, size=4096, sparse=False)
   ecfp_bits = ECFP(
       aspirin, depth=2, size=4096, sparse=False, count_bits=False
   )
   assert ecfp_counts.shape == (4096,)
   assert ecfp_bits.dtype == np.bool_
   assert np.array_equal(
       ecfp_counts, sparse_to_dense(ecfp_a, size=4096)
   )
   assert np.array_equal(dense_to_sparse(ecfp_counts), np.sort(ecfp_a))

Use the same ``size``, depths, feature settings, and count/presence policy
throughout a comparison. Folding maps many possible environments into a finite
feature space, so a shared index can be a collision rather than a unique
chemical motif. Sparse index dtypes and dense count dtypes have different
purposes; neither is a fixed dtype guarantee across all parameter choices.

PLEC: pair protein and ligand environments
------------------------------------------

.. code-block:: python

   plec_options = {
       'depth_ligand': 2,
       'depth_protein': 4,
       'distance_cutoff': 4.5,
       'size': 16384,
       'count_bits': True,
       'ignore_hoh': True,
   }
   plec_sparse = PLEC(ligand, protein, **plec_options)
   query_plec = PLEC(query_pose, protein, **plec_options)
   plec_dense = PLEC(ligand, protein, sparse=False, **plec_options)
   assert plec_dense.shape == (16384,)
   assert np.array_equal(
       plec_dense, sparse_to_dense(plec_sparse, size=16384)
   )
   print('PLEC Dice:', dice(plec_sparse, query_plec, sparse=True))

PLEC first finds heavy-atom pairs within the distance cutoff, then combines
their circular environments. The depths above are defaults. When one side
reaches its maximum depth first, that final environment is reused while the
other side continues expanding. PLEC encodes close-contact chemistry; it does
not first require a hydrogen bond, hydrophobic contact, or other strict
interaction classification.

``ignore_hoh=True`` excludes protein-side contact roots whose residue name
is exactly ``HOH``. It does not remove every possible water naming variant
or every cofactor, and it is not a general structure-preparation operation.
Hydrogens are excluded from contact roots and environment traversal.

Trace a PLEC feature to its environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   bit_info = {}
   explained_plec = PLEC(
       ligand, protein, bits_info=bit_info, **plec_options
   )
   assert np.array_equal(explained_plec, plec_sparse)
   if bit_info:
       bit = int(next(iter(bit_info)))
       for record in sorted(bit_info[bit]):
           print(
               bit,
               record.ligand_root_atom_idx, record.ligand_depth,
               record.protein_root_atom_idx, record.protein_depth,
           )

``bits_info`` is modified in place. Each folded index maps to a set of
records describing ligand/protein root atom indices and depths. Several
environment pairs can map to one folded index; the set of explanations does
not retain every repeated contribution to a count. Save the prepared molecules
or an atom-index mapping when you need to inspect those roots later.

SPLIF: include spatial agreement
--------------------------------

.. code-block:: python

   reference_splif = SPLIF(
       ligand, protein, depth=1, size=4096, distance_cutoff=4.5
   )
   query_splif = SPLIF(
       query_pose, protein, depth=1, size=4096, distance_cutoff=4.5
   )
   assert reference_splif.dtype.names == (
       'hash', 'ligand_coords', 'protein_coords'
   )
   print('SPLIF similarity:', similarity_SPLIF(
       reference_splif, query_splif, rmsd_cutoff=1.0
   ))
   assert np.isclose(
       similarity_SPLIF(reference_splif, reference_splif), 1.0
   )

SPLIF returns structured contact records, with a folded hash and coordinate
arrays for the root atom and neighboring atoms on each side. Use
``similarity_SPLIF``, rather than treating it as a vector for Dice or
Tanimoto.

Its similarity function checks matching hashes and coordinate agreement
against ``rmsd_cutoff``; it does not superpose the input complexes. Align
complexes into a common frame before comparing independently oriented
structures, applying each alignment transform to both protein and ligand.

Search a library by interaction similarity
------------------------------------------

Reuse the receptor and reference fingerprint while evaluating candidate poses:

.. code-block:: python

   reference = SimpleInteractionFingerprint(ligand, protein)
   threshold = 0.7
   matches = []
   for candidate in (ligand, query_pose):
       candidate_fp = SimpleInteractionFingerprint(candidate, protein)
       score = dice(reference, candidate_fp)
       if score > threshold:
           matches.append((candidate.title, score))
   assert matches
   assert all(score > threshold for _, score in matches)

For a pose library, replace the two example poses with an iterator from
``oddt.toolkit.readfile('sdf', 'poses.sdf')`` and apply the same preparation
to every candidate before calculating its fingerprint. All poses must share
the receptor's coordinate frame. The threshold controls this particular
search; calibrate it to your data and representation.

Build a sparse feature matrix
-----------------------------

For a library, convert each PLEC index list into a CSR row and stack the rows:

.. code-block:: python

   import json
   from scipy.sparse import vstack, save_npz

   poses = [ligand, query_pose]
   feature_rows = [
       sparse_to_csr_matrix(
           PLEC(pose, protein, **plec_options), size=16384
       )
       for pose in poses
   ]
   features = vstack(feature_rows, format='csr')
   assert features.shape == (2, 16384)
   save_npz('plec-features.npz', features)

   settings = {
       'fingerprint': 'PLEC',
       'oddt_version': oddt.__version__,
       'backend': oddt.toolkit.backend,
       'protein_file': str(protein_path),
       'parameters': plec_options,
       'row_titles': [pose.title for pose in poses],
   }
   with open('plec-settings.json', 'w') as output:
       json.dump(settings, output, indent=2)

Record pose identifiers and preparation details alongside these settings.
Count dtypes may differ between rows; the stacked matrix promotes to a common
dtype. Keep training and prediction feature settings identical.

Interpret similarity consistently
---------------------------------

* ``dice`` uses the minimum count at each shared feature and divides twice
  that overlap by the total counts. It supports count and Boolean vectors.
* ``tanimoto`` compares presence/absence. Dense integer values are converted
  to Boolean values, and repeated sparse indices are treated as one feature.
* Pass ``sparse=True`` when supplying ECFP/PLEC index arrays. The default
  similarity input is a dense vector, even though the default ECFP/PLEC output
  is sparse.
* Both functions return 0.0 for two empty fingerprints. A nonempty
  fingerprint compared with itself scores 1.0.
* IFP needs a consistent residue mapping. SIFP removes residue positions.
  PLEC and ECFP require identical hashing/folding settings. SPLIF also requires
  coordinate-frame agreement.

These comparisons answer different questions. A high molecular-connectivity
similarity does not establish the same bound pose, and an interaction
fingerprint score is not a calibrated binding affinity.

See :doc:`../references` for the ECFP, SILIRID/SIFP, SPLIF, and PLEC method
papers.
