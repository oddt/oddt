Molecular shape comparison
==========================

:mod:`oddt.shape` provides alignment-free descriptors of individual 3D
conformers. These describe a molecule's shape; protein–ligand contact patterns
are covered in :doc:`fingerprints`.

Choose a descriptor
-------------------

.. list-table::
   :header-rows: 1

   * - Function
     - Length
     - Information
   * - :func:`~oddt.shape.usr`
     - 12
     - Moments of distances to four reference points.
   * - :func:`~oddt.shape.usr_cat`
     - 60
     - USR plus hydrophobic, aromatic, acceptor, and donor selections.
   * - :func:`~oddt.shape.electroshape`
     - 15
     - Spatial coordinates augmented with partial charges.

Compare two conformers
----------------------

.. code-block:: python

   import numpy as np
   import oddt
   from oddt.shape import usr, usr_cat, usr_similarity

   ligand = next(oddt.toolkit.readfile(
       'sdf', 'tests/data/pdb/4lb3_ligand.sdf'
   ))
   translated = ligand.clone
   translated.coords = translated.coords + [5.0, 0.0, 0.0]

   query_shape = usr(ligand)
   candidate_shape = usr(translated)
   score = usr_similarity(query_shape, candidate_shape)
   assert query_shape.shape == (12,)
   assert np.isclose(score, 1.0)

   query_cat = usr_cat(ligand)
   assert query_cat.shape == (60,)
   assert np.isclose(usr_similarity(query_cat, query_cat), 1.0)

The translation leaves the shape descriptor unchanged. In contrast,
translating only the ligand changes its interactions with a fixed protein.
Compare descriptors produced by the same method and preparation settings;
``usr_similarity`` checks descriptor lengths.

For Electroshape, calculate partial charges before generating the descriptor:

.. code-block:: python

   from oddt.shape import electroshape

   ligand.calccharges('gasteiger')
   charged_shape = electroshape(ligand)
   assert charged_shape.shape == (15,)
   assert np.isfinite(charged_shape).all()

Use consistent hydrogen and charge preparation throughout a library. Flexible
molecules may need several conformers; one descriptor represents only the
coordinates currently stored in the molecule.

The underlying papers are listed in :doc:`../references`.
