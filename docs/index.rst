.. _oddt:
.. highlight:: python

.. _welcome-to-oddt-s-documentation:

***************************
Open Drug Discovery Toolkit
***************************

.. container:: oddt-lead

   A Python toolkit for cheminformatics, molecular modeling, and virtual screening.
   Work with RDKit and Open Babel through one consistent interface.

.. container:: oddt-meta

   Python 3.12+ · RDKit & Open Babel · BSD licensed

.. container:: oddt-grid

   .. container:: oddt-card

      **01 / Get started**

      :ref:`Install ODDT <installation-guide>`

      Set up your environment and choose a chemistry backend.

   .. container:: oddt-card

      **02 / Explore**

      :ref:`User guide <usage-guide>`

      Read molecules, analyze interactions, and build screening workflows.

   .. container:: oddt-card

      **03 / Build**

      :doc:`API reference <rst/modules>`

      Explore the modules, classes, and functions behind your workflow.

`View on GitHub <https://github.com/oddt/oddt>`_ ·
`Report an issue <https://github.com/oddt/oddt/issues>`_

Analysis guides
===============

* :ref:`NumPy dictionaries <numpy-dictionaries>`: work with typed molecular
  arrays, vectorize calculations, and control memory use on large systems.
* :doc:`user-guide/interactions`: inspect hydrogen bonds, halogen bonds,
  hydrophobic contacts, ionic interactions, and metal coordination.
* :doc:`user-guide/fingerprints`: encode binding modes, choose a representation,
  compare poses, and build sparse feature matrices.

.. toctree::
   :caption: Get started
   :maxdepth: 1

   installation
   user-guide/index

.. toctree::
   :caption: API reference
   :maxdepth: 2

   rst/modules

.. toctree::
   :caption: Project
   :maxdepth: 1

   contributing
   references

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
