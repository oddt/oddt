.. _oddt::
.. highlight:: python

********************************
Welcome to ODDT's documentation!
********************************

.. contents::
    :depth: 5

Installation
============

Requirements
````````````

* Python 2.7+ or 3.4+
* OpenBabel (2.3.2+) or/and RDKit (2016.03)
* Numpy (1.8+)
* Scipy (0.14+)
* Sklearn (0.18+)
* joblib (0.8+)
* pandas (0.17.1+)
* Skimage (0.10+) (optional, only for surface generation)

.. note:: All installation methods assume that one of toolkits is installed. For detailed installation procedure visit toolkit’s website (OpenBabel, RDKit)

Most convenient way of installing ODDT is using PIP. All required python modules will be installed automatically, although toolkits, either OpenBabel (``pip install openbabel``) or RDKit need to be installed manually

.. code-block:: bash

    pip install oddt

If you want to install cutting edge version (master branch from GitHub) of ODDT also using PIP

.. code-block:: bash

    pip install git+https://github.com/oddt/oddt.git@master

Finally you can install ODDT straight from the source

.. code-block:: bash

    wget https://github.com/oddt/oddt/archive/0.5.tar.gz
    tar zxvf 0.5.tar.gz
    cd oddt-0.5/
    python setup.py install

Common installation problems
````````````````````````````


Usage Instructions
==================
Toolkits
--------

You can use any supported toolkit united under common API (for reference see `Pybel <https://open-babel.readthedocs.org/en/latest/UseTheLibrary/Python_Pybel.html>`_ or `Cinfony <https://code.google.com/p/cinfony/>`_). All methods and software which based on Pybel/Cinfony should be drop in compatible with ODDT toolkits. In contrast to its predecessors, which were aimed to have minimalistic API, ODDT introduces extended methods and additional handles. This extensions allow to use toolkits at all its grace and some features may be backported from others to introduce missing functionalities.
To name a few:

* coordinates are returned as Numpy Arrays
* atoms and residues methods of Molecule class are lazy, ie. not returning a list of pointers, rather an object which allows indexing and iterating through atoms/residues
* Bond object (similar to Atom)
* `atom_dict`_, `ring_dict`_, `res_dict`_ - comprehensive Numpy Arrays containing common information about given entity, particularly useful for high performance computing, ie. interactions, scoring etc.
* lazy Molecule (asynchronous), which is not converted to an object in reading phase, rather passed as a string and read in when underlying object is called
* pickling introduced for Pybel Molecule (internally saved to mol2 string)

Molecules
---------

Atom, residues, bonds iteration
```````````````````````````````

One of the most common operation would be iterating through molecules atoms

.. code-block:: Python

    mol = oddt.toolkit.readstring('smi', 'c1cccc1')
    for atom in mol:
        print(atom.idx)

.. note:: mol.atoms, returns an object (:class:`~oddt.toolkit.AtomStack`) which can be access via indexes or iterated

Iterating over residues is also very convenient, especially for proteins

.. code-block:: python

    for res in mol.residues:
        print(res.name)

Additionally residues can fetch atoms belonging to them:

.. code-block:: python

    for res in mol.residues:
        for atom in res:
            print(atom.idx)

Bonds are also iterable, similar to residues:

.. code-block:: python

    for bond in mol.bonds:
        print(bond.order)
        for atom in bond:
            print(atom.idx)

Reading molecules
`````````````````

Reading molecules is mostly identical to `Pybel <https://open-babel.readthedocs.org/en/latest/UseTheLibrary/Python_Pybel.html>`_.

Reading from file

.. code-block:: python

    for mol in oddt.toolkit.readfile('smi', 'test.smi'):
        print(mol.title)

Reading from string

.. code-block:: python

    mol = oddt.toolkit.readstring('smi', 'c1ccccc1 benzene'):
        print(mol.title)

.. note:: You can force molecules to be read in asynchronously, aka “lazy molecules”. Current default is not to produce lazy molecules due to OpenBabel’s Memory Leaks in OBConverter. Main advantage of lazy molecules is using them in multiprocessing, then conversion is spreaded on all jobs.

Reading molecules from file in asynchronous manner

.. code-block:: python

    for mol in oddt.toolkit.readfile('smi', 'test.smi', lazy=True):
        pass

This example will execute instantaneously, since no molecules were evaluated.

Numpy Dictionaries - store your molecule as an uniform structure
````````````````````````````````````````````````````````````````

Most important and handy property of Molecule in ODDT are Numpy dictionaries containing most properties of supplied molecule. Some of them are straightforward, other require some calculation, ie. atom features. Dictionaries are provided for major entities of molecule: atoms, bonds, residues and rings. It was primarily used for interactions calculations, although it is applicable for any other calculation. The main benefit is marvelous Numpy broadcasting and subsetting.


Each dictionary is defined as a format in Numpy.

atom_dict
---------

Atom basic information

* '*coords*', type: ``float32``, shape: (3) - atom coordinates
* '*charge*', type: ``float32`` - atom's charge
* '*atomicnum*', type: ``int8`` - atomic number
* '*atomtype*', type: ``a4`` - Sybyl atom's type
* '*hybridization*', type: ``int8`` - atoms hybrydization
* '*neighbors*', type: ``float32``, shape: (4,3) - coordinates of non-H neighbors coordinates for angles (max of 4 neighbors should be enough)

Residue information for current atom

* '*resid*', type: ``int16`` - residue ID
* '*resnumber*', type: ``int16`` - residue number
* '*resname*', type: ``a3`` - Residue name (3 letters)
* '*isbackbone*', type: ``bool`` - is atom part of backbone

Atom properties

* '*isacceptor*', type: ``bool`` - is atom H-bond acceptor
* '*isdonor*', type: ``bool`` - is atom H-bond donor
* '*isdonorh*', type: ``bool`` - is atom H-bond donor Hydrogen
* '*ismetal*', type: ``bool`` - is atom a metal
* '*ishydrophobe*', type: ``bool`` - is atom hydrophobic
* '*isaromatic*', type: ``bool`` - is atom aromatic
* '*isminus*', type: ``bool`` - is atom negatively charged/chargable
* '*isplus*', type: ``bool`` - is atom positively charged/chargable
* '*ishalogen*', type: ``bool`` - is atom a halogen

Secondary structure

* '*isalpha*', type: ``bool`` - is atom a part of alpha helix
* '*isbeta*', type: ``bool'`` - is atom a part of beta strand


ring_dict
---------

* '*centroid*', type: ``float32``, shape: 3 - coordinates of ring's centroid
* '*vector*', type: ``float32``, shape: 3 - normal vector for ring
* '*isalpha*', type: ``bool`` - is ring a part of alpha helix
* '*isbeta*', type: ``bool'`` - is ring a part of beta strand

res_dict
--------

* '*id*', type: ``int16`` - residue ID
* '*resnumber*', type: ``int16`` - residue number
* '*resname*', type: ``a3`` - Residue name (3 letters)
* '*N*', type: ``float32``, shape: 3 - cordinates of backbone N atom
* '*CA*', type: ``float32``, shape: 3 - cordinates of backbone CA atom
* '*C*', type: ``float32``, shape: 3 - cordinates of backbone C atom
* '*isalpha*', type: ``bool`` - is residue a part of alpha helix
* '*isbeta*', type: ``bool'`` - is residue a part of beta strand


.. note:: All aforementioned dictionaries are generated “on demand”, and are cached for molecule, thus can be shared between calculations. Caching of dictionaries brings incredible performance gain, since in some applications their generation is the major time consuming task.

Get all acceptor atoms:

.. code-block:: python

    mol.atom_dict['isacceptor']


Interaction Fingerprints
````````````````````````
Module, where interactions between two molecules are calculated and stored in fingerprint.

The most common usage
---------------------

Firstly, loading files

.. code-block:: python

    protein = next(oddt.toolkit.readfile('pdb', 'protein.pdb'))
    protein.protein = True
    ligand = next(oddt.toolkit.readfile('sdf', 'ligand.sdf'))

.. note:: You have to mark a variable with file as protein, otherwise You won't be able to get access to e.g. 'resname; , 'resid' etc. It can be done as above.

File with more than one molecule

.. code-block:: python

  mols = list(oddt.toolkit.readfile('sdf', 'ligands.sdf'))

When files are loaded, You can check interactions between molecules. Let's find out, which amino acids creates hydrogen bonds
::
  protein_atoms, ligand_atoms, strict = hbonds(protein, ligand)
  print(protein_atoms['resname'])

Or check hydrophobic contacts between molecules
::
  protein_atoms, ligand_atoms = hydrophobic_contacts(protein, ligand)
  print(protein_atoms, ligand_atoms)

But instead of checking interactions one by one, You can use fingerprints module.

.. code-block:: python

  IFP = InteractionFingerprint(ligand, protein)
  SIFP = SimpleInteractionFingerprint(ligand, protein)

Very often we're looking for similar molecules. We can easily accomplish this by e.g.

.. code-block:: python

  results = []
  reference = SimpleInteractionFingerprint(ligand, protein)
  for el in query:
      fp_query = SimpleInteractionFingerprint(el, protein)
      # similarity score for current query
      cur_score = dice(reference, fp_query)
      # score is the lowest, required similarity
      if cur_score > score:
          results.append(el)
  return results

Molecular shape comparison
``````````````````````````
Three methods for molecular shape comparison are supported: USR and its two derivatives: USRCAT and Electroshape.

* USR (Ultrafast Shape Recognition) - function usr(molecule)
    Ballester PJ, Richards WG (2007). Ultrafast shape recognition to search
    compound databases for similar molecular shapes. Journal of
    computational chemistry, 28(10):1711-23.
    http://dx.doi.org/10.1002/jcc.20681

* USRCAT (USR with Credo Atom Types) - function usr_cat(molecule)
    Adrian M Schreyer, Tom Blundell (2012). USRCAT: real-time ultrafast
    shape recognition with pharmacophoric constraints. Journal of
    Cheminformatics, 2012 4:27.
    http://dx.doi.org/10.1186/1758-2946-4-27

* Electroshape - function electroshape(molecule)
    Armstrong, M. S. et al. ElectroShape: fast molecular similarity
    calculations incorporating shape, chirality and electrostatics.
    J Comput Aided Mol Des 24, 789-801 (2010).
    http://dx.doi.org/doi:10.1007/s10822-010-9374-0

    Aside from spatial coordinates, atoms' charges are also used
    as the fourth dimension to describe shape of the molecule.

To find most similar molecules from the given set, each of these methods can be used.

Loading files:

.. code-block:: python

    query = next(oddt.toolkit.readfile('sdf', 'query.sdf'))
    database = list(oddt.toolkit.readfile('sdf', 'database.sdf'))

Example code to find similar molecules:

.. code-block:: python

    results = []
    query_shape = usr(query)
    for mol in database:
        mol_shape = usr(mol)
        similarity = usr_similarity(query_shape, mol_shape)
        if similarity > 0.7:
            results.append(mol)

To use another method, replace usr(mol) with usr_cat(mol) or electroshape(mol).

ODDT command line interface (CLI)
=================================

There is an `oddt` command to interface with Open Drug Discovery Toolkit from terminal, without any programming knowleadge.
It simply reproduces :class:`oddt.virtualscreening.virtualscreening`.
One can filter, dock and score ligands using methods implemented or compatible with ODDT.
All positional arguments are treated as input ligands, whereas output must be assigned using `-O` option (following `obabel` convention).
Input and output formats are defined using `-i` and `-o` accordingly.
If output format is present and no output file is assigned, then molecules are printed to STDOUT.



To list all the available options issue `-h` option:

.. code-block:: bash

    oddt_cli -h

Examples
--------

1. Docking ligand using Autodock Vina (construct box using ligand from crystal structure) with additional RFscore v2 rescoring:

.. code-block:: bash

    oddt_cli input_ligands.sdf --dock autodock_vina --receptor rec.mol2 --auto_ligand crystal_ligand.mol2 --score rfscore_v2 -O output_ligands.sdf


2. Filtering ligands using Lipinski RO5 and PAINS. Afterwards dock with Autodock Vina:

.. code-block:: bash

    oddt_cli input_ligands.sdf --filter ro5 --filter pains --dock autodock_vina --receptor rec.mol2 --auto_ligand crystal_ligand.mol2 -O output_ligands.sdf

3. Dock with Autodock Vina, with precise box position and dimensions. Fix seed for reproducibility and increase exhaustiveness:

.. code-block:: bash

    oddt_cli ampc/actives_final.mol2.gz --dock autodock_vina --receptor ampc/receptor.pdb --size '(8,8,8)' --center '(1,2,0.5)' --exhaustiveness 20 --seed 1 -O ampc_docked.sdf

4. Rescore ligands using 3 versions of RFscore and pre-trained scoring function (either pickle from ODDT or any other SF implementing :class:`oddt.scoring.scorer` API):

.. code-block:: bash

    oddt_cli docked_ligands.sdf --receptor rec.mol2 --score rfscore_v1 --score rfscore_v2 --score rfscore_v3 --score TrainedNN.pickle -O docked_ligands_rescored.sdf

Development and contributions guide
===========================================

1. Indicies
All indicies within toolkit are 0-based, but for backward compatibility with OpenBabel there is ``mol.idx`` property.
If you develop using ODDT you are encouraged to use 0-based indicies and/or ``mol.idx0`` and ``mol.idx1`` properties to be exact which convention you adhere to.
Otherwise you can run into bags which are hard to catch, when writing toolkit independent code.

ODDT API documentation
======================

.. toctree:: rst/oddt.rst

References
==========

To be announced.

Documentation Indices and tables
=================================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
