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
* OpenBabel (2.3.2+) or/and RDKit (2014.03)
* Numpy (1.8+)
* Scipy (0.13+)
* Sklearn (0.18+)
* joblib (0.8+)
* pandas (0.17+)

.. note:: All installation methods assume that one of toolkits is installed. For detailed installation procedure visit toolkit’s website (OpenBabel, RDKit)

Most convenient way of installing ODDT is using PIP. All required python modules will be installed automatically, although toolkits, either OpenBabel (``pip install openbabel``) or RDKit need to be installed manually
::
    pip install oddt

If you want to install cutting edge version (master branch from GitHub) of ODDT also using PIP
::
    pip install git+https://github.com/oddt/oddt.git@master

Finally you can install ODDT straight from the source
::
    wget https://github.com/oddt/oddt/archive/0.2.0.tar.gz
    tar zxvf 0.2.0.tar.gz
    cd oddt-0.2.0/
    python setup.py install

Common installation problems
````````````````````````````


Usage Instructions
==================
Toolkits
--------

You can use any supported toolkit united under common API (for reference see `Pybel <https://open-babel.readthedocs.org/en/latest/UseTheLibrary/Python_Pybel.html>`_ or `Cinfony <https://code.google.com/p/cinfony/>`_). All methods and software which based on Pybel/Cinfony should be drop in compatible with ODDT toolkits. In contrast to it’s predecessors, which were aimed to have minimalistic API, ODDT introduces extended methods and additional handles. This extensions allow to use toolkits at all it’s grace and some features may be backported from others to introduce missing functionalities.
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
::
    mol = oddt.toolkit.readstring(‘smi’, ‘c1cccc1’)
    for atom in mol:
        print(atom.idx)

.. note:: mol.atoms, returns an object (:class:`~oddt.toolkit.AtomStack`) which can be access via indexes or iterated

Iterating over residues is also very convenient, especially for proteins
::
    for res in mol.residues:
        print(res.name)

Additionally residues can fetch atoms belonging to them:
::
    for res in mol.residues:
        for atom in res:
            print(atom.idx)

Bonds are also iterable, similar to residues:
::
    for bond in mol.bonds:
        print(bond.order)
        for atom in bond:
            print(atom.idx)

Reading molecules
`````````````````

Reading molecules is mostly identical to `Pybel <https://open-babel.readthedocs.org/en/latest/UseTheLibrary/Python_Pybel.html>`_.

Reading from file
::
    for mol in oddt.toolkit.readfile(‘smi’, ‘test.smi’):
        print(mol.title)

Reading from string
::
    mol = oddt.toolkit.readstring(‘smi’, ‘c1ccccc1 benzene’):
        print(mol.title)

.. note:: You can force molecules to be read in asynchronously, aka “lazy molecules”. Current default is not to produce lazy molecules due to OpenBabel’s Memory Leaks in OBConverter. Main advantage of lazy molecules is using them in multiprocessing, then conversion is spreaded on all jobs.

Reading molecules from file in asynchronous manner
::
    for mol in oddt.toolkit.readfile(‘smi’, ‘test.smi’, lazy=True):
        pass

This example will execute instantaneously, since no molecules were evaluated.

Numpy Dictionaries - store your molecule as an uniform structure
```````````````````````````````````````````````

Most important and handy property of Molecule in ODDT are Numpy dictionaries containing most properties of supplied molecule. Some of them are straightforward, other require some calculation, ie. atom features. Dictionaries are provided for major entities of molecule: atoms, bonds, residues and rings. It was primarily used for interactions calculations, although it is applicable for any other calculation. The main benefit is marvelous Numpy broadcasting and subsetting.


Each dictionary is defined as a format in Numpy.

atom_dict
---------

Atom basic information

* '*coords*', type: ``float32``, shape: (3) - atom coordinates
* '*charge*', type: ``float32`` - atom's charge
* '*atomicnum*', type: ``int8`` - atomic number
* '*atomtype', type: ``a4`` - Sybyl atom's type
* '*hybridization*', type: ``int8`` - atoms hybrydization
* '*neighbors*', type: ``float32``, shape: (4,3) - coordinates of non-H neighbors coordinates for angles (max of 4 neighbors should be enough)

Residue information for current atom

* '*resid*', type: ``int16`` - residue ID
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
* '*resname*', type: ``a3`` - Residue name (3 letters)
* '*N*', type: ``float32``, shape: 3 - cordinates of backbone N atom
* '*CA*', type: ``float32``, shape: 3 - cordinates of backbone CA atom
* '*C*', type: ``float32``, shape: 3 - cordinates of backbone C atom
* '*isalpha*', type: ``bool`` - is residue a part of alpha helix
* '*isbeta*', type: ``bool'`` - is residue a part of beta strand


.. note:: All aforementioned dictionaries are generated “on demand”, and are cached for molecule, thus can be shared between calculations. Caching of dictionaries brings incredible performance gain, since in some applications their generation is the major time consuming task.

Get all acceptor atoms:
::
    mol.atom_dict[‘is_acceptor’]

ODDT command line interface (CLI)
=================================

There is an `oddt` command to interface with Open Drug Discovery Toolkit from terminal, without any programming knowleadge.
It simply reproduces :class:`oddt.virtualscreening.virtualscreening`.
One can filter, dock and score ligands using methods implemented or compatible with ODDT.
All positional arguments are treated as input ligands, whereas output must be assigned using `-O` option (following `obabel` convention).
Input and output formats are defined using `-i` and `-o` accordingly.
If output format is present and no output file is assigned, then molecules are printed to STDOUT.



To list all the available options issue `-h` option:
::
    oddt_cli -h

Examples
--------

1. Docking ligand using Autodock Vina (construct box using ligand from crystal structure) with additional RFscore v2 rescoring:
::
    oddt_cli input_ligands.sdf --dock autodock_vina --receptor rec.mol2 --auto_ligand crystal_ligand.mol2 --score rfscore_v2 -O output_ligands.sdf


2. Filtering ligands using Lipinski RO5 and PAINS. Afterwards dock with Autodock Vina:
::
    oddt_cli input_ligands.sdf --filter ro5 --filter pains --dock autodock_vina --receptor rec.mol2 --auto_ligand crystal_ligand.mol2 -O output_ligands.sdf

3. Dock with Autodock Vina, with precise box position and dimensions. Fix seed for reproducibility and increase exhaustiveness:
::
    oddt_cli ampc/actives_final.mol2.gz --dock autodock_vina --receptor ampc/receptor.pdb --size '(8,8,8)' --center '(1,2,0.5)' --exhaustiveness 20 --seed 1 -O ampc_docked.sdf

4. Rescore ligands using 3 versions of RFscore and pre-trained scoring function (either pickle from ODDT or any other SF implementing :class:`oddt.scoring.scorer` API):
::
    oddt_cli docked_ligands.sdf --receptor rec.mol2 --score rfscore_v1 --score rfscore_v2 --score rfscore_v3 --score TrainedNN.pickle -O docked_ligands_rescored.sdf


ODDT API documentation
======================

.. toctree:: rst/oddt.rst

References
==========

To be announced.

Docuimentation Indices and tables
=================================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
