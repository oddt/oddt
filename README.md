# Open Drug Discovery Toolkit

Open Drug Discovery Toolkit (ODDT) is modular and comprehensive toolkit for use in cheminformatics, molecular modeling etc. ODDT is written in Python, and make extensive use of Numpy/Scipy

[![Documentation Status](https://readthedocs.org/projects/oddt/badge/?version=latest)](http://oddt.readthedocs.org/en/latest/)


### Requrements
   * Python 2.7.X
   * OpenBabel (2.3.2+) or/and RDKit (2012.03)
   * Numpy (1.6.2+)
   * Scipy (0.10+)
   * Sklearn (0.11+)
   * ffnet (0.7.1+) only for neural network functionality.

### Install
When all requirements are met, then installation process is simple
> python setup.py install

You can also use pip. All requirements besides toolkits (OpenBabel, RDKit) are installed if necessary.
Installing inside virtualenv is encouraged, but not necessary.
> pip install oddt

### Documentation
Automatic documentation for ODDT is available on [Readthedocs.org](https://oddt.readthedocs.org/). Additionally it can be build localy:
   > cd docs
   
   > make html
   
   > make latexpdf

### License
ODDT is released under permissive [3-clause BSD license](./LICENSE)

[![Analytics](https://ga-beacon.appspot.com/UA-44788495-3/oddt/oddt)](https://github.com/igrigorik/ga-beacon)
