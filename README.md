# Open Drug Discovery Toolkit

Open Drug Discovery Toolkit (ODDT) is modular and comprehensive toolkit for use in cheminformatics, molecular modeling etc. ODDT is written in Python, and make extensive use of Numpy/Scipy

[![Documentation Status](https://readthedocs.org/projects/oddt/badge/?version=latest)](http://oddt.readthedocs.org/en/latest/)
[![Build Status](https://travis-ci.org/oddt/oddt.svg?branch=master)](https://travis-ci.org/oddt/oddt)
[![Coverage Status](https://coveralls.io/repos/github/oddt/oddt/badge.svg?branch=master)](https://coveralls.io/github/oddt/oddt?branch=master)
[![Code Health](https://landscape.io/github/oddt/oddt/master/landscape.svg?style=flat)](https://landscape.io/github/oddt/oddt/master)
[![Conda packages](https://anaconda.org/oddt/oddt/badges/version.svg?style=flat)](https://anaconda.org/oddt/oddt)
[![Latest Version](https://img.shields.io/pypi/v/oddt.svg)](https://pypi.python.org/pypi/oddt/)

## Documentation, Discussion and Contribution:
  * Documentation: http://oddt.readthedocs.org/en/latest
  * Mailing list: oddt@googlegroups.com  http://groups.google.com/d/forum/oddt
  * Issues: https://github.com/oddt/oddt/issues

## Requrements
  * Python 2.7+ or 3.4+
  * OpenBabel (2.3.2+) or/and RDKit (2016.03)
  * Numpy (1.8+)
  * Scipy (0.14+)
  * Sklearn (0.18+)
  * joblib (0.8+)
  * pandas (0.17.1+)
  * Skimage (0.10+) (optional, only for surface generation)

## Install

### Using PyPi (pip)
  When all requirements are met, then installation process is simple
  > python setup.py install

  You can also use pip. All requirements besides toolkits (OpenBabel, RDKit) are installed if necessary.
  Installing inside virtualenv is encouraged, but not necessary.
  > pip install oddt

  To upgrade oddt using pip (without upgrading dependencies):
  > pip install -U --no-deps oddt

### Using conda
  Install a clean [Miniconda environment](https://conda.io/miniconda.html), if you already don't have one.

  Install ODDT:
  > conda install -c oddt oddt

  You can add a toolkit of your choice or install them along with oddt:
  > conda install -c oddt oddt rdkit openbabel

  (Optionally) Install OpenBabel (using [official  channel](https://anaconda.org/openbabel/openbabel)):
  > conda install -c openbabel openbabel

  (Optionally) install RDKit (using [official channel](https://anaconda.org/rdkit/rdkit)):
  > conda install -c rdkit rdkit

  Upgrading procedure using conda is straightforward:
  > conda update -c oddt oddt

### Documentation
Automatic documentation for ODDT is available on [Readthedocs.org](https://oddt.readthedocs.org/). Additionally it can be build localy:
   > cd docs

   > make html

   > make latexpdf

### License
ODDT is released under permissive [3-clause BSD license](./LICENSE)

### Reference
If you found Open Drug Discovery Toolkit useful for your research, please cite us!

1. Wójcikowski, M., Zielenkiewicz, P., & Siedlecki, P. (2015). Open Drug Discovery Toolkit (ODDT): a new open-source player in the drug discovery field. Journal of Cheminformatics, 7(1), 26. [doi:10.1186/s13321-015-0078-2](https://dx.doi.org/10.1186/s13321-015-0078-2)


[![Analytics](https://ga-beacon.appspot.com/UA-44788495-3/oddt/oddt?flat)](https://github.com/igrigorik/ga-beacon)
