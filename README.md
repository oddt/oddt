# Open Drug Discovery Toolkit

Open Drug Discovery Toolkit (ODDT) is modular and comprehensive toolkit for use in cheminformatics, molecular modeling etc. ODDT is written in Python, and make extensive use of Numpy/Scipy

[![Documentation Status](https://readthedocs.org/projects/oddt/badge/?version=latest)](http://oddt.readthedocs.org/en/latest/)
[![Latest Version](https://img.shields.io/pypi/v/oddt.svg)](https://pypi.python.org/pypi/oddt/)
[![Downloads](https://img.shields.io/pypi/dm/oddt.svg)](https://pypi.python.org/pypi/oddt/)

## Documentation, Discussion and Contribution:
  * Documentation: http://oddt.readthedocs.org/en/latest
  * Mailing list: oddt@googlegroups.com  http://groups.google.com/d/forum/oddt
  * Issues: https://github.com/oddt/oddt/issues

## Requrements
  * Python 2.7+ or 3.4+
  * OpenBabel (2.3.2+) or/and RDKit (2014.03)
  * Numpy (1.8+)
  * Scipy (0.13+)
  * Sklearn (0.18+)
  * joblib (0.8+)
  * pandas (0.17+)

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
  Add conda channel which contains OpenBabel and ODDT [[link](https://anaconda.org/mwojcikowski)]
  > conda config --add channels mwojcikowski

  Optionally you can add RDKit's channel
  > conda config --add channels rdkit

  Install ODDT:
  > conda install oddt

  Optionally install RDKit:
  > conda install rdkit

  Upgrading procedure using conda is straightforward:
  > conda update oddt

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
