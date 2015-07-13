# Open Drug Discovery Toolkit

Open Drug Discovery Toolkit (ODDT) is modular and comprehensive toolkit for use in cheminformatics, molecular modeling etc. ODDT is written in Python, and make extensive use of Numpy/Scipy

[![Documentation Status](https://readthedocs.org/projects/oddt/badge/?version=latest)](http://oddt.readthedocs.org/en/latest/)
[![Latest Version](https://img.shields.io/pypi/v/oddt.svg)](https://pypi.python.org/pypi/oddt/)
[![Downloads](https://img.shields.io/pypi/dm/oddt.svg)](https://pypi.python.org/pypi/oddt/)

### Documentation, Discussion and Contribution:
 * Documentation: http://oddt.readthedocs.org/en/latest
 * Mailing list: oddt@googlegroups.com  http://groups.google.com/d/forum/oddt
 * Issues: https://github.com/oddt/oddt/issues

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

### Upgrading
To upgrade oddt using pip (without upgrading dependencies):
> pip install -U --no-deps oddt

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
