#!/usr/bin/env python
from setuptools import setup, find_packages
from oddt import __version__ as VERSION

setup(name='oddt',
      version=VERSION,
      description='Open Drug Discovery Toolkit',
      author='Maciej Wojcikowski',
      author_email='mwojcikowski@ibb.waw.pl',
      url='https://github.com/oddt/oddt',
      license='BSD',
      packages=find_packages(),
      package_data={'oddt.scoring.functions': ['NNScore/*.csv',
                                               'RFScore/*.csv',
                                               'PLECscore/*.json',
                                               ],
                    'oddt.toolkits.extras': ['pdb_residue_templates.smi'],
                    },
      setup_requires=['numpy'],
      install_requires=open('requirements.txt', 'r').readlines(),
      download_url='https://github.com/oddt/oddt/tarball/%s' % VERSION,
      keywords=['cheminformatics', 'qsar', 'virtual screening', 'docking', 'pipeline'],
      scripts=['bin/oddt_cli'],
      )
