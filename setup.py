#!/usr/bin/env python

from distutils.core import setup

setup(name='ODDT',
        version='0.1.0',
        description='Open Drug Discovery Toolkit',
        author='Maciej Wojcikowski',
        author_email='mwojcikowski@ibb.waw.pl',
        url='https://github.com/oddt/oddt',
        packages=['oddt',
                    'oddt.toolkits',
                    'oddt.scoring',
                    'oddt.scoring.models',
                    'oddt.scoring.functions',
                    'oddt.scoring.descriptors',
                    'oddt.docking',
                    ],
    package_data={'oddt.scoring.functions': ['NNScore/*.csv', 'RFScore/*.csv']},
    )
