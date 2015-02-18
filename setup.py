#!/usr/bin/env python
import setuptools
from setuptools import setup,find_packages

setup(name='oddt',
        version='0.1.1',
        description='Open Drug Discovery Toolkit',
        author='Maciej Wojcikowski',
        author_email='mwojcikowski@ibb.waw.pl',
        url='https://github.com/oddt/oddt',
        license = 'BSD',
        packages=find_packages(),
        package_data={'oddt.scoring.functions': ['NNScore/*.csv', 'RFScore/*.csv']},
        setup_requires = ['numpy>=1.6.2'],
        install_requires = open('requirements.txt', 'r').readlines(),
    )
