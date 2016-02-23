#!/bin/bash
# anaconda login
# anaconda auth -c -n travis --max-age 307584000 --url https://anaconda.org/USERNAME/PACKAGENAME --scopes "api:write api:read"
conda skeleton pypi oddt --output-dir oddt-conda --version $TRAVIS_TAG
anaconda upload $HOME/miniconda/conda-bld/linux-64/oddt-*.tar.bz2
