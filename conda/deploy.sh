#!/bin/bash
# anaconda login
# anaconda auth -c -n travis --max-age 307584000 --url https://anaconda.org/USERNAME/PACKAGENAME --scopes "api:write api:read"
mkdir -p ./conda-recipe/ && cd ./conda-recipe/ && \
conda skeleton pypi --extra-specs=six --noarch-python oddt && \
conda build -c openbabel -c rdkit --py=$TRAVIS_PYTHON_VERSION ./oddt && \
anaconda -t $ANACONDA_TOKEN upload --force $HOME/miniconda/conda-bld/*/oddt-*.tar.bz2
