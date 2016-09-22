#!/bin/bash
# anaconda login
# anaconda auth -c -n travis --max-age 307584000 --url https://anaconda.org/USERNAME/PACKAGENAME --scopes "api:write api:read"
mkdir -p ./conda-recipe/ && cd ./conda-recipe/ && \
conda skeleton pypi oddt && \
conda build -c mwojcikowski -c rdkit ./oddt && \
anaconda -t $ANACONDA_TOKEN upload $HOME/miniconda/conda-bld/linux-64/oddt-*.tar.bz2
