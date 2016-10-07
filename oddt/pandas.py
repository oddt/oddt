""" Pandas extension for chemical analysis """
from __future__ import absolute_import
from collections import deque
import pandas as pd

import oddt

pd.set_option("display.max_colwidth", 999999)


def _mol_reader(fmt='sdf',
                filepath_or_buffer=None,
                usecols=None,
                molecule_column='mol',
                molecule_name_column='mol_name',
                smiles_column=None,
                skip_bad_mols=False,
                chunksize=None,
                **kwargs):

    # capture options for reader
    reader_kwargs = {}
    if 'opt' in kwargs:
        reader_kwargs['opt'] = kwargs.pop('opt')
    if 'sanitize' in kwargs:
        reader_kwargs['sanitize'] = kwargs.pop('sanitize')

    chunk = []
    for n, mol in enumerate(oddt.toolkit.readfile(fmt, filepath_or_buffer, **reader_kwargs)):
        if skip_bad_mols and mol is None:
            continue  # add warning with number of skipped molecules
        if usecols:
            mol_data = dict((k, mol.data[k]) for k in usecols)
        else:
            mol_data = mol.data.to_dict()
        if molecule_column:
            mol_data[molecule_column] = mol
        if molecule_name_column:
            mol_data[molecule_name_column] = mol.title
        if smiles_column:
            mol_data[smiles_column] = mol.write('smi').split()[0]
        chunk.append(mol_data)
        if chunksize and (n + 1) % chunksize == 0:
            yield ChemDataFrame(chunk, **kwargs)
            chunk = []
    if chunk or chunksize is None:
        yield ChemDataFrame(chunk, **kwargs)


def read_sdf(filepath_or_buffer=None,
             usecols=None,
             molecule_column='mol',
             molecule_name_column='mol_name',
             smiles_column=None,
             skip_bad_mols=False,
             chunksize=None,
             **kwargs):
    result = _mol_reader(fmt='sdf',
                         filepath_or_buffer=filepath_or_buffer,
                         usecols=usecols,
                         molecule_column=molecule_column,
                         molecule_name_column=molecule_name_column,
                         smiles_column=smiles_column,
                         skip_bad_mols=skip_bad_mols,
                         chunksize=chunksize,
                         **kwargs)
    if chunksize:
        return result
    else:
        return deque(result, maxlen=1).pop()


def read_mol2(filepath_or_buffer=None,
              usecols=None,
              molecule_column='mol',
              molecule_name_column='mol_name',
              smiles_column=None,
              skip_bad_mols=False,
              chunksize=None,
              **kwargs):
    result = _mol_reader(fmt='mol2',
                         filepath_or_buffer=filepath_or_buffer,
                         usecols=usecols,
                         molecule_column=molecule_column,
                         molecule_name_column=molecule_name_column,
                         smiles_column=smiles_column,
                         skip_bad_mols=skip_bad_mols,
                         chunksize=chunksize,
                         **kwargs)
    if chunksize:
        return result
    else:
        return deque(result, maxlen=1).pop()


class ChemSeries(pd.Series):
    @property
    def _constructor(self):
        return ChemSeries


class ChemDataFrame(pd.DataFrame):
    """

    Note:
    Thanks to: http://blog.snapdragon.cc/2015/05/05/subclass-pandas-dataframe-to-save-custom-attributes/
    """
    def to_sdf(self,
               filepath_or_buffer=None,
               update_properties=True,
               molecule_column='mol',
               columns=None):
        out = oddt.toolkit.Outputfile('sdf', filepath_or_buffer, overwrite=True)
        for ix, row in self.iterrows():
            mol = row[molecule_column].clone
            if update_properties:
                new_data = row.to_dict()
                del new_data[molecule_column]
                mol.data.update(new_data)
            if columns:
                for k in mol.data.keys():
                    if k not in columns:
                        del mol.data[k]
            out.write(mol)
        out.close()

    def to_html(self, *args, **kwargs):
        kwargs['escape'] = False
        return super(ChemDataFrame, self).to_html(*args, **kwargs)

    @property
    def _constructor(self):
        """ Force new class to be usead as sconstructor when slicing """
        return ChemDataFrame

    _constructor_sliced = ChemSeries
