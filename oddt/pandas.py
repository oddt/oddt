""" Pandas extension for chemical analysis """
from __future__ import absolute_import
import pandas as pd

import oddt

pd.set_option("display.max_colwidth", 999999)


def _mol_dict_reader(fmt='sdf',
                     filepath_or_buffer=None,
                     usecols=None,
                     molecule_column='mol',
                     molecule_name='mol_name',
                     smiles_column=None,
                     skip_bad_mols=False,
                     **kwargs):
    out = []
    for mol in oddt.toolkit.readfile(fmt, filepath_or_buffer):
        if skip_bad_mols and mol is None:
            continue  # add warning with number of skipped molecules
        if usecols:
            mol_data = dict((k, v) for k, v in mol.data.items() if k in usecols)
        else:
            mol_data = dict(mol.data)
        if molecule_column:
            mol_data[molecule_column] = mol
        if molecule_name:
            mol_data[molecule_name] = mol.title
        if smiles_column:
            mol_data[smiles_column] = mol.write('smi').split()[0]
        out.append(mol_data)
    return ChemDataFrame(out, **kwargs)


def read_sdf(filepath_or_buffer=None,
             usecols=None,
             molecule_column='mol',
             molecule_name='mol_name',
             smiles_column=None,
             skip_bad_mols=False,
             **kwargs):
    return _mol_dict_reader(fmt='sdf',
                            filepath_or_buffer=filepath_or_buffer,
                            usecols=usecols,
                            molecule_column=molecule_column,
                            molecule_name=molecule_name,
                            smiles_column=smiles_column,
                            skip_bad_mols=skip_bad_mols,
                            **kwargs)


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
