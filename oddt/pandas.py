""" Pandas extension for chemical analysis """
from __future__ import absolute_import
import pandas as pd

import oddt

pd.set_option("display.max_colwidth", 999999)

def _mol_dict_reader(fmt='sdf', filepath_or_buffer=None, names=None, molecule_column='mol', append_smiles=False):
    out = []
    for mol in oddt.toolkit.readfile(fmt, filepath_or_buffer):
        if names:
            mol_data = dict((k, v) for k, v in mol.data.items() if k in names)
        else:
            mol_data = dict(mol.data)
        if molecule_column:
            mol_data[molecule_column] = mol
        if append_smiles:
            mol_data['smiles'] = mol.smiles
        out.append(mol_data)
    return ChemDataFrame(out)

def read_sdf(filepath_or_buffer=None, names=None, molecule_column='mol', append_smiles=False):
    return _mol_dict_reader(fmt='sdf', filepath_or_buffer=filepath_or_buffer, names=None, molecule_column='mol', append_smiles=False)

class ChemSeries(pd.Series):
    @property
    def _constructor(self):
        return ChemSeries


class ChemDataFrame(pd.DataFrame):
    """

    Note:
    Thanks to: http://blog.snapdragon.cc/2015/05/05/subclass-pandas-dataframe-to-save-custom-attributes/
    """
    def to_html(self, *args, **kwargs):
        kwargs['escape'] = False
        return super(ChemDataFrame, self).to_html(*args, **kwargs)

    @property
    def _constructor(self):
        """ Force new class to be usead as sconstructor when slicing """
        return ChemDataFrame

    _constructor_sliced = ChemSeries
