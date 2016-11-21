""" Pandas extension for chemical analysis """
from __future__ import absolute_import
from collections import deque
from six import BytesIO
import pandas as pd

import oddt

pd.set_option("display.max_colwidth", 999999)
image_backend = 'png'  # png or svg


def _mol_reader(fmt='sdf',
                filepath_or_buffer=None,
                usecols=None,
                molecule_column='mol',
                molecule_name_column='mol_name',
                smiles_column=None,
                skip_bad_mols=False,
                chunksize=None,
                **kwargs):
    """Universal reading function for private use.

    .. versionadded:: 0.3

    Parameters
    ----------
        fmt : string
            The format of molecular file

        filepath_or_buffer : string or None
            File path

        usecols : list or None, optional (default=None)
            A list of columns to read from file. If None then all available
            fields are read.

        molecule_column : string or None, optional (default='mol')
            Name of molecule column. If None the molecules will be skipped and
            the reading will be speed up significantly.

        molecule_name_column : string or None, optional (default='mol_name')
            Column name which will contain molecules' title/name. Column is
            skipped when set to None.

        smiles_column  : string or None, optional (default=None)
            Column name containg molecules' SMILES, by default it is disabled.

        skip_bad_mols : bool, optional (default=False)
            Switch to skip empty (bad) molecules. Useful for RDKit, which Returns
            None if molecule can not sanitize.

        chunksize : int or None, optional (default=None)
            Size of chunk to return. If set to None whole set is returned.

    Returns
    -------
        chunk :
            A `ChemDataFrame` containg `chunksize` molecules.

    """
    # capture options for reader
    reader_kwargs = {}
    if 'opt' in kwargs:
        reader_kwargs['opt'] = kwargs.pop('opt')
    if 'sanitize' in kwargs:
        reader_kwargs['sanitize'] = kwargs.pop('sanitize')

    # when you dont read molecules you can skip parsing them
    if molecule_column is None:
        if oddt.toolkit.backend == 'ob' and fmt == 'sdf':
            if 'opt' in reader_kwargs:
                reader_kwargs['opt']['P'] = None
            else:
                reader_kwargs['opt'] = {'P': None}
        elif oddt.toolkit.backend == 'rdk':
            reader_kwargs['sanitize'] = False

    chunk = []
    for n, mol in enumerate(oddt.toolkit.readfile(fmt, filepath_or_buffer, **reader_kwargs)):
        if skip_bad_mols and mol is None:
            continue  # add warning with number of skipped molecules
        if usecols is None:
            mol_data = mol.data.to_dict()
        else:
            mol_data = dict((k, mol.data[k]) for k in usecols)

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


def _mol_writer(dataframe,
                fmt='sdf',
                filepath_or_buffer=None,
                update_properties=True,
                molecule_column='mol',
                columns=None):
    out = oddt.toolkit.Outputfile(fmt, filepath_or_buffer, overwrite=True)
    for ix, row in dataframe.iterrows():
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


def read_sdf(filepath_or_buffer=None,
             usecols=None,
             molecule_column='mol',
             molecule_name_column='mol_name',
             smiles_column=None,
             skip_bad_mols=False,
             chunksize=None,
             **kwargs):
    """Read SDF/MDL multi molecular file to ChemDataFrame

    .. versionadded:: 0.3

    Parameters
    ----------
        filepath_or_buffer : string or None
            File path

        usecols : list or None, optional (default=None)
            A list of columns to read from file. If None then all available
            fields are read.

        molecule_column : string or None, optional (default='mol')
            Name of molecule column. If None the molecules will be skipped and
            the reading will be speed up significantly.

        molecule_name_column : string or None, optional (default='mol_name')
            Column name which will contain molecules' title/name. Column is
            skipped when set to None.

        smiles_column  : string or None, optional (default=None)
            Column name containg molecules' SMILES, by default it is disabled.

        skip_bad_mols : bool, optional (default=False)
            Switch to skip empty (bad) molecules. Useful for RDKit, which Returns
            None if molecule can not sanitize.

        chunksize : int or None, optional (default=None)
            Size of chunk to return. If set to None whole set is returned.

    Returns
    -------
        result :
            A `ChemDataFrame` containg all molecules if `chunksize` is None
            or genrerator of `ChemDataFrame` with `chunksize` molecules.

    """
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
    """Read Mol2 multi molecular file to ChemDataFrame. UCSF Dock 6 comments
    style is supported, i.e. `#### var_name: value` before molecular block.

    .. versionadded:: 0.3

    Parameters
    ----------
        filepath_or_buffer : string or None
            File path

        usecols : list or None, optional (default=None)
            A list of columns to read from file. If None then all available
            fields are read.

        molecule_column : string or None, optional (default='mol')
            Name of molecule column. If None the molecules will be skipped and
            the reading will be speed up significantly.

        molecule_name_column : string or None, optional (default='mol_name')
            Column name which will contain molecules' title/name. Column is
            skipped when set to None.

        smiles_column  : string or None, optional (default=None)
            Column name containg molecules' SMILES, by default it is disabled.

        skip_bad_mols : bool, optional (default=False)
            Switch to skip empty (bad) molecules. Useful for RDKit, which Returns
            None if molecule can not sanitize.

        chunksize : int or None, optional (default=None)
            Size of chunk to return. If set to None whole set is returned.

    Returns
    -------
        result :
            A `ChemDataFrame` containg all molecules if `chunksize` is None
            or genrerator of `ChemDataFrame` with `chunksize` molecules.

    """
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
    """Pandas Series modified to adapt `oddt.toolkit.Molecule` objects and apply
    molecular methods easily.
    """

    @property
    def _constructor(self):
        """ Force new class to be usead as constructor """
        return ChemSeries

    @property
    def _constructor_expanddim(self):
        """ Force new class to be usead as constructor when expandig dims """
        return ChemDataFrame


class ChemDataFrame(pd.DataFrame):
    """Chemical DataFrame object, which contains molecules column of
    `oddt.toolkit.Molecule` objects. Rich display of moleucles (2D) is available
    in iPython Notebook. Additional `to_sdf` and `to_mol2` methods make writing
    to molecular formats easy.

    Note:
    Thanks to: http://blog.snapdragon.cc/2015/05/05/subclass-pandas-dataframe-to-save-custom-attributes/
    """

    def to_sdf(self,
               filepath_or_buffer=None,
               update_properties=True,
               molecule_column='mol',
               columns=None):
        _mol_writer(self,
                    fmt='sdf',
                    filepath_or_buffer=filepath_or_buffer,
                    update_properties=update_properties,
                    molecule_column=molecule_column,
                    columns=columns)

    def to_mol2(self,
                filepath_or_buffer=None,
                update_properties=True,
                molecule_column='mol',
                columns=None):
        _mol_writer(self,
                    fmt='mol2',
                    filepath_or_buffer=filepath_or_buffer,
                    update_properties=update_properties,
                    molecule_column=molecule_column,
                    columns=columns)

    def to_html(self, *args, **kwargs):
        kwargs['escape'] = False
        return super(ChemDataFrame, self).to_html(*args, **kwargs)

    def to_excel(self, *args, **kwargs):
        columns = kwargs['columns'] if 'columns' in kwargs else self.columns.tolist()
        molecule_column = 'mol'  # TODO: Get appropriate molecule_column
        molecule_column_idx = columns.index(molecule_column)
        size = (200, 200)
        excel_writer = pd.ExcelWriter(args[0], engine='xlsxwriter')

        super(ChemDataFrame, self).to_excel(excel_writer, *args[1:], **kwargs)

        sheet = excel_writer.sheets['Sheet1']  # TODO: Get appropriate sheet name
        sheet.set_column(molecule_column_idx + 1, molecule_column_idx + 1, width=size[1] / 4.5)
        for i, mol in enumerate(self[molecule_column]):
            img = BytesIO()
            img.write(mol.write('png').encode('utf-8', errors='surrogateescape'))
            mol.write('png', 'tmp.png', overwrite=True)
            sheet.write_string(i + 1, molecule_column_idx + 1, "")
            sheet.insert_image(i + 1,
                               molecule_column_idx + 1,
                               'tmp.png',
                               {'image_data': img,
                                'positioning': 2,
                                'x_offset': 1,
                                'y_offset': 1})
            sheet.set_row(i + 1, height=size[0] * (1.15))  # TODO: get actual size
        excel_writer.save()

    @property
    def _constructor(self):
        """ Force new class to be usead as constructor """
        return ChemDataFrame

    @property
    def _constructor_sliced(self):
        """ Force new class to be usead as constructor when slicing """
        return ChemSeries

    @property
    def _constructor_expanddim(self):
        """ Force new class to be usead as constructor when expandig dims """
        return ChemPanel
# Copy some docscrings from upstream classes
for method in ['to_html', 'to_excel']:
    try:
        getattr(ChemDataFrame, method).__doc__ = getattr(pd.DataFrame, method).__doc__
    except AttributeError:  # Python 2 compatible
        getattr(ChemDataFrame, method).__func__.__doc__ = getattr(pd.DataFrame, method).__func__.__doc__


class ChemPanel(pd.Panel):
    """Modified `pandas.Panel` to adopt higher dimension data than
    `ChemDataFrame`. Main purpose is to store molecular fingerprints in one
    column and keep 2D numpy array underneath.

    """

    @property
    def _constructor(self):
        """ Force new class to be usead as constructor """
        return ChemPanel

    @property
    def _constructor_sliced(self):
        """ Force new class to be usead as constructor when slicing """
        return ChemDataFrame
