""" Pandas extension for chemical analysis """
from __future__ import absolute_import
from collections import deque
from six import BytesIO, StringIO
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
            mol_data[smiles_column] = mol.smiles
        chunk.append(mol_data)
        if chunksize and (n + 1) % chunksize == 0:
            chunk_frm = ChemDataFrame(chunk, **kwargs)
            chunk_frm._molecule_column = molecule_column
            yield chunk_frm
            chunk = []
    if chunk or chunksize is None:
        chunk_frm = ChemDataFrame(chunk, **kwargs)
        chunk_frm._molecule_column = molecule_column
        yield chunk_frm


def _mol_writer(data,
                fmt='sdf',
                filepath_or_buffer=None,
                update_properties=True,
                molecule_column=None,
                columns=None):
    if filepath_or_buffer is None:
        out = StringIO()
    elif hasattr(filepath_or_buffer, 'write'):
        out = filepath_or_buffer
    else:
        out = oddt.toolkit.Outputfile(fmt, filepath_or_buffer, overwrite=True)
    if isinstance(data, pd.DataFrame):
        molecule_column = molecule_column or data._molecule_column
        for ix, row in data.iterrows():
            mol = row[molecule_column].clone
            if update_properties:
                new_data = row.to_dict()
                del new_data[molecule_column]
                mol.data.update(new_data)
            if columns:
                for k in mol.data.keys():
                    if k not in columns:
                        del mol.data[k]
            if filepath_or_buffer is None or hasattr(filepath_or_buffer, 'write'):
                out.write(mol.write(fmt))
            else:
                out.write(mol)
    elif isinstance(data, pd.Series):
        for mol in data:
            if filepath_or_buffer is None or hasattr(filepath_or_buffer, 'write'):
                out.write(mol.write(fmt))
            else:
                out.write(mol)
    if filepath_or_buffer is None:
        return out.getvalue()
    elif not hasattr(filepath_or_buffer, 'write'):  # dont close foreign buffer
        out.close()


def read_csv(*args, **kwargs):
    """ TODO: Support Chunks """
    smiles_to_molecule = kwargs.pop('smiles_to_molecule', None)
    molecule_column = kwargs.pop('molecule_column', 'mol')
    data = pd.read_csv(*args, **kwargs)
    if smiles_to_molecule is not None:
        data[molecule_column] = data[smiles_to_molecule].map(lambda x: oddt.toolkit.readstring('smi', x))
    return data


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
    def __lt__(self, other):
        """ Substructure searching.
        `chemseries < mol`: are molecules in series substructures of a `mol`
        """
        assert(isinstance(other, oddt.toolkit.Molecule))
        assert(isinstance(self[0], oddt.toolkit.Molecule))
        return self.map(lambda x: oddt.toolkit.Smarts(x.smiles).match(other))

    def __gt__(self, other):
        """ Substructure searching.
        `chemseries > mol`: is `mol` a substructure of molecules in series
        """
        assert(isinstance(other, oddt.toolkit.Molecule))
        assert(isinstance(self[0], oddt.toolkit.Molecule))
        smarts = oddt.toolkit.Smarts(other.smiles)
        return self.map(lambda x: smarts.match(x))

    def __or__(self, other):
        """ Tanimoto coefficient """
        if (isinstance(self[0], oddt.toolkit.Fingerprint) and
           isinstance(other, oddt.toolkit.Fingerprint)):
            return self.map(lambda x: x | other)
        else:
            return super(ChemSeries, self).__or__(other)

    def calcfp(self, *args, **kwargs):
        assert(isinstance(self[0], oddt.toolkit.Molecule))
        return self.map(lambda x: x.calcfp(*args, **kwargs))

    def to_smiles(self, filepath_or_buffer=None):
        return _mol_writer(self, fmt='smi', filepath_or_buffer=filepath_or_buffer)

    def to_sdf(self, filepath_or_buffer=None):
        return _mol_writer(self, fmt='sdf', filepath_or_buffer=filepath_or_buffer)

    def to_mol2(self, filepath_or_buffer=None):
        return _mol_writer(self, fmt='mol2', filepath_or_buffer=filepath_or_buffer)

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
    _metadata = ['_molecule_column']
    _molecule_column = None

    def to_sdf(self,
               filepath_or_buffer=None,
               update_properties=True,
               molecule_column=None,
               columns=None):
        molecule_column = molecule_column or self._molecule_column
        return _mol_writer(self,
                           filepath_or_buffer=filepath_or_buffer,
                           update_properties=update_properties,
                           fmt='sdf',
                           molecule_column=molecule_column,
                           columns=columns)

    def to_mol2(self,
                filepath_or_buffer=None,
                update_properties=True,
                molecule_column='mol',
                columns=None):
        molecule_column = molecule_column or self._molecule_column
        return _mol_writer(self,
                           fmt='mol2',
                           filepath_or_buffer=filepath_or_buffer,
                           update_properties=update_properties,
                           molecule_column=molecule_column,
                           columns=columns)

    def to_html(self, *args, **kwargs):
        kwargs['escape'] = False
        return super(ChemDataFrame, self).to_html(*args, **kwargs)

    def to_csv(self, *args, **kwargs):
        if self._molecule_column:
            frm_copy = self.copy(deep=False)
            frm_copy[self._molecule_column] = frm_copy[self._molecule_column].map(lambda x: x.smiles).values
            return super(ChemDataFrame, frm_copy).to_csv(*args, **kwargs)
        else:
            return super(ChemDataFrame, self).to_csv(*args, **kwargs)

    def to_excel(self, *args, **kwargs):
        columns = kwargs['columns'] if 'columns' in kwargs else self.columns.tolist()
        if 'molecule_column' in kwargs:
            molecule_column = kwargs['molecule_column']
        else:
            molecule_column = self._molecule_column
        molecule_column_idx = columns.index(molecule_column)
        size = kwargs.pop('size') if 'size' in kwargs else (200, 200)
        excel_writer = pd.ExcelWriter(args[0], engine='xlsxwriter')

        super(ChemDataFrame, self).to_excel(excel_writer, *args[1:], **kwargs)

        sheet = excel_writer.sheets['Sheet1']  # TODO: Get appropriate sheet name
        sheet.set_column(molecule_column_idx + 1, molecule_column_idx + 1, width=size[1] / 6.)
        for i, mol in enumerate(self[molecule_column]):
            img = BytesIO()
            png = mol.clone.write('png', size=size)
            if type(png) is str:
                png = png.encode('utf-8', errors='surrogateescape')
            img.write(png)
            sheet.write_string(i + 1, molecule_column_idx + 1, "")
            sheet.insert_image(i + 1,
                               molecule_column_idx + 1,
                               'dummy',
                               {'image_data': img,
                                'positioning': 2,
                                'x_offset': 1,
                                'y_offset': 1})
            sheet.set_row(i + 1, height=size[0])
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
for method in ['to_html', 'to_csv', 'to_excel']:
    try:
        getattr(ChemDataFrame, method).__doc__ = getattr(pd.DataFrame, method).__doc__
    except AttributeError:  # Python 2 compatible
        getattr(ChemDataFrame, method).__func__.__doc__ = getattr(pd.DataFrame, method).__func__.__doc__


class ChemPanel(pd.Panel):
    """Modified `pandas.Panel` to adopt higher dimension data than
    `ChemDataFrame`. Main purpose is to store molecular fingerprints in one
    column and keep 2D numpy array underneath.

    """
    _metadata = ['_molecule_column']
    _molecule_column = None

    @property
    def _constructor(self):
        """ Force new class to be usead as constructor """
        return ChemPanel

    @property
    def _constructor_sliced(self):
        """ Force new class to be usead as constructor when slicing """
        return ChemDataFrame
