"""ODDT pipeline framework for virtual screening"""
from __future__ import print_function
import sys
import csv
from os.path import dirname, isfile, join
from multiprocessing import Pool
from itertools import chain
from functools import partial
import warnings

import six
from six.moves import filter
# from joblib import Parallel, delayed

import oddt
from oddt.utils import is_molecule, compose_iter, chunker, method_caller
from oddt.scoring import scorer
from oddt.fingerprints import (InteractionFingerprint,
                               SimpleInteractionFingerprint,
                               dice)
from oddt.shape import usr, usr_cat, electroshape, usr_similarity


def _filter_smarts(mols, smarts, soft_fail=0):
    """Filter out molecule list (exhaustive) by smarts occurances. Allow molecules
    to pass if them match up to `soft_fail` matches.
    """
    out = []
    for mol in mols:
        if isinstance(smarts, six.string_types):
            compiled_smarts = oddt.toolkit.Smarts(smarts)
            if len(compiled_smarts.findall(mol)) == 0:
                out.append(mol)
        else:
            compiled_smarts = [oddt.toolkit.Smarts(s) for s in smarts]
            fail = 0
            for s in compiled_smarts:
                if len(s.findall(mol)) > 0:
                    fail += 1
                if fail > soft_fail:
                    break
            if fail <= soft_fail:
                out.append(mol)
    return out


def _filter(mols, expression, soft_fail=0):
    """Filter molecule by a generic expression, such as `mol.logp > 1`."""
    out = []
    for mol in mols:
        if isinstance(expression, list):
            fail = 0
            for e in expression:
                if not eval(e):
                    fail += 1
                if fail > soft_fail:
                    break
            if fail <= soft_fail:
                out.append(mol)
        else:
            if eval(expression):
                out.append(mol)
    return out


def _filter_similarity(mols, distance, generator, query_fps, cutoff):
    """Filter molecules by a certain distance to the reference fingerprints.
    User must supply distance funtion, FP generator, query FPs and cutoff."""
    return list(filter(
        lambda q: any(distance(generator(q), q_fp) >= float(cutoff)
                      for q_fp in query_fps), mols))


class virtualscreening:
    def __init__(self, n_cpu=-1, verbose=False, chunksize=100):
        """Virtual Screening pipeline stack

        Parameters
        ----------
        n_cpu: int (default=-1)
            The number of parallel procesors to use

        verbose: bool (default=False)
            Verbosity flag for some methods
        """
        self._pipe = []
        self._mol_feed = []
        self.n_cpu = n_cpu if n_cpu else -1
        self.num_input = 0
        self.num_output = 0
        self.verbose = verbose
        self.chunksize = chunksize

    def load_ligands(self, fmt, ligands_file, **kwargs):
        """Loads file with ligands.

        Parameters
        ----------
        file_type: string
            Type of molecular file

        ligands_file: string
            Path to a file, which is loaded to pipeline

        """
        if fmt == 'mol2' and oddt.toolkit.backend == 'ob':
            if 'opt' in kwargs:
                kwargs['opt']['c'] = None
            else:
                kwargs['opt'] = {'c': None}
        self._mol_feed = chain(self._mol_feed,
                               oddt.toolkit.readfile(fmt,
                                                     ligands_file,
                                                     **kwargs))

    def apply_filter(self, expression, soft_fail=0):
        """Filtering method, can use raw expressions (strings to be evaled
        in if statement, can use oddt.toolkit.Molecule methods, eg.
        `mol.molwt < 500`)
        Currently supported presets:
            * Lipinski Rule of 5 ('ro5' or 'l5')
            * Fragment Rule of 3 ('ro3')
            * PAINS filter ('pains')

        Parameters
        ----------
        expression: string or list of strings
            Expresion(s) to be used while filtering.

        soft_fail: int (default=0)
            The number of faulures molecule can have to pass filter, aka.
            soft-fails.
        """
        if expression in ['l5', 'ro5', 'ro3', 'pains']:
            # define presets
            # TODO: move presets to another config file
            # Lipinski rule of 5's
            if expression.lower() in ['l5', 'ro5']:
                self._pipe.append((partial(_filter,
                                           expression=['mol.molwt < 500',
                                                       'mol.HBA1 <= 10',
                                                       'mol.HBD <= 5',
                                                       'mol.logP <= 5'],
                                           soft_fail=soft_fail)))
            # Rule of three
            elif expression.lower() == 'ro3':
                self._pipe.append((partial(_filter,
                                           expression=['mol.molwt < 300',
                                                       'mol.HBA1 <= 3',
                                                       'mol.HBD <= 3',
                                                       'mol.logP <= 3'],
                                           soft_fail=soft_fail)))
            # PAINS filter
            elif expression.lower() == 'pains':
                pains_smarts = {}
                with open(join(dirname(__file__),
                               'filter', 'pains.smarts')) as pains_file:
                    csv_reader = csv.reader(pains_file, delimiter="\t")
                    for line in csv_reader:
                        if len(line) > 1:
                            pains_smarts[line[1][8:-2]] = line[0]
                self._pipe.append((partial(_filter_smarts,
                                           smarts=list(pains_smarts.values()),
                                           soft_fail=soft_fail)))
        else:
            self._pipe.append((partial(_filter,
                                       expression=expression,
                                       soft_fail=soft_fail)))

    def similarity(self, method, query, cutoff=0.9, protein=None):
        """Similarity filter. Supported structural methods:
            * ift: interaction fingerprints
            * sift: simple interaction fingerprints
            * usr: Ultrafast Shape recognition
            * usr_cat: Ultrafast Shape recognition, Credo Atom Types
            * electroshape: Electroshape, an USR method including partial charges

        Parameters
        ----------
        method: string
            Similarity method used to compare molecules. Avaiale methods:
            * `ifp` - interaction fingerprint (requires a receptor)
            * `sifp` - simple interaction fingerprint (requires a receptor)
            * `usr` - Ultrafast Shape Reckognition
            * `usr_cat` - USR, with CREDO atom types
            * `electroshape` - Electroshape, USR with moments representing
            partial charge

        query: oddt.toolkit.Molecule or list of oddt.toolkit.Molecule
            Query molecules to compare the pipeline to.

        cutoff: float
            Similarity cutoff for filtering molecules. Any similarity lower
            than it will be filtered out.

        protein: oddt.toolkit.Molecule (default = None)
            Protein for underling method. By default it's empty, but
            sturctural fingerprints need one.

        """
        if is_molecule(query):
            query = [query]

        # choose fp/usr and appropriate distance
        if method.lower() == 'ifp':
            gen = partial(InteractionFingerprint, protein=protein)
            dist = dice
        elif method.lower() == 'sifp':
            gen = partial(SimpleInteractionFingerprint, protein=protein)
            dist = dice
        elif method.lower() == 'usr':
            gen = usr
            dist = usr_similarity
        elif method.lower() == 'usr_cat':
            gen = usr_cat
            dist = usr_similarity
        elif method.lower() == 'electroshape':
            gen = electroshape
            dist = usr_similarity
        else:
            raise ValueError('Similarity filter "%s" is not supported.' % method)
        # generate FPs for query molecules once
        query_fps = [gen(q) for q in query]
        self._pipe.append(partial(_filter_similarity,
                                  distance=dist,
                                  generator=gen,  # same generator for pipe mols
                                  query_fps=query_fps,
                                  cutoff=cutoff))

    def dock(self, engine, protein, *args, **kwargs):
        """Docking procedure.

        Parameters
        ----------
        engine: string
            Which docking engine to use.

        Notes
        -----
        Additional parameters are passed directly to the engine.
        Following docking engines are supported:

        1. Audodock Vina (```engine="autodock_vina"```), see
        :class:`oddt.docking.autodock_vina`.
        """
        if engine.lower() == 'autodock_vina':
            from oddt.docking import autodock_vina
            engine = autodock_vina(protein, *args, **kwargs)
        else:
            raise ValueError('Docking engine %s was not implemented in ODDT'
                             % engine)
        self._pipe.append(partial(method_caller, engine, 'dock'))

    def score(self, function, protein=None, *args, **kwargs):
        """Scoring procedure compatible with any scoring function implemented
        in ODDT and other pickled SFs which are subclasses of
        `oddt.scoring.scorer`.

        Parameters
        ----------
        function: string
            Which scoring function to use.

        protein: oddt.toolkit.Molecule
            Default protein to use as reference

        Notes
        -----
        Additional parameters are passed directly to the scoring function.
        """
        if isinstance(protein, six.string_types):
            extension = protein.split('.')[-1]
            protein = next(oddt.toolkit.readfile(extension, protein))
            protein.protein = True
        elif protein is None:
            raise ValueError('Protein needs to be set for structure based '
                             'scoring')
        # trigger cache
        protein.atom_dict

        if isinstance(function, six.string_types):
            if isfile(function):
                sf = scorer.load(function)
                sf.set_protein(protein)
            elif function.lower().startswith('rfscore'):
                from oddt.scoring.functions.RFScore import rfscore
                new_kwargs = {}
                for bit in function.lower().split('_'):
                    if bit.startswith('pdbbind'):
                        new_kwargs['pdbbind_version'] = int(bit.replace('pdbbind', ''))
                    elif bit.startswith('v'):
                        new_kwargs['version'] = int(bit.replace('v', ''))
                sf = rfscore.load(**new_kwargs)
                sf.set_protein(protein)
            elif function.lower().startswith('nnscore'):
                from oddt.scoring.functions.NNScore import nnscore
                new_kwargs = {}
                for bit in function.lower().split('_'):
                    if bit.startswith('pdbbind'):
                        new_kwargs['pdbbind_version'] = int(bit.replace('pdbbind', ''))
                sf = nnscore.load(**new_kwargs)
                sf.set_protein(protein)
            elif function.lower().startswith('plec'):
                from oddt.scoring.functions.PLECscore import PLECscore
                new_kwargs = {}
                for bit in function.lower().split('_'):
                    if bit.startswith('pdbbind'):
                        new_kwargs['pdbbind_version'] = int(bit.replace('pdbbind', ''))
                    elif bit.startswith('plec'):
                        new_kwargs['version'] = bit.replace('plec', '')
                    elif bit.startswith('p'):
                        new_kwargs['depth_protein'] = int(bit.replace('p', ''))
                    elif bit.startswith('l'):
                        new_kwargs['depth_ligand'] = int(bit.replace('l', ''))
                    elif bit.startswith('s'):
                        new_kwargs['size'] = int(bit.replace('s', ''))
                sf = PLECscore.load(**new_kwargs)
                sf.set_protein(protein)
            elif function.lower() == 'autodock_vina':
                from oddt.docking import autodock_vina
                sf = autodock_vina(protein, *args, **kwargs)
                sf.set_protein(protein)
            else:
                raise ValueError('Scoring Function %s was not implemented in '
                                 'ODDT' % function)
        else:
            if isinstance(function, scorer):
                sf = function
                sf.set_protein(protein)
            else:
                raise ValueError('Supplied object "%s" is not an ODDT scoring '
                                 'funtion' % function.__name__)
        self._pipe.append(partial(method_caller, sf, 'predict_ligands'))

    def fetch(self):
        """A method to exhaust the pipeline. Itself it is lazy (a generator)"""
        chunk_feed = chunker(self._mol_feed, chunksize=self.chunksize)
        # get first chunk and check if it is saturated
        try:
            first_chunk = next(chunk_feed)
        except StopIteration:
            raise StopIteration('There are no molecules loaded to the pipeline.')

        if len(first_chunk) == 0:
            warnings.warn('There is **zero** molecules at the output of the VS'
                          ' pipeline. Output file will be empty.')
        elif len(first_chunk) < self.chunksize and self.n_cpu > 1:
            warnings.warn('The chunksize (%i) seams to be to large.'
                          % self.chunksize)

            # use methods multithreading when we have less molecules than cores
            if len(first_chunk) < self.n_cpu:
                warnings.warn('Falling back to sub-methods multithreading as '
                              'the number of molecules is less than cores '
                              '(%i < %i)' % (len(first_chunk),  self.n_cpu))
                for func in self._pipe:
                    if hasattr(func, 'n_cpu'):
                        func.n_cpu = self.n_cpu
                    elif hasattr(func, 'n_jobs'):
                        func.n_jobs = self.n_cpu
                    elif isinstance(func, partial):
                        for func2 in func.args:
                            if hasattr(func2, 'n_cpu'):
                                func2.n_cpu = self.n_cpu
                            elif hasattr(func2, 'n_jobs'):
                                func2.n_jobs = self.n_cpu
                # turn off VS multiprocessing
                self.n_cpu = 1

        # TODO add some verbosity or progress bar
        if self.n_cpu != 1:
            out = (Pool(self.n_cpu if self.n_cpu > 0 else None)
                   .imap(partial(compose_iter, funcs=self._pipe),
                         (chunk for chunk in chain([first_chunk], chunk_feed))))
        else:
            out = (compose_iter(chunk, self._pipe)
                   for chunk in chain([first_chunk], chunk_feed))

        # FIXME use joblib version as soon as it gets return_generator merged
        # out = Parallel(n_jobs=self.n_cpu)(
        #     delayed(compose_iter)(chunk, self._pipe)
        #     for chunk in chain([first_chunk], chunk_feed))

        # merge chunks into one iterable
        return chain.from_iterable(out)

    def write(self, fmt, filename, csv_filename=None, **kwargs):
        """Outputs molecules to a file

        Parameters
        ----------
        file_type: string
            Type of molecular file

        ligands_file: string
            Path to a output file

        csv_filename: string
            Optional path to a CSV file
        """
        if fmt == 'mol2' and oddt.toolkit.backend == 'ob':
            if 'opt' in kwargs:
                kwargs['opt']['c'] = None
            else:
                kwargs['opt'] = {'c': None}
        output_mol_file = oddt.toolkit.Outputfile(fmt,
                                                  filename,
                                                  overwrite=True,
                                                  **kwargs)
        if csv_filename:
            f = open(csv_filename, 'w')
            csv_file = None
        for mol in self.fetch():
            if csv_filename:
                data = mol.data.to_dict()
                # filter some internal data
                blacklist_keys = ['OpenBabel Symmetry Classes',
                                  'MOL Chiral Flag',
                                  'PartialCharges',
                                  'TORSDO',
                                  'REMARK']
                for b in blacklist_keys:
                    if b in data:
                        del data[b]
                if len(data) > 0:
                    data['name'] = mol.title
                else:
                    print("There is no data to write in CSV file",
                          file=sys.stderr)
                    return False
                if csv_file is None:
                    csv_file = csv.DictWriter(f, data.keys(), **kwargs)
                    csv_file.writeheader()
                csv_file.writerow(data)
            # write ligand
            output_mol_file.write(mol)
        output_mol_file.close()
        if csv_filename:
            f.close()
        # TODO keep_pipe using hdf5 to store molecules
        if isfile(filename):
            kwargs.pop('overwrite', None)  # this argument is unsupported
            self.load_ligands(fmt, filename, **kwargs)

    def write_csv(self, csv_filename, fields=None, keep_pipe=False, **kwargs):
        """Outputs molecules to a csv file

        Parameters
        ----------
        csv_filename: string
            Optional path to a CSV file

        fields: list (default None)
            List of fields to save in CSV file

        keep_pipe: bool (default=False)
            If set to True, the ligand pipe is sustained.
        """
        if hasattr(csv_filename, 'write'):
            f = csv_filename
        else:
            f = open(csv_filename, 'w')
        csv_file = None
        for mol in self.fetch():
            data = mol.data.to_dict()
            # filter some internal data
            blacklist_keys = ['OpenBabel Symmetry Classes',
                              'MOL Chiral Flag',
                              'PartialCharges',
                              'TORSDO',
                              'REMARK']
            for b in blacklist_keys:
                if b in data:
                    del data[b]
            if len(data) > 0:
                data['name'] = mol.title
            else:
                print("There is no data to write in CSV file", file=sys.stderr)
                return False
            if csv_file is None:
                csv_file = csv.DictWriter(f, fields or data.keys(),
                                          extrasaction='ignore', **kwargs)
                csv_file.writeheader()
            csv_file.writerow(data)
            # TODO keep_pipe using hdf5 to store molecules
        f.close()
