"""ODDT pipeline framework for virtual screening"""
from __future__ import print_function
import sys
import csv
import six
from six.moves import filter
from os.path import dirname, isfile
# from multiprocessing.dummy import Pool # threading
from multiprocessing import Pool  # process
from itertools import chain
from functools import partial
import warnings

from oddt import toolkit
from oddt.scoring import scorer
from oddt.fingerprints import (InteractionFingerprint,
                               SimpleInteractionFingerprint,
                               dice)
from oddt.shape import usr, usr_cat, electroshape, usr_similarity


def _parallel_helper(obj, methodname, kwargs):
    """Private helper to workaround Python 2 methods picklinh limitations"""
    return getattr(obj, methodname)(**kwargs)


class virtualscreening:
    def __init__(self, n_cpu=-1, verbose=False):
        """Virtual Screening pipeline stack

        Parameters
        ----------
            n_cpu: int (default=-1)
                The number of parallel procesors to use

            verbose: bool (default=False)
                Verbosity flag for some methods
        """
        self._pipe = None
        self.n_cpu = n_cpu if n_cpu else -1
        self.num_input = 0
        self.num_output = 0
        self.verbose = verbose

    def load_ligands(self, fmt, ligands_file, *args, **kwargs):
        """Loads file with ligands.

        Parameters
        ----------
            file_type: string
                Type of molecular file

            ligands_file: string
                Path to a file, which is loaded to pipeline

        """
        if fmt == 'mol2' and toolkit.backend == 'ob':
            if 'opt' in kwargs:
                kwargs['opt']['c'] = None
            else:
                kwargs['opt'] = {'c': None}
        new_pipe = self._ligand_pipe(toolkit.readfile(fmt, ligands_file, *args,
                                                      **kwargs))
        self._pipe = chain(self._pipe, new_pipe) if self._pipe else new_pipe

    def _ligand_pipe(self, ligands):
        for mol in ligands:
            if mol:
                self.num_input += 1
                yield mol

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
                self._pipe = self._filter(self._pipe,
                                          ['mol.molwt < 500',
                                           'mol.HBA1 <= 10',
                                           'mol.HBD <= 5',
                                           'mol.logP <= 5'],
                                          soft_fail=soft_fail)
            # Rule of three
            elif expression.lower() in ['ro3']:
                self._pipe = self._filter(self._pipe,
                                          ['mol.molwt < 300',
                                           'mol.HBA1 <= 3',
                                           'mol.HBD <= 3',
                                           'mol.logP <= 3'],
                                          soft_fail=soft_fail)
            # PAINS filter
            elif expression.lower() in ['pains']:
                pains_smarts = {}
                with open(dirname(__file__)+'/filter/pains.smarts') as pains_file:
                    csv_reader = csv.reader(pains_file, delimiter="\t")
                    for line in csv_reader:
                        if len(line) > 1:
                            pains_smarts[line[1][8:-2]] = line[0]
                self._pipe = self._filter_smarts(self._pipe,
                                                 pains_smarts.values(),
                                                 soft_fail=soft_fail)
        else:
            self._pipe = self._filter(self._pipe, expression, soft_fail=soft_fail)

    def _filter_smarts(self, pipe, smarts, soft_fail=0):
        for mol in pipe:
            if isinstance(smarts, six.string_types):
                compiled_smarts = toolkit.Smarts(smarts)
                if len(compiled_smarts.findall(mol)) == 0:
                    yield mol
            else:
                compiled_smarts = [toolkit.Smarts(s) for s in smarts]
                fail = 0
                for s in compiled_smarts:
                    if len(s.findall(mol)) > 0:
                        fail += 1
                    if fail > soft_fail:
                        break
                if fail <= soft_fail:
                    yield mol

    def _filter(self, pipe, expression, soft_fail=0):
        for mol in pipe:
            if isinstance(expression, list):
                fail = 0
                for e in expression:
                    if not eval(e):
                        fail += 1
                    if fail > soft_fail:
                        break
                if fail <= soft_fail:
                    yield mol
            else:
                if eval(expression):
                    yield mol

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
        if isinstance(query, toolkit.Molecule):
            query = [query]

        # choose fp/usr and appropriate distance
        if method.lower() == 'ifp':
            gen = InteractionFingerprint
            dist = dice
        elif method.lower() == 'sifp':
            gen = SimpleInteractionFingerprint
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
        query_fps = [(gen(q) if protein is None else gen(q, protein))
                     for q in query]
        self._pipe = filter(lambda q: any(dist(gen(q) if protein is None
                                               else gen(q, protein),
                                               q_fp) >= float(cutoff)
                                          for q_fp in query_fps),
                            self._pipe)

    def dock(self, engine, protein, *args, **kwargs):
        """Docking procedure.

        Parameters
        ----------
            engine: string
                Which docking engine to use.

        Returns
        -------
            None

        Note
        ----
            Additional parameters are passed directly to the engine.
            Following docking engines are supported:

            1. Audodock Vina (``engine="autodock_vina"``), see
            `oddt.docking.autodock_vina`.
        """
        if engine.lower() == 'autodock_vina':
            from oddt.docking import autodock_vina
            engine = autodock_vina(protein, *args, **kwargs)
        else:
            raise ValueError('Docking engine %s was not implemented in ODDT'
                             % engine)
        if self.n_cpu != 1:
            _parallel_helper_partial = partial(_parallel_helper, engine, 'dock')
            docking_results = (
                Pool(self.n_cpu if self.n_cpu > 0 else None)
                .imap(_parallel_helper_partial, ({'ligands': lig}
                                                 for lig in self._pipe)))
        else:
            docking_results = (engine.dock(lig) for lig in self._pipe)
        self._pipe = chain.from_iterable(docking_results)

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

        Note
        ----
            Additional parameters are passed directly to the scoring function.
        """
        if isinstance(protein, six.string_types):
            extension = protein.split('.')[-1]
            protein = six.next(toolkit.readfile(extension, protein))
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
        if self.n_cpu != 1:
            _parallel_helper_partial = partial(_parallel_helper, sf,
                                               'predict_ligand')
            self._pipe = (Pool(self.n_cpu if self.n_cpu > 0 else None)
                          .imap(_parallel_helper_partial, ({'ligand': lig}
                                                           for lig in self._pipe),
                                chunksize=100))
        else:
            self._pipe = sf.predict_ligands(self._pipe)

    def fetch(self):
        """A method to exhaust the pipeline. Itself it is lazy (a generator)"""
        for mol in self._pipe:
            self.num_output += 1
            if self.verbose and self.num_input % 100 == 0:
                print("Passed: %i (%.2f%%)\tTotal: %i\r" %
                      (self.num_output,
                       float(self.num_output) / float(self.num_input) * 100,
                       self.num_input),
                      file=sys.stderr, end=" ")
            yield mol
        if self.verbose:
            print('', file=sys.stderr)
        if self.num_output == 0:
            warnings.warn('There is **zero** molecules at the output of the VS'
                          ' pipeline. Output file will be empty.')

    # Consume the pipe
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
        if fmt == 'mol2' and toolkit.backend == 'ob':
            if 'opt' in kwargs:
                kwargs['opt']['c'] = None
            else:
                kwargs['opt'] = {'c': None}
        output_mol_file = toolkit.Outputfile(fmt, filename, overwrite=True, **kwargs)
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
#        if 'keep_pipe' in kwargs and kwargs['keep_pipe']:
        if isfile(filename):
            kwargs.pop('overwrite', None)  # this argument is unsupported in readfile
            self._pipe = toolkit.readfile(fmt, filename, **kwargs)

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
            if keep_pipe:
                # write ligand using pickle
                pass
        f.close()
