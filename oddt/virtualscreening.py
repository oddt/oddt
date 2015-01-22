"""ODDT pipeline framework for virtual screening"""
import csv
from multiprocessing.dummy import Pool
from oddt import toolkit

def _parallel_helper(args):
    """Private helper to workaround Python 2 pickle limitations to paralelize methods"""
    obj, methodname, arg = args
    return getattr(obj, methodname)(**arg)

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
        self.n_cpu = n_cpu
        self.num_input = 0
        self.num_output = 0
        self.verbose = verbose
        # setup pool
        self._pool = Pool(n_cpu if n_cpu > 0 else None)
        
    def load_ligands(self, file_type, ligands_file):
        """Loads file with ligands.
        
        Parameters
        ----------
            file_type: string
                Type of molecular file
            
            ligands_file: string
                Path to a file, which is loaded to pipeline
        
        """
        self._pipe = self._ligand_pipe(toolkit.readfile(file_type, ligands_file))
    
    def _ligand_pipe(self, ligands):
        for n, mol in enumerate(ligands):
            self.num_input = n+1 
            yield mol
    
    def apply_filter(self, expression, filter_type='expression', soft_fail = 0):
        """Filtering method, can use raw expressions (strings to be evaled in if statement, can use oddt.toolkit.Molecule methods, eg. 'mol.molwt < 500')
        Currently supported presets:
            * Lipinski Rule of 5 ('r5' or 'l5')
            * Fragment Rule of 3 ('r3')
        
        Parameters
        ----------
            expression: string or list of strings
                Expresion(s) to be used while filtering.
            
            filter_type: 'expression' or 'preset' (default='expression')
                Specify filter type: 'expression' or 'preset'. Default strings are treated as expressions.
            
            soft_fail: int (default=0)
                The number of faulures molecule can have to pass filter, aka. soft-fails.
        """
        if filter_type == 'expression':
            self._pipe = self._filter(self._pipe, expression, soft_fail = soft_fail)
        elif filter_type == 'preset':
            # define presets
            # TODO: move presets to another config file
            # Lipinski rule of 5's
            if expression.lower() in ['l5', 'ro5']:
                self._pipe = self._filter(self._pipe, ['mol.molwt < 500', 'mol.calcdesc(["HBA1"])["HBA1"] <= 10', 'mol.calcdesc(["HBD"])["HBD"] <= 5', 'mol.calcdesc(["logP"])["logP"] <= 5'], soft_fail = soft_fail)
            # Rule of three
            elif expression.lower() in ['ro3']:
                self._pipe = self._filter(self._pipe, ['mol.molwt < 300', 'mol.calcdesc(["HBA1"])["HBA1"] <= 3', 'mol.calcdesc(["HBD"])["HBD"] <= 3', 'mol.calcdesc(["logP"])["logP"] <= 3'], soft_fail = soft_fail)
            # PAINS filter
            elif expression.lower() in ['pains']:
                pains_smarts = {}
                with open(dirname(__file__)+'filter/pains.smarts') as pains_file:
                    csv_reader = csv.reader(pains_file, delimiter="\t")
                    for line in csv_reader:
                        if len(line) > 1:
                            pains_smarts[line[1][8:-2]] = line[0]
                self._pipe = self._filter_smarts(self._pipe, pains_smarts.values(), soft_fail = soft_fail)
    
    def _filter_smarts(self, pipe, smarts, soft_fail = 0):
        for mol in pipe:
            if type(smarts) is list:
                compiled_smarts = [toolkit.Smarts(s) for s in smarts]
                fail = 0
                for s in compiled_smarts:
                    if len(s.findall(mol)) > 0:
                        fail += 1
                    if fail > soft_fail:
                        break
                if fail <= soft_fail:
                    yield mol
            else:
                compiled_smarts = toolkit.Smarts(smarts)
                if len(compiled_smiles.findall(mol)) == 0:
                    yield mol
    
    def _filter(self, pipe, expression, soft_fail = 0):
        for mol in pipe:
            if type(expression) is list:
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
    
    def dock(self, engine, protein, *args, **kwargs):
        """Docking procedure.
        
        Parameters
        ----------
            engine: string
                Which docking engine to use.
        
        Note
        ----
            Additional parameters are passed directly to the engine.
        """
        if engine.lower() == 'autodock_vina':
            from .docking.autodock_vina import autodock_vina
            engine = autodock_vina(protein, *args, **kwargs)
        else:
            raise ValueError('Docking engine %s was not implemented in ODDT' % engine)
        def _iter_conf(results):
            """ Generator to go through docking results, and put them to pipe """
            for confs in results:
                for conf in confs:
                    yield conf
        if self.n_cpu != 1:
            docking_results = self._pool.imap(_parallel_helper, ((engine, "dock", {'ligands':lig, 'single': True}) for lig in self._pipe))
        else:
            docking_results = (engine.dock(lig, single=True) for lig in self._pipe)
        self._pipe = _iter_conf(docking_results)
        
    def score(self, function, protein, *args, **kwargs):
        """Scoring procedure.
        
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
        if type(protein) is str:
            extension = protein.split('.')[-1]
            protein = toolkit.readfile(extension, protein).next()
            protein.protein = True
        
        if function.lower() == 'rfscore':
            from .scoring.functions.RFScore import rfscore
            sf = rfscore.load()
            sf.set_protein(protein)
        elif function.lower() == 'nnscore':
            from .scoring.functions.NNScore import nnscore
            sf = nnscore.load()
            sf.set_protein(protein)
        else:
            raise ValueError('Scoring Function %s was not implemented in ODDT' % function)
        if self.n_cpu != 1:
            self._pipe = self._pool.imap(_parallel_helper, ((sf, 'predict_ligand', {'ligand': lig}) for lig in self._pipe))
        else:
            self._pipe = sf.predict_ligands(self._pipe)
    
    def fetch(self):
        for n, mol in enumerate(self._pipe):
            self.num_output = n+1 
            if self.verbose and self.num_input % 100 == 0:
                print "\rPassed: %i (%.2f%%)\tTotal: %i" % (self.num_output, float(self.num_output)/float(self.num_input)*100, self.num_input),
            yield mol
        if self.verbose:
            print ""
    
    # Consume the pipe
    def write(self, fmt, filename, csv_filename = None, **kwargs):
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
        output_mol_file = toolkit.Outputfile(fmt, filename, **kwargs)
        if csv_filename:
            f = open(csv_filename, 'w')
            csv_file = None
        for mol in self.fetch():
            if csv_filename:
                data = dict(mol.data)
                #filter some internal data
                blacklist_keys = ['OpenBabel Symmetry Classes', 'MOL Chiral Flag', 'PartialCharges', 'TORSDO', 'REMARK']
                for b in blacklist_keys:
                    if data.has_key(b):
                        del data[b]
                if len(data) > 0:
                    data['name'] = mol.title
                else:
                    print "There is no data to write in CSV file"
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
#        if kwargs.has_key('keep_pipe') and kwargs['keep_pipe']:
        #FIXME destroys data
        self._pipe = toolkit.readfile(fmt, filename)
    
    def write_csv(self, csv_filename, keep_pipe = False, **kwargs):
        """Outputs molecules to a csv file
        
        Parameters
        ----------
            csv_filename: string
                Optional path to a CSV file
            
            keep_pipe: bool (default=False)
                If set to True, the ligand pipe is sustained.
        """
        f = open(csv_filename, 'w')
        csv_file = None
        for mol in self.fetch():
            data = dict(mol.data)
            #filter some internal data
            blacklist_keys = ['OpenBabel Symmetry Classes', 'MOL Chiral Flag', 'PartialCharges', 'TORSDO', 'REMARK']
            for b in blacklist_keys:
                if data.has_key(b):
                    del data[b]
            if len(data) > 0:
                data['name'] = mol.title
            else:
                print "There is no data to write in CSV file"
                return False
            if csv_file is None:
                csv_file = csv.DictWriter(f, data.keys(), **kwargs)
                csv_file.writeheader()
            csv_file.writerow(data)
            if keep_pipe:
                #write ligand using pickle
                pass
        f.close()
