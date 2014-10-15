import csv
from multiprocessing.dummy import Pool
from oddt import toolkit

def _parallel_helper(args):
    """Private helper to workaround Python 2 pickle limitations"""
    obj, methodname, arg = args
    return getattr(obj, methodname)(**arg)

class virtualscreening:
    def __init__(self, n_cpu=-1, verbose=False):
        self._pipe = None
        self.n_cpu = n_cpu
        self.num_input = 0
        self.num_output = 0
        self.verbose = verbose
        # setup pool
        self._pool = Pool(n_cpu if n_cpu > 0 else None)
        
    def load_ligands(self, file_type, ligands_file):
        self._pipe = self._ligand_pipe(toolkit.readfile(file_type, ligands_file))
    
    def _ligand_pipe(self, ligands):
        for n, mol in enumerate(ligands):
            self.num_input = n+1 
            yield mol
    
    def apply_filter(self, expression, filter_type='expression', soft_fail = 0):
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
    
    def _filter(self, pipe, expression, soft_fail = 0):
        for mol in pipe:
            if type(expression) is list:
                fail = 0
                for e in expression:
                    if not eval(e):
                        fail += 1
                if fail <= soft_fail:
                    yield mol
            else:
                if eval(expression):
                    yield mol
    
    def dock(self, engine, protein, *args, **kwargs):
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
