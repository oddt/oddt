""" Datasets wrapped in conviniet models """
import csv
from os.path import isfile

from oddt import toolkit

# skip comments and merge multiple spaces
def _csv_file_filter(f):
    for row in open(f, 'rb'):
        if row[0] == '#':
            continue
        yield ' '.join(row.split())

class pdbbind(object):
    def __init__(self,home, version = None, default_set = None, data_file = None):
        self.home = '%s/v%i' % (home, version) if version else home
        self.default_set = default_set if default_set else 'general'
        self.sets = {}
        if version:
            pdbind_sets = ['core', 'refined', 'general']
            for pdbind_set in pdbind_sets:
                if str(version) == '2007':
                    csv_file = '%s/INDEX.%s.%s.data' % (self.home, version, pdbind_set)
                else:
                    csv_file = '%s/INDEX_%s_data.%s' % (self.home, pdbind_set, version)
                if isfile(csv_file):
                    self.sets[pdbind_set] = {}
                    for row in csv.reader(_csv_file_filter(csv_file), delimiter=' '):
                        pdbid = row[0]
                        if not isfile('%s/%s/%s_pocket.pdb' % (self.home, pdbid, pdbid)):
                            continue
                        self.sets[pdbind_set][pdbid] = float(row[3])
                if len(self.sets) == 0:
                    raise Exception('There is no PDBbind sets availabe')
        else:
            pass # list directory, but no metadata then

    @property
    def ids(self):
        return self.sets[self.default_set].keys()

    @property
    def activities(self):
        return self.sets[self.default_set].values()

    def __iter__(self):
        for id in self.ids:
            yield _pdbbind_id(self.home, id)

    def __getitem__(self,id):
        if id in self.ids:
            return _pdbbind_id(self.home, id)
        else:
            if type(id) is int:
                return _pdbbind_id(self.home + '', self.ids[id])
            return None

class _pdbbind_id(object):
    def __init__(self, home, id):
        self.home = home
        self.id = id
    @property
    def protein(self):
        return toolkit.readfile('pdb', '%s/%s/%s_protein.pdb' % (self.home, self.id,self.id), lazy=True).next()
    @property
    def pocket(self):
        return toolkit.readfile('pdb', '%s/%s/%s_pocket.pdb' % (self.home, self.id,self.id), lazy=True).next()
    @property
    def ligand(self):
        return toolkit.readfile('mol2', '%s/%s/%s_ligand.mol2' % (self.home, self.id,self.id), lazy=True).next()
