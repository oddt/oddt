""" Datasets wrapped in conviniet models """

from oddt import toolkit

class pdbbind(object):
    def __init__(self,home, data_file = None):
        self.home = home
        self.ids = []
        self.activities = []
        if data_file:
            with open(data_file) as f:
                activities = {}
                for line in f:

                    if line[:1] == '#':
                        continue
                    data = line.split()
                    self.ids.append(data[0])
                    activities[data[0]] = float(data[3])
            self.ids = sorted(self.ids)
            self.activities = [activities[i] for i in self.ids]
        else:
            pass # list directory, but no metadata then
    def __iter__(self):
        for id in self.ids:
            yield _pdbbind_id(self.home, id)

    def __getitem__(self,id):
        if id in self.ids:
            return _pdbbind_id(self.home, id)
        else:
            if type(id) is int:
                return _pdbbind_id(self.home, self.ids[id])
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
        return toolkit.readfile('sdf', '%s/%s/%s_ligand.sdf' % (self.home, self.id,self.id), lazy=True).next()
