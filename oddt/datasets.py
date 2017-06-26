""" Datasets wrapped in conviniet models """
import os
import sys
import csv
from os.path import isfile, join

from six import next
from oddt import toolkit


# skip comments and merge multiple spaces
def _csv_file_filter(f):
    for row in open(f, 'rb'):
        row = row.decode('utf-8', 'ignore')
        if row[0] == '#':
            continue
        yield ' '.join(row.split())


class pdbbind(object):
    def __init__(self,
                 home,
                 version=None,
                 default_set=None,
                 data_file=None,
                 opt=None):
        version = int(version)
        self.home = home
        if default_set:
            self.default_set = default_set
        else:
            if version == 2007:
                self.default_set = 'general'
            else:
                self.default_set = 'general_PL'
        self.opt = opt or {}
        self.sets = {}
        self._set_ids = {}
        self._set_act = {}

        if version:
            if version == 2007:
                pdbind_sets = ['core', 'refined', 'general']
            else:
                pdbind_sets = ['core', 'refined', 'general_PL']
            for pdbind_set in pdbind_sets:
                if data_file:
                    csv_file = data_file
                elif version == 2007:
                    csv_file = join(self.home,
                                    'INDEX.%i.%s.data' % (version, pdbind_set))
                elif version == 2016:
                    csv_file = join(self.home,
                                    'index',
                                    'INDEX_%s_data.%i' % (pdbind_set, version))
                else:
                    csv_file = join(self.home,
                                    'INDEX_%s_data.%i' % (pdbind_set, version))

                if isfile(csv_file):
                    self._set_ids[pdbind_set] = []
                    self._set_act[pdbind_set] = []
                    for row in csv.reader(_csv_file_filter(csv_file), delimiter=' '):
                        pdbid = row[0]
                        f = join(self.home, self.id, '%s_pocket.pdb' % self.id)
                        if not isfile(f):
                            continue
                        self._set_ids[pdbind_set].append(pdbid)
                        self._set_act[pdbind_set].append(float(row[3]))
                    self.sets[pdbind_set] = dict(zip(self._set_ids[pdbind_set],
                                                     self._set_act[pdbind_set]))
            if len(self.sets) == 0:
                raise Exception('There is no PDBbind set availabe')
        else:
            pass  # list directory, but no metadata then

    @property
    def ids(self):
        # return sorted(self.sets[self.default_set].keys())
        return self._set_ids[self.default_set]

    @property
    def activities(self):
        return self._set_act[self.default_set]

    def __iter__(self):
        for pdbid in self.ids:
            yield _pdbbind_id(self.home, pdbid, opt=self.opt)

    def __getitem__(self, pdbid):
        if pdbid in self.ids:
            return _pdbbind_id(self.home, pdbid, opt=self.opt)
        else:
            if type(pdbid) is int:
                return _pdbbind_id(self.home + '', self.ids[pdbid], opt=self.opt)
            return None


class _pdbbind_id(object):
    def __init__(self, home, pdbid, opt=None):
        self.home = home
        self.id = pdbid
        self.opt = opt or {}

    @property
    def protein(self):
        f = join(self.home, self.id, '%s_protein.pdb' % self.id)
        if isfile(f):
            return next(toolkit.readfile('pdb', f, lazy=True, opt=self.opt))
        else:
            return None

    @property
    def pocket(self):
        f = join(self.home, self.id, '%s_pocket.pdb' % self.id)
        if isfile(f):
            return next(toolkit.readfile('pdb', f, lazy=True, opt=self.opt))
        else:
            return None

    @property
    def ligand(self):
        f = join(self.home, self.id, '%s_ligand.sdf' % self.id)
        if isfile(f):
            return next(toolkit.readfile('sdf', f, lazy=True, opt=self.opt))
        else:
            return None


class dude(object):

    def __init__(self, home):
        self.home = home
        if not os.path.isdir(home):
            raise Exception("Directory doesn't exist")

        self.ids = []
        files = ['receptor.pdb', 'crystal_ligand.mol2', 'actives_final.mol2.gz', 'decoys_final.mol2.gz']
        all_ids = ['aa2ar', 'aldr', 'cp2c9', 'esr1', 'gcr', 'hs90a',
                   'lck', 'nos1', 'ppard', 'sahh', 'wee1', 'abl1',
                   'ampc', 'cp3a4', 'esr2', 'glcm', 'hxk4', 'lkha4',
                   'nram', 'pparg', 'src', 'xiap', 'ace', 'andr',
                   'csf1r', 'fa10', 'gria2', 'igf1r', 'mapk2', 'pa2ga',
                   'prgr', 'tgfr1', 'aces', 'aofb', 'cxcr4', 'fa7',
                   'grik1', 'inha', 'mcr', 'parp1', 'ptn1', 'thb',
                   'ada', 'bace1', 'def', 'fabp4', 'hdac2', 'ital',
                   'met', 'pde5a', 'pur2', 'thrb', 'ada17', 'braf',
                   'dhi1', 'fak1', 'hdac8', 'jak2', 'mk01', 'pgh1',
                   'pygm', 'try1', 'adrb1', 'cah2', 'dpp4', 'fgfr1',
                   'hivint', 'kif11', 'mk10', 'pgh2', 'pyrd', 'tryb1',
                   'adrb2', 'casp3', 'drd3', 'fkb1a', 'hivpr', 'kit',
                   'mk14', 'plk1', 'reni', 'tysy', 'akt1', 'cdk2',
                   'dyr', 'fnta', 'hivrt', 'kith', 'mmp13', 'pnph',
                   'rock1', 'urok', 'akt2', 'comt', 'egfr', 'fpps',
                   'hmdh', 'kpcb', 'mp2k1', 'ppara', 'rxra', 'vgfr2']
        for i in all_ids:
            if os.path.isdir(home + i):
                self.ids.append(i)
                for file in files:
                    if not os.path.isfile(home + i + '/' + file):
                        print("Target " + i + " doesn't have file " + file, file=sys.stderr)

    def __iter__(self):
        for dude_id in self.ids:
            yield _dude_target(self.home, dude_id)

    def __getitem__(self, dude_id):
        if dude_id in self.ids:
            return _dude_target(self.home, dude_id)
        else:
            raise Exception("Directory doesn't exist")

class _dude_target(object):

    def __init__(self, home, dude_id):
        self.home = home
        self.id = dude_id

    @property
    def protein(self):
        if isfile(self.home + self.id + "/receptor.pdb"):
            return next(toolkit.readfile("pdb", self.home + self.id + "/receptor.pdb"))
        else:
            return None

    @property
    def ligand(self):
        if isfile(self.home + self.id + "/crystal_ligand.mol2"):
            return next(toolkit.readfile("mol2", self.home + self.id + "/crystal_ligand.mol2"))
        else:
            return None

    @property
    def actives(self):
        if isfile(self.home + self.id + "/actives_final.mol2.gz"):
            return list(toolkit.readfile("mol2", self.home + self.id + "/actives_final.mol2.gz"))
        else:
            return None

    @property
    def decoys(self):
        if isfile(self.home + self.id + "/decoys_final.mol2.gz"):
            return list(toolkit.readfile("mol2", self.home + self.id + "/decoys_final.mol2.gz"))
        else:
            return None
