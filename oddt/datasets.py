""" Datasets wrapped in conviniet models """
from __future__ import print_function
import os
import sys
import csv

from six import next
import pandas as pd

from oddt import toolkit


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
                    csv_file = os.path.join(self.home,
                                            'INDEX.%i.%s.data' % (version, pdbind_set))
                elif version == 2016:
                    csv_file = os.path.join(self.home,
                                            'index',
                                            'INDEX_%s_data.%i' % (pdbind_set, version))
                else:
                    csv_file = os.path.join(self.home,
                                            'INDEX_%s_data.%i' % (pdbind_set, version))

                if os.path.isfile(csv_file):
                    data = pd.read_csv(csv_file,
                                       sep='\s+',
                                       usecols=[0, 1, 2, 3],
                                       names=['pdbid',
                                              'resolution',
                                              'release_year',
                                              'act'],
                                       comment='#')
                    self._set_ids[pdbind_set] = data['pdbid'].tolist()
                    self._set_act[pdbind_set] = data['act'].tolist()
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
        f = os.path.join(self.home, self.id, '%s_protein.pdb' % self.id)
        if os.path.isfile(f):
            return next(toolkit.readfile('pdb', f, lazy=True, opt=self.opt))
        else:
            return None

    @property
    def pocket(self):
        f = os.path.join(self.home, self.id, '%s_pocket.pdb' % self.id)
        if os.path.isfile(f):
            return next(toolkit.readfile('pdb', f, lazy=True, opt=self.opt))
        else:
            return None

    @property
    def ligand(self):
        f = os.path.join(self.home, self.id, '%s_ligand.sdf' % self.id)
        if os.path.isfile(f):
            return next(toolkit.readfile('sdf', f, lazy=True, opt=self.opt))
        else:
            return None


class dude(object):

    def __init__(self, home):
        """A wrapper for DUD-E (A Database of Useful Decoys: Enhanced)
        http://dude.docking.org/

        Parameters
        ----------
        home : str
            Path to files from dud-e

        """
        self.home = home
        if not os.path.isdir(self.home):
            raise Exception('Directory %s doesn\'t exist' % self.home)

        self.ids = []
        files = ['receptor.pdb', 'crystal_ligand.mol2', 'actives_final.mol2.gz', 'decoys_final.mol2.gz']
        # ids sorted by size of protein
        all_ids = ['fnta', 'dpp4', 'mmp13', 'hivpr', 'ada17', 'mk14', 'egfr', 'src', 'drd3', 'aa2ar',
                   'cah2', 'parp1', 'cdk2', 'lck', 'pde5a', 'thrb', 'aces', 'try1', 'pparg', 'vgfr2',
                   'pgh2', 'esr1', 'fa10', 'esr2', 'ppara', 'dhi1', 'hivrt', 'bace1', 'ace', 'dyr',
                   'akt1', 'adrb1', 'prgr', 'gcr', 'adrb2', 'andr', 'ppard', 'csf1r', 'gria2', 'cp3a4',
                   'met', 'pgh1', 'abl1', 'casp3', 'kit', 'hdac8', 'hdac2', 'braf', 'urok', 'lkha4',
                   'igf1r', 'aldr', 'fpps', 'hmdh', 'kpcb', 'tgfr1', 'ital', 'mp2k1', 'nos1', 'tryb1',
                   'rxra', 'thb', 'cp2c9', 'ptn1', 'reni', 'pnph', 'tysy', 'akt2', 'kif11', 'aofb',
                   'plk1', 'hivint', 'mk10', 'pyrd', 'grik1', 'jak2', 'rock1', 'fa7', 'mapk2', 'nram',
                   'wee1', 'fkb1a', 'def', 'ada', 'fak1', 'mcr', 'pa2ga', 'xiap', 'hs90a', 'hxk4',
                   'mk01', 'pygm', 'glcm', 'comt', 'sahh', 'cxcr4', 'kith', 'ampc', 'pur2', 'fabp4',
                   'inha', 'fgfr1']
        for i in all_ids:
            if os.path.isdir(os.path.join(self.home, i)):
                self.ids.append(i)
                for file in files:
                    f = os.path.join(self.home, i, file)
                    if not os.path.isfile(f) and not (file[-3:] == '.gz' and os.path.isfile(f[:-3])):
                        print('Target %s doesn\'t have file %s' % (i, file), file=sys.stderr)
        if not self.ids:
            print('No targets in directory %s' % (self.home), file=sys.stderr)

    def __iter__(self):
        for dude_id in self.ids:
            yield _dude_target(self.home, dude_id)

    def __getitem__(self, dude_id):
        if dude_id in self.ids:
            return _dude_target(self.home, dude_id)
        else:
            raise Exception('Directory %s doesn\'t exist' % self.home)


class _dude_target(object):

    def __init__(self, home, dude_id):
        """Allows to read files of the dude target

        Parameters
        ----------
        home : str
            Directory to files from dud-e

        dude_id : str
            Target id
        """
        self.home = home
        self.dude_id = dude_id

    @property
    def protein(self):
        """Read a protein file"""
        f = os.path.join(self.home, self.dude_id, 'receptor.pdb')
        if os.path.isfile(f):
            return next(toolkit.readfile('pdb', f))
        else:
            return None

    @property
    def ligand(self):
        """Read a ligand file"""
        f = os.path.join(self.home, self.dude_id, 'crystal_ligand.mol2')
        if os.path.isfile(f):
            return next(toolkit.readfile('mol2', f))
        else:
            return None

    @property
    def actives(self):
        """Read an actives file"""
        f = os.path.join(self.home, self.dude_id, 'actives_final.mol2.gz')
        if os.path.isfile(f):
            return toolkit.readfile('mol2', f)
        # check if file is unpacked
        elif os.path.isfile(f[:-3]):
            return toolkit.readfile('mol2', f[:-3])
        else:
            return None

    @property
    def decoys(self):
        """Read a decoys file"""
        f = os.path.join(self.home, self.dude_id, 'decoys_final.mol2.gz')
        if os.path.isfile(f):
            return toolkit.readfile('mol2', f)
        # check if file is unpacked
        elif os.path.isfile(f[:-3]):
            return toolkit.readfile('mol2', f[:-3])
        else:
            return None
