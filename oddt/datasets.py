""" Datasets wrapped in convenient models """
from __future__ import print_function
import sys
import os
import logging
from os.path import isfile, isdir
from os import listdir

import six
import pandas as pd

from oddt import toolkit


CASF_2016_IDS = {
    '1a30', '1nc3', '1r5y', '2al5', '2qbp', '2weg', '2yfe', '3arv', '3dx1', '3fv1',
    '3jvs', '3nx7', '3rr4', '3ui7', '4agq', '4e5w', '4ih5', '4kzq', '4twp', '1bcu',
    '1nvq', '1s38', '2br1', '2qbq', '2wer', '2yge', '3ary', '3dx2', '3fv2', '3jya',
    '3o9i', '3rsx', '3uo4', '4bkt', '4e6q', '4ih7', '4kzu', '4ty7', '1bzc', '1o0h',
    '1sqa', '2brb', '2qbr', '2wn9', '2yki', '3b1m', '3dxg', '3g0w', '3k5v', '3oe4',
    '3ryj', '3up2', '4cig', '4ea2', '4ivb', '4llx', '4u4s', '1c5z', '1o3f', '1syi',
    '2c3i', '2qe4', '2wnc', '2ymd', '3b27', '3e5a', '3g2n', '3kgp', '3oe5', '3syr',
    '3uri', '4ciw', '4eky', '4ivc', '4lzs', '4w9c', '1e66', '1o5b', '1u1b', '2cbv',
    '2qnq', '2wtv', '2zb1', '3b5r', '3e92', '3g2z', '3kr8', '3ozs', '3tsk', '3utu',
    '4cr9', '4eo8', '4ivd', '4m0y', '4w9h', '1eby', '1owh', '1uto', '2cet', '2r9w',
    '2wvt', '2zcq', '3b65', '3e93', '3g31', '3kwa', '3ozt', '3twp', '3uuo', '4cra',
    '4eor', '4j21', '4m0z', '4w9i', '1g2k', '1oyt', '1vso', '2fvd', '2v00', '2x00',
    '2zcr', '3b68', '3ebp', '3gbb', '3l7b', '3p5o', '3u5j', '3wtj', '4crc', '4f09',
    '4j28', '4mgd', '4w9l', '1gpk', '1p1n', '1w4o', '2fxs', '2v7a', '2xb8', '2zda',
    '3bgz', '3ehy', '3gc5', '3lka', '3prs', '3u8k', '3wz8', '4ddh', '4f2w', '4j3l',
    '4mme', '4wiv', '1gpn', '1p1q', '1y6r', '2hb1', '2vkm', '2xbv', '2zy1', '3bv9',
    '3ejr', '3ge7', '3mss', '3pww', '3u8n', '3zdg', '4ddk', '4f3c', '4jfs', '4ogj',
    '4x6p', '1h22', '1ps3', '1yc1', '2iwx', '2vvn', '2xdl', '3acw', '3cj4', '3f3a',
    '3gnw', '3myg', '3pxf', '3u9q', '3zso', '4de1', '4f9w', '4jia', '4owm', '5a7b',
    '1h23', '1pxn', '1ydr', '2j78', '2vw5', '2xii', '3ag9', '3coy', '3f3c', '3gr2',
    '3n76', '3pyy', '3udh', '3zsx', '4de2', '4gfm', '4jsz', '4pcs', '5aba', '1k1i',
    '1q8t', '1ydt', '2j7h', '2w4x', '2xj7', '3ao4', '3coz', '3f3d', '3gv9', '3n7a',
    '3qgy', '3ueu', '3zt2', '4de3', '4gid', '4jxs', '4qac', '5c28', '1lpg', '1q8u',
    '1z6e', '2p15', '2w66', '2xnb', '3arp', '3d4z', '3f3e', '3gy4', '3n86', '3qqs',
    '3uev', '4abg', '4djv', '4gkm', '4k18', '4qd6', '5c2h', '1mq6', '1qf1', '1z95',
    '2p4y', '2wbg', '2xys', '3arq', '3d6q', '3fcq', '3ivg', '3nq9', '3r88', '3uew',
    '4agn', '4dld', '4gr0', '4k77', '4rfm', '5dwr', '1nc1', '1qkt', '1z9g', '2pog',
    '2wca', '2y5h', '3aru', '3dd0', '3fur', '3jvr', '3nw9', '3rlr', '3uex', '4agp',
    '4dli', '4hge', '4kz6', '4tmn', '5tmn'
}


class pdbbind(object):
    def __init__(self,
                 home,
                 version=None,
                 default_set=None,
                 opt=None):

        if version is None:
            raise ValueError('PDBbind version not specified')
        else:
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

        # list of protein ids that are known to segfault toolkits
        self.protein_blacklist = {
            'ob': {'1e8h', '1ntk', '1nu1', '1rbo', '1sqb', '1sqp', '1sqq',
                   '2f2h', '2wig', '2wij', '2wik', '3axk', '3axm', '3cf1',
                   # Following segfault on systems with smaller RAM
                   '1px4', '1pyg', '1zyr', '3a2c', '3dxj', '3dyo', '3eql',
                   '3f33', '3f34', '3f35', '3f36', '3f37', '3f38', '3f39',
                   '3i3b', '3i3d', '3k1j', '3muz', '3mv0', '3n75', '3t08',
                   '3t09', '3t0b', '3t0d', '3t2p', '3t2q', '3vd4', '3vd7',
                   '3vd9', '3vdb', '3vdc', '3wi6', '4kmu', '4kn4', '4kn7',
                   '7gpb',
                   # extended use segfaults (not only reading problem)
                   '1l7x',
                   },
            'rdk': {}
        }

        if version == 2007:
            self.pdbind_sets = ['core', 'refined', 'general']
        else:
            self.pdbind_sets = ['core', 'refined', 'general_PL']
        for pdbind_set in self.pdbind_sets:
            if version == 2007:
                csv_file = os.path.join(self.home, 'INDEX.%i.%s.data'
                                        % (version, pdbind_set))
            elif version >= 2016:
                pdbind_set_csv = 'refined' if pdbind_set == 'core' else pdbind_set
                csv_file = os.path.join(self.home, 'index', 'INDEX_%s_data.%i'
                                        % (pdbind_set_csv, version))
            else:
                csv_file = os.path.join(self.home, 'INDEX_%s_data.%i'
                                        % (pdbind_set, version))

            if os.path.isfile(csv_file):
                data = pd.read_csv(csv_file,
                                   sep='\s+',
                                   usecols=[0, 1, 2, 3],
                                   names=['pdbid',
                                          'resolution',
                                          'release_year',
                                          'act'],
                                   comment='#')
                if version >= 2016 and pdbind_set == 'core':
                    data = data[data['pdbid'].isin(CASF_2016_IDS)]
                self._set_ids[pdbind_set] = data['pdbid'].tolist()
                self._set_act[pdbind_set] = data['act'].tolist()
                self.sets[pdbind_set] = dict(zip(self._set_ids[pdbind_set],
                                                 self._set_act[pdbind_set]))
            else:
                raise IOError('PDBbind v%i set %s not found' % (version, pdbind_set))
        if len(self.sets) == 0:
            raise IOError('There is no PDBbind set availabe')

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
        warn_msg = ('A protein "%s" is blacklisted (known to segfault) for '
                    'current toolkit. Proceed at your own risk.' % pdbid)
        if pdbid in self.ids:
            if pdbid in self.protein_blacklist[toolkit.backend]:
                logging.warning(warn_msg)
            return _pdbbind_id(self.home, pdbid, opt=self.opt)
        elif (isinstance(pdbid, int) and
              pdbid < len(self.ids) and
              pdbid >= -len(self.ids)):
            if self.ids[pdbid] in self.protein_blacklist[toolkit.backend]:
                logging.warning(warn_msg)
            return _pdbbind_id(self.home + '', self.ids[pdbid], opt=self.opt)
        else:
            raise KeyError('There is no such target ("%s")' % pdbid)


class _pdbbind_id(object):
    def __init__(self, home, pdbid, opt=None):
        self.home = home
        self.id = pdbid
        self.opt = opt or {}

    @property
    def protein(self):
        f = os.path.join(self.home, self.id, '%s_protein.pdb' % self.id)
        if os.path.isfile(f):
            protein = next(toolkit.readfile('pdb', f, lazy=True, opt=self.opt))
            if protein is not None:
                protein.protein = True
            return protein
        else:
            return None

    @property
    def pocket(self):
        f = os.path.join(self.home, self.id, '%s_pocket.pdb' % self.id)
        if os.path.isfile(f):
            pocket = next(toolkit.readfile('pdb', f, lazy=True, opt=self.opt))
            if pocket is not None:
                pocket.protein = True
            return pocket
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
        files = ['receptor.pdb', 'crystal_ligand.mol2',
                 'actives_final.mol2.gz', 'decoys_final.mol2.gz']
        # ids sorted by size of protein
        all_ids = [
            'fnta', 'dpp4', 'mmp13', 'hivpr', 'ada17', 'mk14', 'egfr', 'src',
            'drd3', 'aa2ar', 'cah2', 'parp1', 'cdk2', 'lck', 'pde5a', 'thrb',
            'aces', 'try1', 'pparg', 'vgfr2', 'pgh2', 'esr1', 'fa10', 'esr2',
            'ppara', 'dhi1', 'hivrt', 'bace1', 'ace', 'dyr', 'akt1', 'adrb1',
            'prgr', 'gcr', 'adrb2', 'andr', 'ppard', 'csf1r', 'gria2', 'cp3a4',
            'met', 'pgh1', 'abl1', 'casp3', 'kit', 'hdac8', 'hdac2', 'braf',
            'urok', 'lkha4', 'igf1r', 'aldr', 'fpps', 'hmdh', 'kpcb', 'tgfr1',
            'ital', 'mp2k1', 'nos1', 'tryb1', 'rxra', 'thb', 'cp2c9', 'ptn1',
            'reni', 'pnph', 'tysy', 'akt2', 'kif11', 'aofb', 'plk1', 'hivint',
            'mk10', 'pyrd', 'grik1', 'jak2', 'rock1', 'fa7', 'mapk2', 'nram',
            'wee1', 'fkb1a', 'def', 'ada', 'fak1', 'mcr', 'pa2ga', 'xiap',
            'hs90a', 'hxk4', 'mk01', 'pygm', 'glcm', 'comt', 'sahh', 'cxcr4',
            'kith', 'ampc', 'pur2', 'fabp4', 'inha', 'fgfr1',
        ]

        for i in all_ids:
            if os.path.isdir(os.path.join(self.home, i)):
                self.ids.append(i)
                for fname in files:
                    f = os.path.join(self.home, i, fname)
                    if not (os.path.isfile(f) or
                            (fname[-3:] == '.gz' and os.path.isfile(f[:-3]))):
                        print('Target %s doesn\'t have file %s' % (i, fname),
                              file=sys.stderr)
        if not self.ids:
            print('No targets in directory %s' % (self.home), file=sys.stderr)

    def __iter__(self):
        for dude_id in self.ids:
            yield _dude_target(self.home, dude_id)

    def __getitem__(self, dude_id):
        if dude_id in self.ids:
            return _dude_target(self.home, dude_id)
        else:
            raise KeyError('There is no such target ("%s")' % dude_id)


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


class CASF:
    """Load CASF dataset as described in
    Li, Y. et al. Comparative Assessment of Scoring Functions
    on an Updated Benchmark: 2. Evaluation Methods and General
    Results. J. Chem. Inf. Model. 54, 1717-1736. (2014)
    http://dx.doi.org/10.1021/ci500081m

    Parameters
    ----------
    home: string
        Path to CASF dataset main directory
    """

    def __init__(self, home):
        self.home = home
        self.index = '%s/coreset/index/' % self.home

        if isdir(self.index):
            filepath = '%s/2013_core_data.lst' % self.index
            self.index_data = pd.read_csv(filepath,
                                          sep=r'\s+',
                                          comment='#',
                                          header=None,
                                          names=['pdbid', 'act', 'cluster'],
                                          usecols=[0, 1, 5])
            self.pdbids = self.index_data['pdbid']

    def __iter__(self):
        for pdbid in self.pdbids:
            yield _CASFTarget(self.home, pdbid)

    def __getitem__(self, item):
        if item in self.pdbids:
            return _CASFTarget(self.home, item)
        elif isinstance(int, item) and item < len(self.pdbids):
            return _CASFTarget(self.home, self.pdbids[item])
        else:
            raise KeyError

    def precomputed_score(self, scoring_function=None):
        """Load precomputed results of scoring power
        test for various scoring functions.

        Parameters
        ----------
        scoring_function: string (default=None)
            Name of the scoring function to get results
            If None, all results are returned.
        """
        examples_dir = '%s/power_scoring/examples' % self.home
        if scoring_function is not None:
            functions = [scoring_function]
        else:
            functions = listdir(examples_dir)
            functions.remove('README')

        frames = []

        for fun in functions:
            file_score = '%s/%s' % (examples_dir, fun)
            if not isfile(file_score):
                raise FileNotFoundError('Invalid scoring function name')

            score = pd.read_csv(file_score, comment='#',
                                sep=r'\s+', header=None,
                                names=['pdbid', 'score_crystal', 'score_opt'])
            act = self.index_data[['pdbid', 'act']]

            scores = pd.merge(score, act)
            scores['scoring_function'] = pd.Series([fun] * 195,
                                                   name='Scoring function')
            frames.append(scores)

        return pd.concat(frames)

    def precomputed_screening(self, scoring_function=None, cluster_id=None):
        """Load precomputed results of screening power
        test for various scoring functions

        Parameters
        ----------
        scoring_function: string (default=None)
            Name of the scoring function to get results
            If None, all results are returned

        cluster_id: int (default=None)
            Number of the protein cluster to get results
            If None, all results are returned
        """
        screening_dir = '%s/power_screening' % self.home
        examples_dir = '%s/examples' % screening_dir
        if scoring_function is not None:
            functions = [scoring_function]
        else:
            functions = listdir(examples_dir)

        cluster_frame = pd.DataFrame(columns=['cluster_id',
                                              'protein_structure',
                                              'cluster_proteins'])
        data_file = open('%s/TargetInfo.dat' % screening_dir)
        for cluster, line in enumerate(filter(lambda x: not x.startswith('#'),
                                              data_file.readlines())):
            line = line.split()
            protein_structure = line[0]
            cluster_proteins = line[1:]
            cluster_frame.loc[cluster] = [cluster + 1,
                                          protein_structure, cluster_proteins]

        frames = []
        for fun in functions:
            file_dir = '%s/%s' % (examples_dir, fun)
            if not isdir(file_dir):
                raise FileNotFoundError('Invalid scoring function name')
            if cluster_id:
                protein = cluster_frame.iloc[cluster_id - 1]['protein_structure']
                frame = pd.read_csv('%s/%s_score.dat' % (file_dir, protein),
                                    sep=r'\s+', header=None,
                                    names=['name', 'score'])
                frame['pdbid'] = [name[:4] for name in frame['name']]
                frame['scoring_function'] = [fun] * len(frame)
                frame = frame.merge(self.index_data[['pdbid', 'act']])
                frames.append(frame)

            else:
                for row in cluster_frame.itertuples():
                    protein = row[2]
                    frame = pd.read_csv('%s/%s_score.dat' % (file_dir, protein),
                                        sep=r'\s+', header=None,
                                        names=['name', 'score'])
                    x = row[1]
                    frame['cluster_id'] = [x] * len(frame)
                    frame['protein_structure'] = [protein] * len(frame)
                    frame['cluster_proteins'] = [row[3]] * len(frame)
                    frame['pdbid'] = [name[:4] for name in frame['name']]
                    frame['scoring_function'] = [fun] * len(frame)
                    frame = frame.merge(self.index_data[['pdbid', 'act']])
                    frames.append(frame)

        return pd.concat(frames, ignore_index=True)


class _CASFTarget:
    """
    Used by CASF class.
    Load CASF target (protein and ligand) with given ID.

    Parameters
    ----------
    home: string
        Path to CASF dataset main directory
    pdbid: string
        ID of target protein
    """
    def __init__(self, home, pdbid):
        self.home = home
        self.pdbid = pdbid

    @property
    def protein(self):
        """Load target protein from mol2 file as ob.Molecule object"""
        filepath = '%s/coreset/%s/%s_protein.mol2' % (
            self.home, self.pdbid, self.pdbid)
        if isfile(filepath):
            protein = six.next(toolkit.readfile('mol2', filepath))
            return protein
        return None

    @property
    def ligand(self):
        """Load target ligand from mol2 file as ob.Molecule object"""
        filepath = '%s/coreset/%s/%s_ligand.mol2' % (
            self.home, self.pdbid, self.pdbid)
        if isfile(filepath):
            ligand = six.next(toolkit.readfile('mol2', filepath))
            return ligand
        return None

    @property
    def decoys_docking(self):
        """Load decoys used for docking from mol2
        file as list of ob.Molecule objects"""
        filepath = '%s/decoys_docking/%s_decoys.mol2' % (self.home, self.pdbid)
        if isfile(filepath):
            decoys = list(toolkit.readfile('mol2', filepath))
            return decoys
        return None

    @property
    def decoys_screening(self):
        """Load decoys used for screening from mol2
        files as list of ob.Molecule objects"""
        dirpath = '%s/decoys_screening/%s' % (self.home, self.pdbid)
        if isdir(dirpath):
            decoys = []
            for file in listdir(dirpath):
                decoys.append(six.next(
                    toolkit.readfile('mol2', dirpath + '/' + file)))
            return decoys
        return None
