import csv
from os.path import dirname, isfile
import numpy as np
from multiprocessing import Pool
import warnings
from sklearn.externals.joblib import Parallel, delayed

from oddt import toolkit
from oddt.scoring import scorer, ensemble_model
from oddt.scoring.descriptors.binana import binana_descriptor
from oddt.scoring.models.regressors import neuralnetwork

# numpy after pickling gives Runtime Warnings
warnings.simplefilter("ignore", RuntimeWarning)

# define sub-function for paralelization
def generate_descriptor(packed):
    pdbid, gen, pdbbind_dir, pdbbind_version = packed
    protein_file = pdbbind_dir + "/v" + pdbbind_version + "/%s/%s_pocket.pdb" % (pdbid, pdbid)
    if not isfile(protein_file):
        protein_file = pdbbind_dir + "/v" + pdbbind_version + "/%s/%s_protein.pdb" % (pdbid, pdbid)
    ligand_file = pdbbind_dir + "/v" + pdbbind_version + "/%s/%s_ligand.sdf" % (pdbid, pdbid)
    protein = toolkit.readfile("pdb", protein_file).next()
    # mark it as a protein
    protein.protein = True
    ligand = toolkit.readfile("sdf", ligand_file).next()
    return gen.build([ligand], protein).flatten()

# skip comments and merge multiple spaces
def _csv_file_filter(f):
    for row in open(f, 'rb'):
        if row[0] == '#':
            continue
        yield ' '.join(row.split())

def _parallel_helper(obj, methodname, *args, **kwargs):
    """Private helper to workaround Python 2 pickle limitations"""
    return getattr(obj, methodname)(*args, **kwargs)

class nnscore(scorer):
    def __init__(self, protein = None, n_jobs = -1, **kwargs):
        self.protein = protein
        self.n_jobs = n_jobs
        model = None
        decsriptors = binana_descriptor(protein)
        super(nnscore,self).__init__(model, decsriptors, score_title='nnscore')
    
    def gen_training_data(self, pdbbind_dir, pdbbind_version = '2007', sf_pickle = ''):
        # build train and test 
        cpus = self.n_jobs if self.n_jobs > 0 else None
        pool = Pool(processes=cpus)
        
        core_act = np.zeros(1, dtype=float)
        core_set = []
        pdb_set = 'core'
        if pdbbind_version == '2007':
            csv_file = '%s/v%s/INDEX.%s.%s.data' % (pdbbind_dir, pdbbind_version, pdbbind_version, pdb_set)
        else:
            csv_file = '%s/v%s/INDEX_%s_data.%s' % (pdbbind_dir, pdbbind_version, pdb_set, pdbbind_version)
        for row in csv.reader(_csv_file_filter(csv_file), delimiter=' '):
            pdbid = row[0]
            if not isfile('%s/v%s/%s/%s_pocket.pdb' % (pdbbind_dir, pdbbind_version, pdbid, pdbid)):
                continue
            act = float(row[3])
            core_set.append(pdbid)
            core_act = np.vstack((core_act, act))
        
        result = pool.map(generate_descriptor, [(pdbid, self.descriptor_generator, pdbbind_dir, pdbbind_version) for pdbid in core_set])
        core_desc = np.vstack(result)
        core_act = core_act[1:]
        
        refined_act = np.zeros(1, dtype=float)
        refined_set = []
        pdb_set = 'refined'
        if pdbbind_version == '2007':
            csv_file = '%s/v%s/INDEX.%s.%s.data' % (pdbbind_dir, pdbbind_version, pdbbind_version, pdb_set)
        else:
            csv_file = '%s/v%s/INDEX_%s_data.%s' % (pdbbind_dir, pdbbind_version, pdb_set, pdbbind_version)
        for row in csv.reader(_csv_file_filter(csv_file), delimiter=' '):
            pdbid = row[0]
            if not isfile('%s/v%s/%s/%s_pocket.pdb' % (pdbbind_dir, pdbbind_version, pdbid, pdbid)):
                continue
            act = float(row[3])
            if pdbid in core_set:
                continue
            refined_set.append(pdbid)
            refined_act = np.vstack((refined_act, act))
        
        result = pool.map(generate_descriptor, [(pdbid, self.descriptor_generator, pdbbind_dir, pdbbind_version) for pdbid in refined_set])
        refined_desc = np.vstack(result)
        refined_act = refined_act[1:]
        
        self.train_descs = refined_desc
        self.train_target = refined_act.flatten()
        self.test_descs = core_desc
        self.test_target = core_act.flatten()
        
        # save numpy arrays
        np.savetxt(dirname(__file__) + '/NNScore/train_descs.csv', self.train_descs, fmt='%.5g', delimiter=',')
        np.savetxt(dirname(__file__) + '/NNScore/train_target.csv', self.train_target, fmt='%.2f', delimiter=',')
        np.savetxt(dirname(__file__) + '/NNScore/test_descs.csv', self.test_descs, fmt='%.5g', delimiter=',')
        np.savetxt(dirname(__file__) + '/NNScore/test_target.csv', self.test_target, fmt='%.2f', delimiter=',')
        
        
    def train(self, sf_pickle = ''):
        # load precomputed descriptors and target values
        self.train_descs = np.loadtxt(dirname(__file__) + '/NNScore/train_descs.csv', delimiter=',', dtype=float)
        self.train_target = np.loadtxt(dirname(__file__) + '/NNScore/train_target.csv', delimiter=',', dtype=float)
        self.test_descs = np.loadtxt(dirname(__file__) + '/NNScore/test_descs.csv', delimiter=',', dtype=float)
        self.test_target = np.loadtxt(dirname(__file__) + '/NNScore/test_target.csv', delimiter=',', dtype=float)
        
        n_dim = (~((self.train_descs == 0).all(axis=0) | (self.train_descs.min(axis=0) == self.train_descs.max(axis=0)))).sum()
        
        # number of network to sample; original implementation did 1000, but 100 give results good enough.
        n = 1000
        trained_nets = Parallel(n_jobs=self.n_jobs)(delayed(_parallel_helper)(neuralnetwork([n_dim,5,1]), 'fit', self.train_descs, self.train_target, train_alg='tnc', maxfun=1000) for i in xrange(n))
        # get 20 best
        best_idx = np.array([net.score(self.test_descs, self.test_target.flatten()) for net in trained_nets]).argsort()[::-1][:20]
        self.model = ensemble_model([trained_nets[i] for i in best_idx])
        
        r2 = self.model.score(self.test_descs, self.test_target)
        r = np.sqrt(r2)
        print 'Test set: R**2:', r2, ' R:', r
        
        r2 = self.model.score(self.train_descs, self.train_target)
        r = np.sqrt(r2)
        print 'Train set: R**2:', r2, ' R:', r
        
        if sf_pickle:
            return self.save(sf_pickle)
        else:
            return self.save('NNScore.pickle')
    
    @classmethod
    def load(self, filename = ''):
        if not filename:
            for f in ['NNScore.pickle', dirname(__file__) + '/NNScore.pickle']:
                if isfile(f):
                    filename = f
                    break
        # if still no pickle found - train function from pregenerated descriptors
        if not filename:
            print "No pickle, training new scoring function."
            nn = nnscore()
            filename = nn.train()
        return scorer.load(filename)
