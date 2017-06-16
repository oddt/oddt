import os
import sys
import oddt
import pandas as pd
from oddt import metrics
from oddt.shape import *
from os.path import isfile
from joblib import Parallel, delayed
from scipy.spatial import distance

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
            return next(oddt.toolkit.readfile("pdb", self.home + self.id + "/receptor.pdb"))
        else:
            return None
        
    @property
    def ligand(self):
        if isfile(self.home + self.id + "/crystal_ligand.mol2"):
            return next(oddt.toolkit.readfile("mol2", self.home + self.id + "/crystal_ligand.mol2"))
        else:
            return None
    
    @property
    def actives(self):
        if isfile(self.home + self.id + "/actives_final.mol2.gz"):
            return list(oddt.toolkit.readfile("mol2", self.home + self.id + "/actives_final.mol2.gz"))
        else:
            return None
        
    @property
    def decoys(self):
        if isfile(self.home + self.id + "/decoys_final.mol2.gz"):
            return list(oddt.toolkit.readfile("mol2", self.home + self.id + "/decoys_final.mol2.gz"))
        else:
            return None
        
def dude_function(target, usr_method):

    path = "/home/paulina/Documents/STUDIA/Praktyki/all/"
    
    if usr_method == "usr":
        usr_actives = np.array([usr(mol) for mol in target.actives])
        usr_decoys = np.array([usr(mol) for mol in target.decoys])
        similarity_function = usr_similarity
    elif usr_method == "usr_cat":
        usr_actives = np.array([usr_cat(mol) for mol in target.actives])
        usr_decoys = np.array([usr_cat(mol) for mol in target.decoys])
        similarity_function = usr_similarity
    elif usr_method == "electroshape":
        usr_actives = np.array([electroshape(mol) for mol in oddt.toolkit.readfile("mol2", path + target + "/actives_final.mol2.gz")])
        usr_decoys = np.array([electroshape(mol) for mol in oddt.toolkit.readfile("mol2", path + target + "/decoys_final.mol2.gz")])
        similarity_function = usr_similarity
    else:
        raise Exception("Wrong usr method")
        
    sim = distance.cdist(usr_actives, np.vstack((usr_actives, usr_decoys)), metric=similarity_function)
    

    act = np.ones(len(usr_actives))
    dec = np.zeros(len(usr_decoys))
    act_dec = np.append(act, dec)
    
    usr_enr_factor = []
    usr_roc_auc = []

    for row in sim:
        
        mols = np.matrix((row, act_dec))
        mols = pd.DataFrame(mols.T)
        mols = mols.sort_values(0, ascending=False)
        mols = mols[1:]
    
        usr_enr_factor.append(oddt.metrics.enrichment_factor(y_true = mols[1].values, y_score = mols[0].values))
        usr_roc_auc.append(oddt.metrics.roc_auc(y_true = mols[1].values, y_score = mols[0].values, ascending_score=False))
    
    data = {} 
    data["target_id"] = target.id
    data["ef1"] = np.array(usr_enr_factor)
    data["roc"] = np.array(usr_roc_auc)
    data["usr_type"] = usr_method
    
    return pd.DataFrame(data)
