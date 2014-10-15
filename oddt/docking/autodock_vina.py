from tempfile import mkdtemp
from shutil import rmtree
from os.path import exists
from os import remove
import subprocess
import numpy as np
import re
from random import random
from oddt import toolkit

class autodock_vina(object):
    def __init__(self, protein=None, size=(10,10,10), center=(0,0,0), auto_ligand=None, exhaustivness=8, num_modes=9, energy_range=3, seed=None, prefix_dir='/tmp', n_cpu=1, executable=None, autocleanup=True):
        self.dir = prefix_dir
        self._tmp_dir = None
        # define binding site
        self.size = size
        self.center = center
        # center automaticaly on ligand
        if auto_ligand:
            if type(auto_ligand) is str:
                extension = auto_ligand.split('.')[-1]
                auto_ligand = toolkit.readfile(extension, auto_ligand).next()
            self.center = tuple(np.array([atom.coords for atom in auto_ligand], dtype=np.float16).mean(axis=0))
        # autodetect Vina executable
        if not executable:
            self.executable = subprocess.check_output(['which', 'vina']).split('\n')[0]
        else:
            self.executable = executable
        # detect version
        self.version = subprocess.check_output([self.executable, '--version']).split(' ')[2]
        self.autocleanup = autocleanup
        self.cleanup_dirs = set()
        
        # share protein to class
        if protein:
            self.set_protein(protein)
        
        #pregenerate common Vina parameters
        self.params = []
        self.params = self.params + ['--center_x', str(self.center[0]), '--center_y', str(self.center[1]), '--center_z', str(self.center[2])]
        self.params = self.params + ['--size_x', str(self.size[0]), '--size_y', str(self.size[1]), '--size_z', str(self.size[2])]
        self.params = self.params + ['--cpu', str(n_cpu)]
        self.params = self.params + ['--exhaustiveness', str(exhaustivness)]
        if not seed is None:
            self.params = self.params + ['--seed', str(seed)]
        self.params = self.params + ['--num_modes', str(num_modes)]
        self.params = self.params + ['--energy_range', str(energy_range)]
    
    @property
    def tmp_dir(self):
        if not self._tmp_dir:
            self._tmp_dir = mkdtemp(dir = self.dir, prefix='autodock_vina_')
            self.cleanup_dirs.add(self._tmp_dir)
        return self._tmp_dir
    
    @tmp_dir.setter
    def tmp_dir(self, value):
        self._tmp_dir = value
    
    def set_protein(self, protein):
        # generate new directory
        self._tmp_dir = None
        self.protein = protein
        if type(protein) is str:
            extension = protein.split('.')[-1]
            if extension == 'pdbqt':
                self.protein_file = protein
                self.protein = toolkit.readfile(extension, protein).next()
            else:
                self.protein = toolkit.readfile(extension, protein).next()
                self.protein.protein = True
                self.protein_file = self.tmp_dir  + '/protein.pdbqt'
                self.protein.write('pdbqt', self.protein_file, opt={'r':None,}, overwrite=True)
        else:
            # write protein to file
            self.protein_file = self.tmp_dir  + '/protein.pdbqt'
            self.protein.write('pdbqt', self.protein_file, opt={'r':None,}, overwrite=True)
    
    def score(self, ligands, protein = None, single = False):
        if protein:
            self.set_protein(protein)
        if single:
            ligands = [ligands]
        ligand_dir = mkdtemp(dir = self.tmp_dir, prefix='ligands_')
        output_array = []
        for n, ligand in enumerate(ligands):
            # write ligand to file
            ligand_file = ligand_dir + '/' + str(n) + '_' + ligand.title + '.pdbqt'
            ligand.write('pdbqt', ligand_file, overwrite=True)
            scores = parse_vina_scoring_output(subprocess.check_output([self.executable, '--score_only', '--receptor', self.protein_file, '--ligand', ligand_file] + self.params, stderr=subprocess.STDOUT))
            ligand.data.update(scores)
            output_array.append(ligand)
        rmtree(ligand_dir)
        return output_array
            
    def dock(self, ligands, protein = None, single = False):
        if protein:
            self.set_protein(protein)
        if single:
            ligands = [ligands]
        ligand_dir = mkdtemp(dir = self.tmp_dir, prefix='ligands_')
        output_array = []
        for n, ligand in enumerate(ligands):
            # write ligand to file
            ligand_file = ligand_dir + '/' + str(n) + '_' + ligand.title + '.pdbqt'
            ligand_outfile = ligand_dir + '/' + str(n) + '_' + ligand.title + '_out.pdbqt'
            ligand.write('pdbqt', ligand_file, overwrite=True)
            vina = parse_vina_docking_output(subprocess.check_output([self.executable, '--receptor', self.protein_file, '--ligand', ligand_file, '--out', ligand_outfile] + self.params, stderr=subprocess.STDOUT))
            ### HACK # overcome connectivity problems in obabel
            source_ligand = toolkit.readfile('pdbqt', ligand_file).next()
            for lig, scores in zip([lig for lig in toolkit.readfile('pdbqt', ligand_outfile, opt={'b': None})], vina):
                ### HACK # copy data from source
                clone = source_ligand.clone
                clone.clone_coords(lig)
                clone.data.update(scores)
                output_array.append(clone)
        rmtree(ligand_dir)
        return output_array
    
    def clean(self):
        for d in self.cleanup_dirs:
            rmtree(d)
    
def parse_vina_scoring_output(output):
    out = {}
    r = re.compile('^(Affinity:|\s{4})')
    for line in output.split('\n')[13:]: # skip some output
        if r.match(line):
            m = line.replace(' ','').split(':')
            if m[0] == 'Affinity':
                m[1] = m[1].replace('(kcal/mol)','')
            out['vina_'+m[0].lower()] = float(m[1])
    return out
    
def parse_vina_docking_output(output):
    out = []
    r = re.compile('^\s+\d\s+')
    for line in output.split('\n')[13:]: # skip some output
        if r.match(line):
            s = line.split()
            out.append({'vina_affinity': s[1], 'vina_rmsd_lb': s[2], 'vina_rmsd_ub': s[3]})
    return out
    
