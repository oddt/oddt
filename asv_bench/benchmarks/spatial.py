import os
import oddt
from oddt.spatial import distance

test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

class BenchDistance(object):
    """Spatial functions"""

    def setup(self):
        self.mols = list(oddt.toolkit.readfile('sdf', '%s/tests/data/dude/xiap/actives_docked.sdf' % test_data_dir))[:10]
        self.protein = list(oddt.toolkit.readfile('pdb', '%s/tests/data/dude/xiap/receptor_rdkit.pdb' % test_data_dir))[0]

    def time_distance_protein(self):
        distance(self.protein.coords, self.protein.coords)

    def peakmem_distance_protein(self):
        distance(self.protein.coords, self.protein.coords)

    def time_distance_mol(self):
        for mol in self.mols:
            distance(mol.coords, mol.coords)

    def peakmem_distance_mol(self):
        for mol in self.mols:
            distance(mol.coords, mol.coords)

    def time_distance_complex(self):
        for mol in self.mols:
            distance(mol.coords, self.protein.coords)

    def peakmem_distance_complex(self):
        for mol in self.mols:
            distance(mol.coords, self.protein.coords)
