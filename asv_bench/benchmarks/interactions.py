import os
import oddt
from oddt.interactions import close_contacts, hbonds

test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

class BenchInteractions(object):
    """Spatial functions"""

    def setup(self):
        self.mols = list(oddt.toolkit.readfile('sdf', '%s/tests/data/dude/xiap/actives_docked.sdf' % test_data_dir))[:10]
        for mol in self.mols:
            mol.addh(only_polar=True)
            mol.atom_dict
        self.protein = list(oddt.toolkit.readfile('pdb', '%s/tests/data/dude/xiap/receptor_rdkit.pdb' % test_data_dir))[0]
        self.protein.atom_dict
        self.protein.addh(only_polar=True)

    def time_close_contacts(self):
        for mol in self.mols:
            close_contacts(mol.atom_dict, self.protein.atom_dict, cutoff=10.)

    def peakmem_close_contacts(self):
        self.time_close_contacts()

    def time_hbonds(self):
        for mol in self.mols:
            hbonds(mol, self.protein)

    def peakmem_hbonds(self):
        self.time_hbonds()
