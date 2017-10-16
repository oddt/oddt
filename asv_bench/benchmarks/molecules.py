import os
import oddt

test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

class CommonBenchMolecule(object):
    def setup(self):
        raise NotImplementedError

    def time_atom_dict(self):
        for mol in self.mols:
            mol.atom_dict
            mol._clear_cache()

    def time_atom_iter(self):
        for mol in self.mols:
            for atom in mol:
                pass

    def time_atom_getter(self):
        for mol in self.mols:
            for i in range(len(mol.atoms)):
                atom = mol.atoms[i]


class BenchSmallMolecule(CommonBenchMolecule):
    """Test molecule methods"""
    def setup(self):
        self.mols = list(oddt.toolkit.readfile('sdf', '%s/tests/data/dude/xiap/actives_docked.sdf' % test_data_dir))[:10]


class BenchProteinMolecule(CommonBenchMolecule):
    """Test molecule methods"""
    goal_time = 1.

    def setup(self):
        self.mols = list(oddt.toolkit.readfile('pdb', '%s/tests/data/dude/xiap/receptor_rdkit.pdb' % test_data_dir))
        for mol in self.mols:
            mol.protein = True
