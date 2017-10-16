import os
import oddt
from oddt.fingerprints import (ECFP,
                               _ECFP_atom_repr,
                               _ECFP_atom_hash,
                               PLEC)

test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

class BenchECFP(object):
    goal_time = 0.5

    def setup(self):
        self.mols = list(oddt.toolkit.readfile('sdf', '%s/tests/data/dude/xiap/actives_docked.sdf' % test_data_dir))[:10]
        for mol in self.mols:
            mol.atom_dict

    def time_ecfp(self):
        for mol in self.mols:
            ECFP(mol, depth=4)

    def time_ecfp_dense(self):
        for mol in self.mols:
            ECFP(mol, depth=4, sparse=False)

    def time_ecfp_pharm(self):
        for mol in self.mols:
            ECFP(mol, depth=4, use_pharm_features=True)

    def time_ecfp_atom_repr(self):
        for mol in self.mols:
            for atom in mol.atoms:
                if atom.atomicnum > 1:
                    _ECFP_atom_repr(mol, atom.idx0)

    def time_ecfp_atom_hash(self):
        for mol in self.mols:
            for atom in mol.atoms:
                if atom.atomicnum > 1:
                    _ECFP_atom_hash(mol, atom.idx0, depth=8)
                    break  # one atom is enough


class BenchPLEC(object):
    def setup(self):
        self.mols = list(oddt.toolkit.readfile('sdf', '%s/tests/data/dude/xiap/actives_docked.sdf' % test_data_dir))[:10]
        for mol in self.mols:
            mol.atom_dict

        self.rec = list(oddt.toolkit.readfile('pdb', '%s/tests/data/dude/xiap/receptor_rdkit.pdb' % test_data_dir))[0]
        self.rec.protein = True
        self.rec.atom_dict

    def time_plec(self):
        for mol in self.mols:
            PLEC(mol, self.rec, depth_ligand=2, depth_protein=4)


class BenchPLECwithHs(BenchPLEC):
    def setup(self):
        super(BenchPLECwithHs, self).setup()
        for mol in self.mols:
            mol.addh()
            mol.atom_dict
        self.rec.addh()
        self.rec.atom_dict
