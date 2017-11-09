import os
from rdkit import Chem

from rdkit_fixer import (PreparePDBMol,
                         ExtractPocketAndLigand,
                         AtomListToSubMol)

test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', '..', 'test_data')


class BenchRdkitFixer(object):
    """Spatial functions"""

    def setup(self):
        self.mol = Chem.MolFromPDBFile('%s/5ar7.pdb' % test_data_dir)

    def time_prepare(self):
        PreparePDBMol(self.mol)

    def peakmem_prepare(self):
        self.time_prepare()

    def time_extract(self):
        ExtractPocketAndLigand(self.mol)

    def peakmem_extract(self):
        self.time_prepare()

    def time_submol_small(self):
        AtomListToSubMol(self.mol, range(10))

    def peakmem_submol_small(self):
        self.time_submol_small()

    def time_submol_pocket(self):
        AtomListToSubMol(self.mol, range(1000))

    def peakmem_submol_pocket(self):
        self.time_submol_pocket()
