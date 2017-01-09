import os
from tempfile import NamedTemporaryFile

import numpy as np

from nose.tools import assert_in, assert_not_in, assert_equal
from sklearn.utils.testing import (assert_true,
                                   assert_array_equal,
                                   assert_array_almost_equal)

import oddt
from oddt.scoring.functions import rfscore

test_data_dir = os.path.dirname(os.path.abspath(__file__))


def test_rfscore():
    mols = list(oddt.toolkit.readfile('sdf', os.path.join(test_data_dir, 'data/dude/xiap/actives_docked.sdf')))
    rec = next(oddt.toolkit.readfile('pdb', os.path.join(test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb')))

    # Delete molecule which has differences in Acceptor-Donor def in RDK and OB
    del mols[65]

    # test descriptors
    for v in [1, 2, 3]:
        descs = rfscore(version=v, protein=rec).descriptor_generator.build(mols)
        # save correct results (for future use)
        # np.savetxt(os.path.join(test_data_dir,
        #                         'data/results/xiap/rfscore_v%i_descs.csv' % v),
        #            descs,
        #            fmt='%.16g',
        #            delimiter=',')
        descs_correct = np.loadtxt(os.path.join(test_data_dir, 'data/results/xiap/rfscore_v%i_descs.csv' % v), delimiter=',')
        assert_array_almost_equal(descs, descs_correct, decimal=4)
