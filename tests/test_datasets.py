import os

from nose.tools import assert_equal, assert_is_instance

import oddt
from oddt.datasets import pdbbind

test_data_dir = os.path.dirname(os.path.abspath(__file__))


def test_pdbbind():
    for year in [2007, 2013, 2016]:
        pdbbind_db = pdbbind(home=os.path.join(test_data_dir, 'data', 'pdbbind'),
                             version=year, default_set='core')

        # mock dataset has 4 targets
        assert_equal(len([i for i in pdbbind_db]), 4)

        # getting by name
        assert_equal(pdbbind_db['10gs'].id, '10gs')

        # getting by id
        assert_equal(pdbbind_db[3].id, '10gs')

        assert_equal(pdbbind_db.ids, ['1ps3', '4x6p', '1y6r', '10gs'])
        assert_equal(pdbbind_db.activities, [2.28, 8.30, 10.11, 6.4])

        # TODO: getting by name error
        # assert_raises(KeyError, pdbbind_db.__getitem__, 'xxxx')

        # TODO: getting by id error
        # assert_raises(KeyError, pdbbind_db.__getitem__, 123)

        # test all pockets
        for pid in pdbbind_db:
            assert_is_instance(pid.pocket, oddt.toolkit.Molecule)
            assert_is_instance(pid.ligand, oddt.toolkit.Molecule)
            if pid.id == '1ps3':
                assert_equal(pid.protein, None)
            else:
                assert_is_instance(pid.protein, oddt.toolkit.Molecule)

        pid = pdbbind_db['10gs']
        # get ligand
        ligand = pid.ligand
        ligand.removeh()
        assert_equal(len(ligand.atoms), 33)

        # get pocket
        pocket = pid.pocket
        pocket.removeh()
        assert_equal(len(pocket.atoms), 603)

        # protein do exist
        protein = pid.protein
        protein.removeh()
        assert_equal(len(protein.atoms), 3431)
