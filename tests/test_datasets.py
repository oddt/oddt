import os

from nose.tools import assert_equal, assert_is_instance, assert_greater

import oddt
from oddt.datasets import pdbbind

test_data_dir = os.path.dirname(os.path.abspath(__file__))


def test_pdbbind():

    results = {
        'core': (['1y6r', '10gs'],
                 [10.11, 6.4]),
        'refined': (['4up5', '4x6p', '1y6r', '10gs'],
                    [2.60, 8.30, 10.11, 6.4]),
        'general_PL': (['4uyh', '4qf8', '4up5', '4x6p', '1y6r', '10gs'],
                       [6.30, 7.51, 2.60, 8.30, 10.11, 6.4]),
    }

    for year in [2007, 2013, 2016]:
        pdbbind_db = pdbbind(home=os.path.join(test_data_dir, 'data', 'pdbbind'),
                             version=year, default_set='core')

        for set_name, (ids, activities) in results.items():
            if set_name == 'general_PL' and year == 2007:
                set_name = 'general'
            pdbbind_db.default_set = set_name
            assert_equal(pdbbind_db.ids, ids)
            assert_equal(pdbbind_db.activities, activities)

            for pid in pdbbind_db:
                assert_is_instance(pid.pocket, oddt.toolkit.Molecule)
                assert_greater(len(pid.pocket.atoms), 0)
                assert_is_instance(pid.ligand, oddt.toolkit.Molecule)
                assert_greater(len(pid.ligand.atoms), 0)
                if pid.id == '1y6r':
                    assert_equal(pid.protein, None)
                else:
                    assert_is_instance(pid.protein, oddt.toolkit.Molecule)
                    assert_greater(len(pid.protein.atoms), 0)

        # getting by name
        assert_equal(pdbbind_db['10gs'].id, '10gs')

        # getting by id
        assert_equal(pdbbind_db[-1].id, '10gs')

        # TODO: getting by name error
        # assert_raises(KeyError, pdbbind_db.__getitem__, 'xxxx')

        # TODO: getting by id error
        # assert_raises(KeyError, pdbbind_db.__getitem__, 123)

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
