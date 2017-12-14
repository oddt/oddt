import os

from nose.tools import (assert_equal, assert_is_instance, assert_greater,
                        assert_raises)

import oddt
from oddt.datasets import pdbbind, dude

test_data_dir = os.path.dirname(os.path.abspath(__file__))


def test_pdbbind():

    results = {
        'core': (['4yef', '10gs'],
                 [5.35, 6.4]),
        'refined': (['1nlp', '1imx', '4yef', '10gs'],
                    [4.96, 3.52, 5.35, 6.4]),
        'general_PL': (['1k9q', '1nlo', '1nlp', '1imx', '4yef', '10gs'],
                       [3.15, 5.47, 4.96, 3.52, 5.35, 6.4]),
    }

    assert_raises(ValueError, pdbbind, home=os.path.join(test_data_dir,
                                                         'data', 'pdbbind'))

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
                if pid.id == '10gs':
                    assert_equal(pid.protein, None)
                else:
                    assert_is_instance(pid.protein, oddt.toolkit.Molecule)
                    assert_greater(len(pid.protein.atoms), 0)

        # reset the pdbbind set
        pdbbind_db.default_set = 'refined'

        # getting by name
        assert_equal(pdbbind_db['1imx'].id, '1imx')

        # getting by id
        assert_equal(pdbbind_db[-3].id, '1imx')
        assert_equal(pdbbind_db[1].id, '1imx')

        assert_raises(KeyError, pdbbind_db.__getitem__, 'xxxx')
        assert_raises(KeyError, pdbbind_db.__getitem__, 123456)
        assert_raises(KeyError, pdbbind_db.__getitem__, -123456)

        pid = pdbbind_db['1imx']
        # get ligand
        ligand = pid.ligand
        ligand.removeh()
        assert_equal(len(ligand.atoms), 60)

        # get pocket
        pocket = pid.pocket
        pocket.removeh()
        assert_equal(len(pocket.atoms), 234)

        # protein do exist
        protein = pid.protein
        protein.removeh()
        assert_equal(len(protein.atoms), 478)


def test_dude():
    results = {
        'fabp4': (1022, 36, 57, 2855),
        'inha': (1857, 22, 71, 2318),
    }

    dude_db = dude(home=os.path.join(test_data_dir, 'data', 'dude'))

    for target in dude_db:
        if target.dude_id == 'xiap':
            # different file names
            assert_equal(target.protein, None)
            assert_equal(target.ligand, None)
            assert_equal(target.actives, None)
            assert_equal(target.decoys, None)
            continue

        prot_atoms, lig_atoms, num_act, num_dec = results[target.dude_id]

        prot = target.protein
        prot.removeh()
        assert_equal(len(prot.atoms), prot_atoms)
        lig = target.ligand
        lig.removeh()
        assert_equal(len(lig.atoms), lig_atoms)

        assert_equal(len(list(target.actives)), num_act)
        for a in target.actives:
            assert_greater(len(a.atoms), 0)
        assert_equal(len(list(target.decoys)), num_dec)
        for d in target.decoys:
            assert_greater(len(d.atoms), 0)

        assert_raises(KeyError, dude_db.__getitem__, 'xxxx')
