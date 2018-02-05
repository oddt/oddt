import os

import pytest

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

    with pytest.raises(ValueError):
        pdbbind(home=os.path.join(test_data_dir, 'data', 'pdbbind'))

    for year in [2007, 2013, 2016]:
        pdbbind_db = pdbbind(home=os.path.join(test_data_dir, 'data', 'pdbbind'),
                             version=year, default_set='core')

        for set_name, (ids, activities) in results.items():
            if set_name == 'general_PL' and year == 2007:
                set_name = 'general'
            pdbbind_db.default_set = set_name
            assert pdbbind_db.ids == ids
            assert pdbbind_db.activities == activities

            for pid in pdbbind_db:
                assert isinstance(pid.pocket, oddt.toolkit.Molecule)
                assert len(pid.pocket.atoms) > 0
                assert isinstance(pid.ligand, oddt.toolkit.Molecule)
                assert len(pid.ligand.atoms) > 0
                if pid.id == '10gs':
                    assert pid.protein is None
                else:
                    assert isinstance(pid.protein, oddt.toolkit.Molecule)
                    assert len(pid.protein.atoms) > 0

        # reset the pdbbind set
        pdbbind_db.default_set = 'refined'

        # getting by name
        assert pdbbind_db['1imx'].id == '1imx'

        # getting by id
        assert pdbbind_db[-3].id == '1imx'
        assert pdbbind_db[1].id == '1imx'

        with pytest.raises(KeyError):
            pdbbind_db['xxxx']
        with pytest.raises(KeyError):
            pdbbind_db[123456]
        with pytest.raises(KeyError):
            pdbbind_db[-123456]

        pid = pdbbind_db['1imx']
        # get ligand
        ligand = pid.ligand
        ligand.removeh()
        assert len(ligand.atoms) == 60

        # get pocket
        pocket = pid.pocket
        pocket.removeh()
        assert len(pocket.atoms) == 234

        # protein do exist
        protein = pid.protein
        protein.removeh()
        assert len(protein.atoms) == 478


def test_dude():
    results = {
        'fabp4': (1022, 36, 57, 2855),
        'inha': (1857, 22, 71, 2318),
    }

    dude_db = dude(home=os.path.join(test_data_dir, 'data', 'dude'))

    for target in dude_db:
        if target.dude_id == 'xiap':
            # different file names
            assert target.protein is None
            assert target.ligand is None
            assert target.actives is None
            assert target.decoys is None
            continue

        prot_atoms, lig_atoms, num_act, num_dec = results[target.dude_id]

        prot = target.protein
        prot.removeh()
        assert len(prot.atoms) == prot_atoms
        lig = target.ligand
        lig.removeh()
        assert len(lig.atoms) == lig_atoms

        assert len(list(target.actives)) == num_act
        for a in target.actives:
            assert len(a.atoms) > 0
        assert len(list(target.decoys)) == num_dec
        for d in target.decoys:
            assert len(d.atoms) > 0

        with pytest.raises(KeyError):
            dude_db['xxxx']
