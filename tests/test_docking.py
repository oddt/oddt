import os
import numpy as np
from numpy.testing import assert_almost_equal
import oddt
from oddt.docking.dock import Dock


test_data_dir = os.path.dirname(os.path.abspath(__file__))
receptor = next(oddt.toolkit.readfile('pdb', os.path.join(
    test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb')))
receptor.protein = True
receptor.addh(only_polar=True)

mols = list(oddt.toolkit.readfile('sdf', os.path.join(
        test_data_dir, 'data/dude/xiap/actives_docked.sdf')))
mols = list(filter(lambda x: x.title == '312335', mols))
_ = list(map(lambda x: x.addh(only_polar=True), mols))
lig = mols[0]


# TODO
# def test_genetic_algorithm():
#     ga_params = {'n_population': 20, 'n_generations': 20, 'top_individuals': 2,
#                  'top_parents': 5, 'crossover_prob': 0.9, 'seed': 123}
#
#     engine = Dock(receptor, lig, docking_type='GeneticAlgorithm', sampling_params=ga_params)
#     _, score = engine.output
#
#     if oddt.toolkit.backend == 'ob':
#         target_score = np.array([1])
#     else:
#         target_score = np.array([1])
#
#     assert_almost_equal(score, target_score, decimal=2)
