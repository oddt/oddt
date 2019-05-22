from oddt.docking.AutodockVina import autodock_vina
from oddt.docking.GeneticAlgorithm import GeneticAlgorithm
from oddt.docking.CustomEngine import CustomEngine


class Dock(object):
    """
    Universal pipeline for molecular docking. There are two types of engines:
    1. extended AutodockVina with auto-centering on ligand
    2. CustomEngine with separately defined space sampling method and scoring function

    Parameters
    ----------
    receptor: Molecule
        Protein object to be used while generating descriptors.

    ligands: Molecule or list of them
        Molecules, to dock. Currently mcmc/ga methods supports only single molecule.

    docking_type: string
        Docking engine type:
        1. AutodockVina (from oddt.docking.AutodockVina)
        2. GeneticAlgorithm (from oddt.docking.GeneticAlgorithm)
        3. MCMC (from oddt.docking.MCMC)

    scoring_func: String
        Scoring functions are applied only for CustomEngine.
        1. nnscore ( from oddt.scoring.nnscore )
        2. rfscore ( from oddt.scoring.rfscore )
        3. interaction energy between receptor and ligand (default)

    sampling_params: dict
        Parameters unique for corresponding sampling method (MCMC or GeneticAlgorithm).

    Returns
    -------
    conformation, score: np.array, float
        Best conformation with its score.

    """

    def __init__(self, receptor, ligands, docking_type, scoring_func=None, sampling_params={}):
        self.docking_type = docking_type
        self.ligands = ligands
        self.scoring_function = scoring_func
        self.output = None
        if self.docking_type == 'AutodockVina':
            self.engine = autodock_vina(protein=receptor)
        else:
            if isinstance(self.ligands, list):
                raise Exception('Currently MCMC/GeneticAlgorithms methods supports only single molecule.')
            self.custom_engine = CustomEngine(receptor, lig=ligands, scoring_func=scoring_func)
            if self.docking_type == 'MCMC':
                # self.engine = MCMCAlgorithm(self.custom_engine, sampling_params)
                pass
            elif self.docking_type == 'GeneticAlgorithm':
                self.engine = GeneticAlgorithm(self.custom_engine, **sampling_params)
            else:
                raise Exception('Choose supported sampling algorithm.')

        self.perform()

    def perform(self):
        if self.docking_type == 'AutodockVina':
            self.engine.dock(self.ligand)
        else:  # MCMC / GeneticAlgorithm
            self.output = self.engine.perform()

        # TODO
        # self.save_output()
