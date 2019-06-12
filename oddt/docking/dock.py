from oddt.docking.AutodockVina import autodock_vina
from oddt.docking.GeneticAlgorithm import GeneticAlgorithm
from oddt.docking.CustomEngine import CustomEngine
from oddt.docking.internal import write_ligand_to_pdbqt


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

    additional_params: dict
        Additional parameters specific for chosen engine.

    Returns
    -------
    conformation, score: np.array, float
        Best conformation with its score.

    """

    def __init__(self, receptor, ligands, docking_type, scoring_func=None, additional_params={}):
        self.docking_type = docking_type
        self.ligands = ligands
        self.scoring_function = scoring_func
        self.output = None
        if self.docking_type == 'AutodockVina':
            self.engine = autodock_vina(protein=receptor, **additional_params)
        else:
            if isinstance(self.ligands, list):
                raise Exception('Currently MCMC/GeneticAlgorithms methods supports only single molecule.')
            self.custom_engine = CustomEngine(receptor, lig=ligands, scoring_func=scoring_func)
            if self.docking_type == 'MCMC':
                # self.engine = MCMCAlgorithm(self.custom_engine, **additional_params)
                pass
            elif self.docking_type == 'GeneticAlgorithm':
                self.engine = GeneticAlgorithm(self.custom_engine, **additional_params)
            else:
                raise Exception('Choose supported sampling algorithm.')

    def perform(self, directory=None):
        if self.docking_type == 'AutodockVina':
            self.engine.dock(self.ligands)
        else:  # MCMC / GeneticAlgorithm
            self.output = self.engine.perform()

            # save found conformation
            if directory:
                conformation, score = self.output
                self.ligands.set_coords(conformation)
                write_ligand_to_pdbqt(directory, self.ligands)
