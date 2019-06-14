from joblib import Parallel, delayed
from oddt.docking.AutodockVina import autodock_vina
from oddt.docking.GeneticAlgorithm import GeneticAlgorithm
from oddt.docking.MCMCAlgorithm import MCMCAlgorithm
from oddt.docking.CustomEngine import CustomEngine
from oddt.docking.internal import write_ligand_to_pdbqt


def dock_single_molecule(docker, directory):
    conformation, score = docker.perform()
    ligand = docker.engine.ligand.clone
    ligand.coords = conformation
    if directory:
        # if directory is given, then molecule is saved to file
        write_ligand_to_pdbqt(directory, ligand)
    return ligand, score


class Dock(object):
    """
    Universal pipeline for molecular docking. There are two types of engines:
    1. extended AutodockVina with auto-centering on ligand
    2. CustomEngine with separately defined space sampling method and scoring function

    Parameters
    ----------
    receptor: Molecule or file name
        Protein object to be used while generating descriptors.

    ligands: Molecule or list of them or sdf file name with ligands
        Molecules, to dock.

    docking_type: string
        Docking engine type:
        1. AutodockVina (from oddt.docking.AutodockVina)
        2. GeneticAlgorithm (from oddt.docking.GeneticAlgorithm)
        3. MCMC (from oddt.docking.MCMC)

    scoring_func: String
        Scoring functions are applied only for CustomEngine.
        1. interaction_energy between receptor and ligand (default)
        2. nnscore ( from oddt.scoring.nnscore )
        3. rfscore ( from oddt.scoring.rfscore )
        4. ri_score ( from.oddt.scoring.ri_score)

     n_jobs: int (default=-1)
            Number of cores to use for docking, -1 means all are allocated.

    additional_params: dict
        Additional parameters specific for chosen engine.

    """

    def __init__(self, receptor, ligands, docking_type, scoring_func='interaction_energy', n_jobs=-1,
                 additional_params={}):
        self.docking_type = docking_type
        self.ligands = ligands
        self.scoring_function = scoring_func
        self.n_jobs = n_jobs
        self.output = []
        if not isinstance(ligands, list):
            self.ligands = [ligands]

        if self.docking_type == 'AutodockVina':
            self.engine = autodock_vina(protein=receptor, **additional_params)
        else:
            # CustomEngine
            self.custom_engines = []
            # for every ligand there's separate engine spawned
            for ligand in self.ligands:
                custom_engine = CustomEngine(receptor, lig=ligand, scoring_func=scoring_func)
                if self.docking_type == 'MCMC':
                    self.custom_engines.append(MCMCAlgorithm(custom_engine, **additional_params))
                    pass
                elif self.docking_type == 'GeneticAlgorithm':
                    self.custom_engines.append(GeneticAlgorithm(custom_engine, **additional_params))
                else:
                    raise Exception('Choose supported docking type.')

    def dock(self, directory=None):
        if self.docking_type == 'AutodockVina':
            self.output = self.engine.dock(self.ligands)
        else:
            # MCMC / GeneticAlgorithm
            self.output = Parallel(n_jobs=self.n_jobs, backend='threading', verbose=13, pre_dispatch='all')(
                delayed(dock_single_molecule)(docker, directory) for docker in self.custom_engines)
        return self.output
