import os
import re
import oddt
from oddt.utils import is_openbabel_molecule
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
        self.output = []
        if self.docking_type == 'AutodockVina':
            self.engine = autodock_vina(protein=receptor, **additional_params)
        else:
            # CustomEngine
            if isinstance(receptor, str):
                rec_format = receptor.split('.')[-1]
                try:
                    self.receptor = next(oddt.toolkit.readfile(rec_format, receptor))
                except:
                    raise Exception("Unsupported receptor file format.")
            else:
                self.receptor = receptor
            self.receptor.protein = True
            self.receptor.addh(only_polar=True)

            if isinstance(self.ligands, str):
                self.ligands = list(oddt.toolkit.readfile('sdf', ligands))
            if isinstance(self.ligands, oddt.toolkit.Molecule):
                self.ligands = [self.ligands]
            _ = list(map(lambda x: x.addh(only_polar=True), self.ligands))

            self.custom_engines = []
            for ligand in self.ligands:
                assert(isinstance(ligand, oddt.toolkit.Molecule))
                custom_engine = CustomEngine(receptor, lig=ligand, scoring_func=scoring_func)
                if self.docking_type == 'MCMC':
                    # custom_engines.append(MCMCAlgorithm(self.custom_engine, **additional_params))
                    pass
                elif self.docking_type == 'GeneticAlgorithm':
                    self.custom_engines.append(GeneticAlgorithm(custom_engine, **additional_params))
                else:
                    raise Exception('Choose supported sampling algorithm.')

    def perform(self, directory=None):
        if self.docking_type == 'AutodockVina':
            self.engine.dock(self.ligands)
        else:  # MCMC / GeneticAlgorithm
            for engine in self.custom_engines:
                new_output = engine.perform()
                self.output.append(new_output)

                # save found conformation
                if directory:
                    conformation, score = new_output
                    engine.ligand.set_coords(conformation)
                    write_ligand_to_pdbqt(directory, engine.ligand)
