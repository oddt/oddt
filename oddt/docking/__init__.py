from .AutodockVina import AutoDockVina
from ._compat import deprecated_class_alias as _deprecated_class_alias
from .base import DockingBase, LigandPose
from .internal import OddtVinaDocking, OddtVinaScoring, OddtVinaGA, OddtVinaMC, OddtVinaDE, OddtVinaACO, OddtVinaBO
from .vinardo import OddtVinardoScoring, OddtVinardoGA, OddtVinardoMC, OddtVinardoDE, OddtVinardoACO, OddtVinardoBO
from .ad4 import OddtAD4Scoring, OddtAD4GA, OddtAD4MC, OddtAD4DE, OddtAD4ACO, OddtAD4BO
from .plp import (
    OddtPLPScoring,
    OddtPLPGA,
    OddtPLPMC,
    OddtPLPDE,
    OddtPLPACO,
    OddtPLPBO,
    OddtChemPLPScoring,
    OddtChemPLPGA,
    OddtChemPLPMC,
    OddtChemPLPDE,
    OddtChemPLPACO,
    OddtChemPLPBO,
)
from .search import (
    AntColonyMixin,
    BayesianOptimizationMixin,
    DifferentialEvolutionMixin,
    GeneticAlgorithmMixin,
    MonteCarloMixin,
)

__all__ = [
    "AutoDockVina",
    "DockingBase",
    "LigandPose",
    "OddtVinaDocking",
    "OddtVinaScoring",
    "OddtVinaGA",
    "OddtVinaMC",
    "OddtVinaDE",
    "OddtVinaACO",
    "OddtVinaBO",
    "OddtVinardoScoring",
    "OddtVinardoGA",
    "OddtVinardoMC",
    "OddtVinardoDE",
    "OddtVinardoACO",
    "OddtVinardoBO",
    "OddtAD4Scoring",
    "OddtAD4GA",
    "OddtAD4MC",
    "OddtAD4DE",
    "OddtAD4ACO",
    "OddtAD4BO",
    "OddtPLPScoring",
    "OddtPLPGA",
    "OddtPLPMC",
    "OddtPLPDE",
    "OddtPLPACO",
    "OddtPLPBO",
    "OddtChemPLPScoring",
    "OddtChemPLPGA",
    "OddtChemPLPMC",
    "OddtChemPLPDE",
    "OddtChemPLPACO",
    "OddtChemPLPBO",
    "AntColonyMixin",
    "BayesianOptimizationMixin",
    "DifferentialEvolutionMixin",
    "GeneticAlgorithmMixin",
    "MonteCarloMixin",
]

_DEPRECATED_ALIASES = {
    "autodock_vina": "AutoDockVina",
    "vina_docking": "OddtVinaDocking",
    "vina_scoring": "OddtVinaScoring",
    "vina_ga": "OddtVinaGA",
    "vina_mc": "OddtVinaMC",
    "vina_de": "OddtVinaDE",
    "vinardo_scoring": "OddtVinardoScoring",
    "vinardo_ga": "OddtVinardoGA",
    "vinardo_mc": "OddtVinardoMC",
    "vinardo_de": "OddtVinardoDE",
    "ad4_scoring": "OddtAD4Scoring",
    "ad4_ga": "OddtAD4GA",
    "ad4_mc": "OddtAD4MC",
    "ad4_de": "OddtAD4DE",
}
_DEPRECATED_ALIASES.update({name[4:]: name for name in _DEPRECATED_ALIASES.values() if name.startswith("Oddt")})
__all__ += list(_DEPRECATED_ALIASES)


def __getattr__(name):
    return _deprecated_class_alias(name, globals(), _DEPRECATED_ALIASES)


def __dir__():
    return sorted(set(globals()) | _DEPRECATED_ALIASES.keys())
