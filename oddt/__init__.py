"""Open Drug Discovery Toolkit
==============================
Universal and easy to use resource for various drug discovery tasks, ie docking, virutal screening, rescoring.

    Attributes
    ----------
    toolkit : module,
        Toolkits backend module, currenlty OpenBabel [ob] and RDKit [rdk].
        This setting is toolkit-wide, and sets given toolkit as default
"""

from numpy.random import seed as np_seed
from random import seed as python_seed
from .toolkits import ob, rdk

toolkit = ob
__all__ = ['toolkit']

def random_seed(i):
    """
    Set global random seed for all underlying components. Use 'brute-force' approach, by setting undelying libraries' seeds.
    
    Parameters
    ----------
        i: int 
            integer used as seed for random number generators
    """
    # python's random module
    python_seed(i)
    # numpy random module
    np_seed(i)
