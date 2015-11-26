"""Open Drug Discovery Toolkit
==============================
Universal and easy to use resource for various drug discovery tasks, ie docking, virutal screening, rescoring.

    Attributes
    ----------
    toolkit : module,
        Toolkits backend module, currenlty OpenBabel [ob] and RDKit [rdk].
        This setting is toolkit-wide, and sets given toolkit as default
"""

import os, subprocess
from numpy.random import seed as np_seed
from random import seed as python_seed
try:
    from .toolkits import ob
except ImportError:
    ob = None
try:
    from .toolkits import rdk
except ImportError:
    rdk = None
if ob:
    toolkit = ob
elif rdk:
    toolkit = rdk
else:
    raise Exception('You need at least one toolkit for ODDT.')

def get_version():
    home = os.path.dirname(__file__)
    v = None
    if os.path.isdir(home + '/../.git'):
        try:
            v = subprocess.check_output(['git', 'describe', '--tags'], cwd=home).strip()
        except CalledProcessError: # catch errors, eg. no git installed
            pass
    if not v:
        v = '0.1.4'
    return v

__version__ = get_version()
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
