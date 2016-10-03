"""Open Drug Discovery Toolkit
==============================
Universal and easy to use resource for various drug discovery tasks, ie docking, virutal screening, rescoring.

    Attributes
    ----------
    toolkit : module,
        Toolkits backend module, currenlty OpenBabel [ob] and RDKit [rdk].
        This setting is toolkit-wide, and sets given toolkit as default
"""

import os
import subprocess
try:
    from .toolkits import ob
except ImportError:
    ob = None
try:
    from .toolkits import rdk
except ImportError:
    rdk = None

toolkit = None
if ob:
    toolkit = ob
elif rdk:
    toolkit = rdk

try:
    if get_ipython().config:
        ipython_notebook = True
    else:
        ipython_notebook = False
except:
    ipython_notebook = False


def get_version():
    home = os.path.dirname(__file__)
    git_v = None
    v = '0.2.0'
    if os.path.isdir(home + '/../.git'):
        try:
            git_v = str(subprocess.check_output(['git', 'describe', '--tags'], cwd=home).strip())
        except subprocess.CalledProcessError:  # catch errors, eg. no git installed
            pass
    if git_v:
        v = git_v
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
    from numpy.random import seed as np_seed
    from random import seed as python_seed

    # python's random module
    python_seed(i)
    # numpy random module
    np_seed(i)
