"""Open Drug Discovery Toolkit
==============================
Universal and easy to use resource for various drug discovery tasks,
ie docking, virutal screening, rescoring.

    Attributes
    ----------
    toolkit : module,
        Toolkits backend module, currenlty OpenBabel [ob] and RDKit [rdk].
        This setting is toolkit-wide, and sets given toolkit as default
"""
from __future__ import absolute_import
import os
import subprocess
try:
    from oddt.toolkits import ob
except ImportError:
    ob = None
try:
    from oddt.toolkits import rdk
except ImportError:
    rdk = None

toolkit = None
if 'ODDT_TOOLKIT' in os.environ:
    if os.environ['ODDT_TOOLKIT'] == 'ob':
        if ob is None:
            raise Exception('OpenBabel toolkit is forced by ODDT_TOOLKIT, '
                            'but can\'t be imported')
        toolkit = ob
    elif os.environ['ODDT_TOOLKIT'] == 'rdk':
        if rdk is None:
            raise Exception('RDKit toolkit is forced by ODDT_TOOLKIT, '
                            'but can\'t be imported')
        toolkit = rdk
elif ob:
    toolkit = ob
elif rdk:
    toolkit = rdk
else:
    raise Exception('No toolkit is present. Install OpenBabel or RDKit')


def get_version():
    home = os.path.dirname(__file__)
    git_v = None
    v = '0.3.0'
    if os.path.isdir(home + '/../.git'):
        try:
            git_v = str(subprocess.check_output(['git',
                                                 'describe',
                                                 '--tags'], cwd=home).strip())
        except subprocess.CalledProcessError:  # catch errors, eg. no git installed
            pass
    if git_v:
        v = git_v
    return v

__version__ = get_version()
__all__ = ['toolkit']


def random_seed(i):
    """
    Set global random seed for all underlying components.
    Use 'brute-force' approach, by setting undelying libraries' seeds.

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
