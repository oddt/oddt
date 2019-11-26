from enum import Enum
import numpy as np
from scipy.optimize import minimize

from oddt import random_seed
from oddt.docking.MCMCUtils import MCMCUtils
from oddt.docking.internal import generate_rotor_vector


class OptimizationMethod(Enum):
    """
    method of optimization (one of SIMPLEX/NO_OPTIM, NELDER_MEAD or LBFGSB)
    """
    SIMPLEX = 1
    NO_OPTIM = 1
    NELDER_MEAD = 2
    LBFGSB = 3


class MCMCAlgorithm(object):
    def __init__(self, engine, optim=OptimizationMethod.NELDER_MEAD, optim_iter=10, mc_steps=50,
                 mut_steps=100, seed=None, ):
        """

        Parameters
        ----------
        engine: CustomEngine
            engine with prepared molecules and defined scoring function

        optim: OptimizationMethod
            method of optimization (or SIMPLEX/NO_OPTIM if none) around locally chosen conformation point,
            must be one of the values in OptimizationMethod enumeration

        optim_iter: int (default=10)
            number of iterations for local optimization, in the scipy.optimize.minimize

        mc_steps: int (default=50)
            number of steps performed by the MCMC algorithm

        mut_steps: int (100)
            number of mutation steps (while choosing next random conformation)

        seed: int
            seed for the pseudonumber generators
        """

        self.engine = engine
        self.ligand = engine.lig
        self.optim = optim
        self.optim_iter = optim_iter
        self.mc_steps = mc_steps
        self.mut_steps = mut_steps
        if seed:
            random_seed(seed)

        self.num_rotors = len(self.engine.rotors)

        self.lig_dict = self.engine.lig_dict
        self.mcmc_utils = MCMCUtils()

    def perform(self):
        """
        performs the algorithm

        Returns
        -------
        conformation, score: float[], float
            best conformation and best score for this conformation

        """

        x1 = generate_rotor_vector(self.num_rotors)
        c1 = self.engine.lig.mutate(x1)
        e1 = self.engine.score(c1)
        out = {'score': e1, 'conformation': c1.copy().tolist()}

        for _ in range(self.mc_steps):
            c2, x2 = self.generate_conformation(x1)
            e2 = self.engine.score(c2)
            e3, x3 = self._optimize(e2, x2)

            delta = e3 - e1

            if delta < 0 or np.exp(-delta) > np.random.uniform():  # Metropolis criterion
                x1 = x3
                if delta < 0:
                    e1 = e3
                    conformation = self.engine.lig.mutate(x1.copy())
                    out = {'score': e1, 'conformation': conformation.tolist()}

        return out['conformation'], out['score']

    def _optimize(self, e2, x2):

        bounds = ((-1., 1.), (-1., 1.), (-1., 1.), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi))
        for i in range(len(self.engine.rotors)):
            bounds += ((-np.pi, np.pi),)
        bounds = np.array(bounds)

        if self.optim == OptimizationMethod.SIMPLEX:
            return e2, x2
        elif self.optim == OptimizationMethod.NELDER_MEAD:
            return self._minimize_nelder_mead(x2)
        elif self.optim == OptimizationMethod.LBFGSB:
            return self._minimize_lbfgsb(bounds, x2)
        return e2, x2

    def _minimize_nelder_mead(self, x2):

        m = minimize(self.mcmc_utils.score_coords, x2, args=(self.engine, self.engine.score),
                     method='Nelder-Mead')
        e3, x3 = self._extract_from_scipy_minimize(m)
        return e3, x3

    def _extract_from_scipy_minimize(self, m):

        x3 = m.x
        x3 = self.mcmc_utils.keep_bound(x3)
        e3 = m.fun
        return e3, x3

    def _minimize_lbfgsb(self, bounds, x2):

        m = minimize(self.mcmc_utils.score_coords, x2, method='L-BFGS-B',
                     jac=self.mcmc_utils.score_coords_jac,
                     args=(self.engine, self.engine.score), bounds=bounds, options={'maxiter': self.optim_iter})
        e3, x3 = self._extract_from_scipy_minimize(m)
        return e3, x3

    def generate_conformation(self, x1):

        for _ in range(self.mut_steps):
            x2 = self.mcmc_utils.rand_mutate_big(x1.copy())
            c2 = self.engine.lig.mutate(x2)
        return c2, x2
