import numpy as np


class MCMCUtils(object):
    def __init__(self):
        """
        class containing MCMC utils for MCMC algorithm
        """

    def score_coords(self, x, engine, scoring_func):
        """
        score using given coordinates

        Parameters
        ----------
        x: float[]
            conformation

        engine: CustomEngine
            engine with prepared molecules and defined scoring function

        Returns
        -------
        score: float
            score

        """
        c1 = engine.lig.mutate(x)
        return scoring_func(c1)

    def score_coords_jac(self, x, engine, scoring_func, step=1e-2):
        """
        jac function for scipy.optimize.minimize

        Parameters
        ----------
        x: float[]
            conformation

        engine: CustomEngine
            engine with prepared molecules and defined scoring function

        step:
            infinitesimal change in x (for computing the gradient)

        Returns
        -------
        grad: float[]
            gradient of the scoring function

        """
        c1 = engine.lig.mutate(x)
        e1 = scoring_func(c1)
        grad = []
        for i in range(len(x)):
            x_g = x.copy()
            x_g[i] += step  # if i < 3 else 1e-2
            cg = engine.lig.mutate(x_g)
            grad.append(scoring_func(cg))
        return (np.array(grad) - e1) / step

    def _random_angle(self, size=1):
        """
        generation of random angles, in range from -pi to +pi

        Parameters
        ----------
        size: int
            number of random angles to generate

        Returns
        -------
        random: float or float[size]
            random angle (float) if size=1 or list of random angles (float[]) if size>1
        """
        if size > 1:
            return np.random.uniform(-np.pi, np.pi, size=size)
        return np.random.uniform(-np.pi, np.pi)

    def keep_bound(self, x):
        """
        keep bound of conformation

        Parameters
        ----------
        x: float[]
            conformation

        Returns
        -------
        x: float[]
            bounded conformation

        """
        x[:3] = np.clip(x[:3], -1, 1)
        x[3:] = np.clip(x[3:], -np.pi, np.pi)
        return x

    def rand_mutate_big(self, x):
        """
        change elements of conformation randomly

        Parameters
        ----------
        x: float[]
            conformation

        Returns
        -------
        x: float[]
            conformation with randomly changed elements

        """
        x = x.copy()
        m = np.random.randint(0, len(x) - 4)
        if m == 0:  # do random translation
            x[:3] = np.random.uniform(-0.3, 0.3, size=3)
        elif m == 1:  # do random rotation step
            x[3:6] = self._random_angle(size=3)
        else:  # do random dihedral change
            ix = 6 + m - 2
            x[ix] = self._random_angle()
        return x
