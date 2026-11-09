"""Independent PLP/ChemPLP potentials and Cartesian derivatives.

Equations 3.7--3.20 in O. Korb's 2008 thesis, urn:nbn:de:bsz:352-opus-64838.
Only scientific equations and numerical facts are used, not upstream code.
"""

import numpy as np


def plp_potential(r, a, b, c, d, e, f=20.0):
    """Attractive PLP potential, and its distance derivative (Eq. 3.7)."""
    r, a, b, c, d, e, f = np.broadcast_arrays(r, a, b, c, d, e, f)
    value = np.zeros(r.shape)
    slope = np.zeros(r.shape)
    for mask, v, g in (
        (r < a, f * (a - r) / a, -f / a),
        ((r >= a) & (r < b), e * (r - a) / (b - a), e / (b - a)),
        ((r >= b) & (r < c), e, np.zeros(r.shape)),
        ((r >= c) & (r <= d), e * (d - r) / (d - c), -e / (d - c)),
    ):
        value[mask], slope[mask] = v[mask], g[mask]
    return value, slope


def repulsive_potential(r, weight):
    """Purely repulsive PLP potential (Eq. 3.8, Table 3.4)."""
    r = np.asarray(r, dtype=float)
    first = r < 3.2
    value = np.where(first, 20 + r * (0.1 - 20) / 3.2, 0.1 * (5 - r) / 1.8)
    slope = np.where(first, (0.1 - 20) / 3.2, -0.1 / 1.8)
    return weight * np.where(r < 5, value, 0), weight * np.where(r < 5, slope, 0)


def ramp(x, lower, upper):
    """ChemScore decreasing piecewise-linear factor and derivative (Eq. 3.13)."""
    x = np.asarray(x, dtype=float)
    return np.clip((upper - x) / (upper - lower), 0, 1), np.where((x > lower) & (x < upper), -1 / (upper - lower), 0)


def angle_factor(first, center, last, lower, upper, gradient=True):
    """Factor f(|angle - 180|) with gradients on all three defining atoms."""
    u, v = first - center, last - center
    ru, rv = np.linalg.norm(u, axis=-1), np.linalg.norm(v, axis=-1)
    if np.any(ru < 1e-10) or np.any(rv < 1e-10):
        raise ValueError("ChemPLP angular scoring requires nonzero bonded and contact distances")
    u, v = u / ru[..., None], v / rv[..., None]
    cosine = np.clip(np.sum(u * v, axis=-1), -1, 1)
    angle = np.arccos(cosine)
    value, derivative = ramp(180 - np.rad2deg(angle), lower, upper)
    if not gradient:
        return value, None
    factor = derivative * (180 / np.pi) / np.maximum(np.sqrt(1 - cosine**2), 1e-12)
    g_first = factor[..., None] * (v - cosine[..., None] * u) / ru[..., None]
    g_last = factor[..., None] * (u - cosine[..., None] * v) / rv[..., None]
    return value, (g_first, -g_first - g_last, g_last)


def product_factors(factors):
    """Product and partial derivatives, including at zero factors."""
    factors = np.asarray(factors)
    before = np.concatenate((np.ones_like(factors[:1]), np.cumprod(factors, axis=0)[:-1]), axis=0)
    after = np.concatenate((np.cumprod(factors[::-1], axis=0)[-2::-1], np.ones_like(factors[:1])), axis=0)
    return factors.prod(axis=0), before * after


def torsional_potential(coords, indices, barriers, signs, periods, gradient=True):
    """Four-atom cosine torsions (Eq. 3.12), with no angle branch-cut singularity."""
    if not len(indices):
        return 0.0, np.zeros_like(coords) if gradient else None
    points = coords[indices]
    b0, b1, b2 = points[:, 1] - points[:, 0], points[:, 2] - points[:, 1], points[:, 3] - points[:, 2]
    n0, n1 = np.cross(b0, b1), np.cross(b1, b2)
    r0, r1 = np.linalg.norm(n0, axis=1), np.linalg.norm(n1, axis=1)
    if np.any(r0 < 1e-10) or np.any(r1 < 1e-10):
        raise ValueError("PLP torsion scoring requires noncollinear bonded coordinates")
    n0, n1 = n0 / r0[:, None], n1 / r1[:, None]
    cosine = np.clip(np.sum(n0 * n1, axis=1), -1, 1)
    # cos(n phi) = T_n(cos(phi)); differentiating the polynomial avoids
    # dividing by sin(phi) at perfectly planar conformations.
    previous, current = np.ones_like(cosine), cosine
    d_previous, d_current = np.zeros_like(cosine), np.ones_like(cosine)
    values, derivatives = current.copy(), d_current.copy()
    for order in range(2, int(np.max(periods)) + 1):
        previous, current, d_previous, d_current = (
            current,
            2 * cosine * current - previous,
            d_current,
            2 * current + 2 * cosine * d_current - d_previous,
        )
        mask = periods == order
        values[mask], derivatives[mask] = current[mask], d_current[mask]
    energy = float(np.sum(0.5 * barriers * (1 + signs * values)))
    if not gradient:
        return energy, None
    coefficient = (0.5 * barriers * signs * derivatives)[:, None]
    a0 = coefficient * (n1 - cosine[:, None] * n0) / r0[:, None]
    a1 = coefficient * (n0 - cosine[:, None] * n1) / r1[:, None]
    g0, g1, g2 = np.cross(b1, a0), np.cross(a0, b0) + np.cross(b2, a1), np.cross(a1, b1)
    forces = np.stack((-g0, g0 - g1, g1 - g2, g2), axis=1)
    cartesian = np.zeros_like(coords)
    np.add.at(cartesian, indices.ravel(), forces.reshape(-1, 3))
    return energy, cartesian
