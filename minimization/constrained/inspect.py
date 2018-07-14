import numpy as np
from numpy.dual import norm

from minimization.constrained.constrained_newton_infeasible import ConstrainedNewtonInfeasible


def inspect_2d_feasible(f, h, x, ν, k, lambda_square):
    return np.concatenate([
        [f(x)],
        *x,
        lambda_square
    ])


def inspect_feasible(f, h, x, ν, k, lambda_square):
    return np.concatenate([
        [f(x)],
        lambda_square
    ])


def inspect_2d_infeasible(f, h, x, ν, k):
    return np.concatenate([
        [f(x)],
        *x,
        [norm(ConstrainedNewtonInfeasible.r_primal(f, h, x, ν)), norm(ConstrainedNewtonInfeasible.r_dual(h, x))]
    ])


def inspect_infeasible(f, h, x, ν, k):
    return np.concatenate([
        [f(x)],
        [norm(ConstrainedNewtonInfeasible.r_primal(f, h, x, ν)), norm(ConstrainedNewtonInfeasible.r_dual(h, x))]
    ])
