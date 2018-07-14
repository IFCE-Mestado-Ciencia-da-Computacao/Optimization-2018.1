import timeit
from itertools import product

import numpy as np
import pandas as pd
from numpy.linalg import norm

from minimization.constrained.backtracking_line_search_infeasible import BacktrackingLineSearchInfeasible
from minimization.constrained.constrained_newton_feasible import ConstrainedNewtonFeasible
from minimization.constrained.constrained_newton_infeasible import ConstrainedNewtonInfeasible
from minimization.constrained.equality_restrictions import LinearRestrictions
from minimization.constrained.kkt.kkt_solver import KKTSolver, lapark_method, ldl_method, inverse_method, \
    pseudo_inverse_method, kkt_elimination
from minimization.function import Function
from minimization.unconstrained.line_search import BacktrackingLineSearch


class HomeworkFunction(Function):

    def __call__(self, x):
        return (x**(3/2)).sum()

    def gradient(self, x):
        return 3/2 * x**(1/2)

    def hessian(self, x):
        diagonal = np.eye(len(x))
        return diagonal * 3/4 * x**(-1/2)


class HomeworkRestrictions(LinearRestrictions):

    def __init__(self):
        self.A = np.array([[1, 1]])
        self.b = np.array([[8]]).T


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


f = HomeworkFunction()
h = HomeworkRestrictions()
α = .3
β = .8
x_feasible = np.array([[2, 6]]).T
x_infeasible = np.array([[9, 6]]).T


methods = [lapark_method, ldl_method, inverse_method, pseudo_inverse_method, kkt_elimination]
newton_feasible = (ConstrainedNewtonFeasible, BacktrackingLineSearch)
newton_infeasible = (ConstrainedNewtonInfeasible, BacktrackingLineSearchInfeasible)


for method, (Newton, Backtracking) in product(methods, [newton_feasible]):
    solver = KKTSolver(method)

    newton = Newton(Backtracking(alpha=α, beta=β), solver)
    start = timeit.default_timer()
    history = newton.minimize(f, h, x_feasible, tolerance=10**-12)
    stop = timeit.default_timer()

    name = method.__name__ + '-' + Newton.__name__ + '-' + Backtracking.__name__

    data = pd.DataFrame(data=history, columns=['f', 'x_0', 'x_1', 'error'])
    data.to_csv('results/' + name + '.csv', sep=',', index=False)

    print(stop-start)

print()

for method, (Newton, Backtracking) in product(methods, [newton_infeasible]):
    solver = KKTSolver(method)

    newton = Newton(Backtracking(alpha=α, beta=β), solver, inspect_2d_infeasible)

    start = timeit.default_timer()
    history = newton.minimize(f, h, x_infeasible, tolerance=10**-12)
    stop = timeit.default_timer()

    name = method.__name__ + '-' + Newton.__name__ + '-' + Backtracking.__name__

    data = pd.DataFrame(data=history, columns=['f', 'x_0', 'x_1', 'norm-r_primal', 'norm-r_dual'])
    data.to_csv('results/' + name + '.csv', sep=',', index=False)

    print(stop-start)
