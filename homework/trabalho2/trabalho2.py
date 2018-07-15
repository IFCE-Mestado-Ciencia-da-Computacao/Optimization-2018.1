import timeit
from itertools import product

import numpy as np
import pandas as pd

from minimization.constrained.inspect import inspect_2d_feasible, inspect_2d_infeasible, inspect_feasible, inspect_infeasible
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


class HomeworkRestrictionsFull(LinearRestrictions):

    def __init__(self, x_0, p):
        """
        :param x_0: First position of x
        :param p: Number of restrictions
        """
        n = x_0.shape[0]

        self.A = self.generate_A(n, p)
        self.b = self.A@x_0

    def generate_A(self, n, p):
        """
        :param n: Number of variables
        :param p: Number of restrictions
        """
        a = np.ones((p, p))
        np.fill_diagonal(a, 0)
        b = np.ones((p, n - p))

        return np.concatenate((a, b), axis=1)


def random_x_0(n):
    return np.random.randint(low=10, high=100, size=(n, 1)).astype(dtype=np.float32)


np.random.seed(42)

# Duas variáveis
f = HomeworkFunction()
#h = HomeworkRestrictions()
#x_feasible = np.array([[2, 6]]).T
#x_infeasible = np.array([[9, 6]]).T
#columns_feasible = ['f', 'x_0', 'x_1', 'error']
#columns_infeasible = ['f', 'x_0', 'x_1', 'norm-r_primal', 'norm-r_dual']
#inspect_feasible_function = inspect_2d_feasible
#inspect_infeasible_function = inspect_2d_infeasible


# n variáveis
n = 200
p = 120

x_feasibles = [random_x_0(n) for i in range(5)]
x_infeasibles = [random_x_0(n) for i in range(5)]

columns_feasible = ['f', 'error', *['x_' + str(i) for i in range(1, n+1)]]
columns_infeasible = ['f', 'norm-r_primal', 'norm-r_dual', 'norm_r', *['x_' + str(i) for i in range(1, n)]]
inspect_feasible_function = inspect_feasible
inspect_infeasible_function = inspect_infeasible

methods = [lapark_method, ldl_method, inverse_method, pseudo_inverse_method, kkt_elimination]
newton_feasible = (ConstrainedNewtonFeasible, BacktrackingLineSearch)
newton_infeasible = (ConstrainedNewtonInfeasible, BacktrackingLineSearchInfeasible)

α = .3
β = .8

data_times = []

for index, x_feasible in enumerate(x_infeasibles):
    for method, (Newton, Backtracking) in product(methods, [newton_feasible]):
        h = HomeworkRestrictionsFull(x_feasible, 120)

        solver = KKTSolver(method)

        newton = Newton(Backtracking(alpha=α, beta=β), solver, inspect=inspect_feasible_function)
        start = timeit.default_timer()
        history = newton.minimize(f, h, x_feasible, tolerance=10**-10)
        stop = timeit.default_timer()

        name = h.__class__.__name__ + '-' + Newton.__name__ + '-' + str(index) + '-' + method.__name__ + '-' + Backtracking.__name__

        data = pd.DataFrame(data=history, columns=columns_feasible)
        data.to_csv('results/' + name + '.csv', sep=',', index=False)

        data_times.append([stop-start, *name.split('-')])


for index, x_infeasible in enumerate(x_infeasibles):
    for method, (Newton, Backtracking) in product(methods, [newton_infeasible]):
        h = HomeworkRestrictionsFull(x_feasibles[index], 120)

        solver = KKTSolver(method)

        newton = Newton(Backtracking(alpha=α, beta=β), solver, inspect=inspect_infeasible_function)

        start = timeit.default_timer()
        history = newton.minimize(f, h, x_infeasible, tolerance=10**-10)
        stop = timeit.default_timer()

        name = h.__class__.__name__ + '-' + Newton.__name__ + '-' + str(index) + '-' + method.__name__ + '-' + Backtracking.__name__

        data = pd.DataFrame(data=history, columns=columns_infeasible)
        data.to_csv('results/' + name + '.csv', sep=',', index=False)

        data_times.append([stop - start, *name.split('-')])

times = pd.DataFrame(data=data_times, columns=['time', 'restriction', 'newton', 'x_0', 'method', 'backtracking'])
times.to_csv('results/times.csv', sep=',', index=False)
print(times)
