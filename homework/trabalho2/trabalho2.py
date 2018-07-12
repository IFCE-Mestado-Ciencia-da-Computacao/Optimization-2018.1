import numpy as np
import pandas as pd

from minimization.constrained.constrained_newton import ConstrainedNewtonFeasible
from minimization.constrained.equality_restrictions import LinearRestrictions
from minimization.constrained.kkt.kkt_solver import KKTSolver, lapark_method, ldl_method
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


f = HomeworkFunction()
h = HomeworkRestrictions()
α = .3#25
β = .8#7
x = np.array([[2., 6.]]).T

#solver = KKTSolver(lapark_method)
solver = KKTSolver(ldl_method)

newton = ConstrainedNewtonFeasible(BacktrackingLineSearch(alpha=α, beta=β), solver)
history = newton.minimize(f, h, x, tolerance=10**-20)

len(history)

data = pd.DataFrame(data=history, columns=['f', 'x_0', 'x_1', 'error'])
#data = pd.DataFrame(data=history, columns=['f', 'x', 'y'])
data.to_csv('results.csv', sep=',', index=False)
