from minimization.constrained.ConstrainedNewton import ConstrainedNewtonFeasible
from minimization.constrained.LinearRestrictions import LinearRestrictions
from minimization.function import Function
from minimization.unconstrained.line_search import BacktrackingLineSearch

import numpy as np


class HomeworkFunction(Function):

    def __call__(self, x):
        return x**3/2

    def gradient(self, x):
        return 3/2 * x ** 1/2

    def hessian(self, x):
        diagonal = np.eye(len(x))
        return diagonal * 3/4 * x**(-1/2)


class HomeworkRestrictions(LinearRestrictions):

    def __init__(self):
        self.A = np.array([[1, 1]])
        self.b = np.array([[8]]).T


f = HomeworkFunction()
h = HomeworkRestrictions()
α = .2
β = .5

newton = ConstrainedNewtonFeasible(BacktrackingLineSearch(alpha=α, beta=β))
x = [0, 8]
newton.minimize(f, x, h, tolerance=10**-8)
