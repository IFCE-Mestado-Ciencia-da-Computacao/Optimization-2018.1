import numpy as np
from minimization.function import Function
from minimization.unconstrained.gradient_descent import GradientDescent
from minimization.unconstrained.line_search import ConstantLineSearch, BacktrackingLineSearch, ExactLineSearch
from minimization.unconstrained.steepest_descent import SteepestDescent


class FunctionTeam2(Function):

    def __init__(self, gama):
        """
        :param float gama:
        """
        self.gama = gama

    def __call__(self, x):
        return 3/2 * (self.gama * x[0]**2 + x[1]**2)

    def gradient(self, x):
        return np.asarray([
            3 * self.gama * x[0],
            3 * x[1]
        ])


f = FunctionTeam2(gama=2)

x_0 = np.asarray([9.5, 5])
#search = GradientDescent(ConstantLineSearch(constant=.25))
#search = GradientDescent(BacktrackingLineSearch(alpha=.5, beta=.1))
search = SteepestDescent(ExactLineSearch())
#search = SteepestDescent(BacktrackingLineSearch(alpha=.5, beta=.1))
#search = SteepestDescent(ConstantLineSearch(constant=.25))
history = search.minimize(f, x_0)

print(history)
print(history[0])
print(history[-1])

with open('data.data', mode='wb') as file:
    np.save(file, history)
