import numpy as np

from minimization.function import Function
from minimization.unconstrained.gradient_descent import GradientDescent
from minimization.unconstrained.line_search import ConstantLineSearch, BacktrackingLineSearch, ExactLineSearch
from minimization.unconstrained.newton import Newton
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

    def hessian(self, x):
        return np.asarray([
            [3 * self.gama, 0],
            [0            , 2]
        ])


methods_exact_line_search = [
    #GradientDescent(ConstantLineSearch(constant=.25)),
    SteepestDescent(ExactLineSearch()),
    SteepestDescent(ExactLineSearch()),
]
methods_backtracking_line_search = [
    GradientDescent,
    SteepestDescent,
    Newton
]

from itertools import product
import pandas as pd
parameters = pd.read_csv('parameters.csv', sep=';')

data = []

alphas = np.arange(0.01, 0.9, (0.9 - 0.01)/200)
betas = np.arange(0.01, 0.9, (0.9 - 0.01)/200)

for id_parameter, row in parameters.iterrows():
    x_0, x_1, gama = row
    x = np.asarray([x_0, x_1])

    f = FunctionTeam2(gama=gama)
    for method in methods_exact_line_search:
        history = method.minimize(f, x)

        for index_step, step in enumerate(history):
            data.append([method.__class__.__name__, 'ExactLineSearch',
                         id_parameter, None, None,
                         index_step, step[0], step[1]])

    for Method in methods_backtracking_line_search:
        for alpha, beta in product(alphas, betas):
            method = Method(BacktrackingLineSearch(alpha=alpha, beta=beta))
            history = method.minimize(f, x)

            for index_step, step in enumerate(history):
                data.append([method.__class__.__name__, 'BacktrackingLineSearch',
                             id_parameter, alpha, beta,
                             index_step, step[0], step[1]])


dataframe = pd.DataFrame(data=data, columns=['method', 'line_search', 'id_parameter', 'alpha', 'beta', 'step', 'x_0', 'x_1'])
dataframe.to_csv('results.csv', sep=';', index=False)

#with open('data.data', mode='wb') as file:
#    np.save(file, history)
