import numpy as np

from minimization.function import Function
from minimization.unconstrained.gradient_descent import GradientDescent
from minimization.unconstrained.line_search import ConstantLineSearch, BacktrackingLineSearch, ExactLineSearch
from minimization.unconstrained.newton import Newton
from minimization.unconstrained.steepest_descent import SteepestDescent


class FunctionTeam2(Function):

    def __init__(self, gamma):
        """
        :param float gamma:
        """
        self.gamma = gamma

    def __call__(self, x):
        return 3/2 * (self.gamma * x[0]**2 + x[1]**2)

    def gradient(self, x):
        return np.asarray([
            3 * self.gamma * x[0],
            3 * x[1]
        ])

    def hessian(self, x):
        return np.asarray([
            [3 * self.gamma, 0],
            [0             , 3]
        ])


class Data(object):
    def __init__(self):
        self.data = []

    def save_history_exact(self, method, id_parameter, history):
        for index_step, step in enumerate(history):
            x_0, x_1 = step[0]
            error = step[1]
            self.append(method, 'exact', id_parameter, None, None, index_step, x_0, x_1, error)

    def save_history_backtracking(self, method, id_parameter, alpha, beta, history):
        for index_step, step in enumerate(history):
            x_0, x_1 = step[0]
            error = step[1]

            self.append(method, 'backtracking', id_parameter, alpha, beta, index_step, x_0, x_1, error)

    def append(self, method, line_search, id_parameter, alpha, beta, index_step, x_0, x_1, error):
        self.data.append([
            method.__class__.__name__, line_search,
            id_parameter, alpha, beta,
            index_step, x_0, x_1,
            error
        ])

    def to_dataframe(self):
        return pd.DataFrame(
            data=self.data,
            columns=['method', 'line_search', 'id_parameter', 'alpha', 'beta', 'step', 'x_0', 'x_1', 'error']
        )



methods_exact_line_search = [
    #GradientDescent(ConstantLineSearch(constant=.25)),
    GradientDescent(ExactLineSearch()),
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

data = Data()

alphas = np.arange(0.01, 0.5, (0.5 - 0.01)/40)
betas = np.arange(0.01, 0.9, (0.9 - 0.01)/40)


for id_parameter, row in parameters.iterrows():
    x_0, x_1, gamma = row
    x = np.asarray([x_0, x_1])

    f = FunctionTeam2(gamma=gamma)

    for method in methods_exact_line_search:
        history = method.minimize(f, x)

        data.save_history_exact(method, id_parameter, history)

    for Method in methods_backtracking_line_search:
        for alpha, beta in product(alphas, betas):
            method = Method(BacktrackingLineSearch(alpha=alpha, beta=beta))
            history = method.minimize(f, x)

            data.save_history_backtracking(method, id_parameter, alpha, beta, history)


data.to_dataframe().to_csv('results.csv', sep=';', index=False)
