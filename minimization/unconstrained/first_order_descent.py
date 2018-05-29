import numpy as np
from numpy.linalg import norm
from minimization.unconstrained.general_descent_method import GeneralDescentMethod
from abc import ABCMeta, abstractmethod


class FirstOrderDescent(GeneralDescentMethod, metaclass=ABCMeta):

    def __init__(self, line_search):
        """
        :param LineSearch line_search:
        """
        self.line_search = line_search

    def minimize(self, f, x, tolerance=1e-4, iterations=60):
        """
        k = iteration
        Δx = 'step' or 'search direction'
        t = 'step size' or 'step length at iteration k'

        $$x^(k+1) = x + t * Δx$$

        stopping criterion usually of the form ||\gradient f(x)||_2 <= tolerance

        :param Function f:
        :param ndarray x:
        :param float tolerance: $\eta$
        :param int iterations:
        """
        history = [x]

        for k in range(iterations):
            delta_x = self.calculate_delta_x(f, x)
            t = self.line_search.calculate(f, x, delta_x)

            x = x + t * delta_x

            history.append(x)

            if norm(-delta_x) <= tolerance:
                break

        return np.asarray(history)

    @abstractmethod
    def calculate_delta_x(self, f, x):
        pass
