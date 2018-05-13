import numpy as np
from numpy.linalg import norm
from minimization.unconstrained.general_descent_method import GeneralDescentMethod


class GradientDescent(GeneralDescentMethod):

    def __init__(self, line_search):
        """
        :param LineSearch line_search:
        """
        self.line_search = line_search

    def minimize(self, f, x, tolerance=1e-4, iterations=int(1e3)):
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
            delta_x = - f.gradient(x)
            t = self.line_search.calculate(f, x)

            x = x + t * delta_x

            history.append(x)

            if norm(-delta_x) <= tolerance:
                break

        return np.asarray(history)
