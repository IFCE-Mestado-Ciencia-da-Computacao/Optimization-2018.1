import numpy as np
from numpy.linalg import inv

from minimization.unconstrained.general_descent_method import GeneralDescentMethod


class Newton(GeneralDescentMethod):

    def __init__(self, line_search):
        """
        :param LineSearch line_search:
        """
        self.line_search = line_search

    def minimize(self, f, x, tolerance=1e-4, iterations=60):
        """
        given a starting point x ∈ dom f , tolerance ǫ > 0.

        repeat
            1. Compute the Newton step and decrement.
             ∆x_{nt} = −∇^2f(x)^{−1} ∇f (x)
             λ^2 = ∇f(x).T ∇^2f(x)^{−1} ∇f(x).
            2. Stopping criterion. quit if λ^2/2 ≤ tolerance.
            3. Line search. Choose step size t by backtracking line search.
            4. Update. x := x + t∆x nt .

        :param Function f:
        :param numpy.ndarray x:
        :param float tolerance: $\eta$
        :param int iterations:
        """
        history = [x]

        for k in range(iterations):
            delta_x = - inv(f.hessian(x)) @ f.gradient(x)

            lambda_square = f.gradient(x).T @ inv(f.hessian(x)) @ f.gradient(x)
            if lambda_square/2 <= tolerance:
                break

            t = self.line_search.calculate(f, x, delta_x)

            x = x + t * delta_x
            history.append(x)

        return np.asarray(history)
