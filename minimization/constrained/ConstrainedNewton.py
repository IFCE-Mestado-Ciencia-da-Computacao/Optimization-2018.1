import numpy as np
from numpy.linalg import inv, pinv, ldl

from minimization.unconstrained.general_descent_method import GeneralDescentMethod


class ConstrainedNewtonFeasible(GeneralDescentMethod):

    def __init__(self, line_search):
        """
        :param LineSearch line_search:
        """
        self.line_search = line_search

    def minimize(self, f, x, h, tolerance=1e-4, iterations=120):
        """
        :param h: Restrictions
        """
        history = []

        for k in range(iterations):
            # 1. Compute the Newton step and decrement Δx_{nt}, λ(x)
            delta_x = RESOLVER KKT
            lambda_square = f.gradient(x).T @ inv(f.hessian(x)) @ f.gradient(x)

            # 2. Stopping criterion. quit if λ^2/2 <= tolerance
            if lambda_square/2 <= tolerance:
                break

            # 3. Line search. Choose step size t by backtracking line search.
            t = self.line_search.calculate(f, x, delta_x)

            # 4. Update. x := x + t∆x_{nt}
            x = x + t * delta_x

        return np.array(history)

    def delta_x(self):
        """
        |∇2f(x) A^T| = |delta_x| = | ∇f(x)|
        |A        0| = |      w| = |Ax - b|
        """
        A = A

        de
        pass


class ConstrainedNewtonInfeasible(object):

    def minimize(self, f, x, tolerance=1e-4, iterations=60):
        history = []

        for k in range(iterations):
            # Primal Newton step
            delta_x = - inv(f.hessian(x)) @ f.gradient(x)
            # Dual Newton step
            delta_v

            # Backtracking line search on ||r||_2
            t = 1
            while
                t = beta * t

            t = self.line_search.calculate(f, x, delta_x)

            x = x + t * delta_x
            v = v + t * delta_v

            if not A*x == b or not ||r(x, v)||_2 <= tolerance:
                break

        return np.array(history)

    def delta_x(self):
        """
        |∇2f(x) A^T| = |delta_x| = | ∇f(x)|
        |A        0| = |      w| = |Ax - b|
        """
        A = A

        de
        pass


def KKT_LDL():
    """
    file:///home/paulo/Downloads/LDL%20factorization%20for%20dummies.pdf
    :return:
    """
    return ldl()
